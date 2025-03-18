"""
This module contains the Results functionality of the flixOpt framework.
It provides high level functions to analyze the results of a calculation.
It leverages the plotting.py module to plot the results.
The results can also be analyzed without this module, as the results are stored in a widely supported format.
"""

import datetime
import json
import logging
import pathlib
import timeit
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly
import yaml

from flixOpt import plotting, utils

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import pyvis

logger = logging.getLogger('flixOpt')


class ElementResults:
    def __init__(self, infos: Dict, results: Dict):
        self.all_infos = infos
        self.all_results = results
        self.label = self.all_infos['label']

    def __repr__(self):
        return f'{self.__class__.__name__}({self.label})'

    @property
    def variables_flat(self) -> Dict[str, Union[int, float, np.ndarray]]:
        return flatten_dict(self.all_results)


class CalculationResults:
    def __init__(self, calculation_name: str, folder: str) -> None:
        self.name = calculation_name
        self.folder = pathlib.Path(folder)
        self._path_infos = self.folder / f'{calculation_name}_infos.yaml'
        self._path_data = self.folder / f'{calculation_name}_data.json'
        self._path_results = self.folder / f'{calculation_name}_results.json'

        start_time = timeit.default_timer()
        with open(self._path_infos, 'rb') as f:
            self.calculation_infos: Dict = yaml.safe_load(f)
        logger.info(f'Loading Calculation Infos from .yaml took {(timeit.default_timer() - start_time):>8.2f} seconds')

        start_time = timeit.default_timer()
        with open(self._path_results, 'rb') as f:
            self.all_results: Dict = json.load(f)
        self.all_results = utils.convert_numeric_lists_to_arrays(self.all_results)
        logger.info(f'Loading results from .json took {(timeit.default_timer() - start_time):>8.2f} seconds')

        start_time = timeit.default_timer()
        with open(self._path_data, 'rb') as f:
            self.all_data: Dict = json.load(f)
        self.all_data = utils.convert_numeric_lists_to_arrays(self.all_data)
        logger.info(f'Loading data from .json took {(timeit.default_timer() - start_time):>8.2f} seconds')

        self.component_results: Dict[str, ComponentResults] = {}
        self.effect_results: Dict[str, EffectResults] = {}
        self.bus_results: Dict[str, BusResults] = {}

        self.time_with_end = np.array(
            [datetime.datetime.fromisoformat(date) for date in self.all_results['Time']]
        ).astype('datetime64')
        self.time = self.time_with_end[:-1]
        self.time_intervals_in_hours = np.array(self.all_results['Time intervals in hours'])

        self._construct_component_results()
        self._construct_bus_results()
        self._construct_effect_results()

    def _construct_component_results(self):
        comp_results = self.all_results['Components']
        comp_infos = self.all_data['Components']
        if not comp_results.keys() == comp_infos.keys():
            logger.warning(f'Missing Component or mismatched keys: {comp_results.keys() ^ comp_infos.keys()}')

        for key in comp_results.keys():
            infos, results = comp_infos.get(key, {}), comp_results.get(key, {})
            res = ComponentResults(infos, results)
            self.component_results[res.label] = res

    def _construct_effect_results(self):
        effect_results = self.all_results['Effects']
        effect_infos = self.all_data['Effects']
        effect_infos['penalty'] = {'label': 'Penalty'}
        if not effect_results.keys() == effect_infos.keys():
            logger.warning(f'Missing Effect or mismatched keys: {effect_results.keys() ^ effect_infos.keys()}')

        for key in effect_results.keys():
            infos, results = effect_infos.get(key, {}), effect_results.get(key, {})
            res = EffectResults(infos, results)
            self.effect_results[res.label] = res

    def _construct_bus_results(self):
        """This has to be called after _construct_component_results(), as its using the Flows from the Components"""
        bus_results = self.all_results['Buses']
        bus_infos = self.all_data['Buses']
        if not bus_results.keys() == bus_infos.keys():
            logger.warning(f'Missing Bus or mismatched keys: {bus_results.keys() ^ bus_infos.keys()}')

        for bus_label in bus_results.keys():
            infos, results = bus_infos.get(bus_label, {}), bus_results.get(bus_label, {})
            inputs = [
                flow
                for flow in self.flow_results().values()
                if bus_label == flow.bus_label and not flow.is_input_in_component
            ]
            outputs = [
                flow
                for flow in self.flow_results().values()
                if bus_label == flow.bus_label and flow.is_input_in_component
            ]
            res = BusResults(infos, results, inputs, outputs)
            self.bus_results[res.label] = res

    def flow_results(self) -> Dict[str, 'FlowResults']:
        return {
            flow.label_full: flow for comp in self.component_results.values() for flow in comp.inputs + comp.outputs
        }

    def to_dataframe(
        self,
        label: str,
        variable_name: str = 'flow_rate',
        input_factor: Optional[Literal[1, -1]] = -1,
        output_factor: Optional[Literal[1, -1]] = 1,
        threshold: Optional[float] = 1e-5,
        with_last_time_step: bool = True,
    ) -> pd.DataFrame:
        """
        Convert results of a specified element to a DataFrame.

        Parameters
        ----------
        label : str
            The label of the element (Component, Bus, or Flow) to retrieve data for.
        variable_name : str, default='flow_rate'
            The name of the variable to extract from the element's data.
        input_factor : Optional[Literal[1, -1]], default=-1
            Factor to apply to input values.
        output_factor : Optional[Literal[1, -1]], default=1
            Factor to apply to output values.
        threshold : Optional[float], default=1e-5
            Minimum absolute value for data inclusion in the DataFrame.
        with_last_time_step : bool, default=True
            Whether to include the last time step in the DataFrame index.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the specified variable's data with a datetime index.
            Dataframe is empty (no index), if no values are left after filtering.

        Raises
        ------
        ValueError
            If no data is found for the specified variable.
        """

        comp_or_bus = {**self.component_results, **self.bus_results}.get(label, None)
        flow = self.flow_results().get(label, None)

        if comp_or_bus is not None and flow is not None:
            raise Exception(f'{label=} matches both a Flow and a Component/Bus. That is an internal Error!')
        elif comp_or_bus is not None:
            df = comp_or_bus.to_dataframe(variable_name, input_factor, output_factor)
        elif flow is not None:
            df = flow.to_dataframe(variable_name)
        else:
            raise ValueError(f'No Element found with {label=}')

        if threshold is not None:
            df = df.loc[:, ((df > threshold) | (df < -1 * threshold)).any()]  # Check if any value exceeds the threshold
        if df.empty:  # If no values are left, return an empty DataFrame
            return df

        if with_last_time_step:
            if len(df) == len(self.time):
                df.loc[len(df)] = df.iloc[-1]
            df.index = self.time_with_end
        elif len(df) == len(self.time_with_end):
            df.index = self.time_with_end
        else:
            df.index = self.time

        return df

    def plot_operation(
        self,
        label: str,
        mode: Literal['bar', 'line', 'area', 'heatmap'] = 'area',
        variable_name: str = 'flow_rate',
        heatmap_periods: Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'] = 'D',
        heatmap_steps_per_period: Literal['W', 'D', 'h', '15min', 'min'] = 'h',
        colors: Union[str, List[str]] = 'viridis',
        engine: Literal['plotly', 'matplotlib'] = 'plotly',
        invert: bool = True,
        show: bool = True,
        save: bool = False,
        path: Union[str, pathlib.Path, Literal['auto']] = 'auto',
    ) -> Union['go.Figure', Tuple['plt.Figure', 'plt.Axes']]:
        """
        Plots the operation results for a specified Element using the chosen plotting engine and mode.

        Parameters
        ----------
        label : str
            The label of the element to plot (e.g., a component or bus).
        mode : {'bar', 'line', 'area', 'heatmap'}, default='area'
            The type of plot to generate.
        variable_name : str, default='flow_rate'
            The variable to plot from the element's data.
        heatmap_periods : {'YS', 'MS', 'W', 'D', 'h', '15min', 'min'}, default='D'
            The period for heatmap plotting.
        heatmap_steps_per_period : {'W', 'D', 'h', '15min', 'min'}, default='h'
            The steps per period for heatmap plotting.
        colors : str or List[str], default='viridis'
            The colors or colorscale to use for the plot.
        engine : {'plotly', 'matplotlib'}, default='plotly'
            The plotting engine to use.
        invert : bool, default=False
            Whether to invert the input and output factors.
        show : bool, default=True
            Whether to display the plot immediately. (This includes saving the plot to file when engine='plotly')
        save : bool, default=False
            Whether to save the plot to a file.
        path : Union[str, pathlib.Path, Literal['auto']], default='auto'
            The path to save the plot to. If 'auto', the plot is saved to an automatically named file.

        Returns
        -------
        Union[go.Figure, Tuple[plt.Figure, plt.Axes]]
            The generated plot object, either a Plotly figure or a Matplotlib figure and axes.

        Raises
        ------
        ValueError
            If an invalid engine or color configuration is provided for heatmap mode.
        """

        if mode == 'heatmap' and not np.all(self.time_intervals_in_hours == self.time_intervals_in_hours[0]):
            logger.warning(
                'Heat map plotting with irregular time intervals in time series can lead to unwanted effects'
            )
        if mode == 'heatmap' and not isinstance(colors, str):
            raise ValueError(
                f'For a heatmap, you need to pass the colors as a valid name of a colormap, not {colors=}.'
                f'Try "Turbo", "Hot", or "Viridis" instead.'
            )

        title = f'{variable_name.replace("_", " ").title()} of {label}'
        if path == 'auto':
            file_suffix = 'html' if engine == 'plotly' else 'png'
            if mode == 'heatmap':
                path = self.folder / f'{title} ({mode} {heatmap_periods}-{heatmap_steps_per_period}).{file_suffix}'
            else:
                path = self.folder / f'{title} ({mode}).{file_suffix}'

        data = self.to_dataframe(
            label, variable_name, input_factor=-1 if not invert else 1, output_factor=1 if not invert else -1
        )
        if mode == 'heatmap':
            heatmap_data = plotting.heat_map_data_from_df(data, heatmap_periods, heatmap_steps_per_period, 'ffill')

        if engine == 'plotly':
            if mode == 'heatmap':
                return plotting.heat_map_plotly(
                    heatmap_data, title=title, color_map=colors, show=show, save=save, path=path
                )
            else:
                return plotting.with_plotly(
                    data=data, mode=mode, show=show, title=title, colors=colors, save=save, path=path
                )

        elif engine == 'matplotlib':
            if mode == 'heatmap':
                return plotting.heat_map_matplotlib(
                    heatmap_data, color_map=colors, show=show, path=path if save else None
                )
            else:
                return plotting.with_matplotlib(
                    data=data, mode=mode, colors=colors, show=show, path=path if save else None
                )
        else:
            raise ValueError(f'Unknown Engine: {engine=}')

    def plot_storage(
        self,
        label: str,
        variable_name: str = 'flow_rate',
        mode: Literal['bar', 'line', 'area'] = 'area',
        colors: Union[str, List[str]] = 'viridis',
        invert: bool = True,
        show: bool = True,
        save: bool = False,
        path: Union[str, pathlib.Path, Literal['auto']] = 'auto',
    ):
        """
        Plots the storage operation results for a specified Storage Element, including its charge state.

        Parameters
        ----------
        label : str
            The label of the Storage to plot
        variable_name : str, default='flow_rate'
            The variable to plot from the element's data.
        mode : {'bar', 'line', 'area'}, default='area'
            The type of plot to generate.
        colors : str or List[str], default='viridis'
            The colors or colorscale to use for the plot.
        invert : bool, default=True
            Whether to invert the input and output factors.
        show : bool, default=True
            Whether to display the plot immediately. (This includes saving the plot to file when engine='plotly')
        save : bool, default=False
            Whether to save the plot to a file.
        path : Union[str, pathlib.Path, Literal['auto']], default='auto'
            The path to save the plot to. If 'auto', the plot is saved to an automatically named file.

        Returns
        -------
        plotly.graph_objs.Figure
            The generated Plotly figure object with the storage operation plot.
        """
        fig = self.plot_operation(
            label, mode, variable_name, invert=invert, engine='plotly', show=False, colors=colors, save=False
        )
        fig.add_trace(
            plotly.graph_objs.Scatter(
                x=self.time_with_end,
                y={**self.component_results, **self.bus_results}[label].variables['charge_state'],
                mode='lines',
                name='Charge State',
            )
        )

        title = f'{variable_name.replace("_", " ").title()} and Charge State of {label}'
        fig.update_layout(title=title)

        if path == 'auto':
            path = self.folder / f'{title} ({mode}).html'
            path = path.as_posix()
        if show:
            plotly.offline.plot(fig, filename=path)
        elif save:  # If show, the file is saved anyway
            fig.write_html(path)

        return fig

    def visualize_network(
        self,
        path: Union[bool, str, pathlib.Path] = 'results/network.html',
        controls: Union[
            bool,
            List[
                Literal['nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer']
            ],
        ] = True,
        show: bool = True,
    ) -> Optional['pyvis.network.Network']:
        """
        Visualizes the network structure of a FlowSystem using PyVis, saving it as an interactive HTML file.

        Parameters
        ----------
        path : Union[bool, str, pathlib.Path], default='results/network.html'
            Path to save the HTML visualization. If False, the visualization is created but not saved.
        controls : Union[bool, List[str]], default=True
            UI controls to add to the visualization. True enables all available controls, or specify a list of controls.
        show : bool, default=True
            Whether to open the visualization in the web browser.

        Returns
        -------
        Optional[pyvis.network.Network]
            The Network instance representing the visualization, or None if pyvis is not installed.

        Notes
        -----
        This function requires pyvis. If not installed, the function prints a warning and returns None.
        Nodes are styled based on type (e.g., circles for buses, boxes for components) and annotated with node information.
        """
        from . import plotting

        return plotting.visualize_network(
            self.calculation_infos['Network']['Nodes'], self.calculation_infos['Network']['Edges'], path, controls, show
        )


class FlowResults(ElementResults):
    def __init__(self, infos: Dict, results: Dict, label_of_component: str) -> None:
        super().__init__(infos, results)
        self.is_input_in_component = self.all_infos['is_input_in_component']
        self.component_label = label_of_component
        self.bus_label = self.all_infos['bus']['label']
        self.label_full = f'{label_of_component}__{self.label}'
        self.variables = self.all_results

    def to_dataframe(self, variable_name: str = 'flow_rate') -> pd.DataFrame:
        return pd.DataFrame({variable_name: self.variables[variable_name]})


class ComponentResults(ElementResults):
    def __init__(self, infos: Dict, results: Dict):
        super().__init__(infos, results)
        inputs, outputs = self._create_flow_results()
        self.inputs: List[FlowResults] = inputs
        self.outputs: List[FlowResults] = outputs
        self.variables = {key: val for key, val in self.all_results.items() if key not in self.inputs + self.outputs}

    def _create_flow_results(self) -> Tuple[List[FlowResults], List[FlowResults]]:
        flow_infos = {flow['label']: flow for flow in self.all_infos['inputs'] + self.all_infos['outputs']}
        flow_results = {flow_info['label']: self.all_results[flow_info['label']] for flow_info in flow_infos.values()}
        flows = [
            FlowResults(flow_info, flow_result, self.label)
            for flow_info, flow_result in zip(flow_infos.values(), flow_results.values(), strict=False)
        ]
        inputs = [flow for flow in flows if flow.is_input_in_component]
        outputs = [flow for flow in flows if not flow.is_input_in_component]
        return inputs, outputs

    def to_dataframe(
        self,
        variable_name: str = 'flow_rate',
        input_factor: Optional[Literal[1, -1]] = -1,
        output_factor: Optional[Literal[1, -1]] = 1,
    ) -> pd.DataFrame:
        inputs, outputs = {}, {}
        if input_factor is not None:
            inputs = {flow.label_full: (flow.variables[variable_name] * input_factor) for flow in self.inputs}
        if output_factor is not None:
            outputs = {flow.label_full: flow.variables[variable_name] * output_factor for flow in self.outputs}

        return pd.DataFrame(data={**inputs, **outputs})


class BusResults(ElementResults):
    def __init__(self, infos: Dict, results: Dict, inputs: List[FlowResults], outputs: List[FlowResults]):
        super().__init__(infos, results)
        self.inputs = inputs
        self.outputs = outputs
        self.variables = {key: val for key, val in self.all_results.items() if key not in self.inputs + self.outputs}

    def to_dataframe(
        self,
        variable_name: str = 'flow_rate',
        input_factor: Optional[Literal[1, -1]] = -1,
        output_factor: Optional[Literal[1, -1]] = 1,
    ) -> pd.DataFrame:
        inputs, outputs = {}, {}
        if input_factor is not None:
            inputs = {flow.label_full: (flow.variables[variable_name] * input_factor) for flow in self.inputs}
            if 'excess_input' in self.variables:
                inputs['Excess Input'] = self.variables['excess_input'] * input_factor
        if output_factor is not None:
            outputs = {flow.label_full: flow.variables[variable_name] * output_factor for flow in self.outputs}
            if 'excess_output' in self.variables:
                outputs['Excess Output'] = self.variables['excess_output'] * output_factor

        return pd.DataFrame(data={**inputs, **outputs})


class EffectResults(ElementResults):
    pass


def extract_single_result(
    results_data: dict[str, Dict[str, Union[int, float, np.ndarray, dict]]], keys: List[str]
) -> Optional[Union[int, float, np.ndarray]]:
    """Goes through a nested dictionary with the given keys. Returns the value if found. Else returns None"""
    for key in keys:
        if isinstance(results_data, dict):
            results_data = results_data.get(key, None)
        else:
            return None
    return results_data


def extract_results(
    results_data: dict[str, Dict[str, Union[int, float, np.ndarray, dict]]], keys: List[str], keep_none: bool = False
) -> Dict[str, Union[int, float, np.ndarray]]:
    """For each item in a dictionary, goes through its sub dictionaries.
    Returns the value if found. Else returns None. If specified, removes all None values
    """
    data = {kind: extract_single_result(results_data.get(kind, {}), keys) for kind in results_data.keys()}
    if keep_none:
        return data
    else:
        return {key: value for key, value in data.items() if value is not None}


def flatten_dict(d, parent_key='', sep='__'):
    """
    Recursively flattens a nested dictionary.

    Parameters:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key for the current recursion level.
        sep (str): The separator to use when concatenating keys.

    Returns:
        dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k  # Combine parent key and current key
        if isinstance(v, dict):  # If the value is a nested dictionary, recurse
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:  # Otherwise, just add the key-value pair
            if new_key not in items:
                items.append((new_key, v))
            else:
                for i in range(100000):
                    new_key = f'{new_key}_#{i}'
                    if new_key not in items:
                        items.append((new_key, v))
                        break
    return dict(items)


if __name__ == '__main__':
    results = CalculationResults(
        'Sim1', '/Users/felix/Documents/Dokumente - eigene/Neuer Ordner/flixOpt-Fork/examples/Ex02_complex/results'
    )

    results.to_dataframe('Kessel')
    results.plot_flow_rate('Kessel__Q_fu', 'heatmap')
    plotting.heat_map_plotly(
        plotting.heat_map_data_from_df(
            pd.DataFrame(results.component_results['Speicher'].variables['charge_state'], index=results.time_with_end),
            periods='D',
            steps_per_period='15min',
        )
    )

    results.plot_operation('Fernwärme', 'area', engine='plotly')
    fig = results.plot_operation('Fernwärme', 'area', engine='plotly')
    fig = plotting.with_plotly(results.to_dataframe('Wärmelast'), 'line', fig=fig)
    import plotly.offline

    plotly.offline.plot(fig)

    extract_results(results.all_results['Components'], ['Q_th', 'flow_rate'])
    extract_single_result(results.all_results['Components'], ['Kessel', 'Q_th', 'flow_rate'])

    fig = plotting.with_plotly(
        pd.DataFrame(extract_results(results.all_results['Components'], ['OnOff', 'on']), index=results.time),
        mode='bar',
    )
    fig.update_layout(barmode='group', bargap=0.2, bargroupgap=0.1)
    plotly.offline.plot(fig)
