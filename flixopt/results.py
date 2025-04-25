import datetime
import json
import logging
import pathlib
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import linopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import xarray as xr
import yaml

from . import io as fx_io
from . import plotting
from .core import TimeSeriesCollection

if TYPE_CHECKING:
    import pyvis

    from .calculation import Calculation, SegmentedCalculation


logger = logging.getLogger('flixopt')


class CalculationResults:
    """Results container for Calculation results.

    This class is used to collect the results of a Calculation.
    It provides access to component, bus, and effect
    results, and includes methods for filtering, plotting, and saving results.

    The recommended way to create instances is through the class methods
    `from_file()` or `from_calculation()`, rather than direct initialization.

    Attributes:
        solution (xr.Dataset): Dataset containing optimization results.
        flow_system (xr.Dataset): Dataset containing the flow system.
        summary (Dict): Information about the calculation.
        name (str): Name identifier for the calculation.
        model (linopy.Model): The optimization model (if available).
        folder (pathlib.Path): Path to the results directory.
        components (Dict[str, ComponentResults]): Results for each component.
        buses (Dict[str, BusResults]): Results for each bus.
        effects (Dict[str, EffectResults]): Results for each effect.
        timesteps_extra (pd.DatetimeIndex): The extended timesteps.
        hours_per_timestep (xr.DataArray): Duration of each timestep in hours.

    Example:
        Load results from saved files:

        >>> results = CalculationResults.from_file('results_dir', 'optimization_run_1')
        >>> element_result = results['Boiler']
        >>> results.plot_heatmap('Boiler(Q_th)|flow_rate')
        >>> results.to_file(compression=5)
        >>> results.to_file(folder='new_results_dir', compression=5)  # Save the results to a new folder
    """

    @classmethod
    def from_file(cls, folder: Union[str, pathlib.Path], name: str):
        """Create CalculationResults instance by loading from saved files.

        This method loads the calculation results from previously saved files,
        including the solution, flow system, model (if available), and metadata.

        Args:
            folder: Path to the directory containing the saved files.
            name: Base name of the saved files (without file extensions).

        Returns:
            CalculationResults: A new instance containing the loaded data.

        Raises:
            FileNotFoundError: If required files cannot be found.
            ValueError: If files exist but cannot be properly loaded.
        """
        folder = pathlib.Path(folder)
        paths = fx_io.CalculationResultsPaths(folder, name)

        model = None
        if paths.linopy_model.exists():
            try:
                logger.info(f'loading the linopy model "{name}" from file ("{paths.linopy_model}")')
                model = linopy.read_netcdf(paths.linopy_model)
            except Exception as e:
                logger.critical(f'Could not load the linopy model "{name}" from file ("{paths.linopy_model}"): {e}')

        with open(paths.summary, 'r', encoding='utf-8') as f:
            summary = yaml.load(f, Loader=yaml.FullLoader)

        return cls(
            solution=fx_io.load_dataset_from_netcdf(paths.solution),
            flow_system=fx_io.load_dataset_from_netcdf(paths.flow_system),
            name=name,
            folder=folder,
            model=model,
            summary=summary,
        )

    @classmethod
    def from_calculation(cls, calculation: 'Calculation'):
        """Create CalculationResults directly from a Calculation object.

        This method extracts the solution, flow system, and other relevant
        information directly from an existing Calculation object.

        Args:
            calculation: A Calculation object containing a solved model.

        Returns:
            CalculationResults: A new instance containing the results from
                the provided calculation.

        Raises:
            AttributeError: If the calculation doesn't have required attributes.
        """
        return cls(
            solution=calculation.model.solution,
            flow_system=calculation.flow_system.as_dataset(constants_in_dataset=True),
            summary=calculation.summary,
            model=calculation.model,
            name=calculation.name,
            folder=calculation.folder,
        )

    def __init__(
        self,
        solution: xr.Dataset,
        flow_system: xr.Dataset,
        name: str,
        summary: Dict,
        folder: Optional[pathlib.Path] = None,
        model: Optional[linopy.Model] = None,
    ):
        """
        Args:
            solution: The solution of the optimization.
            flow_system: The flow_system that was used to create the calculation as a datatset.
            name: The name of the calculation.
            summary: Information about the calculation,
            folder: The folder where the results are saved.
            model: The linopy model that was used to solve the calculation.
        """
        self.solution = solution
        self.flow_system = flow_system
        self.summary = summary
        self.name = name
        self.model = model
        self.folder = pathlib.Path(folder) if folder is not None else pathlib.Path.cwd() / 'results'
        self.components = {
            label: ComponentResults.from_json(self, infos) for label, infos in self.solution.attrs['Components'].items()
        }

        self.buses = {label: BusResults.from_json(self, infos) for label, infos in self.solution.attrs['Buses'].items()}

        self.effects = {
            label: EffectResults.from_json(self, infos) for label, infos in self.solution.attrs['Effects'].items()
        }

        self.timesteps_extra = self.solution.indexes['time']
        self.hours_per_timestep = TimeSeriesCollection.calculate_hours_per_timestep(self.timesteps_extra)

    def __getitem__(self, key: str) -> Union['ComponentResults', 'BusResults', 'EffectResults']:
        if key in self.components:
            return self.components[key]
        if key in self.buses:
            return self.buses[key]
        if key in self.effects:
            return self.effects[key]
        raise KeyError(f'No element with label {key} found.')

    @property
    def storages(self) -> List['ComponentResults']:
        """All storages in the results."""
        return [comp for comp in self.components.values() if comp.is_storage]

    @property
    def objective(self) -> float:
        """The objective result of the optimization."""
        return self.summary['Main Results']['Objective']

    @property
    def variables(self) -> linopy.Variables:
        """The variables of the optimization. Only available if the linopy.Model is available."""
        if self.model is None:
            raise ValueError('The linopy model is not available.')
        return self.model.variables

    @property
    def constraints(self) -> linopy.Constraints:
        """The constraints of the optimization. Only available if the linopy.Model is available."""
        if self.model is None:
            raise ValueError('The linopy model is not available.')
        return self.model.constraints

    def filter_solution(
        self, variable_dims: Optional[Literal['scalar', 'time']] = None, element: Optional[str] = None
    ) -> xr.Dataset:
        """
        Filter the solution to a specific variable dimension and element.
        If no element is specified, all elements are included.

        Args:
            variable_dims: The dimension of the variables to filter for.
            element: The element to filter for.
        """
        if element is not None:
            return filter_dataset(self[element].solution, variable_dims)
        return filter_dataset(self.solution, variable_dims)

    def plot_heatmap(
        self,
        variable_name: str,
        heatmap_timeframes: Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'] = 'D',
        heatmap_timesteps_per_frame: Literal['W', 'D', 'h', '15min', 'min'] = 'h',
        color_map: str = 'portland',
        save: Union[bool, pathlib.Path] = False,
        show: bool = True,
        engine: plotting.PlottingEngine = 'plotly',
    ) -> Union[plotly.graph_objs.Figure, Tuple[plt.Figure, plt.Axes]]:
        return plot_heatmap(
            dataarray=self.solution[variable_name],
            name=variable_name,
            folder=self.folder,
            heatmap_timeframes=heatmap_timeframes,
            heatmap_timesteps_per_frame=heatmap_timesteps_per_frame,
            color_map=color_map,
            save=save,
            show=show,
            engine=engine,
        )

    def plot_network(
        self,
        controls: Union[
            bool,
            List[
                Literal['nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer']
            ],
        ] = True,
        path: Optional[pathlib.Path] = None,
        show: bool = False,
    ) -> 'pyvis.network.Network':
        """See flixopt.flow_system.FlowSystem.plot_network"""
        try:
            from .flow_system import FlowSystem

            flow_system = FlowSystem.from_dataset(self.flow_system)
        except Exception as e:
            logger.critical(f'Could not reconstruct the flow_system from dataset: {e}')
            return None
        if path is None:
            path = self.folder / f'{self.name}--network.html'
        return flow_system.plot_network(controls=controls, path=path, show=show)

    def to_file(
        self,
        folder: Optional[Union[str, pathlib.Path]] = None,
        name: Optional[str] = None,
        compression: int = 5,
        document_model: bool = True,
        save_linopy_model: bool = False,
    ):
        """
        Save the results to a file
        Args:
            folder: The folder where the results should be saved. Defaults to the folder of the calculation.
            name: The name of the results file. If not provided, Defaults to the name of the calculation.
            compression: The compression level to use when saving the solution file (0-9). 0 means no compression.
            document_model: Wether to document the mathematical formulations in the model.
            save_linopy_model: Wether to save the model to file. If True, the (linopy) model is saved as a .nc4 file.
                The model file size is rougly 100 times larger than the solution file.
        """
        folder = self.folder if folder is None else pathlib.Path(folder)
        name = self.name if name is None else name
        if not folder.exists():
            try:
                folder.mkdir(parents=False)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f'Folder {folder} and its parent do not exist. Please create them first.'
                ) from e

        paths = fx_io.CalculationResultsPaths(folder, name)

        fx_io.save_dataset_to_netcdf(self.solution, paths.solution, compression=compression)
        fx_io.save_dataset_to_netcdf(self.flow_system, paths.flow_system, compression=compression)

        with open(paths.summary, 'w', encoding='utf-8') as f:
            yaml.dump(self.summary, f, allow_unicode=True, sort_keys=False, indent=4, width=1000)

        if save_linopy_model:
            if self.model is None:
                logger.critical('No model in the CalculationResults. Saving the model is not possible.')
            else:
                self.model.to_netcdf(paths.linopy_model)

        if document_model:
            if self.model is None:
                logger.critical('No model in the CalculationResults. Documenting the model is not possible.')
            else:
                fx_io.document_linopy_model(self.model, path=paths.model_documentation)

        logger.info(f'Saved calculation results "{name}" to {paths.model_documentation.parent}')


class _ElementResults:
    @classmethod
    def from_json(cls, calculation_results, json_data: Dict) -> '_ElementResults':
        return cls(calculation_results, json_data['label'], json_data['variables'], json_data['constraints'])

    def __init__(
        self, calculation_results: CalculationResults, label: str, variables: List[str], constraints: List[str]
    ):
        self._calculation_results = calculation_results
        self.label = label
        self._variable_names = variables
        self._constraint_names = constraints

        self.solution = self._calculation_results.solution[self._variable_names]

    @property
    def variables(self) -> linopy.Variables:
        """
        Returns the variables of the element.

        Raises:
            ValueError: If the linopy model is not availlable.
        """
        if self._calculation_results.model is None:
            raise ValueError('The linopy model is not available.')
        return self._calculation_results.model.variables[self._variable_names]

    @property
    def constraints(self) -> linopy.Constraints:
        """
        Returns the variables of the element.

        Raises:
            ValueError: If the linopy model is not availlable.
        """
        if self._calculation_results.model is None:
            raise ValueError('The linopy model is not available.')
        return self._calculation_results.model.constraints[self._constraint_names]

    def filter_solution(self, variable_dims: Optional[Literal['scalar', 'time']] = None) -> xr.Dataset:
        """
        Filter the solution of the element by dimension.

        Args:
            variable_dims: The dimension of the variables to filter for.
        """
        return filter_dataset(self.solution, variable_dims)


class _NodeResults(_ElementResults):
    @classmethod
    def from_json(cls, calculation_results, json_data: Dict) -> '_NodeResults':
        return cls(
            calculation_results,
            json_data['label'],
            json_data['variables'],
            json_data['constraints'],
            json_data['inputs'],
            json_data['outputs'],
        )

    def __init__(
        self,
        calculation_results: CalculationResults,
        label: str,
        variables: List[str],
        constraints: List[str],
        inputs: List[str],
        outputs: List[str],
    ):
        super().__init__(calculation_results, label, variables, constraints)
        self.inputs = inputs
        self.outputs = outputs

    def plot_node_balance(
        self,
        save: Union[bool, pathlib.Path] = False,
        show: bool = True,
        colors: plotting.ColorType = 'viridis',
        engine: plotting.PlottingEngine = 'plotly',
    ) -> Union[plotly.graph_objs.Figure, Tuple[plt.Figure, plt.Axes]]:
        """
        Plots the node balance of the Component or Bus.
        Args:
            save: Whether to save the plot or not. If a path is provided, the plot will be saved at that location.
            show: Whether to show the plot or not.
            engine: The engine to use for plotting. Can be either 'plotly' or 'matplotlib'.
        """
        if engine == 'plotly':
            figure_like = plotting.with_plotly(
                self.node_balance(with_last_timestep=True).to_dataframe(),
                colors=colors,
                mode='area',
                title=f'Flow rates of {self.label}',
            )
            default_filetype = '.html'
        elif engine == 'matplotlib':
            figure_like = plotting.with_matplotlib(
                self.node_balance(with_last_timestep=True).to_dataframe(),
                colors=colors,
                mode='bar',
                title=f'Flow rates of {self.label}',
            )
            default_filetype = '.png'
        else:
            raise ValueError(f'Engine "{engine}" not supported. Use "plotly" or "matplotlib"')

        return plotting.export_figure(
            figure_like=figure_like,
            default_path=self._calculation_results.folder / f'{self.label} (flow rates)',
            default_filetype=default_filetype,
            user_path=None if isinstance(save, bool) else pathlib.Path(save),
            show=show,
            save=True if save else False,
        )

    def plot_node_balance_pie(
        self,
        lower_percentage_group: float = 5,
        colors: plotting.ColorType = 'viridis',
        text_info: str = 'percent+label+value',
        save: Union[bool, pathlib.Path] = False,
        show: bool = True,
        engine: plotting.PlottingEngine = 'plotly',
    ) -> plotly.graph_objects.Figure:
        """
        Plots a pie chart of the flow hours of the inputs and outputs of buses or components.

        Args:
            colors: a colorscale or a list of colors to use for the plot
            lower_percentage_group: The percentage of flow_hours that is grouped in "Others" (0...100)
            text_info: What information to display on the pie plot
            save: Whether to save the figure.
            show: Whether to show the figure.
            engine: Plotting engine to use. Only 'plotly' is implemented atm.
        """
        inputs = (
            sanitize_dataset(
                ds=self.solution[self.inputs],
                threshold=1e-5,
                drop_small_vars=True,
                zero_small_values=True,
            )
            * self._calculation_results.hours_per_timestep
        )
        outputs = (
            sanitize_dataset(
                ds=self.solution[self.outputs],
                threshold=1e-5,
                drop_small_vars=True,
                zero_small_values=True,
            )
            * self._calculation_results.hours_per_timestep
        )

        if engine == 'plotly':
            figure_like = plotting.dual_pie_with_plotly(
                inputs.to_dataframe().sum(),
                outputs.to_dataframe().sum(),
                colors=colors,
                title=f'Flow hours of {self.label}',
                text_info=text_info,
                subtitles=('Inputs', 'Outputs'),
                legend_title='Flows',
                lower_percentage_group=lower_percentage_group,
            )
            default_filetype = '.html'
        elif engine == 'matplotlib':
            logger.debug('Parameter text_info is not supported for matplotlib')
            figure_like = plotting.dual_pie_with_matplotlib(
                inputs.to_dataframe().sum(),
                outputs.to_dataframe().sum(),
                colors=colors,
                title=f'Total flow hours of {self.label}',
                subtitles=('Inputs', 'Outputs'),
                legend_title='Flows',
                lower_percentage_group=lower_percentage_group,
            )
            default_filetype = '.png'
        else:
            raise ValueError(f'Engine "{engine}" not supported. Use "plotly" or "matplotlib"')

        return plotting.export_figure(
            figure_like=figure_like,
            default_path=self._calculation_results.folder / f'{self.label} (total flow hours)',
            default_filetype=default_filetype,
            user_path=None if isinstance(save, bool) else pathlib.Path(save),
            show=show,
            save=True if save else False,
        )

    def node_balance(
        self,
        negate_inputs: bool = True,
        negate_outputs: bool = False,
        threshold: Optional[float] = 1e-5,
        with_last_timestep: bool = False,
    ) -> xr.Dataset:
        return sanitize_dataset(
            ds=self.solution[self.inputs + self.outputs],
            threshold=threshold,
            timesteps=self._calculation_results.timesteps_extra if with_last_timestep else None,
            negate=(
                self.outputs + self.inputs
                if negate_outputs and negate_inputs
                else self.outputs
                if negate_outputs
                else self.inputs
                if negate_inputs
                else None
            ),
        )


class BusResults(_NodeResults):
    """Results for a Bus"""


class ComponentResults(_NodeResults):
    """Results for a Component"""

    @property
    def is_storage(self) -> bool:
        return self._charge_state in self._variable_names

    @property
    def _charge_state(self) -> str:
        return f'{self.label}|charge_state'

    @property
    def charge_state(self) -> xr.DataArray:
        """Get the solution of the charge state of the Storage."""
        if not self.is_storage:
            raise ValueError(f'Cant get charge_state. "{self.label}" is not a storage')
        return self.solution[self._charge_state]

    def plot_charge_state(
        self,
        save: Union[bool, pathlib.Path] = False,
        show: bool = True,
        colors: plotting.ColorType = 'viridis',
        engine: plotting.PlottingEngine = 'plotly',
    ) -> plotly.graph_objs.Figure:
        """
        Plots the charge state of a Storage.
        Args:
            save: Whether to save the plot or not. If a path is provided, the plot will be saved at that location.
            show: Whether to show the plot or not.
            colors: The c
            engine: Plotting engine to use. Only 'plotly' is implemented atm.

        Raises:
            ValueError: If the Component is not a Storage.
        """
        if engine != 'plotly':
            raise NotImplementedError(
                f'Plotting engine "{engine}" not implemented for ComponentResults.plot_charge_state.'
            )

        if not self.is_storage:
            raise ValueError(f'Cant plot charge_state. "{self.label}" is not a storage')

        fig = plotting.with_plotly(
            self.node_balance(with_last_timestep=True).to_dataframe(),
            colors=colors,
            mode='area',
            title=f'Operation Balance of {self.label}',
        )

        # TODO: Use colors for charge state?

        charge_state = self.charge_state.to_dataframe()
        fig.add_trace(
            plotly.graph_objs.Scatter(
                x=charge_state.index, y=charge_state.values.flatten(), mode='lines', name=self._charge_state
            )
        )

        return plotting.export_figure(
            fig,
            default_path=self._calculation_results.folder / f'{self.label} (charge state)',
            default_filetype='.html',
            user_path=None if isinstance(save, bool) else pathlib.Path(save),
            show=show,
            save=True if save else False,
        )

    def node_balance_with_charge_state(
        self, negate_inputs: bool = True, negate_outputs: bool = False, threshold: Optional[float] = 1e-5
    ) -> xr.Dataset:
        """
        Returns a dataset with the node balance of the Storage including its charge state.
        Args:
            negate_inputs: Whether to negate the inputs of the Storage.
            negate_outputs: Whether to negate the outputs of the Storage.
            threshold: The threshold for small values.

        Raises:
            ValueError: If the Component is not a Storage.
        """
        if not self.is_storage:
            raise ValueError(f'Cant get charge_state. "{self.label}" is not a storage')
        variable_names = self.inputs + self.outputs + [self._charge_state]
        return sanitize_dataset(
            ds=self.solution[variable_names],
            threshold=threshold,
            timesteps=self._calculation_results.timesteps_extra,
            negate=(
                self.outputs + self.inputs
                if negate_outputs and negate_inputs
                else self.outputs
                if negate_outputs
                else self.inputs
                if negate_inputs
                else None
            ),
        )


class EffectResults(_ElementResults):
    """Results for an Effect"""

    def get_shares_from(self, element: str):
        """Get the shares from an Element (without subelements) to the Effect"""
        return self.solution[[name for name in self._variable_names if name.startswith(f'{element}->')]]


class SegmentedCalculationResults:
    """
    Class to store the results of a SegmentedCalculation.
    """

    @classmethod
    def from_calculation(cls, calculation: 'SegmentedCalculation'):
        return cls(
            [calc.results for calc in calculation.sub_calculations],
            all_timesteps=calculation.all_timesteps,
            timesteps_per_segment=calculation.timesteps_per_segment,
            overlap_timesteps=calculation.overlap_timesteps,
            name=calculation.name,
            folder=calculation.folder,
        )

    @classmethod
    def from_file(cls, folder: Union[str, pathlib.Path], name: str):
        """Create SegmentedCalculationResults directly from file"""
        folder = pathlib.Path(folder)
        path = folder / name
        nc_file = path.with_suffix('.nc4')
        logger.info(f'loading calculation "{name}" from file ("{nc_file}")')
        with open(path.with_suffix('.json'), 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        return cls(
            [CalculationResults.from_file(folder, name) for name in meta_data['sub_calculations']],
            all_timesteps=pd.DatetimeIndex(
                [datetime.datetime.fromisoformat(date) for date in meta_data['all_timesteps']], name='time'
            ),
            timesteps_per_segment=meta_data['timesteps_per_segment'],
            overlap_timesteps=meta_data['overlap_timesteps'],
            name=name,
            folder=folder,
        )

    def __init__(
        self,
        segment_results: List[CalculationResults],
        all_timesteps: pd.DatetimeIndex,
        timesteps_per_segment: int,
        overlap_timesteps: int,
        name: str,
        folder: Optional[pathlib.Path] = None,
    ):
        self.segment_results = segment_results
        self.all_timesteps = all_timesteps
        self.timesteps_per_segment = timesteps_per_segment
        self.overlap_timesteps = overlap_timesteps
        self.name = name
        self.folder = pathlib.Path(folder) if folder is not None else pathlib.Path.cwd() / 'results'
        self.hours_per_timestep = TimeSeriesCollection.calculate_hours_per_timestep(self.all_timesteps)

    @property
    def meta_data(self) -> Dict[str, Union[int, List[str]]]:
        return {
            'all_timesteps': [datetime.datetime.isoformat(date) for date in self.all_timesteps],
            'timesteps_per_segment': self.timesteps_per_segment,
            'overlap_timesteps': self.overlap_timesteps,
            'sub_calculations': [calc.name for calc in self.segment_results],
        }

    @property
    def segment_names(self) -> List[str]:
        return [segment.name for segment in self.segment_results]

    def solution_without_overlap(self, variable_name: str) -> xr.DataArray:
        """Returns the solution of a variable without overlapping timesteps"""
        dataarrays = [
            result.solution[variable_name].isel(time=slice(None, self.timesteps_per_segment))
            for result in self.segment_results[:-1]
        ] + [self.segment_results[-1].solution[variable_name]]
        return xr.concat(dataarrays, dim='time')

    def plot_heatmap(
        self,
        variable_name: str,
        heatmap_timeframes: Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'] = 'D',
        heatmap_timesteps_per_frame: Literal['W', 'D', 'h', '15min', 'min'] = 'h',
        color_map: str = 'portland',
        save: Union[bool, pathlib.Path] = False,
        show: bool = True,
        engine: plotting.PlottingEngine = 'plotly',
    ) -> Union[plotly.graph_objs.Figure, Tuple[plt.Figure, plt.Axes]]:
        """
        Plots a heatmap of the solution of a variable.

        Args:
            variable_name: The name of the variable to plot.
            heatmap_timeframes: The timeframes to use for the heatmap.
            heatmap_timesteps_per_frame: The timesteps per frame to use for the heatmap.
            color_map: The color map to use for the heatmap.
            save: Whether to save the plot or not. If a path is provided, the plot will be saved at that location.
            show: Whether to show the plot or not.
            engine: The engine to use for plotting. Can be either 'plotly' or 'matplotlib'.
        """
        return plot_heatmap(
            dataarray=self.solution_without_overlap(variable_name),
            name=variable_name,
            folder=self.folder,
            heatmap_timeframes=heatmap_timeframes,
            heatmap_timesteps_per_frame=heatmap_timesteps_per_frame,
            color_map=color_map,
            save=save,
            show=show,
            engine=engine,
        )

    def to_file(
        self, folder: Optional[Union[str, pathlib.Path]] = None, name: Optional[str] = None, compression: int = 5
    ):
        """Save the results to a file"""
        folder = self.folder if folder is None else pathlib.Path(folder)
        name = self.name if name is None else name
        path = folder / name
        if not folder.exists():
            try:
                folder.mkdir(parents=False)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f'Folder {folder} and its parent do not exist. Please create them first.'
                ) from e
        for segment in self.segment_results:
            segment.to_file(folder=folder, name=f'{name}-{segment.name}', compression=compression)

        with open(path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(self.meta_data, f, indent=4, ensure_ascii=False)
        logger.info(f'Saved calculation "{name}" to {path}')


def plot_heatmap(
    dataarray: xr.DataArray,
    name: str,
    folder: pathlib.Path,
    heatmap_timeframes: Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'] = 'D',
    heatmap_timesteps_per_frame: Literal['W', 'D', 'h', '15min', 'min'] = 'h',
    color_map: str = 'portland',
    save: Union[bool, pathlib.Path] = False,
    show: bool = True,
    engine: plotting.PlottingEngine = 'plotly',
):
    """
    Plots a heatmap of the solution of a variable.

    Args:
        dataarray: The dataarray to plot.
        name: The name of the variable to plot.
        folder: The folder to save the plot to.
        heatmap_timeframes: The timeframes to use for the heatmap.
        heatmap_timesteps_per_frame: The timesteps per frame to use for the heatmap.
        color_map: The color map to use for the heatmap.
        save: Whether to save the plot or not. If a path is provided, the plot will be saved at that location.
        show: Whether to show the plot or not.
        engine: The engine to use for plotting. Can be either 'plotly' or 'matplotlib'.
    """
    heatmap_data = plotting.heat_map_data_from_df(
        dataarray.to_dataframe(name), heatmap_timeframes, heatmap_timesteps_per_frame, 'ffill'
    )

    xlabel, ylabel = f'timeframe [{heatmap_timeframes}]', f'timesteps [{heatmap_timesteps_per_frame}]'

    if engine == 'plotly':
        figure_like = plotting.heat_map_plotly(
            heatmap_data, title=name, color_map=color_map, xlabel=xlabel, ylabel=ylabel
        )
        default_filetype = '.html'
    elif engine == 'matplotlib':
        figure_like = plotting.heat_map_matplotlib(
            heatmap_data, title=name, color_map=color_map, xlabel=xlabel, ylabel=ylabel
        )
        default_filetype = '.png'
    else:
        raise ValueError(f'Engine "{engine}" not supported. Use "plotly" or "matplotlib"')

    return plotting.export_figure(
        figure_like=figure_like,
        default_path=folder / f'{name} ({heatmap_timeframes}-{heatmap_timesteps_per_frame})',
        default_filetype=default_filetype,
        user_path=None if isinstance(save, bool) else pathlib.Path(save),
        show=show,
        save=True if save else False,
    )


def sanitize_dataset(
    ds: xr.Dataset,
    timesteps: Optional[pd.DatetimeIndex] = None,
    threshold: Optional[float] = 1e-5,
    negate: Optional[List[str]] = None,
    drop_small_vars: bool = True,
    zero_small_values: bool = False,
) -> xr.Dataset:
    """
    Sanitizes a dataset by handling small values (dropping or zeroing) and optionally reindexing the time axis.

    Args:
        ds: The dataset to sanitize.
        timesteps: The timesteps to reindex the dataset to. If None, the original timesteps are kept.
        threshold: The threshold for small values processing. If None, no processing is done.
        negate: The variables to negate. If None, no variables are negated.
        drop_small_vars: If True, drops variables where all values are below threshold.
        zero_small_values: If True, sets values below threshold to zero.

    Returns:
        xr.Dataset: The sanitized dataset.
    """
    # Create a copy to avoid modifying the original
    ds = ds.copy()

    # Step 1: Negate specified variables
    if negate is not None:
        for var in negate:
            if var in ds:
                ds[var] = -ds[var]

    # Step 2: Handle small values
    if threshold is not None:
        ds_no_nan_abs = xr.apply_ufunc(np.abs, ds).fillna(0)  # Replace NaN with 0 (below threshold) for the comparison

        # Option 1: Drop variables where all values are below threshold
        if drop_small_vars:
            vars_to_drop = [var for var in ds.data_vars if (ds_no_nan_abs[var] <= threshold).all()]
            ds = ds.drop_vars(vars_to_drop)

        # Option 2: Set small values to zero
        if zero_small_values:
            for var in ds.data_vars:
                # Create a boolean mask of values below threshold
                mask = ds_no_nan_abs[var] <= threshold
                # Only proceed if there are values to zero out
                if mask.any():
                    # Create a copy to ensure we don't modify data with views
                    ds[var] = ds[var].copy()
                    # Set values below threshold to zero
                    ds[var] = ds[var].where(~mask, 0)

    # Step 3: Reindex to specified timesteps if needed
    if timesteps is not None and not ds.indexes['time'].equals(timesteps):
        ds = ds.reindex({'time': timesteps}, fill_value=np.nan)

    return ds


def filter_dataset(
    ds: xr.Dataset,
    variable_dims: Optional[Literal['scalar', 'time']] = None,
) -> xr.Dataset:
    """
    Filters a dataset by its dimensions.

    Args:
        ds: The dataset to filter.
        variable_dims: The dimension of the variables to filter for.
    """
    if variable_dims is None:
        return ds

    if variable_dims == 'scalar':
        return ds[[name for name, da in ds.data_vars.items() if len(da.dims) == 0]]
    elif variable_dims == 'time':
        return ds[[name for name, da in ds.data_vars.items() if 'time' in da.dims]]
    else:
        raise ValueError(f'Not allowed value for "filter_dataset()": {variable_dims=}')
