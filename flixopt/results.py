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
            flow_system_data=fx_io.load_dataset_from_netcdf(paths.flow_system),
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
            flow_system_data=calculation.flow_system.as_dataset(constants_in_dataset=True),
            summary=calculation.summary,
            model=calculation.model,
            name=calculation.name,
            folder=calculation.folder,
        )

    def __init__(
        self,
        solution: xr.Dataset,
        flow_system_data: xr.Dataset,
        name: str,
        summary: Dict,
        folder: Optional[pathlib.Path] = None,
        model: Optional[linopy.Model] = None,
    ):
        """
        Args:
            solution: The solution of the optimization.
            flow_system_data: The flow_system that was used to create the calculation as a datatset.
            name: The name of the calculation.
            summary: Information about the calculation,
            folder: The folder where the results are saved.
            model: The linopy model that was used to solve the calculation.
        Deprecated:
            flow_system: Use flow_system_data instead.
        """
        self.solution = solution
        self.flow_system_data = flow_system_data
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
        self.scenarios = self.solution.indexes['scenario'] if 'scenario' in self.solution.indexes else None

        self._effect_share_factors = None
        self._flow_system = None
        self._flow_rates = None
        self._flow_hours = None
        self._effects_per_component = {'operation': None, 'invest': None, 'total': None}

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

    @property
    def effect_share_factors(self):
        if self._effect_share_factors is None:
            effect_share_factors = self.flow_system.effects.calculate_effect_share_factors()
            self._effect_share_factors = {'operation': effect_share_factors[0],
                                          'invest': effect_share_factors[1]}
        return self._effect_share_factors

    @property
    def flow_system(self):
        """ The restored flow_system that was used to create the calculation.
        Contains all input parameters."""
        if self._flow_system is None:
            from . import FlowSystem
            current_logger_level = logger.getEffectiveLevel()
            logger.setLevel(logging.CRITICAL)
            self._flow_system = FlowSystem.from_dataset(self.flow_system_data)
            self._flow_system._connect_network()
            logger.setLevel(current_logger_level)
        return self._flow_system

    def filter_solution(
        self,
        variable_dims: Optional[Literal['scalar', 'time', 'scenario', 'timeonly', 'scenarioonly']] = None,
        element: Optional[str] = None,
        timesteps: Optional[pd.DatetimeIndex] = None,
        scenarios: Optional[pd.Index] = None,
        contains: Optional[Union[str, List[str]]] = None,
        startswith: Optional[Union[str, List[str]]] = None,
    ) -> xr.Dataset:
        """
        Filter the solution to a specific variable dimension and element.
        If no element is specified, all elements are included.

        Args:
            variable_dims: The dimension of which to get variables from.
                - 'scalar': Get scalar variables (without dimensions)
                - 'time': Get time-dependent variables (with a time dimension)
                - 'scenario': Get scenario-dependent variables (with ONLY a scenario dimension)
                - 'timeonly': Get time-dependent variables (with ONLY a time dimension)
                - 'scenarioonly': Get scenario-dependent variables (with ONLY a scenario dimension)
            element: The element to filter for.
            timesteps: Optional time indexes to select. Can be:
                - pd.DatetimeIndex: Multiple timesteps
                - str/pd.Timestamp: Single timestep
                Defaults to all available timesteps.
            scenarios: Optional scenario indexes to select. Can be:
                - pd.Index: Multiple scenarios
                - str/int: Single scenario (int is treated as a label, not an index position)
                Defaults to all available scenarios.
            contains: Filter variables that contain this string or strings.
                If a list is provided, variables must contain ALL strings in the list.
            startswith: Filter variables that start with this string or strings.
                If a list is provided, variables must start with ANY of the strings in the list.
        """
        return filter_dataset(
            self.solution if element is None else self[element].solution,
            variable_dims=variable_dims,
            timesteps=timesteps,
            scenarios=scenarios,
            contains=contains,
            startswith=startswith,
        )

    def get_effects_per_component(self, mode: Literal['operation', 'invest', 'total'] = 'total') -> xr.DataArray:
        """Returns a dataset containing effect totals for each components (including their flows).

        Args:
            mode: Which effects to contain. (operation, invest, total)

        Returns:
            An xarray Dataset with an additional component dimension and effects as variables.
        """
        if mode not in ['operation', 'invest', 'total']:
            raise ValueError(f'Invalid mode {mode}')
        if self._effects_per_component[mode] is None:
            self._effects_per_component[mode] = self._create_effects_dataarray(mode)
        return self._effects_per_component[mode]

    def flow_rates(
        self, start: Optional[Union[str, List[str]]] = None, end: Optional[Union[str, List[str]]] = None
    ) -> xr.DataArray:
        """Returns a DataArray containing the flow rates of each Flow.

        Args:
            start: Optional source node(s) to filter by. Can be a single node name or a list of names.
            end: Optional destination node(s) to filter by. Can be a single node name or a list of names.
        """
        if self._flow_rates is None:
            self._flow_rates = self._create_flow_rates_dataarray()
        return filter_edges_dataset(self._flow_rates, start=start, end=end)

    def flow_hours(
        self, start: Optional[Union[str, List[str]]] = None, end: Optional[Union[str, List[str]]] = None
    ) -> xr.DataArray:
        """Returns a DataArray containing the flow hours of each Flow.

        Flow hours represent the total energy/material transferred over time,
        calculated by multiplying flow rates by the duration of each timestep.

        Args:
            start: Optional source node(s) to filter by. Can be a single node name or a list of names.
            end: Optional destination node(s) to filter by. Can be a single node name or a list of names.
        """
        if self._flow_hours is None:
            self._flow_hours = (self.flow_rates() * self.hours_per_timestep).rename('flow_hours')
        return filter_edges_dataset(self._flow_hours, start=start, end=end)

    def _create_flow_rates_dataarray(self) -> xr.DataArray:
        """Creates a DataArray containing flow rates with network topology coordinates.

        Extracts flow rates from the solution dataset and adds network topology
        information (start/end nodes) as coordinates.
        """
        # Step 1: Extract all flow rates and their metadata in one loop
        flow_data = {}

        for flow_name, flow in self.flow_system.flows.items():
            flow_data[flow_name] = {
                'flow_rate': self.solution[f'{flow_name}|flow_rate'].rename(flow_name),
                'start': flow.bus if flow.is_input_in_component else flow.component,
                'end': flow.component if flow.is_input_in_component else flow.bus,
            }

        # Step 2: Combine into one DataArray (preserving all original coords)
        flow_da = xr.concat([flow['flow_rate'] for flow in flow_data.values()],
                            dim=pd.Index(flow_data.keys(), name='flow'))

        # Step 3: Add start and end coordinates
        flow_da = flow_da.assign_coords({
            'start': ('flow', [flow['start'] for flow in flow_data.values()]),
            'end': ('flow', [flow['end'] for flow in flow_data.values()])
        })

        # Step 4: Ensure flow is the last dimension if needed
        existing_dims = [d for d in flow_da.dims if d != 'flow']
        flow_da = flow_da.transpose(*(existing_dims + ['flow']))

        return flow_da.rename('flow_rates')

    def get_effect_shares(
        self,
        element: str,
        effect: str,
        mode: Optional[Literal['operation', 'invest']] = None,
        include_flows: bool = False
    ) -> xr.Dataset:
        """Retrieves individual effect shares for a specific element and effect.
        Either for operation, investment, or both modes combined.
        Only includes the direct shares.

        Args:
            element: The element identifier for which to retrieve effect shares.
            effect: The effect identifier for which to retrieve shares.
            mode: Optional. The mode to retrieve shares for. Can be 'operation', 'invest',
                or None to retrieve both. Defaults to None.

        Returns:
            An xarray Dataset containing the requested effect shares. If mode is None,
            returns a merged Dataset containing both operation and investment shares.

        Raises:
            ValueError: If the specified effect is not available or if mode is invalid.
        """
        if effect not in self.effects:
            raise ValueError(f'Effect {effect} is not available.')

        if mode is None:
            return xr.merge([self.get_effect_shares(element=element, effect=effect, mode='operation', include_flows=include_flows),
                             self.get_effect_shares(element=element, effect=effect, mode='invest', include_flows=include_flows)])

        if mode not in ['operation', 'invest']:
            raise ValueError(f'Mode {mode} is not available. Choose between "operation" and "invest".')

        ds = xr.Dataset()

        label = f'{element}->{effect}({mode})'
        if label in self.solution:
            ds =  xr.Dataset({label: self.solution[label]})

        if include_flows:
            if element not in self.components:
                raise ValueError(f'Only use Components when retrieving Effects including flows. Got {element}')
            flows = [label.split('|')[0] for label in self.components[element].inputs + self.components[element].outputs]
            return xr.merge(
                [ds] + [self.get_effect_shares(element=flow, effect=effect, mode=mode, include_flows=False)
                        for flow in flows]
            )

        return ds

    def _compute_effect_total(
        self,
        element: str,
        effect: str,
        mode: Literal['operation', 'invest', 'total'] = 'total',
        include_flows: bool = False,
    ) -> xr.DataArray:
        """Calculates the total effect for a specific element and effect.

        This method computes the total direct and indirect effects for a given element
        and effect, considering the conversion factors between different effects.

        Args:
            element: The element identifier for which to calculate total effects.
            effect: The effect identifier to calculate.
            mode: The calculation mode. Options are:
                'operation': Returns operation-specific effects.
                'invest': Returns investment-specific effects.
                'total': Returns the sum of operation effects (across all timesteps)
                    and investment effects. Defaults to 'total'.
            include_flows: Whether to include effects from flows connected to this element.

        Returns:
            An xarray DataArray containing the total effects, named with pattern
            '{element}->{effect}' for mode='total' or '{element}->{effect}({mode})'
            for other modes.

        Raises:
            ValueError: If the specified effect is not available.
        """
        if effect not in self.effects:
            raise ValueError(f'Effect {effect} is not available.')

        if mode == 'total':
            operation = self._compute_effect_total(element=element, effect=effect, mode='operation', include_flows=include_flows)
            invest = self._compute_effect_total(element=element, effect=effect, mode='invest', include_flows=include_flows)
            if invest.isnull().all() and operation.isnull().all():
                return xr.DataArray(np.nan)
            if operation.isnull().all():
                return invest.rename(f'{element}->{effect}')
            operation = operation.sum('time')
            if invest.isnull().all():
                return operation.rename(f'{element}->{effect}')
            if 'time' in operation.indexes:
                operation = operation.sum('time')
            return invest + operation

        total = xr.DataArray(0)
        share_exists = False

        relevant_conversion_factors = {
            key[0]: value for key, value in self.effect_share_factors[mode].items() if key[1] == effect
        }
        relevant_conversion_factors[effect] = 1  # Share to itself is 1

        for target_effect, conversion_factor in relevant_conversion_factors.items():
            label = f'{element}->{target_effect}({mode})'
            if label in self.solution:
                share_exists = True
                da = self.solution[label]
                total = da * conversion_factor + total

            if include_flows:
                if element not in self.components:
                    raise ValueError(f'Only use Components when retrieving Effects including flows. Got {element}')
                flows = [label.split('|')[0] for label in
                         self.components[element].inputs + self.components[element].outputs]
                for flow in flows:
                    label = f'{flow}->{target_effect}({mode})'
                    if label in self.solution:
                        share_exists = True
                        da = self.solution[label]
                        total = da * conversion_factor + total
        if not share_exists:
            total = xr.DataArray(np.nan)
        return total.rename(f'{element}->{effect}({mode})')

    def _create_effects_dataarray(self, mode: Literal['operation', 'invest', 'total'] = 'total') -> xr.DataArray:
        """Creates a dataarray containing effect totals for all components (including their flows).
        The dataarray does contain the direct as well as the indirect effects of each component.

        Args:
            mode: The calculation mode ('operation', 'invest', or 'total').

        Returns:
            An xarray DataArray with components and effects as dimensions.
        """
        # Create a list of DataArrays, one for each effect
        effect_arrays = []

        for effect in self.effects:
            # Create a list of DataArrays, one for each component
            component_arrays = [
                self._compute_effect_total(element=component, effect=effect, mode=mode, include_flows=True).expand_dims(
                    component=[component]
                )  # Add component dimension to each array
                for component in list(self.components)
            ]

            # Combine all components into one DataArray for this effect
            if component_arrays:
                effect_array = xr.concat(component_arrays, dim='component', coords='minimal')
                # Add effect dimension
                effect_array = effect_array.expand_dims(effect=[effect])
                effect_arrays.append(effect_array)

        # Combine all effects into one DataArray
        if effect_arrays:
            combined_array = xr.concat(effect_arrays, dim='effect', coords='minimal')
        else:
            # Create empty DataArray if no effects
            combined_array = xr.DataArray()

        # For now include a test to ensure correctness
        suffix = {
            'operation': '(operation)|total_per_timestep',
            'invest': '(invest)|total',
            'total': '|total',
        }
        for effect in self.effects:
            label = f'{effect}{suffix[mode]}'
            computed = combined_array.sel(effect=effect).sum('component')
            found = self.solution[label]
            if not np.allclose(computed.values, found.fillna(0).values):
                logger.critical(
                    f'Results for {effect}({mode}) in effects_dataarray doesnt match {label}\n{computed=}\n, {found=}'
                )

        return combined_array.rename(f'Effects ({mode})')

    def plot_heatmap(
        self,
        variable_name: str,
        heatmap_timeframes: Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'] = 'D',
        heatmap_timesteps_per_frame: Literal['W', 'D', 'h', '15min', 'min'] = 'h',
        color_map: str = 'portland',
        save: Union[bool, pathlib.Path] = False,
        show: bool = True,
        engine: plotting.PlottingEngine = 'plotly',
        scenario: Optional[Union[str, int]] = None,
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
            scenario: The scenario to plot. Defaults to the first scenario. Has no effect without scenarios present
        """
        dataarray = self.solution[variable_name]

        scenario_suffix = ''
        if 'scenario' in dataarray.indexes:
            chosen_scenario = scenario or self.scenarios[0]
            dataarray = dataarray.sel(scenario=chosen_scenario).drop_vars('scenario')
            scenario_suffix = f'--{chosen_scenario}'

        return plot_heatmap(
            dataarray=dataarray,
            name=f'{variable_name}{scenario_suffix}',
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
        if path is None:
            path = self.folder / f'{self.name}--network.html'
        return self.flow_system.plot_network(controls=controls, path=path, show=show)

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
        fx_io.save_dataset_to_netcdf(self.flow_system_data, paths.flow_system, compression=compression)

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
        return self._calculation_results.model.constraints[self._variable_names]

    def filter_solution(
        self,
        variable_dims: Optional[Literal['scalar', 'time', 'scenario', 'timeonly', 'scenarioonly']] = None,
        timesteps: Optional[pd.DatetimeIndex] = None,
        scenarios: Optional[pd.Index] = None,
        contains: Optional[Union[str, List[str]]] = None,
        startswith: Optional[Union[str, List[str]]] = None,
    ) -> xr.Dataset:
        """
        Filter the solution to a specific variable dimension and element.
        If no element is specified, all elements are included.

        Args:
            variable_dims: The dimension of which to get variables from.
                - 'scalar': Get scalar variables (without dimensions)
                - 'time': Get time-dependent variables (with a time dimension)
                - 'scenario': Get scenario-dependent variables (with ONLY a scenario dimension)
                - 'timeonly': Get time-dependent variables (with ONLY a time dimension)
                - 'scenarioonly': Get scenario-dependent variables (with ONLY a scenario dimension)
            timesteps: Optional time indexes to select. Can be:
                - pd.DatetimeIndex: Multiple timesteps
                - str/pd.Timestamp: Single timestep
                Defaults to all available timesteps.
            scenarios: Optional scenario indexes to select. Can be:
                - pd.Index: Multiple scenarios
                - str/int: Single scenario (int is treated as a label, not an index position)
                Defaults to all available scenarios.
            contains: Filter variables that contain this string or strings.
                If a list is provided, variables must contain ALL strings in the list.
            startswith: Filter variables that start with this string or strings.
                If a list is provided, variables must start with ANY of the strings in the list.
        """
        return filter_dataset(
            self.solution,
            variable_dims=variable_dims,
            timesteps=timesteps,
            scenarios=scenarios,
            contains=contains,
            startswith=startswith,
        )


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
        scenario: Optional[Union[str, int]] = None,
        mode: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        style: Literal['area', 'stacked_bar', 'line'] = 'stacked_bar',
        drop_suffix: bool = True,
    ) -> Union[plotly.graph_objs.Figure, Tuple[plt.Figure, plt.Axes]]:
        """
        Plots the node balance of the Component or Bus.
        Args:
            save: Whether to save the plot or not. If a path is provided, the plot will be saved at that location.
            show: Whether to show the plot or not.
            colors: The colors to use for the plot. See `flixopt.plotting.ColorType` for options.
            engine: The engine to use for plotting. Can be either 'plotly' or 'matplotlib'.
            scenario: The scenario to plot. Defaults to the first scenario. Has no effect without scenarios present
            mode: The mode to use for the dataset. Can be 'flow_rate' or 'flow_hours'.
                - 'flow_rate': Returns the flow_rates of the Node.
                - 'flow_hours': Returns the flow_hours of the Node. [flow_hours(t) = flow_rate(t) * dt(t)]. Renames suffixes to |flow_hours.
            drop_suffix: Whether to drop the suffix from the variable names.
        """
        ds = self.node_balance(with_last_timestep=True, mode=mode, drop_suffix=drop_suffix)

        title = f'{self.label} (flow rates)' if mode == 'flow_rate' else f'{self.label} (flow hours)'

        if 'scenario' in ds.indexes:
            chosen_scenario = scenario or self._calculation_results.scenarios[0]
            ds = ds.sel(scenario=chosen_scenario).drop_vars('scenario')
            title = f'{title} - {chosen_scenario}'

        if engine == 'plotly':
            figure_like = plotting.with_plotly(
                ds.to_dataframe(),
                colors=colors,
                style=style,
                title=title,
            )
            default_filetype = '.html'
        elif engine == 'matplotlib':
            figure_like = plotting.with_matplotlib(
                ds.to_dataframe(),
                colors=colors,
                style=style,
                title=title,
            )
            default_filetype = '.png'
        else:
            raise ValueError(f'Engine "{engine}" not supported. Use "plotly" or "matplotlib"')

        return plotting.export_figure(
            figure_like=figure_like,
            default_path=self._calculation_results.folder / title,
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
        scenario: Optional[Union[str, int]] = None,
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
            scenario: If scenarios are present: The scenario to plot. If None, the first scenario is used.
            drop_suffix: Whether to drop the suffix from the variable names.
        """
        inputs = sanitize_dataset(
            ds=self.solution[self.inputs] * self._calculation_results.hours_per_timestep,
            threshold=1e-5,
            drop_small_vars=True,
            zero_small_values=True,
            drop_suffix='|',
        )
        outputs = sanitize_dataset(
            ds=self.solution[self.outputs] * self._calculation_results.hours_per_timestep,
            threshold=1e-5,
            drop_small_vars=True,
            zero_small_values=True,
            drop_suffix='|',
        )
        inputs = inputs.sum('time')
        outputs = outputs.sum('time')

        title = f'{self.label} (total flow hours)'

        if 'scenario' in inputs.indexes:
            chosen_scenario = scenario or self._calculation_results.scenarios[0]
            inputs = inputs.sel(scenario=chosen_scenario).drop_vars('scenario')
            outputs = outputs.sel(scenario=chosen_scenario).drop_vars('scenario')
            title = f'{title} - {chosen_scenario}'

        if engine == 'plotly':
            figure_like = plotting.dual_pie_with_plotly(
                data_left=inputs.to_pandas(),
                data_right=outputs.to_pandas(),
                colors=colors,
                title=title,
                text_info=text_info,
                subtitles=('Inputs', 'Outputs'),
                legend_title='Flows',
                lower_percentage_group=lower_percentage_group,
            )
            default_filetype = '.html'
        elif engine == 'matplotlib':
            logger.debug('Parameter text_info is not supported for matplotlib')
            figure_like = plotting.dual_pie_with_matplotlib(
                data_left=inputs.to_pandas(),
                data_right=outputs.to_pandas(),
                colors=colors,
                title=title,
                subtitles=('Inputs', 'Outputs'),
                legend_title='Flows',
                lower_percentage_group=lower_percentage_group,
            )
            default_filetype = '.png'
        else:
            raise ValueError(f'Engine "{engine}" not supported. Use "plotly" or "matplotlib"')

        return plotting.export_figure(
            figure_like=figure_like,
            default_path=self._calculation_results.folder / title,
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
        mode: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        drop_suffix: bool = False,
    ) -> xr.Dataset:
        """
        Returns a dataset with the node balance of the Component or Bus.
        Args:
            negate_inputs: Whether to negate the input flow_rates of the Node.
            negate_outputs: Whether to negate the output flow_rates of the Node.
            threshold: The threshold for small values. Variables with all values below the threshold are dropped.
            with_last_timestep: Whether to include the last timestep in the dataset.
            mode: The mode to use for the dataset. Can be 'flow_rate' or 'flow_hours'.
                - 'flow_rate': Returns the flow_rates of the Node.
                - 'flow_hours': Returns the flow_hours of the Node. [flow_hours(t) = flow_rate(t) * dt(t)]. Renames suffixes to |flow_hours.
            drop_suffix: Whether to drop the suffix from the variable names.
        """
        ds = self.solution[self.inputs + self.outputs]

        ds = sanitize_dataset(
            ds=ds,
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
            drop_suffix='|' if drop_suffix else None,
        )

        if mode == 'flow_hours':
            ds = ds * self._calculation_results.hours_per_timestep
            ds = ds.rename_vars({var: var.replace('flow_rate', 'flow_hours') for var in ds.data_vars})

        return ds


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
        style: Literal['area', 'stacked_bar', 'line'] = 'stacked_bar',
        scenario: Optional[Union[str, int]] = None,
    ) -> plotly.graph_objs.Figure:
        """
        Plots the charge state of a Storage.
        Args:
            save: Whether to save the plot or not. If a path is provided, the plot will be saved at that location.
            show: Whether to show the plot or not.
            colors: The c
            engine: Plotting engine to use. Only 'plotly' is implemented atm.
            style: The plotting mode for the flow_rate
            scenario: The scenario to plot. Defaults to the first scenario. Has no effect without scenarios present

        Raises:
            ValueError: If the Component is not a Storage.
        """
        if not self.is_storage:
            raise ValueError(f'Cant plot charge_state. "{self.label}" is not a storage')

        ds = self.node_balance(with_last_timestep=True)
        charge_state = self.charge_state

        scenario_suffix = ''
        if 'scenario' in ds.indexes:
            chosen_scenario = scenario or self._calculation_results.scenarios[0]
            ds = ds.sel(scenario=chosen_scenario).drop_vars('scenario')
            charge_state = charge_state.sel(scenario=chosen_scenario).drop_vars('scenario')
            scenario_suffix = f'--{chosen_scenario}'
        if engine == 'plotly':
            fig = plotting.with_plotly(
                ds.to_dataframe(),
                colors=colors,
                style=style,
                title=f'Operation Balance of {self.label}{scenario_suffix}',
            )

            # TODO: Use colors for charge state?

            charge_state = charge_state.to_dataframe()
            fig.add_trace(
                plotly.graph_objs.Scatter(
                    x=charge_state.index, y=charge_state.values.flatten(), mode='lines', name=self._charge_state
                )
            )
        elif engine=='matplotlib':
            fig, ax = plotting.with_matplotlib(
                ds.to_dataframe(),
                colors=colors,
                style=style,
                title=f'Operation Balance of {self.label}{scenario_suffix}',
            )

            charge_state = charge_state.to_dataframe()
            ax.plot(charge_state.index, charge_state.values.flatten(), label=self._charge_state)
            fig.tight_layout()
            fig = fig, ax

        return plotting.export_figure(
            fig,
            default_path=self._calculation_results.folder / f'{self.label} (charge state){scenario_suffix}',
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
    drop_suffix: Optional[str] = None,
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
        drop_suffix: Drop suffix of data var names. Split by the provided str.

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

    if drop_suffix is not None:
        if not isinstance(drop_suffix, str):
            raise ValueError(f'Only pass str values to drop suffixes. Got {drop_suffix}')
        unique_dict = {}
        for var in ds.data_vars:
            new_name = var.split(drop_suffix)[0]

            # If name already exists, keep original name
            if new_name in unique_dict.values():
                unique_dict[var] = var
            else:
                unique_dict[var] = new_name
        ds = ds.rename(unique_dict)

    return ds


def filter_dataset(
    ds: xr.Dataset,
    variable_dims: Optional[Literal['scalar', 'time', 'scenario', 'timeonly', 'scenarioonly']] = None,
    timesteps: Optional[Union[pd.DatetimeIndex, str, pd.Timestamp]] = None,
    scenarios: Optional[Union[pd.Index, str, int]] = None,
    contains: Optional[Union[str, List[str]]] = None,
    startswith: Optional[Union[str, List[str]]] = None,
) -> xr.Dataset:
    """
    Filters a dataset by its dimensions, indexes, and with string filters for variable names.

    Args:
        ds: The dataset to filter.
        variable_dims: The dimension of which to get variables from.
            - 'scalar': Get scalar variables (without dimensions)
            - 'time': Get time-dependent variables (with a time dimension)
            - 'scenario': Get scenario-dependent variables (with ONLY a scenario dimension)
            - 'timeonly': Get time-dependent variables (with ONLY a time dimension)
            - 'scenarioonly': Get scenario-dependent variables (with ONLY a scenario dimension)
        timesteps: Optional time indexes to select. Can be:
            - pd.DatetimeIndex: Multiple timesteps
            - str/pd.Timestamp: Single timestep
            Defaults to all available timesteps.
        scenarios: Optional scenario indexes to select. Can be:
            - pd.Index: Multiple scenarios
            - str/int: Single scenario (int is treated as a label, not an index position)
            Defaults to all available scenarios.
        contains: Filter variables that contain this string or strings.
            If a list is provided, variables must contain ALL strings in the list.
        startswith: Filter variables that start with this string or strings.
            If a list is provided, variables must start with ANY of the strings in the list.

    Returns:
        Filtered dataset with specified variables and indexes.
    """
    # First filter by dimensions
    filtered_ds = ds.copy()
    if variable_dims is not None:
        if variable_dims == 'scalar':
            filtered_ds = filtered_ds[[v for v in filtered_ds.data_vars if not filtered_ds[v].dims]]
        elif variable_dims == 'time':
            filtered_ds = filtered_ds[[v for v in filtered_ds.data_vars if 'time' in filtered_ds[v].dims]]
        elif variable_dims == 'scenario':
            filtered_ds = filtered_ds[[v for v in filtered_ds.data_vars if 'scenario' in filtered_ds[v].dims]]
        elif variable_dims == 'timeonly':
            filtered_ds = filtered_ds[[v for v in filtered_ds.data_vars if filtered_ds[v].dims == ('time',)]]
        elif variable_dims == 'scenarioonly':
            filtered_ds = filtered_ds[[v for v in filtered_ds.data_vars if filtered_ds[v].dims == ('scenario',)]]
        else:
            raise ValueError(f'Unknown variable_dims "{variable_dims}" for filter_dataset')

    # Filter by 'contains' parameter
    if contains is not None:
        if isinstance(contains, str):
            # Single string - keep variables that contain this string
            filtered_ds = filtered_ds[[v for v in filtered_ds.data_vars if contains in v]]
        elif isinstance(contains, list) and all(isinstance(s, str) for s in contains):
            # List of strings - keep variables that contain ALL strings in the list
            filtered_ds = filtered_ds[[v for v in filtered_ds.data_vars if all(s in v for s in contains)]]
        else:
            raise TypeError(f"'contains' must be a string or list of strings, got {type(contains)}")

    # Filter by 'startswith' parameter
    if startswith is not None:
        if isinstance(startswith, str):
            # Single string - keep variables that start with this string
            filtered_ds = filtered_ds[[v for v in filtered_ds.data_vars if v.startswith(startswith)]]
        elif isinstance(startswith, list) and all(isinstance(s, str) for s in startswith):
            # List of strings - keep variables that start with ANY of the strings in the list
            filtered_ds = filtered_ds[[v for v in filtered_ds.data_vars if any(v.startswith(s) for s in startswith)]]
        else:
            raise TypeError(f"'startswith' must be a string or list of strings, got {type(startswith)}")

    # Handle time selection if needed
    if timesteps is not None and 'time' in filtered_ds.dims:
        try:
            filtered_ds = filtered_ds.sel(time=timesteps)
        except KeyError as e:
            available_times = set(filtered_ds.indexes['time'])
            requested_times = set([timesteps]) if not isinstance(timesteps, pd.Index) else set(timesteps)
            missing_times = requested_times - available_times
            raise ValueError(
                f'Timesteps not found in dataset: {missing_times}. Available times: {available_times}'
            ) from e

    # Handle scenario selection if needed
    if scenarios is not None and 'scenario' in filtered_ds.dims:
        try:
            filtered_ds = filtered_ds.sel(scenario=scenarios)
        except KeyError as e:
            available_scenarios = set(filtered_ds.indexes['scenario'])
            requested_scenarios = set([scenarios]) if not isinstance(scenarios, pd.Index) else set(scenarios)
            missing_scenarios = requested_scenarios - available_scenarios
            raise ValueError(
                f'Scenarios not found in dataset: {missing_scenarios}. Available scenarios: {available_scenarios}'
            ) from e

    return filtered_ds


def filter_edges_dataset(
    da: xr.DataArray, start: Optional[Union[str, List[str]]] = None, end: Optional[Union[str, List[str]]] = None
) -> xr.DataArray:
    """Filter the edges/flows in a DataArray by start and/or end nodes.

    Args:
        da: DataArray containing flow data with 'start' and 'end' coordinates
            along the 'edge' dimension.
        start: Optional source node(s) to filter by. Can be a single node name
            or a list of names. If None, no filtering is applied on source nodes.
        end: Optional destination node(s) to filter by. Can be a single node name
            or a list of names. If None, no filtering is applied on destination nodes.

    Returns:
        Filtered DataArray containing only the edges that match the specified
        start and/or end nodes.

    Raises:
        ValueError: If specified start or end nodes don't exist in the DataArray,
            or if no edges match the specified criteria.
        AttributeError: If the DataArray doesn't have 'start' or 'end' coordinates.
    """
    # Check if the DataArray has the required coordinates
    required_coords = []
    if start is not None:
        required_coords.append('start')
    if end is not None:
        required_coords.append('end')

    for coord in required_coords:
        if coord not in da.coords:
            raise AttributeError(f"DataArray is missing required coordinate '{coord}'")

    filtered_da = da

    # Check if start nodes exist
    if start is not None:
        start_values = [start] if isinstance(start, str) else start
        available_starts = set(da.start.values)
        missing_starts = [s for s in start_values if s not in available_starts]
        if missing_starts:
            raise ValueError(f'Start node(s) not found in the network: {missing_starts}')

        if isinstance(start, list):
            filtered_da = filtered_da.where(filtered_da.start.isin(start), drop=True)
        else:
            filtered_da = filtered_da.where(filtered_da.start == start, drop=True)

    # Check if end nodes exist
    if end is not None:
        end_values = [end] if isinstance(end, str) else end
        available_ends = set(da.end.values)
        missing_ends = [e for e in end_values if e not in available_ends]
        if missing_ends:
            raise ValueError(f'End node(s) not found in the network: {missing_ends}')

        if isinstance(end, list):
            filtered_da = filtered_da.where(filtered_da.end.isin(end), drop=True)
        else:
            filtered_da = filtered_da.where(filtered_da.end == end, drop=True)

    # Check if any data remains after filtering
    if filtered_da.size == 0:
        raise ValueError(f'No edges match the specified criteria (start={start}, end={end})')

    return filtered_da
