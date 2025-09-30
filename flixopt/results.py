from __future__ import annotations

import datetime
import json
import logging
import pathlib
import warnings
from typing import TYPE_CHECKING, Any, Literal

import linopy
import numpy as np
import pandas as pd
import plotly
import xarray as xr
import yaml

from . import io as fx_io
from . import plotting
from .flow_system import FlowSystem

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import pyvis

    from .calculation import Calculation, SegmentedCalculation
    from .core import FlowSystemDimensions


logger = logging.getLogger('flixopt')


class _FlowSystemRestorationError(Exception):
    """Exception raised when a FlowSystem cannot be restored from dataset."""

    pass


class CalculationResults:
    """Comprehensive container for optimization calculation results and analysis tools.

    This class provides unified access to all optimization results including flow rates,
    component states, bus balances, and system effects. It offers powerful analysis
    capabilities through filtering, plotting, and export functionality, making it
    the primary interface for post-processing optimization results.

    Key Features:
        **Unified Access**: Single interface to all solution variables and constraints
        **Element Results**: Direct access to component, bus, and effect-specific results
        **Visualization**: Built-in plotting methods for heatmaps, time series, and networks
        **Persistence**: Save/load functionality with compression for large datasets
        **Analysis Tools**: Filtering, aggregation, and statistical analysis methods

    Result Organization:
        - **Components**: Equipment-specific results (flows, states, constraints)
        - **Buses**: Network node balances and energy flows
        - **Effects**: System-wide impacts (costs, emissions, resource consumption)
        - **Solution**: Raw optimization variables and their values
        - **Metadata**: Calculation parameters, timing, and system configuration

    Attributes:
        solution: Dataset containing all optimization variable solutions
        flow_system_data: Dataset with complete system configuration and parameters. Restore the used FlowSystem for further analysis.
        summary: Calculation metadata including solver status, timing, and statistics
        name: Unique identifier for this calculation
        model: Original linopy optimization model (if available)
        folder: Directory path for result storage and loading
        components: Dictionary mapping component labels to ComponentResults objects
        buses: Dictionary mapping bus labels to BusResults objects
        effects: Dictionary mapping effect names to EffectResults objects
        timesteps_extra: Extended time index including boundary conditions
        hours_per_timestep: Duration of each timestep for proper energy calculations

    Examples:
        Load and analyze saved results:

        ```python
        # Load results from file
        results = CalculationResults.from_file('results', 'annual_optimization')

        # Access specific component results
        boiler_results = results['Boiler_01']
        heat_pump_results = results['HeatPump_02']

        # Plot component flow rates
        results.plot_heatmap('Boiler_01(Natural_Gas)|flow_rate')
        results['Boiler_01'].plot_node_balance()

        # Access raw solution dataarrays
        electricity_flows = results.solution[['Generator_01(Grid)|flow_rate', 'HeatPump_02(Grid)|flow_rate']]

        # Filter and analyze results
        peak_demand_hours = results.filter_solution(variable_dims='time')
        costs_solution = results.effects['cost'].solution
        ```

        Advanced filtering and aggregation:

        ```python
        # Filter by variable type
        scalar_results = results.filter_solution(variable_dims='scalar')
        time_series = results.filter_solution(variable_dims='time')

        # Custom data analysis leveraging xarray
        peak_power = results.solution['Generator_01(Grid)|flow_rate'].max()
        avg_efficiency = (
            results.solution['HeatPump(Heat)|flow_rate'] / results.solution['HeatPump(Electricity)|flow_rate']
        ).mean()
        ```

    Design Patterns:
        **Factory Methods**: Use `from_file()` and `from_calculation()` for creation or access directly from `Calculation.results`
        **Dictionary Access**: Use `results[element_label]` for element-specific results
        **Lazy Loading**: Results objects created on-demand for memory efficiency
        **Unified Interface**: Consistent API across different result types

    """

    @classmethod
    def from_file(cls, folder: str | pathlib.Path, name: str) -> CalculationResults:
        """Load CalculationResults from saved files.

        Args:
            folder: Directory containing saved files.
            name: Base name of saved files (without extensions).

        Returns:
            CalculationResults: Loaded instance.
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

        with open(paths.summary, encoding='utf-8') as f:
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
    def from_calculation(cls, calculation: Calculation) -> CalculationResults:
        """Create CalculationResults from a Calculation object.

        Args:
            calculation: Calculation object with solved model.

        Returns:
            CalculationResults: New instance with extracted results.
        """
        return cls(
            solution=calculation.model.solution,
            flow_system_data=calculation.flow_system.to_dataset(),
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
        summary: dict,
        folder: pathlib.Path | None = None,
        model: linopy.Model | None = None,
        **kwargs,  # To accept old "flow_system" parameter
    ):
        """Initialize CalculationResults with optimization data.
        Usually, this class is instantiated by the Calculation class, or by loading from file.

        Args:
            solution: Optimization solution dataset.
            flow_system_data: Flow system configuration dataset.
            name: Calculation name.
            summary: Calculation metadata.
            folder: Results storage folder.
            model: Linopy optimization model.
        Deprecated:
            flow_system: Use flow_system_data instead.
        """
        # Handle potential old "flow_system" parameter for backward compatibility
        if 'flow_system' in kwargs and flow_system_data is None:
            flow_system_data = kwargs.pop('flow_system')
            warnings.warn(
                "The 'flow_system' parameter is deprecated. Use 'flow_system_data' instead."
                "Acess is now by '.flow_system_data', while '.flow_system' returns the restored FlowSystem.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.solution = solution
        self.flow_system_data = flow_system_data
        self.summary = summary
        self.name = name
        self.model = model
        self.folder = pathlib.Path(folder) if folder is not None else pathlib.Path.cwd() / 'results'
        self.components = {
            label: ComponentResults(self, **infos) for label, infos in self.solution.attrs['Components'].items()
        }

        self.buses = {label: BusResults(self, **infos) for label, infos in self.solution.attrs['Buses'].items()}

        self.effects = {label: EffectResults(self, **infos) for label, infos in self.solution.attrs['Effects'].items()}

        if 'Flows' not in self.solution.attrs:
            warnings.warn(
                'No Data about flows found in the results. This data is only included since v2.2.0. Some functionality '
                'is not availlable. We recommend to evaluate your results with a version <2.2.0.',
                stacklevel=2,
            )
            self.flows = {}
        else:
            self.flows = {
                label: FlowResults(self, **infos) for label, infos in self.solution.attrs.get('Flows', {}).items()
            }

        self.timesteps_extra = self.solution.indexes['time']
        self.hours_per_timestep = FlowSystem.calculate_hours_per_timestep(self.timesteps_extra)
        self.scenarios = self.solution.indexes['scenario'] if 'scenario' in self.solution.indexes else None

        self._effect_share_factors = None
        self._flow_system = None

        self._flow_rates = None
        self._flow_hours = None
        self._sizes = None
        self._effects_per_component = None

    def __getitem__(self, key: str) -> ComponentResults | BusResults | EffectResults:
        if key in self.components:
            return self.components[key]
        if key in self.buses:
            return self.buses[key]
        if key in self.effects:
            return self.effects[key]
        if key in self.flows:
            return self.flows[key]
        raise KeyError(f'No element with label {key} found.')

    @property
    def storages(self) -> list[ComponentResults]:
        """Get all storage components in the results."""
        return [comp for comp in self.components.values() if comp.is_storage]

    @property
    def objective(self) -> float:
        """Get optimization objective value."""
        # Deprecated. Fallback
        if 'objective' not in self.solution:
            logger.warning('Objective not found in solution. Fallback to summary (rounded value). This is deprecated')
            return self.summary['Main Results']['Objective']

        return self.solution['objective'].item()

    @property
    def variables(self) -> linopy.Variables:
        """Get optimization variables (requires linopy model)."""
        if self.model is None:
            raise ValueError('The linopy model is not available.')
        return self.model.variables

    @property
    def constraints(self) -> linopy.Constraints:
        """Get optimization constraints (requires linopy model)."""
        if self.model is None:
            raise ValueError('The linopy model is not available.')
        return self.model.constraints

    @property
    def effect_share_factors(self):
        if self._effect_share_factors is None:
            effect_share_factors = self.flow_system.effects.calculate_effect_share_factors()
            self._effect_share_factors = {'temporal': effect_share_factors[0], 'periodic': effect_share_factors[1]}
        return self._effect_share_factors

    @property
    def flow_system(self) -> FlowSystem:
        """The restored flow_system that was used to create the calculation.
        Contains all input parameters."""
        if self._flow_system is None:
            old_level = logger.level
            logger.level = logging.CRITICAL
            try:
                self._flow_system = FlowSystem.from_dataset(self.flow_system_data)
                self._flow_system._connect_network()
            except Exception as e:
                logger.critical(
                    f'Not able to restore FlowSystem from dataset. Some functionality is not availlable. {e}'
                )
                raise _FlowSystemRestorationError(f'Not able to restore FlowSystem from dataset. {e}') from e
            finally:
                logger.level = old_level
        return self._flow_system

    def filter_solution(
        self,
        variable_dims: Literal['scalar', 'time', 'scenario', 'timeonly', 'scenarioonly'] | None = None,
        element: str | None = None,
        timesteps: pd.DatetimeIndex | None = None,
        scenarios: pd.Index | None = None,
        contains: str | list[str] | None = None,
        startswith: str | list[str] | None = None,
    ) -> xr.Dataset:
        """Filter solution by variable dimension and/or element.

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

    @property
    def effects_per_component(self) -> xr.Dataset:
        """Returns a dataset containing effect results for each mode, aggregated by Component

        Returns:
            An xarray Dataset with an additional component dimension and effects as variables.
        """
        if self._effects_per_component is None:
            self._effects_per_component = xr.Dataset(
                {
                    mode: self._create_effects_dataset(mode).to_dataarray('effect', name=mode)
                    for mode in ['temporal', 'periodic', 'total']
                }
            )
            dim_order = ['time', 'period', 'scenario', 'component', 'effect']
            self._effects_per_component = self._effects_per_component.transpose(*dim_order, missing_dims='ignore')

        return self._effects_per_component

    def flow_rates(
        self,
        start: str | list[str] | None = None,
        end: str | list[str] | None = None,
        component: str | list[str] | None = None,
    ) -> xr.DataArray:
        """Returns a DataArray containing the flow rates of each Flow.

        Args:
            start: Optional source node(s) to filter by. Can be a single node name or a list of names.
            end: Optional destination node(s) to filter by. Can be a single node name or a list of names.
            component: Optional component(s) to filter by. Can be a single component name or a list of names.

        Further usage:
            Convert the dataarray to a dataframe:
            >>>results.flow_rates().to_pandas()
            Get the max or min over time:
            >>>results.flow_rates().max('time')
            Sum up the flow rates of flows with the same start and end:
            >>>results.flow_rates(end='Fernwärme').groupby('start').sum(dim='flow')
            To recombine filtered dataarrays, use `xr.concat` with dim 'flow':
            >>>xr.concat([results.flow_rates(start='Fernwärme'), results.flow_rates(end='Fernwärme')], dim='flow')
        """
        if self._flow_rates is None:
            self._flow_rates = self._assign_flow_coords(
                xr.concat(
                    [flow.flow_rate.rename(flow.label) for flow in self.flows.values()],
                    dim=pd.Index(self.flows.keys(), name='flow'),
                )
            ).rename('flow_rates')
        filters = {k: v for k, v in {'start': start, 'end': end, 'component': component}.items() if v is not None}
        return filter_dataarray_by_coord(self._flow_rates, **filters)

    def flow_hours(
        self,
        start: str | list[str] | None = None,
        end: str | list[str] | None = None,
        component: str | list[str] | None = None,
    ) -> xr.DataArray:
        """Returns a DataArray containing the flow hours of each Flow.

        Flow hours represent the total energy/material transferred over time,
        calculated by multiplying flow rates by the duration of each timestep.

        Args:
            start: Optional source node(s) to filter by. Can be a single node name or a list of names.
            end: Optional destination node(s) to filter by. Can be a single node name or a list of names.
            component: Optional component(s) to filter by. Can be a single component name or a list of names.

        Further usage:
            Convert the dataarray to a dataframe:
            >>>results.flow_hours().to_pandas()
            Sum up the flow hours over time:
            >>>results.flow_hours().sum('time')
            Sum up the flow hours of flows with the same start and end:
            >>>results.flow_hours(end='Fernwärme').groupby('start').sum(dim='flow')
            To recombine filtered dataarrays, use `xr.concat` with dim 'flow':
            >>>xr.concat([results.flow_hours(start='Fernwärme'), results.flow_hours(end='Fernwärme')], dim='flow')

        """
        if self._flow_hours is None:
            self._flow_hours = (self.flow_rates() * self.hours_per_timestep).rename('flow_hours')
        filters = {k: v for k, v in {'start': start, 'end': end, 'component': component}.items() if v is not None}
        return filter_dataarray_by_coord(self._flow_hours, **filters)

    def sizes(
        self,
        start: str | list[str] | None = None,
        end: str | list[str] | None = None,
        component: str | list[str] | None = None,
    ) -> xr.DataArray:
        """Returns a dataset with the sizes of the Flows.
        Args:
            start: Optional source node(s) to filter by. Can be a single node name or a list of names.
            end: Optional destination node(s) to filter by. Can be a single node name or a list of names.
            component: Optional component(s) to filter by. Can be a single component name or a list of names.

        Further usage:
            Convert the dataarray to a dataframe:
            >>>results.sizes().to_pandas()
            To recombine filtered dataarrays, use `xr.concat` with dim 'flow':
            >>>xr.concat([results.sizes(start='Fernwärme'), results.sizes(end='Fernwärme')], dim='flow')

        """
        if self._sizes is None:
            self._sizes = self._assign_flow_coords(
                xr.concat(
                    [flow.size.rename(flow.label) for flow in self.flows.values()],
                    dim=pd.Index(self.flows.keys(), name='flow'),
                )
            ).rename('flow_sizes')
        filters = {k: v for k, v in {'start': start, 'end': end, 'component': component}.items() if v is not None}
        return filter_dataarray_by_coord(self._sizes, **filters)

    def _assign_flow_coords(self, da: xr.DataArray):
        # Add start and end coordinates
        da = da.assign_coords(
            {
                'start': ('flow', [flow.start for flow in self.flows.values()]),
                'end': ('flow', [flow.end for flow in self.flows.values()]),
                'component': ('flow', [flow.component for flow in self.flows.values()]),
            }
        )

        # Ensure flow is the last dimension if needed
        existing_dims = [d for d in da.dims if d != 'flow']
        da = da.transpose(*(existing_dims + ['flow']))
        return da

    def get_effect_shares(
        self,
        element: str,
        effect: str,
        mode: Literal['temporal', 'periodic'] | None = None,
        include_flows: bool = False,
    ) -> xr.Dataset:
        """Retrieves individual effect shares for a specific element and effect.
        Either for temporal, investment, or both modes combined.
        Only includes the direct shares.

        Args:
            element: The element identifier for which to retrieve effect shares.
            effect: The effect identifier for which to retrieve shares.
            mode: Optional. The mode to retrieve shares for. Can be 'temporal', 'periodic',
                or None to retrieve both. Defaults to None.

        Returns:
            An xarray Dataset containing the requested effect shares. If mode is None,
            returns a merged Dataset containing both temporal and investment shares.

        Raises:
            ValueError: If the specified effect is not available or if mode is invalid.
        """
        if effect not in self.effects:
            raise ValueError(f'Effect {effect} is not available.')

        if mode is None:
            return xr.merge(
                [
                    self.get_effect_shares(
                        element=element, effect=effect, mode='temporal', include_flows=include_flows
                    ),
                    self.get_effect_shares(
                        element=element, effect=effect, mode='periodic', include_flows=include_flows
                    ),
                ]
            )

        if mode not in ['temporal', 'periodic']:
            raise ValueError(f'Mode {mode} is not available. Choose between "temporal" and "periodic".')

        ds = xr.Dataset()

        label = f'{element}->{effect}({mode})'
        if label in self.solution:
            ds = xr.Dataset({label: self.solution[label]})

        if include_flows:
            if element not in self.components:
                raise ValueError(f'Only use Components when retrieving Effects including flows. Got {element}')
            flows = [
                label.split('|')[0] for label in self.components[element].inputs + self.components[element].outputs
            ]
            return xr.merge(
                [ds]
                + [
                    self.get_effect_shares(element=flow, effect=effect, mode=mode, include_flows=False)
                    for flow in flows
                ]
            )

        return ds

    def _compute_effect_total(
        self,
        element: str,
        effect: str,
        mode: Literal['temporal', 'periodic', 'total'] = 'total',
        include_flows: bool = False,
    ) -> xr.DataArray:
        """Calculates the total effect for a specific element and effect.

        This method computes the total direct and indirect effects for a given element
        and effect, considering the conversion factors between different effects.

        Args:
            element: The element identifier for which to calculate total effects.
            effect: The effect identifier to calculate.
            mode: The calculation mode. Options are:
                'temporal': Returns temporal effects.
                'periodic': Returns investment-specific effects.
                'total': Returns the sum of temporal effects and periodic effects. Defaults to 'total'.
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
            temporal = self._compute_effect_total(
                element=element, effect=effect, mode='temporal', include_flows=include_flows
            )
            periodic = self._compute_effect_total(
                element=element, effect=effect, mode='periodic', include_flows=include_flows
            )
            if periodic.isnull().all() and temporal.isnull().all():
                return xr.DataArray(np.nan)
            if temporal.isnull().all():
                return periodic.rename(f'{element}->{effect}')
            temporal = temporal.sum('time')
            if periodic.isnull().all():
                return temporal.rename(f'{element}->{effect}')
            if 'time' in temporal.indexes:
                temporal = temporal.sum('time')
            return periodic + temporal

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
                flows = [
                    label.split('|')[0] for label in self.components[element].inputs + self.components[element].outputs
                ]
                for flow in flows:
                    label = f'{flow}->{target_effect}({mode})'
                    if label in self.solution:
                        share_exists = True
                        da = self.solution[label]
                        total = da * conversion_factor + total
        if not share_exists:
            total = xr.DataArray(np.nan)
        return total.rename(f'{element}->{effect}({mode})')

    def _create_effects_dataset(self, mode: Literal['temporal', 'periodic', 'total']) -> xr.Dataset:
        """Creates a dataset containing effect totals for all components (including their flows).
        The dataset does contain the direct as well as the indirect effects of each component.

        Args:
            mode: The calculation mode ('temporal', 'periodic', or 'total').

        Returns:
            An xarray Dataset with components as dimension and effects as variables.
        """
        ds = xr.Dataset()
        all_arrays = {}
        template = None  # Template is needed to determine the dimensions of the arrays. This handles the case of no shares for an effect

        components_list = list(self.components)

        # First pass: collect arrays and find template
        for effect in self.effects:
            effect_arrays = []
            for component in components_list:
                da = self._compute_effect_total(element=component, effect=effect, mode=mode, include_flows=True)
                effect_arrays.append(da)

                if template is None and (da.dims or not da.isnull().all()):
                    template = da

            all_arrays[effect] = effect_arrays

        # Ensure we have a template
        if template is None:
            raise ValueError(
                f"No template with proper dimensions found for mode '{mode}'. "
                f'All computed arrays are scalars, which indicates a data issue.'
            )

        # Second pass: process all effects (guaranteed to include all)
        for effect in self.effects:
            dataarrays = all_arrays[effect]
            component_arrays = []

            for component, arr in zip(components_list, dataarrays, strict=False):
                # Expand scalar NaN arrays to match template dimensions
                if not arr.dims and np.isnan(arr.item()):
                    arr = xr.full_like(template, np.nan, dtype=float).rename(arr.name)

                component_arrays.append(arr.expand_dims(component=[component]))

            ds[effect] = xr.concat(component_arrays, dim='component', coords='minimal').rename(effect)

        # For now include a test to ensure correctness
        suffix = {
            'temporal': '(temporal)|per_timestep',
            'periodic': '(periodic)',
            'total': '',
        }
        for effect in self.effects:
            label = f'{effect}{suffix[mode]}'
            computed = ds[effect].sum('component')
            found = self.solution[label]
            if not np.allclose(computed.values, found.fillna(0).values):
                logger.critical(
                    f'Results for {effect}({mode}) in effects_dataset doesnt match {label}\n{computed=}\n, {found=}'
                )

        return ds

    def plot_heatmap(
        self,
        variable_name: str,
        heatmap_timeframes: Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'] = 'D',
        heatmap_timesteps_per_frame: Literal['W', 'D', 'h', '15min', 'min'] = 'h',
        color_map: str = 'portland',
        save: bool | pathlib.Path = False,
        show: bool = True,
        engine: plotting.PlottingEngine = 'plotly',
        indexer: dict[FlowSystemDimensions, Any] | None = None,
    ) -> plotly.graph_objs.Figure | tuple[plt.Figure, plt.Axes]:
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
            indexer: Optional selection dict, e.g., {'scenario': 'base', 'period': 2024}.
                 If None, uses first value for each dimension.
                 If empty dict {}, uses all values.

        Examples:
            Basic usage (uses first scenario, first period, all time):

            >>> results.plot_heatmap('Battery|charge_state')

            Select specific scenario and period:

            >>> results.plot_heatmap('Boiler(Qth)|flow_rate', indexer={'scenario': 'base', 'period': 2024})

            Time filtering (summer months only):

            >>> results.plot_heatmap(
            ...     'Boiler(Qth)|flow_rate',
            ...     indexer={
            ...         'scenario': 'base',
            ...         'time': results.solution.time[results.solution.time.dt.month.isin([6, 7, 8])],
            ...     },
            ... )

            Save to specific location:

            >>> results.plot_heatmap(
            ...     'Boiler(Qth)|flow_rate', indexer={'scenario': 'base'}, save='path/to/my_heatmap.html'
            ... )
        """
        dataarray = self.solution[variable_name]

        return plot_heatmap(
            dataarray=dataarray,
            name=variable_name,
            folder=self.folder,
            heatmap_timeframes=heatmap_timeframes,
            heatmap_timesteps_per_frame=heatmap_timesteps_per_frame,
            color_map=color_map,
            save=save,
            show=show,
            engine=engine,
            indexer=indexer,
        )

    def plot_network(
        self,
        controls: (
            bool
            | list[
                Literal['nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer']
            ]
        ) = True,
        path: pathlib.Path | None = None,
        show: bool = False,
    ) -> pyvis.network.Network | None:
        """Plot interactive network visualization of the system.

        Args:
            controls: Enable/disable interactive controls.
            path: Save path for network HTML.
            show: Whether to display the plot.
        """
        if path is None:
            path = self.folder / f'{self.name}--network.html'
        return self.flow_system.plot_network(controls=controls, path=path, show=show)

    def to_file(
        self,
        folder: str | pathlib.Path | None = None,
        name: str | None = None,
        compression: int = 5,
        document_model: bool = True,
        save_linopy_model: bool = False,
    ):
        """Save results to files.

        Args:
            folder: Save folder (defaults to calculation folder).
            name: File name (defaults to calculation name).
            compression: Compression level 0-9.
            document_model: Whether to document model formulations as yaml.
            save_linopy_model: Whether to save linopy model file.
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
                self.model.to_netcdf(paths.linopy_model, engine='h5netcdf')

        if document_model:
            if self.model is None:
                logger.critical('No model in the CalculationResults. Documenting the model is not possible.')
            else:
                fx_io.document_linopy_model(self.model, path=paths.model_documentation)

        logger.info(f'Saved calculation results "{name}" to {paths.model_documentation.parent}')


class _ElementResults:
    def __init__(
        self, calculation_results: CalculationResults, label: str, variables: list[str], constraints: list[str]
    ):
        self._calculation_results = calculation_results
        self.label = label
        self._variable_names = variables
        self._constraint_names = constraints

        self.solution = self._calculation_results.solution[self._variable_names]

    @property
    def variables(self) -> linopy.Variables:
        """Get element variables (requires linopy model).

        Raises:
            ValueError: If linopy model is unavailable.
        """
        if self._calculation_results.model is None:
            raise ValueError('The linopy model is not available.')
        return self._calculation_results.model.variables[self._variable_names]

    @property
    def constraints(self) -> linopy.Constraints:
        """Get element constraints (requires linopy model).

        Raises:
            ValueError: If linopy model is unavailable.
        """
        if self._calculation_results.model is None:
            raise ValueError('The linopy model is not available.')
        return self._calculation_results.model.constraints[self._constraint_names]

    def filter_solution(
        self,
        variable_dims: Literal['scalar', 'time', 'scenario', 'timeonly', 'scenarioonly'] | None = None,
        timesteps: pd.DatetimeIndex | None = None,
        scenarios: pd.Index | None = None,
        contains: str | list[str] | None = None,
        startswith: str | list[str] | None = None,
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
    def __init__(
        self,
        calculation_results: CalculationResults,
        label: str,
        variables: list[str],
        constraints: list[str],
        inputs: list[str],
        outputs: list[str],
        flows: list[str],
    ):
        super().__init__(calculation_results, label, variables, constraints)
        self.inputs = inputs
        self.outputs = outputs
        self.flows = flows

    def plot_node_balance(
        self,
        save: bool | pathlib.Path = False,
        show: bool = True,
        colors: plotting.ColorType = 'viridis',
        engine: plotting.PlottingEngine = 'plotly',
        indexer: dict[FlowSystemDimensions, Any] | None = None,
        mode: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        style: Literal['area', 'stacked_bar', 'line'] = 'stacked_bar',
        drop_suffix: bool = True,
    ) -> plotly.graph_objs.Figure | tuple[plt.Figure, plt.Axes]:
        """
        Plots the node balance of the Component or Bus.
        Args:
            save: Whether to save the plot or not. If a path is provided, the plot will be saved at that location.
            show: Whether to show the plot or not.
            colors: The colors to use for the plot. See `flixopt.plotting.ColorType` for options.
            engine: The engine to use for plotting. Can be either 'plotly' or 'matplotlib'.
            indexer: Optional selection dict, e.g., {'scenario': 'base', 'period': 2024}.
                 If None, uses first value for each dimension (except time).
                 If empty dict {}, uses all values.
            style: The style to use for the dataset. Can be 'flow_rate' or 'flow_hours'.
                - 'flow_rate': Returns the flow_rates of the Node.
                - 'flow_hours': Returns the flow_hours of the Node. [flow_hours(t) = flow_rate(t) * dt(t)]. Renames suffixes to |flow_hours.
            drop_suffix: Whether to drop the suffix from the variable names.
        """
        ds = self.node_balance(with_last_timestep=True, mode=mode, drop_suffix=drop_suffix, indexer=indexer)

        ds, suffix_parts = _apply_indexer_to_data(ds, indexer, drop=True)
        suffix = '--' + '-'.join(suffix_parts) if suffix_parts else ''

        title = f'{self.label} (flow rates){suffix}' if mode == 'flow_rate' else f'{self.label} (flow hours){suffix}'

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
        save: bool | pathlib.Path = False,
        show: bool = True,
        engine: plotting.PlottingEngine = 'plotly',
        indexer: dict[FlowSystemDimensions, Any] | None = None,
    ) -> plotly.graph_objs.Figure | tuple[plt.Figure, list[plt.Axes]]:
        """Plot pie chart of flow hours distribution.
        Args:
            lower_percentage_group: Percentage threshold for "Others" grouping.
            colors: Color scheme. Also see plotly.
            text_info: Information to display on pie slices.
            save: Whether to save plot.
            show: Whether to display plot.
            engine: Plotting engine ('plotly' or 'matplotlib').
            indexer: Optional selection dict, e.g., {'scenario': 'base', 'period': 2024}.
                 If None, uses first value for each dimension.
                 If empty dict {}, uses all values.
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

        inputs, suffix_parts = _apply_indexer_to_data(inputs, indexer, drop=True)
        outputs, suffix_parts = _apply_indexer_to_data(outputs, indexer, drop=True)
        suffix = '--' + '-'.join(suffix_parts) if suffix_parts else ''

        title = f'{self.label} (total flow hours){suffix}'

        inputs = inputs.sum('time')
        outputs = outputs.sum('time')

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
        threshold: float | None = 1e-5,
        with_last_timestep: bool = False,
        mode: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        drop_suffix: bool = False,
        indexer: dict[FlowSystemDimensions, Any] | None = None,
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
            indexer: Optional selection dict, e.g., {'scenario': 'base', 'period': 2024}.
                 If None, uses first value for each dimension.
                 If empty dict {}, uses all values.
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

        ds, _ = _apply_indexer_to_data(ds, indexer, drop=True)

        if mode == 'flow_hours':
            ds = ds * self._calculation_results.hours_per_timestep
            ds = ds.rename_vars({var: var.replace('flow_rate', 'flow_hours') for var in ds.data_vars})

        return ds


class BusResults(_NodeResults):
    """Results container for energy/material balance nodes in the system."""


class ComponentResults(_NodeResults):
    """Results container for individual system components with specialized analysis tools."""

    @property
    def is_storage(self) -> bool:
        return self._charge_state in self._variable_names

    @property
    def _charge_state(self) -> str:
        return f'{self.label}|charge_state'

    @property
    def charge_state(self) -> xr.DataArray:
        """Get storage charge state solution."""
        if not self.is_storage:
            raise ValueError(f'Cant get charge_state. "{self.label}" is not a storage')
        return self.solution[self._charge_state]

    def plot_charge_state(
        self,
        save: bool | pathlib.Path = False,
        show: bool = True,
        colors: plotting.ColorType = 'viridis',
        engine: plotting.PlottingEngine = 'plotly',
        style: Literal['area', 'stacked_bar', 'line'] = 'stacked_bar',
        indexer: dict[FlowSystemDimensions, Any] | None = None,
    ) -> plotly.graph_objs.Figure:
        """Plot storage charge state over time, combined with the node balance.

        Args:
            save: Whether to save the plot or not. If a path is provided, the plot will be saved at that location.
            show: Whether to show the plot or not.
            colors: Color scheme. Also see plotly.
            engine: Plotting engine to use. Only 'plotly' is implemented atm.
            style: The colors to use for the plot. See `flixopt.plotting.ColorType` for options.
            indexer: Optional selection dict, e.g., {'scenario': 'base', 'period': 2024}.
                 If None, uses first value for each dimension.
                 If empty dict {}, uses all values.

        Raises:
            ValueError: If component is not a storage.
        """
        if not self.is_storage:
            raise ValueError(f'Cant plot charge_state. "{self.label}" is not a storage')

        ds = self.node_balance(with_last_timestep=True, indexer=indexer)
        charge_state = self.charge_state

        ds, suffix_parts = _apply_indexer_to_data(ds, indexer, drop=True)
        charge_state, suffix_parts = _apply_indexer_to_data(charge_state, indexer, drop=True)
        suffix = '--' + '-'.join(suffix_parts) if suffix_parts else ''

        title = f'Operation Balance of {self.label}{suffix}'

        if engine == 'plotly':
            fig = plotting.with_plotly(
                ds.to_dataframe(),
                colors=colors,
                style=style,
                title=title,
            )

            # TODO: Use colors for charge state?

            charge_state = charge_state.to_dataframe()
            fig.add_trace(
                plotly.graph_objs.Scatter(
                    x=charge_state.index, y=charge_state.values.flatten(), mode='lines', name=self._charge_state
                )
            )
        elif engine == 'matplotlib':
            fig, ax = plotting.with_matplotlib(
                ds.to_dataframe(),
                colors=colors,
                style=style,
                title=title,
            )

            charge_state = charge_state.to_dataframe()
            ax.plot(charge_state.index, charge_state.values.flatten(), label=self._charge_state)
            fig.tight_layout()
            fig = fig, ax

        return plotting.export_figure(
            fig,
            default_path=self._calculation_results.folder / title,
            default_filetype='.html',
            user_path=None if isinstance(save, bool) else pathlib.Path(save),
            show=show,
            save=True if save else False,
        )

    def node_balance_with_charge_state(
        self, negate_inputs: bool = True, negate_outputs: bool = False, threshold: float | None = 1e-5
    ) -> xr.Dataset:
        """Get storage node balance including charge state.

        Args:
            negate_inputs: Whether to negate input flows.
            negate_outputs: Whether to negate output flows.
            threshold: Threshold for small values.

        Returns:
            xr.Dataset: Node balance with charge state.

        Raises:
            ValueError: If component is not a storage.
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
        """Get effect shares from specific element.

        Args:
            element: Element label to get shares from.

        Returns:
            xr.Dataset: Element shares to this effect.
        """
        return self.solution[[name for name in self._variable_names if name.startswith(f'{element}->')]]


class FlowResults(_ElementResults):
    def __init__(
        self,
        calculation_results: CalculationResults,
        label: str,
        variables: list[str],
        constraints: list[str],
        start: str,
        end: str,
        component: str,
    ):
        super().__init__(calculation_results, label, variables, constraints)
        self.start = start
        self.end = end
        self.component = component

    @property
    def flow_rate(self) -> xr.DataArray:
        return self.solution[f'{self.label}|flow_rate']

    @property
    def flow_hours(self) -> xr.DataArray:
        return (self.flow_rate * self._calculation_results.hours_per_timestep).rename(f'{self.label}|flow_hours')

    @property
    def size(self) -> xr.DataArray:
        name = f'{self.label}|size'
        if name in self.solution:
            return self.solution[name]
        try:
            return self._calculation_results.flow_system.flows[self.label].size.rename(name)
        except _FlowSystemRestorationError:
            logger.critical(f'Size of flow {self.label}.size not availlable. Returning NaN')
            return xr.DataArray(np.nan).rename(name)


class SegmentedCalculationResults:
    """Results container for segmented optimization calculations with temporal decomposition.

    This class manages results from SegmentedCalculation runs where large optimization
    problems are solved by dividing the time horizon into smaller, overlapping segments.
    It provides unified access to results across all segments while maintaining the
    ability to analyze individual segment behavior.

    Key Features:
        **Unified Time Series**: Automatically assembles results from all segments into
        continuous time series, removing overlaps and boundary effects
        **Segment Analysis**: Access individual segment results for debugging and validation
        **Consistency Checks**: Verify solution continuity at segment boundaries
        **Memory Efficiency**: Handles large datasets that exceed single-segment memory limits

    Temporal Handling:
        The class manages the complex task of combining overlapping segment solutions
        into coherent time series, ensuring proper treatment of:
        - Storage state continuity between segments
        - Flow rate transitions at segment boundaries
        - Aggregated results over the full time horizon

    Examples:
        Load and analyze segmented results:

        ```python
        # Load segmented calculation results
        results = SegmentedCalculationResults.from_file('results', 'annual_segmented')

        # Access unified results across all segments
        full_timeline = results.all_timesteps
        total_segments = len(results.segment_results)

        # Analyze individual segments
        for i, segment in enumerate(results.segment_results):
            print(f'Segment {i + 1}: {len(segment.solution.time)} timesteps')
            segment_costs = segment.effects['cost'].total_value

        # Check solution continuity at boundaries
        segment_boundaries = results.get_boundary_analysis()
        max_discontinuity = segment_boundaries['max_storage_jump']
        ```

        Create from segmented calculation:

        ```python
        # After running segmented calculation
        segmented_calc = SegmentedCalculation(
            name='annual_system',
            flow_system=system,
            timesteps_per_segment=730,  # Monthly segments
            overlap_timesteps=48,  # 2-day overlap
        )
        segmented_calc.do_modeling_and_solve(solver='gurobi')

        # Extract unified results
        results = SegmentedCalculationResults.from_calculation(segmented_calc)

        # Save combined results
        results.to_file(compression=5)
        ```

        Performance analysis across segments:

        ```python
        # Compare segment solve times
        solve_times = [seg.summary['durations']['solving'] for seg in results.segment_results]
        avg_solve_time = sum(solve_times) / len(solve_times)

        # Verify solution quality consistency
        segment_objectives = [seg.summary['objective_value'] for seg in results.segment_results]

        # Storage continuity analysis
        if 'Battery' in results.segment_results[0].components:
            storage_continuity = results.check_storage_continuity('Battery')
        ```

    Design Considerations:
        **Boundary Effects**: Monitor solution quality at segment interfaces where
        foresight is limited compared to full-horizon optimization.

        **Memory Management**: Individual segment results are maintained for detailed
        analysis while providing unified access for system-wide metrics.

        **Validation Tools**: Built-in methods to verify temporal consistency and
        identify potential issues from segmentation approach.

    Common Use Cases:
        - **Large-Scale Analysis**: Annual or multi-period optimization results
        - **Memory-Constrained Systems**: Results from systems exceeding hardware limits
        - **Segment Validation**: Verifying segmentation approach effectiveness
        - **Performance Monitoring**: Comparing segmented vs. full-horizon solutions
        - **Debugging**: Identifying issues specific to temporal decomposition

    """

    @classmethod
    def from_calculation(cls, calculation: SegmentedCalculation):
        return cls(
            [calc.results for calc in calculation.sub_calculations],
            all_timesteps=calculation.all_timesteps,
            timesteps_per_segment=calculation.timesteps_per_segment,
            overlap_timesteps=calculation.overlap_timesteps,
            name=calculation.name,
            folder=calculation.folder,
        )

    @classmethod
    def from_file(cls, folder: str | pathlib.Path, name: str):
        """Load SegmentedCalculationResults from saved files.

        Args:
            folder: Directory containing saved files.
            name: Base name of saved files.

        Returns:
            SegmentedCalculationResults: Loaded instance.
        """
        folder = pathlib.Path(folder)
        path = folder / name
        logger.info(f'loading calculation "{name}" from file ("{path.with_suffix(".nc4")}")')
        with open(path.with_suffix('.json'), encoding='utf-8') as f:
            meta_data = json.load(f)
        return cls(
            [CalculationResults.from_file(folder, sub_name) for sub_name in meta_data['sub_calculations']],
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
        segment_results: list[CalculationResults],
        all_timesteps: pd.DatetimeIndex,
        timesteps_per_segment: int,
        overlap_timesteps: int,
        name: str,
        folder: pathlib.Path | None = None,
    ):
        self.segment_results = segment_results
        self.all_timesteps = all_timesteps
        self.timesteps_per_segment = timesteps_per_segment
        self.overlap_timesteps = overlap_timesteps
        self.name = name
        self.folder = pathlib.Path(folder) if folder is not None else pathlib.Path.cwd() / 'results'
        self.hours_per_timestep = FlowSystem.calculate_hours_per_timestep(self.all_timesteps)

    @property
    def meta_data(self) -> dict[str, int | list[str]]:
        return {
            'all_timesteps': [datetime.datetime.isoformat(date) for date in self.all_timesteps],
            'timesteps_per_segment': self.timesteps_per_segment,
            'overlap_timesteps': self.overlap_timesteps,
            'sub_calculations': [calc.name for calc in self.segment_results],
        }

    @property
    def segment_names(self) -> list[str]:
        return [segment.name for segment in self.segment_results]

    def solution_without_overlap(self, variable_name: str) -> xr.DataArray:
        """Get variable solution removing segment overlaps.

        Args:
            variable_name: Name of variable to extract.

        Returns:
            xr.DataArray: Continuous solution without overlaps.
        """
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
        save: bool | pathlib.Path = False,
        show: bool = True,
        engine: plotting.PlottingEngine = 'plotly',
    ) -> plotly.graph_objs.Figure | tuple[plt.Figure, plt.Axes]:
        """Plot heatmap of variable solution across segments.

        Args:
            variable_name: Variable to plot.
            heatmap_timeframes: Time aggregation level.
            heatmap_timesteps_per_frame: Timesteps per frame.
            color_map: Color scheme. Also see plotly.
            save: Whether to save plot.
            show: Whether to display plot.
            engine: Plotting engine.

        Returns:
            Figure object.
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

    def to_file(self, folder: str | pathlib.Path | None = None, name: str | None = None, compression: int = 5):
        """Save segmented results to files.

        Args:
            folder: Save folder (defaults to instance folder).
            name: File name (defaults to instance name).
            compression: Compression level 0-9.
        """
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
            segment.to_file(folder=folder, name=segment.name, compression=compression)

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
    save: bool | pathlib.Path = False,
    show: bool = True,
    engine: plotting.PlottingEngine = 'plotly',
    indexer: dict[str, Any] | None = None,
):
    """Plot heatmap of time series data.

    Args:
        dataarray: Data to plot.
        name: Variable name for title.
        folder: Save folder.
        heatmap_timeframes: Time aggregation level.
        heatmap_timesteps_per_frame: Timesteps per frame.
        color_map: Color scheme. Also see plotly.
        save: Whether to save plot.
        show: Whether to display plot.
        engine: Plotting engine.
        indexer: Optional selection dict, e.g., {'scenario': 'base', 'period': 2024}.
             If None, uses first value for each dimension.
             If empty dict {}, uses all values.
    """
    dataarray, suffix_parts = _apply_indexer_to_data(dataarray, indexer, drop=True)
    suffix = '--' + '-'.join(suffix_parts) if suffix_parts else ''
    name = name if not suffix_parts else name + suffix

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
    timesteps: pd.DatetimeIndex | None = None,
    threshold: float | None = 1e-5,
    negate: list[str] | None = None,
    drop_small_vars: bool = True,
    zero_small_values: bool = False,
    drop_suffix: str | None = None,
) -> xr.Dataset:
    """Clean dataset by handling small values and reindexing time.

    Args:
        ds: Dataset to sanitize.
        timesteps: Time index for reindexing (optional).
        threshold: Threshold for small values processing.
        negate: Variables to negate.
        drop_small_vars: Whether to drop variables below threshold.
        zero_small_values: Whether to zero values below threshold.
        drop_suffix: Drop suffix of data var names. Split by the provided str.
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
            vars_to_drop = [var for var in ds.data_vars if (ds_no_nan_abs[var] <= threshold).all().item()]
            ds = ds.drop_vars(vars_to_drop)

        # Option 2: Set small values to zero
        if zero_small_values:
            for var in ds.data_vars:
                # Create a boolean mask of values below threshold
                mask = ds_no_nan_abs[var] <= threshold
                # Only proceed if there are values to zero out
                if bool(mask.any().item()):
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
    variable_dims: Literal['scalar', 'time', 'scenario', 'timeonly', 'scenarioonly'] | None = None,
    timesteps: pd.DatetimeIndex | str | pd.Timestamp | None = None,
    scenarios: pd.Index | str | int | None = None,
    contains: str | list[str] | None = None,
    startswith: str | list[str] | None = None,
) -> xr.Dataset:
    """Filter dataset by variable dimensions, indexes, and with string filters for variable names.

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


def filter_dataarray_by_coord(da: xr.DataArray, **kwargs: str | list[str] | None) -> xr.DataArray:
    """Filter flows by node and component attributes.

    Filters are applied in the order they are specified. All filters must match for an edge to be included.

    To recombine filtered dataarrays, use `xr.concat`.

    xr.concat([res.sizes(start='Fernwärme'), res.sizes(end='Fernwärme')], dim='flow')

    Args:
        da: Flow DataArray with network metadata coordinates.
        **kwargs: Coord filters as name=value pairs.

    Returns:
        Filtered DataArray with matching edges.

    Raises:
        AttributeError: If required coordinates are missing.
        ValueError: If specified nodes don't exist or no matches found.
    """

    # Helper function to process filters
    def apply_filter(array, coord_name: str, coord_values: Any | list[Any]):
        # Verify coord exists
        if coord_name not in array.coords:
            raise AttributeError(f"Missing required coordinate '{coord_name}'")

        # Convert single value to list
        val_list = [coord_values] if isinstance(coord_values, str) else coord_values

        # Verify coord_values exist
        available = set(array[coord_name].values)
        missing = [v for v in val_list if v not in available]
        if missing:
            raise ValueError(f'{coord_name.title()} value(s) not found: {missing}')

        # Apply filter
        return array.where(
            array[coord_name].isin(val_list) if isinstance(coord_values, list) else array[coord_name] == coord_values,
            drop=True,
        )

    # Apply filters from kwargs
    filters = {k: v for k, v in kwargs.items() if v is not None}
    try:
        for coord, values in filters.items():
            da = apply_filter(da, coord, values)
    except ValueError as e:
        raise ValueError(f'No edges match criteria: {filters}') from e

    # Verify results exist
    if da.size == 0:
        raise ValueError(f'No edges match criteria: {filters}')

    return da


def _apply_indexer_to_data(
    data: xr.DataArray | xr.Dataset, indexer: dict[str, Any] | None = None, drop=False
) -> tuple[xr.DataArray | xr.Dataset, list[str]]:
    """
    Apply indexer selection or auto-select first values for non-time dimensions.

    Args:
        data: xarray Dataset or DataArray
        indexer: Optional selection dict
            If None, uses first value for each dimension (except time).
            If empty dict {}, uses all values.

    Returns:
        Tuple of (selected_data, selection_string)
    """
    selection_string = []

    if indexer is not None:
        # User provided indexer
        data = data.sel(indexer, drop=drop)
        selection_string.extend(f'{v}[{k}]' for k, v in indexer.items())
    else:
        # Auto-select first value for each dimension except 'time'
        selection = {}
        for dim in data.dims:
            if dim != 'time' and dim in data.coords:
                first_value = data.coords[dim].values[0]
                selection[dim] = first_value
                selection_string.append(f'{first_value}[{dim}]')
        if selection:
            data = data.sel(selection, drop=drop)

    return data, selection_string
