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
import xarray as xr
import yaml

from . import io as fx_io
from . import plotting
from .flow_system import FlowSystem

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import plotly
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
        color_manager: Optional ComponentColorManager for automatic component-based coloring in plots.
            When set, all plotting methods automatically use this manager when colors='auto'
            (the default). Use `create_color_manager()` to create and configure one, or assign
            an existing manager directly. Set to None to disable automatic coloring.

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

        Configure automatic color management for plots:

        ```python
        # Create and configure a color manager for pattern-based coloring
        manager = results.create_color_manager()
        manager.add_grouping_rule('Solar', 'renewables', 'oranges', match_type='prefix')
        manager.add_grouping_rule('Wind', 'renewables', 'blues', match_type='prefix')
        manager.add_grouping_rule('Battery', 'storage', 'greens', match_type='prefix')
        manager.add_grouping_rule('Gas', 'fossil', 'reds', match_type='prefix')
        manager.auto_group_components()

        # All plots automatically use the manager (colors='auto' is the default)
        results['ElectricityBus'].plot_node_balance()  # Uses configured colors
        results['Battery'].plot_charge_state()  # Also uses configured colors

        # Override when needed
        results['ElectricityBus'].plot_node_balance(colors='viridis')  # Ignores mapper
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
                "The 'flow_system' parameter is deprecated. Use 'flow_system_data' instead. "
                "Access is now via '.flow_system_data', while '.flow_system' returns the restored FlowSystem.",
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

        # Color manager for intelligent plot coloring
        self.color_manager: plotting.ComponentColorManager | None = None

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

    def create_color_manager(self) -> plotting.ComponentColorManager:
        """Create and assign a new ComponentColorManager for this results instance.

        The color manager is automatically used by all plotting methods when colors='auto'
        (the default). Configure it with grouping rules to define pattern-based color families.

        You can also assign an existing manager directly via `results.color_manager = manager`.

        Returns:
            The newly created ComponentColorManager with all components registered, ready to be configured.

        Examples:
            Create and configure a new manager:

            >>> manager = results.create_color_manager()
            >>> manager.add_grouping_rule('Solar', 'renewables', 'oranges', match_type='prefix')
            >>> manager.add_grouping_rule('Wind', 'renewables', 'blues', match_type='prefix')
            >>> manager.add_grouping_rule('Gas', 'fossil', 'reds', match_type='prefix')
            >>> manager.auto_group_components()
            >>> results['ElectricityBus'].plot_node_balance()  # Uses manager automatically

            Or assign an existing manager:

            >>> my_manager = plotting.ComponentColorManager(list(results.components.keys()))
            >>> my_manager.add_grouping_rule('Renewable', 'renewables', 'greens', match_type='prefix')
            >>> my_manager.auto_group_components()
            >>> results.color_manager = my_manager

            Override with explicit colors if needed:

            >>> results['ElectricityBus'].plot_node_balance(colors='viridis')  # Ignores manager
        """
        component_names = list(self.components.keys())
        self.color_manager = plotting.ComponentColorManager(component_names)
        self.color_manager.auto_group_components()
        return self.color_manager

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

            ds[effect] = xr.concat(component_arrays, dim='component', coords='minimal', join='outer').rename(effect)

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
        variable_name: str | list[str],
        save: bool | pathlib.Path = False,
        show: bool = True,
        colors: plotting.ColorType = 'viridis',
        engine: plotting.PlottingEngine = 'plotly',
        select: dict[FlowSystemDimensions, Any] | None = None,
        facet_by: str | list[str] | None = 'scenario',
        animate_by: str | None = 'period',
        facet_cols: int = 3,
        reshape_time: tuple[Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'], Literal['W', 'D', 'h', '15min', 'min']]
        | Literal['auto']
        | None = 'auto',
        fill: Literal['ffill', 'bfill'] | None = 'ffill',
        # Deprecated parameters (kept for backwards compatibility)
        indexer: dict[FlowSystemDimensions, Any] | None = None,
        heatmap_timeframes: Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'] | None = None,
        heatmap_timesteps_per_frame: Literal['W', 'D', 'h', '15min', 'min'] | None = None,
        color_map: str | None = None,
        **plot_kwargs: Any,
    ) -> plotly.graph_objs.Figure | tuple[plt.Figure, plt.Axes]:
        """
        Plots a heatmap visualization of a variable using imshow or time-based reshaping.

        Supports multiple visualization features that can be combined:
        - **Multi-variable**: Plot multiple variables on a single heatmap (creates 'variable' dimension)
        - **Time reshaping**: Converts 'time' dimension into 2D (e.g., hours vs days)
        - **Faceting**: Creates subplots for different dimension values
        - **Animation**: Animates through dimension values (Plotly only)

        Args:
            variable_name: The name of the variable to plot, or a list of variable names.
                When a list is provided, variables are combined into a single DataArray
                with a new 'variable' dimension.
            save: Whether to save the plot or not. If a path is provided, the plot will be saved at that location.
            show: Whether to show the plot or not.
            colors: Color scheme for the heatmap. See `flixopt.plotting.ColorType` for options.
            engine: The engine to use for plotting. Can be either 'plotly' or 'matplotlib'.
            select: Optional data selection dict. Supports single values, lists, slices, and index arrays.
                Applied BEFORE faceting/animation/reshaping.
            facet_by: Dimension(s) to create facets (subplots) for. Can be a single dimension name (str)
                or list of dimensions. Each unique value combination creates a subplot. Ignored if not found.
            animate_by: Dimension to animate over (Plotly only). Creates animation frames that cycle through
                dimension values. Only one dimension can be animated. Ignored if not found.
            facet_cols: Number of columns in the facet grid layout (default: 3).
            reshape_time: Time reshaping configuration (default: 'auto'):
                - 'auto': Automatically applies ('D', 'h') when only 'time' dimension remains
                - Tuple: Explicit reshaping, e.g. ('D', 'h') for days vs hours,
                         ('MS', 'D') for months vs days, ('W', 'h') for weeks vs hours
                - None: Disable auto-reshaping (will error if only 1D time data)
                Supported timeframes: 'YS', 'MS', 'W', 'D', 'h', '15min', 'min'
            fill: Method to fill missing values after reshape: 'ffill' (forward fill) or 'bfill' (backward fill).
                Default is 'ffill'.
            **plot_kwargs: Additional plotting customization options.
                Common options:

                - **dpi** (int): Export resolution for saved plots. Default: 300.

                For heatmaps specifically:

                - **vmin** (float): Minimum value for color scale (both engines).
                - **vmax** (float): Maximum value for color scale (both engines).

                For Matplotlib heatmaps:

                - **imshow_kwargs** (dict): Additional kwargs for matplotlib's imshow (e.g., interpolation, aspect).
                - **cbar_kwargs** (dict): Additional kwargs for colorbar customization.

        Examples:
            Direct imshow mode (default):

            >>> results.plot_heatmap('Battery|charge_state', select={'scenario': 'base'})

            Facet by scenario:

            >>> results.plot_heatmap('Boiler(Qth)|flow_rate', facet_by='scenario', facet_cols=2)

            Animate by period:

            >>> results.plot_heatmap('Boiler(Qth)|flow_rate', select={'scenario': 'base'}, animate_by='period')

            Time reshape mode - daily patterns:

            >>> results.plot_heatmap('Boiler(Qth)|flow_rate', select={'scenario': 'base'}, reshape_time=('D', 'h'))

            Combined: time reshaping with faceting and animation:

            >>> results.plot_heatmap(
            ...     'Boiler(Qth)|flow_rate', facet_by='scenario', animate_by='period', reshape_time=('D', 'h')
            ... )

            Multi-variable heatmap (variables as one axis):

            >>> results.plot_heatmap(
            ...     ['Boiler(Q_th)|flow_rate', 'CHP(Q_th)|flow_rate', 'HeatStorage|charge_state'],
            ...     select={'scenario': 'base', 'period': 1},
            ...     reshape_time=None,
            ... )

            Multi-variable with time reshaping:

            >>> results.plot_heatmap(
            ...     ['Boiler(Q_th)|flow_rate', 'CHP(Q_th)|flow_rate'],
            ...     facet_by='scenario',
            ...     animate_by='period',
            ...     reshape_time=('D', 'h'),
            ... )

            High-resolution export with custom color range:

            >>> results.plot_heatmap('Battery|charge_state', save=True, dpi=600, vmin=0, vmax=100)

            Matplotlib heatmap with custom imshow settings:

            >>> results.plot_heatmap(
            ...     'Boiler(Q_th)|flow_rate',
            ...     engine='matplotlib',
            ...     imshow_kwargs={'interpolation': 'bilinear', 'aspect': 'auto'},
            ... )
        """
        # Delegate to module-level plot_heatmap function
        return plot_heatmap(
            data=self.solution[variable_name],
            name=variable_name if isinstance(variable_name, str) else None,
            folder=self.folder,
            colors=colors,
            save=save,
            show=show,
            engine=engine,
            select=select,
            facet_by=facet_by,
            animate_by=animate_by,
            facet_cols=facet_cols,
            reshape_time=reshape_time,
            fill=fill,
            indexer=indexer,
            heatmap_timeframes=heatmap_timeframes,
            heatmap_timesteps_per_frame=heatmap_timesteps_per_frame,
            color_map=color_map,
            **plot_kwargs,
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
        colors: plotting.ColorType | Literal['auto'] = 'auto',
        engine: plotting.PlottingEngine = 'plotly',
        select: dict[FlowSystemDimensions, Any] | None = None,
        unit_type: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        mode: Literal['area', 'stacked_bar', 'line'] = 'stacked_bar',
        drop_suffix: bool = True,
        facet_by: str | list[str] | None = 'scenario',
        animate_by: str | None = 'period',
        facet_cols: int = 3,
        # Deprecated parameter (kept for backwards compatibility)
        indexer: dict[FlowSystemDimensions, Any] | None = None,
        **plot_kwargs: Any,
    ) -> plotly.graph_objs.Figure | tuple[plt.Figure, plt.Axes]:
        """
        Plots the node balance of the Component or Bus with optional faceting and animation.

        Args:
            save: Whether to save the plot or not. If a path is provided, the plot will be saved at that location.
            show: Whether to show the plot or not.
            colors: The colors to use for the plot. Options:
                - 'auto' (default): Use `self.color_manager` if configured, else fall back to 'viridis'
                - Colormap name string (e.g., 'viridis', 'plasma')
                - List of color strings
                - Dict mapping variable names to colors
                Set `results.color_manager` to a `ComponentColorManager` for automatic component-based grouping.
            engine: The engine to use for plotting. Can be either 'plotly' or 'matplotlib'.
            select: Optional data selection dict. Supports:
                - Single values: {'scenario': 'base', 'period': 2024}
                - Multiple values: {'scenario': ['base', 'high', 'renewable']}
                - Slices: {'time': slice('2024-01', '2024-06')}
                - Index arrays: {'time': time_array}
                Note: Applied BEFORE faceting/animation.
            unit_type: The unit type to use for the dataset. Can be 'flow_rate' or 'flow_hours'.
                - 'flow_rate': Returns the flow_rates of the Node.
                - 'flow_hours': Returns the flow_hours of the Node. [flow_hours(t) = flow_rate(t) * dt(t)]. Renames suffixes to |flow_hours.
            mode: The plotting mode. Use 'stacked_bar' for stacked bar charts, 'line' for stepped lines, or 'area' for stacked area charts.
            drop_suffix: Whether to drop the suffix from the variable names.
            facet_by: Dimension(s) to create facets (subplots) for. Can be a single dimension name (str)
                or list of dimensions. Each unique value combination creates a subplot. Ignored if not found.
                Example: 'scenario' creates one subplot per scenario.
                Example: ['scenario', 'period'] creates a grid of subplots for each scenario-period combination.
            animate_by: Dimension to animate over (Plotly only). Creates animation frames that cycle through
                dimension values. Only one dimension can be animated. Ignored if not found.
            facet_cols: Number of columns in the facet grid layout (default: 3).
            **plot_kwargs: Additional plotting customization options passed to underlying plotting functions.

                Common options:

                - **dpi** (int): Export resolution in dots per inch. Default: 300.

                **For Plotly engine** (`engine='plotly'`):

                - **trace_kwargs** (dict): Customize traces via `fig.update_traces()`.
                  Example: `trace_kwargs={'line': {'width': 5, 'dash': 'dot'}}`
                - **layout_kwargs** (dict): Customize layout via `fig.update_layout()`.
                  Example: `layout_kwargs={'width': 1200, 'height': 600, 'template': 'plotly_dark'}`
                - Any Plotly Express parameter for px.bar()/px.line()/px.area()

                **For Matplotlib engine** (`engine='matplotlib'`):

                - **plot_kwargs** (dict): Customize plot via `ax.bar()` or `ax.step()`.
                  Example: `plot_kwargs={'linewidth': 3, 'alpha': 0.7, 'edgecolor': 'black'}`

                See :func:`flixopt.plotting.with_plotly` and :func:`flixopt.plotting.with_matplotlib`
                for complete parameter reference.

        Examples:
            Basic plot (current behavior):

            >>> results['Boiler'].plot_node_balance()

            Facet by scenario:

            >>> results['Boiler'].plot_node_balance(facet_by='scenario', facet_cols=2)

            Animate by period:

            >>> results['Boiler'].plot_node_balance(animate_by='period')

            Facet by scenario AND animate by period:

            >>> results['Boiler'].plot_node_balance(facet_by='scenario', animate_by='period')

            Select single scenario, then facet by period:

            >>> results['Boiler'].plot_node_balance(select={'scenario': 'base'}, facet_by='period')

            Select multiple scenarios and facet by them:

            >>> results['Boiler'].plot_node_balance(
            ...     select={'scenario': ['base', 'high', 'renewable']}, facet_by='scenario'
            ... )

            Time range selection (summer months only):

            >>> results['Boiler'].plot_node_balance(select={'time': slice('2024-06', '2024-08')}, facet_by='scenario')

            High-resolution export for publication:

            >>> results['Boiler'].plot_node_balance(engine='matplotlib', save='figure.png', dpi=600)

            Custom Plotly theme and layout:

            >>> results['Boiler'].plot_node_balance(
            ...     layout_kwargs={'template': 'plotly_dark', 'width': 1200, 'height': 600}
            ... )

            Custom line styling:

            >>> results['Boiler'].plot_node_balance(mode='line', trace_kwargs={'line': {'width': 5, 'dash': 'dot'}})

            Custom matplotlib appearance:

            >>> results['Boiler'].plot_node_balance(engine='matplotlib', plot_kwargs={'linewidth': 3, 'alpha': 0.7})
        """
        # Handle deprecated indexer parameter
        if indexer is not None:
            # Check for conflict with new parameter
            if select is not None:
                raise ValueError(
                    "Cannot use both deprecated parameter 'indexer' and new parameter 'select'. Use only 'select'."
                )

            import warnings

            warnings.warn(
                "The 'indexer' parameter is deprecated and will be removed in a future version. Use 'select' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            select = indexer

        if engine not in {'plotly', 'matplotlib'}:
            raise ValueError(f'Engine "{engine}" not supported. Use one of ["plotly", "matplotlib"]')

        # Extract dpi for export_figure
        dpi = plot_kwargs.pop('dpi', 300)

        # Don't pass select/indexer to node_balance - we'll apply it afterwards
        ds = self.node_balance(with_last_timestep=False, unit_type=unit_type, drop_suffix=drop_suffix)

        ds, suffix_parts = _apply_selection_to_data(ds, select=select, drop=True)

        # Resolve colors to a dict (handles auto, mapper, etc.)
        colors_to_use = (
            self._calculation_results.color_manager
            if colors == 'auto' and self._calculation_results.color_manager is not None
            else 'viridis'
            if colors == 'auto'
            else colors
        )
        resolved_colors = plotting.resolve_colors(ds, colors_to_use, engine=engine)

        # Matplotlib requires only 'time' dimension; check for extras after selection
        if engine == 'matplotlib':
            extra_dims = [d for d in ds.dims if d != 'time']
            if extra_dims:
                raise ValueError(
                    f'Matplotlib engine only supports a single time axis, but found extra dimensions: {extra_dims}. '
                    f'Please use select={{...}} to reduce dimensions or switch to engine="plotly" for faceting/animation.'
                )
        suffix = '--' + '-'.join(suffix_parts) if suffix_parts else ''

        title = (
            f'{self.label} (flow rates){suffix}' if unit_type == 'flow_rate' else f'{self.label} (flow hours){suffix}'
        )

        if engine == 'plotly':
            figure_like = plotting.with_plotly(
                ds,
                facet_by=facet_by,
                animate_by=animate_by,
                colors=resolved_colors,
                mode=mode,
                title=title,
                facet_cols=facet_cols,
                xlabel='Time in h',
                **plot_kwargs,
            )
            default_filetype = '.html'
        else:
            figure_like = plotting.with_matplotlib(
                ds,
                colors=resolved_colors,
                mode=mode,
                title=title,
                **plot_kwargs,
            )
            default_filetype = '.png'

        return plotting.export_figure(
            figure_like=figure_like,
            default_path=self._calculation_results.folder / title,
            default_filetype=default_filetype,
            user_path=None if isinstance(save, bool) else pathlib.Path(save),
            show=show,
            save=True if save else False,
            dpi=dpi,
        )

    def plot_node_balance_pie(
        self,
        lower_percentage_group: float = 5,
        colors: plotting.ColorType | Literal['auto'] = 'auto',
        text_info: str = 'percent+label+value',
        save: bool | pathlib.Path = False,
        show: bool = True,
        engine: plotting.PlottingEngine = 'plotly',
        select: dict[FlowSystemDimensions, Any] | None = None,
        # Deprecated parameter (kept for backwards compatibility)
        indexer: dict[FlowSystemDimensions, Any] | None = None,
        **plot_kwargs: Any,
    ) -> plotly.graph_objs.Figure | tuple[plt.Figure, list[plt.Axes]]:
        """Plot pie chart of flow hours distribution.

        Note:
            Pie charts require scalar data (no extra dimensions beyond time).
            If your data has dimensions like 'scenario' or 'period', either:

            - Use `select` to choose specific values: `select={'scenario': 'base', 'period': 2024}`
            - Let auto-selection choose the first value (a warning will be logged)

        Args:
            lower_percentage_group: Percentage threshold for "Others" grouping.
            colors: Color scheme. Also see plotly.
            text_info: Information to display on pie slices.
            save: Whether to save plot.
            show: Whether to display plot.
            engine: Plotting engine ('plotly' or 'matplotlib').
            select: Optional data selection dict. Supports single values, lists, slices, and index arrays.
                Use this to select specific scenario/period before creating the pie chart.
            **plot_kwargs: Additional plotting customization options.

                Common options:

                - **dpi** (int): Export resolution in dots per inch. Default: 300.
                - **hover_template** (str): Hover text template (Plotly only).
                  Example: `hover_template='%{label}: %{value} (%{percent})'`
                - **text_position** (str): Text position ('inside', 'outside', 'auto').
                - **hole** (float): Size of donut hole (0.0 to 1.0).

                See :func:`flixopt.plotting.dual_pie_with_plotly` for complete reference.

        Examples:
            Basic usage (auto-selects first scenario/period if present):

            >>> results['Bus'].plot_node_balance_pie()

            Explicitly select a scenario and period:

            >>> results['Bus'].plot_node_balance_pie(select={'scenario': 'high_demand', 'period': 2030})

            Create a donut chart with custom hover text:

            >>> results['Bus'].plot_node_balance_pie(hole=0.4, hover_template='%{label}: %{value:.2f} (%{percent})')

            High-resolution export:

            >>> results['Bus'].plot_node_balance_pie(save='figure.png', dpi=600)
        """
        # Handle deprecated indexer parameter
        if indexer is not None:
            # Check for conflict with new parameter
            if select is not None:
                raise ValueError(
                    "Cannot use both deprecated parameter 'indexer' and new parameter 'select'. Use only 'select'."
                )

            import warnings

            warnings.warn(
                "The 'indexer' parameter is deprecated and will be removed in a future version. Use 'select' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            select = indexer

        # Extract dpi for export_figure
        dpi = plot_kwargs.pop('dpi', 300)

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

        inputs, suffix_parts = _apply_selection_to_data(inputs, select=select, drop=True)
        outputs, suffix_parts = _apply_selection_to_data(outputs, select=select, drop=True)

        # Sum over time dimension
        inputs = inputs.sum('time')
        outputs = outputs.sum('time')

        # Auto-select first value for any remaining dimensions (scenario, period, etc.)
        # Pie charts need scalar data, so we automatically reduce extra dimensions
        extra_dims_inputs = [dim for dim in inputs.dims if dim != 'time']
        extra_dims_outputs = [dim for dim in outputs.dims if dim != 'time']
        extra_dims = list(set(extra_dims_inputs + extra_dims_outputs))

        if extra_dims:
            auto_select = {}
            for dim in extra_dims:
                # Get first value of this dimension
                if dim in inputs.coords:
                    first_val = inputs.coords[dim].values[0]
                elif dim in outputs.coords:
                    first_val = outputs.coords[dim].values[0]
                else:
                    continue
                auto_select[dim] = first_val
                logger.info(
                    f'Pie chart auto-selected {dim}={first_val} (first value). '
                    f'Use select={{"{dim}": value}} to choose a different value.'
                )

            # Apply auto-selection
            inputs = inputs.sel(auto_select)
            outputs = outputs.sel(auto_select)

            # Update suffix with auto-selected values
            auto_suffix_parts = [f'{dim}={val}' for dim, val in auto_select.items()]
            suffix_parts.extend(auto_suffix_parts)

        suffix = '--' + '-'.join(suffix_parts) if suffix_parts else ''
        title = f'{self.label} (total flow hours){suffix}'

        # Combine inputs and outputs to resolve colors for all variables
        combined_ds = xr.Dataset({**inputs.data_vars, **outputs.data_vars})
        colors_to_use = (
            self._calculation_results.color_manager
            if colors == 'auto' and self._calculation_results.color_manager is not None
            else 'viridis'
            if colors == 'auto'
            else colors
        )
        resolved_colors = plotting.resolve_colors(combined_ds, colors_to_use, engine=engine)

        if engine == 'plotly':
            figure_like = plotting.dual_pie_with_plotly(
                data_left=inputs,
                data_right=outputs,
                colors=resolved_colors,
                title=title,
                text_info=text_info,
                subtitles=('Inputs', 'Outputs'),
                legend_title='Flows',
                lower_percentage_group=lower_percentage_group,
                **plot_kwargs,
            )
            default_filetype = '.html'
        elif engine == 'matplotlib':
            logger.debug('Parameter text_info is not supported for matplotlib')
            figure_like = plotting.dual_pie_with_matplotlib(
                data_left=inputs.to_pandas(),
                data_right=outputs.to_pandas(),
                colors=resolved_colors,
                title=title,
                subtitles=('Inputs', 'Outputs'),
                legend_title='Flows',
                lower_percentage_group=lower_percentage_group,
                **plot_kwargs,
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
            dpi=dpi,
        )

    def node_balance(
        self,
        negate_inputs: bool = True,
        negate_outputs: bool = False,
        threshold: float | None = 1e-5,
        with_last_timestep: bool = False,
        unit_type: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        drop_suffix: bool = False,
        select: dict[FlowSystemDimensions, Any] | None = None,
        # Deprecated parameter (kept for backwards compatibility)
        indexer: dict[FlowSystemDimensions, Any] | None = None,
    ) -> xr.Dataset:
        """
        Returns a dataset with the node balance of the Component or Bus.
        Args:
            negate_inputs: Whether to negate the input flow_rates of the Node.
            negate_outputs: Whether to negate the output flow_rates of the Node.
            threshold: The threshold for small values. Variables with all values below the threshold are dropped.
            with_last_timestep: Whether to include the last timestep in the dataset.
            unit_type: The unit type to use for the dataset. Can be 'flow_rate' or 'flow_hours'.
                - 'flow_rate': Returns the flow_rates of the Node.
                - 'flow_hours': Returns the flow_hours of the Node. [flow_hours(t) = flow_rate(t) * dt(t)]. Renames suffixes to |flow_hours.
            drop_suffix: Whether to drop the suffix from the variable names.
            select: Optional data selection dict. Supports single values, lists, slices, and index arrays.
        """
        # Handle deprecated indexer parameter
        if indexer is not None:
            # Check for conflict with new parameter
            if select is not None:
                raise ValueError(
                    "Cannot use both deprecated parameter 'indexer' and new parameter 'select'. Use only 'select'."
                )

            import warnings

            warnings.warn(
                "The 'indexer' parameter is deprecated and will be removed in a future version. Use 'select' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            select = indexer

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

        ds, _ = _apply_selection_to_data(ds, select=select, drop=True)

        if unit_type == 'flow_hours':
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
        colors: plotting.ColorType | Literal['auto'] = 'auto',
        engine: plotting.PlottingEngine = 'plotly',
        mode: Literal['area', 'stacked_bar', 'line'] = 'area',
        select: dict[FlowSystemDimensions, Any] | None = None,
        facet_by: str | list[str] | None = 'scenario',
        animate_by: str | None = 'period',
        facet_cols: int = 3,
        # Deprecated parameter (kept for backwards compatibility)
        indexer: dict[FlowSystemDimensions, Any] | None = None,
        **plot_kwargs: Any,
    ) -> plotly.graph_objs.Figure:
        """Plot storage charge state over time, combined with the node balance with optional faceting and animation.

        Args:
            save: Whether to save the plot or not. If a path is provided, the plot will be saved at that location.
            show: Whether to show the plot or not.
            colors: Color scheme. Also see plotly.
            engine: Plotting engine to use. Only 'plotly' is implemented atm.
            mode: The plotting mode. Use 'stacked_bar' for stacked bar charts, 'line' for stepped lines, or 'area' for stacked area charts.
            select: Optional data selection dict. Supports single values, lists, slices, and index arrays.
                Applied BEFORE faceting/animation.
            facet_by: Dimension(s) to create facets (subplots) for. Can be a single dimension name (str)
                or list of dimensions. Each unique value combination creates a subplot. Ignored if not found.
            animate_by: Dimension to animate over (Plotly only). Creates animation frames that cycle through
                dimension values. Only one dimension can be animated. Ignored if not found.
            facet_cols: Number of columns in the facet grid layout (default: 3).
            **plot_kwargs: Additional plotting customization options passed to underlying plotting functions.

                Common options:

                - **dpi** (int): Export resolution in dots per inch. Default: 300.

                **For Plotly engine:**

                - **trace_kwargs** (dict): Customize traces via `fig.update_traces()`.
                - **layout_kwargs** (dict): Customize layout via `fig.update_layout()`.
                - Any Plotly Express parameter for px.bar()/px.line()/px.area()

                **For Matplotlib engine:**

                - **plot_kwargs** (dict): Customize plot via `ax.bar()` or `ax.step()`.

                See :func:`flixopt.plotting.with_plotly` and :func:`flixopt.plotting.with_matplotlib`
                for complete parameter reference.

        Raises:
            ValueError: If component is not a storage.

        Examples:
            Basic plot:

            >>> results['Storage'].plot_charge_state()

            Facet by scenario:

            >>> results['Storage'].plot_charge_state(facet_by='scenario', facet_cols=2)

            Animate by period:

            >>> results['Storage'].plot_charge_state(animate_by='period')

            Facet by scenario AND animate by period:

            >>> results['Storage'].plot_charge_state(facet_by='scenario', animate_by='period')

            Custom layout:

            >>> results['Storage'].plot_charge_state(layout_kwargs={'template': 'plotly_dark', 'height': 800})

            High-resolution export:

            >>> results['Storage'].plot_charge_state(save='storage.png', dpi=600)
        """
        # Handle deprecated indexer parameter
        if indexer is not None:
            # Check for conflict with new parameter
            if select is not None:
                raise ValueError(
                    "Cannot use both deprecated parameter 'indexer' and new parameter 'select'. Use only 'select'."
                )

            import warnings

            warnings.warn(
                "The 'indexer' parameter is deprecated and will be removed in a future version. Use 'select' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            select = indexer

        # Extract dpi for export_figure
        dpi = plot_kwargs.pop('dpi', 300)

        if not self.is_storage:
            raise ValueError(f'Cant plot charge_state. "{self.label}" is not a storage')

        # Get node balance and charge state
        ds = self.node_balance(with_last_timestep=True).fillna(0)
        charge_state_da = self.charge_state

        # Apply select filtering
        ds, suffix_parts = _apply_selection_to_data(ds, select=select, drop=True)
        charge_state_da, _ = _apply_selection_to_data(charge_state_da, select=select, drop=True)
        suffix = '--' + '-'.join(suffix_parts) if suffix_parts else ''

        title = f'Operation Balance of {self.label}{suffix}'

        # Combine flow balance and charge state for color resolution
        # We need to include both in the color map for consistency
        combined_ds = ds.assign({self._charge_state: charge_state_da})

        # Resolve colors to a dict (handles auto, mapper, etc.)
        colors_to_use = (
            self._calculation_results.color_manager
            if colors == 'auto' and self._calculation_results.color_manager is not None
            else 'viridis'
            if colors == 'auto'
            else colors
        )
        resolved_colors = plotting.resolve_colors(combined_ds, colors_to_use, engine=engine)

        if engine == 'plotly':
            # Plot flows (node balance) with the specified mode
            figure_like = plotting.with_plotly(
                ds,
                facet_by=facet_by,
                animate_by=animate_by,
                colors=resolved_colors,
                mode=mode,
                title=title,
                facet_cols=facet_cols,
                xlabel='Time in h',
                **plot_kwargs,
            )

            # Prepare charge_state as Dataset for plotting
            charge_state_ds = xr.Dataset({self._charge_state: charge_state_da})

            # Plot charge_state with mode='line' to get Scatter traces
            charge_state_fig = plotting.with_plotly(
                charge_state_ds,
                facet_by=facet_by,
                animate_by=animate_by,
                colors=resolved_colors,
                mode='line',  # Always line for charge_state
                title='',  # No title needed for this temp figure
                facet_cols=facet_cols,
                xlabel='Time in h',
                **plot_kwargs,
            )

            # Add charge_state traces to the main figure
            # This preserves subplot assignments and animation frames
            for trace in charge_state_fig.data:
                trace.line.width = 2  # Make charge_state line more prominent
                trace.line.shape = 'linear'  # Smooth line for charge state (not stepped like flows)
                trace.line.color = 'black'
                figure_like.add_trace(trace)

            # Also add traces from animation frames if they exist
            # Both figures use the same animate_by parameter, so they should have matching frames
            if hasattr(charge_state_fig, 'frames') and charge_state_fig.frames:
                # Add charge_state traces to each frame
                for i, frame in enumerate(charge_state_fig.frames):
                    if i < len(figure_like.frames):
                        for trace in frame.data:
                            trace.line.width = 2
                            trace.line.shape = 'linear'  # Smooth line for charge state
                            figure_like.frames[i].data = figure_like.frames[i].data + (trace,)

            default_filetype = '.html'
        elif engine == 'matplotlib':
            # Matplotlib requires only 'time' dimension; check for extras after selection
            extra_dims = [d for d in ds.dims if d != 'time']
            if extra_dims:
                raise ValueError(
                    f'Matplotlib engine only supports a single time axis, but found extra dimensions: {extra_dims}. '
                    f'Please use select={{...}} to reduce dimensions or switch to engine="plotly" for faceting/animation.'
                )
            # For matplotlib, plot flows (node balance), then add charge_state as line
            fig, ax = plotting.with_matplotlib(
                ds,
                colors=resolved_colors,
                mode=mode,
                title=title,
                **plot_kwargs,
            )

            # Add charge_state as a line overlay
            charge_state_df = charge_state_da.to_dataframe()
            ax.plot(
                charge_state_df.index,
                charge_state_df.values.flatten(),
                label=self._charge_state,
                linewidth=2,
                color='black',
            )
            ax.legend()
            fig.tight_layout()

            figure_like = fig, ax
            default_filetype = '.png'

        return plotting.export_figure(
            figure_like=figure_like,
            default_path=self._calculation_results.folder / title,
            default_filetype=default_filetype,
            user_path=None if isinstance(save, bool) else pathlib.Path(save),
            show=show,
            save=True if save else False,
            dpi=dpi,
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

    def get_shares_from(self, element: str) -> xr.Dataset:
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

    Attributes:
        segment_results: List of CalculationResults for each segment
        all_timesteps: Complete time index spanning all segments
        timesteps_per_segment: Number of timesteps in each segment
        overlap_timesteps: Number of overlapping timesteps between segments
        name: Identifier for this segmented calculation
        folder: Directory path for result storage and loading
        hours_per_timestep: Duration of each timestep
        color_manager: Optional ComponentColorManager for automatic component-based coloring in plots.
            When set, it is automatically propagated to all segment results, ensuring
            consistent coloring across segments. Use `create_color_manager()` to create
            and configure one, or assign an existing manager directly.

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

        Configure color management for consistent plotting across segments:

        ```python
        # Create and configure a color manager
        manager = results.create_color_manager()
        manager.add_grouping_rule('Solar', 'renewables', 'oranges', match_type='prefix')
        manager.add_grouping_rule('Wind', 'renewables', 'blues', match_type='prefix')
        manager.add_grouping_rule('Battery', 'storage', 'greens', match_type='prefix')
        manager.auto_group_components()

        # Plot using any segment - colors are consistent across all segments
        results.segment_results[0]['ElectricityBus'].plot_node_balance()
        results.segment_results[1]['ElectricityBus'].plot_node_balance()
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
    def from_file(cls, folder: str | pathlib.Path, name: str) -> SegmentedCalculationResults:
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

        # Color manager for intelligent plot coloring
        self.color_manager: plotting.ComponentColorManager | None = None

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

    def create_color_manager(self) -> plotting.ComponentColorManager:
        """Create and assign a new ComponentColorManager for this segmented results instance.

        The color manager is automatically propagated to all segment results,
        ensuring consistent coloring across all segments when using plotting methods.

        Returns:
            The newly created ComponentColorManager with all components registered.

        Examples:
            Create and configure a manager for segmented results:

            >>> manager = segmented_results.create_color_manager()
            >>> manager.add_grouping_rule('Solar', 'renewables', 'oranges', match_type='prefix')
            >>> manager.add_grouping_rule('Wind', 'renewables', 'blues', match_type='prefix')
            >>> manager.auto_group_components()
            >>> # The manager is now available on all segments
            >>> segmented_results.segment_results[0]['ElectricityBus'].plot_node_balance()
        """
        # Get component names from first segment (all segments should have same components)
        component_names = list(self.segment_results[0].components.keys()) if self.segment_results else []
        self.color_manager = plotting.ComponentColorManager(component_names)
        # Propagate to all segment results for consistent coloring
        for segment in self.segment_results:
            segment.color_manager = self.color_manager
        return self.color_manager

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
        reshape_time: tuple[Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'], Literal['W', 'D', 'h', '15min', 'min']]
        | Literal['auto']
        | None = 'auto',
        colors: str = 'portland',
        save: bool | pathlib.Path = False,
        show: bool = True,
        engine: plotting.PlottingEngine = 'plotly',
        facet_by: str | list[str] | None = None,
        animate_by: str | None = None,
        facet_cols: int = 3,
        fill: Literal['ffill', 'bfill'] | None = 'ffill',
        # Deprecated parameters (kept for backwards compatibility)
        heatmap_timeframes: Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'] | None = None,
        heatmap_timesteps_per_frame: Literal['W', 'D', 'h', '15min', 'min'] | None = None,
        color_map: str | None = None,
        **plot_kwargs: Any,
    ) -> plotly.graph_objs.Figure | tuple[plt.Figure, plt.Axes]:
        """Plot heatmap of variable solution across segments.

        Args:
            variable_name: Variable to plot.
            reshape_time: Time reshaping configuration (default: 'auto'):
                - 'auto': Automatically applies ('D', 'h') when only 'time' dimension remains
                - Tuple like ('D', 'h'): Explicit reshaping (days vs hours)
                - None: Disable time reshaping
            colors: Color scheme. See plotting.ColorType for options.
            save: Whether to save plot.
            show: Whether to display plot.
            engine: Plotting engine.
            facet_by: Dimension(s) to create facets (subplots) for.
            animate_by: Dimension to animate over (Plotly only).
            facet_cols: Number of columns in the facet grid layout.
            fill: Method to fill missing values: 'ffill' or 'bfill'.
            heatmap_timeframes: (Deprecated) Use reshape_time instead.
            heatmap_timesteps_per_frame: (Deprecated) Use reshape_time instead.
            color_map: (Deprecated) Use colors instead.
            **plot_kwargs: Additional plotting customization options.
                Common options:

                - **dpi** (int): Export resolution for saved plots. Default: 300.
                - **vmin** (float): Minimum value for color scale.
                - **vmax** (float): Maximum value for color scale.

                For Matplotlib heatmaps:

                - **imshow_kwargs** (dict): Additional kwargs for matplotlib's imshow.
                - **cbar_kwargs** (dict): Additional kwargs for colorbar customization.

        Returns:
            Figure object.
        """
        # Handle deprecated parameters
        if heatmap_timeframes is not None or heatmap_timesteps_per_frame is not None:
            # Check for conflict with new parameter
            if reshape_time != 'auto':  # Check if user explicitly set reshape_time
                raise ValueError(
                    "Cannot use both deprecated parameters 'heatmap_timeframes'/'heatmap_timesteps_per_frame' "
                    "and new parameter 'reshape_time'. Use only 'reshape_time'."
                )

            import warnings

            warnings.warn(
                "The 'heatmap_timeframes' and 'heatmap_timesteps_per_frame' parameters are deprecated. "
                "Use 'reshape_time=(timeframes, timesteps_per_frame)' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Override reshape_time if old parameters provided
            if heatmap_timeframes is not None and heatmap_timesteps_per_frame is not None:
                reshape_time = (heatmap_timeframes, heatmap_timesteps_per_frame)

        if color_map is not None:
            # Check for conflict with new parameter
            if colors != 'portland':  # Check if user explicitly set colors
                raise ValueError(
                    "Cannot use both deprecated parameter 'color_map' and new parameter 'colors'. Use only 'colors'."
                )

            import warnings

            warnings.warn(
                "The 'color_map' parameter is deprecated. Use 'colors' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            colors = color_map

        return plot_heatmap(
            data=self.solution_without_overlap(variable_name),
            name=variable_name,
            folder=self.folder,
            reshape_time=reshape_time,
            colors=colors,
            save=save,
            show=show,
            engine=engine,
            facet_by=facet_by,
            animate_by=animate_by,
            facet_cols=facet_cols,
            fill=fill,
            **plot_kwargs,
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
    data: xr.DataArray | xr.Dataset,
    name: str | None = None,
    folder: pathlib.Path | None = None,
    colors: plotting.ColorType = 'viridis',
    save: bool | pathlib.Path = False,
    show: bool = True,
    engine: plotting.PlottingEngine = 'plotly',
    select: dict[str, Any] | None = None,
    facet_by: str | list[str] | None = None,
    animate_by: str | None = None,
    facet_cols: int = 3,
    reshape_time: tuple[Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'], Literal['W', 'D', 'h', '15min', 'min']]
    | Literal['auto']
    | None = 'auto',
    fill: Literal['ffill', 'bfill'] | None = 'ffill',
    # Deprecated parameters (kept for backwards compatibility)
    indexer: dict[str, Any] | None = None,
    heatmap_timeframes: Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'] | None = None,
    heatmap_timesteps_per_frame: Literal['W', 'D', 'h', '15min', 'min'] | None = None,
    color_map: str | None = None,
    **plot_kwargs: Any,
):
    """Plot heatmap visualization with support for multi-variable, faceting, and animation.

    This function provides a standalone interface to the heatmap plotting capabilities,
    supporting the same modern features as CalculationResults.plot_heatmap().

    Args:
        data: Data to plot. Can be a single DataArray or an xarray Dataset.
            When a Dataset is provided, all data variables are combined along a new 'variable' dimension.
        name: Optional name for the title. If not provided, uses the DataArray name or
            generates a default title for Datasets.
        folder: Save folder for the plot. Defaults to current directory if not provided.
        colors: Color scheme for the heatmap. See `flixopt.plotting.ColorType` for options.
        save: Whether to save the plot or not. If a path is provided, the plot will be saved at that location.
        show: Whether to show the plot or not.
        engine: The engine to use for plotting. Can be either 'plotly' or 'matplotlib'.
        select: Optional data selection dict. Supports single values, lists, slices, and index arrays.
        facet_by: Dimension(s) to create facets (subplots) for. Can be a single dimension name (str)
            or list of dimensions. Each unique value combination creates a subplot.
        animate_by: Dimension to animate over (Plotly only). Creates animation frames.
        facet_cols: Number of columns in the facet grid layout (default: 3).
        reshape_time: Time reshaping configuration (default: 'auto'):
            - 'auto': Automatically applies ('D', 'h') when only 'time' dimension remains
            - Tuple: Explicit reshaping, e.g. ('D', 'h') for days vs hours
            - None: Disable auto-reshaping
        fill: Method to fill missing values after reshape: 'ffill' (forward fill) or 'bfill' (backward fill).
            Default is 'ffill'.

    Examples:
        Single DataArray with time reshaping:

        >>> plot_heatmap(data, name='Temperature', folder=Path('.'), reshape_time=('D', 'h'))

        Dataset with multiple variables (facet by variable):

        >>> dataset = xr.Dataset({'Boiler': data1, 'CHP': data2, 'Storage': data3})
        >>> plot_heatmap(
        ...     dataset,
        ...     folder=Path('.'),
        ...     facet_by='variable',
        ...     reshape_time=('D', 'h'),
        ... )

        Dataset with animation by variable:

        >>> plot_heatmap(dataset, animate_by='variable', reshape_time=('D', 'h'))
    """
    # Handle deprecated heatmap time parameters
    if heatmap_timeframes is not None or heatmap_timesteps_per_frame is not None:
        # Check for conflict with new parameter
        if reshape_time != 'auto':  # User explicitly set reshape_time
            raise ValueError(
                "Cannot use both deprecated parameters 'heatmap_timeframes'/'heatmap_timesteps_per_frame' "
                "and new parameter 'reshape_time'. Use only 'reshape_time'."
            )

        import warnings

        warnings.warn(
            "The 'heatmap_timeframes' and 'heatmap_timesteps_per_frame' parameters are deprecated. "
            "Use 'reshape_time=(timeframes, timesteps_per_frame)' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Override reshape_time if both old parameters provided
        if heatmap_timeframes is not None and heatmap_timesteps_per_frame is not None:
            reshape_time = (heatmap_timeframes, heatmap_timesteps_per_frame)

    # Handle deprecated color_map parameter
    if color_map is not None:
        # Check for conflict with new parameter
        if colors != 'viridis':  # User explicitly set colors
            raise ValueError(
                "Cannot use both deprecated parameter 'color_map' and new parameter 'colors'. Use only 'colors'."
            )

        import warnings

        warnings.warn(
            "The 'color_map' parameter is deprecated. Use 'colors' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        colors = color_map

    # Handle deprecated indexer parameter
    if indexer is not None:
        # Check for conflict with new parameter
        if select is not None:  # User explicitly set select
            raise ValueError(
                "Cannot use both deprecated parameter 'indexer' and new parameter 'select'. Use only 'select'."
            )

        import warnings

        warnings.warn(
            "The 'indexer' parameter is deprecated. Use 'select' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        select = indexer

    # Convert Dataset to DataArray with 'variable' dimension
    if isinstance(data, xr.Dataset):
        # Extract all data variables from the Dataset
        variable_names = list(data.data_vars)
        dataarrays = [data[var] for var in variable_names]

        # Combine into single DataArray with 'variable' dimension
        data = xr.concat(dataarrays, dim='variable')
        data = data.assign_coords(variable=variable_names)

        # Use Dataset variable names for title if name not provided
        if name is None:
            title_name = f'Heatmap of {len(variable_names)} variables'
        else:
            title_name = name
    else:
        # Single DataArray
        if name is None:
            title_name = data.name if data.name else 'Heatmap'
        else:
            title_name = name

    # Apply select filtering
    data, suffix_parts = _apply_selection_to_data(data, select=select, drop=True)
    suffix = '--' + '-'.join(suffix_parts) if suffix_parts else ''

    # Matplotlib heatmaps require at most 2D data
    # Time dimension will be reshaped to 2D (timeframe × timestep), so can't have other dims alongside it
    if engine == 'matplotlib':
        dims = list(data.dims)

        # If 'time' dimension exists and will be reshaped, we can't have any other dimensions
        if 'time' in dims and len(dims) > 1 and reshape_time is not None:
            extra_dims = [d for d in dims if d != 'time']
            raise ValueError(
                f'Matplotlib heatmaps with time reshaping cannot have additional dimensions. '
                f'Found extra dimensions: {extra_dims}. '
                f'Use select={{...}} to reduce to time only, use "reshape_time=None" or switch to engine="plotly" or use for multi-dimensional support.'
            )
        # If no 'time' dimension (already reshaped or different data), allow at most 2 dimensions
        elif 'time' not in dims and len(dims) > 2:
            raise ValueError(
                f'Matplotlib heatmaps support at most 2 dimensions, but data has {len(dims)}: {dims}. '
                f'Use select={{...}} to reduce dimensions or switch to engine="plotly".'
            )

    # Build title
    title = f'{title_name}{suffix}'
    if isinstance(reshape_time, tuple):
        timeframes, timesteps_per_frame = reshape_time
        title += f' ({timeframes} vs {timesteps_per_frame})'

    # Extract dpi before passing to plotting functions
    dpi = plot_kwargs.pop('dpi', 300)

    # Plot with appropriate engine
    if engine == 'plotly':
        figure_like = plotting.heatmap_with_plotly(
            data=data,
            facet_by=facet_by,
            animate_by=animate_by,
            colors=colors,
            title=title,
            facet_cols=facet_cols,
            reshape_time=reshape_time,
            fill=fill,
            **plot_kwargs,
        )
        default_filetype = '.html'
    elif engine == 'matplotlib':
        figure_like = plotting.heatmap_with_matplotlib(
            data=data,
            colors=colors,
            title=title,
            reshape_time=reshape_time,
            fill=fill,
            **plot_kwargs,
        )
        default_filetype = '.png'
    else:
        raise ValueError(f'Engine "{engine}" not supported. Use "plotly" or "matplotlib"')

    # Set default folder if not provided
    if folder is None:
        folder = pathlib.Path('.')

    return plotting.export_figure(
        figure_like=figure_like,
        default_path=folder / title,
        default_filetype=default_filetype,
        user_path=None if isinstance(save, bool) else pathlib.Path(save),
        show=show,
        save=True if save else False,
        dpi=dpi,
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

        # Normalize to list for sequence-like inputs (excluding strings)
        if isinstance(coord_values, str):
            val_list = [coord_values]
        elif isinstance(coord_values, (list, tuple, np.ndarray, pd.Index)):
            val_list = list(coord_values)
        else:
            val_list = [coord_values]

        # Verify coord_values exist
        available = set(array[coord_name].values)
        missing = [v for v in val_list if v not in available]
        if missing:
            raise ValueError(f'{coord_name.title()} value(s) not found: {missing}')

        # Apply filter
        return array.where(
            array[coord_name].isin(val_list) if len(val_list) > 1 else array[coord_name] == val_list[0],
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


def _apply_selection_to_data(
    data: xr.DataArray | xr.Dataset,
    select: dict[str, Any] | None = None,
    drop=False,
) -> tuple[xr.DataArray | xr.Dataset, list[str]]:
    """
    Apply selection to data.

    Args:
        data: xarray Dataset or DataArray
        select: Optional selection dict
        drop: Whether to drop dimensions after selection

    Returns:
        Tuple of (selected_data, selection_string)
    """
    selection_string = []

    if select:
        data = data.sel(select, drop=drop)
        selection_string.extend(f'{dim}={val}' for dim, val in select.items())

    return data, selection_string
