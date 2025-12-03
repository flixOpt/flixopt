from __future__ import annotations

import copy
import datetime
import logging
import pathlib
import warnings
from typing import TYPE_CHECKING, Any, Literal

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from . import io as fx_io
from . import plotting
from .color_processing import process_colors
from .config import CONFIG, DEPRECATION_REMOVAL_VERSION, SUCCESS_LEVEL
from .flow_system import FlowSystem
from .plot_accessors import ElementPlotAccessor, PlotAccessor
from .structure import CompositeContainerMixin, ResultsContainer

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import plotly
    import pyvis

    from .core import FlowSystemDimensions
    from .optimization import Optimization, SegmentedOptimization

logger = logging.getLogger('flixopt')


def load_mapping_from_file(path: pathlib.Path) -> dict[str, str | list[str]]:
    """Load color mapping from JSON or YAML file.

    Tries loader based on file suffix first, with fallback to the other format.

    Args:
        path: Path to config file (.json or .yaml/.yml)

    Returns:
        Dictionary mapping components to colors or colorscales to component lists

    Raises:
        ValueError: If file cannot be loaded as JSON or YAML
    """
    return fx_io.load_config_file(path)


class _FlowSystemRestorationError(Exception):
    """Exception raised when a FlowSystem cannot be restored from dataset."""

    pass


class Results(CompositeContainerMixin['ComponentResults | BusResults | EffectResults | FlowResults']):
    """Comprehensive container for optimization results and analysis tools.

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
        - **Metadata**: Optimization parameters, timing, and system configuration

    Attributes:
        solution: Dataset containing all optimization variable solutions
        flow_system_data: Dataset with complete system configuration and parameters. Restore the used FlowSystem for further analysis.
        summary: Optimization metadata including solver status, timing, and statistics
        name: Unique identifier for this optimization
        model: Original linopy optimization model (if available)
        folder: Directory path for result storage and loading
        components: Dictionary mapping component labels to ComponentResults objects
        buses: Dictionary mapping bus labels to BusResults objects
        effects: Dictionary mapping effect names to EffectResults objects
        timesteps_extra: Extended time index including boundary conditions
        hours_per_timestep: Duration of each timestep for proper energy optimizations

    Examples:
        Load and analyze saved results:

        ```python
        # Load results from file
        results = Results.from_file('results', 'annual_optimization')

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
        # Dict-based configuration:
        results.setup_colors({'Solar*': 'Oranges', 'Wind*': 'Blues', 'Battery': 'green'})

        # All plots automatically use configured colors (colors=None is the default)
        results['ElectricityBus'].plot_node_balance()
        results['Battery'].plot_charge_state()

        # Override when needed
        results['ElectricityBus'].plot_node_balance(colors='turbo')  # Ignores setup
        ```

    Design Patterns:
        **Factory Methods**: Use `from_file()` and `from_optimization()` for creation or access directly from `Optimization.results`
        **Dictionary Access**: Use `results[element_label]` for element-specific results
        **Lazy Loading**: Results objects created on-demand for memory efficiency
        **Unified Interface**: Consistent API across different result types

    """

    model: linopy.Model | None

    @classmethod
    def from_file(cls, folder: str | pathlib.Path, name: str) -> Results:
        """Load Results from saved files.

        Args:
            folder: Directory containing saved files.
            name: Base name of saved files (without extensions).

        Returns:
            Results: Loaded instance.
        """
        folder = pathlib.Path(folder)
        paths = fx_io.ResultsPaths(folder, name)

        model = None
        if paths.linopy_model.exists():
            try:
                logger.info(f'loading the linopy model "{name}" from file ("{paths.linopy_model}")')
                model = linopy.read_netcdf(paths.linopy_model)
            except Exception as e:
                logger.critical(f'Could not load the linopy model "{name}" from file ("{paths.linopy_model}"): {e}')

        summary = fx_io.load_yaml(paths.summary)

        return cls(
            solution=fx_io.load_dataset_from_netcdf(paths.solution),
            flow_system_data=fx_io.load_dataset_from_netcdf(paths.flow_system),
            name=name,
            folder=folder,
            model=model,
            summary=summary,
        )

    @classmethod
    def from_optimization(cls, optimization: Optimization) -> Results:
        """Create Results from an Optimization instance.

        Args:
            optimization: The Optimization instance to extract results from.

        Returns:
            Results: New instance containing the optimization results.
        """
        return cls(
            solution=optimization.model.solution,
            flow_system_data=optimization.flow_system.to_dataset(),
            summary=optimization.summary,
            model=optimization.model,
            name=optimization.name,
            folder=optimization.folder,
        )

    def __init__(
        self,
        solution: xr.Dataset,
        flow_system_data: xr.Dataset,
        name: str,
        summary: dict,
        folder: pathlib.Path | None = None,
        model: linopy.Model | None = None,
    ):
        """Initialize Results with optimization data.
        Usually, this class is instantiated by an Optimization object via `Results.from_optimization()`
        or by loading from file using `Results.from_file()`.

        Args:
            solution: Optimization solution dataset.
            flow_system_data: Flow system configuration dataset.
            name: Optimization name.
            summary: Optimization metadata.
            folder: Results storage folder.
            model: Linopy optimization model.
        """

        self.solution = solution
        self.flow_system_data = flow_system_data
        self.summary = summary
        self.name = name
        self.model = model
        self.folder = pathlib.Path(folder) if folder is not None else pathlib.Path.cwd() / 'results'

        # Create ResultsContainers for better access patterns
        components_dict = {
            label: ComponentResults(self, **infos) for label, infos in self.solution.attrs['Components'].items()
        }
        self.components = ResultsContainer(
            elements=components_dict, element_type_name='component results', truncate_repr=10
        )

        buses_dict = {label: BusResults(self, **infos) for label, infos in self.solution.attrs['Buses'].items()}
        self.buses = ResultsContainer(elements=buses_dict, element_type_name='bus results', truncate_repr=10)

        effects_dict = {label: EffectResults(self, **infos) for label, infos in self.solution.attrs['Effects'].items()}
        self.effects = ResultsContainer(elements=effects_dict, element_type_name='effect results', truncate_repr=10)

        if 'Flows' not in self.solution.attrs:
            warnings.warn(
                'No Data about flows found in the results. This data is only included since v2.2.0. Some functionality '
                'is not availlable. We recommend to evaluate your results with a version <2.2.0.',
                stacklevel=2,
            )
            flows_dict = {}
            self._has_flow_data = False
        else:
            flows_dict = {
                label: FlowResults(self, **infos) for label, infos in self.solution.attrs.get('Flows', {}).items()
            }
            self._has_flow_data = True
        self.flows = ResultsContainer(elements=flows_dict, element_type_name='flow results', truncate_repr=10)

        self.timesteps_extra = self.solution.indexes['time']
        self.hours_per_timestep = FlowSystem.calculate_hours_per_timestep(self.timesteps_extra)
        self.scenarios = self.solution.indexes['scenario'] if 'scenario' in self.solution.indexes else None
        self.periods = self.solution.indexes['period'] if 'period' in self.solution.indexes else None

        self._effect_share_factors = None
        self._flow_system = None

        self._flow_rates = None
        self._flow_hours = None
        self._sizes = None
        self._effects_per_component = None

        self.colors: dict[str, str] = {}

        # Plot accessor for new plotting API
        self.plot = PlotAccessor(self)

    def _get_container_groups(self) -> dict[str, ResultsContainer]:
        """Return ordered container groups for CompositeContainerMixin."""
        return {
            'Components': self.components,
            'Buses': self.buses,
            'Effects': self.effects,
            'Flows': self.flows,
        }

    def __repr__(self) -> str:
        """Return grouped representation of all results."""
        r = fx_io.format_title_with_underline(self.__class__.__name__, '=')
        r += f'Name: "{self.name}"\nFolder: {self.folder}\n'
        # Add grouped container view
        r += '\n' + self._format_grouped_containers()
        return r

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
        """The restored flow_system that was used to create the optimization.
        Contains all input parameters."""
        if self._flow_system is None:
            # Temporarily disable all logging to suppress messages during restoration
            flixopt_logger = logging.getLogger('flixopt')
            original_level = flixopt_logger.level
            flixopt_logger.setLevel(logging.CRITICAL + 1)  # Disable all logging
            try:
                self._flow_system = FlowSystem.from_dataset(self.flow_system_data)
                self._flow_system._connect_network()
            except Exception as e:
                flixopt_logger.setLevel(original_level)  # Re-enable before logging
                logger.critical(
                    f'Not able to restore FlowSystem from dataset. Some functionality is not availlable. {e}'
                )
                raise _FlowSystemRestorationError(f'Not able to restore FlowSystem from dataset. {e}') from e
            finally:
                flixopt_logger.setLevel(original_level)  # Restore original level
        return self._flow_system

    def setup_colors(
        self,
        config: dict[str, str | list[str]] | str | pathlib.Path | None = None,
        default_colorscale: str | None = None,
    ) -> dict[str, str]:
        """
        Setup colors for all variables across all elements. Overwrites existing ones.

        Args:
            config: Configuration for color assignment. Can be:
                - dict: Maps components to colors/colorscales:
                    * 'component1': 'red'  # Single component to single color
                    * 'component1': '#FF0000'  # Single component to hex color
                    - OR maps colorscales to multiple components:
                    * 'colorscale_name': ['component1', 'component2']  # Colorscale across components
                - str: Path to a JSON/YAML config file or a colorscale name to apply to all
                - Path: Path to a JSON/YAML config file
                - None: Use default_colorscale for all components
            default_colorscale: Default colorscale for unconfigured components (default: 'turbo')

        Examples:
            setup_colors({
                # Direct component-to-color mappings
                'Boiler1': '#FF0000',
                'CHP': 'darkred',
                # Colorscale for multiple components
                'Oranges': ['Solar1', 'Solar2'],
                'Blues': ['Wind1', 'Wind2'],
                'Greens': ['Battery1', 'Battery2', 'Battery3'],
            })

        Returns:
            Complete variable-to-color mapping dictionary
        """

        def get_all_variable_names(comp: str) -> list[str]:
            """Collect all variables from the component, including flows and flow_hours."""
            comp_object = self.components[comp]
            var_names = [comp] + list(comp_object.variable_names)
            for flow in comp_object.flows:
                var_names.extend([flow, f'{flow}|flow_hours'])
            return var_names

        # Set default colorscale if not provided
        if default_colorscale is None:
            default_colorscale = CONFIG.Plotting.default_qualitative_colorscale

        # Handle different config input types
        if config is None:
            # Apply default colorscale to all components
            config_dict = {}
        elif isinstance(config, (str, pathlib.Path)):
            # Try to load from file first
            config_path = pathlib.Path(config)
            if config_path.exists():
                # Load config from file using helper
                config_dict = load_mapping_from_file(config_path)
            else:
                # Treat as colorscale name to apply to all components
                all_components = list(self.components.keys())
                config_dict = {config: all_components}
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise TypeError(f'config must be dict, str, Path, or None, got {type(config)}')

        # Step 1: Build component-to-color mapping
        component_colors: dict[str, str] = {}

        # Track which components are configured
        configured_components = set()

        # Process each configuration entry
        for key, value in config_dict.items():
            # Check if value is a list (colorscale -> [components])
            # or a string (component -> color OR colorscale -> [components])

            if isinstance(value, list):
                # key is colorscale, value is list of components
                # Format: 'Blues': ['Wind1', 'Wind2']
                components = value
                colorscale_name = key

                # Validate components exist
                for component in components:
                    if component not in self.components:
                        raise ValueError(f"Component '{component}' not found")

                configured_components.update(components)

                # Use process_colors to get one color per component from the colorscale
                colors_for_components = process_colors(colorscale_name, components)
                component_colors.update(colors_for_components)

            elif isinstance(value, str):
                # Check if key is an existing component
                if key in self.components:
                    # Format: 'CHP': 'red' (component -> color)
                    component, color = key, value

                    configured_components.add(component)
                    component_colors[component] = color
                else:
                    raise ValueError(f"Component '{key}' not found")
            else:
                raise TypeError(f'Config value must be str or list, got {type(value)}')

        # Step 2: Assign colors to remaining unconfigured components
        remaining_components = list(set(self.components.keys()) - configured_components)
        if remaining_components:
            # Use default colorscale to assign one color per remaining component
            default_colors = process_colors(default_colorscale, remaining_components)
            component_colors.update(default_colors)

        # Step 3: Build variable-to-color mapping
        # Clear existing colors to avoid stale keys
        self.colors = {}
        # Each component's variables all get the same color as the component
        for component, color in component_colors.items():
            variable_names = get_all_variable_names(component)
            for var_name in variable_names:
                self.colors[var_name] = color

        return self.colors

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

        .. deprecated::
            Use `results.plot.all_flow_rates` (Dataset) or
            `results.flows['FlowLabel'].flow_rate` (DataArray) instead.

            **Note**: The new API differs from this method:

            - Returns ``xr.Dataset`` (not ``DataArray``) with flow labels as variable names
            - No ``'flow'`` dimension - each flow is a separate variable
            - No filtering parameters - use Dataset indexing instead:
              ``results.plot.all_flow_rates[['Flow1', 'Flow2']]``

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
        warnings.warn(
            'results.flow_rates() is deprecated. '
            'Use results.plot.all_flow_rates instead (returns Dataset, not DataArray). '
            'Note: The new API has no filtering parameters and uses flow labels as variable names. '
            f'Will be removed in {DEPRECATION_REMOVAL_VERSION}.',
            DeprecationWarning,
            stacklevel=2,
        )
        if not self._has_flow_data:
            raise ValueError('Flow data is not available in this results object (pre-v2.2.0).')
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

        .. deprecated::
            Use `results.plot.all_flow_hours` (Dataset) or
            `results.flows['FlowLabel'].flow_rate * results.hours_per_timestep` instead.

            **Note**: The new API differs from this method:

            - Returns ``xr.Dataset`` (not ``DataArray``) with flow labels as variable names
            - No ``'flow'`` dimension - each flow is a separate variable
            - No filtering parameters - use Dataset indexing instead:
              ``results.plot.all_flow_hours[['Flow1', 'Flow2']]``

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
        warnings.warn(
            'results.flow_hours() is deprecated. '
            'Use results.plot.all_flow_hours instead (returns Dataset, not DataArray). '
            'Note: The new API has no filtering parameters and uses flow labels as variable names. '
            f'Will be removed in {DEPRECATION_REMOVAL_VERSION}.',
            DeprecationWarning,
            stacklevel=2,
        )
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

        .. deprecated::
            Use `results.plot.all_sizes` (Dataset) or
            `results.flows['FlowLabel'].size` (DataArray) instead.

            **Note**: The new API differs from this method:

            - Returns ``xr.Dataset`` (not ``DataArray``) with flow labels as variable names
            - No ``'flow'`` dimension - each flow is a separate variable
            - No filtering parameters - use Dataset indexing instead:
              ``results.plot.all_sizes[['Flow1', 'Flow2']]``

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
        warnings.warn(
            'results.sizes() is deprecated. '
            'Use results.plot.all_sizes instead (returns Dataset, not DataArray). '
            'Note: The new API has no filtering parameters and uses flow labels as variable names. '
            f'Will be removed in {DEPRECATION_REMOVAL_VERSION}.',
            DeprecationWarning,
            stacklevel=2,
        )
        if not self._has_flow_data:
            raise ValueError('Flow data is not available in this results object (pre-v2.2.0).')
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
        flows_list = list(self.flows.values())
        da = da.assign_coords(
            {
                'start': ('flow', [flow.start for flow in flows_list]),
                'end': ('flow', [flow.end for flow in flows_list]),
                'component': ('flow', [flow.component for flow in flows_list]),
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
            mode: The optimization mode. Options are:
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

    def _create_template_for_mode(self, mode: Literal['temporal', 'periodic', 'total']) -> xr.DataArray:
        """Create a template DataArray with the correct dimensions for a given mode.

        Args:
            mode: The optimization mode ('temporal', 'periodic', or 'total').

        Returns:
            A DataArray filled with NaN, with dimensions appropriate for the mode.
        """
        coords = {}
        if mode == 'temporal':
            coords['time'] = self.timesteps_extra
        if self.periods is not None:
            coords['period'] = self.periods
        if self.scenarios is not None:
            coords['scenario'] = self.scenarios

        # Create template with appropriate shape
        if coords:
            shape = tuple(len(coords[dim]) for dim in coords)
            return xr.DataArray(np.full(shape, np.nan, dtype=float), coords=coords, dims=list(coords.keys()))
        else:
            return xr.DataArray(np.nan)

    def _create_effects_dataset(self, mode: Literal['temporal', 'periodic', 'total']) -> xr.Dataset:
        """Creates a dataset containing effect totals for all components (including their flows).
        The dataset does contain the direct as well as the indirect effects of each component.

        Args:
            mode: The optimization mode ('temporal', 'periodic', or 'total').

        Returns:
            An xarray Dataset with components as dimension and effects as variables.
        """
        # Create template with correct dimensions for this mode
        template = self._create_template_for_mode(mode)

        ds = xr.Dataset()
        all_arrays = {}
        components_list = list(self.components)

        # Collect arrays for all effects and components
        for effect in self.effects:
            effect_arrays = []
            for component in components_list:
                da = self._compute_effect_total(element=component, effect=effect, mode=mode, include_flows=True)
                effect_arrays.append(da)

            all_arrays[effect] = effect_arrays

        # Process all effects: expand scalar NaN arrays to match template dimensions
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
        show: bool | None = None,
        colors: plotting.ColorType | None = None,
        engine: plotting.PlottingEngine = 'plotly',
        select: dict[FlowSystemDimensions, Any] | None = None,
        facet_by: str | list[str] | None = 'scenario',
        animate_by: str | None = 'period',
        facet_cols: int | None = None,
        reshape_time: tuple[Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'], Literal['W', 'D', 'h', '15min', 'min']]
        | Literal['auto']
        | None = 'auto',
        fill: Literal['ffill', 'bfill'] | None = 'ffill',
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
        show: bool | None = None,
    ) -> pyvis.network.Network | None:
        """Plot interactive network visualization of the system.

        Args:
            controls: Enable/disable interactive controls.
            path: Save path for network HTML.
            show: Whether to display the plot. If None, uses CONFIG.Plotting.default_show.
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
        overwrite: bool = False,
    ):
        """Save results to files.

        Args:
            folder: Save folder (defaults to optimization folder).
            name: File name (defaults to optimization name).
            compression: Compression level 0-9.
            document_model: Whether to document model formulations as yaml.
            save_linopy_model: Whether to save linopy model file.
            overwrite: If False, raise error if results files already exist. If True, overwrite existing files.

        Raises:
            FileExistsError: If overwrite=False and result files already exist.
        """
        folder = self.folder if folder is None else pathlib.Path(folder)
        name = self.name if name is None else name

        # Ensure folder exists, creating parent directories as needed
        folder.mkdir(parents=True, exist_ok=True)

        paths = fx_io.ResultsPaths(folder, name)

        # Check if files already exist (unless overwrite is True)
        if not overwrite:
            existing_files = []
            for file_path in paths.all_paths().values():
                if file_path.exists():
                    existing_files.append(file_path.name)

            if existing_files:
                raise FileExistsError(
                    f'Results files already exist in {folder}: {", ".join(existing_files)}. '
                    f'Use overwrite=True to overwrite existing files.'
                )

        fx_io.save_dataset_to_netcdf(self.solution, paths.solution, compression=compression)
        fx_io.save_dataset_to_netcdf(self.flow_system_data, paths.flow_system, compression=compression)

        fx_io.save_yaml(self.summary, paths.summary, compact_numeric_lists=True)

        if save_linopy_model:
            if self.model is None:
                logger.critical('No model in the Results. Saving the model is not possible.')
            else:
                self.model.to_netcdf(paths.linopy_model, engine='netcdf4')

        if document_model:
            if self.model is None:
                logger.critical('No model in the Results. Documenting the model is not possible.')
            else:
                fx_io.document_linopy_model(self.model, path=paths.model_documentation)

        logger.log(SUCCESS_LEVEL, f'Saved optimization results "{name}" to {paths.model_documentation.parent}')


class _ElementResults:
    def __init__(self, results: Results, label: str, variables: list[str], constraints: list[str]):
        self._results = results
        self.label = label
        self.variable_names = variables
        self._constraint_names = constraints

        self.solution = self._results.solution[self.variable_names]

    @property
    def variables(self) -> linopy.Variables:
        """Get element variables (requires linopy model).

        Raises:
            ValueError: If linopy model is unavailable.
        """
        if self._results.model is None:
            raise ValueError('The linopy model is not available.')
        return self._results.model.variables[self.variable_names]

    @property
    def constraints(self) -> linopy.Constraints:
        """Get element constraints (requires linopy model).

        Raises:
            ValueError: If linopy model is unavailable.
        """
        if self._results.model is None:
            raise ValueError('The linopy model is not available.')
        return self._results.model.constraints[self._constraint_names]

    def __repr__(self) -> str:
        """Return string representation with element info and dataset preview."""
        class_name = self.__class__.__name__
        header = f'{class_name}: "{self.label}"'
        sol = self.solution.copy(deep=False)
        sol.attrs = {}
        return f'{header}\n{"-" * len(header)}\n{repr(sol)}'

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
        results: Results,
        label: str,
        variables: list[str],
        constraints: list[str],
        inputs: list[str],
        outputs: list[str],
        flows: list[str],
    ):
        super().__init__(results, label, variables, constraints)
        self.inputs = inputs
        self.outputs = outputs
        self.flows = flows

        # Plot accessor for new plotting API
        self.plot = ElementPlotAccessor(self)

    def plot_node_balance(
        self,
        save: bool | pathlib.Path = False,
        show: bool | None = None,
        colors: plotting.ColorType | None = None,
        engine: plotting.PlottingEngine = 'plotly',
        select: dict[FlowSystemDimensions, Any] | None = None,
        unit_type: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        mode: Literal['area', 'stacked_bar', 'line'] = 'stacked_bar',
        drop_suffix: bool = True,
        facet_by: str | list[str] | None = 'scenario',
        animate_by: str | None = 'period',
        facet_cols: int | None = None,
        **plot_kwargs: Any,
    ) -> plotly.graph_objs.Figure | tuple[plt.Figure, plt.Axes]:
        """
        Plots the node balance of the Component or Bus with optional faceting and animation.

        Args:
            save: Whether to save the plot or not. If a path is provided, the plot will be saved at that location.
            show: Whether to show the plot or not.
            colors: The colors to use for the plot. See `flixopt.plotting.ColorType` for options.
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
                Example: ['period', 'scenario'] creates a grid of subplots for each scenario-period combination.
            animate_by: Dimension to animate over (Plotly only). Creates animation frames that cycle through
                dimension values. Only one dimension can be animated. Ignored if not found.
            facet_cols: Number of columns in the facet grid layout (default: 3).
            **plot_kwargs: Additional plotting customization options passed to underlying plotting functions.

                Common options:

                - **dpi** (int): Export resolution in dots per inch. Default: 300.

                **For Plotly engine** (`engine='plotly'`):

                - Any Plotly Express parameter for px.bar()/px.line()/px.area()
                  Example: `range_y=[0, 100]`, `line_shape='linear'`

                **For Matplotlib engine** (`engine='matplotlib'`):

                - **plot_kwargs** (dict): Customize plot via `ax.bar()` or `ax.step()`.
                  Example: `plot_kwargs={'linewidth': 3, 'alpha': 0.7, 'edgecolor': 'black'}`

                See :func:`flixopt.plotting.with_plotly` and :func:`flixopt.plotting.with_matplotlib`
                for complete parameter reference.

                Note: For Plotly, you can further customize the returned figure using `fig.update_traces()`
                and `fig.update_layout()` after calling this method.

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

            Plotly Express customization (e.g., set y-axis range):

            >>> results['Boiler'].plot_node_balance(range_y=[0, 100])

            Custom matplotlib appearance:

            >>> results['Boiler'].plot_node_balance(engine='matplotlib', plot_kwargs={'linewidth': 3, 'alpha': 0.7})

            Further customize Plotly figure after creation:

            >>> fig = results['Boiler'].plot_node_balance(mode='line', show=False)
            >>> fig.update_traces(line={'width': 5, 'dash': 'dot'})
            >>> fig.update_layout(template='plotly_dark', width=1200, height=600)
            >>> fig.show()
        """
        if engine not in {'plotly', 'matplotlib'}:
            raise ValueError(f'Engine "{engine}" not supported. Use one of ["plotly", "matplotlib"]')

        # Extract dpi for export_figure
        dpi = plot_kwargs.pop('dpi', None)  # None uses CONFIG.Plotting.default_dpi

        # Don't pass select/indexer to node_balance - we'll apply it afterwards
        ds = self.node_balance(with_last_timestep=False, unit_type=unit_type, drop_suffix=drop_suffix)

        ds, suffix_parts = _apply_selection_to_data(ds, select=select, drop=True)

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
                colors=colors if colors is not None else self._results.colors,
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
                colors=colors if colors is not None else self._results.colors,
                mode=mode,
                title=title,
                **plot_kwargs,
            )
            default_filetype = '.png'

        return plotting.export_figure(
            figure_like=figure_like,
            default_path=self._results.folder / title,
            default_filetype=default_filetype,
            user_path=None if isinstance(save, bool) else pathlib.Path(save),
            show=show,
            save=True if save else False,
            dpi=dpi,
        )

    def plot_node_balance_pie(
        self,
        lower_percentage_group: float = 5,
        colors: plotting.ColorType | None = None,
        text_info: str = 'percent+label+value',
        save: bool | pathlib.Path = False,
        show: bool | None = None,
        engine: plotting.PlottingEngine = 'plotly',
        select: dict[FlowSystemDimensions, Any] | None = None,
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
        # Extract dpi for export_figure
        dpi = plot_kwargs.pop('dpi', None)  # None uses CONFIG.Plotting.default_dpi

        inputs = sanitize_dataset(
            ds=self.solution[self.inputs] * self._results.hours_per_timestep,
            threshold=1e-5,
            drop_small_vars=True,
            zero_small_values=True,
            drop_suffix='|',
        )
        outputs = sanitize_dataset(
            ds=self.solution[self.outputs] * self._results.hours_per_timestep,
            threshold=1e-5,
            drop_small_vars=True,
            zero_small_values=True,
            drop_suffix='|',
        )

        inputs, suffix_parts_in = _apply_selection_to_data(inputs, select=select, drop=True)
        outputs, suffix_parts_out = _apply_selection_to_data(outputs, select=select, drop=True)
        suffix_parts = suffix_parts_in + suffix_parts_out

        # Sum over time dimension
        inputs = inputs.sum('time')
        outputs = outputs.sum('time')

        # Auto-select first value for any remaining dimensions (scenario, period, etc.)
        # Pie charts need scalar data, so we automatically reduce extra dimensions
        extra_dims_inputs = [dim for dim in inputs.dims if dim != 'time']
        extra_dims_outputs = [dim for dim in outputs.dims if dim != 'time']
        extra_dims = sorted(set(extra_dims_inputs + extra_dims_outputs))

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

            # Apply auto-selection only for coords present in each dataset
            inputs = inputs.sel({k: v for k, v in auto_select.items() if k in inputs.coords})
            outputs = outputs.sel({k: v for k, v in auto_select.items() if k in outputs.coords})

            # Update suffix with auto-selected values
            auto_suffix_parts = [f'{dim}={val}' for dim, val in auto_select.items()]
            suffix_parts.extend(auto_suffix_parts)

        suffix = '--' + '-'.join(sorted(set(suffix_parts))) if suffix_parts else ''
        title = f'{self.label} (total flow hours){suffix}'

        if engine == 'plotly':
            figure_like = plotting.dual_pie_with_plotly(
                data_left=inputs,
                data_right=outputs,
                colors=colors if colors is not None else self._results.colors,
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
                colors=colors if colors is not None else self._results.colors,
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
            default_path=self._results.folder / title,
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
        ds = self.solution[self.inputs + self.outputs]

        ds = sanitize_dataset(
            ds=ds,
            threshold=threshold,
            timesteps=self._results.timesteps_extra if with_last_timestep else None,
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
            ds = ds * self._results.hours_per_timestep
            ds = ds.rename_vars({var: var.replace('flow_rate', 'flow_hours') for var in ds.data_vars})

        return ds


class BusResults(_NodeResults):
    """Results container for energy/material balance nodes in the system."""


class ComponentResults(_NodeResults):
    """Results container for individual system components with specialized analysis tools."""

    @property
    def is_storage(self) -> bool:
        return self._charge_state in self.variable_names

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
        show: bool | None = None,
        colors: plotting.ColorType | None = None,
        engine: plotting.PlottingEngine = 'plotly',
        mode: Literal['area', 'stacked_bar', 'line'] = 'area',
        select: dict[FlowSystemDimensions, Any] | None = None,
        facet_by: str | list[str] | None = 'scenario',
        animate_by: str | None = 'period',
        facet_cols: int | None = None,
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

                - Any Plotly Express parameter for px.bar()/px.line()/px.area()
                  Example: `range_y=[0, 100]`, `line_shape='linear'`

                **For Matplotlib engine:**

                - **plot_kwargs** (dict): Customize plot via `ax.bar()` or `ax.step()`.

                See :func:`flixopt.plotting.with_plotly` and :func:`flixopt.plotting.with_matplotlib`
                for complete parameter reference.

                Note: For Plotly, you can further customize the returned figure using `fig.update_traces()`
                and `fig.update_layout()` after calling this method.

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

            Custom layout after creation:

            >>> fig = results['Storage'].plot_charge_state(show=False)
            >>> fig.update_layout(template='plotly_dark', height=800)
            >>> fig.show()

            High-resolution export:

            >>> results['Storage'].plot_charge_state(save='storage.png', dpi=600)
        """
        # Extract dpi for export_figure
        dpi = plot_kwargs.pop('dpi', None)  # None uses CONFIG.Plotting.default_dpi

        # Extract charge state line color (for overlay customization)
        overlay_color = plot_kwargs.pop('charge_state_line_color', 'black')

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

        if engine == 'plotly':
            # Plot flows (node balance) with the specified mode
            figure_like = plotting.with_plotly(
                ds,
                facet_by=facet_by,
                animate_by=animate_by,
                colors=colors if colors is not None else self._results.colors,
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
                colors=colors if colors is not None else self._results.colors,
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
                trace.line.color = overlay_color
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
                            trace.line.color = overlay_color
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
                colors=colors if colors is not None else self._results.colors,
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
                color=overlay_color,
            )
            # Recreate legend with the same styling as with_matplotlib
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                labels,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                ncol=5,
                frameon=False,
            )
            fig.tight_layout()

            figure_like = fig, ax
            default_filetype = '.png'

        return plotting.export_figure(
            figure_like=figure_like,
            default_path=self._results.folder / title,
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
            timesteps=self._results.timesteps_extra,
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
        return self.solution[[name for name in self.variable_names if name.startswith(f'{element}->')]]


class FlowResults(_ElementResults):
    def __init__(
        self,
        results: Results,
        label: str,
        variables: list[str],
        constraints: list[str],
        start: str,
        end: str,
        component: str,
    ):
        super().__init__(results, label, variables, constraints)
        self.start = start
        self.end = end
        self.component = component

    @property
    def flow_rate(self) -> xr.DataArray:
        return self.solution[f'{self.label}|flow_rate']

    @property
    def flow_hours(self) -> xr.DataArray:
        return (self.flow_rate * self._results.hours_per_timestep).rename(f'{self.label}|flow_hours')

    @property
    def size(self) -> xr.DataArray:
        name = f'{self.label}|size'
        if name in self.solution:
            return self.solution[name]
        try:
            return self._results.flow_system.flows[self.label].size.rename(name)
        except _FlowSystemRestorationError:
            logger.critical(f'Size of flow {self.label}.size not availlable. Returning NaN')
            return xr.DataArray(np.nan).rename(name)


class SegmentedResults:
    """Results container for segmented optimization optimizations with temporal decomposition.

    This class manages results from SegmentedOptimization runs where large optimization
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
        # Load segmented optimization results
        results = SegmentedResults.from_file('results', 'annual_segmented')

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

        Create from segmented optimization:

        ```python
        # After running segmented optimization
        segmented_opt = SegmentedOptimization(
            name='annual_system',
            flow_system=system,
            timesteps_per_segment=730,  # Monthly segments
            overlap_timesteps=48,  # 2-day overlap
        )
        segmented_opt.do_modeling_and_solve(solver='gurobi')

        # Extract unified results
        results = SegmentedResults.from_optimization(segmented_opt)

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
    def from_optimization(cls, optimization: SegmentedOptimization) -> SegmentedResults:
        """Create SegmentedResults from a SegmentedOptimization instance.

        Args:
            optimization: The SegmentedOptimization instance to extract results from.

        Returns:
            SegmentedResults: New instance containing the optimization results.
        """
        return cls(
            [calc.results for calc in optimization.sub_optimizations],
            all_timesteps=optimization.all_timesteps,
            timesteps_per_segment=optimization.timesteps_per_segment,
            overlap_timesteps=optimization.overlap_timesteps,
            name=optimization.name,
            folder=optimization.folder,
        )

    @classmethod
    def from_file(cls, folder: str | pathlib.Path, name: str) -> SegmentedResults:
        """Load SegmentedResults from saved files.

        Args:
            folder: Directory containing saved files.
            name: Base name of saved files.

        Returns:
            SegmentedResults: Loaded instance.
        """
        folder = pathlib.Path(folder)
        path = folder / name
        meta_data_path = path.with_suffix('.json')
        logger.info(f'loading segemented optimization meta data from file ("{meta_data_path}")')
        meta_data = fx_io.load_json(meta_data_path)

        # Handle both new 'sub_optimizations' and legacy 'sub_calculations' keys
        sub_names = meta_data.get('sub_optimizations') or meta_data.get('sub_calculations')
        if sub_names is None:
            raise KeyError(
                "Missing 'sub_optimizations' (or legacy 'sub_calculations') key in segmented results metadata."
            )

        return cls(
            [Results.from_file(folder, sub_name) for sub_name in sub_names],
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
        segment_results: list[Results],
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
        self._colors = {}

    @property
    def meta_data(self) -> dict[str, int | list[str]]:
        return {
            'all_timesteps': [datetime.datetime.isoformat(date) for date in self.all_timesteps],
            'timesteps_per_segment': self.timesteps_per_segment,
            'overlap_timesteps': self.overlap_timesteps,
            'sub_optimizations': [calc.name for calc in self.segment_results],
        }

    @property
    def segment_names(self) -> list[str]:
        return [segment.name for segment in self.segment_results]

    @property
    def colors(self) -> dict[str, str]:
        return self._colors

    @colors.setter
    def colors(self, colors: dict[str, str]):
        """Applies colors to all segments"""
        self._colors = colors
        for segment in self.segment_results:
            segment.colors = copy.deepcopy(colors)

    def setup_colors(
        self,
        config: dict[str, str | list[str]] | str | pathlib.Path | None = None,
        default_colorscale: str | None = None,
    ) -> dict[str, str]:
        """
        Setup colors for all variables across all segment results.

        This method applies the same color configuration to all segments, ensuring
        consistent visualization across the entire segmented optimization. The color
        mapping is propagated to each segment's Results instance.

        Args:
            config: Configuration for color assignment. Can be:
                - dict: Maps components to colors/colorscales:
                    * 'component1': 'red'  # Single component to single color
                    * 'component1': '#FF0000'  # Single component to hex color
                    - OR maps colorscales to multiple components:
                    * 'colorscale_name': ['component1', 'component2']  # Colorscale across components
                - str: Path to a JSON/YAML config file or a colorscale name to apply to all
                - Path: Path to a JSON/YAML config file
                - None: Use default_colorscale for all components
            default_colorscale: Default colorscale for unconfigured components (default: 'turbo')

        Examples:
            ```python
            # Apply colors to all segments
            segmented_results.setup_colors(
                {
                    'CHP': 'red',
                    'Blues': ['Storage1', 'Storage2'],
                    'Oranges': ['Solar1', 'Solar2'],
                }
            )

            # Use a single colorscale for all components in all segments
            segmented_results.setup_colors('portland')
            ```

        Returns:
            Complete variable-to-color mapping dictionary from the first segment
            (all segments will have the same mapping)
        """
        if not self.segment_results:
            raise ValueError('No segment_results available; cannot setup colors on an empty SegmentedResults.')

        self.colors = self.segment_results[0].setup_colors(config=config, default_colorscale=default_colorscale)

        return self.colors

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
        colors: plotting.ColorType | None = None,
        save: bool | pathlib.Path = False,
        show: bool | None = None,
        engine: plotting.PlottingEngine = 'plotly',
        facet_by: str | list[str] | None = None,
        animate_by: str | None = None,
        facet_cols: int | None = None,
        fill: Literal['ffill', 'bfill'] | None = 'ffill',
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

    def to_file(
        self,
        folder: str | pathlib.Path | None = None,
        name: str | None = None,
        compression: int = 5,
        overwrite: bool = False,
    ):
        """Save segmented results to files.

        Args:
            folder: Save folder (defaults to instance folder).
            name: File name (defaults to instance name).
            compression: Compression level 0-9.
            overwrite: If False, raise error if results files already exist. If True, overwrite existing files.

        Raises:
            FileExistsError: If overwrite=False and result files already exist.
        """
        folder = self.folder if folder is None else pathlib.Path(folder)
        name = self.name if name is None else name
        path = folder / name

        # Ensure folder exists, creating parent directories as needed
        folder.mkdir(parents=True, exist_ok=True)

        # Check if metadata file already exists (unless overwrite is True)
        metadata_file = path.with_suffix('.json')
        if not overwrite and metadata_file.exists():
            raise FileExistsError(
                f'Segmented results file already exists: {metadata_file}. '
                f'Use overwrite=True to overwrite existing files.'
            )

        # Save segments (they will check for overwrite themselves)
        for segment in self.segment_results:
            segment.to_file(folder=folder, name=segment.name, compression=compression, overwrite=overwrite)

        fx_io.save_json(self.meta_data, metadata_file)
        logger.info(f'Saved optimization "{name}" to {path}')


def plot_heatmap(
    data: xr.DataArray | xr.Dataset,
    name: str | None = None,
    folder: pathlib.Path | None = None,
    colors: plotting.ColorType | None = None,
    save: bool | pathlib.Path = False,
    show: bool | None = None,
    engine: plotting.PlottingEngine = 'plotly',
    select: dict[str, Any] | None = None,
    facet_by: str | list[str] | None = None,
    animate_by: str | None = None,
    facet_cols: int | None = None,
    reshape_time: tuple[Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'], Literal['W', 'D', 'h', '15min', 'min']]
    | Literal['auto']
    | None = 'auto',
    fill: Literal['ffill', 'bfill'] | None = 'ffill',
    **plot_kwargs: Any,
):
    """Plot heatmap visualization with support for multi-variable, faceting, and animation.

    This function provides a standalone interface to the heatmap plotting capabilities,
    supporting the same modern features as Results.plot_heatmap().

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
    dpi = plot_kwargs.pop('dpi', None)  # None uses CONFIG.Plotting.default_dpi

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
