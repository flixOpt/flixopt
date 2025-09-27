from __future__ import annotations

import datetime
import json
import logging
import pathlib
import warnings
from typing import TYPE_CHECKING, Literal

import linopy
import numpy as np
import pandas as pd
import plotly
import xarray as xr
import yaml

from . import io as fx_io
from . import plotting
from .core import TimeSeriesCollection

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import pyvis

    from .calculation import Calculation, SegmentedCalculation


logger = logging.getLogger('flixopt')


class BackwardsCompatibleDataset:
    """Wrapper around xarray.Dataset to provide backwards compatibility for renamed variables.

    This class handles multiple types of backwards compatibility:
    1. Direct variable name mappings (e.g., 'costs|total' -> 'costs')
    2. Pattern-based substitutions (e.g., (operation) -> (temporal))
    3. Dynamic renaming rules for complex variable naming schemes
    """

    # Direct mapping from old variable names to new variable names
    DEPRECATED_VARIABLE_MAPPING = {
        # Effect variable names
        'costs|total': 'costs',
    }

    # Pattern-based substitution rules (old_pattern, new_pattern, condition_func)
    PATTERN_SUBSTITUTION_RULES = [
        # Cross-effect variables: (operation) -> (temporal), (invest) -> (nontemporal)
        ('(operation)', '(temporal)', lambda key: '->' in key),
        ('(invest)', '(nontemporal)', lambda key: '->' in key),
    ]

    # Regex-based renaming patterns for more complex cases
    @staticmethod
    def _get_regex_patterns():
        """Get regex patterns for complex variable renaming.

        Returns:
            List of tuples (pattern, replacement, description) for regex-based renaming.
        """
        import re

        return [
            # Pattern for effect parameter names: *_operation -> *_temporal
            (re.compile(r'(.+)_operation($|_)'), r'\1_temporal\2', 'operation suffix to temporal'),
            # Pattern for effect parameter names: *_invest -> *_nontemporal
            (re.compile(r'(.+)_invest($|_)'), r'\1_nontemporal\2', 'invest suffix to nontemporal'),
            # Pattern for per-hour variants: *_operation_per_hour -> *_temporal_per_hour
            (
                re.compile(r'(.+)_operation_per_hour'),
                r'\1_temporal_per_hour',
                'operation_per_hour to temporal_per_hour',
            ),
        ]

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset
        # Cache for resolved mappings to improve performance
        self._mapping_cache = {}

    def _resolve_deprecated_key(self, key):
        """Resolve a deprecated variable name to its current equivalent.

        Args:
            key: Variable name to resolve

        Returns:
            Tuple of (new_key, found) where found indicates if a mapping was found
        """
        # Check cache first
        if key in self._mapping_cache:
            return self._mapping_cache[key], True

        # 1. Direct mapping
        if key in self.DEPRECATED_VARIABLE_MAPPING:
            new_key = self.DEPRECATED_VARIABLE_MAPPING[key]
            if new_key in self._dataset:
                self._mapping_cache[key] = new_key
                return new_key, True

        # 2. Pattern-based substitution
        for old_pattern, new_pattern, condition_func in self.PATTERN_SUBSTITUTION_RULES:
            if condition_func(key) and old_pattern in key:
                new_key = key.replace(old_pattern, new_pattern)
                if new_key in self._dataset:
                    self._mapping_cache[key] = new_key
                    return new_key, True

        # 3. Regex-based patterns
        for pattern, replacement, _description in self._get_regex_patterns():
            if pattern.search(key):
                new_key = pattern.sub(replacement, key)
                if new_key in self._dataset:
                    self._mapping_cache[key] = new_key
                    return new_key, True

        # 4. Check if key exists as-is in dataset
        if key in self._dataset:
            return key, True

        return key, False

    def __getitem__(self, key):
        """Access dataset variables with backwards compatibility."""
        new_key, found = self._resolve_deprecated_key(key)

        if found and new_key != key:
            # Issue deprecation warning for renamed variables
            warnings.warn(
                f"Variable name '{key}' is deprecated. Use '{new_key}' instead.", DeprecationWarning, stacklevel=2
            )
            return self._dataset[new_key]
        elif found:
            # Key exists as-is
            return self._dataset[key]
        else:
            # Key not found - let dataset handle the error
            return self._dataset[key]

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped dataset."""
        return getattr(self._dataset, name)

    def __contains__(self, key):
        """Check if key exists in dataset (with backwards compatibility)."""
        new_key, found = self._resolve_deprecated_key(key)
        return found

    def __iter__(self):
        """Iterate over dataset variables."""
        return iter(self._dataset)

    def __len__(self):
        """Get number of variables in dataset."""
        return len(self._dataset)

    def keys(self):
        """Get dataset variable names."""
        return self._dataset.keys()

    def values(self):
        """Get dataset variable values."""
        return self._dataset.values()

    def items(self):
        """Get dataset variable items."""
        return self._dataset.items()

    def get_deprecated_mappings(self):
        """Get all currently active deprecated mappings for debugging.

        Returns:
            Dict mapping deprecated names to current names for variables in this dataset.
        """
        mappings = {}

        # Check all variables in dataset against all deprecation rules
        for var_name in self._dataset.data_vars:
            # Check if any deprecated patterns would map TO this variable

            # Direct mappings (reverse lookup)
            for old_name, new_name in self.DEPRECATED_VARIABLE_MAPPING.items():
                if new_name == var_name and old_name not in self._dataset:
                    mappings[old_name] = var_name

            # Pattern-based rules (generate potential old names)
            for old_pattern, new_pattern, condition_func in self.PATTERN_SUBSTITUTION_RULES:
                if new_pattern in var_name:
                    potential_old_name = var_name.replace(new_pattern, old_pattern)
                    if condition_func(potential_old_name) and potential_old_name not in self._dataset:
                        mappings[potential_old_name] = var_name

            # Regex patterns (reverse lookup)
            # This is complex for reverse lookup, so we'll skip for now
            # Could be added if needed for debugging

        return mappings

    @property
    def _raw_dataset(self):
        """Access to the underlying dataset without backwards compatibility."""
        return self._dataset


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
        flow_system: Dataset with complete system configuration and parameters. Restore the used FlowSystem for further analysis.
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
            flow_system=fx_io.load_dataset_from_netcdf(paths.flow_system),
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
        summary: dict,
        folder: pathlib.Path | None = None,
        model: linopy.Model | None = None,
    ):
        """Initialize CalculationResults with optimization data.
        Usually, this class is instantiated by the Calculation class, or by loading from file.

        Args:
            solution: Optimization solution dataset.
            flow_system: Flow system configuration dataset.
            name: Calculation name.
            summary: Calculation metadata.
            folder: Results storage folder.
            model: Linopy optimization model.
        """
        self.solution = BackwardsCompatibleDataset(solution)
        self.flow_system = flow_system
        self.summary = summary
        self.name = name
        self.model = model
        self.folder = pathlib.Path(folder) if folder is not None else pathlib.Path.cwd() / 'results'
        self.components = {
            label: ComponentResults.from_json(self, infos)
            for label, infos in self.solution._raw_dataset.attrs['Components'].items()
        }

        self.buses = {
            label: BusResults.from_json(self, infos)
            for label, infos in self.solution._raw_dataset.attrs['Buses'].items()
        }

        self.effects = {
            label: EffectResults.from_json(self, infos)
            for label, infos in self.solution._raw_dataset.attrs['Effects'].items()
        }

        self.timesteps_extra = self.solution.indexes['time']
        self.hours_per_timestep = TimeSeriesCollection.calculate_hours_per_timestep(self.timesteps_extra)

    def __getitem__(self, key: str) -> ComponentResults | BusResults | EffectResults:
        if key in self.components:
            return self.components[key]
        if key in self.buses:
            return self.buses[key]
        if key in self.effects:
            return self.effects[key]
        raise KeyError(f'No element with label {key} found.')

    @property
    def storages(self) -> list[ComponentResults]:
        """Get all storage components in the results."""
        return [comp for comp in self.components.values() if comp.is_storage]

    @property
    def objective(self) -> float:
        """Get optimization objective value."""
        return self.summary['Main Results']['Objective']

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

    def filter_solution(
        self, variable_dims: Literal['scalar', 'time'] | None = None, element: str | None = None
    ) -> xr.Dataset:
        """Filter solution by variable dimension and/or element.

        Args:
            variable_dims: Variable dimension to filter ('scalar' or 'time').
            element: Element label to filter.

        Returns:
            xr.Dataset: Filtered solution dataset.
        """
        if element is not None:
            return filter_dataset(self[element].solution, variable_dims)
        return filter_dataset(self.solution._raw_dataset, variable_dims)

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

        fx_io.save_dataset_to_netcdf(self.solution._raw_dataset, paths.solution, compression=compression)
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
    def from_json(cls, calculation_results, json_data: dict) -> _ElementResults:
        return cls(calculation_results, json_data['label'], json_data['variables'], json_data['constraints'])

    def __init__(
        self, calculation_results: CalculationResults, label: str, variables: list[str], constraints: list[str]
    ):
        self._calculation_results = calculation_results
        self.label = label
        self._variable_names = variables
        self._constraint_names = constraints

        self.solution = self._calculation_results.solution._raw_dataset[self._variable_names]

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

    def filter_solution(self, variable_dims: Literal['scalar', 'time'] | None = None) -> xr.Dataset:
        """Filter element solution by dimension.

        Args:
            variable_dims: Variable dimension to filter.

        Returns:
            xr.Dataset: Filtered solution dataset.
        """
        return filter_dataset(self.solution, variable_dims)


class _NodeResults(_ElementResults):
    @classmethod
    def from_json(cls, calculation_results, json_data: dict) -> _NodeResults:
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
        variables: list[str],
        constraints: list[str],
        inputs: list[str],
        outputs: list[str],
    ):
        super().__init__(calculation_results, label, variables, constraints)
        self.inputs = inputs
        self.outputs = outputs

    def plot_node_balance(
        self,
        save: bool | pathlib.Path = False,
        show: bool = True,
        colors: plotting.ColorType = 'viridis',
        engine: plotting.PlottingEngine = 'plotly',
    ) -> plotly.graph_objs.Figure | tuple[plt.Figure, plt.Axes]:
        """Plot node balance flows.

        Args:
            save: Whether to save plot (path or boolean).
            show: Whether to display plot.
            colors: Color scheme. Also see plotly.
            engine: Plotting engine ('plotly' or 'matplotlib').

        Returns:
            Figure object.
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
        save: bool | pathlib.Path = False,
        show: bool = True,
        engine: plotting.PlottingEngine = 'plotly',
    ) -> plotly.graph_objs.Figure | tuple[plt.Figure, list[plt.Axes]]:
        """Plot pie chart of flow hours distribution.

        Args:
            lower_percentage_group: Percentage threshold for "Others" grouping.
            colors: Color scheme. Also see plotly.
            text_info: Information to display on pie slices.
            save: Whether to save plot.
            show: Whether to display plot.
            engine: Plotting engine ('plotly' or 'matplotlib').
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
        threshold: float | None = 1e-5,
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
    ) -> plotly.graph_objs.Figure:
        """Plot storage charge state over time, combined with the node balance.

        Args:
            save: Whether to save plot.
            show: Whether to display plot.
            colors: Color scheme. Also see plotly.
            engine: Plotting engine (only 'plotly' supported).

        Returns:
            plotly.graph_objs.Figure: Charge state plot.

        Raises:
            ValueError: If component is not a storage.
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
        - **Large-Scale Analysis**: Annual or multi-year optimization results
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
        self.hours_per_timestep = TimeSeriesCollection.calculate_hours_per_timestep(self.all_timesteps)

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
    save: bool | pathlib.Path = False,
    show: bool = True,
    engine: plotting.PlottingEngine = 'plotly',
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

    Returns:
        Figure object.
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
    timesteps: pd.DatetimeIndex | None = None,
    threshold: float | None = 1e-5,
    negate: list[str] | None = None,
    drop_small_vars: bool = True,
    zero_small_values: bool = False,
) -> xr.Dataset:
    """Clean dataset by handling small values and reindexing time.

    Args:
        ds: Dataset to sanitize.
        timesteps: Time index for reindexing (optional).
        threshold: Threshold for small values processing.
        negate: Variables to negate.
        drop_small_vars: Whether to drop variables below threshold.
        zero_small_values: Whether to zero values below threshold.

    Returns:
        xr.Dataset: Sanitized dataset.
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

    return ds


def filter_dataset(
    ds: xr.Dataset,
    variable_dims: Literal['scalar', 'time'] | None = None,
) -> xr.Dataset:
    """Filter dataset by variable dimensions.

    Args:
        ds: Dataset to filter.
        variable_dims: Variable dimension to filter ('scalar' or 'time').

    Returns:
        xr.Dataset: Filtered dataset.
    """
    if variable_dims is None:
        return ds

    if variable_dims == 'scalar':
        return ds[[name for name, da in ds.data_vars.items() if len(da.dims) == 0]]
    elif variable_dims == 'time':
        return ds[[name for name, da in ds.data_vars.items() if 'time' in da.dims]]
    else:
        raise ValueError(f'Not allowed value for "filter_dataset()": {variable_dims=}')
