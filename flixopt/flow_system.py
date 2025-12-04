"""
This module contains the FlowSystem class, which is used to collect instances of many other classes by the end User.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from . import io as fx_io
from .config import CONFIG
from .core import (
    ConversionError,
    DataConverter,
    FlowSystemDimensions,
    TimeSeriesData,
)
from .effects import Effect, EffectCollection
from .elements import Bus, Component, Flow
from .optimize_accessor import OptimizeAccessor
from .structure import CompositeContainerMixin, Element, ElementContainer, FlowSystemModel, Interface

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Collection

    import pyvis

    from .solvers import _Solver
    from .types import Effect_TPS, Numeric_S, Numeric_TPS, NumericOrBool

logger = logging.getLogger('flixopt')


class FlowSystem(Interface, CompositeContainerMixin[Element]):
    """
    A FlowSystem organizes the high level Elements (Components, Buses, Effects & Flows).

    This is the main container class that users work with to build and manage their energy or material flow system.
    FlowSystem provides both direct container access (via .components, .buses, .effects, .flows) and a unified
    dict-like interface for accessing any element by label across all container types.

    Args:
        timesteps: The timesteps of the model.
        periods: The periods of the model.
        scenarios: The scenarios of the model.
        hours_of_last_timestep: Duration of the last timestep. If None, computed from the last time interval.
        hours_of_previous_timesteps: Duration of previous timesteps. If None, computed from the first time interval.
            Can be a scalar (all previous timesteps have same duration) or array (different durations).
            Used to calculate previous values (e.g., uptime and downtime).
        weight_of_last_period: Weight/duration of the last period. If None, computed from the last period interval.
            Used for calculating sums over periods in multi-period models.
        scenario_weights: The weights of each scenario. If None, all scenarios have the same weight (normalized to 1).
            Period weights are always computed internally from the period index (like hours_per_timestep for time).
            The final `weights` array (accessible via `flow_system.model.objective_weights`) is computed as period_weights × normalized_scenario_weights, with normalization applied to the scenario weights by default.
        scenario_independent_sizes: Controls whether investment sizes are equalized across scenarios.
            - True: All sizes are shared/equalized across scenarios
            - False: All sizes are optimized separately per scenario
            - list[str]: Only specified components (by label_full) are equalized across scenarios
        scenario_independent_flow_rates: Controls whether flow rates are equalized across scenarios.
            - True: All flow rates are shared/equalized across scenarios
            - False: All flow rates are optimized separately per scenario
            - list[str]: Only specified flows (by label_full) are equalized across scenarios

    Examples:
        Creating a FlowSystem and accessing elements:

        >>> import flixopt as fx
        >>> import pandas as pd
        >>> timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
        >>> flow_system = fx.FlowSystem(timesteps)
        >>>
        >>> # Add elements to the system
        >>> boiler = fx.Component('Boiler', inputs=[heat_flow], status_parameters=...)
        >>> heat_bus = fx.Bus('Heat', imbalance_penalty_per_flow_hour=1e4)
        >>> costs = fx.Effect('costs', is_objective=True, is_standard=True)
        >>> flow_system.add_elements(boiler, heat_bus, costs)

        Unified dict-like access (recommended for most cases):

        >>> # Access any element by label, regardless of type
        >>> boiler = flow_system['Boiler']  # Returns Component
        >>> heat_bus = flow_system['Heat']  # Returns Bus
        >>> costs = flow_system['costs']  # Returns Effect
        >>>
        >>> # Check if element exists
        >>> if 'Boiler' in flow_system:
        ...     print('Boiler found in system')
        >>>
        >>> # Iterate over all elements
        >>> for label in flow_system.keys():
        ...     element = flow_system[label]
        ...     print(f'{label}: {type(element).__name__}')
        >>>
        >>> # Get all element labels and objects
        >>> all_labels = list(flow_system.keys())
        >>> all_elements = list(flow_system.values())
        >>> for label, element in flow_system.items():
        ...     print(f'{label}: {element}')

        Direct container access for type-specific operations:

        >>> # Access specific container when you need type filtering
        >>> for component in flow_system.components.values():
        ...     print(f'{component.label}: {len(component.inputs)} inputs')
        >>>
        >>> # Access buses directly
        >>> for bus in flow_system.buses.values():
        ...     print(f'{bus.label}')
        >>>
        >>> # Flows are automatically collected from all components

        Power user pattern - Efficient chaining without conversion overhead:

        >>> # Instead of chaining (causes multiple conversions):
        >>> result = flow_system.sel(time='2020-01').resample('2h')  # Slow
        >>>
        >>> # Use dataset methods directly (single conversion):
        >>> ds = flow_system.to_dataset()
        >>> ds = FlowSystem._dataset_sel(ds, time='2020-01')
        >>> ds = flow_system._dataset_resample(ds, freq='2h', method='mean')
        >>> result = FlowSystem.from_dataset(ds)  # Fast!
        >>>
        >>> # Available dataset methods:
        >>> # - FlowSystem._dataset_sel(dataset, time=..., period=..., scenario=...)
        >>> # - FlowSystem._dataset_isel(dataset, time=..., period=..., scenario=...)
        >>> # - flow_system._dataset_resample(dataset, freq=..., method=..., **kwargs)
        >>> for flow in flow_system.flows.values():
        ...     print(f'{flow.label_full}: {flow.size}')
        >>>
        >>> # Access effects
        >>> for effect in flow_system.effects.values():
        ...     print(f'{effect.label}')

    Notes:
        - The dict-like interface (`flow_system['element']`) searches across all containers
          (components, buses, effects, flows) to find the element with the matching label.
        - Element labels must be unique across all container types. Attempting to add
          elements with duplicate labels will raise an error, ensuring each label maps to exactly one element.
        - Direct container access (`.components`, `.buses`, `.effects`, `.flows`) is useful
          when you need type-specific filtering or operations.
        - The `.flows` container is automatically populated from all component inputs and outputs.
        - Creates an empty registry for components and buses, an empty EffectCollection, and a placeholder for a SystemModel.
        - The instance starts disconnected (self._connected_and_transformed == False) and will be
          connected_and_transformed automatically when trying to optimize.
    """

    model: FlowSystemModel | None

    def __init__(
        self,
        timesteps: pd.DatetimeIndex,
        periods: pd.Index | None = None,
        scenarios: pd.Index | None = None,
        hours_of_last_timestep: int | float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
        weight_of_last_period: int | float | None = None,
        scenario_weights: Numeric_S | None = None,
        scenario_independent_sizes: bool | list[str] = True,
        scenario_independent_flow_rates: bool | list[str] = False,
    ):
        self.timesteps = self._validate_timesteps(timesteps)

        # Compute all time-related metadata using shared helper
        (
            self.timesteps_extra,
            self.hours_of_last_timestep,
            self.hours_of_previous_timesteps,
            hours_per_timestep,
        ) = self._compute_time_metadata(self.timesteps, hours_of_last_timestep, hours_of_previous_timesteps)

        self.periods = None if periods is None else self._validate_periods(periods)
        self.scenarios = None if scenarios is None else self._validate_scenarios(scenarios)

        self.hours_per_timestep = self.fit_to_model_coords('hours_per_timestep', hours_per_timestep)

        self.scenario_weights = scenario_weights  # Use setter

        # Compute all period-related metadata using shared helper
        (self.periods_extra, self.weight_of_last_period, weight_per_period) = self._compute_period_metadata(
            self.periods, weight_of_last_period
        )

        self.period_weights: xr.DataArray | None = weight_per_period

        # Element collections
        self.components: ElementContainer[Component] = ElementContainer(
            element_type_name='components', truncate_repr=10
        )
        self.buses: ElementContainer[Bus] = ElementContainer(element_type_name='buses', truncate_repr=10)
        self.effects: EffectCollection = EffectCollection(truncate_repr=10)
        self.model: FlowSystemModel | None = None

        self._connected_and_transformed = False
        self._used_in_optimization = False

        self._network_app = None
        self._flows_cache: ElementContainer[Flow] | None = None

        # Solution dataset - populated after optimization or loaded from file
        self.solution: xr.Dataset | None = None

        # Use properties to validate and store scenario dimension settings
        self.scenario_independent_sizes = scenario_independent_sizes
        self.scenario_independent_flow_rates = scenario_independent_flow_rates

    @staticmethod
    def _validate_timesteps(timesteps: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Validate timesteps format and rename if needed."""
        if not isinstance(timesteps, pd.DatetimeIndex):
            raise TypeError('timesteps must be a pandas DatetimeIndex')
        if len(timesteps) < 2:
            raise ValueError('timesteps must contain at least 2 timestamps')
        if timesteps.name != 'time':
            timesteps.name = 'time'
        if not timesteps.is_monotonic_increasing:
            raise ValueError('timesteps must be sorted')
        return timesteps

    @staticmethod
    def _validate_scenarios(scenarios: pd.Index) -> pd.Index:
        """
        Validate and prepare scenario index.

        Args:
            scenarios: The scenario index to validate
        """
        if not isinstance(scenarios, pd.Index) or len(scenarios) == 0:
            raise ConversionError('Scenarios must be a non-empty Index')

        if scenarios.name != 'scenario':
            scenarios = scenarios.rename('scenario')

        return scenarios

    @staticmethod
    def _validate_periods(periods: pd.Index) -> pd.Index:
        """
        Validate and prepare period index.

        Args:
            periods: The period index to validate
        """
        if not isinstance(periods, pd.Index) or len(periods) == 0:
            raise ConversionError(f'Periods must be a non-empty Index. Got {periods}')

        if not (
            periods.dtype.kind == 'i'  # integer dtype
            and periods.is_monotonic_increasing  # rising
            and periods.is_unique
        ):
            raise ConversionError(f'Periods must be a monotonically increasing and unique Index. Got {periods}')

        if periods.name != 'period':
            periods = periods.rename('period')

        return periods

    @staticmethod
    def _create_timesteps_with_extra(
        timesteps: pd.DatetimeIndex, hours_of_last_timestep: float | None
    ) -> pd.DatetimeIndex:
        """Create timesteps with an extra step at the end."""
        if hours_of_last_timestep is None:
            hours_of_last_timestep = (timesteps[-1] - timesteps[-2]) / pd.Timedelta(hours=1)

        last_date = pd.DatetimeIndex([timesteps[-1] + pd.Timedelta(hours=hours_of_last_timestep)], name='time')
        return pd.DatetimeIndex(timesteps.append(last_date), name='time')

    @staticmethod
    def calculate_hours_per_timestep(timesteps_extra: pd.DatetimeIndex) -> xr.DataArray:
        """Calculate duration of each timestep as a 1D DataArray."""
        hours_per_step = np.diff(timesteps_extra) / pd.Timedelta(hours=1)
        return xr.DataArray(
            hours_per_step, coords={'time': timesteps_extra[:-1]}, dims='time', name='hours_per_timestep'
        )

    @staticmethod
    def _calculate_hours_of_previous_timesteps(
        timesteps: pd.DatetimeIndex, hours_of_previous_timesteps: float | np.ndarray | None
    ) -> float | np.ndarray:
        """Calculate duration of regular timesteps."""
        if hours_of_previous_timesteps is not None:
            return hours_of_previous_timesteps
        # Calculate from the first interval
        first_interval = timesteps[1] - timesteps[0]
        return first_interval.total_seconds() / 3600  # Convert to hours

    @staticmethod
    def _create_periods_with_extra(periods: pd.Index, weight_of_last_period: int | float | None) -> pd.Index:
        """Create periods with an extra period at the end.

        Args:
            periods: The period index (must be monotonically increasing integers)
            weight_of_last_period: Weight of the last period. If None, computed from the period index.

        Returns:
            Period index with an extra period appended at the end
        """
        if weight_of_last_period is None:
            if len(periods) < 2:
                raise ValueError(
                    'FlowSystem: weight_of_last_period must be provided explicitly when only one period is defined.'
                )
            # Calculate weight from difference between last two periods
            weight_of_last_period = int(periods[-1]) - int(periods[-2])

        # Create the extra period value
        last_period_value = int(periods[-1]) + weight_of_last_period
        periods_extra = periods.append(pd.Index([last_period_value], name='period'))
        return periods_extra

    @staticmethod
    def calculate_weight_per_period(periods_extra: pd.Index) -> xr.DataArray:
        """Calculate weight of each period from period index differences.

        Args:
            periods_extra: Period index with an extra period at the end

        Returns:
            DataArray with weights for each period (1D, 'period' dimension)
        """
        weights = np.diff(periods_extra.to_numpy().astype(int))
        return xr.DataArray(weights, coords={'period': periods_extra[:-1]}, dims='period', name='weight_per_period')

    @classmethod
    def _compute_time_metadata(
        cls,
        timesteps: pd.DatetimeIndex,
        hours_of_last_timestep: int | float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
    ) -> tuple[pd.DatetimeIndex, float, float | np.ndarray, xr.DataArray]:
        """
        Compute all time-related metadata from timesteps.

        This is the single source of truth for time metadata computation, used by both
        __init__ and dataset operations (sel/isel/resample) to ensure consistency.

        Args:
            timesteps: The time index to compute metadata from
            hours_of_last_timestep: Duration of the last timestep. If None, computed from the time index.
            hours_of_previous_timesteps: Duration of previous timesteps. If None, computed from the time index.
                Can be a scalar or array.

        Returns:
            Tuple of (timesteps_extra, hours_of_last_timestep, hours_of_previous_timesteps, hours_per_timestep)
        """
        # Create timesteps with extra step at the end
        timesteps_extra = cls._create_timesteps_with_extra(timesteps, hours_of_last_timestep)

        # Calculate hours per timestep
        hours_per_timestep = cls.calculate_hours_per_timestep(timesteps_extra)

        # Extract hours_of_last_timestep if not provided
        if hours_of_last_timestep is None:
            hours_of_last_timestep = hours_per_timestep.isel(time=-1).item()

        # Compute hours_of_previous_timesteps (handles both None and provided cases)
        hours_of_previous_timesteps = cls._calculate_hours_of_previous_timesteps(timesteps, hours_of_previous_timesteps)

        return timesteps_extra, hours_of_last_timestep, hours_of_previous_timesteps, hours_per_timestep

    @classmethod
    def _compute_period_metadata(
        cls, periods: pd.Index | None, weight_of_last_period: int | float | None = None
    ) -> tuple[pd.Index | None, int | float | None, xr.DataArray | None]:
        """
        Compute all period-related metadata from periods.

        This is the single source of truth for period metadata computation, used by both
        __init__ and dataset operations to ensure consistency.

        Args:
            periods: The period index to compute metadata from (or None if no periods)
            weight_of_last_period: Weight of the last period. If None, computed from the period index.

        Returns:
            Tuple of (periods_extra, weight_of_last_period, weight_per_period)
            All return None if periods is None
        """
        if periods is None:
            return None, None, None

        # Create periods with extra period at the end
        periods_extra = cls._create_periods_with_extra(periods, weight_of_last_period)

        # Calculate weight per period
        weight_per_period = cls.calculate_weight_per_period(periods_extra)

        # Extract weight_of_last_period if not provided
        if weight_of_last_period is None:
            weight_of_last_period = weight_per_period.isel(period=-1).item()

        return periods_extra, weight_of_last_period, weight_per_period

    @classmethod
    def _update_time_metadata(
        cls,
        dataset: xr.Dataset,
        hours_of_last_timestep: int | float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
    ) -> xr.Dataset:
        """
        Update time-related attributes and data variables in dataset based on its time index.

        Recomputes hours_of_last_timestep, hours_of_previous_timesteps, and hours_per_timestep
        from the dataset's time index when these parameters are None. This ensures time metadata
        stays synchronized with the actual timesteps after operations like resampling or selection.

        Args:
            dataset: Dataset to update (will be modified in place)
            hours_of_last_timestep: Duration of the last timestep. If None, computed from the time index.
            hours_of_previous_timesteps: Duration of previous timesteps. If None, computed from the time index.
                Can be a scalar or array.

        Returns:
            The same dataset with updated time-related attributes and data variables
        """
        new_time_index = dataset.indexes.get('time')
        if new_time_index is not None and len(new_time_index) >= 2:
            # Use shared helper to compute all time metadata
            _, hours_of_last_timestep, hours_of_previous_timesteps, hours_per_timestep = cls._compute_time_metadata(
                new_time_index, hours_of_last_timestep, hours_of_previous_timesteps
            )

            # Update hours_per_timestep DataArray if it exists in the dataset
            # This prevents stale data after resampling operations
            if 'hours_per_timestep' in dataset.data_vars:
                dataset['hours_per_timestep'] = hours_per_timestep

        # Update time-related attributes only when new values are provided/computed
        # This preserves existing metadata instead of overwriting with None
        if hours_of_last_timestep is not None:
            dataset.attrs['hours_of_last_timestep'] = hours_of_last_timestep
        if hours_of_previous_timesteps is not None:
            dataset.attrs['hours_of_previous_timesteps'] = hours_of_previous_timesteps

        return dataset

    @classmethod
    def _update_period_metadata(
        cls,
        dataset: xr.Dataset,
        weight_of_last_period: int | float | None = None,
    ) -> xr.Dataset:
        """
        Update period-related attributes and data variables in dataset based on its period index.

        Recomputes weight_of_last_period and period_weights from the dataset's
        period index. This ensures period metadata stays synchronized with the actual
        periods after operations like selection.

        This is analogous to _update_time_metadata() for time-related metadata.

        Args:
            dataset: Dataset to update (will be modified in place)
            weight_of_last_period: Weight of the last period. If None, reused from dataset attrs
                (essential for single-period subsets where it cannot be inferred from intervals).

        Returns:
            The same dataset with updated period-related attributes and data variables
        """
        new_period_index = dataset.indexes.get('period')
        if new_period_index is not None and len(new_period_index) >= 1:
            # Reuse stored weight_of_last_period when not explicitly overridden.
            # This is essential for single-period subsets where it cannot be inferred from intervals.
            if weight_of_last_period is None:
                weight_of_last_period = dataset.attrs.get('weight_of_last_period')

            # Use shared helper to compute all period metadata
            _, weight_of_last_period, period_weights = cls._compute_period_metadata(
                new_period_index, weight_of_last_period
            )

            # Update period_weights DataArray if it exists in the dataset
            if 'period_weights' in dataset.data_vars:
                dataset['period_weights'] = period_weights

        # Update period-related attributes only when new values are provided/computed
        if weight_of_last_period is not None:
            dataset.attrs['weight_of_last_period'] = weight_of_last_period

        return dataset

    def _create_reference_structure(self) -> tuple[dict, dict[str, xr.DataArray]]:
        """
        Override Interface method to handle FlowSystem-specific serialization.
        Combines custom FlowSystem logic with Interface pattern for nested objects.

        Returns:
            Tuple of (reference_structure, extracted_arrays_dict)
        """
        # Start with Interface base functionality for constructor parameters
        reference_structure, all_extracted_arrays = super()._create_reference_structure()

        # Remove timesteps, as it's directly stored in dataset index
        reference_structure.pop('timesteps', None)

        # Extract from components
        components_structure = {}
        for comp_label, component in self.components.items():
            comp_structure, comp_arrays = component._create_reference_structure()
            all_extracted_arrays.update(comp_arrays)
            components_structure[comp_label] = comp_structure
        reference_structure['components'] = components_structure

        # Extract from buses
        buses_structure = {}
        for bus_label, bus in self.buses.items():
            bus_structure, bus_arrays = bus._create_reference_structure()
            all_extracted_arrays.update(bus_arrays)
            buses_structure[bus_label] = bus_structure
        reference_structure['buses'] = buses_structure

        # Extract from effects
        effects_structure = {}
        for effect in self.effects.values():
            effect_structure, effect_arrays = effect._create_reference_structure()
            all_extracted_arrays.update(effect_arrays)
            effects_structure[effect.label] = effect_structure
        reference_structure['effects'] = effects_structure

        return reference_structure, all_extracted_arrays

    def to_dataset(self) -> xr.Dataset:
        """
        Convert the FlowSystem to an xarray Dataset.
        Ensures FlowSystem is connected before serialization.

        If a solution is present, it will be included in the dataset with variable names
        prefixed by 'solution|' to avoid conflicts with FlowSystem configuration variables.
        Solution time coordinates are renamed to 'solution_time' to preserve them
        independently of the FlowSystem's time coordinates.

        Returns:
            xr.Dataset: Dataset containing all DataArrays with structure in attributes
        """
        if not self.connected_and_transformed:
            logger.warning('FlowSystem is not connected_and_transformed. Connecting and transforming data now.')
            self.connect_and_transform()

        ds = super().to_dataset()

        # Include solution data if present
        if self.solution is not None:
            # Rename 'time' to 'solution_time' in solution variables to preserve full solution
            # (linopy solution may have extra timesteps, e.g., for final charge states)
            solution_renamed = (
                self.solution.rename({'time': 'solution_time'}) if 'time' in self.solution.dims else self.solution
            )
            # Add solution variables with 'solution|' prefix to avoid conflicts
            solution_vars = {f'solution|{name}': var for name, var in solution_renamed.data_vars.items()}
            ds = ds.assign(solution_vars)
            # Also add the solution_time coordinate if it exists
            if 'solution_time' in solution_renamed.coords:
                ds = ds.assign_coords(solution_time=solution_renamed.coords['solution_time'])
            ds.attrs['has_solution'] = True
        else:
            ds.attrs['has_solution'] = False

        return ds

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> FlowSystem:
        """
        Create a FlowSystem from an xarray Dataset.
        Handles FlowSystem-specific reconstruction logic.

        If the dataset contains solution data (variables prefixed with 'solution|'),
        the solution will be restored to the FlowSystem. Solution time coordinates
        are renamed back from 'solution_time' to 'time'.

        Args:
            ds: Dataset containing the FlowSystem data

        Returns:
            FlowSystem instance
        """
        # Get the reference structure from attrs
        reference_structure = dict(ds.attrs)

        # Separate solution variables from config variables
        solution_prefix = 'solution|'
        solution_vars = {}
        config_vars = {}
        for name, array in ds.data_vars.items():
            if name.startswith(solution_prefix):
                # Remove prefix for solution dataset
                original_name = name[len(solution_prefix) :]
                solution_vars[original_name] = array
            else:
                config_vars[name] = array

        # Create arrays dictionary from config variables only
        arrays_dict = config_vars

        # Create FlowSystem instance with constructor parameters
        flow_system = cls(
            timesteps=ds.indexes['time'],
            periods=ds.indexes.get('period'),
            scenarios=ds.indexes.get('scenario'),
            hours_of_last_timestep=reference_structure.get('hours_of_last_timestep'),
            hours_of_previous_timesteps=reference_structure.get('hours_of_previous_timesteps'),
            weight_of_last_period=reference_structure.get('weight_of_last_period'),
            scenario_weights=cls._resolve_dataarray_reference(reference_structure['scenario_weights'], arrays_dict)
            if 'scenario_weights' in reference_structure
            else None,
            scenario_independent_sizes=reference_structure.get('scenario_independent_sizes', True),
            scenario_independent_flow_rates=reference_structure.get('scenario_independent_flow_rates', False),
        )

        # Restore components
        components_structure = reference_structure.get('components', {})
        for comp_label, comp_data in components_structure.items():
            component = cls._resolve_reference_structure(comp_data, arrays_dict)
            if not isinstance(component, Component):
                logger.critical(f'Restoring component {comp_label} failed.')
            flow_system._add_components(component)

        # Restore buses
        buses_structure = reference_structure.get('buses', {})
        for bus_label, bus_data in buses_structure.items():
            bus = cls._resolve_reference_structure(bus_data, arrays_dict)
            if not isinstance(bus, Bus):
                logger.critical(f'Restoring bus {bus_label} failed.')
            flow_system._add_buses(bus)

        # Restore effects
        effects_structure = reference_structure.get('effects', {})
        for effect_label, effect_data in effects_structure.items():
            effect = cls._resolve_reference_structure(effect_data, arrays_dict)
            if not isinstance(effect, Effect):
                logger.critical(f'Restoring effect {effect_label} failed.')
            flow_system._add_effects(effect)

        # Restore solution if present
        if reference_structure.get('has_solution', False) and solution_vars:
            solution_ds = xr.Dataset(solution_vars)
            # Rename 'solution_time' back to 'time' if present
            if 'solution_time' in solution_ds.dims:
                solution_ds = solution_ds.rename({'solution_time': 'time'})
            flow_system.solution = solution_ds

        return flow_system

    def to_netcdf(self, path: str | pathlib.Path, compression: int = 0):
        """
        Save the FlowSystem to a NetCDF file.
        Ensures FlowSystem is connected before saving.

        Args:
            path: The path to the netCDF file.
            compression: The compression level to use when saving the file.
        """
        if not self.connected_and_transformed:
            logger.warning('FlowSystem is not connected. Calling connect_and_transform() now.')
            self.connect_and_transform()

        super().to_netcdf(path, compression)
        logger.info(f'Saved FlowSystem to {path}')

    def get_structure(self, clean: bool = False, stats: bool = False) -> dict:
        """
        Get FlowSystem structure.
        Ensures FlowSystem is connected before getting structure.

        Args:
            clean: If True, remove None and empty dicts and lists.
            stats: If True, replace DataArray references with statistics
        """
        if not self.connected_and_transformed:
            logger.warning('FlowSystem is not connected. Calling connect_and_transform() now.')
            self.connect_and_transform()

        return super().get_structure(clean, stats)

    def to_json(self, path: str | pathlib.Path):
        """
        Save the flow system to a JSON file.
        Ensures FlowSystem is connected before saving.

        Args:
            path: The path to the JSON file.
        """
        if not self.connected_and_transformed:
            logger.warning(
                'FlowSystem needs to be connected and transformed before saving to JSON. Calling connect_and_transform() now.'
            )
            self.connect_and_transform()

        super().to_json(path)

    def fit_to_model_coords(
        self,
        name: str,
        data: NumericOrBool | None,
        dims: Collection[FlowSystemDimensions] | None = None,
    ) -> xr.DataArray | None:
        """
        Fit data to model coordinate system (currently time, but extensible).

        Args:
            name: Name of the data
            data: Data to fit to model coordinates (accepts any dimensionality including scalars)
            dims: Collection of dimension names to use for fitting. If None, all dimensions are used.

        Returns:
            xr.DataArray aligned to model coordinate system. If data is None, returns None.
        """
        if data is None:
            return None

        coords = self.coords

        if dims is not None:
            coords = {k: coords[k] for k in dims if k in coords}

        # Rest of your method stays the same, just pass coords
        if isinstance(data, TimeSeriesData):
            try:
                data.name = name  # Set name of previous object!
                return data.fit_to_coords(coords)
            except ConversionError as e:
                raise ConversionError(
                    f'Could not convert time series data "{name}" to DataArray:\n{data}\nOriginal Error: {e}'
                ) from e

        try:
            return DataConverter.to_dataarray(data, coords=coords).rename(name)
        except ConversionError as e:
            raise ConversionError(f'Could not convert data "{name}" to DataArray:\n{data}\nOriginal Error: {e}') from e

    def fit_effects_to_model_coords(
        self,
        label_prefix: str | None,
        effect_values: Effect_TPS | Numeric_TPS | None,
        label_suffix: str | None = None,
        dims: Collection[FlowSystemDimensions] | None = None,
        delimiter: str = '|',
    ) -> Effect_TPS | None:
        """
        Transform EffectValues from the user to Internal Datatypes aligned with model coordinates.
        """
        if effect_values is None:
            return None

        effect_values_dict = self.effects.create_effect_values_dict(effect_values)

        return {
            effect: self.fit_to_model_coords(
                str(delimiter).join(filter(None, [label_prefix, effect, label_suffix])),
                value,
                dims=dims,
            )
            for effect, value in effect_values_dict.items()
        }

    def connect_and_transform(self):
        """Transform data for all elements using the new simplified approach."""
        if self.connected_and_transformed:
            logger.debug('FlowSystem already connected and transformed')
            return

        self._connect_network()
        for element in chain(self.components.values(), self.effects.values(), self.buses.values()):
            element.transform_data()

        # Validate cross-element references immediately after transformation
        self._validate_system_integrity()

        self._connected_and_transformed = True

    def add_elements(self, *elements: Element) -> None:
        """
        Add Components(Storages, Boilers, Heatpumps, ...), Buses or Effects to the FlowSystem

        Args:
            *elements: childs of  Element like Boiler, HeatPump, Bus,...
                modeling Elements
        """
        if self.connected_and_transformed:
            warnings.warn(
                'You are adding elements to an already connected FlowSystem. This is not recommended (But it works).',
                stacklevel=2,
            )
            self._connected_and_transformed = False

        for new_element in list(elements):
            # Validate element type first
            if not isinstance(new_element, (Component, Effect, Bus)):
                raise TypeError(
                    f'Tried to add incompatible object to FlowSystem: {type(new_element)=}: {new_element=} '
                )

            # Common validations for all element types (before any state changes)
            self._check_if_element_already_assigned(new_element)
            self._check_if_element_is_unique(new_element)

            # Dispatch to type-specific handlers
            if isinstance(new_element, Component):
                self._add_components(new_element)
            elif isinstance(new_element, Effect):
                self._add_effects(new_element)
            elif isinstance(new_element, Bus):
                self._add_buses(new_element)

            # Log registration
            element_type = type(new_element).__name__
            logger.info(f'Registered new {element_type}: {new_element.label_full}')

    def create_model(self, normalize_weights: bool = True) -> FlowSystemModel:
        """
        Create a linopy model from the FlowSystem.

        Args:
            normalize_weights: Whether to automatically normalize the weights (periods and scenarios) to sum up to 1 when solving.
        """
        if not self.connected_and_transformed:
            raise RuntimeError(
                'FlowSystem is not connected_and_transformed. Call FlowSystem.connect_and_transform() first.'
            )
        # System integrity was already validated in connect_and_transform()
        self.model = FlowSystemModel(self, normalize_weights)
        return self.model

    def build_model(self, normalize_weights: bool = True) -> FlowSystem:
        """
        Build the optimization model for this FlowSystem.

        This method prepares the FlowSystem for optimization by:
        1. Connecting and transforming all elements (if not already done)
        2. Creating the FlowSystemModel with all variables and constraints

        After calling this method, `self.model` will be available for inspection
        before solving.

        Args:
            normalize_weights: Whether to normalize scenario/period weights to sum to 1.

        Returns:
            Self, for method chaining.

        Examples:
            >>> flow_system.build_model()
            >>> print(flow_system.model.variables)  # Inspect variables before solving
            >>> flow_system.solve(solver)
        """
        self.connect_and_transform()
        self.create_model(normalize_weights)
        self.model.do_modeling()
        return self

    def solve(self, solver: _Solver) -> FlowSystem:
        """
        Solve the optimization model and populate the solution.

        This method solves the previously built model using the specified solver.
        After solving, `self.solution` will contain the optimization results,
        and each element's `.solution` property will provide access to its
        specific variables.

        Args:
            solver: The solver to use (e.g., HighsSolver, GurobiSolver).

        Returns:
            Self, for method chaining.

        Raises:
            RuntimeError: If the model has not been built yet (call build_model first).
            RuntimeError: If the model is infeasible.

        Examples:
            >>> flow_system.build_model()
            >>> flow_system.solve(HighsSolver())
            >>> print(flow_system.solution)
        """
        if self.model is None:
            raise RuntimeError('Model has not been built. Call build_model() first.')

        self.model.solve(
            solver_name=solver.name,
            **solver.options,
        )

        if self.model.status == 'warning':
            if CONFIG.Solving.document_infeasibility:
                self._document_infeasibility()
            raise RuntimeError(f'Model was infeasible. Status: {self.model.status}. Check your constraints and bounds.')

        # Store solution on FlowSystem for direct Element access
        self.solution = self.model.solution

        logger.info(f'Optimization solved successfully. Objective: {self.model.objective.value:.4f}')

        return self

    def _document_infeasibility(self) -> None:
        """Save model documentation and FlowSystem to help debug infeasibility."""
        import tempfile
        from pathlib import Path

        # Create a temporary directory for infeasibility docs
        infeas_dir = Path(tempfile.gettempdir()) / 'flixopt_infeasibility'
        infeas_dir.mkdir(parents=True, exist_ok=True)

        # Save model documentation
        model_doc_path = infeas_dir / 'model_documentation.yaml'
        fx_io.document_linopy_model(self.model, model_doc_path)

        # Save FlowSystem
        fs_path = infeas_dir / 'flow_system.nc4'
        self.to_netcdf(fs_path)

        logger.error(
            f'Model infeasibility documentation saved to:\n  Model docs: {model_doc_path}\n  FlowSystem: {fs_path}'
        )

    @property
    def optimize(self) -> OptimizeAccessor:
        """
        Access optimization methods for this FlowSystem.

        This property returns an OptimizeAccessor that can be called directly
        for standard optimization, or used to access specialized optimization modes.

        Returns:
            An OptimizeAccessor instance.

        Examples:
            Standard optimization (call directly):

            >>> flow_system.optimize(HighsSolver())
            >>> print(flow_system.solution['Boiler(Q_th)|flow_rate'])

            Access element solutions directly:

            >>> flow_system.optimize(solver)
            >>> boiler = flow_system.components['Boiler']
            >>> print(boiler.solution)

            Future specialized modes:

            >>> flow_system.optimize.clustered(solver, aggregation=params)
            >>> flow_system.optimize.mga(solver, alternatives=5)
        """
        return OptimizeAccessor(self)

    @property
    def main_results(self) -> dict[str, Any]:
        """
        Get main results summary after optimization.

        Returns a dictionary containing:
        - Objective value
        - Penalty breakdown (temporal, periodic, total)
        - Effects summary with temporal, periodic, and total values
        - Investment decisions (invested vs not invested)
        - Buses with excess (if any)

        Returns:
            Dictionary with main optimization results.

        Raises:
            RuntimeError: If the model has not been solved yet.

        Examples:
            >>> flow_system.optimize(solver)
            >>> print(flow_system.main_results)
        """
        from .effects import PENALTY_EFFECT_LABEL
        from .features import InvestmentModel

        if self.model is None or self.solution is None:
            raise RuntimeError('FlowSystem has not been solved yet. Call optimize() or solve() first.')

        main_results = {
            'Objective': self.model.objective.value,
            'Penalty': {
                'temporal': self.effects.penalty_effect.submodel.temporal.total.solution.values,
                'periodic': self.effects.penalty_effect.submodel.periodic.total.solution.values,
                'total': self.effects.penalty_effect.submodel.total.solution.values,
            },
            'Effects': {
                f'{effect.label} [{effect.unit}]': {
                    'temporal': effect.submodel.temporal.total.solution.values,
                    'periodic': effect.submodel.periodic.total.solution.values,
                    'total': effect.submodel.total.solution.values,
                }
                for effect in sorted(self.effects.values(), key=lambda e: e.label_full.upper())
                if effect.label_full != PENALTY_EFFECT_LABEL
            },
            'Sizes': {
                'Invested': {
                    model.label_of_element: model.size.solution
                    for component in self.components.values()
                    for model in component.submodel.all_submodels
                    if isinstance(model, InvestmentModel)
                    and model.size.solution.max().item() >= CONFIG.Modeling.epsilon
                },
                'Not invested': {
                    model.label_of_element: model.size.solution
                    for component in self.components.values()
                    for model in component.submodel.all_submodels
                    if isinstance(model, InvestmentModel) and model.size.solution.max().item() < CONFIG.Modeling.epsilon
                },
            },
            'Buses with imbalance': [
                {
                    bus.label_full: {
                        'virtual_supply': (bus.submodel.virtual_supply.solution * self.hours_per_timestep).sum('time'),
                        'virtual_demand': (bus.submodel.virtual_demand.solution * self.hours_per_timestep).sum('time'),
                    }
                }
                for bus in self.buses.values()
                if bus.allows_imbalance
                and (
                    bus.submodel.virtual_supply.solution.sum().item() > 1e-3
                    or bus.submodel.virtual_demand.solution.sum().item() > 1e-3
                )
            ],
        }

        return fx_io.round_nested_floats(main_results)

    @property
    def summary(self) -> dict[str, Any]:
        """
        Get a summary of the FlowSystem and optimization.

        Returns a dictionary containing:
        - Number of timesteps
        - Number of constraints and variables (if modeled)
        - Main results (if solved)
        - Config settings

        Returns:
            Dictionary with FlowSystem summary.

        Raises:
            RuntimeError: If the model has not been solved yet.

        Examples:
            >>> flow_system.optimize(solver)
            >>> print(flow_system.summary)
        """
        if self.model is None or self.solution is None:
            raise RuntimeError('FlowSystem has not been solved yet. Call optimize() or solve() first.')

        return {
            'Number of timesteps': len(self.timesteps),
            'Constraints': self.model.constraints.ncons,
            'Variables': self.model.variables.nvars,
            'Main Results': self.main_results,
            'Config': CONFIG.to_dict(),
        }

    def plot_network(
        self,
        path: bool | str | pathlib.Path = 'flow_system.html',
        controls: bool
        | list[
            Literal['nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer']
        ] = True,
        show: bool | None = None,
    ) -> pyvis.network.Network | None:
        """
        Visualizes the network structure of a FlowSystem using PyVis, saving it as an interactive HTML file.

        Args:
            path: Path to save the HTML visualization.
                - `False`: Visualization is created but not saved.
                - `str` or `Path`: Specifies file path (default: 'flow_system.html').
            controls: UI controls to add to the visualization.
                - `True`: Enables all available controls.
                - `List`: Specify controls, e.g., ['nodes', 'layout'].
                - Options: 'nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer'.
            show: Whether to open the visualization in the web browser.

        Returns:
        - 'pyvis.network.Network' | None: The `Network` instance representing the visualization, or `None` if `pyvis` is not installed.

        Examples:
            >>> flow_system.plot_network()
            >>> flow_system.plot_network(show=False)
            >>> flow_system.plot_network(path='output/custom_network.html', controls=['nodes', 'layout'])

        Notes:
        - This function requires `pyvis`. If not installed, the function prints a warning and returns `None`.
        - Nodes are styled based on type (e.g., circles for buses, boxes for components) and annotated with node information.
        """
        from . import plotting

        node_infos, edge_infos = self.network_infos()
        return plotting.plot_network(
            node_infos, edge_infos, path, controls, show if show is not None else CONFIG.Plotting.default_show
        )

    def start_network_app(self):
        """Visualizes the network structure of a FlowSystem using Dash, Cytoscape, and networkx.
        Requires optional dependencies: dash, dash-cytoscape, dash-daq, networkx, flask, werkzeug.
        """
        from .network_app import DASH_CYTOSCAPE_AVAILABLE, VISUALIZATION_ERROR, flow_graph, shownetwork

        warnings.warn(
            'The network visualization is still experimental and might change in the future.',
            stacklevel=2,
            category=UserWarning,
        )

        if not DASH_CYTOSCAPE_AVAILABLE:
            raise ImportError(
                f'Network visualization requires optional dependencies. '
                f'Install with: `pip install flixopt[network_viz]`, `pip install flixopt[full]` '
                f'or: `pip install dash dash-cytoscape dash-daq networkx werkzeug`. '
                f'Original error: {VISUALIZATION_ERROR}'
            )

        if not self._connected_and_transformed:
            self._connect_network()

        if self._network_app is not None:
            logger.warning('The network app is already running. Restarting it.')
            self.stop_network_app()

        self._network_app = shownetwork(flow_graph(self))

    def stop_network_app(self):
        """Stop the network visualization server."""
        from .network_app import DASH_CYTOSCAPE_AVAILABLE, VISUALIZATION_ERROR

        if not DASH_CYTOSCAPE_AVAILABLE:
            raise ImportError(
                f'Network visualization requires optional dependencies. '
                f'Install with: `pip install flixopt[network_viz]`, `pip install flixopt[full]` '
                f'or: `pip install dash dash-cytoscape dash-daq networkx werkzeug`. '
                f'Original error: {VISUALIZATION_ERROR}'
            )

        if self._network_app is None:
            logger.warning("No network app is currently running. Can't stop it")
            return

        try:
            logger.info('Stopping network visualization server...')
            self._network_app.server_instance.shutdown()
            logger.info('Network visualization stopped.')
        except Exception as e:
            logger.error(f'Failed to stop the network visualization app: {e}')
        finally:
            self._network_app = None

    def network_infos(self) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
        if not self.connected_and_transformed:
            self.connect_and_transform()
        nodes = {
            node.label_full: {
                'label': node.label,
                'class': 'Bus' if isinstance(node, Bus) else 'Component',
                'infos': node.__str__(),
            }
            for node in chain(self.components.values(), self.buses.values())
        }

        edges = {
            flow.label_full: {
                'label': flow.label,
                'start': flow.bus if flow.is_input_in_component else flow.component,
                'end': flow.component if flow.is_input_in_component else flow.bus,
                'infos': flow.__str__(),
            }
            for flow in self.flows.values()
        }

        return nodes, edges

    def _check_if_element_is_unique(self, element: Element) -> None:
        """
        checks if element or label of element already exists in list

        Args:
            element: new element to check
        """
        # check if name is already used:
        if element.label_full in self:
            raise ValueError(f'Label of Element {element.label_full} already used in another element!')

    def _check_if_element_already_assigned(self, element: Element) -> None:
        """
        Check if element already belongs to another FlowSystem.

        Args:
            element: Element to check

        Raises:
            ValueError: If element is already assigned to a different FlowSystem
        """
        if element._flow_system is not None and element._flow_system is not self:
            raise ValueError(
                f'Element "{element.label_full}" is already assigned to another FlowSystem. '
                f'Each element can only belong to one FlowSystem at a time. '
                f'To use this element in multiple systems, create a copy: '
                f'flow_system.add_elements(element.copy())'
            )

    def _validate_system_integrity(self) -> None:
        """
        Validate cross-element references to ensure system consistency.

        This performs system-level validation that requires knowledge of multiple elements:
        - Validates that all Flow.bus references point to existing buses
        - Can be extended for other cross-element validations

        Should be called after connect_and_transform and before create_model.

        Raises:
            ValueError: If any cross-element reference is invalid
        """
        # Validate bus references in flows
        for flow in self.flows.values():
            if flow.bus not in self.buses:
                available_buses = list(self.buses.keys())
                raise ValueError(
                    f'Flow "{flow.label_full}" references bus "{flow.bus}" which does not exist in FlowSystem. '
                    f'Available buses: {available_buses}. '
                    f'Did you forget to add the bus using flow_system.add_elements(Bus("{flow.bus}"))?'
                )

    def _add_effects(self, *args: Effect) -> None:
        for effect in args:
            effect._set_flow_system(self)  # Link element to FlowSystem
        self.effects.add_effects(*args)

    def _add_components(self, *components: Component) -> None:
        for new_component in list(components):
            new_component._set_flow_system(self)  # Link element to FlowSystem
            self.components.add(new_component)  # Add to existing components
        # Invalidate cache once after all additions
        if components:
            self._flows_cache = None

    def _add_buses(self, *buses: Bus):
        for new_bus in list(buses):
            new_bus._set_flow_system(self)  # Link element to FlowSystem
            self.buses.add(new_bus)  # Add to existing buses
        # Invalidate cache once after all additions
        if buses:
            self._flows_cache = None

    def _connect_network(self):
        """Connects the network of components and buses. Can be rerun without changes if no elements were added"""
        for component in self.components.values():
            for flow in component.inputs + component.outputs:
                flow.component = component.label_full
                flow.is_input_in_component = True if flow in component.inputs else False

                # Connect Buses
                bus = self.buses.get(flow.bus)
                if bus is None:
                    raise KeyError(
                        f'Bus {flow.bus} not found in the FlowSystem, but used by "{flow.label_full}". '
                        f'Please add it first.'
                    )
                if flow.is_input_in_component and flow not in bus.outputs:
                    bus.outputs.append(flow)
                elif not flow.is_input_in_component and flow not in bus.inputs:
                    bus.inputs.append(flow)

        # Count flows manually to avoid triggering cache rebuild
        flow_count = sum(len(c.inputs) + len(c.outputs) for c in self.components.values())
        logger.debug(
            f'Connected {len(self.buses)} Buses and {len(self.components)} '
            f'via {flow_count} Flows inside the FlowSystem.'
        )

    def __repr__(self) -> str:
        """Return a detailed string representation showing all containers."""
        r = fx_io.format_title_with_underline('FlowSystem', '=')

        # Timestep info
        time_period = f'{self.timesteps[0].date()} to {self.timesteps[-1].date()}'
        freq_str = str(self.timesteps.freq).replace('<', '').replace('>', '') if self.timesteps.freq else 'irregular'
        r += f'Timesteps: {len(self.timesteps)} ({freq_str}) [{time_period}]\n'

        # Add periods if present
        if self.periods is not None:
            period_names = ', '.join(str(p) for p in self.periods[:3])
            if len(self.periods) > 3:
                period_names += f' ... (+{len(self.periods) - 3} more)'
            r += f'Periods: {len(self.periods)} ({period_names})\n'
        else:
            r += 'Periods: None\n'

        # Add scenarios if present
        if self.scenarios is not None:
            scenario_names = ', '.join(str(s) for s in self.scenarios[:3])
            if len(self.scenarios) > 3:
                scenario_names += f' ... (+{len(self.scenarios) - 3} more)'
            r += f'Scenarios: {len(self.scenarios)} ({scenario_names})\n'
        else:
            r += 'Scenarios: None\n'

        # Add status
        status = '✓' if self.connected_and_transformed else '⚠'
        r += f'Status: {status}\n'

        # Add grouped container view
        r += '\n' + self._format_grouped_containers()

        return r

    def __eq__(self, other: FlowSystem):
        """Check if two FlowSystems are equal by comparing their dataset representations."""
        if not isinstance(other, FlowSystem):
            raise NotImplementedError('Comparison with other types is not implemented for class FlowSystem')

        ds_me = self.to_dataset()
        ds_other = other.to_dataset()

        try:
            xr.testing.assert_equal(ds_me, ds_other)
        except AssertionError:
            return False

        if ds_me.attrs != ds_other.attrs:
            return False

        return True

    def _get_container_groups(self) -> dict[str, ElementContainer]:
        """Return ordered container groups for CompositeContainerMixin."""
        return {
            'Components': self.components,
            'Buses': self.buses,
            'Effects': self.effects,
            'Flows': self.flows,
        }

    @property
    def flows(self) -> ElementContainer[Flow]:
        if self._flows_cache is None:
            flows = [f for c in self.components.values() for f in c.inputs + c.outputs]
            # Deduplicate by id and sort for reproducibility
            flows = sorted({id(f): f for f in flows}.values(), key=lambda f: f.label_full.lower())
            self._flows_cache = ElementContainer(flows, element_type_name='flows', truncate_repr=10)
        return self._flows_cache

    @property
    def coords(self) -> dict[FlowSystemDimensions, pd.Index]:
        active_coords = {'time': self.timesteps}
        if self.periods is not None:
            active_coords['period'] = self.periods
        if self.scenarios is not None:
            active_coords['scenario'] = self.scenarios
        return active_coords

    @property
    def used_in_calculation(self) -> bool:
        return self._used_in_optimization

    @property
    def scenario_weights(self) -> xr.DataArray | None:
        """
        Weights for each scenario.

        Returns:
            xr.DataArray: Scenario weights with 'scenario' dimension
        """
        return self._scenario_weights

    @scenario_weights.setter
    def scenario_weights(self, value: Numeric_S | None) -> None:
        """
        Set scenario weights.

        Args:
            value: Scenario weights to set (will be converted to DataArray with 'scenario' dimension)
                or None to clear weights.

        Raises:
            ValueError: If value is not None and no scenarios are defined in the FlowSystem.
        """
        if value is None:
            self._scenario_weights = None
            return

        if self.scenarios is None:
            raise ValueError(
                'FlowSystem.scenario_weights cannot be set when no scenarios are defined. '
                'Either define scenarios in FlowSystem(scenarios=...) or set scenario_weights to None.'
            )

        self._scenario_weights = self.fit_to_model_coords('scenario_weights', value, dims=['scenario'])

    def _validate_scenario_parameter(self, value: bool | list[str], param_name: str, element_type: str) -> None:
        """
        Validate scenario parameter value.

        Args:
            value: The value to validate
            param_name: Name of the parameter (for error messages)
            element_type: Type of elements expected in list (e.g., 'component label_full', 'flow label_full')

        Raises:
            TypeError: If value is not bool or list[str]
            ValueError: If list contains non-string elements
        """
        if isinstance(value, bool):
            return  # Valid
        elif isinstance(value, list):
            if not all(isinstance(item, str) for item in value):
                raise ValueError(f'{param_name} list must contain only strings ({element_type} values)')
        else:
            raise TypeError(f'{param_name} must be bool or list[str], got {type(value).__name__}')

    @property
    def scenario_independent_sizes(self) -> bool | list[str]:
        """
        Controls whether investment sizes are equalized across scenarios.

        Returns:
            bool or list[str]: Configuration for scenario-independent sizing
        """
        return self._scenario_independent_sizes

    @scenario_independent_sizes.setter
    def scenario_independent_sizes(self, value: bool | list[str]) -> None:
        """
        Set whether investment sizes should be equalized across scenarios.

        Args:
            value: True (all equalized), False (all vary), or list of component label_full strings to equalize

        Raises:
            TypeError: If value is not bool or list[str]
            ValueError: If list contains non-string elements
        """
        self._validate_scenario_parameter(value, 'scenario_independent_sizes', 'Element.label_full')
        self._scenario_independent_sizes = value

    @property
    def scenario_independent_flow_rates(self) -> bool | list[str]:
        """
        Controls whether flow rates are equalized across scenarios.

        Returns:
            bool or list[str]: Configuration for scenario-independent flow rates
        """
        return self._scenario_independent_flow_rates

    @scenario_independent_flow_rates.setter
    def scenario_independent_flow_rates(self, value: bool | list[str]) -> None:
        """
        Set whether flow rates should be equalized across scenarios.

        Args:
            value: True (all equalized), False (all vary), or list of flow label_full strings to equalize

        Raises:
            TypeError: If value is not bool or list[str]
            ValueError: If list contains non-string elements
        """
        self._validate_scenario_parameter(value, 'scenario_independent_flow_rates', 'Flow.label_full')
        self._scenario_independent_flow_rates = value

    @classmethod
    def _dataset_sel(
        cls,
        dataset: xr.Dataset,
        time: str | slice | list[str] | pd.Timestamp | pd.DatetimeIndex | None = None,
        period: int | slice | list[int] | pd.Index | None = None,
        scenario: str | slice | list[str] | pd.Index | None = None,
        hours_of_last_timestep: int | float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
    ) -> xr.Dataset:
        """
        Select subset of dataset by label (for power users to avoid conversion overhead).

        This method operates directly on xarray Datasets, allowing power users to chain
        operations efficiently without repeated FlowSystem conversions:

        Example:
            # Power user pattern (single conversion):
            >>> ds = flow_system.to_dataset()
            >>> ds = FlowSystem._dataset_sel(ds, time='2020-01')
            >>> ds = FlowSystem._dataset_resample(ds, freq='2h', method='mean')
            >>> result = FlowSystem.from_dataset(ds)

            # vs. simple pattern (multiple conversions):
            >>> result = flow_system.sel(time='2020-01').resample('2h')

        Args:
            dataset: xarray Dataset from FlowSystem.to_dataset()
            time: Time selection (e.g., '2020-01', slice('2020-01-01', '2020-06-30'))
            period: Period selection (e.g., 2020, slice(2020, 2022))
            scenario: Scenario selection (e.g., 'Base Case', ['Base Case', 'High Demand'])
            hours_of_last_timestep: Duration of the last timestep. If None, computed from the selected time index.
            hours_of_previous_timesteps: Duration of previous timesteps. If None, computed from the selected time index.
                Can be a scalar or array.

        Returns:
            xr.Dataset: Selected dataset
        """
        indexers = {}
        if time is not None:
            indexers['time'] = time
        if period is not None:
            indexers['period'] = period
        if scenario is not None:
            indexers['scenario'] = scenario

        if not indexers:
            return dataset

        result = dataset.sel(**indexers)

        # Update time-related attributes if time was selected
        if 'time' in indexers:
            result = cls._update_time_metadata(result, hours_of_last_timestep, hours_of_previous_timesteps)

        # Update period-related attributes if period was selected
        # This recalculates period_weights and weights from the new period index
        if 'period' in indexers:
            result = cls._update_period_metadata(result)

        return result

    def sel(
        self,
        time: str | slice | list[str] | pd.Timestamp | pd.DatetimeIndex | None = None,
        period: int | slice | list[int] | pd.Index | None = None,
        scenario: str | slice | list[str] | pd.Index | None = None,
    ) -> FlowSystem:
        """
        Select a subset of the flowsystem by label.

        For power users: Use FlowSystem._dataset_sel() to chain operations on datasets
        without conversion overhead. See _dataset_sel() documentation.

        Args:
            time: Time selection (e.g., slice('2023-01-01', '2023-12-31'), '2023-06-15')
            period: Period selection (e.g., slice(2023, 2024), or list of periods)
            scenario: Scenario selection (e.g., 'scenario1', or list of scenarios)

        Returns:
            FlowSystem: New FlowSystem with selected data
        """
        if time is None and period is None and scenario is None:
            return self.copy()

        if not self.connected_and_transformed:
            self.connect_and_transform()

        ds = self.to_dataset()
        ds = self._dataset_sel(ds, time=time, period=period, scenario=scenario)
        return self.__class__.from_dataset(ds)

    @classmethod
    def _dataset_isel(
        cls,
        dataset: xr.Dataset,
        time: int | slice | list[int] | None = None,
        period: int | slice | list[int] | None = None,
        scenario: int | slice | list[int] | None = None,
        hours_of_last_timestep: int | float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
    ) -> xr.Dataset:
        """
        Select subset of dataset by integer index (for power users to avoid conversion overhead).

        See _dataset_sel() for usage pattern.

        Args:
            dataset: xarray Dataset from FlowSystem.to_dataset()
            time: Time selection by index (e.g., slice(0, 100), [0, 5, 10])
            period: Period selection by index
            scenario: Scenario selection by index
            hours_of_last_timestep: Duration of the last timestep. If None, computed from the selected time index.
            hours_of_previous_timesteps: Duration of previous timesteps. If None, computed from the selected time index.
                Can be a scalar or array.

        Returns:
            xr.Dataset: Selected dataset
        """
        indexers = {}
        if time is not None:
            indexers['time'] = time
        if period is not None:
            indexers['period'] = period
        if scenario is not None:
            indexers['scenario'] = scenario

        if not indexers:
            return dataset

        result = dataset.isel(**indexers)

        # Update time-related attributes if time was selected
        if 'time' in indexers:
            result = cls._update_time_metadata(result, hours_of_last_timestep, hours_of_previous_timesteps)

        # Update period-related attributes if period was selected
        # This recalculates period_weights and weights from the new period index
        if 'period' in indexers:
            result = cls._update_period_metadata(result)

        return result

    def isel(
        self,
        time: int | slice | list[int] | None = None,
        period: int | slice | list[int] | None = None,
        scenario: int | slice | list[int] | None = None,
    ) -> FlowSystem:
        """
        Select a subset of the flowsystem by integer indices.

        For power users: Use FlowSystem._dataset_isel() to chain operations on datasets
        without conversion overhead. See _dataset_sel() documentation.

        Args:
            time: Time selection by integer index (e.g., slice(0, 100), 50, or [0, 5, 10])
            period: Period selection by integer index (e.g., slice(0, 100), 50, or [0, 5, 10])
            scenario: Scenario selection by integer index (e.g., slice(0, 3), 50, or [0, 5, 10])

        Returns:
            FlowSystem: New FlowSystem with selected data
        """
        if time is None and period is None and scenario is None:
            return self.copy()

        if not self.connected_and_transformed:
            self.connect_and_transform()

        ds = self.to_dataset()
        ds = self._dataset_isel(ds, time=time, period=period, scenario=scenario)
        return self.__class__.from_dataset(ds)

    @classmethod
    def _resample_by_dimension_groups(
        cls,
        time_dataset: xr.Dataset,
        time: str,
        method: str,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Resample variables grouped by their dimension structure to avoid broadcasting.

        This method groups variables by their non-time dimensions before resampling,
        which provides two key benefits:

        1. **Performance**: Resampling many variables with the same dimensions together
           is significantly faster than resampling each variable individually.

        2. **Safety**: Prevents xarray from broadcasting variables with different
           dimensions into a larger dimensional space filled with NaNs, which would
           cause memory bloat and computational inefficiency.

        Example:
            Without grouping (problematic):
                var1: (time, location, tech)  shape (8000, 10, 2)
                var2: (time, region)          shape (8000, 5)
                concat → (variable, time, location, tech, region)  ← Unwanted broadcasting!

            With grouping (safe and fast):
                Group 1: [var1, var3, ...] with dims (time, location, tech)
                Group 2: [var2, var4, ...] with dims (time, region)
                Each group resampled separately → No broadcasting, optimal performance!

        Args:
            time_dataset: Dataset containing only variables with time dimension
            time: Resampling frequency (e.g., '2h', '1D', '1M')
            method: Resampling method name (e.g., 'mean', 'sum', 'first')
            **kwargs: Additional arguments passed to xarray.resample()

        Returns:
            Resampled dataset with original dimension structure preserved
        """
        # Group variables by dimensions (excluding time)
        dim_groups = defaultdict(list)
        for var_name, var in time_dataset.data_vars.items():
            dims_key = tuple(sorted(d for d in var.dims if d != 'time'))
            dim_groups[dims_key].append(var_name)

        # Handle empty case: no time-dependent variables
        if not dim_groups:
            return getattr(time_dataset.resample(time=time, **kwargs), method)()

        # Resample each group separately using DataArray concat (faster)
        resampled_groups = []
        for var_names in dim_groups.values():
            # Skip empty groups
            if not var_names:
                continue

            # Concat variables into a single DataArray with 'variable' dimension
            # Use combine_attrs='drop_conflicts' to handle attribute conflicts
            stacked = xr.concat(
                [time_dataset[name] for name in var_names],
                dim=pd.Index(var_names, name='variable'),
                combine_attrs='drop_conflicts',
            )

            # Resample the DataArray (faster than resampling Dataset)
            resampled = getattr(stacked.resample(time=time, **kwargs), method)()

            # Convert back to Dataset using the 'variable' dimension
            resampled_dataset = resampled.to_dataset(dim='variable')
            resampled_groups.append(resampled_dataset)

        # Merge all resampled groups, handling empty list case
        if not resampled_groups:
            return time_dataset  # Return empty dataset as-is

        if len(resampled_groups) == 1:
            return resampled_groups[0]

        # Merge multiple groups with combine_attrs to avoid conflicts
        return xr.merge(resampled_groups, combine_attrs='drop_conflicts')

    @classmethod
    def _dataset_resample(
        cls,
        dataset: xr.Dataset,
        freq: str,
        method: Literal['mean', 'sum', 'max', 'min', 'first', 'last', 'std', 'var', 'median', 'count'] = 'mean',
        hours_of_last_timestep: int | float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Resample dataset along time dimension (for power users to avoid conversion overhead).
        Preserves only the attrs of the Dataset.

        Uses optimized _resample_by_dimension_groups() to avoid broadcasting issues.
        See _dataset_sel() for usage pattern.

        Args:
            dataset: xarray Dataset from FlowSystem.to_dataset()
            freq: Resampling frequency (e.g., '2h', '1D', '1M')
            method: Resampling method (e.g., 'mean', 'sum', 'first')
            hours_of_last_timestep: Duration of the last timestep after resampling. If None, computed from the last time interval.
            hours_of_previous_timesteps: Duration of previous timesteps after resampling. If None, computed from the first time interval.
                Can be a scalar or array.
            **kwargs: Additional arguments passed to xarray.resample()

        Returns:
            xr.Dataset: Resampled dataset
        """
        # Validate method
        available_methods = ['mean', 'sum', 'max', 'min', 'first', 'last', 'std', 'var', 'median', 'count']
        if method not in available_methods:
            raise ValueError(f'Unsupported resampling method: {method}. Available: {available_methods}')

        # Preserve original dataset attributes (especially the reference structure)
        original_attrs = dict(dataset.attrs)

        # Separate time and non-time variables
        time_var_names = [v for v in dataset.data_vars if 'time' in dataset[v].dims]
        non_time_var_names = [v for v in dataset.data_vars if v not in time_var_names]

        # Only resample variables that have time dimension
        time_dataset = dataset[time_var_names]

        # Resample with dimension grouping to avoid broadcasting
        resampled_time_dataset = cls._resample_by_dimension_groups(time_dataset, freq, method, **kwargs)

        # Combine resampled time variables with non-time variables
        if non_time_var_names:
            non_time_dataset = dataset[non_time_var_names]
            result = xr.merge([resampled_time_dataset, non_time_dataset])
        else:
            result = resampled_time_dataset

        # Restore original attributes (xr.merge can drop them)
        result.attrs.update(original_attrs)

        # Update time-related attributes based on new time index
        return cls._update_time_metadata(result, hours_of_last_timestep, hours_of_previous_timesteps)

    def resample(
        self,
        time: str,
        method: Literal['mean', 'sum', 'max', 'min', 'first', 'last', 'std', 'var', 'median', 'count'] = 'mean',
        hours_of_last_timestep: int | float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
        **kwargs: Any,
    ) -> FlowSystem:
        """
        Create a resampled FlowSystem by resampling data along the time dimension (like xr.Dataset.resample()).
        Only resamples data variables that have a time dimension.

        For power users: Use FlowSystem._dataset_resample() to chain operations on datasets
        without conversion overhead. See _dataset_sel() documentation.

        Args:
            time: Resampling frequency (e.g., '3h', '2D', '1M')
            method: Resampling method. Recommended: 'mean', 'first', 'last', 'max', 'min'
            hours_of_last_timestep: Duration of the last timestep after resampling. If None, computed from the last time interval.
            hours_of_previous_timesteps: Duration of previous timesteps after resampling. If None, computed from the first time interval.
                Can be a scalar or array.
            **kwargs: Additional arguments passed to xarray.resample()

        Returns:
            FlowSystem: New resampled FlowSystem
        """
        if not self.connected_and_transformed:
            self.connect_and_transform()

        ds = self.to_dataset()
        ds = self._dataset_resample(
            ds,
            freq=time,
            method=method,
            hours_of_last_timestep=hours_of_last_timestep,
            hours_of_previous_timesteps=hours_of_previous_timesteps,
            **kwargs,
        )
        return self.__class__.from_dataset(ds)

    @property
    def connected_and_transformed(self) -> bool:
        return self._connected_and_transformed
