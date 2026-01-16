"""
This module contains the FlowSystem class, which is used to collect instances of many other classes by the end User.
"""

from __future__ import annotations

import json
import logging
import pathlib
import warnings
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from . import __version__
from . import io as fx_io
from .components import Storage
from .config import CONFIG, DEPRECATION_REMOVAL_VERSION
from .core import (
    ConversionError,
    DataConverter,
    FlowSystemDimensions,
    TimeSeriesData,
)
from .effects import Effect, EffectCollection
from .elements import Bus, Component, Flow
from .optimize_accessor import OptimizeAccessor
from .statistics_accessor import StatisticsAccessor
from .structure import CompositeContainerMixin, Element, ElementContainer, FlowSystemModel, Interface
from .topology_accessor import TopologyAccessor
from .transform_accessor import TransformAccessor

if TYPE_CHECKING:
    from collections.abc import Collection

    import pyvis

    from .clustering import Clustering
    from .solvers import _Solver
    from .types import Effect_TPS, Numeric_S, Numeric_TPS, NumericOrBool

from .carrier import Carrier, CarrierContainer

# Register clustering classes for IO (deferred to avoid circular imports)
from .clustering.base import _register_clustering_classes

_register_clustering_classes()

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
            Period weights are always computed internally from the period index (like timestep_duration for time).
            The final `weights` array (accessible via `flow_system.model.objective_weights`) is computed as period_weights × normalized_scenario_weights, with normalization applied to the scenario weights by default.
        cluster_weight: Weight for each cluster.
            If None (default), all clusters have weight 1.0. Used by cluster() to specify
            how many original timesteps each cluster represents. Multiply with timestep_duration
            for proper time aggregation in clustered models.
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
        timesteps: pd.DatetimeIndex | pd.RangeIndex,
        periods: pd.Index | None = None,
        scenarios: pd.Index | None = None,
        clusters: pd.Index | None = None,
        hours_of_last_timestep: int | float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
        weight_of_last_period: int | float | None = None,
        scenario_weights: Numeric_S | None = None,
        cluster_weight: Numeric_TPS | None = None,
        scenario_independent_sizes: bool | list[str] = True,
        scenario_independent_flow_rates: bool | list[str] = False,
        name: str | None = None,
        timestep_duration: xr.DataArray | None = None,
    ):
        self.timesteps = self._validate_timesteps(timesteps)

        # Compute all time-related metadata using shared helper
        (
            self.timesteps_extra,
            self.hours_of_last_timestep,
            self.hours_of_previous_timesteps,
            computed_timestep_duration,
        ) = self._compute_time_metadata(self.timesteps, hours_of_last_timestep, hours_of_previous_timesteps)

        self.periods = None if periods is None else self._validate_periods(periods)
        self.scenarios = None if scenarios is None else self._validate_scenarios(scenarios)
        self.clusters = clusters  # Cluster dimension for clustered FlowSystems

        # Use provided timestep_duration if given (for segmented systems), otherwise use computed value
        # For RangeIndex (segmented systems), computed_timestep_duration is None
        if timestep_duration is not None:
            self.timestep_duration = timestep_duration
        elif computed_timestep_duration is not None:
            self.timestep_duration = self.fit_to_model_coords('timestep_duration', computed_timestep_duration)
        else:
            self.timestep_duration = None

        # Cluster weight for cluster() optimization (default 1.0)
        # Represents how many original timesteps each cluster represents
        # May have period/scenario dimensions if cluster() was used with those
        self.cluster_weight: xr.DataArray | None = (
            self.fit_to_model_coords(
                'cluster_weight',
                cluster_weight,
            )
            if cluster_weight is not None
            else None
        )

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
        self._storages_cache: ElementContainer[Storage] | None = None

        # Solution dataset - populated after optimization or loaded from file
        self._solution: xr.Dataset | None = None

        # Aggregation info - populated by transform.cluster()
        self.clustering: Clustering | None = None

        # Statistics accessor cache - lazily initialized, invalidated on new solution
        self._statistics: StatisticsAccessor | None = None

        # Topology accessor cache - lazily initialized, invalidated on structure change
        self._topology: TopologyAccessor | None = None

        # Carrier container - local carriers override CONFIG.Carriers
        self._carriers: CarrierContainer = CarrierContainer()

        # Cached flow→carrier mapping (built lazily after connect_and_transform)
        self._flow_carriers: dict[str, str] | None = None

        # Use properties to validate and store scenario dimension settings
        self.scenario_independent_sizes = scenario_independent_sizes
        self.scenario_independent_flow_rates = scenario_independent_flow_rates

        # Optional name for identification (derived from filename on load)
        self.name = name

    @staticmethod
    def _validate_timesteps(
        timesteps: pd.DatetimeIndex | pd.RangeIndex,
    ) -> pd.DatetimeIndex | pd.RangeIndex:
        """Validate timesteps format and rename if needed.

        Accepts either DatetimeIndex (standard) or RangeIndex (for segmented systems).
        """
        if not isinstance(timesteps, (pd.DatetimeIndex, pd.RangeIndex)):
            raise TypeError('timesteps must be a pandas DatetimeIndex or RangeIndex')
        if len(timesteps) < 2:
            raise ValueError('timesteps must contain at least 2 timestamps')
        if timesteps.name != 'time':
            timesteps = timesteps.rename('time')
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
        timesteps: pd.DatetimeIndex | pd.RangeIndex, hours_of_last_timestep: float | None
    ) -> pd.DatetimeIndex | pd.RangeIndex:
        """Create timesteps with an extra step at the end.

        For DatetimeIndex, adds an extra timestep using hours_of_last_timestep.
        For RangeIndex (segmented systems), simply appends the next integer.
        """
        if isinstance(timesteps, pd.RangeIndex):
            # For RangeIndex, just add one more integer
            return pd.RangeIndex(len(timesteps) + 1, name='time')

        if hours_of_last_timestep is None:
            hours_of_last_timestep = (timesteps[-1] - timesteps[-2]) / pd.Timedelta(hours=1)

        last_date = pd.DatetimeIndex([timesteps[-1] + pd.Timedelta(hours=hours_of_last_timestep)], name='time')
        return pd.DatetimeIndex(timesteps.append(last_date), name='time')

    @staticmethod
    def calculate_timestep_duration(
        timesteps_extra: pd.DatetimeIndex | pd.RangeIndex,
    ) -> xr.DataArray | None:
        """Calculate duration of each timestep in hours as a 1D DataArray.

        For RangeIndex (segmented systems), returns None since duration cannot be
        computed from the index. Use timestep_duration parameter instead.
        """
        if isinstance(timesteps_extra, pd.RangeIndex):
            # Cannot compute duration from RangeIndex - must be provided externally
            return None

        hours_per_step = np.diff(timesteps_extra) / pd.Timedelta(hours=1)
        return xr.DataArray(
            hours_per_step, coords={'time': timesteps_extra[:-1]}, dims='time', name='timestep_duration'
        )

    @staticmethod
    def _calculate_hours_of_previous_timesteps(
        timesteps: pd.DatetimeIndex | pd.RangeIndex, hours_of_previous_timesteps: float | np.ndarray | None
    ) -> float | np.ndarray | None:
        """Calculate duration of regular timesteps.

        For RangeIndex (segmented systems), returns None if not provided.
        """
        if hours_of_previous_timesteps is not None:
            return hours_of_previous_timesteps
        if isinstance(timesteps, pd.RangeIndex):
            # Cannot compute from RangeIndex
            return None
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
        timesteps: pd.DatetimeIndex | pd.RangeIndex,
        hours_of_last_timestep: int | float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
    ) -> tuple[
        pd.DatetimeIndex | pd.RangeIndex,
        float | None,
        float | np.ndarray | None,
        xr.DataArray | None,
    ]:
        """
        Compute all time-related metadata from timesteps.

        This is the single source of truth for time metadata computation, used by both
        __init__ and dataset operations (sel/isel/resample) to ensure consistency.

        For RangeIndex (segmented systems), timestep_duration cannot be calculated from
        the index and must be provided externally after FlowSystem creation.

        Args:
            timesteps: The time index to compute metadata from (DatetimeIndex or RangeIndex)
            hours_of_last_timestep: Duration of the last timestep. If None, computed from the time index.
            hours_of_previous_timesteps: Duration of previous timesteps. If None, computed from the time index.
                Can be a scalar or array.

        Returns:
            Tuple of (timesteps_extra, hours_of_last_timestep, hours_of_previous_timesteps, timestep_duration)
            For RangeIndex, hours_of_last_timestep and timestep_duration may be None.
        """
        # Create timesteps with extra step at the end
        timesteps_extra = cls._create_timesteps_with_extra(timesteps, hours_of_last_timestep)

        # Calculate timestep duration (returns None for RangeIndex)
        timestep_duration = cls.calculate_timestep_duration(timesteps_extra)

        # Extract hours_of_last_timestep if not provided
        if hours_of_last_timestep is None and timestep_duration is not None:
            hours_of_last_timestep = timestep_duration.isel(time=-1).item()

        # Compute hours_of_previous_timesteps (handles both None and provided cases)
        hours_of_previous_timesteps = cls._calculate_hours_of_previous_timesteps(timesteps, hours_of_previous_timesteps)

        return timesteps_extra, hours_of_last_timestep, hours_of_previous_timesteps, timestep_duration

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

        Recomputes hours_of_last_timestep, hours_of_previous_timesteps, and timestep_duration
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
            _, hours_of_last_timestep, hours_of_previous_timesteps, timestep_duration = cls._compute_time_metadata(
                new_time_index, hours_of_last_timestep, hours_of_previous_timesteps
            )

            # Update timestep_duration DataArray if it exists in the dataset
            # This prevents stale data after resampling operations
            if 'timestep_duration' in dataset.data_vars:
                dataset['timestep_duration'] = timestep_duration

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

        When the period dimension is dropped (single value selected), this method
        removes the scalar coordinate, period_weights DataArray, and cleans up attributes.

        This is analogous to _update_time_metadata() for time-related metadata.

        Args:
            dataset: Dataset to update (will be modified in place)
            weight_of_last_period: Weight of the last period. If None, reused from dataset attrs
                (essential for single-period subsets where it cannot be inferred from intervals).

        Returns:
            The same dataset with updated period-related attributes and data variables
        """
        new_period_index = dataset.indexes.get('period')

        if new_period_index is None:
            # Period dimension was dropped (single value selected)
            if 'period' in dataset.coords:
                dataset = dataset.drop_vars('period')
            dataset = dataset.drop_vars(['period_weights'], errors='ignore')
            dataset.attrs.pop('weight_of_last_period', None)
            return dataset

        if len(new_period_index) >= 1:
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

    @classmethod
    def _update_scenario_metadata(cls, dataset: xr.Dataset) -> xr.Dataset:
        """
        Update scenario-related attributes and data variables in dataset based on its scenario index.

        Recomputes or removes scenario weights. This ensures scenario metadata stays synchronized with the actual
        scenarios after operations like selection.

        When the scenario dimension is dropped (single value selected), this method
        removes the scalar coordinate, scenario_weights DataArray, and cleans up attributes.

        This is analogous to _update_period_metadata() for time-related metadata.

        Args:
            dataset: Dataset to update (will be modified in place)

        Returns:
            The same dataset with updated scenario-related attributes and data variables
        """
        new_scenario_index = dataset.indexes.get('scenario')

        if new_scenario_index is None:
            # Scenario dimension was dropped (single value selected)
            if 'scenario' in dataset.coords:
                dataset = dataset.drop_vars('scenario')
            dataset = dataset.drop_vars(['scenario_weights'], errors='ignore')
            dataset.attrs.pop('scenario_weights', None)
            return dataset

        if len(new_scenario_index) <= 1:
            dataset.attrs.pop('scenario_weights', None)

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

    def to_dataset(self, include_solution: bool = True, include_original_data: bool = True) -> xr.Dataset:
        """
        Convert the FlowSystem to an xarray Dataset.
        Ensures FlowSystem is connected before serialization.

        Data is stored in minimal form (scalars stay scalar, 1D arrays stay 1D) without
        broadcasting to full model dimensions. This provides significant memory savings
        for multi-period and multi-scenario models.

        If a solution is present and `include_solution=True`, it will be included
        in the dataset with variable names prefixed by 'solution|' to avoid conflicts
        with FlowSystem configuration variables. Solution time coordinates are renamed
        to 'solution_time' to preserve them independently of the FlowSystem's time coordinates.

        Args:
            include_solution: Whether to include the optimization solution in the dataset.
                Defaults to True. Set to False to get only the FlowSystem structure
                without solution data (useful for copying or saving templates).
            include_original_data: Whether to include clustering.original_data in the dataset.
                Defaults to True. Set to False for smaller files (~38% reduction) when
                clustering.plot.compare() isn't needed after loading. The core workflow
                (optimize → expand) works without original_data.

        Returns:
            xr.Dataset: Dataset containing all DataArrays with structure in attributes
        """
        if not self.connected_and_transformed:
            logger.info('FlowSystem is not connected_and_transformed. Connecting and transforming data now.')
            self.connect_and_transform()

        ds = super().to_dataset()

        # Include solution data if present and requested
        if include_solution and self.solution is not None:
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

        # Include carriers if any are registered
        if self._carriers:
            carriers_structure = {}
            for name, carrier in self._carriers.items():
                carrier_ref, _ = carrier._create_reference_structure()
                carriers_structure[name] = carrier_ref
            ds.attrs['carriers'] = json.dumps(carriers_structure)

        # Serialize Clustering object for full reconstruction in from_dataset()
        if self.clustering is not None:
            clustering_ref, clustering_arrays = self.clustering._create_reference_structure(
                include_original_data=include_original_data
            )
            # Add clustering arrays with prefix
            for name, arr in clustering_arrays.items():
                ds[f'clustering|{name}'] = arr
            ds.attrs['clustering'] = json.dumps(clustering_ref)

        # Add version info
        ds.attrs['flixopt_version'] = __version__

        # Ensure model coordinates are always present in the Dataset
        # (even if no data variable uses them, they define the model structure)
        model_coords = {'time': self.timesteps}
        if self.periods is not None:
            model_coords['period'] = self.periods
        if self.scenarios is not None:
            model_coords['scenario'] = self.scenarios
        if self.clusters is not None:
            model_coords['cluster'] = self.clusters
        ds = ds.assign_coords(model_coords)

        return ds

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> FlowSystem:
        """
        Create a FlowSystem from an xarray Dataset.
        Handles FlowSystem-specific reconstruction logic.

        If the dataset contains solution data (variables prefixed with 'solution|'),
        the solution will be restored to the FlowSystem. Solution time coordinates
        are renamed back from 'solution_time' to 'time'.

        Supports clustered datasets with (cluster, time) dimensions. When detected,
        creates a synthetic DatetimeIndex for compatibility and stores the clustered
        data structure for later use.

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

        # Extract cluster index if present (clustered FlowSystem)
        clusters = ds.indexes.get('cluster')

        # For clustered datasets, cluster_weight is (cluster,) shaped - set separately
        if clusters is not None:
            cluster_weight_for_constructor = None
        else:
            cluster_weight_for_constructor = (
                cls._resolve_dataarray_reference(reference_structure['cluster_weight'], arrays_dict)
                if 'cluster_weight' in reference_structure
                else None
            )

        # Resolve scenario_weights only if scenario dimension exists
        scenario_weights = None
        if ds.indexes.get('scenario') is not None and 'scenario_weights' in reference_structure:
            scenario_weights = cls._resolve_dataarray_reference(reference_structure['scenario_weights'], arrays_dict)

        # Resolve timestep_duration if present as DataArray reference (for segmented systems with variable durations)
        timestep_duration = None
        if 'timestep_duration' in reference_structure:
            ref_value = reference_structure['timestep_duration']
            # Only resolve if it's a DataArray reference (starts with ":::")
            # For non-segmented systems, it may be stored as a simple list/scalar
            if isinstance(ref_value, str) and ref_value.startswith(':::'):
                timestep_duration = cls._resolve_dataarray_reference(ref_value, arrays_dict)

        # Get timesteps - convert integer index to RangeIndex for segmented systems
        time_index = ds.indexes['time']
        if not isinstance(time_index, pd.DatetimeIndex):
            # Segmented systems use RangeIndex (stored as integer array in NetCDF)
            time_index = pd.RangeIndex(len(time_index), name='time')

        # Create FlowSystem instance with constructor parameters
        flow_system = cls(
            timesteps=time_index,
            periods=ds.indexes.get('period'),
            scenarios=ds.indexes.get('scenario'),
            clusters=clusters,
            hours_of_last_timestep=reference_structure.get('hours_of_last_timestep'),
            hours_of_previous_timesteps=reference_structure.get('hours_of_previous_timesteps'),
            weight_of_last_period=reference_structure.get('weight_of_last_period'),
            scenario_weights=scenario_weights,
            cluster_weight=cluster_weight_for_constructor,
            scenario_independent_sizes=reference_structure.get('scenario_independent_sizes', True),
            scenario_independent_flow_rates=reference_structure.get('scenario_independent_flow_rates', False),
            name=reference_structure.get('name'),
            timestep_duration=timestep_duration,
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

        # Restore carriers if present
        if 'carriers' in reference_structure:
            carriers_structure = json.loads(reference_structure['carriers'])
            for carrier_data in carriers_structure.values():
                carrier = cls._resolve_reference_structure(carrier_data, {})
                flow_system._carriers.add(carrier)

        # Restore Clustering object if present
        if 'clustering' in reference_structure:
            clustering_structure = json.loads(reference_structure['clustering'])
            # Collect clustering arrays (prefixed with 'clustering|')
            clustering_arrays = {}
            for name, arr in ds.data_vars.items():
                if name.startswith('clustering|'):
                    # Remove 'clustering|' prefix (11 chars) from both key and DataArray name
                    # This ensures that if the FlowSystem is serialized again, the arrays
                    # won't get double-prefixed (clustering|clustering|...)
                    arr_name = name[11:]
                    clustering_arrays[arr_name] = arr.rename(arr_name)
            clustering = cls._resolve_reference_structure(clustering_structure, clustering_arrays)
            flow_system.clustering = clustering

            # Reconstruct aggregated_data from FlowSystem's main data arrays
            # (aggregated_data is not serialized to avoid redundant storage)
            if clustering.aggregated_data is None:
                from .core import drop_constant_arrays

                # Get non-clustering variables and filter to time-varying only
                main_vars = {name: arr for name, arr in ds.data_vars.items() if not name.startswith('clustering|')}
                if main_vars:
                    clustering.aggregated_data = drop_constant_arrays(xr.Dataset(main_vars), dim='time')

            # Restore cluster_weight from clustering's representative_weights
            # This is needed because cluster_weight_for_constructor was set to None for clustered datasets
            if hasattr(clustering, 'representative_weights'):
                flow_system.cluster_weight = clustering.representative_weights

        # Reconnect network to populate bus inputs/outputs (not stored in NetCDF).
        flow_system.connect_and_transform()

        return flow_system

    def to_netcdf(
        self,
        path: str | pathlib.Path,
        compression: int = 5,
        overwrite: bool = False,
        include_original_data: bool = True,
    ):
        """
        Save the FlowSystem to a NetCDF file.
        Ensures FlowSystem is connected before saving.

        The FlowSystem's name is automatically set from the filename
        (without extension) when saving.

        Args:
            path: The path to the netCDF file. Parent directories are created if they don't exist.
            compression: The compression level to use when saving the file (0-9).
            overwrite: If True, overwrite existing file. If False, raise error if file exists.
            include_original_data: Whether to include clustering.original_data in the file.
                Defaults to True. Set to False for smaller files (~38% reduction) when
                clustering.plot.compare() isn't needed after loading.

        Raises:
            FileExistsError: If overwrite=False and file already exists.
        """
        if not self.connected_and_transformed:
            logger.warning('FlowSystem is not connected. Calling connect_and_transform() now.')
            self.connect_and_transform()

        path = pathlib.Path(path)

        if not overwrite and path.exists():
            raise FileExistsError(f'File already exists: {path}. Use overwrite=True to overwrite existing file.')

        path.parent.mkdir(parents=True, exist_ok=True)

        # Set name from filename (without extension)
        self.name = path.stem

        try:
            ds = self.to_dataset(include_original_data=include_original_data)
            fx_io.save_dataset_to_netcdf(ds, path, compression=compression)
            logger.info(f'Saved FlowSystem to {path}')
        except Exception as e:
            raise OSError(f'Failed to save FlowSystem to NetCDF file {path}: {e}') from e

    @classmethod
    def from_netcdf(cls, path: str | pathlib.Path) -> FlowSystem:
        """
        Load a FlowSystem from a NetCDF file.

        The FlowSystem's name is automatically derived from the filename
        (without extension), overriding any name that may have been stored.

        Args:
            path: Path to the NetCDF file

        Returns:
            FlowSystem instance with name set from filename
        """
        path = pathlib.Path(path)
        flow_system = super().from_netcdf(path)
        # Derive name from filename (without extension)
        flow_system.name = path.stem
        return flow_system

    @classmethod
    def from_old_results(cls, folder: str | pathlib.Path, name: str) -> FlowSystem:
        """
        Load a FlowSystem from old-format Results files (pre-v5 API).

        This method loads results saved with the deprecated Results API
        (which used multiple files: ``*--flow_system.nc4``, ``*--solution.nc4``)
        and converts them to a FlowSystem with the solution attached.

        The method performs the following:

        - Loads the old multi-file format
        - Renames deprecated parameters in the FlowSystem structure
          (e.g., ``on_off_parameters`` → ``status_parameters``)
        - Attaches the solution data to the FlowSystem

        Args:
            folder: Directory containing the saved result files
            name: Base name of the saved files (without extensions)

        Returns:
            FlowSystem instance with solution attached

        Warning:
            This is a best-effort migration for accessing old results:

            - **Solution variable names are NOT renamed** - only basic variables
              work (flow rates, sizes, charge states, effect totals)
            - Advanced variable access may require using the original names
            - Summary metadata (solver info, timing) is not loaded

            For full compatibility, re-run optimizations with the new API.

        Examples:
            ```python
            # Load old results
            fs = FlowSystem.from_old_results('results_folder', 'my_optimization')

            # Access basic solution data
            fs.solution['Boiler(Q_th)|flow_rate'].plot()

            # Save in new single-file format
            fs.to_netcdf('my_optimization.nc')
            ```

        Deprecated:
            This method will be removed in v6.
        """
        warnings.warn(
            f'from_old_results() is deprecated and will be removed in v{DEPRECATION_REMOVAL_VERSION}. '
            'This utility is only for migrating results from flixopt versions before v5.',
            DeprecationWarning,
            stacklevel=2,
        )
        from flixopt.io import load_dataset_from_netcdf

        folder = pathlib.Path(folder)
        flow_system_path = folder / f'{name}--flow_system.nc4'
        solution_path = folder / f'{name}--solution.nc4'

        # Load FlowSystem using from_old_dataset (suppress its deprecation warning)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            flow_system = cls.from_old_dataset(flow_system_path)
        flow_system.name = name

        # Attach solution (convert attrs from dicts to JSON strings for consistency)
        solution = load_dataset_from_netcdf(solution_path)
        for key in ['Components', 'Buses', 'Effects', 'Flows']:
            if key in solution.attrs and isinstance(solution.attrs[key], dict):
                solution.attrs[key] = json.dumps(solution.attrs[key])
        flow_system.solution = solution

        return flow_system

    @classmethod
    def from_old_dataset(cls, path: str | pathlib.Path) -> FlowSystem:
        """
        Load a FlowSystem from an old-format dataset file (pre-v5 API).

        This method loads a FlowSystem saved with older versions of flixopt
        (the ``*--flow_system.nc4`` file) and converts parameter names to the
        current API. Unlike :meth:`from_old_results`, this does not require
        a solution file and returns a FlowSystem without solution data.

        The method performs the following:

        - Loads the old netCDF format
        - Renames deprecated parameters in the FlowSystem structure
          (e.g., ``on_off_parameters`` → ``status_parameters``)

        Args:
            path: Path to the old-format FlowSystem file (typically ``*--flow_system.nc4``)

        Returns:
            FlowSystem instance without solution

        Warning:
            This is a best-effort migration for loading old FlowSystem definitions.
            For full compatibility, consider re-saving with the new API after loading.

        Examples:
            ```python
            # Load old FlowSystem file
            fs = FlowSystem.from_old_dataset('results/my_run--flow_system.nc4')

            # Modify and optimize with current API
            fs.optimize(solver)

            # Save in new single-file format
            fs.to_netcdf('my_run.nc')
            ```

        Deprecated:
            This method will be removed in v6.
        """
        warnings.warn(
            f'from_old_dataset() is deprecated and will be removed in v{DEPRECATION_REMOVAL_VERSION}. '
            'This utility is only for migrating FlowSystems from flixopt versions before v5.',
            DeprecationWarning,
            stacklevel=2,
        )
        from flixopt.io import convert_old_dataset, load_dataset_from_netcdf

        path = pathlib.Path(path)

        # Load dataset
        flow_system_data = load_dataset_from_netcdf(path)

        # Convert to new parameter names
        convert_old_dataset(flow_system_data)

        # Reconstruct FlowSystem
        flow_system = cls.from_dataset(flow_system_data)
        flow_system.name = path.stem.replace('--flow_system', '')

        # Set previous_flow_rate=0 for flows of components with status_parameters
        # In v4 API, previous_flow_rate=None defaulted to previous_status=0 (off)
        # Now previous_flow_rate=None means relaxed (no constraint at t=0)
        for comp in flow_system.components.values():
            if getattr(comp, 'status_parameters', None) is not None:
                for flow in comp.inputs + comp.outputs:
                    if flow.previous_flow_rate is None:
                        flow.previous_flow_rate = 0

        return flow_system

    def copy(self) -> FlowSystem:
        """Create a copy of the FlowSystem without optimization state.

        Creates a new FlowSystem with copies of all elements, but without:
        - The solution dataset
        - The optimization model
        - Element submodels and variable/constraint names

        This is useful for creating variations of a FlowSystem for different
        optimization scenarios without affecting the original.

        Returns:
            A new FlowSystem instance that can be modified and optimized independently.

        Examples:
            >>> original = FlowSystem(timesteps)
            >>> original.add_elements(boiler, bus)
            >>> original.optimize(solver)  # Original now has solution
            >>>
            >>> # Create a copy to try different parameters
            >>> variant = original.copy()  # No solution, can be modified
            >>> variant.add_elements(new_component)
            >>> variant.optimize(solver)
        """
        ds = self.to_dataset(include_solution=False)
        return FlowSystem.from_dataset(ds.copy(deep=True))

    def __copy__(self):
        """Support for copy.copy()."""
        return self.copy()

    def __deepcopy__(self, memo):
        """Support for copy.deepcopy()."""
        return self.copy()

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

        coords = self.indexes

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
        """Connect the network and transform all element data to model coordinates.

        This method performs the following steps:

        1. Connects flows to buses (establishing the network topology)
        2. Registers any missing carriers from CONFIG defaults
        3. Assigns colors to elements without explicit colors
        4. Transforms all element data to xarray DataArrays aligned with
           FlowSystem coordinates (time, period, scenario)
        5. Validates system integrity

        This is called automatically by :meth:`build_model` and :meth:`optimize`.

        Warning:
            After this method runs, element attributes (e.g., ``flow.size``,
            ``flow.relative_minimum``) contain transformed xarray DataArrays,
            not the original input values. If you modify element attributes after
            transformation, call :meth:`invalidate` to ensure the changes take
            effect on the next optimization.

        Note:
            This method is idempotent within a single model lifecycle - calling
            it multiple times has no effect once ``connected_and_transformed``
            is True. Use :meth:`invalidate` to reset this flag.
        """
        if self.connected_and_transformed:
            logger.debug('FlowSystem already connected and transformed')
            return

        self._connect_network()
        self._register_missing_carriers()
        self._assign_element_colors()

        for element in chain(self.components.values(), self.effects.values(), self.buses.values()):
            element.transform_data()

        # Validate cross-element references immediately after transformation
        self._validate_system_integrity()

        self._connected_and_transformed = True

    def _register_missing_carriers(self) -> None:
        """Auto-register carriers from CONFIG for buses that reference unregistered carriers."""
        for bus in self.buses.values():
            if not bus.carrier:
                continue
            carrier_key = bus.carrier.lower()
            if carrier_key not in self._carriers:
                # Try to get from CONFIG defaults (try original case first, then lowercase)
                default_carrier = getattr(CONFIG.Carriers, bus.carrier, None) or getattr(
                    CONFIG.Carriers, carrier_key, None
                )
                if default_carrier is not None:
                    self._carriers[carrier_key] = default_carrier
                    logger.debug(f"Auto-registered carrier '{carrier_key}' from CONFIG")

    def _assign_element_colors(self) -> None:
        """Auto-assign colors to elements that don't have explicit colors set.

        Components and buses without explicit colors are assigned colors from the
        default qualitative colorscale. This ensures zero-config color support
        while still allowing users to override with explicit colors.
        """
        from .color_processing import process_colors

        # Collect elements without colors (components only - buses use carrier colors)
        # Use label_full for consistent keying with ElementContainer
        elements_without_colors = [comp.label_full for comp in self.components.values() if comp.color is None]

        if not elements_without_colors:
            return

        # Generate colors from the default colorscale
        colorscale = CONFIG.Plotting.default_qualitative_colorscale
        color_mapping = process_colors(colorscale, elements_without_colors)

        # Assign colors to elements
        for label_full, color in color_mapping.items():
            self.components[label_full].color = color
            logger.debug(f"Auto-assigned color '{color}' to component '{label_full}'")

    def add_elements(self, *elements: Element) -> None:
        """
        Add Components(Storages, Boilers, Heatpumps, ...), Buses or Effects to the FlowSystem

        Args:
            *elements: childs of  Element like Boiler, HeatPump, Bus,...
                modeling Elements

        Raises:
            RuntimeError: If the FlowSystem is locked (has a solution).
                Call `reset()` to unlock it first.
        """
        if self.is_locked:
            raise RuntimeError(
                'Cannot add elements to a FlowSystem that has a solution. '
                'Call `reset()` first to clear the solution and allow modifications.'
            )

        if self.model is not None:
            warnings.warn(
                'Adding elements to a FlowSystem with an existing model. The model will be invalidated.',
                stacklevel=2,
            )
        # Always invalidate when adding elements to ensure new elements get transformed
        if self.model is not None or self._connected_and_transformed:
            self._invalidate_model()

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

    def add_carriers(self, *carriers: Carrier) -> None:
        """Register a custom carrier for this FlowSystem.

        Custom carriers registered on the FlowSystem take precedence over
        CONFIG.Carriers defaults when resolving colors and units for buses.

        Args:
            carriers: Carrier objects defining the carrier properties.

        Raises:
            RuntimeError: If the FlowSystem is locked (has a solution).
                Call `reset()` to unlock it first.

        Examples:
            ```python
            import flixopt as fx

            fs = fx.FlowSystem(timesteps)

            # Define and register custom carriers
            biogas = fx.Carrier('biogas', '#228B22', 'kW', 'Biogas fuel')
            fs.add_carriers(biogas)

            # Now buses can reference this carrier by name
            bus = fx.Bus('BioGasNetwork', carrier='biogas')
            fs.add_elements(bus)

            # The carrier color will be used in plots automatically
            ```
        """
        if self.is_locked:
            raise RuntimeError(
                'Cannot add carriers to a FlowSystem that has a solution. '
                'Call `reset()` first to clear the solution and allow modifications.'
            )

        if self.model is not None:
            warnings.warn(
                'Adding carriers to a FlowSystem with an existing model. The model will be invalidated.',
                stacklevel=2,
            )
        # Always invalidate when adding carriers to ensure proper re-transformation
        if self.model is not None or self._connected_and_transformed:
            self._invalidate_model()

        for carrier in list(carriers):
            if not isinstance(carrier, Carrier):
                raise TypeError(f'Expected Carrier object, got {type(carrier)}')
            self._carriers.add(carrier)
            logger.debug(f'Adding carrier {carrier} to FlowSystem')

    def get_carrier(self, label: str) -> Carrier | None:
        """Get the carrier for a bus or flow.

        Args:
            label: Bus label (e.g., 'Fernwärme') or flow label (e.g., 'Boiler(Q_th)').

        Returns:
            Carrier or None if not found.

        Note:
            To access a carrier directly by name, use ``flow_system.carriers['electricity']``.

        Raises:
            RuntimeError: If FlowSystem is not connected_and_transformed.
        """
        if not self.connected_and_transformed:
            raise RuntimeError(
                'FlowSystem is not connected_and_transformed. Call FlowSystem.connect_and_transform() first.'
            )

        # Try as bus label
        bus = self.buses.get(label)
        if bus and bus.carrier:
            return self._carriers.get(bus.carrier.lower())

        # Try as flow label
        flow = self.flows.get(label)
        if flow and flow.bus:
            bus = self.buses.get(flow.bus)
            if bus and bus.carrier:
                return self._carriers.get(bus.carrier.lower())

        return None

    @property
    def carriers(self) -> CarrierContainer:
        """Carriers registered on this FlowSystem."""
        return self._carriers

    @property
    def flow_carriers(self) -> dict[str, str]:
        """Cached mapping of flow labels to carrier names.

        Returns:
            Dict mapping flow label to carrier name (lowercase).
            Flows without a carrier are not included.

        Raises:
            RuntimeError: If FlowSystem is not connected_and_transformed.
        """
        if not self.connected_and_transformed:
            raise RuntimeError(
                'FlowSystem is not connected_and_transformed. Call FlowSystem.connect_and_transform() first.'
            )

        if self._flow_carriers is None:
            self._flow_carriers = {}
            for flow_label, flow in self.flows.items():
                bus = self.buses.get(flow.bus)
                if bus and bus.carrier:
                    self._flow_carriers[flow_label] = bus.carrier.lower()

        return self._flow_carriers

    def create_model(self, normalize_weights: bool | None = None) -> FlowSystemModel:
        """
        Create a linopy model from the FlowSystem.

        Args:
            normalize_weights: Deprecated. Scenario weights are now always normalized in FlowSystem.
        """
        if normalize_weights is not None:
            warnings.warn(
                f'\n\nnormalize_weights parameter is deprecated and will be removed in {DEPRECATION_REMOVAL_VERSION}. '
                'Scenario weights are now always normalized when set on FlowSystem.\n',
                DeprecationWarning,
                stacklevel=2,
            )
        if not self.connected_and_transformed:
            raise RuntimeError(
                'FlowSystem is not connected_and_transformed. Call FlowSystem.connect_and_transform() first.'
            )
        # System integrity was already validated in connect_and_transform()
        self.model = FlowSystemModel(self)
        return self.model

    def build_model(self, normalize_weights: bool | None = None) -> FlowSystem:
        """
        Build the optimization model for this FlowSystem.

        This method prepares the FlowSystem for optimization by:
        1. Connecting and transforming all elements (if not already done)
        2. Creating the FlowSystemModel with all variables and constraints
        3. Adding clustering constraints (if this is a clustered FlowSystem)
        4. Adding typical periods modeling (if this is a reduced FlowSystem)

        After calling this method, `self.model` will be available for inspection
        before solving.

        Args:
            normalize_weights: Deprecated. Scenario weights are now always normalized in FlowSystem.

        Returns:
            Self, for method chaining.

        Examples:
            >>> flow_system.build_model()
            >>> print(flow_system.model.variables)  # Inspect variables before solving
            >>> flow_system.solve(solver)
        """
        if normalize_weights is not None:
            warnings.warn(
                f'\n\nnormalize_weights parameter is deprecated and will be removed in {DEPRECATION_REMOVAL_VERSION}. '
                'Scenario weights are now always normalized when set on FlowSystem.\n',
                DeprecationWarning,
                stacklevel=2,
            )
        self.connect_and_transform()
        self.create_model()

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
            progress=CONFIG.Solving.log_to_console,
            **solver.options,
        )

        if self.model.termination_condition in ('infeasible', 'infeasible_or_unbounded'):
            if CONFIG.Solving.compute_infeasibilities:
                import io
                from contextlib import redirect_stdout

                f = io.StringIO()

                # Redirect stdout to our buffer
                with redirect_stdout(f):
                    self.model.print_infeasibilities()

                infeasibilities = f.getvalue()
                logger.error('Successfully extracted infeasibilities: \n%s', infeasibilities)
            raise RuntimeError(f'Model was infeasible. Status: {self.model.status}. Check your constraints and bounds.')

        # Store solution on FlowSystem for direct Element access
        self.solution = self.model.solution

        logger.info(f'Optimization solved successfully. Objective: {self.model.objective.value:.4f}')

        return self

    @property
    def solution(self) -> xr.Dataset | None:
        """
        Access the optimization solution as an xarray Dataset.

        The solution is indexed by ``timesteps_extra`` (the original timesteps plus
        one additional timestep at the end). Variables that do not have data for the
        extra timestep (most variables except storage charge states) will contain
        NaN values at the final timestep.

        Returns:
            xr.Dataset: The solution dataset with all optimization variable results,
                or None if the model hasn't been solved yet.

        Example:
            >>> flow_system.optimize(solver)
            >>> flow_system.solution.isel(time=slice(None, -1))  # Exclude trailing NaN (and final charge states)
        """
        return self._solution

    @solution.setter
    def solution(self, value: xr.Dataset | None) -> None:
        """Set the solution dataset and invalidate statistics cache."""
        self._solution = value
        self._statistics = None  # Invalidate cached statistics

    @property
    def is_locked(self) -> bool:
        """Check if the FlowSystem is locked (has a solution).

        A locked FlowSystem cannot be modified. Use `reset()` to unlock it.
        """
        return self._solution is not None

    def _invalidate_model(self) -> None:
        """Invalidate the model and element submodels when structure changes.

        This clears the model, resets the ``connected_and_transformed`` flag,
        clears all element submodels and variable/constraint names, and invalidates
        the topology accessor cache.

        Called internally by :meth:`add_elements`, :meth:`add_carriers`,
        :meth:`reset`, and :meth:`invalidate`.

        See Also:
            :meth:`invalidate`: Public method for manual invalidation.
            :meth:`reset`: Clears solution and invalidates (for locked FlowSystems).
        """
        self.model = None
        self._connected_and_transformed = False
        self._topology = None  # Invalidate topology accessor (and its cached colors)
        self._flow_carriers = None  # Invalidate flow-to-carrier mapping
        for element in self.values():
            element.submodel = None
            element._variable_names = []
            element._constraint_names = []

    def reset(self) -> FlowSystem:
        """Clear optimization state to allow modifications.

        This method unlocks the FlowSystem by clearing:
        - The solution dataset
        - The optimization model
        - All element submodels and variable/constraint names
        - The connected_and_transformed flag

        After calling reset(), the FlowSystem can be modified again
        (e.g., adding elements or carriers).

        Returns:
            Self, for method chaining.

        Examples:
            >>> flow_system.optimize(solver)  # FlowSystem is now locked
            >>> flow_system.add_elements(new_bus)  # Raises RuntimeError
            >>> flow_system.reset()  # Unlock the FlowSystem
            >>> flow_system.add_elements(new_bus)  # Now works
        """
        self.solution = None  # Also clears _statistics via setter
        self._invalidate_model()
        return self

    def invalidate(self) -> FlowSystem:
        """Invalidate the model to allow re-transformation after modifying elements.

        Call this after modifying existing element attributes (e.g., ``flow.size``,
        ``flow.relative_minimum``) to ensure changes take effect on the next
        optimization. The next call to :meth:`optimize` or :meth:`build_model`
        will re-run :meth:`connect_and_transform`.

        Note:
            Adding new elements via :meth:`add_elements` automatically invalidates
            the model. This method is only needed when modifying attributes of
            elements that are already part of the FlowSystem.

        Returns:
            Self, for method chaining.

        Raises:
            RuntimeError: If the FlowSystem has a solution. Call :meth:`reset`
                first to clear the solution.

        Examples:
            Modify a flow's size and re-optimize:

            >>> flow_system.optimize(solver)
            >>> flow_system.reset()  # Clear solution first
            >>> flow_system.components['Boiler'].inputs[0].size = 200
            >>> flow_system.invalidate()
            >>> flow_system.optimize(solver)  # Re-runs connect_and_transform

            Modify before first optimization:

            >>> flow_system.connect_and_transform()
            >>> # Oops, need to change something
            >>> flow_system.components['Boiler'].inputs[0].size = 200
            >>> flow_system.invalidate()
            >>> flow_system.optimize(solver)  # Changes take effect
        """
        if self.is_locked:
            raise RuntimeError(
                'Cannot invalidate a FlowSystem with a solution. Call `reset()` first to clear the solution.'
            )
        self._invalidate_model()
        return self

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
    def transform(self) -> TransformAccessor:
        """
        Access transformation methods for this FlowSystem.

        This property returns a TransformAccessor that provides methods to create
        transformed versions of this FlowSystem (e.g., clustered for time aggregation).

        Returns:
            A TransformAccessor instance.

        Examples:
            Clustered optimization:

            >>> params = ClusteringParameters(hours_per_period=24, nr_of_periods=8)
            >>> clustered_fs = flow_system.transform.cluster(params)
            >>> clustered_fs.optimize(solver)
            >>> print(clustered_fs.solution)
        """
        return TransformAccessor(self)

    @property
    def statistics(self) -> StatisticsAccessor:
        """
        Access statistics and plotting methods for optimization results.

        This property returns a StatisticsAccessor that provides methods to analyze
        and visualize optimization results stored in this FlowSystem's solution.

        Note:
            The FlowSystem must have a solution (from optimize() or solve()) before
            most statistics methods can be used.

        Returns:
            A cached StatisticsAccessor instance.

        Examples:
            After optimization:

            >>> flow_system.optimize(solver)
            >>> flow_system.statistics.plot.balance('ElectricityBus')
            >>> flow_system.statistics.plot.heatmap('Boiler|on')
            >>> ds = flow_system.statistics.flow_rates  # Get data for analysis
        """
        if self._statistics is None:
            self._statistics = StatisticsAccessor(self)
        return self._statistics

    @property
    def topology(self) -> TopologyAccessor:
        """
        Access network topology inspection and visualization methods.

        This property returns a cached TopologyAccessor that provides methods to inspect
        the network structure and visualize it. The accessor is invalidated when the
        FlowSystem structure changes (via reset() or invalidate()).

        Returns:
            A cached TopologyAccessor instance.

        Examples:
            Visualize the network:

            >>> flow_system.topology.plot()
            >>> flow_system.topology.plot(path='my_network.html', show=True)

            Interactive visualization:

            >>> flow_system.topology.start_app()
            >>> # ... interact with the visualization ...
            >>> flow_system.topology.stop_app()

            Get network structure info:

            >>> nodes, edges = flow_system.topology.infos()
        """
        if self._topology is None:
            self._topology = TopologyAccessor(self)
        return self._topology

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
        Deprecated: Use `flow_system.topology.plot()` instead.

        Visualizes the network structure of a FlowSystem using PyVis.
        """
        warnings.warn(
            f'plot_network() is deprecated and will be removed in v{DEPRECATION_REMOVAL_VERSION}. '
            'Use flow_system.topology.plot() instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.topology.plot_legacy(path=path, controls=controls, show=show)

    def start_network_app(self) -> None:
        """
        Deprecated: Use `flow_system.topology.start_app()` instead.

        Visualizes the network structure using Dash and Cytoscape.
        """
        warnings.warn(
            f'start_network_app() is deprecated and will be removed in v{DEPRECATION_REMOVAL_VERSION}. '
            'Use flow_system.topology.start_app() instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        self.topology.start_app()

    def stop_network_app(self) -> None:
        """
        Deprecated: Use `flow_system.topology.stop_app()` instead.

        Stop the network visualization server.
        """
        warnings.warn(
            f'stop_network_app() is deprecated and will be removed in v{DEPRECATION_REMOVAL_VERSION}. '
            'Use flow_system.topology.stop_app() instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        self.topology.stop_app()

    def network_infos(self) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
        """
        Deprecated: Use `flow_system.topology.infos()` instead.

        Get network topology information as dictionaries.
        """
        warnings.warn(
            f'network_infos() is deprecated and will be removed in v{DEPRECATION_REMOVAL_VERSION}. '
            'Use flow_system.topology.infos() instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.topology.infos()

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
            effect.link_to_flow_system(self)  # Link element to FlowSystem
        self.effects.add_effects(*args)

    def _add_components(self, *components: Component) -> None:
        for new_component in list(components):
            new_component.link_to_flow_system(self)  # Link element to FlowSystem
            self.components.add(new_component)  # Add to existing components
        # Invalidate cache once after all additions
        if components:
            self._flows_cache = None
            self._storages_cache = None

    def _add_buses(self, *buses: Bus):
        for new_bus in list(buses):
            new_bus.link_to_flow_system(self)  # Link element to FlowSystem
            self.buses.add(new_bus)  # Add to existing buses
        # Invalidate cache once after all additions
        if buses:
            self._flows_cache = None
            self._storages_cache = None

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

        # Timestep info - handle both DatetimeIndex and RangeIndex (segmented)
        if self.is_segmented:
            r += f'Timesteps: {len(self.timesteps)} segments (segmented)\n'
        else:
            time_period = f'{self.timesteps[0].date()} to {self.timesteps[-1].date()}'
            freq_str = (
                str(self.timesteps.freq).replace('<', '').replace('>', '') if self.timesteps.freq else 'irregular'
            )
            r += f'Timesteps: {len(self.timesteps)} ({freq_str}) [{time_period}]\n'

        # Add clusters if present
        if self.clusters is not None:
            r += f'Clusters: {len(self.clusters)}\n'

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
    def storages(self) -> ElementContainer[Storage]:
        """All storage components as an ElementContainer.

        Returns:
            ElementContainer containing all Storage components in the FlowSystem,
            sorted by label for reproducibility.
        """
        if self._storages_cache is None:
            storages = [c for c in self.components.values() if isinstance(c, Storage)]
            storages = sorted(storages, key=lambda s: s.label_full.lower())
            self._storages_cache = ElementContainer(storages, element_type_name='storages', truncate_repr=10)
        return self._storages_cache

    @property
    def dims(self) -> list[str]:
        """Active dimension names.

        Returns:
            List of active dimension names in order.

        Example:
            >>> fs.dims
            ['time']  # simple case
            >>> fs_clustered.dims
            ['cluster', 'time', 'period', 'scenario']  # full case
        """
        result = []
        if self.clusters is not None:
            result.append('cluster')
        result.append('time')
        if self.periods is not None:
            result.append('period')
        if self.scenarios is not None:
            result.append('scenario')
        return result

    @property
    def indexes(self) -> dict[str, pd.Index]:
        """Indexes for active dimensions.

        Returns:
            Dict mapping dimension names to pandas Index objects.

        Example:
            >>> fs.indexes['time']
            DatetimeIndex(['2024-01-01', ...], dtype='datetime64[ns]', name='time')
        """
        result: dict[str, pd.Index] = {}
        if self.clusters is not None:
            result['cluster'] = self.clusters
        result['time'] = self.timesteps
        if self.periods is not None:
            result['period'] = self.periods
        if self.scenarios is not None:
            result['scenario'] = self.scenarios
        return result

    @property
    def temporal_dims(self) -> list[str]:
        """Temporal dimensions for summing over time.

        Returns ['time', 'cluster'] for clustered systems, ['time'] otherwise.
        """
        if self.clusters is not None:
            return ['time', 'cluster']
        return ['time']

    @property
    def temporal_weight(self) -> xr.DataArray:
        """Combined temporal weight (timestep_duration × cluster_weight).

        Use for converting rates to totals before summing.
        Note: cluster_weight is used even without a clusters dimension.
        """
        # Use cluster_weight directly if set, otherwise check weights dict, fallback to 1.0
        cluster_weight = self.weights.get('cluster', self.cluster_weight if self.cluster_weight is not None else 1.0)
        return self.weights['time'] * cluster_weight

    @property
    def coords(self) -> dict[FlowSystemDimensions, pd.Index]:
        """Active coordinates for variable creation.

        .. deprecated::
            Use :attr:`indexes` instead.

        Returns a dict of dimension names to coordinate arrays. When clustered,
        includes 'cluster' dimension before 'time'.

        Returns:
            Dict mapping dimension names to coordinate arrays.
        """
        warnings.warn(
            f'FlowSystem.coords is deprecated and will be removed in v{DEPRECATION_REMOVAL_VERSION}. '
            'Use FlowSystem.indexes instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.indexes

    @property
    def _use_true_cluster_dims(self) -> bool:
        """Check if true (cluster, time) dimensions should be used."""
        return self.clusters is not None

    @property
    def _cluster_n_clusters(self) -> int | None:
        """Get number of clusters."""
        return len(self.clusters) if self.clusters is not None else None

    @property
    def _cluster_timesteps_per_cluster(self) -> int | None:
        """Get timesteps per cluster (same as len(timesteps) for clustered systems)."""
        return len(self.timesteps) if self.clusters is not None else None

    @property
    def _cluster_time_coords(self) -> pd.DatetimeIndex | pd.RangeIndex | None:
        """Get time coordinates for clustered system (same as timesteps)."""
        return self.timesteps if self.clusters is not None else None

    @property
    def is_segmented(self) -> bool:
        """Check if this FlowSystem uses segmented time (RangeIndex instead of DatetimeIndex).

        Segmented systems have variable timestep durations stored in timestep_duration,
        and use a RangeIndex for time coordinates instead of DatetimeIndex.
        """
        return isinstance(self.timesteps, pd.RangeIndex)

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps (within each cluster if clustered)."""
        if self.is_clustered:
            return self.clustering.timesteps_per_cluster
        return len(self.timesteps)

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
        Set scenario weights (always normalized to sum to 1).

        Args:
            value: Scenario weights to set (will be converted to DataArray with 'scenario' dimension
                and normalized to sum to 1), or None to clear weights.

        Raises:
            ValueError: If value is not None and no scenarios are defined in the FlowSystem.
            ValueError: If weights sum to zero (cannot normalize).
        """
        if value is None:
            self._scenario_weights = None
            return

        if self.scenarios is None:
            raise ValueError(
                'FlowSystem.scenario_weights cannot be set when no scenarios are defined. '
                'Either define scenarios in FlowSystem(scenarios=...) or set scenario_weights to None.'
            )

        weights = self.fit_to_model_coords('scenario_weights', value, dims=['scenario'])

        # Normalize to sum to 1
        norm = weights.sum('scenario')
        if np.isclose(norm, 0.0).any().item():
            # Provide detailed error for multi-dimensional weights
            if norm.ndim > 0:
                zero_locations = np.argwhere(np.isclose(norm.values, 0.0))
                coords_info = ', '.join(
                    f'{dim}={norm.coords[dim].values[idx]}'
                    for idx, dim in zip(zero_locations[0], norm.dims, strict=False)
                )
                raise ValueError(
                    f'scenario_weights sum to 0 at {coords_info}; cannot normalize. '
                    f'Ensure all scenario weight combinations sum to a positive value.'
                )
            raise ValueError('scenario_weights sum to 0; cannot normalize.')
        self._scenario_weights = weights / norm

    def _unit_weight(self, dim: str) -> xr.DataArray:
        """Create a unit weight DataArray (all 1.0) for a dimension."""
        index = self.indexes[dim]
        return xr.DataArray(
            np.ones(len(index), dtype=float),
            coords={dim: index},
            dims=[dim],
            name=f'{dim}_weight',
        )

    @property
    def weights(self) -> dict[str, xr.DataArray]:
        """Weights for active dimensions (unit weights if not explicitly set).

        Returns:
            Dict mapping dimension names to weight DataArrays.
            Keys match :attr:`dims` and :attr:`indexes`.

        Example:
            >>> fs.weights['time']  # timestep durations
            >>> fs.weights['cluster']  # cluster weights (unit if not set)
        """
        result: dict[str, xr.DataArray] = {'time': self.timestep_duration}
        if self.clusters is not None:
            result['cluster'] = self.cluster_weight if self.cluster_weight is not None else self._unit_weight('cluster')
        if self.periods is not None:
            result['period'] = self.period_weights if self.period_weights is not None else self._unit_weight('period')
        if self.scenarios is not None:
            result['scenario'] = (
                self.scenario_weights if self.scenario_weights is not None else self._unit_weight('scenario')
            )
        return result

    def sum_temporal(self, data: xr.DataArray) -> xr.DataArray:
        """Sum data over temporal dimensions with full temporal weighting.

        Applies both timestep_duration and cluster_weight, then sums over temporal dimensions.
        Use this to convert rates to totals (e.g., flow_rate → total_energy).

        Args:
            data: Data with time dimension (and optionally cluster).
                  Typically a rate (e.g., flow_rate in MW, status as 0/1).

        Returns:
            Data summed over temporal dims with full temporal weighting applied.

        Example:
            >>> total_energy = fs.sum_temporal(flow_rate)  # MW → MWh total
            >>> active_hours = fs.sum_temporal(status)  # count → hours
        """
        return (data * self.temporal_weight).sum(self.temporal_dims)

    @property
    def is_clustered(self) -> bool:
        """Check if this FlowSystem uses time series clustering.

        Returns:
            True if the FlowSystem was created with transform.cluster(),
            False otherwise.

        Example:
            >>> fs_clustered = flow_system.transform.cluster(n_clusters=8, cluster_duration='1D')
            >>> fs_clustered.is_clustered
            True
            >>> flow_system.is_clustered
            False
        """
        return getattr(self, 'clustering', None) is not None

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
        warnings.warn(
            f'\n_dataset_sel() is deprecated and will be removed in {DEPRECATION_REMOVAL_VERSION}. '
            'Use TransformAccessor._dataset_sel() instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        from .transform_accessor import TransformAccessor

        return TransformAccessor._dataset_sel(
            dataset,
            time=time,
            period=period,
            scenario=scenario,
            hours_of_last_timestep=hours_of_last_timestep,
            hours_of_previous_timesteps=hours_of_previous_timesteps,
        )

    def sel(
        self,
        time: str | slice | list[str] | pd.Timestamp | pd.DatetimeIndex | None = None,
        period: int | slice | list[int] | pd.Index | None = None,
        scenario: str | slice | list[str] | pd.Index | None = None,
    ) -> FlowSystem:
        """
        Select a subset of the flowsystem by label.

        .. deprecated::
            Use ``flow_system.transform.sel()`` instead. Will be removed in v6.0.0.

        Args:
            time: Time selection (e.g., slice('2023-01-01', '2023-12-31'), '2023-06-15')
            period: Period selection (e.g., slice(2023, 2024), or list of periods)
            scenario: Scenario selection (e.g., 'scenario1', or list of scenarios)

        Returns:
            FlowSystem: New FlowSystem with selected data (no solution).
        """
        warnings.warn(
            f'\nsel() is deprecated and will be removed in {DEPRECATION_REMOVAL_VERSION}. '
            'Use flow_system.transform.sel() instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.transform.sel(time=time, period=period, scenario=scenario)

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
        warnings.warn(
            f'\n_dataset_isel() is deprecated and will be removed in {DEPRECATION_REMOVAL_VERSION}. '
            'Use TransformAccessor._dataset_isel() instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        from .transform_accessor import TransformAccessor

        return TransformAccessor._dataset_isel(
            dataset,
            time=time,
            period=period,
            scenario=scenario,
            hours_of_last_timestep=hours_of_last_timestep,
            hours_of_previous_timesteps=hours_of_previous_timesteps,
        )

    def isel(
        self,
        time: int | slice | list[int] | None = None,
        period: int | slice | list[int] | None = None,
        scenario: int | slice | list[int] | None = None,
    ) -> FlowSystem:
        """
        Select a subset of the flowsystem by integer indices.

        .. deprecated::
            Use ``flow_system.transform.isel()`` instead. Will be removed in v6.0.0.

        Args:
            time: Time selection by integer index (e.g., slice(0, 100), 50, or [0, 5, 10])
            period: Period selection by integer index
            scenario: Scenario selection by integer index

        Returns:
            FlowSystem: New FlowSystem with selected data (no solution).
        """
        warnings.warn(
            f'\nisel() is deprecated and will be removed in {DEPRECATION_REMOVAL_VERSION}. '
            'Use flow_system.transform.isel() instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.transform.isel(time=time, period=period, scenario=scenario)

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
        warnings.warn(
            f'\n_dataset_resample() is deprecated and will be removed in {DEPRECATION_REMOVAL_VERSION}. '
            'Use TransformAccessor._dataset_resample() instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        from .transform_accessor import TransformAccessor

        return TransformAccessor._dataset_resample(
            dataset,
            freq=freq,
            method=method,
            hours_of_last_timestep=hours_of_last_timestep,
            hours_of_previous_timesteps=hours_of_previous_timesteps,
            **kwargs,
        )

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

        .. deprecated::
            Use ``TransformAccessor._resample_by_dimension_groups()`` instead.
            Will be removed in v6.0.0.
        """
        warnings.warn(
            f'\n_resample_by_dimension_groups() is deprecated and will be removed in {DEPRECATION_REMOVAL_VERSION}. '
            'Use TransformAccessor._resample_by_dimension_groups() instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        from .transform_accessor import TransformAccessor

        return TransformAccessor._resample_by_dimension_groups(time_dataset, time, method, **kwargs)

    def resample(
        self,
        time: str,
        method: Literal['mean', 'sum', 'max', 'min', 'first', 'last', 'std', 'var', 'median', 'count'] = 'mean',
        hours_of_last_timestep: int | float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
        **kwargs: Any,
    ) -> FlowSystem:
        """
        Create a resampled FlowSystem by resampling data along the time dimension.

        .. deprecated::
            Use ``flow_system.transform.resample()`` instead. Will be removed in v6.0.0.

        Args:
            time: Resampling frequency (e.g., '3h', '2D', '1M')
            method: Resampling method. Recommended: 'mean', 'first', 'last', 'max', 'min'
            hours_of_last_timestep: Duration of the last timestep after resampling.
            hours_of_previous_timesteps: Duration of previous timesteps after resampling.
            **kwargs: Additional arguments passed to xarray.resample()

        Returns:
            FlowSystem: New resampled FlowSystem (no solution).
        """
        warnings.warn(
            f'\nresample() is deprecated and will be removed in {DEPRECATION_REMOVAL_VERSION}. '
            'Use flow_system.transform.resample() instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.transform.resample(
            time=time,
            method=method,
            hours_of_last_timestep=hours_of_last_timestep,
            hours_of_previous_timesteps=hours_of_previous_timesteps,
            **kwargs,
        )

    @property
    def connected_and_transformed(self) -> bool:
        return self._connected_and_transformed
