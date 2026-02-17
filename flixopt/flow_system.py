"""
This module contains the FlowSystem class, which is used to collect instances of many other classes by the end User.
"""

from __future__ import annotations

import json
import logging
import pathlib
import warnings
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import xarray as xr

from . import io as fx_io
from .batched import BatchedAccessor
from .components import Converter, Port, Storage, Transmission
from .config import CONFIG, DEPRECATION_REMOVAL_VERSION
from .core import (
    ConversionError,
    DataConverter,
    FlowSystemDimensions,
    TimeSeriesData,
)
from .effects import Effect, EffectCollection
from .elements import Bus, Component, Flow
from .flow_system_status import FlowSystemStatus, get_status, invalidate_to_status
from .id_list import IdList, element_id_list
from .model_coordinates import ModelCoordinates
from .optimize_accessor import OptimizeAccessor
from .statistics_accessor import StatisticsAccessor
from .structure import (
    CompositeContainerMixin,
    Element,
    FlowSystemModel,
    create_reference_structure,
    replace_references_with_stats,
)
from .topology_accessor import TopologyAccessor
from .transform_accessor import TransformAccessor

if TYPE_CHECKING:
    from collections.abc import Collection

    import numpy as np
    import pyvis

    from .clustering import Clustering
    from .solvers import _Solver
    from .types import Effect_TPS, Numeric_S, Numeric_TPS, NumericOrBool

from .carrier import Carrier, CarrierContainer

# Register clustering classes for IO (deferred to avoid circular imports)
from .clustering.base import _register_clustering_classes

_register_clustering_classes()

logger = logging.getLogger('flixopt')


class LegacySolutionWrapper:
    """Wrapper for xr.Dataset that provides legacy solution access patterns.

    When CONFIG.Legacy.solution_access is True, this wrapper intercepts
    __getitem__ calls to translate legacy access patterns like:
        fs.solution['costs'] -> fs.solution['effect|total'].sel(effect='costs')
        fs.solution['Src(heat)|flow_rate'] -> fs.solution['flow|rate'].sel(flow='Src(heat)')

    All other operations are proxied directly to the underlying Dataset.
    """

    __slots__ = ('_dataset',)

    # Mapping from old variable suffixes to new type|variable format
    # Format: old_suffix -> (dimension, new_variable_suffix)
    _LEGACY_VAR_MAP = {
        # Flow variables
        'flow_rate': ('flow', 'rate'),
        'size': ('flow', 'size'),  # For flows: Comp(flow)|size
        'status': ('flow', 'status'),
        'invested': ('flow', 'invested'),
    }

    # Storage-specific mappings (no parentheses in id, e.g., 'Battery|size')
    _LEGACY_STORAGE_VAR_MAP = {
        'size': ('storage', 'size'),
        'invested': ('storage', 'invested'),
        'charge_state': ('storage', 'charge'),  # Old: charge_state -> New: charge
    }

    def __init__(self, dataset: xr.Dataset) -> None:
        object.__setattr__(self, '_dataset', dataset)

    def __getitem__(self, key):
        ds = object.__getattribute__(self, '_dataset')
        try:
            return ds[key]
        except KeyError as e:
            if not isinstance(key, str):
                raise e

            # Try legacy effect access patterns
            if 'effect' in ds.coords:
                # Pattern: 'costs' -> 'effect|total'.sel(effect='costs')
                if key in ds.coords['effect'].values:
                    warnings.warn(
                        f"Legacy solution access: solution['{key}'] is deprecated. "
                        f"Use solution['effect|total'].sel(effect='{key}') instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    return ds['effect|total'].sel(effect=key)

                # Pattern: 'costs(periodic)' -> 'effect|periodic'.sel(effect='costs')
                # Pattern: 'costs(temporal)' -> 'effect|temporal'.sel(effect='costs')
                import re

                match = re.match(r'^(.+)\((periodic|temporal)\)$', key)
                if match:
                    effect_name, aspect = match.groups()
                    if effect_name in ds.coords['effect'].values:
                        new_key = f'effect|{aspect}'
                        if new_key in ds:
                            warnings.warn(
                                f"Legacy solution access: solution['{key}'] is deprecated. "
                                f"Use solution['{new_key}'].sel(effect='{effect_name}') instead.",
                                DeprecationWarning,
                                stacklevel=2,
                            )
                            return ds[new_key].sel(effect=effect_name)

                # Pattern: 'costs(temporal)|per_timestep' -> 'effect|per_timestep'.sel(effect='costs')
                if '|' in key:
                    prefix, suffix = key.rsplit('|', 1)
                    match = re.match(r'^(.+)\((temporal|periodic)\)$', prefix)
                    if match:
                        effect_name, aspect = match.groups()
                        if effect_name in ds.coords['effect'].values:
                            new_key = f'effect|{suffix}'
                            if new_key in ds:
                                warnings.warn(
                                    f"Legacy solution access: solution['{key}'] is deprecated. "
                                    f"Use solution['{new_key}'].sel(effect='{effect_name}') instead.",
                                    DeprecationWarning,
                                    stacklevel=2,
                                )
                                return ds[new_key].sel(effect=effect_name)

            # Try legacy flow/storage access: solution['Src(heat)|flow_rate'] -> solution['flow|rate'].sel(flow='Src(heat)')
            if '|' in key:
                parts = key.rsplit('|', 1)
                if len(parts) == 2:
                    element_id, var_suffix = parts

                    # Try flow variables first (ids have parentheses like 'Src(heat)')
                    if var_suffix in self._LEGACY_VAR_MAP:
                        dim, var_name = self._LEGACY_VAR_MAP[var_suffix]
                        new_key = f'{dim}|{var_name}'
                        if new_key in ds and dim in ds.coords and element_id in ds.coords[dim].values:
                            warnings.warn(
                                f"Legacy solution access: solution['{key}'] is deprecated. "
                                f"Use solution['{new_key}'].sel({dim}='{element_id}') instead.",
                                DeprecationWarning,
                                stacklevel=2,
                            )
                            return ds[new_key].sel({dim: element_id})

                    # Try storage variables (ids without parentheses like 'Battery')
                    if var_suffix in self._LEGACY_STORAGE_VAR_MAP:
                        dim, var_name = self._LEGACY_STORAGE_VAR_MAP[var_suffix]
                        new_key = f'{dim}|{var_name}'
                        if new_key in ds and dim in ds.coords and element_id in ds.coords[dim].values:
                            warnings.warn(
                                f"Legacy solution access: solution['{key}'] is deprecated. "
                                f"Use solution['{new_key}'].sel({dim}='{element_id}') instead.",
                                DeprecationWarning,
                                stacklevel=2,
                            )
                            return ds[new_key].sel({dim: element_id})

            raise e

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_dataset'), name)

    def __setattr__(self, name, value):
        if name == '_dataset':
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, '_dataset'), name, value)

    def __repr__(self):
        return repr(object.__getattribute__(self, '_dataset'))

    def __iter__(self):
        return iter(object.__getattribute__(self, '_dataset'))

    def __len__(self):
        return len(object.__getattribute__(self, '_dataset'))

    def __contains__(self, key):
        return key in object.__getattribute__(self, '_dataset')


class FlowSystem(CompositeContainerMixin[Element]):
    """
    A FlowSystem organizes the high level Elements (Components, Buses, Effects & Flows).

    This is the main container class that users work with to build and manage their energy or material flow system.
    FlowSystem provides both direct container access (via .components, .buses, .effects, .flows) and a unified
    dict-like interface for accessing any element by id across all container types.

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
            - list[str]: Only specified components (by id) are equalized across scenarios
        scenario_independent_flow_rates: Controls whether flow rates are equalized across scenarios.
            - True: All flow rates are shared/equalized across scenarios
            - False: All flow rates are optimized separately per scenario
            - list[str]: Only specified flows (by id) are equalized across scenarios

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
        >>> flow_system.add(boiler, heat_bus, costs)

        Unified dict-like access (recommended for most cases):

        >>> # Access any element by id, regardless of type
        >>> boiler = flow_system['Boiler']  # Returns Component
        >>> heat_bus = flow_system['Heat']  # Returns Bus
        >>> costs = flow_system['costs']  # Returns Effect
        >>>
        >>> # Check if element exists
        >>> if 'Boiler' in flow_system:
        ...     print('Boiler found in system')
        >>>
        >>> # Iterate over all elements
        >>> for element_id in flow_system.keys():
        ...     element = flow_system[element_id]
        ...     print(f'{element_id}: {type(element).__name__}')
        >>>
        >>> # Get all element ids and objects
        >>> all_ids = list(flow_system.keys())
        >>> all_elements = list(flow_system.values())
        >>> for element_id, element in flow_system.items():
        ...     print(f'{element_id}: {element}')

        Direct container access for type-specific operations:

        >>> # Access specific container when you need type filtering
        >>> for component in flow_system.components.values():
        ...     print(f'{component.id}: {len(component.inputs)} inputs')
        >>>
        >>> # Access buses directly
        >>> for bus in flow_system.buses.values():
        ...     print(f'{bus.id}')
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
        ...     print(f'{flow.id}: {flow.size}')
        >>>
        >>> # Access effects
        >>> for effect in flow_system.effects.values():
        ...     print(f'{effect.id}')

    Notes:
        - The dict-like interface (`flow_system['element']`) searches across all containers
          (components, buses, effects, flows) to find the element with the matching id.
        - Element ids must be unique across all container types. Attempting to add
          elements with duplicate ids will raise an error, ensuring each id maps to exactly one element.
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
        self.model_coords = ModelCoordinates(
            timesteps=timesteps,
            periods=periods,
            scenarios=scenarios,
            clusters=clusters,
            hours_of_last_timestep=hours_of_last_timestep,
            hours_of_previous_timesteps=hours_of_previous_timesteps,
            weight_of_last_period=weight_of_last_period,
            scenario_weights=scenario_weights,
            cluster_weight=cluster_weight,
            timestep_duration=timestep_duration,
            fit_to_model_coords=self.fit_to_model_coords,
        )

        # Element collections — component sub-containers
        self.converters: IdList[Converter] = element_id_list(display_name='converters', truncate_repr=10)
        self.ports: IdList[Port] = element_id_list(display_name='ports', truncate_repr=10)
        self.storages: IdList[Storage] = element_id_list(display_name='storages', truncate_repr=10)
        self.transmissions: IdList[Transmission] = element_id_list(display_name='transmissions', truncate_repr=10)
        self.buses: IdList[Bus] = element_id_list(display_name='buses', truncate_repr=10)
        self.effects: EffectCollection = EffectCollection(truncate_repr=10)
        self.model: FlowSystemModel | None = None

        self._connected_and_transformed = False
        self._used_in_optimization = False

        # Registry for runtime state (populated during model building, not stored on elements)
        self._element_variable_names: dict[str, list[str]] = {}
        self._element_constraint_names: dict[str, list[str]] = {}
        self._registered_elements: set[int] = set()  # Python id() for ownership check

        self._network_app = None
        self._flows_cache: IdList[Flow] | None = None
        self._components_cache: IdList | None = None

        # Solution dataset - populated after optimization or loaded from file
        self._solution: xr.Dataset | None = None

        # Aggregation info - populated by transform.cluster()
        self._clustering: Clustering | None = None

        # Statistics accessor cache - lazily initialized, invalidated on new solution
        self._statistics: StatisticsAccessor | None = None

        # Topology accessor cache - lazily initialized, invalidated on structure change
        self._topology: TopologyAccessor | None = None

        # Batched data accessor - provides indexed/batched access to element properties
        self._batched: BatchedAccessor | None = None

        # Carrier container - local carriers override CONFIG.Carriers
        self._carriers: CarrierContainer = CarrierContainer()

        # Cached flow→carrier mapping (built lazily after connect_and_transform)
        self._flow_carriers: dict[str, str] | None = None

        # Use properties to validate and store scenario dimension settings
        self.scenario_independent_sizes = scenario_independent_sizes
        self.scenario_independent_flow_rates = scenario_independent_flow_rates

        # Optional name for identification (derived from filename on load)
        self.name = name

    def _create_reference_structure(self) -> tuple[dict, dict[str, xr.DataArray]]:
        """
        Override Interface method to handle FlowSystem-specific serialization.

        Uses path-based DataArray keys via standalone ``create_reference_structure``:
        ``components.{id}.param``, ``buses.{id}.param``, ``effects.{id}.param``.

        Returns:
            Tuple of (reference_structure, extracted_arrays_dict)
        """
        coords = self.indexes

        # Start with standalone function for FlowSystem's own constructor params
        reference_structure, all_extracted_arrays = create_reference_structure(self, coords=coords)

        # Remove timesteps, as it's directly stored in dataset index
        reference_structure.pop('timesteps', None)

        # Extract from component containers with path prefix
        for container_key, container in [
            ('converters', self.converters),
            ('ports', self.ports),
            ('storages', self.storages),
            ('transmissions', self.transmissions),
        ]:
            container_structure = {}
            for comp_id, component in container.items():
                comp_structure, comp_arrays = create_reference_structure(
                    component, f'{container_key}|{comp_id}', coords=coords
                )
                all_extracted_arrays.update(comp_arrays)
                container_structure[comp_id] = comp_structure
            if container_structure:
                reference_structure[container_key] = container_structure

        # Extract from buses with path prefix
        buses_structure = {}
        for bus_id, bus in self.buses.items():
            bus_structure, bus_arrays = create_reference_structure(bus, f'buses|{bus_id}', coords=coords)
            all_extracted_arrays.update(bus_arrays)
            buses_structure[bus_id] = bus_structure
        reference_structure['buses'] = buses_structure

        # Extract from effects with path prefix
        effects_structure = {}
        for effect in self.effects.values():
            effect_structure, effect_arrays = create_reference_structure(effect, f'effects|{effect.id}', coords=coords)
            all_extracted_arrays.update(effect_arrays)
            effects_structure[effect.id] = effect_structure
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

        See Also:
            from_dataset: Create FlowSystem from dataset
            to_netcdf: Save to NetCDF file
        """
        if not self.connected_and_transformed:
            logger.info('FlowSystem is not connected_and_transformed. Connecting and transforming data now.')
            self.connect_and_transform()

        # Build base dataset from FlowSystem's own _create_reference_structure
        reference_structure, extracted_arrays = self._create_reference_structure()
        base_ds = xr.Dataset(extracted_arrays, attrs=reference_structure)

        # Add FlowSystem-specific data (solution, clustering, metadata)
        return fx_io.flow_system_to_dataset(self, base_ds, include_solution, include_original_data)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> FlowSystem:
        """
        Create a FlowSystem from an xarray Dataset.

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

        See Also:
            to_dataset: Convert FlowSystem to dataset
            from_netcdf: Load from NetCDF file
        """
        return fx_io.restore_flow_system_from_dataset(ds)

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
        try:
            ds = fx_io.load_dataset_from_netcdf(path)
            flow_system = cls.from_dataset(ds)
        except Exception as e:
            raise OSError(f'Failed to load FlowSystem from NetCDF file {path}: {e}') from e
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

        # Convert to new parameter names and reduce constant dimensions
        flow_system_data = convert_old_dataset(flow_system_data)

        # Reconstruct FlowSystem
        flow_system = cls.from_dataset(flow_system_data)
        flow_system.name = path.stem.replace('--flow_system', '')

        # Set previous_flow_rate=0 for flows of components with status_parameters
        # In v4 API, previous_flow_rate=None defaulted to previous_status=0 (off)
        # Now previous_flow_rate=None means relaxed (no constraint at t=0)
        for comp in flow_system.components.values():
            if getattr(comp, 'status_parameters', None) is not None:
                for flow in comp.flows.values():
                    if flow.previous_flow_rate is None:
                        flow.previous_flow_rate = 0

        return flow_system

    def copy(self) -> FlowSystem:
        """Create a copy of the FlowSystem without optimization state.

        Creates a new FlowSystem with copies of all elements, but without:
        - The solution dataset
        - The optimization model
        - Element variable/constraint names

        This is useful for creating variations of a FlowSystem for different
        optimization scenarios without affecting the original.

        Returns:
            A new FlowSystem instance that can be modified and optimized independently.

        Examples:
            >>> original = FlowSystem(timesteps)
            >>> original.add(boiler, bus)
            >>> original.optimize(solver)  # Original now has solution
            >>>
            >>> # Create a copy to try different parameters
            >>> variant = original.copy()  # No solution, can be modified
            >>> variant.add(new_component)
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

        reference_structure, extracted_arrays = self._create_reference_structure()

        if stats:
            reference_structure = replace_references_with_stats(reference_structure, extracted_arrays)

        if clean:
            return fx_io.remove_none_and_empty(reference_structure)
        return reference_structure

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

        try:
            data = self.get_structure(clean=True, stats=True)
            fx_io.save_json(data, path)
        except Exception as e:
            raise OSError(f'Failed to save FlowSystem to JSON file {path}: {e}') from e

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

        # Create penalty effect if needed (must happen before validation)
        self._prepare_effects()

        # Propagate status parameters from components to flows
        self._propagate_all_status_parameters()

        # Validate cross-element references after transformation
        self._validate_system_integrity()

        # Unified validation AFTER transformation (config + DataArray checks)
        self._run_validation()

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
        # Use id for consistent keying with IdList
        elements_without_colors = [comp.id for comp in self.components.values() if comp.color is None]

        if not elements_without_colors:
            return

        # Generate colors from the default colorscale
        colorscale = CONFIG.Plotting.default_qualitative_colorscale
        color_mapping = process_colors(colorscale, elements_without_colors)

        # Assign colors to elements
        for element_id, color in color_mapping.items():
            self.components[element_id].color = color
            logger.debug(f"Auto-assigned color '{color}' to component '{element_id}'")

    def add(self, *elements: Element) -> None:
        """Add elements (Converters, Ports, Storages, Buses, Effects, ...) to the FlowSystem.

        Args:
            *elements: Element instances to add (Converter, Port, Storage, Bus, Effect, ...).

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
        if self.status > FlowSystemStatus.INITIALIZED:
            self._invalidate_model()

        for new_element in list(elements):
            # Validate element type first
            if not isinstance(new_element, (Converter, Port, Storage, Transmission, Component, Effect, Bus)):
                raise TypeError(
                    f'Tried to add incompatible object to FlowSystem: {type(new_element)=}: {new_element=} '
                )

            # Common validations for all element types (before any state changes)
            self._check_if_element_already_assigned(new_element)
            self._check_if_element_is_unique(new_element)

            # Dispatch to type-specific handlers
            if isinstance(new_element, (Converter, Port, Storage, Transmission, Component)):
                self._add_components(new_element)
            elif isinstance(new_element, Effect):
                self._add_effects(new_element)
            elif isinstance(new_element, Bus):
                self._add_buses(new_element)

            # Log registration
            element_type = type(new_element).__name__
            logger.info(f'Registered new {element_type}: {new_element.id}')

    def add_elements(self, *elements: Element) -> None:
        """Deprecated. Use :meth:`add` instead."""
        warnings.warn(
            'add_elements() is deprecated. Use add() instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        self.add(*elements)

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
            fs.add(bus)

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
        if self.status > FlowSystemStatus.INITIALIZED:
            self._invalidate_model()

        for carrier in list(carriers):
            if not isinstance(carrier, Carrier):
                raise TypeError(f'Expected Carrier object, got {type(carrier)}')
            self._carriers.add(carrier)
            logger.debug(f'Adding carrier {carrier} to FlowSystem')

    def get_carrier(self, element_id: str) -> Carrier | None:
        """Get the carrier for a bus or flow.

        Args:
            element_id: Bus id (e.g., 'Fernwärme') or flow id (e.g., 'Boiler(Q_th)').

        Returns:
            Carrier or None if not found.

        Note:
            To access a carrier directly by name, use ``flow_system.carriers['electricity']``.

        Raises:
            RuntimeError: If FlowSystem is not connected_and_transformed.
        """
        self._require_status(FlowSystemStatus.CONNECTED, 'get carrier')

        # Try as bus id
        bus = self.buses.get(element_id)
        if bus and bus.carrier:
            return self._carriers.get(bus.carrier.lower())

        # Try as flow id
        flow = self.flows.get(element_id)
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
        """Cached mapping of flow ids to carrier names.

        Returns:
            Dict mapping flow id to carrier name (lowercase).
            Flows without a carrier are not included.

        Raises:
            RuntimeError: If FlowSystem is not connected_and_transformed.
        """
        self._require_status(FlowSystemStatus.CONNECTED, 'access flow_carriers')

        if self._flow_carriers is None:
            self._flow_carriers = {}
            for flow_id, flow in self.flows.items():
                bus = self.buses.get(flow.bus)
                if bus and bus.carrier:
                    self._flow_carriers[flow_id] = bus.carrier.lower()

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
        self._require_status(FlowSystemStatus.CONNECTED, 'create model')
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
        self.model.build_model()
        return self

    def solve(self, solver: _Solver, log_fn: pathlib.Path | str | None = None, progress: bool = True) -> FlowSystem:
        """
        Solve the optimization model and populate the solution.

        This method solves the previously built model using the specified solver.
        After solving, `self.solution` will contain the optimization results,
        and each element's `.solution` property will provide access to its
        specific variables.

        Args:
            solver: The solver to use (e.g., HighsSolver, GurobiSolver).
            log_fn: Path to write the solver log file. If *None* and
                ``capture_solver_log`` is enabled, a temporary file is used
                (deleted after streaming). If a path is provided, the solver
                log is persisted there regardless of capture settings.
            progress: Whether to show a tqdm progress bar during solving.

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
        self._require_status(FlowSystemStatus.MODEL_BUILT, 'solve')

        log_path = pathlib.Path(log_fn) if log_fn is not None else None
        if CONFIG.Solving.capture_solver_log:
            with fx_io.stream_solver_log(log_fn=log_path) as captured_path:
                self.model.solve(
                    log_fn=captured_path,
                    solver_name=solver.name,
                    progress=progress,
                    **solver.options,
                )
        else:
            self.model.solve(
                **({'log_fn': log_path} if log_path is not None else {}),
                solver_name=solver.name,
                progress=progress,
                **solver.options,
            )

        if self.model.termination_condition in ('infeasible', 'infeasible_or_unbounded'):
            self._log_infeasibilities()
            raise RuntimeError(f'Model was infeasible. Status: {self.model.status}. Check your constraints and bounds.')

        # Store solution on FlowSystem for direct Element access
        self.solution = self.model.solution

        logger.info(f'Optimization solved successfully. Objective: {self.model.objective.value:.4f}')

        return self

    def _log_infeasibilities(self) -> None:
        """Log infeasibility details if configured and model supports it."""
        if not CONFIG.Solving.compute_infeasibilities:
            return

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            self.model.print_infeasibilities()

        infeasibilities = f.getvalue()
        logger.error('Successfully extracted infeasibilities: \n%s', infeasibilities)

    @property
    def solution(self) -> xr.Dataset | LegacySolutionWrapper | None:
        """
        Access the optimization solution as an xarray Dataset.

        The solution is indexed by ``timesteps_extra`` (the original timesteps plus
        one additional timestep at the end). Variables that do not have data for the
        extra timestep (most variables except storage charge states) will contain
        NaN values at the final timestep.

        When ``CONFIG.Legacy.solution_access`` is True, returns a wrapper that
        supports legacy access patterns like ``solution['effect_name']``.

        Returns:
            xr.Dataset: The solution dataset with all optimization variable results,
                or None if the model hasn't been solved yet.

        Example:
            >>> flow_system.optimize(solver)
            >>> flow_system.solution.isel(time=slice(None, -1))  # Exclude trailing NaN (and final charge states)
        """
        if self._solution is None:
            return None
        if CONFIG.Legacy.solution_access:
            return LegacySolutionWrapper(self._solution)
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

        This is equivalent to ``status >= FlowSystemStatus.SOLVED``.
        """
        return self.status >= FlowSystemStatus.SOLVED

    @property
    def status(self) -> FlowSystemStatus:
        """Current lifecycle status of this FlowSystem.

        The status progresses through these stages:

        - INITIALIZED: FlowSystem created, elements can be added
        - CONNECTED: Network connected, data transformed to xarray
        - MODEL_CREATED: linopy Model instantiated
        - MODEL_BUILT: Variables and constraints populated
        - SOLVED: Optimization complete, solution exists

        Use this to check what operations are available or what has been done.

        Examples:
            >>> fs = FlowSystem(timesteps)
            >>> fs.status
            <FlowSystemStatus.INITIALIZED: 1>
            >>> fs.add(bus, component)
            >>> fs.connect_and_transform()
            >>> fs.status
            <FlowSystemStatus.CONNECTED: 2>
            >>> fs.optimize(solver)
            >>> fs.status
            <FlowSystemStatus.SOLVED: 5>
        """
        return get_status(self)

    def _require_status(self, minimum: FlowSystemStatus, action: str) -> None:
        """Raise if FlowSystem is not in the required status.

        Args:
            minimum: The minimum required status.
            action: Description of the action being attempted (for error message).

        Raises:
            RuntimeError: If current status is below the minimum required.
        """
        current = self.status
        if current < minimum:
            raise RuntimeError(
                f'Cannot {action}: FlowSystem is in status {current.name}, but requires at least {minimum.name}.'
            )

    def _invalidate_to(self, target: FlowSystemStatus) -> None:
        """Invalidate FlowSystem down to target status.

        This clears all data/caches associated with statuses above the target.
        If the FlowSystem is already at or below the target status, this is a no-op.

        Args:
            target: The target status to invalidate down to.

        See Also:
            :meth:`invalidate`: Public method for manual invalidation.
            :meth:`reset`: Clears solution and invalidates (for locked FlowSystems).
        """
        invalidate_to_status(self, target)

    def _invalidate_model(self) -> None:
        """Invalidate the model when structure changes.

        This clears the model, resets the ``connected_and_transformed`` flag,
        clears all element variable/constraint names, and invalidates the
        topology accessor cache.

        Called internally by :meth:`add_elements`, :meth:`add_carriers`,
        :meth:`reset`, and :meth:`invalidate`.

        See Also:
            :meth:`invalidate`: Public method for manual invalidation.
            :meth:`reset`: Clears solution and invalidates (for locked FlowSystems).
        """
        self._invalidate_to(FlowSystemStatus.INITIALIZED)

    def reset(self) -> FlowSystem:
        """Clear optimization state to allow modifications.

        This method unlocks the FlowSystem by clearing:
        - The solution dataset
        - The optimization model
        - All element variable/constraint names
        - The connected_and_transformed flag

        After calling reset(), the FlowSystem can be modified again
        (e.g., adding elements or carriers).

        Returns:
            Self, for method chaining.

        Examples:
            >>> flow_system.optimize(solver)  # FlowSystem is now locked
            >>> flow_system.add(new_bus)  # Raises RuntimeError
            >>> flow_system.reset()  # Unlock the FlowSystem
            >>> flow_system.add(new_bus)  # Now works
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

            Access solution data:

            >>> flow_system.optimize(solver)
            >>> print(flow_system.solution['flow|rate'])

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
    def stats(self) -> StatisticsAccessor:
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
            >>> flow_system.stats.plot.balance('ElectricityBus')
            >>> flow_system.stats.plot.heatmap('Boiler|on')
            >>> ds = flow_system.stats.flow_rates  # Get data for analysis
        """
        if self._statistics is None:
            self._statistics = StatisticsAccessor(self)
        return self._statistics

    @property
    def statistics(self) -> StatisticsAccessor:
        """Deprecated: Use :attr:`stats` instead."""
        warnings.warn(
            "The 'statistics' accessor is deprecated. Use 'stats' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.stats

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

    @property
    def batched(self) -> BatchedAccessor:
        """
        Access batched data containers for element properties.

        This property returns a BatchedAccessor that provides indexed/batched
        access to element properties as xarray DataArrays with element dimensions.

        Returns:
            A cached BatchedAccessor instance.

        Examples:
            Access flow categorizations:

            >>> flow_system.batched.flows.with_status  # List of flow IDs with status
            >>> flow_system.batched.flows.with_investment  # List of flow IDs with investment

            Access batched parameters:

            >>> flow_system.batched.flows.relative_minimum  # DataArray with flow dimension
            >>> flow_system.batched.flows.effective_size_upper  # DataArray with flow dimension

            Access individual flows:

            >>> flow = flow_system.batched.flows['Boiler(gas_in)']
        """
        if self._batched is None:
            self._batched = BatchedAccessor(self)
        return self._batched

    @property
    def clustering(self) -> Clustering | None:
        """Clustering metadata for this FlowSystem.

        This property is populated by `transform.cluster()` or when loading
        a clustered FlowSystem from file. It contains information about the
        original timesteps, cluster assignments, and aggregation metrics.

        Setting this property resets the batched accessor cache to ensure
        storage categorization (basic vs intercluster) is correctly computed
        based on the new clustering state.

        Returns:
            Clustering object if this is a clustered FlowSystem, None otherwise.
        """
        return self._clustering

    @clustering.setter
    def clustering(self, value: Clustering | None) -> None:
        """Set clustering and reset batched accessor cache."""
        self._clustering = value
        # Reset batched accessor so storage categorization is recomputed
        # with the new clustering state (basic vs intercluster storages)
        if self._batched is not None:
            self._batched._reset()

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
        checks if element or id of element already exists in list

        Args:
            element: new element to check
        """
        # check if id is already used:
        if element.id in self:
            raise ValueError(f'ID of Element {element.id} already used in another element!')

    def _check_if_element_already_assigned(self, element: Element) -> None:
        """
        Check if element already belongs to another FlowSystem.

        Args:
            element: Element to check

        Raises:
            ValueError: If element is already assigned to a different FlowSystem
        """
        if id(element) in self._registered_elements:
            return  # Already registered to this FlowSystem
        # Check if any other FlowSystem has claimed this element — not possible to detect
        # with id()-based tracking alone, but duplicates are caught by _check_if_element_is_unique

    def _propagate_all_status_parameters(self) -> None:
        """Propagate status parameters from components to their flows.

        Components with status_parameters or prevent_simultaneous_flows require
        certain flows to have StatusParameters. Transmissions with absolute_losses
        additionally need status variables on input flows.
        """
        for component in self.components.values():
            component._propagate_status_parameters()

    def _prepare_effects(self) -> None:
        """Create the penalty effect if needed.

        Validation is done after transformation via _run_validation().
        """
        if self.effects._penalty_effect is None:
            penalty = self.effects._create_penalty_effect()
            self._registered_elements.add(id(penalty))

    def _run_validation(self) -> None:
        """Run all validation through batched *Data classes.

        Each *Data.validate() method handles both:
        - Config validation (simple checks)
        - DataArray validation (post-transformation checks)

        Called during connect_and_transform(). The cached *Data instances are
        reused during model building.
        """
        batched = self.batched
        # Validate buses first - catches "Bus with no flows" before FlowsData fails on empty arrays
        batched.buses.validate()
        batched.effects.validate()
        batched.flows.validate()
        batched.storages.validate()
        batched.intercluster_storages.validate()
        batched.converters.validate()
        batched.transmissions.validate()
        batched.components.validate()

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
                    f'Flow "{flow.id}" references bus "{flow.bus}" which does not exist in FlowSystem. '
                    f'Available buses: {available_buses}. '
                    f'Did you forget to add the bus using flow_system.add(Bus("{flow.bus}"))?'
                )

    def _add_effects(self, *args: Effect) -> None:
        for effect in args:
            self._registered_elements.add(id(effect))
        self.effects.add_effects(*args)

    def _add_components(self, *components) -> None:
        for new_component in list(components):
            self._registered_elements.add(id(new_component))
            for flow in new_component.flows.values():
                self._registered_elements.add(id(flow))
            # Dispatch to the right container
            if isinstance(new_component, Converter):
                self.converters.add(new_component)
            elif isinstance(new_component, Port):
                self.ports.add(new_component)
            elif isinstance(new_component, Storage):
                self.storages.add(new_component)
            elif isinstance(new_component, Transmission):
                self.transmissions.add(new_component)
            else:
                # Legacy Component subclass (Source, Sink, SourceAndSink) → ports
                self.ports.add(new_component)
        # Invalidate caches once after all additions
        if components:
            self._flows_cache = None
            self._components_cache = None

    def _add_buses(self, *buses: Bus):
        for new_bus in list(buses):
            self._registered_elements.add(id(new_bus))
            self.buses.add(new_bus)  # Add to existing buses
        # Invalidate cache once after all additions
        if buses:
            self._flows_cache = None

    def _connect_network(self):
        """Connect flows to their buses. Flow ownership is already set in each component's __init__."""
        for flow in self.flows.values():
            bus = self.buses.get(flow.bus)
            if bus is None:
                raise KeyError(
                    f'Bus {flow.bus} not found in the FlowSystem, but used by "{flow.id}". Please add it first.'
                )
            if flow.is_input_in_component and flow.id not in bus.outputs:
                bus.outputs.add(flow)
            elif not flow.is_input_in_component and flow.id not in bus.inputs:
                bus.inputs.add(flow)

        logger.debug(
            f'Connected {len(self.buses)} Buses and {len(self.components)} Components via {len(self.flows)} Flows.'
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

    def _get_container_groups(self) -> dict[str, IdList]:
        """Return ordered container groups for CompositeContainerMixin."""
        groups: dict[str, IdList] = {}
        if self.converters:
            groups['Converters'] = self.converters
        if self.ports:
            groups['Ports'] = self.ports
        if self.storages:
            groups['Storages'] = self.storages
        if self.transmissions:
            groups['Transmissions'] = self.transmissions
        groups['Buses'] = self.buses
        groups['Effects'] = self.effects
        groups['Flows'] = self.flows
        return groups

    @property
    def components(self) -> IdList:
        """All component-like elements as a combined IdList (backward compat).

        Prefer accessing specific containers directly:
        ``self.converters``, ``self.ports``, ``self.storages``, ``self.transmissions``.
        """
        if self._components_cache is None:
            all_comps = (
                list(self.converters.values())
                + list(self.ports.values())
                + list(self.storages.values())
                + list(self.transmissions.values())
            )
            all_comps.sort(key=lambda c: c.id.lower())
            self._components_cache = element_id_list(all_comps, display_name='components', truncate_repr=10)
        return self._components_cache

    @property
    def flows(self) -> IdList[Flow]:
        if self._flows_cache is None:
            flows = []
            for container in (self.converters, self.ports, self.storages, self.transmissions):
                for c in container.values():
                    flows.extend(c.flows.values())
            # Deduplicate by id and sort for reproducibility
            flows = sorted({id(f): f for f in flows}.values(), key=lambda f: f.id.lower())
            self._flows_cache = element_id_list(flows, display_name='flows', truncate_repr=10)
        return self._flows_cache

    # --- Forwarding properties for model coordinate state ---

    @property
    def timesteps(self):
        return self.model_coords.timesteps

    @timesteps.setter
    def timesteps(self, value):
        self.model_coords.timesteps = value

    @property
    def timesteps_extra(self):
        return self.model_coords.timesteps_extra

    @timesteps_extra.setter
    def timesteps_extra(self, value):
        self.model_coords.timesteps_extra = value

    @property
    def hours_of_last_timestep(self):
        return self.model_coords.hours_of_last_timestep

    @hours_of_last_timestep.setter
    def hours_of_last_timestep(self, value):
        self.model_coords.hours_of_last_timestep = value

    @property
    def hours_of_previous_timesteps(self):
        return self.model_coords.hours_of_previous_timesteps

    @hours_of_previous_timesteps.setter
    def hours_of_previous_timesteps(self, value):
        self.model_coords.hours_of_previous_timesteps = value

    @property
    def timestep_duration(self):
        return self.model_coords.timestep_duration

    @timestep_duration.setter
    def timestep_duration(self, value):
        self.model_coords.timestep_duration = value

    @property
    def periods(self):
        return self.model_coords.periods

    @periods.setter
    def periods(self, value):
        self.model_coords.periods = value

    @property
    def periods_extra(self):
        return self.model_coords.periods_extra

    @periods_extra.setter
    def periods_extra(self, value):
        self.model_coords.periods_extra = value

    @property
    def weight_of_last_period(self):
        return self.model_coords.weight_of_last_period

    @weight_of_last_period.setter
    def weight_of_last_period(self, value):
        self.model_coords.weight_of_last_period = value

    @property
    def period_weights(self):
        return self.model_coords.period_weights

    @period_weights.setter
    def period_weights(self, value):
        self.model_coords.period_weights = value

    @property
    def scenarios(self):
        return self.model_coords.scenarios

    @scenarios.setter
    def scenarios(self, value):
        self.model_coords.scenarios = value

    @property
    def clusters(self):
        return self.model_coords.clusters

    @clusters.setter
    def clusters(self, value):
        self.model_coords.clusters = value

    @property
    def cluster_weight(self):
        return self.model_coords.cluster_weight

    @cluster_weight.setter
    def cluster_weight(self, value):
        self.model_coords.cluster_weight = value

    @property
    def dims(self) -> list[str]:
        """Active dimension names."""
        return self.model_coords.dims

    @property
    def indexes(self) -> dict[str, pd.Index]:
        """Indexes for active dimensions."""
        return self.model_coords.indexes

    @property
    def temporal_dims(self) -> list[str]:
        """Temporal dimensions for summing over time."""
        return self.model_coords.temporal_dims

    @property
    def temporal_weight(self) -> xr.DataArray:
        """Combined temporal weight (timestep_duration x cluster_weight)."""
        return self.model_coords.temporal_weight

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
        """Weights for each scenario."""
        return self.model_coords.scenario_weights

    @scenario_weights.setter
    def scenario_weights(self, value: Numeric_S | None) -> None:
        """Set scenario weights (always normalized to sum to 1)."""
        self.model_coords.scenario_weights = value

    def _unit_weight(self, dim: str) -> xr.DataArray:
        """Create a unit weight DataArray (all 1.0) for a dimension."""
        return self.model_coords._unit_weight(dim)

    @property
    def weights(self) -> dict[str, xr.DataArray]:
        """Weights for active dimensions (unit weights if not explicitly set)."""
        return self.model_coords.weights

    def sum_temporal(self, data: xr.DataArray) -> xr.DataArray:
        """Sum data over temporal dimensions with full temporal weighting."""
        return self.model_coords.sum_temporal(data)

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
            element_type: Type of elements expected in list (e.g., 'Element.id', 'Flow.id')

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
            value: True (all equalized), False (all vary), or list of component id strings to equalize

        Raises:
            TypeError: If value is not bool or list[str]
            ValueError: If list contains non-string elements
        """
        self._validate_scenario_parameter(value, 'scenario_independent_sizes', 'Element.id')
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
            value: True (all equalized), False (all vary), or list of flow id strings to equalize

        Raises:
            TypeError: If value is not bool or list[str]
            ValueError: If list contains non-string elements
        """
        self._validate_scenario_parameter(value, 'scenario_independent_flow_rates', 'Flow.id')
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
        """Check if the FlowSystem has been connected and transformed.

        This is equivalent to ``status >= FlowSystemStatus.CONNECTED``.
        """
        return self.status >= FlowSystemStatus.CONNECTED
