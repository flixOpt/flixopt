"""
This module contains the core structure of the flixopt framework.
These classes are not directly used by the end user, but are used by other modules.
"""

from __future__ import annotations

import inspect
import json
import logging
import pathlib
import re
import warnings
from dataclasses import dataclass
from difflib import get_close_matches
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
)

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from . import io as fx_io
from .config import DEPRECATION_REMOVAL_VERSION
from .core import FlowSystemDimensions, TimeSeriesData, get_dataarray_stats

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from collections.abc import Collection, ItemsView, Iterator

    from .effects import EffectCollectionModel
    from .flow_system import FlowSystem
    from .types import Effect_TPS, Numeric_TPS, NumericOrBool

logger = logging.getLogger('flixopt')

CLASS_REGISTRY = {}


def register_class_for_io(cls):
    """Register a class for serialization/deserialization."""
    name = cls.__name__
    if name in CLASS_REGISTRY:
        raise ValueError(
            f'Class {name} already registered! Use a different Name for the class! '
            f'This error should only happen in developement'
        )
    CLASS_REGISTRY[name] = cls
    return cls


class SubmodelsMixin:
    """Mixin that provides submodel functionality for both FlowSystemModel and Submodel."""

    submodels: Submodels

    @property
    def all_submodels(self) -> list[Submodel]:
        """Get all submodels including nested ones recursively."""
        direct_submodels = list(self.submodels.values())

        # Recursively collect nested sub-models
        nested_submodels = []
        for submodel in direct_submodels:
            nested_submodels.extend(submodel.all_submodels)

        return direct_submodels + nested_submodels

    def add_submodels(self, submodel: Submodel, short_name: str = None) -> Submodel:
        """Register a sub-model with the model"""
        if short_name is None:
            short_name = submodel.__class__.__name__
        if short_name in self.submodels:
            raise ValueError(f'Short name "{short_name}" already assigned to model')
        self.submodels.add(submodel, name=short_name)

        return submodel


class FlowSystemModel(linopy.Model, SubmodelsMixin):
    """
    The FlowSystemModel is the linopy Model that is used to create the mathematical model of the flow_system.
    It is used to create and store the variables and constraints for the flow_system.

    Args:
        flow_system: The flow_system that is used to create the model.
    """

    def __init__(self, flow_system: FlowSystem):
        super().__init__(force_dim_names=True)
        self.flow_system = flow_system
        self.effects: EffectCollectionModel | None = None
        self.submodels: Submodels = Submodels({})

    def do_modeling(self):
        # Create all element models
        self.effects = self.flow_system.effects.create_model(self)
        for component in self.flow_system.components.values():
            component.create_model(self)
        for bus in self.flow_system.buses.values():
            bus.create_model(self)

        # Add scenario equality constraints after all elements are modeled
        self._add_scenario_equality_constraints()

        # Populate _variable_names and _constraint_names on each Element
        self._populate_element_variable_names()

    def _populate_element_variable_names(self):
        """Populate _variable_names and _constraint_names on each Element from its submodel."""
        for element in self.flow_system.values():
            if element.submodel is not None:
                element._variable_names = list(element.submodel.variables)
                element._constraint_names = list(element.submodel.constraints)

    def _add_scenario_equality_for_parameter_type(
        self,
        parameter_type: Literal['flow_rate', 'size'],
        config: bool | list[str],
    ):
        """Add scenario equality constraints for a specific parameter type.

        Args:
            parameter_type: The type of parameter ('flow_rate' or 'size')
            config: Configuration value (True = equalize all, False = equalize none, list = equalize these)
        """
        if config is False:
            return  # All vary per scenario, no constraints needed

        suffix = f'|{parameter_type}'
        if config is True:
            # All should be scenario-independent
            vars_to_constrain = [var for var in self.variables if var.endswith(suffix)]
        else:
            # Only those in the list should be scenario-independent
            all_vars = [var for var in self.variables if var.endswith(suffix)]
            to_equalize = {f'{element}{suffix}' for element in config}
            vars_to_constrain = [var for var in all_vars if var in to_equalize]

        # Validate that all specified variables exist
        missing_vars = [v for v in vars_to_constrain if v not in self.variables]
        if missing_vars:
            param_name = 'scenario_independent_sizes' if parameter_type == 'size' else 'scenario_independent_flow_rates'
            raise ValueError(f'{param_name} contains invalid labels: {missing_vars}')

        logger.debug(f'Adding scenario equality constraints for {len(vars_to_constrain)} {parameter_type} variables')
        for var in vars_to_constrain:
            self.add_constraints(
                self.variables[var].isel(scenario=0) == self.variables[var].isel(scenario=slice(1, None)),
                name=f'{var}|scenario_independent',
            )

    def _add_scenario_equality_constraints(self):
        """Add equality constraints to equalize variables across scenarios based on FlowSystem configuration."""
        # Only proceed if we have scenarios
        if self.flow_system.scenarios is None or len(self.flow_system.scenarios) <= 1:
            return

        self._add_scenario_equality_for_parameter_type('flow_rate', self.flow_system.scenario_independent_flow_rates)
        self._add_scenario_equality_for_parameter_type('size', self.flow_system.scenario_independent_sizes)

    @property
    def solution(self):
        """Build solution dataset, reindexing to timesteps_extra for consistency."""
        # Suppress the linopy warning about coordinate mismatch.
        # This warning is expected when storage charge_state has one more timestep than other variables.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                category=UserWarning,
                message='Coordinates across variables not equal',
            )
            solution = super().solution
        solution['objective'] = self.objective.value
        # Store attrs as JSON strings for netCDF compatibility
        solution.attrs = {
            'Components': json.dumps(
                {
                    comp.label_full: comp.submodel.results_structure()
                    for comp in sorted(
                        self.flow_system.components.values(), key=lambda component: component.label_full.upper()
                    )
                }
            ),
            'Buses': json.dumps(
                {
                    bus.label_full: bus.submodel.results_structure()
                    for bus in sorted(self.flow_system.buses.values(), key=lambda bus: bus.label_full.upper())
                }
            ),
            'Effects': json.dumps(
                {
                    effect.label_full: effect.submodel.results_structure()
                    for effect in sorted(
                        self.flow_system.effects.values(), key=lambda effect: effect.label_full.upper()
                    )
                }
            ),
            'Flows': json.dumps(
                {
                    flow.label_full: flow.submodel.results_structure()
                    for flow in sorted(self.flow_system.flows.values(), key=lambda flow: flow.label_full.upper())
                }
            ),
        }
        # Ensure solution is always indexed by timesteps_extra for consistency.
        # Variables without extra timestep data will have NaN at the final timestep.
        if 'time' in solution.coords:
            if not solution.indexes['time'].equals(self.flow_system.timesteps_extra):
                solution = solution.reindex(time=self.flow_system.timesteps_extra)
        return solution

    @property
    def timestep_duration(self) -> xr.DataArray:
        """Duration of each timestep in hours."""
        return self.flow_system.timestep_duration

    @property
    def hours_of_previous_timesteps(self):
        return self.flow_system.hours_of_previous_timesteps

    @property
    def dims(self) -> list[str]:
        """Active dimension names."""
        return self.flow_system.dims

    @property
    def indexes(self) -> dict[str, pd.Index]:
        """Indexes for active dimensions."""
        return self.flow_system.indexes

    @property
    def weights(self) -> dict[str, xr.DataArray]:
        """Weights for active dimensions (unit weights if not set).

        Scenario weights are always normalized (handled by FlowSystem).
        """
        return self.flow_system.weights

    @property
    def temporal_dims(self) -> list[str]:
        """Temporal dimensions for summing over time.

        Returns ['time', 'cluster'] for clustered systems, ['time'] otherwise.
        """
        return self.flow_system.temporal_dims

    @property
    def temporal_weight(self) -> xr.DataArray:
        """Combined temporal weight (timestep_duration × cluster_weight)."""
        return self.flow_system.temporal_weight

    def sum_temporal(self, data: xr.DataArray) -> xr.DataArray:
        """Sum data over temporal dimensions with full temporal weighting.

        Example:
            >>> total_energy = model.sum_temporal(flow_rate)
        """
        return self.flow_system.sum_temporal(data)

    @property
    def scenario_weights(self) -> xr.DataArray:
        """Scenario weights of model.

        Returns:
            - Scalar 1 if no scenarios defined
            - Unit weights (all 1.0) if scenarios exist but no explicit weights set
            - Normalized explicit weights if set via FlowSystem.scenario_weights
        """
        if self.flow_system.scenarios is None:
            return xr.DataArray(1)

        if self.flow_system.scenario_weights is None:
            return self.flow_system._unit_weight('scenario')

        return self.flow_system.scenario_weights

    @property
    def objective_weights(self) -> xr.DataArray:
        """
        Objective weights of model (period_weights × scenario_weights).
        """
        period_weights = self.flow_system.effects.objective_effect.submodel.period_weights
        scenario_weights = self.scenario_weights

        return period_weights * scenario_weights

    def get_coords(
        self,
        dims: Collection[str] | None = None,
        extra_timestep: bool = False,
    ) -> xr.Coordinates | None:
        """
        Returns the coordinates of the model

        Args:
            dims: The dimensions to include in the coordinates. If None, includes all dimensions
            extra_timestep: If True, uses extra timesteps instead of regular timesteps.
                For clustered FlowSystems, extends time by 1 (for charge_state boundaries).

        Returns:
            The coordinates of the model, or None if no coordinates are available

        Raises:
            ValueError: If extra_timestep=True but 'time' is not in dims
        """
        if extra_timestep and dims is not None and 'time' not in dims:
            raise ValueError('extra_timestep=True requires "time" to be included in dims')

        if dims is None:
            coords = dict(self.flow_system.indexes)
        else:
            # In clustered systems, 'time' is always paired with 'cluster'
            # So when 'time' is requested, also include 'cluster' if available
            effective_dims = set(dims)
            if 'time' in dims and 'cluster' in self.flow_system.indexes:
                effective_dims.add('cluster')
            coords = {k: v for k, v in self.flow_system.indexes.items() if k in effective_dims}

        if extra_timestep and coords:
            coords['time'] = self.flow_system.timesteps_extra

        return xr.Coordinates(coords) if coords else None

    def __repr__(self) -> str:
        """
        Return a string representation of the FlowSystemModel, borrowed from linopy.Model.
        """
        # Extract content from existing representations
        sections = {
            f'Variables: [{len(self.variables)}]': self.variables.__repr__().split('\n', 2)[2],
            f'Constraints: [{len(self.constraints)}]': self.constraints.__repr__().split('\n', 2)[2],
            f'Submodels: [{len(self.submodels)}]': self.submodels.__repr__().split('\n', 2)[2],
            'Status': self.status,
        }

        # Format sections with headers and underlines
        formatted_sections = fx_io.format_sections_with_headers(sections)

        title = f'FlowSystemModel ({self.type})'
        all_sections = '\n'.join(formatted_sections)

        return f'{title}\n{"=" * len(title)}\n\n{all_sections}'


class Interface:
    """
    Base class for all Elements and Models in flixopt that provides serialization capabilities.

    This class enables automatic serialization/deserialization of objects containing xarray DataArrays
    and nested Interface objects to/from xarray Datasets and NetCDF files. It uses introspection
    of constructor parameters to automatically handle most serialization scenarios.

    Key Features:
        - Automatic extraction and restoration of xarray DataArrays
        - Support for nested Interface objects
        - NetCDF and JSON export/import
        - Recursive handling of complex nested structures

    Subclasses must implement:
        transform_data(): Transform data to match FlowSystem dimensions
    """

    # Class-level defaults for attributes set by link_to_flow_system()
    # These provide type hints and default values without requiring __init__ in subclasses
    _flow_system: FlowSystem | None = None
    _prefix: str = ''

    def transform_data(self) -> None:
        """Transform the data of the interface to match the FlowSystem's dimensions.

        Uses `self._prefix` (set during `link_to_flow_system()`) to name transformed data.

        Raises:
            NotImplementedError: Must be implemented by subclasses

        Note:
            The FlowSystem reference is available via self._flow_system (for Interface objects)
            or self.flow_system property (for Element objects). Elements must be registered
            to a FlowSystem before calling this method.
        """
        raise NotImplementedError('Every Interface subclass needs a transform_data() method')

    @property
    def prefix(self) -> str:
        """The prefix used for naming transformed data (e.g., 'Boiler(Q_th)|status_parameters')."""
        return self._prefix

    def _sub_prefix(self, name: str) -> str:
        """Build a prefix for a nested interface by appending name to current prefix."""
        return f'{self._prefix}|{name}' if self._prefix else name

    def link_to_flow_system(self, flow_system: FlowSystem, prefix: str = '') -> None:
        """Link this interface and all nested interfaces to a FlowSystem.

        This method is called automatically during element registration to enable
        elements to access FlowSystem properties without passing the reference
        through every method call. It also sets the prefix used for naming
        transformed data.

        Subclasses with nested Interface objects should override this method
        to propagate the link to their nested interfaces by calling
        `super().link_to_flow_system(flow_system, prefix)` first, then linking
        nested objects with appropriate prefixes.

        Args:
            flow_system: The FlowSystem to link to
            prefix: The prefix for naming transformed data (e.g., 'Boiler(Q_th)')

        Examples:
            Override in a subclass with nested interfaces:

            ```python
            def link_to_flow_system(self, flow_system, prefix: str = '') -> None:
                super().link_to_flow_system(flow_system, prefix)
                if self.nested_interface is not None:
                    self.nested_interface.link_to_flow_system(flow_system, f'{prefix}|nested' if prefix else 'nested')
            ```

            Creating an Interface dynamically during modeling:

            ```python
            # In a Model class
            if flow.status_parameters is None:
                flow.status_parameters = StatusParameters()
                flow.status_parameters.link_to_flow_system(self._model.flow_system, f'{flow.label_full}')
            ```
        """
        self._flow_system = flow_system
        self._prefix = prefix

    @property
    def flow_system(self) -> FlowSystem:
        """Access the FlowSystem this interface is linked to.

        Returns:
            The FlowSystem instance this interface belongs to.

        Raises:
            RuntimeError: If interface has not been linked to a FlowSystem yet.

        Note:
            For Elements, this is set during add_elements().
            For parameter classes, this is set recursively when the parent Element is registered.
        """
        if self._flow_system is None:
            raise RuntimeError(
                f'{self.__class__.__name__} is not linked to a FlowSystem. '
                f'Ensure the parent element is registered via flow_system.add_elements() first.'
            )
        return self._flow_system

    def _fit_coords(
        self, name: str, data: NumericOrBool | None, dims: Collection[FlowSystemDimensions] | None = None
    ) -> xr.DataArray | None:
        """Convenience wrapper for FlowSystem.fit_to_model_coords().

        Args:
            name: The name for the data variable
            data: The data to transform
            dims: Optional dimension names

        Returns:
            Transformed data aligned to FlowSystem coordinates
        """
        return self.flow_system.fit_to_model_coords(name, data, dims=dims)

    def _fit_effect_coords(
        self,
        prefix: str | None,
        effect_values: Effect_TPS | Numeric_TPS | None,
        suffix: str | None = None,
        dims: Collection[FlowSystemDimensions] | None = None,
    ) -> Effect_TPS | None:
        """Convenience wrapper for FlowSystem.fit_effects_to_model_coords().

        Args:
            prefix: Label prefix for effect names
            effect_values: The effect values to transform
            suffix: Optional label suffix
            dims: Optional dimension names

        Returns:
            Transformed effect values aligned to FlowSystem coordinates
        """
        return self.flow_system.fit_effects_to_model_coords(prefix, effect_values, suffix, dims=dims)

    def _create_reference_structure(self) -> tuple[dict, dict[str, xr.DataArray]]:
        """
        Convert all DataArrays to references and extract them.
        This is the core method that both to_dict() and to_dataset() build upon.

        Returns:
            Tuple of (reference_structure, extracted_arrays_dict)

        Raises:
            ValueError: If DataArrays don't have unique names or are duplicated
        """
        # Get constructor parameters using caching for performance
        if not hasattr(self, '_cached_init_params'):
            self._cached_init_params = list(inspect.signature(self.__init__).parameters.keys())

        # Process all constructor parameters
        reference_structure = {'__class__': self.__class__.__name__}
        all_extracted_arrays = {}

        for name in self._cached_init_params:
            if name == 'self':  # Skip self and timesteps. Timesteps are directly stored in Datasets
                continue

            value = getattr(self, name, None)

            if value is None:
                continue
            if isinstance(value, pd.Index):
                logger.debug(f'Skipping {name=} because it is an Index')
                continue

            # Extract arrays and get reference structure
            processed_value, extracted_arrays = self._extract_dataarrays_recursive(value, name)

            # Check for array name conflicts
            conflicts = set(all_extracted_arrays.keys()) & set(extracted_arrays.keys())
            if conflicts:
                raise ValueError(
                    f'DataArray name conflicts detected: {conflicts}. '
                    f'Each DataArray must have a unique name for serialization.'
                )

            # Add extracted arrays to the collection
            all_extracted_arrays.update(extracted_arrays)

            # Only store in structure if it's not None/empty after processing
            if processed_value is not None and not self._is_empty_container(processed_value):
                reference_structure[name] = processed_value

        return reference_structure, all_extracted_arrays

    @staticmethod
    def _is_empty_container(obj) -> bool:
        """Check if object is an empty container (dict, list, tuple, set)."""
        return isinstance(obj, (dict, list, tuple, set)) and len(obj) == 0

    def _extract_dataarrays_recursive(self, obj, context_name: str = '') -> tuple[Any, dict[str, xr.DataArray]]:
        """
        Recursively extract DataArrays from nested structures.

        Args:
            obj: Object to process
            context_name: Name context for better error messages

        Returns:
            Tuple of (processed_object_with_references, extracted_arrays_dict)

        Raises:
            ValueError: If DataArrays don't have unique names
        """
        extracted_arrays = {}

        # Handle DataArrays directly - use their unique name
        if isinstance(obj, xr.DataArray):
            if not obj.name:
                raise ValueError(
                    f'DataArrays must have a unique name for serialization. '
                    f'Unnamed DataArray found in {context_name}. Please set array.name = "unique_name"'
                )

            array_name = str(obj.name)  # Ensure string type
            if array_name in extracted_arrays:
                raise ValueError(
                    f'DataArray name "{array_name}" is duplicated in {context_name}. '
                    f'Each DataArray must have a unique name for serialization.'
                )

            extracted_arrays[array_name] = obj
            return f':::{array_name}', extracted_arrays

        # Handle Interface objects - extract their DataArrays too
        elif isinstance(obj, Interface):
            try:
                interface_structure, interface_arrays = obj._create_reference_structure()
                extracted_arrays.update(interface_arrays)
                return interface_structure, extracted_arrays
            except Exception as e:
                raise ValueError(f'Failed to process nested Interface object in {context_name}: {e}') from e

        # Handle sequences (lists, tuples)
        elif isinstance(obj, (list, tuple)):
            processed_items = []
            for i, item in enumerate(obj):
                item_context = f'{context_name}[{i}]' if context_name else f'item[{i}]'
                processed_item, nested_arrays = self._extract_dataarrays_recursive(item, item_context)
                extracted_arrays.update(nested_arrays)
                processed_items.append(processed_item)
            return processed_items, extracted_arrays

        # Handle dictionaries
        elif isinstance(obj, dict):
            processed_dict = {}
            for key, value in obj.items():
                key_context = f'{context_name}.{key}' if context_name else str(key)
                processed_value, nested_arrays = self._extract_dataarrays_recursive(value, key_context)
                extracted_arrays.update(nested_arrays)
                processed_dict[key] = processed_value
            return processed_dict, extracted_arrays

        # Handle sets (convert to list for JSON compatibility)
        elif isinstance(obj, set):
            processed_items = []
            for i, item in enumerate(obj):
                item_context = f'{context_name}.set_item[{i}]' if context_name else f'set_item[{i}]'
                processed_item, nested_arrays = self._extract_dataarrays_recursive(item, item_context)
                extracted_arrays.update(nested_arrays)
                processed_items.append(processed_item)
            return processed_items, extracted_arrays

        # For all other types, serialize to basic types
        else:
            return self._serialize_to_basic_types(obj), extracted_arrays

    def _handle_deprecated_kwarg(
        self,
        kwargs: dict,
        old_name: str,
        new_name: str,
        current_value: Any = None,
        transform: callable = None,
        check_conflict: bool = True,
        additional_warning_message: str = '',
    ) -> Any:
        """
        Handle a deprecated keyword argument by issuing a warning and returning the appropriate value.

        This centralizes the deprecation pattern used across multiple classes (Source, Sink, InvestParameters, etc.).

        Args:
            kwargs: Dictionary of keyword arguments to check and modify
            old_name: Name of the deprecated parameter
            new_name: Name of the replacement parameter
            current_value: Current value of the new parameter (if already set)
            transform: Optional callable to transform the old value before returning (e.g., lambda x: [x] to wrap in list)
            check_conflict: Whether to check if both old and new parameters are specified (default: True).
                Note: For parameters with non-None default values (e.g., bool parameters with default=False),
                set check_conflict=False since we cannot distinguish between an explicit value and the default.
            additional_warning_message: Add a custom message which gets appended with a line break to the default warning.

        Returns:
            The value to use (either from old parameter or current_value)

        Raises:
            ValueError: If both old and new parameters are specified and check_conflict is True

        Example:
            # For parameters where None is the default (conflict checking works):
            value = self._handle_deprecated_kwarg(kwargs, 'old_param', 'new_param', current_value)

            # For parameters with non-None defaults (disable conflict checking):
            mandatory = self._handle_deprecated_kwarg(
                kwargs, 'optional', 'mandatory', mandatory,
                transform=lambda x: not x,
                check_conflict=False  # Cannot detect if mandatory was explicitly passed
            )
        """
        import warnings

        old_value = kwargs.pop(old_name, None)
        if old_value is not None:
            # Build base warning message
            base_warning = f'The use of the "{old_name}" argument is deprecated. Use the "{new_name}" argument instead. Will be removed in v{DEPRECATION_REMOVAL_VERSION}.'

            # Append additional message on a new line if provided
            if additional_warning_message:
                # Normalize whitespace: strip leading/trailing whitespace
                extra_msg = additional_warning_message.strip()
                if extra_msg:
                    base_warning += '\n' + extra_msg

            warnings.warn(
                base_warning,
                DeprecationWarning,
                stacklevel=3,  # Stack: this method -> __init__ -> caller
            )
            # Check for conflicts: only raise error if both were explicitly provided
            if check_conflict and current_value is not None:
                raise ValueError(f'Either {old_name} or {new_name} can be specified, but not both.')

            # Apply transformation if provided
            if transform is not None:
                return transform(old_value)
            return old_value

        return current_value

    def _validate_kwargs(self, kwargs: dict, class_name: str = None) -> None:
        """
        Validate that no unexpected keyword arguments are present in kwargs.

        This method uses inspect to get the actual function signature and filters out
        any parameters that are not defined in the __init__ method, while also
        handling the special case of 'kwargs' itself which can appear during deserialization.

        Args:
            kwargs: Dictionary of keyword arguments to validate
            class_name: Optional class name for error messages. If None, uses self.__class__.__name__

        Raises:
            TypeError: If unexpected keyword arguments are found
        """
        if not kwargs:
            return

        import inspect

        sig = inspect.signature(self.__init__)
        known_params = set(sig.parameters.keys()) - {'self', 'kwargs'}
        # Also filter out 'kwargs' itself which can appear during deserialization
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in known_params and k != 'kwargs'}

        if extra_kwargs:
            class_name = class_name or self.__class__.__name__
            unexpected_params = ', '.join(f"'{param}'" for param in extra_kwargs.keys())
            raise TypeError(f'{class_name}.__init__() got unexpected keyword argument(s): {unexpected_params}')

    @staticmethod
    def _has_value(param: Any) -> bool:
        """Check if a parameter has a meaningful value.

        Args:
            param: The parameter to check.

        Returns:
            False for:
                - None
                - Empty collections (dict, list, tuple, set, frozenset)

            True for all other values, including:
                - Non-empty collections
                - xarray DataArrays (even if they contain NaN/empty data)
                - Scalar values (0, False, empty strings, etc.)
                - NumPy arrays (even if empty - use .size to check those explicitly)
        """
        if param is None:
            return False

        # Check for empty collections (but not strings, arrays, or DataArrays)
        if isinstance(param, (dict, list, tuple, set, frozenset)) and len(param) == 0:
            return False

        return True

    @classmethod
    def _resolve_dataarray_reference(
        cls, reference: str, arrays_dict: dict[str, xr.DataArray]
    ) -> xr.DataArray | TimeSeriesData:
        """
        Resolve a single DataArray reference (:::name) to actual DataArray or TimeSeriesData.

        Args:
            reference: Reference string starting with ":::"
            arrays_dict: Dictionary of available DataArrays

        Returns:
            Resolved DataArray or TimeSeriesData object

        Raises:
            ValueError: If referenced array is not found
        """
        array_name = reference[3:]  # Remove ":::" prefix
        if array_name not in arrays_dict:
            raise ValueError(f"Referenced DataArray '{array_name}' not found in dataset")

        array = arrays_dict[array_name]

        # Handle null values with warning
        if array.isnull().any():
            logger.error(f"DataArray '{array_name}' contains null values. Dropping all-null along present dims.")
            if 'time' in array.dims:
                array = array.dropna(dim='time', how='all')

        # Check if this should be restored as TimeSeriesData
        if TimeSeriesData.is_timeseries_data(array):
            return TimeSeriesData.from_dataarray(array)

        return array

    @classmethod
    def _resolve_reference_structure(cls, structure, arrays_dict: dict[str, xr.DataArray]):
        """
        Convert reference structure back to actual objects using provided arrays.

        Args:
            structure: Structure containing references (:::name) or special type markers
            arrays_dict: Dictionary of available DataArrays

        Returns:
            Structure with references resolved to actual DataArrays or objects

        Raises:
            ValueError: If referenced arrays are not found or class is not registered
        """
        # Handle DataArray references
        if isinstance(structure, str) and structure.startswith(':::'):
            return cls._resolve_dataarray_reference(structure, arrays_dict)

        elif isinstance(structure, list):
            resolved_list = []
            for item in structure:
                resolved_item = cls._resolve_reference_structure(item, arrays_dict)
                if resolved_item is not None:  # Filter out None values from missing references
                    resolved_list.append(resolved_item)
            return resolved_list

        elif isinstance(structure, dict):
            if structure.get('__class__'):
                class_name = structure['__class__']
                if class_name not in CLASS_REGISTRY:
                    raise ValueError(
                        f"Class '{class_name}' not found in CLASS_REGISTRY. "
                        f'Available classes: {list(CLASS_REGISTRY.keys())}'
                    )

                # This is a nested Interface object - restore it recursively
                nested_class = CLASS_REGISTRY[class_name]
                # Remove the __class__ key and process the rest
                nested_data = {k: v for k, v in structure.items() if k != '__class__'}
                # Resolve references in the nested data
                resolved_nested_data = cls._resolve_reference_structure(nested_data, arrays_dict)

                try:
                    # Get valid constructor parameters for this class
                    init_params = set(inspect.signature(nested_class.__init__).parameters.keys())

                    # Check for deferred init attributes (defined as class attribute on Element subclasses)
                    # These are serialized but set after construction, not passed to child __init__
                    deferred_attr_names = getattr(nested_class, '_deferred_init_attrs', set())
                    deferred_attrs = {k: v for k, v in resolved_nested_data.items() if k in deferred_attr_names}
                    constructor_data = {k: v for k, v in resolved_nested_data.items() if k not in deferred_attr_names}

                    # Check for unknown parameters - these could be typos or renamed params
                    unknown_params = set(constructor_data.keys()) - init_params
                    if unknown_params:
                        raise TypeError(
                            f'{class_name}.__init__() got unexpected keyword arguments: {unknown_params}. '
                            f'This may indicate renamed parameters that need conversion. '
                            f'Valid parameters are: {init_params - {"self"}}'
                        )

                    # Create instance with constructor parameters
                    instance = nested_class(**constructor_data)

                    # Set internal attributes after construction
                    for attr_name, attr_value in deferred_attrs.items():
                        setattr(instance, attr_name, attr_value)

                    return instance
                except TypeError as e:
                    raise ValueError(f'Failed to create instance of {class_name}: {e}') from e
                except Exception as e:
                    raise ValueError(f'Failed to create instance of {class_name}: {e}') from e
            else:
                # Regular dictionary - resolve references in values
                resolved_dict = {}
                for key, value in structure.items():
                    resolved_value = cls._resolve_reference_structure(value, arrays_dict)
                    if resolved_value is not None or value is None:  # Keep None values if they were originally None
                        resolved_dict[key] = resolved_value
                return resolved_dict

        else:
            return structure

    def _serialize_to_basic_types(self, obj):
        """
        Convert object to basic Python types only (no DataArrays, no custom objects).

        Args:
            obj: Object to serialize

        Returns:
            Object converted to basic Python types (str, int, float, bool, list, dict)
        """
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.ndarray, pd.Series, pd.DataFrame)):
            return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize_to_basic_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_to_basic_types(item) for item in obj]
        elif isinstance(obj, set):
            return [self._serialize_to_basic_types(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):  # Custom objects with attributes
            logger.warning(f'Converting custom object {type(obj)} to dict representation: {obj}')
            return {str(k): self._serialize_to_basic_types(v) for k, v in obj.__dict__.items()}
        else:
            # For any other object, try to convert to string as fallback
            logger.error(f'Converting unknown type {type(obj)} to string: {obj}')
            return str(obj)

    def to_dataset(self) -> xr.Dataset:
        """
        Convert the object to an xarray Dataset representation.
        All DataArrays become dataset variables, everything else goes to attrs.

        Its recommended to only call this method on Interfaces with all numeric data stored as xr.DataArrays.
        Interfaces inside a FlowSystem are automatically converted this form after connecting and transforming the FlowSystem.

        Returns:
            xr.Dataset: Dataset containing all DataArrays with basic objects only in attributes

        Raises:
            ValueError: If serialization fails due to naming conflicts or invalid data
        """
        try:
            reference_structure, extracted_arrays = self._create_reference_structure()
            # Create the dataset with extracted arrays as variables and structure as attrs
            return xr.Dataset(extracted_arrays, attrs=reference_structure)
        except Exception as e:
            raise ValueError(
                f'Failed to convert {self.__class__.__name__} to dataset. Its recommended to only call this method on '
                f'a fully connected and transformed FlowSystem, or Interfaces inside such a FlowSystem.'
                f'Original Error: {e}'
            ) from e

    def to_netcdf(self, path: str | pathlib.Path, compression: int = 5, overwrite: bool = False):
        """
        Save the object to a NetCDF file.

        Args:
            path: Path to save the NetCDF file. Parent directories are created if they don't exist.
            compression: Compression level (0-9)
            overwrite: If True, overwrite existing file. If False, raise error if file exists.

        Raises:
            FileExistsError: If overwrite=False and file already exists.
            ValueError: If serialization fails
            IOError: If file cannot be written
        """
        path = pathlib.Path(path)

        # Check if file exists (unless overwrite is True)
        if not overwrite and path.exists():
            raise FileExistsError(f'File already exists: {path}. Use overwrite=True to overwrite existing file.')

        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            ds = self.to_dataset()
            fx_io.save_dataset_to_netcdf(ds, path, compression=compression)
        except Exception as e:
            raise OSError(f'Failed to save {self.__class__.__name__} to NetCDF file {path}: {e}') from e

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Interface:
        """
        Create an instance from an xarray Dataset.

        Args:
            ds: Dataset containing the object data

        Returns:
            Interface instance

        Raises:
            ValueError: If dataset format is invalid or class mismatch
        """
        try:
            # Get class name and verify it matches
            class_name = ds.attrs.get('__class__')
            if class_name and class_name != cls.__name__:
                logger.warning(f"Dataset class '{class_name}' doesn't match target class '{cls.__name__}'")

            # Get the reference structure from attrs
            reference_structure = dict(ds.attrs)

            # Remove the class name since it's not a constructor parameter
            reference_structure.pop('__class__', None)

            # Create arrays dictionary from dataset variables
            arrays_dict = {name: array for name, array in ds.data_vars.items()}

            # Resolve all references using the centralized method
            resolved_params = cls._resolve_reference_structure(reference_structure, arrays_dict)

            return cls(**resolved_params)
        except Exception as e:
            raise ValueError(f'Failed to create {cls.__name__} from dataset: {e}') from e

    @classmethod
    def from_netcdf(cls, path: str | pathlib.Path) -> Interface:
        """
        Load an instance from a NetCDF file.

        Args:
            path: Path to the NetCDF file

        Returns:
            Interface instance

        Raises:
            IOError: If file cannot be read
            ValueError: If file format is invalid
        """
        try:
            ds = fx_io.load_dataset_from_netcdf(path)
            return cls.from_dataset(ds)
        except Exception as e:
            raise OSError(f'Failed to load {cls.__name__} from NetCDF file {path}: {e}') from e

    def get_structure(self, clean: bool = False, stats: bool = False) -> dict:
        """
        Get object structure as a dictionary.

        Args:
            clean: If True, remove None and empty dicts and lists.
            stats: If True, replace DataArray references with statistics

        Returns:
            Dictionary representation of the object structure
        """
        reference_structure, extracted_arrays = self._create_reference_structure()

        if stats:
            # Replace references with statistics
            reference_structure = self._replace_references_with_stats(reference_structure, extracted_arrays)

        if clean:
            return fx_io.remove_none_and_empty(reference_structure)
        return reference_structure

    def _replace_references_with_stats(self, structure, arrays_dict: dict[str, xr.DataArray]):
        """Replace DataArray references with statistical summaries."""
        if isinstance(structure, str) and structure.startswith(':::'):
            array_name = structure[3:]
            if array_name in arrays_dict:
                return get_dataarray_stats(arrays_dict[array_name])
            return structure

        elif isinstance(structure, dict):
            return {k: self._replace_references_with_stats(v, arrays_dict) for k, v in structure.items()}

        elif isinstance(structure, list):
            return [self._replace_references_with_stats(item, arrays_dict) for item in structure]

        return structure

    def to_json(self, path: str | pathlib.Path):
        """
        Save the object to a JSON file.
        This is meant for documentation and comparison, not for reloading.

        Args:
            path: The path to the JSON file.

        Raises:
            IOError: If file cannot be written
        """
        try:
            # Use the stats mode for JSON export (cleaner output)
            data = self.get_structure(clean=True, stats=True)
            fx_io.save_json(data, path)
        except Exception as e:
            raise OSError(f'Failed to save {self.__class__.__name__} to JSON file {path}: {e}') from e

    def __repr__(self):
        """Return a detailed string representation for debugging."""
        return fx_io.build_repr_from_init(self, excluded_params={'self', 'label', 'kwargs'})

    def copy(self) -> Interface:
        """
        Create a copy of the Interface object.

        Uses the existing serialization infrastructure to ensure proper copying
        of all DataArrays and nested objects.

        Returns:
            A new instance of the same class with copied data.
        """
        # Convert to dataset, copy it, and convert back
        dataset = self.to_dataset().copy(deep=True)
        return self.__class__.from_dataset(dataset)

    def __copy__(self):
        """Support for copy.copy()."""
        return self.copy()

    def __deepcopy__(self, memo):
        """Support for copy.deepcopy()."""
        return self.copy()


class Element(Interface):
    """This class is the basic Element of flixopt. Every Element has a label"""

    submodel: ElementModel | None

    # Attributes that are serialized but set after construction (not passed to child __init__)
    # These are internal state populated during modeling, not user-facing parameters
    _deferred_init_attrs: ClassVar[set[str]] = {'_variable_names', '_constraint_names'}

    def __init__(
        self,
        label: str,
        meta_data: dict | None = None,
        color: str | None = None,
        _variable_names: list[str] | None = None,
        _constraint_names: list[str] | None = None,
    ):
        """
        Args:
            label: The label of the element
            meta_data: used to store more information about the Element. Is not used internally, but saved in the results. Only use python native types.
            color: Optional color for visualizations (e.g., '#FF6B6B'). If not provided, a color will be automatically assigned during FlowSystem.connect_and_transform().
            _variable_names: Internal. Variable names for this element (populated after modeling).
            _constraint_names: Internal. Constraint names for this element (populated after modeling).
        """
        self.label = Element._valid_label(label)
        self.meta_data = meta_data if meta_data is not None else {}
        self.color = color
        self.submodel = None
        self._flow_system: FlowSystem | None = None
        # Variable/constraint names - populated after modeling, serialized for results
        self._variable_names: list[str] = _variable_names if _variable_names is not None else []
        self._constraint_names: list[str] = _constraint_names if _constraint_names is not None else []

    def _plausibility_checks(self) -> None:
        """This function is used to do some basic plausibility checks for each Element during initialization.
        This is run after all data is transformed to the correct format/type"""
        raise NotImplementedError('Every Element needs a _plausibility_checks() method')

    def create_model(self, model: FlowSystemModel) -> ElementModel:
        raise NotImplementedError('Every Element needs a create_model() method')

    @property
    def label_full(self) -> str:
        return self.label

    @property
    def solution(self) -> xr.Dataset:
        """Solution data for this element's variables.

        Returns a view into FlowSystem.solution containing only this element's variables.

        Raises:
            ValueError: If no solution is available (optimization not run or not solved).
        """
        if self._flow_system is None:
            raise ValueError(f'Element "{self.label}" is not linked to a FlowSystem.')
        if self._flow_system.solution is None:
            raise ValueError(f'No solution available for "{self.label}". Run optimization first or load results.')
        if not self._variable_names:
            raise ValueError(f'No variable names available for "{self.label}". Element may not have been modeled yet.')
        return self._flow_system.solution[self._variable_names]

    def _create_reference_structure(self) -> tuple[dict, dict[str, xr.DataArray]]:
        """
        Override to include _variable_names and _constraint_names in serialization.

        These attributes are defined in Element but may not be in subclass constructors,
        so we need to add them explicitly.
        """
        reference_structure, all_extracted_arrays = super()._create_reference_structure()

        # Always include variable/constraint names for solution access after loading
        if self._variable_names:
            reference_structure['_variable_names'] = self._variable_names
        if self._constraint_names:
            reference_structure['_constraint_names'] = self._constraint_names

        return reference_structure, all_extracted_arrays

    def __repr__(self) -> str:
        """Return string representation."""
        return fx_io.build_repr_from_init(self, excluded_params={'self', 'label', 'kwargs'}, skip_default_size=True)

    @staticmethod
    def _valid_label(label: str) -> str:
        """Checks if the label is valid. If not, it is replaced by the default label.

        Raises:
            ValueError: If the label is not valid.
        """
        not_allowed = ['(', ')', '|', '->', '\\', '-slash-']  # \\ is needed to check for \
        if any([sign in label for sign in not_allowed]):
            raise ValueError(
                f'Label "{label}" is not valid. Labels cannot contain the following characters: {not_allowed}. '
                f'Use any other symbol instead'
            )
        if label.endswith(' '):
            logger.error(f'Label "{label}" ends with a space. This will be removed.')
            return label.rstrip()
        return label


# Precompiled regex pattern for natural sorting
_NATURAL_SPLIT = re.compile(r'(\d+)')


def _natural_sort_key(text):
    """Sort key for natural ordering (e.g., bus1, bus2, bus10 instead of bus1, bus10, bus2)."""
    return [int(c) if c.isdigit() else c.lower() for c in _NATURAL_SPLIT.split(text)]


# Type variable for containers
T = TypeVar('T')


class ContainerMixin(dict[str, T]):
    """
    Mixin providing shared container functionality with nice repr and error messages.

    Subclasses must implement _get_label() to extract the label from elements.
    """

    def __init__(
        self,
        elements: list[T] | dict[str, T] | None = None,
        element_type_name: str = 'elements',
        truncate_repr: int | None = None,
        item_name: str | None = None,
    ):
        """
        Args:
            elements: Initial elements to add (list or dict)
            element_type_name: Name for display (e.g., 'components', 'buses')
            truncate_repr: Maximum number of items to show in repr. If None, show all items. Default: None
            item_name: Singular name for error messages (e.g., 'Component', 'Carrier').
                If None, inferred from first added item's class name.
        """
        super().__init__()
        self._element_type_name = element_type_name
        self._truncate_repr = truncate_repr
        self._item_name = item_name

        if elements is not None:
            if isinstance(elements, dict):
                for element in elements.values():
                    self.add(element)
            else:
                for element in elements:
                    self.add(element)

    def _get_label(self, element: T) -> str:
        """
        Extract label from element. Must be implemented by subclasses.

        Args:
            element: Element to get label from

        Returns:
            Label string
        """
        raise NotImplementedError('Subclasses must implement _get_label()')

    def _get_item_name(self) -> str:
        """Get the singular item name for error messages.

        Returns the explicitly set item_name, or infers from the first item's class name.
        Falls back to 'Item' if container is empty and no name was set.
        """
        if self._item_name is not None:
            return self._item_name
        # Infer from first item's class name
        if self:
            first_item = next(iter(self.values()))
            return first_item.__class__.__name__
        return 'Item'

    def add(self, element: T) -> None:
        """Add an element to the container."""
        label = self._get_label(element)
        if label in self:
            item_name = element.__class__.__name__
            raise ValueError(
                f'{item_name} with label "{label}" already exists in {self._element_type_name}. '
                f'Each {item_name.lower()} must have a unique label.'
            )
        self[label] = element

    def __setitem__(self, label: str, element: T) -> None:
        """Set element with validation."""
        element_label = self._get_label(element)
        if label != element_label:
            raise ValueError(
                f'Key "{label}" does not match element label "{element_label}". '
                f'Use the correct label as key or use .add() method.'
            )
        super().__setitem__(label, element)

    def __getitem__(self, label: str) -> T:
        """
        Get element by label with helpful error messages.

        Args:
            label: Label of the element to retrieve

        Returns:
            The element with the given label

        Raises:
            KeyError: If element is not found, with suggestions for similar labels
        """
        try:
            return super().__getitem__(label)
        except KeyError:
            # Provide helpful error with close matches suggestions
            item_name = self._get_item_name()
            suggestions = get_close_matches(label, self.keys(), n=3, cutoff=0.6)
            error_msg = f'{item_name} "{label}" not found in {self._element_type_name}.'
            if suggestions:
                error_msg += f' Did you mean: {", ".join(suggestions)}?'
            else:
                available = list(self.keys())
                if len(available) <= 5:
                    error_msg += f' Available: {", ".join(available)}'
                else:
                    error_msg += f' Available: {", ".join(available[:5])} ... (+{len(available) - 5} more)'
            raise KeyError(error_msg) from None

    def _get_repr(self, max_items: int | None = None) -> str:
        """
        Get string representation with optional truncation.

        Args:
            max_items: Maximum number of items to show. If None, uses instance default (self._truncate_repr).
                      If still None, shows all items.

        Returns:
            Formatted string representation
        """
        # Use provided max_items, or fall back to instance default
        limit = max_items if max_items is not None else self._truncate_repr

        count = len(self)
        title = f'{self._element_type_name.capitalize()} ({count} item{"s" if count != 1 else ""})'

        if not self:
            r = fx_io.format_title_with_underline(title)
            r += '<empty>\n'
        else:
            r = fx_io.format_title_with_underline(title)
            sorted_names = sorted(self.keys(), key=_natural_sort_key)

            if limit is not None and limit > 0 and len(sorted_names) > limit:
                # Show truncated list
                for name in sorted_names[:limit]:
                    r += f' * {name}\n'
                r += f' ... (+{len(sorted_names) - limit} more)\n'
            else:
                # Show all items
                for name in sorted_names:
                    r += f' * {name}\n'

        return r

    def __repr__(self) -> str:
        """Return a string representation using the instance's truncate_repr setting."""
        return self._get_repr()


class ElementContainer(ContainerMixin[T]):
    """
    Container for Element objects (Component, Bus, Flow, Effect).

    Uses element.label_full for keying.
    """

    def _get_label(self, element: T) -> str:
        """Extract label_full from Element."""
        return element.label_full


class ResultsContainer(ContainerMixin[T]):
    """
    Container for Results objects (ComponentResults, BusResults, etc).

    Uses element.label for keying.
    """

    def _get_label(self, element: T) -> str:
        """Extract label from Results object."""
        return element.label


T_element = TypeVar('T_element')


class CompositeContainerMixin(Generic[T_element]):
    """
    Mixin providing unified dict-like access across multiple typed containers.

    This mixin enables classes that manage multiple containers (e.g., components,
    buses, effects, flows) to provide a unified interface for accessing elements
    across all containers, as if they were a single collection.

    Type Parameter:
        T_element: The type of elements stored in the containers. Can be a union type
            for containers holding multiple types (e.g., 'ComponentResults | BusResults').

    Key Features:
        - Dict-like access: `obj['element_name']` searches all containers
        - Iteration: `for label in obj:` iterates over all elements
        - Membership: `'element' in obj` checks across all containers
        - Standard dict methods: keys(), values(), items()
        - Grouped display: Formatted repr showing elements by type
        - Type hints: Full IDE and type checker support

    Subclasses must implement:
        _get_container_groups() -> dict[str, dict]:
            Returns a dictionary mapping group names (e.g., 'Components', 'Buses')
            to container dictionaries. Containers are displayed in the order returned.

    Example:
        ```python
        class MySystem(CompositeContainerMixin[Component | Bus]):
            def __init__(self):
                self.components = {'Boiler': Component(...), 'CHP': Component(...)}
                self.buses = {'Heat': Bus(...), 'Power': Bus(...)}

            def _get_container_groups(self):
                return {
                    'Components': self.components,
                    'Buses': self.buses,
                }


        system = MySystem()
        comp = system['Boiler']  # Type: Component | Bus (with proper IDE support)
        'Heat' in system  # True
        labels = system.keys()  # Type: list[str]
        elements = system.values()  # Type: list[Component | Bus]
        ```

    Integration with ContainerMixin:
        This mixin is designed to work alongside ContainerMixin-based containers
        (ElementContainer, ResultsContainer) by aggregating them into a unified
        interface while preserving their individual functionality.
    """

    def _get_container_groups(self) -> dict[str, ContainerMixin[Any]]:
        """
        Return ordered dict of container groups to aggregate.

        Returns:
            Dictionary mapping group names to container objects (e.g., ElementContainer, ResultsContainer).
            Group names should be capitalized (e.g., 'Components', 'Buses').
            Order determines display order in __repr__.

        Example:
            ```python
            return {
                'Components': self.components,
                'Buses': self.buses,
                'Effects': self.effects,
            }
            ```
        """
        raise NotImplementedError('Subclasses must implement _get_container_groups()')

    def __getitem__(self, key: str) -> T_element:
        """
        Get element by label, searching all containers.

        Args:
            key: Element label to find

        Returns:
            The element with the given label

        Raises:
            KeyError: If element not found, with helpful suggestions
        """
        # Search all containers in order
        for container in self._get_container_groups().values():
            if key in container:
                return container[key]

        # Element not found - provide helpful error
        all_elements = {}
        for container in self._get_container_groups().values():
            all_elements.update(container)

        suggestions = get_close_matches(key, all_elements.keys(), n=3, cutoff=0.6)
        error_msg = f'Element "{key}" not found.'

        if suggestions:
            error_msg += f' Did you mean: {", ".join(suggestions)}?'
        else:
            available = list(all_elements.keys())
            if len(available) <= 5:
                error_msg += f' Available: {", ".join(available)}'
            else:
                error_msg += f' Available: {", ".join(available[:5])} ... (+{len(available) - 5} more)'

        raise KeyError(error_msg)

    def __iter__(self):
        """Iterate over all element labels across all containers."""
        for container in self._get_container_groups().values():
            yield from container.keys()

    def __len__(self) -> int:
        """Return total count of elements across all containers."""
        return sum(len(container) for container in self._get_container_groups().values())

    def __contains__(self, key: str) -> bool:
        """Check if element exists in any container."""
        return any(key in container for container in self._get_container_groups().values())

    def keys(self) -> list[str]:
        """Return all element labels across all containers."""
        return list(self)

    def values(self) -> list[T_element]:
        """Return all element objects across all containers."""
        vals = []
        for container in self._get_container_groups().values():
            vals.extend(container.values())
        return vals

    def items(self) -> list[tuple[str, T_element]]:
        """Return (label, element) pairs for all elements."""
        items = []
        for container in self._get_container_groups().values():
            items.extend(container.items())
        return items

    def _format_grouped_containers(self, title: str | None = None) -> str:
        """
        Format containers as grouped string representation using each container's repr.

        Args:
            title: Optional title for the representation. If None, no title is shown.

        Returns:
            Formatted string with groups and their elements.
            Empty groups are automatically hidden.

        Example output:
            ```
            Components (1 item)
            -------------------
             * Boiler

            Buses (2 items)
            ---------------
             * Heat
             * Power
            ```
        """
        parts = []

        if title:
            parts.append(fx_io.format_title_with_underline(title))

        container_groups = self._get_container_groups()
        for container in container_groups.values():
            if container:  # Only show non-empty groups
                if parts:  # Add spacing between sections
                    parts.append('')
                # Use container's __repr__ which respects its truncate_repr setting
                parts.append(repr(container).rstrip('\n'))

        return '\n'.join(parts)


class Submodel(SubmodelsMixin):
    """Stores Variables and Constraints. Its a subset of a FlowSystemModel.
    Variables and constraints are stored in the main FlowSystemModel, and are referenced here.
    Can have other Submodels assigned, and can be a Submodel of another Submodel.
    """

    def __init__(self, model: FlowSystemModel, label_of_element: str, label_of_model: str | None = None):
        """
        Args:
            model: The FlowSystemModel that is used to create the model.
            label_of_element: The label of the parent (Element). Used to construct the full label of the model.
            label_of_model: The label of the model. Used as a prefix in all variables and constraints.
        """
        self._model = model
        self.label_of_element = label_of_element
        self.label_of_model = label_of_model if label_of_model is not None else self.label_of_element

        self._variables: dict[str, linopy.Variable] = {}  # Mapping from short name to variable
        self._constraints: dict[str, linopy.Constraint] = {}  # Mapping from short name to constraint
        self.submodels: Submodels = Submodels({})

        logger.debug(f'Creating {self.__class__.__name__}  "{self.label_full}"')
        self._do_modeling()

    def add_variables(self, short_name: str = None, **kwargs) -> linopy.Variable:
        """Create and register a variable in one step"""
        if kwargs.get('name') is None:
            if short_name is None:
                raise ValueError('Short name must be provided when no name is given')
            kwargs['name'] = f'{self.label_of_model}|{short_name}'

        variable = self._model.add_variables(**kwargs)
        self.register_variable(variable, short_name)
        return variable

    def add_constraints(self, expression, short_name: str = None, **kwargs) -> linopy.Constraint:
        """Create and register a constraint in one step"""
        if kwargs.get('name') is None:
            if short_name is None:
                raise ValueError('Short name must be provided when no name is given')
            kwargs['name'] = f'{self.label_of_model}|{short_name}'

        constraint = self._model.add_constraints(expression, **kwargs)
        self.register_constraint(constraint, short_name)
        return constraint

    def register_variable(self, variable: linopy.Variable, short_name: str = None) -> linopy.Variable:
        """Register a variable with the model"""
        if short_name is None:
            short_name = variable.name
        elif short_name in self._variables:
            raise ValueError(f'Short name "{short_name}" already assigned to model variables')

        self._variables[short_name] = variable
        return variable

    def register_constraint(self, constraint: linopy.Constraint, short_name: str = None) -> linopy.Constraint:
        """Register a constraint with the model"""
        if short_name is None:
            short_name = constraint.name
        elif short_name in self._constraints:
            raise ValueError(f'Short name "{short_name}" already assigned to model constraint')

        self._constraints[short_name] = constraint
        return constraint

    def __getitem__(self, key: str) -> linopy.Variable:
        """Get a variable by its short name"""
        if key in self._variables:
            return self._variables[key]
        raise KeyError(f'Variable "{key}" not found in model "{self.label_full}"')

    def __contains__(self, name: str) -> bool:
        """Check if a variable exists in the model"""
        return name in self._variables or name in self.variables

    def get(self, name: str, default=None):
        """Get variable by short name, returning default if not found"""
        try:
            return self[name]
        except KeyError:
            return default

    def get_coords(
        self,
        dims: Collection[str] | None = None,
        extra_timestep: bool = False,
    ) -> xr.Coordinates | None:
        return self._model.get_coords(dims=dims, extra_timestep=extra_timestep)

    def filter_variables(
        self,
        filter_by: Literal['binary', 'continuous', 'integer'] | None = None,
        length: Literal['scalar', 'time'] | None = None,
    ):
        if filter_by is None:
            all_variables = self.variables
        elif filter_by == 'binary':
            all_variables = self.variables.binaries
        elif filter_by == 'integer':
            all_variables = self.variables.integers
        elif filter_by == 'continuous':
            all_variables = self.variables.continuous
        else:
            raise ValueError(f'Invalid filter_by "{filter_by}", must be one of "binary", "continous", "integer"')
        if length is None:
            return all_variables
        elif length == 'scalar':
            return all_variables[[name for name in all_variables if all_variables[name].ndim == 0]]
        elif length == 'time':
            return all_variables[[name for name in all_variables if 'time' in all_variables[name].dims]]
        raise ValueError(f'Invalid length "{length}", must be one of "scalar", "time" or None')

    @property
    def label_full(self) -> str:
        return self.label_of_model

    @property
    def variables_direct(self) -> linopy.Variables:
        """Variables of the model, excluding those of sub-models"""
        return self._model.variables[[var.name for var in self._variables.values()]]

    @property
    def constraints_direct(self) -> linopy.Constraints:
        """Constraints of the model, excluding those of sub-models"""
        return self._model.constraints[[con.name for con in self._constraints.values()]]

    @property
    def constraints(self) -> linopy.Constraints:
        """All constraints of the model, including those of all sub-models"""
        names = list(self.constraints_direct) + [
            constraint_name for submodel in self.submodels.values() for constraint_name in submodel.constraints
        ]

        return self._model.constraints[names]

    @property
    def variables(self) -> linopy.Variables:
        """All variables of the model, including those of all sub-models"""
        names = list(self.variables_direct) + [
            variable_name for submodel in self.submodels.values() for variable_name in submodel.variables
        ]

        return self._model.variables[names]

    def __repr__(self) -> str:
        """
        Return a string representation of the linopy model.
        """
        # Extract content from existing representations
        sections = {
            f'Variables: [{len(self.variables)}/{len(self._model.variables)}]': self.variables.__repr__().split(
                '\n', 2
            )[2],
            f'Constraints: [{len(self.constraints)}/{len(self._model.constraints)}]': self.constraints.__repr__().split(
                '\n', 2
            )[2],
            f'Submodels: [{len(self.submodels)}]': self.submodels.__repr__().split('\n', 2)[2],
        }

        # Format sections with headers and underlines
        formatted_sections = fx_io.format_sections_with_headers(sections)

        model_string = f'Submodel "{self.label_of_model}":'
        all_sections = '\n'.join(formatted_sections)

        return f'{model_string}\n{"=" * len(model_string)}\n\n{all_sections}'

    @property
    def timestep_duration(self):
        return self._model.timestep_duration

    def _do_modeling(self):
        """
        Override in subclasses to create variables, constraints, and submodels.

        This method is called during __init__. Create all nested submodels first
        (so their variables exist), then create constraints that reference those variables.
        """
        pass


@dataclass(repr=False)
class Submodels:
    """A simple collection for storing submodels with easy access and representation."""

    data: dict[str, Submodel]

    def __getitem__(self, name: str) -> Submodel:
        """Get a submodel by its name."""
        return self.data[name]

    def __getattr__(self, name: str) -> Submodel:
        """Get a submodel by attribute access."""
        if name in self.data:
            return self.data[name]
        raise AttributeError(f"Submodels has no attribute '{name}'")

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __contains__(self, name: str) -> bool:
        return name in self.data

    def __repr__(self) -> str:
        """Simple representation of the submodels collection."""
        if not self.data:
            return fx_io.format_title_with_underline('flixopt.structure.Submodels') + ' <empty>\n'

        total_vars = sum(len(submodel.variables) for submodel in self.data.values())
        total_cons = sum(len(submodel.constraints) for submodel in self.data.values())

        title = (
            f'flixopt.structure.Submodels ({total_vars} vars, {total_cons} constraints, {len(self.data)} submodels):'
        )

        result = fx_io.format_title_with_underline(title)
        for name, submodel in self.data.items():
            type_name = submodel.__class__.__name__
            var_count = len(submodel.variables)
            con_count = len(submodel.constraints)
            result += f' * {name} [{type_name}] ({var_count}v/{con_count}c)\n'

        return result

    def items(self) -> ItemsView[str, Submodel]:
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def add(self, submodel: Submodel, name: str) -> None:
        """Add a submodel to the collection."""
        self.data[name] = submodel

    def get(self, name: str, default=None):
        """Get submodel by name, returning default if not found."""
        return self.data.get(name, default)


class ElementModel(Submodel):
    """
    Stores the mathematical Variables and Constraints for Elements.
    ElementModels are directly registered in the main FlowSystemModel
    """

    def __init__(self, model: FlowSystemModel, element: Element):
        """
        Args:
            model: The FlowSystemModel that is used to create the model.
            element: The element this model is created for.
        """
        self.element = element
        super().__init__(model, label_of_element=element.label_full, label_of_model=element.label_full)
        self._model.add_submodels(self, short_name=self.label_of_model)

    def results_structure(self):
        return {
            'label': self.label_full,
            'variables': list(self.variables),
            'constraints': list(self.constraints),
        }
