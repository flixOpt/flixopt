"""
This module contains the core structure of the flixopt framework.
These classes are not directly used by the end user, but are used by other modules.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import re
import warnings
from abc import ABC, abstractmethod
from difflib import get_close_matches
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
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
from .core import TimeSeriesData, align_to_coords, get_dataarray_stats
from .id_list import IdList

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from collections.abc import Collection

    from .effects import EffectsModel
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


def _ensure_coords(
    data: xr.DataArray | float | int,
    coords: xr.Coordinates | dict,
) -> xr.DataArray | float:
    """Broadcast data to coords if needed.

    This is used at the linopy interface to ensure bounds are properly broadcasted
    to the target variable shape. Linopy needs at least one bound to have all
    dimensions to determine the variable shape.

    Note: Infinity values (-inf, inf) are kept as scalars because linopy uses
    special checks like `if (lower != -inf)` that fail with DataArrays.
    """
    # Handle both dict and xr.Coordinates
    if isinstance(coords, dict):
        coord_dims = list(coords.keys())
    else:
        coord_dims = list(coords.dims)

    # Handle None (no bound specified)
    if data is None:
        return data

    # Keep infinity values as scalars (linopy uses them for special checks)
    if not isinstance(data, xr.DataArray):
        if np.isscalar(data) and np.isinf(data):
            return data
        # Finite scalar - create full DataArray
        return xr.DataArray(data, coords=coords, dims=coord_dims)

    if set(data.dims) == set(coord_dims):
        # Has all dims - ensure correct order
        if data.dims != tuple(coord_dims):
            return data.transpose(*coord_dims)
        return data

    # Broadcast to full coords using np.broadcast_to (zero-copy view).
    # We avoid xarray's broadcast_like because it creates lazy views whose internal
    # dim ordering can leak through xr.broadcast in linopy, causing wrong dim order.
    target_shape = tuple(len(coords[d]) for d in coord_dims)
    existing_dims = [d for d in coord_dims if d in data.dims]
    data_transposed = data.transpose(*existing_dims)
    shape_for_broadcast = tuple(len(coords[d]) if d in data.dims else 1 for d in coord_dims)
    values = np.broadcast_to(data_transposed.values.reshape(shape_for_broadcast), target_shape)
    return xr.DataArray(values, coords=coords, dims=coord_dims)


class ExpansionMode(Enum):
    """How a variable is expanded when converting clustered segments back to full time series."""

    REPEAT = 'repeat'
    INTERPOLATE = 'interpolate'
    DIVIDE = 'divide'
    FIRST_TIMESTEP = 'first_timestep'


# =============================================================================
# New Categorization Enums for Type-Level Models
# =============================================================================


class ConstraintType(Enum):
    """What kind of constraint this is.

    Provides semantic meaning for constraints to enable batch processing.
    """

    # === Tracking equations ===
    TRACKING = 'tracking'  # var = sum(other) or var = expression

    # === Bounds ===
    UPPER_BOUND = 'upper_bound'  # var <= bound
    LOWER_BOUND = 'lower_bound'  # var >= bound

    # === Balance ===
    BALANCE = 'balance'  # sum(inflows) == sum(outflows)

    # === Linking ===
    LINKING = 'linking'  # var[t+1] = f(var[t])

    # === State transitions ===
    STATE_TRANSITION = 'state_transition'  # status, startup, shutdown relationships

    # === Piecewise ===
    PIECEWISE = 'piecewise'  # SOS2, lambda constraints

    # === Other ===
    OTHER = 'other'  # Uncategorized


# =============================================================================
# Central Variable/Constraint Naming
# =============================================================================


class FlowVarName:
    """Central variable naming for Flow type-level models.

    All variable and constraint names for FlowsModel should reference these constants.
    Pattern: flow|{variable_name} (max 2 levels for variables)
    """

    # === Flow Variables ===
    RATE = 'flow|rate'
    HOURS = 'flow|hours'
    TOTAL_FLOW_HOURS = 'flow|total_flow_hours'
    STATUS = 'flow|status'
    SIZE = 'flow|size'
    INVESTED = 'flow|invested'

    # === Status Tracking Variables (for flows with status) ===
    ACTIVE_HOURS = 'flow|active_hours'
    STARTUP = 'flow|startup'
    SHUTDOWN = 'flow|shutdown'
    INACTIVE = 'flow|inactive'
    STARTUP_COUNT = 'flow|startup_count'

    # === Duration Tracking Variables ===
    UPTIME = 'flow|uptime'
    DOWNTIME = 'flow|downtime'


# Constraint names for FlowsModel (references FlowVarName)
class _FlowConstraint:
    """Constraint names for FlowsModel.

    Constraints can have 3 levels: flow|{var}|{constraint_type}
    """

    HOURS_EQ = 'flow|hours_eq'
    RATE_STATUS_LB = 'flow|rate_status_lb'
    RATE_STATUS_UB = 'flow|rate_status_ub'
    ACTIVE_HOURS = FlowVarName.ACTIVE_HOURS  # Same as variable (tracking constraint)
    COMPLEMENTARY = 'flow|complementary'
    SWITCH_TRANSITION = 'flow|switch_transition'
    SWITCH_MUTEX = 'flow|switch_mutex'
    SWITCH_INITIAL = 'flow|switch_initial'
    STARTUP_COUNT = FlowVarName.STARTUP_COUNT  # Same as variable
    CLUSTER_CYCLIC = 'flow|cluster_cyclic'

    # Uptime tracking constraints (built from variable name)
    UPTIME_UB = f'{FlowVarName.UPTIME}|ub'
    UPTIME_FORWARD = f'{FlowVarName.UPTIME}|forward'
    UPTIME_BACKWARD = f'{FlowVarName.UPTIME}|backward'
    UPTIME_INITIAL = f'{FlowVarName.UPTIME}|initial'
    UPTIME_INITIAL_CONTINUATION = f'{FlowVarName.UPTIME}|initial_continuation'

    # Downtime tracking constraints (built from variable name)
    DOWNTIME_UB = f'{FlowVarName.DOWNTIME}|ub'
    DOWNTIME_FORWARD = f'{FlowVarName.DOWNTIME}|forward'
    DOWNTIME_BACKWARD = f'{FlowVarName.DOWNTIME}|backward'
    DOWNTIME_INITIAL = f'{FlowVarName.DOWNTIME}|initial'
    DOWNTIME_INITIAL_CONTINUATION = f'{FlowVarName.DOWNTIME}|initial_continuation'


FlowVarName.Constraint = _FlowConstraint


class ComponentVarName:
    """Central variable naming for Component type-level models.

    All variable and constraint names for ComponentsModel should reference these constants.
    Pattern: {element_type}|{variable_suffix}
    """

    # === Component Status Variables ===
    STATUS = 'component|status'
    ACTIVE_HOURS = 'component|active_hours'
    STARTUP = 'component|startup'
    SHUTDOWN = 'component|shutdown'
    INACTIVE = 'component|inactive'
    STARTUP_COUNT = 'component|startup_count'

    # === Duration Tracking Variables ===
    UPTIME = 'component|uptime'
    DOWNTIME = 'component|downtime'


# Constraint names for ComponentsModel (references ComponentVarName)
class _ComponentConstraint:
    """Constraint names for ComponentsModel.

    Constraints can have 3 levels: component|{var}|{constraint_type}
    """

    ACTIVE_HOURS = ComponentVarName.ACTIVE_HOURS
    COMPLEMENTARY = 'component|complementary'
    SWITCH_TRANSITION = 'component|switch_transition'
    SWITCH_MUTEX = 'component|switch_mutex'
    SWITCH_INITIAL = 'component|switch_initial'
    STARTUP_COUNT = ComponentVarName.STARTUP_COUNT
    CLUSTER_CYCLIC = 'component|cluster_cyclic'

    # Uptime tracking constraints
    UPTIME_UB = f'{ComponentVarName.UPTIME}|ub'
    UPTIME_FORWARD = f'{ComponentVarName.UPTIME}|forward'
    UPTIME_BACKWARD = f'{ComponentVarName.UPTIME}|backward'
    UPTIME_INITIAL = f'{ComponentVarName.UPTIME}|initial'
    UPTIME_INITIAL_CONTINUATION = f'{ComponentVarName.UPTIME}|initial_continuation'

    # Downtime tracking constraints
    DOWNTIME_UB = f'{ComponentVarName.DOWNTIME}|ub'
    DOWNTIME_FORWARD = f'{ComponentVarName.DOWNTIME}|forward'
    DOWNTIME_BACKWARD = f'{ComponentVarName.DOWNTIME}|backward'
    DOWNTIME_INITIAL = f'{ComponentVarName.DOWNTIME}|initial'
    DOWNTIME_INITIAL_CONTINUATION = f'{ComponentVarName.DOWNTIME}|initial_continuation'


ComponentVarName.Constraint = _ComponentConstraint


class BusVarName:
    """Central variable naming for Bus type-level models."""

    VIRTUAL_SUPPLY = 'bus|virtual_supply'
    VIRTUAL_DEMAND = 'bus|virtual_demand'


class StorageVarName:
    """Central variable naming for Storage type-level models.

    All variable and constraint names for StoragesModel should reference these constants.
    """

    # === Storage Variables ===
    CHARGE = 'storage|charge'
    NETTO = 'storage|netto'
    SIZE = 'storage|size'
    INVESTED = 'storage|invested'


class InterclusterStorageVarName:
    """Central variable naming for InterclusterStoragesModel."""

    CHARGE_STATE = 'intercluster_storage|charge_state'
    NETTO_DISCHARGE = 'intercluster_storage|netto_discharge'
    SOC_BOUNDARY = 'intercluster_storage|SOC_boundary'
    SIZE = 'intercluster_storage|size'
    INVESTED = 'intercluster_storage|invested'


class ConverterVarName:
    """Central variable naming for Converter type-level models.

    All variable and constraint names for ConvertersModel should reference these constants.
    Pattern: converter|{variable_name}
    """

    # === Piecewise Conversion Variables ===
    # Prefix for all piecewise-related names (used by PiecewiseBuilder)
    PIECEWISE_PREFIX = 'converter|piecewise_conversion'

    # Full variable names (prefix + suffix added by PiecewiseBuilder)
    PIECEWISE_INSIDE = f'{PIECEWISE_PREFIX}|inside_piece'
    PIECEWISE_LAMBDA0 = f'{PIECEWISE_PREFIX}|lambda0'
    PIECEWISE_LAMBDA1 = f'{PIECEWISE_PREFIX}|lambda1'


# Constraint names for ConvertersModel
class _ConverterConstraint:
    """Constraint names for ConvertersModel.

    Constraints can have 3 levels: converter|{var}|{constraint_type}
    """

    # Linear conversion constraints (indexed by equation number)
    CONVERSION = 'conversion'

    # Piecewise conversion constraints
    PIECEWISE_LAMBDA_SUM = 'piecewise_conversion|lambda_sum'
    PIECEWISE_SINGLE_SEGMENT = 'piecewise_conversion|single_segment'
    PIECEWISE_COUPLING = 'piecewise_conversion|coupling'


ConverterVarName.Constraint = _ConverterConstraint


class TransmissionVarName:
    """Central variable naming for Transmission type-level models.

    All variable and constraint names for TransmissionsModel should reference these constants.
    Pattern: transmission|{variable_name}

    Note: Transmissions currently don't create variables (only constraints linking flows).
    """

    pass  # No variables yet - transmissions only create constraints


# Constraint names for TransmissionsModel
class _TransmissionConstraint:
    """Constraint names for TransmissionsModel.

    Batched constraints with transmission dimension: transmission|{constraint_type}
    """

    # Efficiency constraints (batched with transmission dimension)
    DIR1 = 'dir1'
    DIR2 = 'dir2'

    # Size constraints
    BALANCED = 'balanced'

    # Status coupling (for absolute losses)
    IN1_STATUS_COUPLING = 'in1_status_coupling'
    IN2_STATUS_COUPLING = 'in2_status_coupling'


TransmissionVarName.Constraint = _TransmissionConstraint


class EffectVarName:
    """Central variable naming for Effect models."""

    # === Effect Variables ===
    PERIODIC = 'effect|periodic'
    TEMPORAL = 'effect|temporal'
    PER_TIMESTEP = 'effect|per_timestep'
    TOTAL = 'effect|total'


NAME_TO_EXPANSION: dict[str, ExpansionMode] = {
    StorageVarName.CHARGE: ExpansionMode.INTERPOLATE,
    InterclusterStorageVarName.CHARGE_STATE: ExpansionMode.INTERPOLATE,
    FlowVarName.STARTUP: ExpansionMode.FIRST_TIMESTEP,
    FlowVarName.SHUTDOWN: ExpansionMode.FIRST_TIMESTEP,
    ComponentVarName.STARTUP: ExpansionMode.FIRST_TIMESTEP,
    ComponentVarName.SHUTDOWN: ExpansionMode.FIRST_TIMESTEP,
    EffectVarName.PER_TIMESTEP: ExpansionMode.DIVIDE,
    'share|temporal': ExpansionMode.DIVIDE,
}


# =============================================================================
# TypeModel Base Class
# =============================================================================


class TypeModel(ABC):
    """Base class for type-level models that handle ALL elements of a type.

    Unlike Submodel (one per element instance), TypeModel handles ALL elements
    of a given type (e.g., FlowsModel for ALL Flows) in a single instance.

    This enables true vectorized batch creation:
    - One variable with 'flow' dimension for all flows
    - One constraint call for all elements

    Variable/Constraint Naming Convention:
        - Variables: '{dim_name}|{var_name}' e.g., 'flow|rate', 'storage|charge'
        - Constraints: '{dim_name}|{constraint_name}' e.g., 'flow|rate_ub'

    Dimension Naming:
        - Each element type uses its own dimension name: 'flow', 'storage', 'effect', 'component'
        - This prevents unwanted broadcasting when merging into solution Dataset

    Attributes:
        model: The FlowSystemModel to create variables/constraints in.
        data: Data object providing element_ids, dim_name, and elements.
        elements: IdList of elements this model manages.
        element_ids: List of element identifiers.
        dim_name: Dimension name for this element type (e.g., 'flow', 'storage').

    Example:
        >>> class FlowsModel(TypeModel):
        ...     def create_variables(self):
        ...         self.add_variables(
        ...             'flow|rate',  # Creates 'flow|rate' with 'flow' dimension
        ...             lower=data.lower_bounds,
        ...             upper=data.upper_bounds,
        ...         )
    """

    def __init__(self, model: FlowSystemModel, data):
        """Initialize the type-level model.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            data: Data object providing element_ids, dim_name, and elements.
        """
        self.model = model
        self.data = data

        # Storage for created variables and constraints
        self._variables: dict[str, linopy.Variable] = {}
        self._constraints: dict[str, linopy.Constraint] = {}

    @property
    def elements(self) -> IdList:
        """IdList of elements in this model."""
        return self.data.elements

    @property
    def element_ids(self) -> list[str]:
        """List of element IDs (label_full) in this model."""
        return self.data.element_ids

    @property
    def dim_name(self) -> str:
        """Dimension name for this element type (e.g., 'flow', 'storage')."""
        return self.data.dim_name

    @abstractmethod
    def create_variables(self) -> None:
        """Create all batched variables for this element type.

        Implementations should use add_variables() to create variables
        with the element dimension already included.
        """

    @abstractmethod
    def create_constraints(self) -> None:
        """Create all batched constraints for this element type.

        Implementations should create vectorized constraints that operate
        on the full element dimension at once.
        """

    def add_variables(
        self,
        name: str,
        lower: xr.DataArray | float = -np.inf,
        upper: xr.DataArray | float = np.inf,
        dims: tuple[str, ...] | None = ('time',),
        element_ids: list[str] | None = None,
        mask: xr.DataArray | None = None,
        extra_timestep: bool = False,
        **kwargs,
    ) -> linopy.Variable:
        """Create a batched variable with element dimension.

        Args:
            name: Variable name (e.g., 'flow|rate'). Used as-is for the linopy variable.
            lower: Lower bounds (scalar or per-element DataArray).
            upper: Upper bounds (scalar or per-element DataArray).
            dims: Dimensions beyond 'element'. None means ALL model dimensions.
            element_ids: Subset of element IDs. None means all elements.
            mask: Optional boolean mask. If provided, automatically reindexed and broadcast
                to match the built coords. True = create variable, False = skip.
            extra_timestep: If True, extends time dimension by 1 (for charge_state boundaries).
            **kwargs: Additional arguments passed to model.add_variables().

        Returns:
            The created linopy Variable with element dimension.
        """
        coords = self._build_coords(dims, element_ids=element_ids, extra_timestep=extra_timestep)

        # Broadcast mask to match coords if needed
        if mask is not None:
            mask = mask.reindex({self.dim_name: coords[self.dim_name]}, fill_value=False)
            dim_order = list(coords.keys())
            for dim in dim_order:
                if dim not in mask.dims:
                    mask = mask.expand_dims({dim: coords[dim]})
            mask = mask.astype(bool)
            kwargs['mask'] = mask.transpose(*dim_order)

        variable = self.model.add_variables(
            lower=lower,
            upper=upper,
            coords=coords,
            name=name,
            **kwargs,
        )

        # Store reference
        self._variables[name] = variable
        return variable

    def add_constraints(
        self,
        expression: linopy.expressions.LinearExpression,
        name: str,
        **kwargs,
    ) -> linopy.Constraint:
        """Create a batched constraint for all elements.

        Args:
            expression: The constraint expression (e.g., lhs == rhs, lhs <= rhs).
            name: Constraint name (will be prefixed with element type).
            **kwargs: Additional arguments passed to model.add_constraints().

        Returns:
            The created linopy Constraint.
        """
        full_name = f'{self.dim_name}|{name}'
        constraint = self.model.add_constraints(expression, name=full_name, **kwargs)
        self._constraints[name] = constraint
        return constraint

    def _build_coords(
        self,
        dims: tuple[str, ...] | None = ('time',),
        element_ids: list[str] | None = None,
        extra_timestep: bool = False,
    ) -> xr.Coordinates:
        """Build coordinate dict with element-type dimension + model dimensions.

        Args:
            dims: Tuple of dimension names from the model. If None, includes ALL model dimensions.
            element_ids: Subset of element IDs. If None, uses all self.element_ids.
            extra_timestep: If True, extends time dimension by 1 (for charge_state boundaries).

        Returns:
            xarray Coordinates with element-type dim (e.g., 'flow') + requested dims.
        """
        if element_ids is None:
            element_ids = self.element_ids

        # Use element-type-specific dimension name (e.g., 'flow', 'storage')
        coord_dict: dict[str, Any] = {self.dim_name: pd.Index(element_ids, name=self.dim_name)}

        # Add model dimensions
        model_coords = self.model.get_coords(dims=dims, extra_timestep=extra_timestep)
        if model_coords is not None:
            # Add all returned model coords (get_coords handles auto-pairing, e.g., cluster+time)
            for dim, coord in model_coords.items():
                coord_dict[dim] = coord

        return xr.Coordinates(coord_dict)

    def _broadcast_to_model_coords(
        self,
        data: xr.DataArray | float,
        dims: list[str] | None = None,
    ) -> xr.DataArray:
        """Broadcast data to include model dimensions.

        Args:
            data: Input data (scalar or DataArray).
            dims: Model dimensions to include. None = all (time, period, scenario).

        Returns:
            DataArray broadcast to include model dimensions and element dimension.
        """
        # Get model coords for broadcasting
        model_coords = self.model.get_coords(dims=dims)

        # Convert scalar to DataArray with element dimension
        if np.isscalar(data):
            # Start with just element dimension
            result = xr.DataArray(
                [data] * len(self.element_ids),
                dims=[self.dim_name],
                coords={self.dim_name: self.element_ids},
            )
            if model_coords is not None:
                # Broadcast to include model coords
                template = xr.DataArray(coords=model_coords)
                result = result.broadcast_like(template)
            return result

        if not isinstance(data, xr.DataArray):
            data = xr.DataArray(data)

        if model_coords is None:
            return data

        # Create template with all required dims
        template = xr.DataArray(coords=model_coords)
        return data.broadcast_like(template)

    def __getitem__(self, name: str) -> linopy.Variable:
        """Get a variable by name (e.g., model['flow|rate'])."""
        return self._variables[name]

    def __contains__(self, name: str) -> bool:
        """Check if a variable exists (e.g., 'flow|rate' in model)."""
        return name in self._variables

    def get(self, name: str, default=None) -> linopy.Variable | None:
        """Get a variable by name, returning default if not found."""
        return self._variables.get(name, default)

    def get_variable(self, name: str, element_id: str | None = None) -> linopy.Variable:
        """Get a variable, optionally sliced to a specific element.

        Args:
            name: Variable name (e.g., 'flow|rate').
            element_id: If provided, return slice for this element only.

        Returns:
            Full batched variable or element slice.
        """
        variable = self._variables[name]
        if element_id is not None:
            return variable.sel({self.dim_name: element_id})
        return variable

    def get_constraint(self, name: str) -> linopy.Constraint:
        """Get a constraint by name.

        Args:
            name: Constraint name.

        Returns:
            The constraint.
        """
        return self._constraints[name]

    @property
    def variables(self) -> dict[str, linopy.Variable]:
        """All variables created by this type model."""
        return self._variables

    @property
    def constraints(self) -> dict[str, linopy.Constraint]:
        """All constraints created by this type model."""
        return self._constraints

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'elements={len(self.elements)}, '
            f'vars={len(self._variables)}, '
            f'constraints={len(self._constraints)})'
        )


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


# =============================================================================
# Standalone Serialization Functions (path-based DataArray naming)
# =============================================================================


def _is_numeric(obj: Any) -> bool:
    """Check if an object is a numeric value that should be stored as a DataArray.

    Matches arrays (np.ndarray, pd.Series, pd.DataFrame) and scalars
    (int, float, np.integer, np.floating). Excludes bool (subclass of int).

    Storing numerics as DataArrays enables:
    - Dataset operations (resampling, selection, etc.)
    - Efficient binary storage in NetCDF
    - Dtype preservation
    """
    if isinstance(obj, bool):
        return False
    return isinstance(obj, (np.ndarray, pd.Series, pd.DataFrame, int, float, np.integer, np.floating))


def create_reference_structure(
    obj, path_prefix: str = '', coords: dict[str, pd.Index] | None = None
) -> tuple[dict, dict[str, xr.DataArray]]:
    """Extract DataArrays from any registered object, using path-based keys.

    Works with
    any object whose class is in CLASS_REGISTRY, any dataclass, or any object
    with an inspectable ``__init__``.

    DataArray keys are deterministic paths built from the object hierarchy:
    ``element_id.param_name`` for top-level, ``element_id.param.sub_param`` for nested.

    Args:
        obj: Object to serialize.
        path_prefix: Path prefix for DataArray keys (e.g., ``'components.Boiler'``).
        coords: Model coordinates for aligning numeric arrays. When provided,
            numpy arrays / pandas objects are converted to properly-dimensioned
            DataArrays via ``align_to_coords``, ensuring they participate in
            dataset operations (resampling, selection) and avoid dimension conflicts.

    Returns:
        Tuple of (reference_structure dict, extracted_arrays dict).
    """
    structure: dict[str, Any] = {'__class__': obj.__class__.__name__}
    all_arrays: dict[str, xr.DataArray] = {}

    params = _get_serializable_params(obj)

    for name, value in params.items():
        if value is None:
            continue
        if isinstance(value, pd.Index):
            logger.debug(f'Skipping {name=} because it is an Index')
            continue

        param_path = f'{path_prefix}.{name}' if path_prefix else name
        processed, arrays = _extract_recursive(value, param_path, coords)
        all_arrays.update(arrays)
        if processed is not None and not _is_empty(processed):
            structure[name] = processed

    return structure, all_arrays


def _extract_recursive(
    obj: Any, path: str, coords: dict[str, pd.Index] | None = None
) -> tuple[Any, dict[str, xr.DataArray]]:
    """Recursively extract DataArrays, using *path* as the array key.

    Handles DataArrays, numeric arrays (np.ndarray, pd.Series, pd.DataFrame),
    registered classes, plain dataclasses, dicts, lists, tuples, sets, IdList,
    and scalar/basic types.

    When *coords* is provided, numeric arrays are aligned to model dimensions
    via ``align_to_coords`` to get proper dimension names.
    """
    arrays: dict[str, xr.DataArray] = {}

    if isinstance(obj, xr.DataArray):
        arrays[path] = obj.rename(path)
        return f':::{path}', arrays

    # Numeric values → DataArray for dataset operations and binary NetCDF storage.
    if coords is not None and _is_numeric(obj):
        da = align_to_coords(obj, coords, name=path)
        arrays[path] = da.rename(path)
        return f':::{path}', arrays

    if obj.__class__.__name__ in CLASS_REGISTRY:
        return create_reference_structure(obj, path_prefix=path, coords=coords)

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        structure: dict[str, Any] = {'__class__': obj.__class__.__name__}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            if value is None:
                continue
            processed, field_arrays = _extract_recursive(value, f'{path}.{field.name}', coords)
            arrays.update(field_arrays)
            if processed is not None and not _is_empty(processed):
                structure[field.name] = processed
        return structure, arrays

    if isinstance(obj, IdList):
        processed_list: list[Any] = []
        for key, item in obj.items():
            p, a = _extract_recursive(item, f'{path}.{key}', coords)
            arrays.update(a)
            processed_list.append(p)
        return processed_list, arrays

    if isinstance(obj, dict):
        processed_dict = {}
        for key, value in obj.items():
            p, a = _extract_recursive(value, f'{path}.{key}', coords)
            arrays.update(a)
            processed_dict[key] = p
        return processed_dict, arrays

    if isinstance(obj, (list, tuple)):
        processed_list: list[Any] = []
        for i, item in enumerate(obj):
            p, a = _extract_recursive(item, f'{path}.{i}', coords)
            arrays.update(a)
            processed_list.append(p)
        return processed_list, arrays

    if isinstance(obj, set):
        processed_list = []
        for i, item in enumerate(obj):
            p, a = _extract_recursive(item, f'{path}.{i}', coords)
            arrays.update(a)
            processed_list.append(p)
        return processed_list, arrays

    # Scalar / basic type
    return _to_basic_type(obj), arrays


def _has_dataclass_init(cls: type) -> bool:
    """Check if a class uses a dataclass-generated __init__ (not a custom override).

    Returns True only when @dataclass was applied directly to ``cls`` with init=True.
    Classes that merely inherit from a dataclass (e.g. Boiler(LinearConverter))
    but define their own __init__ return False.
    """
    params = cls.__dict__.get('__dataclass_params__')
    return params is not None and params.init


def _get_serializable_params(obj) -> dict[str, Any]:
    """Get name->value pairs for serialization from ``__init__`` parameters."""
    _skip = {'self', 'label', 'label_as_positional', 'args', 'kwargs'}

    # Class-level exclusion set for IO serialization
    io_exclude = getattr(obj.__class__, '_io_exclude', set())
    _skip |= io_exclude

    # Prefer dataclass fields when class uses dataclass-generated __init__
    if _has_dataclass_init(obj.__class__):
        return {f.name: getattr(obj, f.name, None) for f in dataclasses.fields(obj) if f.name not in _skip and f.init}

    # Fallback for non-dataclass or custom-__init__ classes
    sig = inspect.signature(obj.__init__)
    return {name: getattr(obj, name, None) for name in sig.parameters if name not in _skip}


def _to_basic_type(obj: Any) -> Any:
    """Convert a single value to a JSON-compatible basic Python type."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.ndarray, pd.Series, pd.DataFrame)):
        return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
    if isinstance(obj, dict):
        return {k: _to_basic_type(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_basic_type(item) for item in obj]
    if isinstance(obj, set):
        return [_to_basic_type(item) for item in obj]
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    if hasattr(obj, '__dict__'):
        logger.warning(f'Converting custom object {type(obj)} to dict representation: {obj}')
        return {str(k): _to_basic_type(v) for k, v in obj.__dict__.items()}
    logger.error(f'Converting unknown type {type(obj)} to string: {obj}')
    return str(obj)


def _is_empty(obj: Any) -> bool:
    """Check if object is an empty container (dict, list, tuple, set)."""
    return isinstance(obj, (dict, list, tuple, set)) and len(obj) == 0


def resolve_reference_structure(structure: Any, arrays_dict: dict[str, xr.DataArray]) -> Any:
    """Resolve a reference structure back to actual objects.

    Resolves ``:::path`` DataArray references and ``__class__`` markers back to objects.
    Handles ``:::path`` DataArray references, registered classes, lists, and dicts.

    Args:
        structure: Structure containing ``:::path`` references or ``__class__`` markers.
        arrays_dict: Dictionary mapping path keys to DataArrays.

    Returns:
        Resolved structure with DataArrays and reconstructed objects.
    """
    # Handle DataArray references
    if isinstance(structure, str) and structure.startswith(':::'):
        return _resolve_dataarray_reference(structure, arrays_dict)

    if isinstance(structure, list):
        resolved_list = []
        for item in structure:
            resolved_item = resolve_reference_structure(item, arrays_dict)
            if resolved_item is not None:
                resolved_list.append(resolved_item)
        return resolved_list

    if isinstance(structure, dict):
        if structure.get('__class__'):
            class_name = structure['__class__']
            if class_name not in CLASS_REGISTRY:
                raise ValueError(
                    f"Class '{class_name}' not found in CLASS_REGISTRY. "
                    f'Available classes: {list(CLASS_REGISTRY.keys())}'
                )

            nested_class = CLASS_REGISTRY[class_name]
            nested_data = {k: v for k, v in structure.items() if k != '__class__'}
            resolved_nested_data = resolve_reference_structure(nested_data, arrays_dict)

            try:
                # Discover init parameters — prefer dataclass fields
                if _has_dataclass_init(nested_class):
                    init_params = {f.name for f in dataclasses.fields(nested_class) if f.init} | {'self'}
                else:
                    init_params = set(inspect.signature(nested_class.__init__).parameters.keys())

                # Filter out legacy runtime attrs from old serialized files
                _legacy_deferred = {'_variable_names', '_constraint_names'}
                constructor_data = {k: v for k, v in resolved_nested_data.items() if k not in _legacy_deferred}

                # Handle renamed parameters from old serialized data
                if 'label' in constructor_data and 'label' not in init_params:
                    new_key = 'flow_id' if 'flow_id' in init_params else 'id'
                    constructor_data[new_key] = constructor_data.pop('label')
                if 'id' in constructor_data and 'id' not in init_params and 'flow_id' in init_params:
                    constructor_data['flow_id'] = constructor_data.pop('id')

                # Check for unknown parameters
                unknown_params = set(constructor_data.keys()) - init_params
                if unknown_params:
                    raise TypeError(
                        f'{class_name}.__init__() got unexpected keyword arguments: {unknown_params}. '
                        f'This may indicate renamed parameters that need conversion. '
                        f'Valid parameters are: {init_params - {"self"}}'
                    )

                instance = nested_class(**constructor_data)

                return instance
            except TypeError as e:
                raise ValueError(f'Failed to create instance of {class_name}: {e}') from e
            except Exception as e:
                raise ValueError(f'Failed to create instance of {class_name}: {e}') from e
        else:
            # Regular dictionary
            resolved_dict = {}
            for key, value in structure.items():
                resolved_value = resolve_reference_structure(value, arrays_dict)
                if resolved_value is not None or value is None:
                    resolved_dict[key] = resolved_value
            return resolved_dict

    return structure


def _resolve_dataarray_reference(reference: str, arrays_dict: dict[str, xr.DataArray]) -> xr.DataArray | TimeSeriesData:
    """Resolve a single ``:::path`` DataArray reference.

    Args:
        reference: Reference string starting with ``:::``.
        arrays_dict: Dictionary of available DataArrays.

    Returns:
        Resolved DataArray or TimeSeriesData object.
    """
    array_name = reference[3:]
    if array_name not in arrays_dict:
        raise ValueError(f"Referenced DataArray '{array_name}' not found in dataset")

    array = arrays_dict[array_name]

    # Handle null values with warning
    has_nulls = (np.issubdtype(array.dtype, np.floating) and np.any(np.isnan(array.values))) or (
        array.dtype == object and pd.isna(array.values).any()
    )
    if has_nulls:
        logger.error(f"DataArray '{array_name}' contains null values. Dropping all-null along present dims.")
        if 'time' in array.dims:
            array = array.dropna(dim='time', how='all')

    if TimeSeriesData.is_timeseries_data(array):
        return TimeSeriesData.from_dataarray(array)

    # Unwrap 0-d DataArrays back to Python scalars
    if array.ndim == 0:
        return array.item()

    return array


def replace_references_with_stats(structure, arrays_dict: dict[str, xr.DataArray]):
    """Replace ``:::path`` DataArray references with statistical summaries."""
    if isinstance(structure, str) and structure.startswith(':::'):
        array_name = structure[3:]
        if array_name in arrays_dict:
            return get_dataarray_stats(arrays_dict[array_name])
        return structure
    elif isinstance(structure, dict):
        return {k: replace_references_with_stats(v, arrays_dict) for k, v in structure.items()}
    elif isinstance(structure, list):
        return [replace_references_with_stats(item, arrays_dict) for item in structure]
    return structure


class _BuildTimer:
    """Simple timing helper for build_model profiling."""

    def __init__(self):
        import time

        self._time = time
        self._records: list[tuple[str, float]] = [('start', time.perf_counter())]

    def record(self, name: str) -> None:
        self._records.append((name, self._time.perf_counter()))

    def print_summary(self) -> None:
        print('\n  Type-Level Modeling Timing Breakdown:')
        for i in range(1, len(self._records)):
            name = self._records[i][0]
            elapsed = (self._records[i][1] - self._records[i - 1][1]) * 1000
            print(f'    {name:30s}: {elapsed:8.2f}ms')
        total = (self._records[-1][1] - self._records[0][1]) * 1000
        print(f'    {"TOTAL":30s}: {total:8.2f}ms')


class FlowSystemModel(linopy.Model):
    """
    The FlowSystemModel is the linopy Model that is used to create the mathematical model of the flow_system.
    It is used to create and store the variables and constraints for the flow_system.

    Args:
        flow_system: The flow_system that is used to create the model.
    """

    def __init__(self, flow_system: FlowSystem):
        super().__init__(force_dim_names=True)
        self.flow_system = flow_system
        self.effects: EffectsModel | None = None
        self._flows_model: TypeModel | None = None  # Reference to FlowsModel
        self._buses_model: TypeModel | None = None  # Reference to BusesModel
        self._storages_model = None  # Reference to StoragesModel
        self._components_model = None  # Reference to ComponentsModel
        self._converters_model = None  # Reference to ConvertersModel
        self._transmissions_model = None  # Reference to TransmissionsModel
        self._is_built: bool = False  # Set True after build_model() completes

    def add_variables(
        self,
        lower: xr.DataArray | float | None = None,
        upper: xr.DataArray | float | None = None,
        coords: xr.Coordinates | None = None,
        binary: bool = False,
        **kwargs,
    ) -> linopy.Variable:
        """Override to ensure bounds are broadcasted to coords shape.

        Linopy uses the union of all DataArray dimensions to determine variable shape.
        This override ensures at least one bound has all target dimensions when coords
        is provided, allowing internal data to remain compact (scalars, 1D arrays).
        """
        # Binary variables cannot have bounds in linopy
        if binary:
            return super().add_variables(coords=coords, binary=True, **kwargs)

        # Apply default bounds for non-binary variables
        if lower is None:
            lower = -np.inf
        if upper is None:
            upper = np.inf

        if coords is not None:
            lower = _ensure_coords(lower, coords)
            upper = _ensure_coords(upper, coords)
        return super().add_variables(lower=lower, upper=upper, coords=coords, **kwargs)

    def _populate_element_variable_names(self):
        """Populate _variable_names and _constraint_names on each Element from type-level models."""
        # Use type-level models to populate variable/constraint names for each element
        self._populate_names_from_type_level_models()

    def _populate_names_from_type_level_models(self):
        """Populate element variable/constraint names in FlowSystem registry."""
        var_names = self.flow_system._element_variable_names
        con_names = self.flow_system._element_constraint_names

        # Helper to find batched variables that contain a specific element ID in a dimension
        def _find_vars_for_element(element_id: str, dim_name: str) -> list[str]:
            """Find all batched variable names that have this element in their dimension.

            Returns the batched variable names (e.g., 'flow|rate', 'storage|charge').
            """
            result = []
            for var_name in self.variables:
                var = self.variables[var_name]
                if dim_name in var.dims:
                    try:
                        if element_id in var.coords[dim_name].values:
                            result.append(var_name)
                    except (KeyError, AttributeError):
                        pass
            return result

        def _find_constraints_for_element(element_id: str, dim_name: str) -> list[str]:
            """Find all constraint names that have this element in their dimension."""
            result = []
            for con_name in self.constraints:
                con = self.constraints[con_name]
                if dim_name in con.dims:
                    try:
                        if element_id in con.coords[dim_name].values:
                            result.append(con_name)
                    except (KeyError, AttributeError):
                        pass
                # Also check for element-specific constraints (e.g., bus|BusLabel|balance)
                elif element_id in con_name.split('|'):
                    result.append(con_name)
            return result

        # Populate flows
        for flow in self.flow_system.flows.values():
            var_names[flow.id] = _find_vars_for_element(flow.id, 'flow')
            con_names[flow.id] = _find_constraints_for_element(flow.id, 'flow')

        # Populate buses
        for bus in self.flow_system.buses.values():
            var_names[bus.id] = _find_vars_for_element(bus.id, 'bus')
            con_names[bus.id] = _find_constraints_for_element(bus.id, 'bus')

        # Populate storages
        from .components import Storage

        for comp in self.flow_system.components.values():
            if isinstance(comp, Storage):
                comp_vars = _find_vars_for_element(comp.id, 'storage')
                comp_cons = _find_constraints_for_element(comp.id, 'storage')
                # Also add flow variables (storages have charging/discharging flows)
                for flow in comp.flows.values():
                    comp_vars.extend(var_names[flow.id])
                    comp_cons.extend(con_names[flow.id])
                var_names[comp.id] = comp_vars
                con_names[comp.id] = comp_cons
            else:
                # Generic component - collect from child flows
                comp_vars = []
                comp_cons = []
                # Add component-level variables (status, etc.)
                comp_vars.extend(_find_vars_for_element(comp.id, 'component'))
                comp_cons.extend(_find_constraints_for_element(comp.id, 'component'))
                # Add flow variables
                for flow in comp.flows.values():
                    comp_vars.extend(var_names[flow.id])
                    comp_cons.extend(con_names[flow.id])
                var_names[comp.id] = comp_vars
                con_names[comp.id] = comp_cons

        # Populate effects
        for effect in self.flow_system.effects.values():
            var_names[effect.id] = _find_vars_for_element(effect.id, 'effect')
            con_names[effect.id] = _find_constraints_for_element(effect.id, 'effect')

    def _build_results_structure(self) -> dict[str, dict]:
        """Build results structure for all elements using type-level models."""
        var_names = self.flow_system._element_variable_names
        con_names = self.flow_system._element_constraint_names

        results = {
            'Components': {},
            'Buses': {},
            'Effects': {},
            'Flows': {},
        }

        # Components
        for comp in sorted(self.flow_system.components.values(), key=lambda c: c.id.upper()):
            flow_ids = [f.id for f in comp.flows.values()]
            results['Components'][comp.id] = {
                'id': comp.id,
                'variables': var_names.get(comp.id, []),
                'constraints': con_names.get(comp.id, []),
                'inputs': ['flow|rate'] * len(comp.inputs),
                'outputs': ['flow|rate'] * len(comp.outputs),
                'flows': flow_ids,
            }

        # Buses
        for bus in sorted(self.flow_system.buses.values(), key=lambda b: b.id.upper()):
            input_vars = ['flow|rate'] * len(bus.inputs)
            output_vars = ['flow|rate'] * len(bus.outputs)
            if bus.allows_imbalance:
                input_vars.append('bus|virtual_supply')
                output_vars.append('bus|virtual_demand')
            results['Buses'][bus.id] = {
                'id': bus.id,
                'variables': var_names.get(bus.id, []),
                'constraints': con_names.get(bus.id, []),
                'inputs': input_vars,
                'outputs': output_vars,
                'flows': [f.id for f in bus.flows.values()],
            }

        # Effects
        for effect in sorted(self.flow_system.effects.values(), key=lambda e: e.id.upper()):
            results['Effects'][effect.id] = {
                'id': effect.id,
                'variables': var_names.get(effect.id, []),
                'constraints': con_names.get(effect.id, []),
            }

        # Flows
        for flow in sorted(self.flow_system.flows.values(), key=lambda f: f.id.upper()):
            results['Flows'][flow.id] = {
                'id': flow.id,
                'variables': var_names.get(flow.id, []),
                'constraints': con_names.get(flow.id, []),
                'start': flow.bus if flow.is_input_in_component else flow.component,
                'end': flow.component if flow.is_input_in_component else flow.bus,
                'component': flow.component,
            }

        return results

    def build_model(self, timing: bool = False):
        """Build the model using type-level models (one model per element TYPE).

        Uses TypeModel classes (e.g., FlowsModel, BusesModel) which handle ALL
        elements of a type in a single instance with true vectorized operations.

        Args:
            timing: If True, print detailed timing breakdown.
        """
        from .components import InterclusterStoragesModel, StoragesModel
        from .effects import EffectsModel
        from .elements import (
            BusesModel,
            ComponentsModel,
            ConvertersModel,
            FlowsModel,
            TransmissionsModel,
        )

        timer = _BuildTimer() if timing else None

        # Use cached *Data from BatchedAccessor (same instances used for validation)
        batched = self.flow_system.batched

        self.effects = EffectsModel(self, batched.effects)
        if timer:
            timer.record('effects')

        self._flows_model = FlowsModel(self, batched.flows)
        if timer:
            timer.record('flows')

        self._buses_model = BusesModel(self, batched.buses, self._flows_model)
        if timer:
            timer.record('buses')

        self._storages_model = StoragesModel(self, batched.storages, self._flows_model)
        if timer:
            timer.record('storages')

        self._intercluster_storages_model = InterclusterStoragesModel(
            self, batched.intercluster_storages, self._flows_model
        )
        if timer:
            timer.record('intercluster_storages')

        self._components_model = ComponentsModel(self, batched.components, self._flows_model)
        if timer:
            timer.record('components')

        self._converters_model = ConvertersModel(self, batched.converters, self._flows_model)
        if timer:
            timer.record('converters')

        self._transmissions_model = TransmissionsModel(self, batched.transmissions, self._flows_model)
        if timer:
            timer.record('transmissions')

        self._add_scenario_equality_constraints()
        self._populate_element_variable_names()
        self.effects.finalize_shares()

        if timer:
            timer.record('finalize')
        if timer:
            timer.print_summary()

        self._is_built = True

        logger.info(
            f'Type-level modeling complete: {len(self.variables)} variables, {len(self.constraints)} constraints'
        )

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

        # Map parameter types to batched variable names
        batched_var_map = {'flow_rate': 'flow|rate', 'size': 'flow|size'}
        batched_var_name = batched_var_map[parameter_type]

        if batched_var_name not in self.variables:
            return  # Variable doesn't exist (e.g., no flows with investment)

        batched_var = self.variables[batched_var_name]
        if 'scenario' not in batched_var.dims:
            return  # No scenario dimension, nothing to equalize

        all_flow_ids = list(batched_var.coords['flow'].values)

        if config is True:
            # All flows should be scenario-independent
            flows_to_constrain = all_flow_ids
        else:
            # Only those in the list should be scenario-independent
            flows_to_constrain = [f for f in config if f in all_flow_ids]
            # Validate that all specified flows exist
            missing = [f for f in config if f not in all_flow_ids]
            if missing:
                param_name = (
                    'scenario_independent_sizes' if parameter_type == 'size' else 'scenario_independent_flow_rates'
                )
                logger.warning(f'{param_name} contains ids not in {batched_var_name}: {missing}')

        logger.debug(f'Adding scenario equality constraints for {len(flows_to_constrain)} {parameter_type} variables')
        for flow_id in flows_to_constrain:
            var_slice = batched_var.sel(flow=flow_id)
            self.add_constraints(
                var_slice.isel(scenario=0) == var_slice.isel(scenario=slice(1, None)),
                name=f'{flow_id}|{parameter_type}|scenario_independent',
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
        # Use _build_results_structure to build from type-level models
        results_structure = self._build_results_structure()
        solution.attrs = {
            'Components': json.dumps(results_structure['Components']),
            'Buses': json.dumps(results_structure['Buses']),
            'Effects': json.dumps(results_structure['Effects']),
            'Flows': json.dumps(results_structure['Flows']),
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
        obj_effect = self.flow_system.effects.objective_effect
        # Compute period_weights directly from effect
        effect_weights = obj_effect.period_weights
        default_weights = self.flow_system.period_weights
        if effect_weights is not None:
            period_weights = effect_weights
        elif default_weights is not None:
            period_weights = default_weights
        else:
            period_weights = align_to_coords(1, self.flow_system.indexes, name='period_weights', dims=['period'])

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
            'Status': self.status,
        }

        # Format sections with headers and underlines
        formatted_sections = fx_io.format_sections_with_headers(sections)

        title = f'FlowSystemModel ({self.type})'
        all_sections = '\n'.join(formatted_sections)

        return f'{title}\n{"=" * len(title)}\n\n{all_sections}'


def valid_id(id: str) -> str:
    """Check if the id is valid and return it (possibly stripped).

    Raises:
        ValueError: If the id contains forbidden characters.
    """
    not_allowed = ['(', ')', '|', '->', '\\', '-slash-']  # \\ is needed to check for \
    if any([sign in id for sign in not_allowed]):
        raise ValueError(
            f'Id "{id}" is not valid. Ids cannot contain the following characters: {not_allowed}. '
            f'Use any other symbol instead'
        )
    if id.endswith(' '):
        logger.error(f'Id "{id}" ends with a space. This will be removed.')
        return id.rstrip()
    return id


class Element:
    """Mixin for all elements in flixopt. Provides deprecated label properties.

    Subclasses (Effect, Bus, Flow, Component) are @dataclass classes that declare
    their own ``id`` field. Element does NOT define ``id`` — each subclass owns it.

    Runtime state (variable names, constraint names) is stored in FlowSystem registries,
    not on the element objects themselves.
    """

    @property
    def label(self) -> str:
        """Deprecated: Use ``id`` instead."""
        warnings.warn(
            f'Accessing ".label" is deprecated. Use ".id" instead. Will be removed in v{DEPRECATION_REMOVAL_VERSION}.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.id

    @label.setter
    def label(self, value: str) -> None:
        warnings.warn(
            f'Setting ".label" is deprecated. Use ".id" instead. Will be removed in v{DEPRECATION_REMOVAL_VERSION}.',
            DeprecationWarning,
            stacklevel=2,
        )
        self.id = value

    @property
    def label_full(self) -> str:
        """Deprecated: Use ``id`` instead."""
        warnings.warn(
            f'Accessing ".label_full" is deprecated. Use ".id" instead. Will be removed in v{DEPRECATION_REMOVAL_VERSION}.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.id

    @property
    def id_full(self) -> str:
        """Deprecated: Use ``id`` instead."""
        warnings.warn(
            f'Accessing ".id_full" is deprecated. Use ".id" instead. Will be removed in v{DEPRECATION_REMOVAL_VERSION}.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.id


# Precompiled regex pattern for natural sorting
_NATURAL_SPLIT = re.compile(r'(\d+)')


def _natural_sort_key(text):
    """Sort key for natural ordering (e.g., bus1, bus2, bus10 instead of bus1, bus10, bus2)."""
    return [int(c) if c.isdigit() else c.lower() for c in _NATURAL_SPLIT.split(text)]


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

    def _get_container_groups(self) -> dict[str, IdList[Any]]:
        """
        Return ordered dict of container groups to aggregate.

        Returns:
            Dictionary mapping group names to IdList containers.
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
        all_keys: list[str] = []
        for container in self._get_container_groups().values():
            all_keys.extend(container.keys())

        suggestions = get_close_matches(key, all_keys, n=3, cutoff=0.6)
        error_msg = f'Element "{key}" not found.'

        if suggestions:
            error_msg += f' Did you mean: {", ".join(suggestions)}?'
        else:
            available = all_keys
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
