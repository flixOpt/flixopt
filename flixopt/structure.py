"""
This module contains the core structure of the flixopt framework.
These classes are not directly used by the end user, but are used by other modules.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import pathlib
import re
import warnings
from abc import ABC, abstractmethod
from difflib import get_close_matches
from enum import Enum
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
        """Populate element variable/constraint names from type-level models."""

        # Helper to find batched variables that contain a specific element ID in a dimension
        def _find_vars_for_element(element_id: str, dim_name: str) -> list[str]:
            """Find all batched variable names that have this element in their dimension.

            Returns the batched variable names (e.g., 'flow|rate', 'storage|charge').
            """
            var_names = []
            for var_name in self.variables:
                var = self.variables[var_name]
                if dim_name in var.dims:
                    try:
                        if element_id in var.coords[dim_name].values:
                            var_names.append(var_name)
                    except (KeyError, AttributeError):
                        pass
            return var_names

        def _find_constraints_for_element(element_id: str, dim_name: str) -> list[str]:
            """Find all constraint names that have this element in their dimension."""
            con_names = []
            for con_name in self.constraints:
                con = self.constraints[con_name]
                if dim_name in con.dims:
                    try:
                        if element_id in con.coords[dim_name].values:
                            con_names.append(con_name)
                    except (KeyError, AttributeError):
                        pass
                # Also check for element-specific constraints (e.g., bus|BusLabel|balance)
                elif element_id in con_name.split('|'):
                    con_names.append(con_name)
            return con_names

        # Populate flows
        for flow in self.flow_system.flows.values():
            flow._variable_names = _find_vars_for_element(flow.id, 'flow')
            flow._constraint_names = _find_constraints_for_element(flow.id, 'flow')

        # Populate buses
        for bus in self.flow_system.buses.values():
            bus._variable_names = _find_vars_for_element(bus.id, 'bus')
            bus._constraint_names = _find_constraints_for_element(bus.id, 'bus')

        # Populate storages
        from .components import Storage

        for comp in self.flow_system.components.values():
            if isinstance(comp, Storage):
                comp._variable_names = _find_vars_for_element(comp.id, 'storage')
                comp._constraint_names = _find_constraints_for_element(comp.id, 'storage')
                # Also add flow variables (storages have charging/discharging flows)
                for flow in comp.flows.values():
                    comp._variable_names.extend(flow._variable_names)
                    comp._constraint_names.extend(flow._constraint_names)
            else:
                # Generic component - collect from child flows
                comp._variable_names = []
                comp._constraint_names = []
                # Add component-level variables (status, etc.)
                comp._variable_names.extend(_find_vars_for_element(comp.id, 'component'))
                comp._constraint_names.extend(_find_constraints_for_element(comp.id, 'component'))
                # Add flow variables
                for flow in comp.flows.values():
                    comp._variable_names.extend(flow._variable_names)
                    comp._constraint_names.extend(flow._constraint_names)

        # Populate effects
        for effect in self.flow_system.effects.values():
            effect._variable_names = _find_vars_for_element(effect.id, 'effect')
            effect._constraint_names = _find_constraints_for_element(effect.id, 'effect')

    def _build_results_structure(self) -> dict[str, dict]:
        """Build results structure for all elements using type-level models."""

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
                'variables': comp._variable_names,
                'constraints': comp._constraint_names,
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
                'variables': bus._variable_names,
                'constraints': bus._constraint_names,
                'inputs': input_vars,
                'outputs': output_vars,
                'flows': [f.id for f in bus.flows.values()],
            }

        # Effects
        for effect in sorted(self.flow_system.effects.values(), key=lambda e: e.id.upper()):
            results['Effects'][effect.id] = {
                'id': effect.id,
                'variables': effect._variable_names,
                'constraints': effect._constraint_names,
            }

        # Flows
        for flow in sorted(self.flow_system.flows.values(), key=lambda f: f.id.upper()):
            results['Flows'][flow.id] = {
                'id': flow.id,
                'variables': flow._variable_names,
                'constraints': flow._constraint_names,
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
    """

    # Class-level defaults for attributes set by link_to_flow_system()
    # These provide type hints and default values without requiring __init__ in subclasses
    _flow_system: FlowSystem | None = None
    _prefix: str = ''

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
                flow.status_parameters.link_to_flow_system(self._model.flow_system, f'{flow.id}')
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

        # Deprecated init params that should not be serialized (they alias other params)
        _deprecated_init_params = {'label', 'label_as_positional'}
        # On Flow, 'id' is deprecated in favor of 'flow_id'
        if 'flow_id' in self._cached_init_params:
            _deprecated_init_params.add('id')

        for name in self._cached_init_params:
            if name == 'self' or name in _deprecated_init_params:
                continue

            # For 'id' or 'flow_id' param, use _short_id to get the raw constructor value
            # (Flow.id property returns qualified name, but constructor expects short name)
            if name in ('id', 'flow_id') and hasattr(self, '_short_id'):
                value = self._short_id
            else:
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
                # Use context name as fallback (e.g. attribute path) if no explicit name
                obj = obj.rename(context_name)

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

        # Handle plain dataclasses (not Interface) - serialize via fields
        elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            structure = {'__class__': obj.__class__.__name__}
            arrays = {}
            for field in dataclasses.fields(obj):
                value = getattr(obj, field.name)
                if value is None:
                    continue
                field_context = f'{context_name}.{field.name}' if context_name else field.name
                processed, field_arrays = self._extract_dataarrays_recursive(value, field_context)
                if processed is not None and not self._is_empty_container(processed):
                    structure[field.name] = processed
                arrays.update(field_arrays)
            extracted_arrays.update(arrays)
            return structure, extracted_arrays

        # Handle sequences (lists, tuples)
        elif isinstance(obj, (list, tuple)):
            processed_items = []
            for i, item in enumerate(obj):
                item_context = f'{context_name}[{i}]' if context_name else f'item[{i}]'
                processed_item, nested_arrays = self._extract_dataarrays_recursive(item, item_context)
                extracted_arrays.update(nested_arrays)
                processed_items.append(processed_item)
            return processed_items, extracted_arrays

        # Handle IdList containers (treat as dict for serialization)
        elif isinstance(obj, IdList):
            processed_dict = {}
            for key, value in obj.items():
                key_context = f'{context_name}.{key}' if context_name else str(key)
                processed_value, nested_arrays = self._extract_dataarrays_recursive(value, key_context)
                extracted_arrays.update(nested_arrays)
                processed_dict[key] = processed_value
            return processed_dict, extracted_arrays

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

        # Handle null values with warning (use numpy for performance - 200x faster than xarray)
        has_nulls = (np.issubdtype(array.dtype, np.floating) and np.any(np.isnan(array.values))) or (
            array.dtype == object and pd.isna(array.values).any()
        )
        if has_nulls:
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

                    # Handle renamed parameters from old serialized data
                    if 'label' in constructor_data and 'label' not in init_params:
                        # label → id for most elements, label → flow_id for Flow
                        new_key = 'flow_id' if 'flow_id' in init_params else 'id'
                        constructor_data[new_key] = constructor_data.pop('label')
                    if 'id' in constructor_data and 'id' not in init_params and 'flow_id' in init_params:
                        # id → flow_id for Flow (from recently serialized data)
                        constructor_data['flow_id'] = constructor_data.pop('id')

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
            # Use ds.variables with coord_cache for faster DataArray construction
            variables = ds.variables
            coord_cache = {k: ds.coords[k] for k in ds.coords}
            arrays_dict = {
                name: xr.DataArray(
                    variables[name],
                    coords={k: coord_cache[k] for k in variables[name].dims if k in coord_cache},
                    name=name,
                )
                for name in ds.data_vars
            }

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
        return fx_io.build_repr_from_init(self, excluded_params={'self', 'id', 'label', 'kwargs'})

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
    """This class is the basic Element of flixopt. Every Element has an id."""

    # Attributes that are serialized but set after construction (not passed to child __init__)
    # These are internal state populated during modeling, not user-facing parameters
    _deferred_init_attrs: ClassVar[set[str]] = {'_variable_names', '_constraint_names'}

    def __init__(
        self,
        id: str | None = None,
        meta_data: dict | None = None,
        color: str | None = None,
        _variable_names: list[str] | None = None,
        _constraint_names: list[str] | None = None,
        **kwargs,
    ):
        """
        Args:
            id: The id of the element
            meta_data: used to store more information about the Element. Is not used internally, but saved in the results. Only use python native types.
            color: Optional color for visualizations (e.g., '#FF6B6B'). If not provided, a color will be automatically assigned during FlowSystem.connect_and_transform().
            _variable_names: Internal. Variable names for this element (populated after modeling).
            _constraint_names: Internal. Constraint names for this element (populated after modeling).
        """
        id = self._handle_deprecated_kwarg(kwargs, 'label', 'id', id)
        if id is None:
            raise TypeError(f'{self.__class__.__name__}.__init__() requires an "id" argument.')
        self._validate_kwargs(kwargs)
        self._short_id: str = Element._valid_id(id)
        self.meta_data = meta_data if meta_data is not None else {}
        self.color = color
        self._flow_system: FlowSystem | None = None
        # Variable/constraint names - populated after modeling, serialized for results
        self._variable_names: list[str] = _variable_names if _variable_names is not None else []
        self._constraint_names: list[str] = _constraint_names if _constraint_names is not None else []

    @property
    def id(self) -> str:
        """The unique identifier of this element.

        For most elements this is the name passed to the constructor.
        For flows this returns the qualified form: ``component(short_id)``.
        """
        return self._short_id

    @id.setter
    def id(self, value: str) -> None:
        self._short_id = value

    @property
    def label(self) -> str:
        """Deprecated: Use ``id`` instead."""
        warnings.warn(
            f'Accessing ".label" is deprecated. Use ".id" instead. Will be removed in v{DEPRECATION_REMOVAL_VERSION}.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self._short_id

    @label.setter
    def label(self, value: str) -> None:
        warnings.warn(
            f'Setting ".label" is deprecated. Use ".id" instead. Will be removed in v{DEPRECATION_REMOVAL_VERSION}.',
            DeprecationWarning,
            stacklevel=2,
        )
        self._short_id = value

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

    @property
    def solution(self) -> xr.Dataset:
        """Solution data for this element's variables.

        Returns a Dataset built by selecting this element from batched variables
        in FlowSystem.solution.

        Raises:
            ValueError: If no solution is available (optimization not run or not solved).
        """
        if self._flow_system is None:
            raise ValueError(f'Element "{self.id}" is not linked to a FlowSystem.')
        if self._flow_system.solution is None:
            raise ValueError(f'No solution available for "{self.id}". Run optimization first or load results.')
        if not self._variable_names:
            raise ValueError(f'No variable names available for "{self.id}". Element may not have been modeled yet.')
        full_solution = self._flow_system.solution
        data_vars = {}
        for var_name in self._variable_names:
            if var_name not in full_solution:
                continue
            var = full_solution[var_name]
            # Select this element from the appropriate dimension
            for dim in var.dims:
                if dim in ('time', 'period', 'scenario', 'cluster'):
                    continue
                if self.id in var.coords[dim].values:
                    var = var.sel({dim: self.id}, drop=True)
                    break
            data_vars[var_name] = var
        return xr.Dataset(data_vars)

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
        return fx_io.build_repr_from_init(self, excluded_params={'self', 'id', 'kwargs'}, skip_default_size=True)

    @staticmethod
    def _valid_id(id: str) -> str:
        """Checks if the id is valid.

        Raises:
            ValueError: If the id is not valid.
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

    @staticmethod
    def _valid_label(label: str) -> str:
        """Deprecated: Use ``_valid_id`` instead."""
        warnings.warn(
            f'_valid_label is deprecated. Use _valid_id instead. Will be removed in v{DEPRECATION_REMOVAL_VERSION}.',
            DeprecationWarning,
            stacklevel=2,
        )
        return Element._valid_id(label)


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

    def __add__(self, other: ContainerMixin[T]) -> ContainerMixin[T]:
        """Concatenate two containers."""
        result = self.__class__(element_type_name=self._element_type_name)
        for element in self.values():
            result.add(element)
        for element in other.values():
            result.add(element)
        return result

    def __repr__(self) -> str:
        """Return a string representation using the instance's truncate_repr setting."""
        return self._get_repr()


class FlowContainer(ContainerMixin[T]):
    """Container for Flow objects with dual access: by index or by id.

    Supports:
        - container['Boiler(Q_th)']  # id-based access
        - container['Q_th']          # short-id access (when all flows share same component)
        - container[0]               # index-based access
        - container.add(flow)
        - for flow in container.values()
        - container1 + container2    # concatenation

    Examples:
        >>> boiler = Boiler(id='Boiler', inputs=[Flow('heat_bus')])
        >>> boiler.inputs[0]  # Index access
        >>> boiler.inputs['Boiler(heat_bus)']  # Full id access
        >>> boiler.inputs['heat_bus']  # Short id access (same component)
        >>> for flow in boiler.inputs.values():
        ...     print(flow.id)
    """

    def _get_label(self, flow: T) -> str:
        """Extract id from Flow."""
        return flow.id

    def __getitem__(self, key: str | int) -> T:
        """Get flow by id, short id, or index."""
        if isinstance(key, int):
            try:
                return list(self.values())[key]
            except IndexError:
                raise IndexError(f'Flow index {key} out of range (container has {len(self)} flows)') from None

        if dict.__contains__(self, key):
            return super().__getitem__(key)

        # Try short-id match if all flows share the same component
        if len(self) > 0:
            components = {flow.component for flow in self.values()}
            if len(components) == 1:
                component = next(iter(components))
                full_key = f'{component}({key})'
                if dict.__contains__(self, full_key):
                    return super().__getitem__(full_key)

        raise KeyError(f"'{key}' not found in {self._element_type_name}")

    def __contains__(self, key: object) -> bool:
        """Check if key exists (supports id or short id)."""
        if not isinstance(key, str):
            return False
        if dict.__contains__(self, key):
            return True
        if len(self) > 0:
            components = {flow.component for flow in self.values()}
            if len(components) == 1:
                component = next(iter(components))
                full_key = f'{component}({key})'
                return dict.__contains__(self, full_key)
        return False


class ElementContainer(ContainerMixin[T]):
    """
    Container for Element objects (Component, Bus, Flow, Effect).

    Uses element.id for keying.
    """

    def _get_label(self, element: T) -> str:
        """Extract id from Element."""
        return element.id


class ResultsContainer(ContainerMixin[T]):
    """
    Container for Results objects (ComponentResults, BusResults, etc).

    Uses element.id for keying.
    """

    def _get_label(self, element: T) -> str:
        """Extract id from Results object."""
        return element.id


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
