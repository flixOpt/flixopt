"""
Vectorized modeling infrastructure for flixopt.

This module implements the Declaration-Collection-Execution (DCE) pattern
for efficient batch creation of variables and constraints across many elements.

Key concepts:
- VariableSpec: Immutable specification of a variable an element needs
- ConstraintSpec: Specification of a constraint with deferred evaluation
- VariableRegistry: Collects specs and batch-creates variables
- ConstraintRegistry: Collects specs and batch-creates constraints
- VariableHandle: Provides element access to their slice of batched variables

Usage:
    Elements declare what they need via `declare_variables()` and `declare_constraints()`.
    The FlowSystemModel collects all declarations, then batch-creates them.
    Elements receive handles to access their variables.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Callable

    import linopy

    from .structure import FlowSystemModel, VariableCategory

logger = logging.getLogger('flixopt')


# =============================================================================
# Specifications (Declaration Phase)
# =============================================================================


@dataclass(frozen=True)
class VariableSpec:
    """Immutable specification of a variable an element needs.

    This is a declaration - no linopy calls are made when creating a VariableSpec.
    The spec is later collected by a VariableRegistry and batch-created with
    other specs of the same category.

    Attributes:
        category: Variable category for grouping (e.g., 'flow_rate', 'status').
            All specs with the same category are batch-created together.
        element_id: Unique identifier of the element (e.g., 'Boiler(Q_th)').
            Used as a coordinate in the batched variable.
        lower: Lower bound (scalar, array, or DataArray). Default -inf.
        upper: Upper bound (scalar, array, or DataArray). Default +inf.
        integer: If True, variable is integer-valued.
        binary: If True, variable is binary (0 or 1).
        dims: Dimensions this variable spans beyond 'element'.
            Common values: ('time',), ('time', 'scenario'), ('period', 'scenario'), ().
        mask: Optional mask for sparse creation (True = create, False = skip).
        var_category: VariableCategory enum for segment expansion handling.

    Example:
        >>> spec = VariableSpec(
        ...     category='flow_rate',
        ...     element_id='Boiler(Q_th)',
        ...     lower=0,
        ...     upper=100,
        ...     dims=('time', 'scenario'),
        ... )
    """

    category: str
    element_id: str
    lower: float | xr.DataArray = -np.inf
    upper: float | xr.DataArray = np.inf
    integer: bool = False
    binary: bool = False
    dims: tuple[str, ...] = ('time',)
    mask: xr.DataArray | None = None
    var_category: VariableCategory | None = None


@dataclass
class ConstraintSpec:
    """Specification of a constraint with deferred evaluation.

    The constraint expression is not built until variables exist. A build
    function is provided that will be called during the execution phase.

    Attributes:
        category: Constraint category for grouping (e.g., 'flow_rate_bounds').
        element_id: Unique identifier of the element.
        build_fn: Callable that builds the constraint. Called as:
            build_fn(model, handles) -> ConstraintResult
            where handles is a dict mapping category -> VariableHandle.
        sense: Constraint sense ('==', '<=', '>=').
        mask: Optional mask for sparse creation.

    Example:
        >>> def build_flow_bounds(model, handles):
        ...     flow_rate = handles['flow_rate'].variable
        ...     return ConstraintResult(
        ...         lhs=flow_rate,
        ...         rhs=100,
        ...         sense='<=',
        ...     )
        >>> spec = ConstraintSpec(
        ...     category='flow_rate_upper',
        ...     element_id='Boiler(Q_th)',
        ...     build_fn=build_flow_bounds,
        ... )
    """

    category: str
    element_id: str
    build_fn: Callable[[FlowSystemModel, dict[str, VariableHandle]], ConstraintResult]
    mask: xr.DataArray | None = None


@dataclass
class ConstraintResult:
    """Result of a constraint build function.

    Attributes:
        lhs: Left-hand side expression (linopy Variable or LinearExpression).
        rhs: Right-hand side (expression, scalar, or DataArray).
        sense: Constraint sense ('==', '<=', '>=').
    """

    lhs: linopy.Variable | linopy.expressions.LinearExpression | xr.DataArray
    rhs: linopy.Variable | linopy.expressions.LinearExpression | xr.DataArray | float
    sense: Literal['==', '<=', '>='] = '=='


@dataclass
class EffectShareSpec:
    """Specification of an effect share for batch creation.

    Effect shares link flow rates to effects (costs, emissions, etc.).
    Instead of creating them one at a time, we collect specs and batch-create.

    Attributes:
        element_id: The flow's unique identifier (e.g., 'Boiler(gas_in)').
        effect_name: The effect to add to (e.g., 'costs', 'CO2').
        factor: Multiplier for flow_rate * timestep_duration.
        target: 'temporal' for time-varying or 'periodic' for period totals.
    """

    element_id: str
    effect_name: str
    factor: float | xr.DataArray
    target: Literal['temporal', 'periodic'] = 'temporal'


# =============================================================================
# Variable Handle (Element Access)
# =============================================================================


@dataclass
class VariableHandle:
    """Handle providing element access to a batched variable.

    When variables are batch-created across elements, each element needs
    a way to access its slice. The handle stores a reference to the
    element's portion of the batched variable.

    Attributes:
        variable: The element's slice of the batched variable.
            This is typically `batched_var.sel(element=element_id)`.
        category: The variable category this handle is for.
        element_id: The element this handle belongs to.
        full_variable: Optional reference to the full batched variable.

    Example:
        >>> handle = registry.get_handle('flow_rate', 'Boiler(Q_th)')
        >>> flow_rate = handle.variable  # Access the variable
        >>> total = flow_rate.sum('time')  # Use in expressions
    """

    variable: linopy.Variable
    category: str
    element_id: str
    full_variable: linopy.Variable | None = None

    def __repr__(self) -> str:
        dims = list(self.variable.dims) if hasattr(self.variable, 'dims') else []
        return f"VariableHandle(category='{self.category}', element='{self.element_id}', dims={dims})"


# =============================================================================
# Variable Registry (Collection & Execution)
# =============================================================================


class VariableRegistry:
    """Collects variable specifications and batch-creates them.

    The registry implements the Collection and Execution phases of DCE:
    1. Elements register their VariableSpecs via `register()`
    2. `create_all()` groups specs by category and batch-creates them
    3. Elements retrieve handles via `get_handle()`

    Variables are created with an 'element' dimension containing all element IDs
    for that category. Each element then gets a handle to its slice.

    Attributes:
        model: The FlowSystemModel to create variables in.

    Example:
        >>> registry = VariableRegistry(model)
        >>> registry.register(VariableSpec(category='flow_rate', element_id='Boiler', ...))
        >>> registry.register(VariableSpec(category='flow_rate', element_id='CHP', ...))
        >>> registry.create_all()  # Creates one variable with element=['Boiler', 'CHP']
        >>> handle = registry.get_handle('flow_rate', 'Boiler')
    """

    def __init__(self, model: FlowSystemModel):
        self.model = model
        self._specs_by_category: dict[str, list[VariableSpec]] = defaultdict(list)
        self._handles: dict[str, dict[str, VariableHandle]] = {}  # category -> element_id -> handle
        self._full_variables: dict[str, linopy.Variable] = {}  # category -> full batched variable
        self._created = False

    def register(self, spec: VariableSpec) -> None:
        """Register a variable specification for batch creation.

        Args:
            spec: The variable specification to register.

        Raises:
            RuntimeError: If variables have already been created.
            ValueError: If element_id is already registered for this category.
        """
        if self._created:
            raise RuntimeError('Cannot register specs after variables have been created')

        # Check for duplicate element_id in same category
        existing_ids = {s.element_id for s in self._specs_by_category[spec.category]}
        if spec.element_id in existing_ids:
            raise ValueError(f"Element '{spec.element_id}' already registered for category '{spec.category}'")

        self._specs_by_category[spec.category].append(spec)

    def create_all(self) -> None:
        """Batch-create all registered variables.

        Groups specs by category and creates one linopy variable per category
        with an 'element' dimension. Creates handles for each element.

        Raises:
            RuntimeError: If already called.
        """
        if self._created:
            raise RuntimeError('Variables have already been created')

        for category, specs in self._specs_by_category.items():
            if specs:
                self._create_batch(category, specs)

        self._created = True
        logger.debug(
            f'VariableRegistry created {len(self._full_variables)} batched variables '
            f'for {sum(len(h) for h in self._handles.values())} elements'
        )

    def _create_batch(self, category: str, specs: list[VariableSpec]) -> None:
        """Create all variables of a category in one linopy call.

        Args:
            category: The variable category name.
            specs: List of specs for this category.
        """
        if not specs:
            return

        # Extract element IDs and verify homogeneity
        element_ids = [s.element_id for s in specs]
        reference_spec = specs[0]

        # Verify all specs have same dims, binary, integer flags
        for spec in specs[1:]:
            if spec.dims != reference_spec.dims:
                raise ValueError(
                    f"Inconsistent dims in category '{category}': "
                    f"'{spec.element_id}' has {spec.dims}, "
                    f"'{reference_spec.element_id}' has {reference_spec.dims}"
                )
            if spec.binary != reference_spec.binary:
                raise ValueError(f"Inconsistent binary flag in category '{category}'")
            if spec.integer != reference_spec.integer:
                raise ValueError(f"Inconsistent integer flag in category '{category}'")

        # Build coordinates: element + model dimensions
        coords = self._build_coords(element_ids, reference_spec.dims)

        # Stack bounds into arrays with element dimension
        # Note: Binary variables cannot have explicit bounds in linopy
        if reference_spec.binary:
            lower = None
            upper = None
        else:
            lower = self._stack_bounds([s.lower for s in specs], element_ids, reference_spec.dims)
            upper = self._stack_bounds([s.upper for s in specs], element_ids, reference_spec.dims)

        # Combine masks if any
        mask = self._combine_masks(specs, element_ids, reference_spec.dims)

        # Build kwargs, only including bounds for non-binary variables
        kwargs = {
            'coords': coords,
            'name': category,
            'binary': reference_spec.binary,
            'integer': reference_spec.integer,
            'mask': mask,
        }
        if lower is not None:
            kwargs['lower'] = lower
        if upper is not None:
            kwargs['upper'] = upper

        # Single linopy call for all elements!
        variable = self.model.add_variables(**kwargs)

        # Register category if specified
        if reference_spec.var_category is not None:
            self.model.variable_categories[variable.name] = reference_spec.var_category

        # Store full variable
        self._full_variables[category] = variable

        # Create handles for each element
        self._handles[category] = {}
        for spec in specs:
            element_slice = variable.sel(element=spec.element_id)
            handle = VariableHandle(
                variable=element_slice,
                category=category,
                element_id=spec.element_id,
                full_variable=variable,
            )
            self._handles[category][spec.element_id] = handle

    def _build_coords(self, element_ids: list[str], dims: tuple[str, ...]) -> xr.Coordinates:
        """Build coordinate dict with element dimension + model dimensions.

        Args:
            element_ids: List of element identifiers.
            dims: Tuple of dimension names from the model.

        Returns:
            xarray Coordinates with 'element' + requested dims.
        """
        # Start with element dimension
        coord_dict = {'element': pd.Index(element_ids, name='element')}

        # Add model dimensions
        model_coords = self.model.get_coords(dims=dims)
        if model_coords is not None:
            for dim in dims:
                if dim in model_coords:
                    coord_dict[dim] = model_coords[dim]

        return xr.Coordinates(coord_dict)

    def _stack_bounds(
        self,
        bounds: list[float | xr.DataArray],
        element_ids: list[str],
        dims: tuple[str, ...],
    ) -> xr.DataArray | float:
        """Stack per-element bounds into array with element dimension.

        Args:
            bounds: List of bounds (one per element).
            element_ids: List of element identifiers.
            dims: Dimension tuple for the variable.

        Returns:
            Stacked DataArray with element dimension, or scalar if all identical.
        """
        # Extract scalar values from 0-d DataArrays or plain scalars
        scalar_values = []
        has_multidim = False

        for b in bounds:
            if isinstance(b, xr.DataArray):
                if b.ndim == 0:
                    # 0-d DataArray - extract scalar
                    scalar_values.append(float(b.values))
                else:
                    # Multi-dimensional - need full concat
                    has_multidim = True
                    break
            else:
                scalar_values.append(float(b))

        # Fast path: all scalars (including 0-d DataArrays)
        if not has_multidim:
            # Check if all identical (common case: all 0 or all inf)
            unique_values = set(scalar_values)
            if len(unique_values) == 1:
                return scalar_values[0]  # Return scalar - linopy will broadcast

            # Build array directly from scalars
            return xr.DataArray(
                np.array(scalar_values),
                coords={'element': element_ids},
                dims=['element'],
            )

        # Slow path: need full concat for multi-dimensional bounds
        arrays_to_stack = []
        for bound, eid in zip(bounds, element_ids, strict=False):
            if isinstance(bound, xr.DataArray):
                arr = bound.expand_dims(element=[eid])
            else:
                arr = xr.DataArray(bound, coords={'element': [eid]}, dims=['element'])
            arrays_to_stack.append(arr)

        stacked = xr.concat(arrays_to_stack, dim='element')

        # Ensure element is first dimension for consistency
        if 'element' in stacked.dims and stacked.dims[0] != 'element':
            dim_order = ['element'] + [d for d in stacked.dims if d != 'element']
            stacked = stacked.transpose(*dim_order)

        return stacked

    def _combine_masks(
        self,
        specs: list[VariableSpec],
        element_ids: list[str],
        dims: tuple[str, ...],
    ) -> xr.DataArray | None:
        """Combine per-element masks into a single mask array.

        Args:
            specs: List of variable specs.
            element_ids: List of element identifiers.
            dims: Dimension tuple.

        Returns:
            Combined mask DataArray, or None if no masks specified.
        """
        masks = [s.mask for s in specs]
        if all(m is None for m in masks):
            return None

        # Build mask array
        mask_arrays = []
        for mask, eid in zip(masks, element_ids, strict=False):
            if mask is None:
                # No mask = all True
                arr = xr.DataArray(True, coords={'element': [eid]}, dims=['element'])
            else:
                arr = mask.expand_dims(element=[eid])
            mask_arrays.append(arr)

        combined = xr.concat(mask_arrays, dim='element')
        return combined

    def get_handle(self, category: str, element_id: str) -> VariableHandle:
        """Get the handle for an element's variable.

        Args:
            category: Variable category.
            element_id: Element identifier.

        Returns:
            VariableHandle for the element.

        Raises:
            KeyError: If category or element_id not found.
        """
        if category not in self._handles:
            available = list(self._handles.keys())
            raise KeyError(f"Category '{category}' not found. Available: {available}")

        if element_id not in self._handles[category]:
            available = list(self._handles[category].keys())
            raise KeyError(f"Element '{element_id}' not found in category '{category}'. Available: {available}")

        return self._handles[category][element_id]

    def get_handles_for_element(self, element_id: str) -> dict[str, VariableHandle]:
        """Get all handles for a specific element.

        Args:
            element_id: Element identifier.

        Returns:
            Dict mapping category -> VariableHandle for this element.
        """
        handles = {}
        for category, element_handles in self._handles.items():
            if element_id in element_handles:
                handles[category] = element_handles[element_id]
        return handles

    def get_full_variable(self, category: str) -> linopy.Variable:
        """Get the full batched variable for a category.

        Args:
            category: Variable category.

        Returns:
            The full linopy Variable with element dimension.

        Raises:
            KeyError: If category not found.
        """
        if category not in self._full_variables:
            available = list(self._full_variables.keys())
            raise KeyError(f"Category '{category}' not found. Available: {available}")
        return self._full_variables[category]

    def get_element_ids(self, category: str) -> list[str]:
        """Get the list of element IDs for a category.

        Args:
            category: Variable category.

        Returns:
            List of element IDs in the order they appear in the batched variable.

        Raises:
            KeyError: If category not found.
        """
        if category not in self._handles:
            available = list(self._handles.keys())
            raise KeyError(f"Category '{category}' not found. Available: {available}")
        return list(self._handles[category].keys())

    @property
    def categories(self) -> list[str]:
        """List of all registered categories."""
        return list(self._specs_by_category.keys())

    @property
    def element_count(self) -> int:
        """Total number of element registrations across all categories."""
        return sum(len(specs) for specs in self._specs_by_category.values())

    def __repr__(self) -> str:
        status = 'created' if self._created else 'pending'
        return (
            f'VariableRegistry(categories={len(self._specs_by_category)}, '
            f'elements={self.element_count}, status={status})'
        )


# =============================================================================
# Constraint Registry (Collection & Execution)
# =============================================================================


class ConstraintRegistry:
    """Collects constraint specifications and batch-creates them.

    Constraints are evaluated after variables exist. The build function
    in each spec is called to generate the constraint expression.

    Attributes:
        model: The FlowSystemModel to create constraints in.
        variable_registry: The VariableRegistry to get handles from.

    Example:
        >>> registry = ConstraintRegistry(model, var_registry)
        >>> registry.register(
        ...     ConstraintSpec(
        ...         category='flow_bounds',
        ...         element_id='Boiler',
        ...         build_fn=lambda m, h: ConstraintResult(h['flow_rate'].variable, 100, '<='),
        ...     )
        ... )
        >>> registry.create_all()
    """

    def __init__(self, model: FlowSystemModel, variable_registry: VariableRegistry):
        self.model = model
        self.variable_registry = variable_registry
        self._specs_by_category: dict[str, list[ConstraintSpec]] = defaultdict(list)
        self._created = False

    def register(self, spec: ConstraintSpec) -> None:
        """Register a constraint specification for batch creation.

        Args:
            spec: The constraint specification to register.

        Raises:
            RuntimeError: If constraints have already been created.
        """
        if self._created:
            raise RuntimeError('Cannot register specs after constraints have been created')
        self._specs_by_category[spec.category].append(spec)

    def create_all(self) -> None:
        """Batch-create all registered constraints.

        Calls each spec's build function with the model and variable handles,
        then groups results by category for batch creation.

        Raises:
            RuntimeError: If already called.
        """
        if self._created:
            raise RuntimeError('Constraints have already been created')

        for category, specs in self._specs_by_category.items():
            if specs:
                self._create_batch(category, specs)

        self._created = True
        logger.debug(f'ConstraintRegistry created {len(self._specs_by_category)} constraint categories')

    def _create_batch(self, category: str, specs: list[ConstraintSpec]) -> None:
        """Create all constraints of a category.

        Attempts to use true vectorized batching for known constraint patterns.
        Falls back to individual creation for complex constraints.

        Args:
            category: The constraint category name.
            specs: List of specs for this category.
        """
        # Try vectorized batching for known patterns
        if self._try_vectorized_batch(category, specs):
            return

        # Fall back to individual creation
        self._create_individual(category, specs)

    def _try_vectorized_batch(self, category: str, specs: list[ConstraintSpec]) -> bool:
        """Try to create constraints using true vectorized batching.

        Returns True if successful, False to fall back to individual creation.
        """
        # Known batchable constraint patterns
        if category == 'total_flow_hours_eq':
            return self._batch_total_flow_hours_eq(specs)
        elif category == 'flow_hours_over_periods_eq':
            return self._batch_flow_hours_over_periods_eq(specs)
        elif category == 'flow_rate_ub':
            return self._batch_flow_rate_ub(specs)
        elif category == 'flow_rate_lb':
            return self._batch_flow_rate_lb(specs)

        return False

    def _get_flow_elements(self) -> dict[str, Any]:
        """Build a mapping from element_id (label_full) to Flow element."""
        if not hasattr(self, '_flow_element_map'):
            self._flow_element_map = {}
            for comp in self.model.flow_system.components.values():
                for flow in comp.inputs + comp.outputs:
                    self._flow_element_map[flow.label_full] = flow
        return self._flow_element_map

    def _batch_total_flow_hours_eq(self, specs: list[ConstraintSpec]) -> bool:
        """Batch create: total_flow_hours = sum_temporal(flow_rate)"""
        try:
            # Get full batched variables
            flow_rate = self.variable_registry.get_full_variable('flow_rate')
            total_flow_hours = self.variable_registry.get_full_variable('total_flow_hours')

            # Vectorized sum across time dimension
            rhs = self.model.sum_temporal(flow_rate)

            # Single constraint call for all elements
            self.model.add_constraints(total_flow_hours == rhs, name='total_flow_hours_eq')

            logger.debug(f'Batched {len(specs)} total_flow_hours_eq constraints')
            return True
        except Exception as e:
            logger.warning(f'Failed to batch total_flow_hours_eq, falling back: {e}')
            return False

    def _batch_flow_hours_over_periods_eq(self, specs: list[ConstraintSpec]) -> bool:
        """Batch create: flow_hours_over_periods = sum(total_flow_hours * period_weight)"""
        try:
            # Get full batched variables
            total_flow_hours = self.variable_registry.get_full_variable('total_flow_hours')
            flow_hours_over_periods = self.variable_registry.get_full_variable('flow_hours_over_periods')

            # Vectorized weighted sum
            period_weights = self.model.flow_system.period_weights
            if period_weights is None:
                period_weights = 1.0
            weighted = (total_flow_hours * period_weights).sum('period')

            # Single constraint call for all elements
            self.model.add_constraints(flow_hours_over_periods == weighted, name='flow_hours_over_periods_eq')

            logger.debug(f'Batched {len(specs)} flow_hours_over_periods_eq constraints')
            return True
        except Exception as e:
            logger.warning(f'Failed to batch flow_hours_over_periods_eq, falling back: {e}')
            return False

    def _batch_flow_rate_ub(self, specs: list[ConstraintSpec]) -> bool:
        """Batch create: flow_rate <= status * size * relative_max"""
        try:
            # Get element_ids from specs (subset of all flows - only those with status)
            spec_element_ids = [spec.element_id for spec in specs]

            # Get full batched variables and select only relevant elements
            flow_rate_full = self.variable_registry.get_full_variable('flow_rate')
            status_full = self.variable_registry.get_full_variable('status')

            flow_rate = flow_rate_full.sel(element=spec_element_ids)
            status = status_full.sel(element=spec_element_ids)

            # Build upper bounds array from flow elements
            flow_elements = self._get_flow_elements()
            upper_bounds = xr.concat(
                [flow_elements[eid].size * flow_elements[eid].relative_maximum for eid in spec_element_ids],
                dim='element',
            ).assign_coords(element=spec_element_ids)

            # Create vectorized constraint: flow_rate <= status * upper_bounds
            rhs = status * upper_bounds
            self.model.add_constraints(flow_rate <= rhs, name='flow_rate_ub')

            logger.debug(f'Batched {len(specs)} flow_rate_ub constraints')
            return True
        except Exception as e:
            logger.warning(f'Failed to batch flow_rate_ub, falling back: {e}')
            return False

    def _batch_flow_rate_lb(self, specs: list[ConstraintSpec]) -> bool:
        """Batch create: flow_rate >= status * epsilon"""
        try:
            from .config import CONFIG

            # Get element_ids from specs (subset of all flows - only those with status)
            spec_element_ids = [spec.element_id for spec in specs]

            # Get full batched variables and select only relevant elements
            flow_rate_full = self.variable_registry.get_full_variable('flow_rate')
            status_full = self.variable_registry.get_full_variable('status')

            flow_rate = flow_rate_full.sel(element=spec_element_ids)
            status = status_full.sel(element=spec_element_ids)

            # Build lower bounds array from flow elements
            # epsilon = max(CONFIG.Modeling.epsilon, size * relative_minimum)
            flow_elements = self._get_flow_elements()
            lower_bounds = xr.concat(
                [
                    np.maximum(
                        CONFIG.Modeling.epsilon,
                        flow_elements[eid].size * flow_elements[eid].relative_minimum,
                    )
                    for eid in spec_element_ids
                ],
                dim='element',
            ).assign_coords(element=spec_element_ids)

            # Create vectorized constraint: flow_rate >= status * lower_bounds
            rhs = status * lower_bounds
            self.model.add_constraints(flow_rate >= rhs, name='flow_rate_lb')

            logger.debug(f'Batched {len(specs)} flow_rate_lb constraints')
            return True
        except Exception as e:
            logger.warning(f'Failed to batch flow_rate_lb, falling back: {e}')
            return False

    def _create_individual(self, category: str, specs: list[ConstraintSpec]) -> None:
        """Create constraints individually (fallback for complex constraints)."""
        for spec in specs:
            # Get handles for this element
            handles = self.variable_registry.get_handles_for_element(spec.element_id)

            # Build the constraint
            try:
                result = spec.build_fn(self.model, handles)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to build constraint '{category}' for element '{spec.element_id}': {e}"
                ) from e

            # Create the constraint
            constraint_name = f'{spec.element_id}|{category}'
            if result.sense == '==':
                self.model.add_constraints(result.lhs == result.rhs, name=constraint_name)
            elif result.sense == '<=':
                self.model.add_constraints(result.lhs <= result.rhs, name=constraint_name)
            elif result.sense == '>=':
                self.model.add_constraints(result.lhs >= result.rhs, name=constraint_name)
            else:
                raise ValueError(f'Invalid constraint sense: {result.sense}')

    @property
    def categories(self) -> list[str]:
        """List of all registered categories."""
        return list(self._specs_by_category.keys())

    def __repr__(self) -> str:
        status = 'created' if self._created else 'pending'
        total_specs = sum(len(specs) for specs in self._specs_by_category.values())
        return f'ConstraintRegistry(categories={len(self._specs_by_category)}, specs={total_specs}, status={status})'


# =============================================================================
# System Constraint Registry (Cross-Element Constraints)
# =============================================================================


@dataclass
class SystemConstraintSpec:
    """Specification for constraints that span multiple elements.

    These are constraints like bus balance that aggregate across elements.

    Attributes:
        category: Constraint category (e.g., 'bus_balance').
        build_fn: Callable that builds the constraint. Called as:
            build_fn(model, variable_registry) -> list[ConstraintResult] or ConstraintResult
    """

    category: str
    build_fn: Callable[[FlowSystemModel, VariableRegistry], ConstraintResult | list[ConstraintResult]]


class SystemConstraintRegistry:
    """Registry for system-wide constraints that span multiple elements.

    These constraints are created after element constraints and have access
    to the full variable registry.

    Example:
        >>> registry = SystemConstraintRegistry(model, var_registry)
        >>> registry.register(
        ...     SystemConstraintSpec(
        ...         category='bus_balance',
        ...         build_fn=build_bus_balance,
        ...     )
        ... )
        >>> registry.create_all()
    """

    def __init__(self, model: FlowSystemModel, variable_registry: VariableRegistry):
        self.model = model
        self.variable_registry = variable_registry
        self._specs: list[SystemConstraintSpec] = []
        self._created = False

    def register(self, spec: SystemConstraintSpec) -> None:
        """Register a system constraint specification."""
        if self._created:
            raise RuntimeError('Cannot register specs after constraints have been created')
        self._specs.append(spec)

    def create_all(self) -> None:
        """Create all registered system constraints."""
        if self._created:
            raise RuntimeError('System constraints have already been created')

        for spec in self._specs:
            try:
                results = spec.build_fn(self.model, self.variable_registry)
            except Exception as e:
                raise RuntimeError(f"Failed to build system constraint '{spec.category}': {e}") from e

            # Handle single or multiple results
            if isinstance(results, ConstraintResult):
                results = [results]

            for i, result in enumerate(results):
                name = f'{spec.category}' if len(results) == 1 else f'{spec.category}_{i}'
                if result.sense == '==':
                    self.model.add_constraints(result.lhs == result.rhs, name=name)
                elif result.sense == '<=':
                    self.model.add_constraints(result.lhs <= result.rhs, name=name)
                elif result.sense == '>=':
                    self.model.add_constraints(result.lhs >= result.rhs, name=name)

        self._created = True

    def __repr__(self) -> str:
        status = 'created' if self._created else 'pending'
        return f'SystemConstraintRegistry(specs={len(self._specs)}, status={status})'


# =============================================================================
# Effect Share Registry (Batch Effect Share Creation)
# =============================================================================


class EffectShareRegistry:
    """Collects effect share specifications and batch-creates them.

    Effect shares link flow rates to effects (costs, emissions, etc.).
    Traditional approach creates them one at a time; this batches them.

    The key insight: all flow_rate variables are already batched with an
    element dimension. We can create ONE effect share variable for all
    flows contributing to an effect, then ONE constraint.

    Example:
        >>> registry = EffectShareRegistry(model, var_registry)
        >>> registry.register(EffectShareSpec('Boiler(gas_in)', 'costs', 30.0))
        >>> registry.register(EffectShareSpec('HeatPump(elec_in)', 'costs', 100.0))
        >>> registry.create_all()  # One batched call instead of two!
    """

    def __init__(self, model: FlowSystemModel, variable_registry: VariableRegistry):
        self.model = model
        self.variable_registry = variable_registry
        # Group by (effect_name, target) for batching
        self._specs_by_effect: dict[tuple[str, str], list[EffectShareSpec]] = defaultdict(list)
        self._created = False

    def register(self, spec: EffectShareSpec) -> None:
        """Register an effect share specification."""
        if self._created:
            raise RuntimeError('Cannot register specs after shares have been created')
        key = (spec.effect_name, spec.target)
        self._specs_by_effect[key].append(spec)

    def create_all(self) -> None:
        """Batch-create all registered effect shares.

        For each (effect, target) combination:
        1. Build a factors array aligned with element dimension
        2. Compute batched expression: flow_rate * timestep_duration * factors
        3. Add ONE share to the effect with the sum across elements
        """
        if self._created:
            raise RuntimeError('Effect shares have already been created')

        for (effect_name, target), specs in self._specs_by_effect.items():
            self._create_batch(effect_name, target, specs)

        self._created = True
        logger.debug(f'EffectShareRegistry created shares for {len(self._specs_by_effect)} effect/target combinations')

    def _create_batch(self, effect_name: str, target: str, specs: list[EffectShareSpec]) -> None:
        """Create batched effect shares for one effect/target combination.

        The key insight: instead of creating one complex constraint with a sum
        of 200+ terms, we create a BATCHED share variable with element dimension,
        then ONE simple vectorized constraint where each entry is just:
            share_var[e,t] = flow_rate[e,t] * timestep_duration * factor[e]

        This is much faster because linopy can process the simple per-element
        constraint efficiently.
        """
        import time

        logger.debug(f'_create_batch called for {effect_name}/{target} with {len(specs)} specs')
        try:
            # Get the full batched flow_rate variable
            flow_rate = self.variable_registry.get_full_variable('flow_rate')
            element_ids = self.variable_registry.get_element_ids('flow_rate')

            # Build factors: map element_id -> factor (0 for elements without effects)
            spec_map = {spec.element_id: spec.factor for spec in specs}
            factors_list = [spec_map.get(eid, 0) for eid in element_ids]

            # Stack factors into DataArray with element dimension
            # xarray handles broadcasting of scalars and DataArrays automatically
            factors_da = xr.concat(
                [xr.DataArray(f) if not isinstance(f, xr.DataArray) else f for f in factors_list],
                dim='element',
            ).assign_coords(element=element_ids)

            # Compute batched expression: flow_rate * timestep_duration * factors
            # Broadcasting handles (element, time, ...) * (element,) or (element, time, ...)
            t1 = time.perf_counter()
            expression = flow_rate * self.model.timestep_duration * factors_da
            t2 = time.perf_counter()

            # Get the effect model
            effect = self.model.effects.effects[effect_name]

            if target == 'temporal':
                # Create ONE batched share variable with element dimension
                # Combine element coord with temporal coords
                temporal_coords = self.model.get_coords(self.model.temporal_dims)
                share_var = self.model.add_variables(
                    coords=xr.Coordinates(
                        {'element': element_ids, **{dim: temporal_coords[dim] for dim in temporal_coords}}
                    ),
                    name=f'flow_effects->{effect_name}(temporal)',
                )
                t3 = time.perf_counter()

                # ONE vectorized constraint (simple per-element equality)
                self.model.add_constraints(
                    share_var == expression,
                    name=f'flow_effects->{effect_name}(temporal)',
                )
                t4 = time.perf_counter()

                # Add sum of shares to the effect's total_per_timestep equation
                # Sum across elements to get contribution at each timestep
                effect.submodel.temporal._eq_total_per_timestep.lhs -= share_var.sum('element')
                t5 = time.perf_counter()

                logger.debug(
                    f'{effect_name}: expr={(t2 - t1) * 1000:.1f}ms var={(t3 - t2) * 1000:.1f}ms con={(t4 - t3) * 1000:.1f}ms mod={(t5 - t4) * 1000:.1f}ms'
                )

            elif target == 'periodic':
                # Similar for periodic, but sum over time first
                all_coords = self.model.get_coords()
                periodic_coords = {dim: all_coords[dim] for dim in ['period', 'scenario'] if dim in all_coords}
                if periodic_coords:
                    periodic_coords['element'] = element_ids

                    share_var = self.model.add_variables(
                        coords=xr.Coordinates(periodic_coords),
                        name=f'flow_effects->{effect_name}(periodic)',
                    )

                    # Sum expression over time
                    periodic_expression = expression.sum(self.model.temporal_dims)

                    self.model.add_constraints(
                        share_var == periodic_expression,
                        name=f'flow_effects->{effect_name}(periodic)',
                    )

                    effect.submodel.periodic._eq_total.lhs -= share_var.sum('element')

            logger.debug(f'Batched {len(specs)} effect shares for {effect_name}/{target}')

        except Exception as e:
            logger.warning(f'Failed to batch effect shares for {effect_name}/{target}: {e}')
            # Fall back to individual creation
            for spec in specs:
                self._create_individual(effect_name, target, [spec])

    def _create_individual(self, effect_name: str, target: str, specs: list[EffectShareSpec]) -> None:
        """Fall back to individual effect share creation."""
        logger.debug(f'_create_individual called for {effect_name}/{target} with {len(specs)} specs')
        for spec in specs:
            handles = self.variable_registry.get_handles_for_element(spec.element_id)
            if 'flow_rate' not in handles:
                continue

            flow_rate = handles['flow_rate'].variable
            expression = flow_rate * self.model.timestep_duration * spec.factor

            effect = self.model.effects.effects[effect_name]
            if target == 'temporal':
                effect.submodel.temporal.add_share(
                    spec.element_id,
                    expression,
                    dims=('time', 'period', 'scenario'),
                )
            elif target == 'periodic':
                periodic_expression = expression.sum(self.model.temporal_dims)
                effect.submodel.periodic.add_share(
                    spec.element_id,
                    periodic_expression,
                    dims=('period', 'scenario'),
                )

    @property
    def effect_count(self) -> int:
        """Number of distinct effect/target combinations."""
        return len(self._specs_by_effect)

    @property
    def total_specs(self) -> int:
        """Total number of registered specs."""
        return sum(len(specs) for specs in self._specs_by_effect.values())

    def __repr__(self) -> str:
        status = 'created' if self._created else 'pending'
        return f'EffectShareRegistry(effects={self.effect_count}, specs={self.total_specs}, status={status})'
