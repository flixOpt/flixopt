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
from typing import TYPE_CHECKING, Literal

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
        # Check if all bounds are identical scalars (common case: all inf)
        if all(isinstance(b, (int, float)) and not isinstance(b, xr.DataArray) for b in bounds):
            if len(set(bounds)) == 1:
                return bounds[0]  # Return scalar - linopy will broadcast

        # Need to stack into DataArray
        arrays_to_stack = []
        for bound, eid in zip(bounds, element_ids, strict=False):
            if isinstance(bound, xr.DataArray):
                # Ensure proper dimension order
                arr = bound.expand_dims(element=[eid])
            else:
                # Scalar - create DataArray
                arr = xr.DataArray(
                    bound,
                    coords={'element': [eid]},
                    dims=['element'],
                )
            arrays_to_stack.append(arr)

        # Concatenate along element dimension
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

        return False

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
