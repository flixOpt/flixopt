"""Monkey-patch linopy.Model with disjunctive piecewise linear constraint support.

Backports ``add_disjunctive_piecewise_constraints`` from linopy PR #576
(not yet merged as of linopy 0.6). Becomes a no-op once linopy ships the
method natively.

Only the **disjunctive** (disaggregated convex combination) formulation is
included — sufficient for flixopt's Piece-based interface where every segment
is represented explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import linopy
import pandas as pd
import xarray as xr
from linopy.expressions import LinearExpression
from linopy.variables import Variable

if TYPE_CHECKING:
    from linopy import Constraint
    from xarray import DataArray

# ---------------------------------------------------------------------------
# Constants (match linopy PR #576 constants.py additions)
# ---------------------------------------------------------------------------
PWL_LAMBDA_SUFFIX = '_lambda'
PWL_CONVEX_SUFFIX = '_convex'
PWL_LINK_SUFFIX = '_link'
PWL_BINARY_SUFFIX = '_binary'
PWL_SELECT_SUFFIX = '_select'
DEFAULT_BREAKPOINT_DIM = 'breakpoint'
DEFAULT_SEGMENT_DIM = 'segment'


# ---------------------------------------------------------------------------
# Methods to be patched onto linopy.Model
# ---------------------------------------------------------------------------


def _add_disjunctive_piecewise_constraints(
    self: linopy.Model,
    expr: Variable | LinearExpression | dict[str, Variable | LinearExpression],
    breakpoints: DataArray,
    link_dim: str | None = None,
    dim: str = DEFAULT_BREAKPOINT_DIM,
    segment_dim: str = DEFAULT_SEGMENT_DIM,
    mask: DataArray | None = None,
    name: str | None = None,
    skip_nan_check: bool = False,
) -> Constraint:
    """Add a disjunctive piecewise linear constraint for disconnected segments.

    Uses the disaggregated convex combination formulation:
    1. Binary y_k in {0,1} per segment, sum(y_k) = 1
    2. Lambda lam_{k,i} in [0,1] per breakpoint in each segment
    3. Convexity: sum_i(lam_{k,i}) = y_k
    4. SOS2 within each segment (along breakpoint dim)
    5. Linking: expr = sum_k sum_i lam_{k,i} * bp_{k,i}
    """
    # --- Input validation ---
    if dim not in breakpoints.dims:
        raise ValueError(f"breakpoints must have dimension '{dim}', but only has dimensions {list(breakpoints.dims)}")
    if segment_dim not in breakpoints.dims:
        raise ValueError(
            f"breakpoints must have dimension '{segment_dim}', but only has dimensions {list(breakpoints.dims)}"
        )
    if dim == segment_dim:
        raise ValueError(f"dim and segment_dim must be different, both are '{dim}'")
    if not pd.api.types.is_numeric_dtype(breakpoints.coords[dim]):
        raise ValueError(
            f"Breakpoint dimension '{dim}' must have numeric coordinates "
            f'for SOS2 weights, but got {breakpoints.coords[dim].dtype}'
        )

    # --- Generate name using shared counter ---
    if name is None:
        name = f'pwl{self._pwlCounter}'
    self._pwlCounter += 1

    # --- Determine target expression ---
    is_single = isinstance(expr, (Variable, LinearExpression))
    is_dict = isinstance(expr, dict)

    if not is_single and not is_dict:
        raise ValueError(f"'expr' must be a Variable, LinearExpression, or dict of these, got {type(expr)}")

    if is_single:
        target_expr = _to_linexpr(self, expr)
        resolved_link_dim = None
        computed_mask = _compute_pwl_mask(mask, breakpoints, skip_nan_check)
        lambda_mask = computed_mask
    else:
        assert isinstance(expr, dict)
        expr_keys = set(expr.keys())
        resolved_link_dim = _resolve_pwl_link_dim(
            link_dim,
            breakpoints,
            dim,
            expr_keys,
            exclude_dims={dim, segment_dim},
        )
        computed_mask = _compute_pwl_mask(mask, breakpoints, skip_nan_check)
        lambda_mask = computed_mask.any(dim=resolved_link_dim) if computed_mask is not None else None
        target_expr = _build_stacked_expr(self, expr, breakpoints, resolved_link_dim)

    # Build coordinate lists excluding special dimensions
    exclude_dims_set = {dim, segment_dim, resolved_link_dim} - {None}
    extra_coords = [
        pd.Index(breakpoints.coords[d].values, name=d) for d in breakpoints.dims if d not in exclude_dims_set
    ]

    # Also include extra dimensions from the target expression (e.g. 'time')
    # that are not in the breakpoints. This ensures lambda/binary variables
    # can vary independently along those dimensions.
    expr_data = target_expr.data
    for d in expr_data.dims:
        if d in exclude_dims_set or d == '_term':
            continue
        if not any(idx.name == d for idx in extra_coords):
            extra_coords.append(pd.Index(expr_data.coords[d].values, name=d))

    lambda_coords = extra_coords + [
        pd.Index(breakpoints.coords[segment_dim].values, name=segment_dim),
        pd.Index(breakpoints.coords[dim].values, name=dim),
    ]
    binary_coords = extra_coords + [
        pd.Index(breakpoints.coords[segment_dim].values, name=segment_dim),
    ]

    # Binary mask: valid if any breakpoint in segment is valid
    binary_mask = lambda_mask.any(dim=dim) if lambda_mask is not None else None

    return _add_dpwl_sos2(
        self,
        name,
        breakpoints,
        dim,
        segment_dim,
        target_expr,
        lambda_coords,
        lambda_mask,
        binary_coords,
        binary_mask,
    )


def _add_dpwl_sos2(
    model: linopy.Model,
    name: str,
    breakpoints: DataArray,
    dim: str,
    segment_dim: str,
    target_expr: LinearExpression,
    lambda_coords: list[pd.Index],
    lambda_mask: DataArray | None,
    binary_coords: list[pd.Index],
    binary_mask: DataArray | None,
) -> Constraint:
    """Disaggregated convex combination formulation."""
    binary_name = f'{name}{PWL_BINARY_SUFFIX}'
    select_name = f'{name}{PWL_SELECT_SUFFIX}'
    lambda_name = f'{name}{PWL_LAMBDA_SUFFIX}'
    convex_name = f'{name}{PWL_CONVEX_SUFFIX}'
    link_name = f'{name}{PWL_LINK_SUFFIX}'

    # 1. Binary variables y_k in {0,1}, one per segment
    binary_var = model.add_variables(
        binary=True,
        coords=binary_coords,
        name=binary_name,
        mask=binary_mask,
    )

    # 2. Selection constraint: sum(y_k) = 1
    select_con = model.add_constraints(
        binary_var.sum(dim=segment_dim) == 1,
        name=select_name,
    )

    # 3. Lambda variables lam_{k,i} in [0,1], per segment per breakpoint
    lambda_var = model.add_variables(
        lower=0,
        upper=1,
        coords=lambda_coords,
        name=lambda_name,
        mask=lambda_mask,
    )

    # 4. SOS2 within each segment (along breakpoint dim)
    # Only needed when >2 breakpoints per segment; with 2 breakpoints the
    # adjacency constraint is trivially satisfied and we avoid requiring
    # solver SOS support (e.g. HiGHS).
    if breakpoints.sizes[dim] > 2:
        model.add_sos_constraints(lambda_var, sos_type=2, sos_dim=dim)

    # 5. Convexity: sum_i(lam_{k,i}) = y_k (lambdas sum to binary indicator)
    model.add_constraints(lambda_var.sum(dim=dim) == binary_var, name=convex_name)

    # 6. Linking: expr = sum_k sum_i lam_{k,i} * bp_{k,i}
    weighted_sum = (lambda_var * breakpoints).sum(dim=[segment_dim, dim])
    model.add_constraints(target_expr == weighted_sum, name=link_name)

    return select_con


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_linexpr(model: linopy.Model, expr: Variable | LinearExpression) -> LinearExpression:
    if isinstance(expr, LinearExpression):
        return expr
    return expr.to_linexpr()


def _compute_pwl_mask(
    mask: DataArray | None,
    breakpoints: DataArray,
    skip_nan_check: bool,
) -> DataArray | None:
    if mask is not None:
        return mask
    if skip_nan_check:
        return None
    return ~breakpoints.isnull()


def _resolve_pwl_link_dim(
    link_dim: str | None,
    breakpoints: DataArray,
    dim: str,
    expr_keys: set[str],
    exclude_dims: set[str] | None = None,
) -> str:
    if exclude_dims is None:
        exclude_dims = {dim}
    if link_dim is None:
        for d in breakpoints.dims:
            if d in exclude_dims:
                continue
            coords_set = {str(c) for c in breakpoints.coords[d].values}
            if coords_set == expr_keys:
                return str(d)
        raise ValueError(
            'Could not auto-detect link_dim. Please specify it explicitly. '
            f'Breakpoint dimensions: {list(breakpoints.dims)}, '
            f'expression keys: {list(expr_keys)}'
        )

    if link_dim not in breakpoints.dims:
        raise ValueError(f"link_dim '{link_dim}' not found in breakpoints dimensions {list(breakpoints.dims)}")
    coords_set = {str(c) for c in breakpoints.coords[link_dim].values}
    if coords_set != expr_keys:
        raise ValueError(f"link_dim '{link_dim}' coordinates {coords_set} don't match expression keys {expr_keys}")
    return link_dim


def _build_stacked_expr(
    model: linopy.Model,
    expr_dict: dict[str, Variable | LinearExpression],
    breakpoints: DataArray,
    link_dim: str,
) -> LinearExpression:
    """Build a stacked LinearExpression from a dict of Variables/Expressions."""
    link_coords = list(breakpoints.coords[link_dim].values)

    expr_data_list = []
    for k in link_coords:
        e = expr_dict[str(k)]
        linexpr = _to_linexpr(model, e)
        data = linexpr.data
        # Drop scalar coords that would conflict during concat (e.g. 'effect'
        # after .sel(effect='costs') leaves a scalar coord with value 'costs')
        scalar_coords = [c for c in data.coords if c not in data.dims and c != link_dim]
        if scalar_coords:
            data = data.drop_vars(scalar_coords)
        expr_data_list.append(data.expand_dims({link_dim: [k]}))

    stacked_data = xr.concat(expr_data_list, dim=link_dim)
    return LinearExpression(stacked_data, model)


# ---------------------------------------------------------------------------
# Patch function
# ---------------------------------------------------------------------------


def patch_linopy_model() -> None:
    """Monkey-patch ``add_disjunctive_piecewise_constraints`` onto linopy.Model.

    Safe to call multiple times — skips if the method already exists (i.e. when
    linopy ships native support).
    """
    if hasattr(linopy.Model, 'add_disjunctive_piecewise_constraints'):
        return

    linopy.Model.add_disjunctive_piecewise_constraints = _add_disjunctive_piecewise_constraints

    # Ensure _pwlCounter is initialised on every Model instance.
    if not hasattr(linopy.Model, '_pwlCounter'):
        _original_init = linopy.Model.__init__

        def _patched_init(self, *args, **kwargs):
            _original_init(self, *args, **kwargs)
            if not hasattr(self, '_pwlCounter'):
                self._pwlCounter = 0

        linopy.Model.__init__ = _patched_init
