"""
This module contains the features of the flixopt framework.
Features extend the functionality of Elements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    import linopy

    from .interface import (
        InvestParameters,
    )
    from .structure import FlowSystemModel


# =============================================================================
# Helper functions for shared constraint math
# =============================================================================


def sparse_weighted_sum(var, coeffs: xr.DataArray, sum_dim: str, group_dim: str):
    """Compute (var * coeffs).sum(sum_dim) efficiently using sparse groupby.

    When coeffs is a sparse array (most entries zero) with dims (group_dim, sum_dim, ...),
    the naive dense broadcast creates a huge intermediate linopy expression.
    This function selects only the non-zero (group, sum_dim) pairs and uses
    groupby to aggregate, avoiding the dense broadcast entirely.

    Args:
        var: linopy Variable or LinearExpression with sum_dim as a dimension.
        coeffs: xr.DataArray with at least (group_dim, sum_dim) dims.
            Additional dims (e.g., equation_idx, time) are preserved.
        sum_dim: Dimension to sum over (e.g., 'flow').
        group_dim: Dimension to group by (e.g., 'converter', 'component').

    Returns:
        linopy expression with sum_dim removed, group_dim present.
    """
    import linopy

    coeffs_values = coeffs.values
    group_ids = list(coeffs.coords[group_dim].values)
    sum_ids = list(coeffs.coords[sum_dim].values)

    # Find which (group, sum_dim) pairs have any non-zero coefficient.
    # The group_dim and sum_dim may not be the first two axes, so locate them.
    group_axis = coeffs.dims.index(group_dim)
    sum_axis = coeffs.dims.index(sum_dim)

    # Collapse all axes except group and sum to find any non-zero entry
    reduce_axes = tuple(i for i in range(coeffs_values.ndim) if i not in (group_axis, sum_axis))
    if reduce_axes:
        nonzero_2d = np.any(coeffs_values != 0, axis=reduce_axes)
    else:
        nonzero_2d = coeffs_values != 0

    # Ensure shape is (group, sum_dim) regardless of original axis order
    if group_axis > sum_axis:
        nonzero_2d = nonzero_2d.T
    group_idx, sum_idx = np.nonzero(nonzero_2d)

    if len(group_idx) == 0:
        return (var * coeffs).sum(sum_dim)

    pair_sum_ids = [sum_ids[s] for s in sum_idx]
    pair_group_ids = [group_ids[g] for g in group_idx]

    # Extract per-pair coefficients: select along group_dim and sum_dim axes
    # Build indexing tuple for the original array
    idx = [slice(None)] * coeffs_values.ndim
    pair_coeffs_list = []
    for g, s in zip(group_idx, sum_idx, strict=False):
        idx[group_axis] = g
        idx[sum_axis] = s
        pair_coeffs_list.append(coeffs_values[tuple(idx)])
    pair_coeffs_data = np.array(pair_coeffs_list)

    # Build DataArray for pair coefficients with remaining dims
    remaining_dims = [d for d in coeffs.dims if d not in (group_dim, sum_dim)]
    remaining_coords = {d: coeffs.coords[d] for d in remaining_dims if d in coeffs.coords}
    pair_coeffs = xr.DataArray(
        pair_coeffs_data,
        dims=['pair'] + remaining_dims,
        coords=remaining_coords,
    )

    # Select var for active pairs and multiply by coefficients.
    # Convert to LinearExpression first to avoid linopy Variable coord issues.
    selected = (var * 1).sel({sum_dim: xr.DataArray(pair_sum_ids, dims=['pair'])})
    # Drop the dangling sum_dim coordinate that sel() leaves behind
    selected = linopy.LinearExpression(selected.data.drop_vars(sum_dim, errors='ignore'), selected.model)
    weighted = selected * pair_coeffs

    # Groupby to sum back to group dimension
    mapping = xr.DataArray(pair_group_ids, dims=['pair'], name=group_dim)
    result = weighted.groupby(mapping).sum()

    # Reindex to original group order (groupby sorts alphabetically)
    return result.sel({group_dim: group_ids})


def fast_notnull(arr: xr.DataArray) -> xr.DataArray:
    """Fast notnull check using numpy (~55x faster than xr.DataArray.notnull()).

    Args:
        arr: DataArray to check for non-null values.

    Returns:
        Boolean DataArray with True where values are not NaN.
    """
    return xr.DataArray(~np.isnan(arr.values), dims=arr.dims, coords=arr.coords)


def fast_isnull(arr: xr.DataArray) -> xr.DataArray:
    """Fast isnull check using numpy (~55x faster than xr.DataArray.isnull()).

    Args:
        arr: DataArray to check for null values.

    Returns:
        Boolean DataArray with True where values are NaN.
    """
    return xr.DataArray(np.isnan(arr.values), dims=arr.dims, coords=arr.coords)


def stack_along_dim(
    values: list[float | xr.DataArray],
    dim: str,
    coords: list,
    target_coords: dict | None = None,
) -> xr.DataArray:
    """Stack per-element values into a DataArray along a new labeled dimension.

    Handles mixed inputs: scalars, 0-d DataArrays, and N-d DataArrays with
    potentially different dimensions. Uses fast numpy pre-allocation instead
    of xr.concat for performance.

    Args:
        values: Per-element values to stack (scalars or DataArrays).
        dim: Name of the new dimension.
        coords: Coordinate labels for the new dimension.
        target_coords: Optional coords to broadcast to (e.g., {'time': ..., 'period': ...}).
            Order determines output dimension order after dim.

    Returns:
        DataArray with dim as first dimension.
    """
    target_coords = target_coords or {}

    # Classify values and collect extra dimension info
    scalar_values = []
    has_array = False
    collected_coords: dict = {}

    for v in values:
        if isinstance(v, xr.DataArray):
            if v.ndim == 0:
                scalar_values.append(float(v.values))
            else:
                has_array = True
                for d in v.dims:
                    if d not in collected_coords:
                        collected_coords[d] = v.coords[d].values
        elif isinstance(v, (int, float, np.integer, np.floating)):
            scalar_values.append(float(v))
        else:
            has_array = True

    # Fast path: all scalars, no target_coords to broadcast to
    if not has_array and not target_coords:
        return xr.DataArray(
            np.array(scalar_values),
            coords={dim: coords},
            dims=[dim],
        )

    # Merge target_coords (takes precedence) with collected coords
    final_coords = dict(target_coords)
    for d, c in collected_coords.items():
        if d not in final_coords:
            final_coords[d] = c

    # All scalars but need broadcasting to target_coords
    if not has_array:
        n = len(scalar_values)
        extra_dims = list(final_coords.keys())
        extra_shape = [len(c) for c in final_coords.values()]
        data = np.broadcast_to(
            np.array(scalar_values).reshape([n] + [1] * len(extra_dims)),
            [n] + extra_shape,
        ).copy()
        full_coords = {dim: coords}
        full_coords.update(final_coords)
        return xr.DataArray(data, coords=full_coords, dims=[dim] + extra_dims)

    # General path: pre-allocate numpy array and fill
    n_elements = len(values)
    extra_dims = list(final_coords.keys())
    extra_shape = [len(c) for c in final_coords.values()]
    full_shape = [n_elements] + extra_shape
    full_dims = [dim] + extra_dims

    data = np.full(full_shape, np.nan)

    # Create template for broadcasting only if needed
    template = xr.DataArray(coords=final_coords, dims=extra_dims) if final_coords else None

    for i, v in enumerate(values):
        if isinstance(v, xr.DataArray):
            if v.ndim == 0:
                data[i, ...] = float(v.values)
            elif template is not None:
                broadcasted = v.broadcast_like(template)
                data[i, ...] = broadcasted.values
            else:
                data[i, ...] = v.values
        elif isinstance(v, float) and np.isnan(v):
            pass  # leave as NaN
        else:
            data[i, ...] = float(v)

    full_coords = {dim: coords}
    full_coords.update(final_coords)
    return xr.DataArray(data, coords=full_coords, dims=full_dims)


class InvestmentBuilder:
    """Static helper methods for investment constraint creation.

    These helpers contain the shared math for investment constraints,
    used by FlowsModel and StoragesModel.
    """

    @staticmethod
    def add_optional_size_bounds(
        model: FlowSystemModel,
        size_var: linopy.Variable,
        invested_var: linopy.Variable,
        min_bounds: xr.DataArray,
        max_bounds: xr.DataArray,
        element_ids: list[str],
        dim_name: str,
        name_prefix: str,
    ) -> None:
        """Add state-controlled bounds for optional (non-mandatory) investments.

        Creates constraints: invested * min <= size <= invested * max

        Args:
            model: The FlowSystemModel to add constraints to.
            size_var: Size variable (already selected to non-mandatory elements).
            invested_var: Binary invested variable.
            min_bounds: Minimum size bounds DataArray.
            max_bounds: Maximum size bounds DataArray.
            element_ids: List of element IDs for these constraints.
            dim_name: Dimension name (e.g., 'flow', 'storage').
            name_prefix: Prefix for constraint names (e.g., 'flow', 'storage').
        """
        from .config import CONFIG

        epsilon = CONFIG.Modeling.epsilon
        effective_min = xr.where(min_bounds > epsilon, min_bounds, epsilon)

        size_subset = size_var.sel({dim_name: element_ids})

        model.add_constraints(
            size_subset >= invested_var * effective_min,
            name=f'{name_prefix}|size|lb',
        )
        model.add_constraints(
            size_subset <= invested_var * max_bounds,
            name=f'{name_prefix}|size|ub',
        )

    @staticmethod
    def add_linked_periods_constraints(
        model: FlowSystemModel,
        size_var: linopy.Variable,
        params: dict[str, InvestParameters],
        element_ids: list[str],
        dim_name: str,
    ) -> None:
        """Add linked periods constraints for elements that have them.

        For elements with linked_periods, constrains size to be equal
        across linked periods.

        Uses batched mask approach: builds a validity mask for all elements
        and creates a single batched constraint.

        Args:
            model: The FlowSystemModel to add constraints to.
            size_var: Size variable.
            params: Dict mapping element_id -> InvestParameters.
            element_ids: List of all element IDs.
            dim_name: Dimension name (e.g., 'flow', 'storage').
        """
        element_ids_with_linking = [eid for eid in element_ids if params[eid].linked_periods is not None]
        if not element_ids_with_linking or 'period' not in size_var.dims:
            return

        periods = size_var.coords['period'].values
        if len(periods) < 2:
            return

        # Build linking mask: (element, period) - True where period is linked
        # Stack the linked_periods arrays for all elements with linking
        mask_data = np.full((len(element_ids_with_linking), len(periods)), np.nan)
        for i, eid in enumerate(element_ids_with_linking):
            linked = params[eid].linked_periods
            if isinstance(linked, xr.DataArray):
                # Reindex to match periods
                linked_reindexed = linked.reindex(period=periods, fill_value=np.nan)
                mask_data[i, :] = linked_reindexed.values
            else:
                # Scalar or None - fill all
                mask_data[i, :] = 1.0 if linked else np.nan

        linking_mask = xr.DataArray(
            mask_data,
            dims=[dim_name, 'period'],
            coords={dim_name: element_ids_with_linking, 'period': periods},
        )

        # Select size variable for elements with linking
        size_subset = size_var.sel({dim_name: element_ids_with_linking})

        # Create constraint: size[period_i] == size[period_i+1] for linked periods
        # Loop over period pairs (typically few periods, so this is fast)
        # The batching is over elements, which is where the speedup comes from
        for i in range(len(periods) - 1):
            period_prev = periods[i]
            period_next = periods[i + 1]

            # Check which elements are linked in both periods
            mask_prev = linking_mask.sel(period=period_prev)
            mask_next = linking_mask.sel(period=period_next)
            # valid_mask: True = KEEP constraint (element is linked in both periods)
            valid_mask = fast_notnull(mask_prev) & fast_notnull(mask_next)

            # Skip if none valid
            if not valid_mask.any():
                continue

            # Select size for this period pair
            size_prev = size_subset.sel(period=period_prev)
            size_next = size_subset.sel(period=period_next)

            # Use linopy's mask parameter: True = KEEP constraint
            model.add_constraints(
                size_prev == size_next,
                name=f'{dim_name}|linked_periods|{period_prev}->{period_next}',
                mask=valid_mask,
            )

    @staticmethod
    def collect_effects(
        params: dict[str, InvestParameters],
        element_ids: list[str],
        attr: str,
        dim_name: str,
    ) -> dict[str, xr.DataArray]:
        """Collect effects dict from params into a dict of DataArrays.

        Args:
            params: Dict mapping element_id -> InvestParameters.
            element_ids: List of element IDs to collect from.
            attr: Attribute name on InvestParameters (e.g., 'effects_of_investment_per_size').
            dim_name: Dimension name for the DataArrays.

        Returns:
            Dict mapping effect_name -> DataArray with element dimension.
        """
        # Find all effect names across all elements
        all_effects: set[str] = set()
        for eid in element_ids:
            effects = getattr(params[eid], attr) or {}
            all_effects.update(effects.keys())

        if not all_effects:
            return {}

        # Build DataArray for each effect
        result = {}
        for effect_name in all_effects:
            values = []
            for eid in element_ids:
                effects = getattr(params[eid], attr) or {}
                values.append(effects.get(effect_name, np.nan))
            result[effect_name] = xr.DataArray(values, dims=[dim_name], coords={dim_name: element_ids})

        return result

    @staticmethod
    def build_effect_factors(
        effects_dict: dict[str, xr.DataArray],
        element_ids: list[str],
        dim_name: str,
    ) -> xr.DataArray | None:
        """Build factor array with (element, effect, ...) dims from effects dict.

        Args:
            effects_dict: Dict mapping effect_name -> DataArray(element_dim) or DataArray(element_dim, time).
            element_ids: Element IDs (for ordering).
            dim_name: Element dimension name.

        Returns:
            DataArray with (element, effect) or (element, effect, time) dims, or None if empty.
        """
        if not effects_dict:
            return None

        effect_ids = list(effects_dict.keys())
        effect_arrays = [effects_dict[eff] for eff in effect_ids]
        result = stack_along_dim(effect_arrays, 'effect', effect_ids)

        # Transpose to put element first, then effect, then any other dims (like time)
        dims_order = [dim_name, 'effect'] + [d for d in result.dims if d not in (dim_name, 'effect')]
        return result.transpose(*dims_order)


class StatusBuilder:
    """Static helper methods for status constraint creation.

    These helpers contain the shared math for status constraints,
    used by FlowsModel and ComponentsModel.
    """

    @staticmethod
    def compute_previous_duration(
        previous_status: xr.DataArray,
        target_state: int,
        timestep_duration: xr.DataArray | float,
    ) -> float:
        """Compute consecutive duration of target_state at end of previous_status.

        Args:
            previous_status: Previous status DataArray (time dimension).
            target_state: 1 for active (uptime), 0 for inactive (downtime).
            timestep_duration: Duration per timestep.

        Returns:
            Total duration in state at end of previous period.
        """
        values = previous_status.values
        count = 0
        for v in reversed(values):
            if (target_state == 1 and v > 0) or (target_state == 0 and v == 0):
                count += 1
            else:
                break

        # Multiply by timestep_duration
        if hasattr(timestep_duration, 'mean'):
            duration = float(timestep_duration.mean()) * count
        else:
            duration = timestep_duration * count
        return duration

    @staticmethod
    def add_batched_duration_tracking(
        model: FlowSystemModel,
        state: linopy.Variable,
        name: str,
        dim_name: str,
        timestep_duration: xr.DataArray,
        minimum_duration: xr.DataArray | None = None,
        maximum_duration: xr.DataArray | None = None,
        previous_duration: xr.DataArray | None = None,
    ) -> linopy.Variable:
        """Add batched consecutive duration tracking constraints for binary state variables.

        This is a vectorized version that operates on batched state variables
        with an element dimension.

        Creates:
        - duration variable: tracks consecutive time in state for all elements
        - upper bound: duration[e,t] <= state[e,t] * M[e]
        - forward constraint: duration[e,t+1] <= duration[e,t] + dt[t]
        - backward constraint: duration[e,t+1] >= duration[e,t] + dt[t] + (state[e,t+1] - 1) * M[e]
        - optional initial constraints if previous_duration provided

        Args:
            model: The FlowSystemModel to add constraints to.
            state: Binary state variable with (element_dim, time) dims.
            name: Full name for the duration variable (e.g., 'flow|uptime').
            dim_name: Element dimension name (e.g., 'flow', 'component').
            timestep_duration: Duration per timestep (time,).
            minimum_duration: Optional minimum duration per element (element_dim,). NaN = no constraint.
            maximum_duration: Optional maximum duration per element (element_dim,). NaN = no constraint.
            previous_duration: Optional previous duration per element (element_dim,). NaN = no previous.

        Returns:
            The created duration variable with (element_dim, time) dims.
        """
        duration_dim = 'time'
        element_ids = state.coords[dim_name].values

        # Big-M value per element - broadcast to element dimension
        mega_base = timestep_duration.sum(duration_dim)
        if previous_duration is not None:
            mega = mega_base + previous_duration.fillna(0)
        else:
            mega = mega_base

        # Upper bound per element: use max_duration where provided, else mega
        if maximum_duration is not None:
            upper_bound = xr.where(fast_notnull(maximum_duration), maximum_duration, mega)
        else:
            upper_bound = mega

        # Duration variable with (element_dim, time) dims
        duration = model.add_variables(
            lower=0,
            upper=upper_bound,
            coords=state.coords,
            name=name,
        )

        # Upper bound: duration[e,t] <= state[e,t] * M[e]
        model.add_constraints(duration <= state * mega, name=f'{name}|ub')

        # Forward constraint: duration[e,t+1] <= duration[e,t] + dt[t]
        model.add_constraints(
            duration.isel({duration_dim: slice(1, None)})
            <= duration.isel({duration_dim: slice(None, -1)}) + timestep_duration.isel({duration_dim: slice(None, -1)}),
            name=f'{name}|forward',
        )

        # Backward constraint: duration[e,t+1] >= duration[e,t] + dt[t] + (state[e,t+1] - 1) * M[e]
        model.add_constraints(
            duration.isel({duration_dim: slice(1, None)})
            >= duration.isel({duration_dim: slice(None, -1)})
            + timestep_duration.isel({duration_dim: slice(None, -1)})
            + (state.isel({duration_dim: slice(1, None)}) - 1) * mega,
            name=f'{name}|backward',
        )

        # Initial constraints for elements with previous_duration
        if previous_duration is not None:
            # Mask for elements that have previous_duration (not NaN)
            has_previous = fast_notnull(previous_duration)
            if has_previous.any():
                elem_with_prev = [eid for eid, has in zip(element_ids, has_previous.values, strict=False) if has]
                prev_vals = previous_duration.sel({dim_name: elem_with_prev})
                state_init = state.sel({dim_name: elem_with_prev}).isel({duration_dim: 0})
                duration_init = duration.sel({dim_name: elem_with_prev}).isel({duration_dim: 0})
                dt_init = timestep_duration.isel({duration_dim: 0})
                mega_subset = mega.sel({dim_name: elem_with_prev}) if dim_name in mega.dims else mega

                model.add_constraints(
                    duration_init <= state_init * (prev_vals + dt_init),
                    name=f'{name}|initial_ub',
                )
                model.add_constraints(
                    duration_init >= (state_init - 1) * mega_subset + prev_vals + state_init * dt_init,
                    name=f'{name}|initial_lb',
                )

        return duration

    @staticmethod
    def add_active_hours_constraint(
        model: FlowSystemModel,
        active_hours_var: linopy.Variable,
        status_var: linopy.Variable,
        name: str,
    ) -> None:
        """Constrain active_hours == sum_temporal(status)."""
        model.add_constraints(
            active_hours_var == model.sum_temporal(status_var),
            name=name,
        )

    @staticmethod
    def add_complementary_constraint(
        model: FlowSystemModel,
        status_var: linopy.Variable,
        inactive_var: linopy.Variable,
        name: str,
    ) -> None:
        """Constrain status + inactive == 1."""
        model.add_constraints(
            status_var + inactive_var == 1,
            name=name,
        )

    @staticmethod
    def add_switch_transition_constraint(
        model: FlowSystemModel,
        status_var: linopy.Variable,
        startup_var: linopy.Variable,
        shutdown_var: linopy.Variable,
        name: str,
    ) -> None:
        """Constrain startup[t] - shutdown[t] == status[t] - status[t-1] for t > 0."""
        model.add_constraints(
            startup_var.isel(time=slice(1, None)) - shutdown_var.isel(time=slice(1, None))
            == status_var.isel(time=slice(1, None)) - status_var.isel(time=slice(None, -1)),
            name=name,
        )

    @staticmethod
    def add_switch_mutex_constraint(
        model: FlowSystemModel,
        startup_var: linopy.Variable,
        shutdown_var: linopy.Variable,
        name: str,
    ) -> None:
        """Constrain startup + shutdown <= 1."""
        model.add_constraints(
            startup_var + shutdown_var <= 1,
            name=name,
        )

    @staticmethod
    def add_switch_initial_constraint(
        model: FlowSystemModel,
        status_t0: linopy.Variable,
        startup_t0: linopy.Variable,
        shutdown_t0: linopy.Variable,
        prev_state: xr.DataArray,
        name: str,
    ) -> None:
        """Constrain startup[0] - shutdown[0] == status[0] - previous_status[-1].

        All variables should be pre-selected to t=0 and to the relevant element subset.
        prev_state should be the last timestep of the previous period.
        """
        model.add_constraints(
            startup_t0 - shutdown_t0 == status_t0 - prev_state,
            name=name,
        )

    @staticmethod
    def add_startup_count_constraint(
        model: FlowSystemModel,
        startup_count_var: linopy.Variable,
        startup_var: linopy.Variable,
        dim_name: str,
        name: str,
    ) -> None:
        """Constrain startup_count == sum(startup) over temporal dims.

        startup_var should be pre-selected to the relevant element subset.
        """
        temporal_dims = [d for d in startup_var.dims if d not in ('period', 'scenario', dim_name)]
        model.add_constraints(
            startup_count_var == startup_var.sum(temporal_dims),
            name=name,
        )

    @staticmethod
    def add_cluster_cyclic_constraint(
        model: FlowSystemModel,
        status_var: linopy.Variable,
        name: str,
    ) -> None:
        """Constrain status[0] == status[-1] for cyclic cluster mode.

        status_var should be pre-selected to only the cyclic elements.
        """
        model.add_constraints(
            status_var.isel(time=0) == status_var.isel(time=-1),
            name=name,
        )


class MaskHelpers:
    """Static helper methods for batched constraint creation using mask matrices.

    These helpers enable batching of constraints across elements with
    variable-length relationships (e.g., component -> flows mapping).

    Pattern:
        1. Build membership dict: element_id -> list of related item_ids
        2. Create mask matrix: (element_dim, item_dim) = 1 if item belongs to element
        3. Apply mask: (variable * mask).sum(item_dim) creates batched aggregation
    """

    @staticmethod
    def build_mask(
        row_dim: str,
        row_ids: list[str],
        col_dim: str,
        col_ids: list[str],
        membership: dict[str, list[str]],
    ) -> xr.DataArray:
        """Build a binary mask matrix indicating membership between two dimensions.

        Creates a (row, col) DataArray where value is 1 if the column element
        belongs to the row element, 0 otherwise.

        Args:
            row_dim: Name for the row dimension (e.g., 'component', 'storage').
            row_ids: List of row identifiers.
            col_dim: Name for the column dimension (e.g., 'flow').
            col_ids: List of column identifiers.
            membership: Dict mapping row_id -> list of col_ids that belong to it.

        Returns:
            DataArray with dims (row_dim, col_dim), values 0 or 1.

        Example:
            >>> membership = {'storage1': ['charge', 'discharge'], 'storage2': ['in', 'out']}
            >>> mask = MaskHelpers.build_mask(
            ...     'storage', ['storage1', 'storage2'], 'flow', ['charge', 'discharge', 'in', 'out'], membership
            ... )
            >>> # Use with: (status * mask).sum('flow') <= 1
        """
        mask_data = np.zeros((len(row_ids), len(col_ids)))

        for i, row_id in enumerate(row_ids):
            for col_id in membership.get(row_id, []):
                if col_id in col_ids:
                    j = col_ids.index(col_id)
                    mask_data[i, j] = 1

        return xr.DataArray(
            mask_data,
            dims=[row_dim, col_dim],
            coords={row_dim: row_ids, col_dim: col_ids},
        )

    @staticmethod
    def build_flow_membership(
        elements: list,
        get_flows: callable,
    ) -> dict[str, list[str]]:
        """Build membership dict from elements to their flows.

        Args:
            elements: List of elements (components, storages, etc.).
            get_flows: Function that returns list of flows for an element.

        Returns:
            Dict mapping element label -> list of flow label_full.

        Example:
            >>> membership = MaskHelpers.build_flow_membership(storages, lambda s: s.inputs + s.outputs)
        """
        return {e.label: [f.label_full for f in get_flows(e)] for e in elements}


class PiecewiseBuilder:
    """Static helper methods for batched piecewise linear modeling.

    Enables batching of piecewise constraints across multiple elements with
    potentially different segment counts using the "pad to max" approach.

    Pattern:
        1. Collect segment counts from elements
        2. Build segment mask (valid vs padded segments)
        3. Pad breakpoints to max segment count
        4. Create batched variables (inside_piece, lambda0, lambda1)
        5. Create batched constraints

    Variables created (all with element and segment dimensions):
        - inside_piece: binary, 1 if segment is active
        - lambda0: continuous [0,1], weight for segment start
        - lambda1: continuous [0,1], weight for segment end

    Constraints:
        - lambda0 + lambda1 == inside_piece (per element, segment)
        - sum(inside_piece, segment) <= 1 or zero_point (per element)
        - var == sum(lambda0 * starts + lambda1 * ends) (coupling)
    """

    @staticmethod
    def collect_segment_info(
        element_ids: list[str],
        segment_counts: dict[str, int],
        dim_name: str,
    ) -> tuple[int, xr.DataArray]:
        """Collect segment counts and build validity mask.

        Args:
            element_ids: List of element identifiers.
            segment_counts: Dict mapping element_id -> number of segments.
            dim_name: Name for the element dimension.

        Returns:
            max_segments: Maximum segment count across all elements.
            segment_mask: (element, segment) DataArray, 1=valid, 0=padded.
        """
        max_segments = max(segment_counts.values())

        # Build segment validity mask
        mask_data = np.zeros((len(element_ids), max_segments))
        for i, eid in enumerate(element_ids):
            n_segments = segment_counts[eid]
            mask_data[i, :n_segments] = 1

        segment_mask = xr.DataArray(
            mask_data,
            dims=[dim_name, 'segment'],
            coords={dim_name: element_ids, 'segment': list(range(max_segments))},
        )

        return max_segments, segment_mask

    @staticmethod
    def pad_breakpoints(
        element_ids: list[str],
        breakpoints: dict[str, tuple[list, list]],
        max_segments: int,
        dim_name: str,
        time_coords: xr.DataArray | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Pad breakpoints to (element, segment) or (element, segment, time) arrays.

        Handles both scalar and time-varying (array) breakpoints.

        Args:
            element_ids: List of element identifiers.
            breakpoints: Dict mapping element_id -> (starts, ends) lists.
                Values can be scalars or time-varying arrays.
            max_segments: Maximum segment count to pad to.
            dim_name: Name for the element dimension.
            time_coords: Optional time coordinates for time-varying breakpoints.

        Returns:
            starts: (element, segment) or (element, segment, time) DataArray.
            ends: (element, segment) or (element, segment, time) DataArray.
        """
        # Detect if any breakpoints are time-varying (arrays/xr.DataArray with dim > 0)
        is_time_varying = False
        time_length = None
        for eid in element_ids:
            element_starts, element_ends = breakpoints[eid]
            for val in list(element_starts) + list(element_ends):
                if isinstance(val, xr.DataArray):
                    # Check if it has any dimensions (not a scalar)
                    if val.ndim > 0:
                        is_time_varying = True
                        time_length = val.shape[0]
                        break
                elif isinstance(val, np.ndarray):
                    # Check if it's not a 0-d array
                    if val.ndim > 0 and val.size > 1:
                        is_time_varying = True
                        time_length = len(val)
                        break
            if is_time_varying:
                break

        if is_time_varying and time_length is not None:
            # 3D arrays: (element, segment, time)
            starts_data = np.zeros((len(element_ids), max_segments, time_length))
            ends_data = np.zeros((len(element_ids), max_segments, time_length))

            for i, eid in enumerate(element_ids):
                element_starts, element_ends = breakpoints[eid]
                n_segments = len(element_starts)
                for j in range(n_segments):
                    start_val = element_starts[j]
                    end_val = element_ends[j]
                    # Handle scalar vs array values
                    if isinstance(start_val, (np.ndarray, xr.DataArray)):
                        starts_data[i, j, :] = np.asarray(start_val)
                    else:
                        starts_data[i, j, :] = start_val
                    if isinstance(end_val, (np.ndarray, xr.DataArray)):
                        ends_data[i, j, :] = np.asarray(end_val)
                    else:
                        ends_data[i, j, :] = end_val

            # Build coordinates including time if available
            coords = {dim_name: element_ids, 'segment': list(range(max_segments))}
            if time_coords is not None:
                coords['time'] = time_coords
            starts = xr.DataArray(starts_data, dims=[dim_name, 'segment', 'time'], coords=coords)
            ends = xr.DataArray(ends_data, dims=[dim_name, 'segment', 'time'], coords=coords)
        else:
            # 2D arrays: (element, segment) - scalar breakpoints
            starts_data = np.zeros((len(element_ids), max_segments))
            ends_data = np.zeros((len(element_ids), max_segments))

            for i, eid in enumerate(element_ids):
                element_starts, element_ends = breakpoints[eid]
                n_segments = len(element_starts)
                starts_data[i, :n_segments] = element_starts
                ends_data[i, :n_segments] = element_ends

            coords = {dim_name: element_ids, 'segment': list(range(max_segments))}
            starts = xr.DataArray(starts_data, dims=[dim_name, 'segment'], coords=coords)
            ends = xr.DataArray(ends_data, dims=[dim_name, 'segment'], coords=coords)

        return starts, ends

    @staticmethod
    def create_piecewise_variables(
        model: FlowSystemModel,
        element_ids: list[str],
        max_segments: int,
        dim_name: str,
        segment_mask: xr.DataArray,
        base_coords: xr.Coordinates | None,
        name_prefix: str,
    ) -> dict[str, linopy.Variable]:
        """Create batched piecewise variables.

        Args:
            model: The FlowSystemModel.
            element_ids: List of element identifiers.
            max_segments: Number of segments (after padding).
            dim_name: Name for the element dimension.
            segment_mask: (element, segment) validity mask.
            base_coords: Additional coordinates (time, period, scenario).
            name_prefix: Prefix for variable names.

        Returns:
            Dict with 'inside_piece', 'lambda0', 'lambda1' variables.
        """
        import pandas as pd

        # Build coordinates
        coords_dict = {
            dim_name: pd.Index(element_ids, name=dim_name),
            'segment': pd.Index(list(range(max_segments)), name='segment'),
        }
        if base_coords is not None:
            coords_dict.update(dict(base_coords))

        full_coords = xr.Coordinates(coords_dict)

        # inside_piece: binary, but upper=0 for padded segments
        inside_piece = model.add_variables(
            lower=0,
            upper=segment_mask,  # 0 for padded, 1 for valid
            binary=True,
            coords=full_coords,
            name=f'{name_prefix}|inside_piece',
        )

        # lambda0, lambda1: continuous [0, 1], but upper=0 for padded segments
        lambda0 = model.add_variables(
            lower=0,
            upper=segment_mask,
            coords=full_coords,
            name=f'{name_prefix}|lambda0',
        )

        lambda1 = model.add_variables(
            lower=0,
            upper=segment_mask,
            coords=full_coords,
            name=f'{name_prefix}|lambda1',
        )

        return {
            'inside_piece': inside_piece,
            'lambda0': lambda0,
            'lambda1': lambda1,
        }

    @staticmethod
    def create_piecewise_constraints(
        model: FlowSystemModel,
        variables: dict[str, linopy.Variable],
        segment_mask: xr.DataArray,
        zero_point: linopy.Variable | xr.DataArray | None,
        dim_name: str,
        name_prefix: str,
    ) -> None:
        """Create batched piecewise constraints.

        Creates:
            - lambda0 + lambda1 == inside_piece (for valid segments only)
            - sum(inside_piece, segment) <= 1 or zero_point

        Args:
            model: The FlowSystemModel.
            variables: Dict with 'inside_piece', 'lambda0', 'lambda1'.
            segment_mask: (element, segment) validity mask.
            zero_point: Optional variable/array for zero-point constraint.
            dim_name: Name for the element dimension.
            name_prefix: Prefix for constraint names.
        """
        inside_piece = variables['inside_piece']
        lambda0 = variables['lambda0']
        lambda1 = variables['lambda1']

        # Constraint: lambda0 + lambda1 == inside_piece (only for valid segments)
        # For padded segments, all variables are 0, so constraint is 0 == 0 (trivially satisfied)
        model.add_constraints(
            lambda0 + lambda1 == inside_piece,
            name=f'{name_prefix}|lambda_sum',
        )

        # Constraint: sum(inside_piece) <= 1 (or <= zero_point)
        # This ensures at most one segment is active per element
        rhs = 1 if zero_point is None else zero_point
        model.add_constraints(
            inside_piece.sum('segment') <= rhs,
            name=f'{name_prefix}|single_segment',
        )

    @staticmethod
    def create_coupling_constraint(
        model: FlowSystemModel,
        target_var: linopy.Variable,
        lambda0: linopy.Variable,
        lambda1: linopy.Variable,
        starts: xr.DataArray,
        ends: xr.DataArray,
        name: str,
    ) -> None:
        """Create variable coupling constraint.

        Creates: target_var == sum(lambda0 * starts + lambda1 * ends, segment)

        Args:
            model: The FlowSystemModel.
            target_var: The variable to couple (e.g., flow_rate, size).
            lambda0: Lambda0 variable from create_piecewise_variables.
            lambda1: Lambda1 variable from create_piecewise_variables.
            starts: (element, segment) array of segment start values.
            ends: (element, segment) array of segment end values.
            name: Name for the constraint.
        """
        reconstructed = (lambda0 * starts + lambda1 * ends).sum('segment')
        model.add_constraints(target_var == reconstructed, name=name)
