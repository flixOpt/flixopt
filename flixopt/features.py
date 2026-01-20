"""
This module contains the features of the flixopt framework.
Features extend the functionality of Elements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import linopy
import numpy as np
import xarray as xr

from .modeling import BoundingPatterns
from .structure import FlowSystemModel, Submodel, VariableCategory

if TYPE_CHECKING:
    from collections.abc import Collection

    from .core import FlowSystemDimensions
    from .interface import (
        InvestParameters,
        Piecewise,
        StatusParameters,
    )
    from .types import Numeric_PS, Numeric_TPS


# =============================================================================
# Helper functions for shared constraint math
# =============================================================================


class InvestmentHelpers:
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

        Args:
            model: The FlowSystemModel to add constraints to.
            size_var: Size variable.
            params: Dict mapping element_id -> InvestParameters.
            element_ids: List of all element IDs.
            dim_name: Dimension name (e.g., 'flow', 'storage').
        """
        element_ids_with_linking = [eid for eid in element_ids if params[eid].linked_periods is not None]
        if not element_ids_with_linking:
            return

        for element_id in element_ids_with_linking:
            linked = params[element_id].linked_periods
            element_size = size_var.sel({dim_name: element_id})
            masked_size = element_size.where(linked, drop=True)
            if 'period' in masked_size.dims and masked_size.sizes.get('period', 0) > 1:
                model.add_constraints(
                    masked_size.isel(period=slice(None, -1)) == masked_size.isel(period=slice(1, None)),
                    name=f'{element_id}|linked_periods',
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
        """Build factor array with (element, effect) dims from effects dict.

        Args:
            effects_dict: Dict mapping effect_name -> DataArray(element_dim).
            element_ids: Element IDs (for ordering).
            dim_name: Element dimension name.

        Returns:
            DataArray with (element, effect) dims, or None if empty.
        """
        if not effects_dict:
            return None

        effect_ids = list(effects_dict.keys())
        effect_arrays = [effects_dict[eff] for eff in effect_ids]
        result = xr.concat(effect_arrays, dim='effect').assign_coords(effect=effect_ids)

        return result.transpose(dim_name, 'effect')

    @staticmethod
    def stack_bounds(
        bounds: list[float | xr.DataArray],
        element_ids: list[str],
        dim_name: str,
    ) -> xr.DataArray | float:
        """Stack per-element bounds into array with element dimension.

        Args:
            bounds: List of bounds (one per element).
            element_ids: List of element IDs (same order as bounds).
            dim_name: Dimension name (e.g., 'flow', 'storage').

        Returns:
            Stacked DataArray with element dimension, or scalar if all identical.
        """
        # Extract scalar values from 0-d DataArrays or plain scalars
        scalar_values = []
        has_multidim = False

        for b in bounds:
            if isinstance(b, xr.DataArray):
                if b.ndim == 0:
                    scalar_values.append(float(b.values))
                else:
                    has_multidim = True
                    break
            else:
                scalar_values.append(float(b))

        # Fast path: all scalars
        if not has_multidim:
            unique_values = set(scalar_values)
            if len(unique_values) == 1:
                return scalar_values[0]  # Return scalar - linopy will broadcast

            return xr.DataArray(
                np.array(scalar_values),
                coords={dim_name: element_ids},
                dims=[dim_name],
            )

        # Slow path: need full concat for multi-dimensional bounds
        arrays_to_stack = []
        for bound, eid in zip(bounds, element_ids, strict=False):
            if isinstance(bound, xr.DataArray):
                arr = bound.expand_dims({dim_name: [eid]})
            else:
                arr = xr.DataArray(bound, coords={dim_name: [eid]}, dims=[dim_name])
            arrays_to_stack.append(arr)

        # Find union of all non-element dimensions and their coords
        all_dims = {}
        for arr in arrays_to_stack:
            for d in arr.dims:
                if d != dim_name and d not in all_dims:
                    all_dims[d] = arr.coords[d].values

        # Expand each array to have all dimensions
        expanded = []
        for arr in arrays_to_stack:
            for d, coords in all_dims.items():
                if d not in arr.dims:
                    arr = arr.expand_dims({d: coords})
            expanded.append(arr)

        return xr.concat(expanded, dim=dim_name, coords='minimal')


class StatusHelpers:
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
    def collect_status_effects(
        params: dict[str, StatusParameters],
        element_ids: list[str],
        attr: str,
        dim_name: str,
    ) -> dict[str, xr.DataArray]:
        """Collect status effects from params into a dict of DataArrays.

        Args:
            params: Dict mapping element_id -> StatusParameters.
            element_ids: List of element IDs to collect from.
            attr: Attribute name on StatusParameters (e.g., 'effects_per_active_hour').
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
            upper_bound = xr.where(maximum_duration.notnull(), maximum_duration, mega)
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
            has_previous = previous_duration.notnull()
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
    def create_status_features(
        model: FlowSystemModel,
        status: linopy.Variable,
        params: dict[str, StatusParameters],
        dim_name: str,
        var_names,  # FlowVarName or ComponentVarName class
        previous_status: dict[str, xr.DataArray] | None = None,
        has_clusters: bool = False,
    ) -> dict[str, linopy.Variable]:
        """Create all status-derived variables and constraints.

        This is the main entry point for status feature creation. Given a status
        variable (created by the caller), this method creates all derived variables
        and constraints for status tracking.

        Creates variables:
        - active_hours: For all elements with status
        - startup, shutdown: For elements needing startup tracking
        - inactive: For elements needing downtime tracking
        - startup_count: For elements with startup limit
        - uptime, downtime: Duration tracking variables

        Creates constraints:
        - active_hours tracking
        - complementary (status + inactive == 1)
        - switch_transition, switch_mutex, switch_initial
        - startup_count tracking
        - uptime/downtime duration tracking
        - cluster_cyclic (if has_clusters)

        Args:
            model: The FlowSystemModel to add variables/constraints to.
            status: Batched binary status variable with (element_dim, time) dims.
            params: Dict mapping element_id -> StatusParameters.
            dim_name: Element dimension name (e.g., 'flow', 'component').
            var_names: Class with variable/constraint name constants (e.g., FlowVarName).
            previous_status: Optional dict mapping element_id -> previous status DataArray.
            has_clusters: Whether to check for cluster cyclic constraints.

        Returns:
            Dict of created variables (active_hours, startup, shutdown, inactive, startup_count, uptime, downtime).
        """
        import pandas as pd

        if previous_status is None:
            previous_status = {}

        element_ids = list(params.keys())
        variables: dict[str, linopy.Variable] = {}

        # === Compute category lists ===
        startup_tracking_ids = [
            eid
            for eid in element_ids
            if (
                params[eid].effects_per_startup
                or params[eid].min_uptime is not None
                or params[eid].max_uptime is not None
                or params[eid].startup_limit is not None
                or params[eid].force_startup_tracking
            )
        ]
        downtime_tracking_ids = [
            eid for eid in element_ids if params[eid].min_downtime is not None or params[eid].max_downtime is not None
        ]
        uptime_tracking_ids = [
            eid for eid in element_ids if params[eid].min_uptime is not None or params[eid].max_uptime is not None
        ]
        startup_limit_ids = [eid for eid in element_ids if params[eid].startup_limit is not None]

        # === Get coords ===
        base_coords = model.get_coords(['period', 'scenario'])
        base_coords_dict = dict(base_coords) if base_coords is not None else {}
        temporal_coords = model.get_coords()
        total_hours = model.temporal_weight.sum(model.temporal_dims)
        timestep_duration = model.timestep_duration

        # === VARIABLES ===

        # active_hours: For ALL elements with status
        active_hours_min_vals = [params[eid].active_hours_min or 0 for eid in element_ids]
        active_hours_min = xr.DataArray(active_hours_min_vals, dims=[dim_name], coords={dim_name: element_ids})

        active_hours_max_list = [params[eid].active_hours_max for eid in element_ids]
        has_max = xr.DataArray(
            [v is not None for v in active_hours_max_list], dims=[dim_name], coords={dim_name: element_ids}
        )
        max_vals = xr.DataArray(
            [v if v is not None else 0 for v in active_hours_max_list], dims=[dim_name], coords={dim_name: element_ids}
        )
        active_hours_max = xr.where(has_max, max_vals, total_hours)

        active_hours_coords = xr.Coordinates({dim_name: pd.Index(element_ids, name=dim_name), **base_coords_dict})
        variables['active_hours'] = model.add_variables(
            lower=active_hours_min,
            upper=active_hours_max,
            coords=active_hours_coords,
            name=var_names.ACTIVE_HOURS,
        )

        # startup, shutdown: For elements with startup tracking
        if startup_tracking_ids:
            startup_coords = xr.Coordinates(
                {dim_name: pd.Index(startup_tracking_ids, name=dim_name), **dict(temporal_coords)}
            )
            variables['startup'] = model.add_variables(binary=True, coords=startup_coords, name=var_names.STARTUP)
            variables['shutdown'] = model.add_variables(binary=True, coords=startup_coords, name=var_names.SHUTDOWN)

        # inactive: For elements with downtime tracking
        if downtime_tracking_ids:
            inactive_coords = xr.Coordinates(
                {dim_name: pd.Index(downtime_tracking_ids, name=dim_name), **dict(temporal_coords)}
            )
            variables['inactive'] = model.add_variables(binary=True, coords=inactive_coords, name=var_names.INACTIVE)

        # startup_count: For elements with startup limit
        if startup_limit_ids:
            startup_limit_vals = [params[eid].startup_limit for eid in startup_limit_ids]
            startup_limit = xr.DataArray(startup_limit_vals, dims=[dim_name], coords={dim_name: startup_limit_ids})
            startup_count_coords = xr.Coordinates(
                {dim_name: pd.Index(startup_limit_ids, name=dim_name), **base_coords_dict}
            )
            variables['startup_count'] = model.add_variables(
                lower=0, upper=startup_limit, coords=startup_count_coords, name=var_names.STARTUP_COUNT
            )

        # === CONSTRAINTS ===

        # active_hours tracking: sum(status * weight) == active_hours
        model.add_constraints(
            variables['active_hours'] == model.sum_temporal(status),
            name=var_names.Constraint.ACTIVE_HOURS,
        )

        # inactive complementary: status + inactive == 1
        if downtime_tracking_ids:
            status_subset = status.sel({dim_name: downtime_tracking_ids})
            inactive = variables['inactive']
            model.add_constraints(status_subset + inactive == 1, name=var_names.Constraint.COMPLEMENTARY)

        # State transitions: startup, shutdown
        if startup_tracking_ids:
            status_subset = status.sel({dim_name: startup_tracking_ids})
            startup = variables['startup']
            shutdown = variables['shutdown']

            # Transition constraint for t > 0
            model.add_constraints(
                startup.isel(time=slice(1, None)) - shutdown.isel(time=slice(1, None))
                == status_subset.isel(time=slice(1, None)) - status_subset.isel(time=slice(None, -1)),
                name=var_names.Constraint.SWITCH_TRANSITION,
            )

            # Mutex constraint
            model.add_constraints(startup + shutdown <= 1, name=var_names.Constraint.SWITCH_MUTEX)

            # Initial constraint for t = 0 (if previous_status available)
            elements_with_initial = [eid for eid in startup_tracking_ids if eid in previous_status]
            if elements_with_initial:
                prev_arrays = [previous_status[eid].expand_dims({dim_name: [eid]}) for eid in elements_with_initial]
                prev_status_batched = xr.concat(prev_arrays, dim=dim_name)
                prev_state = prev_status_batched.isel(time=-1)
                startup_subset = startup.sel({dim_name: elements_with_initial})
                shutdown_subset = shutdown.sel({dim_name: elements_with_initial})
                status_initial = status_subset.sel({dim_name: elements_with_initial}).isel(time=0)

                model.add_constraints(
                    startup_subset.isel(time=0) - shutdown_subset.isel(time=0) == status_initial - prev_state,
                    name=var_names.Constraint.SWITCH_INITIAL,
                )

        # startup_count: sum(startup) == startup_count
        if startup_limit_ids:
            startup = variables['startup'].sel({dim_name: startup_limit_ids})
            startup_count = variables['startup_count']
            startup_temporal_dims = [d for d in startup.dims if d not in ('period', 'scenario', dim_name)]
            model.add_constraints(
                startup_count == startup.sum(startup_temporal_dims), name=var_names.Constraint.STARTUP_COUNT
            )

        # Uptime tracking (batched)
        if uptime_tracking_ids:
            min_uptime = xr.DataArray(
                [params[eid].min_uptime or np.nan for eid in uptime_tracking_ids],
                dims=[dim_name],
                coords={dim_name: uptime_tracking_ids},
            )
            max_uptime = xr.DataArray(
                [params[eid].max_uptime or np.nan for eid in uptime_tracking_ids],
                dims=[dim_name],
                coords={dim_name: uptime_tracking_ids},
            )
            # Build previous uptime DataArray
            previous_uptime_values = []
            for eid in uptime_tracking_ids:
                if eid in previous_status and params[eid].min_uptime is not None:
                    prev = StatusHelpers.compute_previous_duration(
                        previous_status[eid], target_state=1, timestep_duration=timestep_duration
                    )
                    previous_uptime_values.append(prev)
                else:
                    previous_uptime_values.append(np.nan)
            previous_uptime = xr.DataArray(
                previous_uptime_values, dims=[dim_name], coords={dim_name: uptime_tracking_ids}
            )

            variables['uptime'] = StatusHelpers.add_batched_duration_tracking(
                model=model,
                state=status.sel({dim_name: uptime_tracking_ids}),
                name=var_names.UPTIME,
                dim_name=dim_name,
                timestep_duration=timestep_duration,
                minimum_duration=min_uptime,
                maximum_duration=max_uptime,
                previous_duration=previous_uptime if previous_uptime.notnull().any() else None,
            )

        # Downtime tracking (batched)
        if downtime_tracking_ids:
            min_downtime = xr.DataArray(
                [params[eid].min_downtime or np.nan for eid in downtime_tracking_ids],
                dims=[dim_name],
                coords={dim_name: downtime_tracking_ids},
            )
            max_downtime = xr.DataArray(
                [params[eid].max_downtime or np.nan for eid in downtime_tracking_ids],
                dims=[dim_name],
                coords={dim_name: downtime_tracking_ids},
            )
            # Build previous downtime DataArray
            previous_downtime_values = []
            for eid in downtime_tracking_ids:
                if eid in previous_status and params[eid].min_downtime is not None:
                    prev = StatusHelpers.compute_previous_duration(
                        previous_status[eid], target_state=0, timestep_duration=timestep_duration
                    )
                    previous_downtime_values.append(prev)
                else:
                    previous_downtime_values.append(np.nan)
            previous_downtime = xr.DataArray(
                previous_downtime_values, dims=[dim_name], coords={dim_name: downtime_tracking_ids}
            )

            variables['downtime'] = StatusHelpers.add_batched_duration_tracking(
                model=model,
                state=variables['inactive'],
                name=var_names.DOWNTIME,
                dim_name=dim_name,
                timestep_duration=timestep_duration,
                minimum_duration=min_downtime,
                maximum_duration=max_downtime,
                previous_duration=previous_downtime if previous_downtime.notnull().any() else None,
            )

        # Cluster cyclic constraints
        if has_clusters:
            cyclic_ids = [eid for eid in element_ids if params[eid].cluster_mode == 'cyclic']
            if cyclic_ids:
                status_cyclic = status.sel({dim_name: cyclic_ids})
                model.add_constraints(
                    status_cyclic.isel(time=0) == status_cyclic.isel(time=-1),
                    name=var_names.Constraint.CLUSTER_CYCLIC,
                )

        return variables


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


class PiecewiseHelpers:
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
        breakpoints: dict[str, tuple[list[float], list[float]]],
        max_segments: int,
        dim_name: str,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Pad breakpoints to (element, segment) arrays.

        Args:
            element_ids: List of element identifiers.
            breakpoints: Dict mapping element_id -> (starts, ends) lists.
            max_segments: Maximum segment count to pad to.
            dim_name: Name for the element dimension.

        Returns:
            starts: (element, segment) DataArray of segment start values.
            ends: (element, segment) DataArray of segment end values.
        """
        starts_data = np.zeros((len(element_ids), max_segments))
        ends_data = np.zeros((len(element_ids), max_segments))

        for i, eid in enumerate(element_ids):
            element_starts, element_ends = breakpoints[eid]
            n_segments = len(element_starts)
            starts_data[i, :n_segments] = element_starts
            ends_data[i, :n_segments] = element_ends
            # Padded segments remain 0, which is fine since they're masked out

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


class InvestmentModel(Submodel):
    """Mathematical model implementation for investment decisions.

    Creates optimization variables and constraints for investment sizing decisions,
    supporting both binary and continuous sizing with comprehensive effect modeling.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/features/InvestParameters/>

    Args:
        model: The optimization model instance
        label_of_element: The label of the parent (Element). Used to construct the full label of the model.
        parameters: The parameters of the feature model.
        label_of_model: The label of the model. This is needed to construct the full label of the model.
        size_category: Category for the size variable (FLOW_SIZE, STORAGE_SIZE, or SIZE for generic).
    """

    parameters: InvestParameters

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        parameters: InvestParameters,
        label_of_model: str | None = None,
        size_category: VariableCategory = VariableCategory.SIZE,
    ):
        self.piecewise_effects: PiecewiseEffectsModel | None = None
        self.parameters = parameters
        self._size_category = size_category
        super().__init__(model, label_of_element=label_of_element, label_of_model=label_of_model)

    def _do_modeling(self):
        super()._do_modeling()
        self._create_variables_and_constraints()
        self._add_effects()

    def _create_variables_and_constraints(self):
        size_min, size_max = (self.parameters.minimum_or_fixed_size, self.parameters.maximum_or_fixed_size)
        if self.parameters.linked_periods is not None:
            # Mask size bounds: linked_periods is a binary DataArray that zeros out non-linked periods
            size_min = size_min * self.parameters.linked_periods
            size_max = size_max * self.parameters.linked_periods

        self.add_variables(
            short_name='size',
            lower=size_min if self.parameters.mandatory else 0,
            upper=size_max,
            coords=self._model.get_coords(['period', 'scenario']),
            category=self._size_category,
        )

        if not self.parameters.mandatory:
            self.add_variables(
                binary=True,
                coords=self._model.get_coords(['period', 'scenario']),
                short_name='invested',
                category=VariableCategory.INVESTED,
            )
            BoundingPatterns.bounds_with_state(
                self,
                variable=self.size,
                state=self._variables['invested'],
                bounds=(self.parameters.minimum_or_fixed_size, self.parameters.maximum_or_fixed_size),
            )

        if self.parameters.linked_periods is not None:
            masked_size = self.size.where(self.parameters.linked_periods, drop=True)
            self.add_constraints(
                masked_size.isel(period=slice(None, -1)) == masked_size.isel(period=slice(1, None)),
                short_name='linked_periods',
            )

    def _add_effects(self):
        """Add investment effects"""
        if self.parameters.effects_of_investment:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.invested * factor if self.invested is not None else factor
                    for effect, factor in self.parameters.effects_of_investment.items()
                },
                target='periodic',
            )

        if self.parameters.effects_of_retirement and not self.parameters.mandatory:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: -self.invested * factor + factor
                    for effect, factor in self.parameters.effects_of_retirement.items()
                },
                target='periodic',
            )

        if self.parameters.effects_of_investment_per_size:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.size * factor
                    for effect, factor in self.parameters.effects_of_investment_per_size.items()
                },
                target='periodic',
            )

        if self.parameters.piecewise_effects_of_investment:
            self.piecewise_effects = self.add_submodels(
                PiecewiseEffectsModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    label_of_model=f'{self.label_of_element}|PiecewiseEffects',
                    piecewise_origin=(self.size.name, self.parameters.piecewise_effects_of_investment.piecewise_origin),
                    piecewise_shares=self.parameters.piecewise_effects_of_investment.piecewise_shares,
                    zero_point=self.invested,
                ),
                short_name='segments',
            )

    @property
    def size(self) -> linopy.Variable:
        """Investment size variable"""
        return self._variables['size']

    @property
    def invested(self) -> linopy.Variable | None:
        """Binary investment decision variable"""
        if 'invested' not in self._variables:
            return None
        return self._variables['invested']


class InvestmentProxy:
    """Proxy providing access to investment variables for a specific element.

    This class provides the same interface as InvestmentModel.size/invested
    but returns slices from the batched variables in FlowsModel/StoragesModel.
    """

    def __init__(self, parent_model, element_id: str, dim_name: str = 'flow'):
        self._parent_model = parent_model
        self._element_id = element_id
        self._dim_name = dim_name

    @property
    def size(self):
        """Investment size variable for this element."""
        size_var = self._parent_model._variables.get('size')
        if size_var is None:
            return None
        if self._element_id in size_var.coords.get(self._dim_name, []):
            return size_var.sel({self._dim_name: self._element_id})
        return None

    @property
    def invested(self):
        """Binary investment decision variable for this element (if non-mandatory)."""
        invested_var = self._parent_model._variables.get('invested')
        if invested_var is None:
            return None
        if self._element_id in invested_var.coords.get(self._dim_name, []):
            return invested_var.sel({self._dim_name: self._element_id})
        return None


# =============================================================================
# DEPRECATED: InvestmentsModel classes have been inlined into FlowsModel and StoragesModel
# The investment logic now lives directly in:
# - FlowsModel.create_investment_model() in elements.py
# - StoragesModel.create_investment_model() in components.py
# Using InvestmentHelpers for shared constraint math.
# =============================================================================


class StatusProxy:
    """Proxy providing access to batched status variables for a specific element.

    Provides access to status-related variables for a specific element.
    Returns slices from batched variables. Works with both FlowsModel
    (for flows) and ComponentsModel (for components).
    """

    def __init__(self, model, element_id: str):
        """Initialize proxy.

        Args:
            model: FlowsModel or StatusesModel with get_variable method and _previous_status dict.
            element_id: Element identifier for selecting from batched variables.
        """
        self._model = model
        self._element_id = element_id

    @property
    def status(self):
        """Binary status variable for this element."""
        return self._model.get_variable('status', self._element_id)

    @property
    def active_hours(self):
        """Total active hours variable for this element."""
        return self._model.get_variable('active_hours', self._element_id)

    @property
    def startup(self):
        """Startup variable for this element."""
        return self._model.get_variable('startup', self._element_id)

    @property
    def shutdown(self):
        """Shutdown variable for this element."""
        return self._model.get_variable('shutdown', self._element_id)

    @property
    def inactive(self):
        """Inactive variable for this element."""
        return self._model.get_variable('inactive', self._element_id)

    @property
    def startup_count(self):
        """Startup count variable for this element."""
        return self._model.get_variable('startup_count', self._element_id)

    @property
    def _previous_status(self):
        """Previous status for this element."""
        # Handle both FlowsModel (_previous_status) and StatusesModel (previous_status)
        prev_dict = getattr(self._model, '_previous_status', None) or getattr(self._model, 'previous_status', {})
        return prev_dict.get(self._element_id)


class PieceModel(Submodel):
    """Class for modeling a linear piece of one or more variables in parallel"""

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        label_of_model: str,
        dims: Collection[FlowSystemDimensions] | None,
    ):
        self.inside_piece: linopy.Variable | None = None
        self.lambda0: linopy.Variable | None = None
        self.lambda1: linopy.Variable | None = None
        self.dims = dims

        super().__init__(model, label_of_element, label_of_model)

    def _do_modeling(self):
        """Create variables, constraints, and nested submodels"""
        super()._do_modeling()

        # Create variables
        self.inside_piece = self.add_variables(
            binary=True,
            short_name='inside_piece',
            coords=self._model.get_coords(dims=self.dims),
            category=VariableCategory.INSIDE_PIECE,
        )
        self.lambda0 = self.add_variables(
            lower=0,
            upper=1,
            short_name='lambda0',
            coords=self._model.get_coords(dims=self.dims),
            category=VariableCategory.LAMBDA0,
        )

        self.lambda1 = self.add_variables(
            lower=0,
            upper=1,
            short_name='lambda1',
            coords=self._model.get_coords(dims=self.dims),
            category=VariableCategory.LAMBDA1,
        )

        # Create constraints
        # eq:  lambda0(t) + lambda1(t) = inside_piece(t)
        self.add_constraints(self.inside_piece == self.lambda0 + self.lambda1, short_name='inside_piece')


class PiecewiseModel(Submodel):
    """Mathematical model implementation for piecewise linear approximations.

    Creates optimization variables and constraints for piecewise linear relationships,
    including lambda variables, piece activation binaries, and coupling constraints.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/features/Piecewise/>
    """

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        label_of_model: str,
        piecewise_variables: dict[str, Piecewise],
        zero_point: bool | linopy.Variable | None,
        dims: Collection[FlowSystemDimensions] | None,
    ):
        """
        Modeling a Piecewise relation between miultiple variables.
        The relation is defined by a list of Pieces, which are assigned to the variables.
        Each Piece is a tuple of (start, end).

        Args:
            model: The FlowSystemModel that is used to create the model.
            label_of_element: The label of the parent (Element). Used to construct the full label of the model.
            label_of_model: The label of the model. Used to construct the full label of the model.
            piecewise_variables: The variables to which the Pieces are assigned.
            zero_point: A variable that can be used to define a zero point for the Piecewise relation. If None or False, no zero point is defined.
            dims: The dimensions used for variable creation. If None, all dimensions are used.
        """
        self._piecewise_variables = piecewise_variables
        self._zero_point = zero_point
        self.dims = dims

        self.pieces: list[PieceModel] = []
        self.zero_point: linopy.Variable | None = None
        super().__init__(model, label_of_element=label_of_element, label_of_model=label_of_model)

    def _do_modeling(self):
        """Create variables, constraints, and nested submodels"""
        super()._do_modeling()

        # Validate all piecewise variables have the same number of segments
        segment_counts = [len(pw) for pw in self._piecewise_variables.values()]
        if not all(count == segment_counts[0] for count in segment_counts):
            raise ValueError(f'All piecewises must have the same number of pieces, got {segment_counts}')

        # Create PieceModel submodels (which creates their variables and constraints)
        for i in range(len(list(self._piecewise_variables.values())[0])):
            new_piece = self.add_submodels(
                PieceModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    label_of_model=f'{self.label_of_element}|Piece_{i}',
                    dims=self.dims,
                ),
                short_name=f'Piece_{i}',
            )
            self.pieces.append(new_piece)

        for var_name in self._piecewise_variables:
            variable = self._model.variables[var_name]
            self.add_constraints(
                variable
                == sum(
                    [
                        piece_model.lambda0 * piece_bounds.start + piece_model.lambda1 * piece_bounds.end
                        for piece_model, piece_bounds in zip(
                            self.pieces, self._piecewise_variables[var_name], strict=False
                        )
                    ]
                ),
                name=f'{self.label_full}|{var_name}|lambda',
                short_name=f'{var_name}|lambda',
            )

            # a) eq: Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 1                Aufenthalt nur in Segmenten erlaubt
            # b) eq: -On(t) + Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 0       zusätzlich kann alles auch Null sein
            if isinstance(self._zero_point, linopy.Variable):
                self.zero_point = self._zero_point
                rhs = self.zero_point
            elif self._zero_point is True:
                self.zero_point = self.add_variables(
                    coords=self._model.get_coords(self.dims),
                    binary=True,
                    short_name='zero_point',
                    category=VariableCategory.ZERO_POINT,
                )
                rhs = self.zero_point
            else:
                rhs = 1

            # This constraint ensures at most one segment is active at a time.
            # When zero_point is a binary variable, it acts as a gate:
            # - zero_point=1: at most one segment can be active (normal piecewise operation)
            # - zero_point=0: all segments must be inactive (effectively disables the piecewise)
            self.add_constraints(
                sum([piece.inside_piece for piece in self.pieces]) <= rhs,
                name=f'{self.label_full}|{variable.name}|single_segment',
                short_name=f'{var_name}|single_segment',
            )


class PiecewiseEffectsModel(Submodel):
    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        label_of_model: str,
        piecewise_origin: tuple[str, Piecewise],
        piecewise_shares: dict[str, Piecewise],
        zero_point: bool | linopy.Variable | None,
    ):
        origin_count = len(piecewise_origin[1])
        share_counts = [len(pw) for pw in piecewise_shares.values()]
        if not all(count == origin_count for count in share_counts):
            raise ValueError(
                f'Piece count mismatch: piecewise_origin has {origin_count} segments, '
                f'but piecewise_shares have {share_counts}'
            )
        self._zero_point = zero_point
        self._piecewise_origin = piecewise_origin
        self._piecewise_shares = piecewise_shares
        self.shares: dict[str, linopy.Variable] = {}

        self.piecewise_model: PiecewiseModel | None = None

        super().__init__(model, label_of_element=label_of_element, label_of_model=label_of_model)

    def _do_modeling(self):
        """Create variables, constraints, and nested submodels"""
        super()._do_modeling()

        # Create variables
        self.shares = {
            effect: self.add_variables(coords=self._model.get_coords(['period', 'scenario']), short_name=effect)
            for effect in self._piecewise_shares
        }

        piecewise_variables = {
            self._piecewise_origin[0]: self._piecewise_origin[1],
            **{
                self.shares[effect_label].name: self._piecewise_shares[effect_label]
                for effect_label in self._piecewise_shares
            },
        }

        # Create piecewise model (which creates its variables and constraints)
        self.piecewise_model = self.add_submodels(
            PiecewiseModel(
                model=self._model,
                label_of_element=self.label_of_element,
                piecewise_variables=piecewise_variables,
                zero_point=self._zero_point,
                dims=('period', 'scenario'),
                label_of_model=f'{self.label_of_element}|PiecewiseEffects',
            ),
            short_name='PiecewiseEffects',
        )

        # Add shares to effects
        self._model.effects.add_share_to_effects(
            name=self.label_of_element,
            expressions={effect: variable * 1 for effect, variable in self.shares.items()},
            target='periodic',
        )


class ShareAllocationModel(Submodel):
    def __init__(
        self,
        model: FlowSystemModel,
        dims: list[FlowSystemDimensions],
        label_of_element: str | None = None,
        label_of_model: str | None = None,
        total_max: Numeric_PS | None = None,
        total_min: Numeric_PS | None = None,
        max_per_hour: Numeric_TPS | None = None,
        min_per_hour: Numeric_TPS | None = None,
    ):
        if 'time' not in dims and (max_per_hour is not None or min_per_hour is not None):
            raise ValueError("max_per_hour and min_per_hour require 'time' dimension in dims")

        self._dims = dims
        self.total_per_timestep: linopy.Variable | None = None
        self.total: linopy.Variable | None = None
        self.shares: dict[str, linopy.Variable] = {}
        self.share_constraints: dict[str, linopy.Constraint] = {}

        self._eq_total_per_timestep: linopy.Constraint | None = None
        self._eq_total: linopy.Constraint | None = None

        # Parameters
        self._total_max = total_max
        self._total_min = total_min
        self._max_per_hour = max_per_hour
        self._min_per_hour = min_per_hour

        super().__init__(model, label_of_element=label_of_element, label_of_model=label_of_model)

    def _do_modeling(self):
        """Create variables, constraints, and nested submodels"""
        super()._do_modeling()

        # Create variables
        self.total = self.add_variables(
            lower=self._total_min if self._total_min is not None else -np.inf,
            upper=self._total_max if self._total_max is not None else np.inf,
            coords=self._model.get_coords([dim for dim in self._dims if dim != 'time']),
            name=self.label_full,
            short_name='total',
            category=VariableCategory.TOTAL,
        )
        # eq: sum = sum(share_i) # skalar
        self._eq_total = self.add_constraints(self.total == 0, name=self.label_full)

        if 'time' in self._dims:
            self.total_per_timestep = self.add_variables(
                lower=-np.inf if (self._min_per_hour is None) else self._min_per_hour * self._model.timestep_duration,
                upper=np.inf if (self._max_per_hour is None) else self._max_per_hour * self._model.timestep_duration,
                coords=self._model.get_coords(self._dims),
                short_name='per_timestep',
                category=VariableCategory.PER_TIMESTEP,
            )

            self._eq_total_per_timestep = self.add_constraints(self.total_per_timestep == 0, short_name='per_timestep')

            # Add it to the total (cluster_weight handles cluster representation, defaults to 1.0)
            # Sum over all temporal dimensions (time, and cluster if present)
            weighted_per_timestep = self.total_per_timestep * self._model.weights.get('cluster', 1.0)
            self._eq_total.lhs -= weighted_per_timestep.sum(dim=self._model.temporal_dims)

    def add_share(
        self,
        name: str,
        expression: linopy.LinearExpression,
        dims: list[FlowSystemDimensions] | None = None,
    ):
        """
        Add a share to the share allocation model. If the share already exists, the expression is added to the existing share.
        The expression is added to the right hand side (rhs) of the constraint.
        The variable representing the total share is on the left hand side (lhs) of the constraint.
        var_total = sum(expressions)

        Args:
            name: The name of the share.
            expression: The expression of the share. Added to the right hand side of the constraint.
            dims: The dimensions of the share. Defaults to all dimensions. Dims are ordered automatically
        """
        if dims is None:
            dims = self._dims
        else:
            if 'time' in dims and 'time' not in self._dims:
                raise ValueError('Cannot add share with time-dim to a model without time-dim')
            if 'period' in dims and 'period' not in self._dims:
                raise ValueError('Cannot add share with period-dim to a model without period-dim')
            if 'scenario' in dims and 'scenario' not in self._dims:
                raise ValueError('Cannot add share with scenario-dim to a model without scenario-dim')

        if name in self.shares:
            self.share_constraints[name].lhs -= expression
        else:
            # Temporal shares (with 'time' dim) are segment totals that need division
            category = VariableCategory.SHARE if 'time' in dims else None
            self.shares[name] = self.add_variables(
                coords=self._model.get_coords(dims),
                name=f'{name}->{self.label_full}',
                short_name=name,
                category=category,
            )

            self.share_constraints[name] = self.add_constraints(
                self.shares[name] == expression, name=f'{name}->{self.label_full}'
            )

            if 'time' not in dims:
                self._eq_total.lhs -= self.shares[name]
            else:
                self._eq_total_per_timestep.lhs -= self.shares[name]
