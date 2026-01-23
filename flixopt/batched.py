"""
Batched data containers for FlowSystem elements.

These classes provide indexed/batched access to element properties,
separating data management from mathematical modeling.

Usage:
    flow_system.batched.flows  # Access FlowsData
    flow_system.batched.storages  # Access StoragesData (future)
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from .features import concat_with_coords
from .interface import InvestParameters, StatusParameters
from .structure import ElementContainer

if TYPE_CHECKING:
    from .elements import Flow
    from .flow_system import FlowSystem


class FlowsData:
    """Batched data container for all flows with indexed access.

    Provides:
    - Element lookup by label: `flows['Boiler(gas_in)']` or `flows.get('label')`
    - Categorizations as list[str]: `flows.with_status`, `flows.with_investment`
    - Batched parameters as xr.DataArray with flow dimension

    This separates data access from mathematical modeling (FlowsModel).
    """

    def __init__(self, flows: list[Flow], flow_system: FlowSystem):
        """Initialize FlowsData.

        Args:
            flows: List of all Flow elements.
            flow_system: Parent FlowSystem for model coordinates.
        """
        self.elements: ElementContainer[Flow] = ElementContainer(flows)
        self._fs = flow_system

    def __getitem__(self, label: str) -> Flow:
        """Get a flow by its label_full."""
        return self.elements[label]

    def get(self, label: str, default: Flow | None = None) -> Flow | None:
        """Get a flow by label, returning default if not found."""
        return self.elements.get(label, default)

    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self):
        """Iterate over flow IDs."""
        return iter(self.elements)

    @property
    def ids(self) -> list[str]:
        """List of all flow IDs (label_full)."""
        return list(self.elements.keys())

    # === Flow Categorizations ===
    # All return list[str] of label_full IDs.

    @cached_property
    def with_status(self) -> list[str]:
        """IDs of flows with status parameters."""
        return [f.label_full for f in self.elements.values() if f.status_parameters is not None]

    @cached_property
    def with_startup_tracking(self) -> list[str]:
        """IDs of flows that need startup/shutdown tracking.

        Includes flows with: effects_per_startup, min/max_uptime, startup_limit, or force_startup_tracking.
        """
        result = []
        for fid in self.with_status:
            p = self.status_params[fid]
            if (
                p.effects_per_startup
                or p.min_uptime is not None
                or p.max_uptime is not None
                or p.startup_limit is not None
                or p.force_startup_tracking
            ):
                result.append(fid)
        return result

    @cached_property
    def with_downtime_tracking(self) -> list[str]:
        """IDs of flows that need downtime (inactive) tracking."""
        return [
            fid
            for fid in self.with_status
            if self.status_params[fid].min_downtime is not None or self.status_params[fid].max_downtime is not None
        ]

    @cached_property
    def with_uptime_tracking(self) -> list[str]:
        """IDs of flows that need uptime duration tracking."""
        return [
            fid
            for fid in self.with_status
            if self.status_params[fid].min_uptime is not None or self.status_params[fid].max_uptime is not None
        ]

    @cached_property
    def with_startup_limit(self) -> list[str]:
        """IDs of flows with startup limit."""
        return [fid for fid in self.with_status if self.status_params[fid].startup_limit is not None]

    @cached_property
    def without_size(self) -> list[str]:
        """IDs of flows with status parameters."""
        return [f.label_full for f in self.elements.values() if f.size is None]

    @cached_property
    def with_investment(self) -> list[str]:
        """IDs of flows with investment parameters."""
        return [f.label_full for f in self.elements.values() if isinstance(f.size, InvestParameters)]

    @cached_property
    def with_optional_investment(self) -> list[str]:
        """IDs of flows with optional (non-mandatory) investment."""
        return [fid for fid in self.with_investment if not self[fid].size.mandatory]

    @cached_property
    def with_mandatory_investment(self) -> list[str]:
        """IDs of flows with mandatory investment."""
        return [fid for fid in self.with_investment if self[fid].size.mandatory]

    @cached_property
    def with_flow_hours_min(self) -> list[str]:
        """IDs of flows with explicit flow_hours_min constraint."""
        return [f.label_full for f in self.elements.values() if f.flow_hours_min is not None]

    @cached_property
    def with_flow_hours_max(self) -> list[str]:
        """IDs of flows with explicit flow_hours_max constraint."""
        return [f.label_full for f in self.elements.values() if f.flow_hours_max is not None]

    @cached_property
    def with_flow_hours_over_periods_min(self) -> list[str]:
        """IDs of flows with explicit flow_hours_min_over_periods constraint."""
        return [f.label_full for f in self.elements.values() if f.flow_hours_min_over_periods is not None]

    @cached_property
    def with_flow_hours_over_periods_max(self) -> list[str]:
        """IDs of flows with explicit flow_hours_max_over_periods constraint."""
        return [f.label_full for f in self.elements.values() if f.flow_hours_max_over_periods is not None]

    @cached_property
    def with_load_factor_min(self) -> list[str]:
        """IDs of flows with explicit load_factor_min constraint."""
        return [f.label_full for f in self.elements.values() if f.load_factor_min is not None]

    @cached_property
    def with_load_factor_max(self) -> list[str]:
        """IDs of flows with explicit load_factor_max constraint."""
        return [f.label_full for f in self.elements.values() if f.load_factor_max is not None]

    @cached_property
    def with_effects(self) -> list[str]:
        """IDs of flows with effects_per_flow_hour defined."""
        return [f.label_full for f in self.elements.values() if f.effects_per_flow_hour]

    @cached_property
    def with_previous_flow_rate(self) -> list[str]:
        """IDs of flows with previous_flow_rate defined (for startup/shutdown tracking)."""
        return [f.label_full for f in self.elements.values() if f.previous_flow_rate is not None]

    # === Parameter Dicts ===

    @cached_property
    def invest_params(self) -> dict[str, InvestParameters]:
        """Investment parameters for flows with investment, keyed by label_full."""
        return {fid: self[fid].size for fid in self.with_investment}

    @cached_property
    def status_params(self) -> dict[str, StatusParameters]:
        """Status parameters for flows with status, keyed by label_full."""
        return {fid: self[fid].status_parameters for fid in self.with_status}

    # === Batched Parameters ===
    # Properties return xr.DataArray only for relevant flows (based on categorizations).

    @cached_property
    def flow_hours_minimum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - minimum total flow hours for flows with explicit min."""
        flow_ids = self.with_flow_hours_min
        if not flow_ids:
            return None
        values = [self[fid].flow_hours_min for fid in flow_ids]
        return self._stack_values_for_subset(flow_ids, values, dims=['period', 'scenario'])

    @cached_property
    def flow_hours_maximum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - maximum total flow hours for flows with explicit max."""
        flow_ids = self.with_flow_hours_max
        if not flow_ids:
            return None
        values = [self[fid].flow_hours_max for fid in flow_ids]
        return self._stack_values_for_subset(flow_ids, values, dims=['period', 'scenario'])

    @cached_property
    def flow_hours_minimum_over_periods(self) -> xr.DataArray | None:
        """(flow, scenario) - minimum flow hours over all periods for flows with explicit min."""
        flow_ids = self.with_flow_hours_over_periods_min
        if not flow_ids:
            return None
        values = [self[fid].flow_hours_min_over_periods for fid in flow_ids]
        return self._stack_values_for_subset(flow_ids, values, dims=['scenario'])

    @cached_property
    def flow_hours_maximum_over_periods(self) -> xr.DataArray | None:
        """(flow, scenario) - maximum flow hours over all periods for flows with explicit max."""
        flow_ids = self.with_flow_hours_over_periods_max
        if not flow_ids:
            return None
        values = [self[fid].flow_hours_max_over_periods for fid in flow_ids]
        return self._stack_values_for_subset(flow_ids, values, dims=['scenario'])

    @cached_property
    def load_factor_minimum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - minimum load factor for flows with explicit min."""
        flow_ids = self.with_load_factor_min
        if not flow_ids:
            return None
        values = [self[fid].load_factor_min for fid in flow_ids]
        return self._stack_values_for_subset(flow_ids, values, dims=['period', 'scenario'])

    @cached_property
    def load_factor_maximum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - maximum load factor for flows with explicit max."""
        flow_ids = self.with_load_factor_max
        if not flow_ids:
            return None
        values = [self[fid].load_factor_max for fid in flow_ids]
        return self._stack_values_for_subset(flow_ids, values, dims=['period', 'scenario'])

    @cached_property
    def relative_minimum(self) -> xr.DataArray:
        """(flow, time, period, scenario) - relative lower bound on flow rate."""
        values = [f.relative_minimum for f in self.elements.values()]
        return self._broadcast_to_coords(self._stack_values(values), dims=None)

    @cached_property
    def relative_maximum(self) -> xr.DataArray:
        """(flow, time, period, scenario) - relative upper bound on flow rate."""
        values = [f.relative_maximum for f in self.elements.values()]
        return self._broadcast_to_coords(self._stack_values(values), dims=None)

    @cached_property
    def fixed_relative_profile(self) -> xr.DataArray:
        """(flow, time, period, scenario) - fixed profile. NaN = not fixed."""
        values = [
            f.fixed_relative_profile if f.fixed_relative_profile is not None else np.nan for f in self.elements.values()
        ]
        return self._broadcast_to_coords(self._stack_values(values), dims=None)

    @cached_property
    def effective_relative_minimum(self) -> xr.DataArray:
        """(flow, time, period, scenario) - effective lower bound (uses fixed_profile if set)."""
        fixed = self.fixed_relative_profile
        rel_min = self.relative_minimum
        return xr.where(fixed.notnull(), fixed, rel_min)

    @cached_property
    def effective_relative_maximum(self) -> xr.DataArray:
        """(flow, time, period, scenario) - effective upper bound (uses fixed_profile if set)."""
        fixed = self.fixed_relative_profile
        rel_max = self.relative_maximum
        return xr.where(fixed.notnull(), fixed, rel_max)

    @cached_property
    def fixed_size(self) -> xr.DataArray:
        """(flow, period, scenario) - fixed size for non-investment flows. NaN for investment/no-size flows."""
        values = []
        for f in self.elements.values():
            if f.size is None or isinstance(f.size, InvestParameters):
                values.append(np.nan)
            else:
                values.append(f.size)
        return self._broadcast_to_coords(self._stack_values(values), dims=['period', 'scenario'])

    @cached_property
    def effective_size_lower(self) -> xr.DataArray:
        """(flow, period, scenario) - effective lower size for bounds.

        - Fixed size flows: the size value
        - Investment flows: minimum_or_fixed_size
        - No size: NaN
        """
        values = []
        for f in self.elements.values():
            if f.size is None:
                values.append(np.nan)
            elif isinstance(f.size, InvestParameters):
                values.append(f.size.minimum_or_fixed_size)
            else:
                values.append(f.size)
        return self._broadcast_to_coords(self._stack_values(values), dims=['period', 'scenario'])

    @cached_property
    def effective_size_upper(self) -> xr.DataArray:
        """(flow, period, scenario) - effective upper size for bounds.

        - Fixed size flows: the size value
        - Investment flows: maximum_or_fixed_size
        - No size: NaN
        """
        values = []
        for f in self.elements.values():
            if f.size is None:
                values.append(np.nan)
            elif isinstance(f.size, InvestParameters):
                values.append(f.size.maximum_or_fixed_size)
            else:
                values.append(f.size)
        return self._broadcast_to_coords(self._stack_values(values), dims=['period', 'scenario'])

    @cached_property
    def absolute_lower_bounds(self) -> xr.DataArray:
        """(flow, time, period, scenario) - absolute lower bounds for flow rate.

        Logic:
        - Status flows → 0 (status variable controls activation)
        - Optional investment → 0 (invested variable controls)
        - Mandatory investment → relative_min * effective_size_lower
        - Fixed size → relative_min * effective_size_lower
        - No size → 0
        """
        # Base: relative_min * size_lower
        base = self.effective_relative_minimum * self.effective_size_lower

        # Build mask for flows that should have lb=0
        flow_ids = xr.DataArray(self.ids, dims=['flow'], coords={'flow': self.ids})
        is_status = flow_ids.isin(self.with_status)
        is_optional_invest = flow_ids.isin(self.with_optional_investment)
        has_no_size = self.effective_size_lower.isnull()

        is_zero = is_status | is_optional_invest | has_no_size
        return xr.where(is_zero, 0.0, base).fillna(0.0)

    @cached_property
    def absolute_upper_bounds(self) -> xr.DataArray:
        """(flow, time, period, scenario) - absolute upper bounds for flow rate.

        Logic:
        - Investment flows → relative_max * effective_size_upper
        - Fixed size → relative_max * effective_size_upper
        - No size → inf
        """
        # Base: relative_max * size_upper
        base = self.effective_relative_maximum * self.effective_size_upper

        # Inf for flows without size
        return xr.where(self.effective_size_upper.isnull(), np.inf, base)

    # --- Investment Bounds (for size variable) ---

    @cached_property
    def investment_size_minimum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - minimum size for flows with investment.

        For mandatory: minimum_or_fixed_size
        For optional: 0 (invested variable controls actual minimum)
        """
        if not self.with_investment:
            return None
        flow_ids = self.with_investment
        values = []
        for fid in flow_ids:
            params = self.invest_params[fid]
            if params.mandatory:
                values.append(params.minimum_or_fixed_size)
            else:
                values.append(0)  # Optional: lower bound is 0
        return self._stack_values_for_subset(flow_ids, values, dims=['period', 'scenario'])

    @cached_property
    def investment_size_maximum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - maximum size for flows with investment."""
        if not self.with_investment:
            return None
        flow_ids = self.with_investment
        values = [self.invest_params[fid].maximum_or_fixed_size for fid in flow_ids]
        return self._stack_values_for_subset(flow_ids, values, dims=['period', 'scenario'])

    @cached_property
    def optional_investment_size_minimum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - minimum size for optional investment flows.

        Used in constraints: size >= min * invested
        """
        if not self.with_optional_investment:
            return None
        flow_ids = self.with_optional_investment
        values = [self.invest_params[fid].minimum_or_fixed_size for fid in flow_ids]
        return self._stack_values_for_subset(flow_ids, values, dims=['period', 'scenario'])

    @cached_property
    def optional_investment_size_maximum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - maximum size for optional investment flows.

        Used in constraints: size <= max * invested
        """
        if not self.with_optional_investment:
            return None
        flow_ids = self.with_optional_investment
        values = [self.invest_params[fid].maximum_or_fixed_size for fid in flow_ids]
        return self._stack_values_for_subset(flow_ids, values, dims=['period', 'scenario'])

    @cached_property
    def effects_per_flow_hour(self) -> xr.DataArray | None:
        """(flow, effect, ...) - effect factors per flow hour.

        Missing (flow, effect) combinations are NaN - the xarray convention for
        missing data. This distinguishes "no effect defined" from "effect is zero".

        Use `.fillna(0)` to fill for computation, `.notnull()` as mask.
        """
        if not self.with_effects:
            return None

        effect_ids = list(self._fs.effects.keys())
        if not effect_ids:
            return None

        flow_ids = self.with_effects

        # Use np.nan for missing effects (not 0!) to distinguish "not defined" from "zero"
        # Use coords='minimal' to handle dimension mismatches (some effects may have 'time', some scalars)
        flow_factors = [
            xr.concat(
                [xr.DataArray(self[fid].effects_per_flow_hour.get(eff, np.nan)) for eff in effect_ids],
                dim='effect',
                coords='minimal',
            ).assign_coords(effect=effect_ids)
            for fid in flow_ids
        ]

        # Use coords='minimal' to handle dimension mismatches (some effects may have 'period', some don't)
        return concat_with_coords(flow_factors, 'flow', flow_ids)

    # --- Investment Parameters ---

    @cached_property
    def linked_periods(self) -> xr.DataArray | None:
        """(flow, period) - period linking mask. 1=linked, 0=not linked, NaN=no linking."""
        has_linking = any(
            isinstance(f.size, InvestParameters) and f.size.linked_periods is not None for f in self.elements.values()
        )
        if not has_linking:
            return None

        values = []
        for f in self.elements.values():
            if not isinstance(f.size, InvestParameters) or f.size.linked_periods is None:
                values.append(np.nan)
            else:
                values.append(f.size.linked_periods)
        return self._broadcast_to_coords(self._stack_values(values), dims=['period'])

    # --- Status Effects ---

    @cached_property
    def status_effects_per_active_hour(self) -> xr.DataArray | None:
        """(flow, effect, ...) - effect factors per active hour for flows with status."""
        if not self.with_status:
            return None

        from .features import InvestmentHelpers, StatusHelpers

        element_ids = [fid for fid in self.with_status if self.status_params[fid].effects_per_active_hour]
        if not element_ids:
            return None

        time_coords = self._fs.timesteps
        effects_dict = StatusHelpers.collect_status_effects(
            self.status_params, element_ids, 'effects_per_active_hour', 'flow', time_coords
        )
        return InvestmentHelpers.build_effect_factors(effects_dict, element_ids, 'flow')

    @cached_property
    def status_effects_per_startup(self) -> xr.DataArray | None:
        """(flow, effect, ...) - effect factors per startup for flows with status."""
        if not self.with_status:
            return None

        from .features import InvestmentHelpers, StatusHelpers

        element_ids = [fid for fid in self.with_status if self.status_params[fid].effects_per_startup]
        if not element_ids:
            return None

        time_coords = self._fs.timesteps
        effects_dict = StatusHelpers.collect_status_effects(
            self.status_params, element_ids, 'effects_per_startup', 'flow', time_coords
        )
        return InvestmentHelpers.build_effect_factors(effects_dict, element_ids, 'flow')

    # --- Previous Status ---

    @cached_property
    def previous_states(self) -> dict[str, xr.DataArray]:
        """Previous status for flows with previous_flow_rate, keyed by label_full.

        Returns:
            Dict mapping flow_id -> binary DataArray (time dimension).
        """
        from .config import CONFIG
        from .modeling import ModelingUtilitiesAbstract

        result = {}
        for fid in self.with_previous_flow_rate:
            flow = self[fid]
            if flow.previous_flow_rate is not None:
                result[fid] = ModelingUtilitiesAbstract.to_binary(
                    values=xr.DataArray(
                        [flow.previous_flow_rate] if np.isscalar(flow.previous_flow_rate) else flow.previous_flow_rate,
                        dims='time',
                    ),
                    epsilon=CONFIG.Modeling.epsilon,
                    dims='time',
                )
        return result

    # --- Status Bounds (for duration tracking) ---

    @cached_property
    def min_uptime(self) -> xr.DataArray | None:
        """(flow,) - minimum uptime for flows with uptime tracking. NaN = no constraint."""
        flow_ids = self.with_uptime_tracking
        if not flow_ids:
            return None
        values = [self.status_params[fid].min_uptime or np.nan for fid in flow_ids]
        return xr.DataArray(values, dims=['flow'], coords={'flow': flow_ids})

    @cached_property
    def max_uptime(self) -> xr.DataArray | None:
        """(flow,) - maximum uptime for flows with uptime tracking. NaN = no constraint."""
        flow_ids = self.with_uptime_tracking
        if not flow_ids:
            return None
        values = [self.status_params[fid].max_uptime or np.nan for fid in flow_ids]
        return xr.DataArray(values, dims=['flow'], coords={'flow': flow_ids})

    @cached_property
    def min_downtime(self) -> xr.DataArray | None:
        """(flow,) - minimum downtime for flows with downtime tracking. NaN = no constraint."""
        flow_ids = self.with_downtime_tracking
        if not flow_ids:
            return None
        values = [self.status_params[fid].min_downtime or np.nan for fid in flow_ids]
        return xr.DataArray(values, dims=['flow'], coords={'flow': flow_ids})

    @cached_property
    def max_downtime(self) -> xr.DataArray | None:
        """(flow,) - maximum downtime for flows with downtime tracking. NaN = no constraint."""
        flow_ids = self.with_downtime_tracking
        if not flow_ids:
            return None
        values = [self.status_params[fid].max_downtime or np.nan for fid in flow_ids]
        return xr.DataArray(values, dims=['flow'], coords={'flow': flow_ids})

    @cached_property
    def startup_limit_values(self) -> xr.DataArray | None:
        """(flow,) - startup limit for flows with startup limit."""
        flow_ids = self.with_startup_limit
        if not flow_ids:
            return None
        values = [self.status_params[fid].startup_limit for fid in flow_ids]
        return xr.DataArray(values, dims=['flow'], coords={'flow': flow_ids})

    @cached_property
    def previous_uptime(self) -> xr.DataArray | None:
        """(flow,) - previous uptime duration for flows with uptime tracking and previous state.

        Computed from previous_states using StatusHelpers.compute_previous_duration().
        NaN for flows without previous state or without min_uptime.
        """
        from .features import StatusHelpers

        flow_ids = self.with_uptime_tracking
        if not flow_ids:
            return None

        # Need timestep_duration for computation
        timestep_duration = self._fs.timestep_duration

        values = []
        for fid in flow_ids:
            params = self.status_params[fid]
            if fid in self.previous_states and params.min_uptime is not None:
                prev = StatusHelpers.compute_previous_duration(
                    self.previous_states[fid], target_state=1, timestep_duration=timestep_duration
                )
                values.append(prev)
            else:
                values.append(np.nan)

        return xr.DataArray(values, dims=['flow'], coords={'flow': flow_ids})

    @cached_property
    def previous_downtime(self) -> xr.DataArray | None:
        """(flow,) - previous downtime duration for flows with downtime tracking and previous state.

        Computed from previous_states using StatusHelpers.compute_previous_duration().
        NaN for flows without previous state or without min_downtime.
        """
        from .features import StatusHelpers

        flow_ids = self.with_downtime_tracking
        if not flow_ids:
            return None

        # Need timestep_duration for computation
        timestep_duration = self._fs.timestep_duration

        values = []
        for fid in flow_ids:
            params = self.status_params[fid]
            if fid in self.previous_states and params.min_downtime is not None:
                prev = StatusHelpers.compute_previous_duration(
                    self.previous_states[fid], target_state=0, timestep_duration=timestep_duration
                )
                values.append(prev)
            else:
                values.append(np.nan)

        return xr.DataArray(values, dims=['flow'], coords={'flow': flow_ids})

    # === Helper Methods ===

    def _stack_values_for_subset(
        self, flow_ids: list[str], values: list, dims: list[str] | None = None
    ) -> xr.DataArray | None:
        """Stack values for a subset of flows and broadcast to coords.

        Args:
            flow_ids: List of flow IDs to include.
            values: List of values corresponding to flow_ids.
            dims: Model dimensions to broadcast to. None = all (time, period, scenario).

        Returns:
            DataArray with flow dimension, or None if flow_ids is empty.
        """
        if not flow_ids:
            return None

        dim = 'flow'

        # Check for multi-dimensional values
        has_multidim = any(isinstance(v, xr.DataArray) and v.ndim > 0 for v in values)

        if not has_multidim:
            # Fast path: all scalars
            scalar_values = [float(v.values) if isinstance(v, xr.DataArray) else float(v) for v in values]
            arr = xr.DataArray(
                np.array(scalar_values),
                coords={dim: flow_ids},
                dims=[dim],
            )
        else:
            # Slow path: concat multi-dimensional arrays
            arrays_to_stack = []
            for val, fid in zip(values, flow_ids, strict=True):
                if isinstance(val, xr.DataArray):
                    arr_item = val.expand_dims({dim: [fid]})
                else:
                    arr_item = xr.DataArray(val, coords={dim: [fid]}, dims=[dim])
                arrays_to_stack.append(arr_item)
            arr = xr.concat(arrays_to_stack, dim=dim)

        return self._broadcast_to_coords(arr, dims=dims)

    def _stack_values(self, values: list) -> xr.DataArray | float:
        """Stack per-element values into array with flow dimension.

        Returns scalar if all values are identical scalars.
        """
        dim = 'flow'

        # Extract scalar values
        scalar_values = []
        has_multidim = False

        for v in values:
            if isinstance(v, xr.DataArray):
                if v.ndim == 0:
                    scalar_values.append(float(v.values))
                else:
                    has_multidim = True
                    break
            else:
                scalar_values.append(float(v) if not (isinstance(v, float) and np.isnan(v)) else np.nan)

        # Fast path: all scalars
        if not has_multidim:
            unique_values = set(v for v in scalar_values if not (isinstance(v, float) and np.isnan(v)))
            nan_count = sum(1 for v in scalar_values if isinstance(v, float) and np.isnan(v))
            if len(unique_values) == 1 and nan_count == 0:
                return list(unique_values)[0]

            return xr.DataArray(
                np.array(scalar_values),
                coords={dim: self.ids},
                dims=[dim],
            )

        # Slow path: concat multi-dimensional arrays
        arrays_to_stack = []
        for val, fid in zip(values, self.ids, strict=False):
            if isinstance(val, xr.DataArray):
                arr = val.expand_dims({dim: [fid]})
            else:
                arr = xr.DataArray(val, coords={dim: [fid]}, dims=[dim])
            arrays_to_stack.append(arr)

        return xr.concat(arrays_to_stack, dim=dim)

    def _broadcast_to_coords(
        self,
        arr: xr.DataArray | float,
        dims: list[str] | None,
    ) -> xr.DataArray:
        """Broadcast array to include model coordinates.

        Args:
            arr: Array with flow dimension (or scalar).
            dims: Model dimensions to include. None = all (time, period, scenario).

        Returns:
            DataArray with dimensions in canonical order: (flow, time, period, scenario)
        """
        if isinstance(arr, (int, float)):
            # Scalar - create array with flow dim first
            arr = xr.DataArray(
                np.full(len(self.ids), arr),
                coords={'flow': self.ids},
                dims=['flow'],
            )

        # Get model coordinates from FlowSystem.indexes
        if dims is None:
            dims = ['time', 'period', 'scenario']

        indexes = self._fs.indexes
        coords_to_add = {dim: indexes[dim] for dim in dims if dim in indexes}

        if not coords_to_add:
            return arr

        # Broadcast to include new dimensions
        for dim_name, coord in coords_to_add.items():
            if dim_name not in arr.dims:
                arr = arr.expand_dims({dim_name: coord})

        # Enforce canonical dimension order: (flow, time, period, scenario)
        canonical_order = ['flow', 'time', 'period', 'scenario']
        actual_dims = [d for d in canonical_order if d in arr.dims]
        if list(arr.dims) != actual_dims:
            arr = arr.transpose(*actual_dims)

        return arr


class BatchedAccessor:
    """Accessor for batched data containers on FlowSystem.

    Usage:
        flow_system.batched.flows  # Access FlowsData
    """

    def __init__(self, flow_system: FlowSystem):
        self._fs = flow_system
        self._flows: FlowsData | None = None

    @property
    def flows(self) -> FlowsData:
        """Get or create FlowsData for all flows in the system."""
        if self._flows is None:
            all_flows = list(self._fs.flows.values())
            self._flows = FlowsData(all_flows, self._fs)
        return self._flows

    def _reset(self) -> None:
        """Reset cached data (called when FlowSystem changes)."""
        self._flows = None
