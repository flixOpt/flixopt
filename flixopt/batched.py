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
import pandas as pd
import xarray as xr

from .features import InvestmentHelpers, concat_with_coords
from .interface import InvestParameters, StatusParameters
from .structure import ElementContainer

if TYPE_CHECKING:
    from .elements import Flow
    from .flow_system import FlowSystem


class StatusData:
    """Batched access to StatusParameters for a group of elements.

    Provides efficient batched access to status-related data as xr.DataArrays.
    Used internally by FlowsData and can be reused by ComponentsModel.

    Args:
        params: Dict mapping element_id -> StatusParameters.
        dim_name: Dimension name for arrays (e.g., 'flow', 'component').
        effect_ids: List of effect IDs for building effect arrays.
        timestep_duration: Duration per timestep (for previous duration computation).
        previous_states: Optional dict of previous status arrays for duration computation.
    """

    def __init__(
        self,
        params: dict[str, StatusParameters],
        dim_name: str,
        effect_ids: list[str] | None = None,
        timestep_duration: xr.DataArray | float | None = None,
        previous_states: dict[str, xr.DataArray] | None = None,
    ):
        self._params = params
        self._dim = dim_name
        self._ids = list(params.keys())
        self._effect_ids = effect_ids or []
        self._timestep_duration = timestep_duration
        self._previous_states = previous_states or {}

    @property
    def ids(self) -> list[str]:
        """All element IDs with status."""
        return self._ids

    # === Categorizations ===

    @cached_property
    def with_startup_tracking(self) -> list[str]:
        """IDs needing startup/shutdown tracking."""
        return [
            eid
            for eid in self._ids
            if (
                self._params[eid].effects_per_startup
                or self._params[eid].min_uptime is not None
                or self._params[eid].max_uptime is not None
                or self._params[eid].startup_limit is not None
                or self._params[eid].force_startup_tracking
            )
        ]

    @cached_property
    def with_downtime_tracking(self) -> list[str]:
        """IDs needing downtime (inactive) tracking."""
        return [
            eid
            for eid in self._ids
            if self._params[eid].min_downtime is not None or self._params[eid].max_downtime is not None
        ]

    @cached_property
    def with_uptime_tracking(self) -> list[str]:
        """IDs needing uptime duration tracking."""
        return [
            eid
            for eid in self._ids
            if self._params[eid].min_uptime is not None or self._params[eid].max_uptime is not None
        ]

    @cached_property
    def with_startup_limit(self) -> list[str]:
        """IDs with startup limit."""
        return [eid for eid in self._ids if self._params[eid].startup_limit is not None]

    @cached_property
    def with_effects_per_active_hour(self) -> list[str]:
        """IDs with effects_per_active_hour defined."""
        return [eid for eid in self._ids if self._params[eid].effects_per_active_hour]

    @cached_property
    def with_effects_per_startup(self) -> list[str]:
        """IDs with effects_per_startup defined."""
        return [eid for eid in self._ids if self._params[eid].effects_per_startup]

    # === Bounds (combined min/max in single pass) ===

    def _build_bounds(self, ids: list[str], min_attr: str, max_attr: str) -> tuple[xr.DataArray, xr.DataArray] | None:
        """Build min/max bound arrays in a single pass."""
        if not ids:
            return None
        min_vals = np.empty(len(ids), dtype=float)
        max_vals = np.empty(len(ids), dtype=float)
        for i, eid in enumerate(ids):
            p = self._params[eid]
            min_vals[i] = getattr(p, min_attr) or np.nan
            max_vals[i] = getattr(p, max_attr) or np.nan
        return (
            xr.DataArray(min_vals, dims=[self._dim], coords={self._dim: ids}),
            xr.DataArray(max_vals, dims=[self._dim], coords={self._dim: ids}),
        )

    @cached_property
    def _uptime_bounds(self) -> tuple[xr.DataArray, xr.DataArray] | None:
        """Cached (min_uptime, max_uptime) tuple."""
        return self._build_bounds(self.with_uptime_tracking, 'min_uptime', 'max_uptime')

    @cached_property
    def _downtime_bounds(self) -> tuple[xr.DataArray, xr.DataArray] | None:
        """Cached (min_downtime, max_downtime) tuple."""
        return self._build_bounds(self.with_downtime_tracking, 'min_downtime', 'max_downtime')

    @property
    def min_uptime(self) -> xr.DataArray | None:
        """(element,) - minimum uptime. NaN = no constraint."""
        return self._uptime_bounds[0] if self._uptime_bounds else None

    @property
    def max_uptime(self) -> xr.DataArray | None:
        """(element,) - maximum uptime. NaN = no constraint."""
        return self._uptime_bounds[1] if self._uptime_bounds else None

    @property
    def min_downtime(self) -> xr.DataArray | None:
        """(element,) - minimum downtime. NaN = no constraint."""
        return self._downtime_bounds[0] if self._downtime_bounds else None

    @property
    def max_downtime(self) -> xr.DataArray | None:
        """(element,) - maximum downtime. NaN = no constraint."""
        return self._downtime_bounds[1] if self._downtime_bounds else None

    @cached_property
    def startup_limit(self) -> xr.DataArray | None:
        """(element,) - startup limit for elements with startup limit."""
        ids = self.with_startup_limit
        if not ids:
            return None
        values = np.array([self._params[eid].startup_limit for eid in ids], dtype=float)
        return xr.DataArray(values, dims=[self._dim], coords={self._dim: ids})

    # === Previous Durations ===

    def _build_previous_durations(self, ids: list[str], target_state: int, min_attr: str) -> xr.DataArray | None:
        """Build previous duration array for elements with previous state."""
        if not ids or self._timestep_duration is None:
            return None

        from .features import StatusHelpers

        values = np.full(len(ids), np.nan, dtype=float)
        for i, eid in enumerate(ids):
            if eid in self._previous_states and getattr(self._params[eid], min_attr) is not None:
                values[i] = StatusHelpers.compute_previous_duration(
                    self._previous_states[eid], target_state=target_state, timestep_duration=self._timestep_duration
                )

        return xr.DataArray(values, dims=[self._dim], coords={self._dim: ids})

    @cached_property
    def previous_uptime(self) -> xr.DataArray | None:
        """(element,) - previous uptime duration. NaN where not applicable."""
        return self._build_previous_durations(self.with_uptime_tracking, target_state=1, min_attr='min_uptime')

    @cached_property
    def previous_downtime(self) -> xr.DataArray | None:
        """(element,) - previous downtime duration. NaN where not applicable."""
        return self._build_previous_durations(self.with_downtime_tracking, target_state=0, min_attr='min_downtime')

    # === Effects ===

    def _build_effects(self, attr: str) -> xr.DataArray | None:
        """Build effect factors array for a status effect attribute."""
        ids = [eid for eid in self._ids if getattr(self._params[eid], attr)]
        if not ids or not self._effect_ids:
            return None

        flow_factors = [
            xr.concat(
                [xr.DataArray(getattr(self._params[eid], attr).get(eff, np.nan)) for eff in self._effect_ids],
                dim='effect',
                coords='minimal',
            ).assign_coords(effect=self._effect_ids)
            for eid in ids
        ]

        return concat_with_coords(flow_factors, self._dim, ids)

    @cached_property
    def effects_per_active_hour(self) -> xr.DataArray | None:
        """(element, effect, ...) - effect factors per active hour."""
        return self._build_effects('effects_per_active_hour')

    @cached_property
    def effects_per_startup(self) -> xr.DataArray | None:
        """(element, effect, ...) - effect factors per startup."""
        return self._build_effects('effects_per_startup')


class InvestmentData:
    """Batched access to InvestParameters for a group of elements.

    Provides efficient batched access to investment-related data as xr.DataArrays.
    Used internally by FlowsData and can be reused by StoragesModel.

    Args:
        params: Dict mapping element_id -> InvestParameters.
        dim_name: Dimension name for arrays (e.g., 'flow', 'storage').
        effect_ids: List of effect IDs for building effect arrays.
    """

    def __init__(
        self,
        params: dict[str, InvestParameters],
        dim_name: str,
        effect_ids: list[str] | None = None,
    ):
        self._params = params
        self._dim = dim_name
        self._ids = list(params.keys())
        self._effect_ids = effect_ids or []

    @property
    def ids(self) -> list[str]:
        """All element IDs with investment."""
        return self._ids

    # === Categorizations ===

    @cached_property
    def with_optional(self) -> list[str]:
        """IDs with optional (non-mandatory) investment."""
        return [eid for eid in self._ids if not self._params[eid].mandatory]

    @cached_property
    def with_mandatory(self) -> list[str]:
        """IDs with mandatory investment."""
        return [eid for eid in self._ids if self._params[eid].mandatory]

    @cached_property
    def with_effects_per_size(self) -> list[str]:
        """IDs with effects_of_investment_per_size defined."""
        return [eid for eid in self._ids if self._params[eid].effects_of_investment_per_size]

    @cached_property
    def with_effects_of_investment(self) -> list[str]:
        """IDs with effects_of_investment defined (optional only)."""
        return [eid for eid in self.with_optional if self._params[eid].effects_of_investment]

    @cached_property
    def with_effects_of_retirement(self) -> list[str]:
        """IDs with effects_of_retirement defined (optional only)."""
        return [eid for eid in self.with_optional if self._params[eid].effects_of_retirement]

    @cached_property
    def with_linked_periods(self) -> list[str]:
        """IDs with linked_periods defined."""
        return [eid for eid in self._ids if self._params[eid].linked_periods is not None]

    @cached_property
    def with_piecewise_effects(self) -> list[str]:
        """IDs with piecewise_effects_of_investment defined."""
        return [eid for eid in self._ids if self._params[eid].piecewise_effects_of_investment is not None]

    # === Size Bounds ===

    @cached_property
    def size_minimum(self) -> xr.DataArray:
        """(element, [period, scenario]) - minimum size for all investment elements.

        For mandatory: minimum_or_fixed_size
        For optional: 0 (invested variable controls actual minimum)
        """
        bounds = [self._params[eid].minimum_or_fixed_size if self._params[eid].mandatory else 0.0 for eid in self._ids]
        return InvestmentHelpers.stack_bounds(bounds, self._ids, self._dim)

    @cached_property
    def size_maximum(self) -> xr.DataArray:
        """(element, [period, scenario]) - maximum size for all investment elements."""
        bounds = [self._params[eid].maximum_or_fixed_size for eid in self._ids]
        return InvestmentHelpers.stack_bounds(bounds, self._ids, self._dim)

    @cached_property
    def optional_size_minimum(self) -> xr.DataArray | None:
        """(element, [period, scenario]) - minimum size for optional investment."""
        ids = self.with_optional
        if not ids:
            return None
        bounds = [self._params[eid].minimum_or_fixed_size for eid in ids]
        return InvestmentHelpers.stack_bounds(bounds, ids, self._dim)

    @cached_property
    def optional_size_maximum(self) -> xr.DataArray | None:
        """(element, [period, scenario]) - maximum size for optional investment."""
        ids = self.with_optional
        if not ids:
            return None
        bounds = [self._params[eid].maximum_or_fixed_size for eid in ids]
        return InvestmentHelpers.stack_bounds(bounds, ids, self._dim)

    @cached_property
    def linked_periods(self) -> xr.DataArray | None:
        """(element, period) - period linking mask. 1=linked, NaN=not linked."""
        ids = self.with_linked_periods
        if not ids:
            return None
        bounds = [self._params[eid].linked_periods for eid in ids]
        return InvestmentHelpers.stack_bounds(bounds, ids, self._dim)

    # === Effects ===

    def _build_effects(self, attr: str, ids: list[str] | None = None) -> xr.DataArray | None:
        """Build effect factors array for an investment effect attribute."""
        if ids is None:
            ids = [eid for eid in self._ids if getattr(self._params[eid], attr)]
        if not ids or not self._effect_ids:
            return None

        factors = [
            xr.concat(
                [xr.DataArray(getattr(self._params[eid], attr).get(eff, np.nan)) for eff in self._effect_ids],
                dim='effect',
                coords='minimal',
            ).assign_coords(effect=self._effect_ids)
            for eid in ids
        ]

        return concat_with_coords(factors, self._dim, ids)

    @cached_property
    def effects_per_size(self) -> xr.DataArray | None:
        """(element, effect) - effects per unit size."""
        return self._build_effects('effects_of_investment_per_size', self.with_effects_per_size)

    @cached_property
    def effects_of_investment(self) -> xr.DataArray | None:
        """(element, effect) - fixed effects of investment (optional only)."""
        return self._build_effects('effects_of_investment', self.with_effects_of_investment)

    @cached_property
    def effects_of_retirement(self) -> xr.DataArray | None:
        """(element, effect) - effects of retirement (optional only)."""
        return self._build_effects('effects_of_retirement', self.with_effects_of_retirement)

    @cached_property
    def effects_of_investment_mandatory(self) -> list[tuple[str, dict[str, float | xr.DataArray]]]:
        """List of (element_id, effects_dict) for mandatory investments with fixed effects."""
        result = []
        for eid in self.with_mandatory:
            effects = self._params[eid].effects_of_investment
            if effects:
                effects_dict = {
                    k: v for k, v in effects.items() if v is not None and not (np.isscalar(v) and np.isnan(v))
                }
                if effects_dict:
                    result.append((eid, effects_dict))
        return result

    @cached_property
    def effects_of_retirement_constant(self) -> list[tuple[str, dict[str, float | xr.DataArray]]]:
        """List of (element_id, effects_dict) for retirement constant parts."""
        result = []
        for eid in self.with_optional:
            effects = self._params[eid].effects_of_retirement
            if effects:
                effects_dict = {
                    k: v for k, v in effects.items() if v is not None and not (np.isscalar(v) and np.isnan(v))
                }
                if effects_dict:
                    result.append((eid, effects_dict))
        return result


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

    @cached_property
    def _ids_index(self) -> pd.Index:
        """Cached pd.Index of flow IDs for fast DataArray creation."""
        return pd.Index(self.ids)

    # === Flow Categorizations ===
    # All return list[str] of label_full IDs.

    @cached_property
    def with_status(self) -> list[str]:
        """IDs of flows with status parameters."""
        return [f.label_full for f in self.elements.values() if f.status_parameters is not None]

    @property
    def with_startup_tracking(self) -> list[str]:
        """IDs of flows that need startup/shutdown tracking."""
        return self._status_data.with_startup_tracking if self._status_data else []

    @property
    def with_downtime_tracking(self) -> list[str]:
        """IDs of flows that need downtime (inactive) tracking."""
        return self._status_data.with_downtime_tracking if self._status_data else []

    @property
    def with_uptime_tracking(self) -> list[str]:
        """IDs of flows that need uptime duration tracking."""
        return self._status_data.with_uptime_tracking if self._status_data else []

    @property
    def with_startup_limit(self) -> list[str]:
        """IDs of flows with startup limit."""
        return self._status_data.with_startup_limit if self._status_data else []

    @cached_property
    def without_size(self) -> list[str]:
        """IDs of flows with status parameters."""
        return [f.label_full for f in self.elements.values() if f.size is None]

    @cached_property
    def with_investment(self) -> list[str]:
        """IDs of flows with investment parameters."""
        return [f.label_full for f in self.elements.values() if isinstance(f.size, InvestParameters)]

    @property
    def with_optional_investment(self) -> list[str]:
        """IDs of flows with optional (non-mandatory) investment."""
        return self._investment_data.with_optional if self._investment_data else []

    @property
    def with_mandatory_investment(self) -> list[str]:
        """IDs of flows with mandatory investment."""
        return self._investment_data.with_mandatory if self._investment_data else []

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

    @cached_property
    def _status_data(self) -> StatusData | None:
        """Batched status data for flows with status."""
        if not self.with_status:
            return None
        return StatusData(
            params=self.status_params,
            dim_name='flow',
            effect_ids=list(self._fs.effects.keys()),
            timestep_duration=self._fs.timestep_duration,
            previous_states=self.previous_states,
        )

    @cached_property
    def _investment_data(self) -> InvestmentData | None:
        """Batched investment data for flows with investment."""
        if not self.with_investment:
            return None
        return InvestmentData(
            params=self.invest_params,
            dim_name='flow',
            effect_ids=list(self._fs.effects.keys()),
        )

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
        # Use DataArray.where (faster than xr.where)
        return rel_min.where(fixed.isnull(), fixed)

    @cached_property
    def effective_relative_maximum(self) -> xr.DataArray:
        """(flow, time, period, scenario) - effective upper bound (uses fixed_profile if set)."""
        fixed = self.fixed_relative_profile
        rel_max = self.relative_maximum
        # Use DataArray.where (faster than xr.where)
        return rel_max.where(fixed.isnull(), fixed)

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
        flow_ids = xr.DataArray(self._ids_index, dims=['flow'], coords={'flow': self._ids_index})
        is_status = flow_ids.isin(self.with_status)
        is_optional_invest = flow_ids.isin(self.with_optional_investment)
        has_no_size = self.effective_size_lower.isnull()

        is_zero = is_status | is_optional_invest | has_no_size
        # Use DataArray.where (faster than xr.where)
        return base.where(~is_zero, 0.0).fillna(0.0)

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

        # Inf for flows without size (use DataArray.where, faster than xr.where)
        return base.where(self.effective_size_upper.notnull(), np.inf)

    # --- Investment Bounds (delegated to InvestmentData) ---

    @property
    def investment_size_minimum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - minimum size for flows with investment."""
        if not self._investment_data:
            return None
        return self._broadcast_to_coords(self._investment_data.size_minimum, dims=['period', 'scenario'])

    @property
    def investment_size_maximum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - maximum size for flows with investment."""
        if not self._investment_data:
            return None
        return self._broadcast_to_coords(self._investment_data.size_maximum, dims=['period', 'scenario'])

    @property
    def optional_investment_size_minimum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - minimum size for optional investment flows."""
        if not self._investment_data or not self._investment_data.optional_size_minimum is not None:
            return None
        raw = self._investment_data.optional_size_minimum
        if raw is None:
            return None
        return self._broadcast_to_coords(raw, dims=['period', 'scenario'])

    @property
    def optional_investment_size_maximum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - maximum size for optional investment flows."""
        if not self._investment_data:
            return None
        raw = self._investment_data.optional_size_maximum
        if raw is None:
            return None
        return self._broadcast_to_coords(raw, dims=['period', 'scenario'])

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

        # Check what extra dimensions are present (time, period, scenario)
        extra_dims: set[str] = set()
        for fid in flow_ids:
            flow_effects = self[fid].effects_per_flow_hour
            for val in flow_effects.values():
                if isinstance(val, xr.DataArray) and val.ndim > 0:
                    extra_dims.update(val.dims)

        if extra_dims:
            # Has multi-dimensional effects - use concat approach
            # But optimize by only doing inner concat once per flow
            flow_factors = []
            for fid in flow_ids:
                flow_effects = self[fid].effects_per_flow_hour
                effect_arrays = []
                for eff in effect_ids:
                    val = flow_effects.get(eff)
                    if val is None:
                        effect_arrays.append(xr.DataArray(np.nan))
                    elif isinstance(val, xr.DataArray):
                        effect_arrays.append(val)
                    else:
                        effect_arrays.append(xr.DataArray(float(val)))

                flow_factor = xr.concat(effect_arrays, dim='effect', coords='minimal')
                flow_factor = flow_factor.assign_coords(effect=effect_ids)
                flow_factors.append(flow_factor)

            return concat_with_coords(flow_factors, 'flow', flow_ids)

        # Fast path: all scalars - build numpy array directly
        data = np.full((len(flow_ids), len(effect_ids)), np.nan)

        for i, fid in enumerate(flow_ids):
            flow_effects = self[fid].effects_per_flow_hour
            for j, eff in enumerate(effect_ids):
                val = flow_effects.get(eff)
                if val is not None:
                    if isinstance(val, xr.DataArray):
                        data[i, j] = float(val.values)
                    else:
                        data[i, j] = float(val)

        return xr.DataArray(
            data,
            coords={'flow': pd.Index(flow_ids), 'effect': pd.Index(effect_ids)},
            dims=['flow', 'effect'],
        )

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

    # --- Status Effects (delegated to StatusData) ---

    @property
    def effects_per_active_hour(self) -> xr.DataArray | None:
        """(flow, effect, ...) - effect factors per active hour for flows with status."""
        return self._status_data.effects_per_active_hour if self._status_data else None

    @property
    def effects_per_startup(self) -> xr.DataArray | None:
        """(flow, effect, ...) - effect factors per startup for flows with status."""
        return self._status_data.effects_per_startup if self._status_data else None

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

    # --- Status Bounds (delegated to StatusData) ---

    @property
    def min_uptime(self) -> xr.DataArray | None:
        """(flow,) - minimum uptime for flows with uptime tracking. NaN = no constraint."""
        return self._status_data.min_uptime if self._status_data else None

    @property
    def max_uptime(self) -> xr.DataArray | None:
        """(flow,) - maximum uptime for flows with uptime tracking. NaN = no constraint."""
        return self._status_data.max_uptime if self._status_data else None

    @property
    def min_downtime(self) -> xr.DataArray | None:
        """(flow,) - minimum downtime for flows with downtime tracking. NaN = no constraint."""
        return self._status_data.min_downtime if self._status_data else None

    @property
    def max_downtime(self) -> xr.DataArray | None:
        """(flow,) - maximum downtime for flows with downtime tracking. NaN = no constraint."""
        return self._status_data.max_downtime if self._status_data else None

    @property
    def startup_limit_values(self) -> xr.DataArray | None:
        """(flow,) - startup limit for flows with startup limit."""
        return self._status_data.startup_limit if self._status_data else None

    @property
    def previous_uptime(self) -> xr.DataArray | None:
        """(flow,) - previous uptime duration for flows with uptime tracking."""
        return self._status_data.previous_uptime if self._status_data else None

    @property
    def previous_downtime(self) -> xr.DataArray | None:
        """(flow,) - previous downtime duration for flows with downtime tracking."""
        return self._status_data.previous_downtime if self._status_data else None

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
                coords={dim: self._ids_index},
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

        return xr.concat(arrays_to_stack, dim=dim, coords='minimal')

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
                np.full(len(self._ids_index), arr),
                coords={'flow': self._ids_index},
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
