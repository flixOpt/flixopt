"""
Batched data containers for FlowSystem elements.

These classes provide indexed/batched access to element properties,
separating data management from mathematical modeling.

Usage:
    flow_system.batched.flows  # Access FlowsData
    flow_system.batched.storages  # Access StoragesData (future)
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from .core import PlausibilityError
from .features import fast_isnull, fast_notnull, stack_along_dim
from .interface import InvestParameters, StatusParameters
from .modeling import _scalar_safe_isel_drop
from .structure import ElementContainer

if TYPE_CHECKING:
    from .components import LinearConverter, Transmission
    from .effects import Effect, EffectCollection
    from .elements import Bus, Component, Flow
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


def build_effects_array(
    effect_dicts: dict[str, dict[str, float | xr.DataArray]],
    effect_ids: list[str],
    dim_name: str,
) -> xr.DataArray | None:
    """Build effect factors array from per-element effect dicts.

    Args:
        effect_dicts: Dict mapping element_id -> {effect_id -> factor}.
            Missing effects default to 0.
        effect_ids: List of effect IDs for the effect dimension.
        dim_name: Element dimension name ('flow', 'storage', etc.).

    Returns:
        DataArray with (dim_name, effect, ...) or None if empty.
    """
    if not effect_dicts or not effect_ids:
        return None

    ids = list(effect_dicts.keys())

    # Scan for extra dimensions from time-varying effect values
    extra_dims: dict[str, np.ndarray] = {}
    for ed in effect_dicts.values():
        for val in ed.values():
            if isinstance(val, xr.DataArray) and val.ndim > 0:
                for d in val.dims:
                    if d not in extra_dims:
                        extra_dims[d] = val.coords[d].values

    # Build shape: (n_elements, n_effects, *extra_dims)
    shape = [len(ids), len(effect_ids)] + [len(c) for c in extra_dims.values()]
    data = np.zeros(shape)

    # Fill values directly
    for i, ed in enumerate(effect_dicts.values()):
        for j, eff in enumerate(effect_ids):
            val = ed.get(eff, 0.0)
            if isinstance(val, xr.DataArray):
                if val.ndim == 0:
                    data[i, j, ...] = float(val.values)
                else:
                    data[i, j, ...] = val.values
            else:
                data[i, j, ...] = float(val)

    coords = {dim_name: ids, 'effect': effect_ids}
    coords.update(extra_dims)
    dims = [dim_name, 'effect'] + list(extra_dims.keys())
    return xr.DataArray(data, coords=coords, dims=dims)


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

    def _categorize(self, condition) -> list[str]:
        """Return IDs where condition(params) is True."""
        return [eid for eid in self._ids if condition(self._params[eid])]

    @cached_property
    def with_startup_tracking(self) -> list[str]:
        """IDs needing startup/shutdown tracking."""
        return self._categorize(
            lambda p: (
                p.effects_per_startup
                or p.min_uptime is not None
                or p.max_uptime is not None
                or p.startup_limit is not None
                or p.force_startup_tracking
            )
        )

    @cached_property
    def with_downtime_tracking(self) -> list[str]:
        """IDs needing downtime (inactive) tracking."""
        return self._categorize(lambda p: p.min_downtime is not None or p.max_downtime is not None)

    @cached_property
    def with_uptime_tracking(self) -> list[str]:
        """IDs needing uptime duration tracking."""
        return self._categorize(lambda p: p.min_uptime is not None or p.max_uptime is not None)

    @cached_property
    def with_startup_limit(self) -> list[str]:
        """IDs with startup limit."""
        return self._categorize(lambda p: p.startup_limit is not None)

    @cached_property
    def with_effects_per_active_hour(self) -> list[str]:
        """IDs with effects_per_active_hour defined."""
        return self._categorize(lambda p: p.effects_per_active_hour)

    @cached_property
    def with_effects_per_startup(self) -> list[str]:
        """IDs with effects_per_startup defined."""
        return self._categorize(lambda p: p.effects_per_startup)

    # === Bounds (combined min/max in single pass) ===

    def _build_bounds(self, ids: list[str], min_attr: str, max_attr: str) -> tuple[xr.DataArray, xr.DataArray] | None:
        """Build min/max bound arrays in a single pass."""
        if not ids:
            return None

        def _get_scalar_or_nan(value) -> float:
            """Convert value to scalar float, handling arrays and None."""
            if value is None:
                return np.nan
            if isinstance(value, (xr.DataArray, np.ndarray)):
                # For time-varying values, use the minimum for min_* and maximum for max_*
                # This provides conservative bounds for the duration tracking
                return float(np.nanmin(value)) if np.any(np.isfinite(value)) else np.nan
            return float(value) if value else np.nan

        min_vals = np.empty(len(ids), dtype=float)
        max_vals = np.empty(len(ids), dtype=float)
        for i, eid in enumerate(ids):
            p = self._params[eid]
            min_vals[i] = _get_scalar_or_nan(getattr(p, min_attr))
            max_vals[i] = _get_scalar_or_nan(getattr(p, max_attr))
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

    def _build_previous_durations(
        self, ids: list[str], target_state: int, min_attr: str, max_attr: str
    ) -> xr.DataArray | None:
        """Build previous duration array for elements with previous state."""
        if not ids or self._timestep_duration is None:
            return None

        from .features import StatusBuilder

        values = np.full(len(ids), np.nan, dtype=float)
        for i, eid in enumerate(ids):
            # Compute previous duration if element has previous state AND has either min or max constraint
            has_constraint = (
                getattr(self._params[eid], min_attr) is not None or getattr(self._params[eid], max_attr) is not None
            )
            if eid in self._previous_states and has_constraint:
                values[i] = StatusBuilder.compute_previous_duration(
                    self._previous_states[eid], target_state=target_state, timestep_duration=self._timestep_duration
                )

        return xr.DataArray(values, dims=[self._dim], coords={self._dim: ids})

    @cached_property
    def previous_uptime(self) -> xr.DataArray | None:
        """(element,) - previous uptime duration. NaN where not applicable."""
        return self._build_previous_durations(
            self.with_uptime_tracking, target_state=1, min_attr='min_uptime', max_attr='max_uptime'
        )

    @cached_property
    def previous_downtime(self) -> xr.DataArray | None:
        """(element,) - previous downtime duration. NaN where not applicable."""
        return self._build_previous_durations(
            self.with_downtime_tracking, target_state=0, min_attr='min_downtime', max_attr='max_downtime'
        )

    # === Effects ===

    def _build_effects(self, attr: str) -> xr.DataArray | None:
        """Build effect factors array for a status effect attribute."""
        ids = self._categorize(lambda p: getattr(p, attr))
        dicts = {eid: getattr(self._params[eid], attr) for eid in ids}
        return build_effects_array(dicts, self._effect_ids, self._dim)

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

    def _categorize(self, condition) -> list[str]:
        """Return IDs where condition(params) is True."""
        return [eid for eid in self._ids if condition(self._params[eid])]

    @cached_property
    def with_optional(self) -> list[str]:
        """IDs with optional (non-mandatory) investment."""
        return self._categorize(lambda p: not p.mandatory)

    @cached_property
    def with_mandatory(self) -> list[str]:
        """IDs with mandatory investment."""
        return self._categorize(lambda p: p.mandatory)

    @cached_property
    def with_effects_per_size(self) -> list[str]:
        """IDs with effects_of_investment_per_size defined."""
        return self._categorize(lambda p: p.effects_of_investment_per_size)

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
        return self._categorize(lambda p: p.linked_periods is not None)

    @cached_property
    def with_piecewise_effects(self) -> list[str]:
        """IDs with piecewise_effects_of_investment defined."""
        return self._categorize(lambda p: p.piecewise_effects_of_investment is not None)

    # === Size Bounds ===

    @cached_property
    def size_minimum(self) -> xr.DataArray:
        """(element, [period, scenario]) - minimum size for all investment elements.

        For mandatory: minimum_or_fixed_size
        For optional: 0 (invested variable controls actual minimum)
        """
        bounds = [self._params[eid].minimum_or_fixed_size if self._params[eid].mandatory else 0.0 for eid in self._ids]
        return stack_along_dim(bounds, self._dim, self._ids)

    @cached_property
    def size_maximum(self) -> xr.DataArray:
        """(element, [period, scenario]) - maximum size for all investment elements."""
        bounds = [self._params[eid].maximum_or_fixed_size for eid in self._ids]
        return stack_along_dim(bounds, self._dim, self._ids)

    @cached_property
    def optional_size_minimum(self) -> xr.DataArray | None:
        """(element, [period, scenario]) - minimum size for optional investment."""
        ids = self.with_optional
        if not ids:
            return None
        bounds = [self._params[eid].minimum_or_fixed_size for eid in ids]
        return stack_along_dim(bounds, self._dim, ids)

    @cached_property
    def optional_size_maximum(self) -> xr.DataArray | None:
        """(element, [period, scenario]) - maximum size for optional investment."""
        ids = self.with_optional
        if not ids:
            return None
        bounds = [self._params[eid].maximum_or_fixed_size for eid in ids]
        return stack_along_dim(bounds, self._dim, ids)

    @cached_property
    def linked_periods(self) -> xr.DataArray | None:
        """(element, period) - period linking mask. 1=linked, NaN=not linked."""
        ids = self.with_linked_periods
        if not ids:
            return None
        bounds = [self._params[eid].linked_periods for eid in ids]
        return stack_along_dim(bounds, self._dim, ids)

    # === Effects ===

    def _build_effects(self, attr: str, ids: list[str] | None = None) -> xr.DataArray | None:
        """Build effect factors array for an investment effect attribute."""
        if ids is None:
            ids = self._categorize(lambda p: getattr(p, attr))
        dicts = {eid: getattr(self._params[eid], attr) for eid in ids}
        return build_effects_array(dicts, self._effect_ids, self._dim)

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
    def effects_of_investment_mandatory(self) -> xr.DataArray | None:
        """(element, effect) - fixed effects of investment for mandatory elements."""
        ids = [eid for eid in self.with_mandatory if self._params[eid].effects_of_investment]
        return self._build_effects('effects_of_investment', ids)

    @cached_property
    def effects_of_retirement_constant(self) -> xr.DataArray | None:
        """(element, effect) - constant retirement effects for optional elements."""
        ids = [eid for eid in self.with_optional if self._params[eid].effects_of_retirement]
        return self._build_effects('effects_of_retirement', ids)

    # === Piecewise Effects Data ===

    @cached_property
    def _piecewise_raw(self) -> dict:
        """Compute all piecewise data in one pass. Returns dict with all arrays or empty dict."""
        from .features import PiecewiseBuilder

        ids = self.with_piecewise_effects
        if not ids:
            return {}

        dim = self._dim
        params = self._params

        # Segment counts and mask
        segment_counts = {eid: len(params[eid].piecewise_effects_of_investment.piecewise_origin) for eid in ids}
        max_segments, segment_mask = PiecewiseBuilder.collect_segment_info(ids, segment_counts, dim)

        # Origin breakpoints (for size coupling)
        origin_breakpoints = {}
        for eid in ids:
            pieces = params[eid].piecewise_effects_of_investment.piecewise_origin
            origin_breakpoints[eid] = ([p.start for p in pieces], [p.end for p in pieces])
        origin_starts, origin_ends = PiecewiseBuilder.pad_breakpoints(ids, origin_breakpoints, max_segments, dim)

        # Effect breakpoints as (dim, segment, effect)
        all_effect_names: set[str] = set()
        for eid in ids:
            all_effect_names.update(params[eid].piecewise_effects_of_investment.piecewise_shares.keys())
        effect_names = sorted(all_effect_names)

        effect_starts_list, effect_ends_list = [], []
        for effect_name in effect_names:
            breakpoints = {}
            for eid in ids:
                shares = params[eid].piecewise_effects_of_investment.piecewise_shares
                if effect_name in shares:
                    piecewise = shares[effect_name]
                    breakpoints[eid] = ([p.start for p in piecewise], [p.end for p in piecewise])
                else:
                    breakpoints[eid] = ([0.0] * segment_counts[eid], [0.0] * segment_counts[eid])
            s, e = PiecewiseBuilder.pad_breakpoints(ids, breakpoints, max_segments, dim)
            effect_starts_list.append(s.expand_dims(effect=[effect_name]))
            effect_ends_list.append(e.expand_dims(effect=[effect_name]))

        return {
            'element_ids': ids,
            'max_segments': max_segments,
            'segment_mask': segment_mask,
            'origin_starts': origin_starts,
            'origin_ends': origin_ends,
            'effect_starts': xr.concat(effect_starts_list, dim='effect'),
            'effect_ends': xr.concat(effect_ends_list, dim='effect'),
            'effect_names': effect_names,
        }

    @cached_property
    def piecewise_element_ids(self) -> list[str]:
        return self._piecewise_raw.get('element_ids', [])

    @cached_property
    def piecewise_max_segments(self) -> int:
        return self._piecewise_raw.get('max_segments', 0)

    @cached_property
    def piecewise_segment_mask(self) -> xr.DataArray | None:
        return self._piecewise_raw.get('segment_mask')

    @cached_property
    def piecewise_origin_starts(self) -> xr.DataArray | None:
        return self._piecewise_raw.get('origin_starts')

    @cached_property
    def piecewise_origin_ends(self) -> xr.DataArray | None:
        return self._piecewise_raw.get('origin_ends')

    @cached_property
    def piecewise_effect_starts(self) -> xr.DataArray | None:
        return self._piecewise_raw.get('effect_starts')

    @cached_property
    def piecewise_effect_ends(self) -> xr.DataArray | None:
        return self._piecewise_raw.get('effect_ends')

    @cached_property
    def piecewise_effect_names(self) -> list[str]:
        return self._piecewise_raw.get('effect_names', [])


class StoragesData:
    """Batched data container for storage categorization and investment data.

    Provides categorization and batched data for a list of storages,
    separating data management from mathematical modeling.
    Used by both StoragesModel and InterclusterStoragesModel.
    """

    def __init__(
        self, storages: list, dim_name: str, effect_ids: list[str], timesteps_extra: pd.DatetimeIndex | None = None
    ):
        """Initialize StoragesData.

        Args:
            storages: List of Storage elements.
            dim_name: Dimension name for arrays ('storage' or 'intercluster_storage').
            effect_ids: List of effect IDs for building effect arrays.
            timesteps_extra: Extended timesteps (time + 1 final step) for charge state bounds.
                Required for StoragesModel, None for InterclusterStoragesModel.
        """
        self._storages = storages
        self._dim_name = dim_name
        self._effect_ids = effect_ids
        self._timesteps_extra = timesteps_extra
        self._by_label = {s.label_full: s for s in storages}

    @cached_property
    def ids(self) -> list[str]:
        """All storage IDs (label_full)."""
        return [s.label_full for s in self._storages]

    @property
    def element_ids(self) -> list[str]:
        """All storage IDs (alias for ids)."""
        return self.ids

    @property
    def dim_name(self) -> str:
        """Dimension name for this data container."""
        return self._dim_name

    @cached_property
    def elements(self) -> ElementContainer:
        """ElementContainer of storages."""
        return ElementContainer(self._storages)

    def __getitem__(self, label: str):
        """Get a storage by its label_full."""
        return self._by_label[label]

    def __len__(self) -> int:
        return len(self._storages)

    # === Categorization ===

    @cached_property
    def with_investment(self) -> list[str]:
        """IDs of storages with investment parameters."""
        return [s.label_full for s in self._storages if isinstance(s.capacity_in_flow_hours, InvestParameters)]

    @cached_property
    def with_optional_investment(self) -> list[str]:
        """IDs of storages with optional (non-mandatory) investment."""
        return [sid for sid in self.with_investment if not self._by_label[sid].capacity_in_flow_hours.mandatory]

    @cached_property
    def with_mandatory_investment(self) -> list[str]:
        """IDs of storages with mandatory investment."""
        return [sid for sid in self.with_investment if self._by_label[sid].capacity_in_flow_hours.mandatory]

    @cached_property
    def with_balanced(self) -> list[str]:
        """IDs of storages with balanced charging/discharging flow sizes."""
        return [s.label_full for s in self._storages if s.balanced]

    # === Investment Data ===

    @cached_property
    def invest_params(self) -> dict[str, InvestParameters]:
        """Investment parameters for storages with investment, keyed by label_full."""
        return {sid: self._by_label[sid].capacity_in_flow_hours for sid in self.with_investment}

    @cached_property
    def investment_data(self) -> InvestmentData | None:
        """Batched investment data for storages with investment."""
        if not self.with_investment:
            return None
        return InvestmentData(
            params=self.invest_params,
            dim_name=self._dim_name,
            effect_ids=self._effect_ids,
        )

    # === Stacked Storage Parameters ===

    @cached_property
    def eta_charge(self) -> xr.DataArray:
        """(element, [time]) - charging efficiency."""
        return stack_along_dim([s.eta_charge for s in self._storages], self._dim_name, self.ids)

    @cached_property
    def eta_discharge(self) -> xr.DataArray:
        """(element, [time]) - discharging efficiency."""
        return stack_along_dim([s.eta_discharge for s in self._storages], self._dim_name, self.ids)

    @cached_property
    def relative_loss_per_hour(self) -> xr.DataArray:
        """(element, [time]) - relative loss per hour."""
        return stack_along_dim([s.relative_loss_per_hour for s in self._storages], self._dim_name, self.ids)

    @cached_property
    def relative_minimum_charge_state(self) -> xr.DataArray:
        """(element, [time]) - relative minimum charge state."""
        return stack_along_dim([s.relative_minimum_charge_state for s in self._storages], self._dim_name, self.ids)

    @cached_property
    def relative_maximum_charge_state(self) -> xr.DataArray:
        """(element, [time]) - relative maximum charge state."""
        return stack_along_dim([s.relative_maximum_charge_state for s in self._storages], self._dim_name, self.ids)

    @cached_property
    def charging_flow_ids(self) -> list[str]:
        """Flow IDs for charging flows, aligned with self.ids."""
        return [s.charging.label_full for s in self._storages]

    @cached_property
    def discharging_flow_ids(self) -> list[str]:
        """Flow IDs for discharging flows, aligned with self.ids."""
        return [s.discharging.label_full for s in self._storages]

    # === Capacity and Charge State Bounds ===

    @cached_property
    def capacity_lower(self) -> xr.DataArray:
        """(storage, [period, scenario]) - lower capacity per storage (0 for None, min_size for invest, cap for fixed)."""
        values = []
        for s in self._storages:
            if s.capacity_in_flow_hours is None:
                values.append(0.0)
            elif isinstance(s.capacity_in_flow_hours, InvestParameters):
                values.append(s.capacity_in_flow_hours.minimum_or_fixed_size)
            else:
                values.append(s.capacity_in_flow_hours)
        return stack_along_dim(values, self._dim_name, self.ids)

    @cached_property
    def capacity_upper(self) -> xr.DataArray:
        """(storage, [period, scenario]) - upper capacity per storage (inf for None, max_size for invest, cap for fixed)."""
        values = []
        for s in self._storages:
            if s.capacity_in_flow_hours is None:
                values.append(np.inf)
            elif isinstance(s.capacity_in_flow_hours, InvestParameters):
                values.append(s.capacity_in_flow_hours.maximum_or_fixed_size)
            else:
                values.append(s.capacity_in_flow_hours)
        return stack_along_dim(values, self._dim_name, self.ids)

    def _relative_bounds_extra(self) -> tuple[xr.DataArray, xr.DataArray]:
        """Compute relative charge state bounds extended with final timestep values.

        Returns stacked (storage, time_extra) arrays for relative min and max bounds.
        """
        assert self._timesteps_extra is not None, 'timesteps_extra required for charge state bounds'

        rel_mins = []
        rel_maxs = []
        for s in self._storages:
            rel_min = s.relative_minimum_charge_state
            rel_max = s.relative_maximum_charge_state

            # Get final values
            if s.relative_minimum_final_charge_state is None:
                min_final_value = _scalar_safe_isel_drop(rel_min, 'time', -1)
            else:
                min_final_value = s.relative_minimum_final_charge_state

            if s.relative_maximum_final_charge_state is None:
                max_final_value = _scalar_safe_isel_drop(rel_max, 'time', -1)
            else:
                max_final_value = s.relative_maximum_final_charge_state

            # Build bounds arrays for timesteps_extra
            if 'time' in rel_min.dims:
                min_final_da = (
                    min_final_value.expand_dims('time') if 'time' not in min_final_value.dims else min_final_value
                )
                min_final_da = min_final_da.assign_coords(time=[self._timesteps_extra[-1]])
                min_bounds = xr.concat([rel_min, min_final_da], dim='time')
            else:
                min_bounds = rel_min.expand_dims(time=self._timesteps_extra)

            if 'time' in rel_max.dims:
                max_final_da = (
                    max_final_value.expand_dims('time') if 'time' not in max_final_value.dims else max_final_value
                )
                max_final_da = max_final_da.assign_coords(time=[self._timesteps_extra[-1]])
                max_bounds = xr.concat([rel_max, max_final_da], dim='time')
            else:
                max_bounds = rel_max.expand_dims(time=self._timesteps_extra)

            min_bounds, max_bounds = xr.broadcast(min_bounds, max_bounds)
            rel_mins.append(min_bounds)
            rel_maxs.append(max_bounds)

        rel_min_stacked = stack_along_dim(rel_mins, self._dim_name, self.ids)
        rel_max_stacked = stack_along_dim(rel_maxs, self._dim_name, self.ids)
        return rel_min_stacked, rel_max_stacked

    @cached_property
    def _relative_bounds_extra_cached(self) -> tuple[xr.DataArray, xr.DataArray]:
        """Cached relative bounds extended with final timestep."""
        return self._relative_bounds_extra()

    @cached_property
    def relative_minimum_charge_state_extra(self) -> xr.DataArray:
        """(storage, time_extra) - relative min charge state bounds including final timestep."""
        return self._relative_bounds_extra_cached[0]

    @cached_property
    def relative_maximum_charge_state_extra(self) -> xr.DataArray:
        """(storage, time_extra) - relative max charge state bounds including final timestep."""
        return self._relative_bounds_extra_cached[1]

    @cached_property
    def charge_state_lower_bounds(self) -> xr.DataArray:
        """(storage, time_extra) - absolute lower bounds = relative_min * capacity_lower."""
        return self.relative_minimum_charge_state_extra * self.capacity_lower

    @cached_property
    def charge_state_upper_bounds(self) -> xr.DataArray:
        """(storage, time_extra) - absolute upper bounds = relative_max * capacity_upper."""
        return self.relative_maximum_charge_state_extra * self.capacity_upper

    # === Validation ===

    def validate(self) -> None:
        """Validate all storages (config + DataArray checks).

        Performs both:
        - Config validation via Storage.validate_config()
        - DataArray validation (post-transformation checks)

        Raises:
            PlausibilityError: If any validation check fails.
        """
        from .modeling import _scalar_safe_isel

        errors: list[str] = []

        for storage in self._storages:
            storage.validate_config()
            sid = storage.label_full

            # Capacity required for non-default relative bounds (DataArray checks)
            if storage.capacity_in_flow_hours is None:
                if np.any(storage.relative_minimum_charge_state > 0):
                    errors.append(
                        f'Storage "{sid}" has relative_minimum_charge_state > 0 but no capacity_in_flow_hours. '
                        f'A capacity is required because the lower bound is capacity * relative_minimum_charge_state.'
                    )
                if np.any(storage.relative_maximum_charge_state < 1):
                    errors.append(
                        f'Storage "{sid}" has relative_maximum_charge_state < 1 but no capacity_in_flow_hours. '
                        f'A capacity is required because the upper bound is capacity * relative_maximum_charge_state.'
                    )

            # Initial charge state vs capacity bounds (DataArray checks)
            if storage.capacity_in_flow_hours is not None:
                if isinstance(storage.capacity_in_flow_hours, InvestParameters):
                    minimum_capacity = storage.capacity_in_flow_hours.minimum_or_fixed_size
                    maximum_capacity = storage.capacity_in_flow_hours.maximum_or_fixed_size
                else:
                    maximum_capacity = storage.capacity_in_flow_hours
                    minimum_capacity = storage.capacity_in_flow_hours

                min_initial_at_max_capacity = maximum_capacity * _scalar_safe_isel(
                    storage.relative_minimum_charge_state, {'time': 0}
                )
                max_initial_at_min_capacity = minimum_capacity * _scalar_safe_isel(
                    storage.relative_maximum_charge_state, {'time': 0}
                )

                initial_equals_final = isinstance(storage.initial_charge_state, str)
                if not initial_equals_final and storage.initial_charge_state is not None:
                    if (storage.initial_charge_state > max_initial_at_min_capacity).any():
                        errors.append(
                            f'{sid}: initial_charge_state={storage.initial_charge_state} '
                            f'is constraining the investment decision. Choose a value <= {max_initial_at_min_capacity}.'
                        )
                    if (storage.initial_charge_state < min_initial_at_max_capacity).any():
                        errors.append(
                            f'{sid}: initial_charge_state={storage.initial_charge_state} '
                            f'is constraining the investment decision. Choose a value >= {min_initial_at_max_capacity}.'
                        )

            # Balanced charging/discharging size compatibility (DataArray checks)
            if storage.balanced:
                charging_min = storage.charging.size.minimum_or_fixed_size
                charging_max = storage.charging.size.maximum_or_fixed_size
                discharging_min = storage.discharging.size.minimum_or_fixed_size
                discharging_max = storage.discharging.size.maximum_or_fixed_size

                if (charging_min > discharging_max).any() or (charging_max < discharging_min).any():
                    errors.append(
                        f'Balancing charging and discharging Flows in {sid} need compatible minimum and maximum sizes. '
                        f'Got: charging.size.minimum={charging_min}, charging.size.maximum={charging_max} and '
                        f'discharging.size.minimum={discharging_min}, discharging.size.maximum={discharging_max}.'
                    )

        if errors:
            raise PlausibilityError('\n'.join(errors))


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

    @property
    def element_ids(self) -> list[str]:
        """List of all flow IDs (alias for ids)."""
        return self.ids

    @cached_property
    def _ids_index(self) -> pd.Index:
        """Cached pd.Index of flow IDs for fast DataArray creation."""
        return pd.Index(self.ids)

    def _categorize(self, condition) -> list[str]:
        """Return IDs of flows matching condition(flow) -> bool."""
        return [f.label_full for f in self.elements.values() if condition(f)]

    def _mask(self, condition) -> xr.DataArray:
        """Return boolean DataArray mask for condition(flow) -> bool."""
        return xr.DataArray(
            [condition(f) for f in self.elements.values()],
            dims=['flow'],
            coords={'flow': self._ids_index},
        )

    # === Flow Categorizations ===
    # All return list[str] of label_full IDs.

    @cached_property
    def with_status(self) -> list[str]:
        """IDs of flows with status parameters."""
        return self._categorize(lambda f: f.status_parameters is not None)

    # === Boolean Masks (PyPSA-style) ===
    # These enable efficient batched constraint creation using linopy's mask= parameter.

    @cached_property
    def has_status(self) -> xr.DataArray:
        """(flow,) - boolean mask for flows with status parameters."""
        return self._mask(lambda f: f.status_parameters is not None)

    @cached_property
    def has_investment(self) -> xr.DataArray:
        """(flow,) - boolean mask for flows with investment parameters."""
        return self._mask(lambda f: isinstance(f.size, InvestParameters))

    @cached_property
    def has_optional_investment(self) -> xr.DataArray:
        """(flow,) - boolean mask for flows with optional (non-mandatory) investment."""
        return self._mask(lambda f: isinstance(f.size, InvestParameters) and not f.size.mandatory)

    @cached_property
    def has_mandatory_investment(self) -> xr.DataArray:
        """(flow,) - boolean mask for flows with mandatory investment."""
        return self._mask(lambda f: isinstance(f.size, InvestParameters) and f.size.mandatory)

    @cached_property
    def has_fixed_size(self) -> xr.DataArray:
        """(flow,) - boolean mask for flows with fixed (non-investment) size."""
        return self._mask(lambda f: f.size is not None and not isinstance(f.size, InvestParameters))

    @cached_property
    def has_size(self) -> xr.DataArray:
        """(flow,) - boolean mask for flows with any size (fixed or investment)."""
        return self._mask(lambda f: f.size is not None)

    @cached_property
    def has_effects(self) -> xr.DataArray:
        """(flow,) - boolean mask for flows with effects_per_flow_hour."""
        return self._mask(lambda f: bool(f.effects_per_flow_hour))

    @cached_property
    def has_flow_hours_min(self) -> xr.DataArray:
        """(flow,) - boolean mask for flows with flow_hours_min constraint."""
        return self._mask(lambda f: f.flow_hours_min is not None)

    @cached_property
    def has_flow_hours_max(self) -> xr.DataArray:
        """(flow,) - boolean mask for flows with flow_hours_max constraint."""
        return self._mask(lambda f: f.flow_hours_max is not None)

    @cached_property
    def has_load_factor_min(self) -> xr.DataArray:
        """(flow,) - boolean mask for flows with load_factor_min constraint."""
        return self._mask(lambda f: f.load_factor_min is not None)

    @cached_property
    def has_load_factor_max(self) -> xr.DataArray:
        """(flow,) - boolean mask for flows with load_factor_max constraint."""
        return self._mask(lambda f: f.load_factor_max is not None)

    @cached_property
    def has_startup_tracking(self) -> xr.DataArray:
        """(flow,) - boolean mask for flows needing startup/shutdown tracking."""
        mask = np.zeros(len(self.ids), dtype=bool)
        if self._status_data:
            for i, fid in enumerate(self.ids):
                mask[i] = fid in self._status_data.with_startup_tracking
        return xr.DataArray(mask, dims=['flow'], coords={'flow': self._ids_index})

    @cached_property
    def has_uptime_tracking(self) -> xr.DataArray:
        """(flow,) - boolean mask for flows needing uptime duration tracking."""
        mask = np.zeros(len(self.ids), dtype=bool)
        if self._status_data:
            for i, fid in enumerate(self.ids):
                mask[i] = fid in self._status_data.with_uptime_tracking
        return xr.DataArray(mask, dims=['flow'], coords={'flow': self._ids_index})

    @cached_property
    def has_downtime_tracking(self) -> xr.DataArray:
        """(flow,) - boolean mask for flows needing downtime tracking."""
        mask = np.zeros(len(self.ids), dtype=bool)
        if self._status_data:
            for i, fid in enumerate(self.ids):
                mask[i] = fid in self._status_data.with_downtime_tracking
        return xr.DataArray(mask, dims=['flow'], coords={'flow': self._ids_index})

    @cached_property
    def has_startup_limit(self) -> xr.DataArray:
        """(flow,) - boolean mask for flows with startup limit."""
        mask = np.zeros(len(self.ids), dtype=bool)
        if self._status_data:
            for i, fid in enumerate(self.ids):
                mask[i] = fid in self._status_data.with_startup_limit
        return xr.DataArray(mask, dims=['flow'], coords={'flow': self._ids_index})

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
        """IDs of flows without size."""
        return self._categorize(lambda f: f.size is None)

    @cached_property
    def with_investment(self) -> list[str]:
        """IDs of flows with investment parameters."""
        return self._categorize(lambda f: isinstance(f.size, InvestParameters))

    @property
    def with_optional_investment(self) -> list[str]:
        """IDs of flows with optional (non-mandatory) investment."""
        return self._investment_data.with_optional if self._investment_data else []

    @property
    def with_mandatory_investment(self) -> list[str]:
        """IDs of flows with mandatory investment."""
        return self._investment_data.with_mandatory if self._investment_data else []

    @cached_property
    def with_status_only(self) -> list[str]:
        """IDs of flows with status but no investment and a fixed size."""
        return sorted(set(self.with_status) - set(self.with_investment) - set(self.without_size))

    @cached_property
    def with_investment_only(self) -> list[str]:
        """IDs of flows with investment but no status."""
        return sorted(set(self.with_investment) - set(self.with_status))

    @cached_property
    def with_status_and_investment(self) -> list[str]:
        """IDs of flows with both status and investment."""
        return sorted(set(self.with_status) & set(self.with_investment))

    @cached_property
    def with_flow_hours_min(self) -> list[str]:
        """IDs of flows with explicit flow_hours_min constraint."""
        return self._categorize(lambda f: f.flow_hours_min is not None)

    @cached_property
    def with_flow_hours_max(self) -> list[str]:
        """IDs of flows with explicit flow_hours_max constraint."""
        return self._categorize(lambda f: f.flow_hours_max is not None)

    @cached_property
    def with_flow_hours_over_periods_min(self) -> list[str]:
        """IDs of flows with explicit flow_hours_min_over_periods constraint."""
        return self._categorize(lambda f: f.flow_hours_min_over_periods is not None)

    @cached_property
    def with_flow_hours_over_periods_max(self) -> list[str]:
        """IDs of flows with explicit flow_hours_max_over_periods constraint."""
        return self._categorize(lambda f: f.flow_hours_max_over_periods is not None)

    @cached_property
    def with_load_factor_min(self) -> list[str]:
        """IDs of flows with explicit load_factor_min constraint."""
        return self._categorize(lambda f: f.load_factor_min is not None)

    @cached_property
    def with_load_factor_max(self) -> list[str]:
        """IDs of flows with explicit load_factor_max constraint."""
        return self._categorize(lambda f: f.load_factor_max is not None)

    @cached_property
    def with_effects(self) -> list[str]:
        """IDs of flows with effects_per_flow_hour defined."""
        return self._categorize(lambda f: f.effects_per_flow_hour)

    @cached_property
    def with_previous_flow_rate(self) -> list[str]:
        """IDs of flows with previous_flow_rate defined (for startup/shutdown tracking)."""
        return self._categorize(lambda f: f.previous_flow_rate is not None)

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
        return self._batched_parameter(self.with_flow_hours_min, 'flow_hours_min', ['period', 'scenario'])

    @cached_property
    def flow_hours_maximum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - maximum total flow hours for flows with explicit max."""
        return self._batched_parameter(self.with_flow_hours_max, 'flow_hours_max', ['period', 'scenario'])

    @cached_property
    def flow_hours_minimum_over_periods(self) -> xr.DataArray | None:
        """(flow, scenario) - minimum flow hours over all periods for flows with explicit min."""
        return self._batched_parameter(
            self.with_flow_hours_over_periods_min, 'flow_hours_min_over_periods', ['scenario']
        )

    @cached_property
    def flow_hours_maximum_over_periods(self) -> xr.DataArray | None:
        """(flow, scenario) - maximum flow hours over all periods for flows with explicit max."""
        return self._batched_parameter(
            self.with_flow_hours_over_periods_max, 'flow_hours_max_over_periods', ['scenario']
        )

    @cached_property
    def load_factor_minimum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - minimum load factor for flows with explicit min."""
        return self._batched_parameter(self.with_load_factor_min, 'load_factor_min', ['period', 'scenario'])

    @cached_property
    def load_factor_maximum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - maximum load factor for flows with explicit max."""
        return self._batched_parameter(self.with_load_factor_max, 'load_factor_max', ['period', 'scenario'])

    @cached_property
    def relative_minimum(self) -> xr.DataArray:
        """(flow, time, period, scenario) - relative lower bound on flow rate."""
        values = [f.relative_minimum for f in self.elements.values()]
        arr = stack_along_dim(values, 'flow', self.ids, self._model_coords(None))
        return self._ensure_canonical_order(arr)

    @cached_property
    def relative_maximum(self) -> xr.DataArray:
        """(flow, time, period, scenario) - relative upper bound on flow rate."""
        values = [f.relative_maximum for f in self.elements.values()]
        arr = stack_along_dim(values, 'flow', self.ids, self._model_coords(None))
        return self._ensure_canonical_order(arr)

    @cached_property
    def fixed_relative_profile(self) -> xr.DataArray:
        """(flow, time, period, scenario) - fixed profile. NaN = not fixed."""
        values = [
            f.fixed_relative_profile if f.fixed_relative_profile is not None else np.nan for f in self.elements.values()
        ]
        arr = stack_along_dim(values, 'flow', self.ids, self._model_coords(None))
        return self._ensure_canonical_order(arr)

    @cached_property
    def effective_relative_minimum(self) -> xr.DataArray:
        """(flow, time, period, scenario) - effective lower bound (uses fixed_profile if set)."""
        fixed = self.fixed_relative_profile
        rel_min = self.relative_minimum
        # Use DataArray.where with fast_isnull (faster than xr.where)
        return rel_min.where(fast_isnull(fixed), fixed)

    @cached_property
    def effective_relative_maximum(self) -> xr.DataArray:
        """(flow, time, period, scenario) - effective upper bound (uses fixed_profile if set)."""
        fixed = self.fixed_relative_profile
        rel_max = self.relative_maximum
        # Use DataArray.where with fast_isnull (faster than xr.where)
        return rel_max.where(fast_isnull(fixed), fixed)

    @cached_property
    def fixed_size(self) -> xr.DataArray:
        """(flow, period, scenario) - fixed size for non-investment flows. NaN for investment/no-size flows."""
        values = []
        for f in self.elements.values():
            if f.size is None or isinstance(f.size, InvestParameters):
                values.append(np.nan)
            else:
                values.append(f.size)
        arr = stack_along_dim(values, 'flow', self.ids, self._model_coords(['period', 'scenario']))
        return self._ensure_canonical_order(arr)

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
        arr = stack_along_dim(values, 'flow', self.ids, self._model_coords(['period', 'scenario']))
        return self._ensure_canonical_order(arr)

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
        arr = stack_along_dim(values, 'flow', self.ids, self._model_coords(['period', 'scenario']))
        return self._ensure_canonical_order(arr)

    @cached_property
    def absolute_lower_bounds(self) -> xr.DataArray:
        """(flow, cluster, time, period, scenario) - absolute lower bounds for flow rate.

        Logic:
        - Status flows → 0 (status variable controls activation)
        - Optional investment → 0 (invested variable controls)
        - Mandatory investment → relative_min * effective_size_lower
        - Fixed size → relative_min * effective_size_lower
        - No size → 0
        """
        # Base: relative_min * size_lower
        base = self.effective_relative_minimum * self.effective_size_lower

        # Build mask for flows that should have lb=0 (use pre-computed boolean masks)
        is_zero = self.has_status | self.has_optional_investment | fast_isnull(self.effective_size_lower)
        # Use DataArray.where (faster than xr.where)
        result = base.where(~is_zero, 0.0).fillna(0.0)
        return self._ensure_canonical_order(result)

    @cached_property
    def absolute_upper_bounds(self) -> xr.DataArray:
        """(flow, cluster, time, period, scenario) - absolute upper bounds for flow rate.

        Logic:
        - Investment flows → relative_max * effective_size_upper
        - Fixed size → relative_max * effective_size_upper
        - No size → inf
        """
        # Base: relative_max * size_upper
        base = self.effective_relative_maximum * self.effective_size_upper

        # Inf for flows without size (use DataArray.where, faster than xr.where)
        result = base.where(fast_notnull(self.effective_size_upper), np.inf)
        return self._ensure_canonical_order(result)

    # --- Investment Bounds (delegated to InvestmentData) ---

    @property
    def investment_size_minimum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - minimum size for flows with investment."""
        if not self._investment_data:
            return None
        # InvestmentData.size_minimum already has flow dim via stack_along_dim
        raw = self._investment_data.size_minimum
        return self._broadcast_existing(raw, dims=['period', 'scenario'])

    @property
    def investment_size_maximum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - maximum size for flows with investment."""
        if not self._investment_data:
            return None
        raw = self._investment_data.size_maximum
        return self._broadcast_existing(raw, dims=['period', 'scenario'])

    @property
    def optional_investment_size_minimum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - minimum size for optional investment flows."""
        if not self._investment_data:
            return None
        raw = self._investment_data.optional_size_minimum
        if raw is None:
            return None
        return self._broadcast_existing(raw, dims=['period', 'scenario'])

    @property
    def optional_investment_size_maximum(self) -> xr.DataArray | None:
        """(flow, period, scenario) - maximum size for optional investment flows."""
        if not self._investment_data:
            return None
        raw = self._investment_data.optional_size_maximum
        if raw is None:
            return None
        return self._broadcast_existing(raw, dims=['period', 'scenario'])

    # --- All-Flows Bounds (for mask-based variable creation) ---

    @cached_property
    def size_minimum_all(self) -> xr.DataArray:
        """(flow, period, scenario) - size minimum for ALL flows. NaN for non-investment flows."""
        if self.investment_size_minimum is not None:
            return self.investment_size_minimum.reindex({self.dim_name: self._ids_index})
        return xr.DataArray(
            np.nan,
            dims=[self.dim_name],
            coords={self.dim_name: self._ids_index},
        )

    @cached_property
    def size_maximum_all(self) -> xr.DataArray:
        """(flow, period, scenario) - size maximum for ALL flows. NaN for non-investment flows."""
        if self.investment_size_maximum is not None:
            return self.investment_size_maximum.reindex({self.dim_name: self._ids_index})
        return xr.DataArray(
            np.nan,
            dims=[self.dim_name],
            coords={self.dim_name: self._ids_index},
        )

    @cached_property
    def dim_name(self) -> str:
        """Dimension name for this data container."""
        return 'flow'

    @cached_property
    def effects_per_flow_hour(self) -> xr.DataArray | None:
        """(flow, effect, ...) - effect factors per flow hour.

        Missing (flow, effect) combinations are 0 (pre-filled for efficient computation).
        """
        if not self.with_effects:
            return None

        effect_ids = list(self._fs.effects.keys())
        if not effect_ids:
            return None

        dicts = {fid: self[fid].effects_per_flow_hour for fid in self.with_effects}
        return build_effects_array(dicts, effect_ids, 'flow')

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
        arr = stack_along_dim(values, 'flow', self.ids, self._model_coords(['period']))
        return self._ensure_canonical_order(arr)

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

    def _batched_parameter(
        self,
        ids: list[str],
        attr: str,
        dims: list[str] | None,
    ) -> xr.DataArray | None:
        """Build a batched parameter array from per-flow attributes.

        Args:
            ids: Flow IDs to include (typically from a with_* property).
            attr: Attribute name to extract from each Flow.
            dims: Model dimensions to broadcast to (e.g., ['period', 'scenario']).

        Returns:
            DataArray with (flow, *dims) or None if ids is empty.
        """
        if not ids:
            return None
        values = [getattr(self[fid], attr) for fid in ids]
        arr = stack_along_dim(values, 'flow', ids, self._model_coords(dims))
        return self._ensure_canonical_order(arr)

    def _model_coords(self, dims: list[str] | None = None) -> dict[str, pd.Index | np.ndarray]:
        """Get model coordinates for broadcasting.

        Args:
            dims: Dimensions to include. None = all (time, period, scenario).

        Returns:
            Dict of dim name -> coordinate values.
        """
        if dims is None:
            dims = ['time', 'period', 'scenario']
        indexes = self._fs.indexes
        return {dim: indexes[dim] for dim in dims if dim in indexes}

    def _ensure_canonical_order(self, arr: xr.DataArray) -> xr.DataArray:
        """Ensure array has canonical dimension order and coord dict order.

        Args:
            arr: Input DataArray.

        Returns:
            DataArray with dims in order (flow, cluster, time, period, scenario, ...) and
            coords dict matching dims order. Additional dims are appended at the end.
        """
        # Note: cluster comes before time to match FlowSystem.dims ordering
        canonical_order = ['flow', 'cluster', 'time', 'period', 'scenario']
        # Start with canonical dims that exist in arr
        actual_dims = [d for d in canonical_order if d in arr.dims]
        # Append any additional dims not in canonical order
        for d in arr.dims:
            if d not in actual_dims:
                actual_dims.append(d)

        if list(arr.dims) != actual_dims:
            arr = arr.transpose(*actual_dims)

        # Ensure coords dict order matches dims order (linopy uses coords order)
        if list(arr.coords.keys()) != list(arr.dims):
            ordered_coords = {d: arr.coords[d] for d in arr.dims}
            arr = xr.DataArray(arr.values, dims=arr.dims, coords=ordered_coords)

        return arr

    def _broadcast_existing(self, arr: xr.DataArray, dims: list[str] | None = None) -> xr.DataArray:
        """Broadcast an existing DataArray (with element dim) to model coordinates.

        Use this for arrays that already have the flow dimension (e.g., from InvestmentData).

        Args:
            arr: DataArray with flow dimension.
            dims: Model dimensions to add. None = all (time, period, scenario).

        Returns:
            DataArray with dimensions in canonical order: (flow, time, period, scenario)
        """
        coords_to_add = self._model_coords(dims)

        if not coords_to_add:
            return self._ensure_canonical_order(arr)

        # Broadcast to include new dimensions
        for dim_name, coord in coords_to_add.items():
            if dim_name not in arr.dims:
                arr = arr.expand_dims({dim_name: coord})

        return self._ensure_canonical_order(arr)

    # === Validation ===

    def validate(self) -> None:
        """Validate all flows (config + DataArray checks).

        Performs both:
        - Config validation via Flow.validate_config()
        - DataArray validation (post-transformation checks)

        Raises:
            PlausibilityError: If any validation check fails.
        """
        # Early return if no flows (avoids empty DataArray operations)
        if not self.elements:
            return

        for flow in self.elements.values():
            flow.validate_config()

        errors: list[str] = []

        # Batched checks: relative_minimum <= relative_maximum
        invalid_bounds = (self.relative_minimum > self.relative_maximum).any(
            dim=[d for d in self.relative_minimum.dims if d != 'flow']
        )
        if invalid_bounds.any():
            bad_flows = [fid for fid, bad in zip(self.ids, invalid_bounds.values, strict=False) if bad]
            errors.append(f'relative_minimum > relative_maximum for flows: {bad_flows}')

        # Check: size required when relative_minimum > 0
        has_nonzero_min = (self.relative_minimum > 0).any(dim=[d for d in self.relative_minimum.dims if d != 'flow'])
        needs_size_for_min = has_nonzero_min & ~self.has_size
        if needs_size_for_min.any():
            bad_flows = [fid for fid, bad in zip(self.ids, needs_size_for_min.values, strict=False) if bad]
            errors.append(
                f'relative_minimum > 0 but no size defined for flows: {bad_flows}. '
                f'A size is required because the lower bound is size * relative_minimum.'
            )

        # Check: size required when relative_maximum < 1
        has_nondefault_max = (self.relative_maximum < 1).any(dim=[d for d in self.relative_maximum.dims if d != 'flow'])
        needs_size_for_max = has_nondefault_max & ~self.has_size
        if needs_size_for_max.any():
            bad_flows = [fid for fid, bad in zip(self.ids, needs_size_for_max.values, strict=False) if bad]
            errors.append(
                f'relative_maximum < 1 but no size defined for flows: {bad_flows}. '
                f'A size is required because the upper bound is size * relative_maximum.'
            )

        # Warning: relative_minimum > 0 without status_parameters prevents switching inactive
        has_nonzero_min_no_status = has_nonzero_min & ~self.has_status
        if has_nonzero_min_no_status.any():
            warn_flows = [fid for fid, warn in zip(self.ids, has_nonzero_min_no_status.values, strict=False) if warn]
            logger.warning(
                f'Flows {warn_flows} have relative_minimum > 0 and no status_parameters. '
                f'This prevents the flow from switching inactive (flow_rate = 0). '
                f'Consider using status_parameters to allow switching active and inactive.'
            )

        # Warning: status_parameters with relative_minimum=0 allows status=1 with flow=0
        has_zero_min_with_status = ~has_nonzero_min & self.has_status
        if has_zero_min_with_status.any():
            warn_flows = [fid for fid, warn in zip(self.ids, has_zero_min_with_status.values, strict=False) if warn]
            logger.warning(
                f'Flows {warn_flows} have status_parameters but relative_minimum=0. '
                f'This allows status=1 with flow=0, which may lead to unexpected behavior. '
                f'Consider setting relative_minimum > 0 to ensure the unit produces when active.'
            )

        if errors:
            raise PlausibilityError('\n'.join(errors))


class EffectsData:
    """Batched data container for all effects.

    Provides indexed access to effect properties as stacked xr.DataArrays
    with an 'effect' dimension. Separates data access from mathematical
    modeling (EffectsModel).
    """

    def __init__(self, effect_collection: EffectCollection):
        self._collection = effect_collection
        self._effects: list[Effect] = list(effect_collection.values())

    @cached_property
    def effect_ids(self) -> list[str]:
        return [e.label for e in self._effects]

    @property
    def element_ids(self) -> list[str]:
        """Alias for effect_ids."""
        return self.effect_ids

    @property
    def dim_name(self) -> str:
        """Dimension name for this data container."""
        return 'effect'

    @cached_property
    def effect_index(self) -> pd.Index:
        return pd.Index(self.effect_ids, name='effect')

    @property
    def objective_effect_id(self) -> str:
        return self._collection.objective_effect.label

    @property
    def penalty_effect_id(self) -> str:
        return self._collection.penalty_effect.label

    def _effect_values(self, attr_name: str, default: float) -> list:
        """Extract per-effect attribute values, substituting default for None."""
        values = []
        for effect in self._effects:
            val = getattr(effect, attr_name, None)
            values.append(default if val is None else val)
        return values

    @cached_property
    def minimum_periodic(self) -> xr.DataArray:
        return stack_along_dim(self._effect_values('minimum_periodic', -np.inf), 'effect', self.effect_ids)

    @cached_property
    def maximum_periodic(self) -> xr.DataArray:
        return stack_along_dim(self._effect_values('maximum_periodic', np.inf), 'effect', self.effect_ids)

    @cached_property
    def minimum_temporal(self) -> xr.DataArray:
        return stack_along_dim(self._effect_values('minimum_temporal', -np.inf), 'effect', self.effect_ids)

    @cached_property
    def maximum_temporal(self) -> xr.DataArray:
        return stack_along_dim(self._effect_values('maximum_temporal', np.inf), 'effect', self.effect_ids)

    @cached_property
    def minimum_per_hour(self) -> xr.DataArray:
        return stack_along_dim(self._effect_values('minimum_per_hour', -np.inf), 'effect', self.effect_ids)

    @cached_property
    def maximum_per_hour(self) -> xr.DataArray:
        return stack_along_dim(self._effect_values('maximum_per_hour', np.inf), 'effect', self.effect_ids)

    @cached_property
    def minimum_total(self) -> xr.DataArray:
        return stack_along_dim(self._effect_values('minimum_total', -np.inf), 'effect', self.effect_ids)

    @cached_property
    def maximum_total(self) -> xr.DataArray:
        return stack_along_dim(self._effect_values('maximum_total', np.inf), 'effect', self.effect_ids)

    @cached_property
    def minimum_over_periods(self) -> xr.DataArray:
        return stack_along_dim(self._effect_values('minimum_over_periods', -np.inf), 'effect', self.effect_ids)

    @cached_property
    def maximum_over_periods(self) -> xr.DataArray:
        return stack_along_dim(self._effect_values('maximum_over_periods', np.inf), 'effect', self.effect_ids)

    @cached_property
    def effects_with_over_periods(self) -> list[Effect]:
        return [e for e in self._effects if e.minimum_over_periods is not None or e.maximum_over_periods is not None]

    @property
    def period_weights(self) -> dict[str, xr.DataArray]:
        """Get period weights for each effect, keyed by effect label."""
        result = {}
        for effect in self._effects:
            effect_weights = effect.period_weights
            default_weights = effect._flow_system.period_weights
            if effect_weights is not None:
                result[effect.label] = effect_weights
            elif default_weights is not None:
                result[effect.label] = default_weights
            else:
                result[effect.label] = effect._fit_coords(name='period_weights', data=1, dims=['period'])
        return result

    def effects(self) -> list[Effect]:
        """Access the underlying effect objects."""
        return self._effects

    def __getitem__(self, label: str) -> Effect:
        """Look up an effect by label (delegates to the collection)."""
        return self._collection[label]

    def values(self):
        """Iterate over Effect objects."""
        return self._effects

    def validate(self) -> None:
        """Validate all effects and the effect collection structure.

        Performs both:
        - Individual effect config validation
        - Collection-level validation (circular loops in share mappings, unknown effect refs)
        """
        for effect in self._effects:
            effect.validate_config()

        # Collection-level validation (share structure)
        self._validate_share_structure()

    def _validate_share_structure(self) -> None:
        """Validate effect share mappings for cycles and unknown references."""
        from .effects import detect_cycles, tuples_to_adjacency_list

        temporal, periodic = self._collection.calculate_effect_share_factors()

        # Validate all referenced effects exist
        edges = list(temporal.keys()) + list(periodic.keys())
        unknown_sources = {src for src, _ in edges if src not in self._collection}
        unknown_targets = {tgt for _, tgt in edges if tgt not in self._collection}
        unknown = unknown_sources | unknown_targets
        if unknown:
            raise KeyError(f'Unknown effects used in effect share mappings: {sorted(unknown)}')

        # Check for circular dependencies
        temporal_cycles = detect_cycles(tuples_to_adjacency_list([key for key in temporal]))
        periodic_cycles = detect_cycles(tuples_to_adjacency_list([key for key in periodic]))

        if temporal_cycles:
            cycle_str = '\n'.join([' -> '.join(cycle) for cycle in temporal_cycles])
            raise ValueError(f'Error: circular temporal-shares detected:\n{cycle_str}')

        if periodic_cycles:
            cycle_str = '\n'.join([' -> '.join(cycle) for cycle in periodic_cycles])
            raise ValueError(f'Error: circular periodic-shares detected:\n{cycle_str}')


class BusesData:
    """Batched data container for buses."""

    def __init__(self, buses: list[Bus]):
        self._buses = buses
        self.elements: ElementContainer = ElementContainer(buses)

    @property
    def element_ids(self) -> list[str]:
        return list(self.elements.keys())

    @property
    def dim_name(self) -> str:
        return 'bus'

    @cached_property
    def with_imbalance(self) -> list[str]:
        """IDs of buses allowing imbalance."""
        return [b.label_full for b in self._buses if b.allows_imbalance]

    @cached_property
    def imbalance_elements(self) -> list[Bus]:
        """Bus objects that allow imbalance."""
        return [b for b in self._buses if b.allows_imbalance]

    def validate(self) -> None:
        """Validate all buses (config + DataArray checks).

        Performs both:
        - Config validation via Bus.validate_config()
        - DataArray validation (post-transformation checks)
        """
        for bus in self._buses:
            bus.validate_config()
            # Warning: imbalance_penalty == 0 (DataArray check)
            if bus.imbalance_penalty_per_flow_hour is not None:
                zero_penalty = np.all(np.equal(bus.imbalance_penalty_per_flow_hour, 0))
                if zero_penalty:
                    logger.warning(
                        f'In Bus {bus.label_full}, the imbalance_penalty_per_flow_hour is 0. Use "None" or a value > 0.'
                    )


class ComponentsData:
    """Batched data container for components with status."""

    def __init__(self, components_with_status: list[Component], all_components: list[Component]):
        self._components_with_status = components_with_status
        self._all_components = all_components
        self.elements: ElementContainer = ElementContainer(components_with_status)

    @property
    def element_ids(self) -> list[str]:
        return list(self.elements.keys())

    @property
    def dim_name(self) -> str:
        return 'component'

    @property
    def all_components(self) -> list[Component]:
        return self._all_components

    def validate(self) -> None:
        """Validate generic components (config checks only).

        Note: Storage, Transmission, and LinearConverter are validated
        through their specialized *Data classes, so we skip them here.
        """
        from .components import LinearConverter, Storage, Transmission

        for component in self._all_components:
            if not isinstance(component, (Storage, LinearConverter, Transmission)):
                component.validate_config()


class ConvertersData:
    """Batched data container for converters."""

    def __init__(self, converters: list[LinearConverter]):
        self._converters = converters
        self.elements: ElementContainer = ElementContainer(converters)

    @property
    def element_ids(self) -> list[str]:
        return list(self.elements.keys())

    @property
    def dim_name(self) -> str:
        return 'converter'

    @cached_property
    def with_factors(self) -> list[LinearConverter]:
        """Converters with conversion_factors."""
        return [c for c in self._converters if c.conversion_factors]

    @cached_property
    def with_piecewise(self) -> list[LinearConverter]:
        """Converters with piecewise_conversion."""
        return [c for c in self._converters if c.piecewise_conversion]

    def validate(self) -> None:
        """Validate all converters (config checks, no DataArray operations needed)."""
        for converter in self._converters:
            converter.validate_config()


class TransmissionsData:
    """Batched data container for transmissions."""

    def __init__(self, transmissions: list[Transmission]):
        self._transmissions = transmissions
        self.elements: ElementContainer = ElementContainer(transmissions)

    @property
    def element_ids(self) -> list[str]:
        return list(self.elements.keys())

    @property
    def dim_name(self) -> str:
        return 'transmission'

    @cached_property
    def bidirectional(self) -> list[Transmission]:
        """Transmissions that are bidirectional."""
        return [t for t in self._transmissions if t.in2 is not None]

    @cached_property
    def balanced(self) -> list[Transmission]:
        """Transmissions with balanced flow sizes."""
        return [t for t in self._transmissions if t.balanced]

    def validate(self) -> None:
        """Validate all transmissions (config + DataArray checks).

        Performs both:
        - Config validation via Transmission.validate_config()
        - DataArray validation (post-transformation checks)

        Raises:
            PlausibilityError: If any validation check fails.
        """
        for transmission in self._transmissions:
            transmission.validate_config()

        errors: list[str] = []

        for transmission in self._transmissions:
            tid = transmission.label_full

            # Balanced size compatibility (DataArray check)
            if transmission.balanced:
                in1_min = transmission.in1.size.minimum_or_fixed_size
                in1_max = transmission.in1.size.maximum_or_fixed_size
                in2_min = transmission.in2.size.minimum_or_fixed_size
                in2_max = transmission.in2.size.maximum_or_fixed_size

                if (in1_min > in2_max).any() or (in1_max < in2_min).any():
                    errors.append(
                        f'Balanced Transmission {tid} needs compatible minimum and maximum sizes. '
                        f'Got: in1.size.minimum={in1_min}, in1.size.maximum={in1_max} and '
                        f'in2.size.minimum={in2_min}, in2.size.maximum={in2_max}.'
                    )

        if errors:
            raise PlausibilityError('\n'.join(errors))


class BatchedAccessor:
    """Accessor for batched data containers on FlowSystem.

    Provides cached access to *Data containers for all element types.
    The same cached instances are used for both validation (during connect_and_transform)
    and model building, ensuring consistency and avoiding duplicate object creation.

    Usage:
        flow_system.batched.flows      # Access FlowsData
        flow_system.batched.storages   # Access StoragesData
        flow_system.batched.buses      # Access BusesData
    """

    def __init__(self, flow_system: FlowSystem):
        self._fs = flow_system
        self._flows: FlowsData | None = None
        self._storages: StoragesData | None = None
        self._intercluster_storages: StoragesData | None = None
        self._buses: BusesData | None = None
        self._effects: EffectsData | None = None
        self._components: ComponentsData | None = None
        self._converters: ConvertersData | None = None
        self._transmissions: TransmissionsData | None = None

    @property
    def flows(self) -> FlowsData:
        """Get or create FlowsData for all flows in the system."""
        if self._flows is None:
            all_flows = list(self._fs.flows.values())
            self._flows = FlowsData(all_flows, self._fs)
        return self._flows

    @property
    def storages(self) -> StoragesData:
        """Get or create StoragesData for basic storages (excludes intercluster)."""
        if self._storages is None:
            from .components import Storage

            clustering = self._fs.clustering
            basic_storages = [
                c
                for c in self._fs.components.values()
                if isinstance(c, Storage)
                and not (clustering is not None and c.cluster_mode in ('intercluster', 'intercluster_cyclic'))
            ]
            effect_ids = list(self._fs.effects.keys())
            self._storages = StoragesData(
                basic_storages, 'storage', effect_ids, timesteps_extra=self._fs.timesteps_extra
            )
        return self._storages

    @property
    def intercluster_storages(self) -> StoragesData:
        """Get or create StoragesData for intercluster storages."""
        if self._intercluster_storages is None:
            from .components import Storage

            clustering = self._fs.clustering
            intercluster = [
                c
                for c in self._fs.components.values()
                if isinstance(c, Storage)
                and clustering is not None
                and c.cluster_mode in ('intercluster', 'intercluster_cyclic')
            ]
            effect_ids = list(self._fs.effects.keys())
            self._intercluster_storages = StoragesData(intercluster, 'intercluster_storage', effect_ids)
        return self._intercluster_storages

    @property
    def buses(self) -> BusesData:
        """Get or create BusesData for all buses."""
        if self._buses is None:
            self._buses = BusesData(list(self._fs.buses.values()))
        return self._buses

    @property
    def effects(self) -> EffectsData:
        """Get or create EffectsData for all effects."""
        if self._effects is None:
            self._effects = EffectsData(self._fs.effects)
        return self._effects

    @property
    def components(self) -> ComponentsData:
        """Get or create ComponentsData for all components."""
        if self._components is None:
            all_components = list(self._fs.components.values())
            components_with_status = [c for c in all_components if c.status_parameters is not None]
            self._components = ComponentsData(components_with_status, all_components)
        return self._components

    @property
    def converters(self) -> ConvertersData:
        """Get or create ConvertersData for all converters."""
        if self._converters is None:
            from .components import LinearConverter

            converters = [c for c in self._fs.components.values() if isinstance(c, LinearConverter)]
            self._converters = ConvertersData(converters)
        return self._converters

    @property
    def transmissions(self) -> TransmissionsData:
        """Get or create TransmissionsData for all transmissions."""
        if self._transmissions is None:
            from .components import Transmission

            transmissions = [c for c in self._fs.components.values() if isinstance(c, Transmission)]
            self._transmissions = TransmissionsData(transmissions)
        return self._transmissions

    def _reset(self) -> None:
        """Reset all cached data (called when FlowSystem is invalidated)."""
        self._flows = None
        self._storages = None
        self._intercluster_storages = None
        self._buses = None
        self._effects = None
        self._components = None
        self._converters = None
        self._transmissions = None
