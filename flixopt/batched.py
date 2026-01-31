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
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from .features import InvestmentHelpers, concat_with_coords, fast_isnull, fast_notnull
from .interface import InvestParameters, StatusParameters
from .structure import ElementContainer

if TYPE_CHECKING:
    from .effects import Effect, EffectCollection
    from .elements import Flow
    from .flow_system import FlowSystem


def stack_and_broadcast(
    values: list[float | xr.DataArray],
    element_ids: list[str] | pd.Index,
    element_dim: str,
    target_coords: dict[str, pd.Index | np.ndarray] | None = None,
) -> xr.DataArray:
    """Stack per-element values and broadcast to target coordinates.

    Always returns a DataArray with element_dim as first dimension,
    followed by target dimensions in the order provided.

    Args:
        values: Per-element values (scalars or DataArrays with any dims).
        element_ids: Element IDs for the stacking dimension.
        element_dim: Name of element dimension ('flow', 'storage', etc.).
        target_coords: Coords to broadcast to (e.g., {'time': ..., 'period': ...}).
            Order determines output dimension order after element_dim.

    Returns:
        DataArray with dims (element_dim, *target_dims) and all values broadcast
        to the full shape.
    """
    if not isinstance(element_ids, pd.Index):
        element_ids = pd.Index(element_ids)

    target_coords = target_coords or {}

    # Collect coords from input arrays (may have subset of target dims)
    collected_coords: dict[str, Any] = {}
    for v in values:
        if isinstance(v, xr.DataArray) and v.ndim > 0:
            for d in v.dims:
                if d not in collected_coords:
                    collected_coords[d] = v.coords[d].values

    # Merge: target_coords take precedence, add any from collected
    final_coords = dict(target_coords)
    for d, c in collected_coords.items():
        if d not in final_coords:
            final_coords[d] = c

    # Build full shape: (n_elements, *target_dims)
    n_elements = len(element_ids)
    extra_dims = list(final_coords.keys())
    extra_shape = [len(c) for c in final_coords.values()]
    full_shape = [n_elements] + extra_shape
    full_dims = [element_dim] + extra_dims

    # Pre-allocate with NaN
    data = np.full(full_shape, np.nan)

    # Create template for broadcasting (if we have extra dims)
    template = xr.DataArray(coords=final_coords, dims=extra_dims) if final_coords else None

    # Fill in values
    for i, v in enumerate(values):
        if isinstance(v, xr.DataArray):
            if v.ndim == 0:
                data[i, ...] = float(v.values)
            elif template is not None:
                # Broadcast to template shape
                broadcasted = v.broadcast_like(template)
                data[i, ...] = broadcasted.values
            else:
                data[i, ...] = v.values
        elif not (isinstance(v, float) and np.isnan(v)):
            data[i, ...] = float(v)
        # else: leave as NaN

    # Build coords with element_dim first
    full_coords = {element_dim: element_ids}
    full_coords.update(final_coords)

    return xr.DataArray(data, coords=full_coords, dims=full_dims)


def build_effects_array(
    params: dict[str, Any],
    attr: str,
    ids: list[str],
    effect_ids: list[str],
    dim_name: str,
) -> xr.DataArray | None:
    """Build effect factors array from per-element effect dicts.

    Args:
        params: Dict mapping element_id -> parameter object with effect attributes.
        attr: Attribute name on the parameter object (e.g., 'effects_per_startup').
        ids: Element IDs to include (must have truthy attr values).
        effect_ids: List of effect IDs for the effect dimension.
        dim_name: Element dimension name ('flow', 'storage', etc.).

    Returns:
        DataArray with (dim_name, effect, ...) or None if ids or effect_ids empty.
    """
    if not ids or not effect_ids:
        return None

    factors = [
        xr.concat(
            [xr.DataArray(getattr(params[eid], attr).get(eff, 0.0)) for eff in effect_ids],
            dim='effect',
            coords='minimal',
        ).assign_coords(effect=effect_ids)
        for eid in ids
    ]

    return concat_with_coords(factors, dim_name, ids)


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
        ids = self._categorize(lambda p: getattr(p, attr))
        return build_effects_array(self._params, attr, ids, self._effect_ids, self._dim)

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
            ids = self._categorize(lambda p: getattr(p, attr))
        return build_effects_array(self._params, attr, ids, self._effect_ids, self._dim)

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
        from .features import PiecewiseHelpers

        ids = self.with_piecewise_effects
        if not ids:
            return {}

        dim = self._dim
        params = self._params

        # Segment counts and mask
        segment_counts = {eid: len(params[eid].piecewise_effects_of_investment.piecewise_origin) for eid in ids}
        max_segments, segment_mask = PiecewiseHelpers.collect_segment_info(ids, segment_counts, dim)

        # Origin breakpoints (for size coupling)
        origin_breakpoints = {}
        for eid in ids:
            pieces = params[eid].piecewise_effects_of_investment.piecewise_origin
            origin_breakpoints[eid] = ([p.start for p in pieces], [p.end for p in pieces])
        origin_starts, origin_ends = PiecewiseHelpers.pad_breakpoints(ids, origin_breakpoints, max_segments, dim)

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
            s, e = PiecewiseHelpers.pad_breakpoints(ids, breakpoints, max_segments, dim)
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

    def __init__(self, storages: list, dim_name: str, effect_ids: list[str]):
        """Initialize StoragesData.

        Args:
            storages: List of Storage elements.
            dim_name: Dimension name for arrays ('storage' or 'intercluster_storage').
            effect_ids: List of effect IDs for building effect arrays.
        """
        self._storages = storages
        self._dim_name = dim_name
        self._effect_ids = effect_ids
        self._by_label = {s.label_full: s for s in storages}

    @cached_property
    def ids(self) -> list[str]:
        """All storage IDs (label_full)."""
        return [s.label_full for s in self._storages]

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

    # === Bounds ===

    @cached_property
    def charge_state_lower(self) -> xr.DataArray:
        """(element,) - minimum size for investment storages."""
        element_ids = self.with_investment
        values = [self._by_label[sid].capacity_in_flow_hours.minimum_or_fixed_size for sid in element_ids]
        return InvestmentHelpers.stack_bounds(values, element_ids, self._dim_name)

    @cached_property
    def charge_state_upper(self) -> xr.DataArray:
        """(element,) - maximum size for investment storages."""
        element_ids = self.with_investment
        values = [self._by_label[sid].capacity_in_flow_hours.maximum_or_fixed_size for sid in element_ids]
        return InvestmentHelpers.stack_bounds(values, element_ids, self._dim_name)


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
        arr = stack_and_broadcast(values, self.ids, 'flow', self._model_coords(None))
        return self._ensure_canonical_order(arr)

    @cached_property
    def relative_maximum(self) -> xr.DataArray:
        """(flow, time, period, scenario) - relative upper bound on flow rate."""
        values = [f.relative_maximum for f in self.elements.values()]
        arr = stack_and_broadcast(values, self.ids, 'flow', self._model_coords(None))
        return self._ensure_canonical_order(arr)

    @cached_property
    def fixed_relative_profile(self) -> xr.DataArray:
        """(flow, time, period, scenario) - fixed profile. NaN = not fixed."""
        values = [
            f.fixed_relative_profile if f.fixed_relative_profile is not None else np.nan for f in self.elements.values()
        ]
        arr = stack_and_broadcast(values, self.ids, 'flow', self._model_coords(None))
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
        arr = stack_and_broadcast(values, self.ids, 'flow', self._model_coords(['period', 'scenario']))
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
        arr = stack_and_broadcast(values, self.ids, 'flow', self._model_coords(['period', 'scenario']))
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
        arr = stack_and_broadcast(values, self.ids, 'flow', self._model_coords(['period', 'scenario']))
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
        # InvestmentData.size_minimum already has flow dim via InvestmentHelpers.stack_bounds
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

        flow_ids = self.with_effects

        # Determine required dimensions by scanning all effect values
        extra_dims: dict[str, pd.Index] = {}
        for fid in flow_ids:
            flow_effects = self[fid].effects_per_flow_hour
            for val in flow_effects.values():
                if isinstance(val, xr.DataArray) and val.ndim > 0:
                    for dim in val.dims:
                        if dim not in extra_dims:
                            extra_dims[dim] = val.coords[dim].values

        # Build shape and coords
        shape = [len(flow_ids), len(effect_ids)]
        dims = ['flow', 'effect']
        coords: dict = {'flow': pd.Index(flow_ids), 'effect': pd.Index(effect_ids)}

        for dim, coord_vals in extra_dims.items():
            shape.append(len(coord_vals))
            dims.append(dim)
            coords[dim] = pd.Index(coord_vals)

        # Pre-allocate numpy array with zeros (pre-filled, avoids fillna later)
        data = np.zeros(shape)

        # Fill in values
        for i, fid in enumerate(flow_ids):
            flow_effects = self[fid].effects_per_flow_hour
            for j, eff in enumerate(effect_ids):
                val = flow_effects.get(eff)
                if val is None:
                    continue
                elif isinstance(val, xr.DataArray):
                    if val.ndim == 0:
                        # Scalar DataArray - broadcast to all extra dims
                        data[i, j, ...] = float(val.values)
                    else:
                        # Multi-dimensional - place in correct position
                        # Build slice for this value's dimensions
                        data[i, j, ...] = val.values
                else:
                    # Python scalar - broadcast to all extra dims
                    data[i, j, ...] = float(val)

        return xr.DataArray(data, coords=coords, dims=dims)

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
        arr = stack_and_broadcast(values, self.ids, 'flow', self._model_coords(['period']))
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
        arr = stack_and_broadcast(values, ids, 'flow', self._model_coords(dims))
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

    @cached_property
    def effect_index(self) -> pd.Index:
        return pd.Index(self.effect_ids, name='effect')

    @property
    def objective_effect_id(self) -> str:
        return self._collection.objective_effect.label

    @property
    def penalty_effect_id(self) -> str:
        return self._collection.penalty_effect.label

    def _stack_bounds(self, attr_name: str, default: float = np.inf) -> xr.DataArray:
        """Stack per-effect bounds into a single DataArray with effect dimension."""

        def as_dataarray(effect) -> xr.DataArray:
            val = getattr(effect, attr_name, None)
            if val is None:
                return xr.DataArray(default)
            return val if isinstance(val, xr.DataArray) else xr.DataArray(val)

        return xr.concat(
            [as_dataarray(e).expand_dims(effect=[e.label]) for e in self._effects],
            dim='effect',
            fill_value=default,
        )

    @cached_property
    def minimum_periodic(self) -> xr.DataArray:
        return self._stack_bounds('minimum_periodic', -np.inf)

    @cached_property
    def maximum_periodic(self) -> xr.DataArray:
        return self._stack_bounds('maximum_periodic', np.inf)

    @cached_property
    def minimum_temporal(self) -> xr.DataArray:
        return self._stack_bounds('minimum_temporal', -np.inf)

    @cached_property
    def maximum_temporal(self) -> xr.DataArray:
        return self._stack_bounds('maximum_temporal', np.inf)

    @cached_property
    def minimum_per_hour(self) -> xr.DataArray:
        return self._stack_bounds('minimum_per_hour', -np.inf)

    @cached_property
    def maximum_per_hour(self) -> xr.DataArray:
        return self._stack_bounds('maximum_per_hour', np.inf)

    @cached_property
    def minimum_total(self) -> xr.DataArray:
        return self._stack_bounds('minimum_total', -np.inf)

    @cached_property
    def maximum_total(self) -> xr.DataArray:
        return self._stack_bounds('maximum_total', np.inf)

    @cached_property
    def minimum_over_periods(self) -> xr.DataArray:
        return self._stack_bounds('minimum_over_periods', -np.inf)

    @cached_property
    def maximum_over_periods(self) -> xr.DataArray:
        return self._stack_bounds('maximum_over_periods', np.inf)

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
