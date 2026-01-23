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
    def with_flow_hours_over_periods(self) -> list[str]:
        """IDs of flows with flow_hours_over_periods constraints."""
        return [
            f.label_full
            for f in self.elements.values()
            if f.flow_hours_min_over_periods is not None or f.flow_hours_max_over_periods is not None
        ]

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
    # All return xr.DataArray with 'flow' dimension.

    @cached_property
    def flow_hours_minimum(self) -> xr.DataArray:
        """(flow, period, scenario) - minimum total flow hours. NaN = no constraint."""
        values = [f.flow_hours_min if f.flow_hours_min is not None else np.nan for f in self.elements.values()]
        return self._broadcast_to_coords(self._stack_values(values), dims=['period', 'scenario'])

    @cached_property
    def flow_hours_maximum(self) -> xr.DataArray:
        """(flow, period, scenario) - maximum total flow hours. NaN = no constraint."""
        values = [f.flow_hours_max if f.flow_hours_max is not None else np.nan for f in self.elements.values()]
        return self._broadcast_to_coords(self._stack_values(values), dims=['period', 'scenario'])

    @cached_property
    def flow_hours_minimum_over_periods(self) -> xr.DataArray:
        """(flow, scenario) - minimum flow hours summed over all periods. NaN = no constraint."""
        values = [
            f.flow_hours_min_over_periods if f.flow_hours_min_over_periods is not None else np.nan
            for f in self.elements.values()
        ]
        return self._broadcast_to_coords(self._stack_values(values), dims=['scenario'])

    @cached_property
    def flow_hours_maximum_over_periods(self) -> xr.DataArray:
        """(flow, scenario) - maximum flow hours summed over all periods. NaN = no constraint."""
        values = [
            f.flow_hours_max_over_periods if f.flow_hours_max_over_periods is not None else np.nan
            for f in self.elements.values()
        ]
        return self._broadcast_to_coords(self._stack_values(values), dims=['scenario'])

    @cached_property
    def load_factor_minimum(self) -> xr.DataArray:
        """(flow, period, scenario) - minimum load factor. NaN = no constraint."""
        values = [f.load_factor_min if f.load_factor_min is not None else np.nan for f in self.elements.values()]
        return self._broadcast_to_coords(self._stack_values(values), dims=['period', 'scenario'])

    @cached_property
    def load_factor_maximum(self) -> xr.DataArray:
        """(flow, period, scenario) - maximum load factor. NaN = no constraint."""
        values = [f.load_factor_max if f.load_factor_max is not None else np.nan for f in self.elements.values()]
        return self._broadcast_to_coords(self._stack_values(values), dims=['period', 'scenario'])

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
    def size_minimum(self) -> xr.DataArray:
        """(flow, period, scenario) - minimum size. NaN for flows without size."""
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
    def size_maximum(self) -> xr.DataArray:
        """(flow, period, scenario) - maximum size. NaN for flows without size."""
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

    # === Helper Methods ===

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
