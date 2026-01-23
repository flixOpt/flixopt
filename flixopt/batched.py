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

        # Get model coordinates
        if dims is None:
            dims = ['time', 'period', 'scenario']

        coords_to_add = {}
        if 'time' in dims and self._fs.timesteps is not None:
            coords_to_add['time'] = self._fs.timesteps
        if 'period' in dims and self._fs.periods is not None:
            coords_to_add['period'] = self._fs.periods
        if 'scenario' in dims and self._fs.scenarios is not None:
            coords_to_add['scenario'] = self._fs.scenarios

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
