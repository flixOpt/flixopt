"""Helper utilities for inter-cluster storage linking.

This module provides reusable utilities for building inter-cluster storage linking
constraints following the S-N model from Blanke et al. (2022).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..flow_system import FlowSystem
    from ..interface import InvestParameters
    from .base import ClusterStructure


@dataclass
class SliceContext:
    """Context for a (period, scenario) slice during constraint generation.

    Provides the current iteration state when iterating over multi-dimensional
    cluster orders, along with helper methods for constraint naming.
    """

    period: str | int | None
    scenario: str | None
    cluster_order: np.ndarray

    @property
    def suffix(self) -> str:
        """Generate constraint name suffix like '_p2020_shigh'."""
        parts = []
        if self.period is not None:
            parts.append(f'p{self.period}')
        if self.scenario is not None:
            parts.append(f's{self.scenario}')
        return '_' + '_'.join(parts) if parts else ''


class MultiDimIterator:
    """Unified iterator over (period, scenario) combinations.

    Provides a clean interface for iterating over multi-dimensional slices
    with automatic handling of None cases and selector building.

    Example:
        iterator = MultiDimIterator(flow_system, cluster_structure)
        for ctx in iterator:
            # ctx.period, ctx.scenario, ctx.cluster_order available
            selector = iterator.build_selector(ctx, available_dims)
            data_slice = data.sel(**selector) if selector else data
    """

    def __init__(self, flow_system: FlowSystem, cluster_structure: ClusterStructure):
        """Initialize the iterator.

        Args:
            flow_system: The FlowSystem containing period/scenario dimensions.
            cluster_structure: The ClusterStructure with cluster ordering info.
        """
        self.periods = list(flow_system.periods) if flow_system.periods is not None else [None]
        self.scenarios = list(flow_system.scenarios) if flow_system.scenarios is not None else [None]
        self.cluster_structure = cluster_structure

    @property
    def has_periods(self) -> bool:
        """Check if there are period dimensions."""
        return self.periods != [None]

    @property
    def has_scenarios(self) -> bool:
        """Check if there are scenario dimensions."""
        return self.scenarios != [None]

    @property
    def is_multi_dim(self) -> bool:
        """Check if there are any extra dimensions beyond time."""
        return self.has_periods or self.has_scenarios

    def __iter__(self) -> Iterator[SliceContext]:
        """Iterate over all (period, scenario) combinations."""
        for p in self.periods:
            for s in self.scenarios:
                cluster_order = self.cluster_structure.get_cluster_order_for_slice(period=p, scenario=s)
                yield SliceContext(period=p, scenario=s, cluster_order=cluster_order)

    def build_selector(self, ctx: SliceContext, available_dims: set[str]) -> dict:
        """Build xarray selector dict for the given context.

        Args:
            ctx: The current slice context.
            available_dims: Set of dimension names available in the target data.

        Returns:
            Dict suitable for xr.DataArray.sel(**selector).
        """
        selector = {}
        if self.has_periods and ctx.period is not None and 'period' in available_dims:
            selector['period'] = ctx.period
        if self.has_scenarios and ctx.scenario is not None and 'scenario' in available_dims:
            selector['scenario'] = ctx.scenario
        return selector


@dataclass
class CapacityBounds:
    """Extracted capacity bounds for storage SOC_boundary variables."""

    lower: xr.DataArray
    upper: xr.DataArray
    has_investment: bool


def extract_capacity_bounds(
    capacity_param: InvestParameters | int | float,
    boundary_coords: dict,
    boundary_dims: list[str],
) -> CapacityBounds:
    """Extract capacity bounds from storage parameters.

    Handles:
    - Fixed numeric values
    - InvestParameters with fixed_size or maximum_size
    - xr.DataArray with dimensions

    Args:
        capacity_param: The capacity parameter (InvestParameters or scalar).
        boundary_coords: Coordinates for SOC_boundary variable.
        boundary_dims: Dimension names for SOC_boundary variable.

    Returns:
        CapacityBounds with lower/upper bounds and investment flag.
    """
    n_boundaries = len(boundary_coords['cluster_boundary'])
    lb_shape = [n_boundaries] + [len(boundary_coords[d]) for d in boundary_dims[1:]]

    lb = xr.DataArray(np.zeros(lb_shape), coords=boundary_coords, dims=boundary_dims)

    # Determine has_investment and cap_value
    has_investment = hasattr(capacity_param, 'maximum_size')

    if hasattr(capacity_param, 'fixed_size') and capacity_param.fixed_size is not None:
        cap_value = capacity_param.fixed_size
    elif hasattr(capacity_param, 'maximum_size') and capacity_param.maximum_size is not None:
        cap_value = capacity_param.maximum_size
    elif isinstance(capacity_param, (int, float)):
        cap_value = capacity_param
    else:
        cap_value = 1e9  # Large default for unbounded case

    # Build upper bound
    if isinstance(cap_value, xr.DataArray) and cap_value.dims:
        ub = cap_value.expand_dims({'cluster_boundary': n_boundaries}, axis=0)
        ub = ub.assign_coords(cluster_boundary=np.arange(n_boundaries))
        ub = ub.transpose('cluster_boundary', ...)
    else:
        if hasattr(cap_value, 'item'):
            cap_value = float(cap_value.item())
        else:
            cap_value = float(cap_value)
        ub = xr.DataArray(np.full(lb_shape, cap_value), coords=boundary_coords, dims=boundary_dims)

    return CapacityBounds(lower=lb, upper=ub, has_investment=has_investment)


def build_boundary_coords(
    n_original_periods: int,
    flow_system: FlowSystem,
) -> tuple[dict, list[str]]:
    """Build coordinates and dimensions for SOC_boundary variables.

    Args:
        n_original_periods: Number of original (non-aggregated) periods.
        flow_system: The FlowSystem containing period/scenario dimensions.

    Returns:
        Tuple of (coords dict, dims list) ready for variable creation.
    """
    n_boundaries = n_original_periods + 1
    coords = {'cluster_boundary': np.arange(n_boundaries)}
    dims = ['cluster_boundary']

    if flow_system.periods is not None:
        dims.append('period')
        coords['period'] = np.array(list(flow_system.periods))

    if flow_system.scenarios is not None:
        dims.append('scenario')
        coords['scenario'] = np.array(list(flow_system.scenarios))

    return coords, dims
