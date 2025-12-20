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
    from ..flow_system import FlowSystem
    from ..interface import InvestParameters


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
