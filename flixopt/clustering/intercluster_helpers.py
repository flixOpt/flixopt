"""Helper utilities for inter-cluster storage linking.

This module provides utilities for building inter-cluster storage linking
constraints following the S-N model from Blanke et al. (2022).

Background
----------
When time series are clustered (aggregated into representative periods), storage
behavior needs special handling. The S-N linking model introduces:

- **SOC_boundary**: Absolute state-of-charge at the boundary between original periods.
  With N original periods, there are N+1 boundary points.

- **Linking**: SOC_boundary[d+1] = SOC_boundary[d] + delta_SOC[cluster_assignments[d]]
  Each boundary is connected to the next via the net charge change of the
  representative cluster for that period.

These utilities help construct the coordinates and bounds for SOC_boundary variables.

References
----------
- Blanke, T., et al. (2022). "Inter-Cluster Storage Linking for Time Series
  Aggregation in Energy System Optimization Models."
- Kotzur, L., et al. (2018). "Time series aggregation for energy system design:
  Modeling seasonal storage."

See Also
--------
:class:`flixopt.components.InterclusterStorageModel`
    The storage model that uses these utilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ..interface import InvestParameters

if TYPE_CHECKING:
    from ..flow_system import FlowSystem

logger = logging.getLogger('flixopt')

# Default upper bound for unbounded storage capacity.
# Used when no explicit capacity or InvestParameters.maximum_size is provided.
# Set to 1e6 to avoid numerical issues with very large bounds while still
# being effectively unbounded for most practical applications.
DEFAULT_UNBOUNDED_CAPACITY = 1e6


@dataclass
class CapacityBounds:
    """Bounds for SOC_boundary variable creation.

    This dataclass holds the lower and upper bounds for the SOC_boundary variable,
    along with a flag indicating whether investment sizing is used.

    Attributes:
        lower: Lower bound DataArray (typically zeros).
        upper: Upper bound DataArray (capacity or maximum investment size).
        has_investment: True if the storage uses InvestParameters for sizing.
    """

    lower: xr.DataArray
    upper: xr.DataArray
    has_investment: bool


def extract_capacity_bounds(
    capacity_param: InvestParameters | int | float | None,
    boundary_coords: dict,
    boundary_dims: list[str],
) -> CapacityBounds:
    """Extract capacity bounds from storage parameters for SOC_boundary variable.

    This function determines the appropriate bounds for the SOC_boundary variable
    based on the storage's capacity parameter:

    - **Fixed capacity** (numeric): Upper bound is the fixed value.
    - **InvestParameters**: Upper bound is maximum_size (or fixed_size if set).
      The actual bound is enforced via separate constraints linked to investment.size.
    - **None/Unbounded**: Upper bound is set to a large value (1e6).

    The lower bound is always zero (SOC cannot be negative).

    Args:
        capacity_param: Storage capacity specification. Can be:
            - Numeric (int/float): Fixed capacity
            - InvestParameters: Investment-based sizing with min/max
            - None: Unbounded storage
        boundary_coords: Coordinate dictionary for SOC_boundary variable.
            Must contain 'cluster_boundary' key.
        boundary_dims: Dimension names for SOC_boundary variable.
            First dimension must be 'cluster_boundary'.

    Returns:
        CapacityBounds with lower/upper bounds and investment flag.

    Example:
        >>> coords, dims = build_boundary_coords(14, flow_system)
        >>> bounds = extract_capacity_bounds(InvestParameters(maximum_size=10000), coords, dims)
        >>> bounds.has_investment
        True
        >>> bounds.upper.max()
        10000.0
    """
    n_boundaries = len(boundary_coords['cluster_boundary'])
    lb_shape = [n_boundaries] + [len(boundary_coords[d]) for d in boundary_dims[1:]]

    lb = xr.DataArray(np.zeros(lb_shape), coords=boundary_coords, dims=boundary_dims)

    # Determine has_investment and cap_value
    has_investment = isinstance(capacity_param, InvestParameters)
    using_default_bound = False

    if isinstance(capacity_param, InvestParameters):
        if capacity_param.fixed_size is not None:
            cap_value = capacity_param.fixed_size
        elif capacity_param.maximum_size is not None:
            cap_value = capacity_param.maximum_size
        else:
            cap_value = DEFAULT_UNBOUNDED_CAPACITY
            using_default_bound = True
    elif isinstance(capacity_param, (int, float)):
        cap_value = capacity_param
    else:
        cap_value = DEFAULT_UNBOUNDED_CAPACITY
        using_default_bound = True

    if using_default_bound:
        logger.warning(
            f'No explicit capacity bound provided for inter-cluster storage linking. '
            f'Using default upper bound of {DEFAULT_UNBOUNDED_CAPACITY:.0e}. '
            f'Consider setting capacity_in_flow_hours or InvestParameters.maximum_size explicitly.'
        )

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
    n_original_clusters: int,
    flow_system: FlowSystem,
) -> tuple[dict, list[str]]:
    """Build coordinates and dimensions for SOC_boundary variable.

    Creates the coordinate dictionary and dimension list needed to create the
    SOC_boundary variable. The primary dimension is 'cluster_boundary' with
    N+1 values (one for each boundary between N original periods).

    Additional dimensions (period, scenario) are included if present in the
    FlowSystem, ensuring the SOC_boundary variable has the correct shape for
    multi-period or stochastic optimizations.

    Args:
        n_original_clusters: Number of original (non-aggregated) time periods.
            For example, if a year is clustered into 8 typical days but originally
            had 365 days, this would be 365.
        flow_system: The FlowSystem containing optional period/scenario dimensions.

    Returns:
        Tuple of (coords, dims) where:
            - coords: Dictionary mapping dimension names to coordinate arrays
            - dims: List of dimension names in order

    Example:
        >>> coords, dims = build_boundary_coords(14, flow_system)
        >>> dims
        ['cluster_boundary']  # or ['cluster_boundary', 'period'] if periods exist
        >>> coords['cluster_boundary']
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
    """
    n_boundaries = n_original_clusters + 1
    coords = {'cluster_boundary': np.arange(n_boundaries)}
    dims = ['cluster_boundary']

    if flow_system.periods is not None:
        dims.append('period')
        coords['period'] = np.array(list(flow_system.periods))

    if flow_system.scenarios is not None:
        dims.append('scenario')
        coords['scenario'] = np.array(list(flow_system.scenarios))

    return coords, dims
