"""
Inter-cluster storage linking for aggregated optimization.

When using time series aggregation (clustering), timesteps are reduced to only
representative (typical) periods. This module provides the `InterClusterLinking`
model that tracks storage state across the full original time horizon.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..structure import Submodel

if TYPE_CHECKING:
    from ..flow_system import FlowSystem
    from ..structure import FlowSystemModel
    from .base import ClusterStructure

logger = logging.getLogger('flixopt')


class InterClusterLinking(Submodel):
    """Model that links storage state across representative periods.

    When using aggregation (clustering), timesteps are reduced to only representative
    periods. This model creates variables and constraints to track storage state
    across the full original time horizon using boundary state variables.

    The approach:
    1. Create SOC_boundary[d] for each original period d (0 to n_original_periods)
    2. Compute delta_SOC[c] for each representative period c (change in SOC during period)
    3. Link: SOC_boundary[d+1] = SOC_boundary[d] + delta_SOC[cluster_order[d]]
    4. Optionally enforce cyclic: SOC_boundary[0] = SOC_boundary[n_original_periods]

    This allows the optimizer to properly value storage for long-term (seasonal)
    patterns while only solving for the representative timesteps.

    Example:
        >>> from flixopt.aggregation import ClusterStructure, InterClusterLinking
        >>> structure = ClusterStructure(...)
        >>> model = InterClusterLinking(
        ...     model=flow_system.model,
        ...     flow_system=flow_system,
        ...     cluster_structure=structure,
        ... )
        >>> model.do_modeling()
    """

    def __init__(
        self,
        model: FlowSystemModel,
        flow_system: FlowSystem,
        cluster_structure: ClusterStructure,
        storage_cyclic: bool = True,
    ):
        """
        Args:
            model: The FlowSystemModel to add constraints to.
            flow_system: The FlowSystem being optimized.
            cluster_structure: Clustering structure with cluster_order and occurrences.
            storage_cyclic: If True, enforce SOC_boundary[0] = SOC_boundary[end].
        """
        super().__init__(model, label_of_element='InterClusterLinking', label_of_model='InterClusterLinking')
        self.flow_system = flow_system
        self.cluster_structure = cluster_structure
        self.storage_cyclic = storage_cyclic

        # Extract commonly used values from cluster_structure
        self._cluster_order = cluster_structure.cluster_order.values
        self._n_clusters = (
            int(cluster_structure.n_clusters)
            if isinstance(cluster_structure.n_clusters, (int, np.integer))
            else int(cluster_structure.n_clusters.values)
        )
        self._timesteps_per_cluster = cluster_structure.timesteps_per_cluster
        self._n_original_periods = len(self._cluster_order)

    def do_modeling(self):
        """Create SOC boundary variables and inter-period linking constraints.

        For each storage:
        - SOC_boundary[d]: State of charge at start of original period d
        - delta_SOC[c]: Change in SOC during representative period c
        - Linking: SOC_boundary[d+1] = SOC_boundary[d] + delta_SOC[cluster_order[d]]
        """
        storages = list(self.flow_system.storages.values())
        if not storages:
            logger.info('No storages found - skipping inter-cluster linking')
            return

        logger.info(
            f'Adding inter-cluster storage linking for {len(storages)} storages '
            f'({self._n_original_periods} original periods, {self._n_clusters} clusters)'
        )

        for storage in storages:
            self._add_storage_linking(storage)

    def _add_storage_linking(self, storage) -> None:
        """Add inter-cluster linking constraints for a single storage.

        Args:
            storage: Storage component to add linking for.
        """
        import xarray as xr

        label = storage.label

        # Get the charge state variable from the storage's submodel
        charge_state_name = f'{label}|charge_state'
        if charge_state_name not in storage.submodel.variables:
            logger.warning(f'Storage {label} has no charge_state variable - skipping')
            return

        charge_state = storage.submodel.variables[charge_state_name]

        # Get storage capacity bounds (may have period/scenario dimensions)
        capacity = storage.capacity_in_flow_hours
        if hasattr(capacity, 'fixed_size') and capacity.fixed_size is not None:
            cap_value = capacity.fixed_size
        elif hasattr(capacity, 'maximum') and capacity.maximum is not None:
            cap_value = capacity.maximum
        else:
            cap_value = 1e9  # Large default

        # Create SOC_boundary variables for each original period boundary
        # We need n_original_periods + 1 boundaries (start of first through end of last)
        n_boundaries = self._n_original_periods + 1
        boundary_coords = [np.arange(n_boundaries)]
        boundary_dims = ['period_boundary']

        # Build bounds - handle both scalar and multi-dimensional cap_value
        if isinstance(cap_value, xr.DataArray) and cap_value.dims:
            # cap_value has dimensions (e.g., period, scenario) - need to broadcast
            extra_dims = list(cap_value.dims)
            extra_coords = {dim: cap_value.coords[dim].values for dim in extra_dims}

            boundary_dims = ['period_boundary'] + extra_dims
            boundary_coords = [np.arange(n_boundaries)] + [extra_coords[d] for d in extra_dims]

            lb_coords = {'period_boundary': np.arange(n_boundaries), **extra_coords}
            lb_shape = [n_boundaries] + [len(extra_coords[d]) for d in extra_dims]
            lb = xr.DataArray(np.zeros(lb_shape), coords=lb_coords, dims=boundary_dims)

            ub = cap_value.expand_dims({'period_boundary': n_boundaries}, axis=0)
            ub = ub.assign_coords(period_boundary=np.arange(n_boundaries))
        else:
            # Scalar cap_value
            if hasattr(cap_value, 'item'):
                cap_value = float(cap_value.item())
            else:
                cap_value = float(cap_value)
            lb = xr.DataArray(0.0, coords={'period_boundary': np.arange(n_boundaries)}, dims=['period_boundary'])
            ub = xr.DataArray(cap_value, coords={'period_boundary': np.arange(n_boundaries)}, dims=['period_boundary'])

        soc_boundary = self.add_variables(
            lower=lb,
            upper=ub,
            coords=boundary_coords,
            dims=boundary_dims,
            short_name=f'SOC_boundary|{label}',
        )

        # Pre-compute delta_SOC for each representative period
        # delta_SOC[c] = charge_state[c, end] - charge_state[c, start]
        delta_soc_dict = {}
        for c in range(self._n_clusters):
            start_idx = c * self._timesteps_per_cluster
            end_idx = (c + 1) * self._timesteps_per_cluster  # charge_state has extra timestep

            delta_soc_dict[c] = charge_state.isel(time=end_idx) - charge_state.isel(time=start_idx)

        # Create linking constraints:
        # SOC_boundary[d+1] = SOC_boundary[d] + delta_SOC[cluster_order[d]]
        for d in range(self._n_original_periods):
            c = int(self._cluster_order[d])
            lhs = soc_boundary.isel(period_boundary=d + 1) - soc_boundary.isel(period_boundary=d) - delta_soc_dict[c]
            self.add_constraints(lhs == 0, short_name=f'link|{label}|{d}')

        # Cyclic constraint: SOC_boundary[0] = SOC_boundary[end]
        if self.storage_cyclic:
            lhs = soc_boundary.isel(period_boundary=0) - soc_boundary.isel(period_boundary=self._n_original_periods)
            self.add_constraints(lhs == 0, short_name=f'cyclic|{label}')

        logger.debug(f'Added inter-cluster linking for storage {label}')
