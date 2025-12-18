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
        self._n_clusters = (
            int(cluster_structure.n_clusters)
            if isinstance(cluster_structure.n_clusters, (int, np.integer))
            else int(cluster_structure.n_clusters.values)
        )
        self._timesteps_per_cluster = cluster_structure.timesteps_per_cluster
        self._n_original_periods = cluster_structure.n_original_periods
        self._has_multi_dims = cluster_structure.has_multi_dims

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
        boundary_coords = {'cluster_boundary': np.arange(n_boundaries)}
        boundary_dims = ['cluster_boundary']

        # Determine extra dimensions from FlowSystem (period, scenario)
        # These are needed even if cap_value is scalar, because different periods/scenarios
        # may have different cluster assignments
        extra_dims = []
        if self.flow_system.periods is not None:
            extra_dims.append('period')
            boundary_coords['period'] = np.array(list(self.flow_system.periods))
        if self.flow_system.scenarios is not None:
            extra_dims.append('scenario')
            boundary_coords['scenario'] = np.array(list(self.flow_system.scenarios))

        if extra_dims:
            boundary_dims = ['cluster_boundary'] + extra_dims

        # Build bounds shape
        lb_shape = [n_boundaries] + [len(boundary_coords[d]) for d in extra_dims]
        lb = xr.DataArray(np.zeros(lb_shape), coords=boundary_coords, dims=boundary_dims)

        # Get upper bound from capacity
        if isinstance(cap_value, xr.DataArray) and cap_value.dims:
            # cap_value has dimensions - expand to include cluster_boundary
            ub = cap_value.expand_dims({'cluster_boundary': n_boundaries}, axis=0)
            ub = ub.assign_coords(cluster_boundary=np.arange(n_boundaries))
            # Ensure dims are in the right order
            ub = ub.transpose('cluster_boundary', ...)
        else:
            # Scalar cap_value - broadcast to all dims
            if hasattr(cap_value, 'item'):
                cap_value = float(cap_value.item())
            else:
                cap_value = float(cap_value)
            ub = xr.DataArray(np.full(lb_shape, cap_value), coords=boundary_coords, dims=boundary_dims)

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
        if self._has_multi_dims:
            # Multi-dimensional cluster_order: create constraints per (period, scenario) slice
            self._add_linking_constraints_multi_dim(storage, soc_boundary, delta_soc_dict, label)
        else:
            # Simple case: single cluster_order for all slices
            cluster_order = self.cluster_structure.get_cluster_order_for_slice()
            for d in range(self._n_original_periods):
                c = int(cluster_order[d])
                lhs = (
                    soc_boundary.isel(cluster_boundary=d + 1)
                    - soc_boundary.isel(cluster_boundary=d)
                    - delta_soc_dict[c]
                )
                self.add_constraints(lhs == 0, short_name=f'link|{label}|{d}')

        # Cyclic constraint: SOC_boundary[0] = SOC_boundary[end]
        if self.storage_cyclic:
            lhs = soc_boundary.isel(cluster_boundary=0) - soc_boundary.isel(cluster_boundary=self._n_original_periods)
            self.add_constraints(lhs == 0, short_name=f'cyclic|{label}')

        logger.debug(f'Added inter-cluster linking for storage {label}')

    def _add_linking_constraints_multi_dim(
        self,
        storage,
        soc_boundary,
        delta_soc_dict: dict,
        label: str,
    ) -> None:
        """Add linking constraints when cluster_order has period/scenario dimensions.

        When different (period, scenario) slices have different cluster assignments,
        we need to create constraints that select the correct delta_SOC for each slice.

        Args:
            storage: Storage component being linked.
            soc_boundary: SOC boundary variable with dims [cluster_boundary, period?, scenario?].
            delta_soc_dict: Dict mapping cluster ID to delta_SOC expression.
            label: Storage label for constraint naming.
        """
        # Determine which dimensions we're iterating over
        periods = list(self.flow_system.periods) if self.flow_system.periods is not None else [None]
        scenarios = list(self.flow_system.scenarios) if self.flow_system.scenarios is not None else [None]
        has_periods = periods != [None]
        has_scenarios = scenarios != [None]

        # Check which dimensions soc_boundary actually has
        soc_dims = set(soc_boundary.dims)

        # For each (period, scenario) combination, create constraints using the slice's cluster_order
        for p in periods:
            for s in scenarios:
                cluster_order = self.cluster_structure.get_cluster_order_for_slice(period=p, scenario=s)

                # Build selector for this slice - only include dims that exist in soc_boundary
                soc_selector = {}
                if has_periods and p is not None and 'period' in soc_dims:
                    soc_selector['period'] = p
                if has_scenarios and s is not None and 'scenario' in soc_dims:
                    soc_selector['scenario'] = s

                # Select the slice of soc_boundary for this (period, scenario)
                soc_boundary_slice = soc_boundary.sel(**soc_selector) if soc_selector else soc_boundary

                for d in range(self._n_original_periods):
                    c = int(cluster_order[d])
                    delta_soc = delta_soc_dict[c]

                    # Build selector for delta_soc - check which dims it has
                    delta_selector = {}
                    if has_periods and p is not None and 'period' in delta_soc.dims:
                        delta_selector['period'] = p
                    if has_scenarios and s is not None and 'scenario' in delta_soc.dims:
                        delta_selector['scenario'] = s
                    if delta_selector:
                        delta_soc = delta_soc.sel(**delta_selector)

                    lhs = (
                        soc_boundary_slice.isel(cluster_boundary=d + 1)
                        - soc_boundary_slice.isel(cluster_boundary=d)
                        - delta_soc
                    )

                    # Build constraint name with period/scenario info
                    slice_suffix = ''
                    if has_periods and p is not None:
                        slice_suffix += f'|p={p}'
                    if has_scenarios and s is not None:
                        slice_suffix += f'|s={s}'

                    self.add_constraints(lhs == 0, short_name=f'link|{label}|{d}{slice_suffix}')
