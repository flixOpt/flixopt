"""
Transform accessor for FlowSystem.

This module provides the TransformAccessor class that enables
transformations on FlowSystem like clustering and MGA.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .clustering import ClusteringParameters
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


class TransformAccessor:
    """
    Accessor for transformation methods on FlowSystem.

    This class provides transformations that create new FlowSystem instances
    with modified structure or data, accessible via `flow_system.transform`.

    Examples:
        Clustered optimization:

        >>> clustered_fs = flow_system.transform.cluster(params)
        >>> clustered_fs.optimize(solver)
        >>> print(clustered_fs.solution)

        Future MGA:

        >>> mga_fs = flow_system.transform.mga(alternatives=5)
        >>> mga_fs.optimize(solver)
    """

    def __init__(self, flow_system: FlowSystem) -> None:
        """
        Initialize the accessor with a reference to the FlowSystem.

        Args:
            flow_system: The FlowSystem to transform.
        """
        self._fs = flow_system

    def cluster(
        self,
        parameters: ClusteringParameters,
        components_to_clusterize: list | None = None,
    ) -> FlowSystem:
        """
        Create a clustered FlowSystem for time series aggregation.

        This method creates a new FlowSystem that can be optimized with
        clustered time series data. The clustering reduces computational
        complexity by identifying representative time periods.

        The returned FlowSystem:
        - Has the same timesteps as the original (clustering works via constraints, not reduction)
        - Has aggregated time series data (if `aggregate_data_and_fix_non_binary_vars=True`)
        - Will have clustering constraints added during `build_model()`

        Args:
            parameters: Clustering parameters specifying period duration,
                number of periods, and aggregation settings.
            components_to_clusterize: List of components to apply clustering to.
                If None, all components are clustered.

        Returns:
            A new FlowSystem configured for clustered optimization.

        Raises:
            ValueError: If timestep sizes are inconsistent.
            ValueError: If hours_per_period is not a multiple of timestep size.

        Examples:
            Basic clustered optimization:

            >>> from flixopt import ClusteringParameters
            >>> params = ClusteringParameters(
            ...     hours_per_period=24,
            ...     nr_of_periods=8,
            ...     fix_storage_flows=True,
            ...     aggregate_data_and_fix_non_binary_vars=True,
            ... )
            >>> clustered_fs = flow_system.transform.cluster(params)
            >>> clustered_fs.optimize(solver)
            >>> print(clustered_fs.solution)

            With model modifications:

            >>> clustered_fs = flow_system.transform.cluster(params)
            >>> clustered_fs.build_model()
            >>> clustered_fs.model.add_constraints(...)
            >>> clustered_fs.solve(solver)
        """
        import numpy as np

        from .clustering import Clustering
        from .core import DataConverter, TimeSeriesData, drop_constant_arrays

        # Validation
        dt_min = float(self._fs.hours_per_timestep.min().item())
        dt_max = float(self._fs.hours_per_timestep.max().item())
        if not dt_min == dt_max:
            raise ValueError(
                f'Clustering failed due to inconsistent time step sizes: '
                f'delta_t varies from {dt_min} to {dt_max} hours.'
            )
        ratio = parameters.hours_per_period / dt_max
        if not np.isclose(ratio, round(ratio), atol=1e-9):
            raise ValueError(
                f'The selected hours_per_period={parameters.hours_per_period} does not match the time '
                f'step size of {dt_max} hours. It must be an integer multiple of {dt_max} hours.'
            )

        logger.info(f'{"":#^80}')
        logger.info(f'{" Clustering TimeSeries Data ":#^80}')

        # Get dataset representation
        ds = self._fs.to_dataset()
        temporaly_changing_ds = drop_constant_arrays(ds, dim='time')

        # Perform clustering
        clustering = Clustering(
            original_data=temporaly_changing_ds.to_dataframe(),
            hours_per_time_step=float(dt_min),
            hours_per_period=parameters.hours_per_period,
            nr_of_periods=parameters.nr_of_periods,
            weights=self._calculate_clustering_weights(temporaly_changing_ds),
            time_series_for_high_peaks=parameters.labels_for_high_peaks,
            time_series_for_low_peaks=parameters.labels_for_low_peaks,
        )
        clustering.cluster()

        # Create new FlowSystem (with aggregated data if requested)
        if parameters.aggregate_data_and_fix_non_binary_vars:
            ds = self._fs.to_dataset()
            for name, series in clustering.aggregated_data.items():
                da = DataConverter.to_dataarray(series, self._fs.coords).rename(name).assign_attrs(ds[name].attrs)
                if TimeSeriesData.is_timeseries_data(da):
                    da = TimeSeriesData.from_dataarray(da)
                ds[name] = da

            from .flow_system import FlowSystem

            clustered_fs = FlowSystem.from_dataset(ds)
        else:
            # Copy without data modification
            clustered_fs = self._fs.copy()

        # Store clustering info for later use
        clustered_fs._clustering_info = {
            'parameters': parameters,
            'clustering': clustering,
            'components_to_clusterize': components_to_clusterize,
            'original_fs': self._fs,
        }

        return clustered_fs

    @staticmethod
    def _calculate_clustering_weights(ds) -> dict[str, float]:
        """Calculate weights for clustering based on dataset attributes."""
        from collections import Counter

        import numpy as np

        groups = [da.attrs.get('clustering_group') for da in ds.data_vars.values() if 'clustering_group' in da.attrs]
        group_counts = Counter(groups)

        # Calculate weight for each group (1/count)
        group_weights = {group: 1 / count for group, count in group_counts.items()}

        weights = {}
        for name, da in ds.data_vars.items():
            clustering_group = da.attrs.get('clustering_group')
            group_weight = group_weights.get(clustering_group)
            if group_weight is not None:
                weights[name] = group_weight
            else:
                weights[name] = da.attrs.get('clustering_weight', 1)

        if np.all(np.isclose(list(weights.values()), 1, atol=1e-6)):
            logger.info('All Clustering weights were set to 1')

        return weights

    # Future methods can be added here:
    #
    # def mga(self, alternatives: int = 5) -> FlowSystem:
    #     """Create a FlowSystem configured for Modeling to Generate Alternatives."""
    #     ...
