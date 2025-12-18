"""
This module contains the Clustering functionality for the flixopt framework.
Through this, clustering TimeSeriesData is possible.
"""

from __future__ import annotations

import copy
import logging
import timeit
from typing import TYPE_CHECKING

import numpy as np

try:
    import tsam.timeseriesaggregation as tsam

    TSAM_AVAILABLE = True
except ImportError:
    TSAM_AVAILABLE = False

from .color_processing import process_colors
from .components import Storage
from .config import CONFIG
from .plot_result import PlotResult
from .structure import (
    FlowSystemModel,
    Interface,
    Submodel,
    register_class_for_io,
)  # Interface and register_class_for_io used by ClusteringParameters

if TYPE_CHECKING:
    import linopy
    import pandas as pd
    import xarray as xr

    from .core import Scalar, TimeSeriesData
    from .elements import Component
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


class Clustering:
    """
    Clustering organizing class for time series aggregation using tsam.
    """

    def __init__(
        self,
        original_data: pd.DataFrame,
        hours_per_time_step: Scalar,
        hours_per_period: Scalar,
        nr_of_periods: int | None = 8,
        n_segments: int | None = None,
        weights: dict[str, float] | None = None,
        time_series_for_high_peaks: list[str] | None = None,
        time_series_for_low_peaks: list[str] | None = None,
    ):
        """
        Args:
            original_data: The original data to aggregate.
            hours_per_time_step: The duration of each timestep in hours.
            hours_per_period: The duration of each period in hours.
            nr_of_periods: The number of typical periods to use in the aggregation.
                Set to None to skip period clustering and only do segmentation.
            n_segments: Number of segments within each period (inner-period clustering).
                If None, no inner-period segmentation is performed.
            weights: The weights for aggregation. If None, all time series are equally weighted.
            time_series_for_high_peaks: List of time series to use for explicitly selecting periods with high values.
            time_series_for_low_peaks: List of time series to use for explicitly selecting periods with low values.
        """
        if not TSAM_AVAILABLE:
            raise ImportError(
                "The 'tsam' package is required for clustering functionality. Install it with 'pip install tsam'."
            )
        self.original_data = copy.deepcopy(original_data)
        self.hours_per_time_step = hours_per_time_step
        self.hours_per_period = hours_per_period
        self.nr_of_periods = nr_of_periods
        self.n_segments = n_segments
        self.nr_of_time_steps = len(self.original_data.index)
        self.weights = weights or {}
        self.time_series_for_high_peaks = time_series_for_high_peaks or []
        self.time_series_for_low_peaks = time_series_for_low_peaks or []

        self.aggregated_data: pd.DataFrame | None = None
        self.clustering_duration_seconds = None
        self.tsam: tsam.TimeSeriesAggregation | None = None

    def cluster(self) -> None:
        """
        Perform time series clustering/aggregation.
        """
        start_time = timeit.default_timer()

        # Determine number of periods for clustering
        # If nr_of_periods is None, use segmentation only (no inter-period clustering)
        total_periods = int(self.nr_of_time_steps * self.hours_per_time_step / self.hours_per_period)
        n_typical_periods = self.nr_of_periods if self.nr_of_periods is not None else total_periods

        # Create aggregation object
        self.tsam = tsam.TimeSeriesAggregation(
            self.original_data,
            noTypicalPeriods=n_typical_periods,
            hoursPerPeriod=self.hours_per_period,
            resolution=self.hours_per_time_step,
            clusterMethod='k_means',
            extremePeriodMethod='new_cluster_center' if self.use_extreme_periods else 'None',
            weightDict={name: weight for name, weight in self.weights.items() if name in self.original_data.columns},
            addPeakMax=self.time_series_for_high_peaks,
            addPeakMin=self.time_series_for_low_peaks,
            # Inner-period segmentation parameters
            segmentation=self.n_segments is not None,
            noSegments=self.n_segments if self.n_segments is not None else 1,
        )

        self.tsam.createTypicalPeriods()
        self.aggregated_data = self.tsam.predictOriginalData()

        self.clustering_duration_seconds = timeit.default_timer() - start_time
        if logger.isEnabledFor(logging.INFO):
            logger.info(self.describe_clusters())

    def describe_clusters(self) -> str:
        description = {}
        for cluster in self.get_cluster_indices().keys():
            description[cluster] = [
                str(indexVector[0]) + '...' + str(indexVector[-1])
                for indexVector in self.get_cluster_indices()[cluster]
            ]

        if self.use_extreme_periods:
            # Zeitreihe rauslöschen:
            extreme_periods = self.tsam.extremePeriods.copy()
            for key in extreme_periods:
                del extreme_periods[key]['profile']
        else:
            extreme_periods = {}

        return (
            f'{"":#^80}\n'
            f'{" Clustering ":#^80}\n'
            f'periods_order:\n'
            f'{self.tsam.clusterOrder}\n'
            f'clusterPeriodNoOccur:\n'
            f'{self.tsam.clusterPeriodNoOccur}\n'
            f'index_vectors_of_clusters:\n'
            f'{description}\n'
            f'{"":#^80}\n'
            f'extreme_periods:\n'
            f'{extreme_periods}\n'
            f'{"":#^80}'
        )

    @property
    def use_extreme_periods(self):
        return self.time_series_for_high_peaks or self.time_series_for_low_peaks

    def plot(self, colormap: str | None = None, show: bool | None = None) -> PlotResult:
        """Plot original vs aggregated data comparison.

        Visualizes the original time series (dashed lines) overlaid with
        the aggregated/clustered time series (solid lines) for comparison.

        Args:
            colormap: Colorscale name for the time series colors.
                Defaults to CONFIG.Plotting.default_qualitative_colorscale.
            show: Whether to display the figure.
                Defaults to CONFIG.Plotting.default_show.

        Returns:
            PlotResult containing the comparison figure and underlying data.

        Examples:
            >>> clustering.cluster()
            >>> clustering.plot()
            >>> clustering.plot(colormap='Set2', show=False).to_html('clustering.html')
        """
        import plotly.express as px
        import xarray as xr

        df_org = self.original_data.copy().rename(
            columns={col: f'Original - {col}' for col in self.original_data.columns}
        )
        df_agg = self.aggregated_data.copy().rename(
            columns={col: f'Aggregated - {col}' for col in self.aggregated_data.columns}
        )
        colors = list(
            process_colors(colormap or CONFIG.Plotting.default_qualitative_colorscale, list(df_org.columns)).values()
        )

        # Create line plot for original data (dashed)
        index_name = df_org.index.name or 'index'
        df_org_long = df_org.reset_index().melt(id_vars=index_name, var_name='variable', value_name='value')
        fig = px.line(df_org_long, x=index_name, y='value', color='variable', color_discrete_sequence=colors)
        for trace in fig.data:
            trace.update(line=dict(dash='dash'))

        # Add aggregated data (solid lines)
        df_agg_long = df_agg.reset_index().melt(id_vars=index_name, var_name='variable', value_name='value')
        fig2 = px.line(df_agg_long, x=index_name, y='value', color='variable', color_discrete_sequence=colors)
        for trace in fig2.data:
            fig.add_trace(trace)

        fig.update_layout(
            title='Original vs Aggregated Data (original = ---)',
            xaxis_title='Time in h',
            yaxis_title='Value',
        )

        # Build xarray Dataset with both original and aggregated data
        data = xr.Dataset(
            {
                'original': self.original_data.to_xarray().to_array(dim='variable'),
                'aggregated': self.aggregated_data.to_xarray().to_array(dim='variable'),
            }
        )
        result = PlotResult(data=data, figure=fig)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            result.show()

        return result

    def get_cluster_indices(self) -> dict[str, list[np.ndarray]]:
        """
        Generates a dictionary that maps each cluster to a list of index vectors representing the time steps
        assigned to that cluster for each period.

        Returns:
            dict: {cluster_0: [index_vector_3, index_vector_7, ...],
                   cluster_1: [index_vector_1],
                   ...}
        """
        clusters = self.tsam.clusterPeriodNoOccur.keys()
        index_vectors = {cluster: [] for cluster in clusters}

        # Use actual timesteps per period, not segment count
        period_length = int(self.hours_per_period / self.hours_per_time_step)
        total_steps = len(self.tsam.timeSeries)

        for period, cluster_id in enumerate(self.tsam.clusterOrder):
            start_idx = period * period_length
            end_idx = np.min([start_idx + period_length, total_steps])
            index_vectors[cluster_id].append(np.arange(start_idx, end_idx))

        return index_vectors

    def get_equation_indices(self, skip_first_index_of_period: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates pairs of indices for the equations by comparing index vectors of the same cluster.
        If `skip_first_index_of_period` is True, the first index of each period is skipped.

        Args:
            skip_first_index_of_period (bool): Whether to include or skip the first index of each period.

        Returns:
            tuple[np.ndarray, np.ndarray]: Two arrays of indices.
        """
        idx_var1 = []
        idx_var2 = []

        # Iterate through cluster index vectors
        for index_vectors in self.get_cluster_indices().values():
            if len(index_vectors) <= 1:  # Only proceed if cluster has more than one period
                continue

            # Process the first vector, optionally skip first index
            first_vector = index_vectors[0][1:] if skip_first_index_of_period else index_vectors[0]

            # Compare first vector to others in the cluster
            for other_vector in index_vectors[1:]:
                if skip_first_index_of_period:
                    other_vector = other_vector[1:]

                # Compare elements up to the minimum length of both vectors
                min_len = min(len(first_vector), len(other_vector))
                idx_var1.extend(first_vector[:min_len])
                idx_var2.extend(other_vector[:min_len])

        # Convert lists to numpy arrays
        return np.array(idx_var1), np.array(idx_var2)

    def get_equation_groups(self, skip_first_index_of_period: bool = True) -> list[list[int]]:
        """Get groups of timestep indices that should be equal (inter-cluster).

        Each group contains timesteps at the same position within periods of the same cluster.
        E.g., if cluster 0 has periods [0-95] and [192-287], position 5 gives group [5, 197].

        Args:
            skip_first_index_of_period: Skip first timestep of each period (for storage continuity).

        Returns:
            List of groups, where each group is a list of timestep indices to equate.
        """
        groups = []

        for index_vectors in self.get_cluster_indices().values():
            if len(index_vectors) <= 1:
                continue

            # Determine the length and starting offset
            start_offset = 1 if skip_first_index_of_period else 0
            min_len = min(len(v) for v in index_vectors) - start_offset

            # Create a group for each position across all periods in this cluster
            for pos in range(min_len):
                group = [int(v[pos + start_offset]) for v in index_vectors]
                if len(group) > 1:
                    groups.append(group)

        return groups

    def get_segment_equation_groups(self) -> list[list[int]]:
        """Get groups of timestep indices that should be equal (intra-segment).

        Each group contains all timesteps within the same segment.

        Returns:
            List of groups, where each group is a list of timestep indices to equate.
        """
        if self.n_segments is None:
            return []

        groups = []
        period_length = int(self.hours_per_period / self.hours_per_time_step)
        segment_duration_dict = self.tsam.segmentDurationDict['Segment Duration']

        for period_idx, cluster_id in enumerate(self.tsam.clusterOrder):
            period_offset = period_idx * period_length
            start_step = 0

            for seg_idx in range(self.n_segments):
                duration = segment_duration_dict[(cluster_id, seg_idx)]
                if duration > 1:
                    # Group all timesteps in this segment
                    group = [period_offset + start_step + step for step in range(duration)]
                    groups.append(group)
                start_step += duration

        return groups

    def get_segment_equation_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates pairs of indices for intra-segment equalization.

        When segmentation is enabled, all timesteps within the same segment should have
        equal values. This method returns index pairs where each timestep in a segment
        is paired with the first timestep of that segment.

        Returns:
            tuple[np.ndarray, np.ndarray]: Two arrays of indices. For each pair (i, j),
                variable[i] should equal variable[j].

        Note:
            Only generates constraints when n_segments is set. Returns empty arrays otherwise.
        """
        if self.n_segments is None:
            return np.array([]), np.array([])

        idx_var1 = []
        idx_var2 = []

        period_length = int(self.hours_per_period / self.hours_per_time_step)
        segment_duration_dict = self.tsam.segmentDurationDict['Segment Duration']

        for period_idx, cluster_id in enumerate(self.tsam.clusterOrder):
            period_offset = period_idx * period_length
            start_step = 0

            for seg_idx in range(self.n_segments):
                # Get duration for this (cluster, segment)
                duration = segment_duration_dict[(cluster_id, seg_idx)]

                # Equate all timesteps in this segment to the first timestep
                first_ts = period_offset + start_step
                for step in range(1, duration):
                    idx_var1.append(first_ts)
                    idx_var2.append(period_offset + start_step + step)

                start_step += duration

        return np.array(idx_var1), np.array(idx_var2)


def _parse_cluster_duration(duration: str | float) -> float:
    """Convert cluster duration to hours.

    Args:
        duration: Either a pandas-style duration string ('1D', '24h', '6h')
                  or a numeric value in hours.

    Returns:
        Duration in hours.

    Examples:
        >>> _parse_cluster_duration('1D')
        24.0
        >>> _parse_cluster_duration('6h')
        6.0
        >>> _parse_cluster_duration(24)
        24.0
    """
    import pandas as pd

    if isinstance(duration, (int, float)):
        return float(duration)

    # Parse pandas-style duration strings
    td = pd.Timedelta(duration)
    return td.total_seconds() / 3600


@register_class_for_io
class ClusteringParameters(Interface):
    """Parameters for time series clustering.

    This class configures how time series data is clustered into representative
    segments using the tsam (time series aggregation module) package.

    Note:
        The term "cluster" here refers to clustering time segments (e.g., typical days),
        not to be confused with the FlowSystem's "period" dimension (e.g., years).

    Args:
        n_clusters: Number of clusters to create (e.g., 8 typical days).
            Set to None to skip clustering and only do segmentation.
        cluster_duration: Duration of each cluster segment. Can be a pandas-style
            string ('1D', '24h', '6h') or a numeric value in hours.
        n_segments: Number of segments to create within each cluster (inner-period
            clustering). For example, n_segments=4 with cluster_duration='1D' will
            reduce 24 hourly timesteps to 4 representative segments per day.
            Default is None (no inner-period segmentation).
        aggregate_data: If True, aggregate time series data and fix all time-dependent
            variables. If False, only fix binary variables. Default is True.
        include_storage: Whether to include storage flows in clustering constraints.
            If other flows are fixed, fixing storage flows is usually not required.
            Default is True.
        flexibility_percent: Maximum percentage (0-100) of binary values that can
            deviate from the clustered pattern. Default is 0 (no flexibility).
        flexibility_penalty: Penalty added to objective for each deviation.
            Only applies when flexibility_percent > 0. Default is 0.
        time_series_for_high_peaks: List of TimeSeriesData to force inclusion of
            segments with high values.
        time_series_for_low_peaks: List of TimeSeriesData to force inclusion of
            segments with low values.
        cluster_order: Pre-computed cluster assignments. DataArray of shape (cluster_period,)
            specifying which cluster each period belongs to. If provided, tsam clustering
            is skipped.
        period_length: Number of timesteps per clustering-period. Required if cluster_order
            is provided.
        segment_assignment: Pre-computed segment assignments. DataArray of shape (cluster, position)
            specifying segment ID for each position. Optional.
        skip_first_of_period: Whether to skip the first timestep of each period for storage
            constraints (to maintain inter-period continuity). Default is True.

    Examples:
        Basic usage (8 typical days):

        >>> clustered_fs = flow_system.transform.cluster(
        ...     n_clusters=8,
        ...     cluster_duration='1D',
        ... )

        With inner-period segmentation (8 typical days × 4 segments each = 32 timesteps):

        >>> clustered_fs = flow_system.transform.cluster(
        ...     n_clusters=8,
        ...     cluster_duration='1D',
        ...     n_segments=4,  # Reduce 24h to 4 segments per day
        ... )

        With pre-computed cluster assignments (external clustering):

        >>> params = fx.ClusteringParameters(
        ...     n_clusters=8,
        ...     cluster_duration='1D',
        ...     cluster_order=xr.DataArray([0, 1, 2, 0, 1, ...], dims=['cluster_period']),
        ...     period_length=24,
        ... )
        >>> clustered_fs = flow_system.transform.cluster(parameters=params)
    """

    def __init__(
        self,
        n_clusters: int | None,
        cluster_duration: str | float,
        n_segments: int | None = None,
        aggregate_data: bool = True,
        include_storage: bool = True,
        flexibility_percent: float = 0,
        flexibility_penalty: float = 0,
        time_series_for_high_peaks: list[TimeSeriesData] | None = None,
        time_series_for_low_peaks: list[TimeSeriesData] | None = None,
        # Clustering indices (optional - computed from tsam if not provided)
        cluster_order: xr.DataArray | None = None,
        period_length: int | None = None,
        segment_assignment: xr.DataArray | None = None,
        skip_first_of_period: bool = True,
        # External tsam aggregation for data transformation
        tsam_aggregation: tsam.TimeSeriesAggregation | None = None,
    ):
        import xarray as xr

        self.n_clusters = n_clusters
        self.cluster_duration = cluster_duration  # Store original for serialization
        self.cluster_duration_hours = _parse_cluster_duration(cluster_duration)
        self.n_segments = n_segments
        self.aggregate_data = aggregate_data
        self.include_storage = include_storage
        self.flexibility_percent = flexibility_percent
        self.flexibility_penalty = flexibility_penalty
        self.time_series_for_high_peaks: list[TimeSeriesData] = time_series_for_high_peaks or []
        self.time_series_for_low_peaks: list[TimeSeriesData] = time_series_for_low_peaks or []
        self.skip_first_of_period = skip_first_of_period
        self.tsam_aggregation = tsam_aggregation  # Not serialized - runtime only

        # Clustering indices - ensure DataArrays have names for IO
        if cluster_order is not None:
            if isinstance(cluster_order, xr.DataArray):
                self.cluster_order = (
                    cluster_order.rename('cluster_order') if cluster_order.name is None else cluster_order
                )
            else:
                self.cluster_order = xr.DataArray(cluster_order, dims=['cluster_period'], name='cluster_order')
        else:
            self.cluster_order = None

        self.period_length = int(period_length) if period_length is not None else None

        if segment_assignment is not None:
            if isinstance(segment_assignment, xr.DataArray):
                self.segment_assignment = (
                    segment_assignment.rename('segment_assignment')
                    if segment_assignment.name is None
                    else segment_assignment
                )
            else:
                self.segment_assignment = xr.DataArray(
                    segment_assignment, dims=['cluster', 'position'], name='segment_assignment'
                )
        else:
            self.segment_assignment = None

        # Auto-populate indices from tsam if provided
        if tsam_aggregation is not None and not self.has_indices:
            self.populate_from_tsam(tsam_aggregation)

    @property
    def has_indices(self) -> bool:
        """Whether clustering indices have been computed/provided."""
        return self.cluster_order is not None and self.period_length is not None

    @property
    def use_extreme_periods(self) -> bool:
        """Whether extreme segment selection is enabled."""
        return bool(self.time_series_for_high_peaks or self.time_series_for_low_peaks)

    @property
    def use_segmentation(self) -> bool:
        """Whether inner-period segmentation is enabled."""
        return self.n_segments is not None

    @property
    def labels_for_high_peaks(self) -> list[str]:
        """Names of time series used for high peak selection."""
        return [ts.name for ts in self.time_series_for_high_peaks]

    @property
    def labels_for_low_peaks(self) -> list[str]:
        """Names of time series used for low peak selection."""
        return [ts.name for ts in self.time_series_for_low_peaks]

    def populate_from_tsam(self, aggregation: tsam.TimeSeriesAggregation) -> None:
        """Populate clustering indices from a tsam TimeSeriesAggregation object.

        Args:
            aggregation: tsam object after calling createTypicalPeriods().
        """
        import xarray as xr

        if not TSAM_AVAILABLE:
            raise ImportError("The 'tsam' package is required. Install with 'pip install tsam'.")

        self.period_length = int(aggregation.hoursPerPeriod / aggregation.resolution)
        self.cluster_order = xr.DataArray(aggregation.clusterOrder, dims=['cluster_period'], name='cluster_order')

        # Build segment assignment if segmentation is used
        if aggregation.segmentation and hasattr(aggregation, 'segmentDurationDict'):
            n_clusters = aggregation.noTypicalPeriods
            segment_duration_dict = aggregation.segmentDurationDict['Segment Duration']

            # Build (cluster, position) -> segment_id mapping
            arr = np.zeros((n_clusters, self.period_length), dtype=np.int32)
            for cluster_id in range(n_clusters):
                pos = 0
                for seg_idx in range(aggregation.noSegments):
                    duration = segment_duration_dict[(cluster_id, seg_idx)]
                    arr[cluster_id, pos : pos + duration] = seg_idx
                    pos += duration

            self.segment_assignment = xr.DataArray(arr, dims=['cluster', 'position'], name='segment_assignment')

    def get_cluster_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Get inter-cluster equation pairs (i, j) where var[i] == var[j].

        Returns:
            Tuple of (idx_i, idx_j) arrays of timestep indices to equate.
        """
        if self.cluster_order is None or self.period_length is None:
            raise ValueError('Clustering indices not set. Call populate_from_tsam() first or provide cluster_order.')

        cluster_to_periods: dict[int, list[int]] = {}
        for period_idx, cluster_id in enumerate(self.cluster_order.values):
            cluster_to_periods.setdefault(int(cluster_id), []).append(period_idx)

        idx_i, idx_j = [], []
        start_pos = 1 if self.skip_first_of_period else 0

        for periods in cluster_to_periods.values():
            if len(periods) <= 1:
                continue
            first_period = periods[0]
            for pos in range(start_pos, self.period_length):
                first_ts = first_period * self.period_length + pos
                for other_period in periods[1:]:
                    idx_i.append(first_ts)
                    idx_j.append(other_period * self.period_length + pos)

        return np.array(idx_i, dtype=np.int32), np.array(idx_j, dtype=np.int32)

    def get_segment_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Get intra-segment equation pairs (i, j) where var[i] == var[j].

        Returns:
            Tuple of (idx_i, idx_j) arrays of timestep indices to equate.
        """
        if self.segment_assignment is None:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        if self.cluster_order is None or self.period_length is None:
            raise ValueError('Clustering indices not set. Call populate_from_tsam() first or provide cluster_order.')

        idx_i, idx_j = [], []
        seg_arr = self.segment_assignment.values  # (cluster, position)

        for period_idx, cluster_id in enumerate(self.cluster_order.values):
            period_offset = period_idx * self.period_length
            segment_ids = seg_arr[int(cluster_id)]  # (position,)

            # Group positions by segment
            for seg_id in np.unique(segment_ids):
                positions = np.where(segment_ids == seg_id)[0]
                if len(positions) > 1:
                    first_ts = period_offset + positions[0]
                    for pos in positions[1:]:
                        idx_i.append(first_ts)
                        idx_j.append(period_offset + pos)

        return np.array(idx_i, dtype=np.int32), np.array(idx_j, dtype=np.int32)


class ClusteringModel(Submodel):
    """Model that adds clustering constraints to equate variables across clustered time segments.

    Creates equations that equate variable values at corresponding time indices within the same cluster,
    and optionally allows binary variables to deviate with a penalty.
    """

    def __init__(
        self,
        model: FlowSystemModel,
        clustering_parameters: ClusteringParameters,
        flow_system: FlowSystem,
        components_to_clusterize: list[Component] | None = None,
        period_selector: int | str | None = None,
        scenario_selector: str | None = None,
    ):
        """
        Args:
            model: The FlowSystemModel to add constraints to.
            clustering_parameters: Parameters controlling clustering behavior (must have indices populated).
            flow_system: The FlowSystem being optimized.
            components_to_clusterize: Components to apply clustering to. If None, all components.
            period_selector: If provided, only add constraints for this period (for multi-period FlowSystems).
            scenario_selector: If provided, only add constraints for this scenario (for multi-scenario FlowSystems).
        """
        # Include period/scenario in label for multi-dimensional cases
        label_suffix = ''
        if period_selector is not None:
            label_suffix += f'|{period_selector}'
        if scenario_selector is not None:
            label_suffix += f'|{scenario_selector}'

        super().__init__(model, label_of_element='Clustering', label_of_model=f'Clustering{label_suffix}')
        self.flow_system = flow_system
        self.clustering_parameters = clustering_parameters
        self.components_to_clusterize = components_to_clusterize
        self.period_selector = period_selector
        self.scenario_selector = scenario_selector

    def do_modeling(self):
        """Create equality constraints for clustered time indices.

        Equalizes:
        - flow_rate: continuous flow variables (batched into single constraint)
        - status: binary on/off variables (individual constraints)
        - inside_piece: piecewise segment binaries (individual constraints)
        """
        if not self.clustering_parameters.has_indices:
            raise ValueError(
                'ClusteringParameters must have indices populated. '
                'Call populate_from_tsam() or provide cluster_order/period_length directly.'
            )

        components = self.components_to_clusterize or list(self.flow_system.components.values())

        # Collect variables to equalize, grouped by type
        continuous_vars: dict[str, linopy.Variable] = {}
        binary_vars: dict[str, linopy.Variable] = {}

        for component in components:
            if isinstance(component, Storage) and not self.clustering_parameters.include_storage:
                continue

            for flow in component.inputs + component.outputs:
                # Continuous: flow_rate (when aggregating data)
                if self.clustering_parameters.aggregate_data:
                    name = f'{flow.label_full}|flow_rate'
                    if name in component.submodel.variables:
                        continuous_vars[name] = component.submodel.variables[name]

                # Binary: status
                name = f'{flow.label_full}|status'
                if name in component.submodel.variables:
                    binary_vars[name] = component.submodel.variables[name]

            # Binary: piecewise segment selection
            piecewise = getattr(component.submodel, 'piecewise_conversion', None)
            if piecewise is not None:
                for piece in piecewise.pieces:
                    if piece.inside_piece is not None:
                        binary_vars[piece.inside_piece.name] = piece.inside_piece

        # Create constraints from clustering parameters
        params = self.clustering_parameters

        for constraint_type, idx_pair in [
            ('cluster', params.get_cluster_indices()),
            ('segment', params.get_segment_indices()),
        ]:
            if len(idx_pair[0]) == 0:
                continue

            # Batch continuous variables into single constraint
            if continuous_vars:
                self._add_equality_constraint(continuous_vars, idx_pair, f'base_{constraint_type}')

            # Individual constraints for binaries (needed for flexibility correction vars)
            for var in binary_vars.values():
                self._add_equality_constraint(
                    {var.name: var}, idx_pair, f'base_{constraint_type}|{var.name}', allow_flexibility=True
                )

        # Add penalty for flexibility deviations
        self._add_flexibility_penalty()

    def _add_equality_constraint(
        self,
        variables: dict[str, linopy.Variable],
        indices: tuple[np.ndarray, np.ndarray],
        suffix: str,
        allow_flexibility: bool = False,
    ) -> None:
        """Add equality constraint: var[idx_i] == var[idx_j] for all index pairs.

        Args:
            variables: Variables to constrain (batched if multiple).
            indices: Tuple of (idx_i, idx_j) arrays - timesteps to equate.
            suffix: Constraint name suffix.
            allow_flexibility: If True, add correction variables for binaries.
        """
        import linopy

        idx_i, idx_j = indices
        n_equations = len(idx_i)

        # Build constraint expression for each variable
        expressions = []
        for name, var in variables.items():
            if 'time' not in var.dims:
                continue

            # For multi-period/scenario, select only the relevant slice
            # Each period/scenario has its own clustering indices
            if self.period_selector is not None and 'period' in var.dims:
                var = var.sel(period=self.period_selector)
            if self.scenario_selector is not None and 'scenario' in var.dims:
                var = var.sel(scenario=self.scenario_selector)

            # Compute difference: var[idx_i] - var[idx_j]
            diff = var.isel(time=idx_i) - var.isel(time=idx_j)

            # Replace time dim with integer eq_idx (avoids duplicate datetime coords)
            diff = diff.rename({'time': 'eq_idx'}).assign_coords(eq_idx=np.arange(n_equations))
            expressions.append(diff.expand_dims(variable=[name]))

        if not expressions:
            return

        # Merge into single expression with 'variable' dimension
        lhs = linopy.merge(*expressions, dim='variable') if len(expressions) > 1 else expressions[0]

        # Add flexibility for binaries
        if allow_flexibility and self.clustering_parameters.flexibility_percent > 0:
            var_name = next(iter(variables))  # Single variable for binary case
            if var_name in self._model.variables.binaries:
                lhs = self._add_binary_flexibility(lhs, n_equations, suffix, var_name)

        self.add_constraints(lhs == 0, short_name=f'equate_{suffix}')

    def _add_binary_flexibility(self, lhs, n_equations: int, suffix: str, var_name: str):
        """Add correction variables to allow limited binary deviations."""
        coords = [np.arange(n_equations)]
        dims = ['eq_idx']

        k_up = self.add_variables(binary=True, coords=coords, dims=dims, short_name=f'k_up_{suffix}|{var_name}')
        k_down = self.add_variables(binary=True, coords=coords, dims=dims, short_name=f'k_down_{suffix}|{var_name}')

        # Modified equation: diff + k_up - k_down == 0
        lhs = lhs + k_up - k_down

        # At most one correction per equation
        self.add_constraints(k_up + k_down <= 1, short_name=f'lock_k_{suffix}|{var_name}')

        # Limit total corrections
        max_corrections = int(self.clustering_parameters.flexibility_percent / 100 * n_equations)
        self.add_constraints(
            k_up.sum('eq_idx') + k_down.sum('eq_idx') <= max_corrections,
            short_name=f'limit_k_{suffix}|{var_name}',
        )

        return lhs

    def _add_flexibility_penalty(self):
        """Add penalty cost for flexibility correction variables."""
        penalty = self.clustering_parameters.flexibility_penalty
        if self.clustering_parameters.flexibility_percent == 0 or penalty == 0:
            return

        from .effects import PENALTY_EFFECT_LABEL

        for var in self.variables_direct.values():
            sum_dim = 'eq_idx' if 'eq_idx' in var.dims else 'time'
            self._model.effects.add_share_to_effects(
                name='Clustering',
                expressions={PENALTY_EFFECT_LABEL: (var * penalty).sum(sum_dim)},
                target='periodic',
            )


class TypicalPeriodsModel(Submodel):
    """Model that adds storage inter-period linking for typical periods optimization.

    When using cluster_reduce(), timesteps are reduced to only typical (representative)
    periods. This model creates variables and constraints to track storage state
    across the full original time horizon using boundary state variables.

    The approach:
    1. Create SOC_boundary[d] for each original period d (0 to n_original_periods)
    2. Compute delta_SOC[c] for each typical period c (change in SOC during period)
    3. Link: SOC_boundary[d+1] = SOC_boundary[d] + delta_SOC[cluster_order[d]]
    4. Optionally enforce cyclic: SOC_boundary[0] = SOC_boundary[n_original_periods]

    This allows the optimizer to properly value storage for long-term (seasonal)
    patterns while only solving for the typical period timesteps.
    """

    def __init__(
        self,
        model: FlowSystemModel,
        flow_system: FlowSystem,
        cluster_order: np.ndarray | list,
        cluster_occurrences: dict[int, int],
        n_typical_periods: int,
        timesteps_per_period: int,
        storage_cyclic: bool = True,
    ):
        """
        Args:
            model: The FlowSystemModel to add constraints to.
            flow_system: The FlowSystem being optimized.
            cluster_order: Array indicating which typical period (cluster) each original
                period belongs to. Length = n_original_periods.
            cluster_occurrences: Dict mapping cluster_id to number of original periods
                it represents.
            n_typical_periods: Number of typical (representative) periods.
            timesteps_per_period: Number of timesteps in each period.
            storage_cyclic: If True, enforce SOC_boundary[0] = SOC_boundary[end].
        """
        super().__init__(model, label_of_element='TypicalPeriods', label_of_model='TypicalPeriods')
        self.flow_system = flow_system
        self.cluster_order = np.array(cluster_order)
        self.cluster_occurrences = cluster_occurrences
        self.n_typical_periods = n_typical_periods
        self.timesteps_per_period = timesteps_per_period
        self.storage_cyclic = storage_cyclic
        self.n_original_periods = len(self.cluster_order)

    def do_modeling(self):
        """Create SOC boundary variables and inter-period linking constraints.

        For each storage:
        - SOC_boundary[d]: State of charge at start of original period d
        - delta_SOC[c]: Change in SOC during typical period c
        - Linking: SOC_boundary[d+1] = SOC_boundary[d] + delta_SOC[cluster_order[d]]
        """

        storages = list(self.flow_system.storages.values())
        if not storages:
            logger.info('No storages found - skipping inter-period linking')
            return

        logger.info(
            f'Adding inter-period storage linking for {len(storages)} storages '
            f'({self.n_original_periods} original periods, {self.nr_of_typical_periods} typical)'
        )

        for storage in storages:
            self._add_storage_linking(storage)

    def _add_storage_linking(self, storage) -> None:
        """Add inter-period linking constraints for a single storage.

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

        # Get storage capacity bounds
        capacity = storage.capacity_in_flow_hours
        if hasattr(capacity, 'fixed_size') and capacity.fixed_size is not None:
            cap_value = capacity.fixed_size
        elif hasattr(capacity, 'maximum') and capacity.maximum is not None:
            cap_value = float(capacity.maximum.max().item()) if hasattr(capacity.maximum, 'max') else capacity.maximum
        else:
            cap_value = 1e9  # Large default

        # Create SOC_boundary variables for each original period boundary
        # We need n_original_periods + 1 boundaries (start of first period through end of last)
        n_boundaries = self.n_original_periods + 1
        boundary_coords = [np.arange(n_boundaries)]
        boundary_dims = ['period_boundary']

        # Bounds: 0 <= SOC_boundary <= capacity
        lb = xr.DataArray(0.0, coords={'period_boundary': np.arange(n_boundaries)}, dims=['period_boundary'])
        ub = xr.DataArray(cap_value, coords={'period_boundary': np.arange(n_boundaries)}, dims=['period_boundary'])

        soc_boundary = self.add_variables(
            lower=lb,
            upper=ub,
            coords=boundary_coords,
            dims=boundary_dims,
            short_name=f'SOC_boundary|{label}',
        )

        # Pre-compute delta_SOC for each typical period
        # delta_SOC[c] = charge_state[c, end] - charge_state[c, start]
        # We store these as a dict since linopy expressions can't be concat'd with xr.concat
        delta_soc_dict = {}
        for c in range(self.nr_of_typical_periods):
            # Get start and end timestep indices for this typical period
            start_idx = c * self.timesteps_per_period
            end_idx = (c + 1) * self.timesteps_per_period  # charge_state has extra timestep at end

            # charge_state at end - charge_state at start of typical period c
            # Note: charge_state is indexed by time with extra timestep
            delta_soc_dict[c] = charge_state.isel(time=end_idx) - charge_state.isel(time=start_idx)

        # Create linking constraints:
        # SOC_boundary[d+1] = SOC_boundary[d] + delta_SOC[cluster_order[d]]
        for d in range(self.n_original_periods):
            c = int(self.cluster_order[d])  # Which typical period this original period maps to
            lhs = soc_boundary.isel(period_boundary=d + 1) - soc_boundary.isel(period_boundary=d) - delta_soc_dict[c]
            self.add_constraints(lhs == 0, short_name=f'inter_period_link|{label}|{d}')

        # Cyclic constraint: SOC_boundary[0] = SOC_boundary[end]
        if self.storage_cyclic:
            lhs = soc_boundary.isel(period_boundary=0) - soc_boundary.isel(period_boundary=self.n_original_periods)
            self.add_constraints(lhs == 0, short_name=f'cyclic|{label}')

        logger.debug(f'Added inter-period linking for storage {label}')
