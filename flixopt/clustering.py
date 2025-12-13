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
    Submodel,
)

if TYPE_CHECKING:
    import linopy
    import pandas as pd

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

        period_length = len(self.tsam.stepIdx)
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


class ClusteringParameters:
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

        Segmentation only (no clustering, just reduce to 4 segments per day):

        >>> clustered_fs = flow_system.transform.cluster(
        ...     n_clusters=None,  # Skip clustering
        ...     cluster_duration='1D',
        ...     n_segments=4,
        ... )
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
    ):
        self.n_clusters = n_clusters
        self.cluster_duration_hours = _parse_cluster_duration(cluster_duration)
        self.n_segments = n_segments
        self.aggregate_data = aggregate_data
        self.include_storage = include_storage
        self.flexibility_percent = flexibility_percent
        self.flexibility_penalty = flexibility_penalty
        self.time_series_for_high_peaks: list[TimeSeriesData] = time_series_for_high_peaks or []
        self.time_series_for_low_peaks: list[TimeSeriesData] = time_series_for_low_peaks or []

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
        clustering_data: Clustering | dict[tuple, Clustering],
        components_to_clusterize: list[Component] | None,
    ):
        """
        Args:
            model: The FlowSystemModel to add constraints to.
            clustering_parameters: Parameters controlling clustering behavior.
            flow_system: The FlowSystem being optimized.
            clustering_data: Either a single Clustering object (simple case) or a dict
                mapping (period_label, scenario_label) tuples to Clustering objects
                (multi-dimensional case).
            components_to_clusterize: Components to apply clustering to. If None, all components.
        """
        super().__init__(model, label_of_element='Clustering', label_of_model='Clustering')
        self.flow_system = flow_system
        self.clustering_parameters = clustering_parameters
        self.components_to_clusterize = components_to_clusterize

        # Handle both single and multi-dimensional clustering
        if isinstance(clustering_data, dict):
            self.clustering_data_dict = clustering_data
            self.is_multi_dimensional = True
        else:
            self.clustering_data_dict = {(None, None): clustering_data}
            self.is_multi_dimensional = False

    def do_modeling(self):
        if not self.components_to_clusterize:
            components = list(self.flow_system.components.values())
        else:
            components = list(self.components_to_clusterize)

        time_variables: set[str] = {
            name for name in self._model.variables if 'time' in self._model.variables[name].dims
        }
        binary_variables: set[str] = set(self._model.variables.binaries)
        binary_time_variables: set[str] = time_variables & binary_variables

        for component in components:
            if isinstance(component, Storage) and not self.clustering_parameters.include_storage:
                continue  # Skip storage if not included

            all_variables_of_component = set(component.submodel.variables)

            if self.clustering_parameters.aggregate_data:
                relevant_variables = component.submodel.variables[all_variables_of_component & time_variables]
            else:
                relevant_variables = component.submodel.variables[all_variables_of_component & binary_time_variables]

            for variable in relevant_variables:
                self._equate_indices_multi_dimensional(component.submodel.variables[variable])

        # Add penalty for flexibility deviations
        penalty = self.clustering_parameters.flexibility_penalty
        if self.clustering_parameters.flexibility_percent > 0 and penalty != 0:
            from .effects import PENALTY_EFFECT_LABEL

            for variable_name in self.variables_direct:
                variable = self.variables_direct[variable_name]
                self._model.effects.add_share_to_effects(
                    name='Clustering',
                    expressions={PENALTY_EFFECT_LABEL: (variable * penalty).sum('time')},
                    target='periodic',
                )

    def _equate_indices_multi_dimensional(self, variable: linopy.Variable) -> None:
        """Equate indices across clustered segments, handling multi-dimensional cases."""
        var_dims = set(variable.dims)
        has_period = 'period' in var_dims
        has_scenario = 'scenario' in var_dims

        for (period_label, scenario_label), clustering in self.clustering_data_dict.items():
            indices = clustering.get_equation_indices(skip_first_index_of_period=True)

            if len(indices[0]) == 0:
                continue  # No constraints needed for this cluster

            # Build selector for this period/scenario combination
            selector = {}
            if has_period and period_label is not None:
                selector['period'] = period_label
            if has_scenario and scenario_label is not None:
                selector['scenario'] = scenario_label

            # Select variable slice for this dimension combination
            if selector:
                var_slice = variable.sel(**selector)
            else:
                var_slice = variable

            # Create constraint name with dimension info
            dim_suffix = ''
            if period_label is not None:
                dim_suffix += f'_p{period_label}'
            if scenario_label is not None:
                dim_suffix += f'_s{scenario_label}'

            # Equate indices within this slice
            self._equate_indices(var_slice, indices, dim_suffix, variable.name)

    def _equate_indices(
        self,
        variable: linopy.Variable,
        indices: tuple[np.ndarray, np.ndarray],
        dim_suffix: str = '',
        original_var_name: str | None = None,
    ) -> None:
        """Add constraints to equate variable values at corresponding cluster indices."""
        assert len(indices[0]) == len(indices[1]), 'The length of the indices must match!'
        length = len(indices[0])
        var_name = original_var_name or variable.name

        # Main constraint: x(cluster_a, t) - x(cluster_b, t) = 0
        con = self.add_constraints(
            variable.isel(time=indices[0]) - variable.isel(time=indices[1]) == 0,
            short_name=f'equate_indices{dim_suffix}|{var_name}',
        )

        # Add correction variables for binary flexibility
        if var_name in self._model.variables.binaries and self.clustering_parameters.flexibility_percent > 0:
            sel = variable.isel(time=indices[0])
            coords = {d: sel.indexes[d] for d in sel.dims}
            var_k1 = self.add_variables(binary=True, coords=coords, short_name=f'correction1{dim_suffix}|{var_name}')
            var_k0 = self.add_variables(binary=True, coords=coords, short_name=f'correction0{dim_suffix}|{var_name}')

            # Extend equation to allow deviation: On(a,t) - On(b,t) + K1 - K0 = 0
            con.lhs += 1 * var_k1 - 1 * var_k0

            # Interlock K0 and K1: can't both be 1
            self.add_constraints(var_k0 + var_k1 <= 1, short_name=f'lock_k0_and_k1{dim_suffix}|{var_name}')

            # Limit total corrections
            limit = int(np.floor(self.clustering_parameters.flexibility_percent / 100 * length))
            self.add_constraints(
                var_k0.sum(dim='time') + var_k1.sum(dim='time') <= limit,
                short_name=f'limit_corrections{dim_suffix}|{var_name}',
            )
