"""
TSAM (Time Series Aggregation Module) backend for time series aggregation.

This backend wraps the existing flixopt Clustering class which uses the
tsam package to perform k-means clustering of time series into typical periods.

Terminology note:
- TSAM uses "typical periods" to mean representative time chunks (e.g., typical days)
- "cluster" = a group of similar time chunks (e.g., similar days)
- "cluster_duration" = length of each time chunk (e.g., 24h for daily clustering)
- "period" and "scenario" in method signatures refer to the MODEL's dimensions
  (years/months and scenarios), NOT the clustering time chunks
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from .base import AggregationResult, ClusterStructure

logger = logging.getLogger('flixopt')

# Check if tsam is available
try:
    import tsam.timeseriesaggregation as tsam

    TSAM_AVAILABLE = True
except ImportError:
    TSAM_AVAILABLE = False


def _parse_cluster_duration(duration: str | float) -> float:
    """Convert cluster duration to hours.

    Args:
        duration: Either a pandas-style duration string ('1D', '24h', '6h')
                  or a numeric value in hours.

    Returns:
        Duration in hours.
    """
    import pandas as pd

    if isinstance(duration, (int, float)):
        return float(duration)

    # Parse pandas-style duration strings
    td = pd.Timedelta(duration)
    return td.total_seconds() / 3600


class TSAMBackend:
    """TSAM-based time series aggregation backend.

    This backend uses the tsam (Time Series Aggregation Module) package
    to perform k-means clustering of time series into typical periods.

    Features:
        - Inter-period clustering (typical days/weeks)
        - Intra-period segmentation (reduce timesteps within periods)
        - Extreme period preservation (high/low peaks)
        - Custom weighting of time series for clustering

    Args:
        cluster_duration: Duration of each cluster period.
            Can be pandas-style string ('1D', '24h') or hours as float.
        n_segments: Number of segments within each period for intra-period
            clustering. None for no segmentation.
        time_series_for_high_peaks: Column names to preserve high-value periods for.
        time_series_for_low_peaks: Column names to preserve low-value periods for.
        weights: Dict mapping column names to clustering weights.

    Example:
        >>> backend = TSAMBackend(cluster_duration='1D', n_segments=4)
        >>> result = backend.aggregate(data, n_representatives=8)
    """

    def __init__(
        self,
        cluster_duration: str | float = '1D',
        n_segments: int | None = None,
        time_series_for_high_peaks: list[str] | None = None,
        time_series_for_low_peaks: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ):
        if not TSAM_AVAILABLE:
            raise ImportError("The 'tsam' package is required for TSAMBackend. Install it with 'pip install tsam'.")

        self.cluster_duration = cluster_duration
        self.cluster_duration_hours = _parse_cluster_duration(cluster_duration)
        self.n_segments = n_segments
        self.time_series_for_high_peaks = time_series_for_high_peaks or []
        self.time_series_for_low_peaks = time_series_for_low_peaks or []
        self.weights = weights or {}

    @property
    def use_extreme_periods(self) -> bool:
        """Whether extreme period selection is enabled."""
        return bool(self.time_series_for_high_peaks or self.time_series_for_low_peaks)

    def aggregate(
        self,
        data: xr.Dataset,
        n_representatives: int,
        hours_per_timestep: float | None = None,
        **kwargs,
    ) -> AggregationResult:
        """Perform TSAM aggregation on the input data.

        For multi-dimensional data (period/scenario), aggregation is performed
        independently for each (period, scenario) combination.

        Args:
            data: Input time series data as xarray Dataset.
                Must have 'time' dimension.
            n_representatives: Target number of typical periods (clusters).
            hours_per_timestep: Duration of each timestep in hours.
                If None, inferred from time coordinates.
            **kwargs: Additional options passed to tsam.

        Returns:
            AggregationResult with mapping, weights, and aggregated data.
        """
        # Convert Dataset to DataFrame for tsam
        # Handle multi-dimensional case
        has_period = 'period' in data.dims
        has_scenario = 'scenario' in data.dims

        if has_period or has_scenario:
            return self._aggregate_multi_dimensional(data, n_representatives, hours_per_timestep, **kwargs)
        else:
            return self._aggregate_single(data, n_representatives, hours_per_timestep, **kwargs)

    def _aggregate_single(
        self,
        data: xr.Dataset,
        n_representatives: int,
        hours_per_timestep: float | None = None,
        **kwargs,
    ) -> AggregationResult:
        """Aggregate a single-dimensional time series."""
        import pandas as pd

        # Convert to DataFrame
        df = data.to_dataframe()
        if isinstance(df.index, pd.MultiIndex):
            # Flatten multi-index (shouldn't happen for single-dim, but be safe)
            df = df.reset_index(drop=True)

        n_timesteps = len(df)

        # Infer hours_per_timestep if not provided
        if hours_per_timestep is None:
            if 'time' in data.coords and hasattr(data.coords['time'], 'values'):
                time_vals = pd.to_datetime(data.coords['time'].values)
                if len(time_vals) > 1:
                    hours_per_timestep = (time_vals[1] - time_vals[0]).total_seconds() / 3600
                else:
                    hours_per_timestep = 1.0
            else:
                hours_per_timestep = 1.0

        # Calculate number of timesteps per period
        timesteps_per_period = int(self.cluster_duration_hours / hours_per_timestep)
        total_periods = n_timesteps // timesteps_per_period

        # Determine actual number of clusters
        n_clusters = min(n_representatives, total_periods)

        # Create tsam aggregation
        tsam_agg = tsam.TimeSeriesAggregation(
            df,
            noTypicalPeriods=n_clusters,
            hoursPerPeriod=self.cluster_duration_hours,
            resolution=hours_per_timestep,
            clusterMethod='k_means',
            extremePeriodMethod='new_cluster_center' if self.use_extreme_periods else 'None',
            weightDict={name: w for name, w in self.weights.items() if name in df.columns},
            addPeakMax=self.time_series_for_high_peaks,
            addPeakMin=self.time_series_for_low_peaks,
            segmentation=self.n_segments is not None,
            noSegments=self.n_segments if self.n_segments is not None else 1,
        )

        tsam_agg.createTypicalPeriods()
        aggregated_df = tsam_agg.predictOriginalData()

        # Build timestep mapping
        # For each original timestep, find which representative timestep it maps to
        cluster_order = tsam_agg.clusterOrder
        timestep_mapping = np.zeros(n_timesteps, dtype=np.int32)

        for period_idx, cluster_id in enumerate(cluster_order):
            for pos in range(timesteps_per_period):
                original_idx = period_idx * timesteps_per_period + pos
                if original_idx < n_timesteps:
                    representative_idx = cluster_id * timesteps_per_period + pos
                    timestep_mapping[original_idx] = representative_idx

        # Build representative weights (how many originals each representative covers)
        n_representative_timesteps = n_clusters * timesteps_per_period
        representative_weights = np.zeros(n_representative_timesteps, dtype=np.float64)

        for cluster_id, count in tsam_agg.clusterPeriodNoOccur.items():
            for pos in range(timesteps_per_period):
                rep_idx = cluster_id * timesteps_per_period + pos
                if rep_idx < n_representative_timesteps:
                    representative_weights[rep_idx] = count

        # Create cluster structure for storage linking
        cluster_occurrences = xr.DataArray(
            [tsam_agg.clusterPeriodNoOccur.get(c, 0) for c in range(n_clusters)],
            dims=['cluster'],
            name='cluster_occurrences',
        )

        cluster_structure = ClusterStructure(
            cluster_order=xr.DataArray(cluster_order, dims=['original_period'], name='cluster_order'),
            cluster_occurrences=cluster_occurrences,
            n_clusters=n_clusters,
            timesteps_per_cluster=timesteps_per_period,
        )

        # Convert aggregated data to xarray Dataset
        # Extract only the typical period timesteps
        typical_timesteps = n_clusters * timesteps_per_period
        aggregated_ds = xr.Dataset(
            {col: (['time'], aggregated_df[col].values[:typical_timesteps]) for col in aggregated_df.columns},
            coords={'time': np.arange(typical_timesteps)},
        )

        return AggregationResult(
            timestep_mapping=xr.DataArray(timestep_mapping, dims=['original_time'], name='timestep_mapping'),
            n_representatives=n_representative_timesteps,
            representative_weights=xr.DataArray(representative_weights, dims=['time'], name='representative_weights'),
            aggregated_data=aggregated_ds,
            cluster_structure=cluster_structure,
            original_data=data,
        )

    def _aggregate_multi_dimensional(
        self,
        data: xr.Dataset,
        n_representatives: int,
        hours_per_timestep: float | None = None,
        **kwargs,
    ) -> AggregationResult:
        """Aggregate multi-dimensional data (with period/scenario dims).

        Performs independent aggregation for each (period, scenario) combination,
        then combines results into multi-dimensional arrays.
        """

        has_period = 'period' in data.dims
        has_scenario = 'scenario' in data.dims

        periods = data.coords['period'].values if has_period else [None]
        scenarios = data.coords['scenario'].values if has_scenario else [None]

        # Collect results for each combination
        results: dict[tuple, AggregationResult] = {}

        for period in periods:
            for scenario in scenarios:
                # Select slice
                slice_data = data
                if period is not None:
                    slice_data = slice_data.sel(period=period)
                if scenario is not None:
                    slice_data = slice_data.sel(scenario=scenario)

                # Aggregate this slice
                result = self._aggregate_single(slice_data, n_representatives, hours_per_timestep, **kwargs)
                results[(period, scenario)] = result

        # Combine results into multi-dimensional arrays
        # For now, assume all slices have same n_representatives (simplification)
        first_result = next(iter(results.values()))
        n_rep = first_result.n_representatives
        n_original = first_result.n_original_timesteps

        # Build multi-dimensional timestep_mapping
        if has_period and has_scenario:
            mapping_data = np.zeros((n_original, len(periods), len(scenarios)), dtype=np.int32)
            weights_data = np.zeros((n_rep, len(periods), len(scenarios)), dtype=np.float64)
            for (p, s), res in results.items():
                pi = list(periods).index(p)
                si = list(scenarios).index(s)
                mapping_data[:, pi, si] = res.timestep_mapping.values
                weights_data[:, pi, si] = res.representative_weights.values

            timestep_mapping = xr.DataArray(
                mapping_data,
                dims=['original_time', 'period', 'scenario'],
                coords={'original_time': np.arange(n_original), 'period': periods, 'scenario': scenarios},
                name='timestep_mapping',
            )
            representative_weights = xr.DataArray(
                weights_data,
                dims=['time', 'period', 'scenario'],
                coords={'time': np.arange(n_rep), 'period': periods, 'scenario': scenarios},
                name='representative_weights',
            )
        elif has_period:
            mapping_data = np.zeros((n_original, len(periods)), dtype=np.int32)
            weights_data = np.zeros((n_rep, len(periods)), dtype=np.float64)
            for (p, _), res in results.items():
                pi = list(periods).index(p)
                mapping_data[:, pi] = res.timestep_mapping.values
                weights_data[:, pi] = res.representative_weights.values

            timestep_mapping = xr.DataArray(
                mapping_data,
                dims=['original_time', 'period'],
                coords={'original_time': np.arange(n_original), 'period': periods},
                name='timestep_mapping',
            )
            representative_weights = xr.DataArray(
                weights_data,
                dims=['time', 'period'],
                coords={'time': np.arange(n_rep), 'period': periods},
                name='representative_weights',
            )
        else:  # has_scenario only
            mapping_data = np.zeros((n_original, len(scenarios)), dtype=np.int32)
            weights_data = np.zeros((n_rep, len(scenarios)), dtype=np.float64)
            for (_, s), res in results.items():
                si = list(scenarios).index(s)
                mapping_data[:, si] = res.timestep_mapping.values
                weights_data[:, si] = res.representative_weights.values

            timestep_mapping = xr.DataArray(
                mapping_data,
                dims=['original_time', 'scenario'],
                coords={'original_time': np.arange(n_original), 'scenario': scenarios},
                name='timestep_mapping',
            )
            representative_weights = xr.DataArray(
                weights_data,
                dims=['time', 'scenario'],
                coords={'time': np.arange(n_rep), 'scenario': scenarios},
                name='representative_weights',
            )

        # Use cluster structure from first result (for now - could be enhanced)
        # In multi-dimensional case, cluster structure may vary by period/scenario
        cluster_structure = first_result.cluster_structure

        return AggregationResult(
            timestep_mapping=timestep_mapping,
            n_representatives=n_rep,
            representative_weights=representative_weights,
            aggregated_data=first_result.aggregated_data,  # Simplified - use first slice's data
            cluster_structure=cluster_structure,
            original_data=data,
        )


def plot_aggregation(
    result: AggregationResult,
    colormap: str | None = None,
    show: bool | None = None,
):
    """Plot original vs aggregated data comparison.

    Visualizes the original time series (dashed lines) overlaid with
    the aggregated/clustered time series (solid lines) for comparison.

    Args:
        result: AggregationResult containing original and aggregated data.
        colormap: Colorscale name for the time series colors.
            Defaults to CONFIG.Plotting.default_qualitative_colorscale.
        show: Whether to display the figure.
            Defaults to CONFIG.Plotting.default_show.

    Returns:
        PlotResult containing the comparison figure and underlying data.

    Example:
        >>> result = backend.aggregate(data, n_representatives=8)
        >>> plot_aggregation(result)
    """
    import plotly.express as px

    from ..color_processing import process_colors
    from ..config import CONFIG
    from ..plot_result import PlotResult

    if result.original_data is None or result.aggregated_data is None:
        raise ValueError('AggregationResult must contain both original_data and aggregated_data for plotting')

    # Convert xarray to DataFrames
    original_df = result.original_data.to_dataframe()
    aggregated_df = result.aggregated_data.to_dataframe()

    # Expand aggregated data to original length using mapping
    mapping = result.timestep_mapping.values
    expanded_agg = aggregated_df.iloc[mapping].reset_index(drop=True)

    # Rename for legend
    original_df = original_df.rename(columns={col: f'Original - {col}' for col in original_df.columns})
    expanded_agg = expanded_agg.rename(columns={col: f'Aggregated - {col}' for col in expanded_agg.columns})

    colors = list(
        process_colors(colormap or CONFIG.Plotting.default_qualitative_colorscale, list(original_df.columns)).values()
    )

    # Create line plot for original data (dashed)
    original_df = original_df.reset_index()
    index_name = original_df.columns[0]
    df_org_long = original_df.melt(id_vars=index_name, var_name='variable', value_name='value')
    fig = px.line(df_org_long, x=index_name, y='value', color='variable', color_discrete_sequence=colors)
    for trace in fig.data:
        trace.update(line=dict(dash='dash'))

    # Add aggregated data (solid lines)
    expanded_agg[index_name] = original_df[index_name]
    df_agg_long = expanded_agg.melt(id_vars=index_name, var_name='variable', value_name='value')
    fig2 = px.line(df_agg_long, x=index_name, y='value', color='variable', color_discrete_sequence=colors)
    for trace in fig2.data:
        fig.add_trace(trace)

    fig.update_layout(
        title='Original vs Aggregated Data (original = ---)',
        xaxis_title='Time',
        yaxis_title='Value',
    )

    # Build xarray Dataset with both original and aggregated data
    data = xr.Dataset(
        {
            'original': result.original_data.to_array(dim='variable'),
            'aggregated': result.aggregated_data.to_array(dim='variable'),
        }
    )
    plot_result = PlotResult(data=data, figure=fig)

    if show is None:
        show = CONFIG.Plotting.default_show
    if show:
        plot_result.show()

    return plot_result
