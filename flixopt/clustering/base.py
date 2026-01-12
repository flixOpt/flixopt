"""
Clustering classes for time series aggregation.

This module provides a thin wrapper around tsam's clustering functionality,
storing AggregationResult objects directly and deriving properties on-demand.

The key class is `Clustering`, which is stored on FlowSystem after clustering.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from pathlib import Path

    from tsam import AggregationResult

    from ..color_processing import ColorType
    from ..plot_result import PlotResult
    from ..statistics_accessor import SelectType


def _select_dims(da: xr.DataArray, period: Any = None, scenario: Any = None) -> xr.DataArray:
    """Select from DataArray by period/scenario if those dimensions exist."""
    if 'period' in da.dims and period is not None:
        da = da.sel(period=period)
    if 'scenario' in da.dims and scenario is not None:
        da = da.sel(scenario=scenario)
    return da


class Clustering:
    """Clustering information for a FlowSystem.

    Stores tsam AggregationResult objects directly and provides
    convenience accessors for common operations.

    This is a thin wrapper around tsam 3.0's API. The actual clustering
    logic is delegated to tsam, and this class only:
    1. Manages results for multiple (period, scenario) dimensions
    2. Provides xarray-based convenience properties
    3. Handles JSON persistence via tsam's ClusteringResult

    Attributes:
        tsam_results: Dict mapping (period, scenario) tuples to tsam AggregationResult.
            For simple cases without periods/scenarios, use ``{(): result}``.
        dim_names: Names of extra dimensions, e.g., ``['period', 'scenario']``.
        original_timesteps: Original timesteps before clustering.
        cluster_order: Pre-computed DataArray mapping original clusters to representative clusters.
        original_data: Original dataset before clustering (for expand/plotting).
        aggregated_data: Aggregated dataset after clustering (for plotting).

    Example:
        >>> fs_clustered = flow_system.transform.cluster(n_clusters=8, cluster_duration='1D')
        >>> fs_clustered.clustering.n_clusters
        8
        >>> fs_clustered.clustering.cluster_order
        <xarray.DataArray (original_cluster: 365)>
        >>> fs_clustered.clustering.plot.compare()
    """

    # ==========================================================================
    # Core properties derived from first tsam result
    # ==========================================================================

    @property
    def _first_result(self) -> AggregationResult | None:
        """Get the first AggregationResult (for structure info)."""
        if self.tsam_results is None:
            return None
        return next(iter(self.tsam_results.values()))

    @property
    def n_clusters(self) -> int:
        """Number of clusters (typical periods)."""
        if self._cached_n_clusters is not None:
            return self._cached_n_clusters
        if self._first_result is not None:
            return self._first_result.n_clusters
        # Infer from cluster_order
        return int(self.cluster_order.max().item()) + 1

    @property
    def timesteps_per_cluster(self) -> int:
        """Number of timesteps in each cluster."""
        if self._cached_timesteps_per_cluster is not None:
            return self._cached_timesteps_per_cluster
        if self._first_result is not None:
            return self._first_result.n_timesteps_per_period
        # Infer from aggregated_data
        if self.aggregated_data is not None and 'time' in self.aggregated_data.dims:
            return len(self.aggregated_data.time)
        # Fallback
        return len(self.original_timesteps) // self.n_original_clusters

    @property
    def timesteps_per_period(self) -> int:
        """Alias for timesteps_per_cluster."""
        return self.timesteps_per_cluster

    @property
    def n_original_clusters(self) -> int:
        """Number of original periods (before clustering)."""
        return len(self.cluster_order.coords['original_cluster'])

    @property
    def n_representatives(self) -> int:
        """Number of representative timesteps after clustering."""
        return self.n_clusters * self.timesteps_per_cluster

    # ==========================================================================
    # Derived properties (computed from tsam results)
    # ==========================================================================

    @property
    def cluster_occurrences(self) -> xr.DataArray:
        """Count of how many original periods each cluster represents.

        Returns:
            DataArray with dims [cluster] or [cluster, period?, scenario?].
        """
        return self._build_cluster_occurrences()

    @property
    def representative_weights(self) -> xr.DataArray:
        """Weight for each cluster (number of original periods it represents).

        This is the same as cluster_occurrences but named for API consistency.
        Used as cluster_weight in FlowSystem.
        """
        return self.cluster_occurrences.rename('representative_weights')

    @property
    def timestep_mapping(self) -> xr.DataArray:
        """Mapping from original timesteps to representative timestep indices.

        Each value indicates which representative timestep index (0 to n_representatives-1)
        corresponds to each original timestep.
        """
        return self._build_timestep_mapping()

    @property
    def metrics(self) -> xr.Dataset:
        """Clustering quality metrics (RMSE, MAE, etc.).

        Returns:
            Dataset with dims [time_series, period?, scenario?].
        """
        if self._metrics is None:
            self._metrics = self._build_metrics()
        return self._metrics

    @property
    def cluster_start_positions(self) -> np.ndarray:
        """Integer positions where clusters start in reduced timesteps.

        Returns:
            1D array: [0, T, 2T, ...] where T = timesteps_per_cluster.
        """
        n_timesteps = self.n_clusters * self.timesteps_per_cluster
        return np.arange(0, n_timesteps, self.timesteps_per_cluster)

    # ==========================================================================
    # Methods
    # ==========================================================================

    def expand_data(
        self,
        aggregated: xr.DataArray,
        original_time: pd.DatetimeIndex | None = None,
    ) -> xr.DataArray:
        """Expand aggregated data back to original timesteps.

        Uses the timestep_mapping to map each original timestep to its
        representative value from the aggregated data.

        Args:
            aggregated: DataArray with aggregated (cluster, time) or (time,) dimension.
            original_time: Original time coordinates. Defaults to self.original_timesteps.

        Returns:
            DataArray expanded to original timesteps.
        """
        if original_time is None:
            original_time = self.original_timesteps

        timestep_mapping = self.timestep_mapping
        has_cluster_dim = 'cluster' in aggregated.dims
        timesteps_per_cluster = self.timesteps_per_cluster

        def _expand_slice(mapping: np.ndarray, data: xr.DataArray) -> np.ndarray:
            """Expand a single slice using the mapping."""
            if has_cluster_dim:
                cluster_ids = mapping // timesteps_per_cluster
                time_within = mapping % timesteps_per_cluster
                return data.values[cluster_ids, time_within]
            return data.values[mapping]

        # Simple case: no period/scenario dimensions
        extra_dims = [d for d in timestep_mapping.dims if d != 'original_time']
        if not extra_dims:
            expanded_values = _expand_slice(timestep_mapping.values, aggregated)
            return xr.DataArray(
                expanded_values,
                coords={'time': original_time},
                dims=['time'],
                attrs=aggregated.attrs,
            )

        # Multi-dimensional: expand each slice and recombine
        dim_coords = {d: list(timestep_mapping.coords[d].values) for d in extra_dims}
        expanded_slices = {}
        for combo in np.ndindex(*[len(v) for v in dim_coords.values()]):
            selector = {d: dim_coords[d][i] for d, i in zip(extra_dims, combo, strict=True)}
            mapping = _select_dims(timestep_mapping, **selector).values
            data_slice = (
                _select_dims(aggregated, **selector) if any(d in aggregated.dims for d in selector) else aggregated
            )
            expanded_slices[tuple(selector.values())] = xr.DataArray(
                _expand_slice(mapping, data_slice),
                coords={'time': original_time},
                dims=['time'],
            )

        # Concatenate along extra dimensions
        result_arrays = expanded_slices
        for dim in reversed(extra_dims):
            dim_vals = dim_coords[dim]
            grouped = {}
            for key, arr in result_arrays.items():
                rest_key = key[:-1] if len(key) > 1 else ()
                grouped.setdefault(rest_key, []).append(arr)
            result_arrays = {k: xr.concat(v, dim=pd.Index(dim_vals, name=dim)) for k, v in grouped.items()}
        result = list(result_arrays.values())[0]
        return result.transpose('time', ...).assign_attrs(aggregated.attrs)

    def get_result(
        self,
        period: Any = None,
        scenario: Any = None,
    ) -> AggregationResult:
        """Get the AggregationResult for a specific (period, scenario).

        Args:
            period: Period label (if applicable).
            scenario: Scenario label (if applicable).

        Returns:
            The tsam AggregationResult for the specified combination.
        """
        key = self._make_key(period, scenario)
        if key not in self.tsam_results:
            raise KeyError(f'No result found for {dict(zip(self.dim_names, key, strict=False))}')
        return self.tsam_results[key]

    def apply(
        self,
        data: pd.DataFrame,
        period: Any = None,
        scenario: Any = None,
    ) -> AggregationResult:
        """Apply the saved clustering to new data.

        Args:
            data: DataFrame with time series data to cluster.
            period: Period label (if applicable).
            scenario: Scenario label (if applicable).

        Returns:
            tsam AggregationResult with the clustering applied.
        """
        result = self.get_result(period, scenario)
        return result.clustering.apply(data)

    def to_json(self, path: str | Path) -> None:
        """Save the clustering for reuse.

        Uses tsam's ClusteringResult.to_json() for each (period, scenario).
        Can be loaded later with Clustering.from_json() and used with
        flow_system.transform.apply_clustering().

        Args:
            path: Path to save the JSON file.
        """
        data = {
            'dim_names': self.dim_names,
            'results': {},
        }

        for key, result in self.tsam_results.items():
            key_str = '|'.join(str(k) for k in key) if key else '__single__'
            data['results'][key_str] = result.clustering.to_dict()

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        original_timesteps: pd.DatetimeIndex,
    ) -> Clustering:
        """Load a clustering from JSON.

        Note: This creates a Clustering with only ClusteringResult objects
        (not full AggregationResult). Use flow_system.transform.apply_clustering()
        to apply it to data.

        Args:
            path: Path to the JSON file.
            original_timesteps: Original timesteps for the new FlowSystem.

        Returns:
            A Clustering that can be used with apply_clustering().
        """
        # We can't fully reconstruct AggregationResult from JSON
        # (it requires the data). Create a placeholder that stores
        # ClusteringResult for apply().
        # This is a "partial" Clustering - it can only be used with apply_clustering()
        raise NotImplementedError(
            'Clustering.from_json() is not yet implemented. '
            'Use tsam.ClusteringResult.from_json() directly and '
            'pass to flow_system.transform.apply_clustering().'
        )

    # ==========================================================================
    # Visualization
    # ==========================================================================

    @property
    def plot(self) -> ClusteringPlotAccessor:
        """Access plotting methods for clustering visualization.

        Returns:
            ClusteringPlotAccessor with compare(), heatmap(), and clusters() methods.
        """
        return ClusteringPlotAccessor(self)

    # ==========================================================================
    # Private helpers
    # ==========================================================================

    def _make_key(self, period: Any, scenario: Any) -> tuple:
        """Create a key tuple from period and scenario values."""
        key_parts = []
        for dim in self.dim_names:
            if dim == 'period':
                key_parts.append(period)
            elif dim == 'scenario':
                key_parts.append(scenario)
            else:
                raise ValueError(f'Unknown dimension: {dim}')
        return tuple(key_parts)

    def _build_cluster_occurrences(self) -> xr.DataArray:
        """Build cluster_occurrences DataArray from tsam results or cluster_order."""
        cluster_coords = np.arange(self.n_clusters)

        # If tsam_results is None, derive occurrences from cluster_order
        if self.tsam_results is None:
            # Count occurrences from cluster_order
            if self.cluster_order.ndim == 1:
                weights = np.bincount(self.cluster_order.values.astype(int), minlength=self.n_clusters)
                return xr.DataArray(weights, dims=['cluster'], coords={'cluster': cluster_coords})
            else:
                # Multi-dimensional case - compute per slice from cluster_order
                periods = self._get_periods()
                scenarios = self._get_scenarios()

                def _occurrences_from_cluster_order(key: tuple) -> xr.DataArray:
                    kwargs = dict(zip(self.dim_names, key, strict=False)) if key else {}
                    order = _select_dims(self.cluster_order, **kwargs).values if kwargs else self.cluster_order.values
                    weights = np.bincount(order.astype(int), minlength=self.n_clusters)
                    return xr.DataArray(
                        weights,
                        dims=['cluster'],
                        coords={'cluster': cluster_coords},
                    )

                # Build all combinations of periods/scenarios
                slices = {}
                has_periods = periods != [None]
                has_scenarios = scenarios != [None]

                if has_periods and has_scenarios:
                    for p in periods:
                        for s in scenarios:
                            slices[(p, s)] = _occurrences_from_cluster_order((p, s))
                elif has_periods:
                    for p in periods:
                        slices[(p,)] = _occurrences_from_cluster_order((p,))
                elif has_scenarios:
                    for s in scenarios:
                        slices[(s,)] = _occurrences_from_cluster_order((s,))
                else:
                    return _occurrences_from_cluster_order(())

                return self._combine_slices(slices, ['cluster'], periods, scenarios, 'cluster_occurrences')

        periods = self._get_periods()
        scenarios = self._get_scenarios()

        def _occurrences_for_key(key: tuple) -> xr.DataArray:
            result = self.tsam_results[key]
            weights = np.array([result.cluster_weights.get(c, 0) for c in range(self.n_clusters)])
            return xr.DataArray(
                weights,
                dims=['cluster'],
                coords={'cluster': cluster_coords},
            )

        if not self.dim_names:
            return _occurrences_for_key(())

        return self._combine_slices(
            {key: _occurrences_for_key(key) for key in self.tsam_results},
            ['cluster'],
            periods,
            scenarios,
            'cluster_occurrences',
        )

    def _build_timestep_mapping(self) -> xr.DataArray:
        """Build timestep_mapping DataArray from cluster_order."""
        n_original = len(self.original_timesteps)
        timesteps_per_cluster = self.timesteps_per_cluster
        cluster_order = self.cluster_order
        periods = self._get_periods()
        scenarios = self._get_scenarios()

        def _mapping_for_key(key: tuple) -> np.ndarray:
            # Build kwargs dict based on dim_names
            kwargs = dict(zip(self.dim_names, key, strict=False)) if key else {}
            order = _select_dims(cluster_order, **kwargs).values if kwargs else cluster_order.values
            mapping = np.zeros(n_original, dtype=np.int32)
            for period_idx, cluster_id in enumerate(order):
                for pos in range(timesteps_per_cluster):
                    original_idx = period_idx * timesteps_per_cluster + pos
                    if original_idx < n_original:
                        representative_idx = int(cluster_id) * timesteps_per_cluster + pos
                        mapping[original_idx] = representative_idx
            return mapping

        original_time_coord = self.original_timesteps.rename('original_time')

        if not self.dim_names:
            return xr.DataArray(
                _mapping_for_key(()),
                dims=['original_time'],
                coords={'original_time': original_time_coord},
                name='timestep_mapping',
            )

        # Build key combinations from periods/scenarios
        has_periods = periods != [None]
        has_scenarios = scenarios != [None]

        slices = {}
        if has_periods and has_scenarios:
            for p in periods:
                for s in scenarios:
                    key = (p, s)
                    slices[key] = xr.DataArray(
                        _mapping_for_key(key),
                        dims=['original_time'],
                        coords={'original_time': original_time_coord},
                    )
        elif has_periods:
            for p in periods:
                key = (p,)
                slices[key] = xr.DataArray(
                    _mapping_for_key(key),
                    dims=['original_time'],
                    coords={'original_time': original_time_coord},
                )
        elif has_scenarios:
            for s in scenarios:
                key = (s,)
                slices[key] = xr.DataArray(
                    _mapping_for_key(key),
                    dims=['original_time'],
                    coords={'original_time': original_time_coord},
                )

        return self._combine_slices(slices, ['original_time'], periods, scenarios, 'timestep_mapping')

    def _build_metrics(self) -> xr.Dataset:
        """Build metrics Dataset from tsam accuracy results."""
        periods = self._get_periods()
        scenarios = self._get_scenarios()

        # Collect metrics from each result
        metrics_all: dict[tuple, pd.DataFrame] = {}
        for key, result in self.tsam_results.items():
            try:
                accuracy = result.accuracy
                metrics_all[key] = pd.DataFrame(
                    {
                        'RMSE': accuracy.rmse,
                        'MAE': accuracy.mae,
                        'RMSE_duration': accuracy.rmse_duration,
                    }
                )
            except Exception:
                metrics_all[key] = pd.DataFrame()

        # Simple case
        if not self.dim_names:
            first_key = ()
            df = metrics_all.get(first_key, pd.DataFrame())
            if df.empty:
                return xr.Dataset()
            return xr.Dataset(
                {
                    col: xr.DataArray(df[col].values, dims=['time_series'], coords={'time_series': df.index})
                    for col in df.columns
                }
            )

        # Multi-dim case
        non_empty = {k: v for k, v in metrics_all.items() if not v.empty}
        if not non_empty:
            return xr.Dataset()

        sample_df = next(iter(non_empty.values()))
        data_vars = {}
        for metric in sample_df.columns:
            slices = {}
            for key, df in metrics_all.items():
                if df.empty:
                    slices[key] = xr.DataArray(
                        np.full(len(sample_df.index), np.nan),
                        dims=['time_series'],
                        coords={'time_series': list(sample_df.index)},
                    )
                else:
                    slices[key] = xr.DataArray(
                        df[metric].values,
                        dims=['time_series'],
                        coords={'time_series': list(df.index)},
                    )
            data_vars[metric] = self._combine_slices(slices, ['time_series'], periods, scenarios, metric)

        return xr.Dataset(data_vars)

    def _get_periods(self) -> list:
        """Get list of periods or [None] if no periods dimension."""
        if 'period' not in self.dim_names:
            return [None]
        if self.tsam_results is None:
            # Get from cluster_order dimensions
            if 'period' in self.cluster_order.dims:
                return list(self.cluster_order.period.values)
            return [None]
        idx = self.dim_names.index('period')
        return list(set(k[idx] for k in self.tsam_results.keys()))

    def _get_scenarios(self) -> list:
        """Get list of scenarios or [None] if no scenarios dimension."""
        if 'scenario' not in self.dim_names:
            return [None]
        if self.tsam_results is None:
            # Get from cluster_order dimensions
            if 'scenario' in self.cluster_order.dims:
                return list(self.cluster_order.scenario.values)
            return [None]
        idx = self.dim_names.index('scenario')
        return list(set(k[idx] for k in self.tsam_results.keys()))

    def _combine_slices(
        self,
        slices: dict[tuple, xr.DataArray],
        base_dims: list[str],
        periods: list,
        scenarios: list,
        name: str,
    ) -> xr.DataArray:
        """Combine per-(period, scenario) slices into a single DataArray.

        The keys in slices match the keys in tsam_results:
        - No dims: key = ()
        - Only period: key = (period,)
        - Only scenario: key = (scenario,)
        - Both: key = (period, scenario)
        """
        has_periods = periods != [None]
        has_scenarios = scenarios != [None]

        if not has_periods and not has_scenarios:
            return slices[()].rename(name)

        if has_periods and has_scenarios:
            period_arrays = []
            for p in periods:
                scenario_arrays = [slices[(p, s)] for s in scenarios]
                period_arrays.append(xr.concat(scenario_arrays, dim=pd.Index(scenarios, name='scenario')))
            result = xr.concat(period_arrays, dim=pd.Index(periods, name='period'))
        elif has_periods:
            # Keys are (period,) tuples
            result = xr.concat([slices[(p,)] for p in periods], dim=pd.Index(periods, name='period'))
        else:
            # Keys are (scenario,) tuples
            result = xr.concat([slices[(s,)] for s in scenarios], dim=pd.Index(scenarios, name='scenario'))

        # Put base dims first
        dim_order = base_dims + [d for d in result.dims if d not in base_dims]
        return result.transpose(*dim_order).rename(name)

    def _create_reference_structure(self) -> tuple[dict, dict[str, xr.DataArray]]:
        """Create serialization structure for to_dataset().

        Returns:
            Tuple of (reference_dict, arrays_dict).
        """
        arrays = {}

        # Collect original_data arrays
        original_data_refs = None
        if self.original_data is not None:
            original_data_refs = []
            for name, da in self.original_data.data_vars.items():
                ref_name = f'original_data|{name}'
                arrays[ref_name] = da
                original_data_refs.append(f':::{ref_name}')

        # Collect aggregated_data arrays
        aggregated_data_refs = None
        if self.aggregated_data is not None:
            aggregated_data_refs = []
            for name, da in self.aggregated_data.data_vars.items():
                ref_name = f'aggregated_data|{name}'
                arrays[ref_name] = da
                aggregated_data_refs.append(f':::{ref_name}')

        # Collect metrics arrays
        metrics_refs = None
        if self._metrics is not None:
            metrics_refs = []
            for name, da in self._metrics.data_vars.items():
                ref_name = f'metrics|{name}'
                arrays[ref_name] = da
                metrics_refs.append(f':::{ref_name}')

        # Add cluster_order
        arrays['cluster_order'] = self.cluster_order

        reference = {
            '__class__': 'Clustering',
            'dim_names': self.dim_names,
            'original_timesteps': [ts.isoformat() for ts in self.original_timesteps],
            '_cached_n_clusters': self.n_clusters,
            '_cached_timesteps_per_cluster': self.timesteps_per_cluster,
            'cluster_order': ':::cluster_order',
            'tsam_results': None,  # Can't serialize tsam results
            '_original_data_refs': original_data_refs,
            '_aggregated_data_refs': aggregated_data_refs,
            '_metrics_refs': metrics_refs,
        }

        return reference, arrays

    def __init__(
        self,
        tsam_results: dict[tuple, AggregationResult] | None,
        dim_names: list[str],
        original_timesteps: pd.DatetimeIndex | list[str],
        cluster_order: xr.DataArray,
        original_data: xr.Dataset | None = None,
        aggregated_data: xr.Dataset | None = None,
        _metrics: xr.Dataset | None = None,
        _cached_n_clusters: int | None = None,
        _cached_timesteps_per_cluster: int | None = None,
        # These are for reconstruction from serialization
        _original_data_refs: list[str] | None = None,
        _aggregated_data_refs: list[str] | None = None,
        _metrics_refs: list[str] | None = None,
    ):
        """Initialize Clustering object."""
        # Handle ISO timestamp strings from serialization
        if (
            isinstance(original_timesteps, list)
            and len(original_timesteps) > 0
            and isinstance(original_timesteps[0], str)
        ):
            original_timesteps = pd.DatetimeIndex([pd.Timestamp(ts) for ts in original_timesteps])

        self.tsam_results = tsam_results
        self.dim_names = dim_names
        self.original_timesteps = original_timesteps
        self.cluster_order = cluster_order
        self._metrics = _metrics
        self._cached_n_clusters = _cached_n_clusters
        self._cached_timesteps_per_cluster = _cached_timesteps_per_cluster

        # Handle reconstructed data from refs (list of DataArrays)
        if _original_data_refs is not None and isinstance(_original_data_refs, list):
            # These are resolved DataArrays from the structure resolver
            if all(isinstance(da, xr.DataArray) for da in _original_data_refs):
                self.original_data = xr.Dataset({da.name: da for da in _original_data_refs})
            else:
                self.original_data = original_data
        else:
            self.original_data = original_data

        if _aggregated_data_refs is not None and isinstance(_aggregated_data_refs, list):
            if all(isinstance(da, xr.DataArray) for da in _aggregated_data_refs):
                self.aggregated_data = xr.Dataset({da.name: da for da in _aggregated_data_refs})
            else:
                self.aggregated_data = aggregated_data
        else:
            self.aggregated_data = aggregated_data

        if _metrics_refs is not None and isinstance(_metrics_refs, list):
            if all(isinstance(da, xr.DataArray) for da in _metrics_refs):
                self._metrics = xr.Dataset({da.name: da for da in _metrics_refs})

        # Post-init validation
        if self.tsam_results is not None and len(self.tsam_results) == 0:
            raise ValueError('tsam_results cannot be empty')

        # If we have tsam_results, cache the values
        if self.tsam_results is not None:
            first_result = next(iter(self.tsam_results.values()))
            self._cached_n_clusters = first_result.n_clusters
            self._cached_timesteps_per_cluster = first_result.n_timesteps_per_period

    def __repr__(self) -> str:
        return (
            f'Clustering(\n'
            f'  {self.n_original_clusters} periods → {self.n_clusters} clusters\n'
            f'  timesteps_per_cluster={self.timesteps_per_cluster}\n'
            f'  dims={self.dim_names}\n'
            f')'
        )


class ClusteringPlotAccessor:
    """Plot accessor for Clustering objects.

    Provides visualization methods for comparing original vs aggregated data
    and understanding the clustering structure.
    """

    def __init__(self, clustering: Clustering):
        self._clustering = clustering

    def compare(
        self,
        kind: str = 'timeseries',
        variables: str | list[str] | None = None,
        *,
        select: SelectType | None = None,
        colors: ColorType | None = None,
        color: str | None = 'auto',
        line_dash: str | None = 'representation',
        facet_col: str | None = 'auto',
        facet_row: str | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Compare original vs aggregated data.

        Args:
            kind: Type of comparison plot ('timeseries' or 'duration_curve').
            variables: Variable(s) to plot. None for all time-varying variables.
            select: xarray-style selection dict.
            colors: Color specification.
            color: Dimension for line colors.
            line_dash: Dimension for line dash styles.
            facet_col: Dimension for subplot columns.
            facet_row: Dimension for subplot rows.
            show: Whether to display the figure.
            **plotly_kwargs: Additional arguments passed to plotly.

        Returns:
            PlotResult containing the comparison figure and underlying data.
        """
        from ..config import CONFIG
        from ..plot_result import PlotResult
        from ..statistics_accessor import _apply_selection

        if kind not in ('timeseries', 'duration_curve'):
            raise ValueError(f"Unknown kind '{kind}'. Use 'timeseries' or 'duration_curve'.")

        clustering = self._clustering
        if clustering.original_data is None or clustering.aggregated_data is None:
            raise ValueError('No original/aggregated data available for comparison')

        resolved_variables = self._resolve_variables(variables)

        # Build Dataset with variables as data_vars
        data_vars = {}
        for var in resolved_variables:
            original = clustering.original_data[var]
            clustered = clustering.expand_data(clustering.aggregated_data[var])
            combined = xr.concat([original, clustered], dim=pd.Index(['Original', 'Clustered'], name='representation'))
            data_vars[var] = combined
        ds = xr.Dataset(data_vars)

        ds = _apply_selection(ds, select)

        if kind == 'duration_curve':
            sorted_vars = {}
            for var in ds.data_vars:
                for rep in ds.coords['representation'].values:
                    values = np.sort(ds[var].sel(representation=rep).values.flatten())[::-1]
                    sorted_vars[(var, rep)] = values
            n = len(values)
            ds = xr.Dataset(
                {
                    var: xr.DataArray(
                        [sorted_vars[(var, r)] for r in ['Original', 'Clustered']],
                        dims=['representation', 'duration'],
                        coords={'representation': ['Original', 'Clustered'], 'duration': range(n)},
                    )
                    for var in resolved_variables
                }
            )

        title = (
            (
                'Original vs Clustered'
                if len(resolved_variables) > 1
                else f'Original vs Clustered: {resolved_variables[0]}'
            )
            if kind == 'timeseries'
            else ('Duration Curve' if len(resolved_variables) > 1 else f'Duration Curve: {resolved_variables[0]}')
        )

        line_kwargs = {}
        if line_dash is not None:
            line_kwargs['line_dash'] = line_dash
            if line_dash == 'representation':
                line_kwargs['line_dash_map'] = {'Original': 'dot', 'Clustered': 'solid'}

        fig = ds.fxplot.line(
            colors=colors,
            color=color,
            title=title,
            facet_col=facet_col,
            facet_row=facet_row,
            **line_kwargs,
            **plotly_kwargs,
        )
        fig.update_yaxes(matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))

        plot_result = PlotResult(data=ds, figure=fig)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            plot_result.show()

        return plot_result

    def _get_time_varying_variables(self) -> list[str]:
        """Get list of time-varying variables from original data."""
        if self._clustering.original_data is None:
            return []
        return [
            name
            for name in self._clustering.original_data.data_vars
            if 'time' in self._clustering.original_data[name].dims
            and not np.isclose(
                self._clustering.original_data[name].min(),
                self._clustering.original_data[name].max(),
            )
        ]

    def _resolve_variables(self, variables: str | list[str] | None) -> list[str]:
        """Resolve variables parameter to a list of valid variable names."""
        time_vars = self._get_time_varying_variables()
        if not time_vars:
            raise ValueError('No time-varying variables found')

        if variables is None:
            return time_vars
        elif isinstance(variables, str):
            if variables not in time_vars:
                raise ValueError(f"Variable '{variables}' not found. Available: {time_vars}")
            return [variables]
        else:
            invalid = [v for v in variables if v not in time_vars]
            if invalid:
                raise ValueError(f'Variables {invalid} not found. Available: {time_vars}')
            return list(variables)

    def heatmap(
        self,
        *,
        select: SelectType | None = None,
        colors: str | list[str] | None = None,
        facet_col: str | None = 'auto',
        animation_frame: str | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot cluster assignments over time as a heatmap timeline."""
        from ..config import CONFIG
        from ..plot_result import PlotResult
        from ..statistics_accessor import _apply_selection

        clustering = self._clustering
        cluster_order = clustering.cluster_order
        timesteps_per_cluster = clustering.timesteps_per_cluster
        original_time = clustering.original_timesteps

        if select:
            cluster_order = _apply_selection(cluster_order.to_dataset(name='cluster'), select)['cluster']

        # Expand cluster_order to per-timestep
        extra_dims = [d for d in cluster_order.dims if d != 'original_cluster']
        expanded_values = np.repeat(cluster_order.values, timesteps_per_cluster, axis=0)

        coords = {'time': original_time}
        coords.update({d: cluster_order.coords[d].values for d in extra_dims})
        cluster_da = xr.DataArray(expanded_values, dims=['time'] + extra_dims, coords=coords)

        heatmap_da = cluster_da.expand_dims('y', axis=-1).assign_coords(y=['Cluster'])
        heatmap_da.name = 'cluster_assignment'
        heatmap_da = heatmap_da.transpose('time', 'y', ...)

        fig = heatmap_da.fxplot.heatmap(
            colors=colors,
            title='Cluster Assignments',
            facet_col=facet_col,
            animation_frame=animation_frame,
            aspect='auto',
            **plotly_kwargs,
        )

        fig.update_yaxes(showticklabels=False)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))

        cluster_da.name = 'cluster'
        data = xr.Dataset({'cluster': cluster_da})
        plot_result = PlotResult(data=data, figure=fig)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            plot_result.show()

        return plot_result

    def clusters(
        self,
        variables: str | list[str] | None = None,
        *,
        select: SelectType | None = None,
        colors: ColorType | None = None,
        color: str | None = 'auto',
        facet_col: str | None = 'cluster',
        facet_cols: int | None = None,
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot each cluster's typical period profile."""
        from ..config import CONFIG
        from ..plot_result import PlotResult
        from ..statistics_accessor import _apply_selection

        clustering = self._clustering
        if clustering.aggregated_data is None:
            raise ValueError('No aggregated data available')

        aggregated_data = _apply_selection(clustering.aggregated_data, select)
        resolved_variables = self._resolve_variables(variables)

        n_clusters = clustering.n_clusters
        timesteps_per_cluster = clustering.timesteps_per_cluster
        cluster_occurrences = clustering.cluster_occurrences

        # Build cluster labels
        occ_extra_dims = [d for d in cluster_occurrences.dims if d != 'cluster']
        if occ_extra_dims:
            cluster_labels = [f'Cluster {c}' for c in range(n_clusters)]
        else:
            cluster_labels = [
                f'Cluster {c} (×{int(cluster_occurrences.sel(cluster=c).values)})' for c in range(n_clusters)
            ]

        data_vars = {}
        for var in resolved_variables:
            da = aggregated_data[var]
            if 'cluster' in da.dims:
                data_by_cluster = da.values
            else:
                data_by_cluster = da.values.reshape(n_clusters, timesteps_per_cluster)
            data_vars[var] = xr.DataArray(
                data_by_cluster,
                dims=['cluster', 'time'],
                coords={'cluster': cluster_labels, 'time': range(timesteps_per_cluster)},
            )

        ds = xr.Dataset(data_vars)
        title = 'Clusters' if len(resolved_variables) > 1 else f'Clusters: {resolved_variables[0]}'

        fig = ds.fxplot.line(
            colors=colors,
            color=color,
            title=title,
            facet_col=facet_col,
            facet_cols=facet_cols,
            **plotly_kwargs,
        )
        fig.update_yaxes(matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))

        data_vars['occurrences'] = cluster_occurrences
        result_data = xr.Dataset(data_vars)
        plot_result = PlotResult(data=result_data, figure=fig)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            plot_result.show()

        return plot_result


# Backwards compatibility - keep these names for existing code
# TODO: Remove after migration
ClusteringResultCollection = Clustering  # Alias for backwards compat


def _register_clustering_classes():
    """Register clustering classes for IO."""
    from ..structure import CLASS_REGISTRY

    CLASS_REGISTRY['Clustering'] = Clustering
