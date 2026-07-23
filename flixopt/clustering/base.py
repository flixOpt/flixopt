"""
Clustering classes for time series aggregation.

This module provides the `Clustering` class stored on FlowSystem after clustering,
wrapping tsam_xarray's ClusteringResult for structure access and AggregationResult
for full data access (pre-serialization only).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr
    from tsam_xarray import AccuracyMetrics as TsamXarrayAccuracyMetrics
    from tsam_xarray import AggregationResult as TsamXarrayAggregationResult
    from tsam_xarray import ClusteringResult


class Clustering:
    """Clustering information for a FlowSystem.

    Wraps tsam_xarray's ClusteringResult for structure access and optionally
    AggregationResult for full data access (pre-serialization only).

    For advanced access to clustering structure (dims, coords, cluster_centers,
    segment_centers, etc.), use ``clustering_result`` directly.

    Example:
        >>> clustering = fs_clustered.clustering
        >>> clustering.n_clusters
        8
        >>> clustering.clustering_result  # tsam_xarray ClusteringResult for full access
    """

    def __init__(
        self,
        clustering_result: ClusteringResult | dict | None = None,
        original_timesteps: pd.DatetimeIndex | list[str] | None = None,
        # Internal: tsam_xarray AggregationResult for full data access
        _aggregation_result: TsamXarrayAggregationResult | None = None,
        # Internal: mapping from renamed dims back to originals (e.g., _period -> period)
        _unrename_map: dict[str, str] | None = None,
    ):
        # Handle ISO timestamp strings from serialization
        if (
            isinstance(original_timesteps, list)
            and len(original_timesteps) > 0
            and isinstance(original_timesteps[0], str)
        ):
            original_timesteps = pd.DatetimeIndex([pd.Timestamp(ts) for ts in original_timesteps])

        # Store tsam_xarray AggregationResult if provided (full data access)
        self._aggregation_result = _aggregation_result

        # Resolve ClusteringResult from various sources
        if clustering_result is not None:
            if isinstance(clustering_result, dict):
                self._clustering_result = self._clustering_result_from_dict(clustering_result)
            else:
                self._clustering_result = clustering_result
        elif _aggregation_result is not None:
            self._clustering_result = _aggregation_result.clustering
        else:
            raise ValueError('Either clustering_result or _aggregation_result must be provided')

        # Resolve unrename_map: if not explicitly provided, infer from slice_dims
        # (e.g., '_period' in slice_dims → {'_period': 'period'})
        if _unrename_map:
            self._unrename_map = _unrename_map
        else:
            known_renames = {'_period': 'period', '_cluster': 'cluster'}
            self._unrename_map = {k: v for k, v in known_renames.items() if k in self._clustering_result.slice_dims}

        # Flag indicating this was loaded from serialization (missing full AggregationResult data)
        self._from_serialization = _aggregation_result is None

        self.original_timesteps = original_timesteps if original_timesteps is not None else pd.DatetimeIndex([])

    @staticmethod
    def _clustering_result_from_dict(d: dict) -> ClusteringResult:
        """Create ClusteringResult from serialized dict."""
        from tsam_xarray import ClusteringResult as ClusteringResultClass

        return ClusteringResultClass.from_dict(d)

    # ==========================================================================
    # Helper for dim unrenaming
    # ==========================================================================

    def _unrename(self, da: xr.DataArray) -> xr.DataArray:
        """Rename tsam_xarray output dims back to original names (e.g., _period -> period)."""
        if not self._unrename_map:
            return da
        renames = {k: v for k, v in self._unrename_map.items() if k in da.dims}
        return da.rename(renames) if renames else da

    # ==========================================================================
    # Core properties (delegated to ClusteringResult)
    # ==========================================================================

    @property
    def clustering_result(self) -> ClusteringResult:
        """tsam_xarray ClusteringResult for reuse with apply_clustering()."""
        return self._clustering_result

    @property
    def n_clusters(self) -> int:
        """Number of clusters (typical periods)."""
        return self._clustering_result.n_clusters

    @property
    def timesteps_per_cluster(self) -> int:
        """Number of timesteps in each cluster."""
        return self._clustering_result.n_timesteps_per_period

    @property
    def n_original_clusters(self) -> int:
        """Number of original periods (before clustering)."""
        return self._clustering_result.n_original_periods

    @property
    def n_segments(self) -> int | None:
        """Number of segments per cluster, or None if not segmented."""
        return self._clustering_result.n_segments

    @property
    def is_segmented(self) -> bool:
        """Whether intra-period segmentation was used."""
        return self._clustering_result.n_segments is not None

    @property
    def dim_names(self) -> list[str]:
        """Names of extra dimensions, e.g., ['period', 'scenario']."""
        return [self._unrename_map.get(d, d) for d in self._clustering_result.slice_dims]

    # ==========================================================================
    # DataArray properties (delegated to ClusteringResult with unrename)
    # ==========================================================================

    @property
    def cluster_assignments(self) -> xr.DataArray:
        """Mapping from original periods to cluster IDs.

        Returns:
            DataArray with dims [original_cluster, period?, scenario?].
        """
        da = self._clustering_result.cluster_assignments
        # Rename tsam_xarray's 'period' dim to our 'original_cluster' convention
        # (must happen before _unrename to avoid conflict with _period → period rename)
        if 'period' in da.dims:
            da = da.rename({'period': 'original_cluster'})
        da = self._unrename(da)
        # Ensure original_cluster is first dim (tsam_xarray puts slice dims first)
        if 'original_cluster' in da.dims and da.dims[0] != 'original_cluster':
            other_dims = [d for d in da.dims if d != 'original_cluster']
            da = da.transpose('original_cluster', *other_dims)
        return da

    @property
    def cluster_occurrences(self) -> xr.DataArray:
        """How many original clusters map to each typical cluster.

        Returns:
            DataArray with dims [cluster, period?, scenario?].
        """
        return self._unrename(self._clustering_result.cluster_occurrences)

    @property
    def segment_assignments(self) -> xr.DataArray | None:
        """For each timestep within a cluster, which segment it belongs to.

        Returns:
            DataArray with dims [cluster, time, period?, scenario?], or None if not segmented.
        """
        result = self._clustering_result.segment_assignments
        if result is None:
            return None
        # tsam_xarray uses 'timestep', we use 'time'
        if 'timestep' in result.dims:
            result = result.rename({'timestep': 'time'})
        return self._unrename(result)

    @property
    def segment_durations(self) -> xr.DataArray | None:
        """Duration of each segment in timesteps.

        Returns:
            DataArray with dims [cluster, segment, period?, scenario?], or None if not segmented.
        """
        result = self._clustering_result.segment_durations
        if result is None:
            return None
        # tsam_xarray uses 'timestep', we use 'segment'
        if 'timestep' in result.dims:
            result = result.rename({'timestep': 'segment'})
        return self._unrename(result)

    # ==========================================================================
    # Methods
    # ==========================================================================

    def disaggregate(self, data: xr.DataArray) -> xr.DataArray:
        """Expand clustered data back to original timesteps.

        Delegates to tsam_xarray's ClusteringResult.disaggregate(). Handles
        the dim rename from flixopt's ``(cluster, time)`` to tsam_xarray's
        ``(cluster, timestep)`` convention.

        For non-segmented systems, values are repeated for each timestep in the period.
        For segmented systems, values are placed at segment boundaries with NaN
        elsewhere — use ``.ffill()``, ``.interpolate_na()``, or ``.fillna()``
        on the result.

        Args:
            data: DataArray with ``(cluster, time)`` or ``(cluster, segment)`` dims.

        Returns:
            DataArray with ``time`` dim restored to original timesteps.
        """
        # Rename flixopt dim names to tsam_xarray's 'timestep' convention
        flixopt_to_tsam = {'time': 'timestep', 'segment': 'timestep'}
        renames_to_tsam = {k: v for k, v in flixopt_to_tsam.items() if k in data.dims}
        if renames_to_tsam:
            data = data.rename(renames_to_tsam)
        # Rename period/scenario dims to internal names (_period, _scenario)
        reverse_unrename = {v: k for k, v in self._unrename_map.items()}
        renames = {k: v for k, v in reverse_unrename.items() if k in data.dims}
        if renames:
            data = data.rename(renames)
        result = self._clustering_result.disaggregate(data)
        return self._unrename(result)

    def apply(
        self,
        data: xr.DataArray,
    ) -> TsamXarrayAggregationResult:
        """Apply the saved clustering to new data.

        Args:
            data: DataArray with time series data to cluster.

        Returns:
            tsam_xarray AggregationResult with the clustering applied.
        """
        return self._clustering_result.apply(data)

    # ==========================================================================
    # Serialization
    # ==========================================================================

    def to_json(self, path: str | Path) -> None:
        """Save the clustering for reuse.

        Can be loaded later with Clustering.from_json() and used with
        flow_system.transform.apply_clustering().

        Args:
            path: Path to save the JSON file.
        """
        data = {
            'clustering_result': self._clustering_result.to_dict(),
            'original_timesteps': [ts.isoformat() for ts in self.original_timesteps],
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        original_timesteps: pd.DatetimeIndex | None = None,
    ) -> Clustering:
        """Load a clustering from JSON.

        The loaded Clustering has full apply() and disaggregate() support
        because ClusteringResult is fully preserved via serialization.

        Args:
            path: Path to the JSON file.
            original_timesteps: Original timesteps for the new FlowSystem.
                If None, uses the timesteps stored in the JSON.

        Returns:
            A Clustering that can be used with apply_clustering().
        """
        with open(path) as f:
            data = json.load(f)

        if 'clustering_result' not in data:
            raise ValueError('JSON file must contain "clustering_result" key')

        if original_timesteps is None:
            original_timesteps = pd.DatetimeIndex([pd.Timestamp(ts) for ts in data['original_timesteps']])

        return cls(
            clustering_result=data['clustering_result'],
            original_timesteps=original_timesteps,
        )

    def _create_reference_structure(self) -> tuple[dict, dict[str, xr.DataArray]]:
        """Create serialization structure for to_dataset().

        Returns:
            Tuple of (reference_dict, arrays_dict).
        """
        reference = {
            '__class__': 'Clustering',
            'clustering_result': self._clustering_result.to_dict(),
            'original_timesteps': [ts.isoformat() for ts in self.original_timesteps],
        }
        return reference, {}

    # ==========================================================================
    # Access to tsam_xarray AggregationResult
    # ==========================================================================

    @property
    def aggregation_result(self) -> TsamXarrayAggregationResult:
        """The tsam_xarray AggregationResult for full data access.

        Only available before serialization. After loading from file,
        use clustering_result for structure-only access.

        The returned object holds the **raw** tsam_xarray result, on which
        flixopt's reserved-dim renames are still applied (the period dim is
        ``_period``). For a friendlier view with the original dim names, use
        the ``original`` / ``reconstructed`` / ``residuals`` / ``accuracy``
        properties or ``compare()`` instead.

        Raises:
            ValueError: If accessed on a Clustering loaded from JSON/NetCDF.
        """
        self._require_full_data('aggregation_result')
        return self._aggregation_result

    @property
    def original(self) -> xr.DataArray:
        """Original (input) time series fed to clustering, on the original time axis.

        All time-varying inputs are stacked on a ``variable`` dim. Dims are
        ``(*dim_names, variable, time)`` — e.g. ``(period, scenario, variable, time)``.

        Only available before serialization.
        """
        self._require_full_data('original')
        return self._unrename(self._aggregation_result.original)

    @property
    def reconstructed(self) -> xr.DataArray:
        """Clustered profiles mapped back onto the original time axis.

        Same dims and shape as ``original`` (dim order aligned with it), so the
        two can be compared, subtracted, or plotted directly.

        Only available before serialization.
        """
        self._require_full_data('reconstructed')
        da = self._unrename(self._aggregation_result.reconstructed)
        return da.transpose(*self.original.dims)

    @property
    def residuals(self) -> xr.DataArray:
        """``original - reconstructed``, the per-timestep aggregation error.

        Only available before serialization.
        """
        self._require_full_data('residuals')
        return self._unrename(self._aggregation_result.residuals)

    @property
    def accuracy(self) -> TsamXarrayAccuracyMetrics:
        """tsam_xarray ``AccuracyMetrics`` (per-variable and column-weighted).

        Exposes ``rmse`` / ``mae`` / ``rmse_duration`` (dims ``(variable, *dim_names)``)
        and the aggregate ``weighted_rmse`` / ``weighted_mae`` / ``weighted_rmse_duration``
        (dims ``(*dim_names,)``). Dim names are un-renamed to match ``original``.

        Only available before serialization.
        """
        import dataclasses

        self._require_full_data('accuracy')
        acc = self._aggregation_result.accuracy
        return dataclasses.replace(
            acc, **{f.name: self._unrename(getattr(acc, f.name)) for f in dataclasses.fields(acc)}
        )

    def compare(self, variable: str | list[str] | None = None) -> xr.Dataset:
        """Tidy original-vs-clustered comparison, ready for plotting.

        Returns a Dataset with data_vars ``original`` and ``clustered`` on the
        original time axis and matching dim order, so ``.to_dataframe()`` and
        plotting libraries need no reshaping. This is the v7 replacement for the
        removed ``clustering.plot.compare()``.

        Args:
            variable: Optional column name (or list) to select from the
                ``variable`` dim. Available names are
                ``list(clustering.original['variable'].values)``. Defaults to
                all variables.

        Returns:
            xr.Dataset with variables ``original`` and ``clustered``.

        Examples:
            >>> import plotly.express as px
            >>> cmp = clustering.compare('HeatDemand(Q)|fixed_relative_profile')
            >>> px.line(cmp.to_dataframe()[['original', 'clustered']]).show()

        Only available before serialization.
        """
        import xarray as xr

        original = self.original
        reconstructed = self.reconstructed
        if variable is not None:
            original = original.sel(variable=variable)
            reconstructed = reconstructed.sel(variable=variable)
        return xr.Dataset({'original': original, 'clustered': reconstructed})

    def __len__(self) -> int:
        """Number of (period, scenario) combinations."""
        return len(self._clustering_result.clusterings)

    def _require_full_data(self, operation: str) -> None:
        """Raise error if full AggregationResult data is not available."""
        if self._from_serialization or self._aggregation_result is None:
            raise ValueError(
                f'{operation} requires full AggregationResult data, '
                f'but this Clustering was loaded from JSON. '
                f'Use apply_clustering() to get full results.'
            )

    def __repr__(self) -> str:
        return (
            f'Clustering(\n'
            f'  {self.n_original_clusters} periods → {self.n_clusters} clusters\n'
            f'  timesteps_per_cluster={self.timesteps_per_cluster}\n'
            f'  dims={self.dim_names}\n'
            f')'
        )


def _register_clustering_classes():
    """Register clustering classes for IO."""
    from ..structure import CLASS_REGISTRY

    CLASS_REGISTRY['Clustering'] = Clustering
