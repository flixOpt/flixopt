"""Tests for flixopt.clustering.base module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt.clustering import Clustering
from flixopt.clustering.base import _build_timestep_mapping

tsam_xarray = pytest.importorskip('tsam_xarray')


def _make_clustering_info(clusterings: dict, dim_names: list[str]):
    """Create a ClusteringInfo from a dict of tsam ClusteringResult-like objects."""
    return tsam_xarray.ClusteringInfo(
        time_dim='time',
        cluster_dim=['variable'],
        slice_dims=dim_names,
        clusterings=clusterings,
    )


def _make_clustering(clusterings: dict, dim_names: list[str], n_timesteps: int | None = None):
    """Create a Clustering from mock ClusteringResult objects."""
    info = _make_clustering_info(clusterings, dim_names)
    first = next(iter(clusterings.values()))
    if n_timesteps is None:
        n_timesteps = first.n_original_periods * first.n_timesteps_per_period
    original_timesteps = pd.date_range('2024-01-01', periods=n_timesteps, freq='h')
    return Clustering(clustering_info=info, original_timesteps=original_timesteps)


class TestHelperFunctions:
    """Tests for helper functions."""

    @pytest.fixture
    def mock_clustering_result(self):
        """Create a mock tsam ClusteringResult-like object."""

        class MockClusteringResult:
            n_clusters = 3
            n_original_periods = 6
            n_timesteps_per_period = 24
            cluster_assignments = (0, 1, 0, 1, 2, 0)
            period_duration = 24.0
            n_segments = None
            segment_assignments = None
            cluster_centers = (0, 1, 4)

        return MockClusteringResult()

    def test_cluster_occurrences(self, mock_clustering_result):
        """Test cluster_occurrences via Clustering."""
        clustering = _make_clustering({(): mock_clustering_result}, [])
        occurrences = clustering.cluster_occurrences
        # cluster 0: 3 occurrences (indices 0, 2, 5)
        # cluster 1: 2 occurrences (indices 1, 3)
        # cluster 2: 1 occurrence (index 4)
        np.testing.assert_array_equal(occurrences.values, [3, 2, 1])

    def test_build_timestep_mapping(self, mock_clustering_result):
        """Test _build_timestep_mapping helper."""
        mapping = _build_timestep_mapping(mock_clustering_result, n_timesteps=144)
        assert len(mapping) == 144

        # First 24 timesteps should map to cluster 0's representative (0-23)
        np.testing.assert_array_equal(mapping[:24], np.arange(24))

        # Second 24 timesteps (period 1 -> cluster 1) should map to cluster 1's representative (24-47)
        np.testing.assert_array_equal(mapping[24:48], np.arange(24, 48))


class TestClustering:
    """Tests for Clustering class."""

    @pytest.fixture
    def mock_cr(self):
        """Create a mock tsam ClusteringResult."""

        class MockClusteringResult:
            n_clusters = 3
            n_original_periods = 6
            n_timesteps_per_period = 24
            cluster_assignments = (0, 1, 0, 1, 2, 0)
            period_duration = 24.0
            n_segments = None
            segment_assignments = None
            cluster_centers = (0, 1, 4)

        return MockClusteringResult()

    @pytest.fixture
    def basic_clustering(self, mock_cr):
        """Create a basic Clustering instance for testing."""
        return _make_clustering({(): mock_cr}, [])

    def test_basic_creation(self, basic_clustering):
        """Test basic Clustering creation."""
        assert basic_clustering.n_clusters == 3
        assert basic_clustering.timesteps_per_cluster == 24
        assert basic_clustering.n_original_clusters == 6

    def test_n_representatives(self, basic_clustering):
        """Test n_representatives property."""
        assert basic_clustering.n_representatives == 72  # 3 * 24

    def test_cluster_occurrences(self, basic_clustering):
        """Test cluster_occurrences property returns correct values."""
        occurrences = basic_clustering.cluster_occurrences
        assert isinstance(occurrences, xr.DataArray)
        assert 'cluster' in occurrences.dims
        # cluster 0: 3 occurrences, cluster 1: 2 occurrences, cluster 2: 1 occurrence
        assert occurrences.sel(cluster=0).item() == 3
        assert occurrences.sel(cluster=1).item() == 2
        assert occurrences.sel(cluster=2).item() == 1

    def test_representative_weights(self, basic_clustering):
        """Test representative_weights is same as cluster_occurrences."""
        weights = basic_clustering.representative_weights
        occurrences = basic_clustering.cluster_occurrences
        xr.testing.assert_equal(
            weights.drop_vars('cluster', errors='ignore'),
            occurrences.drop_vars('cluster', errors='ignore'),
        )

    def test_timestep_mapping(self, basic_clustering):
        """Test timestep_mapping property."""
        mapping = basic_clustering.timestep_mapping
        assert isinstance(mapping, xr.DataArray)
        assert 'original_time' in mapping.dims
        assert len(mapping) == 144  # Original timesteps

    def test_metrics(self, basic_clustering):
        """Test metrics property returns empty Dataset when no metrics."""
        metrics = basic_clustering.metrics
        assert isinstance(metrics, xr.Dataset)
        assert len(metrics.data_vars) == 0

    def test_repr(self, basic_clustering):
        """Test string representation."""
        repr_str = repr(basic_clustering)
        assert 'Clustering' in repr_str
        assert '6 periods' in repr_str
        assert '3 clusters' in repr_str

    def test_dims_no_extra(self, basic_clustering):
        """Test dims/coords with no extra dimensions."""
        assert basic_clustering.dims == ()
        assert basic_clustering.coords == {}
        assert basic_clustering.dim_names == []


class TestClusteringMultiDim:
    """Tests for Clustering with period/scenario dimensions."""

    @pytest.fixture
    def mock_cr_factory(self):
        """Factory for creating mock ClusteringResult objects."""

        def create_result(cluster_assignments, n_timesteps_per_period=24):
            class MockClusteringResult:
                n_clusters = max(cluster_assignments) + 1 if cluster_assignments else 0
                n_original_periods = len(cluster_assignments)
                period_duration = 24.0
                n_segments = None
                segment_assignments = None
                cluster_centers = tuple(range(max(cluster_assignments) + 1)) if cluster_assignments else ()

                def __init__(self, assignments, n_timesteps):
                    self.cluster_assignments = tuple(assignments)
                    self.n_timesteps_per_period = n_timesteps

            return MockClusteringResult(cluster_assignments, n_timesteps_per_period)

        return create_result

    def test_multi_period_clustering(self, mock_cr_factory):
        """Test Clustering with multiple periods."""
        cr_2020 = mock_cr_factory([0, 1, 0])
        cr_2030 = mock_cr_factory([1, 0, 1])

        clustering = _make_clustering(
            {(2020,): cr_2020, (2030,): cr_2030},
            ['period'],
        )

        assert clustering.n_clusters == 2
        assert 'period' in clustering.cluster_occurrences.dims
        assert clustering.dims == ('period',)
        assert clustering.coords == {'period': [2020, 2030]}


class TestClusteringPlotAccessor:
    """Tests for ClusteringPlotAccessor."""

    @pytest.fixture
    def clustering_with_data(self):
        """Create Clustering with original and aggregated data."""

        class MockClusteringResult:
            n_clusters = 2
            n_original_periods = 3
            n_timesteps_per_period = 24
            cluster_assignments = (0, 1, 0)
            period_duration = 24.0
            n_segments = None
            segment_assignments = None
            cluster_centers = (0, 1)

        info = _make_clustering_info({(): MockClusteringResult()}, [])
        original_timesteps = pd.date_range('2024-01-01', periods=72, freq='h')

        original_data = xr.Dataset(
            {
                'col1': xr.DataArray(np.random.randn(72), dims=['time'], coords={'time': original_timesteps}),
            }
        )
        aggregated_data = xr.Dataset(
            {
                'col1': xr.DataArray(
                    np.random.randn(2, 24),
                    dims=['cluster', 'time'],
                    coords={'cluster': [0, 1], 'time': pd.date_range('2000-01-01', periods=24, freq='h')},
                ),
            }
        )

        return Clustering(
            clustering_info=info,
            original_timesteps=original_timesteps,
            original_data=original_data,
            aggregated_data=aggregated_data,
        )

    def test_plot_accessor_exists(self, clustering_with_data):
        """Test that plot accessor is available."""
        assert hasattr(clustering_with_data, 'plot')
        assert hasattr(clustering_with_data.plot, 'compare')
        assert hasattr(clustering_with_data.plot, 'heatmap')
        assert hasattr(clustering_with_data.plot, 'clusters')

    def test_compare_requires_data(self):
        """Test compare() raises when no data available."""

        class MockClusteringResult:
            n_clusters = 2
            n_original_periods = 2
            n_timesteps_per_period = 24
            cluster_assignments = (0, 1)
            period_duration = 24.0
            n_segments = None
            segment_assignments = None
            cluster_centers = (0, 1)

        info = _make_clustering_info({(): MockClusteringResult()}, [])
        original_timesteps = pd.date_range('2024-01-01', periods=48, freq='h')

        clustering = Clustering(
            clustering_info=info,
            original_timesteps=original_timesteps,
        )

        with pytest.raises(ValueError, match='No original/aggregated data'):
            clustering.plot.compare()
