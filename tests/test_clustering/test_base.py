"""Tests for flixopt.clustering.base module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt.clustering import Clustering

tsam_xarray = pytest.importorskip('tsam_xarray')


def _make_clustering_result(clusterings: dict, dim_names: list[str]):
    """Create a ClusteringResult from a dict of tsam ClusteringResult-like objects."""
    return tsam_xarray.ClusteringResult(
        time_dim='time',
        cluster_dim=['variable'],
        slice_dims=dim_names,
        clusterings=clusterings,
    )


def _make_clustering(clusterings: dict, dim_names: list[str], n_timesteps: int | None = None):
    """Create a Clustering from mock ClusteringResult objects."""
    cr_result = _make_clustering_result(clusterings, dim_names)
    first = next(iter(clusterings.values()))
    if n_timesteps is None:
        n_timesteps = first.n_original_periods * first.n_timesteps_per_period
    original_timesteps = pd.date_range('2024-01-01', periods=n_timesteps, freq='h')
    return Clustering(clustering_result=cr_result, original_timesteps=original_timesteps)


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

    def test_cluster_occurrences(self, basic_clustering):
        """Test cluster_occurrences property returns correct values."""
        occurrences = basic_clustering.cluster_occurrences
        assert isinstance(occurrences, xr.DataArray)
        assert 'cluster' in occurrences.dims
        # cluster 0: 3 occurrences, cluster 1: 2 occurrences, cluster 2: 1 occurrence
        assert occurrences.sel(cluster=0).item() == 3
        assert occurrences.sel(cluster=1).item() == 2
        assert occurrences.sel(cluster=2).item() == 1

    def test_repr(self, basic_clustering):
        """Test string representation."""
        repr_str = repr(basic_clustering)
        assert 'Clustering' in repr_str
        assert '6 periods' in repr_str
        assert '3 clusters' in repr_str

    def test_dim_names_no_extra(self, basic_clustering):
        """Test dim_names with no extra dimensions."""
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
        assert clustering.dim_names == ['period']
