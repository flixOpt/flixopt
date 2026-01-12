"""Tests for flixopt.clustering.base module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt.clustering import Clustering, ClusteringResults
from flixopt.clustering.base import _build_timestep_mapping, _cluster_occurrences


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
            n_segments = None  # None indicates non-segmented
            segment_assignments = None  # None indicates non-segmented

            def to_dict(self):
                return {
                    'n_clusters': self.n_clusters,
                    'n_original_periods': self.n_original_periods,
                    'n_timesteps_per_period': self.n_timesteps_per_period,
                    'cluster_assignments': list(self.cluster_assignments),
                    'period_duration': self.period_duration,
                }

            def apply(self, data):
                """Mock apply method."""
                return {'applied': True}

        return MockClusteringResult()

    def test_cluster_occurrences(self, mock_clustering_result):
        """Test _cluster_occurrences helper."""
        occurrences = _cluster_occurrences(mock_clustering_result)
        # cluster 0: 3 occurrences (indices 0, 2, 5)
        # cluster 1: 2 occurrences (indices 1, 3)
        # cluster 2: 1 occurrence (index 4)
        np.testing.assert_array_equal(occurrences, [3, 2, 1])

    def test_build_timestep_mapping(self, mock_clustering_result):
        """Test _build_timestep_mapping helper."""
        mapping = _build_timestep_mapping(mock_clustering_result, n_timesteps=144)
        assert len(mapping) == 144

        # First 24 timesteps should map to cluster 0's representative (0-23)
        np.testing.assert_array_equal(mapping[:24], np.arange(24))

        # Second 24 timesteps (period 1 -> cluster 1) should map to cluster 1's representative (24-47)
        np.testing.assert_array_equal(mapping[24:48], np.arange(24, 48))


class TestClusteringResults:
    """Tests for ClusteringResults collection class."""

    @pytest.fixture
    def mock_clustering_result_factory(self):
        """Factory for creating mock ClusteringResult objects."""

        def create_result(cluster_assignments, n_timesteps_per_period=24):
            class MockClusteringResult:
                n_clusters = max(cluster_assignments) + 1 if cluster_assignments else 0
                n_original_periods = len(cluster_assignments)
                period_duration = 24.0
                n_segments = None  # None indicates non-segmented
                segment_assignments = None  # None indicates non-segmented

                def __init__(self, assignments, n_timesteps):
                    self.cluster_assignments = tuple(assignments)
                    self.n_timesteps_per_period = n_timesteps

                def to_dict(self):
                    return {
                        'n_clusters': self.n_clusters,
                        'n_original_periods': self.n_original_periods,
                        'n_timesteps_per_period': self.n_timesteps_per_period,
                        'cluster_assignments': list(self.cluster_assignments),
                        'period_duration': self.period_duration,
                    }

                def apply(self, data):
                    return {'applied': True}

            return MockClusteringResult(cluster_assignments, n_timesteps_per_period)

        return create_result

    def test_single_result(self, mock_clustering_result_factory):
        """Test ClusteringResults with single result."""
        cr = mock_clustering_result_factory([0, 1, 0])
        results = ClusteringResults({(): cr}, dim_names=[])

        assert results.n_clusters == 2
        assert results.timesteps_per_cluster == 24
        assert len(results) == 1

    def test_multi_period_results(self, mock_clustering_result_factory):
        """Test ClusteringResults with multiple periods."""
        cr_2020 = mock_clustering_result_factory([0, 1, 0])
        cr_2030 = mock_clustering_result_factory([1, 0, 1])

        results = ClusteringResults(
            {(2020,): cr_2020, (2030,): cr_2030},
            dim_names=['period'],
        )

        assert results.n_clusters == 2
        assert len(results) == 2

        # Access by period
        assert results.sel(period=2020) is cr_2020
        assert results.sel(period=2030) is cr_2030

    def test_dims_property(self, mock_clustering_result_factory):
        """Test dims property returns tuple (xarray-like)."""
        cr = mock_clustering_result_factory([0, 1, 0])
        results = ClusteringResults({(): cr}, dim_names=[])
        assert results.dims == ()

        cr_2020 = mock_clustering_result_factory([0, 1, 0])
        cr_2030 = mock_clustering_result_factory([1, 0, 1])
        results = ClusteringResults(
            {(2020,): cr_2020, (2030,): cr_2030},
            dim_names=['period'],
        )
        assert results.dims == ('period',)

    def test_coords_property(self, mock_clustering_result_factory):
        """Test coords property returns dict (xarray-like)."""
        cr_2020 = mock_clustering_result_factory([0, 1, 0])
        cr_2030 = mock_clustering_result_factory([1, 0, 1])
        results = ClusteringResults(
            {(2020,): cr_2020, (2030,): cr_2030},
            dim_names=['period'],
        )
        assert results.coords == {'period': [2020, 2030]}

    def test_sel_method(self, mock_clustering_result_factory):
        """Test sel() method (xarray-like selection)."""
        cr_2020 = mock_clustering_result_factory([0, 1, 0])
        cr_2030 = mock_clustering_result_factory([1, 0, 1])
        results = ClusteringResults(
            {(2020,): cr_2020, (2030,): cr_2030},
            dim_names=['period'],
        )
        assert results.sel(period=2020) is cr_2020
        assert results.sel(period=2030) is cr_2030

    def test_sel_invalid_key_raises(self, mock_clustering_result_factory):
        """Test sel() raises KeyError for invalid key."""
        cr = mock_clustering_result_factory([0, 1, 0])
        results = ClusteringResults({(2020,): cr}, dim_names=['period'])

        with pytest.raises(KeyError):
            results.sel(period=2030)

    def test_isel_method(self, mock_clustering_result_factory):
        """Test isel() method (xarray-like integer selection)."""
        cr_2020 = mock_clustering_result_factory([0, 1, 0])
        cr_2030 = mock_clustering_result_factory([1, 0, 1])
        results = ClusteringResults(
            {(2020,): cr_2020, (2030,): cr_2030},
            dim_names=['period'],
        )
        assert results.isel(period=0) is cr_2020
        assert results.isel(period=1) is cr_2030

    def test_isel_invalid_index_raises(self, mock_clustering_result_factory):
        """Test isel() raises IndexError for out-of-range index."""
        cr = mock_clustering_result_factory([0, 1, 0])
        results = ClusteringResults({(2020,): cr}, dim_names=['period'])

        with pytest.raises(IndexError):
            results.isel(period=5)

    def test_cluster_assignments_dataarray(self, mock_clustering_result_factory):
        """Test cluster_assignments returns correct DataArray."""
        cr = mock_clustering_result_factory([0, 1, 0])
        results = ClusteringResults({(): cr}, dim_names=[])

        cluster_assignments = results.cluster_assignments
        assert isinstance(cluster_assignments, xr.DataArray)
        assert 'original_cluster' in cluster_assignments.dims
        np.testing.assert_array_equal(cluster_assignments.values, [0, 1, 0])

    def test_cluster_occurrences_dataarray(self, mock_clustering_result_factory):
        """Test cluster_occurrences returns correct DataArray."""
        cr = mock_clustering_result_factory([0, 1, 0])  # 2 x cluster 0, 1 x cluster 1
        results = ClusteringResults({(): cr}, dim_names=[])

        occurrences = results.cluster_occurrences
        assert isinstance(occurrences, xr.DataArray)
        assert 'cluster' in occurrences.dims
        np.testing.assert_array_equal(occurrences.values, [2, 1])


class TestClustering:
    """Tests for Clustering dataclass."""

    @pytest.fixture
    def basic_cluster_results(self):
        """Create basic ClusteringResults for testing."""

        class MockClusteringResult:
            n_clusters = 3
            n_original_periods = 6
            n_timesteps_per_period = 24
            cluster_assignments = (0, 1, 0, 1, 2, 0)
            period_duration = 24.0
            n_segments = None  # None indicates non-segmented
            segment_assignments = None  # None indicates non-segmented

            def to_dict(self):
                return {
                    'n_clusters': self.n_clusters,
                    'n_original_periods': self.n_original_periods,
                    'n_timesteps_per_period': self.n_timesteps_per_period,
                    'cluster_assignments': list(self.cluster_assignments),
                    'period_duration': self.period_duration,
                }

            def apply(self, data):
                return {'applied': True}

        mock_cr = MockClusteringResult()
        return ClusteringResults({(): mock_cr}, dim_names=[])

    @pytest.fixture
    def basic_clustering(self, basic_cluster_results):
        """Create a basic Clustering instance for testing."""
        original_timesteps = pd.date_range('2024-01-01', periods=144, freq='h')

        return Clustering(
            results=basic_cluster_results,
            original_timesteps=original_timesteps,
        )

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
        # No metrics provided, so should be empty
        assert len(metrics.data_vars) == 0

    def test_cluster_start_positions(self, basic_clustering):
        """Test cluster_start_positions property."""
        positions = basic_clustering.cluster_start_positions
        np.testing.assert_array_equal(positions, [0, 24, 48])

    def test_empty_results_raises(self):
        """Test that empty results raises ValueError."""
        with pytest.raises(ValueError, match='cannot be empty'):
            ClusteringResults({}, dim_names=[])

    def test_repr(self, basic_clustering):
        """Test string representation."""
        repr_str = repr(basic_clustering)
        assert 'Clustering' in repr_str
        assert '6 periods' in repr_str
        assert '3 clusters' in repr_str


class TestClusteringMultiDim:
    """Tests for Clustering with period/scenario dimensions."""

    @pytest.fixture
    def mock_clustering_result_factory(self):
        """Factory for creating mock ClusteringResult objects."""

        def create_result(cluster_assignments, n_timesteps_per_period=24):
            class MockClusteringResult:
                n_clusters = max(cluster_assignments) + 1 if cluster_assignments else 0
                n_original_periods = len(cluster_assignments)
                period_duration = 24.0
                n_segments = None  # None indicates non-segmented
                segment_assignments = None  # None indicates non-segmented

                def __init__(self, assignments, n_timesteps):
                    self.cluster_assignments = tuple(assignments)
                    self.n_timesteps_per_period = n_timesteps

                def to_dict(self):
                    return {
                        'n_clusters': self.n_clusters,
                        'n_original_periods': self.n_original_periods,
                        'n_timesteps_per_period': self.n_timesteps_per_period,
                        'cluster_assignments': list(self.cluster_assignments),
                        'period_duration': self.period_duration,
                    }

                def apply(self, data):
                    return {'applied': True}

            return MockClusteringResult(cluster_assignments, n_timesteps_per_period)

        return create_result

    def test_multi_period_clustering(self, mock_clustering_result_factory):
        """Test Clustering with multiple periods."""
        cr_2020 = mock_clustering_result_factory([0, 1, 0])
        cr_2030 = mock_clustering_result_factory([1, 0, 1])

        results = ClusteringResults(
            {(2020,): cr_2020, (2030,): cr_2030},
            dim_names=['period'],
        )
        original_timesteps = pd.date_range('2024-01-01', periods=72, freq='h')

        clustering = Clustering(
            results=results,
            original_timesteps=original_timesteps,
        )

        assert clustering.n_clusters == 2
        assert 'period' in clustering.cluster_occurrences.dims

    def test_get_result(self, mock_clustering_result_factory):
        """Test get_result method."""
        cr = mock_clustering_result_factory([0, 1, 0])
        results = ClusteringResults({(): cr}, dim_names=[])
        original_timesteps = pd.date_range('2024-01-01', periods=72, freq='h')

        clustering = Clustering(
            results=results,
            original_timesteps=original_timesteps,
        )

        retrieved = clustering.get_result()
        assert retrieved is cr

    def test_get_result_invalid_key(self, mock_clustering_result_factory):
        """Test get_result with invalid key raises KeyError."""
        cr = mock_clustering_result_factory([0, 1, 0])
        results = ClusteringResults({(2020,): cr}, dim_names=['period'])
        original_timesteps = pd.date_range('2024-01-01', periods=72, freq='h')

        clustering = Clustering(
            results=results,
            original_timesteps=original_timesteps,
        )

        with pytest.raises(KeyError):
            clustering.get_result(period=2030)


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

            def to_dict(self):
                return {
                    'n_clusters': self.n_clusters,
                    'n_original_periods': self.n_original_periods,
                    'n_timesteps_per_period': self.n_timesteps_per_period,
                    'cluster_assignments': list(self.cluster_assignments),
                    'period_duration': self.period_duration,
                }

            def apply(self, data):
                return {'applied': True}

        mock_cr = MockClusteringResult()
        results = ClusteringResults({(): mock_cr}, dim_names=[])

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
            results=results,
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

            def to_dict(self):
                return {
                    'n_clusters': self.n_clusters,
                    'n_original_periods': self.n_original_periods,
                    'n_timesteps_per_period': self.n_timesteps_per_period,
                    'cluster_assignments': list(self.cluster_assignments),
                    'period_duration': self.period_duration,
                }

            def apply(self, data):
                return {'applied': True}

        mock_cr = MockClusteringResult()
        results = ClusteringResults({(): mock_cr}, dim_names=[])
        original_timesteps = pd.date_range('2024-01-01', periods=48, freq='h')

        clustering = Clustering(
            results=results,
            original_timesteps=original_timesteps,
        )

        with pytest.raises(ValueError, match='No original/aggregated data'):
            clustering.plot.compare()
