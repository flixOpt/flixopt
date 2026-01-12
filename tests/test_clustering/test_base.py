"""Tests for flixopt.clustering.base module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt.clustering import Clustering


class TestClustering:
    """Tests for Clustering dataclass."""

    @pytest.fixture
    def mock_aggregation_result(self):
        """Create a mock AggregationResult-like object for testing."""

        class MockClustering:
            period_duration = 24

        class MockAccuracy:
            rmse = {'col1': 0.1, 'col2': 0.2}
            mae = {'col1': 0.05, 'col2': 0.1}
            rmse_duration = {'col1': 0.15, 'col2': 0.25}

        class MockAggregationResult:
            n_clusters = 3
            n_timesteps_per_period = 24
            cluster_weights = {0: 2, 1: 3, 2: 1}
            cluster_assignments = np.array([0, 1, 0, 1, 2, 0])
            cluster_representatives = pd.DataFrame(
                {
                    'col1': np.arange(72),  # 3 clusters * 24 timesteps
                    'col2': np.arange(72) * 2,
                }
            )
            clustering = MockClustering()
            accuracy = MockAccuracy()

        return MockAggregationResult()

    @pytest.fixture
    def basic_clustering(self, mock_aggregation_result):
        """Create a basic Clustering instance for testing."""
        cluster_order = xr.DataArray([0, 1, 0, 1, 2, 0], dims=['original_cluster'])
        original_timesteps = pd.date_range('2024-01-01', periods=144, freq='h')

        return Clustering(
            tsam_results={(): mock_aggregation_result},
            dim_names=[],
            original_timesteps=original_timesteps,
            cluster_order=cluster_order,
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
        assert occurrences.sel(cluster=0).item() == 2
        assert occurrences.sel(cluster=1).item() == 3
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
        """Test metrics property."""
        metrics = basic_clustering.metrics
        assert isinstance(metrics, xr.Dataset)
        # Should have RMSE, MAE, RMSE_duration
        assert 'RMSE' in metrics.data_vars
        assert 'MAE' in metrics.data_vars
        assert 'RMSE_duration' in metrics.data_vars

    def test_cluster_start_positions(self, basic_clustering):
        """Test cluster_start_positions property."""
        positions = basic_clustering.cluster_start_positions
        np.testing.assert_array_equal(positions, [0, 24, 48])

    def test_empty_tsam_results_raises(self):
        """Test that empty tsam_results raises ValueError."""
        cluster_order = xr.DataArray([0, 1], dims=['original_cluster'])
        original_timesteps = pd.date_range('2024-01-01', periods=48, freq='h')

        with pytest.raises(ValueError, match='cannot be empty'):
            Clustering(
                tsam_results={},
                dim_names=[],
                original_timesteps=original_timesteps,
                cluster_order=cluster_order,
            )

    def test_repr(self, basic_clustering):
        """Test string representation."""
        repr_str = repr(basic_clustering)
        assert 'Clustering' in repr_str
        assert '6 periods' in repr_str
        assert '3 clusters' in repr_str


class TestClusteringMultiDim:
    """Tests for Clustering with period/scenario dimensions."""

    @pytest.fixture
    def mock_aggregation_result_factory(self):
        """Factory for creating mock AggregationResult-like objects."""

        def create_result(cluster_weights, cluster_assignments):
            class MockClustering:
                period_duration = 24

            class MockAccuracy:
                rmse = {'col1': 0.1}
                mae = {'col1': 0.05}
                rmse_duration = {'col1': 0.15}

            class MockAggregationResult:
                n_clusters = 2
                n_timesteps_per_period = 24

            result = MockAggregationResult()
            result.cluster_weights = cluster_weights
            result.cluster_assignments = cluster_assignments
            result.cluster_representatives = pd.DataFrame(
                {
                    'col1': np.arange(48),  # 2 clusters * 24 timesteps
                }
            )
            result.clustering = MockClustering()
            result.accuracy = MockAccuracy()
            return result

        return create_result

    def test_multi_period_clustering(self, mock_aggregation_result_factory):
        """Test Clustering with multiple periods."""
        result_2020 = mock_aggregation_result_factory({0: 2, 1: 1}, np.array([0, 1, 0]))
        result_2030 = mock_aggregation_result_factory({0: 1, 1: 2}, np.array([1, 0, 1]))

        cluster_order = xr.DataArray(
            [[0, 1, 0], [1, 0, 1]],
            dims=['period', 'original_cluster'],
            coords={'period': [2020, 2030]},
        )
        original_timesteps = pd.date_range('2024-01-01', periods=72, freq='h')

        clustering = Clustering(
            tsam_results={(2020,): result_2020, (2030,): result_2030},
            dim_names=['period'],
            original_timesteps=original_timesteps,
            cluster_order=cluster_order,
        )

        assert clustering.n_clusters == 2
        assert 'period' in clustering.cluster_occurrences.dims

    def test_get_result(self, mock_aggregation_result_factory):
        """Test get_result method."""
        result = mock_aggregation_result_factory({0: 2, 1: 1}, np.array([0, 1, 0]))

        cluster_order = xr.DataArray([0, 1, 0], dims=['original_cluster'])
        original_timesteps = pd.date_range('2024-01-01', periods=72, freq='h')

        clustering = Clustering(
            tsam_results={(): result},
            dim_names=[],
            original_timesteps=original_timesteps,
            cluster_order=cluster_order,
        )

        retrieved = clustering.get_result()
        assert retrieved is result

    def test_get_result_invalid_key(self, mock_aggregation_result_factory):
        """Test get_result with invalid key raises KeyError."""
        result = mock_aggregation_result_factory({0: 2, 1: 1}, np.array([0, 1, 0]))

        cluster_order = xr.DataArray([0, 1, 0], dims=['original_cluster'])
        original_timesteps = pd.date_range('2024-01-01', periods=72, freq='h')

        clustering = Clustering(
            tsam_results={(2020,): result},
            dim_names=['period'],
            original_timesteps=original_timesteps,
            cluster_order=cluster_order,
        )

        with pytest.raises(KeyError):
            clustering.get_result(period=2030)


class TestClusteringPlotAccessor:
    """Tests for ClusteringPlotAccessor."""

    @pytest.fixture
    def clustering_with_data(self):
        """Create Clustering with original and aggregated data."""

        class MockClustering:
            period_duration = 24

        class MockAccuracy:
            rmse = {'col1': 0.1}
            mae = {'col1': 0.05}
            rmse_duration = {'col1': 0.15}

        class MockAggregationResult:
            n_clusters = 2
            n_timesteps_per_period = 24
            cluster_weights = {0: 2, 1: 1}
            cluster_assignments = np.array([0, 1, 0])
            cluster_representatives = pd.DataFrame(
                {
                    'col1': np.arange(48),  # 2 clusters * 24 timesteps
                }
            )
            clustering = MockClustering()
            accuracy = MockAccuracy()

        result = MockAggregationResult()
        cluster_order = xr.DataArray([0, 1, 0], dims=['original_cluster'])
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
            tsam_results={(): result},
            dim_names=[],
            original_timesteps=original_timesteps,
            cluster_order=cluster_order,
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

        class MockClustering:
            period_duration = 24

        class MockAccuracy:
            rmse = {}
            mae = {}
            rmse_duration = {}

        class MockAggregationResult:
            n_clusters = 2
            n_timesteps_per_period = 24
            cluster_weights = {0: 1, 1: 1}
            cluster_assignments = np.array([0, 1])
            cluster_representatives = pd.DataFrame({'col1': [1, 2]})
            clustering = MockClustering()
            accuracy = MockAccuracy()

        result = MockAggregationResult()
        cluster_order = xr.DataArray([0, 1], dims=['original_cluster'])
        original_timesteps = pd.date_range('2024-01-01', periods=48, freq='h')

        clustering = Clustering(
            tsam_results={(): result},
            dim_names=[],
            original_timesteps=original_timesteps,
            cluster_order=cluster_order,
        )

        with pytest.raises(ValueError, match='No original/aggregated data'):
            clustering.plot.compare()
