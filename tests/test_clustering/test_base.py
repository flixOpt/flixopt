"""Tests for flixopt.clustering module."""

import numpy as np
import pandas as pd
import xarray as xr

from flixopt.clustering import Clustering


class TestClustering:
    """Tests for Clustering dataclass."""

    def test_basic_creation(self):
        """Test basic Clustering creation."""
        cluster_assignments = xr.DataArray([0, 1, 0, 1, 2, 0], dims=['original_cluster'])
        cluster_weights = xr.DataArray([3, 2, 1], dims=['cluster'], coords={'cluster': [0, 1, 2]})

        clustering = Clustering(
            cluster_assignments=cluster_assignments,
            cluster_weights=cluster_weights,
            original_timesteps=pd.date_range('2024-01-01', periods=144, freq='h'),
        )

        assert clustering.n_clusters == 3
        assert clustering.timesteps_per_cluster == 24
        assert clustering.n_original_clusters == 6

    def test_cluster_order_alias(self):
        """Test cluster_order is alias for cluster_assignments."""
        cluster_assignments = xr.DataArray([0, 1, 0], dims=['original_cluster'])
        cluster_weights = xr.DataArray([2, 1], dims=['cluster'], coords={'cluster': [0, 1]})

        clustering = Clustering(
            cluster_assignments=cluster_assignments,
            cluster_weights=cluster_weights,
            original_timesteps=pd.date_range('2024-01-01', periods=72, freq='h'),
        )

        xr.testing.assert_equal(clustering.cluster_order, clustering.cluster_assignments)

    def test_cluster_occurrences_alias(self):
        """Test cluster_occurrences is alias for cluster_weights."""
        cluster_assignments = xr.DataArray([0, 1, 0], dims=['original_cluster'])
        cluster_weights = xr.DataArray([2, 1], dims=['cluster'], coords={'cluster': [0, 1]})

        clustering = Clustering(
            cluster_assignments=cluster_assignments,
            cluster_weights=cluster_weights,
            original_timesteps=pd.date_range('2024-01-01', periods=72, freq='h'),
        )

        xr.testing.assert_equal(clustering.cluster_occurrences, clustering.cluster_weights)


class TestTimestepMapping:
    """Tests for get_timestep_mapping method."""

    def test_get_timestep_mapping(self):
        """Test timestep mapping computation."""
        # 3 original clusters, 4 timesteps per cluster = 12 original timesteps
        # Cluster assignments: [0, 1, 0] - periods 0 and 2 map to cluster 0, period 1 maps to cluster 1
        cluster_assignments = xr.DataArray([0, 1, 0], dims=['original_cluster'])
        cluster_weights = xr.DataArray([2, 1], dims=['cluster'], coords={'cluster': [0, 1]})

        clustering = Clustering(
            cluster_assignments=cluster_assignments,
            cluster_weights=cluster_weights,
            original_timesteps=pd.date_range('2024-01-01', periods=12, freq='h'),
        )

        mapping = clustering.get_timestep_mapping()

        # Expected mapping:
        # t=0-3 (orig cluster 0) -> cluster 0 * 4 + [0,1,2,3] = [0,1,2,3]
        # t=4-7 (orig cluster 1) -> cluster 1 * 4 + [0,1,2,3] = [4,5,6,7]
        # t=8-11 (orig cluster 2) -> cluster 0 * 4 + [0,1,2,3] = [0,1,2,3]
        expected = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3])
        np.testing.assert_array_equal(mapping, expected)


class TestExpandData:
    """Tests for expand_data method."""

    def test_expand_1d_data(self):
        """Test expanding 1D time series data."""
        cluster_assignments = xr.DataArray([0, 1, 0], dims=['original_cluster'])
        cluster_weights = xr.DataArray([2, 1], dims=['cluster'], coords={'cluster': [0, 1]})
        original_timesteps = pd.date_range('2024-01-01', periods=12, freq='h')

        clustering = Clustering(
            cluster_assignments=cluster_assignments,
            cluster_weights=cluster_weights,
            original_timesteps=original_timesteps,
        )

        # Clustered data: 2 clusters × 4 timesteps = 8 values
        clustered_data = xr.DataArray(
            [[10, 11, 12, 13], [20, 21, 22, 23]],  # cluster 0 and cluster 1
            dims=['cluster', 'time'],
            coords={'cluster': [0, 1], 'time': pd.date_range('2000-01-01', periods=4, freq='h')},
        )

        expanded = clustering.expand_data(clustered_data)

        # Periods 0 and 2 map to cluster 0, period 1 maps to cluster 1
        expected_values = [10, 11, 12, 13, 20, 21, 22, 23, 10, 11, 12, 13]
        np.testing.assert_array_equal(expanded.values, expected_values)
        assert list(expanded.coords['time'].values) == list(original_timesteps)

    def test_expand_flat_data(self):
        """Test expanding flat (no cluster dim) time series data."""
        cluster_assignments = xr.DataArray([0, 1, 0], dims=['original_cluster'])
        cluster_weights = xr.DataArray([2, 1], dims=['cluster'], coords={'cluster': [0, 1]})
        original_timesteps = pd.date_range('2024-01-01', periods=12, freq='h')

        clustering = Clustering(
            cluster_assignments=cluster_assignments,
            cluster_weights=cluster_weights,
            original_timesteps=original_timesteps,
        )

        # Flat clustered data: 8 timesteps (2 clusters × 4)
        clustered_data = xr.DataArray(
            [10, 11, 12, 13, 20, 21, 22, 23],  # flat: cluster 0 values, then cluster 1 values
            dims=['time'],
            coords={'time': pd.date_range('2000-01-01', periods=8, freq='h')},
        )

        expanded = clustering.expand_data(clustered_data)

        # Expected: periods 0, 2 map to indices 0-3, period 1 maps to indices 4-7
        expected_values = [10, 11, 12, 13, 20, 21, 22, 23, 10, 11, 12, 13]
        np.testing.assert_array_equal(expanded.values, expected_values)


class TestIOSerialization:
    """Tests for IO serialization methods."""

    def test_to_reference(self):
        """Test to_reference method."""
        cluster_assignments = xr.DataArray([0, 1, 0], dims=['original_cluster'])
        cluster_weights = xr.DataArray([2, 1], dims=['cluster'], coords={'cluster': [0, 1]})

        clustering = Clustering(
            cluster_assignments=cluster_assignments,
            cluster_weights=cluster_weights,
            original_timesteps=pd.date_range('2024-01-01', periods=72, freq='h'),
        )

        ref, arrays = clustering.to_reference()

        assert ref['__class__'] == 'Clustering'
        assert 'cluster_assignments' in arrays
        assert 'cluster_weights' in arrays
        assert 'original_timesteps' in arrays

    def test_roundtrip_via_from_reference(self):
        """Test roundtrip serialization via to_reference/from_reference."""
        cluster_assignments = xr.DataArray([0, 1, 0], dims=['original_cluster'])
        cluster_weights = xr.DataArray([2, 1], dims=['cluster'], coords={'cluster': [0, 1]})
        original_timesteps = pd.date_range('2024-01-01', periods=72, freq='h')

        clustering = Clustering(
            cluster_assignments=cluster_assignments,
            cluster_weights=cluster_weights,
            original_timesteps=original_timesteps,
        )

        ref, arrays = clustering.to_reference()
        restored = Clustering.from_reference(ref, arrays)

        assert restored.n_clusters == clustering.n_clusters
        assert restored.timesteps_per_cluster == clustering.timesteps_per_cluster
        xr.testing.assert_equal(restored.cluster_assignments, clustering.cluster_assignments)
        xr.testing.assert_equal(restored.cluster_weights, clustering.cluster_weights)
        assert list(restored.original_timesteps) == list(clustering.original_timesteps)


class TestRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        cluster_assignments = xr.DataArray([0, 1, 0], dims=['original_cluster'])
        cluster_weights = xr.DataArray([2, 1], dims=['cluster'], coords={'cluster': [0, 1]})

        clustering = Clustering(
            cluster_assignments=cluster_assignments,
            cluster_weights=cluster_weights,
            original_timesteps=pd.date_range('2024-01-01', periods=72, freq='h'),
        )

        repr_str = repr(clustering)
        assert 'Clustering' in repr_str
        assert 'n_clusters=2' in repr_str
        assert 'timesteps_per_cluster=24' in repr_str
        assert 'n_original_clusters=3' in repr_str
