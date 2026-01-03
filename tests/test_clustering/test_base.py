"""Tests for flixopt.clustering.base module."""

import numpy as np
import pytest
import xarray as xr

from flixopt.clustering import (
    Clustering,
    ClusterResult,
    ClusterStructure,
    create_cluster_structure_from_mapping,
)


class TestClusterStructure:
    """Tests for ClusterStructure dataclass."""

    def test_basic_creation(self):
        """Test basic ClusterStructure creation."""
        cluster_order = xr.DataArray([0, 1, 0, 1, 2, 0], dims=['original_period'])
        cluster_occurrences = xr.DataArray([3, 2, 1], dims=['cluster'])

        structure = ClusterStructure(
            cluster_order=cluster_order,
            cluster_occurrences=cluster_occurrences,
            n_clusters=3,
            timesteps_per_cluster=24,
        )

        assert structure.n_clusters == 3
        assert structure.timesteps_per_cluster == 24
        assert structure.n_original_periods == 6

    def test_creation_from_numpy(self):
        """Test ClusterStructure creation from numpy arrays."""
        structure = ClusterStructure(
            cluster_order=np.array([0, 0, 1, 1, 0]),
            cluster_occurrences=np.array([3, 2]),
            n_clusters=2,
            timesteps_per_cluster=12,
        )

        assert isinstance(structure.cluster_order, xr.DataArray)
        assert isinstance(structure.cluster_occurrences, xr.DataArray)
        assert structure.n_original_periods == 5

    def test_get_cluster_weight_per_timestep(self):
        """Test weight calculation per timestep."""
        structure = ClusterStructure(
            cluster_order=xr.DataArray([0, 1, 0], dims=['original_period']),
            cluster_occurrences=xr.DataArray([2, 1], dims=['cluster']),
            n_clusters=2,
            timesteps_per_cluster=4,
        )

        weights = structure.get_cluster_weight_per_timestep()

        # Cluster 0 has 4 timesteps, each with weight 2
        # Cluster 1 has 4 timesteps, each with weight 1
        assert len(weights) == 8
        assert float(weights.isel(time=0).values) == 2.0
        assert float(weights.isel(time=4).values) == 1.0


class TestClusterResult:
    """Tests for ClusterResult dataclass."""

    def test_basic_creation(self):
        """Test basic ClusterResult creation."""
        result = ClusterResult(
            timestep_mapping=xr.DataArray([0, 0, 1, 1, 2, 2], dims=['original_time']),
            n_representatives=3,
            representative_weights=xr.DataArray([2, 2, 2], dims=['time']),
        )

        assert result.n_representatives == 3
        assert result.n_original_timesteps == 6

    def test_creation_from_numpy(self):
        """Test ClusterResult creation from numpy arrays."""
        result = ClusterResult(
            timestep_mapping=np.array([0, 1, 0, 1]),
            n_representatives=2,
            representative_weights=np.array([2.0, 2.0]),
        )

        assert isinstance(result.timestep_mapping, xr.DataArray)
        assert isinstance(result.representative_weights, xr.DataArray)

    def test_validation_success(self):
        """Test validation passes for valid result."""
        result = ClusterResult(
            timestep_mapping=xr.DataArray([0, 1, 0, 1], dims=['original_time']),
            n_representatives=2,
            representative_weights=xr.DataArray([2.0, 2.0], dims=['time']),
        )

        # Should not raise
        result.validate()

    def test_validation_invalid_mapping(self):
        """Test validation fails for out-of-range mapping."""
        result = ClusterResult(
            timestep_mapping=xr.DataArray([0, 5, 0, 1], dims=['original_time']),  # 5 is out of range
            n_representatives=2,
            representative_weights=xr.DataArray([2.0, 2.0], dims=['time']),
        )

        with pytest.raises(ValueError, match='timestep_mapping contains index'):
            result.validate()

    def test_get_expansion_mapping(self):
        """Test get_expansion_mapping returns named DataArray."""
        result = ClusterResult(
            timestep_mapping=xr.DataArray([0, 1, 0], dims=['original_time']),
            n_representatives=2,
            representative_weights=xr.DataArray([2.0, 1.0], dims=['time']),
        )

        mapping = result.get_expansion_mapping()
        assert mapping.name == 'expansion_mapping'


class TestCreateClusterStructureFromMapping:
    """Tests for create_cluster_structure_from_mapping function."""

    def test_basic_creation(self):
        """Test creating ClusterStructure from timestep mapping."""
        # 12 original timesteps, 4 per period, 3 periods
        # Mapping: period 0 -> cluster 0, period 1 -> cluster 1, period 2 -> cluster 0
        mapping = xr.DataArray(
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3],  # First and third period map to cluster 0
            dims=['original_time'],
        )

        structure = create_cluster_structure_from_mapping(mapping, timesteps_per_cluster=4)

        assert structure.timesteps_per_cluster == 4
        assert structure.n_original_periods == 3


class TestClustering:
    """Tests for Clustering dataclass."""

    def test_creation(self):
        """Test Clustering creation."""
        result = ClusterResult(
            timestep_mapping=xr.DataArray([0, 1], dims=['original_time']),
            n_representatives=2,
            representative_weights=xr.DataArray([1.0, 1.0], dims=['time']),
        )

        info = Clustering(
            result=result,
            backend_name='tsam',
        )

        assert info.backend_name == 'tsam'
