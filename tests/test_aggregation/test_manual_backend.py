"""Tests for flixopt.aggregation.manual module."""

import numpy as np
import pytest
import xarray as xr

from flixopt.aggregation import (
    ManualBackend,
    create_manual_backend_from_labels,
    create_manual_backend_from_selection,
)


class TestManualBackend:
    """Tests for ManualBackend class."""

    def test_basic_creation(self):
        """Test basic ManualBackend creation."""
        mapping = xr.DataArray([0, 1, 0, 1, 2, 2], dims=['original_time'])
        weights = xr.DataArray([2.0, 2.0, 2.0], dims=['time'])

        backend = ManualBackend(timestep_mapping=mapping, representative_weights=weights)

        assert len(backend.timestep_mapping) == 6
        assert len(backend.representative_weights) == 3

    def test_validation_dimension_mismatch(self):
        """Test validation fails for mismatched dimensions."""
        mapping = xr.DataArray([0, 1, 5], dims=['original_time'])  # 5 is out of range
        weights = xr.DataArray([2.0, 2.0], dims=['time'])  # Only 2 weights

        with pytest.raises(ValueError, match='timestep_mapping contains index'):
            ManualBackend(timestep_mapping=mapping, representative_weights=weights)

    def test_aggregate_creates_result(self):
        """Test aggregate method creates proper AggregationResult."""
        mapping = xr.DataArray([0, 1, 0, 1], dims=['original_time'])
        weights = xr.DataArray([2.0, 2.0], dims=['time'])

        backend = ManualBackend(timestep_mapping=mapping, representative_weights=weights)

        # Create test data
        data = xr.Dataset(
            {'var1': (['time'], [1.0, 2.0, 3.0, 4.0])},
            coords={'time': range(4)},
        )

        result = backend.aggregate(data)

        assert result.n_representatives == 2
        assert result.n_original_timesteps == 4
        assert result.aggregated_data is not None

    def test_aggregate_validates_data_dimensions(self):
        """Test aggregate validates data dimensions match mapping."""
        mapping = xr.DataArray([0, 1, 0], dims=['original_time'])  # 3 timesteps
        weights = xr.DataArray([2.0, 1.0], dims=['time'])

        backend = ManualBackend(timestep_mapping=mapping, representative_weights=weights)

        # Data has wrong number of timesteps
        data = xr.Dataset(
            {'var1': (['time'], [1.0, 2.0, 3.0, 4.0, 5.0])},  # 5 timesteps
            coords={'time': range(5)},
        )

        with pytest.raises(ValueError, match='timesteps'):
            backend.aggregate(data)


class TestCreateManualBackendFromLabels:
    """Tests for create_manual_backend_from_labels function."""

    def test_basic_creation(self):
        """Test creating ManualBackend from cluster labels."""
        # 3 periods of 4 timesteps each, labeled [0, 1, 0]
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])

        backend = create_manual_backend_from_labels(labels, timesteps_per_cluster=4)

        assert len(backend.representative_weights) == 8  # 2 clusters x 4 timesteps
        # Cluster 0 appears 2 times, cluster 1 appears 1 time
        assert float(backend.representative_weights.isel(time=0).values) == 2.0
        assert float(backend.representative_weights.isel(time=4).values) == 1.0

    def test_non_consecutive_labels(self):
        """Test handling of non-consecutive cluster labels."""
        # Labels are 0, 2, 0 (skipping 1)
        labels = np.array([0, 0, 2, 2, 0, 0])

        backend = create_manual_backend_from_labels(labels, timesteps_per_cluster=2)

        # Should remap to consecutive 0, 1
        assert len(backend.representative_weights) == 4  # 2 unique clusters x 2 timesteps


class TestCreateManualBackendFromSelection:
    """Tests for create_manual_backend_from_selection function."""

    def test_basic_creation(self):
        """Test creating ManualBackend from selected indices."""
        # Select every 3rd timestep from 12 original timesteps
        selected_indices = np.array([0, 3, 6, 9])
        weights = np.array([3.0, 3.0, 3.0, 3.0])

        backend = create_manual_backend_from_selection(
            selected_indices=selected_indices,
            weights=weights,
            n_original_timesteps=12,
        )

        assert len(backend.representative_weights) == 4
        # Check mapping assigns nearby timesteps to nearest representative
        mapping = backend.timestep_mapping.values
        assert mapping[0] == 0  # Timestep 0 -> representative 0 (at index 0)
        assert mapping[1] == 0  # Timestep 1 -> representative 0 (nearest to 0)
        # Timestep 5 is equidistant from indices 3 and 6, but argmin picks first
        # Actually: distances from 5 to [0,3,6,9] = [5,2,1,4], so nearest is rep 2 (at index 6)
        assert mapping[5] == 2  # Timestep 5 -> representative 2 (at index 6)

    def test_weights_length_mismatch(self):
        """Test error when weights length doesn't match selected indices."""
        selected_indices = np.array([0, 3, 6])
        weights = np.array([3.0, 3.0])  # Wrong length

        with pytest.raises(ValueError, match='weights'):
            create_manual_backend_from_selection(
                selected_indices=selected_indices,
                weights=weights,
                n_original_timesteps=12,
            )
