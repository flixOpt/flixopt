import numpy as np
import pytest
import xarray as xr

from flixopt.modeling import ModelingUtilities


class TestComputeConsecutiveDuration:
    """Tests for the compute_consecutive_hours_in_state static method."""

    @pytest.mark.parametrize(
        'binary_values, hours_per_timestep, expected',
        [
            # Case 1: Single timestep DataArrays
            (xr.DataArray([1], dims=['time']), 5, 5),
            (xr.DataArray([0], dims=['time']), 3, 0),
            # Case 2: Array binary, scalar hours
            (xr.DataArray([0, 0, 1, 1, 1, 0], dims=['time']), 2, 0),
            (xr.DataArray([0, 1, 1, 0, 1, 1], dims=['time']), 1, 2),
            (xr.DataArray([1, 1, 1], dims=['time']), 2, 6),
            # Case 3: Edge cases
            (xr.DataArray([1], dims=['time']), 4, 4),
            (xr.DataArray([0], dims=['time']), 3, 0),
            # Case 4: More complex patterns
            (xr.DataArray([1, 0, 0, 1, 1, 1], dims=['time']), 2, 6),  # 3 consecutive at end * 2 hours
            (xr.DataArray([0, 1, 1, 1, 0, 0], dims=['time']), 1, 0),  # ends with 0
        ],
    )
    def test_compute_duration(self, binary_values, hours_per_timestep, expected):
        """Test compute_consecutive_hours_in_state with various inputs."""
        result = ModelingUtilities.compute_consecutive_hours_in_state(binary_values, hours_per_timestep)
        assert np.isclose(result, expected)

    @pytest.mark.parametrize(
        'binary_values, hours_per_timestep',
        [
            # Case: hours_per_timestep must be scalar
            (xr.DataArray([1, 1, 1, 1, 1], dims=['time']), np.array([1, 2])),
        ],
    )
    def test_compute_duration_raises_error(self, binary_values, hours_per_timestep):
        """Test error conditions."""
        with pytest.raises(TypeError):
            ModelingUtilities.compute_consecutive_hours_in_state(binary_values, hours_per_timestep)


class TestComputePreviousOnStates:
    """Tests for the compute_previous_states static method."""

    @pytest.mark.parametrize(
        'previous_values, expected',
        [
            # Case 1: Single value DataArrays
            (xr.DataArray([0], dims=['time']), xr.DataArray([0], dims=['time'])),
            (xr.DataArray([1], dims=['time']), xr.DataArray([1], dims=['time'])),
            (xr.DataArray([0.001], dims=['time']), xr.DataArray([1], dims=['time'])),  # Using default epsilon
            (xr.DataArray([1e-4], dims=['time']), xr.DataArray([1], dims=['time'])),
            (xr.DataArray([1e-8], dims=['time']), xr.DataArray([0], dims=['time'])),
            # Case 1: Multiple timestep DataArrays
            (xr.DataArray([0, 5, 0], dims=['time']), xr.DataArray([0, 1, 0], dims=['time'])),
            (xr.DataArray([0.1, 0, 0.3], dims=['time']), xr.DataArray([1, 0, 1], dims=['time'])),
            (xr.DataArray([0, 0, 0], dims=['time']), xr.DataArray([0, 0, 0], dims=['time'])),
            (xr.DataArray([0.1, 0, 0.2], dims=['time']), xr.DataArray([1, 0, 1], dims=['time'])),
        ],
    )
    def test_compute_previous_on_states(self, previous_values, expected):
        """Test compute_previous_states with various inputs."""
        result = ModelingUtilities.compute_previous_states(previous_values)
        xr.testing.assert_equal(result, expected)

    @pytest.mark.parametrize(
        'previous_values, epsilon, expected',
        [
            # Testing with different epsilon values
            (xr.DataArray([1e-6, 1e-4, 1e-2], dims=['time']), 1e-3, xr.DataArray([0, 0, 1], dims=['time'])),
            (xr.DataArray([1e-6, 1e-4, 1e-2], dims=['time']), 1e-5, xr.DataArray([0, 1, 1], dims=['time'])),
            (xr.DataArray([1e-6, 1e-4, 1e-2], dims=['time']), 1e-1, xr.DataArray([0, 0, 0], dims=['time'])),
            # Mixed case with custom epsilon
            (xr.DataArray([0.05, 0.005, 0.0005], dims=['time']), 0.01, xr.DataArray([1, 0, 0], dims=['time'])),
        ],
    )
    def test_compute_previous_on_states_with_epsilon(self, previous_values, epsilon, expected):
        """Test compute_previous_states with custom epsilon values."""
        result = ModelingUtilities.compute_previous_states(previous_values, epsilon)
        xr.testing.assert_equal(result, expected)

    @pytest.mark.parametrize(
        'previous_values, expected_shape',
        [
            # Check that output shapes match expected dimensions
            (xr.DataArray([0, 1, 0, 1], dims=['time']), (4,)),
            (xr.DataArray([0, 1], dims=['time']), (2,)),
            (xr.DataArray([1, 0], dims=['time']), (2,)),
        ],
    )
    def test_output_shapes(self, previous_values, expected_shape):
        """Test that output array has the correct shape."""
        result = ModelingUtilities.compute_previous_states(previous_values)
        assert result.shape == expected_shape
