import numpy as np
import pytest

from flixopt.features import ConsecutiveStateModel, StateModel


class TestComputeConsecutiveDuration:
    """Tests for the compute_consecutive_duration static method."""

    @pytest.mark.parametrize("binary_values, hours_per_timestep, expected", [
        # Case 1: Both scalar inputs
        (1, 5, 5),
        (0, 3, 0),

        # Case 2: Scalar binary, array hours
        (1, np.array([1, 2, 3]), 3),
        (0, np.array([2, 4, 6]), 0),

        # Case 3: Array binary, scalar hours
        (np.array([0, 0, 1, 1, 1, 0]), 2, 0),
        (np.array([0, 1, 1, 0, 1, 1]), 1, 2),
        (np.array([1, 1, 1]), 2, 6),

        # Case 4: Both array inputs
        (np.array([0, 1, 1, 0, 1, 1]), np.array([1, 2, 3, 4, 5, 6]), 11),  # 5+6
        (np.array([1, 0, 0, 1, 1, 1]), np.array([2, 2, 2, 3, 4, 5]), 12),  # 3+4+5

        # Case 5: Edge cases
        (np.array([1]), np.array([4]), 4),
        (np.array([0]), np.array([3]), 0),
    ])
    def test_compute_duration(self, binary_values, hours_per_timestep, expected):
        """Test compute_consecutive_duration with various inputs."""
        result = ConsecutiveStateModel.compute_consecutive_hours_in_state(binary_values, hours_per_timestep)
        assert np.isclose(result, expected)

    @pytest.mark.parametrize("binary_values, hours_per_timestep", [
        # Case: Incompatible array lengths
        (np.array([1, 1, 1, 1, 1]), np.array([1, 2])),
    ])
    def test_compute_duration_raises_error(self, binary_values, hours_per_timestep):
        """Test error conditions."""
        with pytest.raises(TypeError):
            ConsecutiveStateModel.compute_consecutive_hours_in_state(binary_values, hours_per_timestep)


class TestComputePreviousOnStates:
    """Tests for the compute_previous_on_states static method."""

    @pytest.mark.parametrize(
        'previous_values, expected',
        [
            # Case 1: Empty list
            ([], np.array([0])),

            # Case 2: All None values
            ([None, None], np.array([0])),

            # Case 3: Single value arrays
            ([np.array([0])], np.array([0])),
            ([np.array([1])], np.array([1])),
            ([np.array([0.001])], np.array([1])),  # Using default epsilon
            ([np.array([1e-4])], np.array([1])),
            ([np.array([1e-8])], np.array([0])),

            # Case 4: Multiple 1D arrays
            ([np.array([0, 5, 0]), np.array([0, 0, 1])], np.array([0, 1, 1])),
            ([np.array([0.1, 0, 0.3]), None, np.array([0, 0, 0])], np.array([1, 0, 1])),
            ([np.array([0, 0, 0]), np.array([0, 1, 0])], np.array([0, 1, 0])),
            ([np.array([0.1, 0, 0]), np.array([0, 0, 0.2])], np.array([1, 0, 1])),

            # Case 6: Mix of None, 1D and 2D arrays
            ([None, np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 0])], np.array([0, 1, 0])),
            ([np.array([0, 0, 0]), None, np.array([0, 0, 0]), np.array([0, 0, 0])], np.array([0, 0, 0])),
        ],
    )
    def test_compute_previous_on_states(self, previous_values, expected):
        """Test compute_previous_on_states with various inputs."""
        result = StateModel.compute_previous_states(previous_values)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("previous_values, epsilon, expected", [
        # Testing with different epsilon values
        ([np.array([1e-6, 1e-4, 1e-2])], 1e-3, np.array([0, 0, 1])),
        ([np.array([1e-6, 1e-4, 1e-2])], 1e-5, np.array([0, 1, 1])),
        ([np.array([1e-6, 1e-4, 1e-2])], 1e-1, np.array([0, 0, 0])),

        # Mixed case with custom epsilon
        ([np.array([0.05, 0.005, 0.0005])], 0.01, np.array([1, 0, 0])),
    ])
    def test_compute_previous_on_states_with_epsilon(self, previous_values, epsilon, expected):
        """Test compute_previous_on_states with custom epsilon values."""
        result = StateModel.compute_previous_states(previous_values, epsilon)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("previous_values, expected_shape", [
        # Check that output shapes match expected dimensions
        ([np.array([0, 1, 0, 1])], (4,)),
        ([np.array([0, 1]), np.array([1, 0]), np.array([0, 0])], (2,)),
        ([np.array([0, 1]), np.array([1, 0])], (2,)),
    ])
    def test_output_shapes(self, previous_values, expected_shape):
        """Test that output array has the correct shape."""
        result = StateModel.compute_previous_states(previous_values)
        assert result.shape == expected_shape
