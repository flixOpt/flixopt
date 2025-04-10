import numpy as np
import pytest

from flixopt.features import OnOffModel


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
        result = OnOffModel.compute_consecutive_duration(binary_values, hours_per_timestep)
        assert np.isclose(result, expected)

    @pytest.mark.parametrize("binary_values, hours_per_timestep", [
        # Case: Incompatible array lengths
        (np.array([1, 1, 1, 1, 1]), np.array([1, 2])),
    ])
    def test_compute_duration_raises_error(self, binary_values, hours_per_timestep):
        """Test error conditions."""
        with pytest.raises(TypeError):
            OnOffModel.compute_consecutive_duration(binary_values, hours_per_timestep)
