import pytest
import xarray as xr

from flixopt.effects import calculate_all_conversion_paths


def test_direct_conversions():
    """Test direct conversions with simple scalar values."""
    conversion_dict = {'A': {'B': xr.DataArray(2.0)}, 'B': {'C': xr.DataArray(3.0)}}

    result = calculate_all_conversion_paths(conversion_dict)

    # Check direct conversions
    assert ('A', 'B') in result
    assert ('B', 'C') in result
    assert result[('A', 'B')].item() == 2.0
    assert result[('B', 'C')].item() == 3.0

    # Check indirect conversion
    assert ('A', 'C') in result
    assert result[('A', 'C')].item() == 6.0  # 2.0 * 3.0


def test_multiple_paths():
    """Test multiple paths between nodes that should be summed."""
    conversion_dict = {
        'A': {'B': xr.DataArray(2.0), 'C': xr.DataArray(3.0)},
        'B': {'D': xr.DataArray(4.0)},
        'C': {'D': xr.DataArray(5.0)},
    }

    result = calculate_all_conversion_paths(conversion_dict)

    # A to D should sum two paths: A->B->D (2*4=8) and A->C->D (3*5=15)
    assert ('A', 'D') in result
    assert result[('A', 'D')].item() == 8.0 + 15.0


def test_xarray_conversions():
    """Test with xarray DataArrays that have dimensions."""
    # Create DataArrays with a time dimension
    time_points = [1, 2, 3]
    a_to_b = xr.DataArray([2.0, 2.1, 2.2], dims='time', coords={'time': time_points})
    b_to_c = xr.DataArray([3.0, 3.1, 3.2], dims='time', coords={'time': time_points})

    conversion_dict = {'A': {'B': a_to_b}, 'B': {'C': b_to_c}}

    result = calculate_all_conversion_paths(conversion_dict)

    # Check indirect conversion preserves dimensions
    assert ('A', 'C') in result
    assert result[('A', 'C')].dims == ('time',)

    # Check values at each time point
    for i, t in enumerate(time_points):
        expected = a_to_b.values[i] * b_to_c.values[i]
        assert pytest.approx(result[('A', 'C')].sel(time=t).item()) == expected


def test_long_paths():
    """Test with longer paths (more than one intermediate node)."""
    conversion_dict = {
        'A': {'B': xr.DataArray(2.0)},
        'B': {'C': xr.DataArray(3.0)},
        'C': {'D': xr.DataArray(4.0)},
        'D': {'E': xr.DataArray(5.0)},
    }

    result = calculate_all_conversion_paths(conversion_dict)

    # Check the full path A->B->C->D->E
    assert ('A', 'E') in result
    expected = 2.0 * 3.0 * 4.0 * 5.0  # 120.0
    assert result[('A', 'E')].item() == expected


def test_diamond_paths():
    """Test with a diamond shape graph with multiple paths to the same destination."""
    conversion_dict = {
        'A': {'B': xr.DataArray(2.0), 'C': xr.DataArray(3.0)},
        'B': {'D': xr.DataArray(4.0)},
        'C': {'D': xr.DataArray(5.0)},
        'D': {'E': xr.DataArray(6.0)},
    }

    result = calculate_all_conversion_paths(conversion_dict)

    # A to E should go through both paths:
    # A->B->D->E (2*4*6=48) and A->C->D->E (3*5*6=90)
    assert ('A', 'E') in result
    expected = 48.0 + 90.0  # 138.0
    assert result[('A', 'E')].item() == expected


def test_effect_shares_example():
    """Test the specific example from the effects share factors test."""
    # Create the conversion dictionary based on test example
    conversion_dict = {
        'costs': {'Effect1': xr.DataArray(0.5)},
        'Effect1': {'Effect2': xr.DataArray(1.1), 'Effect3': xr.DataArray(1.2)},
        'Effect2': {'Effect3': xr.DataArray(5.0)},
    }

    result = calculate_all_conversion_paths(conversion_dict)

    # Test direct paths
    assert result[('costs', 'Effect1')].item() == 0.5
    assert result[('Effect1', 'Effect2')].item() == 1.1
    assert result[('Effect2', 'Effect3')].item() == 5.0

    # Test indirect paths
    # costs -> Effect2 = costs -> Effect1 -> Effect2 = 0.5 * 1.1
    assert result[('costs', 'Effect2')].item() == 0.5 * 1.1

    # costs -> Effect3 has two paths:
    # 1. costs -> Effect1 -> Effect3 = 0.5 * 1.2 = 0.6
    # 2. costs -> Effect1 -> Effect2 -> Effect3 = 0.5 * 1.1 * 5 = 2.75
    # Total = 0.6 + 2.75 = 3.35
    assert result[('costs', 'Effect3')].item() == 0.5 * 1.2 + 0.5 * 1.1 * 5

    # Effect1 -> Effect3 has two paths:
    # 1. Effect1 -> Effect2 -> Effect3 = 1.1 * 5.0 = 5.5
    # 2. Effect1 -> Effect3 = 1.2
    # Total = 0.6 + 2.75 = 3.35
    assert result[('Effect1', 'Effect3')].item() == 1.2 + 1.1 * 5.0


def test_empty_conversion_dict():
    """Test with an empty conversion dictionary."""
    result = calculate_all_conversion_paths({})
    assert len(result) == 0


def test_no_indirect_paths():
    """Test with a dictionary that has no indirect paths."""
    conversion_dict = {'A': {'B': xr.DataArray(2.0)}, 'C': {'D': xr.DataArray(3.0)}}

    result = calculate_all_conversion_paths(conversion_dict)

    # Only direct paths should exist
    assert len(result) == 2
    assert ('A', 'B') in result
    assert ('C', 'D') in result
    assert result[('A', 'B')].item() == 2.0
    assert result[('C', 'D')].item() == 3.0


def test_complex_network():
    """Test with a complex network of many nodes and multiple paths, without circular references."""
    # Create a directed acyclic graph with many nodes
    # Structure resembles a layered network with multiple paths
    conversion_dict = {
        'A': {'B': xr.DataArray(1.5), 'C': xr.DataArray(2.0), 'D': xr.DataArray(0.5)},
        'B': {'E': xr.DataArray(3.0), 'F': xr.DataArray(1.2)},
        'C': {'E': xr.DataArray(0.8), 'G': xr.DataArray(2.5)},
        'D': {'G': xr.DataArray(1.8), 'H': xr.DataArray(3.2)},
        'E': {'I': xr.DataArray(0.7), 'J': xr.DataArray(1.4)},
        'F': {'J': xr.DataArray(2.2), 'K': xr.DataArray(0.9)},
        'G': {'K': xr.DataArray(1.6), 'L': xr.DataArray(2.8)},
        'H': {'L': xr.DataArray(0.4), 'M': xr.DataArray(1.1)},
        'I': {'N': xr.DataArray(2.3)},
        'J': {'N': xr.DataArray(1.9), 'O': xr.DataArray(0.6)},
        'K': {'O': xr.DataArray(3.5), 'P': xr.DataArray(1.3)},
        'L': {'P': xr.DataArray(2.7), 'Q': xr.DataArray(0.8)},
        'M': {'Q': xr.DataArray(2.1)},
        'N': {'R': xr.DataArray(1.7)},
        'O': {'R': xr.DataArray(2.9), 'S': xr.DataArray(1.0)},
        'P': {'S': xr.DataArray(2.4)},
        'Q': {'S': xr.DataArray(1.5)},
    }

    result = calculate_all_conversion_paths(conversion_dict)

    # Check some direct paths
    assert result[('A', 'B')].item() == 1.5
    assert result[('D', 'H')].item() == 3.2
    assert result[('G', 'L')].item() == 2.8

    # Check some two-step paths
    assert result[('A', 'E')].item() == 1.5 * 3.0 + 2.0 * 0.8  # A->B->E + A->C->E
    assert result[('B', 'J')].item() == 3.0 * 1.4 + 1.2 * 2.2  # B->E->J + B->F->J

    # Check some three-step paths
    # A->B->E->I
    # A->C->E->I
    expected_a_to_i = 1.5 * 3.0 * 0.7 + 2.0 * 0.8 * 0.7
    assert pytest.approx(result[('A', 'I')].item()) == expected_a_to_i

    # Check some four-step paths
    # A->B->E->I->N
    # A->C->E->I->N
    expected_a_to_n = 1.5 * 3.0 * 0.7 * 2.3 + 2.0 * 0.8 * 0.7 * 2.3
    expected_a_to_n += 1.5 * 3.0 * 1.4 * 1.9 + 2.0 * 0.8 * 1.4 * 1.9  # A->B->E->J->N + A->C->E->J->N
    expected_a_to_n += 1.5 * 1.2 * 2.2 * 1.9  # A->B->F->J->N
    assert pytest.approx(result[('A', 'N')].item()) == expected_a_to_n

    # Check a very long path from A to S
    # This should include:
    # A->B->E->J->O->S
    # A->B->F->K->O->S
    # A->C->E->J->O->S
    # A->C->G->K->O->S
    # A->D->G->K->O->S
    # A->D->H->L->P->S
    # A->D->H->M->Q->S
    # And many more
    assert ('A', 'S') in result

    # There are many paths to R from A - check their existence
    assert ('A', 'R') in result

    # Check that there's no direct path from A to R
    # But there should be indirect paths
    assert ('A', 'R') in result
    assert 'A' not in conversion_dict.get('R', {})

    # Count the number of paths calculated to verify algorithm explored all connections
    # In a DAG with 19 nodes (A through S), the maximum number of pairs is 19*18 = 342
    # But we won't have all possible connections due to the structure
    # Just verify we have a reasonable number
    assert len(result) > 50


if __name__ == '__main__':
    pytest.main()
