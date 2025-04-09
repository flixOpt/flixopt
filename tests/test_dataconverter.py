import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt.core import (  # Adjust this import to match your project structure
    ConversionError,
    DataConverter,
    TimeSeries,
)


@pytest.fixture
def sample_time_index():
    return pd.date_range('2024-01-01', periods=5, freq='D', name='time')


@pytest.fixture
def sample_scenario_index():
    return pd.Index(['baseline', 'high_demand', 'low_price'], name='scenario')


@pytest.fixture
def multi_index(sample_time_index, sample_scenario_index):
    """Create a sample MultiIndex combining scenarios and times."""
    return pd.MultiIndex.from_product([sample_scenario_index, sample_time_index], names=['scenario', 'time'])


class TestSingleDimensionConversion:
    """Tests for converting data without scenarios (1D: time only)."""

    def test_scalar_conversion(self, sample_time_index):
        """Test converting a scalar value."""
        # Test with integer
        result = DataConverter.as_dataarray(42, sample_time_index)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(sample_time_index),)
        assert result.dims == ('time',)
        assert np.all(result.values == 42)

        # Test with float
        result = DataConverter.as_dataarray(42.5, sample_time_index)
        assert np.all(result.values == 42.5)

        # Test with numpy scalar types
        result = DataConverter.as_dataarray(np.int64(42), sample_time_index)
        assert np.all(result.values == 42)
        result = DataConverter.as_dataarray(np.float32(42.5), sample_time_index)
        assert np.all(result.values == 42.5)

    def test_ndarray_conversion(self, sample_time_index):
        """Test converting a numpy ndarray."""
        # Test with integer 1D array
        arr_1d = np.array([1, 2, 3, 4, 5])
        result = DataConverter.as_dataarray(arr_1d, sample_time_index)
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, arr_1d)

        # Test with float 1D array
        arr_1d = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        result = DataConverter.as_dataarray(arr_1d, sample_time_index)
        assert np.array_equal(result.values, arr_1d)

        # Test with array containing NaN
        arr_1d = np.array([1, np.nan, 3, np.nan, 5])
        result = DataConverter.as_dataarray(arr_1d, sample_time_index)
        assert np.array_equal(np.isnan(result.values), np.isnan(arr_1d))
        assert np.array_equal(result.values[~np.isnan(result.values)], arr_1d[~np.isnan(arr_1d)])

    def test_dataarray_conversion(self, sample_time_index):
        """Test converting an existing xarray DataArray."""
        # Create original DataArray
        original = xr.DataArray(data=np.array([1, 2, 3, 4, 5]), coords={'time': sample_time_index}, dims=['time'])

        # Convert and check
        result = DataConverter.as_dataarray(original, sample_time_index)
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, original.values)

        # Ensure it's a copy
        result[0] = 999
        assert original[0].item() == 1  # Original should be unchanged

        # Test with different time coordinates but same length
        different_times = pd.date_range('2025-01-01', periods=5, freq='D', name='time')
        original = xr.DataArray(data=np.array([1, 2, 3, 4, 5]), coords={'time': different_times}, dims=['time'])

        # Should raise an error for mismatched time coordinates
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(original, sample_time_index)


class TestMultiDimensionConversion:
    """Tests for converting data with scenarios (2D: scenario × time)."""

    def test_scalar_with_scenarios(self, sample_time_index, sample_scenario_index):
        """Test converting scalar values with scenario dimension."""
        # Test with integer
        result = DataConverter.as_dataarray(42, sample_time_index, sample_scenario_index)

        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(sample_time_index), len(sample_scenario_index))
        assert result.dims == ('time', 'scenario')
        assert np.all(result.values == 42)
        assert set(result.coords['scenario'].values) == set(sample_scenario_index.values)
        assert set(result.coords['time'].values) == set(sample_time_index.values)

        # Test with float
        result = DataConverter.as_dataarray(42.5, sample_time_index, sample_scenario_index)
        assert np.all(result.values == 42.5)

    def test_1d_array_with_scenarios(self, sample_time_index, sample_scenario_index):
        """Test converting 1D array with scenario dimension (broadcasting)."""
        # Create 1D array matching timesteps length
        arr_1d = np.array([1, 2, 3, 4, 5])

        # Convert with scenarios
        result = DataConverter.as_dataarray(arr_1d, sample_time_index, sample_scenario_index)

        assert result.shape == (len(sample_time_index), len(sample_scenario_index))
        assert result.dims == ('time', 'scenario')

        # Each scenario should have the same values (broadcasting)
        for scenario in sample_scenario_index:
            scenario_slice = result.sel(scenario=scenario)
            assert np.array_equal(scenario_slice.values, arr_1d)

    def test_2d_array_with_scenarios(self, sample_time_index, sample_scenario_index):
        """Test converting 2D array with scenario dimension."""
        # Create 2D array with different values per scenario
        arr_2d = np.array(
            [
                [1, 2, 3, 4, 5],  # baseline scenario
                [6, 7, 8, 9, 10],  # high_demand scenario
                [11, 12, 13, 14, 15],  # low_price scenario
            ]
        )

        # Convert to DataArray
        result = DataConverter.as_dataarray(arr_2d.T, sample_time_index, sample_scenario_index)

        assert result.shape == (5, 3)
        assert result.dims == ('time', 'scenario')

        # Check that each scenario has correct values
        assert np.array_equal(result.sel(scenario='baseline').values, arr_2d[0])
        assert np.array_equal(result.sel(scenario='high_demand').values, arr_2d[1])
        assert np.array_equal(result.sel(scenario='low_price').values, arr_2d[2])

    def test_dataarray_with_scenarios(self, sample_time_index, sample_scenario_index):
        """Test converting an existing DataArray with scenarios."""
        # Create a multi-scenario DataArray
        original = xr.DataArray(
            data=np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]),
            coords={'scenario': sample_scenario_index, 'time': sample_time_index},
            dims=['scenario', 'time'],
        )

        # Test conversion
        result = DataConverter.as_dataarray(original, sample_time_index, sample_scenario_index)

        assert result.shape == (5, 3)
        assert result.dims == ('time', 'scenario')
        assert np.array_equal(result.values, original.values.T)

        # Ensure it's a copy
        result.loc[:, 'baseline'] = 999
        assert original.sel(scenario='baseline')[0].item() == 1  # Original should be unchanged


class TestInvalidInputs:
    """Tests for invalid inputs and error handling."""

    def test_time_index_validation(self):
        """Test validation of time index."""
        # Test with unnamed index
        unnamed_index = pd.date_range('2024-01-01', periods=5, freq='D')
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(42, unnamed_index)

        # Test with empty index
        empty_index = pd.DatetimeIndex([], name='time')
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(42, empty_index)

        # Test with non-DatetimeIndex
        wrong_type_index = pd.Index([1, 2, 3, 4, 5], name='time')
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(42, wrong_type_index)

    def test_scenario_index_validation(self, sample_time_index):
        """Test validation of scenario index."""
        # Test with unnamed scenario index
        unnamed_index = pd.Index(['baseline', 'high_demand'])
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(42, sample_time_index, unnamed_index)

        # Test with empty scenario index
        empty_index = pd.Index([], name='scenario')
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(42, sample_time_index, empty_index)

        # Test with non-Index scenario
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(42, sample_time_index, ['baseline', 'high_demand'])

    def test_invalid_data_types(self, sample_time_index, sample_scenario_index):
        """Test handling of invalid data types."""
        # Test invalid input type (string)
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray('invalid_string', sample_time_index)

        # Test invalid input type with scenarios
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray('invalid_string', sample_time_index, sample_scenario_index)

        # Test unsupported complex object
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(object(), sample_time_index)

        # Test None value
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(None, sample_time_index)

    def test_mismatched_input_dimensions(self, sample_time_index, sample_scenario_index):
        """Test handling of mismatched input dimensions."""
        # Test mismatched Series index
        mismatched_series = pd.Series(
            [1, 2, 3, 4, 5, 6], index=pd.date_range('2025-01-01', periods=6, freq='D', name='time')
        )
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(mismatched_series, sample_time_index)

        # Test DataFrame with multiple columns
        df_multi_col = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]}, index=sample_time_index)
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(df_multi_col, sample_time_index)

        # Test mismatched array shape for time-only
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(np.array([1, 2, 3]), sample_time_index)  # Wrong length

        # Test mismatched array shape for scenario × time
        # Array shape should be (n_scenarios, n_timesteps)
        wrong_shape_array = np.array(
            [
                [1, 2, 3, 4],  # Missing a timestep
                [5, 6, 7, 8],
                [9, 10, 11, 12],
            ]
        )
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(wrong_shape_array, sample_time_index, sample_scenario_index)

        # Test array with too many dimensions
        with pytest.raises(ConversionError):
            # 3D array not allowed
            DataConverter.as_dataarray(np.ones((3, 5, 2)), sample_time_index, sample_scenario_index)

    def test_dataarray_dimension_mismatch(self, sample_time_index, sample_scenario_index):
        """Test handling of mismatched DataArray dimensions."""
        # Create DataArray with wrong dimensions
        wrong_dims = xr.DataArray(data=np.array([1, 2, 3, 4, 5]), coords={'wrong_dim': range(5)}, dims=['wrong_dim'])
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(wrong_dims, sample_time_index)

        # Create DataArray with scenario but no time
        wrong_dims_2 = xr.DataArray(data=np.array([1, 2, 3]), coords={'scenario': ['a', 'b', 'c']}, dims=['scenario'])
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(wrong_dims_2, sample_time_index, sample_scenario_index)

        # Create DataArray with right dims but wrong length
        wrong_length = xr.DataArray(
            data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            coords={
                'scenario': sample_scenario_index,
                'time': pd.date_range('2024-01-01', periods=3, freq='D', name='time'),
            },
            dims=['scenario', 'time'],
        )
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(wrong_length, sample_time_index, sample_scenario_index)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_single_timestep(self, sample_scenario_index):
        """Test with a single timestep."""
        # Test with only one timestep
        single_timestep = pd.DatetimeIndex(['2024-01-01'], name='time')

        # Scalar conversion
        result = DataConverter.as_dataarray(42, single_timestep)
        assert result.shape == (1,)
        assert result.dims == ('time',)

        # With scenarios
        result_with_scenarios = DataConverter.as_dataarray(42, single_timestep, sample_scenario_index)
        assert result_with_scenarios.shape == (len(sample_scenario_index), 1)
        assert result_with_scenarios.dims == ('time', 'scenario')

    def test_single_scenario(self, sample_time_index):
        """Test with a single scenario."""
        # Test with only one scenario
        single_scenario = pd.Index(['baseline'], name='scenario')

        # Scalar conversion with single scenario
        result = DataConverter.as_dataarray(42, sample_time_index, single_scenario)
        assert result.shape == (1, len(sample_time_index))
        assert result.dims == ('time', 'scenario')

        # Array conversion with single scenario
        arr = np.array([1, 2, 3, 4, 5])
        result_arr = DataConverter.as_dataarray(arr, sample_time_index, single_scenario)
        assert result_arr.shape == (1, 5)
        assert np.array_equal(result_arr.sel(scenario='baseline').values, arr)

        # 2D array with single scenario
        arr_2d = np.array([[1, 2, 3, 4, 5]])  # Note the extra dimension
        result_arr_2d = DataConverter.as_dataarray(arr_2d, sample_time_index, single_scenario)
        assert result_arr_2d.shape == (1, 5)
        assert np.array_equal(result_arr_2d.sel(scenario='baseline').values, arr_2d[0])

    def test_different_scenario_order(self, sample_time_index):
        """Test that scenario order is preserved."""
        # Test with different scenario orders
        scenarios1 = pd.Index(['a', 'b', 'c'], name='scenario')
        scenarios2 = pd.Index(['c', 'b', 'a'], name='scenario')

        # Create DataArray with first order
        data = np.array(
            [
                [1, 2, 3, 4, 5],  # a
                [6, 7, 8, 9, 10],  # b
                [11, 12, 13, 14, 15],  # c
            ]
        )

        result1 = DataConverter.as_dataarray(data, sample_time_index, scenarios1)
        assert np.array_equal(result1.sel(scenario='a').values, [1, 2, 3, 4, 5])
        assert np.array_equal(result1.sel(scenario='c').values, [11, 12, 13, 14, 15])

        # Create DataArray with second order
        result2 = DataConverter.as_dataarray(data, sample_time_index, scenarios2)
        # First row should match 'c' now
        assert np.array_equal(result2.sel(scenario='c').values, [1, 2, 3, 4, 5])
        # Last row should match 'a' now
        assert np.array_equal(result2.sel(scenario='a').values, [11, 12, 13, 14, 15])

    def test_all_nan_data(self, sample_time_index, sample_scenario_index):
        """Test handling of all-NaN data."""
        # Create array of all NaNs
        all_nan_array = np.full(5, np.nan)
        result = DataConverter.as_dataarray(all_nan_array, sample_time_index)
        assert np.all(np.isnan(result.values))

        # With scenarios
        result = DataConverter.as_dataarray(all_nan_array, sample_time_index, sample_scenario_index)
        assert result.shape == (len(sample_time_index), len(sample_scenario_index))
        assert np.all(np.isnan(result.values))

        # Series of all NaNs
        result = DataConverter.as_dataarray(
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan]), sample_time_index, sample_scenario_index
        )
        assert np.all(np.isnan(result.values))

    def test_mixed_data_types(self, sample_time_index, sample_scenario_index):
        """Test conversion of mixed integer and float data."""
        # Create array with mixed types
        mixed_array = np.array([1, 2.5, 3, 4.5, 5])
        result = DataConverter.as_dataarray(mixed_array, sample_time_index)

        # Result should be float dtype
        assert np.issubdtype(result.dtype, np.floating)
        assert np.array_equal(result.values, mixed_array)

        # With scenarios
        result = DataConverter.as_dataarray(mixed_array, sample_time_index, sample_scenario_index)
        assert np.issubdtype(result.dtype, np.floating)
        for scenario in sample_scenario_index:
            assert np.array_equal(result.sel(scenario=scenario).values, mixed_array)


class TestFunctionalUseCase:
    """Tests for realistic use cases combining multiple features."""

    def test_large_dataset(self, sample_scenario_index):
        """Test with a larger dataset to ensure performance."""
        # Create a larger timestep array (e.g., hourly for a year)
        large_timesteps = pd.date_range(
            '2024-01-01',
            periods=8760,  # Hours in a year
            freq='H',
            name='time',
        )

        # Create large 2D array (3 scenarios × 8760 hours)
        large_data = np.random.rand(len(sample_scenario_index), len(large_timesteps))

        # Convert and check
        result = DataConverter.as_dataarray(large_data, large_timesteps, sample_scenario_index)

        assert result.shape == (len(sample_scenario_index), len(large_timesteps))
        assert result.dims == ('time', 'scenario')
        assert np.array_equal(result.values, large_data)


class TestMultiScenarioArrayConversion:
    """Tests specifically focused on array conversion with scenarios."""

    def test_1d_array_broadcasting(self, sample_time_index, sample_scenario_index):
        """Test that 1D arrays are properly broadcast to all scenarios."""
        arr_1d = np.array([1, 2, 3, 4, 5])
        result = DataConverter.as_dataarray(arr_1d, sample_time_index, sample_scenario_index)

        assert result.shape == (len(sample_time_index), len(sample_scenario_index))

        # Each scenario should have identical values
        for i, scenario in enumerate(sample_scenario_index):
            assert np.array_equal(result.sel(scenario=scenario).values, arr_1d)

            # Modify one scenario's values
            result.loc[dict(scenario=scenario)] = np.ones(len(sample_time_index)) * i

        # Ensure modifications are isolated to each scenario
        for i, scenario in enumerate(sample_scenario_index):
            assert np.all(result.sel(scenario=scenario).values == i)

    def test_2d_array_different_shapes(self, sample_time_index):
        """Test different scenario shapes with 2D arrays."""
        # Test with 1 scenario
        single_scenario = pd.Index(['baseline'], name='scenario')
        arr_1_scenario = np.array([[1, 2, 3, 4, 5]])

        result = DataConverter.as_dataarray(arr_1_scenario, sample_time_index, single_scenario)
        assert result.shape == (1, len(sample_time_index))

        # Test with 2 scenarios
        two_scenarios = pd.Index(['baseline', 'high_demand'], name='scenario')
        arr_2_scenarios = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        result = DataConverter.as_dataarray(arr_2_scenarios, sample_time_index, two_scenarios)
        assert result.shape == (2, len(sample_time_index))
        assert np.array_equal(result.sel(scenario='baseline').values, arr_2_scenarios[0])
        assert np.array_equal(result.sel(scenario='high_demand').values, arr_2_scenarios[1])

        # Test mismatched scenarios count
        three_scenarios = pd.Index(['a', 'b', 'c'], name='scenario')
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(arr_2_scenarios, sample_time_index, three_scenarios)

    def test_array_handling_edge_cases(self, sample_time_index, sample_scenario_index):
        """Test array edge cases."""
        # Test with boolean array
        bool_array = np.array([True, False, True, False, True])
        result = DataConverter.as_dataarray(bool_array, sample_time_index, sample_scenario_index)
        assert result.dtype == bool
        assert result.shape == (len(sample_time_index), len(sample_scenario_index))

        # Test with array containing infinite values
        inf_array = np.array([1, np.inf, 3, -np.inf, 5])
        result = DataConverter.as_dataarray(inf_array, sample_time_index, sample_scenario_index)
        for scenario in sample_scenario_index:
            scenario_data = result.sel(scenario=scenario)
            assert np.isinf(scenario_data[1].item())
            assert np.isinf(scenario_data[3].item())
            assert scenario_data[3].item() < 0  # Negative infinity


class TestScenarioReindexing:
    """Tests for reindexing and coordinate preservation in DataConverter."""

    def test_preserving_scenario_order(self, sample_time_index):
        """Test that scenario order is preserved in converted DataArrays."""
        # Define scenarios in a specific order
        scenarios = pd.Index(['scenario3', 'scenario1', 'scenario2'], name='scenario')

        # Create 2D array
        data = np.array(
            [
                [1, 2, 3, 4, 5],  # scenario3
                [6, 7, 8, 9, 10],  # scenario1
                [11, 12, 13, 14, 15],  # scenario2
            ]
        )

        # Convert to DataArray
        result = DataConverter.as_dataarray(data, sample_time_index, scenarios)

        # Verify order of scenarios is preserved
        assert list(result.coords['scenario'].values) == list(scenarios)

        # Verify data for each scenario
        assert np.array_equal(result.sel(scenario='scenario3').values, data[0])
        assert np.array_equal(result.sel(scenario='scenario1').values, data[1])
        assert np.array_equal(result.sel(scenario='scenario2').values, data[2])


if __name__ == '__main__':
    pytest.main()


def test_invalid_inputs(sample_time_index):
    # Test invalid input type
    with pytest.raises(ConversionError):
        DataConverter.as_dataarray('invalid_string', sample_time_index)

    # Test mismatched Series index
    mismatched_series = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('2025-01-01', periods=6, freq='D'))
    with pytest.raises(ConversionError):
        DataConverter.as_dataarray(mismatched_series, sample_time_index)

    # Test DataFrame with multiple columns
    df_multi_col = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]}, index=sample_time_index)
    with pytest.raises(ConversionError):
        DataConverter.as_dataarray(df_multi_col, sample_time_index)

    # Test mismatched array shape
    with pytest.raises(ConversionError):
        DataConverter.as_dataarray(np.array([1, 2, 3]), sample_time_index)  # Wrong length

    # Test multi-dimensional array
    with pytest.raises(ConversionError):
        DataConverter.as_dataarray(np.array([[1, 2], [3, 4]]), sample_time_index)  # 2D array not allowed


def test_time_index_validation():
    # Test with unnamed index
    unnamed_index = pd.date_range('2024-01-01', periods=5, freq='D')
    with pytest.raises(ConversionError):
        DataConverter.as_dataarray(42, unnamed_index)

    # Test with empty index
    empty_index = pd.DatetimeIndex([], name='time')
    with pytest.raises(ConversionError):
        DataConverter.as_dataarray(42, empty_index)

    # Test with non-DatetimeIndex
    wrong_type_index = pd.Index([1, 2, 3, 4, 5], name='time')
    with pytest.raises(ConversionError):
        DataConverter.as_dataarray(42, wrong_type_index)


if __name__ == '__main__':
    pytest.main()
