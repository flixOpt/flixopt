import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt.core import (  # Adjust this import to match your project structure
    ConversionError,
    DataConverter,
    TimeSeriesData,
)


@pytest.fixture
def sample_time_index():
    return pd.date_range('2024-01-01', periods=5, freq='D', name='time')


@pytest.fixture
def sample_scenario_index():
    return pd.Index(['baseline', 'high_demand', 'low_price'], name='scenario')


class TestSingleDimensionConversion:
    """Tests for converting data without scenarios (1D: time only)."""

    def test_scalar_conversion(self, sample_time_index):
        """Test converting a scalar value."""
        # Test with integer
        result = DataConverter.to_dataarray(42, sample_time_index)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(sample_time_index),)
        assert result.dims == ('time',)
        assert np.all(result.values == 42)

        # Test with float
        result = DataConverter.to_dataarray(42.5, sample_time_index)
        assert np.all(result.values == 42.5)

        # Test with numpy scalar types
        result = DataConverter.to_dataarray(np.int64(42), sample_time_index)
        assert np.all(result.values == 42)
        result = DataConverter.to_dataarray(np.float32(42.5), sample_time_index)
        assert np.all(result.values == 42.5)

    def test_ndarray_conversion(self, sample_time_index):
        """Test converting a numpy ndarray."""
        # Test with integer 1D array
        arr_1d = np.array([1, 2, 3, 4, 5])
        result = DataConverter.to_dataarray(arr_1d, sample_time_index)
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, arr_1d)

        # Test with float 1D array
        arr_1d = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        result = DataConverter.to_dataarray(arr_1d, sample_time_index)
        assert np.array_equal(result.values, arr_1d)

        # Test with array containing NaN
        arr_1d = np.array([1, np.nan, 3, np.nan, 5])
        result = DataConverter.to_dataarray(arr_1d, sample_time_index)
        assert np.array_equal(np.isnan(result.values), np.isnan(arr_1d))
        assert np.array_equal(result.values[~np.isnan(result.values)], arr_1d[~np.isnan(arr_1d)])

    def test_dataarray_conversion(self, sample_time_index):
        """Test converting an existing xarray DataArray."""
        # Create original DataArray
        original = xr.DataArray(data=np.array([1, 2, 3, 4, 5]), coords={'time': sample_time_index}, dims=['time'])

        # Convert and check
        result = DataConverter.to_dataarray(original, sample_time_index)
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
            DataConverter.to_dataarray(original, sample_time_index)


class TestMultiDimensionConversion:
    """Tests for converting data with scenarios (2D: scenario × time)."""

    def test_scalar_with_scenarios(self, sample_time_index, sample_scenario_index):
        """Test converting scalar values with scenario dimension."""
        # Test with integer
        result = DataConverter.to_dataarray(42, sample_time_index, sample_scenario_index)

        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(sample_time_index), len(sample_scenario_index))
        assert result.dims == ('time', 'scenario')
        assert np.all(result.values == 42)
        assert set(result.coords['scenario'].values) == set(sample_scenario_index.values)
        assert set(result.coords['time'].values) == set(sample_time_index.values)

        # Test with float
        result = DataConverter.to_dataarray(42.5, sample_time_index, sample_scenario_index)
        assert np.all(result.values == 42.5)

    def test_1d_array_with_scenarios_time_broadcast(self, sample_time_index, sample_scenario_index):
        """Test converting 1D array matching time dimension (broadcasting across scenarios)."""
        # Create 1D array matching timesteps length
        arr_1d = np.array([1, 2, 3, 4, 5])

        # Convert with scenarios
        result = DataConverter.to_dataarray(arr_1d, sample_time_index, sample_scenario_index)

        assert result.shape == (len(sample_time_index), len(sample_scenario_index))
        assert result.dims == ('time', 'scenario')

        # Each scenario should have the same values (broadcasting)
        for scenario in sample_scenario_index:
            scenario_slice = result.sel(scenario=scenario)
            assert np.array_equal(scenario_slice.values, arr_1d)

    def test_1d_array_with_scenarios_scenario_broadcast(self, sample_time_index, sample_scenario_index):
        """Test converting 1D array matching scenario dimension (broadcasting across time)."""
        # Create 1D array matching scenario length
        arr_1d = np.array([10, 20, 30])  # 3 scenarios

        # Convert with time and scenarios
        result = DataConverter.to_dataarray(arr_1d, sample_time_index, sample_scenario_index)

        assert result.shape == (len(sample_time_index), len(sample_scenario_index))
        assert result.dims == ('time', 'scenario')

        # Each time step should have the same scenario values (broadcasting)
        for time in sample_time_index:
            time_slice = result.sel(time=time)
            assert np.array_equal(time_slice.values, arr_1d)

    def test_dataarray_with_scenarios(self, sample_time_index, sample_scenario_index):
        """Test converting an existing DataArray with scenarios."""
        # Create a multi-scenario DataArray with dims in (time, scenario) order
        original = xr.DataArray(
            data=np.array([[1, 6, 11], [2, 7, 12], [3, 8, 13], [4, 9, 14], [5, 10, 15]]),
            coords={'time': sample_time_index, 'scenario': sample_scenario_index},
            dims=['time', 'scenario'],
        )

        # Test conversion
        result = DataConverter.to_dataarray(original, sample_time_index, sample_scenario_index)

        assert result.shape == (5, 3)
        assert result.dims == ('time', 'scenario')
        assert np.array_equal(result.values, original.values)

        # Ensure it's a copy
        result.loc[:, 'baseline'] = 999
        assert original.sel(scenario='baseline')[0].item() == 1  # Original should be unchanged


class TestSeriesConversion:
    """Tests for converting pandas Series to DataArray."""

    def test_series_single_dimension_time(self, sample_time_index):
        """Test converting a pandas Series with time index."""
        # Create a Series with matching time index
        series = pd.Series([10, 20, 30, 40, 50], index=sample_time_index)

        # Convert and check
        result = DataConverter.to_dataarray(series, sample_time_index)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, series.values)
        assert np.array_equal(result.coords['time'].values, sample_time_index.values)

    def test_series_single_dimension_scenario(self, sample_scenario_index):
        """Test converting a pandas Series with scenario index."""
        # Create a Series with scenario index
        series = pd.Series([100, 200, 300], index=sample_scenario_index)

        result = DataConverter.to_dataarray(series, scenarios=sample_scenario_index)
        assert result.shape == (3,)
        assert result.dims == ('scenario',)
        assert np.array_equal(result.values, series.values)
        assert np.array_equal(result.coords['scenario'].values, sample_scenario_index.values)

    def test_series_mismatched_index(self, sample_time_index):
        """Test converting a Series with mismatched index."""
        # Create Series with different time index
        different_times = pd.date_range('2025-01-01', periods=5, freq='D', name='time')
        series = pd.Series([10, 20, 30, 40, 50], index=different_times)

        # Should raise error for mismatched index
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(series, sample_time_index)

    def test_series_broadcast_to_scenarios(self, sample_time_index, sample_scenario_index):
        """Test broadcasting a time-indexed Series across scenarios."""
        # Create a Series with time index
        series = pd.Series([10, 20, 30, 40, 50], index=sample_time_index)

        # Convert with scenarios
        result = DataConverter.to_dataarray(series, sample_time_index, sample_scenario_index)

        assert result.shape == (5, 3)
        assert result.dims == ('time', 'scenario')

        # Check broadcasting - each scenario should have the same values
        for scenario in sample_scenario_index:
            scenario_slice = result.sel(scenario=scenario)
            assert np.array_equal(scenario_slice.values, series.values)

    def test_series_broadcast_to_time(self, sample_time_index, sample_scenario_index):
        """Test broadcasting a scenario-indexed Series across time."""
        # Create a Series with scenario index
        series = pd.Series([100, 200, 300], index=sample_scenario_index)

        # Convert with time
        result = DataConverter.to_dataarray(series, sample_time_index, sample_scenario_index)

        assert result.shape == (5, 3)
        assert result.dims == ('time', 'scenario')

        # Check broadcasting - each time should have the same scenario values
        for time in sample_time_index:
            time_slice = result.sel(time=time)
            assert np.array_equal(time_slice.values, series.values)


class TestTimeSeriesDataConversion:
    """Tests for converting TimeSeriesData objects."""

    def test_timeseries_data_conversion(self, sample_time_index):
        """Test converting TimeSeriesData."""
        # Create TimeSeriesData
        data_array = xr.DataArray([1, 2, 3, 4, 5], coords={'time': sample_time_index}, dims=['time'])
        ts_data = TimeSeriesData(data_array, aggregation_group='test_group')

        # Convert
        result = DataConverter.to_dataarray(ts_data, sample_time_index)

        assert isinstance(result, xr.DataArray)
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, [1, 2, 3, 4, 5])

    def test_timeseries_data_with_scenarios(self, sample_time_index, sample_scenario_index):
        """Test converting TimeSeriesData with broadcasting to scenarios."""
        # Create 1D TimeSeriesData
        data_array = xr.DataArray([1, 2, 3, 4, 5], coords={'time': sample_time_index}, dims=['time'])
        ts_data = TimeSeriesData(data_array)

        # Convert with scenarios (should broadcast)
        result = DataConverter.to_dataarray(ts_data, sample_time_index, sample_scenario_index)

        assert result.shape == (5, 3)
        assert result.dims == ('time', 'scenario')

        # Each scenario should have the same values
        for scenario in sample_scenario_index:
            assert np.array_equal(result.sel(scenario=scenario).values, [1, 2, 3, 4, 5])


class TestDataFrameConversion:
    """Tests for converting single-column pandas DataFrames to DataArray."""

    def test_single_column_dataframe_time(self, sample_time_index):
        """Test converting a single-column DataFrame with time index."""
        # Create DataFrame with one column
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50]}, index=sample_time_index)

        # Convert and check
        result = DataConverter.to_dataarray(df, sample_time_index)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, df['value'].values)

    def test_single_column_dataframe_scenario(self, sample_scenario_index):
        """Test converting a single-column DataFrame with scenario index."""
        # Create DataFrame with one column and scenario index
        df = pd.DataFrame({'value': [100, 200, 300]}, index=sample_scenario_index)

        result = DataConverter.to_dataarray(df, scenarios=sample_scenario_index)
        assert result.shape == (3,)
        assert result.dims == ('scenario',)
        assert np.array_equal(result.values, df['value'].values)

    def test_dataframe_broadcast_to_scenarios(self, sample_time_index, sample_scenario_index):
        """Test broadcasting a time-indexed DataFrame across scenarios."""
        # Create DataFrame with time index
        df = pd.DataFrame({'power': [10, 20, 30, 40, 50]}, index=sample_time_index)

        # Convert with scenarios
        result = DataConverter.to_dataarray(df, sample_time_index, sample_scenario_index)

        assert result.shape == (5, 3)
        assert result.dims == ('time', 'scenario')

        # Check broadcasting - each scenario should have the same values
        for scenario in sample_scenario_index:
            scenario_slice = result.sel(scenario=scenario)
            assert np.array_equal(scenario_slice.values, df['power'].values)

    def test_dataframe_broadcast_to_time(self, sample_time_index, sample_scenario_index):
        """Test broadcasting a scenario-indexed DataFrame across time."""
        # Create DataFrame with scenario index
        df = pd.DataFrame({'cost': [100, 200, 300]}, index=sample_scenario_index)

        # Convert with time
        result = DataConverter.to_dataarray(df, sample_time_index, sample_scenario_index)

        assert result.shape == (5, 3)
        assert result.dims == ('time', 'scenario')

        # Check broadcasting - each time should have the same scenario values
        for time in sample_time_index:
            time_slice = result.sel(time=time)
            assert np.array_equal(time_slice.values, df['cost'].values)

    def test_multi_column_dataframe_fails(self, sample_time_index):
        """Test that multi-column DataFrames are rejected."""
        # Create DataFrame with multiple columns
        df = pd.DataFrame({
            'value1': [10, 20, 30, 40, 50],
            'value2': [15, 25, 35, 45, 55]
        }, index=sample_time_index)

        # Should raise error
        with pytest.raises(ConversionError, match="Only single-column DataFrames are supported"):
            DataConverter.to_dataarray(df, sample_time_index)

    def test_dataframe_mismatched_index(self, sample_time_index):
        """Test DataFrame with mismatched index."""
        # Create DataFrame with different time index
        different_times = pd.date_range('2025-01-01', periods=5, freq='D', name='time')
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50]}, index=different_times)

        # Should raise error for mismatched index
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(df, sample_time_index)

    def test_dataframe_copy_behavior(self, sample_time_index):
        """Test that DataFrame conversion creates a copy."""
        # Create DataFrame
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50]}, index=sample_time_index)

        # Convert
        result = DataConverter.to_dataarray(df, sample_time_index)

        # Modify the result
        result[0] = 999

        # Original DataFrame should be unchanged
        assert df.loc[sample_time_index[0], 'value'] == 10

    def test_empty_dataframe_fails(self, sample_time_index):
        """Test that empty DataFrames are rejected."""
        # DataFrame with no columns
        df = pd.DataFrame(index=sample_time_index)

        with pytest.raises(ConversionError, match="Only single-column DataFrames are supported"):
            DataConverter.to_dataarray(df, sample_time_index)

    def test_dataframe_with_named_column(self, sample_time_index):
        """Test DataFrame with a named column."""
        df = pd.DataFrame(index=sample_time_index)
        df['energy_output'] = [100, 150, 200, 175, 125]

        result = DataConverter.to_dataarray(df, sample_time_index)
        assert result.shape == (5,)
        assert np.array_equal(result.values, [100, 150, 200, 175, 125])


class TestInvalidInputs:
    """Tests for invalid inputs and error handling."""

    def test_time_index_validation(self):
        """Test validation of time index."""
        # Test with unnamed index
        unnamed_index = pd.date_range('2024-01-01', periods=5, freq='D')
        # Should automatically rename to 'time' with a warning, not raise error
        result = DataConverter.to_dataarray(42, unnamed_index)
        assert result.coords['time'].name == 'time'

        # Test with empty index
        empty_index = pd.DatetimeIndex([], name='time')
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(42, empty_index)

        # Test with non-DatetimeIndex
        wrong_type_index = pd.Index([1, 2, 3, 4, 5], name='time')
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(42, wrong_type_index)

    def test_scenario_index_validation(self, sample_time_index):
        """Test validation of scenario index."""
        # Test with unnamed scenario index
        unnamed_index = pd.Index(['baseline', 'high_demand'])
        # Should automatically rename to 'scenario' with a warning, not raise error
        result = DataConverter.to_dataarray(42, sample_time_index, unnamed_index)
        assert result.coords['scenario'].name == 'scenario'

        # Test with empty scenario index
        empty_index = pd.Index([], name='scenario')
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(42, sample_time_index, empty_index)

        # Test with non-Index scenario
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(42, sample_time_index, ['baseline', 'high_demand'])

    def test_invalid_data_types(self, sample_time_index, sample_scenario_index):
        """Test handling of invalid data types."""
        # Test invalid input type (string)
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray('invalid_string', sample_time_index)

        # Test invalid input type with scenarios
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray('invalid_string', sample_time_index, sample_scenario_index)

        # Test unsupported complex object
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(object(), sample_time_index)

        # Test None value
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(None, sample_time_index)

    def test_multidimensional_array_rejection(self, sample_time_index, sample_scenario_index):
        """Test that multidimensional arrays are rejected."""
        # Test 2D array (not supported in simplified version)
        arr_2d = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        with pytest.raises(ConversionError, match="Only 1D arrays supported"):
            DataConverter.to_dataarray(arr_2d, sample_time_index)

        # Test 3D array
        arr_3d = np.ones((2, 3, 4))
        with pytest.raises(ConversionError, match="Only 1D arrays supported"):
            DataConverter.to_dataarray(arr_3d, sample_time_index, sample_scenario_index)

    def test_mismatched_input_dimensions(self, sample_time_index, sample_scenario_index):
        """Test handling of mismatched input dimensions."""
        # Test mismatched Series index
        mismatched_series = pd.Series(
            [1, 2, 3, 4, 5, 6], index=pd.date_range('2025-01-01', periods=6, freq='D', name='time')
        )
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(mismatched_series, sample_time_index)

        # Test mismatched array length for time-only
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(np.array([1, 2, 3]), sample_time_index)  # Wrong length

        # Test array that doesn't match either dimension
        wrong_length_array = np.array([1, 2, 3, 4])  # Doesn't match time (5) or scenario (3)
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(wrong_length_array, sample_time_index, sample_scenario_index)

    def test_dataarray_dimension_mismatch(self, sample_time_index, sample_scenario_index):
        """Test handling of mismatched DataArray dimensions."""
        # Create DataArray with wrong dimensions
        wrong_dims = xr.DataArray(data=np.array([1, 2, 3, 4, 5]), coords={'wrong_dim': range(5)}, dims=['wrong_dim'])
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(wrong_dims, sample_time_index)

        # Create DataArray with right dims but wrong coordinate values
        wrong_coords = xr.DataArray(
            data=np.array([1, 2, 3, 4, 5]),
            coords={'time': pd.date_range('2025-01-01', periods=5, freq='D', name='time')},
            dims=['time']
        )
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(wrong_coords, sample_time_index)


class TestDataArrayBroadcasting:
    """Tests for broadcasting DataArrays."""

    def test_broadcast_1d_array_to_2d_time(self, sample_time_index, sample_scenario_index):
        """Test broadcasting a 1D array (time) to 2D."""
        arr_1d = np.array([1, 2, 3, 4, 5])

        result = DataConverter.to_dataarray(arr_1d, sample_time_index, sample_scenario_index)

        # Should broadcast across scenarios
        expected = np.repeat(arr_1d[:, np.newaxis], len(sample_scenario_index), axis=1)
        assert np.array_equal(result.values, expected)
        assert result.dims == ('time', 'scenario')

    def test_broadcast_1d_array_to_2d_scenario(self, sample_time_index, sample_scenario_index):
        """Test broadcasting a 1D array (scenario) to 2D."""
        arr_1d = np.array([1, 2, 3])  # Matches scenario length

        result = DataConverter.to_dataarray(arr_1d, sample_time_index, sample_scenario_index)

        # Should broadcast across time
        expected = np.repeat(arr_1d[np.newaxis, :], len(sample_time_index), axis=0)
        assert np.array_equal(result.values, expected)
        assert result.dims == ('time', 'scenario')

    def test_broadcast_1d_array_to_1d(self, sample_time_index):
        """Test that 1D array with matching dimension doesn't change."""
        arr_1d = np.array([1, 2, 3, 4, 5])

        result = DataConverter.to_dataarray(arr_1d, sample_time_index)

        assert np.array_equal(result.values, arr_1d)
        assert result.dims == ('time',)

    def test_scalar_dataarray_broadcasting(self, sample_time_index, sample_scenario_index):
        """Test broadcasting scalar DataArray."""
        scalar_da = xr.DataArray(42)

        result = DataConverter.to_dataarray(scalar_da, sample_time_index, sample_scenario_index)

        assert result.shape == (len(sample_time_index), len(sample_scenario_index))
        assert np.all(result.values == 42)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_single_timestep(self, sample_scenario_index):
        """Test with a single timestep."""
        # Test with only one timestep
        single_timestep = pd.DatetimeIndex(['2024-01-01'], name='time')

        # Scalar conversion
        result = DataConverter.to_dataarray(42, single_timestep)
        assert result.shape == (1,)
        assert result.dims == ('time',)

        # With scenarios
        result_with_scenarios = DataConverter.to_dataarray(42, single_timestep, sample_scenario_index)
        assert result_with_scenarios.shape == (1, len(sample_scenario_index))
        assert result_with_scenarios.dims == ('time', 'scenario')

    def test_single_scenario(self, sample_time_index):
        """Test with a single scenario."""
        # Test with only one scenario
        single_scenario = pd.Index(['baseline'], name='scenario')

        # Scalar conversion with single scenario
        result = DataConverter.to_dataarray(42, sample_time_index, single_scenario)
        assert result.shape == (len(sample_time_index), 1)
        assert result.dims == ('time', 'scenario')

        # Array conversion with single scenario
        arr = np.array([1, 2, 3, 4, 5])
        result_arr = DataConverter.to_dataarray(arr, sample_time_index, single_scenario)
        assert result_arr.shape == (5, 1)
        assert np.array_equal(result_arr.sel(scenario='baseline').values, arr)

    def test_all_nan_data(self, sample_time_index, sample_scenario_index):
        """Test handling of all-NaN data."""
        # Create array of all NaNs
        all_nan_array = np.full(5, np.nan)
        result = DataConverter.to_dataarray(all_nan_array, sample_time_index)
        assert np.all(np.isnan(result.values))

        # With scenarios
        result = DataConverter.to_dataarray(all_nan_array, sample_time_index, sample_scenario_index)
        assert result.shape == (len(sample_time_index), len(sample_scenario_index))
        assert np.all(np.isnan(result.values))

    def test_mixed_data_types(self, sample_time_index, sample_scenario_index):
        """Test conversion of mixed integer and float data."""
        # Create array with mixed types
        mixed_array = np.array([1, 2.5, 3, 4.5, 5])
        result = DataConverter.to_dataarray(mixed_array, sample_time_index)

        # Result should be float dtype
        assert np.issubdtype(result.dtype, np.floating)
        assert np.array_equal(result.values, mixed_array)

        # With scenarios
        result = DataConverter.to_dataarray(mixed_array, sample_time_index, sample_scenario_index)
        assert np.issubdtype(result.dtype, np.floating)
        for scenario in sample_scenario_index:
            assert np.array_equal(result.sel(scenario=scenario).values, mixed_array)

    def test_boolean_data(self, sample_time_index, sample_scenario_index):
        """Test handling of boolean data."""
        bool_array = np.array([True, False, True, False, True])
        result = DataConverter.to_dataarray(bool_array, sample_time_index, sample_scenario_index)
        assert result.dtype == bool
        assert result.shape == (len(sample_time_index), len(sample_scenario_index))


class TestNoIndexConversion:
    """Tests for conversion without any indices (scalar results)."""

    def test_scalar_no_dimensions(self):
        """Test scalar conversion without any dimensions."""
        result = DataConverter.to_dataarray(42)
        assert isinstance(result, xr.DataArray)
        assert result.shape == ()
        assert result.dims == ()
        assert result.item() == 42

    def test_single_element_array_no_dimensions(self):
        """Test single-element array without dimensions."""
        arr = np.array([42])
        result = DataConverter.to_dataarray(arr)
        assert result.shape == ()
        assert result.item() == 42

    def test_multi_element_array_no_dimensions_fails(self):
        """Test that multi-element array fails without dimensions."""
        arr = np.array([1, 2, 3])
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(arr)

    def test_series_no_dimensions_fails(self):
        """Test that multi-element Series fails without dimensions."""
        series = pd.Series([1, 2, 3])
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(series)

    def test_single_element_series_no_dimensions(self):
        """Test single-element Series without dimensions."""
        series = pd.Series([42])
        result = DataConverter.to_dataarray(series)
        assert result.shape == ()
        assert result.item() == 42


if __name__ == '__main__':
    pytest.main()
