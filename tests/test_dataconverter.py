import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt.core import ConversionError, DataConverter  # Adjust this import to match your project structure


@pytest.fixture
def sample_time_index():
    return pd.date_range('2024-01-01', periods=5, freq='D', name='time')


@pytest.fixture
def sample_scenario_index():
    return pd.Index(['baseline', 'high_demand', 'low_price'], name='scenario')


class TestSingleDimensionConversion:
    """Tests for converting data without scenarios (1D: time only)"""

    def test_scalar_conversion(self, sample_time_index):
        # Test scalar conversion
        result = DataConverter.as_dataarray(42, sample_time_index)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(sample_time_index),)
        assert result.dims == ('time',)
        assert np.all(result.values == 42)

    def test_series_conversion(self, sample_time_index):
        series = pd.Series([1, 2, 3, 4, 5], index=sample_time_index)

        # Test Series conversion
        result = DataConverter.as_dataarray(series, sample_time_index)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, series.values)

    def test_dataframe_conversion(self, sample_time_index):
        # Create a single-column DataFrame
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5]}, index=sample_time_index)

        # Test DataFrame conversion
        result = DataConverter.as_dataarray(df, sample_time_index)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values.flatten(), df['A'].values)

    def test_ndarray_conversion(self, sample_time_index):
        # Test 1D array conversion
        arr_1d = np.array([1, 2, 3, 4, 5])
        result = DataConverter.as_dataarray(arr_1d, sample_time_index)
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, arr_1d)

    def test_dataarray_conversion(self, sample_time_index):
        # Create a DataArray
        original = xr.DataArray(
            data=np.array([1, 2, 3, 4, 5]),
            coords={'time': sample_time_index},
            dims=['time']
        )

        # Test DataArray conversion
        result = DataConverter.as_dataarray(original, sample_time_index)
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, original.values)

        # Ensure it's a copy
        result[0] = 999
        assert original[0].item() == 1  # Original should be unchanged


class TestMultiDimensionConversion:
    """Tests for converting data with scenarios (2D: scenario × time)"""

    def test_scalar_with_scenarios(self, sample_time_index, sample_scenario_index):
        # Convert scalar with scenario dimension
        result = DataConverter.as_dataarray(42, sample_time_index, sample_scenario_index)

        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(sample_scenario_index), len(sample_time_index))
        assert result.dims == ('scenario', 'time')
        assert np.all(result.values == 42)
        assert set(result.coords['scenario'].values) == set(sample_scenario_index.values)
        assert set(result.coords['time'].values) == set(sample_time_index.values)

    def test_series_with_scenarios(self, sample_time_index, sample_scenario_index):
        # Create time series data
        series = pd.Series([1, 2, 3, 4, 5], index=sample_time_index)

        # Convert with scenario dimension
        result = DataConverter.as_dataarray(series, sample_time_index, sample_scenario_index)

        assert result.shape == (len(sample_scenario_index), len(sample_time_index))
        assert result.dims == ('scenario', 'time')

        # Values should be broadcast to all scenarios
        for scenario in sample_scenario_index:
            scenario_slice = result.sel(scenario=scenario)
            assert np.array_equal(scenario_slice.values, series.values)

    def test_dataframe_with_scenarios(self, sample_time_index, sample_scenario_index):
        # Create a single-column DataFrame
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5]}, index=sample_time_index)

        # Convert with scenario dimension
        result = DataConverter.as_dataarray(df, sample_time_index, sample_scenario_index)

        assert result.shape == (len(sample_scenario_index), len(sample_time_index))
        assert result.dims == ('scenario', 'time')

        # Values should be broadcast to all scenarios
        for scenario in sample_scenario_index:
            scenario_slice = result.sel(scenario=scenario)
            assert np.array_equal(scenario_slice.values, df['A'].values)

    def test_ndarray_with_scenarios(self, sample_time_index, sample_scenario_index):
        # Test multi-scenario array conversion
        # For multi-dimensional, the first dimension should match number of scenarios
        arr_2d = np.array(
            [
                [1, 2, 3, 4, 5],  # baseline scenario
                [6, 7, 8, 9, 10],  # high_demand scenario
                [11, 12, 13, 14, 15],  # low_price scenario
            ]
        )

        result = DataConverter.as_dataarray(arr_2d, sample_time_index, sample_scenario_index)

        assert result.shape == (3, 5)
        assert result.dims == ('scenario', 'time')

        # Check that each scenario has correct values
        assert np.array_equal(result.sel(scenario='baseline').values, arr_2d[0])
        assert np.array_equal(result.sel(scenario='high_demand').values, arr_2d[1])
        assert np.array_equal(result.sel(scenario='low_price').values, arr_2d[2])

    def test_dataarray_with_scenarios(self, sample_time_index, sample_scenario_index):
        # Create a multi-scenario DataArray
        original = xr.DataArray(
            data=np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]),
            coords={'scenario': sample_scenario_index, 'time': sample_time_index},
            dims=['scenario', 'time'],
        )

        # Test conversion
        result = DataConverter.as_dataarray(original, sample_time_index, sample_scenario_index)

        assert result.shape == (3, 5)
        assert result.dims == ('scenario', 'time')
        assert np.array_equal(result.values, original.values)

        # Ensure it's a copy
        result.loc['baseline'] = 999
        assert original.sel(scenario='baseline')[0].item() == 1  # Original should be unchanged


class TestInvalidInputs:
    """Tests for invalid inputs and error handling"""

    def test_time_index_validation(self):
        # Test with unnamed index
        unnamed_index = pd.date_range('2024-01-01', periods=5, freq='D')
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(42, unnamed_index)

        # Test with empty index
        empty_index = pd.DatetimeIndex([], name='time')
        with pytest.raises(ValueError):
            DataConverter.as_dataarray(42, empty_index)

        # Test with non-DatetimeIndex
        wrong_type_index = pd.Index([1, 2, 3, 4, 5], name='time')
        with pytest.raises(ValueError):
            DataConverter.as_dataarray(42, wrong_type_index)

    def test_scenario_index_validation(self, sample_time_index):
        # Test with unnamed scenario index
        unnamed_index = pd.Index(['baseline', 'high_demand'])
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(42, sample_time_index, unnamed_index)

        # Test with empty scenario index
        empty_index = pd.Index([], name='scenario')
        with pytest.raises(ValueError):
            DataConverter.as_dataarray(42, sample_time_index, empty_index)

    def test_invalid_data_types(self, sample_time_index, sample_scenario_index):
        # Test invalid input type (string)
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray('invalid_string', sample_time_index)

        # Test invalid input type with scenarios
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray('invalid_string', sample_time_index, sample_scenario_index)

        # Test unsupported complex object
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(object(), sample_time_index)

    def test_mismatched_input_dimensions(self, sample_time_index, sample_scenario_index):
        # Test mismatched Series index
        mismatched_series = pd.Series(
            [1, 2, 3, 4, 5, 6],
            index=pd.date_range('2025-01-01', periods=6, freq='D', name='time')
        )
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(mismatched_series, sample_time_index)

        # Test DataFrame with multiple columns
        df_multi_col = pd.DataFrame(
            {'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]},
            index=sample_time_index
        )
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(df_multi_col, sample_time_index)

        # Test mismatched array shape for time-only
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(np.array([1, 2, 3]), sample_time_index)  # Wrong length

        # Test mismatched array shape for scenario × time
        # Array shape should be (n_scenarios, n_timesteps)
        wrong_shape_array = np.array([
            [1, 2, 3, 4],  # Missing a timestep
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ])
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(wrong_shape_array, sample_time_index, sample_scenario_index)

        # Test array with too many dimensions
        with pytest.raises(ConversionError):
            # 3D array not allowed
            DataConverter.as_dataarray(np.ones((3, 5, 2)), sample_time_index, sample_scenario_index)

    def test_dataarray_dimension_mismatch(self, sample_time_index, sample_scenario_index):
        # Create DataArray with wrong dimensions
        wrong_dims = xr.DataArray(
            data=np.array([1, 2, 3, 4, 5]),
            coords={'wrong_dim': range(5)},
            dims=['wrong_dim']
        )
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(wrong_dims, sample_time_index)

        # Create DataArray with scenario but no time
        wrong_dims_2 = xr.DataArray(
            data=np.array([1, 2, 3]),
            coords={'scenario': ['a', 'b', 'c']},
            dims=['scenario']
        )
        with pytest.raises(ConversionError):
            DataConverter.as_dataarray(wrong_dims_2, sample_time_index, sample_scenario_index)


class TestEdgeCases:
    """Tests for edge cases and special scenarios"""

    def test_single_timestep(self, sample_scenario_index):
        # Test with only one timestep
        single_timestep = pd.DatetimeIndex(['2024-01-01'], name='time')

        # Scalar conversion
        result = DataConverter.as_dataarray(42, single_timestep)
        assert result.shape == (1,)
        assert result.dims == ('time',)

        # With scenarios
        result_with_scenarios = DataConverter.as_dataarray(42, single_timestep, sample_scenario_index)
        assert result_with_scenarios.shape == (len(sample_scenario_index), 1)
        assert result_with_scenarios.dims == ('scenario', 'time')

    def test_single_scenario(self, sample_time_index):
        # Test with only one scenario
        single_scenario = pd.Index(['baseline'], name='scenario')

        # Scalar conversion with single scenario
        result = DataConverter.as_dataarray(42, sample_time_index, single_scenario)
        assert result.shape == (1, len(sample_time_index))
        assert result.dims == ('scenario', 'time')

        # Array conversion with single scenario
        arr = np.array([1, 2, 3, 4, 5])
        result_arr = DataConverter.as_dataarray(arr, sample_time_index, single_scenario)
        assert result_arr.shape == (1, 5)
        assert np.array_equal(result_arr.sel(scenario='baseline').values, arr)

    def test_different_scenario_order(self, sample_time_index):
        # Test that scenario order is preserved
        scenarios1 = pd.Index(['a', 'b', 'c'], name='scenario')
        scenarios2 = pd.Index(['c', 'b', 'a'], name='scenario')

        # Create DataArray with first order
        data = np.array([
            [1, 2, 3, 4, 5],  # a
            [6, 7, 8, 9, 10],  # b
            [11, 12, 13, 14, 15]  # c
        ])

        result1 = DataConverter.as_dataarray(data, sample_time_index, scenarios1)
        assert np.array_equal(result1.sel(scenario='a').values, [1, 2, 3, 4, 5])
        assert np.array_equal(result1.sel(scenario='c').values, [11, 12, 13, 14, 15])

        # Create DataArray with second order
        result2 = DataConverter.as_dataarray(data, sample_time_index, scenarios2)
        # First row should match 'c' now
        assert np.array_equal(result2.sel(scenario='c').values, [1, 2, 3, 4, 5])
        # Last row should match 'a' now
        assert np.array_equal(result2.sel(scenario='a').values, [11, 12, 13, 14, 15])



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
    with pytest.raises(ValueError):
        DataConverter.as_dataarray(42, empty_index)

    # Test with non-DatetimeIndex
    wrong_type_index = pd.Index([1, 2, 3, 4, 5], name='time')
    with pytest.raises(ValueError):
        DataConverter.as_dataarray(42, wrong_type_index)


if __name__ == '__main__':
    pytest.main()
