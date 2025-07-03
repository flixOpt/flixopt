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
def time_coords():
    return pd.date_range('2024-01-01', periods=5, freq='D', name='time')


@pytest.fixture
def scenario_coords():
    return pd.Index(['baseline', 'high', 'low'], name='scenario')


@pytest.fixture
def region_coords():
    return pd.Index(['north', 'south', 'east'], name='region')


class TestBasicConversion:
    """Test basic data type conversions with different coordinate configurations."""

    def test_scalar_no_coords(self):
        """Scalar without coordinates should create 0D DataArray."""
        result = DataConverter.to_dataarray(42)
        assert result.shape == ()
        assert result.dims == ()
        assert result.item() == 42

    def test_scalar_single_coord(self, time_coords):
        """Scalar with single coordinate should broadcast."""
        result = DataConverter.to_dataarray(42, coords={'time': time_coords})
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.all(result.values == 42)

    def test_scalar_multiple_coords(self, time_coords, scenario_coords):
        """Scalar with multiple coordinates should broadcast to all."""
        result = DataConverter.to_dataarray(42, coords={'time': time_coords, 'scenario': scenario_coords})
        assert result.shape == (5, 3)
        assert result.dims == ('time', 'scenario')
        assert np.all(result.values == 42)

    def test_numpy_scalars(self, time_coords):
        """Test numpy scalar types."""
        for scalar in [np.int32(42), np.int64(42), np.float32(42.5), np.float64(42.5)]:
            result = DataConverter.to_dataarray(scalar, coords={'time': time_coords})
            assert result.shape == (5,)
            assert np.all(result.values == scalar.item())


class TestArrayConversion:
    """Test numpy array conversions."""

    def test_1d_array_no_coords(self):
        """1D array without coords should fail unless single element."""
        # Multi-element fails
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(np.array([1, 2, 3]))

        # Single element succeeds
        result = DataConverter.to_dataarray(np.array([42]))
        assert result.shape == ()
        assert result.item() == 42

    def test_1d_array_matching_coord(self, time_coords):
        """1D array matching coordinate length should work."""
        arr = np.array([10, 20, 30, 40, 50])
        result = DataConverter.to_dataarray(arr, coords={'time': time_coords})
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, arr)

    def test_1d_array_mismatched_coord(self, time_coords):
        """1D array not matching coordinate length should fail."""
        arr = np.array([10, 20, 30])  # Length 3, time_coords has length 5
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(arr, coords={'time': time_coords})

    def test_1d_array_broadcast_to_multiple_coords(self, time_coords, scenario_coords):
        """1D array should broadcast to matching dimension."""
        # Array matching time dimension
        time_arr = np.array([10, 20, 30, 40, 50])
        result = DataConverter.to_dataarray(time_arr, coords={'time': time_coords, 'scenario': scenario_coords})
        assert result.shape == (5, 3)
        assert result.dims == ('time', 'scenario')

        # Each scenario should have the same time values
        for scenario in scenario_coords:
            assert np.array_equal(result.sel(scenario=scenario).values, time_arr)

        # Array matching scenario dimension
        scenario_arr = np.array([100, 200, 300])
        result = DataConverter.to_dataarray(scenario_arr, coords={'time': time_coords, 'scenario': scenario_coords})
        assert result.shape == (5, 3)
        assert result.dims == ('time', 'scenario')

        # Each time should have the same scenario values
        for time in time_coords:
            assert np.array_equal(result.sel(time=time).values, scenario_arr)

    def test_1d_array_ambiguous_length(self):
        """Array length matching multiple dimensions should fail."""
        # Both dimensions have length 3
        coords_3x3 = {
            'time': pd.date_range('2024-01-01', periods=3, freq='D', name='time'),
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario')
        }
        arr = np.array([1, 2, 3])

        with pytest.raises(ConversionError, match="matches multiple dimensions"):
            DataConverter.to_dataarray(arr, coords=coords_3x3)

    def test_multidimensional_array_rejected(self, time_coords):
        """Multidimensional arrays should be rejected."""
        arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ConversionError, match="Only 1D arrays supported"):
            DataConverter.to_dataarray(arr_2d, coords={'time': time_coords})


class TestSeriesConversion:
    """Test pandas Series conversions."""

    def test_series_no_coords(self):
        """Series without coords should fail unless single element."""
        # Multi-element fails
        series = pd.Series([1, 2, 3])
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(series)

        # Single element succeeds
        single_series = pd.Series([42])
        result = DataConverter.to_dataarray(single_series)
        assert result.shape == ()
        assert result.item() == 42

    def test_series_matching_index(self, time_coords, scenario_coords):
        """Series with matching index should work."""
        # Time-indexed series
        time_series = pd.Series([10, 20, 30, 40, 50], index=time_coords)
        result = DataConverter.to_dataarray(time_series, coords={'time': time_coords})
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, time_series.values)

        # Scenario-indexed series
        scenario_series = pd.Series([100, 200, 300], index=scenario_coords)
        result = DataConverter.to_dataarray(scenario_series, coords={'scenario': scenario_coords})
        assert result.shape == (3,)
        assert result.dims == ('scenario',)
        assert np.array_equal(result.values, scenario_series.values)

    def test_series_mismatched_index(self, time_coords):
        """Series with non-matching index should fail."""
        wrong_times = pd.date_range('2025-01-01', periods=5, freq='D', name='time')
        series = pd.Series([10, 20, 30, 40, 50], index=wrong_times)

        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(series, coords={'time': time_coords})

    def test_series_broadcast_to_multiple_coords(self, time_coords, scenario_coords):
        """Series should broadcast to non-matching dimensions."""
        # Time series broadcast to scenarios
        time_series = pd.Series([10, 20, 30, 40, 50], index=time_coords)
        result = DataConverter.to_dataarray(time_series, coords={'time': time_coords, 'scenario': scenario_coords})
        assert result.shape == (5, 3)

        for scenario in scenario_coords:
            assert np.array_equal(result.sel(scenario=scenario).values, time_series.values)

        # Scenario series broadcast to time
        scenario_series = pd.Series([100, 200, 300], index=scenario_coords)
        result = DataConverter.to_dataarray(scenario_series, coords={'time': time_coords, 'scenario': scenario_coords})
        assert result.shape == (5, 3)

        for time in time_coords:
            assert np.array_equal(result.sel(time=time).values, scenario_series.values)

    def test_series_wrong_dimension(self, time_coords, region_coords):
        """Series indexed by dimension not in coords should fail."""
        wrong_series = pd.Series([1, 2, 3], index=region_coords)

        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(wrong_series, coords={'time': time_coords})


class TestDataFrameConversion:
    """Test pandas DataFrame conversions."""

    def test_single_column_dataframe(self, time_coords):
        """Single-column DataFrame should work like Series."""
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50]}, index=time_coords)
        result = DataConverter.to_dataarray(df, coords={'time': time_coords})

        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, df['value'].values)

    def test_multi_column_dataframe_rejected(self, time_coords):
        """Multi-column DataFrame should be rejected."""
        df = pd.DataFrame({
            'value1': [10, 20, 30, 40, 50],
            'value2': [15, 25, 35, 45, 55]
        }, index=time_coords)

        with pytest.raises(ConversionError, match="Array has 2 dimensions but 1 target "):
            DataConverter.to_dataarray(df, coords={'time': time_coords})

    def test_empty_dataframe_rejected(self, time_coords):
        """Empty DataFrame should be rejected."""
        df = pd.DataFrame(index=time_coords)  # No columns

        with pytest.raises(ConversionError, match="DataFrame must have at least one"):
            DataConverter.to_dataarray(df, coords={'time': time_coords})

    def test_dataframe_broadcast(self, time_coords, scenario_coords):
        """Single-column DataFrame should broadcast like Series."""
        df = pd.DataFrame({'power': [10, 20, 30, 40, 50]}, index=time_coords)
        result = DataConverter.to_dataarray(df, coords={'time': time_coords, 'scenario': scenario_coords})

        assert result.shape == (5, 3)
        for scenario in scenario_coords:
            assert np.array_equal(result.sel(scenario=scenario).values, df['power'].values)


class TestDataArrayConversion:
    """Test xarray DataArray conversions."""

    def test_compatible_dataarray(self, time_coords):
        """Compatible DataArray should pass through."""
        original = xr.DataArray([10, 20, 30, 40, 50], coords={'time': time_coords}, dims=['time'])
        result = DataConverter.to_dataarray(original, coords={'time': time_coords})

        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, original.values)

        # Should be a copy
        result[0] = 999
        assert original[0].item() == 10

    def test_incompatible_dataarray_coords(self, time_coords):
        """DataArray with wrong coordinates should fail."""
        wrong_times = pd.date_range('2025-01-01', periods=5, freq='D', name='time')
        original = xr.DataArray([10, 20, 30, 40, 50], coords={'time': wrong_times}, dims=['time'])

        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(original, coords={'time': time_coords})

    def test_incompatible_dataarray_dims(self, time_coords):
        """DataArray with wrong dimensions should fail."""
        original = xr.DataArray([10, 20, 30, 40, 50], coords={'wrong_dim': range(5)}, dims=['wrong_dim'])

        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(original, coords={'time': time_coords})

    def test_dataarray_broadcast(self, time_coords, scenario_coords):
        """DataArray should broadcast to additional dimensions."""
        # 1D time DataArray to 2D time+scenario
        original = xr.DataArray([10, 20, 30, 40, 50], coords={'time': time_coords}, dims=['time'])
        result = DataConverter.to_dataarray(original, coords={'time': time_coords, 'scenario': scenario_coords})

        assert result.shape == (5, 3)
        assert result.dims == ('time', 'scenario')

        for scenario in scenario_coords:
            assert np.array_equal(result.sel(scenario=scenario).values, original.values)

    def test_scalar_dataarray_broadcast(self, time_coords, scenario_coords):
        """Scalar DataArray should broadcast to all dimensions."""
        scalar_da = xr.DataArray(42)
        result = DataConverter.to_dataarray(scalar_da, coords={'time': time_coords, 'scenario': scenario_coords})

        assert result.shape == (5, 3)
        assert np.all(result.values == 42)


class TestTimeSeriesDataConversion:
    """Test TimeSeriesData conversions."""

    def test_timeseries_data_basic(self, time_coords):
        """TimeSeriesData should work like DataArray."""
        data_array = xr.DataArray([10, 20, 30, 40, 50], coords={'time': time_coords}, dims=['time'])
        ts_data = TimeSeriesData(data_array, aggregation_group='test')

        result = DataConverter.to_dataarray(ts_data, coords={'time': time_coords})

        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, [10, 20, 30, 40, 50])

    def test_timeseries_data_broadcast(self, time_coords, scenario_coords):
        """TimeSeriesData should broadcast to additional dimensions."""
        data_array = xr.DataArray([10, 20, 30, 40, 50], coords={'time': time_coords}, dims=['time'])
        ts_data = TimeSeriesData(data_array)

        result = DataConverter.to_dataarray(ts_data, coords={'time': time_coords, 'scenario': scenario_coords})

        assert result.shape == (5, 3)
        for scenario in scenario_coords:
            assert np.array_equal(result.sel(scenario=scenario).values, [10, 20, 30, 40, 50])


class TestThreeDimensionConversion:
    """Test conversions with exactly 3 dimensions for all data types."""

    @pytest.fixture
    def three_d_coords(self, time_coords, scenario_coords):
        """Standard 3D coordinate system with unique lengths."""
        return {
            'time': time_coords,  # length 5
            'scenario': scenario_coords,  # length 3
            'region': pd.Index(['north', 'south'], name='region')  # length 2 - unique!
        }

    def test_scalar_three_dimensions(self, three_d_coords):
        """Scalar should broadcast to 3 dimensions."""
        result = DataConverter.to_dataarray(42, coords=three_d_coords)

        assert result.shape == (5, 3, 2)  # time=5, scenario=3, region=2
        assert result.dims == ('time', 'scenario', 'region')
        assert np.all(result.values == 42)

        # Verify all coordinates are correct
        assert result.indexes['time'].equals(three_d_coords['time'])
        assert result.indexes['scenario'].equals(three_d_coords['scenario'])
        assert result.indexes['region'].equals(three_d_coords['region'])

    def test_numpy_scalar_three_dimensions(self, three_d_coords):
        """Numpy scalars should broadcast to 3 dimensions."""
        for scalar in [np.int32(100), np.float64(3.14)]:
            result = DataConverter.to_dataarray(scalar, coords=three_d_coords)

            assert result.shape == (5, 3, 2)
            assert result.dims == ('time', 'scenario', 'region')
            assert np.all(result.values == scalar.item())

    def test_1d_array_time_to_three_dimensions(self, three_d_coords):
        """1D array matching time should broadcast to 3D."""
        time_arr = np.array([10, 20, 30, 40, 50])
        result = DataConverter.to_dataarray(time_arr, coords=three_d_coords)

        assert result.shape == (5, 3, 2)
        assert result.dims == ('time', 'scenario', 'region')

        # Check broadcasting across scenarios and regions
        for scenario in three_d_coords['scenario']:
            for region in three_d_coords['region']:
                slice_data = result.sel(scenario=scenario, region=region)
                assert np.array_equal(slice_data.values, time_arr)

    def test_1d_array_scenario_to_three_dimensions(self, three_d_coords):
        """1D array matching scenario should broadcast to 3D."""
        scenario_arr = np.array([100, 200, 300])
        result = DataConverter.to_dataarray(scenario_arr, coords=three_d_coords)

        assert result.shape == (5, 3, 2)
        assert result.dims == ('time', 'scenario', 'region')

        # Check broadcasting across time and regions
        for time in three_d_coords['time']:
            for region in three_d_coords['region']:
                slice_data = result.sel(time=time, region=region)
                assert np.array_equal(slice_data.values, scenario_arr)

    def test_1d_array_region_to_three_dimensions(self, three_d_coords):
        """1D array matching region should broadcast to 3D."""
        region_arr = np.array([1000, 2000])  # Length 2 to match region
        result = DataConverter.to_dataarray(region_arr, coords=three_d_coords)

        assert result.shape == (5, 3, 2)
        assert result.dims == ('time', 'scenario', 'region')

        # Check broadcasting across time and scenarios
        for time in three_d_coords['time']:
            for scenario in three_d_coords['scenario']:
                slice_data = result.sel(time=time, scenario=scenario)
                assert np.array_equal(slice_data.values, region_arr)

    def test_series_time_to_three_dimensions(self, three_d_coords):
        """Time-indexed Series should broadcast to 3D."""
        time_series = pd.Series([15, 25, 35, 45, 55], index=three_d_coords['time'])
        result = DataConverter.to_dataarray(time_series, coords=three_d_coords)

        assert result.shape == (5, 3, 2)
        assert result.dims == ('time', 'scenario', 'region')

        # Check broadcasting
        for scenario in three_d_coords['scenario']:
            for region in three_d_coords['region']:
                slice_data = result.sel(scenario=scenario, region=region)
                assert np.array_equal(slice_data.values, time_series.values)

    def test_series_scenario_to_three_dimensions(self, three_d_coords):
        """Scenario-indexed Series should broadcast to 3D."""
        scenario_series = pd.Series([500, 600, 700], index=three_d_coords['scenario'])
        result = DataConverter.to_dataarray(scenario_series, coords=three_d_coords)

        assert result.shape == (5, 3, 2)
        assert result.dims == ('time', 'scenario', 'region')

        # Check broadcasting
        for time in three_d_coords['time']:
            for region in three_d_coords['region']:
                slice_data = result.sel(time=time, region=region)
                assert np.array_equal(slice_data.values, scenario_series.values)

    def test_series_region_to_three_dimensions(self, three_d_coords):
        """Region-indexed Series should broadcast to 3D."""
        region_series = pd.Series([5000, 6000], index=three_d_coords['region'])  # Length 2
        result = DataConverter.to_dataarray(region_series, coords=three_d_coords)

        assert result.shape == (5, 3, 2)
        assert result.dims == ('time', 'scenario', 'region')

        # Check broadcasting
        for time in three_d_coords['time']:
            for scenario in three_d_coords['scenario']:
                slice_data = result.sel(time=time, scenario=scenario)
                assert np.array_equal(slice_data.values, region_series.values)

    def test_dataframe_time_to_three_dimensions(self, three_d_coords):
        """Time-indexed DataFrame should broadcast to 3D."""
        df = pd.DataFrame({'power': [11, 22, 33, 44, 55]}, index=three_d_coords['time'])
        result = DataConverter.to_dataarray(df, coords=three_d_coords)

        assert result.shape == (5, 3, 2)
        assert result.dims == ('time', 'scenario', 'region')

        # Check broadcasting
        for scenario in three_d_coords['scenario']:
            for region in three_d_coords['region']:
                slice_data = result.sel(scenario=scenario, region=region)
                assert np.array_equal(slice_data.values, df['power'].values)

    def test_dataframe_scenario_to_three_dimensions(self, three_d_coords):
        """Scenario-indexed DataFrame should broadcast to 3D."""
        df = pd.DataFrame({'cost': [1100, 1200, 1300]}, index=three_d_coords['scenario'])
        result = DataConverter.to_dataarray(df, coords=three_d_coords)

        assert result.shape == (5, 3, 2)
        assert result.dims == ('time', 'scenario', 'region')

        # Check broadcasting
        for time in three_d_coords['time']:
            for region in three_d_coords['region']:
                slice_data = result.sel(time=time, region=region)
                assert np.array_equal(slice_data.values, df['cost'].values)

    def test_1d_dataarray_time_to_three_dimensions(self, three_d_coords):
        """1D time DataArray should broadcast to 3D."""
        original = xr.DataArray([101, 102, 103, 104, 105],
                                coords={'time': three_d_coords['time']},
                                dims=['time'])
        result = DataConverter.to_dataarray(original, coords=three_d_coords)

        assert result.shape == (5, 3, 2)
        assert result.dims == ('time', 'scenario', 'region')

        # Check broadcasting
        for scenario in three_d_coords['scenario']:
            for region in three_d_coords['region']:
                slice_data = result.sel(scenario=scenario, region=region)
                assert np.array_equal(slice_data.values, original.values)

    def test_1d_dataarray_scenario_to_three_dimensions(self, three_d_coords):
        """1D scenario DataArray should broadcast to 3D."""
        original = xr.DataArray([2001, 2002, 2003],
                                coords={'scenario': three_d_coords['scenario']},
                                dims=['scenario'])
        result = DataConverter.to_dataarray(original, coords=three_d_coords)

        assert result.shape == (5, 3, 2)
        assert result.dims == ('time', 'scenario', 'region')

        # Check broadcasting
        for time in three_d_coords['time']:
            for region in three_d_coords['region']:
                slice_data = result.sel(time=time, region=region)
                assert np.array_equal(slice_data.values, original.values)

    def test_2d_dataarray_to_three_dimensions(self, three_d_coords):
        """2D DataArray should broadcast to 3D."""
        # Create 2D time x scenario DataArray
        data_2d = np.random.rand(5, 3)
        original = xr.DataArray(data_2d,
                                coords={'time': three_d_coords['time'],
                                        'scenario': three_d_coords['scenario']},
                                dims=['time', 'scenario'])

        result = DataConverter.to_dataarray(original, coords=three_d_coords)

        assert result.shape == (5, 3, 2)
        assert result.dims == ('time', 'scenario', 'region')

        # Check that all regions have the same time x scenario data
        for region in three_d_coords['region']:
            slice_data = result.sel(region=region)
            assert np.array_equal(slice_data.values, original.values)

    def test_timeseries_data_to_three_dimensions(self, three_d_coords):
        """TimeSeriesData should broadcast to 3D."""
        data_array = xr.DataArray([99, 88, 77, 66, 55],
                                  coords={'time': three_d_coords['time']},
                                  dims=['time'])
        ts_data = TimeSeriesData(data_array, aggregation_group='test_3d')

        result = DataConverter.to_dataarray(ts_data, coords=three_d_coords)

        assert result.shape == (5, 3, 2)
        assert result.dims == ('time', 'scenario', 'region')

        # Check broadcasting
        for scenario in three_d_coords['scenario']:
            for region in three_d_coords['region']:
                slice_data = result.sel(scenario=scenario, region=region)
                assert np.array_equal(slice_data.values, [99, 88, 77, 66, 55])

    def test_three_d_copy_independence(self, three_d_coords):
        """3D results should be independent copies."""
        original_arr = np.array([10, 20, 30, 40, 50])
        result = DataConverter.to_dataarray(original_arr, coords=three_d_coords)

        # Modify result
        result[0, 0, 0] = 999

        # Original should be unchanged
        assert original_arr[0] == 10

    def test_three_d_special_values(self, three_d_coords):
        """3D conversion should preserve special values."""
        # Array with NaN and inf
        special_arr = np.array([1, np.nan, np.inf, -np.inf, 5])
        result = DataConverter.to_dataarray(special_arr, coords=three_d_coords)

        assert result.shape == (5, 3, 2)

        # Check that special values are preserved in all broadcasts
        for scenario in three_d_coords['scenario']:
            for region in three_d_coords['region']:
                slice_data = result.sel(scenario=scenario, region=region)
                assert np.array_equal(np.isnan(slice_data.values), np.isnan(special_arr))
                assert np.array_equal(np.isinf(slice_data.values), np.isinf(special_arr))

    def test_three_d_ambiguous_length_error(self):
        """Should fail when array length matches multiple dimensions in 3D."""
        # All dimensions have length 3
        coords_3x3x3 = {
            'time': pd.date_range('2024-01-01', periods=3, freq='D', name='time'),
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),
            'region': pd.Index(['X', 'Y', 'Z'], name='region')
        }

        arr = np.array([1, 2, 3])  # Length 3 - matches all dimensions

        with pytest.raises(ConversionError, match="matches multiple dimensions"):
            DataConverter.to_dataarray(arr, coords=coords_3x3x3)

    def test_three_d_custom_dimensions(self):
        """3D conversion with custom dimension names."""
        coords = {
            'product': pd.Index(['A', 'B'], name='product'),
            'factory': pd.Index(['F1', 'F2', 'F3'], name='factory'),
            'quarter': pd.Index(['Q1', 'Q2', 'Q3', 'Q4'], name='quarter')
        }

        # Array matching factory dimension
        factory_arr = np.array([100, 200, 300])
        result = DataConverter.to_dataarray(factory_arr, coords=coords)

        assert result.shape == (2, 3, 4)
        assert result.dims == ('product', 'factory', 'quarter')

        # Check broadcasting
        for product in coords['product']:
            for quarter in coords['quarter']:
                slice_data = result.sel(product=product, quarter=quarter)
                assert np.array_equal(slice_data.values, factory_arr)


class TestMultipleDimensions:
    """Test support for more than 2 dimensions."""

    def test_scalar_many_dimensions(self):
        """Scalar should broadcast to any number of dimensions."""
        coords = {
            'time': pd.date_range('2024-01-01', periods=2, freq='D', name='time'),
            'scenario': pd.Index(['A', 'B'], name='scenario'),
            'region': pd.Index(['north', 'south'], name='region'),
            'technology': pd.Index(['solar', 'wind'], name='technology')
        }

        result = DataConverter.to_dataarray(42, coords=coords)
        assert result.shape == (2, 2, 2, 2)
        assert result.dims == ('time', 'scenario', 'region', 'technology')
        assert np.all(result.values == 42)

    def test_1d_array_broadcast_to_many_dimensions(self):
        """1D array should broadcast to many dimensions."""
        coords = {
            'time': pd.date_range('2024-01-01', periods=3, freq='D', name='time'),
            'scenario': pd.Index(['A', 'B'], name='scenario'),
            'region': pd.Index(['north', 'south'], name='region')
        }

        # Array matching time dimension
        time_arr = np.array([10, 20, 30])
        result = DataConverter.to_dataarray(time_arr, coords=coords)

        assert result.shape == (3, 2, 2)
        assert result.dims == ('time', 'scenario', 'region')

        # Check broadcasting - all scenarios and regions should have same time values
        for scenario in coords['scenario']:
            for region in coords['region']:
                assert np.array_equal(
                    result.sel(scenario=scenario, region=region).values,
                    time_arr
                )

    def test_series_broadcast_to_many_dimensions(self):
        """Series should broadcast to many dimensions."""
        time_coords = pd.date_range('2024-01-01', periods=3, freq='D', name='time')
        coords = {
            'time': time_coords,
            'scenario': pd.Index(['A', 'B'], name='scenario'),
            'region': pd.Index(['north', 'south'], name='region'),
            'product': pd.Index(['X', 'Y', 'Z'], name='product')
        }

        # Time-indexed series
        time_series = pd.Series([100, 200, 300], index=time_coords)
        result = DataConverter.to_dataarray(time_series, coords=coords)

        assert result.shape == (3, 2, 2, 3)
        assert result.dims == ('time', 'scenario', 'region', 'product')

        # Check that all non-time dimensions have the same time series values
        for scenario in coords['scenario']:
            for region in coords['region']:
                for product in coords['product']:
                    assert np.array_equal(
                        result.sel(scenario=scenario, region=region, product=product).values,
                        time_series.values
                    )

    def test_dataarray_broadcast_to_more_dimensions(self):
        """DataArray should broadcast to additional dimensions."""
        time_coords = pd.date_range('2024-01-01', periods=2, freq='D', name='time')
        scenario_coords = pd.Index(['A', 'B'], name='scenario')

        # Start with 2D DataArray
        original = xr.DataArray(
            [[10, 20], [30, 40]],
            coords={'time': time_coords, 'scenario': scenario_coords},
            dims=['time', 'scenario']
        )

        # Broadcast to 3D
        coords = {
            'time': time_coords,
            'scenario': scenario_coords,
            'region': pd.Index(['north', 'south'], name='region')
        }

        result = DataConverter.to_dataarray(original, coords=coords)

        assert result.shape == (2, 2, 2)
        assert result.dims == ('time', 'scenario', 'region')

        # Check that all regions have the same time+scenario values
        for region in coords['region']:
            assert np.array_equal(
                result.sel(region=region).values,
                original.values
            )


class TestCustomDimensions:
    """Test with custom dimension names beyond time/scenario."""

    def test_custom_single_dimension(self, region_coords):
        """Test with custom dimension name."""
        result = DataConverter.to_dataarray(42, coords={'region': region_coords})
        assert result.shape == (3,)
        assert result.dims == ('region',)
        assert np.all(result.values == 42)

    def test_custom_multiple_dimensions(self):
        """Test with multiple custom dimensions."""
        products = pd.Index(['A', 'B'], name='product')
        technologies = pd.Index(['solar', 'wind', 'gas'], name='technology')

        # Array matching technology dimension
        arr = np.array([100, 150, 80])
        result = DataConverter.to_dataarray(arr, coords={'product': products, 'technology': technologies})

        assert result.shape == (2, 3)
        assert result.dims == ('product', 'technology')

        # Should broadcast across products
        for product in products:
            assert np.array_equal(result.sel(product=product).values, arr)

    def test_mixed_dimension_types(self):
        """Test mixing time dimension with custom dimensions."""
        time_coords = pd.date_range('2024-01-01', periods=3, freq='D', name='time')
        regions = pd.Index(['north', 'south'], name='region')

        # Time series should broadcast to regions
        time_series = pd.Series([10, 20, 30], index=time_coords)
        result = DataConverter.to_dataarray(time_series, coords={'time': time_coords, 'region': regions})

        assert result.shape == (3, 2)
        assert result.dims == ('time', 'region')


class TestValidation:
    """Test coordinate validation."""

    def test_empty_coords(self):
        """Empty coordinates should work for scalars."""
        result = DataConverter.to_dataarray(42, coords={})
        assert result.shape == ()
        assert result.item() == 42

    def test_invalid_coord_type(self):
        """Non-pandas Index coordinates should fail."""
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(42, coords={'time': [1, 2, 3]})

    def test_empty_coord_index(self):
        """Empty coordinate index should fail."""
        empty_index = pd.Index([], name='time')
        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(42, coords={'time': empty_index})

    def test_time_coord_validation(self):
        """Time coordinates must be DatetimeIndex."""
        # Non-datetime index with name 'time' should fail
        wrong_time = pd.Index([1, 2, 3], name='time')
        with pytest.raises(ConversionError, match="time coordinates must be a DatetimeIndex"):
            DataConverter.to_dataarray(42, coords={'time': wrong_time})

    def test_coord_naming(self, time_coords):
        """Coordinates should be auto-renamed to match dimension."""
        # Unnamed time index should be renamed
        unnamed_time = time_coords.rename(None)
        result = DataConverter.to_dataarray(42, coords={'time': unnamed_time})
        assert result.coords['time'].name == 'time'


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_unsupported_data_types(self, time_coords):
        """Unsupported data types should fail with clear messages."""
        unsupported = [
            'string',
            object(),
            None,
            {'dict': 'value'},
            [1, 2, 3]
        ]

        for data in unsupported:
            with pytest.raises(ConversionError):
                DataConverter.to_dataarray(data, coords={'time': time_coords})

    def test_dimension_mismatch_messages(self, time_coords, scenario_coords):
        """Error messages should be informative."""
        # Array with wrong length
        wrong_arr = np.array([1, 2])  # Length 2, but no dimension has length 2
        with pytest.raises(ConversionError, match="matches none of the target dimensions"):
            DataConverter.to_dataarray(wrong_arr, coords={'time': time_coords, 'scenario': scenario_coords})


class TestDataIntegrity:
    """Test data copying and integrity."""

    def test_array_copy_independence(self, time_coords):
        """Converted arrays should be independent copies."""
        original_arr = np.array([10, 20, 30, 40, 50])
        result = DataConverter.to_dataarray(original_arr, coords={'time': time_coords})

        # Modify result
        result[0] = 999

        # Original should be unchanged
        assert original_arr[0] == 10

    def test_series_copy_independence(self, time_coords):
        """Converted Series should be independent copies."""
        original_series = pd.Series([10, 20, 30, 40, 50], index=time_coords)
        result = DataConverter.to_dataarray(original_series, coords={'time': time_coords})

        # Modify result
        result[0] = 999

        # Original should be unchanged
        assert original_series.iloc[0] == 10

    def test_dataframe_copy_independence(self, time_coords):
        """Converted DataFrames should be independent copies."""
        original_df = pd.DataFrame({'value': [10, 20, 30, 40, 50]}, index=time_coords)
        result = DataConverter.to_dataarray(original_df, coords={'time': time_coords})

        # Modify result
        result[0] = 999

        # Original should be unchanged
        assert original_df.loc[time_coords[0], 'value'] == 10


class TestSpecialValues:
    """Test handling of special numeric values."""

    def test_nan_values(self, time_coords):
        """NaN values should be preserved."""
        arr_with_nan = np.array([1, np.nan, 3, np.nan, 5])
        result = DataConverter.to_dataarray(arr_with_nan, coords={'time': time_coords})

        assert np.array_equal(np.isnan(result.values), np.isnan(arr_with_nan))
        assert np.array_equal(result.values[~np.isnan(result.values)], arr_with_nan[~np.isnan(arr_with_nan)])

    def test_infinite_values(self, time_coords):
        """Infinite values should be preserved."""
        arr_with_inf = np.array([1, np.inf, 3, -np.inf, 5])
        result = DataConverter.to_dataarray(arr_with_inf, coords={'time': time_coords})

        assert np.array_equal(result.values, arr_with_inf)

    def test_boolean_values(self, time_coords):
        """Boolean values should be preserved."""
        bool_arr = np.array([True, False, True, False, True])
        result = DataConverter.to_dataarray(bool_arr, coords={'time': time_coords})

        assert result.dtype == bool
        assert np.array_equal(result.values, bool_arr)

    def test_mixed_numeric_types(self, time_coords):
        """Mixed integer/float should become float."""
        mixed_arr = np.array([1, 2.5, 3, 4.5, 5])
        result = DataConverter.to_dataarray(mixed_arr, coords={'time': time_coords})

        assert np.issubdtype(result.dtype, np.floating)
        assert np.array_equal(result.values, mixed_arr)


class TestMultiDimensionalArrayConversion:
    """Test multi-dimensional numpy array conversions."""

    @pytest.fixture
    def standard_coords(self):
        """Standard coordinates with unique lengths for easy testing."""
        return {
            'time': pd.date_range('2024-01-01', periods=5, freq='D', name='time'),  # length 5
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),  # length 3
            'region': pd.Index(['north', 'south'], name='region')  # length 2
        }

    def test_2d_array_unique_dimensions(self, standard_coords):
        """2D array with unique dimension lengths should work."""
        # 5x3 array should map to time x scenario
        data_2d = np.random.rand(5, 3)
        result = DataConverter.to_dataarray(data_2d, coords={
            'time': standard_coords['time'],
            'scenario': standard_coords['scenario']
        })

        assert result.shape == (5, 3)
        assert result.dims == ('time', 'scenario')
        assert np.array_equal(result.values, data_2d)

        # 3x5 array should map to scenario x time
        data_2d_flipped = np.random.rand(3, 5)
        result_flipped = DataConverter.to_dataarray(data_2d_flipped, coords={
            'time': standard_coords['time'],
            'scenario': standard_coords['scenario']
        })

        assert result_flipped.shape == (3, 5)
        assert result_flipped.dims == ('scenario', 'time')
        assert np.array_equal(result_flipped.values, data_2d_flipped)

    def test_3d_array_unique_dimensions(self, standard_coords):
        """3D array with unique dimension lengths should work."""
        # 5x3x2 array should map to time x scenario x region
        data_3d = np.random.rand(5, 3, 2)
        result = DataConverter.to_dataarray(data_3d, coords=standard_coords)

        assert result.shape == (5, 3, 2)
        assert result.dims == ('time', 'scenario', 'region')
        assert np.array_equal(result.values, data_3d)

    def test_3d_array_different_permutation(self, standard_coords):
        """3D array with different dimension order should work."""
        # 2x5x3 array should map to region x time x scenario
        data_3d = np.random.rand(2, 5, 3)
        result = DataConverter.to_dataarray(data_3d, coords=standard_coords)

        assert result.shape == (2, 5, 3)
        assert result.dims == ('region', 'time', 'scenario')
        assert np.array_equal(result.values, data_3d)

    def test_4d_array_unique_dimensions(self):
        """4D array with unique dimension lengths should work."""
        coords = {
            'time': pd.date_range('2024-01-01', periods=2, freq='D', name='time'),  # length 2
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),  # length 3
            'region': pd.Index(['north', 'south', 'east', 'west'], name='region'),  # length 4
            'technology': pd.Index(['solar', 'wind', 'gas', 'coal', 'hydro'], name='technology')  # length 5
        }

        # 3x5x2x4 array should map to scenario x technology x time x region
        data_4d = np.random.rand(3, 5, 2, 4)
        result = DataConverter.to_dataarray(data_4d, coords=coords)

        assert result.shape == (3, 5, 2, 4)
        assert result.dims == ('scenario', 'technology', 'time', 'region')
        assert np.array_equal(result.values, data_4d)

    def test_2d_array_ambiguous_dimensions_error(self):
        """2D array with ambiguous dimension lengths should fail."""
        # Both dimensions have length 3
        coords_ambiguous = {
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),  # length 3
            'region': pd.Index(['north', 'south', 'east'], name='region')  # length 3
        }

        data_2d = np.random.rand(3, 3)
        with pytest.raises(ConversionError, match="matches multiple dimension orders"):
            DataConverter.to_dataarray(data_2d, coords=coords_ambiguous)

    def test_3d_array_ambiguous_dimensions_error(self):
        """3D array with ambiguous dimension lengths should fail."""
        # All dimensions have length 2
        coords_ambiguous = {
            'scenario': pd.Index(['A', 'B'], name='scenario'),  # length 2
            'region': pd.Index(['north', 'south'], name='region'),  # length 2
            'technology': pd.Index(['solar', 'wind'], name='technology')  # length 2
        }

        data_3d = np.random.rand(2, 2, 2)
        with pytest.raises(ConversionError, match="matches multiple dimension orders"):
            DataConverter.to_dataarray(data_3d, coords=coords_ambiguous)

    def test_array_dimension_count_mismatch_error(self, standard_coords):
        """Array with wrong number of dimensions should fail."""
        # 2D array with 3D coordinates
        data_2d = np.random.rand(5, 3)
        with pytest.raises(ConversionError, match="Array has 2 dimensions but 3 target dimensions provided"):
            DataConverter.to_dataarray(data_2d, coords=standard_coords)

        # 4D array with 3D coordinates
        data_4d = np.random.rand(5, 3, 2, 4)
        with pytest.raises(ConversionError, match="Array has 4 dimensions but 3 target dimensions provided"):
            DataConverter.to_dataarray(data_4d, coords=standard_coords)

    def test_array_no_matching_dimensions_error(self, standard_coords):
        """Array with no matching dimension lengths should fail."""
        # 7x8 array - no dimension has length 7 or 8
        data_2d = np.random.rand(7, 8)
        coords_2d = {
            'time': standard_coords['time'],  # length 5
            'scenario': standard_coords['scenario']  # length 3
        }

        with pytest.raises(ConversionError, match="Array dimensions do not match any coordinate lengths"):
            DataConverter.to_dataarray(data_2d, coords=coords_2d)

    def test_2d_array_custom_dimensions(self):
        """2D array with custom dimension names should work."""
        coords = {
            'product': pd.Index(['A', 'B', 'C', 'D'], name='product'),  # length 4
            'factory': pd.Index(['F1', 'F2', 'F3'], name='factory')  # length 3
        }

        # 4x3 array should map to product x factory
        data_2d = np.array([[10, 11, 12],
                           [20, 21, 22],
                           [30, 31, 32],
                           [40, 41, 42]])

        result = DataConverter.to_dataarray(data_2d, coords=coords)

        assert result.shape == (4, 3)
        assert result.dims == ('product', 'factory')
        assert np.array_equal(result.values, data_2d)

        # Verify coordinates are correct
        assert result.indexes['product'].equals(coords['product'])
        assert result.indexes['factory'].equals(coords['factory'])

    def test_multid_array_copy_independence(self, standard_coords):
        """Multi-D arrays should be independent copies."""
        original_data = np.random.rand(5, 3)
        result = DataConverter.to_dataarray(original_data, coords={
            'time': standard_coords['time'],
            'scenario': standard_coords['scenario']
        })

        # Modify result
        result[0, 0] = 999

        # Original should be unchanged
        assert original_data[0, 0] != 999

    def test_multid_array_special_values(self, standard_coords):
        """Multi-D arrays should preserve special values."""
        # Create 2D array with special values
        data_2d = np.array([[1.0, np.nan, 3.0],
                           [np.inf, 5.0, -np.inf],
                           [7.0, 8.0, 9.0],
                           [10.0, np.nan, 12.0],
                           [13.0, 14.0, np.inf]])

        result = DataConverter.to_dataarray(data_2d, coords={
            'time': standard_coords['time'],
            'scenario': standard_coords['scenario']
        })

        assert result.shape == (5, 3)
        assert np.array_equal(np.isnan(result.values), np.isnan(data_2d))
        assert np.array_equal(np.isinf(result.values), np.isinf(data_2d))

    def test_multid_array_with_time_dimension(self):
        """Multi-D arrays should work with time dimension."""
        time_coords = pd.date_range('2024-01-01', periods=4, freq='H', name='time')
        scenario_coords = pd.Index(['base', 'high', 'low'], name='scenario')

        # 4x3 time series data
        data_2d = np.array([[100, 110, 120],
                           [200, 210, 220],
                           [300, 310, 320],
                           [400, 410, 420]])

        result = DataConverter.to_dataarray(data_2d, coords={
            'time': time_coords,
            'scenario': scenario_coords
        })

        assert result.shape == (4, 3)
        assert result.dims == ('time', 'scenario')
        assert isinstance(result.indexes['time'], pd.DatetimeIndex)
        assert np.array_equal(result.values, data_2d)

    def test_multid_array_dtype_preservation(self, standard_coords):
        """Multi-D arrays should preserve data types."""
        # Integer array
        int_data = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9],
                            [10, 11, 12],
                            [13, 14, 15]], dtype=np.int32)

        result_int = DataConverter.to_dataarray(int_data, coords={
            'time': standard_coords['time'],
            'scenario': standard_coords['scenario']
        })

        assert result_int.dtype == np.int32
        assert np.array_equal(result_int.values, int_data)

        # Float array
        float_data = np.array([[1.1, 2.2, 3.3],
                              [4.4, 5.5, 6.6],
                              [7.7, 8.8, 9.9],
                              [10.1, 11.1, 12.1],
                              [13.1, 14.1, 15.1]], dtype=np.float64)

        result_float = DataConverter.to_dataarray(float_data, coords={
            'time': standard_coords['time'],
            'scenario': standard_coords['scenario']
        })

        assert result_float.dtype == np.float64
        assert np.array_equal(result_float.values, float_data)

        # Boolean array
        bool_data = np.array([[True, False, True],
                             [False, True, False],
                             [True, True, False],
                             [False, False, True],
                             [True, False, True]])

        result_bool = DataConverter.to_dataarray(bool_data, coords={
            'time': standard_coords['time'],
            'scenario': standard_coords['scenario']
        })

        assert result_bool.dtype == bool
        assert np.array_equal(result_bool.values, bool_data)

    def test_multid_array_no_coords(self):
        """Multi-D arrays without coords should fail unless scalar."""
        # Multi-element fails
        data_2d = np.random.rand(2, 3)
        with pytest.raises(ConversionError, match="Cannot convert multi-element array without target dimensions"):
            DataConverter.to_dataarray(data_2d)

        # Single element succeeds
        single_element = np.array([[42]])
        result = DataConverter.to_dataarray(single_element)
        assert result.shape == ()
        assert result.item() == 42

    def test_multid_array_empty_coords(self, standard_coords):
        """Multi-D arrays with empty coords should fail."""
        data_2d = np.random.rand(5, 3)
        with pytest.raises(ConversionError, match="Cannot convert multi-element array without target dimensions"):
            DataConverter.to_dataarray(data_2d, coords={})

    def test_multid_array_coordinate_validation(self):
        """Multi-D arrays should validate coordinates properly."""
        # Test with time coordinate that's not DatetimeIndex
        wrong_time = pd.Index([1, 2, 3, 4, 5], name='time')
        scenario_coords = pd.Index(['A', 'B', 'C'], name='scenario')

        data_2d = np.random.rand(5, 3)
        with pytest.raises(ConversionError, match="time coordinates must be a DatetimeIndex"):
            DataConverter.to_dataarray(data_2d, coords={
                'time': wrong_time,
                'scenario': scenario_coords
            })

    def test_multid_array_complex_scenario(self):
        """Complex real-world scenario with multi-D array."""
        # Energy system data: time x technology x region
        coords = {
            'time': pd.date_range('2024-01-01', periods=8760, freq='H', name='time'),  # 1 year hourly
            'technology': pd.Index(['solar', 'wind', 'gas', 'coal'], name='technology'),  # 4 technologies
            'region': pd.Index(['north', 'south', 'east'], name='region')  # 3 regions
        }

        # Capacity factors: 8760 x 4 x 3
        capacity_factors = np.random.rand(8760, 4, 3)

        result = DataConverter.to_dataarray(capacity_factors, coords=coords)

        assert result.shape == (8760, 4, 3)
        assert result.dims == ('time', 'technology', 'region')
        assert isinstance(result.indexes['time'], pd.DatetimeIndex)
        assert len(result.indexes['time']) == 8760
        assert len(result.indexes['technology']) == 4
        assert len(result.indexes['region']) == 3

    def test_multid_array_edge_cases(self):
        """Test edge cases for multi-D arrays."""
        # Single dimension with multi-D array should fail
        coords_1d = {'time': pd.date_range('2024-01-01', periods=5, freq='D', name='time')}
        data_2d = np.random.rand(5, 3)

        with pytest.raises(ConversionError, match="Array has 2 dimensions but 1 target dimensions provided"):
            DataConverter.to_dataarray(data_2d, coords=coords_1d)

        # Zero dimensions with multi-D array should fail
        data_1d = np.array([1, 2, 3])
        with pytest.raises(ConversionError, match="Cannot convert multi-element array without target dimensions"):
            DataConverter.to_dataarray(data_1d, coords={})

    def test_multid_array_partial_dimension_match(self):
        """Array where only some dimensions match should fail."""
        coords = {
            'time': pd.date_range('2024-01-01', periods=5, freq='D', name='time'),  # length 5
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario')  # length 3
        }

        # 5x7 array - first dimension matches time (5) but second doesn't match scenario (3)
        data_2d = np.random.rand(5, 7)
        with pytest.raises(ConversionError, match="Array dimensions do not match any coordinate lengths"):
            DataConverter.to_dataarray(data_2d, coords=coords)



if __name__ == '__main__':
    pytest.main()
