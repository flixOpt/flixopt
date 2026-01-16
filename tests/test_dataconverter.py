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


@pytest.fixture
def standard_coords():
    """Standard coordinates with unique lengths for easy testing."""
    return {
        'time': pd.date_range('2024-01-01', periods=5, freq='D', name='time'),  # length 5
        'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),  # length 3
        'region': pd.Index(['north', 'south'], name='region'),  # length 2
    }


class TestScalarConversion:
    """Test scalar data conversions with different coordinate configurations."""

    def test_scalar_no_coords(self):
        """Scalar without coordinates should create 0D DataArray."""
        result = DataConverter.to_dataarray(42)
        assert result.shape == ()
        assert result.dims == ()
        assert result.item() == 42

    def test_scalar_single_coord(self, time_coords):
        """Scalar with single coordinate stays as scalar (no broadcasting at conversion time)."""
        result = DataConverter.to_dataarray(42, coords={'time': time_coords})
        # Scalars stay as scalars - broadcasting happens at linopy interface
        assert result.shape == ()
        assert result.dims == ()
        assert result.item() == 42

    def test_scalar_multiple_coords(self, time_coords, scenario_coords):
        """Scalar with multiple coordinates stays as scalar (no broadcasting at conversion time)."""
        result = DataConverter.to_dataarray(42, coords={'time': time_coords, 'scenario': scenario_coords})
        # Scalars stay as scalars - broadcasting happens at linopy interface
        assert result.shape == ()
        assert result.dims == ()
        assert result.item() == 42

    def test_numpy_scalars(self, time_coords):
        """Test numpy scalar types stay as scalars."""
        for scalar in [np.int32(42), np.int64(42), np.float32(42.5), np.float64(42.5)]:
            result = DataConverter.to_dataarray(scalar, coords={'time': time_coords})
            # Scalars stay as scalars - broadcasting happens at linopy interface
            assert result.shape == ()
            assert result.item() == scalar.item()

    def test_scalar_many_dimensions(self, standard_coords):
        """Scalar stays as scalar regardless of target dimensions."""
        coords = {**standard_coords, 'technology': pd.Index(['solar', 'wind'], name='technology')}

        result = DataConverter.to_dataarray(42, coords=coords)
        # Scalars stay as scalars - broadcasting happens at linopy interface
        assert result.shape == ()
        assert result.dims == ()
        assert result.item() == 42


class TestOneDimensionalArrayConversion:
    """Test 1D numpy array and pandas Series conversions."""

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
        """1D array stays in minimal form, matching only one dimension (no broadcast at conversion)."""
        # Array matching time dimension - stays as 1D with time dim only
        time_arr = np.array([10, 20, 30, 40, 50])
        result = DataConverter.to_dataarray(time_arr, coords={'time': time_coords, 'scenario': scenario_coords})
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, time_arr)

        # Array matching scenario dimension - stays as 1D with scenario dim only
        scenario_arr = np.array([100, 200, 300])
        result = DataConverter.to_dataarray(scenario_arr, coords={'time': time_coords, 'scenario': scenario_coords})
        assert result.shape == (3,)
        assert result.dims == ('scenario',)
        assert np.array_equal(result.values, scenario_arr)

    def test_1d_array_ambiguous_length(self):
        """Array length matching multiple dimensions should fail."""
        # Both dimensions have length 3
        coords_3x3 = {
            'time': pd.date_range('2024-01-01', periods=3, freq='D', name='time'),
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),
        }
        arr = np.array([1, 2, 3])

        with pytest.raises(ConversionError, match='matches multiple dimension'):
            DataConverter.to_dataarray(arr, coords=coords_3x3)

    def test_1d_array_broadcast_to_many_dimensions(self, standard_coords):
        """1D array stays in minimal form with matched dimension only (no broadcast at conversion)."""
        # Array matching time dimension - stays as 1D with time dim only
        time_arr = np.array([10, 20, 30, 40, 50])
        result = DataConverter.to_dataarray(time_arr, coords=standard_coords)

        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, time_arr)


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
        """Series stays in minimal form with matched dimension only (no broadcast at conversion)."""
        # Time series stays as 1D with time dim only
        time_series = pd.Series([10, 20, 30, 40, 50], index=time_coords)
        result = DataConverter.to_dataarray(time_series, coords={'time': time_coords, 'scenario': scenario_coords})
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, time_series.values)

    def test_series_wrong_dimension(self, time_coords, region_coords):
        """Series indexed by dimension not in coords should fail."""
        wrong_series = pd.Series([1, 2, 3], index=region_coords)

        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(wrong_series, coords={'time': time_coords})

    def test_series_broadcast_to_many_dimensions(self, standard_coords):
        """Series stays in minimal form with matched dimension only (no broadcast at conversion)."""
        time_series = pd.Series([100, 200, 300, 400, 500], index=standard_coords['time'])
        result = DataConverter.to_dataarray(time_series, coords=standard_coords)

        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, time_series.values)


class TestDataFrameConversion:
    """Test pandas DataFrame conversions."""

    def test_single_column_dataframe(self, time_coords):
        """Single-column DataFrame should work like Series."""
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50]}, index=time_coords)
        result = DataConverter.to_dataarray(df, coords={'time': time_coords})

        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, df['value'].values)

    def test_multi_column_dataframe_accepted(self, time_coords, scenario_coords):
        """Multi-column DataFrame should now be accepted and converted via numpy array path."""
        df = pd.DataFrame(
            {'value1': [10, 20, 30, 40, 50], 'value2': [15, 25, 35, 45, 55], 'value3': [12, 22, 32, 42, 52]},
            index=time_coords,
        )

        # Should work by converting to numpy array (5x3) and matching to time x scenario
        result = DataConverter.to_dataarray(df, coords={'time': time_coords, 'scenario': scenario_coords})

        assert result.shape == (5, 3)
        assert result.dims == ('time', 'scenario')
        assert np.array_equal(result.values, df.to_numpy())

    def test_empty_dataframe_rejected(self, time_coords):
        """Empty DataFrame should be rejected."""
        df = pd.DataFrame(index=time_coords)  # No columns

        with pytest.raises(ConversionError, match='DataFrame must have at least one column'):
            DataConverter.to_dataarray(df, coords={'time': time_coords})

    def test_dataframe_broadcast(self, time_coords, scenario_coords):
        """Single-column DataFrame stays 1D (no broadcasting at conversion time)."""
        df = pd.DataFrame({'power': [10, 20, 30, 40, 50]}, index=time_coords)
        result = DataConverter.to_dataarray(df, coords={'time': time_coords, 'scenario': scenario_coords})

        # Data stays in minimal form - 1D along time
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, df['power'].values)


class TestMultiDimensionalArrayConversion:
    """Test multi-dimensional numpy array conversions."""

    def test_2d_array_unique_dimensions(self, standard_coords):
        """2D array with unique dimension lengths should work."""
        # 5x3 array should map to time x scenario
        data_2d = np.random.rand(5, 3)
        result = DataConverter.to_dataarray(
            data_2d, coords={'time': standard_coords['time'], 'scenario': standard_coords['scenario']}
        )

        assert result.shape == (5, 3)
        assert result.dims == ('time', 'scenario')
        assert np.array_equal(result.values, data_2d)

        # 3x5 array should map to scenario x time, then transpose to canonical (time, scenario) order
        data_2d_flipped = np.random.rand(3, 5)
        result_flipped = DataConverter.to_dataarray(
            data_2d_flipped, coords={'time': standard_coords['time'], 'scenario': standard_coords['scenario']}
        )

        assert result_flipped.shape == (5, 3)
        assert result_flipped.dims == ('time', 'scenario')
        assert np.array_equal(result_flipped.values.transpose(), data_2d_flipped)

    def test_2d_array_stays_2d(self, standard_coords):
        """2D array stays 2D even when more coords are provided (no broadcasting)."""
        # 2D array (5x3) matches time×scenario, stays 2D (no broadcast to region)
        data_2d = np.random.rand(5, 3)
        result = DataConverter.to_dataarray(data_2d, coords=standard_coords)

        assert result.shape == (5, 3)
        assert result.dims == ('time', 'scenario')
        assert np.array_equal(result.values, data_2d)

    def test_3d_array_unique_dimensions(self, standard_coords):
        """3D array with unique dimension lengths should work."""
        # 5x3x2 array should map to time x scenario x region
        data_3d = np.random.rand(5, 3, 2)
        result = DataConverter.to_dataarray(data_3d, coords=standard_coords)

        assert result.shape == (5, 3, 2)
        assert result.dims == ('time', 'scenario', 'region')
        assert np.array_equal(result.values, data_3d)

    def test_3d_array_different_permutation(self, standard_coords):
        """3D array with different dimension order should be transposed to canonical order."""
        # 2x5x3 array should map to region x time x scenario, then transpose to canonical order
        data_3d = np.random.rand(2, 5, 3)
        result = DataConverter.to_dataarray(data_3d, coords=standard_coords)

        assert result.shape == (5, 3, 2)
        assert result.dims == ('time', 'scenario', 'region')
        assert np.array_equal(result.transpose('region', 'time', 'scenario').values, data_3d)

    def test_4d_array_unique_dimensions(self):
        """4D array with unique dimension lengths should work."""
        coords = {
            'time': pd.date_range('2024-01-01', periods=2, freq='D', name='time'),  # length 2
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),  # length 3
            'region': pd.Index(['north', 'south', 'east', 'west'], name='region'),  # length 4
            'technology': pd.Index(['solar', 'wind', 'gas', 'coal', 'hydro'], name='technology'),  # length 5
        }

        # 3x5x2x4 array should map to scenario x technology x time x region
        data_4d = np.random.rand(3, 5, 2, 4)
        result = DataConverter.to_dataarray(data_4d, coords=coords)

        assert result.shape == (2, 3, 4, 5)
        assert result.dims == ('time', 'scenario', 'region', 'technology')
        assert np.array_equal(result.transpose('scenario', 'technology', 'time', 'region').values, data_4d)

    def test_2d_array_ambiguous_dimensions_error(self):
        """2D array with ambiguous dimension lengths should fail."""
        # Both dimensions have length 3
        coords_ambiguous = {
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),  # length 3
            'region': pd.Index(['north', 'south', 'east'], name='region'),  # length 3
        }

        data_2d = np.random.rand(3, 3)
        with pytest.raises(ConversionError, match='matches multiple dimension combinations'):
            DataConverter.to_dataarray(data_2d, coords=coords_ambiguous)

    def test_multid_array_no_coords(self):
        """Multi-D arrays without coords should fail unless scalar."""
        # Multi-element fails
        data_2d = np.random.rand(2, 3)
        with pytest.raises(ConversionError, match='Cannot convert multi-element array without target dimensions'):
            DataConverter.to_dataarray(data_2d)

        # Single element succeeds
        single_element = np.array([[42]])
        result = DataConverter.to_dataarray(single_element)
        assert result.shape == ()
        assert result.item() == 42

    def test_array_no_matching_dimensions_error(self, standard_coords):
        """Array with no matching dimension lengths should fail."""
        # 7x8 array - no dimension has length 7 or 8
        data_2d = np.random.rand(7, 8)
        coords_2d = {
            'time': standard_coords['time'],  # length 5
            'scenario': standard_coords['scenario'],  # length 3
        }

        with pytest.raises(ConversionError, match='cannot be mapped to any combination'):
            DataConverter.to_dataarray(data_2d, coords=coords_2d)

    def test_multid_array_special_values(self, standard_coords):
        """Multi-D arrays should preserve special values."""
        # Create 2D array with special values
        data_2d = np.array(
            [[1.0, np.nan, 3.0], [np.inf, 5.0, -np.inf], [7.0, 8.0, 9.0], [10.0, np.nan, 12.0], [13.0, 14.0, np.inf]]
        )

        result = DataConverter.to_dataarray(
            data_2d, coords={'time': standard_coords['time'], 'scenario': standard_coords['scenario']}
        )

        assert result.shape == (5, 3)
        assert np.array_equal(np.isnan(result.values), np.isnan(data_2d))
        assert np.array_equal(np.isinf(result.values), np.isinf(data_2d))

    def test_multid_array_dtype_preservation(self, standard_coords):
        """Multi-D arrays should preserve data types."""
        # Integer array
        int_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=np.int32)

        result_int = DataConverter.to_dataarray(
            int_data, coords={'time': standard_coords['time'], 'scenario': standard_coords['scenario']}
        )

        assert result_int.dtype == np.int32
        assert np.array_equal(result_int.values, int_data)

        # Boolean array
        bool_data = np.array(
            [[True, False, True], [False, True, False], [True, True, False], [False, False, True], [True, False, True]]
        )

        result_bool = DataConverter.to_dataarray(
            bool_data, coords={'time': standard_coords['time'], 'scenario': standard_coords['scenario']}
        )

        assert result_bool.dtype == bool
        assert np.array_equal(result_bool.values, bool_data)


class TestDataArrayConversion:
    """Test xarray DataArray conversions."""

    def test_compatible_dataarray(self, time_coords):
        """Compatible DataArray should pass through."""
        original = xr.DataArray([10, 20, 30, 40, 50], coords={'time': time_coords}, dims='time')
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
        original = xr.DataArray([10, 20, 30, 40, 50], coords={'time': wrong_times}, dims='time')

        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(original, coords={'time': time_coords})

    def test_incompatible_dataarray_dims(self, time_coords):
        """DataArray with wrong dimensions should fail."""
        original = xr.DataArray([10, 20, 30, 40, 50], coords={'wrong_dim': range(5)}, dims='wrong_dim')

        with pytest.raises(ConversionError):
            DataConverter.to_dataarray(original, coords={'time': time_coords})

    def test_dataarray_no_broadcast(self, time_coords, scenario_coords):
        """DataArray stays in minimal form (no broadcasting at conversion time)."""
        # 1D time DataArray stays 1D even with additional coords
        original = xr.DataArray([10, 20, 30, 40, 50], coords={'time': time_coords}, dims='time')
        result = DataConverter.to_dataarray(original, coords={'time': time_coords, 'scenario': scenario_coords})

        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, original.values)

    def test_scalar_dataarray_stays_scalar(self, time_coords, scenario_coords):
        """Scalar DataArray stays scalar (no broadcasting at conversion time)."""
        scalar_da = xr.DataArray(42)
        result = DataConverter.to_dataarray(scalar_da, coords={'time': time_coords, 'scenario': scenario_coords})

        assert result.shape == ()
        assert result.item() == 42

    def test_2d_dataarray_stays_2d(self, standard_coords):
        """DataArray stays in minimal form (no broadcasting at conversion time)."""
        # Start with 2D DataArray
        original = xr.DataArray(
            [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120], [130, 140, 150]],
            coords={'time': standard_coords['time'], 'scenario': standard_coords['scenario']},
            dims=('time', 'scenario'),
        )

        # Stays 2D (no broadcast to 3D)
        result = DataConverter.to_dataarray(original, coords=standard_coords)

        assert result.shape == (5, 3)
        assert result.dims == ('time', 'scenario')
        assert np.array_equal(result.values, original.values)


class TestTimeSeriesDataConversion:
    """Test TimeSeriesData conversions."""

    def test_timeseries_data_basic(self, time_coords):
        """TimeSeriesData should work like DataArray."""
        data_array = xr.DataArray([10, 20, 30, 40, 50], coords={'time': time_coords}, dims='time')
        ts_data = TimeSeriesData(data_array, clustering_group='test')

        result = DataConverter.to_dataarray(ts_data, coords={'time': time_coords})

        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, [10, 20, 30, 40, 50])

    def test_timeseries_data_stays_minimal(self, time_coords, scenario_coords):
        """TimeSeriesData stays in minimal form (no broadcasting at conversion time)."""
        data_array = xr.DataArray([10, 20, 30, 40, 50], coords={'time': time_coords}, dims='time')
        ts_data = TimeSeriesData(data_array)

        result = DataConverter.to_dataarray(ts_data, coords={'time': time_coords, 'scenario': scenario_coords})

        # Stays 1D (time only)
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, [10, 20, 30, 40, 50])


class TestAsDataArrayAlias:
    """Test that to_dataarray works as an alias for to_dataarray."""

    def test_to_dataarray_is_alias(self, time_coords, scenario_coords):
        """to_dataarray should work identically to to_dataarray."""
        # Test with scalar
        result_to = DataConverter.to_dataarray(42, coords={'time': time_coords})
        result_as = DataConverter.to_dataarray(42, coords={'time': time_coords})
        assert np.array_equal(result_to.values, result_as.values)
        assert result_to.dims == result_as.dims
        assert result_to.shape == result_as.shape

        # Test with array
        arr = np.array([10, 20, 30, 40, 50])
        result_to_arr = DataConverter.to_dataarray(arr, coords={'time': time_coords})
        result_as_arr = DataConverter.to_dataarray(arr, coords={'time': time_coords})
        assert np.array_equal(result_to_arr.values, result_as_arr.values)
        assert result_to_arr.dims == result_as_arr.dims

        # Test with Series
        series = pd.Series([100, 200, 300, 400, 500], index=time_coords)
        result_to_series = DataConverter.to_dataarray(series, coords={'time': time_coords, 'scenario': scenario_coords})
        result_as_series = DataConverter.to_dataarray(series, coords={'time': time_coords, 'scenario': scenario_coords})
        assert np.array_equal(result_to_series.values, result_as_series.values)
        assert result_to_series.dims == result_as_series.dims


class TestCustomDimensions:
    """Test with custom dimension names beyond time/scenario."""

    def test_custom_single_dimension(self, region_coords):
        """Scalar stays scalar even with custom dimension."""
        result = DataConverter.to_dataarray(42, coords={'region': region_coords})
        assert result.shape == ()
        assert result.dims == ()
        assert result.item() == 42

    def test_custom_multiple_dimensions(self):
        """Array with matching dimension stays 1D (no broadcasting)."""
        products = pd.Index(['A', 'B'], name='product')
        technologies = pd.Index(['solar', 'wind', 'gas'], name='technology')

        # Array matching technology dimension stays 1D
        arr = np.array([100, 150, 80])
        result = DataConverter.to_dataarray(arr, coords={'product': products, 'technology': technologies})

        assert result.shape == (3,)
        assert result.dims == ('technology',)
        assert np.array_equal(result.values, arr)

    def test_mixed_dimension_types(self):
        """Time series stays 1D (no broadcasting to regions)."""
        time_coords = pd.date_range('2024-01-01', periods=3, freq='D', name='time')
        regions = pd.Index(['north', 'south'], name='region')

        time_series = pd.Series([10, 20, 30], index=time_coords)
        result = DataConverter.to_dataarray(time_series, coords={'time': time_coords, 'region': regions})

        # Stays 1D along time
        assert result.shape == (3,)
        assert result.dims == ('time',)

    def test_custom_dimensions_complex(self):
        """Array with matching dimension stays 1D (no broadcasting)."""
        coords = {
            'product': pd.Index(['A', 'B'], name='product'),
            'factory': pd.Index(['F1', 'F2', 'F3'], name='factory'),
            'quarter': pd.Index(['Q1', 'Q2', 'Q3', 'Q4'], name='quarter'),
        }

        # Array matching factory dimension stays 1D
        factory_arr = np.array([100, 200, 300])
        result = DataConverter.to_dataarray(factory_arr, coords=coords)

        assert result.shape == (3,)
        assert result.dims == ('factory',)
        assert np.array_equal(result.values, factory_arr)


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
        with pytest.raises(ConversionError, match='DatetimeIndex'):
            DataConverter.to_dataarray(42, coords={'time': wrong_time})

    def test_coord_naming(self, time_coords):
        """Scalar with coords stays scalar but validates coords."""
        # Scalar stays scalar regardless of coords
        result = DataConverter.to_dataarray(42, coords={'time': time_coords})
        assert result.shape == ()
        assert result.item() == 42


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_unsupported_data_types(self, time_coords):
        """Unsupported data types should fail with clear messages."""
        unsupported = ['string', object(), None, {'dict': 'value'}, [1, 2, 3]]

        for data in unsupported:
            with pytest.raises(ConversionError):
                DataConverter.to_dataarray(data, coords={'time': time_coords})

    def test_dimension_mismatch_messages(self, time_coords, scenario_coords):
        """Error messages should be informative."""
        # Array with wrong length
        wrong_arr = np.array([1, 2])  # Length 2, but no dimension has length 2
        with pytest.raises(ConversionError, match='does not match any target dimension lengths'):
            DataConverter.to_dataarray(wrong_arr, coords={'time': time_coords, 'scenario': scenario_coords})

    def test_multidimensional_array_dimension_count_mismatch(self, standard_coords):
        """Array with wrong number of dimensions should fail with clear error."""
        # 4D array with 3D coordinates
        data_4d = np.random.rand(5, 3, 2, 4)
        with pytest.raises(ConversionError, match='cannot be mapped to any combination'):
            DataConverter.to_dataarray(data_4d, coords=standard_coords)

    def test_error_message_quality(self, standard_coords):
        """Error messages should include helpful information."""
        # Wrong shape array
        data_2d = np.random.rand(7, 8)
        coords_2d = {
            'time': standard_coords['time'],  # length 5
            'scenario': standard_coords['scenario'],  # length 3
        }

        try:
            DataConverter.to_dataarray(data_2d, coords=coords_2d)
            raise AssertionError('Should have raised ConversionError')
        except ConversionError as e:
            error_msg = str(e)
            assert 'Array shape (7, 8)' in error_msg
            assert 'target coordinate lengths:' in error_msg


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

    def test_multid_array_copy_independence(self, standard_coords):
        """Multi-D arrays should be independent copies."""
        original_data = np.random.rand(5, 3)
        result = DataConverter.to_dataarray(
            original_data, coords={'time': standard_coords['time'], 'scenario': standard_coords['scenario']}
        )

        # Modify result
        result[0, 0] = 999

        # Original should be unchanged
        assert original_data[0, 0] != 999


class TestBooleanValues:
    """Test handling of boolean values and arrays."""

    def test_scalar_boolean_to_dataarray(self, time_coords):
        """Scalar boolean values stay scalar (no broadcasting)."""
        result_true = DataConverter.to_dataarray(True, coords={'time': time_coords})
        assert result_true.shape == ()
        assert result_true.dtype == bool
        assert result_true.item() is True

        result_false = DataConverter.to_dataarray(False, coords={'time': time_coords})
        assert result_false.shape == ()
        assert result_false.dtype == bool
        assert result_false.item() is False

    def test_numpy_boolean_scalar(self, time_coords):
        """Numpy boolean scalars stay scalar (no broadcasting)."""
        result_np_true = DataConverter.to_dataarray(np.bool_(True), coords={'time': time_coords})
        assert result_np_true.shape == ()
        assert result_np_true.dtype == bool
        assert result_np_true.item() is True

        result_np_false = DataConverter.to_dataarray(np.bool_(False), coords={'time': time_coords})
        assert result_np_false.shape == ()
        assert result_np_false.dtype == bool
        assert result_np_false.item() is False

    def test_boolean_array_to_dataarray(self, time_coords):
        """Boolean arrays should work with to_dataarray."""
        bool_arr = np.array([True, False, True, False, True])
        result = DataConverter.to_dataarray(bool_arr, coords={'time': time_coords})
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert result.dtype == bool
        assert np.array_equal(result.values, bool_arr)

    def test_boolean_no_coords(self):
        """Boolean scalar without coordinates should create 0D DataArray."""
        result = DataConverter.to_dataarray(True)
        assert result.shape == ()
        assert result.dims == ()
        assert result.item()

        result_as = DataConverter.to_dataarray(False)
        assert result_as.shape == ()
        assert result_as.dims == ()
        assert not result_as.item()

    def test_boolean_scalar_stays_scalar(self, standard_coords):
        """Boolean scalars stay scalar (no broadcasting)."""
        result = DataConverter.to_dataarray(True, coords=standard_coords)
        assert result.shape == ()
        assert result.dtype == bool
        assert result.item() is True

        result_as = DataConverter.to_dataarray(False, coords=standard_coords)
        assert result_as.shape == ()
        assert result_as.dtype == bool
        assert result_as.item() is False

    def test_boolean_series(self, time_coords):
        """Boolean Series should work."""
        bool_series = pd.Series([True, False, True, False, True], index=time_coords)
        result = DataConverter.to_dataarray(bool_series, coords={'time': time_coords})
        assert result.shape == (5,)
        assert result.dtype == bool
        assert np.array_equal(result.values, bool_series.values)

        result_as = DataConverter.to_dataarray(bool_series, coords={'time': time_coords})
        assert result_as.shape == (5,)
        assert result_as.dtype == bool
        assert np.array_equal(result_as.values, bool_series.values)

    def test_boolean_dataframe(self, time_coords):
        """Boolean DataFrame should work."""
        bool_df = pd.DataFrame({'values': [True, False, True, False, True]}, index=time_coords)
        result = DataConverter.to_dataarray(bool_df, coords={'time': time_coords})
        assert result.shape == (5,)
        assert result.dtype == bool
        assert np.array_equal(result.values, bool_df['values'].values)

        result_as = DataConverter.to_dataarray(bool_df, coords={'time': time_coords})
        assert result_as.shape == (5,)
        assert result_as.dtype == bool
        assert np.array_equal(result_as.values, bool_df['values'].values)

    def test_multidimensional_boolean_array(self, standard_coords):
        """Multi-dimensional boolean arrays should work."""
        bool_data = np.array(
            [[True, False, True], [False, True, False], [True, True, False], [False, False, True], [True, False, True]]
        )
        result = DataConverter.to_dataarray(
            bool_data, coords={'time': standard_coords['time'], 'scenario': standard_coords['scenario']}
        )
        assert result.shape == (5, 3)
        assert result.dtype == bool
        assert np.array_equal(result.values, bool_data)

        result_as = DataConverter.to_dataarray(
            bool_data, coords={'time': standard_coords['time'], 'scenario': standard_coords['scenario']}
        )
        assert result_as.shape == (5, 3)
        assert result_as.dtype == bool
        assert np.array_equal(result_as.values, bool_data)


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

    def test_special_values_in_1d_arrays(self, standard_coords):
        """Special values should be preserved in arrays (no broadcasting)."""
        # Array with NaN and inf
        special_arr = np.array([1, np.nan, np.inf, -np.inf, 5])
        result = DataConverter.to_dataarray(special_arr, coords=standard_coords)

        # Stays 1D (no broadcasting)
        assert result.shape == (5,)
        assert result.dims == ('time',)

        # Check that special values are preserved
        assert np.array_equal(np.isnan(result.values), np.isnan(special_arr))
        assert np.array_equal(np.isinf(result.values), np.isinf(special_arr))


class TestAdvancedBroadcasting:
    """Test advanced scenarios (no broadcasting - data stays minimal)."""

    def test_partial_dimension_matching_stays_1d(self, standard_coords):
        """1D array stays 1D (no broadcasting to additional dimensions)."""
        time_arr = np.array([10, 20, 30, 40, 50])  # matches time (length 5)
        result = DataConverter.to_dataarray(time_arr, coords=standard_coords)

        # Stays 1D
        assert result.shape == (5,)
        assert result.dims == ('time',)
        assert np.array_equal(result.values, time_arr)

    def test_complex_multid_scenario(self):
        """2D array stays 2D (no broadcasting to additional dimensions)."""
        coords = {
            'time': pd.date_range('2024-01-01', periods=24, freq='h', name='time'),  # 24 hours
            'technology': pd.Index(['solar', 'wind', 'gas', 'coal'], name='technology'),  # 4 technologies
            'region': pd.Index(['north', 'south', 'east'], name='region'),  # 3 regions
        }

        # Capacity factors: 24 x 4 stays 2D (no broadcast to 24 x 4 x 3)
        capacity_factors = np.random.rand(24, 4)

        result = DataConverter.to_dataarray(capacity_factors, coords=coords)

        assert result.shape == (24, 4)
        assert result.dims == ('time', 'technology')
        assert isinstance(result.indexes['time'], pd.DatetimeIndex)
        assert np.array_equal(result.values, capacity_factors)

    def test_ambiguous_length_handling(self):
        """Test handling of ambiguous length scenarios across different data types."""
        # All dimensions have length 3
        coords_3x3x3 = {
            'time': pd.date_range('2024-01-01', periods=3, freq='D', name='time'),
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),
            'region': pd.Index(['X', 'Y', 'Z'], name='region'),
        }

        # 1D array - should fail
        arr_1d = np.array([1, 2, 3])
        with pytest.raises(ConversionError, match='matches multiple dimension'):
            DataConverter.to_dataarray(arr_1d, coords=coords_3x3x3)

        # 2D array - should fail
        arr_2d = np.random.rand(3, 3)
        with pytest.raises(ConversionError, match='matches multiple dimension'):
            DataConverter.to_dataarray(arr_2d, coords=coords_3x3x3)

        # 3D array - should fail
        arr_3d = np.random.rand(3, 3, 3)
        with pytest.raises(ConversionError, match='matches multiple dimension'):
            DataConverter.to_dataarray(arr_3d, coords=coords_3x3x3)

    def test_no_broadcasting_scenarios(self):
        """Data stays in minimal form (no broadcasting to additional dimensions)."""
        coords = {
            'time': pd.date_range('2024-01-01', periods=4, freq='D', name='time'),  # length 4
            'scenario': pd.Index(['A', 'B'], name='scenario'),  # length 2
            'region': pd.Index(['north', 'south', 'east'], name='region'),  # length 3
            'product': pd.Index(['X', 'Y', 'Z', 'W', 'V'], name='product'),  # length 5
        }

        # Scalar stays scalar
        scalar_result = DataConverter.to_dataarray(42, coords=coords)
        assert scalar_result.shape == ()
        assert scalar_result.item() == 42

        # 1D array (length 4, matches time) stays 1D
        arr_1d = np.array([10, 20, 30, 40])
        arr_result = DataConverter.to_dataarray(arr_1d, coords=coords)
        assert arr_result.shape == (4,)
        assert arr_result.dims == ('time',)
        assert np.array_equal(arr_result.values, arr_1d)

        # 2D array (4x2, matches time×scenario) stays 2D
        arr_2d = np.random.rand(4, 2)
        arr_2d_result = DataConverter.to_dataarray(arr_2d, coords=coords)
        assert arr_2d_result.shape == (4, 2)
        assert arr_2d_result.dims == ('time', 'scenario')
        assert np.array_equal(arr_2d_result.values, arr_2d)


class TestAmbiguousDimensionLengthHandling:
    """Test that DataConverter correctly raises errors when multiple dimensions have the same length."""

    def test_1d_array_ambiguous_dimensions_simple(self):
        """Test 1D array with two dimensions of same length should fail."""
        # Both dimensions have length 3
        coords_ambiguous = {
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),  # length 3
            'region': pd.Index(['north', 'south', 'east'], name='region'),  # length 3
        }

        arr_1d = np.array([1, 2, 3])  # length 3 - matches both dimensions

        with pytest.raises(ConversionError, match='matches multiple dimension'):
            DataConverter.to_dataarray(arr_1d, coords=coords_ambiguous)

    def test_1d_array_ambiguous_dimensions_complex(self):
        """Test 1D array with multiple dimensions of same length."""
        # Three dimensions have length 4
        coords_4x4x4 = {
            'time': pd.date_range('2024-01-01', periods=4, freq='D', name='time'),  # length 4
            'scenario': pd.Index(['A', 'B', 'C', 'D'], name='scenario'),  # length 4
            'region': pd.Index(['north', 'south', 'east', 'west'], name='region'),  # length 4
            'product': pd.Index(['X', 'Y'], name='product'),  # length 2 - unique
        }

        # Array matching the ambiguous length
        arr_1d = np.array([10, 20, 30, 40])  # length 4 - matches time, scenario, region

        with pytest.raises(ConversionError, match='matches multiple dimension'):
            DataConverter.to_dataarray(arr_1d, coords=coords_4x4x4)

        # Array matching the unique length should work (stays 1D, no broadcasting)
        arr_1d_unique = np.array([100, 200])  # length 2 - matches only product
        result = DataConverter.to_dataarray(arr_1d_unique, coords=coords_4x4x4)
        assert result.shape == (2,)  # stays 1D
        assert result.dims == ('product',)

    def test_2d_array_ambiguous_dimensions_both_same(self):
        """Test 2D array where both dimensions have the same ambiguous length."""
        # All dimensions have length 3
        coords_3x3x3 = {
            'time': pd.date_range('2024-01-01', periods=3, freq='D', name='time'),  # length 3
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),  # length 3
            'region': pd.Index(['X', 'Y', 'Z'], name='region'),  # length 3
        }

        # 3x3 array - could be any combination of the three dimensions
        arr_2d = np.random.rand(3, 3)

        with pytest.raises(ConversionError, match='matches multiple dimension'):
            DataConverter.to_dataarray(arr_2d, coords=coords_3x3x3)

    def test_2d_array_one_dimension_ambiguous(self):
        """Test 2D array where only one dimension length is ambiguous."""
        coords_mixed = {
            'time': pd.date_range('2024-01-01', periods=5, freq='D', name='time'),  # length 5 - unique
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),  # length 3
            'region': pd.Index(['X', 'Y', 'Z'], name='region'),  # length 3 - same as scenario
            'product': pd.Index(['P1', 'P2'], name='product'),  # length 2 - unique
        }

        # 5x3 array - first dimension clearly maps to time (unique length 5)
        # but second dimension could be scenario or region (both length 3)
        arr_5x3 = np.random.rand(5, 3)

        with pytest.raises(ConversionError, match='matches multiple dimension'):
            DataConverter.to_dataarray(arr_5x3, coords=coords_mixed)

        # 5x2 array should work - dimensions are unambiguous
        arr_5x2 = np.random.rand(5, 2)
        result = DataConverter.to_dataarray(
            arr_5x2, coords={'time': coords_mixed['time'], 'product': coords_mixed['product']}
        )
        assert result.shape == (5, 2)
        assert result.dims == ('time', 'product')

    def test_3d_array_all_dimensions_ambiguous(self):
        """Test 3D array where all dimension lengths are ambiguous."""
        # All dimensions have length 2
        coords_2x2x2x2 = {
            'scenario': pd.Index(['A', 'B'], name='scenario'),  # length 2
            'region': pd.Index(['north', 'south'], name='region'),  # length 2
            'technology': pd.Index(['solar', 'wind'], name='technology'),  # length 2
            'product': pd.Index(['X', 'Y'], name='product'),  # length 2
        }

        # 2x2x2 array - could be any combination of 3 dimensions from the 4 available
        arr_3d = np.random.rand(2, 2, 2)

        with pytest.raises(ConversionError, match='matches multiple dimension'):
            DataConverter.to_dataarray(arr_3d, coords=coords_2x2x2x2)

    def test_3d_array_partial_ambiguity(self):
        """Test 3D array with partial dimension ambiguity."""
        coords_partial = {
            'time': pd.date_range('2024-01-01', periods=4, freq='D', name='time'),  # length 4 - unique
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),  # length 3
            'region': pd.Index(['X', 'Y', 'Z'], name='region'),  # length 3 - same as scenario
            'technology': pd.Index(['solar', 'wind'], name='technology'),  # length 2 - unique
        }

        # 4x3x2 array - first and third dimensions are unique, middle is ambiguous
        # This should still fail because middle dimension (length 3) could be scenario or region
        arr_4x3x2 = np.random.rand(4, 3, 2)

        with pytest.raises(ConversionError, match='matches multiple dimension'):
            DataConverter.to_dataarray(arr_4x3x2, coords=coords_partial)

    def test_pandas_series_ambiguous_dimensions(self):
        """Test pandas Series with ambiguous dimension lengths."""
        coords_ambiguous = {
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),  # length 3
            'region': pd.Index(['north', 'south', 'east'], name='region'),  # length 3
        }

        # Series with length 3 but index that doesn't match either coordinate exactly
        generic_series = pd.Series([10, 20, 30], index=[0, 1, 2])

        # Should fail because length matches multiple dimensions and index doesn't match any
        with pytest.raises(ConversionError, match='Series index does not match any target dimension coordinates'):
            DataConverter.to_dataarray(generic_series, coords=coords_ambiguous)

        # Series with index that matches one of the ambiguous coordinates should work (stays 1D)
        scenario_series = pd.Series([10, 20, 30], index=coords_ambiguous['scenario'])
        result = DataConverter.to_dataarray(scenario_series, coords=coords_ambiguous)
        assert result.shape == (3,)  # stays 1D
        assert result.dims == ('scenario',)

    def test_edge_case_many_same_lengths(self):
        """Test edge case with many dimensions having the same length."""
        # Five dimensions all have length 2
        coords_many = {
            'dim1': pd.Index(['A', 'B'], name='dim1'),
            'dim2': pd.Index(['X', 'Y'], name='dim2'),
            'dim3': pd.Index(['P', 'Q'], name='dim3'),
            'dim4': pd.Index(['M', 'N'], name='dim4'),
            'dim5': pd.Index(['U', 'V'], name='dim5'),
        }

        # 1D array
        arr_1d = np.array([1, 2])
        with pytest.raises(ConversionError, match='matches multiple dimension'):
            DataConverter.to_dataarray(arr_1d, coords=coords_many)

        # 2D array
        arr_2d = np.random.rand(2, 2)
        with pytest.raises(ConversionError, match='matches multiple dimension'):
            DataConverter.to_dataarray(arr_2d, coords=coords_many)

        # 3D array
        arr_3d = np.random.rand(2, 2, 2)
        with pytest.raises(ConversionError, match='matches multiple dimension'):
            DataConverter.to_dataarray(arr_3d, coords=coords_many)

    def test_mixed_lengths_with_duplicates(self):
        """Test mixed scenario with some duplicate and some unique lengths."""
        coords_mixed = {
            'time': pd.date_range('2024-01-01', periods=8, freq='D', name='time'),  # length 8 - unique
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),  # length 3
            'region': pd.Index(['X', 'Y', 'Z'], name='region'),  # length 3 - same as scenario
            'technology': pd.Index(['solar'], name='technology'),  # length 1 - unique
            'product': pd.Index(['P1', 'P2', 'P3', 'P4', 'P5'], name='product'),  # length 5 - unique
        }

        # Arrays with unique lengths should work (stays minimal, no broadcasting)
        arr_8 = np.arange(8)
        result_8 = DataConverter.to_dataarray(arr_8, coords=coords_mixed)
        assert result_8.dims == ('time',)  # Stays 1D, matches time dimension

        arr_1 = np.array([42])
        result_1 = DataConverter.to_dataarray(arr_1, coords={'technology': coords_mixed['technology']})
        assert result_1.shape == (1,)

        arr_5 = np.arange(5)
        result_5 = DataConverter.to_dataarray(arr_5, coords={'product': coords_mixed['product']})
        assert result_5.shape == (5,)

        # Arrays with ambiguous length should fail
        arr_3 = np.array([1, 2, 3])  # matches both scenario and region
        with pytest.raises(ConversionError, match='matches multiple dimension'):
            DataConverter.to_dataarray(arr_3, coords=coords_mixed)

    def test_dataframe_with_ambiguous_dimensions(self):
        """Test DataFrame handling with ambiguous dimensions."""
        coords_ambiguous = {
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),  # length 3
            'region': pd.Index(['X', 'Y', 'Z'], name='region'),  # length 3
        }

        # Multi-column DataFrame with ambiguous dimensions
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9]})  # 3x3 DataFrame

        # Should fail due to ambiguous dimensions
        with pytest.raises(ConversionError, match='matches multiple dimension'):
            DataConverter.to_dataarray(df, coords=coords_ambiguous)

    def test_error_message_quality_for_ambiguous_dimensions(self):
        """Test that error messages for ambiguous dimensions are helpful."""
        coords_ambiguous = {
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),
            'region': pd.Index(['north', 'south', 'east'], name='region'),
            'technology': pd.Index(['solar', 'wind', 'gas'], name='technology'),
        }

        # 1D array case
        arr_1d = np.array([1, 2, 3])
        try:
            DataConverter.to_dataarray(arr_1d, coords=coords_ambiguous)
            raise AssertionError('Should have raised ConversionError')
        except ConversionError as e:
            error_msg = str(e)
            assert 'matches multiple dimension' in error_msg
            assert 'scenario' in error_msg
            assert 'region' in error_msg
            assert 'technology' in error_msg

        # 2D array case
        arr_2d = np.random.rand(3, 3)
        try:
            DataConverter.to_dataarray(arr_2d, coords=coords_ambiguous)
            raise AssertionError('Should have raised ConversionError')
        except ConversionError as e:
            error_msg = str(e)
            assert 'matches multiple dimension combinations' in error_msg
            assert '(3, 3)' in error_msg

    def test_ambiguous_with_broadcasting_target(self):
        """Test ambiguous dimensions when target includes broadcasting."""
        coords_ambiguous_plus = {
            'time': pd.date_range('2024-01-01', periods=5, freq='D', name='time'),  # length 5
            'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),  # length 3
            'region': pd.Index(['X', 'Y', 'Z'], name='region'),  # length 3 - same as scenario
            'technology': pd.Index(['solar', 'wind'], name='technology'),  # length 2
        }

        # 1D array with ambiguous length, but targeting broadcast scenario
        arr_3 = np.array([10, 20, 30])  # length 3, matches scenario and region

        # Should fail even though it would broadcast to other dimensions
        with pytest.raises(ConversionError, match='matches multiple dimension'):
            DataConverter.to_dataarray(arr_3, coords=coords_ambiguous_plus)

        # 2D array with one ambiguous dimension
        arr_5x3 = np.random.rand(5, 3)  # 5 is unique (time), 3 is ambiguous (scenario/region)

        with pytest.raises(ConversionError, match='matches multiple dimension'):
            DataConverter.to_dataarray(arr_5x3, coords=coords_ambiguous_plus)

    def test_time_dimension_ambiguity(self):
        """Test ambiguity specifically involving time dimension."""
        # Create scenario where time has same length as another dimension
        coords_time_ambiguous = {
            'time': pd.date_range('2024-01-01', periods=3, freq='D', name='time'),  # length 3
            'scenario': pd.Index(['base', 'high', 'low'], name='scenario'),  # length 3 - same as time
            'region': pd.Index(['north', 'south'], name='region'),  # length 2 - unique
        }

        # Time-indexed series should work even with ambiguous lengths (index matching takes precedence)
        # Stays minimal - no broadcasting to other dimensions
        time_series = pd.Series([100, 200, 300], index=coords_time_ambiguous['time'])
        result = DataConverter.to_dataarray(time_series, coords=coords_time_ambiguous)
        assert result.shape == (3,)  # Stays 1D
        assert result.dims == ('time',)  # Matches time via index

        # But generic array with length 3 should still fail
        generic_array = np.array([100, 200, 300])
        with pytest.raises(ConversionError, match='matches multiple dimension'):
            DataConverter.to_dataarray(generic_array, coords=coords_time_ambiguous)


if __name__ == '__main__':
    pytest.main()
