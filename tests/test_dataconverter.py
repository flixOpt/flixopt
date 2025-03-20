import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixOpt.core import ConversionError, DataConverter  # Adjust this import to match your project structure


@pytest.fixture
def sample_time_index(request):
    index = pd.date_range("2024-01-01", periods=5, freq="D", name='time')
    return index


def test_scalar_conversion(sample_time_index):
    # Test scalar conversion
    result = DataConverter.as_dataarray(42, sample_time_index)
    assert isinstance(result, xr.DataArray)
    assert result.shape == (len(sample_time_index),)
    assert result.dims == ('time',)
    assert np.all(result.values == 42)


def test_series_conversion(sample_time_index):
    series = pd.Series([1, 2, 3, 4, 5], index=sample_time_index)

    # Test Series conversion
    result = DataConverter.as_dataarray(series, sample_time_index)
    assert isinstance(result, xr.DataArray)
    assert result.shape == (5,)
    assert result.dims == ('time',)
    assert np.array_equal(result.values, series.values)


def test_dataframe_conversion(sample_time_index):
    # Create a single-column DataFrame
    df = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5]},
        index=sample_time_index
    )

    # Test DataFrame conversion
    result = DataConverter.as_dataarray(df, sample_time_index)
    assert isinstance(result, xr.DataArray)
    assert result.shape == (5,)
    assert result.dims == ('time',)
    assert np.array_equal(result.values.flatten(), df['A'].values)


def test_ndarray_conversion(sample_time_index):
    # Test 1D array conversion
    arr_1d = np.array([1, 2, 3, 4, 5])
    result = DataConverter.as_dataarray(arr_1d, sample_time_index)
    assert result.shape == (5,)
    assert result.dims == ('time',)
    assert np.array_equal(result.values, arr_1d)


def test_dataarray_conversion(sample_time_index):
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


def test_invalid_inputs(sample_time_index):
    # Test invalid input type
    with pytest.raises(ConversionError):
        DataConverter.as_dataarray("invalid_string", sample_time_index)

    # Test mismatched Series index
    mismatched_series = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("2025-01-01", periods=6, freq="D"))
    with pytest.raises(ConversionError):
        DataConverter.as_dataarray(mismatched_series, sample_time_index)

    # Test DataFrame with multiple columns
    df_multi_col = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [6, 7, 8, 9, 10]
    }, index=sample_time_index)
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
    unnamed_index = pd.date_range("2024-01-01", periods=5, freq="D")
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


if __name__ == "__main__":
    pytest.main()
