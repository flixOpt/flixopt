"""Tests for align_to_coords() and align_effects_to_coords()."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt.core import ConversionError, TimeSeriesData, align_effects_to_coords, align_to_coords


@pytest.fixture
def time_coords():
    """Standard time-only coordinates."""
    return {'time': pd.date_range('2020-01-01', periods=5, freq='h', name='time')}


@pytest.fixture
def full_coords():
    """Time + period + scenario coordinates."""
    return {
        'time': pd.date_range('2020-01-01', periods=5, freq='h', name='time'),
        'period': pd.Index([2020, 2030], name='period'),
        'scenario': pd.Index(['A', 'B', 'C'], name='scenario'),
    }


class TestAlignNone:
    def test_none_returns_none(self, time_coords):
        assert align_to_coords(None, time_coords) is None

    def test_none_with_name(self, time_coords):
        assert align_to_coords(None, time_coords, name='test') is None


class TestAlignScalar:
    def test_int(self, time_coords):
        result = align_to_coords(42, time_coords, name='val')
        assert isinstance(result, xr.DataArray)
        assert result.ndim == 0
        assert float(result) == 42.0

    def test_float(self, time_coords):
        result = align_to_coords(0.5, time_coords)
        assert result.ndim == 0
        assert float(result) == 0.5

    def test_bool(self, time_coords):
        result = align_to_coords(True, time_coords)
        assert result.ndim == 0

    def test_np_float(self, time_coords):
        result = align_to_coords(np.float64(3.14), time_coords)
        assert result.ndim == 0
        assert float(result) == pytest.approx(3.14)


class TestAlign1DArray:
    def test_numpy_array_matches_time(self, time_coords):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = align_to_coords(data, time_coords, name='profile')
        assert result.dims == ('time',)
        assert len(result) == 5
        np.testing.assert_array_equal(result.values, data)

    def test_wrong_length_raises(self, time_coords):
        data = np.array([1.0, 2.0, 3.0])  # length 3, time has 5
        with pytest.raises(ConversionError):
            align_to_coords(data, time_coords)

    def test_matches_period_dim(self, full_coords):
        data = np.array([10.0, 20.0])  # length 2 matches period
        result = align_to_coords(data, full_coords, dims=['period', 'scenario'])
        assert result.dims == ('period',)

    def test_matches_scenario_dim(self, full_coords):
        data = np.array([1.0, 2.0, 3.0])  # length 3 matches scenario
        result = align_to_coords(data, full_coords, dims=['period', 'scenario'])
        assert result.dims == ('scenario',)


class TestAlignSeries:
    def test_series_with_datetime_index(self, time_coords):
        idx = time_coords['time']
        data = pd.Series([10, 20, 30, 40, 50], index=idx)
        result = align_to_coords(data, time_coords)
        assert result.dims == ('time',)
        np.testing.assert_array_equal(result.values, [10, 20, 30, 40, 50])

    def test_series_wrong_index_raises(self, time_coords):
        wrong_idx = pd.date_range('2021-01-01', periods=5, freq='h')
        data = pd.Series([1, 2, 3, 4, 5], index=wrong_idx)
        with pytest.raises(ConversionError):
            align_to_coords(data, time_coords)


class TestAlignTimeSeriesData:
    def test_basic_timeseries(self, time_coords):
        data = TimeSeriesData([1, 2, 3, 4, 5])
        result = align_to_coords(data, time_coords, name='ts')
        assert isinstance(result, TimeSeriesData)
        assert result.dims == ('time',)

    def test_clustering_metadata_preserved(self, time_coords):
        data = TimeSeriesData([1, 2, 3, 4, 5], clustering_group='heat')
        result = align_to_coords(data, time_coords, name='ts')
        assert result.clustering_group == 'heat'

    def test_clustering_weight_preserved(self, time_coords):
        data = TimeSeriesData([1, 2, 3, 4, 5], clustering_weight=0.7)
        result = align_to_coords(data, time_coords, name='ts')
        assert result.clustering_weight == 0.7


class TestAlignDataArray:
    def test_already_aligned_passthrough(self, time_coords):
        idx = time_coords['time']
        da = xr.DataArray([1, 2, 3, 4, 5], dims=['time'], coords={'time': idx})
        result = align_to_coords(da, time_coords)
        xr.testing.assert_equal(result, da)

    def test_scalar_dataarray(self, time_coords):
        da = xr.DataArray(42.0)
        result = align_to_coords(da, time_coords)
        assert result.ndim == 0
        assert float(result) == 42.0

    def test_incompatible_dims_raises(self, time_coords):
        da = xr.DataArray([1, 2, 3], dims=['foo'])
        with pytest.raises(ConversionError):
            align_to_coords(da, time_coords)


class TestAlignDimsFilter:
    def test_dims_restricts_alignment(self, full_coords):
        data = np.array([10.0, 20.0])  # length 2 matches period
        result = align_to_coords(data, full_coords, dims=['period'])
        assert result.dims == ('period',)

    def test_dims_none_uses_all(self, time_coords):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = align_to_coords(data, time_coords, dims=None)
        assert result.dims == ('time',)


class TestAlignName:
    def test_name_assigned(self, time_coords):
        result = align_to_coords(42, time_coords, name='my_param')
        assert result.name == 'my_param'

    def test_no_name(self, time_coords):
        result = align_to_coords(42, time_coords)
        # Should not error, name may be None
        assert result is not None


class TestAlignEffects:
    def test_none_returns_none(self, time_coords):
        assert align_effects_to_coords(None, time_coords) is None

    def test_scalar_effects(self, time_coords):
        effects = {'costs': 0.04, 'CO2': 0.3}
        result = align_effects_to_coords(effects, time_coords, prefix='flow')
        assert set(result.keys()) == {'costs', 'CO2'}
        assert float(result['costs']) == pytest.approx(0.04)
        assert result['costs'].name == 'flow|costs'

    def test_array_effects(self, time_coords):
        effects = {'costs': np.array([1, 2, 3, 4, 5])}
        result = align_effects_to_coords(effects, time_coords)
        assert result['costs'].dims == ('time',)

    def test_prefix_suffix(self, time_coords):
        effects = {'costs': 42}
        result = align_effects_to_coords(effects, time_coords, prefix='Boiler', suffix='per_hour')
        assert result['costs'].name == 'Boiler|costs|per_hour'

    def test_empty_dict(self, time_coords):
        result = align_effects_to_coords({}, time_coords)
        assert result == {}
