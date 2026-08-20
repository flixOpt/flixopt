"""Test reshape_data_for_heatmap() for common use cases."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt.plotting import reshape_data_for_heatmap

# Set random seed for reproducible tests
np.random.seed(42)


@pytest.fixture
def hourly_week_data():
    """Typical use case: hourly data for a week."""
    time = pd.date_range('2024-01-01', periods=168, freq='h')
    data = np.random.rand(168) * 100
    return xr.DataArray(data, dims=['time'], coords={'time': time}, name='power')


def test_daily_hourly_pattern():
    """Most common use case: reshape hourly data into days × hours for daily patterns."""
    time = pd.date_range('2024-01-01', periods=72, freq='h')
    data = np.random.rand(72) * 100
    da = xr.DataArray(data, dims=['time'], coords={'time': time})

    result = reshape_data_for_heatmap(da, reshape_time=('D', 'h'))

    assert 'timeframe' in result.dims and 'timestep' in result.dims
    assert result.sizes['timeframe'] == 3  # 3 days
    assert result.sizes['timestep'] == 24  # 24 hours


def test_weekly_daily_pattern(hourly_week_data):
    """Common use case: reshape hourly data into weeks × days."""
    result = reshape_data_for_heatmap(hourly_week_data, reshape_time=('W', 'D'))

    assert 'timeframe' in result.dims and 'timestep' in result.dims
    # 168 hours = 7 days = 1 week
    assert result.sizes['timeframe'] == 1  # 1 week
    assert result.sizes['timestep'] == 7  # 7 days


def test_with_irregular_data():
    """Real-world use case: data with missing timestamps needs filling."""
    time = pd.date_range('2024-01-01', periods=100, freq='15min')
    data = np.random.rand(100)
    # Randomly drop 30% to simulate real data gaps
    keep = np.sort(np.random.choice(100, 70, replace=False))  # Must be sorted
    da = xr.DataArray(data[keep], dims=['time'], coords={'time': time[keep]})

    result = reshape_data_for_heatmap(da, reshape_time=('h', 'min'), fill='ffill')

    assert 'timeframe' in result.dims and 'timestep' in result.dims
    # 100 * 15min = 1500min = 25h; reshaped to hours × minutes
    assert result.sizes['timeframe'] == 25  # 25 hours
    assert result.sizes['timestep'] == 60  # 60 minutes per hour
    # Should handle irregular data without errors


def test_multidimensional_scenarios():
    """Use case: data with scenarios/periods that need to be preserved."""
    time = pd.date_range('2024-01-01', periods=48, freq='h')
    scenarios = ['base', 'high']
    data = np.random.rand(48, 2) * 100

    da = xr.DataArray(data, dims=['time', 'scenario'], coords={'time': time, 'scenario': scenarios}, name='demand')

    result = reshape_data_for_heatmap(da, reshape_time=('D', 'h'))

    # Should preserve scenario dimension
    assert 'scenario' in result.dims
    assert result.sizes['scenario'] == 2
    # 48 hours = 2 days × 24 hours
    assert result.sizes['timeframe'] == 2  # 2 days
    assert result.sizes['timestep'] == 24  # 24 hours


def test_no_reshape_returns_unchanged():
    """Use case: when reshape_time=None, return data as-is."""
    time = pd.date_range('2024-01-01', periods=24, freq='h')
    da = xr.DataArray(np.random.rand(24), dims=['time'], coords={'time': time})

    result = reshape_data_for_heatmap(da, reshape_time=None)

    xr.testing.assert_equal(result, da)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
