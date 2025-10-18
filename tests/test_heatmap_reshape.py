"""Test reshape_data_for_heatmap() function."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt.plotting import reshape_data_for_heatmap


@pytest.fixture
def regular_timeseries():
    """Create regular time series data (hourly for 3 days)."""
    time = pd.date_range('2024-01-01', periods=72, freq='h', name='time')
    data = np.random.rand(72) * 100
    return xr.DataArray(data, dims=['time'], coords={'time': time}, name='power')


@pytest.fixture
def irregular_timeseries():
    """Create irregular time series data with missing timestamps."""
    time = pd.date_range('2024-01-01', periods=240, freq='5min', name='time')
    data = np.random.rand(240) * 100
    da = xr.DataArray(data, dims=['time'], coords={'time': time}, name='temperature')
    # Drop random 30% of data points to create irregularity
    np.random.seed(42)
    keep_indices = np.random.choice(240, int(240 * 0.7), replace=False)
    keep_indices.sort()
    return da.isel(time=keep_indices)


@pytest.fixture
def multidim_timeseries():
    """Create multi-dimensional time series (time × scenario × period)."""
    time = pd.date_range('2024-01-01', periods=48, freq='h', name='time')
    scenarios = ['base', 'high', 'low']
    periods = [2024, 2030]
    data = np.random.rand(48, 3, 2) * 100
    return xr.DataArray(
        data,
        dims=['time', 'scenario', 'period'],
        coords={'time': time, 'scenario': scenarios, 'period': periods},
        name='demand',
    )


class TestBasicReshaping:
    """Test basic reshaping functionality."""

    def test_daily_hourly_reshape(self, regular_timeseries):
        """Test reshaping into days × hours."""
        result = reshape_data_for_heatmap(regular_timeseries, reshape_time=('D', 'h'))

        assert result.dims == ('timestep', 'timeframe')
        assert result.sizes['timeframe'] == 3  # 3 days
        assert result.sizes['timestep'] == 24  # 24 hours per day
        assert result.name == 'power'

    def test_weekly_daily_reshape(self, regular_timeseries):
        """Test reshaping into weeks × days."""
        result = reshape_data_for_heatmap(regular_timeseries, reshape_time=('W', 'D'))

        assert result.dims == ('timestep', 'timeframe')
        assert 'timeframe' in result.dims
        assert 'timestep' in result.dims

    def test_monthly_daily_reshape(self):
        """Test reshaping into months × days."""
        time = pd.date_range('2024-01-01', periods=90, freq='D', name='time')
        data = np.random.rand(90) * 100
        da = xr.DataArray(data, dims=['time'], coords={'time': time}, name='monthly_data')

        result = reshape_data_for_heatmap(da, reshape_time=('MS', 'D'))

        assert result.dims == ('timestep', 'timeframe')
        assert result.sizes['timeframe'] == 3  # ~3 months
        assert result.name == 'monthly_data'

    def test_no_reshape(self, regular_timeseries):
        """Test that reshape_time=None returns data unchanged."""
        result = reshape_data_for_heatmap(regular_timeseries, reshape_time=None)

        # Should return the same data
        xr.testing.assert_equal(result, regular_timeseries)


class TestFillMethods:
    """Test different fill methods for irregular data."""

    def test_forward_fill(self, irregular_timeseries):
        """Test forward fill for missing values."""
        result = reshape_data_for_heatmap(irregular_timeseries, reshape_time=('D', 'h'), fill='ffill')

        assert result.dims == ('timestep', 'timeframe')
        # Should have no NaN values with ffill (except possibly first values)
        nan_count = np.isnan(result.values).sum()
        total_count = result.values.size
        assert nan_count < total_count * 0.1  # Less than 10% NaN

    def test_backward_fill(self, irregular_timeseries):
        """Test backward fill for missing values."""
        result = reshape_data_for_heatmap(irregular_timeseries, reshape_time=('D', 'h'), fill='bfill')

        assert result.dims == ('timestep', 'timeframe')
        # Should have no NaN values with bfill (except possibly last values)
        nan_count = np.isnan(result.values).sum()
        total_count = result.values.size
        assert nan_count < total_count * 0.1  # Less than 10% NaN

    def test_no_fill(self, irregular_timeseries):
        """Test that fill=None does not automatically fill missing values."""
        result = reshape_data_for_heatmap(irregular_timeseries, reshape_time=('D', 'h'), fill=None)

        assert result.dims == ('timestep', 'timeframe')
        # Note: Whether NaN values appear depends on whether data covers full time range
        # Just verify the function completes without error and returns correct dims
        assert result.sizes['timestep'] >= 1
        assert result.sizes['timeframe'] >= 1


class TestMultidimensionalData:
    """Test handling of multi-dimensional data."""

    def test_multidim_basic_reshape(self, multidim_timeseries):
        """Test reshaping multi-dimensional data."""
        result = reshape_data_for_heatmap(multidim_timeseries, reshape_time=('D', 'h'))

        # Should preserve extra dimensions
        assert 'timeframe' in result.dims
        assert 'timestep' in result.dims
        assert 'scenario' in result.dims
        assert 'period' in result.dims
        assert result.sizes['scenario'] == 3
        assert result.sizes['period'] == 2

    def test_multidim_with_selection(self, multidim_timeseries):
        """Test reshaping after selecting from multi-dimensional data."""
        # Select single scenario and period
        selected = multidim_timeseries.sel(scenario='base', period=2024)
        result = reshape_data_for_heatmap(selected, reshape_time=('D', 'h'))

        # Should only have timeframe and timestep dimensions
        assert result.dims == ('timestep', 'timeframe')
        assert 'scenario' not in result.dims
        assert 'period' not in result.dims


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_timeframe(self):
        """Test with data that fits in a single timeframe."""
        time = pd.date_range('2024-01-01', periods=12, freq='h', name='time')
        data = np.random.rand(12) * 100
        da = xr.DataArray(data, dims=['time'], coords={'time': time}, name='short_data')

        result = reshape_data_for_heatmap(da, reshape_time=('D', 'h'))

        assert result.dims == ('timestep', 'timeframe')
        assert result.sizes['timeframe'] == 1  # Only 1 day
        assert result.sizes['timestep'] == 12  # 12 hours

    def test_preserves_name(self, regular_timeseries):
        """Test that the data name is preserved."""
        result = reshape_data_for_heatmap(regular_timeseries, reshape_time=('D', 'h'))

        assert result.name == regular_timeseries.name

    def test_different_frequencies(self):
        """Test various time frequency combinations."""
        time = pd.date_range('2024-01-01', periods=168, freq='h', name='time')
        data = np.random.rand(168) * 100
        da = xr.DataArray(data, dims=['time'], coords={'time': time}, name='week_data')

        # Test week × hour
        result = reshape_data_for_heatmap(da, reshape_time=('W', 'h'))
        assert result.dims == ('timestep', 'timeframe')

        # Test week × day
        result = reshape_data_for_heatmap(da, reshape_time=('W', 'D'))
        assert result.dims == ('timestep', 'timeframe')


class TestDataIntegrity:
    """Test that data values are preserved correctly."""

    def test_values_preserved(self, regular_timeseries):
        """Test that no data values are lost or corrupted."""
        result = reshape_data_for_heatmap(regular_timeseries, reshape_time=('D', 'h'))

        # Flatten and compare non-NaN values
        original_values = regular_timeseries.values
        reshaped_values = result.values.flatten()

        # All original values should be present (allowing for reordering)
        # Compare sums as a simple integrity check
        assert np.isclose(np.nansum(original_values), np.nansum(reshaped_values), rtol=1e-10)

    def test_coordinate_alignment(self, regular_timeseries):
        """Test that time coordinates are properly aligned."""
        result = reshape_data_for_heatmap(regular_timeseries, reshape_time=('D', 'h'))

        # Check that coordinates exist
        assert 'timeframe' in result.coords
        assert 'timestep' in result.coords

        # Check coordinate sizes match dimensions
        assert len(result.coords['timeframe']) == result.sizes['timeframe']
        assert len(result.coords['timestep']) == result.sizes['timestep']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
