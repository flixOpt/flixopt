import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt.core import ConversionError, DataConverter, TimeSeries, TimeSeriesCollection


@pytest.fixture
def sample_timesteps():
    """Create a sample time index with the required 'time' name."""
    return pd.date_range('2023-01-01', periods=5, freq='D', name='time')


@pytest.fixture
def simple_dataarray(sample_timesteps):
    """Create a simple DataArray with time dimension."""
    return xr.DataArray([10, 20, 30, 40, 50], coords={'time': sample_timesteps}, dims=['time'])


@pytest.fixture
def sample_timeseries(simple_dataarray):
    """Create a sample TimeSeries object."""
    return TimeSeries(simple_dataarray, name='Test Series')


class TestTimeSeries:
    """Test suite for TimeSeries class."""

    def test_initialization(self, simple_dataarray):
        """Test basic initialization of TimeSeries."""
        ts = TimeSeries(simple_dataarray, name='Test Series')

        # Check basic properties
        assert ts.name == 'Test Series'
        assert ts.aggregation_weight is None
        assert ts.aggregation_group is None

        # Check data initialization
        assert isinstance(ts.stored_data, xr.DataArray)
        assert ts.stored_data.equals(simple_dataarray)
        assert ts.selected_data.equals(simple_dataarray)

        # Check backup was created
        assert ts._backup.equals(simple_dataarray)

        # Check active timesteps
        assert ts._valid_selector == {}  # No selections initially

    def test_initialization_with_aggregation_params(self, simple_dataarray):
        """Test initialization with aggregation parameters."""
        ts = TimeSeries(
            simple_dataarray, name='Weighted Series', aggregation_weight=0.5, aggregation_group='test_group'
        )

        assert ts.name == 'Weighted Series'
        assert ts.aggregation_weight == 0.5
        assert ts.aggregation_group == 'test_group'

    def test_initialization_validation(self, sample_timesteps):
        """Test validation during initialization."""
        # Test missing time dimension
        invalid_data = xr.DataArray([1, 2, 3], dims=['invalid_dim'])
        with pytest.raises(ValueError, match='DataArray dimensions must be subset of'):
            TimeSeries(invalid_data, name='Invalid Series')

        # Test multi-dimensional data
        multi_dim_data = xr.DataArray(
            [[1, 2, 3], [4, 5, 6]], coords={'dim1': [0, 1], 'time': sample_timesteps[:3]}, dims=['dim1', 'time']
        )
        with pytest.raises(ValueError, match='DataArray dimensions must be subset of'):
            TimeSeries(multi_dim_data, name='Multi-dim Series')

    def test_selection_methods(self, sample_timeseries, sample_timesteps):
        """Test selection methods."""
        # Initial state should have no selections
        assert sample_timeseries._selected_timesteps is None
        assert sample_timeseries._selected_scenarios is None

        # Set to a subset
        subset_index = sample_timesteps[1:3]
        sample_timeseries.set_selection(timesteps=subset_index)
        assert sample_timeseries._selected_timesteps.equals(subset_index)

        # Active data should reflect the subset
        assert sample_timeseries.selected_data.equals(sample_timeseries.stored_data.sel(time=subset_index))

        # Clear selection
        sample_timeseries.set_selection()
        assert sample_timeseries._selected_timesteps is None
        assert sample_timeseries.selected_data.equals(sample_timeseries.stored_data)

    def test_reset(self, sample_timeseries, sample_timesteps):
        """Test reset method."""
        # Set to subset first
        subset_index = sample_timesteps[1:3]
        sample_timeseries.set_selection(timesteps=subset_index)

        # Reset
        sample_timeseries.reset()

        # Should be back to full index (all selections cleared)
        assert sample_timeseries._selected_timesteps is None
        assert sample_timeseries.selected_data.equals(sample_timeseries.stored_data)

    def test_restore_data(self, sample_timeseries, simple_dataarray):
        """Test restore_data method."""
        # Modify the stored data
        new_data = xr.DataArray(
            [1, 2, 3, 4, 5], coords={'time': sample_timeseries.stored_data.coords['time']}, dims=['time']
        )

        # Store original data for comparison
        original_data = sample_timeseries.stored_data

        # Update data
        sample_timeseries.update_stored_data(new_data)
        assert sample_timeseries.stored_data.equals(new_data)

        # Restore from backup
        sample_timeseries.restore_data()

        # Should be back to original data
        assert sample_timeseries.stored_data.equals(original_data)
        assert sample_timeseries.selected_data.equals(original_data)

    def test_update_stored_data(self, sample_timeseries, sample_timesteps):
        """Test update_stored_data method with different data types."""
        # Test with a Series
        series_data = pd.Series([5, 6, 7, 8, 9], index=sample_timesteps)
        sample_timeseries.update_stored_data(series_data)
        assert np.array_equal(sample_timeseries.stored_data.values, series_data.values)

        # Test with a single-column DataFrame
        df_data = pd.DataFrame({'col1': [15, 16, 17, 18, 19]}, index=sample_timesteps)
        sample_timeseries.update_stored_data(df_data)
        assert np.array_equal(sample_timeseries.stored_data.values, df_data['col1'].values)

        # Test with a NumPy array
        array_data = np.array([25, 26, 27, 28, 29])
        sample_timeseries.update_stored_data(array_data)
        assert np.array_equal(sample_timeseries.stored_data.values, array_data)

        # Test with a scalar
        sample_timeseries.update_stored_data(42)
        assert np.all(sample_timeseries.stored_data.values == 42)

        # Test with another DataArray
        another_dataarray = xr.DataArray([30, 31, 32, 33, 34], coords={'time': sample_timesteps}, dims=['time'])
        sample_timeseries.update_stored_data(another_dataarray)
        assert sample_timeseries.stored_data.equals(another_dataarray)

    def test_stored_data_setter_no_change(self, sample_timeseries):
        """Test update_stored_data method when data doesn't change."""
        # Get current data
        current_data = sample_timeseries.stored_data
        current_backup = sample_timeseries._backup

        # Set the same data
        sample_timeseries.update_stored_data(current_data)

        # Backup shouldn't change
        assert sample_timeseries._backup is current_backup  # Should be the same object

    def test_from_datasource(self, sample_timesteps):
        """Test from_datasource class method."""
        # Test with scalar
        ts_scalar = TimeSeries.from_datasource(42, 'Scalar Series', sample_timesteps)
        assert np.all(ts_scalar.stored_data.values == 42)

        # Test with Series
        series_data = pd.Series([1, 2, 3, 4, 5], index=sample_timesteps)
        ts_series = TimeSeries.from_datasource(series_data, 'Series Data', sample_timesteps)
        assert np.array_equal(ts_series.stored_data.values, series_data.values)

        # Test with aggregation parameters
        ts_with_agg = TimeSeries.from_datasource(
            series_data, 'Aggregated Series', sample_timesteps, aggregation_weight=0.7, aggregation_group='group1'
        )
        assert ts_with_agg.aggregation_weight == 0.7
        assert ts_with_agg.aggregation_group == 'group1'

    def test_to_json_from_json(self, sample_timeseries):
        """Test to_json and from_json methods."""
        # Test to_json (dictionary only)
        json_dict = sample_timeseries.to_json()
        assert json_dict['name'] == sample_timeseries.name
        assert 'data' in json_dict
        assert 'coords' in json_dict['data']
        assert 'time' in json_dict['data']['coords']

        # Test to_json with file saving
        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = Path(tmpdirname) / 'timeseries.json'
            sample_timeseries.to_json(filepath)
            assert filepath.exists()

            # Test from_json with file loading
            loaded_ts = TimeSeries.from_json(path=filepath)
            assert loaded_ts.name == sample_timeseries.name
            assert np.array_equal(loaded_ts.stored_data.values, sample_timeseries.stored_data.values)

        # Test from_json with dictionary
        loaded_ts_dict = TimeSeries.from_json(data=json_dict)
        assert loaded_ts_dict.name == sample_timeseries.name
        assert np.array_equal(loaded_ts_dict.stored_data.values, sample_timeseries.stored_data.values)

        # Test validation in from_json
        with pytest.raises(ValueError, match="one of 'path' or 'data'"):
            TimeSeries.from_json(data=json_dict, path='dummy.json')

    def test_all_equal(self, sample_timesteps):
        """Test all_equal property."""
        # All equal values
        equal_data = xr.DataArray([5, 5, 5, 5, 5], coords={'time': sample_timesteps}, dims=['time'])
        ts_equal = TimeSeries(equal_data, 'Equal Series')
        assert ts_equal.all_equal is True

        # Not all equal
        unequal_data = xr.DataArray([5, 5, 6, 5, 5], coords={'time': sample_timesteps}, dims=['time'])
        ts_unequal = TimeSeries(unequal_data, 'Unequal Series')
        assert ts_unequal.all_equal is False

    def test_arithmetic_operations(self, sample_timeseries):
        """Test arithmetic operations."""
        # Create a second TimeSeries for testing
        data2 = xr.DataArray(
            [1, 2, 3, 4, 5], coords={'time': sample_timeseries.stored_data.coords['time']}, dims=['time']
        )
        ts2 = TimeSeries(data2, 'Second Series')

        # Test operations between two TimeSeries objects
        assert np.array_equal(
            (sample_timeseries + ts2).values, sample_timeseries.selected_data.values + ts2.selected_data.values
        )
        assert np.array_equal(
            (sample_timeseries - ts2).values, sample_timeseries.selected_data.values - ts2.selected_data.values
        )
        assert np.array_equal(
            (sample_timeseries * ts2).values, sample_timeseries.selected_data.values * ts2.selected_data.values
        )
        assert np.array_equal(
            (sample_timeseries / ts2).values, sample_timeseries.selected_data.values / ts2.selected_data.values
        )

        # Test operations with DataArrays
        assert np.array_equal((sample_timeseries + data2).values, sample_timeseries.selected_data.values + data2.values)
        assert np.array_equal((data2 + sample_timeseries).values, data2.values + sample_timeseries.selected_data.values)

        # Test operations with scalars
        assert np.array_equal((sample_timeseries + 5).values, sample_timeseries.selected_data.values + 5)
        assert np.array_equal((5 + sample_timeseries).values, 5 + sample_timeseries.selected_data.values)

        # Test unary operations
        assert np.array_equal((-sample_timeseries).values, -sample_timeseries.selected_data.values)
        assert np.array_equal((+sample_timeseries).values, +sample_timeseries.selected_data.values)
        assert np.array_equal((abs(sample_timeseries)).values, abs(sample_timeseries.selected_data.values))

    def test_comparison_operations(self, sample_timesteps):
        """Test comparison operations."""
        data1 = xr.DataArray([10, 20, 30, 40, 50], coords={'time': sample_timesteps}, dims=['time'])
        data2 = xr.DataArray([5, 10, 15, 20, 25], coords={'time': sample_timesteps}, dims=['time'])

        ts1 = TimeSeries(data1, 'Series 1')
        ts2 = TimeSeries(data2, 'Series 2')

        # Test __gt__ method
        assert (ts1 > ts2).all().item()

        # Test with mixed values
        data3 = xr.DataArray([5, 25, 15, 45, 25], coords={'time': sample_timesteps}, dims=['time'])
        ts3 = TimeSeries(data3, 'Series 3')

        assert not (ts1 > ts3).all().item()  # Not all values in ts1 are greater than ts3

    def test_numpy_ufunc(self, sample_timeseries):
        """Test numpy ufunc compatibility."""
        # Test basic numpy functions
        assert np.array_equal(np.add(sample_timeseries, 5).values, np.add(sample_timeseries.selected_data, 5).values)

        assert np.array_equal(
            np.multiply(sample_timeseries, 2).values, np.multiply(sample_timeseries.selected_data, 2).values
        )

        # Test with two TimeSeries objects
        data2 = xr.DataArray(
            [1, 2, 3, 4, 5], coords={'time': sample_timeseries.stored_data.coords['time']}, dims=['time']
        )
        ts2 = TimeSeries(data2, 'Second Series')

        assert np.array_equal(
            np.add(sample_timeseries, ts2).values, np.add(sample_timeseries.selected_data, ts2.selected_data).values
        )

    def test_sel_and_isel_properties(self, sample_timeseries):
        """Test sel and isel properties."""
        # Test that sel property works
        selected = sample_timeseries.sel(time=sample_timeseries.stored_data.coords['time'][0])
        assert selected.item() == sample_timeseries.selected_data.values[0]

        # Test that isel property works
        indexed = sample_timeseries.isel(time=0)
        assert indexed.item() == sample_timeseries.selected_data.values[0]


@pytest.fixture
def sample_scenario_index():
    """Create a sample scenario index with the required 'scenario' name."""
    return pd.Index(['baseline', 'high_demand', 'low_price'], name='scenario')


@pytest.fixture
def simple_scenario_dataarray(sample_timesteps, sample_scenario_index):
    """Create a DataArray with both scenario and time dimensions."""
    data = np.array(
        [
            [10, 20, 30, 40, 50],  # baseline
            [15, 25, 35, 45, 55],  # high_demand
            [5, 15, 25, 35, 45],  # low_price
        ]
    )
    return xr.DataArray(
        data=data, coords={'scenario': sample_scenario_index, 'time': sample_timesteps}, dims=['scenario', 'time']
    )


@pytest.fixture
def sample_scenario_timeseries(simple_scenario_dataarray):
    """Create a sample TimeSeries object with scenario dimension."""
    return TimeSeries(simple_scenario_dataarray, name='Test Scenario Series')


@pytest.fixture
def sample_allocator(sample_timesteps):
    """Create a sample TimeSeriesCollection."""
    return TimeSeriesCollection(sample_timesteps)


@pytest.fixture
def sample_scenario_allocator(sample_timesteps, sample_scenario_index):
    """Create a sample TimeSeriesCollection with scenarios."""
    return TimeSeriesCollection(sample_timesteps, scenarios=sample_scenario_index)


class TestTimeSeriesWithScenarios:
    """Test suite for TimeSeries class with scenarios."""

    def test_initialization_with_scenarios(self, simple_scenario_dataarray):
        """Test initialization of TimeSeries with scenario dimension."""
        ts = TimeSeries(simple_scenario_dataarray, name='Scenario Series')

        # Check basic properties
        assert ts.name == 'Scenario Series'
        assert ts.has_scenario_dim is True
        assert ts._selected_scenarios is None  # No selection initially

        # Check data initialization
        assert isinstance(ts.stored_data, xr.DataArray)
        assert ts.stored_data.equals(simple_scenario_dataarray)
        assert ts.selected_data.equals(simple_scenario_dataarray)

        # Check backup was created
        assert ts._backup.equals(simple_scenario_dataarray)

    def test_reset_with_scenarios(self, sample_scenario_timeseries, simple_scenario_dataarray):
        """Test reset method with scenarios."""
        # Get original full indexes
        full_timesteps = simple_scenario_dataarray.coords['time']
        full_scenarios = simple_scenario_dataarray.coords['scenario']

        # Set to subset timesteps and scenarios
        subset_timesteps = full_timesteps[1:3]
        subset_scenarios = full_scenarios[:2]

        sample_scenario_timeseries.set_selection(timesteps=subset_timesteps, scenarios=subset_scenarios)

        # Verify subsets were set
        assert sample_scenario_timeseries._selected_timesteps.equals(subset_timesteps)
        assert sample_scenario_timeseries._selected_scenarios.equals(subset_scenarios)
        assert sample_scenario_timeseries.selected_data.shape == (len(subset_scenarios), len(subset_timesteps))

        # Reset
        sample_scenario_timeseries.reset()

        # Should be back to full indexes
        assert sample_scenario_timeseries._selected_timesteps is None
        assert sample_scenario_timeseries._selected_scenarios is None
        assert sample_scenario_timeseries.selected_data.shape == (len(full_scenarios), len(full_timesteps))

    def test_scenario_selection(self, sample_scenario_timeseries, sample_scenario_index):
        """Test scenario selection."""
        # Initial state should use all scenarios
        assert sample_scenario_timeseries._selected_scenarios is None

        # Set to a subset
        subset_index = sample_scenario_index[:2]  # First two scenarios
        sample_scenario_timeseries.set_selection(scenarios=subset_index)
        assert sample_scenario_timeseries._selected_scenarios.equals(subset_index)

        # Active data should reflect the subset
        assert sample_scenario_timeseries.selected_data.equals(
            sample_scenario_timeseries.stored_data.sel(scenario=subset_index)
        )

        # Clear selection
        sample_scenario_timeseries.set_selection()
        assert sample_scenario_timeseries._selected_scenarios is None

    def test_all_equal_with_scenarios(self, sample_timesteps, sample_scenario_index):
        """Test all_equal property with scenarios."""
        # All values equal across all scenarios
        equal_data = np.full((3, 5), 5)  # All values are 5
        equal_dataarray = xr.DataArray(
            data=equal_data,
            coords={'scenario': sample_scenario_index, 'time': sample_timesteps},
            dims=['scenario', 'time'],
        )
        ts_equal = TimeSeries(equal_dataarray, 'Equal Scenario Series')
        assert ts_equal.all_equal is True

        # Equal within each scenario but different between scenarios
        per_scenario_equal = np.array(
            [
                [5, 5, 5, 5, 5],  # baseline - all 5
                [10, 10, 10, 10, 10],  # high_demand - all 10
                [15, 15, 15, 15, 15],  # low_price - all 15
            ]
        )
        per_scenario_dataarray = xr.DataArray(
            data=per_scenario_equal,
            coords={'scenario': sample_scenario_index, 'time': sample_timesteps},
            dims=['scenario', 'time'],
        )
        ts_per_scenario = TimeSeries(per_scenario_dataarray, 'Per-Scenario Equal Series')
        assert ts_per_scenario.all_equal is False

    def test_arithmetic_with_scenarios(self, sample_scenario_timeseries, sample_timesteps, sample_scenario_index):
        """Test arithmetic operations with scenarios."""
        # Create a second TimeSeries with scenarios
        data2 = np.ones((3, 5))  # All ones
        second_dataarray = xr.DataArray(
            data=data2, coords={'scenario': sample_scenario_index, 'time': sample_timesteps}, dims=['scenario', 'time']
        )
        ts2 = TimeSeries(second_dataarray, 'Second Series')

        # Test operations between two scenario TimeSeries objects
        result = sample_scenario_timeseries + ts2
        assert result.shape == (3, 5)
        assert result.dims == ('scenario', 'time')

        # First scenario values should be increased by 1
        baseline_original = sample_scenario_timeseries.sel(scenario='baseline').values
        baseline_result = result.sel(scenario='baseline').values
        assert np.array_equal(baseline_result, baseline_original + 1)


class TestTimeSeriesAllocator:
    """Test suite for TimeSeriesCollection class."""

    def test_initialization(self, sample_timesteps):
        """Test basic initialization."""
        allocator = TimeSeriesCollection(sample_timesteps)

        assert allocator.timesteps.equals(sample_timesteps)
        assert len(allocator.timesteps_extra) == len(sample_timesteps) + 1
        assert isinstance(allocator.hours_per_timestep, xr.DataArray)
        assert len(allocator._time_series) == 0

    def test_initialization_with_custom_hours(self, sample_timesteps):
        """Test initialization with custom hour settings."""
        # Test with last timestep duration
        last_timestep_hours = 12
        allocator = TimeSeriesCollection(sample_timesteps, hours_of_last_timestep=last_timestep_hours)

        # Verify the last timestep duration
        extra_step_delta = allocator.timesteps_extra[-1] - allocator.timesteps_extra[-2]
        assert extra_step_delta == pd.Timedelta(hours=last_timestep_hours)

        # Test with previous timestep duration
        hours_per_step = 8
        allocator2 = TimeSeriesCollection(sample_timesteps, hours_of_previous_timesteps=hours_per_step)

        assert allocator2.hours_of_previous_timesteps == hours_per_step

    def test_add_time_series(self, sample_allocator, sample_timesteps):
        """Test adding time series."""
        # Test scalar
        ts1 = sample_allocator.add_time_series('scalar_series', 42)
        assert ts1.name == 'scalar_series'
        assert np.all(ts1.selected_data.values == 42)

        # Test numpy array
        data = np.array([1, 2, 3, 4, 5])
        ts2 = sample_allocator.add_time_series('array_series', data)
        assert np.array_equal(ts2.selected_data.values, data)

        # Test with existing TimeSeries
        existing_ts = TimeSeries.from_datasource(10, 'original_name', sample_timesteps, aggregation_weight=0.7)
        ts3 = sample_allocator.add_time_series('weighted_series', existing_ts)
        assert ts3.name == 'weighted_series'  # Name changed
        assert ts3.aggregation_weight == 0.7  # Weight preserved

        # Test with extra timestep
        ts4 = sample_allocator.add_time_series('extra_series', 5, has_extra_timestep=True)
        assert ts4.name == 'extra_series'
        assert ts4.has_extra_timestep
        assert len(ts4.selected_data) == len(sample_allocator.timesteps_extra)

        # Test duplicate name
        with pytest.raises(KeyError, match='already exists'):
            sample_allocator.add_time_series('scalar_series', 1)

    def test_access_time_series(self, sample_allocator):
        """Test accessing time series."""
        # Add a few time series
        sample_allocator.add_time_series('series1', 42)
        sample_allocator.add_time_series('series2', np.array([1, 2, 3, 4, 5]))

        # Test __getitem__
        ts = sample_allocator['series1']
        assert ts.name == 'series1'

        # Test __contains__ with string
        assert 'series1' in sample_allocator
        assert 'nonexistent_series' not in sample_allocator

        # Test __contains__ with TimeSeries object
        assert sample_allocator['series2'] in sample_allocator

        # Test access to non-existent series
        with pytest.raises(ValueError):
            sample_allocator['nonexistent_series']

    def test_selection_propagation(self, sample_allocator, sample_timesteps):
        """Test that selections propagate to TimeSeries."""
        # Add a few time series
        ts1 = sample_allocator.add_time_series('series1', 42)
        ts2 = sample_allocator.add_time_series('series2', np.array([1, 2, 3, 4, 5]))
        ts3 = sample_allocator.add_time_series('series3', 5, has_extra_timestep=True)

        # Initially no selections
        assert ts1._selected_timesteps is None
        assert ts2._selected_timesteps is None
        assert ts3._selected_timesteps is None

        # Apply selection
        subset_timesteps = sample_timesteps[1:3]
        sample_allocator.set_selection(timesteps=subset_timesteps)

        # Check selection propagated to regular time series
        assert ts1._selected_timesteps.equals(subset_timesteps)
        assert ts2._selected_timesteps.equals(subset_timesteps)

        # Check selection with extra timestep
        assert ts3._selected_timesteps is not None
        assert len(ts3._selected_timesteps) == len(subset_timesteps) + 1

        # Clear selection
        sample_allocator.set_selection()

        # Check selection cleared
        assert ts1._selected_timesteps is None
        assert ts2._selected_timesteps is None
        assert ts3._selected_timesteps is None

    def test_update_time_series(self, sample_allocator):
        """Test updating a time series."""
        # Add a time series
        ts = sample_allocator.add_time_series('series', 42)

        # Update it
        sample_allocator.update_time_series('series', np.array([1, 2, 3, 4, 5]))

        # Check update was applied
        assert np.array_equal(ts.selected_data.values, np.array([1, 2, 3, 4, 5]))

        # Test updating non-existent series
        with pytest.raises(KeyError):
            sample_allocator.update_time_series('nonexistent', 42)

    def test_as_dataset(self, sample_allocator):
        """Test as_dataset method."""
        # Add some time series
        sample_allocator.add_time_series('series1', 42)
        sample_allocator.add_time_series('series2', np.array([1, 2, 3, 4, 5]))

        # Get dataset
        ds = sample_allocator.as_dataset(with_extra_timestep=False)

        # Check dataset contents
        assert isinstance(ds, xr.Dataset)
        assert 'series1' in ds
        assert 'series2' in ds
        assert np.all(ds['series1'].values == 42)
        assert np.array_equal(ds['series2'].values, np.array([1, 2, 3, 4, 5]))


class TestTimeSeriesAllocatorWithScenarios:
    """Test suite for TimeSeriesCollection with scenarios."""

    def test_initialization_with_scenarios(self, sample_timesteps, sample_scenario_index):
        """Test initialization with scenarios."""
        allocator = TimeSeriesCollection(sample_timesteps, scenarios=sample_scenario_index)

        assert allocator.timesteps.equals(sample_timesteps)
        assert allocator.scenarios.equals(sample_scenario_index)
        assert len(allocator._time_series) == 0

    def test_add_time_series_with_scenarios(self, sample_scenario_allocator):
        """Test creating time series with scenarios."""
        # Test scalar (broadcasts to all scenarios)
        ts1 = sample_scenario_allocator.add_time_series('scalar_series', 42)
        assert ts1.has_scenario_dim
        assert ts1.name == 'scalar_series'
        assert ts1.selected_data.shape == (5, 3)  # 5 timesteps, 3 scenarios
        assert np.all(ts1.selected_data.values == 42)

        # Test 1D array (broadcasts to all scenarios)
        data = np.array([1, 2, 3, 4, 5])
        ts2 = sample_scenario_allocator.add_time_series('array_series', data)
        assert ts2.has_scenario_dim
        assert ts2.selected_data.shape == (5, 3)
        # Each scenario should have the same values
        for scenario in sample_scenario_allocator.scenarios:
            assert np.array_equal(ts2.sel(scenario=scenario).values, data)

        # Test 2D array (one row per scenario)
        data_2d = np.array([[10, 20, 30, 40, 50], [15, 25, 35, 45, 55], [5, 15, 25, 35, 45]]).T
        ts3 = sample_scenario_allocator.add_time_series('scenario_specific_series', data_2d)
        assert ts3.has_scenario_dim
        assert ts3.selected_data.shape == (5, 3)
        # Each scenario should have its own values
        assert np.array_equal(ts3.sel(scenario='baseline').values, data_2d[:,0])
        assert np.array_equal(ts3.sel(scenario='high_demand').values, data_2d[:,1])
        assert np.array_equal(ts3.sel(scenario='low_price').values, data_2d[:,2])

    def test_selection_propagation_with_scenarios(
        self, sample_scenario_allocator, sample_timesteps, sample_scenario_index
    ):
        """Test scenario selection propagation."""
        # Add some time series
        ts1 = sample_scenario_allocator.add_time_series('series1', 42)
        ts2 = sample_scenario_allocator.add_time_series('series2', np.array([1, 2, 3, 4, 5]))

        # Initial state - no selections
        assert ts1._selected_scenarios is None
        assert ts2._selected_scenarios is None

        # Select scenarios
        subset_scenarios = sample_scenario_index[:2]
        sample_scenario_allocator.set_selection(scenarios=subset_scenarios)

        # Check selections propagated
        assert ts1._selected_scenarios.equals(subset_scenarios)
        assert ts2._selected_scenarios.equals(subset_scenarios)

        # Check data is filtered
        assert ts1.selected_data.shape == (5, 2)  # 5 timesteps, 2 scenarios
        assert ts2.selected_data.shape == (5, 2)

        # Apply combined selection
        subset_timesteps = sample_timesteps[1:3]
        sample_scenario_allocator.set_selection(timesteps=subset_timesteps, scenarios=subset_scenarios)

        # Check combined selection applied
        assert ts1._selected_timesteps.equals(subset_timesteps)
        assert ts1._selected_scenarios.equals(subset_scenarios)
        assert ts1.selected_data.shape == (2, 2)  # 2 timesteps, 2 scenarios

        # Clear selections
        sample_scenario_allocator.set_selection()
        assert ts1._selected_timesteps is None
        assert ts1.active_timesteps.equals(sample_scenario_allocator.timesteps)
        assert ts1._selected_scenarios is None
        assert ts1.active_scenarios.equals(sample_scenario_allocator.scenarios)
        assert ts1.selected_data.shape == (5, 3)  # Back to full shape

    def test_as_dataset_with_scenarios(self, sample_scenario_allocator):
        """Test as_dataset method with scenarios."""
        # Add some time series
        sample_scenario_allocator.add_time_series('scalar_series', 42)
        sample_scenario_allocator.add_time_series(
            'varying_series', np.array([[10, 20, 30, 40, 50], [15, 25, 35, 45, 55], [5, 15, 25, 35, 45]]).T
        )

        # Get dataset
        ds = sample_scenario_allocator.as_dataset(with_extra_timestep=False)

        # Check dataset dimensions
        assert 'scenario' in ds.dims
        assert 'time' in ds.dims
        assert ds.dims['scenario'] == 3
        assert ds.dims['time'] == 5

        # Check dataset variables
        assert 'scalar_series' in ds
        assert 'varying_series' in ds

        # Check values
        assert np.all(ds['scalar_series'].values == 42)
        baseline_values = ds['varying_series'].sel(scenario='baseline').values
        assert np.array_equal(baseline_values, np.array([10, 20, 30, 40, 50]))

    def test_contains_and_iteration(self, sample_scenario_allocator):
        """Test __contains__ and __iter__ methods."""
        # Add some time series
        ts1 = sample_scenario_allocator.add_time_series('series1', 42)
        sample_scenario_allocator.add_time_series('series2', 10)

        # Test __contains__
        assert 'series1' in sample_scenario_allocator
        assert ts1 in sample_scenario_allocator
        assert 'nonexistent' not in sample_scenario_allocator

        # Test behavior with invalid type
        with pytest.raises(TypeError):
            assert 42 in sample_scenario_allocator

    def test_update_time_series_with_scenarios(self, sample_scenario_allocator, sample_scenario_index):
        """Test updating a time series with scenarios."""
        # Add a time series
        ts = sample_scenario_allocator.add_time_series('series', 42)
        assert ts.has_scenario_dim
        assert np.all(ts.selected_data.values == 42)

        # Update with scenario-specific data
        new_data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]).T
        sample_scenario_allocator.update_time_series('series', new_data)

        # Check update was applied
        assert np.array_equal(ts.selected_data.values, new_data)
        assert ts.has_scenario_dim

        # Check scenario-specific values
        assert np.array_equal(ts.sel(scenario='baseline').values, new_data[:,0])
        assert np.array_equal(ts.sel(scenario='high_demand').values, new_data[:,1])
        assert np.array_equal(ts.sel(scenario='low_price').values, new_data[:,2])


if __name__ == '__main__':
    pytest.main()
