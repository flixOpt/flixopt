import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt.core import ConversionError, DataConverter, TimeSeries, TimeSeriesCollection, TimeSeriesData


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
        assert ts.active_data.equals(simple_dataarray)

        # Check backup was created
        assert ts._backup.equals(simple_dataarray)

        # Check active timesteps
        assert ts.active_timesteps.equals(simple_dataarray.indexes['time'])

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
        with pytest.raises(ValueError, match='must have a "time" index'):
            TimeSeries(invalid_data, name='Invalid Series')

        # Test multi-dimensional data
        multi_dim_data = xr.DataArray(
            [[1, 2, 3], [4, 5, 6]], coords={'dim1': [0, 1], 'time': sample_timesteps[:3]}, dims=['dim1', 'time']
        )
        with pytest.raises(ValueError, match='DataArray dimensions must be subset of'):
            TimeSeries(multi_dim_data, name='Multi-dim Series')

    def test_active_timesteps_getter_setter(self, sample_timeseries, sample_timesteps):
        """Test active_timesteps getter and setter."""
        # Initial state should use all timesteps
        assert sample_timeseries.active_timesteps.equals(sample_timesteps)

        # Set to a subset
        subset_index = sample_timesteps[1:3]
        sample_timeseries.active_timesteps = subset_index
        assert sample_timeseries.active_timesteps.equals(subset_index)

        # Active data should reflect the subset
        assert sample_timeseries.active_data.equals(sample_timeseries.stored_data.sel(time=subset_index))

        # Reset to full index
        sample_timeseries.active_timesteps = None
        assert sample_timeseries.active_timesteps.equals(sample_timesteps)

        # Test invalid type
        with pytest.raises(TypeError, match='must be a pandas DatetimeIndex'):
            sample_timeseries.active_timesteps = 'invalid'

    def test_reset(self, sample_timeseries, sample_timesteps):
        """Test reset method."""
        # Set to subset first
        subset_index = sample_timesteps[1:3]
        sample_timeseries.active_timesteps = subset_index

        # Reset
        sample_timeseries.reset()

        # Should be back to full index
        assert sample_timeseries.active_timesteps.equals(sample_timesteps)
        assert sample_timeseries.active_data.equals(sample_timeseries.stored_data)

    def test_restore_data(self, sample_timeseries, simple_dataarray):
        """Test restore_data method."""
        # Modify the stored data
        new_data = xr.DataArray([1, 2, 3, 4, 5], coords={'time': sample_timeseries.active_timesteps}, dims=['time'])

        # Store original data for comparison
        original_data = sample_timeseries.stored_data

        # Set new data
        sample_timeseries.stored_data = new_data
        assert sample_timeseries.stored_data.equals(new_data)

        # Restore from backup
        sample_timeseries.restore_data()

        # Should be back to original data
        assert sample_timeseries.stored_data.equals(original_data)
        assert sample_timeseries.active_data.equals(original_data)

    def test_stored_data_setter(self, sample_timeseries, sample_timesteps):
        """Test stored_data setter with different data types."""
        # Test with a Series
        series_data = pd.Series([5, 6, 7, 8, 9], index=sample_timesteps)
        sample_timeseries.stored_data = series_data
        assert np.array_equal(sample_timeseries.stored_data.values, series_data.values)

        # Test with a single-column DataFrame
        df_data = pd.DataFrame({'col1': [15, 16, 17, 18, 19]}, index=sample_timesteps)
        sample_timeseries.stored_data = df_data
        assert np.array_equal(sample_timeseries.stored_data.values, df_data['col1'].values)

        # Test with a NumPy array
        array_data = np.array([25, 26, 27, 28, 29])
        sample_timeseries.stored_data = array_data
        assert np.array_equal(sample_timeseries.stored_data.values, array_data)

        # Test with a scalar
        sample_timeseries.stored_data = 42
        assert np.all(sample_timeseries.stored_data.values == 42)

        # Test with another DataArray
        another_dataarray = xr.DataArray([30, 31, 32, 33, 34], coords={'time': sample_timesteps}, dims=['time'])
        sample_timeseries.stored_data = another_dataarray
        assert sample_timeseries.stored_data.equals(another_dataarray)

    def test_stored_data_setter_no_change(self, sample_timeseries):
        """Test stored_data setter when data doesn't change."""
        # Get current data
        current_data = sample_timeseries.stored_data
        current_backup = sample_timeseries._backup

        # Set the same data
        sample_timeseries.stored_data = current_data

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
        data2 = xr.DataArray([1, 2, 3, 4, 5], coords={'time': sample_timeseries.active_timesteps}, dims=['time'])
        ts2 = TimeSeries(data2, 'Second Series')

        # Test operations between two TimeSeries objects
        assert np.array_equal(
            (sample_timeseries + ts2).values, sample_timeseries.active_data.values + ts2.active_data.values
        )
        assert np.array_equal(
            (sample_timeseries - ts2).values, sample_timeseries.active_data.values - ts2.active_data.values
        )
        assert np.array_equal(
            (sample_timeseries * ts2).values, sample_timeseries.active_data.values * ts2.active_data.values
        )
        assert np.array_equal(
            (sample_timeseries / ts2).values, sample_timeseries.active_data.values / ts2.active_data.values
        )

        # Test operations with DataArrays
        assert np.array_equal((sample_timeseries + data2).values, sample_timeseries.active_data.values + data2.values)
        assert np.array_equal((data2 + sample_timeseries).values, data2.values + sample_timeseries.active_data.values)

        # Test operations with scalars
        assert np.array_equal((sample_timeseries + 5).values, sample_timeseries.active_data.values + 5)
        assert np.array_equal((5 + sample_timeseries).values, 5 + sample_timeseries.active_data.values)

        # Test unary operations
        assert np.array_equal((-sample_timeseries).values, -sample_timeseries.active_data.values)
        assert np.array_equal((+sample_timeseries).values, +sample_timeseries.active_data.values)
        assert np.array_equal((abs(sample_timeseries)).values, abs(sample_timeseries.active_data.values))

    def test_comparison_operations(self, sample_timesteps):
        """Test comparison operations."""
        data1 = xr.DataArray([10, 20, 30, 40, 50], coords={'time': sample_timesteps}, dims=['time'])
        data2 = xr.DataArray([5, 10, 15, 20, 25], coords={'time': sample_timesteps}, dims=['time'])

        ts1 = TimeSeries(data1, 'Series 1')
        ts2 = TimeSeries(data2, 'Series 2')

        # Test __gt__ method
        assert (ts1 > ts2) is True  # All values in ts1 are greater than ts2

        # Test with mixed values
        data3 = xr.DataArray([5, 25, 15, 45, 25], coords={'time': sample_timesteps}, dims=['time'])
        ts3 = TimeSeries(data3, 'Series 3')

        assert (ts1 > ts3) is False  # Not all values in ts1 are greater than ts3

    def test_numpy_ufunc(self, sample_timeseries):
        """Test numpy ufunc compatibility."""
        # Test basic numpy functions
        assert np.array_equal(np.add(sample_timeseries, 5).values, np.add(sample_timeseries.active_data, 5).values)

        assert np.array_equal(
            np.multiply(sample_timeseries, 2).values, np.multiply(sample_timeseries.active_data, 2).values
        )

        # Test with two TimeSeries objects
        data2 = xr.DataArray([1, 2, 3, 4, 5], coords={'time': sample_timeseries.active_timesteps}, dims=['time'])
        ts2 = TimeSeries(data2, 'Second Series')

        assert np.array_equal(
            np.add(sample_timeseries, ts2).values, np.add(sample_timeseries.active_data, ts2.active_data).values
        )

    def test_sel_and_isel_properties(self, sample_timeseries):
        """Test sel and isel properties."""
        # Test that sel property works
        selected = sample_timeseries.sel(time=sample_timeseries.active_timesteps[0])
        assert selected.item() == sample_timeseries.active_data.values[0]

        # Test that isel property works
        indexed = sample_timeseries.isel(time=0)
        assert indexed.item() == sample_timeseries.active_data.values[0]


@pytest.fixture
def sample_collection(sample_timesteps):
    """Create a sample TimeSeriesCollection."""
    return TimeSeriesCollection(sample_timesteps)


@pytest.fixture
def populated_collection(sample_collection):
    """Create a TimeSeriesCollection with test data."""
    # Add a constant time series
    sample_collection.create_time_series(42, 'constant_series')

    # Add a varying time series
    varying_data = np.array([10, 20, 30, 40, 50])
    sample_collection.create_time_series(varying_data, 'varying_series')

    # Add a time series with extra timestep
    sample_collection.create_time_series(
        np.array([1, 2, 3, 4, 5, 6]), 'extra_timestep_series', needs_extra_timestep=True
    )

    # Add series with aggregation settings
    sample_collection.create_time_series(
        TimeSeriesData(np.array([5, 5, 5, 5, 5]), agg_group='group1'), 'group1_series1'
    )
    sample_collection.create_time_series(
        TimeSeriesData(np.array([6, 6, 6, 6, 6]), agg_group='group1'), 'group1_series2'
    )
    sample_collection.create_time_series(
        TimeSeriesData(np.array([10, 10, 10, 10, 10]), agg_weight=0.5), 'weighted_series'
    )

    return sample_collection


class TestTimeSeriesCollection:
    """Test suite for TimeSeriesCollection."""

    def test_initialization(self, sample_timesteps):
        """Test basic initialization."""
        collection = TimeSeriesCollection(sample_timesteps)

        assert collection.all_timesteps.equals(sample_timesteps)
        assert len(collection.all_timesteps_extra) == len(sample_timesteps) + 1
        assert isinstance(collection.all_hours_per_timestep, xr.DataArray)
        assert len(collection) == 0

    def test_initialization_with_custom_hours(self, sample_timesteps):
        """Test initialization with custom hour settings."""
        # Test with last timestep duration
        last_timestep_hours = 12
        collection = TimeSeriesCollection(sample_timesteps, hours_of_last_timestep=last_timestep_hours)

        # Verify the last timestep duration
        extra_step_delta = collection.all_timesteps_extra[-1] - collection.all_timesteps_extra[-2]
        assert extra_step_delta == pd.Timedelta(hours=last_timestep_hours)

        # Test with previous timestep duration
        hours_per_step = 8
        collection2 = TimeSeriesCollection(sample_timesteps, hours_of_previous_timesteps=hours_per_step)

        assert collection2.hours_of_previous_timesteps == hours_per_step

    def test_create_time_series(self, sample_collection):
        """Test creating time series."""
        # Test scalar
        ts1 = sample_collection.create_time_series(42, 'scalar_series')
        assert ts1.name == 'scalar_series'
        assert np.all(ts1.active_data.values == 42)

        # Test numpy array
        data = np.array([1, 2, 3, 4, 5])
        ts2 = sample_collection.create_time_series(data, 'array_series')
        assert np.array_equal(ts2.active_data.values, data)

        # Test with TimeSeriesData
        ts3 = sample_collection.create_time_series(TimeSeriesData(10, agg_weight=0.7), 'weighted_series')
        assert ts3.aggregation_weight == 0.7

        # Test with extra timestep
        ts4 = sample_collection.create_time_series(5, 'extra_series', needs_extra_timestep=True)
        assert ts4.needs_extra_timestep
        assert len(ts4.active_data) == len(sample_collection.timesteps_extra)

        # Test duplicate name
        with pytest.raises(ValueError, match='already exists'):
            sample_collection.create_time_series(1, 'scalar_series')

    def test_access_time_series(self, populated_collection):
        """Test accessing time series."""
        # Test __getitem__
        ts = populated_collection['varying_series']
        assert ts.name == 'varying_series'

        # Test __contains__ with string
        assert 'constant_series' in populated_collection
        assert 'nonexistent_series' not in populated_collection

        # Test __contains__ with TimeSeries object
        assert populated_collection['varying_series'] in populated_collection

        # Test __iter__
        names = [ts.name for ts in populated_collection]
        assert len(names) == 6
        assert 'varying_series' in names

        # Test access to non-existent series
        with pytest.raises(KeyError):
            populated_collection['nonexistent_series']

    def test_constants_and_non_constants(self, populated_collection):
        """Test constants and non_constants properties."""
        # Test constants
        constants = populated_collection.constants
        assert len(constants) == 4  # constant_series, group1_series1, group1_series2, weighted_series
        assert all(ts.all_equal for ts in constants)

        # Test non_constants
        non_constants = populated_collection.non_constants
        assert len(non_constants) == 2  # varying_series, extra_timestep_series
        assert all(not ts.all_equal for ts in non_constants)

        # Test modifying a series changes the results
        populated_collection['constant_series'].stored_data = np.array([1, 2, 3, 4, 5])
        updated_constants = populated_collection.constants
        assert len(updated_constants) == 3  # One less constant
        assert 'constant_series' not in [ts.name for ts in updated_constants]

    def test_timesteps_properties(self, populated_collection, sample_timesteps):
        """Test timestep-related properties."""
        # Test default (all) timesteps
        assert populated_collection.timesteps.equals(sample_timesteps)
        assert len(populated_collection.timesteps_extra) == len(sample_timesteps) + 1

        # Test activating a subset
        subset = sample_timesteps[1:3]
        populated_collection.activate_timesteps(subset)

        assert populated_collection.timesteps.equals(subset)
        assert len(populated_collection.timesteps_extra) == len(subset) + 1

        # Check that time series were updated
        assert populated_collection['varying_series'].active_timesteps.equals(subset)
        assert populated_collection['extra_timestep_series'].active_timesteps.equals(
            populated_collection.timesteps_extra
        )

        # Test reset
        populated_collection.reset()
        assert populated_collection.timesteps.equals(sample_timesteps)

    def test_to_dataframe_and_dataset(self, populated_collection):
        """Test conversion to DataFrame and Dataset."""
        # Test to_dataset
        ds = populated_collection.to_dataset()
        assert isinstance(ds, xr.Dataset)
        assert len(ds.data_vars) == 6

        # Test to_dataframe with different filters
        df_all = populated_collection.to_dataframe(filtered='all')
        assert len(df_all.columns) == 6

        df_constant = populated_collection.to_dataframe(filtered='constant')
        assert len(df_constant.columns) == 4

        df_non_constant = populated_collection.to_dataframe(filtered='non_constant')
        assert len(df_non_constant.columns) == 2

        # Test invalid filter
        with pytest.raises(ValueError):
            populated_collection.to_dataframe(filtered='invalid')

    def test_calculate_aggregation_weights(self, populated_collection):
        """Test aggregation weight calculation."""
        weights = populated_collection.calculate_aggregation_weights()

        # Group weights should be 0.5 each (1/2)
        assert populated_collection.group_weights['group1'] == 0.5

        # Series in group1 should have weight 0.5
        assert weights['group1_series1'] == 0.5
        assert weights['group1_series2'] == 0.5

        # Series with explicit weight should have that weight
        assert weights['weighted_series'] == 0.5

        # Series without group or weight should have weight 1
        assert weights['constant_series'] == 1

    def test_insert_new_data(self, populated_collection, sample_timesteps):
        """Test inserting new data."""
        # Create new data
        new_data = pd.DataFrame(
            {
                'constant_series': [100, 100, 100, 100, 100],
                'varying_series': [5, 10, 15, 20, 25],
                # extra_timestep_series is omitted to test partial updates
            },
            index=sample_timesteps,
        )

        # Insert data
        populated_collection.insert_new_data(new_data)

        # Verify updates
        assert np.all(populated_collection['constant_series'].active_data.values == 100)
        assert np.array_equal(populated_collection['varying_series'].active_data.values, np.array([5, 10, 15, 20, 25]))

        # Series not in the DataFrame should be unchanged
        assert np.array_equal(
            populated_collection['extra_timestep_series'].active_data.values[:-1], np.array([1, 2, 3, 4, 5])
        )

        # Test with mismatched index
        bad_index = pd.date_range('2023-02-01', periods=5, freq='D', name='time')
        bad_data = pd.DataFrame({'constant_series': [1, 1, 1, 1, 1]}, index=bad_index)

        with pytest.raises(ValueError, match='must match collection timesteps'):
            populated_collection.insert_new_data(bad_data)

    def test_restore_data(self, populated_collection):
        """Test restoring original data."""
        # Capture original data
        original_values = {name: ts.stored_data.copy() for name, ts in populated_collection.time_series_data.items()}

        # Modify data
        new_data = pd.DataFrame(
            {
                name: np.ones(len(populated_collection.timesteps)) * 999
                for name in populated_collection.time_series_data
                if not populated_collection[name].needs_extra_timestep
            },
            index=populated_collection.timesteps,
        )

        populated_collection.insert_new_data(new_data)

        # Verify data was changed
        assert np.all(populated_collection['constant_series'].active_data.values == 999)

        # Restore data
        populated_collection.restore_data()

        # Verify data was restored
        for name, original in original_values.items():
            restored = populated_collection[name].stored_data
            assert np.array_equal(restored.values, original.values)

    def test_class_method_with_uniform_timesteps(self):
        """Test the with_uniform_timesteps class method."""
        collection = TimeSeriesCollection.with_uniform_timesteps(
            start_time=pd.Timestamp('2023-01-01'), periods=24, freq='H', hours_per_step=1
        )

        assert len(collection.timesteps) == 24
        assert collection.hours_of_previous_timesteps == 1
        assert (collection.timesteps[1] - collection.timesteps[0]) == pd.Timedelta(hours=1)

    def test_hours_per_timestep(self, populated_collection):
        """Test hours_per_timestep calculation."""
        # Standard case - uniform timesteps
        hours = populated_collection.hours_per_timestep.values
        assert np.allclose(hours, 24)  # Default is daily timesteps

        # Create non-uniform timesteps
        non_uniform_times = pd.DatetimeIndex(
            [
                pd.Timestamp('2023-01-01'),
                pd.Timestamp('2023-01-02'),
                pd.Timestamp('2023-01-03 12:00:00'),  # 1.5 days from previous
                pd.Timestamp('2023-01-04'),  # 0.5 days from previous
                pd.Timestamp('2023-01-06'),  # 2 days from previous
            ],
            name='time',
        )

        collection = TimeSeriesCollection(non_uniform_times)
        hours = collection.hours_per_timestep.values

        # Expected hours between timestamps
        expected = np.array([24, 36, 12, 48, 48])
        assert np.allclose(hours, expected)

    def test_validation_and_errors(self, sample_timesteps):
        """Test validation and error handling."""
        # Test non-DatetimeIndex
        with pytest.raises(TypeError, match='must be a pandas DatetimeIndex'):
            TimeSeriesCollection(pd.Index([1, 2, 3, 4, 5]))

        # Test too few timesteps
        with pytest.raises(ValueError, match='must contain at least 2 timestamps'):
            TimeSeriesCollection(pd.DatetimeIndex([pd.Timestamp('2023-01-01')], name='time'))

        # Test invalid active_timesteps
        collection = TimeSeriesCollection(sample_timesteps)
        invalid_timesteps = pd.date_range('2024-01-01', periods=3, freq='D', name='time')

        with pytest.raises(ValueError, match='must be a subset'):
            collection.activate_timesteps(invalid_timesteps)



@pytest.fixture
def sample_scenario_index():
    """Create a sample scenario index with the required 'scenario' name."""
    return pd.Index(['baseline', 'high_demand', 'low_price'], name='scenario')


@pytest.fixture
def sample_multi_index(sample_timesteps, sample_scenario_index):
    """Create a sample MultiIndex with scenarios and timesteps."""
    return pd.MultiIndex.from_product(
        [sample_scenario_index, sample_timesteps],
        names=['scenario', 'time']
    )


@pytest.fixture
def simple_scenario_dataarray(sample_timesteps, sample_scenario_index):
    """Create a DataArray with both scenario and time dimensions."""
    data = np.array([
        [10, 20, 30, 40, 50],    # baseline
        [15, 25, 35, 45, 55],    # high_demand
        [5, 15, 25, 35, 45]      # low_price
    ])
    return xr.DataArray(
        data=data,
        coords={'scenario': sample_scenario_index, 'time': sample_timesteps},
        dims=['scenario', 'time']
    )


@pytest.fixture
def sample_scenario_timeseries(simple_scenario_dataarray):
    """Create a sample TimeSeries object with scenario dimension."""
    return TimeSeries(simple_scenario_dataarray, name='Test Scenario Series')


@pytest.fixture
def sample_scenario_collection(sample_timesteps, sample_scenario_index):
    """Create a sample TimeSeriesCollection with scenarios."""
    return TimeSeriesCollection(sample_timesteps, scenarios=sample_scenario_index)


class TestTimeSeriesWithScenarios:
    """Test suite for TimeSeries class with scenarios."""

    def test_initialization_with_scenarios(self, simple_scenario_dataarray):
        """Test initialization of TimeSeries with scenario dimension."""
        ts = TimeSeries(simple_scenario_dataarray, name='Scenario Series')

        # Check basic properties
        assert ts.name == 'Scenario Series'
        assert ts._has_scenarios is True
        assert ts.active_scenarios is not None
        assert len(ts.active_scenarios) == len(simple_scenario_dataarray.coords['scenario'])

        # Check data initialization
        assert isinstance(ts.stored_data, xr.DataArray)
        assert ts.stored_data.equals(simple_scenario_dataarray)
        assert ts.active_data.equals(simple_scenario_dataarray)

        # Check backup was created
        assert ts._backup.equals(simple_scenario_dataarray)

        # Check active timesteps and scenarios
        assert ts.active_timesteps.equals(simple_scenario_dataarray.indexes['time'])
        assert ts.active_scenarios.equals(simple_scenario_dataarray.indexes['scenario'])

    def test_reset_with_scenarios(self, sample_scenario_timeseries):
        """Test reset method with scenarios."""
        # Get original full indexes
        full_timesteps = sample_scenario_timeseries.active_timesteps
        full_scenarios = sample_scenario_timeseries.active_scenarios

        # Set to subset timesteps and scenarios
        subset_timesteps = full_timesteps[1:3]
        subset_scenarios = full_scenarios[:2]

        sample_scenario_timeseries.active_timesteps = subset_timesteps
        sample_scenario_timeseries.active_scenarios = subset_scenarios

        # Verify subsets were set
        assert sample_scenario_timeseries.active_timesteps.equals(subset_timesteps)
        assert sample_scenario_timeseries.active_scenarios.equals(subset_scenarios)
        assert sample_scenario_timeseries.active_data.shape == (len(subset_scenarios), len(subset_timesteps))

        # Reset
        sample_scenario_timeseries.reset()

        # Should be back to full indexes
        assert sample_scenario_timeseries.active_timesteps.equals(full_timesteps)
        assert sample_scenario_timeseries.active_scenarios.equals(full_scenarios)
        assert sample_scenario_timeseries.active_data.shape == (len(full_scenarios), len(full_timesteps))

    def test_active_scenarios_getter_setter(self, sample_scenario_timeseries, sample_scenario_index):
        """Test active_scenarios getter and setter."""
        # Initial state should use all scenarios
        assert sample_scenario_timeseries.active_scenarios.equals(sample_scenario_index)

        # Set to a subset
        subset_index = sample_scenario_index[:2]  # First two scenarios
        sample_scenario_timeseries.active_scenarios = subset_index
        assert sample_scenario_timeseries.active_scenarios.equals(subset_index)

        # Active data should reflect the subset
        assert sample_scenario_timeseries.active_data.equals(
            sample_scenario_timeseries.stored_data.sel(scenario=subset_index)
        )

        # Reset to full index
        sample_scenario_timeseries.active_scenarios = None
        assert sample_scenario_timeseries.active_scenarios.equals(sample_scenario_index)

        # Test invalid type
        with pytest.raises(TypeError, match='must be a pandas Index'):
            sample_scenario_timeseries.active_scenarios = 'invalid'

        # Test invalid scenario names
        invalid_scenarios = pd.Index(['invalid1', 'invalid2'], name='scenario')
        with pytest.raises(ValueError, match='must be a subset'):
            sample_scenario_timeseries.active_scenarios = invalid_scenarios

    def test_scenario_selection_methods(self, sample_scenario_timeseries):
        """Test scenario selection helper methods."""
        # Test select_scenario
        baseline_data = sample_scenario_timeseries.sel(scenario='baseline')
        assert baseline_data.dims == ('time',)
        assert np.array_equal(baseline_data.values, [10, 20, 30, 40, 50])

        # Test with non-existent scenario
        with pytest.raises(KeyError):
            sample_scenario_timeseries.sel(scenario='nonexistent')

        # Test get_scenario_names
        scenario_names = sample_scenario_timeseries.active_scenarios
        assert len(scenario_names) == 3
        assert set(scenario_names) == {'baseline', 'high_demand', 'low_price'}

    def test_all_equal_with_scenarios(self, sample_timesteps, sample_scenario_index):
        """Test all_equal property with scenarios."""
        # All values equal across all scenarios
        equal_data = np.full((3, 5), 5)  # All values are 5
        equal_dataarray = xr.DataArray(
            data=equal_data,
            coords={'scenario': sample_scenario_index, 'time': sample_timesteps},
            dims=['scenario', 'time']
        )
        ts_equal = TimeSeries(equal_dataarray, 'Equal Scenario Series')
        assert ts_equal.all_equal is True

        # Equal within each scenario but different between scenarios
        per_scenario_equal = np.array([
            [5, 5, 5, 5, 5],    # baseline - all 5
            [10, 10, 10, 10, 10], # high_demand - all 10
            [15, 15, 15, 15, 15]  # low_price - all 15
        ])
        per_scenario_dataarray = xr.DataArray(
            data=per_scenario_equal,
            coords={'scenario': sample_scenario_index, 'time': sample_timesteps},
            dims=['scenario', 'time']
        )
        ts_per_scenario = TimeSeries(per_scenario_dataarray, 'Per-Scenario Equal Series')
        assert ts_per_scenario.all_equal is False

        # Not equal within at least one scenario
        unequal_data = np.array([
            [5, 5, 5, 5, 5],     # baseline - all equal
            [10, 10, 10, 10, 10], # high_demand - all equal
            [15, 15, 20, 15, 15]  # low_price - not all equal
        ])
        unequal_dataarray = xr.DataArray(
            data=unequal_data,
            coords={'scenario': sample_scenario_index, 'time': sample_timesteps},
            dims=['scenario', 'time']
        )
        ts_unequal = TimeSeries(unequal_dataarray, 'Unequal Scenario Series')
        assert ts_unequal.all_equal is False

    def test_stats_with_scenarios(self, sample_timesteps, sample_scenario_index):
        """Test stats property with scenarios."""
        # Create data with different patterns in each scenario
        data = np.array([
            [10, 20, 30, 40, 50],    # baseline - increasing
            [100, 100, 100, 100, 100], # high_demand - constant
            [50, 40, 30, 20, 10]    # low_price - decreasing
        ])
        dataarray = xr.DataArray(
            data=data,
            coords={'scenario': sample_scenario_index, 'time': sample_timesteps},
            dims=['scenario', 'time']
        )
        ts = TimeSeries(dataarray, 'Mixed Stats Series')

        # Get stats string
        stats_str = ts.stats

        # Should include scenario information
        assert "By scenario" in stats_str
        assert "baseline" in stats_str
        assert "high_demand" in stats_str
        assert "low_price" in stats_str

        # Should include actual statistics
        assert "mean" in stats_str
        assert "min" in stats_str
        assert "max" in stats_str
        assert "std" in stats_str
        assert "constant" in stats_str

        # Test with single active scenario
        ts.active_scenarios = pd.Index(['baseline'], name='scenario')
        single_stats_str = ts.stats

        # Should not include scenario breakdown
        assert "By scenario" not in single_stats_str
        assert "mean" in single_stats_str  # Still has regular stats

    def test_stored_data_setter_with_scenarios(self, sample_scenario_timeseries, sample_timesteps, sample_scenario_index):
        """Test stored_data setter with different scenario data types."""
        # Test with 2D array
        array_data = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ])
        sample_scenario_timeseries.stored_data = array_data
        assert np.array_equal(sample_scenario_timeseries.stored_data.values, array_data)

        # Test with scalar (should broadcast to all scenarios and timesteps)
        sample_scenario_timeseries.stored_data = 42
        assert np.all(sample_scenario_timeseries.stored_data.values == 42)

        # Test with another scenario DataArray
        another_dataarray = xr.DataArray(
            data=np.random.rand(3, 5),
            coords={'scenario': sample_scenario_index, 'time': sample_timesteps},
            dims=['scenario', 'time']
        )
        sample_scenario_timeseries.stored_data = another_dataarray
        assert sample_scenario_timeseries.stored_data.equals(another_dataarray)

        # Test with MultiIndex Series
        multi_idx = pd.MultiIndex.from_product(
            [sample_scenario_index, sample_timesteps],
            names=['scenario', 'time']
        )
        series_values = np.arange(15)  # 15 = 3 scenarios * 5 timesteps
        multi_series = pd.Series(series_values, index=multi_idx)

        sample_scenario_timeseries.stored_data = multi_series
        assert sample_scenario_timeseries.stored_data.shape == (3, 5)
        # Verify the first scenario's values
        assert np.array_equal(
            sample_scenario_timeseries.sel(scenario='baseline').values,
            series_values[:5]
        )

    def test_from_datasource_with_scenarios(self, sample_timesteps, sample_scenario_index):
        """Test from_datasource class method with scenarios."""
        # Test with 2D array
        data = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ])
        ts_array = TimeSeries.from_datasource(data, 'Array Series', sample_timesteps, scenarios=sample_scenario_index)
        assert ts_array._has_scenarios
        assert np.array_equal(ts_array.stored_data.values, data)

        # Test with scalar
        ts_scalar = TimeSeries.from_datasource(42, 'Scalar Series', sample_timesteps, scenarios=sample_scenario_index)
        assert ts_scalar._has_scenarios
        assert np.all(ts_scalar.stored_data.values == 42)

        # Test with TimeSeriesData including scenarios

        #TODO: Test with TimeSeriesData including scenarios

    def test_to_json_from_json_with_scenarios(self, sample_scenario_timeseries):
        """Test to_json and from_json methods with scenarios."""
        # Test to_json (dictionary only)
        json_dict = sample_scenario_timeseries.to_json()
        assert json_dict['name'] == sample_scenario_timeseries.name
        assert 'data' in json_dict
        assert 'coords' in json_dict['data']
        assert 'time' in json_dict['data']['coords']
        assert 'scenario' in json_dict['data']['coords']

        # Test to_json with file saving
        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = Path(tmpdirname) / 'scenario_timeseries.json'
            sample_scenario_timeseries.to_json(filepath)
            assert filepath.exists()

            # Test from_json with file loading
            loaded_ts = TimeSeries.from_json(path=filepath)
            assert loaded_ts.name == sample_scenario_timeseries.name
            assert loaded_ts._has_scenarios
            assert np.array_equal(loaded_ts.stored_data.values, sample_scenario_timeseries.stored_data.values)
            assert loaded_ts.active_scenarios.equals(sample_scenario_timeseries.active_scenarios)

        # Test from_json with dictionary
        loaded_ts_dict = TimeSeries.from_json(data=json_dict)
        assert loaded_ts_dict.name == sample_scenario_timeseries.name
        assert loaded_ts_dict._has_scenarios
        assert np.array_equal(loaded_ts_dict.stored_data.values, sample_scenario_timeseries.stored_data.values)
        assert loaded_ts_dict.active_scenarios.equals(sample_scenario_timeseries.active_scenarios)

    def test_arithmetic_with_scenarios(self, sample_scenario_timeseries, sample_timesteps, sample_scenario_index):
        """Test arithmetic operations with scenarios."""
        # Create a second TimeSeries with scenarios
        data2 = np.ones((3, 5))  # All ones
        second_dataarray = xr.DataArray(
            data=data2,
            coords={'scenario': sample_scenario_index, 'time': sample_timesteps},
            dims=['scenario', 'time']
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

        # Test operation with scalar
        result_scalar = sample_scenario_timeseries * 2
        assert result_scalar.shape == (3, 5)
        # All values should be doubled
        assert np.array_equal(
            result_scalar.sel(scenario='baseline').values,
            baseline_original * 2
        )

    def test_repr_and_str(self, sample_scenario_timeseries):
        """Test __repr__ and __str__ methods with scenarios."""
        # Test __repr__
        repr_str = repr(sample_scenario_timeseries)
        assert 'scenarios' in repr_str
        assert str(len(sample_scenario_timeseries.active_scenarios)) in repr_str

        # Test __str__
        str_repr = str(sample_scenario_timeseries)
        assert 'By scenario' in str_repr
        # Should include the name
        assert sample_scenario_timeseries.name in str_repr


class TestTimeSeriesCollectionWithScenarios:
    """Test suite for TimeSeriesCollection with scenarios."""

    def test_initialization_with_scenarios(self, sample_timesteps, sample_scenario_index):
        """Test initialization with scenarios."""
        collection = TimeSeriesCollection(sample_timesteps, scenarios=sample_scenario_index)

        assert collection.all_timesteps.equals(sample_timesteps)
        assert collection.all_scenarios.equals(sample_scenario_index)
        assert len(collection) == 0

    def test_create_time_series_with_scenarios(self, sample_scenario_collection):
        """Test creating time series with scenarios."""
        # Test scalar (broadcasts to all scenarios)
        ts1 = sample_scenario_collection.create_time_series(42, 'scalar_series')
        assert ts1._has_scenarios
        assert ts1.name == 'scalar_series'
        assert ts1.active_data.shape == (3, 5)  # 3 scenarios, 5 timesteps
        assert np.all(ts1.active_data.values == 42)

        # Test 1D array (broadcasts to all scenarios)
        data = np.array([1, 2, 3, 4, 5])
        ts2 = sample_scenario_collection.create_time_series(data, 'array_series')
        assert ts2._has_scenarios
        assert ts2.active_data.shape == (3, 5)
        # Each scenario should have the same values
        for scenario in sample_scenario_collection.scenarios:
            assert np.array_equal(ts2.sel(scenario=scenario).values, data)

        # Test 2D array (one row per scenario)
        data_2d = np.array([
            [10, 20, 30, 40, 50],
            [15, 25, 35, 45, 55],
            [5, 15, 25, 35, 45]
        ])
        ts3 = sample_scenario_collection.create_time_series(data_2d, 'scenario_specific_series')
        assert ts3._has_scenarios
        assert ts3.active_data.shape == (3, 5)
        # Each scenario should have its own values
        assert np.array_equal(ts3.sel(scenario='baseline').values, data_2d[0])
        assert np.array_equal(ts3.sel(scenario='high_demand').values, data_2d[1])
        assert np.array_equal(ts3.sel(scenario='low_price').values, data_2d[2])

    def test_activate_scenarios(self, sample_scenario_collection, sample_scenario_index):
        """Test activating scenarios."""
        # Add some time series
        sample_scenario_collection.create_time_series(42, 'scalar_series')
        sample_scenario_collection.create_time_series(
            np.array([
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15]
            ]),
            'array_series'
        )

        # Activate a subset of scenarios
        subset_scenarios = sample_scenario_index[:2]  # First two scenarios
        sample_scenario_collection.activate_timesteps(active_scenarios=subset_scenarios)

        # Collection should have the subset
        assert sample_scenario_collection.scenarios.equals(subset_scenarios)

        # Time series should have the subset too
        assert sample_scenario_collection['scalar_series'].active_scenarios.equals(subset_scenarios)
        assert sample_scenario_collection['array_series'].active_scenarios.equals(subset_scenarios)

        # Active data should reflect the subset
        assert sample_scenario_collection['array_series'].active_data.shape == (2, 5)  # 2 scenarios, 5 timesteps

        # Reset scenarios
        sample_scenario_collection.reset()
        assert sample_scenario_collection.scenarios.equals(sample_scenario_index)
        assert sample_scenario_collection['scalar_series'].active_scenarios.equals(sample_scenario_index)

    def test_to_dataframe_with_scenarios(self, sample_scenario_collection):
        """Test conversion to DataFrame with scenarios."""
        # Add some time series
        sample_scenario_collection.create_time_series(42, 'constant_series')
        sample_scenario_collection.create_time_series(
            np.array([
                [10, 20, 30, 40, 50],  # baseline
                [15, 25, 35, 45, 55],  # high_demand
                [5, 15, 25, 35, 45]    # low_price
            ]),
            'varying_series'
        )

        # Convert to DataFrame
        df = sample_scenario_collection.to_dataframe('all')

        # DataFrame should have MultiIndex with (scenario, time)
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ['scenario', 'time']

        # Should have correct number of rows (scenarios * timesteps)
        assert len(df) == 18  # 3 scenarios * 5 timesteps (+1)

        # Should have both series as columns
        assert 'constant_series' in df.columns
        assert 'varying_series' in df.columns

        # Check values for specific scenario and time
        baseline_t0 = df.loc[('baseline', sample_scenario_collection.timesteps[0])]
        assert baseline_t0['constant_series'] == 42
        assert baseline_t0['varying_series'] == 10

    def test_to_dataset_with_scenarios(self, sample_scenario_collection):
        """Test conversion to Dataset with scenarios."""
        # Add some time series
        sample_scenario_collection.create_time_series(42, 'constant_series')
        sample_scenario_collection.create_time_series(
            np.array([
                [10, 20, 30, 40, 50],
                [15, 25, 35, 45, 55],
                [5, 15, 25, 35, 45]
            ]),
            'varying_series'
        )

        # Convert to Dataset
        ds = sample_scenario_collection.to_dataset()

        # Dataset should have both dimensions
        assert 'scenario' in ds.dims
        assert 'time' in ds.dims

        # Should have both series as variables
        assert 'constant_series' in ds
        assert 'varying_series' in ds


        # Check values for specific scenario and time
        assert ds['varying_series'].sel(
            scenario='baseline',
            time=sample_scenario_collection.timesteps[0]
        ).item() == 10

    def test_get_scenario_data(self, sample_scenario_collection):
        """Test get_scenario_data method."""
        # Add some time series
        sample_scenario_collection.create_time_series(42, 'constant_series')
        sample_scenario_collection.create_time_series(
            np.array([
                [10, 20, 30, 40, 50],
                [15, 25, 35, 45, 55],
                [5, 15, 25, 35, 45]
            ]),
            'varying_series'
        )

        # Get data for one scenario
        baseline_df = sample_scenario_collection.get_scenario_data('baseline')

        # Should be a DataFrame with time index
        assert isinstance(baseline_df, pd.DataFrame)
        assert baseline_df.index.name == 'time'
        assert len(baseline_df) == 5  # 5 timesteps

        # Should have both series as columns
        assert 'constant_series' in baseline_df.columns
        assert 'varying_series' in baseline_df.columns

        # Check specific values
        assert baseline_df['constant_series'].iloc[0] == 42
        assert baseline_df['varying_series'].iloc[0] == 10

        # Test with invalid scenario
        with pytest.raises(ValueError, match="Scenario 'invalid' not found"):
            sample_scenario_collection.get_scenario_data('invalid')

    def test_compare_scenarios(self, sample_scenario_collection):
        """Test compare_scenarios method."""
        # Add some time series
        sample_scenario_collection.create_time_series(
            np.array([
                [10, 20, 30, 40, 50],  # baseline
                [15, 25, 35, 45, 55],  # high_demand
                [5, 15, 25, 35, 45]    # low_price
            ]),
            'varying_series'
        )

        # Compare two scenarios
        diff_df = sample_scenario_collection.compare_scenarios('baseline', 'high_demand')

        # Should be a DataFrame with time index
        assert isinstance(diff_df, pd.DataFrame)
        assert diff_df.index.name == 'time'

        # Should show differences (baseline - high_demand)
        assert np.array_equal(diff_df['varying_series'].values, np.array([-5, -5, -5, -5, -5]))

        # Compare with specific time series
        diff_specific = sample_scenario_collection.compare_scenarios(
            'baseline', 'low_price', time_series_names=['varying_series']
        )

        # Should only include the specified time series
        assert list(diff_specific.columns) == ['varying_series']

        # Should show correct differences (baseline - low_price)
        assert np.array_equal(diff_specific['varying_series'].values, np.array([5, 5, 5, 5, 5]))

    def test_scenario_summary(self, sample_scenario_collection):
        """Test scenario_summary method."""
        # Add some time series with different patterns
        sample_scenario_collection.create_time_series(
            np.array([
                [10, 20, 30, 40, 50],  # baseline - increasing
                [100, 100, 100, 100, 100],  # high_demand - constant
                [50, 40, 30, 20, 10]   # low_price - decreasing
            ]),
            'varying_series'
        )

        # Get summary
        summary = sample_scenario_collection.scenario_summary()

        # Should be a DataFrame with scenario index and MultiIndex columns
        assert isinstance(summary, pd.DataFrame)
        assert summary.index.name == 'scenario'
        assert isinstance(summary.columns, pd.MultiIndex)

        # Should include statistics for each time series and scenario
        assert ('varying_series', 'mean') in summary.columns
        assert ('varying_series', 'min') in summary.columns
        assert ('varying_series', 'max') in summary.columns

        # Check specific statistics
        # Baseline (increasing): 10,20,30,40,50
        assert summary.loc['baseline', ('varying_series', 'mean')] == 30
        assert summary.loc['baseline', ('varying_series', 'min')] == 10
        assert summary.loc['baseline', ('varying_series', 'max')] == 50

        # high_demand (constant): 100,100,100,100,100
        assert summary.loc['high_demand', ('varying_series', 'mean')] == 100
        assert summary.loc['high_demand', ('varying_series', 'std')] == 0

        # low_price (decreasing): 50,40,30,20,10
        assert summary.loc['low_price', ('varying_series', 'mean')] == 30
        assert summary.loc['low_price', ('varying_series', 'min')] == 10
        assert summary.loc['low_price', ('varying_series', 'max')] == 50

    def test_insert_new_data_with_scenarios(self, sample_scenario_collection, sample_timesteps, sample_scenario_index):
        """Test inserting new data with scenarios."""
        # Add some time series
        sample_scenario_collection.create_time_series(42, 'constant_series')
        sample_scenario_collection.create_time_series(
            np.array([
                [10, 20, 30, 40, 50],
                [15, 25, 35, 45, 55],
                [5, 15, 25, 35, 45]
            ]),
            'varying_series'
        )

        # Create new data with MultiIndex (scenario, time)
        multi_idx = pd.MultiIndex.from_product(
            [sample_scenario_index, sample_timesteps],
            names=['scenario', 'time']
        )

        new_data = pd.DataFrame(
            {
                'constant_series': [100] * 15,  # 3 scenarios * 5 timesteps
                'varying_series': np.arange(15)  # Different value for each scenario-time combination
            },
            index=multi_idx
        )

        # Insert data
        sample_scenario_collection.insert_new_data(new_data)

        # Verify constant series updated
        for scenario in sample_scenario_index:
            assert np.all(
                sample_scenario_collection['constant_series']
                .select_scenario(scenario)
                .values == 100
            )

        # Verify varying series updated with scenario-specific values
        baseline_values = sample_scenario_collection['varying_series'].select_scenario('baseline').values
        assert np.array_equal(baseline_values, np.arange(0, 5))

        high_demand_values = sample_scenario_collection['varying_series'].select_scenario('high_demand').values
        assert np.array_equal(high_demand_values, np.arange(5, 10))

        low_price_values = sample_scenario_collection['varying_series'].select_scenario('low_price').values
        assert np.array_equal(low_price_values, np.arange(10, 15))

        # Test with partial data (missing some scenarios)
        partial_idx = pd.MultiIndex.from_product(
            [sample_scenario_index[:2], sample_timesteps],  # Only first two scenarios
            names=['scenario', 'time']
        )

        partial_data = pd.DataFrame(
            {
                'constant_series': [200] * 10,  # 2 scenarios * 5 timesteps
                'varying_series': np.arange(100, 110)
            },
            index=partial_idx
        )

        # Insert partial data
        sample_scenario_collection.insert_new_data(partial_data)

        # First two scenarios should be updated
        assert np.all(
            sample_scenario_collection['constant_series']
            .select_scenario('baseline')
            .values == 200
        )

        assert np.all(
            sample_scenario_collection['constant_series']
            .select_scenario('high_demand')
            .values == 200
        )

        # Last scenario should remain unchanged
        assert np.all(
            sample_scenario_collection['constant_series']
            .select_scenario('low_price')
            .values == 100
        )

        # Test with mismatched index
        bad_scenarios = pd.Index(['s1', 's2', 's3'], name='scenario')
        bad_idx = pd.MultiIndex.from_product(
            [bad_scenarios, sample_timesteps],
            names=['scenario', 'time']
        )

        bad_data = pd.DataFrame(
            {'constant_series': [1] * 15},
            index=bad_idx
        )

        with pytest.raises(ValueError, match="scenario index doesn't match"):
            sample_scenario_collection.insert_new_data(bad_data)

    def test_with_scenarios_class_method(self):
        """Test the with_scenarios class method."""
        collection = TimeSeriesCollection.with_scenarios(
            start_time=pd.Timestamp('2023-01-01'),
            periods=24,
            freq='H',
            scenario_names=['baseline', 'high', 'low'],
            hours_per_step=1
        )

        assert len(collection.timesteps) == 24
        assert collection.scenarios is not None
        assert len(collection.scenarios) == 3
        assert list(collection.scenarios) == ['baseline', 'high', 'low']
        assert collection.hours_of_previous_timesteps == 1
        assert (collection.timesteps[1] - collection.timesteps[0]) == pd.Timedelta(hours=1)

    def test_string_representation_with_scenarios(self, sample_scenario_collection):
        """Test string representation with scenarios."""
        # Add some time series
        sample_scenario_collection.create_time_series(42, 'constant_series')

        # Get string representation
        str_repr = str(sample_scenario_collection)

        # Should include scenario information
        assert 'scenarios' in str_repr
        assert str(len(sample_scenario_collection.scenarios)) in str_repr

        # Should include time series information
        assert 'constant_series' in str_repr

    def test_restore_data_with_scenarios(self, sample_scenario_collection):
        """Test restoring original data with scenarios."""
        # Add some time series
        sample_scenario_collection.create_time_series(
            np.array([
                [10, 20, 30, 40, 50],
                [15, 25, 35, 45, 55],
                [5, 15, 25, 35, 45]
            ]),
            'varying_series'
        )

        # Capture original data
        original_baseline = sample_scenario_collection['varying_series'].select_scenario('baseline').values.copy()

        # Modify data
        sample_scenario_collection['varying_series'].stored_data = 999

        # Verify data was changed
        assert np.all(sample_scenario_collection['varying_series'].select_scenario('baseline').values == 999)

        # Restore data
        sample_scenario_collection.restore_data()

        # Verify data was restored
        assert np.array_equal(
            sample_scenario_collection['varying_series'].select_scenario('baseline').values,
            original_baseline
        )

        # Verify scenarios were preserved
        assert sample_scenario_collection['varying_series']._has_scenarios
        assert len(sample_scenario_collection['varying_series'].active_scenarios) == 3


class TestIntegrationWithDataConverter:
    """Test integration between DataConverter and TimeSeries with scenarios."""

    def test_from_dataarray_with_scenarios(self, sample_timesteps, sample_scenario_index):
        """Test creating TimeSeries from DataArray with scenarios."""
        # Create a DataArray with scenarios using DataConverter
        data = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ])

        da = DataConverter.as_dataarray(data, sample_timesteps, sample_scenario_index)

        # Create TimeSeries from the DataArray
        ts = TimeSeries(da, name="Converted Series")

        # Verify scenarios were preserved
        assert ts._has_scenarios
        assert ts.active_scenarios.equals(sample_scenario_index)
        assert np.array_equal(ts.stored_data.values, data)

        # Test with different shapes
        # Scalar should broadcast to all scenarios and timesteps
        scalar_da = DataConverter.as_dataarray(42, sample_timesteps, sample_scenario_index)
        scalar_ts = TimeSeries(scalar_da, name="Scalar Series")

        assert scalar_ts._has_scenarios
        assert scalar_ts.active_scenarios.equals(sample_scenario_index)
        assert np.all(scalar_ts.stored_data.values == 42)

        # 1D array should broadcast to all scenarios
        array_1d = np.array([5, 10, 15, 20, 25])
        array_da = DataConverter.as_dataarray(array_1d, sample_timesteps, sample_scenario_index)
        array_ts = TimeSeries(array_da, name="Array Series")

        assert array_ts._has_scenarios
        for scenario in sample_scenario_index:
            assert np.array_equal(array_ts.select_scenario(scenario).values, array_1d)

    def test_multiindex_series_to_timeseries(self, sample_timesteps, sample_scenario_index, sample_multi_index):
        """Test creating TimeSeries from MultiIndex Series."""
        # Create a MultiIndex Series
        series_values = np.arange(15)  # 3 scenarios * 5 timesteps
        multi_series = pd.Series(series_values, index=sample_multi_index)

        # Convert to DataArray
        da = DataConverter.as_dataarray(multi_series, sample_timesteps, sample_scenario_index)

        # Create TimeSeries
        ts = TimeSeries(da, name="From MultiIndex Series")

        # Verify scenarios and data
        assert ts._has_scenarios
        assert ts.active_scenarios.equals(sample_scenario_index)

        # Verify the first scenario's values (first 5 values)
        baseline_values = ts.select_scenario('baseline').values
        assert np.array_equal(baseline_values, series_values[:5])

        # Verify the second scenario's values (second 5 values)
        high_demand_values = ts.select_scenario('high_demand').values
        assert np.array_equal(high_demand_values, series_values[5:10])

        # Verify the third scenario's values (last 5 values)
        low_price_values = ts.select_scenario('low_price').values
        assert np.array_equal(low_price_values, series_values[10:15])

    def test_dataconverter_to_timeseriescollection(self, sample_timesteps, sample_scenario_index):
        """Test end-to-end DataConverter to TimeSeriesCollection flow."""
        # Create a collection with scenarios
        collection = TimeSeriesCollection(sample_timesteps, scenarios=sample_scenario_index)

        # 1. Test with scalar
        scalar_da = DataConverter.as_dataarray(42, sample_timesteps, sample_scenario_index)
        collection.add_time_series(TimeSeries(scalar_da, name="scalar_series"))

        # 2. Test with 1D array
        array_1d = np.array([5, 10, 15, 20, 25])
        array_da = DataConverter.as_dataarray(array_1d, sample_timesteps, sample_scenario_index)
        collection.add_time_series(TimeSeries(array_da, name="array_series"))

        # 3. Test with 2D array
        array_2d = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ])
        array_2d_da = DataConverter.as_dataarray(array_2d, sample_timesteps, sample_scenario_index)
        collection.add_time_series(TimeSeries(array_2d_da, name="array_2d_series"))

        # 4. Test with MultiIndex Series
        multi_idx = pd.MultiIndex.from_product(
            [sample_scenario_index, sample_timesteps],
            names=['scenario', 'time']
        )
        series_values = np.arange(15)
        multi_series = pd.Series(series_values, index=multi_idx)
        series_da = DataConverter.as_dataarray(multi_series, sample_timesteps, sample_scenario_index)
        collection.add_time_series(TimeSeries(series_da, name="multi_series"))

        # Verify all series were added with scenarios
        assert len(collection) == 4
        assert all(ts._has_scenarios for ts in collection)

        # Try getting scenario-specific data
        baseline_df = collection.get_scenario_data('baseline')
        assert len(baseline_df) == 5  # 5 timesteps
        assert len(baseline_df.columns) == 4  # 4 series

        # Values should match expected values for 'baseline' scenario
        assert baseline_df['scalar_series'].iloc[0] == 42
        assert baseline_df['array_series'].iloc[0] == 5
        assert baseline_df['array_2d_series'].iloc[0] == 1
        assert baseline_df['multi_series'].iloc[0] == 0


if __name__ == '__main__':
    pytest.main()
