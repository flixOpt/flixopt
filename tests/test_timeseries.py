import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixOpt.core import ConversionError, DataConverter, TimeSeries, TimeSeriesCollection, TimeSeriesData


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
            simple_dataarray,
            name='Weighted Series',
            aggregation_weight=0.5,
            aggregation_group='test_group'
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
            [[1, 2, 3], [4, 5, 6]],
            coords={'dim1': [0, 1], 'time': sample_timesteps[:3]},
            dims=['dim1', 'time']
        )
        with pytest.raises(ValueError, match='dimensions of DataArray must be 1'):
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
        assert sample_timeseries.active_data.equals(
            sample_timeseries.stored_data.sel(time=subset_index)
        )

        # Reset to full index
        sample_timeseries.active_timesteps = None
        assert sample_timeseries.active_timesteps.equals(sample_timesteps)

        # Test invalid type
        with pytest.raises(TypeError, match="must be a pandas DatetimeIndex"):
            sample_timeseries.active_timesteps = "invalid"

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
        new_data = xr.DataArray(
            [1, 2, 3, 4, 5],
            coords={'time': sample_timeseries.active_timesteps},
            dims=['time']
        )

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
        assert np.array_equal(
            sample_timeseries.stored_data.values,
            series_data.values
        )

        # Test with a single-column DataFrame
        df_data = pd.DataFrame({'col1': [15, 16, 17, 18, 19]}, index=sample_timesteps)
        sample_timeseries.stored_data = df_data
        assert np.array_equal(
            sample_timeseries.stored_data.values,
            df_data['col1'].values
        )

        # Test with a NumPy array
        array_data = np.array([25, 26, 27, 28, 29])
        sample_timeseries.stored_data = array_data
        assert np.array_equal(
            sample_timeseries.stored_data.values,
            array_data
        )

        # Test with a scalar
        sample_timeseries.stored_data = 42
        assert np.all(sample_timeseries.stored_data.values == 42)

        # Test with another DataArray
        another_dataarray = xr.DataArray(
            [30, 31, 32, 33, 34],
            coords={'time': sample_timesteps},
            dims=['time']
        )
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
            series_data,
            'Aggregated Series',
            sample_timesteps,
            aggregation_weight=0.7,
            aggregation_group='group1'
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
            assert np.array_equal(
                loaded_ts.stored_data.values,
                sample_timeseries.stored_data.values
            )

        # Test from_json with dictionary
        loaded_ts_dict = TimeSeries.from_json(data=json_dict)
        assert loaded_ts_dict.name == sample_timeseries.name
        assert np.array_equal(
            loaded_ts_dict.stored_data.values,
            sample_timeseries.stored_data.values
        )

        # Test validation in from_json
        with pytest.raises(ValueError, match="one of 'path' or 'data'"):
            TimeSeries.from_json(data=json_dict, path='dummy.json')

    def test_all_equal(self, sample_timesteps):
        """Test all_equal property."""
        # All equal values
        equal_data = xr.DataArray(
            [5, 5, 5, 5, 5],
            coords={'time': sample_timesteps},
            dims=['time']
        )
        ts_equal = TimeSeries(equal_data, 'Equal Series')
        assert ts_equal.all_equal is True

        # Not all equal
        unequal_data = xr.DataArray(
            [5, 5, 6, 5, 5],
            coords={'time': sample_timesteps},
            dims=['time']
        )
        ts_unequal = TimeSeries(unequal_data, 'Unequal Series')
        assert ts_unequal.all_equal is False

    def test_arithmetic_operations(self, sample_timeseries):
        """Test arithmetic operations."""
        # Create a second TimeSeries for testing
        data2 = xr.DataArray(
            [1, 2, 3, 4, 5],
            coords={'time': sample_timeseries.active_timesteps},
            dims=['time']
        )
        ts2 = TimeSeries(data2, 'Second Series')

        # Test operations between two TimeSeries objects
        assert np.array_equal((sample_timeseries + ts2).values, sample_timeseries.active_data.values + ts2.active_data.values)
        assert np.array_equal((sample_timeseries - ts2).values, sample_timeseries.active_data.values - ts2.active_data.values)
        assert np.array_equal((sample_timeseries * ts2).values, sample_timeseries.active_data.values * ts2.active_data.values)
        assert np.array_equal((sample_timeseries / ts2).values, sample_timeseries.active_data.values / ts2.active_data.values)

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
        assert np.array_equal(
            np.add(sample_timeseries, 5).values,
            np.add(sample_timeseries.active_data, 5).values
        )

        assert np.array_equal(
            np.multiply(sample_timeseries, 2).values,
            np.multiply(sample_timeseries.active_data, 2).values
        )

        # Test with two TimeSeries objects
        data2 = xr.DataArray(
            [1, 2, 3, 4, 5],
            coords={'time': sample_timeseries.active_timesteps},
            dims=['time']
        )
        ts2 = TimeSeries(data2, 'Second Series')

        assert np.array_equal(
            np.add(sample_timeseries, ts2).values,
            np.add(sample_timeseries.active_data, ts2.active_data).values
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
    sample_collection.create_time_series(42, "constant_series")

    # Add a varying time series
    varying_data = np.array([10, 20, 30, 40, 50])
    sample_collection.create_time_series(varying_data, "varying_series")

    # Add a time series with extra timestep
    sample_collection.create_time_series(
        np.array([1, 2, 3, 4, 5, 6]),
        "extra_timestep_series",
        needs_extra_timestep=True
    )

    # Add series with aggregation settings
    sample_collection.create_time_series(
        TimeSeriesData(np.array([5, 5, 5, 5, 5]), agg_group="group1"),
        "group1_series1"
    )
    sample_collection.create_time_series(
        TimeSeriesData(np.array([6, 6, 6, 6, 6]), agg_group="group1"),
        "group1_series2"
    )
    sample_collection.create_time_series(
        TimeSeriesData(np.array([10, 10, 10, 10, 10]), agg_weight=0.5),
        "weighted_series"
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
        collection = TimeSeriesCollection(
            sample_timesteps,
            hours_of_last_timestep=last_timestep_hours
        )

        # Verify the last timestep duration
        extra_step_delta = collection.all_timesteps_extra[-1] - collection.all_timesteps_extra[-2]
        assert extra_step_delta == pd.Timedelta(hours=last_timestep_hours)

        # Test with previous timestep duration
        hours_per_step = 8
        collection2 = TimeSeriesCollection(
            sample_timesteps,
            hours_of_previous_timesteps=hours_per_step
        )

        assert collection2.hours_of_previous_timesteps == hours_per_step

    def test_create_time_series(self, sample_collection):
        """Test creating time series."""
        # Test scalar
        ts1 = sample_collection.create_time_series(42, "scalar_series")
        assert ts1.name == "scalar_series"
        assert np.all(ts1.active_data.values == 42)

        # Test numpy array
        data = np.array([1, 2, 3, 4, 5])
        ts2 = sample_collection.create_time_series(data, "array_series")
        assert np.array_equal(ts2.active_data.values, data)

        # Test with TimeSeriesData
        ts3 = sample_collection.create_time_series(
            TimeSeriesData(10, agg_weight=0.7),
            "weighted_series"
        )
        assert ts3.aggregation_weight == 0.7

        # Test with extra timestep
        ts4 = sample_collection.create_time_series(5, "extra_series", needs_extra_timestep=True)
        assert ts4.needs_extra_timestep
        assert len(ts4.active_data) == len(sample_collection.timesteps_extra)

        # Test duplicate name
        with pytest.raises(ValueError, match="already exists"):
            sample_collection.create_time_series(1, "scalar_series")

    def test_access_time_series(self, populated_collection):
        """Test accessing time series."""
        # Test __getitem__
        ts = populated_collection["varying_series"]
        assert ts.name == "varying_series"

        # Test __contains__ with string
        assert "constant_series" in populated_collection
        assert "nonexistent_series" not in populated_collection

        # Test __contains__ with TimeSeries object
        assert populated_collection["varying_series"] in populated_collection

        # Test __iter__
        names = [ts.name for ts in populated_collection]
        assert len(names) == 6
        assert "varying_series" in names

        # Test access to non-existent series
        with pytest.raises(KeyError):
            populated_collection["nonexistent_series"]

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
        populated_collection["constant_series"].stored_data = np.array([1, 2, 3, 4, 5])
        updated_constants = populated_collection.constants
        assert len(updated_constants) == 3  # One less constant
        assert "constant_series" not in [ts.name for ts in updated_constants]

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
        assert populated_collection["varying_series"].active_timesteps.equals(subset)
        assert populated_collection["extra_timestep_series"].active_timesteps.equals(
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
        assert populated_collection.group_weights["group1"] == 0.5

        # Series in group1 should have weight 0.5
        assert weights["group1_series1"] == 0.5
        assert weights["group1_series2"] == 0.5

        # Series with explicit weight should have that weight
        assert weights["weighted_series"] == 0.5

        # Series without group or weight should have weight 1
        assert weights["constant_series"] == 1

    def test_insert_new_data(self, populated_collection, sample_timesteps):
        """Test inserting new data."""
        # Create new data
        new_data = pd.DataFrame({
            "constant_series": [100, 100, 100, 100, 100],
            "varying_series": [5, 10, 15, 20, 25],
            # extra_timestep_series is omitted to test partial updates
        }, index=sample_timesteps)

        # Insert data
        populated_collection.insert_new_data(new_data)

        # Verify updates
        assert np.all(populated_collection["constant_series"].active_data.values == 100)
        assert np.array_equal(
            populated_collection["varying_series"].active_data.values,
            np.array([5, 10, 15, 20, 25])
        )

        # Series not in the DataFrame should be unchanged
        assert np.array_equal(
            populated_collection["extra_timestep_series"].active_data.values[:-1],
            np.array([1, 2, 3, 4, 5])
        )

        # Test with mismatched index
        bad_index = pd.date_range("2023-02-01", periods=5, freq="D", name="time")
        bad_data = pd.DataFrame({"constant_series": [1, 1, 1, 1, 1]}, index=bad_index)

        with pytest.raises(ValueError, match="must match collection timesteps"):
            populated_collection.insert_new_data(bad_data)

    def test_restore_data(self, populated_collection):
        """Test restoring original data."""
        # Capture original data
        original_values = {name: ts.stored_data.copy() for name, ts in populated_collection.time_series_data.items()}

        # Modify data
        new_data = pd.DataFrame({
            name: np.ones(len(populated_collection.timesteps)) * 999
            for name in populated_collection.time_series_data
            if not populated_collection[name].needs_extra_timestep
        }, index=populated_collection.timesteps)

        populated_collection.insert_new_data(new_data)

        # Verify data was changed
        assert np.all(populated_collection["constant_series"].active_data.values == 999)

        # Restore data
        populated_collection.restore_data()

        # Verify data was restored
        for name, original in original_values.items():
            restored = populated_collection[name].stored_data
            assert np.array_equal(restored.values, original.values)

    def test_class_method_with_uniform_timesteps(self):
        """Test the with_uniform_timesteps class method."""
        collection = TimeSeriesCollection.with_uniform_timesteps(
            start_time=pd.Timestamp("2023-01-01"),
            periods=24,
            freq="H",
            hours_per_step=1
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
        non_uniform_times = pd.DatetimeIndex([
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-01-02"),
            pd.Timestamp("2023-01-03 12:00:00"),  # 1.5 days from previous
            pd.Timestamp("2023-01-04"),  # 0.5 days from previous
            pd.Timestamp("2023-01-06")  # 2 days from previous
        ], name="time")

        collection = TimeSeriesCollection(non_uniform_times)
        hours = collection.hours_per_timestep.values

        # Expected hours between timestamps
        expected = np.array([24, 36, 12, 48, 48])
        assert np.allclose(hours, expected)

    def test_validation_and_errors(self, sample_timesteps):
        """Test validation and error handling."""
        # Test non-DatetimeIndex
        with pytest.raises(TypeError, match="must be a pandas DatetimeIndex"):
            TimeSeriesCollection(pd.Index([1, 2, 3, 4, 5]))

        # Test too few timesteps
        with pytest.raises(ValueError, match="must contain at least 2 timestamps"):
            TimeSeriesCollection(pd.DatetimeIndex([pd.Timestamp("2023-01-01")], name="time"))

        # Test invalid active_timesteps
        collection = TimeSeriesCollection(sample_timesteps)
        invalid_timesteps = pd.date_range("2024-01-01", periods=3, freq="D", name="time")

        with pytest.raises(ValueError, match="must be a subset"):
            collection.activate_timesteps(invalid_timesteps)
