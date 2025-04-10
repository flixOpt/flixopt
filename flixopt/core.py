"""
This module contains the core functionality of the flixopt framework.
It provides Datatypes, logging functionality, and some functions to transform data structures.
"""

import inspect
import json
import logging
import pathlib
from collections import Counter
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger('flixopt')

Scalar = Union[int, float]
"""A type representing a single number, either integer or float."""

NumericData = Union[int, float, np.integer, np.floating, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray]
"""Represents any form of numeric data, from simple scalars to complex data structures."""

NumericDataTS = Union[NumericData, 'TimeSeriesData']
"""Represents either standard numeric data or TimeSeriesData."""


class PlausibilityError(Exception):
    """Error for a failing Plausibility check."""

    pass


class ConversionError(Exception):
    """Base exception for data conversion errors."""

    pass


class DataConverter:
    """
    Converts various data types into xarray.DataArray with a timesteps index.

    Supports: scalars, arrays, Series, DataFrames, and DataArrays.
    """

    @staticmethod
    def as_dataarray(data: NumericData, timesteps: pd.DatetimeIndex) -> xr.DataArray:
        """Convert data to xarray.DataArray with specified timesteps index."""
        if not isinstance(timesteps, pd.DatetimeIndex) or len(timesteps) == 0:
            raise ValueError(f'Timesteps must be a non-empty DatetimeIndex, got {type(timesteps).__name__}')
        if not timesteps.name == 'time':
            raise ConversionError(f'DatetimeIndex is not named correctly. Must be named "time", got {timesteps.name=}')

        coords = [timesteps]
        dims = ['time']
        expected_shape = (len(timesteps),)

        try:
            if isinstance(data, (int, float, np.integer, np.floating)):
                return xr.DataArray(data, coords=coords, dims=dims)
            elif isinstance(data, pd.DataFrame):
                if not data.index.equals(timesteps):
                    raise ConversionError("DataFrame index doesn't match timesteps index")
                if not len(data.columns) == 1:
                    raise ConversionError('DataFrame must have exactly one column')
                return xr.DataArray(data.values.flatten(), coords=coords, dims=dims)
            elif isinstance(data, pd.Series):
                if not data.index.equals(timesteps):
                    raise ConversionError("Series index doesn't match timesteps index")
                return xr.DataArray(data.values, coords=coords, dims=dims)
            elif isinstance(data, np.ndarray):
                if data.ndim != 1:
                    raise ConversionError(f'Array must be 1-dimensional, got {data.ndim}')
                elif data.shape[0] != expected_shape[0]:
                    raise ConversionError(f"Array shape {data.shape} doesn't match expected {expected_shape}")
                return xr.DataArray(data, coords=coords, dims=dims)
            elif isinstance(data, xr.DataArray):
                if data.dims != tuple(dims):
                    raise ConversionError(f"DataArray dimensions {data.dims} don't match expected {dims}")
                if data.sizes[dims[0]] != len(coords[0]):
                    raise ConversionError(
                        f"DataArray length {data.sizes[dims[0]]} doesn't match expected {len(coords[0])}"
                    )
                return data.copy(deep=True)
            else:
                raise ConversionError(f'Unsupported type: {type(data).__name__}')
        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(f'Converting data {type(data)} to xarray.Dataset raised an error: {str(e)}') from e


class TimeSeriesData:
    # TODO: Move to Interface.py
    def __init__(self, data: NumericData, agg_group: Optional[str] = None, agg_weight: Optional[float] = None):
        """
        timeseries class for transmit timeseries AND special characteristics of timeseries,
        i.g. to define weights needed in calculation_type 'aggregated'
            EXAMPLE solar:
            you have several solar timeseries. These should not be overweighted
            compared to the remaining timeseries (i.g. heat load, price)!
            fixed_relative_profile_solar1 = TimeSeriesData(sol_array_1, type = 'solar')
            fixed_relative_profile_solar2 = TimeSeriesData(sol_array_2, type = 'solar')
            fixed_relative_profile_solar3 = TimeSeriesData(sol_array_3, type = 'solar')
            --> this 3 series of same type share one weight, i.e. internally assigned each weight = 1/3
            (instead of standard weight = 1)

        Args:
            data: The timeseries data, which can be a scalar, array, or numpy array.
            agg_group: The group this TimeSeriesData is a part of. agg_weight is split between members of a group. Default is None.
            agg_weight: The weight for calculation_type 'aggregated', should be between 0 and 1. Default is None.

        Raises:
            Exception: If both agg_group and agg_weight are set, an exception is raised.
        """
        self.data = data
        self.agg_group = agg_group
        self.agg_weight = agg_weight
        if (agg_group is not None) and (agg_weight is not None):
            raise ValueError('Either <agg_group> or explicit <agg_weigth> can be used. Not both!')
        self.label: Optional[str] = None

    def __repr__(self):
        # Get the constructor arguments and their current values
        init_signature = inspect.signature(self.__init__)
        init_args = init_signature.parameters

        # Create a dictionary with argument names and their values
        args_str = ', '.join(f'{name}={repr(getattr(self, name, None))}' for name in init_args if name != 'self')
        return f'{self.__class__.__name__}({args_str})'

    def __str__(self):
        return str(self.data)


class TimeSeries:
    """
    A class representing time series data with active and stored states.

    TimeSeries provides a way to store time-indexed data and work with temporal subsets.
    It supports arithmetic operations, aggregation, and JSON serialization.

    Attributes:
        name (str): The name of the time series
        aggregation_weight (Optional[float]): Weight used for aggregation
        aggregation_group (Optional[str]): Group name for shared aggregation weighting
        needs_extra_timestep (bool): Whether this series needs an extra timestep
    """

    @classmethod
    def from_datasource(
        cls,
        data: NumericData,
        name: str,
        timesteps: pd.DatetimeIndex,
        aggregation_weight: Optional[float] = None,
        aggregation_group: Optional[str] = None,
        needs_extra_timestep: bool = False,
    ) -> 'TimeSeries':
        """
        Initialize the TimeSeries from multiple data sources.

        Args:
            data: The time series data
            name: The name of the TimeSeries
            timesteps: The timesteps of the TimeSeries
            aggregation_weight: The weight in aggregation calculations
            aggregation_group: Group this TimeSeries belongs to for aggregation weight sharing
            needs_extra_timestep: Whether this series requires an extra timestep

        Returns:
            A new TimeSeries instance
        """
        return cls(
            DataConverter.as_dataarray(data, timesteps),
            name,
            aggregation_weight,
            aggregation_group,
            needs_extra_timestep,
        )

    @classmethod
    def from_json(cls, data: Optional[Dict[str, Any]] = None, path: Optional[str] = None) -> 'TimeSeries':
        """
        Load a TimeSeries from a dictionary or json file.

        Args:
            data: Dictionary containing TimeSeries data
            path: Path to a JSON file containing TimeSeries data

        Returns:
            A new TimeSeries instance

        Raises:
            ValueError: If both path and data are provided or neither is provided
        """
        if (path is None and data is None) or (path is not None and data is not None):
            raise ValueError("Exactly one of 'path' or 'data' must be provided")

        if path is not None:
            with open(path, 'r') as f:
                data = json.load(f)

        # Convert ISO date strings to datetime objects
        data['data']['coords']['time']['data'] = pd.to_datetime(data['data']['coords']['time']['data'])

        # Create the TimeSeries instance
        return cls(
            data=xr.DataArray.from_dict(data['data']),
            name=data['name'],
            aggregation_weight=data['aggregation_weight'],
            aggregation_group=data['aggregation_group'],
            needs_extra_timestep=data['needs_extra_timestep'],
        )

    def __init__(
        self,
        data: xr.DataArray,
        name: str,
        aggregation_weight: Optional[float] = None,
        aggregation_group: Optional[str] = None,
        needs_extra_timestep: bool = False,
    ):
        """
        Initialize a TimeSeries with a DataArray.

        Args:
            data: The DataArray containing time series data
            name: The name of the TimeSeries
            aggregation_weight: The weight in aggregation calculations
            aggregation_group: Group this TimeSeries belongs to for weight sharing
            needs_extra_timestep: Whether this series requires an extra timestep

        Raises:
            ValueError: If data doesn't have a 'time' index or has more than 1 dimension
        """
        if 'time' not in data.indexes:
            raise ValueError(f'DataArray must have a "time" index. Got {data.indexes}')
        if data.ndim > 1:
            raise ValueError(f'Number of dimensions of DataArray must be 1. Got {data.ndim}')

        self.name = name
        self.aggregation_weight = aggregation_weight
        self.aggregation_group = aggregation_group
        self.needs_extra_timestep = needs_extra_timestep

        # Data management
        self._stored_data = data.copy(deep=True)
        self._backup = self._stored_data.copy(deep=True)
        self._active_timesteps = self._stored_data.indexes['time']
        self._active_data = None
        self._update_active_data()

    def reset(self):
        """
        Reset active timesteps to the full set of stored timesteps.
        """
        self.active_timesteps = None

    def restore_data(self):
        """
        Restore stored_data from the backup and reset active timesteps.
        """
        self._stored_data = self._backup.copy(deep=True)
        self.reset()

    def to_json(self, path: Optional[pathlib.Path] = None) -> Dict[str, Any]:
        """
        Save the TimeSeries to a dictionary or JSON file.

        Args:
            path: Optional path to save JSON file

        Returns:
            Dictionary representation of the TimeSeries
        """
        data = {
            'name': self.name,
            'aggregation_weight': self.aggregation_weight,
            'aggregation_group': self.aggregation_group,
            'needs_extra_timestep': self.needs_extra_timestep,
            'data': self.active_data.to_dict(),
        }

        # Convert datetime objects to ISO strings
        data['data']['coords']['time']['data'] = [date.isoformat() for date in data['data']['coords']['time']['data']]

        # Save to file if path is provided
        if path is not None:
            indent = 4 if len(self.active_timesteps) <= 480 else None
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)

        return data

    @property
    def stats(self) -> str:
        """
        Return a statistical summary of the active data.

        Returns:
            String representation of data statistics
        """
        return get_numeric_stats(self.active_data, padd=0)

    def _update_active_data(self):
        """
        Update the active data based on active_timesteps.
        """
        self._active_data = self._stored_data.sel(time=self.active_timesteps)

    @property
    def all_equal(self) -> bool:
        """Check if all values in the series are equal."""
        return np.unique(self.active_data.values).size == 1

    @property
    def active_timesteps(self) -> pd.DatetimeIndex:
        """Get the current active timesteps."""
        return self._active_timesteps

    @active_timesteps.setter
    def active_timesteps(self, timesteps: Optional[pd.DatetimeIndex]):
        """
        Set active_timesteps and refresh active_data.

        Args:
            timesteps: New timesteps to activate, or None to use all stored timesteps

        Raises:
            TypeError: If timesteps is not a pandas DatetimeIndex or None
        """
        if timesteps is None:
            self._active_timesteps = self.stored_data.indexes['time']
        elif isinstance(timesteps, pd.DatetimeIndex):
            self._active_timesteps = timesteps
        else:
            raise TypeError('active_timesteps must be a pandas DatetimeIndex or None')

        self._update_active_data()

    @property
    def active_data(self) -> xr.DataArray:
        """Get a view of stored_data based on active_timesteps."""
        return self._active_data

    @property
    def stored_data(self) -> xr.DataArray:
        """Get a copy of the full stored data."""
        return self._stored_data.copy()

    @stored_data.setter
    def stored_data(self, value: NumericData):
        """
        Update stored_data and refresh active_data.

        Args:
            value: New data to store
        """
        new_data = DataConverter.as_dataarray(value, timesteps=self.active_timesteps)

        # Skip if data is unchanged to avoid overwriting backup
        if new_data.equals(self._stored_data):
            return

        self._stored_data = new_data
        self.active_timesteps = None  # Reset to full timeline

    @property
    def sel(self):
        return self.active_data.sel

    @property
    def isel(self):
        return self.active_data.isel

    def _apply_operation(self, other, op):
        """Apply an operation between this TimeSeries and another object."""
        if isinstance(other, TimeSeries):
            other = other.active_data
        return op(self.active_data, other)

    def __add__(self, other):
        return self._apply_operation(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self._apply_operation(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self._apply_operation(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self._apply_operation(other, lambda x, y: x / y)

    def __radd__(self, other):
        return other + self.active_data

    def __rsub__(self, other):
        return other - self.active_data

    def __rmul__(self, other):
        return other * self.active_data

    def __rtruediv__(self, other):
        return other / self.active_data

    def __neg__(self) -> xr.DataArray:
        return -self.active_data

    def __pos__(self) -> xr.DataArray:
        return +self.active_data

    def __abs__(self) -> xr.DataArray:
        return abs(self.active_data)

    def __gt__(self, other):
        """
        Compare if this TimeSeries is greater than another.

        Args:
            other: Another TimeSeries to compare with

        Returns:
            True if all values in this TimeSeries are greater than other
        """
        if isinstance(other, TimeSeries):
            return self.active_data > other.active_data
        return self.active_data > other

    def __ge__(self, other):
        """
        Compare if this TimeSeries is greater than or equal to another.

        Args:
            other: Another TimeSeries to compare with

        Returns:
            True if all values in this TimeSeries are greater than or equal to other
        """
        if isinstance(other, TimeSeries):
            return self.active_data >= other.active_data
        return self.active_data >= other

    def __lt__(self, other):
        """
        Compare if this TimeSeries is less than another.

        Args:
            other: Another TimeSeries to compare with

        Returns:
            True if all values in this TimeSeries are less than other
        """
        if isinstance(other, TimeSeries):
            return self.active_data < other.active_data
        return self.active_data < other

    def __le__(self, other):
        """
        Compare if this TimeSeries is less than or equal to another.

        Args:
            other: Another TimeSeries to compare with

        Returns:
            True if all values in this TimeSeries are less than or equal to other
        """
        if isinstance(other, TimeSeries):
            return self.active_data <= other.active_data
        return self.active_data <= other

    def __eq__(self, other):
        """
        Compare if this TimeSeries is equal to another.

        Args:
            other: Another TimeSeries to compare with

        Returns:
            True if all values in this TimeSeries are equal to other
        """
        if isinstance(other, TimeSeries):
            return self.active_data == other.active_data
        return self.active_data == other

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Handle NumPy universal functions.

        This allows NumPy functions to work with TimeSeries objects.
        """
        # Convert any TimeSeries inputs to their active_data
        inputs = [x.active_data if isinstance(x, TimeSeries) else x for x in inputs]
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __repr__(self):
        """
        Get a string representation of the TimeSeries.

        Returns:
            String showing TimeSeries details
        """
        attrs = {
            'name': self.name,
            'aggregation_weight': self.aggregation_weight,
            'aggregation_group': self.aggregation_group,
            'needs_extra_timestep': self.needs_extra_timestep,
            'shape': self.active_data.shape,
            'time_range': f'{self.active_timesteps[0]} to {self.active_timesteps[-1]}',
        }
        attr_str = ', '.join(f'{k}={repr(v)}' for k, v in attrs.items())
        return f'TimeSeries({attr_str})'

    def __str__(self):
        """
        Get a human-readable string representation.

        Returns:
            Descriptive string with statistics
        """
        return f"TimeSeries '{self.name}': {self.stats}"


class TimeSeriesCollection:
    """
    Collection of TimeSeries objects with shared timestep management.

    TimeSeriesCollection handles multiple TimeSeries objects with synchronized
    timesteps, provides operations on collections, and manages extra timesteps.
    """

    def __init__(
        self,
        timesteps: pd.DatetimeIndex,
        hours_of_last_timestep: Optional[float] = None,
        hours_of_previous_timesteps: Optional[Union[float, np.ndarray]] = None,
    ):
        """
        Args:
            timesteps: The timesteps of the Collection.
            hours_of_last_timestep: The duration of the last time step. Uses the last time interval if not specified
            hours_of_previous_timesteps: The duration of previous timesteps.
                If None, the first time increment of time_series is used.
                This is needed to calculate previous durations (for example consecutive_on_hours).
                If you use an array, take care that its long enough to cover all previous values!
        """
        # Prepare and validate timesteps
        self._validate_timesteps(timesteps)
        self.hours_of_previous_timesteps = self._calculate_hours_of_previous_timesteps(
            timesteps, hours_of_previous_timesteps
        )

        # Set up timesteps and hours
        self.all_timesteps = timesteps
        self.all_timesteps_extra = self._create_timesteps_with_extra(timesteps, hours_of_last_timestep)
        self.all_hours_per_timestep = self.calculate_hours_per_timestep(self.all_timesteps_extra)

        # Active timestep tracking
        self._active_timesteps = None
        self._active_timesteps_extra = None
        self._active_hours_per_timestep = None

        # Dictionary of time series by name
        self.time_series_data: Dict[str, TimeSeries] = {}

        # Aggregation
        self.group_weights: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}

    @classmethod
    def with_uniform_timesteps(
        cls, start_time: pd.Timestamp, periods: int, freq: str, hours_per_step: Optional[float] = None
    ) -> 'TimeSeriesCollection':
        """Create a collection with uniform timesteps."""
        timesteps = pd.date_range(start_time, periods=periods, freq=freq, name='time')
        return cls(timesteps, hours_of_previous_timesteps=hours_per_step)

    def create_time_series(
        self, data: Union[NumericData, TimeSeriesData], name: str, needs_extra_timestep: bool = False
    ) -> TimeSeries:
        """
        Creates a TimeSeries from the given data and adds it to the collection.

        Args:
            data: The data to create the TimeSeries from.
            name: The name of the TimeSeries.
            needs_extra_timestep: Whether to create an additional timestep at the end of the timesteps.
            The data to create the TimeSeries from.

        Returns:
            The created TimeSeries.

        """
        # Check for duplicate name
        if name in self.time_series_data:
            raise ValueError(f"TimeSeries '{name}' already exists in this collection")

        # Determine which timesteps to use
        timesteps_to_use = self.timesteps_extra if needs_extra_timestep else self.timesteps

        # Create the time series
        if isinstance(data, TimeSeriesData):
            time_series = TimeSeries.from_datasource(
                name=name,
                data=data.data,
                timesteps=timesteps_to_use,
                aggregation_weight=data.agg_weight,
                aggregation_group=data.agg_group,
                needs_extra_timestep=needs_extra_timestep,
            )
            # Connect the user time series to the created TimeSeries
            data.label = name
        else:
            time_series = TimeSeries.from_datasource(
                name=name, data=data, timesteps=timesteps_to_use, needs_extra_timestep=needs_extra_timestep
            )

        # Add to the collection
        self.add_time_series(time_series)

        return time_series

    def calculate_aggregation_weights(self) -> Dict[str, float]:
        """Calculate and return aggregation weights for all time series."""
        self.group_weights = self._calculate_group_weights()
        self.weights = self._calculate_weights()

        if np.all(np.isclose(list(self.weights.values()), 1, atol=1e-6)):
            logger.info('All Aggregation weights were set to 1')

        return self.weights

    def activate_timesteps(self, active_timesteps: Optional[pd.DatetimeIndex] = None):
        """
        Update active timesteps for the collection and all time series.
        If no arguments are provided, the active timesteps are reset.

        Args:
            active_timesteps: The active timesteps of the model.
                If None, the all timesteps of the TimeSeriesCollection are taken.
        """
        if active_timesteps is None:
            return self.reset()

        if not np.all(np.isin(active_timesteps, self.all_timesteps)):
            raise ValueError('active_timesteps must be a subset of the timesteps of the TimeSeriesCollection')

        # Calculate derived timesteps
        self._active_timesteps = active_timesteps
        first_ts_index = np.where(self.all_timesteps == active_timesteps[0])[0][0]
        last_ts_idx = np.where(self.all_timesteps == active_timesteps[-1])[0][0]
        self._active_timesteps_extra = self.all_timesteps_extra[first_ts_index : last_ts_idx + 2]
        self._active_hours_per_timestep = self.all_hours_per_timestep.isel(time=slice(first_ts_index, last_ts_idx + 1))

        # Update all time series
        self._update_time_series_timesteps()

    def reset(self):
        """Reset active timesteps to defaults for all time series."""
        self._active_timesteps = None
        self._active_timesteps_extra = None
        self._active_hours_per_timestep = None

        for time_series in self.time_series_data.values():
            time_series.reset()

    def restore_data(self):
        """Restore original data for all time series."""
        for time_series in self.time_series_data.values():
            time_series.restore_data()

    def add_time_series(self, time_series: TimeSeries):
        """Add an existing TimeSeries to the collection."""
        if time_series.name in self.time_series_data:
            raise ValueError(f"TimeSeries '{time_series.name}' already exists in this collection")

        self.time_series_data[time_series.name] = time_series

    def insert_new_data(self, data: pd.DataFrame, include_extra_timestep: bool = False):
        """
        Update time series with new data from a DataFrame.

        Args:
            data: DataFrame containing new data with timestamps as index
            include_extra_timestep: Whether the provided data already includes the extra timestep, by default False
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f'data must be a pandas DataFrame, got {type(data).__name__}')

        # Check if the DataFrame index matches the expected timesteps
        expected_timesteps = self.timesteps_extra if include_extra_timestep else self.timesteps
        if not data.index.equals(expected_timesteps):
            raise ValueError(
                f'DataFrame index must match {"collection timesteps with extra timestep" if include_extra_timestep else "collection timesteps"}'
            )

        for name, ts in self.time_series_data.items():
            if name in data.columns:
                if not ts.needs_extra_timestep:
                    # For time series without extra timestep
                    if include_extra_timestep:
                        # If data includes extra timestep but series doesn't need it, exclude the last point
                        ts.stored_data = data[name].iloc[:-1]
                    else:
                        # Use data as is
                        ts.stored_data = data[name]
                else:
                    # For time series with extra timestep
                    if include_extra_timestep:
                        # Data already includes extra timestep
                        ts.stored_data = data[name]
                    else:
                        # Need to add extra timestep - extrapolate from the last value
                        extra_step_value = data[name].iloc[-1]
                        extra_step_index = pd.DatetimeIndex([self.timesteps_extra[-1]], name='time')
                        extra_step_series = pd.Series([extra_step_value], index=extra_step_index)

                        # Combine the regular data with the extra timestep
                        ts.stored_data = pd.concat([data[name], extra_step_series])

                logger.debug(f'Updated data for {name}')

    def to_dataframe(
        self, filtered: Literal['all', 'constant', 'non_constant'] = 'non_constant', include_extra_timestep: bool = True
    ) -> pd.DataFrame:
        """
        Convert collection to DataFrame with optional filtering and timestep control.

        Args:
            filtered: Filter time series by variability, by default 'non_constant'
            include_extra_timestep: Whether to include the extra timestep in the result, by default True

        Returns:
            DataFrame representation of the collection
        """
        include_constants = filtered != 'non_constant'
        ds = self.to_dataset(include_constants=include_constants)

        if not include_extra_timestep:
            ds = ds.isel(time=slice(None, -1))

        df = ds.to_dataframe()

        # Apply filtering
        if filtered == 'all':
            return df
        elif filtered == 'constant':
            return df.loc[:, df.nunique() == 1]
        elif filtered == 'non_constant':
            return df.loc[:, df.nunique() > 1]
        else:
            raise ValueError("filtered must be one of: 'all', 'constant', 'non_constant'")

    def to_dataset(self, include_constants: bool = True) -> xr.Dataset:
        """
        Combine all time series into a single Dataset with all timesteps.

        Args:
            include_constants: Whether to include time series with constant values, by default True

        Returns:
            Dataset containing all selected time series with all timesteps
        """
        # Determine which series to include
        if include_constants:
            series_to_include = self.time_series_data.values()
        else:
            series_to_include = self.non_constants

        # Create individual datasets and merge them
        ds = xr.merge([ts.active_data.to_dataset(name=ts.name) for ts in series_to_include])

        # Ensure the correct time coordinates
        ds = ds.reindex(time=self.timesteps_extra)

        ds.attrs.update(
            {
                'timesteps_extra': f'{self.timesteps_extra[0]} ... {self.timesteps_extra[-1]} | len={len(self.timesteps_extra)}',
                'hours_per_timestep': self._format_stats(self.hours_per_timestep),
            }
        )

        return ds

    def _update_time_series_timesteps(self):
        """Update active timesteps for all time series."""
        for ts in self.time_series_data.values():
            if ts.needs_extra_timestep:
                ts.active_timesteps = self.timesteps_extra
            else:
                ts.active_timesteps = self.timesteps

    @staticmethod
    def _validate_timesteps(timesteps: pd.DatetimeIndex):
        """Validate timesteps format and rename if needed."""
        if not isinstance(timesteps, pd.DatetimeIndex):
            raise TypeError('timesteps must be a pandas DatetimeIndex')

        if len(timesteps) < 2:
            raise ValueError('timesteps must contain at least 2 timestamps')

        # Ensure timesteps has the required name
        if timesteps.name != 'time':
            logger.warning('Renamed timesteps to "time" (was "%s")', timesteps.name)
            timesteps.name = 'time'

    @staticmethod
    def _create_timesteps_with_extra(
        timesteps: pd.DatetimeIndex, hours_of_last_timestep: Optional[float]
    ) -> pd.DatetimeIndex:
        """Create timesteps with an extra step at the end."""
        if hours_of_last_timestep is not None:
            # Create the extra timestep using the specified duration
            last_date = pd.DatetimeIndex([timesteps[-1] + pd.Timedelta(hours=hours_of_last_timestep)], name='time')
        else:
            # Use the last interval as the extra timestep duration
            last_date = pd.DatetimeIndex([timesteps[-1] + (timesteps[-1] - timesteps[-2])], name='time')

        # Combine with original timesteps
        return pd.DatetimeIndex(timesteps.append(last_date), name='time')

    @staticmethod
    def _calculate_hours_of_previous_timesteps(
        timesteps: pd.DatetimeIndex, hours_of_previous_timesteps: Optional[Union[float, np.ndarray]]
    ) -> Union[float, np.ndarray]:
        """Calculate duration of regular timesteps."""
        if hours_of_previous_timesteps is not None:
            return hours_of_previous_timesteps

        # Calculate from the first interval
        first_interval = timesteps[1] - timesteps[0]
        return first_interval.total_seconds() / 3600  # Convert to hours

    @staticmethod
    def calculate_hours_per_timestep(timesteps_extra: pd.DatetimeIndex) -> xr.DataArray:
        """Calculate duration of each timestep."""
        # Calculate differences between consecutive timestamps
        hours_per_step = np.diff(timesteps_extra) / pd.Timedelta(hours=1)

        return xr.DataArray(
            data=hours_per_step, coords={'time': timesteps_extra[:-1]}, dims=('time',), name='hours_per_step'
        )

    def _calculate_group_weights(self) -> Dict[str, float]:
        """Calculate weights for aggregation groups."""
        # Count series in each group
        groups = [ts.aggregation_group for ts in self.time_series_data.values() if ts.aggregation_group is not None]
        group_counts = Counter(groups)

        # Calculate weight for each group (1/count)
        return {group: 1 / count for group, count in group_counts.items()}

    def _calculate_weights(self) -> Dict[str, float]:
        """Calculate weights for all time series."""
        # Calculate weight for each time series
        weights = {}
        for name, ts in self.time_series_data.items():
            if ts.aggregation_group is not None:
                # Use group weight
                weights[name] = self.group_weights.get(ts.aggregation_group, 1)
            else:
                # Use individual weight or default to 1
                weights[name] = ts.aggregation_weight or 1

        return weights

    def _format_stats(self, data) -> str:
        """Format statistics for a data array."""
        if hasattr(data, 'values'):
            values = data.values
        else:
            values = np.asarray(data)

        mean_val = np.mean(values)
        min_val = np.min(values)
        max_val = np.max(values)

        return f'mean: {mean_val:.2f}, min: {min_val:.2f}, max: {max_val:.2f}'

    def __getitem__(self, name: str) -> TimeSeries:
        """Get a TimeSeries by name."""
        try:
            return self.time_series_data[name]
        except KeyError as e:
            raise KeyError(f'TimeSeries "{name}" not found in the TimeSeriesCollection') from e

    def __iter__(self) -> Iterator[TimeSeries]:
        """Iterate through all TimeSeries in the collection."""
        return iter(self.time_series_data.values())

    def __len__(self) -> int:
        """Get the number of TimeSeries in the collection."""
        return len(self.time_series_data)

    def __contains__(self, item: Union[str, TimeSeries]) -> bool:
        """Check if a TimeSeries exists in the collection."""
        if isinstance(item, str):
            return item in self.time_series_data
        elif isinstance(item, TimeSeries):
            return any([item is ts for ts in self.time_series_data.values()])
        return False

    @property
    def non_constants(self) -> List[TimeSeries]:
        """Get time series with varying values."""
        return [ts for ts in self.time_series_data.values() if not ts.all_equal]

    @property
    def constants(self) -> List[TimeSeries]:
        """Get time series with constant values."""
        return [ts for ts in self.time_series_data.values() if ts.all_equal]

    @property
    def timesteps(self) -> pd.DatetimeIndex:
        """Get the active timesteps."""
        return self.all_timesteps if self._active_timesteps is None else self._active_timesteps

    @property
    def timesteps_extra(self) -> pd.DatetimeIndex:
        """Get the active timesteps with extra step."""
        return self.all_timesteps_extra if self._active_timesteps_extra is None else self._active_timesteps_extra

    @property
    def hours_per_timestep(self) -> xr.DataArray:
        """Get the duration of each active timestep."""
        return (
            self.all_hours_per_timestep if self._active_hours_per_timestep is None else self._active_hours_per_timestep
        )

    @property
    def hours_of_last_timestep(self) -> float:
        """Get the duration of the last timestep."""
        return float(self.hours_per_timestep[-1].item())

    def __repr__(self):
        return f'TimeSeriesCollection:\n{self.to_dataset()}'

    def __str__(self):
        longest_name = max([time_series.name for time_series in self.time_series_data], key=len)

        stats_summary = '\n'.join(
            [
                f'  - {time_series.name:<{len(longest_name)}}: {get_numeric_stats(time_series.active_data)}'
                for time_series in self.time_series_data
            ]
        )

        return (
            f'TimeSeriesCollection with {len(self.time_series_data)} series\n'
            f'  Time Range: {self.timesteps[0]} → {self.timesteps[-1]}\n'
            f'  No. of timesteps: {len(self.timesteps)} + 1 extra\n'
            f'  Hours per timestep: {get_numeric_stats(self.hours_per_timestep)}\n'
            f'  Time Series Data:\n'
            f'{stats_summary}'
        )


def get_numeric_stats(data: xr.DataArray, decimals: int = 2, padd: int = 10) -> str:
    """Calculates the mean, median, min, max, and standard deviation of a numeric DataArray."""
    format_spec = f'>{padd}.{decimals}f' if padd else f'.{decimals}f'
    if np.unique(data).size == 1:
        return f'{data.max().item():{format_spec}} (constant)'
    mean = data.mean().item()
    median = data.median().item()
    min_val = data.min().item()
    max_val = data.max().item()
    std = data.std().item()
    return f'{mean:{format_spec}} (mean), {median:{format_spec}} (median), {min_val:{format_spec}} (min), {max_val:{format_spec}} (max), {std:{format_spec}} (std)'
