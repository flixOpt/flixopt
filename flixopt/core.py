"""
This module contains the core functionality of the flixopt framework.
It provides Datatypes, logging functionality, and some functions to transform data structures.
"""

import inspect
import json
import logging
import pathlib
import textwrap
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

TimestepData = NumericData
"""Represents any form of numeric data that corresponds to timesteps."""

ScenarioData = NumericData
"""Represents any form of numeric data that corresponds to scenarios."""


class PlausibilityError(Exception):
    """Error for a failing Plausibility check."""

    pass


class ConversionError(Exception):
    """Base exception for data conversion errors."""

    pass


class DataConverter:
    """
    Converts various data types into xarray.DataArray with optional time and scenario dimension.

    Current implementation handles:
    - Scalar values
    - NumPy arrays
    - xarray.DataArray
    """

    @staticmethod
    def as_dataarray(
        data: TimestepData, timesteps: Optional[pd.DatetimeIndex] = None, scenarios: Optional[pd.Index] = None
    ) -> xr.DataArray:
        """
        Convert data to xarray.DataArray with specified dimensions.

        Args:
            data: The data to convert (scalar, array, or DataArray)
            timesteps: Optional DatetimeIndex for time dimension
            scenarios: Optional Index for scenario dimension

        Returns:
            DataArray with the converted data
        """
        # Prepare dimensions and coordinates
        coords, dims = DataConverter._prepare_dimensions(timesteps, scenarios)

        # Select appropriate converter based on data type
        if isinstance(data, (int, float, np.integer, np.floating)):
            return DataConverter._convert_scalar(data, coords, dims)

        elif isinstance(data, xr.DataArray):
            return DataConverter._convert_dataarray(data, coords, dims)

        elif isinstance(data, np.ndarray):
            return DataConverter._convert_ndarray(data, coords, dims)

        elif isinstance(data, pd.Series):
            return DataConverter._convert_series(data, coords, dims)

        elif isinstance(data, pd.DataFrame):
            return DataConverter._convert_dataframe(data, coords, dims)

        else:
            raise ConversionError(f'Unsupported data type: {type(data).__name__}')

    @staticmethod
    def _validate_timesteps(timesteps: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Validate and prepare time index.

        Args:
            timesteps: The time index to validate

        Returns:
            Validated time index
        """
        if not isinstance(timesteps, pd.DatetimeIndex) or len(timesteps) == 0:
            raise ConversionError('Timesteps must be a non-empty DatetimeIndex')

        if not timesteps.name == 'time':
            raise ConversionError(f'Scenarios must be named "time", got "{timesteps.name}"')

        return timesteps

    @staticmethod
    def _validate_scenarios(scenarios: pd.Index) -> pd.Index:
        """
        Validate and prepare scenario index.

        Args:
            scenarios: The scenario index to validate
        """
        if not isinstance(scenarios, pd.Index) or len(scenarios) == 0:
            raise ConversionError('Scenarios must be a non-empty Index')

        if not scenarios.name == 'scenario':
            raise ConversionError(f'Scenarios must be named "scenario", got "{scenarios.name}"')

        return scenarios

    @staticmethod
    def _prepare_dimensions(
        timesteps: Optional[pd.DatetimeIndex], scenarios: Optional[pd.Index]
    ) -> Tuple[Dict[str, pd.Index], Tuple[str, ...]]:
        """
        Prepare coordinates and dimensions for the DataArray.

        Args:
            timesteps: Optional time index
            scenarios: Optional scenario index

        Returns:
            Tuple of (coordinates dict, dimensions tuple)
        """
        # Validate inputs if provided
        if timesteps is not None:
            timesteps = DataConverter._validate_timesteps(timesteps)

        if scenarios is not None:
            scenarios = DataConverter._validate_scenarios(scenarios)

        # Build coordinates and dimensions
        coords = {}
        dims = []

        if timesteps is not None:
            coords['time'] = timesteps
            dims.append('time')

        if scenarios is not None:
            coords['scenario'] = scenarios
            dims.append('scenario')

        return coords, tuple(dims)

    @staticmethod
    def _convert_scalar(
        data: Union[int, float, np.integer, np.floating], coords: Dict[str, pd.Index], dims: Tuple[str, ...]
    ) -> xr.DataArray:
        """
        Convert a scalar value to a DataArray.

        Args:
            data: The scalar value
            coords: Coordinate dictionary
            dims: Dimension names

        Returns:
            DataArray with the scalar value
        """
        if isinstance(data, (np.integer, np.floating)):
            data = data.item()
        return xr.DataArray(data, coords=coords, dims=dims)

    @staticmethod
    def _convert_dataarray(data: xr.DataArray, coords: Dict[str, pd.Index], dims: Tuple[str, ...]) -> xr.DataArray:
        """
        Convert an existing DataArray to desired dimensions.

        Args:
            data: The source DataArray
            coords: Target coordinates
            dims: Target dimensions

        Returns:
            DataArray with the target dimensions
        """
        # No dimensions case
        if len(dims) == 0:
            if data.size != 1:
                raise ConversionError('When converting to dimensionless DataArray, source must be scalar')
            return xr.DataArray(data.values.item())

        # Check if data already has matching dimensions and coordinates
        if set(data.dims) == set(dims):
            # Check if coordinates match
            is_compatible = True
            for dim in dims:
                if dim in data.dims and not np.array_equal(data.coords[dim].values, coords[dim].values):
                    is_compatible = False
                    break

            if is_compatible:
                # Ensure dimensions are in the correct order
                if data.dims != dims:
                    # Transpose to get dimensions in the right order
                    return data.transpose(*dims).copy(deep=True)
                else:
                    # Return existing DataArray if compatible and order is correct
                    return data.copy(deep=True)

        # Handle dimension broadcasting
        if len(data.dims) == 1 and len(dims) == 2:
            # Single dimension to two dimensions
            if data.dims[0] == 'time' and 'scenario' in dims:
                # Broadcast time dimension to include scenarios
                return DataConverter._broadcast_time_to_scenarios(data, coords, dims)

            elif data.dims[0] == 'scenario' and 'time' in dims:
                # Broadcast scenario dimension to include time
                return DataConverter._broadcast_scenario_to_time(data, coords, dims)

        raise ConversionError(
            f'Cannot convert {data.dims} to {dims}. Source coordinates: {data.coords}, Target coordinates: {coords}'
        )
    @staticmethod
    def _broadcast_time_to_scenarios(
        data: xr.DataArray, coords: Dict[str, pd.Index], dims: Tuple[str, ...]
    ) -> xr.DataArray:
        """
        Broadcast a time-only DataArray to include scenarios.

        Args:
            data: The time-indexed DataArray
            coords: Target coordinates
            dims: Target dimensions

        Returns:
            DataArray with time and scenario dimensions
        """
        # Check compatibility
        if not np.array_equal(data.coords['time'].values, coords['time'].values):
            raise ConversionError("Source time coordinates don't match target time coordinates")

        if len(coords['scenario']) <= 1:
            return data.copy(deep=True)

        # Broadcast values
        values = np.repeat(data.values[:, np.newaxis], len(coords['scenario']), axis=1)
        return xr.DataArray(values.copy(), coords=coords, dims=dims)

    @staticmethod
    def _broadcast_scenario_to_time(
        data: xr.DataArray, coords: Dict[str, pd.Index], dims: Tuple[str, ...]
    ) -> xr.DataArray:
        """
        Broadcast a scenario-only DataArray to include time.

        Args:
            data: The scenario-indexed DataArray
            coords: Target coordinates
            dims: Target dimensions

        Returns:
            DataArray with time and scenario dimensions
        """
        # Check compatibility
        if not np.array_equal(data.coords['scenario'].values, coords['scenario'].values):
            raise ConversionError("Source scenario coordinates don't match target scenario coordinates")

        # Broadcast values
        values = np.repeat(data.values[:, np.newaxis], len(coords['time']), axis=1).T
        return xr.DataArray(values.copy(), coords=coords, dims=dims)

    @staticmethod
    def _convert_ndarray(data: np.ndarray, coords: Dict[str, pd.Index], dims: Tuple[str, ...]) -> xr.DataArray:
        """
        Convert a NumPy array to a DataArray.

        Args:
            data: The NumPy array
            coords: Target coordinates
            dims: Target dimensions

        Returns:
            DataArray from the NumPy array
        """
        # Handle dimensionless case
        if len(dims) == 0:
            if data.size != 1:
                raise ConversionError('Without dimensions, can only convert scalar arrays')
            return xr.DataArray(data.item())

        # Handle single dimension
        elif len(dims) == 1:
            return DataConverter._convert_ndarray_single_dim(data, coords, dims)

        # Handle two dimensions
        elif len(dims) == 2:
            return DataConverter._convert_ndarray_two_dims(data, coords, dims)

        else:
            raise ConversionError('Maximum 2 dimensions supported')

    @staticmethod
    def _convert_ndarray_single_dim(
        data: np.ndarray, coords: Dict[str, pd.Index], dims: Tuple[str, ...]
    ) -> xr.DataArray:
        """
        Convert a NumPy array to a single-dimension DataArray.

        Args:
            data: The NumPy array
            coords: Target coordinates
            dims: Target dimensions (length 1)

        Returns:
            DataArray with single dimension
        """
        dim_name = dims[0]
        dim_length = len(coords[dim_name])

        if data.ndim == 1:
            # 1D array must match dimension length
            if data.shape[0] != dim_length:
                raise ConversionError(f"Array length {data.shape[0]} doesn't match {dim_name} length {dim_length}")
            return xr.DataArray(data, coords=coords, dims=dims)
        else:
            raise ConversionError(f'Expected 1D array for single dimension, got {data.ndim}D')

    @staticmethod
    def _convert_ndarray_two_dims(data: np.ndarray, coords: Dict[str, pd.Index], dims: Tuple[str, ...]) -> xr.DataArray:
        """
        Convert a NumPy array to a two-dimension DataArray.

        Args:
            data: The NumPy array
            coords: Target coordinates
            dims: Target dimensions (length 2)

        Returns:
            DataArray with two dimensions
        """
        scenario_length = len(coords['scenario'])
        time_length = len(coords['time'])

        if data.ndim == 1:
            # For 1D array, create 2D array based on which dimension it matches
            if data.shape[0] == time_length:
                # Broadcast across scenarios
                values = np.repeat(data[:, np.newaxis], scenario_length, axis=1)
                return xr.DataArray(values, coords=coords, dims=dims)
            elif data.shape[0] == scenario_length:
                # Broadcast across time
                values = np.repeat(data[np.newaxis, :], time_length, axis=0)
                return xr.DataArray(values, coords=coords, dims=dims)
            else:
                raise ConversionError(f"1D array length {data.shape[0]} doesn't match either dimension")

        elif data.ndim == 2:
            # For 2D array, shape must match dimensions
            expected_shape = (time_length, scenario_length)
            if data.shape != expected_shape:
                raise ConversionError(f"2D array shape {data.shape} doesn't match expected shape {expected_shape}")
            return xr.DataArray(data, coords=coords, dims=dims)

        else:
            raise ConversionError(f'Expected 1D or 2D array for two dimensions, got {data.ndim}D')

    @staticmethod
    def _convert_series(data: pd.Series, coords: Dict[str, pd.Index], dims: Tuple[str, ...]) -> xr.DataArray:
        """
        Convert pandas Series to xarray DataArray.

        Args:
            data: pandas Series to convert
            coords: Target coordinates
            dims: Target dimensions

        Returns:
            DataArray from the pandas Series
        """
        # Handle single dimension case
        if len(dims) == 1:
            dim_name = dims[0]

            # Check if series index matches the dimension
            if data.index.equals(coords[dim_name]):
                return xr.DataArray(data.values.copy(), coords=coords, dims=dims)
            else:
                raise ConversionError(
                    f"Series index doesn't match {dim_name} coordinates.\n"
                    f'Series index: {data.index}\n'
                    f'Target {dim_name} coordinates: {coords[dim_name]}'
                )

        # Handle two dimensions case
        elif len(dims) == 2:
            # Check if dimensions are time and scenario
            if dims != ('time', 'scenario'):
                raise ConversionError(
                    f'Two-dimensional conversion only supports time and scenario dimensions, got {dims}'
                )

            # Case 1: Series is indexed by time
            if data.index.equals(coords['time']):
                # Broadcast across scenarios
                values = np.repeat(data.values[:, np.newaxis], len(coords['scenario']), axis=1)
                return xr.DataArray(values.copy(), coords=coords, dims=dims)

            # Case 2: Series is indexed by scenario
            elif data.index.equals(coords['scenario']):
                # Broadcast across time
                values = np.repeat(data.values[np.newaxis, :], len(coords['time']), axis=0)
                return xr.DataArray(values.copy(), coords=coords, dims=dims)

            else:
                raise ConversionError(
                    "Series index must match either 'time' or 'scenario' coordinates.\n"
                    f'Series index: {data.index}\n'
                    f'Target time coordinates: {coords["time"]}\n'
                    f'Target scenario coordinates: {coords["scenario"]}'
                )

        else:
            raise ConversionError(f'Maximum 2 dimensions supported, got {len(dims)}')

    @staticmethod
    def _convert_dataframe(data: pd.DataFrame, coords: Dict[str, pd.Index], dims: Tuple[str, ...]) -> xr.DataArray:
        """
        Convert pandas DataFrame to xarray DataArray.
        Only allows time as index and scenarios as columns.

        Args:
            data: pandas DataFrame to convert
            coords: Target coordinates
            dims: Target dimensions

        Returns:
            DataArray from the pandas DataFrame
        """
        # Single dimension case
        if len(dims) == 1:
            # If DataFrame has one column, treat it like a Series
            if len(data.columns) == 1:
                series = data.iloc[:, 0]
                return DataConverter._convert_series(series, coords, dims)

            raise ConversionError(
                f'When converting DataFrame to single-dimension DataArray, DataFrame must have exactly one column, got {len(data.columns)}'
            )

        # Two dimensions case
        elif len(dims) == 2:
            # Check if dimensions are time and scenario
            if dims != ('time', 'scenario'):
                raise ConversionError(
                    f'Two-dimensional conversion only supports time and scenario dimensions, got {dims}'
                )

            # DataFrame must have time as index and scenarios as columns
            if data.index.equals(coords['time']) and data.columns.equals(coords['scenario']):
                # Create DataArray with proper dimension order
                return xr.DataArray(data.values.copy(), coords=coords, dims=dims)
            else:
                raise ConversionError(
                    'DataFrame must have time as index and scenarios as columns.\n'
                    f'DataFrame index: {data.index}\n'
                    f'DataFrame columns: {data.columns}\n'
                    f'Target time coordinates: {coords["time"]}\n'
                    f'Target scenario coordinates: {coords["scenario"]}'
                )

        else:
            raise ConversionError(f'Maximum 2 dimensions supported, got {len(dims)}')


class TimeSeriesData:
    # TODO: Move to Interface.py
    def __init__(self, data: TimestepData, agg_group: Optional[str] = None, agg_weight: Optional[float] = None):
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
        has_extra_timestep (bool): Whether this series needs an extra timestep
    """

    @classmethod
    def from_datasource(
        cls,
        data: NumericDataTS,
        name: str,
        timesteps: pd.DatetimeIndex,
        scenarios: Optional[pd.Index] = None,
        aggregation_weight: Optional[float] = None,
        aggregation_group: Optional[str] = None,
        has_extra_timestep: bool = False,
    ) -> 'TimeSeries':
        """
        Initialize the TimeSeries from multiple data sources.

        Args:
            data: The time series data
            name: The name of the TimeSeries
            timesteps: The timesteps of the TimeSeries
            scenarios: The scenarios of the TimeSeries
            aggregation_weight: The weight in aggregation calculations
            aggregation_group: Group this TimeSeries belongs to for aggregation weight sharing
            has_extra_timestep: Whether this series requires an extra timestep

        Returns:
            A new TimeSeries instance
        """
        return cls(
            DataConverter.as_dataarray(data, timesteps, scenarios),
            name,
            aggregation_weight,
            aggregation_group,
            has_extra_timestep,
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
            has_extra_timestep=data['has_extra_timestep'],
        )

    def __init__(
        self,
        data: xr.DataArray,
        name: str,
        aggregation_weight: Optional[float] = None,
        aggregation_group: Optional[str] = None,
        has_extra_timestep: bool = False,
    ):
        """
        Initialize a TimeSeries with a DataArray.

        Args:
            data: The DataArray containing time series data
            name: The name of the TimeSeries
            aggregation_weight: The weight in aggregation calculations
            aggregation_group: Group this TimeSeries belongs to for weight sharing
            has_extra_timestep: Whether this series requires an extra timestep

        Raises:
            ValueError: If data has unsupported dimensions
        """
        allowed_dims = {'time', 'scenario'}
        if not set(data.dims).issubset(allowed_dims):
            raise ValueError(f'DataArray dimensions must be subset of {allowed_dims}. Got {data.dims}')

        self.name = name
        self.aggregation_weight = aggregation_weight
        self.aggregation_group = aggregation_group
        self.has_extra_timestep = has_extra_timestep

        # Data management
        self._stored_data = data.copy(deep=True)
        self._backup = self._stored_data.copy(deep=True)

        # Selection state
        self._selected_timesteps: Optional[pd.DatetimeIndex] = None
        self._selected_scenarios: Optional[pd.Index] = None

        # Flag for whether this series has various dimensions
        self.has_time_dim = 'time' in data.dims
        self.has_scenario_dim = 'scenario' in data.dims

    def reset(self) -> None:
        """
        Reset selections to include all timesteps and scenarios.
        This is equivalent to clearing all selections.
        """
        self.set_selection(None, None)

    def restore_data(self) -> None:
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
            'has_extra_timestep': self.has_extra_timestep,
            'data': self.selected_data.to_dict(),
        }

        # Convert datetime objects to ISO strings
        data['data']['coords']['time']['data'] = [date.isoformat() for date in data['data']['coords']['time']['data']]

        # Save to file if path is provided
        if path is not None:
            indent = 4 if len(self.selected_timesteps) <= 480 else None
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
        return get_numeric_stats(self.selected_data, padd=0, by_scenario=True)

    @property
    def all_equal(self) -> bool:
        """Check if all values in the series are equal."""
        return np.unique(self.selected_data.values).size == 1

    @property
    def selected_data(self) -> xr.DataArray:
        """
        Get a view of stored_data based on current selections.
        This computes the view dynamically based on the current selection state.
        """
        return self._stored_data.sel(**self._valid_selector)

    @property
    def selected_timesteps(self) -> Optional[pd.DatetimeIndex]:
        """Get the current active timesteps, or None if no time dimension."""
        if not self.has_time_dim:
            return None
        if self._selected_timesteps is None:
            return self._stored_data.indexes['time']
        return self._selected_timesteps

    @property
    def active_scenarios(self) -> Optional[pd.Index]:
        """Get the current active scenarios, or None if no scenario dimension."""
        if not self.has_scenario_dim:
            return None
        if self._selected_scenarios is None:
            return self._stored_data.indexes['scenario']
        return self._selected_scenarios

    @property
    def stored_data(self) -> xr.DataArray:
        """Get a copy of the full stored data."""
        return self._stored_data.copy()

    def update_stored_data(self, value: xr.DataArray) -> None:
        """
        Update stored_data and refresh selected_data.

        Args:
            value: New data to store
        """
        new_data = DataConverter.as_dataarray(
            value,
            timesteps=self.selected_timesteps if self.has_time_dim else None,
            scenarios=self.active_scenarios if self.has_scenario_dim else None,
        )

        # Skip if data is unchanged to avoid overwriting backup
        if new_data.equals(self._stored_data):
            return

        self._stored_data = new_data
        self.set_selection(None, None)  # Reset selections to full dataset

    def set_selection(self, timesteps: Optional[pd.DatetimeIndex] = None, scenarios: Optional[pd.Index] = None) -> None:
        """
        Set active subset for timesteps and scenarios.

        Args:
            timesteps: Timesteps to activate, or None to clear. Ignored if series has no time dimension.
            scenarios: Scenarios to activate, or None to clear. Ignored if series has no scenario dimension.
        """
        # Only update timesteps if the series has time dimension
        if self.has_time_dim:
            if timesteps is None or timesteps.equals(self._stored_data.indexes['time']):
                self._selected_timesteps = None
            else:
                self._selected_timesteps = timesteps

        # Only update scenarios if the series has scenario dimension
        if self.has_scenario_dim:
            if scenarios is None or scenarios.equals(self._stored_data.indexes['scenario']):
                self._selected_scenarios = None
            else:
                self._selected_scenarios = scenarios

    @property
    def sel(self):
        """Direct access to the selected_data's sel method for convenience."""
        return self.selected_data.sel

    @property
    def isel(self):
        """Direct access to the selected_data's isel method for convenience."""
        return self.selected_data.isel

    @property
    def _valid_selector(self) -> Dict[str, pd.Index]:
        """Get the current selection as a dictionary."""
        selector = {}

        # Only include time in selector if series has time dimension
        if self.has_time_dim and self._selected_timesteps is not None:
            selector['time'] = self._selected_timesteps

        # Only include scenario in selector if series has scenario dimension
        if self.has_scenario_dim and self._selected_scenarios is not None:
            selector['scenario'] = self._selected_scenarios

        return selector

    def _apply_operation(self, other, op):
        """Apply an operation between this TimeSeries and another object."""
        if isinstance(other, TimeSeries):
            other = other.selected_data
        return op(self.selected_data, other)

    def __add__(self, other):
        return self._apply_operation(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self._apply_operation(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self._apply_operation(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self._apply_operation(other, lambda x, y: x / y)

    def __radd__(self, other):
        return other + self.selected_data

    def __rsub__(self, other):
        return other - self.selected_data

    def __rmul__(self, other):
        return other * self.selected_data

    def __rtruediv__(self, other):
        return other / self.selected_data

    def __neg__(self) -> xr.DataArray:
        return -self.selected_data

    def __pos__(self) -> xr.DataArray:
        return +self.selected_data

    def __abs__(self) -> xr.DataArray:
        return abs(self.selected_data)

    def __gt__(self, other):
        """
        Compare if this TimeSeries is greater than another.

        Args:
            other: Another TimeSeries to compare with

        Returns:
            True if all values in this TimeSeries are greater than other
        """
        if isinstance(other, TimeSeries):
            return self.selected_data > other.selected_data
        return self.selected_data > other

    def __ge__(self, other):
        """
        Compare if this TimeSeries is greater than or equal to another.

        Args:
            other: Another TimeSeries to compare with

        Returns:
            True if all values in this TimeSeries are greater than or equal to other
        """
        if isinstance(other, TimeSeries):
            return self.selected_data >= other.selected_data
        return self.selected_data >= other

    def __lt__(self, other):
        """
        Compare if this TimeSeries is less than another.

        Args:
            other: Another TimeSeries to compare with

        Returns:
            True if all values in this TimeSeries are less than other
        """
        if isinstance(other, TimeSeries):
            return self.selected_data < other.selected_data
        return self.selected_data < other

    def __le__(self, other):
        """
        Compare if this TimeSeries is less than or equal to another.

        Args:
            other: Another TimeSeries to compare with

        Returns:
            True if all values in this TimeSeries are less than or equal to other
        """
        if isinstance(other, TimeSeries):
            return self.selected_data <= other.selected_data
        return self.selected_data <= other

    def __eq__(self, other):
        """
        Compare if this TimeSeries is equal to another.

        Args:
            other: Another TimeSeries to compare with

        Returns:
            True if all values in this TimeSeries are equal to other
        """
        if isinstance(other, TimeSeries):
            return self.selected_data == other.selected_data
        return self.selected_data == other

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Handle NumPy universal functions.

        This allows NumPy functions to work with TimeSeries objects.
        """
        # Convert any TimeSeries inputs to their selected_data
        inputs = [x.selected_data if isinstance(x, TimeSeries) else x for x in inputs]
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
            'has_extra_timestep': self.has_extra_timestep,
            'shape': self.selected_data.shape,
        }

        attr_str = ', '.join(f'{k}={repr(v)}' for k, v in attrs.items())
        return f'TimeSeries({attr_str})'

    def __str__(self):
        """
        Get a human-readable string representation.

        Returns:
            Descriptive string with statistics
        """
        return f'TimeSeries "{self.name}":\n{textwrap.indent(self.stats, "  ")}'


class TimeSeriesCollection:
    """
    Simplified central manager for time series data with reference tracking.

    Provides a way to store time series data and work with subsets of dimensions
    that automatically update all references when changed.
    """

    def __init__(
        self,
        timesteps: pd.DatetimeIndex,
        scenarios: Optional[pd.Index] = None,
        hours_of_last_timestep: Optional[float] = None,
        hours_of_previous_timesteps: Optional[Union[float, np.ndarray]] = None,
    ):
        """Initialize a TimeSeriesCollection."""
        self._full_timesteps = self._validate_timesteps(timesteps)
        self._full_scenarios = self._validate_scenarios(scenarios)

        self._full_timesteps_extra = self._create_timesteps_with_extra(
            self._full_timesteps,
            self._calculate_hours_of_final_timestep(
                self._full_timesteps, hours_of_final_timestep=hours_of_last_timestep
            ),
        )
        self._full_hours_per_timestep = self.calculate_hours_per_timestep(
            self._full_timesteps_extra, self._full_scenarios
        )

        self.hours_of_previous_timesteps = self._calculate_hours_of_previous_timesteps(
            timesteps, hours_of_previous_timesteps
        )  # TODO: Make dynamic

        # Series that need extra timestep
        self._has_extra_timestep: set = set()

        # Storage for TimeSeries objects
        self._time_series: Dict[str, TimeSeries] = {}

        # Active subset selectors
        self._selected_timesteps: Optional[pd.DatetimeIndex] = None
        self._selected_scenarios: Optional[pd.Index] = None
        self._selected_timesteps_extra: Optional[pd.DatetimeIndex] = None
        self._selected_hours_per_timestep: Optional[xr.DataArray] = None

    def add_time_series(
        self,
        name: str,
        data: Union[NumericDataTS, TimeSeries],
        has_time_dim: bool = True,
        has_scenario_dim: bool = True,
        aggregation_weight: Optional[float] = None,
        aggregation_group: Optional[str] = None,
        has_extra_timestep: bool = False,
    ) -> TimeSeries:
        """
        Add a new TimeSeries to the allocator.

        Args:
            name: Name of the time series
            data: Data for the time series (can be raw data or an existing TimeSeries)
            has_time_dim: Whether the TimeSeries has a time dimension
            has_scenario_dim: Whether the TimeSeries has a scenario dimension
            aggregation_weight: Weight used for aggregation
            aggregation_group: Group name for shared aggregation weighting
            has_extra_timestep: Whether this series needs an extra timestep

        Returns:
            The created TimeSeries object
        """
        if name in self._time_series:
            raise KeyError(f"TimeSeries '{name}' already exists in allocator")
        if not has_time_dim and has_extra_timestep:
            raise ValueError('A not time-indexed TimeSeries cannot have an extra timestep')

        # Choose which timesteps to use
        if has_time_dim:
            target_timesteps = self.timesteps_extra if has_extra_timestep else self.timesteps
        else:
            target_timesteps = None

        target_scenarios = self.scenarios if has_scenario_dim else None

        # Create or adapt the TimeSeries object
        if isinstance(data, TimeSeries):
            # Use the existing TimeSeries but update its parameters
            time_series = data
            # Update the stored data to use our timesteps and scenarios
            data_array = DataConverter.as_dataarray(
                time_series.stored_data, timesteps=target_timesteps, scenarios=target_scenarios
            )
            time_series = TimeSeries(
                data=data_array,
                name=name,
                aggregation_weight=aggregation_weight or time_series.aggregation_weight,
                aggregation_group=aggregation_group or time_series.aggregation_group,
                has_extra_timestep=has_extra_timestep or time_series.has_extra_timestep,
            )
        else:
            # Create a new TimeSeries from raw data
            time_series = TimeSeries.from_datasource(
                data=data,
                name=name,
                timesteps=target_timesteps,
                scenarios=target_scenarios,
                aggregation_weight=aggregation_weight,
                aggregation_group=aggregation_group,
                has_extra_timestep=has_extra_timestep,
            )

        # Add to storage
        self._time_series[name] = time_series

        # Track if it needs extra timestep
        if has_extra_timestep:
            self._has_extra_timestep.add(name)

        # Return the TimeSeries object
        return time_series

    def set_selection(self, timesteps: Optional[pd.DatetimeIndex] = None, scenarios: Optional[pd.Index] = None) -> None:
        """
        Set active subset for timesteps and scenarios.

        Args:
            timesteps: Timesteps to activate, or None to clear
            scenarios: Scenarios to activate, or None to clear
        """
        if timesteps is None:
            self._selected_timesteps = None
            self._selected_timesteps_extra = None
        else:
            self._selected_timesteps = self._validate_timesteps(timesteps, self._full_timesteps)
            self._selected_timesteps_extra = self._create_timesteps_with_extra(
                timesteps, self._calculate_hours_of_final_timestep(timesteps, self._full_timesteps)
            )

        if scenarios is None:
            self._selected_scenarios = None
        else:
            self._selected_scenarios = self._validate_scenarios(scenarios, self._full_scenarios)

        self._selected_hours_per_timestep = self.calculate_hours_per_timestep(self.timesteps_extra, self.scenarios)

        # Apply the selection to all TimeSeries objects
        for ts_name, ts in self._time_series.items():
            if ts.has_time_dim:
                timesteps = self.timesteps_extra if ts_name in self._has_extra_timestep else self.timesteps
            else:
                timesteps = None

            ts.set_selection(timesteps=timesteps, scenarios=self.scenarios if ts.has_scenario_dim else None)
        self._propagate_selection_to_time_series()

    def as_dataset(self, with_extra_timestep: bool = True, with_constants: bool = True) -> xr.Dataset:
        """
        Convert the TimeSeriesCollection to a xarray Dataset, containing the data of each TimeSeries.

        Args:
            with_extra_timestep: Whether to exclude the extra timesteps.
                Effectively, this removes the last timestep for certain TimeSeries, but mitigates the presence of NANs in others.
            with_constants: Whether to exclude TimeSeries with a constant value from the dataset.
        """
        if self.scenarios is None:
            ds = xr.Dataset(coords={'time': self.timesteps_extra})
        else:
            ds = xr.Dataset(coords={'scenario': self.scenarios, 'time': self.timesteps_extra})

        for ts in self._time_series.values():
            if not with_constants and ts.all_equal:
                continue
            ds[ts.name] = ts.selected_data

        if not with_extra_timestep:
            return ds.sel(time=self.timesteps)

        return ds

    @property
    def timesteps(self) -> pd.DatetimeIndex:
        """Get the current active timesteps."""
        if self._selected_timesteps is None:
            return self._full_timesteps
        return self._selected_timesteps

    @property
    def timesteps_extra(self) -> pd.DatetimeIndex:
        """Get the current active timesteps with extra timestep."""
        if self._selected_timesteps_extra is None:
            return self._full_timesteps_extra
        return self._selected_timesteps_extra

    @property
    def hours_per_timestep(self) -> xr.DataArray:
        """Get the current active hours per timestep."""
        if self._selected_hours_per_timestep is None:
            return self._full_hours_per_timestep
        return self._selected_hours_per_timestep

    @property
    def scenarios(self) -> Optional[pd.Index]:
        """Get the current active scenarios."""
        if self._selected_scenarios is None:
            return self._full_scenarios
        return self._selected_scenarios

    def _propagate_selection_to_time_series(self) -> None:
        """Apply the current selection to all TimeSeries objects."""
        for ts_name, ts in self._time_series.items():
            if ts.has_time_dim:
                    timesteps = self.timesteps_extra if ts_name in self._has_extra_timestep else self.timesteps
            else:
                timesteps = None

            ts.set_selection(timesteps=timesteps, scenarios=self.scenarios if ts.has_scenario_dim else None)

    def __getitem__(self, name: str) -> TimeSeries:
        """
        Get a reference to a time series or data array.

        Args:
            name: Name of the data array or time series

        Returns:
            TimeSeries object if it exists, otherwise DataArray with current selection applied
        """
        # First check if this is a TimeSeries
        if name in self._time_series:
            # Return the TimeSeries object (it will handle selection internally)
            return self._time_series[name]
        raise ValueError(f'No TimeSeries named "{name}" found')

    def __contains__(self, value) -> bool:
        if isinstance(value, str):
            return value in self._time_series
        elif isinstance(value, TimeSeries):
            return value.name in self._time_series
        raise TypeError(f'Invalid type for __contains__ of {self.__class__.__name__}: {type(value)}')

    def __iter__(self) -> Iterator[TimeSeries]:
        """Iterate over TimeSeries objects."""
        return iter(self._time_series.values())

    def update_time_series(self, name: str, data: TimestepData) -> TimeSeries:
        """
        Update an existing TimeSeries with new data.

        Args:
            name: Name of the TimeSeries to update
            data: New data to assign

        Returns:
            The updated TimeSeries

        Raises:
            KeyError: If no TimeSeries with the given name exists
        """
        if name not in self._time_series:
            raise KeyError(f"No TimeSeries named '{name}' found")

        # Get the TimeSeries
        ts = self._time_series[name]

        # Determine which timesteps to use if the series has a time dimension
        if ts.has_time_dim:
            target_timesteps = self.timesteps_extra if name in self._has_extra_timestep else self.timesteps
        else:
            target_timesteps = None

        # Convert data to proper format
        data_array = DataConverter.as_dataarray(
            data, timesteps=target_timesteps, scenarios=self.scenarios if ts.has_scenario_dim else None
        )

        # Update the TimeSeries
        ts.update_stored_data(data_array)

        return ts

    def calculate_aggregation_weights(self) -> Dict[str, float]:
        """Calculate and return aggregation weights for all time series."""
        group_weights = self._calculate_group_weights()

        weights = {}
        for name, ts in self._time_series.items():
            if ts.aggregation_group is not None:
                # Use group weight
                weights[name] = group_weights.get(ts.aggregation_group, 1)
            else:
                # Use individual weight or default to 1
                weights[name] = ts.aggregation_weight or 1

        if np.all(np.isclose(list(weights.values()), 1, atol=1e-6)):
            logger.info('All Aggregation weights were set to 1')

        return weights

    def _calculate_group_weights(self) -> Dict[str, float]:
        """Calculate weights for aggregation groups."""
        # Count series in each group
        groups = [ts.aggregation_group for ts in self._time_series.values() if ts.aggregation_group is not None]
        group_counts = Counter(groups)

        # Calculate weight for each group (1/count)
        return {group: 1 / count for group, count in group_counts.items()}

    @staticmethod
    def _validate_timesteps(
        timesteps: pd.DatetimeIndex, present_timesteps: Optional[pd.DatetimeIndex] = None
    ) -> pd.DatetimeIndex:
        """
        Validate timesteps format and rename if needed.
        Args:
            timesteps: The timesteps to validate
            present_timesteps: The timesteps that are present in the dataset

        Raises:
            ValueError: If timesteps is not a pandas DatetimeIndex
            ValueError: If timesteps is not at least 2 timestamps
            ValueError: If timesteps has a different name than 'time'
            ValueError: If timesteps is not sorted
            ValueError: If timesteps contains duplicates
            ValueError: If timesteps is not a subset of present_timesteps
        """
        if not isinstance(timesteps, pd.DatetimeIndex):
            raise TypeError('timesteps must be a pandas DatetimeIndex')

        if len(timesteps) < 2:
            raise ValueError('timesteps must contain at least 2 timestamps')

        # Ensure timesteps has the required name
        if timesteps.name != 'time':
            logger.debug('Renamed timesteps to "time" (was "%s")', timesteps.name)
            timesteps.name = 'time'

        # Ensure timesteps is sorted
        if not timesteps.is_monotonic_increasing:
            raise ValueError('timesteps must be sorted')

        # Ensure timesteps has no duplicates
        if len(timesteps) != len(timesteps.drop_duplicates()):
            raise ValueError('timesteps must not contain duplicates')

        # Ensure timesteps is a subset of present_timesteps
        if present_timesteps is not None and not set(timesteps).issubset(set(present_timesteps)):
            raise ValueError('timesteps must be a subset of present_timesteps')

        return timesteps

    @staticmethod
    def _validate_scenarios(scenarios: pd.Index, present_scenarios: Optional[pd.Index] = None) -> Optional[pd.Index]:
        """
        Validate scenario format and rename if needed.
        Args:
            scenarios: The scenarios to validate
            present_scenarios: The present_scenarios that are present in the dataset

        Raises:
            ValueError: If timesteps is not a pandas DatetimeIndex
            ValueError: If timesteps is not at least 2 timestamps
            ValueError: If timesteps has a different name than 'time'
            ValueError: If timesteps is not sorted
            ValueError: If timesteps contains duplicates
            ValueError: If timesteps is not a subset of present_timesteps
        """
        if scenarios is None:
            return None

        if not isinstance(scenarios, pd.Index):
            logger.warning('Converting scenarios to pandas.Index')
            scenarios = pd.Index(scenarios, name='scenario')

        # Ensure timesteps has the required name
        if scenarios.name != 'scenario':
            logger.debug('Renamed scenarios to "scneario" (was "%s")', scenarios.name)
            scenarios.name = 'scenario'

        # Ensure timesteps is a subset of present_timesteps
        if present_scenarios is not None and not set(scenarios).issubset(set(present_scenarios)):
            raise ValueError('scenarios must be a subset of present_scenarios')

        return scenarios

    @staticmethod
    def _create_timesteps_with_extra(timesteps: pd.DatetimeIndex, hours_of_last_timestep: float) -> pd.DatetimeIndex:
        """Create timesteps with an extra step at the end."""
        last_date = pd.DatetimeIndex([timesteps[-1] + pd.Timedelta(hours=hours_of_last_timestep)], name='time')
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
    def _calculate_hours_of_final_timestep(
        timesteps: pd.DatetimeIndex,
        timesteps_superset: Optional[pd.DatetimeIndex] = None,
        hours_of_final_timestep: Optional[float] = None,
    ) -> float:
        """
        Calculate duration of the final timestep.
        If timesteps_subset is provided, the final timestep is calculated for this subset.
        The hours_of_final_timestep is only used if the final timestep cant be determined from the timesteps.

        Args:
            timesteps: The full timesteps
            timesteps_subset: The subset of timesteps
            hours_of_final_timestep: The duration of the final timestep, if already known

        Returns:
            The duration of the final timestep in hours

        Raises:
            ValueError: If the provided timesteps_subset does not end before the timesteps superset
        """
        if timesteps_superset is None:
            if hours_of_final_timestep is not None:
                return hours_of_final_timestep
            return (timesteps[-1] - timesteps[-2]) / pd.Timedelta(hours=1)

        final_timestep = timesteps[-1]

        if timesteps_superset[-1] == final_timestep:
            if hours_of_final_timestep is not None:
                return hours_of_final_timestep
            return (timesteps_superset[-1] - timesteps_superset[-2]) / pd.Timedelta(hours=1)

        elif timesteps_superset[-1] <= final_timestep:
            raise ValueError(
                f'The provided timesteps ({timesteps}) end after the provided timesteps_superset ({timesteps_superset})'
            )
        else:
            # Get the first timestep in the superset that is after the final timestep of the subset
            extra_timestep = timesteps_superset[timesteps_superset > final_timestep].min()
            return (extra_timestep - final_timestep) / pd.Timedelta(hours=1)

    @staticmethod
    def calculate_hours_per_timestep(
        timesteps_extra: pd.DatetimeIndex, scenarios: Optional[pd.Index] = None
    ) -> xr.DataArray:
        """Calculate duration of each timestep."""
        # Calculate differences between consecutive timestamps
        hours_per_step = np.diff(timesteps_extra) / pd.Timedelta(hours=1)

        return DataConverter.as_dataarray(
            hours_per_step,
            timesteps=timesteps_extra[:-1],
            scenarios=scenarios,
        ).rename('hours_per_step')


def get_numeric_stats(data: xr.DataArray, decimals: int = 2, padd: int = 10, by_scenario: bool = False) -> str:
    """
    Calculates the mean, median, min, max, and standard deviation of a numeric DataArray.

    Args:
        data: The DataArray to analyze
        decimals: Number of decimal places to show
        padd: Padding for alignment
        by_scenario: Whether to break down stats by scenario

    Returns:
        String representation of data statistics
    """
    format_spec = f'>{padd}.{decimals}f' if padd else f'.{decimals}f'

    # If by_scenario is True and there's a scenario dimension with multiple values
    if by_scenario and 'scenario' in data.dims and data.sizes['scenario'] > 1:
        results = []
        for scenario in data.coords['scenario'].values:
            scenario_data = data.sel(scenario=scenario)
            if np.unique(scenario_data).size == 1:
                results.append(f'  {scenario}: {scenario_data.max().item():{format_spec}} (constant)')
            else:
                mean = scenario_data.mean().item()
                median = scenario_data.median().item()
                min_val = scenario_data.min().item()
                max_val = scenario_data.max().item()
                std = scenario_data.std().item()
                results.append(
                    f'  {scenario}: {mean:{format_spec}} (mean), {median:{format_spec}} (median), '
                    f'{min_val:{format_spec}} (min), {max_val:{format_spec}} (max), {std:{format_spec}} (std)'
                )
        return '\n'.join(['By scenario:'] + results)

    # Standard logic for non-scenario data or aggregated stats
    if np.unique(data).size == 1:
        return f'{data.max().item():{format_spec}} (constant)'

    mean = data.mean().item()
    median = data.median().item()
    min_val = data.min().item()
    max_val = data.max().item()
    std = data.std().item()

    return f'{mean:{format_spec}} (mean), {median:{format_spec}} (median), {min_val:{format_spec}} (min), {max_val:{format_spec}} (max), {std:{format_spec}} (std)'


def extract_data(
    data: Optional[Union[int, float, xr.DataArray, TimeSeries]],
    if_none: Any = None
) -> Any:
    """
    Convert data to xr.DataArray.

    Args:
        data: The data to convert (scalar, array, or DataArray)
        if_none: The value to return if data is None

    Returns:
        DataArray with the converted data, or the value specified by if_none
    """
    if data is None:
        return if_none
    if isinstance(data, TimeSeries):
        return data.selected_data
    if isinstance(data, xr.DataArray):
        return data
    if isinstance(data, (int, float, np.integer, np.floating)):
        return data
    raise TypeError(f'Unsupported data type: {type(data).__name__}')
