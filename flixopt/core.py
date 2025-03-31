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


class PlausibilityError(Exception):
    """Error for a failing Plausibility check."""

    pass


class ConversionError(Exception):
    """Base exception for data conversion errors."""

    pass


class DataConverter:
    """
    Converts various data types into xarray.DataArray with timesteps and optional scenarios dimensions.

    Supports:
    - Scalar values (broadcast to all timesteps/scenarios)
    - 1D arrays (mapped to timesteps, broadcast to scenarios if provided)
    - 2D arrays (mapped to scenarios × timesteps if dimensions match)
    - Series with time index (broadcast to scenarios if provided)
    - DataFrames with time index and a single column (broadcast to scenarios if provided)
    - Series/DataFrames with MultiIndex (scenario, time)
    - Existing DataArrays
    """

    #TODO: Allow DataFrame with scenarios as columns

    @staticmethod
    def as_dataarray(
        data: NumericData, timesteps: pd.DatetimeIndex, scenarios: Optional[pd.Index] = None
    ) -> xr.DataArray:
        """
        Convert data to xarray.DataArray with specified timesteps and optional scenarios dimensions.

        Args:
            data: The data to convert (scalar, array, Series, DataFrame, or DataArray)
            timesteps: DatetimeIndex representing the time dimension (must be named 'time')
            scenarios: Optional Index representing scenarios (must be named 'scenario')

        Returns:
            DataArray with the converted data

        Raises:
            ValueError: If timesteps or scenarios are invalid
            ConversionError: If the data cannot be converted to the expected dimensions
        """
        # Validate inputs
        DataConverter._validate_timesteps(timesteps)
        if scenarios is not None:
            DataConverter._validate_scenarios(scenarios)

        # Determine dimensions and coordinates
        coords, dims, expected_shape = DataConverter._get_dimensions(timesteps, scenarios)

        try:
            # Convert different data types using specialized methods
            if isinstance(data, (int, float, np.integer, np.floating)):
                return DataConverter._convert_scalar(data, coords, dims)

            elif isinstance(data, pd.DataFrame):
                return DataConverter._convert_dataframe(data, timesteps, scenarios, coords, dims)

            elif isinstance(data, pd.Series):
                return DataConverter._convert_series(data, timesteps, scenarios, coords, dims)

            elif isinstance(data, np.ndarray):
                return DataConverter._convert_ndarray(data, timesteps, scenarios, coords, dims, expected_shape)

            elif isinstance(data, xr.DataArray):
                return DataConverter._convert_dataarray(data, timesteps, scenarios, coords, dims)

            else:
                raise ConversionError(f'Unsupported type: {type(data).__name__}')

        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(f'Converting {type(data)} to DataArray raised an error: {str(e)}') from e

    @staticmethod
    def _validate_timesteps(timesteps: pd.DatetimeIndex) -> None:
        """
        Validate that timesteps is a properly named non-empty DatetimeIndex.

        Args:
            timesteps: The DatetimeIndex to validate

        Raises:
            ValueError: If timesteps is not a non-empty DatetimeIndex
            ConversionError: If timesteps is not named 'time'
        """
        if not isinstance(timesteps, pd.DatetimeIndex) or len(timesteps) == 0:
            raise ValueError(f'Timesteps must be a non-empty DatetimeIndex, got {type(timesteps).__name__}')
        if timesteps.name != 'time':
            raise ConversionError(f'DatetimeIndex must be named "time", got {timesteps.name=}')

    @staticmethod
    def _validate_scenarios(scenarios: pd.Index) -> None:
        """
        Validate that scenarios is a properly named non-empty Index.

        Args:
            scenarios: The Index to validate

        Raises:
            ValueError: If scenarios is not a non-empty Index
            ConversionError: If scenarios is not named 'scenario'
        """
        if not isinstance(scenarios, pd.Index) or len(scenarios) == 0:
            raise ValueError(f'Scenarios must be a non-empty Index, got {type(scenarios).__name__}')
        if scenarios.name != 'scenario':
            raise ConversionError(f'Scenarios Index must be named "scenario", got {scenarios.name=}')

    @staticmethod
    def _get_dimensions(
        timesteps: pd.DatetimeIndex, scenarios: Optional[pd.Index] = None
    ) -> Tuple[Dict[str, pd.Index], Tuple[str, ...], Tuple[int, ...]]:
        """
        Create the coordinates, dimensions, and expected shape for the output DataArray.

        Args:
            timesteps: The time index
            scenarios: Optional scenario index

        Returns:
            Tuple containing:
            - Dict mapping dimension names to coordinate indexes
            - Tuple of dimension names
            - Tuple of expected shape
        """
        if scenarios is not None:
            coords = {'scenario': scenarios, 'time': timesteps}
            dims = ('scenario', 'time')
            expected_shape = (len(scenarios), len(timesteps))
        else:
            coords = {'time': timesteps}
            dims = ('time',)
            expected_shape = (len(timesteps),)

        return coords, dims, expected_shape

    @staticmethod
    def _convert_scalar(
        data: Union[int, float, np.integer, np.floating], coords: Dict[str, pd.Index], dims: Tuple[str, ...]
    ) -> xr.DataArray:
        """
        Convert a scalar value to a DataArray.

        Args:
            data: The scalar value to convert
            coords: Dictionary mapping dimension names to coordinate indexes
            dims: Tuple of dimension names

        Returns:
            DataArray with the scalar value broadcast to all coordinates
        """
        return xr.DataArray(data, coords=coords, dims=dims)

    @staticmethod
    def _convert_dataframe(
        df: pd.DataFrame,
        timesteps: pd.DatetimeIndex,
        scenarios: Optional[pd.Index],
        coords: Dict[str, pd.Index],
        dims: Tuple[str, ...],
    ) -> xr.DataArray:
        """
        Convert a DataFrame to a DataArray.

        Args:
            df: The DataFrame to convert
            timesteps: The time index
            scenarios: Optional scenario index
            coords: Dictionary mapping dimension names to coordinate indexes
            dims: Tuple of dimension names

        Returns:
            DataArray created from the DataFrame

        Raises:
            ConversionError: If the DataFrame cannot be converted to the expected dimensions
        """
        # Case 1: DataFrame with MultiIndex (scenario, time)
        if (
            isinstance(df.index, pd.MultiIndex)
            and len(df.index.names) == 2
            and 'scenario' in df.index.names
            and 'time' in df.index.names
            and scenarios is not None
        ):
            return DataConverter._convert_multi_index_dataframe(df, timesteps, scenarios, coords, dims)

        # Case 2: Standard DataFrame with time index
        elif not isinstance(df.index, pd.MultiIndex):
            return DataConverter._convert_standard_dataframe(df, timesteps, scenarios, coords, dims)

        else:
            raise ConversionError(f'Unsupported DataFrame index structure: {df}')

    @staticmethod
    def _convert_multi_index_dataframe(
        df: pd.DataFrame,
        timesteps: pd.DatetimeIndex,
        scenarios: pd.Index,
        coords: Dict[str, pd.Index],
        dims: Tuple[str, ...],
    ) -> xr.DataArray:
        """
        Convert a DataFrame with MultiIndex (scenario, time) to a DataArray.

        Args:
            df: The DataFrame with MultiIndex to convert
            timesteps: The time index
            scenarios: The scenario index
            coords: Dictionary mapping dimension names to coordinate indexes
            dims: Tuple of dimension names

        Returns:
            DataArray created from the MultiIndex DataFrame

        Raises:
            ConversionError: If the DataFrame's index doesn't match expected or has multiple columns
        """
        # Validate that the index contains the expected values
        if not set(df.index.get_level_values('time')).issubset(set(timesteps)):
            raise ConversionError("DataFrame time index doesn't match or isn't a subset of timesteps")
        if not set(df.index.get_level_values('scenario')).issubset(set(scenarios)):
            raise ConversionError("DataFrame scenario index doesn't match or isn't a subset of scenarios")

        # Ensure single column
        if len(df.columns) != 1:
            raise ConversionError('DataFrame must have exactly one column')

        # Reindex to ensure complete coverage and correct order
        multi_idx = pd.MultiIndex.from_product([scenarios, timesteps], names=['scenario', 'time'])
        reindexed = df.reindex(multi_idx).iloc[:, 0]

        # Reshape to 2D array
        reshaped = reindexed.values.reshape(len(scenarios), len(timesteps))
        return xr.DataArray(reshaped, coords=coords, dims=dims)

    @staticmethod
    def _convert_standard_dataframe(
        df: pd.DataFrame,
        timesteps: pd.DatetimeIndex,
        scenarios: Optional[pd.Index],
        coords: Dict[str, pd.Index],
        dims: Tuple[str, ...],
    ) -> xr.DataArray:
        """
        Convert a standard DataFrame with time index to a DataArray.

        Args:
            df: The DataFrame to convert
            timesteps: The time index
            scenarios: Optional scenario index
            coords: Dictionary mapping dimension names to coordinate indexes
            dims: Tuple of dimension names

        Returns:
            DataArray created from the DataFrame

        Raises:
            ConversionError: If the DataFrame's index doesn't match timesteps or has multiple columns
        """
        if not df.index.equals(timesteps):
            raise ConversionError("DataFrame index doesn't match timesteps index")
        if len(df.columns) != 1:
            raise ConversionError('DataFrame must have exactly one column')

        # Get values
        values = df.values.flatten()

        if scenarios is not None:
            # Broadcast to scenarios dimension
            values = np.tile(values, (len(scenarios), 1))

        return xr.DataArray(values, coords=coords, dims=dims)

    @staticmethod
    def _convert_series(
        series: pd.Series,
        timesteps: pd.DatetimeIndex,
        scenarios: Optional[pd.Index],
        coords: Dict[str, pd.Index],
        dims: Tuple[str, ...],
    ) -> xr.DataArray:
        """
        Convert a Series to a DataArray.

        Args:
            series: The Series to convert
            timesteps: The time index
            scenarios: Optional scenario index
            coords: Dictionary mapping dimension names to coordinate indexes
            dims: Tuple of dimension names

        Returns:
            DataArray created from the Series

        Raises:
            ConversionError: If the Series cannot be converted to the expected dimensions
        """
        # Case 1: Series with MultiIndex (scenario, time)
        if (
            isinstance(series.index, pd.MultiIndex)
            and len(series.index.names) == 2
            and 'scenario' in series.index.names
            and 'time' in series.index.names
            and scenarios is not None
        ):
            return DataConverter._convert_multi_index_series(series, timesteps, scenarios, coords, dims)

        # Case 2: Standard Series with time index
        elif not isinstance(series.index, pd.MultiIndex):
            return DataConverter._convert_standard_series(series, timesteps, scenarios, coords, dims)

        else:
            raise ConversionError('Unsupported Series index structure')

    @staticmethod
    def _convert_multi_index_series(
        series: pd.Series,
        timesteps: pd.DatetimeIndex,
        scenarios: pd.Index,
        coords: Dict[str, pd.Index],
        dims: Tuple[str, ...],
    ) -> xr.DataArray:
        """
        Convert a Series with MultiIndex (scenario, time) to a DataArray.

        Args:
            series: The Series with MultiIndex to convert
            timesteps: The time index
            scenarios: The scenario index
            coords: Dictionary mapping dimension names to coordinate indexes
            dims: Tuple of dimension names

        Returns:
            DataArray created from the MultiIndex Series

        Raises:
            ConversionError: If the Series' index doesn't match expected
        """
        # Validate that the index contains the expected values
        if not set(series.index.get_level_values('time')).issubset(set(timesteps)):
            raise ConversionError("Series time index doesn't match or isn't a subset of timesteps")
        if not set(series.index.get_level_values('scenario')).issubset(set(scenarios)):
            raise ConversionError("Series scenario index doesn't match or isn't a subset of scenarios")

        # Reindex to ensure complete coverage and correct order
        multi_idx = pd.MultiIndex.from_product([scenarios, timesteps], names=['scenario', 'time'])
        reindexed = series.reindex(multi_idx)

        # Reshape to 2D array
        reshaped = reindexed.values.reshape(len(scenarios), len(timesteps))
        return xr.DataArray(reshaped, coords=coords, dims=dims)

    @staticmethod
    def _convert_standard_series(
        series: pd.Series,
        timesteps: pd.DatetimeIndex,
        scenarios: Optional[pd.Index],
        coords: Dict[str, pd.Index],
        dims: Tuple[str, ...],
    ) -> xr.DataArray:
        """
        Convert a standard Series with time index to a DataArray.

        Args:
            series: The Series to convert
            timesteps: The time index
            scenarios: Optional scenario index
            coords: Dictionary mapping dimension names to coordinate indexes
            dims: Tuple of dimension names

        Returns:
            DataArray created from the Series

        Raises:
            ConversionError: If the Series' index doesn't match timesteps
        """
        if not series.index.equals(timesteps):
            raise ConversionError("Series index doesn't match timesteps index")

        # Get values
        values = series.values

        if scenarios is not None:
            # Broadcast to scenarios dimension
            values = np.tile(values, (len(scenarios), 1))

        return xr.DataArray(values, coords=coords, dims=dims)

    @staticmethod
    def _convert_ndarray(
        arr: np.ndarray,
        timesteps: pd.DatetimeIndex,
        scenarios: Optional[pd.Index],
        coords: Dict[str, pd.Index],
        dims: Tuple[str, ...],
        expected_shape: Tuple[int, ...],
    ) -> xr.DataArray:
        """
        Convert a numpy array to a DataArray.

        Args:
            arr: The numpy array to convert
            timesteps: The time index
            scenarios: Optional scenario index
            coords: Dictionary mapping dimension names to coordinate indexes
            dims: Tuple of dimension names
            expected_shape: Expected shape of the resulting array

        Returns:
            DataArray created from the numpy array

        Raises:
            ConversionError: If the array cannot be converted to the expected dimensions
        """
        # Case 1: With scenarios - array can be 1D or 2D
        if scenarios is not None:
            return DataConverter._convert_ndarray_with_scenarios(
                arr, timesteps, scenarios, coords, dims, expected_shape
            )

        # Case 2: Without scenarios - array must be 1D
        else:
            return DataConverter._convert_ndarray_without_scenarios(arr, timesteps, coords, dims)

    @staticmethod
    def _convert_ndarray_with_scenarios(
        arr: np.ndarray,
        timesteps: pd.DatetimeIndex,
        scenarios: pd.Index,
        coords: Dict[str, pd.Index],
        dims: Tuple[str, ...],
        expected_shape: Tuple[int, ...],
    ) -> xr.DataArray:
        """
        Convert a numpy array to a DataArray with scenarios dimension.

        Args:
            arr: The numpy array to convert
            timesteps: The time index
            scenarios: The scenario index
            coords: Dictionary mapping dimension names to coordinate indexes
            dims: Tuple of dimension names
            expected_shape: Expected shape (scenarios, timesteps)

        Returns:
            DataArray created from the numpy array

        Raises:
            ConversionError: If the array dimensions don't match expected
        """
        if arr.ndim == 1:
            # 1D array should match timesteps and be broadcast to scenarios
            if arr.shape[0] != len(timesteps):
                raise ConversionError(f"1D array length {arr.shape[0]} doesn't match timesteps length {len(timesteps)}")
            # Broadcast to scenarios
            values = np.tile(arr, (len(scenarios), 1))
            return xr.DataArray(values, coords=coords, dims=dims)

        elif arr.ndim == 2:
            # 2D array should match (scenarios, timesteps)
            if arr.shape != expected_shape:
                raise ConversionError(f"2D array shape {arr.shape} doesn't match expected shape {expected_shape}")
            return xr.DataArray(arr, coords=coords, dims=dims)

        else:
            raise ConversionError(f'Array must be 1D or 2D, got {arr.ndim}D')

    @staticmethod
    def _convert_ndarray_without_scenarios(
        arr: np.ndarray, timesteps: pd.DatetimeIndex, coords: Dict[str, pd.Index], dims: Tuple[str, ...]
    ) -> xr.DataArray:
        """
        Convert a numpy array to a DataArray without scenarios dimension.

        Args:
            arr: The numpy array to convert
            timesteps: The time index
            coords: Dictionary mapping dimension names to coordinate indexes
            dims: Tuple of dimension names

        Returns:
            DataArray created from the numpy array

        Raises:
            ConversionError: If the array isn't 1D or doesn't match timesteps length
        """
        if arr.ndim != 1:
            raise ConversionError(f'Without scenarios, array must be 1D, got {arr.ndim}D')
        if arr.shape[0] != len(timesteps):
            raise ConversionError(f"Array shape {arr.shape} doesn't match expected length {len(timesteps)}")
        return xr.DataArray(arr, coords=coords, dims=dims)

    @staticmethod
    def _convert_dataarray(
        da: xr.DataArray,
        timesteps: pd.DatetimeIndex,
        scenarios: Optional[pd.Index],
        coords: Dict[str, pd.Index],
        dims: Tuple[str, ...],
    ) -> xr.DataArray:
        """
        Convert an existing DataArray to a new DataArray with the desired dimensions.

        Args:
            da: The DataArray to convert
            timesteps: The time index
            scenarios: Optional scenario index
            coords: Dictionary mapping dimension names to coordinate indexes
            dims: Tuple of dimension names

        Returns:
            New DataArray with the specified coordinates and dimensions

        Raises:
            ConversionError: If the DataArray dimensions don't match expected
        """
        # Case 1: DataArray with only time dimension when scenarios are provided
        if scenarios is not None and set(da.dims) == {'time'}:
            return DataConverter._broadcast_time_only_dataarray(da, timesteps, scenarios, coords, dims)

        # Case 2: DataArray dimensions should match expected
        elif set(da.dims) != set(dims):
            raise ConversionError(f"DataArray dimensions {da.dims} don't match expected {dims}")

        # Validate dimensions sizes
        for dim in dims:
            if not np.array_equal(da.coords[dim].values, coords[dim].values):
                raise ConversionError(f"DataArray dimension '{dim}' doesn't match expected {coords[dim]}")

        # Create a new DataArray with our coordinates to ensure consistency
        result = xr.DataArray(da.values.copy(), coords=coords, dims=dims)
        return result

    @staticmethod
    def _broadcast_time_only_dataarray(
        da: xr.DataArray,
        timesteps: pd.DatetimeIndex,
        scenarios: pd.Index,
        coords: Dict[str, pd.Index],
        dims: Tuple[str, ...],
    ) -> xr.DataArray:
        """
        Broadcast a time-only DataArray to include the scenarios dimension.

        Args:
            da: The DataArray with only time dimension
            timesteps: The time index
            scenarios: The scenario index
            coords: Dictionary mapping dimension names to coordinate indexes
            dims: Tuple of dimension names

        Returns:
            DataArray with the data broadcast to include scenarios dimension

        Raises:
            ConversionError: If the DataArray time coordinates aren't compatible with timesteps
        """
        # Ensure the time dimension is compatible
        if not np.array_equal(da.coords['time'].values, timesteps.values):
            raise ConversionError("DataArray time coordinates aren't compatible with timesteps")

        # Broadcast to scenarios
        values = np.tile(da.values.copy(), (len(scenarios), 1))
        return xr.DataArray(values, coords=coords, dims=dims)


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
            ValueError: If data doesn't have a 'time' index or has unsupported dimensions
        """
        if 'time' not in data.indexes:
            raise ValueError(f'DataArray must have a "time" index. Got {data.indexes}')

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
        self.clear_selection()

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
    def active_timesteps(self) -> Optional[pd.DatetimeIndex]:
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
            timesteps=self.active_timesteps,
            scenarios=self.active_scenarios if self._has_scenarios else None
        )

        # Skip if data is unchanged to avoid overwriting backup
        if new_data.equals(self._stored_data):
            return

        self._stored_data = new_data
        self.clear_selection()  # Reset selections to full dataset

    def clear_selection(self, timesteps: bool = True, scenarios: bool = True) -> None:
        if timesteps:
            self._selected_timesteps = None
        if scenarios:
            self._selected_scenarios = None

    def set_selection(self, timesteps: Optional[pd.DatetimeIndex] = None, scenarios: Optional[pd.Index] = None) -> None:
        """
        Set active subset for timesteps and scenarios.

        Args:
            timesteps: Timesteps to activate, or None to clear. Ignored if series has no time dimension.
            scenarios: Scenarios to activate, or None to clear. Ignored if series has no scenario dimension.
        """
        # Only update timesteps if the series has time dimension
        if self.has_time_dim:
            if timesteps is None:
                self.clear_selection(timesteps=True, scenarios=False)
            else:
                self._selected_timesteps = timesteps

        # Only update scenarios if the series has scenario dimension
        if self.has_scenario_dim:
            if scenarios is None:
                self.clear_selection(timesteps=False, scenarios=True)
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
            return (self.selected_data > other.selected_data).all().item()
        return NotImplemented

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
        self._validate_timesteps(timesteps)
        self.hours_of_previous_timesteps = self._calculate_hours_of_previous_timesteps(
            timesteps, hours_of_previous_timesteps
        ) #TODO: Make dynamic

        self._full_timesteps = timesteps
        self._full_timesteps_extra = self._create_timesteps_with_extra(timesteps, hours_of_last_timestep)
        self._full_hours_per_timestep = self.calculate_hours_per_timestep(self._full_timesteps_extra)

        self._full_scenarios = scenarios

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

    def clear_selection(self, timesteps: bool = True, scenarios: bool = True) -> None:
        """
        Clear selection for timesteps and/or scenarios.

        Args:
            timesteps: Whether to clear timesteps selection
            scenarios: Whether to clear scenarios selection
        """
        if timesteps:
            self._update_selected_timesteps(timesteps=None)
        if scenarios:
            self._selected_scenarios = None

        # Apply the selection to all TimeSeries objects
        self._propagate_selection_to_time_series()

    def set_selection(self, timesteps: Optional[pd.DatetimeIndex] = None, scenarios: Optional[pd.Index] = None) -> None:
        """
        Set active subset for timesteps and scenarios.

        Args:
            timesteps: Timesteps to activate, or None to clear
            scenarios: Scenarios to activate, or None to clear
        """
        if timesteps is None:
            self.clear_selection(timesteps=True, scenarios=False)
        else:
            self._update_selected_timesteps(timesteps)

        if scenarios is None:
            self.clear_selection(timesteps=False, scenarios=True)
        else:
            self._selected_scenarios = scenarios

        # Apply the selection to all TimeSeries objects
        self._propagate_selection_to_time_series()

    def _update_selected_timesteps(self, timesteps: Optional[pd.DatetimeIndex]) -> None:
        """
        Updates the timestep and related metrics (timesteps_extra, hours_per_timestep) based on the current selection.
        """
        if timesteps is None:
            self._selected_timesteps = None
            self._selected_timesteps_extra = None
            self._selected_hours_per_timestep = None
            return

        self._validate_timesteps(timesteps, self._full_timesteps)

        self._selected_timesteps = timesteps
        self._selected_hours_per_timestep = self._full_hours_per_timestep.sel(time=timesteps)
        self._selected_timesteps_extra = self._create_timesteps_with_extra(
            timesteps, self._selected_hours_per_timestep.isel(time=-1).max().item()
        )

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
                timesteps = (
                    self.timesteps_extra if ts_name in self._has_extra_timestep else self.timesteps
                )
            else:
                timesteps = None

            ts.set_selection(
                timesteps=timesteps,
                scenarios=self.scenarios if ts.has_scenario_dim else None
            )

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

    def update_time_series(self, name: str, data: NumericData) -> TimeSeries:
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
            data,
            timesteps=target_timesteps,
            scenarios=self.scenarios if ts.has_scenario_dim else None
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
    def _validate_timesteps(timesteps: pd.DatetimeIndex, present_timesteps: Optional[pd.DatetimeIndex] = None):
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
            logger.warning('Renamed timesteps to "time" (was "%s")', timesteps.name)
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
