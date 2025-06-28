"""
This module contains the core functionality of the flixopt framework.
It provides Datatypes, logging functionality, and some functions to transform data structures.
"""

import logging
import warnings
from typing import Dict, Optional, Union, Tuple

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger('flixopt')

Scalar = Union[int, float]
"""A single number, either integer or float."""

TemporalDataUser = Union[
    int, float, np.integer, np.floating, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, 'TimeSeriesData'
]
"""User data which might have a time dimension. Internally converted to an xr.DataArray with time dimension."""

TemporalData = Union[xr.DataArray, 'TimeSeriesData']
"""Internally used datatypes for temporal data (data with a time dimension)."""

NonTemporalDataUser = Union[int, float, np.integer, np.floating, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray]
"""User data which has no time dimension. Internally converted to a Scalar or an xr.DataArray without a time dimension."""

NonTemporalData = Union[Scalar, xr.DataArray]
"""Internally used datatypes for non-temporal data. Can be a Scalar or an xr.DataArray."""


class PlausibilityError(Exception):
    """Error for a failing Plausibility check."""

    pass


class ConversionError(Exception):
    """Base exception for data conversion errors."""

    pass


class TimeSeriesData(xr.DataArray):
    """Minimal TimeSeriesData that inherits from xr.DataArray with aggregation metadata."""

    __slots__ = ()  # No additional instance attributes - everything goes in attrs

    def __init__(self, *args, aggregation_group: Optional[str] = None, aggregation_weight: Optional[float] = None,
                 agg_group: Optional[str] = None, agg_weight: Optional[float] = None, **kwargs):
        """
        Args:
            *args: Arguments passed to DataArray
            aggregation_group: Aggregation group name
            aggregation_weight: Aggregation weight (0-1)
            agg_group: Deprecated, use aggregation_group instead
            agg_weight: Deprecated, use aggregation_weight instead
            **kwargs: Additional arguments passed to DataArray
        """
        if agg_group is not None:
            warnings.warn('agg_group is deprecated, use aggregation_group instead', DeprecationWarning, stacklevel=2)
            aggregation_group = agg_group
        if agg_weight is not None:
            warnings.warn('agg_weight is deprecated, use aggregation_weight instead', DeprecationWarning, stacklevel=2)
            aggregation_weight = agg_weight

        if (aggregation_group is not None) and (aggregation_weight is not None):
            raise ValueError('Use either aggregation_group or aggregation_weight, not both')

        # Let xarray handle all the initialization complexity
        super().__init__(*args, **kwargs)

        # Add our metadata to attrs after initialization
        if aggregation_group is not None:
            self.attrs['aggregation_group'] = aggregation_group
        if aggregation_weight is not None:
            self.attrs['aggregation_weight'] = aggregation_weight

        # Always mark as TimeSeriesData
        self.attrs['__timeseries_data__'] = True

    @property
    def aggregation_group(self) -> Optional[str]:
        return self.attrs.get('aggregation_group')

    @property
    def aggregation_weight(self) -> Optional[float]:
        return self.attrs.get('aggregation_weight')

    @classmethod
    def from_dataarray(cls, da: xr.DataArray, aggregation_group: Optional[str] = None, aggregation_weight: Optional[float] = None):
        """Create TimeSeriesData from DataArray, extracting metadata from attrs."""
        # Get aggregation metadata from attrs or parameters
        final_aggregation_group = aggregation_group if aggregation_group is not None else da.attrs.get('aggregation_group')
        final_aggregation_weight = aggregation_weight if aggregation_weight is not None else da.attrs.get('aggregation_weight')

        return cls(da, aggregation_group=final_aggregation_group, aggregation_weight=final_aggregation_weight)

    @classmethod
    def is_timeseries_data(cls, obj) -> bool:
        """Check if an object is TimeSeriesData."""
        return isinstance(obj, xr.DataArray) and obj.attrs.get('__timeseries_data__', False)

    def __repr__(self):
        agg_info = []
        if self.aggregation_group:
            agg_info.append(f"aggregation_group='{self.aggregation_group}'")
        if self.aggregation_weight is not None:
            agg_info.append(f'aggregation_weight={self.aggregation_weight}')

        info_str = f'TimeSeriesData({", ".join(agg_info)})' if agg_info else 'TimeSeriesData'
        return f'{info_str}\n{super().__repr__()}'

    @property
    def agg_group(self):
        warnings.warn('agg_group is deprecated, use aggregation_group instead', DeprecationWarning, stacklevel=2)
        return self._aggregation_group

    @property
    def agg_weight(self):
        warnings.warn('agg_weight is deprecated, use aggregation_weight instead', DeprecationWarning, stacklevel=2)
        return self._aggregation_weight


class DataConverter:
    """
    Converts various data types into xarray.DataArray with optional time and scenario dimension.

    Supports: scalars, arrays, Series, DataFrames, DataArrays, and TimeSeriesData.
    """

    @staticmethod
    def _fix_timeseries_data_indexing(
        data: TimeSeriesData, timesteps: pd.DatetimeIndex, dims: list, coords: list
    ) -> TimeSeriesData:
        """
        Fix TimeSeriesData indexing issues and return properly indexed TimeSeriesData.

        Args:
            data: TimeSeriesData that might have indexing issues
            timesteps: Target timesteps
            dims: Expected dimensions
            coords: Expected coordinates

        Returns:
            TimeSeriesData with correct indexing

        Raises:
            ConversionError: If data cannot be fixed to match expected indexing
        """
        expected_shape = (len(timesteps),)

        # Check if dimensions match
        if data.dims != tuple(dims):
            logger.warning(
                f'TimeSeriesData has dimensions {data.dims}, expected {dims}. Reshaping to match timesteps. To avoid '
                f'this warning, create a correctly shaped DataArray with the correct dimensions in the first place.'
            )
            # Try to reshape the data to match expected dimensions
            if data.size != len(timesteps):
                raise ConversionError(
                    f'TimeSeriesData has {data.size} elements, cannot reshape to match {len(timesteps)} timesteps'
                )
            # Create new DataArray with correct coordinates, preserving metadata
            reshaped_data = xr.DataArray(
                data.values.reshape(expected_shape), coords=coords, dims=dims, name=data.name, attrs=data.attrs.copy()
            )
            return TimeSeriesData(reshaped_data)

        # Check if time coordinate length matches
        elif data.sizes[dims[0]] != len(coords[0]):
            logger.warning(
                f'TimeSeriesData has {data.sizes[dims[0]]} time points, '
                f"expected {len(coords[0])}. Cannot reindex - lengths don't match."
            )
            raise ConversionError(
                f"TimeSeriesData length {data.sizes[dims[0]]} doesn't match expected {len(coords[0])}"
            )

        # Check if time coordinates are identical
        elif not data.coords['time'].equals(timesteps):
            logger.warning(
                'TimeSeriesData has different time coordinates than expected. Replacing with provided timesteps.'
            )
            # Replace time coordinates while preserving data and metadata
            recoordinated_data = xr.DataArray(
                data.values, coords=coords, dims=dims, name=data.name, attrs=data.attrs.copy()
            )
            return TimeSeriesData(recoordinated_data)

        else:
            # Everything matches - return copy to avoid modifying original
            return data.copy(deep=True)

    @staticmethod
    def to_dataarray(
            data: Union[TemporalData, NonTemporalData], timesteps: Optional[pd.DatetimeIndex] = None, scenarios: Optional[pd.Index] = None
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


def get_dataarray_stats(arr: xr.DataArray) -> Dict:
    """Generate statistical summary of a DataArray."""
    stats = {}
    if arr.dtype.kind in 'biufc':  # bool, int, uint, float, complex
        try:
            stats.update(
                {
                    'min': float(arr.min().values),
                    'max': float(arr.max().values),
                    'mean': float(arr.mean().values),
                    'median': float(arr.median().values),
                    'std': float(arr.std().values),
                    'count': int(arr.count().values),  # non-null count
                }
            )

            # Add null count only if there are nulls
            null_count = int(arr.isnull().sum().values)
            if null_count > 0:
                stats['nulls'] = null_count

        except Exception:
            pass

    return stats


def drop_constant_arrays(ds: xr.Dataset, dim='time', drop_arrays_without_dim: bool = True):
    """Drop variables with very low variance (near-constant)."""
    drop_vars = []

    for name, da in ds.data_vars.items():
        if dim in da.dims:
            if da.max(dim) == da.min(dim):
                drop_vars.append(name)
            continue
        elif drop_arrays_without_dim:
            drop_vars.append(name)

    logger.debug(f'Dropping {len(drop_vars)} arrays with constant values')
    return ds.drop_vars(drop_vars)
