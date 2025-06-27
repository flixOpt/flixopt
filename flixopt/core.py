"""
This module contains the core functionality of the flixopt framework.
It provides Datatypes, logging functionality, and some functions to transform data structures.
"""

import logging
from typing import Dict, Optional, Union

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
"""Internally used datatypes for temporal data."""


class PlausibilityError(Exception):
    """Error for a failing Plausibility check."""

    pass


class ConversionError(Exception):
    """Base exception for data conversion errors."""

    pass


class TimeSeriesData(xr.DataArray):
    """Minimal TimeSeriesData that inherits from xr.DataArray with aggregation metadata."""

    __slots__ = ()  # No additional instance attributes - everything goes in attrs

    def __init__(self, *args, aggregation_group: Optional[str] = None, aggregation_weight: Optional[float] = None, **kwargs):
        """
        Args:
            *args: Arguments passed to DataArray
            aggregation_group: Aggregation group name
            aggregation_weight: Aggregation weight (0-1)
            **kwargs: Additional arguments passed to DataArray
        """
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


class DataConverter:
    """
    Converts various data types into xarray.DataArray with a timesteps index.

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
            logger.warning(f'TimeSeriesData has dimensions {data.dims}, expected {dims}. Reshaping to match timesteps.')
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
    def to_dataarray(data: TemporalData, timesteps: pd.DatetimeIndex) -> xr.DataArray:
        """Convert data to xarray.DataArray with specified timesteps index."""
        if not isinstance(timesteps, pd.DatetimeIndex) or len(timesteps) == 0:
            raise ValueError(f'Timesteps must be a non-empty DatetimeIndex, got {type(timesteps).__name__}')
        if not timesteps.name == 'time':
            raise ConversionError(f'DatetimeIndex is not named correctly. Must be named "time", got {timesteps.name=}')

        coords = [timesteps]
        dims = ['time']
        expected_shape = (len(timesteps),)

        try:
            # Handle TimeSeriesData first (before generic DataArray check)
            if isinstance(data, TimeSeriesData):
                return DataConverter._fix_timeseries_data_indexing(data, timesteps, dims, coords)

            elif isinstance(data, (int, float, np.integer, np.floating)):
                # Scalar: broadcast to all timesteps
                scalar_data = np.full(expected_shape, data)
                return xr.DataArray(scalar_data, coords=coords, dims=dims)

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
                        f"DataArray length {data.sizes[dims[0]]} doesn't match expected {len(coords[0])}: {data}"
                    )
                return data.copy(deep=True)

            elif isinstance(data, list):
                logger.warning('Converting list to DataArray. This is not recommended.')
                if len(data) != expected_shape[0]:
                    raise ConversionError(f"List length {len(data)} doesn't match expected {expected_shape[0]}")
                return xr.DataArray(data, coords=coords, dims=dims)

            else:
                raise ConversionError(f'Unsupported type: {type(data).__name__}')

        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(f'Converting data {type(data)} to xarray.DataArray raised an error: {str(e)}') from e


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
