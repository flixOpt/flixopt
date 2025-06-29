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
    Converts scalars and 1D data into xarray.DataArray with optional time and scenario dimensions.

    Only handles:
    - Scalars (int, float, np.number)
    - 1D arrays (np.ndarray, pd.Series)
    - xr.DataArray (for broadcasting/checking)
    """

    @staticmethod
    def to_dataarray(
        data: Union[Scalar, np.ndarray, pd.Series, xr.DataArray, TimeSeriesData],
        timesteps: Optional[pd.DatetimeIndex] = None,
        scenarios: Optional[pd.Index] = None,
    ) -> xr.DataArray:
        """
        Convert data to xarray.DataArray with specified dimensions.

        Args:
            data: Scalar, 1D array/Series, or existing DataArray
            timesteps: Optional DatetimeIndex for time dimension
            scenarios: Optional Index for scenario dimension

        Returns:
            DataArray with the converted data
        """
        coords, dims = DataConverter._prepare_dimensions(timesteps, scenarios)

        # Handle scalars
        if isinstance(data, (int, float, np.integer, np.floating)):
            return DataConverter._convert_scalar(data, coords, dims)

        # Handle 1D numpy arrays
        elif isinstance(data, np.ndarray):
            if data.ndim != 1:
                raise ConversionError(f'Only 1D arrays supported, got {data.ndim}D array')
            return DataConverter._convert_1d_array(data, coords, dims)

        # Handle pandas Series
        elif isinstance(data, pd.Series):
            return DataConverter._convert_series(data, coords, dims)

        # Handle existing DataArrays (including TimeSeriesData)
        elif isinstance(data, xr.DataArray):
            return DataConverter._handle_dataarray(data, coords, dims)

        else:
            raise ConversionError(
                f'Unsupported data type: {type(data).__name__}. Only scalars, 1D arrays, Series, and DataArrays are supported.'
            )

    @staticmethod
    def _prepare_dimensions(
        timesteps: Optional[pd.DatetimeIndex], scenarios: Optional[pd.Index]
    ) -> Tuple[Dict[str, pd.Index], Tuple[str, ...]]:
        """Prepare coordinates and dimensions."""
        coords = {}
        dims = []

        if timesteps is not None:
            if not isinstance(timesteps, pd.DatetimeIndex) or len(timesteps) == 0:
                raise ConversionError('Timesteps must be a non-empty DatetimeIndex')
            if timesteps.name != 'time':
                timesteps = timesteps.rename('time')
            coords['time'] = timesteps
            dims.append('time')

        if scenarios is not None:
            if not isinstance(scenarios, pd.Index) or len(scenarios) == 0:
                raise ConversionError('Scenarios must be a non-empty Index')
            if scenarios.name != 'scenario':
                scenarios = scenarios.rename('scenario')
            coords['scenario'] = scenarios
            dims.append('scenario')

        return coords, tuple(dims)

    @staticmethod
    def _convert_scalar(
        data: Union[int, float, np.integer, np.floating], coords: Dict[str, pd.Index], dims: Tuple[str, ...]
    ) -> xr.DataArray:
        """Convert scalar to DataArray, broadcasting to all dimensions."""
        if isinstance(data, (np.integer, np.floating)):
            data = data.item()
        return xr.DataArray(data, coords=coords, dims=dims)

    @staticmethod
    def _convert_1d_array(data: np.ndarray, coords: Dict[str, pd.Index], dims: Tuple[str, ...]) -> xr.DataArray:
        """Convert 1D array to DataArray."""
        if len(dims) == 0:
            if len(data) != 1:
                raise ConversionError('Cannot convert multi-element array without dimensions')
            return xr.DataArray(data[0])

        elif len(dims) == 1:
            dim_name = dims[0]
            if len(data) != len(coords[dim_name]):
                raise ConversionError(
                    f'Array length {len(data)} does not match {dim_name} length {len(coords[dim_name])}'
                )
            return xr.DataArray(data, coords=coords, dims=dims)

        elif len(dims) == 2:
            # Broadcast 1D array to 2D based on which dimension it matches
            time_len = len(coords['time'])
            scenario_len = len(coords['scenario'])

            if len(data) == time_len:
                # Broadcast across scenarios
                values = np.repeat(data[:, np.newaxis], scenario_len, axis=1)
                return xr.DataArray(values, coords=coords, dims=dims)
            elif len(data) == scenario_len:
                # Broadcast across time
                values = np.repeat(data[np.newaxis, :], time_len, axis=0)
                return xr.DataArray(values, coords=coords, dims=dims)
            else:
                raise ConversionError(
                    f'Array length {len(data)} matches neither time ({time_len}) nor scenario ({scenario_len}) dimensions'
                )

        else:
            raise ConversionError('Maximum 2 dimensions supported')

    @staticmethod
    def _convert_series(data: pd.Series, coords: Dict[str, pd.Index], dims: Tuple[str, ...]) -> xr.DataArray:
        """Convert pandas Series to DataArray."""
        if len(dims) == 0:
            if len(data) != 1:
                raise ConversionError('Cannot convert multi-element Series without dimensions')
            return xr.DataArray(data.iloc[0])

        elif len(dims) == 1:
            dim_name = dims[0]
            if not data.index.equals(coords[dim_name]):
                raise ConversionError(f'Series index does not match {dim_name} coordinates')
            return xr.DataArray(data.values, coords=coords, dims=dims)

        elif len(dims) == 2:
            # Check which dimension the Series index matches
            if data.index.equals(coords['time']):
                # Broadcast across scenarios
                values = np.repeat(data.values[:, np.newaxis], len(coords['scenario']), axis=1)
                return xr.DataArray(values, coords=coords, dims=dims)
            elif data.index.equals(coords['scenario']):
                # Broadcast across time
                values = np.repeat(data.values[np.newaxis, :], len(coords['time']), axis=0)
                return xr.DataArray(values, coords=coords, dims=dims)
            else:
                raise ConversionError('Series index must match either time or scenario coordinates')

        else:
            raise ConversionError('Maximum 2 dimensions supported')

    @staticmethod
    def _handle_dataarray(data: xr.DataArray, coords: Dict[str, pd.Index], dims: Tuple[str, ...]) -> xr.DataArray:
        """Handle existing DataArray - check compatibility or broadcast."""
        # If no target dimensions, data must be scalar
        if len(dims) == 0:
            if data.size != 1:
                raise ConversionError('DataArray must be scalar when no dimensions specified')
            return xr.DataArray(data.values.item())

        # Check if already compatible
        if data.dims == dims:
            # Check if coordinates match
            compatible = True
            for dim in dims:
                if not np.array_equal(data.coords[dim].values, coords[dim].values):
                    compatible = False
                    break
            if compatible:
                return data.copy()

        # Handle broadcasting from smaller to larger dimensions
        if len(data.dims) < len(dims):
            return DataConverter._broadcast_dataarray(data, coords, dims)

        # If dimensions don't match and can't broadcast, raise error
        raise ConversionError(f'Cannot convert DataArray with dims {data.dims} to target dims {dims}')

    @staticmethod
    def _broadcast_dataarray(data: xr.DataArray, coords: Dict[str, pd.Index], dims: Tuple[str, ...]) -> xr.DataArray:
        """Broadcast DataArray to target dimensions."""
        if len(data.dims) == 0:
            # Scalar DataArray - broadcast to all dimensions
            return xr.DataArray(data.values.item(), coords=coords, dims=dims)

        elif len(data.dims) == 1 and len(dims) == 2:
            source_dim = data.dims[0]

            # Check coordinate compatibility
            if not np.array_equal(data.coords[source_dim].values, coords[source_dim].values):
                raise ConversionError(f'Source {source_dim} coordinates do not match target coordinates')

            if source_dim == 'time':
                # Broadcast time to include scenarios
                values = np.repeat(data.values[:, np.newaxis], len(coords['scenario']), axis=1)
                return xr.DataArray(values, coords=coords, dims=dims)
            elif source_dim == 'scenario':
                # Broadcast scenario to include time
                values = np.repeat(data.values[np.newaxis, :], len(coords['time']), axis=0)
                return xr.DataArray(values, coords=coords, dims=dims)

        raise ConversionError(f'Cannot broadcast from {data.dims} to {dims}')


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
