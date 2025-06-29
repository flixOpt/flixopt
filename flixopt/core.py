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
    Converts data into xarray.DataArray with specified coordinates.

    Supports:
    - Scalars (broadcast to all dimensions)
    - 1D data (np.ndarray, pd.Series, single-column DataFrame)
    - xr.DataArray (validated and potentially broadcast)

    Simple 1D data is matched to one dimension and broadcast to others.
    DataArrays can have any number of dimensions.
    """

    @staticmethod
    def _convert_1d_data_to_dataarray(
        data: Union[np.ndarray, pd.Series], coords: Dict[str, pd.Index], target_dims: Tuple[str, ...]
    ) -> xr.DataArray:
        """
        Convert 1D data (array or Series) to DataArray by matching to one dimension.

        Args:
            data: 1D numpy array or pandas Series
            coords: Available coordinates
            target_dims: Target dimension names

        Returns:
            DataArray with the data matched to appropriate dimension
        """
        if len(target_dims) == 0:
            # No target dimensions - data must be single element
            if len(data) != 1:
                raise ConversionError('Cannot convert multi-element data without target dimensions')
            return xr.DataArray(data[0] if isinstance(data, np.ndarray) else data.iloc[0])

        # For Series, try to match index to coordinates
        if isinstance(data, pd.Series):
            for dim_name in target_dims:
                if data.index.equals(coords[dim_name]):
                    return xr.DataArray(data.values.copy(), coords={dim_name: coords[dim_name]}, dims=[dim_name])

            # If no index match, fall through to length matching

        # For arrays or unmatched Series, match by length
        matching_dims = []
        for dim_name in target_dims:
            if len(data) == len(coords[dim_name]):
                matching_dims.append(dim_name)

        if len(matching_dims) == 0:
            dim_info = {dim: len(coords[dim]) for dim in target_dims}
            raise ConversionError(f'Data length {len(data)} matches none of the target dimensions: {dim_info}')
        elif len(matching_dims) > 1:
            raise ConversionError(
                f'Data length {len(data)} matches multiple dimensions: {matching_dims}. Cannot determine which dimension to use.'
            )

        # Match to the single matching dimension
        match_dim = matching_dims[0]
        values = data.values.copy() if isinstance(data, pd.Series) else data.copy()
        return xr.DataArray(values, coords={match_dim: coords[match_dim]}, dims=[match_dim])

    @staticmethod
    def _broadcast_to_target_dims(
        data: xr.DataArray, coords: Dict[str, pd.Index], target_dims: Tuple[str, ...]
    ) -> xr.DataArray:
        """
        Broadcast DataArray to match target dimensions.

        Args:
            data: Source DataArray
            coords: Target coordinates
            target_dims: Target dimension names

        Returns:
            DataArray broadcast to target dimensions
        """
        if len(target_dims) == 0:
            # Target is scalar
            if data.size != 1:
                raise ConversionError('Cannot convert multi-element DataArray to scalar')
            return xr.DataArray(data.values.item())

        # If data already matches target, validate coordinates and return
        if set(data.dims) == set(target_dims) and len(data.dims) == len(target_dims):
            # Check coordinate compatibility
            for dim in data.dims:
                if dim in coords and not np.array_equal(data.coords[dim].values, coords[dim].values):
                    raise ConversionError(f'DataArray {dim} coordinates do not match target coordinates')

            # Ensure correct dimension order
            if data.dims != target_dims:
                data = data.transpose(*target_dims)
            return data.copy()

        # Handle scalar data (0D) - broadcast to all dimensions
        if data.ndim == 0:
            return xr.DataArray(data.item(), coords=coords, dims=target_dims)

        # Handle broadcasting from fewer to more dimensions
        if len(data.dims) < len(target_dims):
            return DataConverter._broadcast_dataarray_to_more_dims(data, coords, target_dims)

        # Cannot handle more dimensions than target
        if len(data.dims) > len(target_dims):
            raise ConversionError(f'Cannot reduce DataArray from {len(data.dims)} to {len(target_dims)} dimensions')

        raise ConversionError(f'Cannot convert DataArray with dims {data.dims} to target dims {target_dims}')

    @staticmethod
    def _broadcast_dataarray_to_more_dims(
        data: xr.DataArray, coords: Dict[str, pd.Index], target_dims: Tuple[str, ...]
    ) -> xr.DataArray:
        """Broadcast DataArray to additional dimensions."""
        # Validate that all source dimensions exist in target
        for dim in data.dims:
            if dim not in target_dims:
                raise ConversionError(f'Source dimension "{dim}" not found in target dimensions {target_dims}')

            # Check coordinate compatibility
            if not np.array_equal(data.coords[dim].values, coords[dim].values):
                raise ConversionError(f'Source {dim} coordinates do not match target coordinates')

        # Build the full coordinate system
        full_coords = {}
        for dim in target_dims:
            full_coords[dim] = coords[dim]

        # Use xarray's broadcast_to functionality
        # Create a template DataArray with target structure
        template_data = np.broadcast_to(data.values, [len(coords[dim]) for dim in target_dims])

        # Create mapping for broadcasting
        # We need to insert new axes for missing dimensions
        expanded_data = data.values
        for i, dim in enumerate(target_dims):
            if dim not in data.dims:
                # Add new axis for this dimension
                expanded_data = np.expand_dims(expanded_data, axis=i)

        # Now broadcast to full shape
        target_shape = tuple(len(coords[dim]) for dim in target_dims)
        broadcasted_data = np.broadcast_to(expanded_data, target_shape)

        return xr.DataArray(broadcasted_data.copy(), coords=full_coords, dims=target_dims)

    @staticmethod
    def to_dataarray(
        data: Union[Scalar, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, TimeSeriesData],
        coords: Optional[Dict[str, pd.Index]] = None,
    ) -> xr.DataArray:
        """
        Convert data to xarray.DataArray with specified coordinates.

        Accepts:
        - Scalars (broadcast to all dimensions)
        - 1D arrays, Series, or single-column DataFrames (matched to one dimension, broadcast to others)
        - xr.DataArray (validated and potentially broadcast to additional dimensions)

        Args:
            data: Data to convert
            coords: Dictionary mapping dimension names to coordinate indices

        Returns:
            DataArray with the converted data
        """
        if coords is None:
            coords = {}

        validated_coords, target_dims = DataConverter._prepare_dimensions(coords)

        # Step 1: Convert to DataArray (with safe 1D/2D logic for simple data)
        if isinstance(data, (int, float, np.integer, np.floating)):
            # Scalars: create 0D DataArray, will be broadcast later
            intermediate = xr.DataArray(data.item() if hasattr(data, 'item') else data)

        elif isinstance(data, np.ndarray):
            if data.ndim != 1:
                raise ConversionError(f'Only 1D arrays supported, got {data.ndim}D array')
            intermediate = DataConverter._convert_1d_data_to_dataarray(data, validated_coords, target_dims)

        elif isinstance(data, pd.Series):
            intermediate = DataConverter._convert_1d_data_to_dataarray(data, validated_coords, target_dims)

        elif isinstance(data, pd.DataFrame):
            if len(data.columns) != 1:
                raise ConversionError(f'Only single-column DataFrames are supported, got {len(data.columns)} columns')
            series = data.iloc[:, 0]
            intermediate = DataConverter._convert_1d_data_to_dataarray(series, validated_coords, target_dims)

        elif isinstance(data, xr.DataArray):
            intermediate = data.copy()

        else:
            raise ConversionError(
                f'Unsupported data type: {type(data).__name__}. Only scalars, 1D arrays, Series, single-column DataFrames, and DataArrays are supported.'
            )

        # Step 2: Broadcast to target dimensions if needed
        return DataConverter._broadcast_to_target_dims(intermediate, validated_coords, target_dims)

    @staticmethod
    def _prepare_dimensions(coords: Dict[str, pd.Index]) -> Tuple[Dict[str, pd.Index], Tuple[str, ...]]:
        """
        Prepare and validate coordinates for the DataArray.

        Args:
            coords: Dictionary mapping dimension names to coordinate indices

        Returns:
            Tuple of (validated coordinates dict, dimensions tuple)
        """
        validated_coords = {}
        dims = []

        for dim_name, coord_index in coords.items():
            # Validate coordinate index
            if not isinstance(coord_index, pd.Index) or len(coord_index) == 0:
                raise ConversionError(f'{dim_name} coordinates must be a non-empty pandas Index')

            # Ensure coordinate index has the correct name
            if coord_index.name != dim_name:
                coord_index = coord_index.rename(dim_name)

            # Special validation for time dimension
            if dim_name == 'time' and not isinstance(coord_index, pd.DatetimeIndex):
                raise ConversionError('time coordinates must be a DatetimeIndex')

            validated_coords[dim_name] = coord_index
            dims.append(dim_name)

        return validated_coords, tuple(dims)

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
            return xr.DataArray(data.copy(), coords=coords, dims=dims)

        elif len(dims) == 2:
            # Broadcast 1D array to 2D based on which dimension it matches
            dim_lengths = {dim: len(coords[dim]) for dim in dims}

            # Find which dimension the array length matches
            matching_dims = [dim for dim, length in dim_lengths.items() if len(data) == length]

            if len(matching_dims) == 0:
                raise ConversionError(f'Array length {len(data)} matches none of the dimensions: {dim_lengths}')
            elif len(matching_dims) > 1:
                raise ConversionError(
                    f'Array length {len(data)} matches multiple dimensions: {matching_dims}. Cannot determine broadcasting direction.'
                )

            # Broadcast along the matching dimension
            match_dim = matching_dims[0]
            other_dim = [d for d in dims if d != match_dim][0]

            if dims.index(match_dim) == 0:  # First dimension
                values = np.repeat(data[:, np.newaxis], len(coords[other_dim]), axis=1)
            else:  # Second dimension
                values = np.repeat(data[np.newaxis, :], len(coords[other_dim]), axis=0)

            return xr.DataArray(values.copy(), coords=coords, dims=dims)

        else:
            raise ConversionError(f'Maximum 2 dimensions currently supported, got {len(dims)}')

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
            return xr.DataArray(data.values.copy(), coords=coords, dims=dims)

        elif len(dims) == 2:
            # Check which dimension the Series index matches
            if 'time' in coords and data.index.equals(coords['time']):
                # Broadcast across other dimensions
                other_dims = [d for d in dims if d != 'time']
                if len(other_dims) == 1:
                    other_dim = other_dims[0]
                    values = np.repeat(data.values[:, np.newaxis], len(coords[other_dim]), axis=1)
                    return xr.DataArray(values.copy(), coords=coords, dims=dims)

            elif len([d for d in dims if d != 'time']) == 1:
                # Check if Series matches the non-time dimension
                other_dim = [d for d in dims if d != 'time'][0]
                if data.index.equals(coords[other_dim]):
                    # Broadcast across time
                    values = np.repeat(data.values[np.newaxis, :], len(coords['time']), axis=0)
                    return xr.DataArray(values.copy(), coords=coords, dims=dims)

            raise ConversionError(f'Series index must match one of the target dimensions: {list(coords.keys())}')

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

            # Check if source dimension exists in target
            if source_dim not in coords:
                raise ConversionError(f'Source dimension "{source_dim}" not found in target coordinates')

            # Check coordinate compatibility
            if not np.array_equal(data.coords[source_dim].values, coords[source_dim].values):
                raise ConversionError(f'Source {source_dim} coordinates do not match target coordinates')

            # Find the other dimension to broadcast to
            other_dim = [d for d in dims if d != source_dim][0]

            # Broadcast based on dimension order
            if dims.index(source_dim) == 0:  # Source is first dimension
                values = np.repeat(data.values[:, np.newaxis], len(coords[other_dim]), axis=1)
            else:  # Source is second dimension
                values = np.repeat(data.values[np.newaxis, :], len(coords[other_dim]), axis=0)

            return xr.DataArray(values.copy(), coords=coords, dims=dims)

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
