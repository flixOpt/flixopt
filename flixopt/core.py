"""
This module contains the core functionality of the flixopt framework.
It provides Datatypes, logging functionality, and some functions to transform data structures.
"""

import logging
import warnings
from typing import Dict, Literal, Optional, Tuple, Union

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

FlowSystemDimensions = Literal['time', 'scenario']
"""Possible dimensions of a FlowSystem."""


class PlausibilityError(Exception):
    """Error for a failing Plausibility check."""

    pass


class ConversionError(Exception):
    """Base exception for data conversion errors."""

    pass


class TimeSeriesData(xr.DataArray):
    """Minimal TimeSeriesData that inherits from xr.DataArray with aggregation metadata."""

    __slots__ = ()  # No additional instance attributes - everything goes in attrs

    def __init__(
            self,
            *args,
            aggregation_group: Optional[str] = None,
            aggregation_weight: Optional[float] = None,
            agg_group: Optional[str] = None,
            agg_weight: Optional[float] = None,
            **kwargs
    ):
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

    def fit_to_coords(
        self,
        coords: Dict[str, pd.Index],
        name: Optional[str] = None,
    ) -> 'TimeSeriesData':
        """Fit the data to the given coordinates. Returns a new TimeSeriesData object if the current coords are different."""
        if self.coords.equals(xr.Coordinates(coords)):
            return self

        da = DataConverter.to_dataarray(self.data, coords=coords)
        return self.__class__(
            da,
            aggregation_group=self.aggregation_group,
            aggregation_weight=self.aggregation_weight,
            name=name if name is not None else self.name
        )

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
    def _convert_series_by_index(
        data: pd.Series, coords: Dict[str, pd.Index], target_dims: Tuple[str, ...]
    ) -> xr.DataArray:
        """
        Convert pandas Series to DataArray by matching index to coordinates.

        Args:
            data: pandas Series
            coords: Available coordinates
            target_dims: Target dimension names

        Returns:
            DataArray with the Series matched to appropriate dimension by index

        Raises:
            ConversionError: If Series index doesn't match any target dimension coordinates
        """
        if len(target_dims) == 0:
            if len(data) != 1:
                raise ConversionError('Cannot convert multi-element Series without target dimensions')
            return xr.DataArray(data.iloc[0])

        # Try to match Series index to coordinates
        for dim_name in target_dims:
            if data.index.equals(coords[dim_name]):
                return xr.DataArray(data.values.copy(), coords={dim_name: coords[dim_name]}, dims=[dim_name])

        # If no index matches, raise error
        raise ConversionError(f'Series index does not match any target dimension coordinates: {target_dims}')

    @staticmethod
    def _convert_1d_array_by_length(
        data: np.ndarray, coords: Dict[str, pd.Index], target_dims: Tuple[str, ...]
    ) -> xr.DataArray:
        """
        Convert 1D numpy array to DataArray by matching length to coordinates.

        Args:
            data: 1D numpy array
            coords: Available coordinates
            target_dims: Target dimension names

        Returns:
            DataArray with the array matched to appropriate dimension by length

        Raises:
            ConversionError: If array length doesn't uniquely match a target dimension
        """
        if len(target_dims) == 0:
            if len(data) != 1:
                raise ConversionError('Cannot convert multi-element array without target dimensions')
            return xr.DataArray(data[0])

        # Match by length
        matching_dims = []
        for dim_name in target_dims:
            if len(data) == len(coords[dim_name]):
                matching_dims.append(dim_name)

        if len(matching_dims) == 0:
            dim_info = {dim: len(coords[dim]) for dim in target_dims}
            raise ConversionError(f'Array length {len(data)} matches none of the target dimensions: {dim_info}')
        elif len(matching_dims) > 1:
            raise ConversionError(
                f'Array length {len(data)} matches multiple dimensions: {matching_dims}. Cannot determine which '
                f'dimension to use. To avoid this error, convert the array to a DataArray with the correct dimensions '
                f'yourself.'
            )

        # Match to the single matching dimension
        match_dim = matching_dims[0]
        return xr.DataArray(data.copy(), coords={match_dim: coords[match_dim]}, dims=[match_dim])

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
            return DataConverter._expand_to_more_dims(data, coords, target_dims)

        # Cannot handle more dimensions than target
        if len(data.dims) > len(target_dims):
            raise ConversionError(f'Cannot reduce DataArray from {len(data.dims)} to {len(target_dims)} dimensions')

        raise ConversionError(f'Cannot convert DataArray with dims {data.dims} to target dims {target_dims}')

    @staticmethod
    def _expand_to_more_dims(
        data: xr.DataArray, coords: Dict[str, pd.Index], target_dims: Tuple[str, ...]
    ) -> xr.DataArray:
        """Expand DataArray to additional dimensions by broadcasting."""
        # Validate that all source dimensions exist in target
        for dim in data.dims:
            if dim not in target_dims:
                raise ConversionError(f'Source dimension "{dim}" not found in target dimensions {target_dims}')

            # Check coordinate compatibility
            if not np.array_equal(data.coords[dim].values, coords[dim].values):
                raise ConversionError(f'Source {dim} coordinates do not match target coordinates')

        # Start with the original data
        result_data = data.values
        result_dims = list(data.dims)
        result_coords = {dim: data.coords[dim] for dim in data.dims}

        # Add missing dimensions one by one
        for target_dim in target_dims:
            if target_dim not in result_dims:
                # Add this dimension at the end
                result_data = np.expand_dims(result_data, axis=-1)
                result_dims.append(target_dim)
                result_coords[target_dim] = coords[target_dim]

                # Broadcast along the new dimension
                new_shape = list(result_data.shape)
                new_shape[-1] = len(coords[target_dim])
                result_data = np.broadcast_to(result_data, new_shape)

        # Reorder dimensions to match target order
        if tuple(result_dims) != target_dims:
            # Create mapping from current to target order
            dim_indices = [result_dims.index(dim) for dim in target_dims]
            result_data = np.transpose(result_data, dim_indices)

        # Build final coordinates dict in target order
        final_coords = {dim: coords[dim] for dim in target_dims}

        return xr.DataArray(result_data.copy(), coords=final_coords, dims=target_dims)

    @staticmethod
    def _convert_multid_array_by_shape(
        data: np.ndarray, coords: Dict[str, pd.Index], target_dims: Tuple[str, ...]
    ) -> xr.DataArray:
        """
        Convert multi-dimensional numpy array to DataArray by matching dimensions by shape.
        Returns a DataArray that may need further broadcasting to target dimensions.

        Args:
            data: Multi-dimensional numpy array
            coords: Available coordinates
            target_dims: Target dimension names

        Returns:
            DataArray with dimensions matched by shape (may be subset of target_dims)

        Raises:
            ConversionError: If array dimensions cannot be uniquely matched to coordinates
        """
        if len(target_dims) == 0:
            if data.size != 1:
                raise ConversionError('Cannot convert multi-element array without target dimensions')
            return xr.DataArray(data.item())

        # Get lengths of each dimension
        array_shape = data.shape
        coord_lengths = {dim: len(coords[dim]) for dim in target_dims}

        # Find all possible ways to match array dimensions to available coordinates
        from itertools import permutations

        # Try all permutations of target_dims that match the array's number of dimensions
        possible_mappings = []
        for dim_subset in permutations(target_dims, data.ndim):
            # Check if this permutation matches the array shape
            if all(array_shape[i] == coord_lengths[dim_subset[i]] for i in range(len(dim_subset))):
                possible_mappings.append(dim_subset)

        if len(possible_mappings) == 0:
            shape_info = f'Array shape: {array_shape}, Coordinate lengths: {coord_lengths}'
            raise ConversionError(f'Array dimensions do not match any coordinate lengths. {shape_info}')

        if len(possible_mappings) > 1:
            raise ConversionError(
                f'Array shape {array_shape} matches multiple dimension orders: {possible_mappings}. '
                'Cannot uniquely determine dimension mapping.'
            )

        # Use the unique mapping found
        matched_dims = possible_mappings[0]
        matched_coords = {dim: coords[dim] for dim in matched_dims}

        # Return DataArray with matched dimensions - broadcasting will happen later if needed
        return xr.DataArray(data.copy(), coords=matched_coords, dims=matched_dims)

    @classmethod
    def to_dataarray(
        cls,
        data: Union[Scalar, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, TimeSeriesData],
        coords: Optional[Dict[str, pd.Index]] = None,
    ) -> xr.DataArray:
        """
        Convert data to xarray.DataArray with specified coordinates.

        Accepts:
        - Scalars (broadcast to all dimensions)
        - 1D arrays or Series (matched to one dimension, broadcast to others)
        - Multi-D arrays or DataFrames (dimensions matched by length, broadcast to remaining)
        - xr.DataArray (validated and potentially broadcast to additional dimensions)

        Args:
            data: Data to convert
            coords: Dictionary mapping dimension names to coordinate indices

        Returns:
            DataArray with the converted data
        """
        if coords is None:
            coords = {}

        validated_coords, target_dims = cls._validate_and_prepare_coords(coords)

        # Step 1: Convert to DataArray (may have fewer dimensions than target)
        if isinstance(data, (int, float, np.integer, np.floating)):
            # Scalars: create 0D DataArray, will be broadcast later
            intermediate = xr.DataArray(data.item() if hasattr(data, 'item') else data)

        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                intermediate = cls._convert_1d_array_by_length(data, validated_coords, target_dims)
            else:
                # Handle multi-dimensional arrays - this now allows partial matching
                intermediate = cls._convert_multid_array_by_shape(data, validated_coords, target_dims)

        elif isinstance(data, pd.Series):
            if isinstance(data.index, pd.MultiIndex):
                raise ConversionError(
                    'Series index must be a single level Index. Multi-index Series are not supported.'
                )
            intermediate = cls._convert_series_by_index(data, validated_coords, target_dims)

        elif isinstance(data, pd.DataFrame):
            if isinstance(data.index, pd.MultiIndex):
                raise ConversionError(
                    'DataFrame index must be a single level Index. Multi-index DataFrames are not supported.'
                )
            if len(data.columns) == 0 or data.empty:
                raise ConversionError('DataFrame must have at least one column.')

            if len(data.columns) == 1:
                intermediate = cls._convert_series_by_index(
                    data.iloc[:, 0], validated_coords, target_dims
                )
            else:
                # Handle multi-column DataFrames - this now allows partial matching
                logger.warning('Converting multi-column DataFrame to xr.DataArray. We advise to do this manually.')
                intermediate = cls._convert_multid_array_by_shape(
                    data.to_numpy(), validated_coords, target_dims
                )

        elif isinstance(data, xr.DataArray):
            intermediate = data.copy()

        else:
            raise ConversionError(
                f'Unsupported data type: {type(data).__name__}. Only scalars, arrays, Series, DataFrames, and DataArrays are supported.'
            )

        # Step 2: Broadcast to target dimensions if needed
        # This now handles cases where intermediate has some but not all target dimensions
        return cls._broadcast_to_target_dims(intermediate, validated_coords, target_dims)

    @staticmethod
    def _validate_and_prepare_coords(coords: Dict[str, pd.Index]) -> Tuple[Dict[str, pd.Index], Tuple[str, ...]]:
        """
        Validate and prepare coordinates for the DataArray.

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
