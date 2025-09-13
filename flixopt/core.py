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

NonTemporalData = xr.DataArray
"""Internally used datatypes for non-temporal data. Can be a Scalar or an xr.DataArray."""

FlowSystemDimensions = Literal['time', 'year', 'scenario']
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
        **kwargs,
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
            name=name if name is not None else self.name,
        )

    @property
    def aggregation_group(self) -> Optional[str]:
        return self.attrs.get('aggregation_group')

    @property
    def aggregation_weight(self) -> Optional[float]:
        return self.attrs.get('aggregation_weight')

    @classmethod
    def from_dataarray(
        cls, da: xr.DataArray, aggregation_group: Optional[str] = None, aggregation_weight: Optional[float] = None
    ):
        """Create TimeSeriesData from DataArray, extracting metadata from attrs."""
        # Get aggregation metadata from attrs or parameters
        final_aggregation_group = (
            aggregation_group if aggregation_group is not None else da.attrs.get('aggregation_group')
        )
        final_aggregation_weight = (
            aggregation_weight if aggregation_weight is not None else da.attrs.get('aggregation_weight')
        )

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
    - Multi-dimensional arrays
    - xr.DataArray (validated and potentially broadcast)

    Simple 1D data is matched to one dimension and broadcast to others.
    DataArrays can have any number of dimensions.
    """

    @staticmethod
    def _match_series_to_dimension(
        data: pd.Series, coords: Dict[str, pd.Index], target_dims: Tuple[str, ...]
    ) -> xr.DataArray:
        """
        Match pandas Series to a dimension by comparing its index to coordinates.

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
                raise ConversionError(
                    f'Cannot convert multi-element Series without target dimensions. Got \n{data}\n and \n{coords}'
                )
            return xr.DataArray(data.iloc[0])

        # Try to match Series index to coordinates
        for dim_name in target_dims:
            if data.index.equals(coords[dim_name]):
                return xr.DataArray(data.values.copy(), coords={dim_name: coords[dim_name]}, dims=dim_name)

        # If no index matches, raise error
        raise ConversionError(f'Series index does not match any target dimension coordinates: {target_dims}')

    @staticmethod
    def _match_array_to_dimension(
        data: np.ndarray, coords: Dict[str, pd.Index], target_dims: Tuple[str, ...]
    ) -> xr.DataArray:
        """
        Match 1D numpy array to a dimension by comparing its length to coordinate lengths.

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

        # Find dimensions with matching lengths
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
        return xr.DataArray(data.copy(), coords={match_dim: coords[match_dim]}, dims=match_dim)

    @staticmethod
    def _match_multidim_array_to_dimensions(
        data: np.ndarray, coords: Dict[str, pd.Index], target_dims: Tuple[str, ...]
    ) -> xr.DataArray:
        """
        Match multi-dimensional numpy array to dimensions by finding the correct shape permutation.

        Args:
            data: Multi-dimensional numpy array
            coords: Available coordinates
            target_dims: Target dimension names

        Returns:
            DataArray with dimensions matched by shape

        Raises:
            ConversionError: If array dimensions cannot be uniquely matched to coordinates
        """
        if len(target_dims) == 0:
            if data.size != 1:
                raise ConversionError('Cannot convert multi-element array without target dimensions')
            return xr.DataArray(data.item())

        from itertools import permutations

        array_shape = data.shape
        coord_lengths = {dim: len(coords[dim]) for dim in target_dims}

        # Find all possible dimension mappings
        possible_mappings = []
        for dim_subset in permutations(target_dims, data.ndim):
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

        matched_dims = possible_mappings[0]
        matched_coords = {dim: coords[dim] for dim in matched_dims}

        return xr.DataArray(data.copy(), coords=matched_coords, dims=matched_dims)

    @staticmethod
    def _broadcast_to_target(
        data: xr.DataArray, coords: Dict[str, pd.Index], target_dims: Tuple[str, ...]
    ) -> xr.DataArray:
        """
        Broadcast DataArray to target dimensions with validation.

        Handles all cases: scalar expansion, dimension validation, coordinate matching,
        and broadcasting to additional dimensions using xarray's capabilities.
        """
        # Cannot reduce dimensions of data
        if len(data.dims) > len(target_dims):
            raise ConversionError(f'Cannot reduce DataArray from {len(data.dims)} to {len(target_dims)} dimensions')

        # Validate coordinate compatibility
        for dim in data.dims:
            if dim not in target_dims:
                raise ConversionError(f'Source dimension "{dim}" not found in target dimensions {target_dims}')

            if not np.array_equal(data.coords[dim].values, coords[dim].values):
                raise ConversionError(f'DataArray {dim} coordinates do not match target coordinates')

        # Use xarray's broadcast_like for efficient expansion and broadcasting
        target_template = xr.DataArray(
            np.empty([len(coords[dim]) for dim in target_dims]), coords=coords, dims=target_dims
        )
        return data.broadcast_like(target_template).transpose(*target_dims)

    @classmethod
    def to_dataarray(
        cls,
        data: Union[float, int, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray],
        coords: Optional[Dict[str, pd.Index]] = None,
    ) -> xr.DataArray:
        """
        Convert various data types to xarray.DataArray with specified coordinates.

        Args:
            data: Data to convert (scalar, array, Series, DataFrame, or DataArray)
            coords: Dictionary mapping dimension names to coordinate indices

        Returns:
            DataArray with the converted data broadcast to target dimensions

        Raises:
            ConversionError: If data cannot be converted or dimensions are ambiguous
        """
        if coords is None:
            coords = {}

        validated_coords, target_dims = cls._prepare_coordinates(coords)

        # Step 1: Convert input data to initial DataArray
        if isinstance(data, (int, float, np.integer, np.floating)):
            # Scalar values
            intermediate = xr.DataArray(data.item() if hasattr(data, 'item') else data)

        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                intermediate = cls._match_array_to_dimension(data, validated_coords, target_dims)
            else:
                intermediate = cls._match_multidim_array_to_dimensions(data, validated_coords, target_dims)

        elif isinstance(data, pd.Series):
            if isinstance(data.index, pd.MultiIndex):
                raise ConversionError(
                    'Series index must be a single level Index. Multi-index Series are not supported.'
                )
            intermediate = cls._match_series_to_dimension(data, validated_coords, target_dims)

        elif isinstance(data, pd.DataFrame):
            if isinstance(data.index, pd.MultiIndex):
                raise ConversionError(
                    'DataFrame index must be a single level Index. Multi-index DataFrames are not supported.'
                )
            if len(data.columns) == 0 or data.empty:
                raise ConversionError('DataFrame must have at least one column.')

            if len(data.columns) == 1:
                # Single-column DataFrame - treat as Series
                intermediate = cls._match_series_to_dimension(data.iloc[:, 0], validated_coords, target_dims)
            else:
                # Multi-column DataFrame - treat as multi-dimensional array
                intermediate = cls._match_multidim_array_to_dimensions(data.to_numpy(), validated_coords, target_dims)

        elif isinstance(data, xr.DataArray):
            intermediate = data.copy()

        else:
            raise ConversionError(f'Unsupported data type: {type(data).__name__}.')

        # Step 2: Broadcast to target dimensions
        return cls._broadcast_to_target(intermediate, validated_coords, target_dims)

    @staticmethod
    def _prepare_coordinates(coords: Dict[str, pd.Index]) -> Tuple[Dict[str, pd.Index], Tuple[str, ...]]:
        """
        Validate coordinates and prepare them for DataArray creation.

        Args:
            coords: Dictionary mapping dimension names to coordinate indices

        Returns:
            Tuple of (validated coordinates dict, dimensions tuple)

        Raises:
            ConversionError: If coordinates are invalid
        """
        validated_coords = {}
        dims = []

        for dim_name, coord_index in coords.items():
            # Basic validation
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
