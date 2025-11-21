"""
This module contains the core functionality of the flixopt framework.
It provides Datatypes, logging functionality, and some functions to transform data structures.
"""

import logging
import warnings
from itertools import permutations
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from .config import DEPRECATION_REMOVAL_VERSION
from .types import NumericOrBool

logger = logging.getLogger('flixopt')

FlowSystemDimensions = Literal['time', 'period', 'scenario']
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
        *args: Any,
        aggregation_group: str | None = None,
        aggregation_weight: float | None = None,
        agg_group: str | None = None,
        agg_weight: float | None = None,
        **kwargs: Any,
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
            warnings.warn(
                f'agg_group is deprecated, use aggregation_group instead. '
                f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}.',
                DeprecationWarning,
                stacklevel=2,
            )
            aggregation_group = agg_group
        if agg_weight is not None:
            warnings.warn(
                f'agg_weight is deprecated, use aggregation_weight instead. '
                f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}.',
                DeprecationWarning,
                stacklevel=2,
            )
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
        coords: dict[str, pd.Index],
        name: str | None = None,
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
    def aggregation_group(self) -> str | None:
        return self.attrs.get('aggregation_group')

    @property
    def aggregation_weight(self) -> float | None:
        return self.attrs.get('aggregation_weight')

    @classmethod
    def from_dataarray(
        cls, da: xr.DataArray, aggregation_group: str | None = None, aggregation_weight: float | None = None
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
        warnings.warn(
            f'agg_group is deprecated, use aggregation_group instead. '
            f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.aggregation_group

    @property
    def agg_weight(self):
        warnings.warn(
            f'agg_weight is deprecated, use aggregation_weight instead. '
            f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.aggregation_weight


class DataConverter:
    """
    Converts various data types into xarray.DataArray with specified target coordinates.

    This converter handles intelligent dimension matching and broadcasting to ensure
    the output DataArray always conforms to the specified coordinate structure.

    Supported input types:
    - Scalars: int, float, np.number (broadcast to all target dimensions)
    - 1D data: np.ndarray, pd.Series, single-column DataFrame (matched by length/index)
    - Multi-dimensional arrays: np.ndarray, DataFrame (matched by shape)
    - xr.DataArray: validated and potentially broadcast to target dimensions

    The converter uses smart matching strategies:
    - Series: matched by exact index comparison
    - 1D arrays: matched by length to target coordinates
    - Multi-dimensional arrays: matched by shape permutation analysis
    - DataArrays: validated for compatibility and broadcast as needed
    """

    @staticmethod
    def _match_series_by_index_alignment(
        data: pd.Series, target_coords: dict[str, pd.Index], target_dims: tuple[str, ...]
    ) -> xr.DataArray:
        """
        Match pandas Series to target dimension by exact index comparison.

        Attempts to find a target dimension whose coordinates exactly match
        the Series index values, ensuring proper alignment.

        Args:
            data: pandas Series to convert
            target_coords: Available target coordinates {dim_name: coordinate_index}
            target_dims: Target dimension names to consider for matching

        Returns:
            DataArray with Series matched to the appropriate dimension

        Raises:
            ConversionError: If Series cannot be matched to any target dimension,
                           or if no target dimensions provided for multi-element Series
        """
        # Handle edge case: no target dimensions
        if len(target_dims) == 0:
            if len(data) != 1:
                raise ConversionError(
                    f'Cannot convert multi-element Series without target dimensions. '
                    f'Series has {len(data)} elements but no target dimensions specified.'
                )
            return xr.DataArray(data.iloc[0])

        # Attempt exact index matching with each target dimension
        for dim_name in target_dims:
            target_index = target_coords[dim_name]
            if data.index.equals(target_index):
                return xr.DataArray(data.values.copy(), coords={dim_name: target_index}, dims=dim_name)

        # No exact matches found
        available_lengths = {dim: len(target_coords[dim]) for dim in target_dims}
        raise ConversionError(
            f'Series index does not match any target dimension coordinates. '
            f'Series length: {len(data)}, available coordinate lengths: {available_lengths}'
        )

    @staticmethod
    def _match_1d_array_by_length(
        data: np.ndarray, target_coords: dict[str, pd.Index], target_dims: tuple[str, ...]
    ) -> xr.DataArray:
        """
        Match 1D numpy array to target dimension by length comparison.

        Finds target dimensions whose coordinate length matches the array length.
        Requires unique length match to avoid ambiguity.

        Args:
            data: 1D numpy array to convert
            target_coords: Available target coordinates {dim_name: coordinate_index}
            target_dims: Target dimension names to consider for matching

        Returns:
            DataArray with array matched to the uniquely identified dimension

        Raises:
            ConversionError: If array length matches zero or multiple target dimensions,
                           or if no target dimensions provided for multi-element array
        """
        # Handle edge case: no target dimensions
        if len(target_dims) == 0:
            if len(data) != 1:
                raise ConversionError(
                    f'Cannot convert multi-element array without target dimensions. Array has {len(data)} elements.'
                )
            return xr.DataArray(data[0])

        # Find all dimensions with matching lengths
        array_length = len(data)
        matching_dims = []
        coordinate_lengths = {}

        for dim_name in target_dims:
            coord_length = len(target_coords[dim_name])
            coordinate_lengths[dim_name] = coord_length
            if array_length == coord_length:
                matching_dims.append(dim_name)

        # Validate matching results
        if len(matching_dims) == 0:
            raise ConversionError(
                f'Array length {array_length} does not match any target dimension lengths: {coordinate_lengths}'
            )
        elif len(matching_dims) > 1:
            raise ConversionError(
                f'Array length {array_length} matches multiple dimensions: {matching_dims}. '
                f'Cannot uniquely determine target dimension. Consider using explicit '
                f'dimension specification or converting to DataArray manually.'
            )

        # Create DataArray with the uniquely matched dimension
        matched_dim = matching_dims[0]
        return xr.DataArray(data.copy(), coords={matched_dim: target_coords[matched_dim]}, dims=matched_dim)

    @staticmethod
    def _match_multidim_array_by_shape_permutation(
        data: np.ndarray, target_coords: dict[str, pd.Index], target_dims: tuple[str, ...]
    ) -> xr.DataArray:
        """
        Match multi-dimensional array to target dimensions using shape permutation analysis.

        Analyzes all possible mappings between array shape and target coordinate lengths
        to find the unique valid dimension assignment.

        Args:
            data: Multi-dimensional numpy array to convert
            target_coords: Available target coordinates {dim_name: coordinate_index}
            target_dims: Target dimension names to consider for matching

        Returns:
            DataArray with array dimensions mapped to target dimensions by shape

        Raises:
            ConversionError: If array shape cannot be uniquely mapped to target dimensions,
                           or if no target dimensions provided for multi-element array
        """
        # Handle edge case: no target dimensions
        if len(target_dims) == 0:
            if data.size != 1:
                raise ConversionError(
                    f'Cannot convert multi-element array without target dimensions. '
                    f'Array has {data.size} elements with shape {data.shape}.'
                )
            return xr.DataArray(data.item())

        array_shape = data.shape
        coordinate_lengths = {dim: len(target_coords[dim]) for dim in target_dims}

        # Find all valid dimension permutations that match the array shape
        valid_mappings = []
        for dim_permutation in permutations(target_dims, data.ndim):
            shape_matches = all(
                array_shape[i] == coordinate_lengths[dim_permutation[i]] for i in range(len(dim_permutation))
            )
            if shape_matches:
                valid_mappings.append(dim_permutation)

        # Validate mapping results
        if len(valid_mappings) == 0:
            raise ConversionError(
                f'Array shape {array_shape} cannot be mapped to any combination of target '
                f'coordinate lengths: {coordinate_lengths}. Consider reshaping the array '
                f'or adjusting target coordinates.'
            )

        if len(valid_mappings) > 1:
            raise ConversionError(
                f'Array shape {array_shape} matches multiple dimension combinations: '
                f'{valid_mappings}. Cannot uniquely determine dimension mapping. '
                f'Consider using explicit dimension specification.'
            )

        # Create DataArray with the uniquely determined mapping
        matched_dims = valid_mappings[0]
        matched_coords = {dim: target_coords[dim] for dim in matched_dims}

        return xr.DataArray(data.copy(), coords=matched_coords, dims=matched_dims)

    @staticmethod
    def _broadcast_dataarray_to_target_specification(
        source_data: xr.DataArray, target_coords: dict[str, pd.Index], target_dims: tuple[str, ...]
    ) -> xr.DataArray:
        """
        Broadcast DataArray to conform to target coordinate and dimension specification.

        Performs comprehensive validation and broadcasting to ensure the result exactly
        matches the target specification. Handles scalar expansion, dimension validation,
        coordinate compatibility checking, and broadcasting to additional dimensions.

        Args:
            source_data: Source DataArray to broadcast
            target_coords: Target coordinates {dim_name: coordinate_index}
            target_dims: Target dimension names in desired order

        Returns:
            DataArray broadcast to target specification with proper dimension ordering

        Raises:
            ConversionError: If broadcasting is impossible due to incompatible dimensions
                           or coordinate mismatches
        """
        # Validate: cannot reduce dimensions
        if len(source_data.dims) > len(target_dims):
            raise ConversionError(
                f'Cannot reduce DataArray dimensionality from {len(source_data.dims)} '
                f'to {len(target_dims)} dimensions. Source dims: {source_data.dims}, '
                f'target dims: {target_dims}'
            )

        # Validate: all source dimensions must exist in target
        missing_dims = set(source_data.dims) - set(target_dims)
        if missing_dims:
            raise ConversionError(
                f'Source DataArray has dimensions {missing_dims} not present in target dimensions {target_dims}'
            )

        # Validate: coordinate compatibility for overlapping dimensions
        for dim in source_data.dims:
            if dim in source_data.coords and dim in target_coords:
                source_coords = source_data.coords[dim]
                target_coords_for_dim = target_coords[dim]

                if not np.array_equal(source_coords.values, target_coords_for_dim.values):
                    raise ConversionError(
                        f'Coordinate mismatch for dimension "{dim}". '
                        f'Source and target coordinates have different values.'
                    )

        # Create target template for broadcasting
        target_template = xr.DataArray(coords=target_coords, dims=target_dims)

        # Perform broadcasting and ensure proper dimension ordering
        broadcasted = source_data.broadcast_like(target_template)
        return broadcasted.transpose(*target_dims)

    @classmethod
    def to_dataarray(
        cls,
        data: NumericOrBool,
        coords: dict[str, pd.Index] | None = None,
    ) -> xr.DataArray:
        """
        Convert various data types to xarray.DataArray with specified target coordinates.

        This is the main conversion method that intelligently handles different input types
        and ensures the result conforms to the specified coordinate structure through
        smart dimension matching and broadcasting.

        Args:
            data: Input data to convert. Supported types:
                - Scalars: int, float, bool, np.integer, np.floating, np.bool_
                - Arrays: np.ndarray (1D and multi-dimensional)
                - Pandas: pd.Series, pd.DataFrame
                - xarray: xr.DataArray
            coords: Target coordinate specification as {dimension_name: coordinate_index}.
                   All coordinate indices must be pandas.Index objects.

        Returns:
            DataArray conforming to the target coordinate specification,
            with input data appropriately matched and broadcast

        Raises:
            ConversionError: If data type is unsupported, conversion fails,
                           or broadcasting to target coordinates is impossible

        Examples:
            # Scalar broadcasting
            >>> coords = {'x': pd.Index([1, 2, 3]), 'y': pd.Index(['a', 'b'])}
            >>> converter.to_dataarray(42, coords)
            # Returns: DataArray with shape (3, 2), all values = 42

            # Series index matching
            >>> series = pd.Series([10, 20, 30], index=[1, 2, 3])
            >>> converter.to_dataarray(series, coords)
            # Returns: DataArray matched to 'x' dimension, broadcast to 'y'

            # Array shape matching
            >>> array = np.array([[1, 2], [3, 4], [5, 6]])  # Shape (3, 2)
            >>> converter.to_dataarray(array, coords)
            # Returns: DataArray with dimensions ('x', 'y') based on shape
        """
        # Prepare and validate target specification
        if coords is None:
            coords = {}

        validated_coords, target_dims = cls._validate_and_prepare_target_coordinates(coords)

        # Convert input data to intermediate DataArray based on type
        if isinstance(data, (int, float, bool, np.integer, np.floating, np.bool_)):
            # Scalar values - create scalar DataArray
            intermediate = xr.DataArray(data.item() if hasattr(data, 'item') else data)

        elif isinstance(data, np.ndarray):
            # NumPy arrays - dispatch based on dimensionality
            if data.ndim == 0:
                # 0-dimensional array (scalar)
                intermediate = xr.DataArray(data.item())
            elif data.ndim == 1:
                # 1-dimensional array
                intermediate = cls._match_1d_array_by_length(data, validated_coords, target_dims)
            else:
                # Multi-dimensional array
                intermediate = cls._match_multidim_array_by_shape_permutation(data, validated_coords, target_dims)

        elif isinstance(data, pd.Series):
            # Pandas Series - validate and match by index
            if isinstance(data.index, pd.MultiIndex):
                raise ConversionError('MultiIndex Series are not supported. Please use a single-level index.')
            intermediate = cls._match_series_by_index_alignment(data, validated_coords, target_dims)

        elif isinstance(data, pd.DataFrame):
            # Pandas DataFrame - validate and convert
            if isinstance(data.index, pd.MultiIndex):
                raise ConversionError('MultiIndex DataFrames are not supported. Please use a single-level index.')
            if len(data.columns) == 0 or data.empty:
                raise ConversionError('DataFrame must have at least one column and cannot be empty.')

            if len(data.columns) == 1:
                # Single-column DataFrame - treat as Series
                series_data = data.iloc[:, 0]
                intermediate = cls._match_series_by_index_alignment(series_data, validated_coords, target_dims)
            else:
                # Multi-column DataFrame - treat as multi-dimensional array
                intermediate = cls._match_multidim_array_by_shape_permutation(
                    data.to_numpy(), validated_coords, target_dims
                )

        elif isinstance(data, xr.DataArray):
            # Existing DataArray - use as-is
            intermediate = data.copy()

        else:
            # Unsupported data type
            supported_types = [
                'int',
                'float',
                'bool',
                'np.integer',
                'np.floating',
                'np.bool_',
                'np.ndarray',
                'pd.Series',
                'pd.DataFrame',
                'xr.DataArray',
            ]
            raise ConversionError(
                f'Unsupported data type: {type(data).__name__}. Supported types: {", ".join(supported_types)}'
            )

        # Broadcast intermediate result to target specification
        return cls._broadcast_dataarray_to_target_specification(intermediate, validated_coords, target_dims)

    @staticmethod
    def _validate_and_prepare_target_coordinates(
        coords: dict[str, pd.Index],
    ) -> tuple[dict[str, pd.Index], tuple[str, ...]]:
        """
        Validate and prepare target coordinate specification for DataArray creation.

        Performs comprehensive validation of coordinate inputs and prepares them
        for use in DataArray construction with appropriate naming and type checking.

        Args:
            coords: Raw coordinate specification {dimension_name: coordinate_index}

        Returns:
            Tuple of (validated_coordinates_dict, dimension_names_tuple)

        Raises:
            ConversionError: If any coordinates are invalid, improperly typed,
                           or have inconsistent naming
        """
        validated_coords = {}
        dimension_names = []

        for dim_name, coord_index in coords.items():
            # Type validation
            if not isinstance(coord_index, pd.Index):
                raise ConversionError(
                    f'Coordinate for dimension "{dim_name}" must be a pandas.Index, got {type(coord_index).__name__}'
                )

            # Non-empty validation
            if len(coord_index) == 0:
                raise ConversionError(f'Coordinate for dimension "{dim_name}" cannot be empty')

            # Ensure coordinate index has consistent naming
            if coord_index.name != dim_name:
                coord_index = coord_index.rename(dim_name)

            # Special validation for time dimensions (common pattern)
            if dim_name == 'time' and not isinstance(coord_index, pd.DatetimeIndex):
                raise ConversionError(
                    f'Dimension named "time" should use DatetimeIndex for proper '
                    f'time-series functionality, got {type(coord_index).__name__}'
                )

            validated_coords[dim_name] = coord_index
            dimension_names.append(dim_name)

        return validated_coords, tuple(dimension_names)


def get_dataarray_stats(arr: xr.DataArray) -> dict:
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


def drop_constant_arrays(ds: xr.Dataset, dim: str = 'time', drop_arrays_without_dim: bool = True) -> xr.Dataset:
    """Drop variables with constant values along a dimension.

    Args:
        ds: Input dataset to filter.
        dim: Dimension along which to check for constant values.
        drop_arrays_without_dim: If True, also drop variables that don't have the specified dimension.

    Returns:
        Dataset with constant variables removed.
    """
    drop_vars = []

    for name, da in ds.data_vars.items():
        # Skip variables without the dimension
        if dim not in da.dims:
            if drop_arrays_without_dim:
                drop_vars.append(name)
            continue

        # Check if variable is constant along the dimension
        if (da.max(dim, skipna=True) == da.min(dim, skipna=True)).all().item():
            drop_vars.append(name)

    if drop_vars:
        drop_vars = sorted(drop_vars)
        logger.debug(
            f'Dropping {len(drop_vars)} constant/dimension-less arrays: {drop_vars[:5]}{"..." if len(drop_vars) > 5 else ""}'
        )

    return ds.drop_vars(drop_vars)
