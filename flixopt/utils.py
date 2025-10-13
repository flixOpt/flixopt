"""
This module contains several utility functions used throughout the flixopt framework.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import xarray as xr

logger = logging.getLogger('flixopt')


def round_nested_floats(obj: dict | list | float | int | Any, decimals: int = 2) -> dict | list | float | int | Any:
    """Recursively round floating point numbers in nested data structures.

    This function traverses nested data structures (dictionaries, lists) and rounds
    any floating point numbers to the specified number of decimal places. It handles
    various data types including NumPy arrays and xarray DataArrays by converting
    them to lists with rounded values.

    Args:
        obj: The object to process. Can be a dict, list, float, int, numpy.ndarray,
            xarray.DataArray, or any other type.
        decimals (int, optional): Number of decimal places to round to. Defaults to 2.

    Returns:
        The processed object with the same structure as the input, but with all floating point numbers rounded to the specified precision. NumPy arrays and xarray DataArrays are converted to lists.

    Examples:
        >>> data = {'a': 3.14159, 'b': [1.234, 2.678]}
        >>> round_nested_floats(data, decimals=2)
        {'a': 3.14, 'b': [1.23, 2.68]}

        >>> import numpy as np
        >>> arr = np.array([1.234, 5.678])
        >>> round_nested_floats(arr, decimals=1)
        [1.2, 5.7]
    """
    if isinstance(obj, dict):
        return {k: round_nested_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_nested_floats(v, decimals) for v in obj]
    elif isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, int):
        return obj
    elif isinstance(obj, np.ndarray):
        return np.round(obj, decimals).tolist()
    elif isinstance(obj, xr.DataArray):
        return obj.round(decimals).values.tolist()
    return obj


def convert_dataarray(
    data: xr.DataArray, mode: Literal['py', 'numpy', 'xarray', 'structure']
) -> list | np.ndarray | xr.DataArray | str:
    """
    Convert a DataArray to a different format.

    Args:
        data: The DataArray to convert.
        mode: The mode to convert to.
            - 'py': Convert to python native types (for json)
            - 'numpy': Convert to numpy array
            - 'xarray': Convert to xarray.DataArray
            - 'structure': Convert to strings (for structure, storing variable names)

    Returns:
        The converted data.

    Raises:
        ValueError: If the mode is unknown.
    """
    if mode == 'numpy':
        return data.values
    elif mode == 'py':
        return data.values.tolist()
    elif mode == 'xarray':
        return data
    elif mode == 'structure':
        return f':::{data.name}'
    else:
        raise ValueError(f'Unknown mode {mode}')
