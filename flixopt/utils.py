"""
This module contains several utility functions used throughout the flixopt framework.
"""

import logging
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import xarray as xr

logger = logging.getLogger('flixopt')


def is_number(number_alias: Union[int, float, str]):
    """Returns True is string is a number."""
    try:
        float(number_alias)
        return True
    except ValueError:
        return False


def round_floats(obj, decimals=2):
    if isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(v, decimals) for v in obj]
    elif isinstance(obj, float):
        return round(obj, decimals)
    return obj


def convert_dataarray(
    data: xr.DataArray, mode: Literal['py', 'numpy', 'xarray', 'structure']
) -> Union[List, np.ndarray, xr.DataArray, str]:
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
