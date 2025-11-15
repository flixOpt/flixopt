"""
Type system for dimension-aware data in flixopt.

This module provides type aliases that clearly communicate which dimensions
data can have. The type system is designed to be self-documenting while
maintaining maximum flexibility for input formats.

Key Concepts
------------
- Type aliases use suffix notation to indicate dimensions:
  - `_T`: Time dimension only
  - `_TS`: Time and Scenario dimensions
  - `_PS`: Period and Scenario dimensions (no time)
  - `_TPS`: Time, Period, and Scenario dimensions
- Data can have any subset of the specified dimensions (including being scalar)
- All standard input formats are supported (scalar, array, Series, DataFrame, DataArray)

Examples
--------
Type hint `Numeric_T` accepts:
    - Scalar: `0.5` (broadcast to all timesteps)
    - 1D array: `np.array([1, 2, 3])` (matched to time dimension)
    - pandas Series: with DatetimeIndex matching flow system
    - xarray DataArray: with 'time' dimension

Type hint `Numeric_TS` accepts:
    - Scalar: `100` (broadcast to all time and scenario combinations)
    - 1D array: matched to time OR scenario dimension
    - 2D array: matched to both dimensions
    - pandas DataFrame: columns as scenarios, index as time
    - xarray DataArray: with any subset of 'time', 'scenario' dimensions

Type hint `Numeric_PS` (periodic data, no time):
    - Used for investment parameters that vary by planning period
    - Accepts scalars, arrays matching periods/scenarios, or DataArrays

Type hint `Scalar`:
    - Only numeric scalars (int, float)
    - Not converted to DataArray, stays as scalar
"""

from typing import TypeAlias

import numpy as np
import pandas as pd
import xarray as xr

# Internal base types
_Numeric: TypeAlias = int | float | np.integer | np.floating | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray
_Bool: TypeAlias = bool | np.bool_ | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray
_Effect: TypeAlias = _Numeric | dict[str, _Numeric]

# Numeric data with dimension combinations
Numeric_TPS: TypeAlias = _Numeric  # Time, Period, Scenario
Numeric_PS: TypeAlias = _Numeric  # Period, Scenario
Numeric_S: TypeAlias = _Numeric  # Scenario

# Boolean data with dimension combinations
Bool_TPS: TypeAlias = _Bool  # Time, Period, Scenario
Bool_PS: TypeAlias = _Bool  # Period, Scenario
Bool_S: TypeAlias = _Bool  # Scenario

# Effect data with dimension combinations
Effect_TPS: TypeAlias = _Effect  # Time, Period, Scenario
Effect_PS: TypeAlias = _Effect  # Period, Scenario
Effect_S: TypeAlias = _Effect  # Scenario

# Scalar (no dimensions)
Scalar: TypeAlias = int | float | np.integer | np.floating

# Export public API
__all__ = [
    'Numeric_TPS',
    'Numeric_PS',
    'Numeric_S',
    'Bool_TPS',
    'Bool_PS',
    'Bool_S',
    'Effect_TPS',
    'Effect_PS',
    'Effect_S',
    'Scalar',
]
