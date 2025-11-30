"""Type system for dimension-aware data in flixopt.

Type aliases use suffix notation to indicate maximum dimensions. Data can have any
subset of these dimensions (including scalars, which are broadcast to all dimensions).

| Suffix | Dimensions | Use Case |
|--------|------------|----------|
| `_TPS` | Time, Period, Scenario | Time-varying data across all dimensions |
| `_PS`  | Period, Scenario | Investment parameters (no time variation) |
| `_S`   | Scenario | Scenario-specific parameters |
| (none) | Scalar only | Single numeric values |

All dimensioned types accept: scalars (`int`, `float`), arrays (`ndarray`),
Series (`pd.Series`), DataFrames (`pd.DataFrame`), or DataArrays (`xr.DataArray`).

Example:
    ```python
    from flixopt.types import Numeric_TPS, Numeric_PS, Scalar


    def create_flow(
        size: Numeric_PS = None,  # Scalar, array, Series, DataFrame, or DataArray
        profile: Numeric_TPS = 1.0,  # Time-varying data
        efficiency: Scalar = 0.95,  # Scalars only
    ): ...


    # All valid:
    create_flow(size=100)  # Scalar broadcast
    create_flow(size=np.array([100, 150]))  # Period-varying
    create_flow(profile=pd.DataFrame(...))  # Time + scenario
    ```

Important:
    Data can have **any subset** of specified dimensions, but **cannot have more
    dimensions than the FlowSystem**. If the FlowSystem has only time dimension,
    you cannot pass period or scenario data. The type hints indicate the maximum
    dimensions that could be used if they exist in the FlowSystem.
"""

from typing import TypeAlias

import numpy as np
import pandas as pd
import xarray as xr

# Internal base types - not exported
_Numeric: TypeAlias = int | float | np.integer | np.floating | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray
_Bool: TypeAlias = bool | np.bool_ | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray
_Effect: TypeAlias = dict[
    str, int | float | np.integer | np.floating | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray
]

# Combined type for numeric or boolean data (no dimension information)
NumericOrBool: TypeAlias = (
    int | float | bool | np.integer | np.floating | np.bool_ | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray
)
"""Numeric or boolean data without dimension metadata. For internal utilities."""


# Numeric data types - Repeating types instead of using common var for better docs rendering
Numeric_TPS: TypeAlias = int | float | np.integer | np.floating | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray
"""Time, Period, Scenario dimensions. For time-varying data across all dimensions."""

Numeric_PS: TypeAlias = int | float | np.integer | np.floating | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray
"""Period, Scenario dimensions. For investment parameters (e.g., size, costs)."""

Numeric_S: TypeAlias = int | float | np.integer | np.floating | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray
"""Scenario dimension. For scenario-specific parameters (e.g., discount rates)."""


# Boolean data types - Repeating types instead of using common var for better docs rendering
Bool_TPS: TypeAlias = bool | np.bool_ | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray
"""Time, Period, Scenario dimensions. For time-varying binary flags/constraints."""

Bool_PS: TypeAlias = bool | np.bool_ | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray
"""Period, Scenario dimensions. For period-specific binary decisions."""

Bool_S: TypeAlias = bool | np.bool_ | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray
"""Scenario dimension. For scenario-specific binary flags."""


# Effect data types
Effect_TPS: TypeAlias = dict[
    str, int | float | np.integer | np.floating | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray
]
"""Time, Period, Scenario dimensions. Dict mapping effect names to numeric values.
For time-varying effects (costs, emissions). Use `Effect_TPS | Numeric_TPS` to accept single values."""

Effect_PS: TypeAlias = dict[
    str, int | float | np.integer | np.floating | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray
]
"""Period, Scenario dimensions. Dict mapping effect names to numeric values.
For period-specific effects (investment costs). Use `Effect_PS | Numeric_PS` to accept single values."""

Effect_S: TypeAlias = dict[
    str, int | float | np.integer | np.floating | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray
]
"""Scenario dimension. Dict mapping effect names to numeric values.
For scenario-specific effects (carbon prices). Use `Effect_S | Numeric_S` to accept single values."""


# Scalar type (no dimensions)
Scalar: TypeAlias = int | float | np.integer | np.floating
"""Scalar numeric values only. Not converted to DataArray (unlike dimensioned types)."""


# Internal types for transformed data (after fit_to_model_coords)
# These are always xr.DataArray after transformation
PeriodicData: TypeAlias = xr.DataArray
"""Periodic data (after transformation). Always an xr.DataArray with period, scenario dims."""

PeriodicEffects: TypeAlias = dict[str, xr.DataArray]
"""Periodic effects (after transformation). Dict mapping effect names to DataArrays."""


# User-facing types for effects (before transformation)
PeriodicEffectsUser: TypeAlias = dict[
    str, int | float | np.integer | np.floating | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray
]
"""User input for periodic effects. Dict mapping effect names to numeric values (any format)."""


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
    'NumericOrBool',
    'PeriodicData',
    'PeriodicEffects',
    'PeriodicEffectsUser',
]
