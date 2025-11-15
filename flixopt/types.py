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
        size: Numeric_PS = None,      # Scalar, array, Series, DataFrame, or DataArray
        profile: Numeric_TPS = 1.0,   # Time-varying data
        efficiency: Scalar = 0.95,    # Scalars only
    ):
        ...

    # All valid:
    create_flow(size=100)                          # Scalar broadcast
    create_flow(size=np.array([100, 150]))         # Period-varying
    create_flow(profile=pd.DataFrame(...))         # Time + scenario
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
_Effect: TypeAlias = _Numeric | dict[str, _Numeric]


# Numeric data types
Numeric_TPS: TypeAlias = _Numeric
"""Time, Period, Scenario dimensions. For time-varying data across all dimensions."""

Numeric_PS: TypeAlias = _Numeric
"""Period, Scenario dimensions. For investment parameters (e.g., size, costs)."""

Numeric_S: TypeAlias = _Numeric
"""Scenario dimension. For scenario-specific parameters (e.g., discount rates)."""


# Boolean data types
Bool_TPS: TypeAlias = _Bool
"""Time, Period, Scenario dimensions. For time-varying binary flags/constraints."""

Bool_PS: TypeAlias = _Bool
"""Period, Scenario dimensions. For period-specific binary decisions."""

Bool_S: TypeAlias = _Bool
"""Scenario dimension. For scenario-specific binary flags."""


# Effect data types
Effect_TPS: TypeAlias = _Effect
"""Time, Period, Scenario dimensions. For time-varying effects (costs, emissions).
Can be single numeric value or dict mapping effect names to values."""

Effect_PS: TypeAlias = _Effect
"""Period, Scenario dimensions. For period-specific effects (investment costs).
Can be single numeric value or dict mapping effect names to values."""

Effect_S: TypeAlias = _Effect
"""Scenario dimension. For scenario-specific effects (carbon prices).
Can be single numeric value or dict mapping effect names to values."""


# Scalar type (no dimensions)
Scalar: TypeAlias = int | float | np.integer | np.floating
"""Scalar numeric values only. Not converted to DataArray (unlike dimensioned types)."""

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
