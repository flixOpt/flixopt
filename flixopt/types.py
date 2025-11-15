"""Type system for dimension-aware data in flixopt.

This module provides type aliases that clearly communicate which dimensions data can
have, making function signatures self-documenting while maintaining maximum flexibility
for input formats.

The type system uses suffix notation to indicate maximum dimensions:
    - ``_TPS``: Time, Period, and Scenario dimensions
    - ``_PS``: Period and Scenario dimensions (no time)
    - ``_S``: Scenario dimension only
    - No suffix: Scalar values only

All dimensioned types accept any subset of their specified dimensions, including scalars
which are automatically broadcast to all dimensions.

Supported Input Formats:
    - Scalars: ``int``, ``float`` (including numpy types)
    - Arrays: ``numpy.ndarray`` (matched by length/shape to dimensions)
    - Series: ``pandas.Series`` (matched by index to dimension coordinates)
    - DataFrames: ``pandas.DataFrame`` (typically columns=scenarios, index=time)
    - DataArrays: ``xarray.DataArray`` (used directly with dimension validation)

Example:
    Basic usage with different dimension combinations::
        ```python
        from flixopt.types import Numeric_TPS, Numeric_PS, Scalar

        def create_flow(
            label: str,
            size: Numeric_PS = None,  # Can be scalar, array, Series, etc.
            profile: Numeric_TPS = 1.0,  # Accepts time-varying data
            efficiency: Scalar = 0.95,  # Only scalars
        ):
            ...

        # All of these are valid:
        create_flow("heat", size=100)  # Scalar broadcast
        create_flow("heat", size=np.array([100, 150]))  # Period-varying
        create_flow("heat", profile=pd.DataFrame(...))  # Time + scenario
        ```

Note:
    Data can have **any subset** of the specified dimensions. For example,
    ``Numeric_TPS`` can accept:
        - Scalar: ``0.5`` (broadcast to all time, period, scenario combinations)
        - 1D array: matched to one dimension
        - 2D array: matched to two dimensions
        - 3D array: matched to all three dimensions
        - ``xarray.DataArray``: with any subset of 'time', 'period', 'scenario' dims

See Also:
    DataConverter.to_dataarray: Implementation of data conversion logic
    FlowSystem.fit_to_model_coords: Fits data to model coordinate system
"""

from typing import TypeAlias

import numpy as np
import pandas as pd
import xarray as xr

# Internal base types - not exported
_Numeric: TypeAlias = int | float | np.integer | np.floating | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray
"""Base numeric type union accepting scalars, arrays, Series, DataFrames, and DataArrays."""

_Bool: TypeAlias = bool | np.bool_ | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray
"""Base boolean type union accepting bool scalars, arrays, Series, DataFrames, and DataArrays."""

_Effect: TypeAlias = _Numeric | dict[str, _Numeric]
"""Base effect type union accepting numeric data or dict of numeric data for named effects."""


# Numeric data type aliases with dimension combinations
Numeric_TPS: TypeAlias = _Numeric
"""Numeric data with at most Time, Period, and Scenario dimensions.

Use this for data that can vary across time steps, planning periods, and scenarios.
Accepts any subset of these dimensions including scalars (broadcast to all dimensions).

Example:
    ::

        efficiency: Numeric_TPS = 0.95  # Scalar broadcast
        efficiency: Numeric_TPS = np.array([...])  # Time-varying
        efficiency: Numeric_TPS = pd.DataFrame(...)  # Time + scenarios
"""

Numeric_PS: TypeAlias = _Numeric
"""Numeric data with at most Period and Scenario dimensions (no time variation).

Use this for investment parameters that vary by planning period and scenario but not
within each period (e.g., investment costs, capacity sizes).

Example:
    ::

        size: Numeric_PS = 100  # Scalar
        size: Numeric_PS = np.array([100, 150, 200])  # Period-varying
        size: Numeric_PS = pd.DataFrame(...)  # Period + scenario combinations
"""

Numeric_S: TypeAlias = _Numeric
"""Numeric data with at most Scenario dimension.

Use this for scenario-specific parameters that don't vary over time or periods.

Example:
    ::

        discount_rate: Numeric_S = 0.05  # Same for all scenarios
        discount_rate: Numeric_S = pd.Series([0.03, 0.05, 0.07])  # Scenario-varying
"""


# Boolean data type aliases with dimension combinations
Bool_TPS: TypeAlias = _Bool
"""Boolean data with at most Time, Period, and Scenario dimensions.

Use this for binary flags or activation states that can vary across time, periods,
and scenarios (e.g., on/off constraints, feasibility indicators).

Example:
    ::

        is_active: Bool_TPS = True  # Always active
        is_active: Bool_TPS = np.array([True, False, True, ...])  # Time-varying
"""

Bool_PS: TypeAlias = _Bool
"""Boolean data with at most Period and Scenario dimensions.

Use this for binary investment decisions or constraints that vary by period and
scenario but not within each period.

Example:
    ::

        can_invest: Bool_PS = True  # Can invest in all periods
        can_invest: Bool_PS = np.array([False, True, True])  # Period-specific
"""

Bool_S: TypeAlias = _Bool
"""Boolean data with at most Scenario dimension.

Use this for scenario-specific binary flags.

Example:
    ::

        high_demand: Bool_S = False  # Same for all scenarios
        high_demand: Bool_S = pd.Series([False, True, True])  # Scenario-varying
"""


# Effect data type aliases with dimension combinations
Effect_TPS: TypeAlias = _Effect
"""Effect data with at most Time, Period, and Scenario dimensions.

Effects represent costs, emissions, or other impacts. Can be a single numeric value
or a dict mapping effect names to numeric values for multiple named effects.

Example:
    ::

        # Single effect
        cost: Effect_TPS = 10.5
        cost: Effect_TPS = np.array([10, 12, 11, ...])

        # Multiple named effects
        effects: Effect_TPS = {
            'CO2': 0.5,
            'costs': np.array([100, 120, 110, ...]),
        }
"""

Effect_PS: TypeAlias = _Effect
"""Effect data with at most Period and Scenario dimensions.

Use this for period-specific effects like investment costs or periodic emissions.

Example:
    ::

        investment_cost: Effect_PS = 1000  # Fixed cost
        investment_cost: Effect_PS = {'capex': 1000, 'opex': 50}  # Multiple effects
"""

Effect_S: TypeAlias = _Effect
"""Effect data with at most Scenario dimension.

Use this for scenario-specific effects.

Example:
    ::

        carbon_price: Effect_S = 50  # Same for all scenarios
        carbon_price: Effect_S = pd.Series([30, 50, 70])  # Scenario-varying
"""


# Scalar type (no dimensions)
Scalar: TypeAlias = int | float | np.integer | np.floating
"""Scalar numeric values only (no arrays or DataArrays).

Use this when you specifically want to accept only scalar values, not arrays.
Unlike dimensioned types, scalars are not converted to DataArrays internally.

Example:
    ::

        efficiency: Scalar = 0.95  # OK
        efficiency: Scalar = np.array([0.95])  # Type error - array not allowed
"""

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
