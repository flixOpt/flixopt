"""
Type system for dimension-aware data in flixopt.

This module provides generic types that clearly communicate which dimensions
data can have. The type system is designed to be self-documenting while
maintaining maximum flexibility for input formats.

Key Concepts
------------
- Dimension markers (`Time`, `Period`, `Scenario`) represent the possible dimensions
- `NumericData[...]` generic type indicates the **maximum** dimensions data can have
- Data can have any subset of the specified dimensions (including being scalar)
- All standard input formats are supported (scalar, array, Series, DataFrame, DataArray)

Examples
--------
Type hint `NumericData[Time]` accepts:
    - Scalar: `0.5` (broadcast to all timesteps)
    - 1D array: `np.array([1, 2, 3])` (matched to time dimension)
    - pandas Series: with DatetimeIndex matching flow system
    - xarray DataArray: with 'time' dimension

Type hint `NumericData[Time, Scenario]` accepts:
    - Scalar: `100` (broadcast to all time and scenario combinations)
    - 1D array: matched to time OR scenario dimension
    - 2D array: matched to both dimensions
    - pandas DataFrame: columns as scenarios, index as time
    - xarray DataArray: with any subset of 'time', 'scenario' dimensions

Type hint `NumericData[Period, Scenario]` (periodic data, no time):
    - Used for investment parameters that vary by planning period
    - Accepts scalars, arrays matching periods/scenarios, or DataArrays

Type hint `Scalar`:
    - Only numeric scalars (int, float)
    - Not converted to DataArray, stays as scalar
"""

from typing import Any, TypeAlias, Union

import numpy as np
import pandas as pd
import xarray as xr


# Dimension marker classes for generic type subscripting
class Time:
    """Marker for the time dimension in Data generic types."""

    pass


class Period:
    """Marker for the period dimension in Data generic types (for multi-period optimization)."""

    pass


class Scenario:
    """Marker for the scenario dimension in Data generic types (for scenario analysis)."""

    pass


class _NumericDataMeta(type):
    """Metaclass for NumericData to enable subscript notation NumericData[Time, Scenario] for numeric data."""

    def __getitem__(cls, dimensions):
        """
        Create a type hint showing maximum dimensions for numeric data.

        The dimensions parameter can be:
        - A single dimension: NumericData[Time]
        - Multiple dimensions: NumericData[Time, Period, Scenario]

        The type hint communicates that data can have **at most** these dimensions.
        Actual data can be:
        - Scalar (broadcast to all dimensions)
        - Have any subset of the specified dimensions
        - Have all specified dimensions

        This is consistent with xarray's broadcasting semantics and the
        framework's data conversion behavior.
        """
        # For type checking purposes, we return the same union type regardless
        # of which dimensions are specified. The dimension parameters serve
        # as documentation rather than runtime validation.

        # Return using | operator for better IDE compatibility
        return int | float | np.integer | np.floating | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray


class _BoolDataMeta(type):
    """Metaclass for BoolData to enable subscript notation BoolData[Time, Scenario] for boolean data."""

    def __getitem__(cls, dimensions):
        """
        Create a type hint showing maximum dimensions for boolean data.

        Same semantics as NumericData, but for boolean values.
        """
        # Return using | operator for better IDE compatibility
        return bool | np.bool_ | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray


class _EffectDataMeta(type):
    """Metaclass for EffectData to enable subscript notation EffectData[Time, Period, Scenario] for effect data."""

    def __getitem__(cls, dimensions):
        """
        Create a type hint showing maximum dimensions for effect data.

        Effect data is a dict with string keys mapping to numeric values
        """
        # Return using | operator for better IDE compatibility
        # EffectData = dict[str, NumericData]
        return dict[str, int | float | np.integer | np.floating | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray]


class NumericData(metaclass=_NumericDataMeta):
    """
    Base type for numeric data that can have various dimensions.

    This is the internal base class. Use `NumericData` publicly for clarity.

    Use subscript notation to specify the maximum dimensions:
    - `NumericData[Time]`: Time-varying numeric data (at most 'time' dimension)
    - `NumericData[Time, Scenario]`: Time-varying with scenarios (at most 'time', 'scenario')
    - `NumericData[Period, Scenario]`: Periodic data without time (at most 'period', 'scenario')
    - `NumericData[Time, Period, Scenario]`: Full dimensionality (rarely used)

    Semantics: "At Most" Dimensions
    --------------------------------
    When you see `NumericData[Time, Scenario]`, it means the data can have:
    - No dimensions (scalar): broadcast to all time and scenario values
    - Just 'time': broadcast across scenarios
    - Just 'scenario': broadcast across time
    - Both 'time' and 'scenario': full dimensionality

    Accepted Input Formats
    ----------------------
    All dimension combinations accept these formats:
    - Scalars: int, float (including numpy types)
    - Arrays: numpy ndarray (matched by length/shape to dimensions)
    - pandas Series: matched by index to dimension coordinates
    - pandas DataFrame: typically columns=scenarios, index=time
    - xarray DataArray: used directly with dimension validation

    Conversion Behavior
    -------------------
    Input data is converted to xarray.DataArray internally:
    - Scalars are broadcast to all specified dimensions
    - Arrays are matched by length (unambiguous) or shape (multi-dimensional)
    - Series are matched by index equality with coordinate values
    - DataArrays are validated and broadcast as needed

    Note
    ----
    This type is for **numeric** data only. For boolean data, use `BoolData`.

    This is the base class - use `NumericData` alias publicly for clarity and symmetry with `BoolData`.

    See Also
    --------
    BoolData : For boolean data with dimensions
    DataConverter.to_dataarray : The conversion implementation
    FlowSystem.fit_to_model_coords : Fits data to the model's coordinate system
    """

    # This class is not meant to be instantiated, only used for type hints
    def __init__(self):
        raise TypeError('NumericData is a type hint only and cannot be instantiated')


class BoolData(metaclass=_BoolDataMeta):
    """
    Generic type for boolean data that can have various dimensions.

    Use subscript notation to specify the maximum dimensions:
    - `BoolData[Time]`: Time-varying boolean data
    - `BoolData[Time, Scenario]`: Boolean data with time and scenario dimensions
    - `BoolData[Period, Scenario]`: Periodic boolean data

    Semantics: "At Most" Dimensions
    --------------------------------
    Same semantics as NumericData, but for boolean values.
    When you see `BoolData[Time, Scenario]`, the data can have:
    - No dimensions (scalar bool): broadcast to all time and scenario values
    - Just 'time': broadcast across scenarios
    - Just 'scenario': broadcast across time
    - Both 'time' and 'scenario': full dimensionality

    Accepted Input Formats (Boolean)
    ---------------------------------
    All dimension combinations accept these formats:
    - Scalars: bool, np.bool_
    - Arrays: numpy ndarray with boolean dtype (matched by length/shape to dimensions)
    - pandas Series: with boolean values, matched by index to dimension coordinates
    - pandas DataFrame: with boolean values
    - xarray DataArray: with boolean values, used directly with dimension validation

    Use Cases
    ---------
    Boolean data is typically used for:
    - Binary decision variables (on/off states)
    - Constraint activation flags
    - Feasibility indicators
    - Conditional parameters

    Examples
    --------
    >>> # Scalar boolean (broadcast to all dimensions)
    >>> active: BoolData[Time] = True
    >>>
    >>> # Time-varying on/off pattern
    >>> import numpy as np
    >>> pattern: BoolData[Time] = np.array([True, False, True, False])
    >>>
    >>> # Scenario-specific activation
    >>> import pandas as pd
    >>> scenario_active: BoolData[Scenario] = pd.Series([True, False, True], index=['low', 'mid', 'high'])

    Note
    ----
    This type is for **boolean** data only. For numeric data, use `NumericData`.

    See Also
    --------
    NumericData : For numeric data with dimensions
    DataConverter.to_dataarray : The conversion implementation
    """

    # This class is not meant to be instantiated, only used for type hints
    def __init__(self):
        raise TypeError('BoolData is a type hint only and cannot be instantiated')


class EffectData(metaclass=_EffectDataMeta):
    """
    Generic type for effect data that can be a single value or a dictionary of values.

    EffectData represents the common pattern for effects: either a single numeric contribution
    or a dictionary with string keys mapping to numeric contributions. This is useful for
    specifying effects where you either have a single effect or multiple named effects.

    Use subscript notation to specify the maximum dimensions:
    - `EffectData[Time]`: Time-varying effect data
    - `EffectData[Period, Scenario]`: Periodic effect data
    - `EffectData[Time, Period, Scenario]`: Full dimensional effect data

    Type Structure
    --------------
    `EffectData[dims]` = `NumericData[dims] | dict[str, NumericData[dims]]`

    This means you can provide:
    - A single numeric value (scalar, array, Series, DataFrame, DataArray)
    - A dict mapping effect names to numeric values

    Semantics: "At Most" Dimensions
    --------------------------------
    When you see `EffectData[Time, Period, Scenario]`, it means the data can have:
    - No dimensions (scalar): broadcast to all time, period, and scenario values
    - Any subset: just time, just period, just scenario, time+period, etc.
    - All dimensions: full 3D data

    Accepted Input Formats
    ----------------------
    Single value:
    - Scalars: int, float (including numpy types)
    - Arrays: numpy ndarray with numeric dtype (matched by length/shape to dimensions)
    - pandas Series: matched by index to dimension coordinates
    - pandas DataFrame: typically columns=scenarios/periods, index=time
    - xarray DataArray: used directly with dimension validation

    Dictionary of values:
    - dict[str, <any of the above formats>]

    Typical Use Cases
    -----------------
    - Single effect: `EffectData[Time] = 10.5` or `np.array([10, 12, 11])`
    - Multiple effects: `EffectData[Time] = {'CO2': 0.5, 'costs': 100}`
    - Cross-effect relationships in Effect class
    - Component effect contributions (effects_per_flow_hour, etc.)

    Examples
    --------
    >>> # Single scalar effect (broadcast to all dimensions)
    >>> single_cost: EffectData[Time, Period, Scenario] = 10.5
    >>>
    >>> # Single time-varying effect
    >>> single_emissions: EffectData[Time] = np.array([100, 120, 110])
    >>>
    >>> # Multiple named effects (dict)
    >>> multiple_effects: EffectData[Time] = {
    ...     'CO2': np.array([0.5, 0.6, 0.5]),
    ...     'costs': 100,  # scalar broadcast to all time
    ... }
    >>>
    >>> # Cross-effect in Effect class
    >>> cost_effect = Effect(
    ...     label='total_costs',
    ...     unit='€',
    ...     is_objective=True,
    ...     share_from_temporal={'CO2': 0.1},  # EffectData[Time, Period, Scenario]
    ... )

    Note
    ----
    EffectData = NumericData | dict[str, NumericData]. This pattern is specific to effects
    and different from NumericData which only represents single numeric values.

    See Also
    --------
    NumericData : General numeric data with dimensions
    BoolData : For boolean data with dimensions
    """

    # This class is not meant to be instantiated, only used for type hints
    def __init__(self):
        raise TypeError('EffectData is a type hint only and cannot be instantiated')


# Simple scalar type for dimension-less numeric values
Scalar: TypeAlias = int | float | np.integer | np.floating

# Export public API
__all__ = [
    'NumericData',  # Primary public type for numeric data
    'BoolData',  # Primary public type for boolean data
    'EffectData',  # Primary public type for effect data
    'Time',
    'Period',
    'Scenario',
    'Scalar',
]
