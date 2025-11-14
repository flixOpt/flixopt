"""
Type system for dimension-aware data in flixopt.

This module provides generic types that clearly communicate which dimensions
data can have. The type system is designed to be self-documenting while
maintaining maximum flexibility for input formats.

Key Concepts
------------
- Dimension markers (`Time`, `Period`, `Scenario`) represent the possible dimensions
- `Data[...]` generic type indicates the **maximum** dimensions data can have
- Data can have any subset of the specified dimensions (including being scalar)
- All standard input formats are supported (scalar, array, Series, DataFrame, DataArray)

Examples
--------
Type hint `Data[Time]` accepts:
    - Scalar: `0.5` (broadcast to all timesteps)
    - 1D array: `np.array([1, 2, 3])` (matched to time dimension)
    - pandas Series: with DatetimeIndex matching flow system
    - xarray DataArray: with 'time' dimension

Type hint `Data[Time, Scenario]` accepts:
    - Scalar: `100` (broadcast to all time and scenario combinations)
    - 1D array: matched to time OR scenario dimension
    - 2D array: matched to both dimensions
    - pandas DataFrame: columns as scenarios, index as time
    - xarray DataArray: with any subset of 'time', 'scenario' dimensions

Type hint `Data[Period, Scenario]` (periodic data, no time):
    - Used for investment parameters that vary by planning period
    - Accepts scalars, arrays matching periods/scenarios, or DataArrays

Type hint `Scalar`:
    - Only numeric scalars (int, float)
    - Not converted to DataArray, stays as scalar
"""

from typing import Any, TypeAlias

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
    """Metaclass for Data to enable subscript notation Data[Time, Scenario] for numeric data."""

    def __getitem__(cls, dimensions):
        """
        Create a type hint showing maximum dimensions for numeric data.

        The dimensions parameter can be:
        - A single dimension: Data[Time]
        - Multiple dimensions: Data[Time, Scenario]

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

        # Return type that includes all possible numeric input formats
        return int | float | np.integer | np.floating | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray


class _BoolDataMeta(type):
    """Metaclass for BoolData to enable subscript notation BoolData[Time, Scenario] for boolean data."""

    def __getitem__(cls, dimensions):
        """
        Create a type hint showing maximum dimensions for boolean data.

        Same semantics as numeric Data, but for boolean values.
        """
        # Return type that includes all possible boolean input formats
        return bool | np.bool_ | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray


class Data(metaclass=_NumericDataMeta):
    """
    Generic type for data that can have various dimensions.

    Use subscript notation to specify the maximum dimensions:
    - `Data[Time]`: Time-varying data (at most 'time' dimension)
    - `Data[Time, Scenario]`: Time-varying with scenarios (at most 'time', 'scenario')
    - `Data[Period, Scenario]`: Periodic data without time (at most 'period', 'scenario')
    - `Data[Time, Period, Scenario]`: Full dimensionality (rarely used)

    Semantics: "At Most" Dimensions
    --------------------------------
    When you see `Data[Time, Scenario]`, it means the data can have:
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

    See Also
    --------
    BoolData : For boolean data with dimensions
    DataConverter.to_dataarray : The conversion implementation
    FlowSystem.fit_to_model_coords : Fits data to the model's coordinate system
    """

    # This class is not meant to be instantiated, only used for type hints
    def __init__(self):
        raise TypeError('Data is a type hint only and cannot be instantiated')


class BoolData(metaclass=_BoolDataMeta):
    """
    Generic type for boolean data that can have various dimensions.

    Use subscript notation to specify the maximum dimensions:
    - `BoolData[Time]`: Time-varying boolean data
    - `BoolData[Time, Scenario]`: Boolean data with time and scenario dimensions
    - `BoolData[Period, Scenario]`: Periodic boolean data

    Semantics: "At Most" Dimensions
    --------------------------------
    Same semantics as Data, but for boolean values.
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
    This type is for **boolean** data only. For numeric data, use `Data`.

    See Also
    --------
    Data : For numeric data with dimensions
    DataConverter.to_dataarray : The conversion implementation
    """

    # This class is not meant to be instantiated, only used for type hints
    def __init__(self):
        raise TypeError('BoolData is a type hint only and cannot be instantiated')


# Simple scalar type for dimension-less numeric values
Scalar: TypeAlias = int | float | np.integer | np.floating

# Export public API
__all__ = [
    'Data',
    'BoolData',
    'Time',
    'Period',
    'Scenario',
    'Scalar',
]
