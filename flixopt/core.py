"""
This module contains the core functionality of the flixopt framework.
It provides Datatypes, logging functionality, and some functions to transform data structures.
"""

import inspect
import json
import logging
import pathlib
from collections import Counter
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger('flixopt')

Scalar = Union[int, float]
"""A type representing a single number, either integer or float."""

NumericData = Union[int, float, np.integer, np.floating, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray]
"""Represents any form of numeric data, from simple scalars to complex data structures."""

NumericDataTS = Union[NumericData, 'TimeSeriesData']
"""Represents either standard numeric data or TimeSeriesData."""


class PlausibilityError(Exception):
    """Error for a failing Plausibility check."""

    pass


class ConversionError(Exception):
    """Base exception for data conversion errors."""

    pass


class DataConverter:
    """
    Converts various data types into xarray.DataArray with a timesteps index.

    Supports: scalars, arrays, Series, DataFrames, and DataArrays.
    """

    @staticmethod
    def to_dataarray(data: NumericData, timesteps: pd.DatetimeIndex) -> xr.DataArray:
        """Convert data to xarray.DataArray with specified timesteps index."""
        if not isinstance(timesteps, pd.DatetimeIndex) or len(timesteps) == 0:
            raise ValueError(f'Timesteps must be a non-empty DatetimeIndex, got {type(timesteps).__name__}')
        if not timesteps.name == 'time':
            raise ConversionError(f'DatetimeIndex is not named correctly. Must be named "time", got {timesteps.name=}')

        coords = [timesteps]
        dims = ['time']
        expected_shape = (len(timesteps),)

        try:
            if isinstance(data, (int, float, np.integer, np.floating)):
                return xr.DataArray(data, coords=coords, dims=dims)
            elif isinstance(data, pd.DataFrame):
                if not data.index.equals(timesteps):
                    raise ConversionError("DataFrame index doesn't match timesteps index")
                if not len(data.columns) == 1:
                    raise ConversionError('DataFrame must have exactly one column')
                return xr.DataArray(data.values.flatten(), coords=coords, dims=dims)
            elif isinstance(data, pd.Series):
                if not data.index.equals(timesteps):
                    raise ConversionError("Series index doesn't match timesteps index")
                return xr.DataArray(data.values, coords=coords, dims=dims)
            elif isinstance(data, np.ndarray):
                if data.ndim != 1:
                    raise ConversionError(f'Array must be 1-dimensional, got {data.ndim}')
                elif data.shape[0] != expected_shape[0]:
                    raise ConversionError(f"Array shape {data.shape} doesn't match expected {expected_shape}")
                return xr.DataArray(data, coords=coords, dims=dims)
            elif isinstance(data, xr.DataArray):
                if data.dims != tuple(dims):
                    raise ConversionError(f"DataArray dimensions {data.dims} don't match expected {dims}")
                if data.sizes[dims[0]] != len(coords[0]):
                    raise ConversionError(
                        f"DataArray length {data.sizes[dims[0]]} doesn't match expected {len(coords[0])}"
                    )
                return data.copy(deep=True)
            elif isinstance(data, list):
                logger.warning(f'Converting list to DataArray. This is not reccomended.')
                return xr.DataArray(data, coords=coords, dims=dims)
            else:
                raise ConversionError(f'Unsupported type: {type(data).__name__}')
        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(f'Converting data {type(data)} to xarray.Dataset raised an error: {str(e)}') from e


class TimeSeries:
    def __init__(self):
        raise NotImplementedError('TimeSeries was removed')


class TimeSeriesCollection:
    """
    Collection of TimeSeries objects with shared timestep management.

    TimeSeriesCollection handles multiple TimeSeries objects with synchronized
    timesteps, provides operations on collections, and manages extra timesteps.
    """

    def __init__(self):
        raise NotImplementedError('TimeSeriesCollection was removed')


def get_numeric_stats(data: xr.DataArray, decimals: int = 2, padd: int = 10) -> str:
    """Calculates the mean, median, min, max, and standard deviation of a numeric DataArray."""
    format_spec = f'>{padd}.{decimals}f' if padd else f'.{decimals}f'
    if np.unique(data).size == 1:
        return f'{data.max().item():{format_spec}} (constant)'
    mean = data.mean().item()
    median = data.median().item()
    min_val = data.min().item()
    max_val = data.max().item()
    std = data.std().item()
    return f'{mean:{format_spec}} (mean), {median:{format_spec}} (median), {min_val:{format_spec}} (min), {max_val:{format_spec}} (max), {std:{format_spec}} (std)'
