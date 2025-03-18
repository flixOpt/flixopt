"""
This module contains the core functionality of the flixOpt framework.
It provides Datatypes, logging functionality, and some functions to transform data structures.
"""

import inspect
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from . import utils

logger = logging.getLogger('flixOpt')

Skalar = Union[int, float]  # Datatype
Numeric = Union[int, float, np.ndarray]  # Datatype


class TimeSeriesData:
    # TODO: Move to Interface.py
    def __init__(self, data: Numeric, agg_group: Optional[str] = None, agg_weight: Optional[float] = None):
        """
        timeseries class for transmit timeseries AND special characteristics of timeseries,
        i.g. to define weights needed in calculation_type 'aggregated'
            EXAMPLE solar:
            you have several solar timeseries. These should not be overweighted
            compared to the remaining timeseries (i.g. heat load, price)!
            fixed_relative_profile_solar1 = TimeSeriesData(sol_array_1, type = 'solar')
            fixed_relative_profile_solar2 = TimeSeriesData(sol_array_2, type = 'solar')
            fixed_relative_profile_solar3 = TimeSeriesData(sol_array_3, type = 'solar')
            --> this 3 series of same type share one weight, i.e. internally assigned each weight = 1/3
            (instead of standard weight = 1)

        Parameters
        ----------
        data : Union[int, float, np.ndarray]
            The timeseries data, which can be a scalar, array, or numpy array.
        agg_group : str, optional
            The group this TimeSeriesData is a part of. agg_weight is split between members of a group. Default is None.
        agg_weight : float, optional
            The weight for calculation_type 'aggregated', should be between 0 and 1. Default is None.

        Raises
        ------
        Exception
            If both agg_group and agg_weight are set, an exception is raised.
        """
        self.data = data
        self.agg_group = agg_group
        self.agg_weight = agg_weight
        if (agg_group is not None) and (agg_weight is not None):
            raise Exception('Either <agg_group> or explicit <agg_weigth> can be used. Not both!')
        self.label: Optional[str] = None

    def __repr__(self):
        # Get the constructor arguments and their current values
        init_signature = inspect.signature(self.__init__)
        init_args = init_signature.parameters

        # Create a dictionary with argument names and their values
        args_str = ', '.join(f'{name}={repr(getattr(self, name, None))}' for name in init_args if name != 'self')
        return f'{self.__class__.__name__}({args_str})'

    def __str__(self):
        return str(self.data)


Numeric_TS = Union[
    Skalar, np.ndarray, TimeSeriesData
]  # TODO: This is not really correct throughozt the codebase. Sometimes its used for TimeSeries aswell?


class TimeSeries:
    """
    Class for data that applies to time series, stored as vector (np.ndarray) or scalar.

    This class represents a vector or scalar value that makes the handling of time series easier.
    It supports various operations such as activation of specific time indices, setting explicit active data, and
    aggregation weight management.

    Attributes
    ----------
    label : str
        The label for the time series.
    data : Optional[Numeric]
        The actual data for the time series. Can be None.
    aggregated_data : Optional[Numeric]
        aggregated_data to use instead of data if provided.
    active_indices : Optional[np.ndarray]
        Indices of the time steps to activate.
    aggregation_weight : float
        Weight for aggregation method, between 0 and 1, normally 1.
    aggregation_group : str
        Group for calculating the aggregation weigth for aggregation method.
    """

    def __init__(self, label: str, data: Optional[Numeric_TS]):
        self.label: str = label
        if isinstance(data, TimeSeriesData):
            self.data = self.make_scalar_if_possible(data.data)
            self.aggregation_weight, self.aggregation_group = data.agg_weight, data.agg_group
            data.label = self.label  # Connecting User_time_series to real Time_series
        else:
            self.data = self.make_scalar_if_possible(data)
            self.aggregation_weight, self.aggregation_group = None, None

        self.active_indices: Optional[Union[range, List[int]]] = None
        self.aggregated_data: Optional[Numeric] = None

    def activate_indices(self, indices: Optional[Union[range, List[int]]], aggregated_data: Optional[Numeric] = None):
        self.active_indices = indices

        if aggregated_data is not None:
            assert len(aggregated_data) == len(self.active_indices) or len(aggregated_data) == 1, (
                f'The aggregated_data has the wrong length for TimeSeries {self.label}. '
                f'Length should be: {len(self.active_indices)} or 1, but is {len(aggregated_data)}'
            )
            self.aggregated_data = self.make_scalar_if_possible(aggregated_data)

    def clear_indices_and_aggregated_data(self):
        self.active_indices = None
        self.aggregated_data = None

    @property
    def active_data(self) -> Numeric:
        if self.aggregated_data is not None:  # Aggregated data is always active, if present
            return self.aggregated_data

        indices_not_applicable = np.isscalar(self.data) or (self.data is None) or (self.active_indices is None)
        if indices_not_applicable:
            return self.data
        else:
            return self.data[self.active_indices]

    @property
    def active_data_vector(self) -> np.ndarray:
        # Always returns the active data as a vector.
        return utils.as_vector(self.active_data, len(self.active_indices))

    @property
    def is_scalar(self) -> bool:
        return np.isscalar(self.data)

    @property
    def is_array(self) -> bool:
        return not self.is_scalar and self.data is not None

    def __repr__(self):
        # Retrieve all attributes and their values
        attrs = vars(self)
        # Format each attribute as 'key=value'
        attrs_str = ', '.join(f'{key}={value!r}' for key, value in attrs.items())
        # Format the output as 'ClassName(attr1=value1, attr2=value2, ...)'
        return f'{self.__class__.__name__}({attrs_str})'

    def __str__(self):
        return str(self.active_data)

    @staticmethod
    def make_scalar_if_possible(data: Optional[Numeric]) -> Optional[Numeric]:
        """
        Convert an array to a scalar if all values are equal, or return the array as-is.
        Can Return None if the passed data is None

        Parameters
        ----------
        data : Numeric, None
            The data to process.

        Returns
        -------
        Numeric
            A scalar if all values in the array are equal, otherwise the array itself. None, if the passed value is None
        """
        # TODO: Should this really return None Values?
        if np.isscalar(data) or data is None:
            return data
        data = np.array(data)
        if np.all(data == data[0]):
            return data[0]
        return data
