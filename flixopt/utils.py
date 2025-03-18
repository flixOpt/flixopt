"""
This module contains several utility functions used throughout the flixopt framework.
"""

import logging
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np

logger = logging.getLogger('flixopt')


def as_vector(value: Union[int, float, np.ndarray, List], length: int) -> np.ndarray:
    """
    Macht aus Skalar einen Vektor. Vektor bleibt Vektor.
    -> Idee dahinter: Aufruf aus abgespeichertem Vektor schneller, als für jede i-te Gleichung zu Checken ob Vektor oder Skalar)

    Parameters
    ----------

    aValue: skalar, list, np.array
    aLen  : skalar
    """
    # dtype = 'float64' # -> muss mit übergeben werden, sonst entstehen evtl. int32 Reihen (dort ist kein +/-inf möglich)
    # TODO: as_vector() -> int32 Vektoren möglich machen

    # Wenn Skalar oder None, return directly as array
    if value is None:
        return np.array([None] * length)
    if np.isscalar(value):
        return np.ones(length) * value

    if len(value) != length:  # Wenn Vektor nicht richtige Länge
        raise Exception(f'error in changing to {length=}; vector has already {len(value)=}')

    if isinstance(value, np.ndarray):
        return value
    else:
        return np.array(value)


def is_number(number_alias: Union[int, float, str]):
    """Returns True is string is a number."""
    try:
        float(number_alias)
        return True
    except ValueError:
        return False


def check_time_series(label: str, time_series: np.ndarray[np.datetime64]):
    # check sowohl für globale Zeitreihe, als auch für chosenIndexe:

    # Zeitdifferenz:
    #              zweites bis Letztes            - erstes bis Vorletztes
    dt = time_series[1:] - time_series[0:-1]
    # dt_in_hours    = dt.total_seconds() / 3600
    dt_in_hours = dt / np.timedelta64(1, 'h')

    # unterschiedliche dt:
    if np.max(dt_in_hours) - np.min(dt_in_hours) != 0:
        logger.warning(f'{label}: !! Achtung !! unterschiedliche delta_t von {min(dt)} h bis  {max(dt)} h')
    # negative dt:
    if np.min(dt_in_hours) < 0:
        raise Exception(label + ': Zeitreihe besitzt Zurücksprünge - vermutlich Zeitumstellung nicht beseitigt!')


def apply_formating(
    data_dict: Dict[str, Union[int, float]],
    key_format: str = '<17',
    value_format: str = '>10.2f',
    indent: int = 0,
    sort_by: Optional[Literal['key', 'value']] = None,
) -> str:
    if sort_by == 'key':
        sorted_keys = sorted(data_dict.keys(), key=str.lower)
    elif sort_by == 'value':
        sorted_keys = sorted(data_dict, key=lambda k: data_dict[k], reverse=True)
    else:
        sorted_keys = data_dict.keys()

    lines = [f'{indent * " "}{key:{key_format}}: {data_dict[key]:{value_format}}' for key in sorted_keys]
    return '\n'.join(lines)


def label_is_valid(label: str) -> bool:
    """Function to make sure '__' is reserved for internal splitting of labels"""
    if label.startswith('_') or label.endswith('_') or '__' in label:
        return False
    return True


def convert_numeric_lists_to_arrays(
    d: Union[Dict[str, Any], List[Any], tuple],
) -> Union[Dict[str, Any], List[Any], tuple]:
    """
    Recursively converts all lists of numeric values in a nested dictionary to numpy arrays.
    Handles nested lists, tuples, and dictionaries. Does not alter the original dictionary.
    """

    def convert_list_to_array_if_numeric(sequence: Union[List[Any], tuple]) -> Union[np.ndarray, List[Any]]:
        """
        Converts a Sequence to a numpy array if all elements are numeric.
        Recursively processes each element.
        Does not alter the original sequence.
        Returns an empty list if the sequence is empty.
        """
        # Check if the list is empty
        if len(sequence) == 0:
            return []
        # Check if all elements are numeric in the list
        elif isinstance(sequence, list) and all(isinstance(item, (int, float)) for item in sequence):
            return np.array(sequence)
        else:
            return [
                convert_numeric_lists_to_arrays(item) if isinstance(item, (dict, list, tuple)) else item
                for item in sequence
            ]

    if isinstance(d, dict):
        d_copy = {}  # Reconstruct the dict from ground up to not modify the original dictionary.'
        for key, value in d.items():
            if isinstance(value, (list, tuple)):
                d_copy[key] = convert_list_to_array_if_numeric(value)
            elif isinstance(value, dict):
                d_copy[key] = convert_numeric_lists_to_arrays(value)  # Recursively process nested dictionaries
            else:
                d_copy[key] = value
        return d_copy
    elif isinstance(d, (list, tuple)):
        # If the input itself is a list or tuple, process it as a sequence
        return convert_list_to_array_if_numeric(d)
    else:
        return d
