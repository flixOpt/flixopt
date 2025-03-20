"""
This module contains the core structure of the flixOpt framework.
These classes are not directly used by the end user, but are used by other modules.
"""

import inspect
import json
import logging
import pathlib
from datetime import datetime
from io import StringIO
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import linopy
import numpy as np
import pandas as pd
import xarray as xr
from rich.console import Console
from rich.pretty import Pretty

from .config import CONFIG
from .core import NumericData, Scalar, TimeSeries, TimeSeriesCollection, TimeSeriesData

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from .effects import EffectCollectionModel
    from .flow_system import FlowSystem

logger = logging.getLogger('flixOpt')


CLASS_REGISTRY = {}

def register_class_for_io(cls):
    """Register a class for serialization/deserialization."""
    name = cls.__name__
    if name in CLASS_REGISTRY:
        raise ValueError(f'Class {name} already registered! Use a different Name for the class! '
                         f'This error should only happen in developement')
    CLASS_REGISTRY[name] = cls
    return cls


class SystemModel(linopy.Model):

    def __init__(self, flow_system: 'FlowSystem'):
        super().__init__(force_dim_names=True)
        self.flow_system = flow_system
        self.time_series_collection = flow_system.time_series_collection
        self.effects: Optional[EffectCollectionModel] = None

    def do_modeling(self):
        self.effects = self.flow_system.effects.create_model(self)
        self.effects.do_modeling()
        component_models = [component.create_model(self) for component in self.flow_system.components.values()]
        bus_models = [bus.create_model(self) for bus in self.flow_system.buses.values()]
        for component_model in component_models:
            component_model.do_modeling()
        for bus_model in bus_models:  # Buses after Components, because FlowModels are created in ComponentModels
            bus_model.do_modeling()

    @property
    def hours_per_step(self):
        return self.time_series_collection.hours_per_timestep

    @property
    def hours_of_previous_timesteps(self):
        return self.time_series_collection.hours_of_previous_timesteps

    @property
    def coords(self) -> Tuple[pd.DatetimeIndex]:
        return (self.time_series_collection.timesteps,)

    @property
    def coords_extra(self) -> Tuple[pd.DatetimeIndex]:
        return (self.time_series_collection.timesteps_extra,)


class Interface:
    """
    This class is used to collect arguments about a Model.
    """

    def transform_data(self, flow_system: 'FlowSystem'):
        """ Transforms the data of the interface to match the FlowSystem's dimensions"""
        raise NotImplementedError('Every Interface needs a transform_data() method')

    def infos(self, use_numpy=True, use_element_label=False) -> Dict:
        """
        Generate a dictionary representation of the object's constructor arguments.
        Excludes default values and empty dictionaries and lists.
        Converts data to be compatible with JSON.

        Parameters:
        -----------
        use_numpy bool:
            Whether to convert NumPy arrays to lists. Defaults to True.
            If True, numeric numpy arrays (`np.ndarray`) are preserved as-is.
            If False, they are converted to lists.
        use_element_label bool:
            Whether to use the element label instead of the infos of the element. Defaults to False.
            Note that Elements used as keys in dictionaries are always converted to their labels.

        Returns:
            Dict: A dictionary representation of the object's constructor arguments.

        """
        # Get the constructor arguments and their default values
        init_params = sorted(
            inspect.signature(self.__init__).parameters.items(),
            key=lambda x: (x[0].lower() != 'label', x[0].lower()),  # Prioritize 'label'
        )
        # Build a dict of attribute=value pairs, excluding defaults
        details = {'class': ':'.join([cls.__name__ for cls in self.__class__.__mro__])}
        for name, param in init_params:
            if name == 'self':
                continue
            value, default = getattr(self, name, None), param.default
            # Ignore default values and empty dicts and list
            if np.all(value == default) or (isinstance(value, (dict, list)) and not value):
                continue
            details[name] = copy_and_convert_datatypes(value, use_numpy, use_element_label)
        return details

    def to_json(self, path: Union[str, pathlib.Path]):
        """
        Saves the element to a json file.
        This not meant to be reloaded and recreate the object, but rather used to document or compare the object.

        Parameters:
        -----------
        path : Union[str, pathlib.Path]
            The path to the json file.
        """
        data = get_compact_representation(self.infos(use_numpy=True, use_element_label=True))
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def to_dict(self) -> Dict:
        """Convert the object to a dictionary representation."""
        data = {'__class__': self.__class__.__name__}

        # Get the constructor parameters
        init_params = inspect.signature(self.__init__).parameters

        for name in init_params:
            if name == 'self':
                continue

            value = getattr(self, name, None)
            data[name] = self._serialize_value(value)

        return data

    def _serialize_value(self, value: Any):
        """Helper method to serialize a value based on its type."""
        if value is None:
            return None
        elif isinstance(value, Interface):
            return value.to_dict()
        elif isinstance(value, (list, tuple)):
            return self._serialize_list(value)
        elif isinstance(value, dict):
            return self._serialize_dict(value)
        else:
            return value

    def _serialize_list(self, items):
        """Serialize a list of items."""
        return [self._serialize_value(item) for item in items]

    def _serialize_dict(self, d):
        """Serialize a dictionary of items."""
        return {k: self._serialize_value(v) for k, v in d.items()}

    @classmethod
    def _deserialize_dict(cls, data: Dict) -> Union[Dict, 'Interface']:
        if '__class__' in data:
            class_name = data.pop('__class__')
            try:
                class_type = CLASS_REGISTRY[class_name]
                if issubclass(class_type, Interface):
                    # Use _deserialize_dict to process the arguments
                    processed_data = {k: cls._deserialize_value(v) for k, v in data.items()}
                    return class_type(**processed_data)
                else:
                    raise ValueError(f'Class "{class_name}" is not an Interface.')
            except (AttributeError, KeyError) as e:
                raise ValueError(f'Class "{class_name}" could not get reconstructed.') from e
        else:
            return {k: cls._deserialize_value(v) for k, v in data.items()}

    @classmethod
    def _deserialize_list(cls, data: List) -> List:
        return [cls._deserialize_value(value) for value in data]

    @classmethod
    def _deserialize_value(cls, value: Any):
        """Helper method to deserialize a value based on its type."""
        if value is None:
            return None
        elif isinstance(value, dict):
            return cls._deserialize_dict(value)
        elif isinstance(value, list):
            return cls._deserialize_list(value)
        return value

    @classmethod
    def from_dict(cls, data: Dict) -> 'Interface':
        """Create an instance from a dictionary representation."""
        return cls._deserialize_dict(data)

    def __repr__(self):
        # Get the constructor arguments and their current values
        init_signature = inspect.signature(self.__init__)
        init_args = init_signature.parameters

        # Create a dictionary with argument names and their values
        args_str = ', '.join(f'{name}={repr(getattr(self, name, None))}' for name in init_args if name != 'self')
        return f'{self.__class__.__name__}({args_str})'

    def __str__(self):
        return get_str_representation(self.infos(use_numpy=True, use_element_label=True))


class Element(Interface):
    """Basic Element of flixOpt"""

    def __init__(self, label: str, meta_data: Dict = None):
        """
        Parameters
        ----------
        label : str
            label of the element
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        """
        self.label = Element._valid_label(label)
        self.meta_data = meta_data if meta_data is not None else {}
        self.model: Optional[ElementModel] = None

    def _plausibility_checks(self) -> None:
        """This function is used to do some basic plausibility checks for each Element during initialization"""
        raise NotImplementedError('Every Element needs a _plausibility_checks() method')

    def create_model(self, model: SystemModel) -> 'ElementModel':
        raise NotImplementedError('Every Element needs a create_model() method')

    @property
    def label_full(self) -> str:
        return self.label

    @staticmethod
    def _valid_label(label: str) -> str:
        """
        Checks if the label is valid. If not, it is replaced by the default label

        Raises
        ------
        ValueError
            If the label is not valid
        """
        not_allowed = ['(', ')', '|', '->', '\\', '-slash-']  # \\ is needed to check for \
        if any([sign in label for sign in not_allowed]):
            raise ValueError(
                f'Label "{label}" is not valid. Labels cannot contain the following characters: {not_allowed}. '
                f'Use any other symbol instead'
            )
        if label.endswith(' '):
            logger.warning(f'Label "{label}" ends with a space. This will be removed.')
            return label.rstrip()
        return label


class Model:
    """Stores Variables and Constraints"""

    def __init__(self, model: SystemModel, label_of_element: str, label: Optional[str] = None, label_full: Optional[str] = None):
        """
        Parameters
        ----------
        label_of_element : str
            The label of the parent (Element). Used to construct the full label of the model.
        label : str
            The label of the model. Used to construct the full label of the model.
        label_full : str
            The full label of the model. Can overwrite the full label constructed from the other labels.
        """
        self._model = model
        self.label_of_element = label_of_element
        self._label = label
        self._label_full = label_full

        self._variables_direct: List[str] = []
        self._constraints_direct: List[str] = []
        self.sub_models: List[Model] = []

        self._variables_short: Dict[str, str] = {}
        self._constraints_short: Dict[str, str] = {}
        self._sub_models_short: Dict[str, str] = {}
        logger.debug(f'Created {self.__class__.__name__}  "{self._label}"')

    def do_modeling(self):
        raise NotImplementedError('Every Model needs a do_modeling() method')

    def add(
        self,
        item: Union[linopy.Variable, linopy.Constraint, 'Model'],
        short_name: Optional[str] = None
    ) -> Union[linopy.Variable, linopy.Constraint, 'Model']:
        """
        Add a variable, constraint or sub-model to the model

        Parameters
        ----------
        item : linopy.Variable, linopy.Constraint, InterfaceModel
            The variable, constraint or sub-model to add to the model
        short_name : str, optional
            The short name of the variable, constraint or sub-model. If not provided, the full name is used.
        """
        # TODO: Check uniquenes of short names
        if isinstance(item, linopy.Variable):
            self._variables_direct.append(item.name)
            self._variables_short[item.name] = short_name or item.name
        elif isinstance(item, linopy.Constraint):
            self._constraints_direct.append(item.name)
            self._constraints_short[item.name] = short_name or item.name
        elif isinstance(item, Model):
            self.sub_models.append(item)
            self._sub_models_short[item.label_full] = short_name or item.label_full
        else:
            raise ValueError(
                f'Item must be a linopy.Variable, linopy.Constraint or flixOpt.structure.Model, got {type(item)}')
        return item

    def filter_variables(self,
                         filter_by: Optional[Literal['binary', 'continuous', 'integer']] = None,
                         length: Literal['scalar', 'time'] = None):
        if filter_by is None:
            all_variables = self.variables
        elif filter_by == 'binary':
            all_variables = self.variables.binaries
        elif filter_by == 'integer':
            all_variables = self.variables.integers
        elif filter_by == 'continuous':
            all_variables = self.variables.continuous
        else:
            raise ValueError(f'Invalid filter_by "{filter_by}", must be one of "binary", "continous", "integer"')
        if length is None:
            return all_variables
        elif length == 'scalar':
            return all_variables[[name for name in all_variables if all_variables[name].ndim == 0]]
        elif length == 'time':
            return all_variables[[name for name in all_variables if 'time' in all_variables[name].dims]]
        raise ValueError(f'Invalid length "{length}", must be one of "scalar", "time" or None')

    @property
    def label(self) -> str:
        return self._label if self._label is not None else self.label_of_element

    @property
    def label_full(self) -> str:
        """ Used to construct the names of variables and constraints """
        if self._label_full is not None:
            return self._label_full
        elif self._label is not None:
            return f'{self.label_of_element}|{self.label}'
        return self.label_of_element

    @property
    def variables_direct(self) -> linopy.Variables:
        return self._model.variables[self._variables_direct]

    @property
    def constraints_direct(self) -> linopy.Constraints:
        return self._model.constraints[self._constraints_direct]

    @property
    def _variables(self) -> List[str]:
        all_variables = self._variables_direct.copy()
        for sub_model in self.sub_models:
            for variable in sub_model._variables:
                if variable in all_variables:
                    raise KeyError(
                        f"Duplicate key found: '{variable}' in both {self.label_full} and {sub_model.label_full}!"
                    )
                all_variables.append(variable)
        return all_variables

    @property
    def _constraints(self) -> List[str]:
        all_constraints = self._constraints_direct.copy()
        for sub_model in self.sub_models:
            for constraint in sub_model._constraints:
                if constraint in all_constraints:
                    raise KeyError(f"Duplicate key found: '{constraint}' in both main model and submodel!")
                all_constraints.append(constraint)
        return all_constraints

    @property
    def variables(self) -> linopy.Variables:
        return self._model.variables[self._variables]

    @property
    def constraints(self) -> linopy.Constraints:
        return self._model.constraints[self._constraints]

    @property
    def all_sub_models(self) -> List['Model']:
        return [model for sub_model in self.sub_models for model in [sub_model] + sub_model.all_sub_models]


class ElementModel(Model):
    """Interface to create the mathematical Variables and Constraints for Elements"""

    def __init__(self, model: SystemModel, element: Element):
        """
        Parameters
        ----------
        element : Element
            The element this model is created for.
        """
        super().__init__(model, label_of_element=element.label_full, label=element.label, label_full=element.label_full)
        self.element = element

    def results_structure(self):
        return {
            'label': self.label,
            'label_full': self.label_full,
            'variables': list(self.variables),
            'constraints': list(self.constraints),
        }


def copy_and_convert_datatypes(data: Any, use_numpy: bool = True, use_element_label: bool = False) -> Any:
    """
    Converts values in a nested data structure into JSON-compatible types while preserving or transforming numpy arrays
    and custom `Element` objects based on the specified options.

    The function handles various data types and transforms them into a consistent, readable format:
    - Primitive types (`int`, `float`, `str`, `bool`, `None`) are returned as-is.
    - Numpy scalars are converted to their corresponding Python scalar types.
    - Collections (`list`, `tuple`, `set`, `dict`) are recursively processed to ensure all elements are compatible.
    - Numpy arrays are preserved or converted to lists, depending on `use_numpy`.
    - Custom `Element` objects can be represented either by their `label` or their initialization parameters as a dictionary.
    - Timestamps (`datetime`) are converted to ISO 8601 strings.

    Parameters
    ----------
    data : Any
        The input data to process, which may be deeply nested and contain a mix of types.
    use_numpy : bool, optional
        If `True`, numeric numpy arrays (`np.ndarray`) are preserved as-is. If `False`, they are converted to lists.
        Default is `True`.
    use_element_label : bool, optional
        If `True`, `Element` objects are represented by their `label`. If `False`, they are converted into a dictionary
        based on their initialization parameters. Default is `False`.

    Returns
    -------
    Any
        A transformed version of the input data, containing only JSON-compatible types:
        - `int`, `float`, `str`, `bool`, `None`
        - `list`, `dict`
        - `np.ndarray` (if `use_numpy=True`. This is NOT JSON-compatible)

    Raises
    ------
    TypeError
        If the data cannot be converted to the specified types.

    Examples
    --------
    >>> copy_and_convert_datatypes({'a': np.array([1, 2, 3]), 'b': Element(label='example')})
    {'a': array([1, 2, 3]), 'b': {'class': 'Element', 'label': 'example'}}

    >>> copy_and_convert_datatypes({'a': np.array([1, 2, 3]), 'b': Element(label='example')}, use_numpy=False)
    {'a': [1, 2, 3], 'b': {'class': 'Element', 'label': 'example'}}

    Notes
    -----
    - The function gracefully handles unexpected types by issuing a warning and returning a deep copy of the data.
    - Empty collections (lists, dictionaries) and default parameter values in `Element` objects are omitted from the output.
    - Numpy arrays with non-numeric data types are automatically converted to lists.
    """
    if isinstance(data, np.integer):  # This must be checked before checking for regular int and float!
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)

    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    elif isinstance(data, datetime):
        return data.isoformat()

    elif isinstance(data, (tuple, set)):
        return copy_and_convert_datatypes([item for item in data], use_numpy, use_element_label)
    elif isinstance(data, dict):
        return {
            copy_and_convert_datatypes(key, use_numpy, use_element_label=True): copy_and_convert_datatypes(
                value, use_numpy, use_element_label
            )
            for key, value in data.items()
        }
    elif isinstance(data, list):  # Shorten arrays/lists to be readable
        if use_numpy and all([isinstance(value, (int, float)) for value in data]):
            return np.array([item for item in data])
        else:
            return [copy_and_convert_datatypes(item, use_numpy, use_element_label) for item in data]

    elif isinstance(data, np.ndarray):
        if not use_numpy:
            return copy_and_convert_datatypes(data.tolist(), use_numpy, use_element_label)
        elif use_numpy and np.issubdtype(data.dtype, np.number):
            return data
        else:
            logger.critical(
                f'An np.array with non-numeric content was found: {data=}.It will be converted to a list instead'
            )
            return copy_and_convert_datatypes(data.tolist(), use_numpy, use_element_label)

    elif isinstance(data, TimeSeries):
        return copy_and_convert_datatypes(data.active_data, use_numpy, use_element_label)
    elif isinstance(data, TimeSeriesData):
        return copy_and_convert_datatypes(data.data, use_numpy, use_element_label)

    elif isinstance(data, Interface):
        if use_element_label and isinstance(data, Element):
            return data.label
        return data.infos(use_numpy, use_element_label)
    elif isinstance(data, xr.DataArray):
        #TODO: This is a temporary basic work around
        return copy_and_convert_datatypes(data.values, use_numpy, use_element_label)
    else:
        raise TypeError(f'copy_and_convert_datatypes() did get unexpected data of type "{type(data)}": {data=}')


def get_compact_representation(data: Any, array_threshold: int = 50, decimals: int = 2) -> Dict:
    """
    Generate a compact json serializable representation of deeply nested data.
    Numpy arrays are statistically described if they exceed a threshold and converted to lists.

    Args:
        data (Any): The data to format and represent.
        array_threshold (int): Maximum length of NumPy arrays to display. Longer arrays are statistically described.
        decimals (int): Number of decimal places in which to describe the arrays.

    Returns:
        Dict: A dictionary representation of the data
    """

    def format_np_array_if_found(value: Any) -> Any:
        """Recursively processes the data, formatting NumPy arrays."""
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        elif isinstance(value, np.ndarray):
            return describe_numpy_arrays(value)
        elif isinstance(value, dict):
            return {format_np_array_if_found(k): format_np_array_if_found(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple, set)):
            return [format_np_array_if_found(v) for v in value]
        else:
            logger.warning(
                f'Unexpected value found when trying to format numpy array numpy array: {type(value)=}; {value=}'
            )
            return value

    def describe_numpy_arrays(arr: np.ndarray) -> Union[str, List]:
        """Shortens NumPy arrays if they exceed the specified length."""

        def normalized_center_of_mass(array: Any) -> float:
            # position in array (0 bis 1 normiert)
            positions = np.linspace(0, 1, len(array))  # weights w_i
            # mass center
            if np.sum(array) == 0:
                return np.nan
            else:
                return np.sum(positions * array) / np.sum(array)

        if arr.size > array_threshold:  # Calculate basic statistics
            fmt = f'.{decimals}f'
            return (
                f'Array (min={np.min(arr):{fmt}}, max={np.max(arr):{fmt}}, mean={np.mean(arr):{fmt}}, '
                f'median={np.median(arr):{fmt}}, std={np.std(arr):{fmt}}, len={len(arr)}, '
                f'center={normalized_center_of_mass(arr):{fmt}})'
            )
        else:
            return np.around(arr, decimals=decimals).tolist()

    # Process the data to handle NumPy arrays
    formatted_data = format_np_array_if_found(copy_and_convert_datatypes(data, use_numpy=True))

    return formatted_data


def get_str_representation(data: Any, array_threshold: int = 50, decimals: int = 2) -> str:
    """
    Generate a string representation of deeply nested data using `rich.print`.
    NumPy arrays are shortened to the specified length and converted to strings.

    Args:
        data (Any): The data to format and represent.
        array_threshold (int): Maximum length of NumPy arrays to display. Longer arrays are statistically described.
        decimals (int): Number of decimal places in which to describe the arrays.

    Returns:
        str: The formatted string representation of the data.
    """

    formatted_data = get_compact_representation(data, array_threshold, decimals)

    # Use Rich to format and print the data
    with StringIO() as output_buffer:
        console = Console(file=output_buffer, width=1000)  # Adjust width as needed
        console.print(Pretty(formatted_data, expand_all=True, indent_guides=True))
        return output_buffer.getvalue()
