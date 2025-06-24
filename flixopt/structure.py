"""
This module contains the core structure of the flixopt framework.
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
from .core import NumericData, Scalar, TimeSeriesCollection, TimeSeries

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from .effects import EffectCollectionModel
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


CLASS_REGISTRY = {}


def register_class_for_io(cls):
    """Register a class for serialization/deserialization."""
    name = cls.__name__
    if name in CLASS_REGISTRY:
        raise ValueError(
            f'Class {name} already registered! Use a different Name for the class! '
            f'This error should only happen in developement'
        )
    CLASS_REGISTRY[name] = cls
    return cls


class SystemModel(linopy.Model):
    """
    The SystemModel is the linopy Model that is used to create the mathematical model of the flow_system.
    It is used to create and store the variables and constraints for the flow_system.
    """

    def __init__(self, flow_system: 'FlowSystem'):
        """
        Args:
            flow_system: The flow_system that is used to create the model.
        """
        super().__init__(force_dim_names=True)
        self.flow_system = flow_system
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
    def solution(self):
        solution = super().solution
        solution.attrs = {
            'Components': {
                comp.label_full: comp.model.results_structure()
                for comp in sorted(
                    self.flow_system.components.values(), key=lambda component: component.label_full.upper()
                )
            },
            'Buses': {
                bus.label_full: bus.model.results_structure()
                for bus in sorted(self.flow_system.buses.values(), key=lambda bus: bus.label_full.upper())
            },
            'Effects': {
                effect.label_full: effect.model.results_structure()
                for effect in sorted(self.flow_system.effects, key=lambda effect: effect.label_full.upper())
            },
        }
        return solution.reindex(time=self.flow_system.timesteps_extra)

    @property
    def hours_per_step(self):
        return self.flow_system.hours_per_timestep

    @property
    def hours_of_previous_timesteps(self):
        return self.flow_system.hours_of_previous_timesteps

    @property
    def coords(self) -> Tuple[pd.DatetimeIndex]:
        return (self.flow_system.timesteps,)

    @property
    def coords_extra(self) -> Tuple[pd.DatetimeIndex]:
        return (self.flow_system.timesteps_extra,)


class Interface:
    """
    This class is used to collect arguments about a Model. Its the base class for all Elements and Models in flixopt.
    """

    def transform_data(self, flow_system: 'FlowSystem'):
        """Transforms the data of the interface to match the FlowSystem's dimensions"""
        raise NotImplementedError('Every Interface needs a transform_data() method')

    def _create_reference_structure(self) -> Tuple[Dict, Dict[str, xr.DataArray]]:
        """
        Convert all DataArrays/TimeSeries to references and extract them.
        This is the core method that both to_dict() and to_dataset() build upon.

        Returns:
            Tuple of (reference_structure, extracted_arrays_dict)
        """
        # Get constructor parameters
        init_params = inspect.signature(self.__init__).parameters

        # Process all constructor parameters
        reference_structure = {'__class__': self.__class__.__name__}
        all_extracted_arrays = {}

        for name in init_params:
            if name == 'self':
                continue

            value = getattr(self, name, None)
            if value is None:
                continue

            # Extract arrays and get reference structure
            processed_value, extracted_arrays = self._extract_dataarrays_recursive(value)

            # Add extracted arrays to the collection
            all_extracted_arrays.update(extracted_arrays)

            # Only store in structure if it's not None/empty after processing
            if processed_value is not None and not (isinstance(processed_value, (dict, list)) and not processed_value):
                reference_structure[name] = processed_value

        return reference_structure, all_extracted_arrays

    def _extract_dataarrays_recursive(self, obj) -> Tuple[Any, Dict[str, xr.DataArray]]:
        """
        Recursively extract DataArrays/TimeSeries from nested structures.

        Args:
            obj: Object to process

        Returns:
            Tuple of (processed_object_with_references, extracted_arrays_dict)
        """
        extracted_arrays = {}

        # Handle DataArrays directly - use their unique name
        if isinstance(obj, xr.DataArray):
            if not obj.name:
                raise ValueError('DataArray must have a unique name for serialization')
            extracted_arrays[obj.name] = obj
            return f':::{obj.name}', extracted_arrays

        # Handle Interface objects - extract their DataArrays too
        elif isinstance(obj, Interface):
            # Get the Interface's reference structure and arrays
            interface_structure, interface_arrays = obj._create_reference_structure()

            # Add all extracted arrays from the nested Interface
            extracted_arrays.update(interface_arrays)
            return interface_structure, extracted_arrays

        # Handle lists
        elif isinstance(obj, list):
            processed_list = []
            for item in obj:
                processed_item, nested_arrays = self._extract_dataarrays_recursive(item)
                extracted_arrays.update(nested_arrays)
                processed_list.append(processed_item)
            return processed_list, extracted_arrays

        # Handle dictionaries
        elif isinstance(obj, dict):
            processed_dict = {}
            for key, value in obj.items():
                processed_value, nested_arrays = self._extract_dataarrays_recursive(value)
                extracted_arrays.update(nested_arrays)
                processed_dict[key] = processed_value
            return processed_dict, extracted_arrays

        # Handle tuples (convert to list for JSON compatibility)
        elif isinstance(obj, tuple):
            processed_list = []
            for item in obj:
                processed_item, nested_arrays = self._extract_dataarrays_recursive(item)
                extracted_arrays.update(nested_arrays)
                processed_list.append(processed_item)
            return processed_list, extracted_arrays

        # For all other types, serialize to basic types
        else:
            return self._serialize_to_basic_types(obj), extracted_arrays

    @classmethod
    def _resolve_reference_structure(cls, structure, arrays_dict: Dict[str, xr.DataArray]):
        """
        Convert reference structure back to actual objects using provided arrays.

        Args:
            structure: Structure containing references (:::name) or special type markers
            arrays_dict: Dictionary of available DataArrays

        Returns:
            Structure with references resolved to actual DataArrays or TimeSeriesData objects
        """
        # Handle DataArray references (including TimeSeriesData)
        if isinstance(structure, str) and structure.startswith(':::'):
            array_name = structure[3:]  # Remove ":::" prefix
            if array_name in arrays_dict:
                array = arrays_dict[array_name]

                # Check if this should be restored as TimeSeriesData
                if TimeSeriesData.is_timeseries_data(array):
                    return TimeSeriesData.from_dataarray(array)
                else:
                    return array
            else:
                logger.critical(f"Referenced DataArray '{array_name}' not found in dataset")
                return None

        elif isinstance(structure, list):
            resolved_list = []
            for item in structure:
                resolved_item = cls._resolve_reference_structure(item, arrays_dict)
                if resolved_item is not None:  # Filter out None values from missing references
                    resolved_list.append(resolved_item)
            return resolved_list

        elif isinstance(structure, dict):
            if structure.get('__class__') and structure['__class__'] in CLASS_REGISTRY:
                # This is a nested Interface object - restore it recursively
                nested_class = CLASS_REGISTRY[structure['__class__']]
                # Remove the __class__ key and process the rest
                nested_data = {k: v for k, v in structure.items() if k != '__class__'}
                # Resolve references in the nested data
                resolved_nested_data = cls._resolve_reference_structure(nested_data, arrays_dict)
                # Create the nested Interface object
                return nested_class(**resolved_nested_data)
            else:
                # Regular dictionary - resolve references in values
                resolved_dict = {}
                for key, value in structure.items():
                    resolved_value = cls._resolve_reference_structure(value, arrays_dict)
                    if resolved_value is not None or value is None:  # Keep None values if they were originally None
                        resolved_dict[key] = resolved_value
                return resolved_dict

        else:
            return structure

    def _serialize_to_basic_types(self, obj):
        """Convert object to basic Python types only (no DataArrays, no custom objects)."""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.ndarray, pd.Series, pd.DataFrame)):
            return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize_to_basic_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_to_basic_types(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            # For any other object, try to convert to string as fallback
            logger.warning(f'Converting unknown type {type(obj)} to string: {obj}')
            return str(obj)

    def to_dataset(self) -> xr.Dataset:
        """
        Convert the object to an xarray Dataset representation.
        All DataArrays and TimeSeries become dataset variables, everything else goes to attrs.

        Returns:
            xr.Dataset: Dataset containing all DataArrays with basic objects only in attributes
        """
        reference_structure, extracted_arrays = self._create_reference_structure()

        # Create the dataset with extracted arrays as variables and structure as attrs
        ds = xr.Dataset(extracted_arrays, attrs=reference_structure)
        return ds

    def to_dict(self) -> Dict:
        """
        Convert the object to a dictionary representation.
        DataArrays/TimeSeries are converted to references, but structure is preserved.

        Returns:
            Dict: Dictionary with references to DataArrays/TimeSeries
        """
        reference_structure, _ = self._create_reference_structure()
        return reference_structure

    def infos(self, use_numpy: bool = True, use_element_label: bool = False) -> Dict:
        """
        Generate a dictionary representation of the object's constructor arguments.
        Built on top of dataset creation for better consistency and analytics capabilities.

        Args:
            use_numpy: Whether to convert NumPy arrays to lists. Defaults to True.
                If True, numeric numpy arrays are preserved as-is.
                If False, they are converted to lists.
            use_element_label: Whether to use element labels instead of full infos for nested objects.

        Returns:
            A dictionary representation optimized for documentation and analysis.
        """
        # Get the core dataset representation
        ds = self.to_dataset()

        # Start with the reference structure from attrs
        info_dict = dict(ds.attrs)

        # Process DataArrays in the dataset based on preferences
        for var_name, data_array in ds.data_vars.items():
            if use_numpy:
                # Keep as DataArray/numpy for analysis
                info_dict[f'_data_{var_name}'] = data_array
            else:
                # Convert to lists for JSON compatibility
                info_dict[f'_data_{var_name}'] = data_array.values.tolist()

        # Apply element label preference to nested structures
        if use_element_label:
            info_dict = self._apply_element_label_preference(info_dict)

        return info_dict

    def _apply_element_label_preference(self, obj):
        """Apply element label preference to nested structures."""
        if isinstance(obj, dict):
            if obj.get('__class__') and 'label' in obj:
                # This looks like an Interface with a label - return just the label
                return obj.get('label', obj.get('__class__'))
            else:
                return {k: self._apply_element_label_preference(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._apply_element_label_preference(item) for item in obj]
        else:
            return obj

    def to_json(self, path: Union[str, pathlib.Path]):
        """
        Save the element to a JSON file for documentation purposes.
        Uses the infos() method for consistent representation.

        Args:
            path: The path to the JSON file.
        """
        data = get_compact_representation(self.infos(use_numpy=False, use_element_label=True))
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def to_netcdf(self, path: Union[str, pathlib.Path], compression: int = 0):
        """
        Save the object to a NetCDF file.

        Args:
            path: Path to save the NetCDF file
            compression: Compression level (0-9)
        """
        from . import io as fx_io  # Assuming fx_io is available

        ds = self.to_dataset()
        fx_io.save_dataset_to_netcdf(ds, path, compression=compression)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> 'Interface':
        """
        Create an instance from an xarray Dataset.

        Args:
            ds: Dataset containing the object data

        Returns:
            Interface instance
        """
        # Get class name and verify it matches
        class_name = ds.attrs.get('__class__')
        if class_name != cls.__name__:
            logger.warning(f"Dataset class '{class_name}' doesn't match target class '{cls.__name__}'")

        # Get the reference structure from attrs
        reference_structure = dict(ds.attrs)

        # Remove the class name since it's not a constructor parameter
        reference_structure.pop('__class__', None)

        # Create arrays dictionary from dataset variables
        arrays_dict = {name: array for name, array in ds.data_vars.items()}

        # Resolve all references using the centralized method
        resolved_params = cls._resolve_reference_structure(reference_structure, arrays_dict)

        return cls(**resolved_params)

    @classmethod
    def from_netcdf(cls, path: Union[str, pathlib.Path]) -> 'Interface':
        """
        Load an instance from a NetCDF file.

        Args:
            path: Path to the NetCDF file

        Returns:
            Interface instance
        """
        from . import io as fx_io  # Assuming fx_io is available

        ds = fx_io.load_dataset_from_netcdf(path)
        return cls.from_dataset(ds)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Interface':
        """
        Create an instance from a dictionary representation.
        This is now a thin wrapper around the reference resolution system.

        Args:
            data: Dictionary containing the data for the object.
        """
        class_name = data.pop('__class__', None)
        if class_name and class_name != cls.__name__:
            logger.warning(f"Dict class '{class_name}' doesn't match target class '{cls.__name__}'")

        # Since dict format doesn't separate arrays, resolve with empty arrays dict
        # References in dict format would need to be handled differently if they exist
        resolved_params = cls._resolve_reference_structure(data, {})
        return cls(**resolved_params)

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
    """This class is the basic Element of flixopt. Every Element has a label"""

    def __init__(self, label: str, meta_data: Dict = None):
        """
        Args:
            label: The label of the element
            meta_data: used to store more information about the Element. Is not used internally, but saved in the results. Only use python native types.
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
    """Stores Variables and Constraints."""

    def __init__(
        self, model: SystemModel, label_of_element: str, label: str = '', label_full: Optional[str] = None
    ):
        """
        Args:
            model: The SystemModel that is used to create the model.
            label_of_element: The label of the parent (Element). Used to construct the full label of the model.
            label: The label of the model. Used to construct the full label of the model.
            label_full: The full label of the model. Can overwrite the full label constructed from the other labels.
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
        logger.debug(f'Created {self.__class__.__name__}  "{self.label_full}"')

    def do_modeling(self):
        raise NotImplementedError('Every Model needs a do_modeling() method')

    def add(
        self, item: Union[linopy.Variable, linopy.Constraint, 'Model'], short_name: Optional[str] = None
    ) -> Union[linopy.Variable, linopy.Constraint, 'Model']:
        """
        Add a variable, constraint or sub-model to the model

        Args:
            item: The variable, constraint or sub-model to add to the model
            short_name: The short name of the variable, constraint or sub-model. If not provided, the full name is used.
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
                f'Item must be a linopy.Variable, linopy.Constraint or flixopt.structure.Model, got {type(item)}'
            )
        return item

    def filter_variables(
        self,
        filter_by: Optional[Literal['binary', 'continuous', 'integer']] = None,
        length: Literal['scalar', 'time'] = None,
    ):
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
        return self._label if self._label else self.label_of_element

    @property
    def label_full(self) -> str:
        """Used to construct the names of variables and constraints"""
        if self._label_full:
            return self._label_full
        elif self._label:
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
    """Stores the mathematical Variables and Constraints for Elements"""

    def __init__(self, model: SystemModel, element: Element):
        """
        Args:
            model: The SystemModel that is used to create the model.
            element: The element this model is created for.
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


class TimeSeriesData(xr.DataArray):
    """Minimal TimeSeriesData that inherits from xr.DataArray with aggregation metadata."""

    def __init__(self, *args, agg_group: Optional[str] = None, agg_weight: Optional[float] = None, **kwargs):
        """
        Args:
            *args: Arguments passed to DataArray
            agg_group: Aggregation group name
            agg_weight: Aggregation weight (0-1)
            **kwargs: Additional arguments passed to DataArray
        """
        if (agg_group is not None) and (agg_weight is not None):
            raise ValueError('Use either agg_group or agg_weight, not both')

        # Let xarray handle all the initialization complexity
        super().__init__(*args, **kwargs)

        # Add our metadata to attrs after initialization
        if agg_group is not None:
            self.attrs['agg_group'] = agg_group
        if agg_weight is not None:
            self.attrs['agg_weight'] = agg_weight

        # Always mark as TimeSeriesData
        self.attrs['__timeseries_data__'] = True

    @property
    def agg_group(self) -> Optional[str]:
        return self.attrs.get('agg_group')

    @property
    def agg_weight(self) -> Optional[float]:
        return self.attrs.get('agg_weight')

    @classmethod
    def from_dataarray(cls, da: xr.DataArray, agg_group: Optional[str] = None, agg_weight: Optional[float] = None):
        """Create TimeSeriesData from DataArray, extracting metadata from attrs."""
        # Get aggregation metadata from attrs or parameters
        final_agg_group = agg_group if agg_group is not None else da.attrs.get('agg_group')
        final_agg_weight = agg_weight if agg_weight is not None else da.attrs.get('agg_weight')

        return cls(da, agg_group=final_agg_group, agg_weight=final_agg_weight)

    @classmethod
    def is_timeseries_data(cls, obj) -> bool:
        """Check if an object is TimeSeriesData."""
        return isinstance(obj, xr.DataArray) and obj.attrs.get('__timeseries_data__', False)

    def __repr__(self):
        agg_info = []
        if self.agg_group:
            agg_info.append(f"agg_group='{self.agg_group}'")
        if self.agg_weight is not None:
            agg_info.append(f'agg_weight={self.agg_weight}')

        info_str = f'TimeSeriesData({", ".join(agg_info)})' if agg_info else 'TimeSeriesData'
        return f'{info_str}\n{super().__repr__()}'


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

    Args:
        data: The input data to process, which may be deeply nested and contain a mix of types.
        use_numpy: If `True`, numeric numpy arrays (`np.ndarray`) are preserved as-is. If `False`, they are converted to lists.
            Default is `True`.
        use_element_label: If `True`, `Element` objects are represented by their `label`. If `False`, they are converted into a dictionary
            based on their initialization parameters. Default is `False`.

    Returns:
        A transformed version of the input data, containing only JSON-compatible types:
        - `int`, `float`, `str`, `bool`, `None`
        - `list`, `dict`
        - `np.ndarray` (if `use_numpy=True`. This is NOT JSON-compatible)

    Raises:
        TypeError: If the data cannot be converted to the specified types.

    Examples:
        >>> copy_and_convert_datatypes({'a': np.array([1, 2, 3]), 'b': Element(label='example')})
        {'a': array([1, 2, 3]), 'b': {'class': 'Element', 'label': 'example'}}

        >>> copy_and_convert_datatypes({'a': np.array([1, 2, 3]), 'b': Element(label='example')}, use_numpy=False)
        {'a': [1, 2, 3], 'b': {'class': 'Element', 'label': 'example'}}

    Notes:
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
        return copy_and_convert_datatypes(data, use_numpy, use_element_label)
    elif isinstance(data, TimeSeriesData):
        return copy_and_convert_datatypes(data.data, use_numpy, use_element_label)

    elif isinstance(data, Interface):
        if use_element_label and isinstance(data, Element):
            return data.label
        return data.infos(use_numpy, use_element_label)
    elif isinstance(data, xr.DataArray):
        # TODO: This is a temporary basic work around
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
