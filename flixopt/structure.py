"""
This module contains the core structure of the flixopt framework.
These classes are not directly used by the end user, but are used by other modules.
"""

import inspect
import json
import logging
import pathlib
from io import StringIO
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union, Collection

import linopy
import numpy as np
import pandas as pd
import xarray as xr
from rich.console import Console
from rich.pretty import Pretty

from . import io as fx_io
from .config import CONFIG
from .core import NonTemporalData, Scalar, TemporalDataUser, TimeSeriesData, get_dataarray_stats, FlowSystemDimensions

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


class FlowSystemModel(linopy.Model):
    """
    The FlowSystemModel is the linopy Model that is used to create the mathematical model of the flow_system.
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
        solution['objective'] = self.objective.value
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
            'Flows': {
                flow.label_full: flow.model.results_structure()
                for flow in sorted(self.flow_system.flows.values(), key=lambda flow: flow.label_full.upper())
            },
        }
        return solution.reindex(time=self.flow_system.timesteps_extra)

    @property
    def hours_per_step(self):
        return self.flow_system.hours_per_timestep

    @property
    def hours_of_previous_timesteps(self):
        return self.flow_system.hours_of_previous_timesteps

    def get_coords(
        self,
        dims: Optional[Collection[str]] = None,
        extra_timestep: bool = False,
    ) -> Optional[xr.Coordinates]:
        """
        Returns the coordinates of the model

        Args:
            dims: The dimensions to include in the coordinates. If None, includes all dimensions
            extra_timestep: If True, uses extra timesteps instead of regular timesteps

        Returns:
            The coordinates of the model, or None if no coordinates are available

        Raises:
            ValueError: If extra_timestep=True but 'time' is not in dims
        """
        if extra_timestep and dims is not None and 'time' not in dims:
            raise ValueError('extra_timestep=True requires "time" to be included in dims')

        if dims is None:
            coords = dict(self.flow_system.coords)
        else:
            coords = {k: v for k, v in self.flow_system.coords.items() if k in dims}

        if extra_timestep and coords:
            coords['time'] = self.flow_system.timesteps_extra

        return xr.Coordinates(coords) if coords else None

    @property
    def weights(self) -> Union[int, xr.DataArray]:
        """Returns the scenario weights of the FlowSystem. If None, return weights that are normalized to 1 (one)"""
        if self.flow_system.weights is None:
            weights = self.flow_system.fit_to_model_coords('weights', 1, has_time_dim=False)

            return weights / weights.sum()

        return self.flow_system.weights


class Interface:
    """
    Base class for all Elements and Models in flixopt that provides serialization capabilities.

    This class enables automatic serialization/deserialization of objects containing xarray DataArrays
    and nested Interface objects to/from xarray Datasets and NetCDF files. It uses introspection
    of constructor parameters to automatically handle most serialization scenarios.

    Key Features:
        - Automatic extraction and restoration of xarray DataArrays
        - Support for nested Interface objects
        - NetCDF and JSON export/import
        - Recursive handling of complex nested structures

    Subclasses must implement:
        transform_data(flow_system): Transform data to match FlowSystem dimensions
    """

    def transform_data(self, flow_system: 'FlowSystem'):
        """Transform the data of the interface to match the FlowSystem's dimensions.

        Args:
            flow_system: The FlowSystem containing timing and dimensional information

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError('Every Interface subclass needs a transform_data() method')

    def _create_reference_structure(self) -> Tuple[Dict, Dict[str, xr.DataArray]]:
        """
        Convert all DataArrays to references and extract them.
        This is the core method that both to_dict() and to_dataset() build upon.

        Returns:
            Tuple of (reference_structure, extracted_arrays_dict)

        Raises:
            ValueError: If DataArrays don't have unique names or are duplicated
        """
        # Get constructor parameters using caching for performance
        if not hasattr(self, '_cached_init_params'):
            self._cached_init_params = list(inspect.signature(self.__init__).parameters.keys())

        # Process all constructor parameters
        reference_structure = {'__class__': self.__class__.__name__}
        all_extracted_arrays = {}

        for name in self._cached_init_params:
            if name == 'self':  # Skip self and timesteps. Timesteps are directly stored in Datasets
                continue

            value = getattr(self, name, None)

            if value is None:
                continue
            if isinstance(value, pd.Index):
                logger.debug(f'Skipping {name=} because it is an Index')
                continue

            # Extract arrays and get reference structure
            processed_value, extracted_arrays = self._extract_dataarrays_recursive(value, name)

            # Check for array name conflicts
            conflicts = set(all_extracted_arrays.keys()) & set(extracted_arrays.keys())
            if conflicts:
                raise ValueError(
                    f'DataArray name conflicts detected: {conflicts}. '
                    f'Each DataArray must have a unique name for serialization.'
                )

            # Add extracted arrays to the collection
            all_extracted_arrays.update(extracted_arrays)

            # Only store in structure if it's not None/empty after processing
            if processed_value is not None and not self._is_empty_container(processed_value):
                reference_structure[name] = processed_value

        return reference_structure, all_extracted_arrays

    @staticmethod
    def _is_empty_container(obj) -> bool:
        """Check if object is an empty container (dict, list, tuple, set)."""
        return isinstance(obj, (dict, list, tuple, set)) and len(obj) == 0

    def _extract_dataarrays_recursive(self, obj, context_name: str = '') -> Tuple[Any, Dict[str, xr.DataArray]]:
        """
        Recursively extract DataArrays from nested structures.

        Args:
            obj: Object to process
            context_name: Name context for better error messages

        Returns:
            Tuple of (processed_object_with_references, extracted_arrays_dict)

        Raises:
            ValueError: If DataArrays don't have unique names
        """
        extracted_arrays = {}

        # Handle DataArrays directly - use their unique name
        if isinstance(obj, xr.DataArray):
            if not obj.name:
                raise ValueError(
                    f'DataArrays must have a unique name for serialization. '
                    f'Unnamed DataArray found in {context_name}. Please set array.name = "unique_name"'
                )

            array_name = str(obj.name)  # Ensure string type
            if array_name in extracted_arrays:
                raise ValueError(
                    f'DataArray name "{array_name}" is duplicated in {context_name}. '
                    f'Each DataArray must have a unique name for serialization.'
                )

            extracted_arrays[array_name] = obj
            return f':::{array_name}', extracted_arrays

        # Handle Interface objects - extract their DataArrays too
        elif isinstance(obj, Interface):
            try:
                interface_structure, interface_arrays = obj._create_reference_structure()
                extracted_arrays.update(interface_arrays)
                return interface_structure, extracted_arrays
            except Exception as e:
                raise ValueError(f'Failed to process nested Interface object in {context_name}: {e}') from e

        # Handle sequences (lists, tuples)
        elif isinstance(obj, (list, tuple)):
            processed_items = []
            for i, item in enumerate(obj):
                item_context = f'{context_name}[{i}]' if context_name else f'item[{i}]'
                processed_item, nested_arrays = self._extract_dataarrays_recursive(item, item_context)
                extracted_arrays.update(nested_arrays)
                processed_items.append(processed_item)
            return processed_items, extracted_arrays

        # Handle dictionaries
        elif isinstance(obj, dict):
            processed_dict = {}
            for key, value in obj.items():
                key_context = f'{context_name}.{key}' if context_name else str(key)
                processed_value, nested_arrays = self._extract_dataarrays_recursive(value, key_context)
                extracted_arrays.update(nested_arrays)
                processed_dict[key] = processed_value
            return processed_dict, extracted_arrays

        # Handle sets (convert to list for JSON compatibility)
        elif isinstance(obj, set):
            processed_items = []
            for i, item in enumerate(obj):
                item_context = f'{context_name}.set_item[{i}]' if context_name else f'set_item[{i}]'
                processed_item, nested_arrays = self._extract_dataarrays_recursive(item, item_context)
                extracted_arrays.update(nested_arrays)
                processed_items.append(processed_item)
            return processed_items, extracted_arrays

        # For all other types, serialize to basic types
        else:
            return self._serialize_to_basic_types(obj), extracted_arrays

    @classmethod
    def _resolve_dataarray_reference(
        cls, reference: str, arrays_dict: Dict[str, xr.DataArray]
    ) -> Union[xr.DataArray, TimeSeriesData]:
        """
        Resolve a single DataArray reference (:::name) to actual DataArray or TimeSeriesData.

        Args:
            reference: Reference string starting with ":::"
            arrays_dict: Dictionary of available DataArrays

        Returns:
            Resolved DataArray or TimeSeriesData object

        Raises:
            ValueError: If referenced array is not found
        """
        array_name = reference[3:]  # Remove ":::" prefix
        if array_name not in arrays_dict:
            raise ValueError(f"Referenced DataArray '{array_name}' not found in dataset")

        array = arrays_dict[array_name]

        # Handle null values with warning
        if array.isnull().any():
            logger.warning(f"DataArray '{array_name}' contains null values. Dropping them.")
            array = array.dropna(dim='time', how='all')

        # Check if this should be restored as TimeSeriesData
        if TimeSeriesData.is_timeseries_data(array):
            return TimeSeriesData.from_dataarray(array)

        return array

    @classmethod
    def _resolve_reference_structure(cls, structure, arrays_dict: Dict[str, xr.DataArray]):
        """
        Convert reference structure back to actual objects using provided arrays.

        Args:
            structure: Structure containing references (:::name) or special type markers
            arrays_dict: Dictionary of available DataArrays

        Returns:
            Structure with references resolved to actual DataArrays or objects

        Raises:
            ValueError: If referenced arrays are not found or class is not registered
        """
        # Handle DataArray references
        if isinstance(structure, str) and structure.startswith(':::'):
            return cls._resolve_dataarray_reference(structure, arrays_dict)

        elif isinstance(structure, list):
            resolved_list = []
            for item in structure:
                resolved_item = cls._resolve_reference_structure(item, arrays_dict)
                if resolved_item is not None:  # Filter out None values from missing references
                    resolved_list.append(resolved_item)
            return resolved_list

        elif isinstance(structure, dict):
            if structure.get('__class__'):
                class_name = structure['__class__']
                if class_name not in CLASS_REGISTRY:
                    raise ValueError(
                        f"Class '{class_name}' not found in CLASS_REGISTRY. "
                        f'Available classes: {list(CLASS_REGISTRY.keys())}'
                    )

                # This is a nested Interface object - restore it recursively
                nested_class = CLASS_REGISTRY[class_name]
                # Remove the __class__ key and process the rest
                nested_data = {k: v for k, v in structure.items() if k != '__class__'}
                # Resolve references in the nested data
                resolved_nested_data = cls._resolve_reference_structure(nested_data, arrays_dict)

                try:
                    return nested_class(**resolved_nested_data)
                except Exception as e:
                    raise ValueError(f'Failed to create instance of {class_name}: {e}') from e
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
        """
        Convert object to basic Python types only (no DataArrays, no custom objects).

        Args:
            obj: Object to serialize

        Returns:
            Object converted to basic Python types (str, int, float, bool, list, dict)
        """
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.ndarray, pd.Series, pd.DataFrame)):
            return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize_to_basic_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_to_basic_types(item) for item in obj]
        elif isinstance(obj, set):
            return [self._serialize_to_basic_types(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):  # Custom objects with attributes
            logger.warning(f'Converting custom object {type(obj)} to dict representation: {obj}')
            return {str(k): self._serialize_to_basic_types(v) for k, v in obj.__dict__.items()}
        else:
            # For any other object, try to convert to string as fallback
            logger.warning(f'Converting unknown type {type(obj)} to string: {obj}')
            return str(obj)

    def to_dataset(self) -> xr.Dataset:
        """
        Convert the object to an xarray Dataset representation.
        All DataArrays become dataset variables, everything else goes to attrs.

        Its recommended to only call this method on Interfaces with all numeric data stored as xr.DataArrays.
        Interfaces inside a FlowSystem are automatically converted this form after connecting and transforming the FlowSystem.

        Returns:
            xr.Dataset: Dataset containing all DataArrays with basic objects only in attributes

        Raises:
            ValueError: If serialization fails due to naming conflicts or invalid data
        """
        try:
            reference_structure, extracted_arrays = self._create_reference_structure()
            # Create the dataset with extracted arrays as variables and structure as attrs
            return xr.Dataset(extracted_arrays, attrs=reference_structure)
        except Exception as e:
            raise ValueError(
                f'Failed to convert {self.__class__.__name__} to dataset. Its recommended to only call this method on '
                f'a fully connected and transformed FlowSystem, or Interfaces inside such a FlowSystem.'
                f'Original Error: {e}') from e

    def to_netcdf(self, path: Union[str, pathlib.Path], compression: int = 0):
        """
        Save the object to a NetCDF file.

        Args:
            path: Path to save the NetCDF file
            compression: Compression level (0-9)

        Raises:
            ValueError: If serialization fails
            IOError: If file cannot be written
        """
        try:
            ds = self.to_dataset()
            fx_io.save_dataset_to_netcdf(ds, path, compression=compression)
        except Exception as e:
            raise IOError(f'Failed to save {self.__class__.__name__} to NetCDF file {path}: {e}') from e

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> 'Interface':
        """
        Create an instance from an xarray Dataset.

        Args:
            ds: Dataset containing the object data

        Returns:
            Interface instance

        Raises:
            ValueError: If dataset format is invalid or class mismatch
        """
        try:
            # Get class name and verify it matches
            class_name = ds.attrs.get('__class__')
            if class_name and class_name != cls.__name__:
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
        except Exception as e:
            raise ValueError(f'Failed to create {cls.__name__} from dataset: {e}') from e

    @classmethod
    def from_netcdf(cls, path: Union[str, pathlib.Path]) -> 'Interface':
        """
        Load an instance from a NetCDF file.

        Args:
            path: Path to the NetCDF file

        Returns:
            Interface instance

        Raises:
            IOError: If file cannot be read
            ValueError: If file format is invalid
        """
        try:
            ds = fx_io.load_dataset_from_netcdf(path)
            return cls.from_dataset(ds)
        except Exception as e:
            raise IOError(f'Failed to load {cls.__name__} from NetCDF file {path}: {e}') from e

    def get_structure(self, clean: bool = False, stats: bool = False) -> Dict:
        """
        Get object structure as a dictionary.

        Args:
            clean: If True, remove None and empty dicts and lists.
            stats: If True, replace DataArray references with statistics

        Returns:
            Dictionary representation of the object structure
        """
        reference_structure, extracted_arrays = self._create_reference_structure()

        if stats:
            # Replace references with statistics
            reference_structure = self._replace_references_with_stats(reference_structure, extracted_arrays)

        if clean:
            return fx_io.remove_none_and_empty(reference_structure)
        return reference_structure

    def _replace_references_with_stats(self, structure, arrays_dict: Dict[str, xr.DataArray]):
        """Replace DataArray references with statistical summaries."""
        if isinstance(structure, str) and structure.startswith(':::'):
            array_name = structure[3:]
            if array_name in arrays_dict:
                return get_dataarray_stats(arrays_dict[array_name])
            return structure

        elif isinstance(structure, dict):
            return {k: self._replace_references_with_stats(v, arrays_dict) for k, v in structure.items()}

        elif isinstance(structure, list):
            return [self._replace_references_with_stats(item, arrays_dict) for item in structure]

        return structure

    def to_json(self, path: Union[str, pathlib.Path]):
        """
        Save the object to a JSON file.
        This is meant for documentation and comparison, not for reloading.

        Args:
            path: The path to the JSON file.

        Raises:
            IOError: If file cannot be written
        """
        try:
            # Use the stats mode for JSON export (cleaner output)
            data = self.get_structure(clean=True, stats=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            raise IOError(f'Failed to save {self.__class__.__name__} to JSON file {path}: {e}') from e

    def __repr__(self):
        """Return a detailed string representation for debugging."""
        try:
            # Get the constructor arguments and their current values
            init_signature = inspect.signature(self.__init__)
            init_args = init_signature.parameters

            # Create a dictionary with argument names and their values, with better formatting
            args_parts = []
            for name in init_args:
                if name == 'self':
                    continue
                value = getattr(self, name, None)
                # Truncate long representations
                value_repr = repr(value)
                if len(value_repr) > 50:
                    value_repr = value_repr[:47] + '...'
                args_parts.append(f'{name}={value_repr}')

            args_str = ', '.join(args_parts)
            return f'{self.__class__.__name__}({args_str})'
        except Exception:
            # Fallback if introspection fails
            return f'{self.__class__.__name__}(<repr_failed>)'

    def __str__(self):
        """Return a user-friendly string representation."""
        try:
            data = self.get_structure(clean=True, stats=True)
            with StringIO() as output_buffer:
                console = Console(file=output_buffer, width=1000)  # Adjust width as needed
                console.print(Pretty(data, expand_all=True, indent_guides=True))
                return output_buffer.getvalue()
        except Exception:
            # Fallback if structure generation fails
            return f'{self.__class__.__name__} instance'

    def copy(self) -> 'Interface':
        """
        Create a copy of the Interface object.

        Uses the existing serialization infrastructure to ensure proper copying
        of all DataArrays and nested objects.

        Returns:
            A new instance of the same class with copied data.
        """
        # Convert to dataset, copy it, and convert back
        dataset = self.to_dataset().copy(deep=True)
        return self.__class__.from_dataset(dataset)

    def __copy__(self):
        """Support for copy.copy()."""
        return self.copy()

    def __deepcopy__(self, memo):
        """Support for copy.deepcopy()."""
        return self.copy()


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

    def create_model(self, model: FlowSystemModel) -> 'ElementModel':
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
        self, model: FlowSystemModel, label_of_element: str, label: str = '', label_full: Optional[str] = None
    ):
        """
        Args:
            model: The FlowSystemModel that is used to create the model.
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
            self._variables_short[short_name] = item.name
        elif isinstance(item, linopy.Constraint):
            self._constraints_direct.append(item.name)
            self._constraints_short[short_name] = item.name
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

    def get_variable_by_short_name(self, short_name: str, default_return = None) -> Optional[linopy.Variable]:
        """Get variable by short name"""
        if short_name not in self._variables_short:
            return default_return
        return self._model.variables[self._variables_short.get(short_name)]

    def get_constraint_by_short_name(self, short_name: str, default_return = None) -> Optional[linopy.Constraint]:
        """Get variable by short name"""
        if short_name not in self._constraints_short:
            return default_return
        return self._model.constraints[self._constraints_short.get(short_name)]


class BaseFeatureModel(Model):
    """Minimal base class for feature models that use factory patterns"""

    def __init__(self, model: FlowSystemModel, label_of_element: str, parameters, label: Optional[str] = None):
        super().__init__(model, label_of_element, label or self.__class__.__name__)
        self.parameters = parameters

    def do_modeling(self):
        """Template method - creates variables and constraints, then effects"""
        self.create_variables_and_constraints()
        self.add_effects()

    def create_variables_and_constraints(self):
        """Override in subclasses to create variables and constraints"""
        raise NotImplementedError('Subclasses must implement create_variables_and_constraints()')

    def add_effects(self):
        """Override in subclasses to add effects"""
        pass  # Default: no effects


class ElementModel(Model):
    """Stores the mathematical Variables and Constraints for Elements"""

    def __init__(self, model: FlowSystemModel, element: Element):
        """
        Args:
            model: The FlowSystemModel that is used to create the model.
            element: The element this model is created for.
        """
        super().__init__(model, label_of_element=element.label_full, label=element.label, label_full=element.label_full)
        self.element = element

    def results_structure(self):
        return {
            'label': self.label_full,
            'variables': list(self.variables),
            'constraints': list(self.constraints),
        }
