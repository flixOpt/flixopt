from __future__ import annotations

import inspect
import json
import logging
import pathlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

if TYPE_CHECKING:
    import linopy

logger = logging.getLogger('flixopt')


def remove_none_and_empty(obj):
    """Recursively removes None and empty dicts and lists values from a dictionary or list."""

    if isinstance(obj, dict):
        return {
            k: remove_none_and_empty(v)
            for k, v in obj.items()
            if not (v is None or (isinstance(v, (list, dict)) and not v))
        }

    elif isinstance(obj, list):
        return [remove_none_and_empty(v) for v in obj if not (v is None or (isinstance(v, (list, dict)) and not v))]

    else:
        return obj


def round_nested_floats(obj: dict | list | float | int | Any, decimals: int = 2) -> dict | list | float | int | Any:
    """Recursively round floating point numbers in nested data structures and convert it to python native types.

    This function traverses nested data structures (dictionaries, lists) and rounds
    any floating point numbers to the specified number of decimal places. It handles
    various data types including NumPy arrays and xarray DataArrays by converting
    them to lists with rounded values.

    Args:
        obj: The object to process. Can be a dict, list, float, int, numpy.ndarray,
            xarray.DataArray, or any other type.
        decimals (int, optional): Number of decimal places to round to. Defaults to 2.

    Returns:
        The processed object with the same structure as the input, but with all floating point numbers rounded to the specified precision. NumPy arrays and xarray DataArrays are converted to lists.

    Examples:
        >>> data = {'a': 3.14159, 'b': [1.234, 2.678]}
        >>> round_nested_floats(data, decimals=2)
        {'a': 3.14, 'b': [1.23, 2.68]}

        >>> import numpy as np
        >>> arr = np.array([1.234, 5.678])
        >>> round_nested_floats(arr, decimals=1)
        [1.2, 5.7]
    """
    if isinstance(obj, dict):
        return {k: round_nested_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_nested_floats(v, decimals) for v in obj]
    elif isinstance(obj, np.floating):
        return round(float(obj), decimals)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, int):
        return obj
    elif isinstance(obj, np.ndarray):
        return np.round(obj, decimals).tolist()
    elif isinstance(obj, xr.DataArray):
        return obj.round(decimals).values.tolist()
    return obj


# ============================================================================
# Centralized JSON and YAML I/O Functions
# ============================================================================


def load_json(path: str | pathlib.Path) -> dict | list:
    """
    Load data from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Loaded data (typically dict or list).

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = pathlib.Path(path)
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_json(
    data: dict | list,
    path: str | pathlib.Path,
    indent: int = 4,
    ensure_ascii: bool = False,
    **kwargs,
) -> None:
    """
    Save data to a JSON file with consistent formatting.

    Args:
        data: Data to save (dict or list).
        path: Path to save the JSON file.
        indent: Number of spaces for indentation (default: 4).
        ensure_ascii: If False, allow Unicode characters (default: False).
        **kwargs: Additional arguments to pass to json.dump().
    """
    path = pathlib.Path(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def load_yaml(path: str | pathlib.Path) -> dict | list:
    """
    Load data from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Loaded data (typically dict or list), or empty dict if file is empty.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        Note: Returns {} for empty YAML files instead of None.
    """
    path = pathlib.Path(path)
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _load_yaml_unsafe(path: str | pathlib.Path) -> dict | list:
    """
    INTERNAL: Load YAML allowing arbitrary tags. Do not use on untrusted input.

    This function exists only for loading internally-generated files that may
    contain custom YAML tags. Never use this on user-provided files.

    Args:
        path: Path to the YAML file.

    Returns:
        Loaded data (typically dict or list), or empty dict if file is empty.
    """
    path = pathlib.Path(path)
    with open(path, encoding='utf-8') as f:
        return yaml.unsafe_load(f) or {}


def save_yaml(
    data: dict | list,
    path: str | pathlib.Path,
    indent: int = 4,
    width: int = 1000,
    allow_unicode: bool = True,
    sort_keys: bool = False,
    **kwargs,
) -> None:
    """
    Save data to a YAML file with consistent formatting.

    Args:
        data: Data to save (dict or list).
        path: Path to save the YAML file.
        indent: Number of spaces for indentation (default: 4).
        width: Maximum line width (default: 1000).
        allow_unicode: If True, allow Unicode characters (default: True).
        sort_keys: If True, sort dictionary keys (default: False).
        **kwargs: Additional arguments to pass to yaml.safe_dump().
    """
    path = pathlib.Path(path)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(
            data,
            f,
            indent=indent,
            width=width,
            allow_unicode=allow_unicode,
            sort_keys=sort_keys,
            default_flow_style=False,
            **kwargs,
        )


def load_config_file(path: str | pathlib.Path) -> dict:
    """
    Load a configuration file, automatically detecting JSON or YAML format.

    This function intelligently tries to load the file based on its extension,
    with fallback support if the primary format fails.

    Supported extensions:
    - .json: Tries JSON first, falls back to YAML
    - .yaml, .yml: Tries YAML first, falls back to JSON
    - Others: Tries YAML, then JSON

    Args:
        path: Path to the configuration file.

    Returns:
        Loaded configuration as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If neither JSON nor YAML parsing succeeds.
    """
    path = pathlib.Path(path)

    if not path.exists():
        raise FileNotFoundError(f'Configuration file not found: {path}')

    # Try based on file extension
    # Normalize extension to lowercase for case-insensitive matching
    suffix = path.suffix.lower()

    if suffix == '.json':
        try:
            return load_json(path)
        except json.JSONDecodeError:
            logger.warning(f'Failed to parse {path} as JSON, trying YAML')
            try:
                return load_yaml(path)
            except yaml.YAMLError as e:
                raise ValueError(f'Failed to parse {path} as JSON or YAML') from e

    elif suffix in ['.yaml', '.yml']:
        try:
            return load_yaml(path)
        except yaml.YAMLError:
            logger.warning(f'Failed to parse {path} as YAML, trying JSON')
            try:
                return load_json(path)
            except json.JSONDecodeError as e:
                raise ValueError(f'Failed to parse {path} as YAML or JSON') from e

    else:
        # Unknown extension, try YAML first (more common for config)
        try:
            return load_yaml(path)
        except yaml.YAMLError:
            try:
                return load_json(path)
            except json.JSONDecodeError as e:
                raise ValueError(f'Failed to parse {path} as YAML or JSON') from e


def _save_yaml_multiline(data, output_file='formatted_output.yaml'):
    """
    Save dictionary data to YAML with proper multi-line string formatting.
    Handles complex string patterns including backticks, special characters,
    and various newline formats.

    Args:
        data (dict): Dictionary containing string data
        output_file (str): Path to output YAML file
    """
    # Process strings to normalize all newlines and handle special patterns
    processed_data = _normalize_complex_data(data)

    # Define a custom representer for strings
    def represent_str(dumper, data):
        # Use literal block style (|) for multi-line strings
        if '\n' in data:
            # Clean up formatting for literal block style
            data = data.strip()  # Remove leading/trailing whitespace
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')

        # Use quoted style for strings with special characters
        elif any(char in data for char in ':`{}[]#,&*!|>%@'):
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

        # Use plain style for simple strings
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    # Configure dumper options for better formatting
    class CustomDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)

    # Bind representer locally to CustomDumper to avoid global side effects
    CustomDumper.add_representer(str, represent_str)

    # Write to file with settings that ensure proper formatting
    with open(output_file, 'w', encoding='utf-8') as file:
        yaml.dump(
            processed_data,
            file,
            Dumper=CustomDumper,
            sort_keys=False,  # Preserve dictionary order
            default_flow_style=False,  # Use block style for mappings
            width=1000,  # Set a reasonable line width
            allow_unicode=True,  # Support Unicode characters
            indent=4,  # Set consistent indentation
        )


def _normalize_complex_data(data):
    """
    Recursively normalize strings in complex data structures.

    Handles dictionaries, lists, and strings, applying various text normalization
    rules while preserving important formatting elements.

    Args:
        data: Any data type (dict, list, str, or primitive)

    Returns:
        Data with all strings normalized according to defined rules
    """
    if isinstance(data, dict):
        return {key: _normalize_complex_data(value) for key, value in data.items()}

    elif isinstance(data, list):
        return [_normalize_complex_data(item) for item in data]

    elif isinstance(data, str):
        return _normalize_string_content(data)

    else:
        return data


def _normalize_string_content(text):
    """
    Apply comprehensive string normalization rules.

    Args:
        text: The string to normalize

    Returns:
        Normalized string with standardized formatting
    """
    # Standardize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Convert escaped newlines to actual newlines (avoiding double-backslashes)
    text = re.sub(r'(?<!\\)\\n', '\n', text)

    # Normalize double backslashes before specific escape sequences
    text = re.sub(r'\\\\([rtn])', r'\\\1', text)

    # Standardize constraint headers format
    text = re.sub(r'Constraint\s*`([^`]+)`\s*(?:\\n|[\s\n]*)', r'Constraint `\1`\n', text)

    # Clean up ellipsis patterns
    text = re.sub(r'[\t ]*(\.\.\.)', r'\1', text)

    # Limit consecutive newlines (max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def document_linopy_model(model: linopy.Model, path: pathlib.Path | None = None) -> dict[str, str]:
    """
    Convert all model variables and constraints to a structured string representation.
    This can take multiple seconds for large models.
    The output can be saved to a yaml file with readable formating applied.

    Args:
        path (pathlib.Path, optional): Path to save the document. Defaults to None.
    """
    documentation = {
        'objective': model.objective.__repr__(),
        'termination_condition': model.termination_condition,
        'status': model.status,
        'nvars': model.nvars,
        'nvarsbin': model.binaries.nvars if len(model.binaries) > 0 else 0,  # Temporary, waiting for linopy to fix
        'nvarscont': model.continuous.nvars if len(model.continuous) > 0 else 0,  # Temporary, waiting for linopy to fix
        'ncons': model.ncons,
        'variables': {variable_name: variable.__repr__() for variable_name, variable in model.variables.items()},
        'constraints': {
            constraint_name: constraint.__repr__() for constraint_name, constraint in model.constraints.items()
        },
        'binaries': list(model.binaries),
        'integers': list(model.integers),
        'continuous': list(model.continuous),
        'infeasible_constraints': '',
    }

    if model.status == 'warning':
        logger.critical(f'The model has a warning status {model.status=}. Trying to extract infeasibilities')
        try:
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()

            # Redirect stdout to our buffer
            with redirect_stdout(f):
                model.print_infeasibilities()

            documentation['infeasible_constraints'] = f.getvalue()
        except NotImplementedError:
            logger.critical(
                'Infeasible constraints could not get retrieved. This functionality is only availlable with gurobi'
            )
            documentation['infeasible_constraints'] = 'Not possible to retrieve infeasible constraints'

    if path is not None:
        if path.suffix not in ['.yaml', '.yml']:
            raise ValueError(f'Invalid file extension for path {path}. Only .yaml and .yml are supported')
        _save_yaml_multiline(documentation, str(path))

    return documentation


def save_dataset_to_netcdf(
    ds: xr.Dataset,
    path: str | pathlib.Path,
    compression: int = 0,
) -> None:
    """
    Save a dataset to a netcdf file. Store all attrs as JSON strings in 'attrs' attributes.

    Args:
        ds: Dataset to save.
        path: Path to save the dataset to.
        compression: Compression level for the dataset (0-9). 0 means no compression. 5 is a good default.

    Raises:
        ValueError: If the path has an invalid file extension.
    """
    path = pathlib.Path(path)
    if path.suffix not in ['.nc', '.nc4']:
        raise ValueError(f'Invalid file extension for path {path}. Only .nc and .nc4 are supported')

    ds = ds.copy(deep=True)
    ds.attrs = {'attrs': json.dumps(ds.attrs)}

    # Convert all DataArray attrs to JSON strings
    for var_name, data_var in ds.data_vars.items():
        if data_var.attrs:  # Only if there are attrs
            ds[var_name].attrs = {'attrs': json.dumps(data_var.attrs)}

    # Also handle coordinate attrs if they exist
    for coord_name, coord_var in ds.coords.items():
        if hasattr(coord_var, 'attrs') and coord_var.attrs:
            ds[coord_name].attrs = {'attrs': json.dumps(coord_var.attrs)}

    ds.to_netcdf(
        path,
        encoding=None
        if compression == 0
        else {data_var: {'zlib': True, 'complevel': compression} for data_var in ds.data_vars},
        engine='netcdf4',
    )


def load_dataset_from_netcdf(path: str | pathlib.Path) -> xr.Dataset:
    """
    Load a dataset from a netcdf file. Load all attrs from 'attrs' attributes.

    Args:
        path: Path to load the dataset from.

    Returns:
        Dataset: Loaded dataset with restored attrs.
    """
    ds = xr.load_dataset(str(path), engine='netcdf4')

    # Restore Dataset attrs
    if 'attrs' in ds.attrs:
        ds.attrs = json.loads(ds.attrs['attrs'])

    # Restore DataArray attrs
    for var_name, data_var in ds.data_vars.items():
        if 'attrs' in data_var.attrs:
            ds[var_name].attrs = json.loads(data_var.attrs['attrs'])

    # Restore coordinate attrs
    for coord_name, coord_var in ds.coords.items():
        if hasattr(coord_var, 'attrs') and 'attrs' in coord_var.attrs:
            ds[coord_name].attrs = json.loads(coord_var.attrs['attrs'])

    return ds


@dataclass
class CalculationResultsPaths:
    """Container for all paths related to saving CalculationResults."""

    folder: pathlib.Path
    name: str

    def __post_init__(self):
        """Initialize all path attributes."""
        self._update_paths()

    def _update_paths(self):
        """Update all path attributes based on current folder and name."""
        self.linopy_model = self.folder / f'{self.name}--linopy_model.nc4'
        self.solution = self.folder / f'{self.name}--solution.nc4'
        self.summary = self.folder / f'{self.name}--summary.yaml'
        self.network = self.folder / f'{self.name}--network.json'
        self.flow_system = self.folder / f'{self.name}--flow_system.nc4'
        self.model_documentation = self.folder / f'{self.name}--model_documentation.yaml'

    def all_paths(self) -> dict[str, pathlib.Path]:
        """Return a dictionary of all paths."""
        return {
            'linopy_model': self.linopy_model,
            'solution': self.solution,
            'summary': self.summary,
            'network': self.network,
            'flow_system': self.flow_system,
            'model_documentation': self.model_documentation,
        }

    def create_folders(self, parents: bool = False) -> None:
        """Ensure the folder exists.
        Args:
            parents: Whether to create the parent folders if they do not exist.
        """
        if not self.folder.exists():
            try:
                self.folder.mkdir(parents=parents)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f'Folder {self.folder} and its parent do not exist. Please create them first.'
                ) from e

    def update(self, new_name: str | None = None, new_folder: pathlib.Path | None = None) -> None:
        """Update name and/or folder and refresh all paths."""
        if new_name is not None:
            self.name = new_name
        if new_folder is not None:
            if not new_folder.is_dir() or not new_folder.exists():
                raise FileNotFoundError(f'Folder {new_folder} does not exist or is not a directory.')
            self.folder = new_folder
        self._update_paths()


def numeric_to_str_for_repr(
    value: int | float | np.integer | np.floating | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray,
    precision: int = 1,
    atol: float = 1e-10,
) -> str:
    """Format value for display in repr methods.

    For single values or uniform arrays, returns the formatted value.
    For arrays with variation, returns a range showing min-max.

    Args:
        value: Numeric value or container (DataArray, array, Series, DataFrame)
        precision: Number of decimal places (default: 1)
        atol: Absolute tolerance for considering values equal (default: 1e-10)

    Returns:
        Formatted string representation:
        - Single/uniform values: "100.0"
        - Nearly uniform values: "~100.0" (values differ slightly but display similarly)
        - Varying values: "50.0-150.0" (shows range from min to max)

    Raises:
        TypeError: If value cannot be converted to numeric format
    """
    # Handle simple scalar types
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f'{float(value):.{precision}f}'

    # Extract array data for variation checking
    arr = None
    if isinstance(value, xr.DataArray):
        arr = value.values.flatten()
    elif isinstance(value, (np.ndarray, pd.Series)):
        arr = np.asarray(value).flatten()
    elif isinstance(value, pd.DataFrame):
        arr = value.values.flatten()
    else:
        # Fallback for unknown types
        try:
            return f'{float(value):.{precision}f}'
        except (TypeError, ValueError) as e:
            raise TypeError(f'Cannot format value of type {type(value).__name__} for repr') from e

    # Check for single value
    if arr.size == 1:
        return f'{float(arr[0]):.{precision}f}'

    # Check if all values are the same or very close
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    # First check: values are essentially identical
    if np.allclose(min_val, max_val, atol=atol):
        return f'{float(np.mean(arr)):.{precision}f}'

    # Second check: display values are the same but actual values differ slightly
    min_str = f'{min_val:.{precision}f}'
    max_str = f'{max_val:.{precision}f}'
    if min_str == max_str:
        return f'~{min_str}'

    # Values vary significantly - show range
    return f'{min_str}-{max_str}'


def build_repr_from_init(
    obj: object,
    excluded_params: set[str] | None = None,
    info: str = '',
    label_as_positional: bool = True,
    skip_default_size: bool = False,
) -> str:
    """Build a repr string from __init__ signature, showing non-default parameter values.

    This utility function extracts common repr logic used across flixopt classes.
    It introspects the __init__ method to build a constructor-style repr showing
    only parameters that differ from their defaults.

    Args:
        obj: The object to create repr for
        excluded_params: Set of parameter names to exclude (e.g., {'self', 'inputs', 'outputs'})
                        Default excludes 'self', 'label', and 'kwargs'
        info: Optional comment to append (e.g., '2 flows (1 in, 1 out)')
        label_as_positional: If True and 'label' param exists, show it as first positional arg
        skip_default_size: If True, skip 'size' parameter when it equals CONFIG.Modeling.big

    Returns:
        Formatted repr string like: ClassName("label", param=value)  # info
    """
    if excluded_params is None:
        excluded_params = {'self', 'label', 'kwargs'}
    else:
        # Always exclude 'self'
        excluded_params = excluded_params | {'self'}

    try:
        # Get the constructor arguments and their current values
        init_signature = inspect.signature(obj.__init__)
        init_params = init_signature.parameters

        # Check if this has a 'label' parameter - if so, show it first as positional
        has_label = 'label' in init_params and label_as_positional

        # Build kwargs for non-default parameters
        kwargs_parts = []
        label_value = None

        for param_name, param in init_params.items():
            if param_name in excluded_params:
                continue

            # Skip *args and **kwargs
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            # Handle label separately if showing as positional
            if param_name == 'label' and has_label:
                label_value = getattr(obj, param_name, None)
                continue

            # Get current value
            value = getattr(obj, param_name, None)

            # Skip if value matches default
            if param.default != inspect.Parameter.empty:
                # Special handling for empty containers (even if default was None)
                if isinstance(value, (dict, list, tuple, set)) and len(value) == 0:
                    if param.default is None or (
                        isinstance(param.default, (dict, list, tuple, set)) and len(param.default) == 0
                    ):
                        continue

                # Handle array comparisons (xarray, numpy)
                elif isinstance(value, (xr.DataArray, np.ndarray)):
                    try:
                        if isinstance(param.default, (xr.DataArray, np.ndarray)):
                            # Compare arrays element-wise
                            if isinstance(value, xr.DataArray) and isinstance(param.default, xr.DataArray):
                                if value.equals(param.default):
                                    continue
                            elif np.array_equal(value, param.default):
                                continue
                    except Exception:
                        pass  # If comparison fails, include in repr

                # Handle numeric comparisons (deals with 0 vs 0.0, int vs float)
                elif isinstance(value, (int, float, np.integer, np.floating)) and isinstance(
                    param.default, (int, float, np.integer, np.floating)
                ):
                    try:
                        if float(value) == float(param.default):
                            continue
                    except (ValueError, TypeError):
                        pass

                elif value == param.default:
                    continue

            # Skip None values if default is None
            if value is None and param.default is None:
                continue

            # Special case: hide CONFIG.Modeling.big for size parameter
            if skip_default_size and param_name == 'size':
                from .config import CONFIG

                try:
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        if float(value) == CONFIG.Modeling.big:
                            continue
                except Exception:
                    pass

            # Format value - use numeric formatter for numbers
            if isinstance(
                value, (int, float, np.integer, np.floating, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray)
            ):
                try:
                    value_repr = numeric_to_str_for_repr(value)
                except Exception:
                    value_repr = repr(value)
                    if len(value_repr) > 50:
                        value_repr = value_repr[:47] + '...'

            elif isinstance(value, dict):
                # Format dicts with numeric/array values nicely
                try:
                    formatted_items = []
                    for k, v in value.items():
                        if isinstance(
                            v, (int, float, np.integer, np.floating, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray)
                        ):
                            v_str = numeric_to_str_for_repr(v)
                        else:
                            v_str = repr(v)
                            if len(v_str) > 30:
                                v_str = v_str[:27] + '...'
                        formatted_items.append(f'{repr(k)}: {v_str}')
                    value_repr = '{' + ', '.join(formatted_items) + '}'
                    if len(value_repr) > 50:
                        value_repr = value_repr[:47] + '...'
                except Exception:
                    value_repr = repr(value)
                    if len(value_repr) > 50:
                        value_repr = value_repr[:47] + '...'

            else:
                value_repr = repr(value)
                if len(value_repr) > 50:
                    value_repr = value_repr[:47] + '...'

            kwargs_parts.append(f'{param_name}={value_repr}')

        # Build args string with label first as positional if present
        if has_label and label_value is not None:
            # Use label_full if available, otherwise label
            if hasattr(obj, 'label_full'):
                label_repr = repr(obj.label_full)
            else:
                label_repr = repr(label_value)

            if len(label_repr) > 50:
                label_repr = label_repr[:47] + '...'
            args_str = label_repr
            if kwargs_parts:
                args_str += ', ' + ', '.join(kwargs_parts)
        else:
            args_str = ', '.join(kwargs_parts)

        # Build final repr
        class_name = obj.__class__.__name__
        if info:
            # Remove leading ' | ' if present (from old format) and format as comment
            info_clean = info.lstrip(' |').strip()
            return f'{class_name}({args_str})  # {info_clean}'
        return f'{class_name}({args_str})'

    except Exception:
        # Fallback if introspection fails
        return f'{obj.__class__.__name__}(<repr_failed>)'
