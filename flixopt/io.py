from __future__ import annotations

import inspect
import json
import logging
import os
import pathlib
import re
import sys
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

if TYPE_CHECKING:
    import linopy

    from .flow_system import FlowSystem
    from .types import Numeric_TPS

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
    **kwargs: Any,
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


def _create_compact_dumper():
    """
    Create a YAML dumper class with custom representer for compact numeric lists.

    Returns:
        A yaml.SafeDumper subclass configured to format numeric lists inline.
    """

    def represent_list(dumper, data):
        """
        Custom representer for lists to format them inline (flow style)
        but only if they contain only numbers or nested numeric lists.
        """
        if data and all(
            isinstance(item, (int, float, np.integer, np.floating))
            or (isinstance(item, list) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in item))
            for item in data
        ):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

    # Create custom dumper with the representer
    class CompactDumper(yaml.SafeDumper):
        pass

    CompactDumper.add_representer(list, represent_list)
    return CompactDumper


def save_yaml(
    data: dict | list,
    path: str | pathlib.Path,
    indent: int = 4,
    width: int = 1000,
    allow_unicode: bool = True,
    sort_keys: bool = False,
    compact_numeric_lists: bool = False,
    **kwargs: Any,
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
        compact_numeric_lists: If True, format numeric lists inline for better readability (default: False).
        **kwargs: Additional arguments to pass to yaml.dump().
    """
    path = pathlib.Path(path)

    if compact_numeric_lists:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(
                data,
                f,
                Dumper=_create_compact_dumper(),
                indent=indent,
                width=width,
                allow_unicode=allow_unicode,
                sort_keys=sort_keys,
                default_flow_style=False,
                **kwargs,
            )
    else:
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


def format_yaml_string(
    data: dict | list,
    indent: int = 4,
    width: int = 1000,
    allow_unicode: bool = True,
    sort_keys: bool = False,
    compact_numeric_lists: bool = False,
    **kwargs: Any,
) -> str:
    """
    Format data as a YAML string with consistent formatting.

    This function provides the same formatting as save_yaml() but returns a string
    instead of writing to a file. Useful for logging or displaying YAML data.

    Args:
        data: Data to format (dict or list).
        indent: Number of spaces for indentation (default: 4).
        width: Maximum line width (default: 1000).
        allow_unicode: If True, allow Unicode characters (default: True).
        sort_keys: If True, sort dictionary keys (default: False).
        compact_numeric_lists: If True, format numeric lists inline for better readability (default: False).
        **kwargs: Additional arguments to pass to yaml.dump().

    Returns:
        Formatted YAML string.
    """
    if compact_numeric_lists:
        return yaml.dump(
            data,
            Dumper=_create_compact_dumper(),
            indent=indent,
            width=width,
            allow_unicode=allow_unicode,
            sort_keys=sort_keys,
            default_flow_style=False,
            **kwargs,
        )
    else:
        return yaml.safe_dump(
            data,
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
        logger.warning(f'The model has a warning status {model.status=}. Trying to extract infeasibilities')
        try:
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()

            # Redirect stdout to our buffer
            with redirect_stdout(f):
                model.print_infeasibilities()

            documentation['infeasible_constraints'] = f.getvalue()
        except NotImplementedError:
            logger.warning(
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
    stack_vars: bool = True,
) -> None:
    """
    Save a dataset to a netcdf file. Store all attrs as JSON strings in 'attrs' attributes.

    Args:
        ds: Dataset to save.
        path: Path to save the dataset to.
        compression: Compression level for the dataset (0-9). 0 means no compression. 5 is a good default.
        stack_vars: If True (default), stack variables with equal dims for faster I/O.
            Variables are automatically unstacked when loading with load_dataset_from_netcdf.

    Raises:
        ValueError: If the path has an invalid file extension.
    """
    path = pathlib.Path(path)
    if path.suffix not in ['.nc', '.nc4']:
        raise ValueError(f'Invalid file extension for path {path}. Only .nc and .nc4 are supported')

    ds = ds.copy(deep=True)

    # Stack variables with equal dims for faster I/O
    if stack_vars:
        ds = _stack_equal_vars(ds)

    ds.attrs = {'attrs': json.dumps(ds.attrs)}

    # Convert all DataArray attrs to JSON strings
    # Use ds.variables to avoid slow _construct_dataarray calls
    variables = ds.variables
    coord_names = set(ds.coords)
    for var_name in variables:
        if var_name in coord_names:
            continue
        var = variables[var_name]
        if var.attrs:  # Only if there are attrs
            var.attrs = {'attrs': json.dumps(var.attrs)}

    # Also handle coordinate attrs if they exist
    for coord_name in ds.coords:
        var = variables[coord_name]
        if var.attrs:
            var.attrs = {'attrs': json.dumps(var.attrs)}

    # Suppress numpy binary compatibility warnings from netCDF4 (numpy 1->2 transition)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='numpy.ndarray size changed')
        ds.to_netcdf(
            path,
            encoding=None
            if compression == 0
            else {name: {'zlib': True, 'complevel': compression} for name in variables if name not in coord_names},
            engine='netcdf4',
        )


def _reduce_constant_arrays(ds: xr.Dataset) -> xr.Dataset:
    """
    Reduce constant dimensions in arrays for more efficient storage.

    For each array, checks each dimension and removes it if values are constant
    along that dimension. This handles cases like:
    - Shape (8760,) all identical → scalar
    - Shape (8760, 2) constant along time → shape (2,)
    - Shape (8760, 2, 3) constant along time → shape (2, 3)

    This is useful for datasets saved with older versions where data was
    broadcast to full dimensions.

    Args:
        ds: Dataset with potentially constant arrays.

    Returns:
        Dataset with constant dimensions reduced.
    """
    new_data_vars = {}
    variables = ds.variables
    coord_names = set(ds.coords)

    for name in variables:
        if name in coord_names:
            continue
        var = variables[name]
        dims = var.dims
        data = var.values

        if not dims or data.size == 0:
            new_data_vars[name] = var
            continue

        # Try to reduce each dimension using numpy operations
        reduced_data = data
        reduced_dims = list(dims)

        for _axis, dim in enumerate(dims):
            if dim not in reduced_dims:
                continue  # Already removed

            current_axis = reduced_dims.index(dim)
            # Check if constant along this axis using numpy
            first_slice = np.take(reduced_data, 0, axis=current_axis)
            # Broadcast first_slice to compare
            expanded = np.expand_dims(first_slice, axis=current_axis)
            is_constant = np.allclose(reduced_data, expanded, equal_nan=True)

            if is_constant:
                # Remove this dimension by taking first slice
                reduced_data = first_slice
                reduced_dims.pop(current_axis)

        new_data_vars[name] = xr.Variable(tuple(reduced_dims), reduced_data, attrs=var.attrs)

    return xr.Dataset(new_data_vars, coords=ds.coords, attrs=ds.attrs)


def _stack_equal_vars(ds: xr.Dataset, stacked_dim: str = '__stacked__') -> xr.Dataset:
    """
    Stack data_vars with equal dims into single DataArrays with a stacked dimension.

    This reduces the number of data_vars in a dataset by grouping variables that
    share the same dimensions. Each group is concatenated along a new stacked
    dimension, with the original variable names stored as coordinates.

    This can significantly improve I/O performance for datasets with many
    variables that share the same shape.

    Args:
        ds: Input dataset
        stacked_dim: Base name for the stacking dimensions (default: '__stacked__')

    Returns:
        Dataset with fewer variables (equal-dim vars stacked together).
        Stacked variables are named 'stacked_{dims}' and have a coordinate
        '{stacked_dim}_{dims}' containing the original variable names.
    """
    # Use ds.variables to avoid slow _construct_dataarray calls
    variables = ds.variables
    coord_names = set(ds.coords)

    # Group data variables by their dimensions (preserve insertion order for deterministic stacking)
    groups = defaultdict(list)
    for name in variables:
        if name not in coord_names:
            groups[variables[name].dims].append(name)

    new_data_vars = {}
    for dims, var_names in groups.items():
        if len(var_names) == 1:
            # Single variable - use Variable directly
            new_data_vars[var_names[0]] = variables[var_names[0]]
        else:
            dim_suffix = '_'.join(dims) if dims else 'scalar'
            group_stacked_dim = f'{stacked_dim}_{dim_suffix}'

            # Stack using numpy directly - much faster than xr.concat
            # All variables in this group have the same dims/shape
            arrays = [variables[name].values for name in var_names]
            stacked_data = np.stack(arrays, axis=0)

            # Capture per-variable attrs before stacking
            per_variable_attrs = {name: dict(variables[name].attrs) for name in var_names}

            # Create new Variable with stacked dimension first
            stacked_var = xr.Variable(
                dims=(group_stacked_dim,) + dims,
                data=stacked_data,
                attrs={'__per_variable_attrs__': per_variable_attrs},
            )
            new_data_vars[f'stacked_{dim_suffix}'] = stacked_var

    # Build result dataset preserving coordinates
    result = xr.Dataset(new_data_vars, coords=ds.coords, attrs=ds.attrs)

    # Add the stacking coordinates (variable names)
    for dims, var_names in groups.items():
        if len(var_names) > 1:
            dim_suffix = '_'.join(dims) if dims else 'scalar'
            group_stacked_dim = f'{stacked_dim}_{dim_suffix}'
            result = result.assign_coords({group_stacked_dim: var_names})

    return result


def _unstack_vars(ds: xr.Dataset, stacked_prefix: str = '__stacked__') -> xr.Dataset:
    """
    Reverse of _stack_equal_vars - unstack back to individual variables.

    Args:
        ds: Dataset with stacked variables (from _stack_equal_vars)
        stacked_prefix: Prefix used for stacking dimensions (default: '__stacked__')

    Returns:
        Dataset with individual variables restored from stacked arrays.
    """
    new_data_vars = {}
    variables = ds.variables
    coord_names = set(ds.coords)

    for name in variables:
        if name in coord_names:
            continue
        var = variables[name]
        # Find stacked dimension (if any)
        stacked_dim = None
        stacked_dim_idx = None
        for i, d in enumerate(var.dims):
            if d.startswith(stacked_prefix):
                stacked_dim = d
                stacked_dim_idx = i
                break

        if stacked_dim is not None:
            # Get labels from the stacked coordinate
            labels = ds.coords[stacked_dim].values
            # Get remaining dims (everything except stacked dim)
            remaining_dims = var.dims[:stacked_dim_idx] + var.dims[stacked_dim_idx + 1 :]
            # Get per-variable attrs if available
            per_variable_attrs = var.attrs.get('__per_variable_attrs__', {})
            # Extract each slice using numpy indexing (much faster than .sel())
            data = var.values
            for idx, label in enumerate(labels):
                # Use numpy indexing to get the slice
                sliced_data = np.take(data, idx, axis=stacked_dim_idx)
                # Restore original attrs if available
                restored_attrs = per_variable_attrs.get(str(label), {})
                new_data_vars[str(label)] = xr.Variable(remaining_dims, sliced_data, attrs=restored_attrs)
        else:
            new_data_vars[name] = var

    # Preserve non-dimension coordinates (filter out stacked dim coords)
    preserved_coords = {k: v for k, v in ds.coords.items() if not k.startswith(stacked_prefix)}
    return xr.Dataset(new_data_vars, coords=preserved_coords, attrs=ds.attrs)


def load_dataset_from_netcdf(path: str | pathlib.Path) -> xr.Dataset:
    """
    Load a dataset from a netcdf file. Load all attrs from 'attrs' attributes.

    Automatically unstacks variables that were stacked during saving with
    save_dataset_to_netcdf(stack_vars=True).

    Args:
        path: Path to load the dataset from.

    Returns:
        Dataset: Loaded dataset with restored attrs and unstacked variables.
    """
    # Suppress numpy binary compatibility warnings from netCDF4 (numpy 1->2 transition)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='numpy.ndarray size changed')
        ds = xr.load_dataset(str(path), engine='netcdf4')

    # Restore Dataset attrs
    if 'attrs' in ds.attrs:
        ds.attrs = json.loads(ds.attrs['attrs'])

    # Restore DataArray attrs (before unstacking, as stacked vars have no individual attrs)
    # Use ds.variables to avoid slow _construct_dataarray calls
    variables = ds.variables
    for var_name in variables:
        var = variables[var_name]
        if 'attrs' in var.attrs:
            var.attrs = json.loads(var.attrs['attrs'])

    # Unstack variables if they were stacked during saving
    # Detection: check if any dataset dimension starts with '__stacked__'
    if any(dim.startswith('__stacked__') for dim in ds.dims):
        ds = _unstack_vars(ds)

    return ds


# Parameter rename mappings for backwards compatibility conversion
# Format: {old_name: new_name}
PARAMETER_RENAMES = {
    # Effect parameters
    'minimum_operation': 'minimum_temporal',
    'maximum_operation': 'maximum_temporal',
    'minimum_invest': 'minimum_periodic',
    'maximum_invest': 'maximum_periodic',
    'minimum_investment': 'minimum_periodic',
    'maximum_investment': 'maximum_periodic',
    'minimum_operation_per_hour': 'minimum_per_hour',
    'maximum_operation_per_hour': 'maximum_per_hour',
    # InvestParameters
    'fix_effects': 'effects_of_investment',
    'specific_effects': 'effects_of_investment_per_size',
    'divest_effects': 'effects_of_retirement',
    'piecewise_effects': 'piecewise_effects_of_investment',
    # Flow/OnOffParameters
    'flow_hours_total_max': 'flow_hours_max',
    'flow_hours_total_min': 'flow_hours_min',
    'on_hours_total_max': 'on_hours_max',
    'on_hours_total_min': 'on_hours_min',
    'switch_on_total_max': 'switch_on_max',
    # Bus
    'excess_penalty_per_flow_hour': 'imbalance_penalty_per_flow_hour',
    # Component parameters (Source/Sink)
    'source': 'outputs',
    'sink': 'inputs',
    'prevent_simultaneous_sink_and_source': 'prevent_simultaneous_flow_rates',
    # LinearConverter flow/efficiency parameters (pre-v4 files)
    # These are needed for very old files that use short flow names
    'Q_fu': 'fuel_flow',
    'P_el': 'electrical_flow',
    'Q_th': 'thermal_flow',
    'Q_ab': 'heat_source_flow',
    'eta': 'thermal_efficiency',
    'eta_th': 'thermal_efficiency',
    'eta_el': 'electrical_efficiency',
    'COP': 'cop',
    # Storage
    # Note: 'lastValueOfSim' → 'equals_final' is a value change, not a key change
    # Class renames (v4.2.0)
    'FullCalculation': 'Optimization',
    'AggregatedCalculation': 'ClusteredOptimization',
    'SegmentedCalculation': 'SegmentedOptimization',
    'CalculationResults': 'Results',
    'SegmentedCalculationResults': 'SegmentedResults',
    'Aggregation': 'Clustering',
    'AggregationParameters': 'ClusteringParameters',
    'AggregationModel': 'ClusteringModel',
    # OnOffParameters → StatusParameters (class and attribute names)
    'OnOffParameters': 'StatusParameters',
    'on_off_parameters': 'status_parameters',
    # StatusParameters attribute renames (applies to both Flow-level and Component-level)
    'effects_per_switch_on': 'effects_per_startup',
    'effects_per_running_hour': 'effects_per_active_hour',
    'consecutive_on_hours_min': 'min_uptime',
    'consecutive_on_hours_max': 'max_uptime',
    'consecutive_off_hours_min': 'min_downtime',
    'consecutive_off_hours_max': 'max_downtime',
    'force_switch_on': 'force_startup_tracking',
    'on_hours_min': 'active_hours_min',
    'on_hours_max': 'active_hours_max',
    'switch_on_max': 'startup_limit',
    # TimeSeriesData
    'agg_group': 'aggregation_group',
    'agg_weight': 'aggregation_weight',
}

# Value renames (for specific parameter values that changed)
VALUE_RENAMES = {
    'initial_charge_state': {'lastValueOfSim': 'equals_final'},
}


# Keys that should NOT have their child keys renamed (they reference flow labels)
_FLOW_LABEL_REFERENCE_KEYS = {'piecewises', 'conversion_factors'}

# Keys that ARE flow parameters on components (should be renamed)
_FLOW_PARAMETER_KEYS = {'Q_fu', 'P_el', 'Q_th', 'Q_ab', 'eta', 'eta_th', 'eta_el', 'COP'}


def _rename_keys_recursive(
    obj: Any,
    key_renames: dict[str, str],
    value_renames: dict[str, dict],
    skip_flow_renames: bool = False,
) -> Any:
    """Recursively rename keys and values in nested data structures.

    Args:
        obj: The object to process (dict, list, or scalar)
        key_renames: Mapping of old key names to new key names
        value_renames: Mapping of key names to {old_value: new_value} dicts
        skip_flow_renames: If True, skip renaming flow parameter keys (for inside piecewises)

    Returns:
        The processed object with renamed keys and values
    """
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            # Determine if we should skip flow renames for children
            child_skip_flow_renames = skip_flow_renames or key in _FLOW_LABEL_REFERENCE_KEYS

            # Rename the key if needed (skip flow params if in reference context)
            if skip_flow_renames and key in _FLOW_PARAMETER_KEYS:
                new_key = key  # Don't rename flow labels inside piecewises etc.
            else:
                new_key = key_renames.get(key, key)

            # Process the value recursively
            new_value = _rename_keys_recursive(value, key_renames, value_renames, child_skip_flow_renames)

            # Check if this key has value renames (lookup by renamed key, fallback to old key)
            vr_key = new_key if new_key in value_renames else key
            if vr_key in value_renames and isinstance(new_value, str):
                new_value = value_renames[vr_key].get(new_value, new_value)

            # Handle __class__ values - rename class names
            if key == '__class__' and isinstance(new_value, str):
                new_value = key_renames.get(new_value, new_value)

            new_dict[new_key] = new_value
        return new_dict

    elif isinstance(obj, list):
        return [_rename_keys_recursive(item, key_renames, value_renames, skip_flow_renames) for item in obj]

    else:
        return obj


def convert_old_dataset(
    ds: xr.Dataset,
    key_renames: dict[str, str] | None = None,
    value_renames: dict[str, dict] | None = None,
    reduce_constants: bool = True,
) -> xr.Dataset:
    """Convert an old FlowSystem dataset to the current format.

    This function performs two conversions:
    1. Renames parameters in the reference structure to current naming conventions
    2. Reduces constant arrays to minimal dimensions (e.g., broadcasted scalars back to scalars)

    This is useful for loading FlowSystem files saved with older versions of flixopt.

    Args:
        ds: The dataset to convert
        key_renames: Custom key renames to apply. If None, uses PARAMETER_RENAMES.
        value_renames: Custom value renames to apply. If None, uses VALUE_RENAMES.
        reduce_constants: If True (default), reduce constant arrays to minimal dimensions.
            Old files may have scalars broadcasted to full (time, period, scenario) shape.

    Returns:
        The converted dataset

    Examples:
        Convert an old netCDF file to new format:

        ```python
        from flixopt import io

        # Load old file
        ds = io.load_dataset_from_netcdf('old_flow_system.nc4')

        # Convert to current format
        ds = io.convert_old_dataset(ds)

        # Now load as FlowSystem
        from flixopt import FlowSystem

        fs = FlowSystem.from_dataset(ds)
        ```
    """
    if key_renames is None:
        key_renames = PARAMETER_RENAMES
    if value_renames is None:
        value_renames = VALUE_RENAMES

    # Convert the attrs (reference_structure)
    ds.attrs = _rename_keys_recursive(ds.attrs, key_renames, value_renames)

    # Reduce constant arrays to minimal dimensions
    if reduce_constants:
        ds = _reduce_constant_arrays(ds)

    return ds


def convert_old_netcdf(
    input_path: str | pathlib.Path,
    output_path: str | pathlib.Path | None = None,
    compression: int = 0,
) -> xr.Dataset:
    """Load an old FlowSystem netCDF file and convert to new parameter names.

    This is a convenience function that combines loading, conversion, and
    optionally saving the converted dataset.

    Args:
        input_path: Path to the old netCDF file
        output_path: If provided, save the converted dataset to this path.
            If None, only returns the converted dataset without saving.
        compression: Compression level (0-9) for saving. Only used if output_path is provided.

    Returns:
        The converted dataset

    Examples:
        Convert and save to new file:

        ```python
        from flixopt import io

        # Convert old file to new format
        ds = io.convert_old_netcdf('old_system.nc4', 'new_system.nc')
        ```

        Convert and load as FlowSystem:

        ```python
        from flixopt import FlowSystem, io

        ds = io.convert_old_netcdf('old_system.nc4')
        fs = FlowSystem.from_dataset(ds)
        ```
    """
    # Load and convert
    ds = load_dataset_from_netcdf(input_path)
    ds = convert_old_dataset(ds)

    # Optionally save
    if output_path is not None:
        save_dataset_to_netcdf(ds, output_path, compression=compression)
        logger.info(f'Converted {input_path} -> {output_path}')

    return ds


@dataclass
class ResultsPaths:
    """Container for all paths related to saving Results."""

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

    def create_folders(self, parents: bool = False, exist_ok: bool = True) -> None:
        """Ensure the folder exists.

        Args:
            parents: If True, create parent directories as needed. If False, parent must exist.
            exist_ok: If True, do not raise error if folder already exists. If False, raise FileExistsError.

        Raises:
            FileNotFoundError: If parents=False and parent directory doesn't exist.
            FileExistsError: If exist_ok=False and folder already exists.
        """
        try:
            self.folder.mkdir(parents=parents, exist_ok=exist_ok)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f'Cannot create folder {self.folder}: parent directory does not exist. '
                f'Use parents=True to create parent directories.'
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
    value: Numeric_TPS,
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

    # Normalize dtype and handle empties
    arr = arr.astype(float, copy=False)
    if arr.size == 0:
        return '?'

    # Filter non-finite values
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 'nan'

    # Check for single value
    if finite.size == 1:
        return f'{float(finite[0]):.{precision}f}'

    # Check if all values are the same or very close
    min_val = float(np.nanmin(finite))
    max_val = float(np.nanmax(finite))

    # First check: values are essentially identical
    if np.allclose(min_val, max_val, atol=atol):
        return f'{float(np.mean(finite)):.{precision}f}'

    # Second check: display values are the same but actual values differ slightly
    min_str = f'{min_val:.{precision}f}'
    max_str = f'{max_val:.{precision}f}'
    if min_str == max_str:
        return f'~{min_str}'

    # Values vary significantly - show range
    return f'{min_str}-{max_str}'


def _format_value_for_repr(value) -> str:
    """Format a single value for display in repr.

    Args:
        value: The value to format

    Returns:
        Formatted string representation of the value
    """
    # Format numeric types using specialized formatter
    if isinstance(value, (int, float, np.integer, np.floating, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray)):
        try:
            return numeric_to_str_for_repr(value)
        except Exception:
            value_repr = repr(value)
            if len(value_repr) > 50:
                value_repr = value_repr[:47] + '...'
            return value_repr

    # Format dicts with numeric/array values nicely
    elif isinstance(value, dict):
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
            return value_repr
        except Exception:
            value_repr = repr(value)
            if len(value_repr) > 50:
                value_repr = value_repr[:47] + '...'
            return value_repr

    # Default repr with truncation
    else:
        value_repr = repr(value)
        if len(value_repr) > 50:
            value_repr = value_repr[:47] + '...'
        return value_repr


def build_repr_from_init(
    obj: object,
    excluded_params: set[str] | None = None,
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
        label_as_positional: If True and 'label' param exists, show it as first positional arg
        skip_default_size: Deprecated. Previously skipped size=CONFIG.Modeling.big, now size=None is default.

    Returns:
        Formatted repr string like: ClassName("label", param=value)
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
            # Skip *args and **kwargs
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            # Handle label separately if showing as positional (check BEFORE excluded_params)
            if param_name == 'label' and has_label:
                label_value = getattr(obj, param_name, None)
                continue

            # Now check if parameter should be excluded
            if param_name in excluded_params:
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
                        elif isinstance(param.default, (int, float, np.integer, np.floating)):
                            # Compare array to scalar (e.g., after transform_data converts scalar to DataArray)
                            if isinstance(value, xr.DataArray):
                                if np.all(value.values == float(param.default)):
                                    continue
                            elif isinstance(value, np.ndarray):
                                if np.all(value == float(param.default)):
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

            # Format value using helper function
            value_repr = _format_value_for_repr(value)
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

        return f'{class_name}({args_str})'

    except Exception:
        # Fallback if introspection fails
        return f'{obj.__class__.__name__}(<repr_failed>)'


def format_flow_details(obj: Any, has_inputs: bool = True, has_outputs: bool = True) -> str:
    """Format inputs and outputs as indented bullet list.

    Args:
        obj: Object with 'inputs' and/or 'outputs' attributes
        has_inputs: Whether to check for inputs
        has_outputs: Whether to check for outputs

    Returns:
        Formatted string with flow details (including leading newline), or empty string if no flows
    """
    flow_lines = []

    if has_inputs and hasattr(obj, 'inputs') and obj.inputs:
        flow_lines.append('  inputs:')
        for flow in obj.inputs.values():
            flow_lines.append(f'    * {repr(flow)}')

    if has_outputs and hasattr(obj, 'outputs') and obj.outputs:
        flow_lines.append('  outputs:')
        for flow in obj.outputs.values():
            flow_lines.append(f'    * {repr(flow)}')

    return '\n' + '\n'.join(flow_lines) if flow_lines else ''


def format_title_with_underline(title: str, underline_char: str = '-') -> str:
    """Format a title with underline of matching length.

    Args:
        title: The title text
        underline_char: Character to use for underline (default: '-')

    Returns:
        Formatted string: "Title\\n-----\\n"
    """
    return f'{title}\n{underline_char * len(title)}\n'


def format_sections_with_headers(sections: dict[str, str], underline_char: str = '-') -> list[str]:
    """Format sections with underlined headers.

    Args:
        sections: Dict mapping section headers to content
        underline_char: Character for underlining headers

    Returns:
        List of formatted section strings
    """
    formatted_sections = []
    for section_header, section_content in sections.items():
        underline = underline_char * len(section_header)
        formatted_sections.append(f'{section_header}\n{underline}\n{section_content}')
    return formatted_sections


def build_metadata_info(parts: list[str], prefix: str = ' | ') -> str:
    """Build metadata info string from parts.

    Args:
        parts: List of metadata strings (empty strings are filtered out)
        prefix: Prefix to add if parts is non-empty

    Returns:
        Formatted info string or empty string
    """
    # Filter out empty strings
    parts = [p for p in parts if p]
    if not parts:
        return ''
    info = ' | '.join(parts)
    return prefix + info if prefix else info


@contextmanager
def suppress_output():
    """
    Suppress all console output including C-level output from solvers.

    WARNING: Not thread-safe. Modifies global file descriptors.
    Use only with sequential execution or multiprocessing.
    """
    # Save original file descriptors
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    devnull_fd = None

    try:
        # Open devnull
        devnull_fd = os.open(os.devnull, os.O_WRONLY)

        # Flush Python buffers before redirecting
        sys.stdout.flush()
        sys.stderr.flush()

        # Redirect file descriptors to devnull
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)

        yield

    finally:
        # Restore original file descriptors with nested try blocks
        # to ensure all cleanup happens even if one step fails
        try:
            # Flush any buffered output in the redirected streams
            sys.stdout.flush()
            sys.stderr.flush()
        except (OSError, ValueError):
            pass  # Stream might be closed or invalid

        try:
            os.dup2(old_stdout_fd, 1)
        except OSError:
            pass  # Failed to restore stdout, continue cleanup

        try:
            os.dup2(old_stderr_fd, 2)
        except OSError:
            pass  # Failed to restore stderr, continue cleanup

        # Close all file descriptors
        for fd in [devnull_fd, old_stdout_fd, old_stderr_fd]:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass  # FD already closed or invalid


# ============================================================================
# FlowSystem Dataset I/O
# ============================================================================


class FlowSystemDatasetIO:
    """Unified I/O handler for FlowSystem dataset serialization and deserialization.

    This class provides optimized methods for converting FlowSystem objects to/from
    xarray Datasets. It uses shared constants for variable prefixes and implements
    fast DataArray construction to avoid xarray's slow _construct_dataarray method.

    Constants:
        SOLUTION_PREFIX: Prefix for solution variables ('solution|')
        CLUSTERING_PREFIX: Prefix for clustering variables ('clustering|')

    Example:
        # Serialization (FlowSystem -> Dataset)
        ds = FlowSystemDatasetIO.to_dataset(flow_system, base_ds)

        # Deserialization (Dataset -> FlowSystem)
        fs = FlowSystemDatasetIO.from_dataset(ds)
    """

    # Shared prefixes for variable namespacing
    SOLUTION_PREFIX = 'solution|'
    CLUSTERING_PREFIX = 'clustering|'

    # --- Deserialization (Dataset -> FlowSystem) ---

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> FlowSystem:
        """Create FlowSystem from dataset.

        This is the main entry point for dataset restoration.
        Called by FlowSystem.from_dataset().

        If the dataset contains solution data (variables prefixed with 'solution|'),
        the solution will be restored to the FlowSystem. Solution time coordinates
        are renamed back from 'solution_time' to 'time'.

        Supports clustered datasets with (cluster, time) dimensions. When detected,
        creates a synthetic DatetimeIndex for compatibility and stores the clustered
        data structure for later use.

        Args:
            ds: Dataset containing the FlowSystem data

        Returns:
            FlowSystem instance with all components, buses, effects, and solution restored
        """
        from .flow_system import FlowSystem

        # Parse dataset structure
        reference_structure = dict(ds.attrs)
        solution_var_names, config_var_names = cls._separate_variables(ds)
        coord_cache = {k: ds.coords[k] for k in ds.coords}
        arrays_dict = {name: cls._fast_get_dataarray(ds, name, coord_cache) for name in config_var_names}

        # Create and populate FlowSystem
        flow_system = cls._create_flow_system(ds, reference_structure, arrays_dict, FlowSystem)
        cls._restore_elements(flow_system, reference_structure, arrays_dict, FlowSystem)
        cls._restore_solution(flow_system, ds, reference_structure, solution_var_names)
        cls._restore_clustering(flow_system, ds, reference_structure, config_var_names, arrays_dict, FlowSystem)
        cls._restore_metadata(flow_system, reference_structure, FlowSystem)
        flow_system.connect_and_transform()
        return flow_system

    @classmethod
    def _separate_variables(cls, ds: xr.Dataset) -> tuple[dict[str, str], list[str]]:
        """Separate solution variables from config variables.

        Args:
            ds: Source dataset

        Returns:
            Tuple of (solution_var_names dict, config_var_names list)
        """
        solution_var_names: dict[str, str] = {}  # Maps original_name -> ds_name
        config_var_names: list[str] = []
        coord_names = set(ds.coords)

        for name in ds.variables:
            if name in coord_names:
                continue
            if name.startswith(cls.SOLUTION_PREFIX):
                solution_var_names[name[len(cls.SOLUTION_PREFIX) :]] = name
            else:
                config_var_names.append(name)

        return solution_var_names, config_var_names

    @staticmethod
    def _fast_get_dataarray(ds: xr.Dataset, name: str, coord_cache: dict[str, xr.DataArray]) -> xr.DataArray:
        """Construct DataArray from Variable without slow coordinate inference.

        This bypasses the slow _construct_dataarray method (~1.5ms -> ~0.1ms per var).

        Args:
            ds: Source dataset
            name: Variable name
            coord_cache: Pre-cached coordinate DataArrays

        Returns:
            Constructed DataArray
        """
        variable = ds.variables[name]
        var_dims = set(variable.dims)
        # Include coordinates whose dims are a subset of the variable's dims
        # This preserves both dimension coordinates and auxiliary coordinates
        coords = {k: v for k, v in coord_cache.items() if set(v.dims).issubset(var_dims)}
        return xr.DataArray(variable, coords=coords, name=name)

    @staticmethod
    def _create_flow_system(
        ds: xr.Dataset,
        reference_structure: dict[str, Any],
        arrays_dict: dict[str, xr.DataArray],
        cls: type[FlowSystem],
    ) -> FlowSystem:
        """Create FlowSystem instance with constructor parameters."""
        # Extract cluster index if present (clustered FlowSystem)
        clusters = ds.indexes.get('cluster')

        # For clustered datasets, cluster_weight is (cluster,) shaped - set separately
        if clusters is not None:
            cluster_weight_for_constructor = None
        else:
            cluster_weight_for_constructor = (
                cls._resolve_dataarray_reference(reference_structure['cluster_weight'], arrays_dict)
                if 'cluster_weight' in reference_structure
                else None
            )

        # Resolve scenario_weights only if scenario dimension exists
        scenario_weights = None
        if ds.indexes.get('scenario') is not None and 'scenario_weights' in reference_structure:
            scenario_weights = cls._resolve_dataarray_reference(reference_structure['scenario_weights'], arrays_dict)

        # Resolve timestep_duration if present
        # For segmented systems, it's stored as a data_var; for others it's computed from timesteps_extra
        timestep_duration = None
        if 'timestep_duration' in arrays_dict:
            # Segmented systems store timestep_duration as a data_var
            timestep_duration = arrays_dict['timestep_duration']
        elif 'timestep_duration' in reference_structure:
            ref_value = reference_structure['timestep_duration']
            if isinstance(ref_value, str) and ref_value.startswith(':::'):
                timestep_duration = cls._resolve_dataarray_reference(ref_value, arrays_dict)
            else:
                # Concrete value (e.g., list from expand())
                timestep_duration = ref_value

        # Get timesteps - convert integer index to RangeIndex for segmented systems
        time_index = ds.indexes['time']
        if not isinstance(time_index, pd.DatetimeIndex):
            time_index = pd.RangeIndex(len(time_index), name='time')

        return cls(
            timesteps=time_index,
            periods=ds.indexes.get('period'),
            scenarios=ds.indexes.get('scenario'),
            clusters=clusters,
            hours_of_last_timestep=reference_structure.get('hours_of_last_timestep'),
            hours_of_previous_timesteps=reference_structure.get('hours_of_previous_timesteps'),
            weight_of_last_period=reference_structure.get('weight_of_last_period'),
            scenario_weights=scenario_weights,
            cluster_weight=cluster_weight_for_constructor,
            scenario_independent_sizes=reference_structure.get('scenario_independent_sizes', True),
            scenario_independent_flow_rates=reference_structure.get('scenario_independent_flow_rates', False),
            name=reference_structure.get('name'),
            timestep_duration=timestep_duration,
        )

    @staticmethod
    def _restore_elements(
        flow_system: FlowSystem,
        reference_structure: dict[str, Any],
        arrays_dict: dict[str, xr.DataArray],
        cls: type[FlowSystem],
    ) -> None:
        """Restore components, buses, and effects to FlowSystem."""
        from .effects import Effect
        from .elements import Bus, Component

        # Restore components
        for comp_label, comp_data in reference_structure.get('components', {}).items():
            component = cls._resolve_reference_structure(comp_data, arrays_dict)
            if not isinstance(component, Component):
                logger.critical(f'Restoring component {comp_label} failed.')
            flow_system._add_components(component)

        # Restore buses
        for bus_label, bus_data in reference_structure.get('buses', {}).items():
            bus = cls._resolve_reference_structure(bus_data, arrays_dict)
            if not isinstance(bus, Bus):
                logger.critical(f'Restoring bus {bus_label} failed.')
            flow_system._add_buses(bus)

        # Restore effects
        for effect_label, effect_data in reference_structure.get('effects', {}).items():
            effect = cls._resolve_reference_structure(effect_data, arrays_dict)
            if not isinstance(effect, Effect):
                logger.critical(f'Restoring effect {effect_label} failed.')
            flow_system._add_effects(effect)

    @classmethod
    def _restore_solution(
        cls,
        flow_system: FlowSystem,
        ds: xr.Dataset,
        reference_structure: dict[str, Any],
        solution_var_names: dict[str, str],
    ) -> None:
        """Restore solution dataset if present."""
        if not reference_structure.get('has_solution', False) or not solution_var_names:
            return

        # Use dataset subsetting (faster than individual ds[name] access)
        solution_ds_names = list(solution_var_names.values())
        solution_ds = ds[solution_ds_names]
        # Rename variables to remove 'solution|' prefix
        rename_map = {ds_name: orig_name for orig_name, ds_name in solution_var_names.items()}
        solution_ds = solution_ds.rename(rename_map)
        # Rename 'solution_time' back to 'time' if present
        if 'solution_time' in solution_ds.dims:
            solution_ds = solution_ds.rename({'solution_time': 'time'})
        flow_system.solution = solution_ds

    @classmethod
    def _restore_clustering(
        cls,
        flow_system: FlowSystem,
        ds: xr.Dataset,
        reference_structure: dict[str, Any],
        config_var_names: list[str],
        arrays_dict: dict[str, xr.DataArray],
        fs_cls: type[FlowSystem],
    ) -> None:
        """Restore Clustering object if present."""
        if 'clustering' not in reference_structure:
            return

        clustering_structure = json.loads(reference_structure['clustering'])

        # Collect clustering arrays (prefixed with 'clustering|')
        clustering_arrays: dict[str, xr.DataArray] = {}
        main_var_names: list[str] = []

        for name in config_var_names:
            if name.startswith(cls.CLUSTERING_PREFIX):
                arr = ds[name]
                arr_name = name[len(cls.CLUSTERING_PREFIX) :]
                clustering_arrays[arr_name] = arr.rename(arr_name)
            else:
                main_var_names.append(name)

        clustering = fs_cls._resolve_reference_structure(clustering_structure, clustering_arrays)
        flow_system.clustering = clustering

        # Reconstruct aggregated_data from FlowSystem's main data arrays
        if clustering.aggregated_data is None and main_var_names:
            from .core import drop_constant_arrays

            main_vars = {name: arrays_dict[name] for name in main_var_names}
            clustering.aggregated_data = drop_constant_arrays(xr.Dataset(main_vars), dim='time')

        # Restore cluster_weight from clustering's representative_weights
        if hasattr(clustering, 'representative_weights'):
            flow_system.cluster_weight = clustering.representative_weights

    @staticmethod
    def _restore_metadata(
        flow_system: FlowSystem,
        reference_structure: dict[str, Any],
        cls: type[FlowSystem],
    ) -> None:
        """Restore carriers and variable categories."""
        from .structure import VariableCategory

        # Restore carriers if present
        if 'carriers' in reference_structure:
            carriers_structure = json.loads(reference_structure['carriers'])
            for carrier_data in carriers_structure.values():
                carrier = cls._resolve_reference_structure(carrier_data, {})
                flow_system._carriers.add(carrier)

        # Restore variable categories if present
        if 'variable_categories' in reference_structure:
            categories_dict = json.loads(reference_structure['variable_categories'])
            restored_categories: dict[str, VariableCategory] = {}
            for name, value in categories_dict.items():
                try:
                    restored_categories[name] = VariableCategory(value)
                except ValueError:
                    logger.warning(f'Unknown VariableCategory value "{value}" for "{name}", skipping')
            flow_system._variable_categories = restored_categories

    # --- Serialization (FlowSystem -> Dataset) ---

    @classmethod
    def to_dataset(
        cls,
        flow_system: FlowSystem,
        base_dataset: xr.Dataset,
        include_solution: bool = True,
        include_original_data: bool = True,
    ) -> xr.Dataset:
        """Convert FlowSystem-specific data to dataset.

        This function adds FlowSystem-specific data (solution, clustering, metadata)
        to a base dataset created by the parent class's to_dataset() method.

        Args:
            flow_system: The FlowSystem to serialize
            base_dataset: Dataset from parent class with basic structure
            include_solution: Whether to include optimization solution
            include_original_data: Whether to include clustering.original_data

        Returns:
            Complete dataset with all FlowSystem data
        """
        from . import __version__

        ds = base_dataset

        # Add solution data
        ds = cls._add_solution_to_dataset(ds, flow_system.solution, include_solution)

        # Add carriers
        ds = cls._add_carriers_to_dataset(ds, flow_system._carriers)

        # Add clustering
        ds = cls._add_clustering_to_dataset(ds, flow_system.clustering, include_original_data)

        # Add variable categories
        ds = cls._add_variable_categories_to_dataset(ds, flow_system._variable_categories)

        # Add version info
        ds.attrs['flixopt_version'] = __version__

        # Ensure model coordinates are present
        ds = cls._add_model_coords(ds, flow_system)

        return ds

    @classmethod
    def _add_solution_to_dataset(
        cls,
        ds: xr.Dataset,
        solution: xr.Dataset | None,
        include_solution: bool,
    ) -> xr.Dataset:
        """Add solution variables to dataset.

        Uses ds.variables directly for fast serialization (avoids _construct_dataarray).
        """
        if include_solution and solution is not None:
            # Rename 'time' to 'solution_time' to preserve full solution
            solution_renamed = solution.rename({'time': 'solution_time'}) if 'time' in solution.dims else solution

            # Use ds.variables directly to avoid slow _construct_dataarray calls
            # Only include data variables (not coordinates)
            data_var_names = set(solution_renamed.data_vars)
            solution_vars = {
                f'{cls.SOLUTION_PREFIX}{name}': var
                for name, var in solution_renamed.variables.items()
                if name in data_var_names
            }
            ds = ds.assign(solution_vars)

            # Add solution_time coordinate if it exists
            if 'solution_time' in solution_renamed.coords:
                ds = ds.assign_coords(solution_time=solution_renamed.coords['solution_time'])

            ds.attrs['has_solution'] = True
        else:
            ds.attrs['has_solution'] = False

        return ds

    @staticmethod
    def _add_carriers_to_dataset(ds: xr.Dataset, carriers: Any) -> xr.Dataset:
        """Add carrier definitions to dataset attributes."""
        if carriers:
            carriers_structure = {}
            for name, carrier in carriers.items():
                carrier_ref, _ = carrier._create_reference_structure()
                carriers_structure[name] = carrier_ref
            ds.attrs['carriers'] = json.dumps(carriers_structure)

        return ds

    @classmethod
    def _add_clustering_to_dataset(
        cls,
        ds: xr.Dataset,
        clustering: Any,
        include_original_data: bool,
    ) -> xr.Dataset:
        """Add clustering object to dataset."""
        if clustering is not None:
            clustering_ref, clustering_arrays = clustering._create_reference_structure(
                include_original_data=include_original_data
            )
            # Add clustering arrays with prefix using batch assignment
            # (individual ds[name] = arr assignments are slow)
            prefixed_arrays = {f'{cls.CLUSTERING_PREFIX}{name}': arr for name, arr in clustering_arrays.items()}
            ds = ds.assign(prefixed_arrays)
            ds.attrs['clustering'] = json.dumps(clustering_ref)

        return ds

    @staticmethod
    def _add_variable_categories_to_dataset(
        ds: xr.Dataset,
        variable_categories: dict,
    ) -> xr.Dataset:
        """Add variable categories to dataset attributes."""
        if variable_categories:
            categories_dict = {name: cat.value for name, cat in variable_categories.items()}
            ds.attrs['variable_categories'] = json.dumps(categories_dict)

        return ds

    @staticmethod
    def _add_model_coords(ds: xr.Dataset, flow_system: FlowSystem) -> xr.Dataset:
        """Ensure model coordinates are present in dataset."""
        model_coords = {'time': flow_system.timesteps}
        if flow_system.periods is not None:
            model_coords['period'] = flow_system.periods
        if flow_system.scenarios is not None:
            model_coords['scenario'] = flow_system.scenarios
        if flow_system.clusters is not None:
            model_coords['cluster'] = flow_system.clusters

        return ds.assign_coords(model_coords)


# =============================================================================
# Public API Functions (delegate to FlowSystemDatasetIO class)
# =============================================================================


def restore_flow_system_from_dataset(ds: xr.Dataset) -> FlowSystem:
    """Create FlowSystem from dataset.

    This is the main entry point for dataset restoration.
    Called by FlowSystem.from_dataset().

    Args:
        ds: Dataset containing the FlowSystem data

    Returns:
        FlowSystem instance with all components, buses, effects, and solution restored

    See Also:
        FlowSystemDatasetIO: Class containing the implementation
    """
    return FlowSystemDatasetIO.from_dataset(ds)


def flow_system_to_dataset(
    flow_system: FlowSystem,
    base_dataset: xr.Dataset,
    include_solution: bool = True,
    include_original_data: bool = True,
) -> xr.Dataset:
    """Convert FlowSystem-specific data to dataset.

    This function adds FlowSystem-specific data (solution, clustering, metadata)
    to a base dataset created by the parent class's to_dataset() method.

    Args:
        flow_system: The FlowSystem to serialize
        base_dataset: Dataset from parent class with basic structure
        include_solution: Whether to include optimization solution
        include_original_data: Whether to include clustering.original_data

    Returns:
        Complete dataset with all FlowSystem data

    See Also:
        FlowSystemDatasetIO: Class containing the implementation
    """
    return FlowSystemDatasetIO.to_dataset(flow_system, base_dataset, include_solution, include_original_data)
