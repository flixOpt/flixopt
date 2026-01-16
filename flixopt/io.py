from __future__ import annotations

import inspect
import json
import logging
import os
import pathlib
import re
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

if TYPE_CHECKING:
    import linopy

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

    # Suppress numpy binary compatibility warnings from netCDF4 (numpy 1->2 transition)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='numpy.ndarray size changed')
        ds.to_netcdf(
            path,
            encoding=None
            if compression == 0
            else {data_var: {'zlib': True, 'complevel': compression} for data_var in ds.data_vars},
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

    for name, da in ds.data_vars.items():
        if not da.dims or da.size == 0:
            new_data_vars[name] = da
            continue

        # Try to reduce each dimension
        reduced = da
        for dim in list(da.dims):
            if dim not in reduced.dims:
                continue  # Already removed
            # Check if constant along this dimension
            first_slice = reduced.isel({dim: 0})
            is_constant = (reduced == first_slice).all()
            if is_constant:
                # Remove this dimension by taking first slice
                reduced = first_slice

        new_data_vars[name] = reduced

    return xr.Dataset(new_data_vars, coords=ds.coords, attrs=ds.attrs)


def load_dataset_from_netcdf(path: str | pathlib.Path) -> xr.Dataset:
    """
    Load a dataset from a netcdf file. Load all attrs from 'attrs' attributes.

    Args:
        path: Path to load the dataset from.

    Returns:
        Dataset: Loaded dataset with restored attrs.
    """
    # Suppress numpy binary compatibility warnings from netCDF4 (numpy 1->2 transition)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='numpy.ndarray size changed')
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
        for flow in obj.inputs:
            flow_lines.append(f'    * {repr(flow)}')

    if has_outputs and hasattr(obj, 'outputs') and obj.outputs:
        flow_lines.append('  outputs:')
        for flow in obj.outputs:
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
