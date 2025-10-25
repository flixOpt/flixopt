from __future__ import annotations

import importlib.util
import json
import logging
import pathlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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


def load_yaml(path: str | pathlib.Path, safe: bool = True) -> dict | list:
    """
    Load data from a YAML file.

    Args:
        path: Path to the YAML file.
        safe: If True, use safe_load for security (default: True).
            If False, use FullLoader (allows arbitrary Python objects - SECURITY RISK).
            Only use safe=False for trusted, internally-generated files.

    Returns:
        Loaded data (typically dict or list), or empty dict if file is empty.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        Note: Returns {} for empty YAML files instead of None.
    """
    path = pathlib.Path(path)
    with open(path, encoding='utf-8') as f:
        if safe:
            return yaml.safe_load(f) or {}
        else:
            return yaml.load(f, Loader=yaml.FullLoader) or {}


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
        **kwargs: Additional arguments to pass to yaml.dump().
    """
    path = pathlib.Path(path)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(
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
    if path.suffix == '.json':
        try:
            return load_json(path)
        except json.JSONDecodeError:
            logger.warning(f'Failed to parse {path} as JSON, trying YAML')
            try:
                return load_yaml(path, safe=True)
            except yaml.YAMLError as e:
                raise ValueError(f'Failed to parse {path} as JSON or YAML') from e

    elif path.suffix in ['.yaml', '.yml']:
        try:
            return load_yaml(path, safe=True)
        except yaml.YAMLError:
            logger.warning(f'Failed to parse {path} as YAML, trying JSON')
            try:
                return load_json(path)
            except json.JSONDecodeError as e:
                raise ValueError(f'Failed to parse {path} as YAML or JSON') from e

    else:
        # Unknown extension, try YAML first (more common for config)
        try:
            return load_yaml(path, safe=True)
        except yaml.YAMLError:
            try:
                return load_json(path)
            except json.JSONDecodeError as e:
                raise ValueError(f'Failed to parse {path} as YAML or JSON') from e


def _save_to_yaml(data, output_file='formatted_output.yaml'):
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

    # Add the string representer to SafeDumper
    yaml.add_representer(str, represent_str, Dumper=yaml.SafeDumper)

    # Configure dumper options for better formatting
    class CustomDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)

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
            indent=2,  # Set consistent indentation
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
        _save_to_yaml(documentation, str(path))

    return documentation


def save_dataset_to_netcdf(
    ds: xr.Dataset,
    path: str | pathlib.Path,
    compression: int = 0,
    engine: Literal['netcdf4', 'scipy', 'h5netcdf'] = 'h5netcdf',
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

    apply_encoding = False
    if compression != 0:
        if importlib.util.find_spec(engine) is not None:
            apply_encoding = True
        else:
            logger.warning(
                f'Dataset was exported without compression due to missing dependency "{engine}".'
                f'Install {engine} via `pip install {engine}`.'
            )

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
        if not apply_encoding
        else {data_var: {'zlib': True, 'complevel': compression} for data_var in ds.data_vars},
        engine=engine,
    )


def load_dataset_from_netcdf(path: str | pathlib.Path) -> xr.Dataset:
    """
    Load a dataset from a netcdf file. Load all attrs from 'attrs' attributes.

    Args:
        path: Path to load the dataset from.

    Returns:
        Dataset: Loaded dataset with restored attrs.
    """
    ds = xr.load_dataset(str(path), engine='h5netcdf')

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
