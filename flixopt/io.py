import importlib.util
import json
import logging
import pathlib
import re
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union

import linopy
import xarray as xr
import yaml

from .core import TimeSeries

logger = logging.getLogger('flixopt')


def replace_timeseries(obj, mode: Literal['name', 'stats', 'data'] = 'name'):
    """Recursively replaces TimeSeries objects with their names prefixed by '::::'."""
    if isinstance(obj, dict):
        return {k: replace_timeseries(v, mode) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_timeseries(v, mode) for v in obj]
    elif isinstance(obj, TimeSeries):  # Adjust this based on the actual class
        if obj.all_equal:
            return obj.active_data.values[0].item()
        elif mode == 'name':
            return f'::::{obj.name}'
        elif mode == 'stats':
            return obj.stats
        elif mode == 'data':
            return obj
        else:
            raise ValueError(f'Invalid mode {mode}')
    else:
        return obj


def insert_dataarray(obj, ds: xr.Dataset):
    """Recursively inserts TimeSeries objects into a dataset."""
    if isinstance(obj, dict):
        return {k: insert_dataarray(v, ds) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [insert_dataarray(v, ds) for v in obj]
    elif isinstance(obj, str) and obj.startswith('::::'):
        da = ds[obj[4:]]
        if da.isel(time=-1).isnull():
            return da.isel(time=slice(0, -1))
        return da
    else:
        return obj


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
    processed_data = _process_complex_strings(data)

    # Define a custom representer for strings
    def represent_str(dumper, data):
        # Use literal block style (|) for any string with newlines
        if '\n' in data:
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')

        # Use quoted style for strings with special characters to ensure proper parsing
        elif any(char in data for char in ':`{}[]#,&*!|>%@'):
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

        # Use plain style for simple strings
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    # Add the string representer to SafeDumper
    yaml.add_representer(str, represent_str, Dumper=yaml.SafeDumper)

    # Write to file with settings that ensure proper formatting
    with open(output_file, 'w', encoding='utf-8') as file:
        yaml.dump(
            processed_data,
            file,
            Dumper=yaml.SafeDumper,
            sort_keys=False,  # Preserve dictionary order
            default_flow_style=False,  # Use block style for mappings
            width=float('inf'),  # Don't wrap long lines
            allow_unicode=True,  # Support Unicode characters
        )


def _process_complex_strings(data):
    """
    Process dictionary data recursively with comprehensive string normalization.
    Handles various types of strings and special formatting.

    Args:
        data: The data to process (dict, list, str, or other)

    Returns:
        Processed data with normalized strings
    """
    if isinstance(data, dict):
        return {k: _process_complex_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_process_complex_strings(item) for item in data]
    elif isinstance(data, str):
        # Step 1: Normalize line endings to \n
        normalized = data.replace('\r\n', '\n').replace('\r', '\n')

        # Step 2: Handle escaped newlines with robust regex
        normalized = re.sub(r'(?<!\\)\\n', '\n', normalized)

        # Step 3: Handle unnecessary double backslashes
        normalized = re.sub(r'\\\\(n)', r'\\\1', normalized)

        # Step 4: Ensure proper formatting of "[time: N]:\n---------"
        normalized = re.sub(r'(\[time: \d+\]):\s*\\?n', r'\1:\n', normalized)

        # Step 5: Ensure "Constraint `...`" patterns are properly formatted
        normalized = re.sub(r'Constraint `([^`]+)`\\?n', r'Constraint `\1`\n', normalized)

        return normalized
    else:
        return data


def document_linopy_model(model: linopy.Model, path: pathlib.Path = None) -> Dict[str, str]:
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
        _save_to_yaml(documentation, path)

    return documentation


def save_dataset_to_netcdf(
    ds: xr.Dataset,
    path: Union[str, pathlib.Path],
    compression: int = 0,
) -> None:
    """
    Save a dataset to a netcdf file. Store the attrs as a json string in the 'attrs' attribute.

    Args:
        ds: Dataset to save.
        path: Path to save the dataset to.
        compression: Compression level for the dataset (0-9). 0 means no compression. 5 is a good default.

    Raises:
        ValueError: If the path has an invalid file extension.
    """
    if path.suffix not in ['.nc', '.nc4']:
        raise ValueError(f'Invalid file extension for path {path}. Only .nc and .nc4 are supported')

    apply_encoding = False
    if compression != 0:
        if importlib.util.find_spec('netCDF4') is not None:
            apply_encoding = True
        else:
            logger.warning(
                'Dataset was exported without compression due to missing dependency "netcdf4".'
                'Install netcdf4 via `pip install netcdf4`.'
            )
    ds = ds.copy(deep=True)
    ds.attrs = {'attrs': json.dumps(ds.attrs)}
    ds.to_netcdf(
        path,
        encoding=None
        if not apply_encoding
        else {data_var: {'zlib': True, 'complevel': 5} for data_var in ds.data_vars},
    )


def load_dataset_from_netcdf(path: Union[str, pathlib.Path]) -> xr.Dataset:
    """
    Load a dataset from a netcdf file. Load the attrs from the 'attrs' attribute.

    Args:
        path: Path to load the dataset from.

    Returns:
        Dataset: Loaded dataset.
    """
    ds = xr.load_dataset(path)
    ds.attrs = json.loads(ds.attrs['attrs'])
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

    def all_paths(self) -> Dict[str, pathlib.Path]:
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

    def update(self, new_name: Optional[str] = None, new_folder: Optional[pathlib.Path] = None) -> None:
        """Update name and/or folder and refresh all paths."""
        if new_name is not None:
            self.name = new_name
        if new_folder is not None:
            if not new_folder.is_dir() or not new_folder.exists():
                raise FileNotFoundError(f'Folder {new_folder} does not exist or is not a directory.')
            self.folder = new_folder
        self._update_paths()
