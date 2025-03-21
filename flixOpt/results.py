import datetime
import json
import logging
import pathlib
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import linopy
import numpy as np
import pandas as pd
import plotly
import xarray as xr
import yaml

from . import io as fx_io
from . import plotting
from .core import TimeSeriesCollection

if TYPE_CHECKING:
    from .calculation import Calculation, SegmentedCalculation


logger = logging.getLogger('flixOpt')


class CalculationResults:
    """Results container for Calculation results.

    This class is used to collect the results of a Calculation.
    It provides access to component, bus, and effect
    results, and includes methods for filtering, plotting, and saving results.

    The recommended way to create instances is through the class methods
    `from_file()` or `from_calculation()`, rather than direct initialization.

    Attributes:
        solution (xr.Dataset): Dataset containing optimization results.
        flow_system (xr.Dataset): Dataset containing the flow system.
        infos (Dict): Information about the calculation.
        network_infos (Dict): Information about the network structure.
        name (str): Name identifier for the calculation.
        model (linopy.Model): The optimization model (if available).
        folder (pathlib.Path): Path to the results directory.
        components (Dict[str, ComponentResults]): Results for each component.
        buses (Dict[str, BusResults]): Results for each bus.
        effects (Dict[str, EffectResults]): Results for each effect.
        timesteps_extra (pd.DatetimeIndex): The extended timesteps.
        hours_per_timestep (xr.DataArray): Duration of each timestep in hours.

    Example:
        Load results from saved files:

        >>> results = CalculationResults.from_file("results_dir", "optimization_run_1")
        >>> element_result = results["Boiler"]
        >>> results.plot_heatmap("Boiler(Q_th)|flow_rate")
        >>> results.to_file(compression=5)
        >>> results.to_file(folder="new_results_dir", compression=5)  # Save the results to a new folder
    """
    @classmethod
    def from_file(cls, folder: Union[str, pathlib.Path], name: str):
        """Create CalculationResults instance by loading from saved files.

        This method loads the calculation results from previously saved files,
        including the solution, flow system, model (if available), and metadata.

        Args:
            folder: Path to the directory containing the saved files.
            name: Base name of the saved files (without file extensions).

        Returns:
            CalculationResults: A new instance containing the loaded data.

        Raises:
            FileNotFoundError: If required files cannot be found.
            ValueError: If files exist but cannot be properly loaded.
        """
        folder = pathlib.Path(folder)

        model_path, solution_path, _, json_path, flow_system_path, _ = cls._get_paths(folder=folder, name=name)

        model = None
        if model_path.exists():
            try:
                logger.info(f'loading the linopy model "{name}" from file ("{model_path}")')
                model = linopy.read_netcdf(model_path)
            except Exception as e:
                logger.critical(f'Could not load the linopy model "{name}" from file ("{model_path}"): {e}')

        with open(json_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)

        return cls(solution=fx_io.load_dataset_from_netcdf(solution_path),
                   flow_system=fx_io.load_dataset_from_netcdf(flow_system_path),
                   name=name,
                   folder=folder,
                   model=model,
                   **meta_data)

    @classmethod
    def from_calculation(cls, calculation: 'Calculation'):
        """Create CalculationResults directly from a Calculation object.

        This method extracts the solution, flow system, and other relevant
        information directly from an existing Calculation object.

        Args:
            calculation: A Calculation object containing a solved model.

        Returns:
            CalculationResults: A new instance containing the results from
                the provided calculation.

        Raises:
            AttributeError: If the calculation doesn't have required attributes.
        """
        solution = calculation.model.solution
        solution.reindex(time=calculation.flow_system.time_series_collection.timesteps_extra)
        solution.attrs = fx_io._results_structure(calculation.flow_system)

        return cls(
            solution=solution,
            flow_system=calculation.flow_system.as_dataset(constants_in_dataset=True),
            infos=calculation.infos,
            network_infos=calculation.flow_system.network_infos(),
            model=calculation.model,
            name=calculation.name,
            folder=calculation.folder,
        )

    def __init__(
        self,
        solution: xr.Dataset,
        flow_system: xr.Dataset,
        name: str,
        infos: Dict,
        network_infos: Dict,
        folder: Optional[pathlib.Path] = None,
        model: Optional[linopy.Model] = None,
    ):
        self.solution = solution
        self.flow_system = flow_system
        self.infos = infos
        self.network_infos = network_infos
        self.name = name
        self.model = model
        self.folder = pathlib.Path(folder) if folder is not None else pathlib.Path.cwd() / 'results'
        self.components = {label: ComponentResults.from_json(self, infos)
                           for label, infos in self.solution.attrs['Components'].items()}

        self.buses = {label: BusResults.from_json(self, infos)
                      for label, infos in self.solution.attrs['Buses'].items()}

        self.effects = {label: EffectResults.from_json(self, infos)
                        for label, infos in self.solution.attrs['Effects'].items()}

        self.timesteps_extra = self.solution.indexes['time']
        self.hours_per_timestep = TimeSeriesCollection.calculate_hours_per_timestep(self.timesteps_extra)

    def __getitem__(self, key: str) -> Union['ComponentResults', 'BusResults', 'EffectResults']:
        if key in self.components:
            return self.components[key]
        if key in self.buses:
            return self.buses[key]
        if key in self.effects:
            return self.effects[key]
        raise KeyError(f'No element with label {key} found.')

    @property
    def storages(self) -> List['ComponentResults']:
        """All storages in the results."""
        return [comp for comp in self.components.values() if comp.is_storage]

    @property
    def objective(self) -> float:
        """ The objective result of the optimization. """
        return self.infos['Main Results']['Objective']

    @property
    def variables(self) -> linopy.Variables:
        """ The variables of the optimization. Only available if the linopy.Model is available. """
        if self.model is None:
            raise ValueError('The linopy model is not available.')
        return self.model.variables

    @property
    def constraints(self) -> linopy.Constraints:
        """The constraints of the optimization. Only available if the linopy.Model is available."""
        if self.model is None:
            raise ValueError('The linopy model is not available.')
        return self.model.constraints

    def filter_solution(self,
               variable_dims: Optional[Literal['scalar', 'time']] = None,
               element: Optional[str] = None) -> xr.Dataset:
        """
        Filter the solution to a specific variable dimension and element.
        If no element is specified, all elements are included.

        Args:
            variable_dims: The dimension of the variables to filter for.
            element: The element to filter for.
        """
        if element is not None:
            return filter_dataset(self[element].solution, variable_dims)
        return filter_dataset(self.solution, variable_dims)

    def plot_heatmap(self,
                     variable_name: str,
                     heatmap_timeframes: Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'] = 'D',
                     heatmap_timesteps_per_frame: Literal['W', 'D', 'h', '15min', 'min'] = 'h',
                     color_map: str = 'portland',
                     save: Union[bool, pathlib.Path] = False,
                     show: bool = True
                     ) -> plotly.graph_objs.Figure:
        return plot_heatmap(
            dataarray=self.solution[variable_name],
            name=variable_name,
            folder=self.folder,
            heatmap_timeframes=heatmap_timeframes,
            heatmap_timesteps_per_frame=heatmap_timesteps_per_frame,
            color_map=color_map,
            save=save,
            show=show)

    def to_file(
        self,
        folder: Optional[Union[str, pathlib.Path]] = None,
        name: Optional[str] = None,
        compression: int = 5,
        document_model: bool = True,
        save_linopy_model: bool = False,
    ):
        """
        Save the results to a file
        Args:
            folder: The folder where the results should be saved. Defaults to the folder of the calculation.
            name: The name of the results file. If not provided, Defaults to the name of the calculation.
            compression: The compression level to use when saving the solution file (0-9). 0 means no compression.
            document_model: Wether to document the mathematical formulations in the model.
            save_linopy_model: Wether to save the model to file. If True, the (linopy) model is saved as a .nc file.
                The model file size is rougly 100 times larger than the solution file.
        """
        folder = self.folder if folder is None else pathlib.Path(folder)
        if not folder.exists():
            try:
                folder.mkdir(parents=False)
            except FileNotFoundError as e:
                raise FileNotFoundError(f'Folder {folder} and its parent do not exist. Please create them first.') from e

        model_path, solution_path, infos_path, json_path, flow_system_path, model_doc_path = self._get_paths(
            folder= folder, name= self.name if name is None else name)

        fx_io.save_dataset_to_netcdf(self.solution, solution_path, compression=compression)
        fx_io.save_dataset_to_netcdf(self.flow_system, flow_system_path, compression=compression)

        with open(infos_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.infos, f, allow_unicode=True, sort_keys=False, indent=4, width=1000)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self._get_meta_data(), f, indent=4, ensure_ascii=False)

        if save_linopy_model:
            if self.model is None:
                logger.critical('No model in the CalculationResults. Saving the model is not possible.')
            else:
                self.model.to_netcdf(model_path)

        if document_model:
            if self.model is None:
                logger.critical('No model in the CalculationResults. Documenting the model is not possible.')
            else:
                fx_io.document_linopy_model(self.model, path=model_doc_path)

        logger.info(f'Saved calculation results "{name}" to {solution_path.parent}')

    def _get_meta_data(self) -> Dict:
        return {
            'infos': self.infos,
            'network_infos': self.network_infos,
        }

    @staticmethod
    def _get_paths(
            folder: pathlib.Path,
            name: str
    ) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
        model_path = folder / f'{name}_model.nc'
        solution_path = folder / f'{name}_solution.nc'
        infos_path = folder / f'{name}_infos.yaml'
        json_path = folder/f'{name}_structure.json'
        flow_system_path = folder / f'{name}_flowsystem.nc'
        model_documentation_path = folder / f'{name}_model_doc.yaml'
        return model_path, solution_path, infos_path, json_path, flow_system_path, model_documentation_path


class _ElementResults:
    @classmethod
    def from_json(cls, calculation_results, json_data: Dict) -> '_ElementResults':
        return cls(calculation_results,
                   json_data['label'],
                   json_data['variables'],
                   json_data['constraints'])

    def __init__(self,
                 calculation_results: CalculationResults,
                 label: str,
                 variables: List[str],
                 constraints: List[str]):
        self._calculation_results = calculation_results
        self.label = label
        self._variable_names = variables
        self._constraint_names = constraints

        self.solution = self._calculation_results.solution[self._variable_names]

    @property
    def variables(self) -> linopy.Variables:
        """
        Returns the variables of the element.

        Raises:
            ValueError: If the linopy model is not availlable.
        """
        if self._calculation_results.model is None:
            raise ValueError('The linopy model is not available.')
        return self._calculation_results.model.variables[self._variable_names]

    @property
    def constraints(self) -> linopy.Constraints:
        """
        Returns the variables of the element.

        Raises:
            ValueError: If the linopy model is not availlable.
        """
        if self._calculation_results.model is None:
            raise ValueError('The linopy model is not available.')
        return self._calculation_results.model.constraints[self._variable_names]

    def filter_solution(self, variable_dims: Optional[Literal['scalar', 'time']] = None) -> xr.Dataset:
        """
        Filter the solution of the element by dimension.

        Args:
            variable_dims: The dimension of the variables to filter for.
        """
        return filter_dataset(self.solution, variable_dims)


class _NodeResults(_ElementResults):
    @classmethod
    def from_json(cls, calculation_results, json_data: Dict)  -> '_NodeResults':
        return cls(calculation_results,
                   json_data['label'],
                   json_data['variables'],
                   json_data['constraints'],
                   json_data['inputs'],
                   json_data['outputs'])

    def __init__(self,
                 calculation_results: CalculationResults,
                 label: str,
                 variables: List[str],
                 constraints: List[str],
                 inputs: List[str],
                 outputs: List[str]):
        super().__init__(calculation_results, label, variables, constraints)
        self.inputs = inputs
        self.outputs = outputs

    def plot_node_balance(self,
                        save: Union[bool, pathlib.Path] = False,
                        show: bool = True):
        fig = plotting.with_plotly(
            self.node_balance(with_last_timestep=True).to_dataframe(), mode='area', title=f'Flow rates of {self.label}'
        )
        return plotly_save_and_show(
            fig,
            self._calculation_results.folder / f'{self.label} (flow rates).html',
            user_filename=None if isinstance(save, bool) else pathlib.Path(save),
            show=show,
            save=True if save else False)

    def node_balance(self,
                   negate_inputs: bool = True,
                   negate_outputs: bool = False,
                   threshold: Optional[float] = 1e-5,
                   with_last_timestep: bool = False) -> xr.Dataset:
        variable_names = [name for name in self._variable_names if name.endswith(('|flow_rate', '|excess_input', '|excess_output'))]
        return sanitize_dataset(
            ds=self.solution[variable_names],
            threshold=threshold,
            timesteps=self._calculation_results.timesteps_extra if with_last_timestep else None,
            negate=(
                self.outputs + self.inputs if negate_outputs and negate_inputs
                else self.outputs if negate_outputs
                else self.inputs if negate_inputs
                else None),
        )


class BusResults(_NodeResults):
    """Results for a Bus"""


class ComponentResults(_NodeResults):
    """Results for a Component"""

    @property
    def is_storage(self) -> bool:
        return self._charge_state in self._variable_names

    @property
    def _charge_state(self) -> str:
        return f'{self.label}|charge_state'

    @property
    def charge_state(self) -> xr.DataArray:
        if not self.is_storage:
            raise ValueError(f'Cant get charge_state. "{self.label}" is not a storage')
        return self.solution[self._charge_state]

    def plot_charge_state(self,
                          save: Union[bool, pathlib.Path] = False,
                          show: bool = True) -> plotly.graph_objs._figure.Figure:
        if not self.is_storage:
            raise ValueError(f'Cant plot charge_state. "{self.label}" is not a storage')
        fig = plotting.with_plotly(self.node_balance(with_last_timestep=True).to_dataframe(),
                                    mode='area',
                                    title=f'Operation Balance of {self.label}',
                                    show=False)
        charge_state = self.charge_state.to_dataframe()
        fig.add_trace(plotly.graph_objs.Scatter(
            x=charge_state.index, y=charge_state.values.flatten(), mode='lines', name=self._charge_state))

        return plotly_save_and_show(
            fig,
            self._calculation_results.folder / f'{self.label} (charge state).html',
            user_filename=None if isinstance(save, bool) else pathlib.Path(save),
            show=show,
            save=True if save else False)

    def node_balance_with_charge_state(
            self,
            negate_inputs: bool = True,
            negate_outputs: bool = False,
            threshold: Optional[float] = 1e-5) -> xr.Dataset:
        if not self.is_storage:
            raise ValueError(f'Cant get charge_state. "{self.label}" is not a storage')
        variable_names = self.inputs + self.outputs + [self._charge_state]
        return sanitize_dataset(
            ds=self.solution[variable_names],
            threshold=threshold,
            timesteps=self._calculation_results.timesteps_extra,
            negate=(
                self.outputs + self.inputs if negate_outputs and negate_inputs
                else self.outputs if negate_outputs
                else self.inputs if negate_inputs
                else None),
        )


class EffectResults(_ElementResults):
    """Results for an Effect"""

    def get_shares_from(self, element: str):
        """ Get the shares from an Element (without subelements) to the Effect"""
        return self.solution[[name for name in self._variable_names if name.startswith(f'{element}->')]]


class SegmentedCalculationResults:
    """
    Class to store the results of a SegmentedCalculation.
    """
    @classmethod
    def from_calculation(cls, calculation: 'SegmentedCalculation'):
        return cls([calc.results for calc in calculation.sub_calculations],
                   all_timesteps=calculation.all_timesteps,
                   timesteps_per_segment=calculation.timesteps_per_segment,
                   overlap_timesteps=calculation.overlap_timesteps,
                   name=calculation.name,
                   folder=calculation.folder)

    @classmethod
    def from_file(cls, folder: Union[str, pathlib.Path], name: str):
        """ Create SegmentedCalculationResults directly from file"""
        folder = pathlib.Path(folder)
        path = folder / name
        nc_file = path.with_suffix('.nc')
        logger.info(f'loading calculation "{name}" from file ("{nc_file}")')
        with open(path.with_suffix('.json'), 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        return cls(
            [CalculationResults.from_file(folder, name) for name in meta_data['sub_calculations']],
            all_timesteps=pd.DatetimeIndex([datetime.datetime.fromisoformat(date)
                                            for date in meta_data['all_timesteps']], name='time'),
            timesteps_per_segment=meta_data['timesteps_per_segment'],
            overlap_timesteps=meta_data['overlap_timesteps'],
            name=name,
            folder=folder
        )

    def __init__(self,
                 segment_results: List[CalculationResults],
                 all_timesteps: pd.DatetimeIndex,
                 timesteps_per_segment: int,
                 overlap_timesteps: int,
                 name: str,
                 folder: Optional[pathlib.Path] = None):
        self.segment_results = segment_results
        self.all_timesteps = all_timesteps
        self.timesteps_per_segment = timesteps_per_segment
        self.overlap_timesteps = overlap_timesteps
        self.name = name
        self.folder = pathlib.Path(folder) if folder is not None else pathlib.Path.cwd() / 'results'
        self.hours_per_timestep = TimeSeriesCollection.calculate_hours_per_timestep(self.all_timesteps)

    @property
    def meta_data(self) -> Dict[str, Union[int, List[str]]]:
        return {
            'all_timesteps': [datetime.datetime.isoformat(date) for date in self.all_timesteps],
            'timesteps_per_segment': self.timesteps_per_segment,
            'overlap_timesteps': self.overlap_timesteps,
            'sub_calculations': [calc.name for calc in self.segment_results]
        }

    @property
    def segment_names(self) -> List[str]:
        return [segment.name for segment in self.segment_results]

    def solution_without_overlap(self, variable_name: str) -> xr.DataArray:
        """Returns the solution of a variable without overlap"""
        dataarrays = [result.solution[variable_name].isel(time=slice(None, self.timesteps_per_segment))
                      for result in self.segment_results[:-1]
                      ] + [self.segment_results[-1].solution[variable_name]]
        return xr.concat(dataarrays, dim='time')


    def plot_heatmap(
        self,
        variable_name: str,
        heatmap_timeframes: Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'] = 'D',
        heatmap_timesteps_per_frame: Literal['W', 'D', 'h', '15min', 'min'] = 'h',
        color_map: str = 'portland',
        save: Union[bool, pathlib.Path] = False,
        show: bool = True
    ) -> plotly.graph_objs.Figure:
        return plot_heatmap(
            dataarray=self.solution_without_overlap(variable_name),
            name=variable_name,
            folder=self.folder,
            heatmap_timeframes=heatmap_timeframes,
            heatmap_timesteps_per_frame=heatmap_timesteps_per_frame,
            color_map=color_map,
            save=save,
            show=show)

    def to_file(self,
                folder: Optional[Union[str, pathlib.Path]] = None,
                name: Optional[str] = None,
                compression: int = 5):
        """Save the results to a file"""
        folder = self.folder if folder is None else pathlib.Path(folder)
        name = self.name if name is None else name
        path = folder / name
        if not folder.exists():
            try:
                folder.mkdir(parents=False)
            except FileNotFoundError as e:
                raise FileNotFoundError(f'Folder {folder} and its parent do not exist. Please create them first.') from e
        for segment in self.segment_results:
            segment.to_file(folder=folder, name=f'{name}-{segment.name}', compression=compression)

        with open(path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(self.meta_data, f, indent=4, ensure_ascii=False)
        logger.info(f'Saved calculation "{name}" to {path}')


def plotly_save_and_show(fig: plotly.graph_objs.Figure,
                         default_filename: pathlib.Path,
                         user_filename: Optional[pathlib.Path] = None,
                         show: bool = True,
                         save: bool = False) -> plotly.graph_objs.Figure:
    """
    Optionally saves and/or displays a Plotly figure.

    Parameters:
    - fig (go.Figure): The Plotly figure to display or save.
    - default_filename (Path): The default file path if no user filename is provided.
    - user_filename (Optional[Path]): An optional user-specified file path.
    - show (bool): Whether to display the figure (default: True).
    - save (bool): Whether to save the figure (default: False).

    Returns:
    - go.Figure: The input figure.
    """
    filename = user_filename or default_filename
    if show and not save:
        fig.show()
    elif save and show:
        plotly.offline.plot(fig, filename=str(filename))
    elif save and not show:
        fig.write_html(filename)
    return fig


def plot_heatmap(
    dataarray: xr.DataArray,
    name: str,
    folder: pathlib.Path,
    heatmap_timeframes: Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'] = 'D',
    heatmap_timesteps_per_frame: Literal['W', 'D', 'h', '15min', 'min'] = 'h',
    color_map: str = 'portland',
    save: Union[bool, pathlib.Path] = False,
    show: bool = True
):
    heatmap_data = plotting.heat_map_data_from_df(
        dataarray.to_dataframe(name), heatmap_timeframes, heatmap_timesteps_per_frame, 'ffill')
    fig = plotting.heat_map_plotly(
        heatmap_data, title=name, color_map=color_map,
        xlabel=f'timeframe [{heatmap_timeframes}]', ylabel=f'timesteps [{heatmap_timesteps_per_frame}]'
    )
    return plotly_save_and_show(
        fig,
        folder / f'{name} ({heatmap_timeframes}-{heatmap_timesteps_per_frame}).html',
        user_filename=None if isinstance(save, bool) else pathlib.Path(save),
        show=show,
        save=True if save else False)


def sanitize_dataset(
        ds: xr.Dataset,
        timesteps: Optional[pd.DatetimeIndex] = None,
        threshold: Optional[float] = 1e-5,
        negate: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Sanitizes a dataset by dropping variables with small values and optionally reindexing the time axis.

    Parameters:
    - ds (xr.Dataset): The dataset to sanitize.
    - timesteps (Optional[pd.DatetimeIndex]): The timesteps to reindex the dataset to. If None, the original timesteps are kept.
    - threshold (Optional[float]): The threshold for dropping variables. If None, no variables are dropped.
    - negate (Optional[List[str]]): The variables to negate. If None, no variables are negated.

    Returns:
    - xr.Dataset: The sanitized dataset.
    """
    if negate is not None:
        for var in negate:
            ds[var] = -ds[var]
    if threshold is not None:
        abs_ds = xr.apply_ufunc(np.abs, ds)
        vars_to_drop = [var for var in ds.data_vars if (abs_ds[var] <= threshold).all()]
        ds = ds.drop_vars(vars_to_drop)
    if timesteps is not None and not ds.indexes['time'].equals(timesteps):
        ds = ds.reindex({'time': timesteps}, fill_value=np.nan)
    return ds


def filter_dataset(
        ds: xr.Dataset,
        variable_dims: Optional[Literal['scalar', 'time']] = None,
) -> xr.Dataset:
    """
    Filters a dataset by its dimensions.

    Args:
        ds: The dataset to filter.
        variable_dims: The dimension of the variables to filter for.
    """
    if variable_dims is None:
        return ds

    if variable_dims == 'scalar':
        return ds[[name for name, da in ds.data_vars.items() if len(da.dims) == 0]]
    elif variable_dims == 'time':
        return ds[[name for name, da in ds.data_vars.items() if 'time' in da.dims]]
    else:
        raise ValueError(f'Not allowed value for "filter_dataset()": {variable_dims=}')
