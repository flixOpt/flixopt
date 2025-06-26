"""
This module contains the FlowSystem class, which is used to collect instances of many other classes by the end User.
"""

import json
import logging
import pathlib
import warnings
from io import StringIO
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from rich.console import Console
from rich.pretty import Pretty

from . import io as fx_io
from .core import ConversionError, DataConverter, NumericDataInternal, NumericDataUser, TimeSeriesData
from .effects import Effect, EffectCollection, EffectValuesInternal, EffectValuesUser
from .elements import Bus, Component, Flow
from .structure import Element, Interface, SystemModel

if TYPE_CHECKING:
    import pyvis

logger = logging.getLogger('flixopt')


class FlowSystem(Interface):
    """
    FlowSystem serves as the main container for energy system modeling, organizing
    high-level elements including Components (like boilers, heat pumps, storages),
    Buses (connection points), and Effects (system-wide influences). It handles
    time series data management, network connectivity, and provides serialization
    capabilities for saving and loading complete system configurations.

    The system uses xarray.Dataset for efficient time series data handling. It can be exported and restored to NETCDF.

    See Also:
        Component: Base class for system components like boilers, heat pumps.
        Bus: Connection points for flows between components.
        Effect: System-wide effects, like the optimization objective.
    """

    def __init__(
        self,
        timesteps: pd.DatetimeIndex,
        hours_of_last_timestep: Optional[float] = None,
        hours_of_previous_timesteps: Optional[Union[int, float, np.ndarray]] = None,
    ):
        """
        Args:
            timesteps: The timesteps of the model.
            hours_of_last_timestep: The duration of the last time step. Uses the last time interval if not specified
            hours_of_previous_timesteps: The duration of previous timesteps.
                If None, the first time increment of time_series is used.
                This is needed to calculate previous durations (for example consecutive_on_hours).
                If you use an array, take care that its long enough to cover all previous values!
        """
        # Store timing information directly
        self.timesteps = self._validate_timesteps(timesteps)
        self.timesteps_extra = self._create_timesteps_with_extra(timesteps, hours_of_last_timestep)
        self.hours_per_timestep = self.calculate_hours_per_timestep(self.timesteps_extra)
        self.hours_of_previous_timesteps = self._calculate_hours_of_previous_timesteps(
            timesteps, hours_of_previous_timesteps
        )

        # Element collections
        self.components: Dict[str, Component] = {}
        self.buses: Dict[str, Bus] = {}
        self.effects: EffectCollection = EffectCollection()
        self.model: Optional[SystemModel] = None

        self._connected_and_transformed = False
        self._used_in_calculation = False

    @staticmethod
    def _validate_timesteps(timesteps: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Validate timesteps format and rename if needed."""
        if not isinstance(timesteps, pd.DatetimeIndex):
            raise TypeError('timesteps must be a pandas DatetimeIndex')
        if len(timesteps) < 2:
            raise ValueError('timesteps must contain at least 2 timestamps')
        if timesteps.name != 'time':
            timesteps.name = 'time'
        if not timesteps.is_monotonic_increasing:
            raise ValueError('timesteps must be sorted')
        return timesteps

    @staticmethod
    def _create_timesteps_with_extra(
            timesteps: pd.DatetimeIndex, hours_of_last_timestep: Optional[float]
    ) -> pd.DatetimeIndex:
        """Create timesteps with an extra step at the end."""
        if hours_of_last_timestep is None:
            hours_of_last_timestep = (timesteps[-1] - timesteps[-2]) / pd.Timedelta(hours=1)

        last_date = pd.DatetimeIndex([timesteps[-1] + pd.Timedelta(hours=hours_of_last_timestep)], name='time')
        return pd.DatetimeIndex(timesteps.append(last_date), name='time')

    @staticmethod
    def calculate_hours_per_timestep(timesteps_extra: pd.DatetimeIndex) -> xr.DataArray:
        """Calculate duration of each timestep."""
        hours_per_step = np.diff(timesteps_extra) / pd.Timedelta(hours=1)
        return xr.DataArray(
            hours_per_step, coords={'time': timesteps_extra[:-1]}, dims=['time'], name='hours_per_timestep'
        )

    @staticmethod
    def _calculate_hours_of_previous_timesteps(
            timesteps: pd.DatetimeIndex, hours_of_previous_timesteps: Optional[Union[float, np.ndarray]]
    ) -> Union[float, np.ndarray]:
        """Calculate duration of regular timesteps."""
        if hours_of_previous_timesteps is not None:
            return hours_of_previous_timesteps
        # Calculate from the first interval
        first_interval = timesteps[1] - timesteps[0]
        return first_interval.total_seconds() / 3600  # Convert to hours

    def _create_reference_structure(self) -> Tuple[Dict, Dict[str, xr.DataArray]]:
        """
        Override Interface method to handle FlowSystem-specific serialization.
        Combines custom FlowSystem logic with Interface pattern for nested objects.

        Returns:
            Tuple of (reference_structure, extracted_arrays_dict)
        """
        # Start with Interface base functionality for constructor parameters
        reference_structure, all_extracted_arrays = super()._create_reference_structure()

        # Override timesteps serialization (we need timesteps_extra instead of timesteps)
        reference_structure['timesteps_extra'] = [date.isoformat() for date in self.timesteps_extra]

        # Remove timesteps from structure since we're using timesteps_extra
        reference_structure.pop('timesteps', None)

        # Add timing arrays directly (not handled by Interface introspection)
        all_extracted_arrays['hours_per_timestep'] = self.hours_per_timestep

        # Extract from components
        components_structure = {}
        for comp_label, component in self.components.items():
            comp_structure, comp_arrays = component._create_reference_structure()
            all_extracted_arrays.update(comp_arrays)
            components_structure[comp_label] = comp_structure
        reference_structure['components'] = components_structure

        # Extract from buses
        buses_structure = {}
        for bus_label, bus in self.buses.items():
            bus_structure, bus_arrays = bus._create_reference_structure()
            all_extracted_arrays.update(bus_arrays)
            buses_structure[bus_label] = bus_structure
        reference_structure['buses'] = buses_structure

        # Extract from effects
        effects_structure = {}
        for effect in self.effects:
            effect_structure, effect_arrays = effect._create_reference_structure()
            all_extracted_arrays.update(effect_arrays)
            effects_structure[effect.label] = effect_structure
        reference_structure['effects'] = effects_structure

        return reference_structure, all_extracted_arrays

    def to_dataset(self) -> xr.Dataset:
        """
        Convert the FlowSystem to an xarray Dataset.
        Ensures FlowSystem is connected before serialization.

        Returns:
            xr.Dataset: Dataset containing all DataArrays with structure in attributes
        """
        if not self._connected_and_transformed:
            logger.warning('FlowSystem is not connected_and_transformed..')
            self.connect_and_transform()

        return super().to_dataset()

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> 'FlowSystem':
        """
        Create a FlowSystem from an xarray Dataset.
        Handles FlowSystem-specific reconstruction logic.

        Args:
            ds: Dataset containing the FlowSystem data

        Returns:
            FlowSystem instance
        """
        # Get the reference structure from attrs
        reference_structure = dict(ds.attrs)

        # Extract FlowSystem constructor parameters
        timesteps_extra = ds.indexes['time']
        hours_of_previous_timesteps = reference_structure['hours_of_previous_timesteps']

        # Calculate hours_of_last_timestep from the timesteps
        hours_of_last_timestep = float((timesteps_extra[-1] - timesteps_extra[-2]) / pd.Timedelta(hours=1))

        # Create FlowSystem instance with constructor parameters
        flow_system = cls(
            timesteps=timesteps_extra[:-1],
            hours_of_last_timestep=hours_of_last_timestep,
            hours_of_previous_timesteps=hours_of_previous_timesteps,
        )

        # Create arrays dictionary from dataset variables
        arrays_dict = {name: array for name, array in ds.data_vars.items()}

        # Restore components
        components_structure = reference_structure.get('components', {})
        for comp_label, comp_data in components_structure.items():
            component = cls._resolve_reference_structure(comp_data, arrays_dict)
            if not isinstance(component, Component):
                logger.critical(f'Restoring component {comp_label} failed.')
            flow_system._add_components(component)

        # Restore buses
        buses_structure = reference_structure.get('buses', {})
        for bus_label, bus_data in buses_structure.items():
            bus = cls._resolve_reference_structure(bus_data, arrays_dict)
            if not isinstance(bus, Bus):
                logger.critical(f'Restoring bus {bus_label} failed.')
            flow_system._add_buses(bus)

        # Restore effects
        effects_structure = reference_structure.get('effects', {})
        for effect_label, effect_data in effects_structure.items():
            effect = cls._resolve_reference_structure(effect_data, arrays_dict)
            if not isinstance(effect, Effect):
                logger.critical(f'Restoring effect {effect_label} failed.')
            flow_system._add_effects(effect)

        return flow_system

    def to_netcdf(self, path: Union[str, pathlib.Path], compression: int = 0):
        """
        Save the FlowSystem to a NetCDF file.
        Ensures FlowSystem is connected before saving.

        Args:
            path: The path to the netCDF file.
            compression: The compression level to use when saving the file.
        """
        if not self._connected_and_transformed:
            logger.warning('FlowSystem is not connected. Calling connect_and_transform() now.')
            self.connect_and_transform()

        super().to_netcdf(path, compression)
        logger.info(f'Saved FlowSystem to {path}')

    def get_structure(self, clean: bool = False, stats: bool = False) -> Dict:
        """
        Get FlowSystem structure.
        Ensures FlowSystem is connected before getting structure.

        Args:
            clean: If True, remove None and empty dicts and lists.
            stats: If True, replace DataArray references with statistics
        """
        if not self._connected_and_transformed:
            logger.warning('FlowSystem is not connected. Calling connect_and_transform() now.')
            self.connect_and_transform()

        return super().get_structure(clean, stats)

    def to_json(self, path: Union[str, pathlib.Path]):
        """
        Save the flow system to a JSON file.
        Ensures FlowSystem is connected before saving.

        Args:
            path: The path to the JSON file.
        """
        if not self._connected_and_transformed:
            logger.warning('FlowSystem needs to be connected and transformed before saving to JSON. Calling connect_and_transform() now.')
            self.connect_and_transform()

        super().to_json(path)

    def fit_to_model_coords(
        self,
        name: str,
        data: Optional[NumericDataUser],
        needs_extra_timestep: bool = False,
    ) -> Optional[NumericDataInternal]:
        """
        Fit data to model coordinate system (currently time, but extensible).

        Args:
            name: Name of the data
            data: Data to fit to model coordinates
            needs_extra_timestep: Whether to use extended time coordinates

        Returns:
            xr.DataArray aligned to model coordinate system
        """
        if data is None:
            return None

        # Choose appropriate timesteps
        target_timesteps = self.timesteps_extra if needs_extra_timestep else self.timesteps

        if isinstance(data, TimeSeriesData):
            try:
                return TimeSeriesData(
                    DataConverter.to_dataarray(data, timesteps=target_timesteps),
                    agg_group=data.agg_group, agg_weight=data.agg_weight
                ).rename(name)
            except ConversionError as e:
                logger.critical(f'Could not convert time series data "{name}" to DataArray: {e}. \n'
                                f'Take care to use the correct (time) index.')
        else:
            return DataConverter.to_dataarray(data, timesteps=target_timesteps).rename(name)

    def fit_effects_to_model_coords(
        self,
        label_prefix: Optional[str],
        effect_values: Optional[EffectValuesUser],
        label_suffix: Optional[str] = None,
    ) -> Optional[EffectValuesInternal]:
        """
        Transform EffectValues from the user to Internal Datatypes aligned with model coordinates.
        """
        if effect_values is None:
            return None

        effect_values_dict = self.effects.create_effect_values_dict(effect_values)

        return {
            effect: self.fit_to_model_coords('|'.join(filter(None, [label_prefix, effect, label_suffix])), value)
            for effect, value in effect_values_dict.items()
        }

    def connect_and_transform(self):
        """Transform data for all elements using the new simplified approach."""
        if not self._connected_and_transformed:
            self._connect_network()
            for element in self.all_elements.values():
                element.transform_data(self)
        self._connected_and_transformed = True

    def add_elements(self, *elements: Element) -> None:
        """
        Add Components(Storages, Boilers, Heatpumps, ...), Buses or Effects to the FlowSystem

        Args:
            *elements: childs of  Element like Boiler, HeatPump, Bus,...
                modeling Elements
        """
        if self._connected_and_transformed:
            warnings.warn(
                'You are adding elements to an already connected FlowSystem. This is not recommended (But it works).',
                stacklevel=2,
            )
            self._connected_and_transformed = False
        for new_element in list(elements):
            if isinstance(new_element, Component):
                self._add_components(new_element)
            elif isinstance(new_element, Effect):
                self._add_effects(new_element)
            elif isinstance(new_element, Bus):
                self._add_buses(new_element)
            else:
                raise TypeError(
                    f'Tried to add incompatible object to FlowSystem: {type(new_element)=}: {new_element=} '
                )

    def create_model(self) -> SystemModel:
        if not self._connected_and_transformed:
            raise RuntimeError('FlowSystem is not connected_and_transformed. Call FlowSystem.connect_and_transform() first.')
        self.model = SystemModel(self)
        return self.model

    def plot_network(
        self,
        path: Union[bool, str, pathlib.Path] = 'flow_system.html',
        controls: Union[
            bool,
            List[
                Literal['nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer']
            ],
        ] = True,
        show: bool = False,
    ) -> Optional['pyvis.network.Network']:
        """
        Visualizes the network structure of a FlowSystem using PyVis, saving it as an interactive HTML file.
        """
        from . import plotting

        node_infos, edge_infos = self.network_infos()
        return plotting.plot_network(node_infos, edge_infos, path, controls, show)

    def network_infos(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
        if not self._connected_and_transformed:
            self.connect_and_transform()
        nodes = {
            node.label_full: {
                'label': node.label,
                'class': 'Bus' if isinstance(node, Bus) else 'Component',
                'infos': node.__str__(),
            }
            for node in list(self.components.values()) + list(self.buses.values())
        }

        edges = {
            flow.label_full: {
                'label': flow.label,
                'start': flow.bus if flow.is_input_in_component else flow.component,
                'end': flow.component if flow.is_input_in_component else flow.bus,
                'infos': flow.__str__(),
            }
            for flow in self.flows.values()
        }

        return nodes, edges

    def _check_if_element_is_unique(self, element: Element) -> None:
        """
        checks if element or label of element already exists in list

        Args:
            element: new element to check
        """
        if element in self.all_elements.values():
            raise ValueError(f'Element {element.label} already added to FlowSystem!')
        # check if name is already used:
        if element.label_full in self.all_elements:
            raise ValueError(f'Label of Element {element.label} already used in another element!')

    def _add_effects(self, *args: Effect) -> None:
        self.effects.add_effects(*args)

    def _add_components(self, *components: Component) -> None:
        for new_component in list(components):
            logger.info(f'Registered new Component: {new_component.label}')
            self._check_if_element_is_unique(new_component)  # check if already exists:
            self.components[new_component.label] = new_component  # Add to existing components

    def _add_buses(self, *buses: Bus):
        for new_bus in list(buses):
            logger.info(f'Registered new Bus: {new_bus.label}')
            self._check_if_element_is_unique(new_bus)  # check if already exists:
            self.buses[new_bus.label] = new_bus  # Add to existing components

    def _connect_network(self):
        """Connects the network of components and buses. Can be rerun without changes if no elements were added"""
        for component in self.components.values():
            for flow in component.inputs + component.outputs:
                flow.component = component.label_full
                flow.is_input_in_component = True if flow in component.inputs else False

                # Add Bus if not already added (deprecated)
                if flow._bus_object is not None and flow._bus_object not in self.buses.values():
                    self._add_buses(flow._bus_object)
                    warnings.warn(
                        f'The Bus {flow._bus_object.label} was added to the FlowSystem from {flow.label_full}.'
                        f'This is deprecated and will be removed in the future. '
                        f'Please pass the Bus.label to the Flow and the Bus to the FlowSystem instead.',
                        UserWarning,
                        stacklevel=1,
                    )

                # Connect Buses
                bus = self.buses.get(flow.bus)
                if bus is None:
                    raise KeyError(
                        f'Bus {flow.bus} not found in the FlowSystem, but used by "{flow.label_full}". '
                        f'Please add it first.'
                    )
                if flow.is_input_in_component and flow not in bus.outputs:
                    bus.outputs.append(flow)
                elif not flow.is_input_in_component and flow not in bus.inputs:
                    bus.inputs.append(flow)
        logger.debug(
            f'Connected {len(self.buses)} Buses and {len(self.components)} '
            f'via {len(self.flows)} Flows inside the FlowSystem.'
        )

    def __repr__(self) -> str:
        """Compact representation for debugging."""
        status = '✓' if self._connected_and_transformed else '⚠'
        return (
            f'FlowSystem({len(self.timesteps)} timesteps '
            f'[{self.timesteps[0].strftime("%Y-%m-%d")} to {self.timesteps[-1].strftime("%Y-%m-%d")}], '
            f'{len(self.components)} Components,  {len(self.buses)} Buses, {len(self.effects)} Effects, {status})'
        )

    def __str__(self) -> str:
        """Structured summary for users."""

        def format_elements(element_names: list, label: str, alignment: int = 12):
            name_list = ', '.join(element_names[:3])
            if len(element_names) > 3:
                name_list += f' ... (+{len(element_names) - 3} more)'

            suffix = f' ({name_list})' if element_names else ''
            padding = alignment - len(label) - 1  # -1 for the colon
            return f'{label}:{"":<{padding}} {len(element_names)}{suffix}'

        time_period = f'Time period: {self.timesteps[0].date()} to {self.timesteps[-1].date()}'
        freq_str = str(self.timesteps.freq).replace('<', '').replace('>', '') if self.timesteps.freq else 'irregular'

        lines = [
            'FlowSystem Overview:',
            f'{"─" * 50}',
            time_period,
            f'Timesteps:   {len(self.timesteps)} ({freq_str})',
            format_elements(list(self.components.keys()), 'Components'),
            format_elements(list(self.buses.keys()), 'Buses'),
            format_elements(list(self.effects.effects.keys()), 'Effects'),
            f'Status:      {"Connected & Transformed" if self._connected_and_transformed else "Not connected"}',
        ]

        return '\n'.join(lines)

    def __eq__(self, other: 'FlowSystem'):
        """Check if two FlowSystems are equal by comparing their dataset representations."""
        if not isinstance(other, FlowSystem):
            raise NotImplementedError('Comparison with other types is not implemented for class FlowSystem')

        ds_me = self.to_dataset()
        ds_other = other.to_dataset()

        try:
            xr.testing.assert_equal(ds_me, ds_other)
        except AssertionError:
            return False

        if ds_me.attrs != ds_other.attrs:
            return False

        return True

    @property
    def flows(self) -> Dict[str, Flow]:
        set_of_flows = {flow for comp in self.components.values() for flow in comp.inputs + comp.outputs}
        return {flow.label_full: flow for flow in set_of_flows}

    @property
    def all_elements(self) -> Dict[str, Element]:
        return {**self.components, **self.effects.effects, **self.flows, **self.buses}

    @property
    def used_in_calculation(self) -> bool:
        return self._used_in_calculation

    def sel(self, **indexers) -> 'FlowSystem':
        """Select a subset of the flowsystem like dataset.sel(time=slice('2023-01', '2023-06'))"""
        if not self._connected_and_transformed:
            self.connect_and_transform()

        # Convert to dataset, select, then convert back
        dataset = self.to_dataset()

        # Extend time selection and handle NaN preservation
        if 'time' in indexers:
            indexers = self._extend_time_selection(indexers, dataset)
            selected_dataset = dataset.sel(**indexers)
            selected_dataset = self._preserve_nan_pattern(selected_dataset, dataset)
        else:
            selected_dataset = dataset.sel(**indexers)

        return self.__class__.from_dataset(selected_dataset)

    def isel(self, **indexers) -> 'FlowSystem':
        """Select by integer index like dataset.isel(time=slice(0, 100))"""
        if not self._connected_and_transformed:
            self.connect_and_transform()

        # Convert to dataset, select, then convert back
        dataset = self.to_dataset()

        # Extend time selection and handle NaN preservation
        if 'time' in indexers:
            indexers = self._extend_time_iselection(indexers, dataset)
            selected_dataset = dataset.isel(**indexers)
            selected_dataset = self._preserve_nan_pattern(selected_dataset, dataset)
        else:
            selected_dataset = dataset.isel(**indexers)

        return self.__class__.from_dataset(selected_dataset)

    def _preserve_nan_pattern(self, processed_dataset: xr.Dataset, original_dataset: xr.Dataset) -> xr.Dataset:
        """
        Preserve NaN pattern at the last timestep for arrays that originally had NaN at the end.
        Works for both selection and resampling operations.
        """
        for var_name, processed_array in processed_dataset.data_vars.items():
            if var_name in original_dataset.data_vars:
                original_array = original_dataset.data_vars[var_name]

                # Check if original array had NaN at the last timestep
                if len(original_array.time) > 0 and len(processed_array.time) > 0:
                    last_original = original_array.isel(time=-1)

                    if last_original.isnull().all():  # All values at last timestep are NaN
                        # Set all values at last timestep to NaN
                        processed_array = processed_array.copy()
                        processed_array.values[..., -1] = np.nan
                        processed_dataset[var_name] = processed_array
                    elif last_original.isnull().any():  # Some values at last timestep are NaN
                        # Preserve the specific NaN pattern (if dimensions allow)
                        processed_array = processed_array.copy()
                        try:
                            nan_mask = last_original.isnull().values
                            processed_array.values[..., -1][nan_mask] = np.nan
                        except (IndexError, ValueError):
                            # Fallback: set entire last timestep to NaN if dimensions don't match
                            processed_array.values[..., -1] = np.nan
                        processed_dataset[var_name] = processed_array

        return processed_dataset

    def _extend_time_selection(self, indexers: dict, dataset: xr.Dataset) -> dict:
        """Extend time selection to include the next timestep for proper boundaries."""
        new_indexers = indexers.copy()
        time_sel = indexers['time']

        if isinstance(time_sel, slice):
            # For slice, extend the stop point
            if time_sel.stop is not None:
                time_coord = dataset.coords['time']
                try:
                    # Find the index of the stop time and add 1
                    stop_idx = time_coord.get_index('time').get_indexer([time_sel.stop], method='nearest')[0]
                    if stop_idx < len(time_coord) - 1:  # Don't go beyond bounds
                        next_time = time_coord.isel(time=stop_idx + 1).values
                        new_indexers['time'] = slice(time_sel.start, next_time, time_sel.step)
                except Exception:
                    pass  # Keep original if extension fails

        return new_indexers

    def _extend_time_iselection(self, indexers: dict, dataset: xr.Dataset) -> dict:
        """Extend integer time selection to include the next timestep."""
        new_indexers = indexers.copy()
        time_sel = indexers['time']

        if isinstance(time_sel, slice):
            # For slice, extend the stop point by 1
            stop = time_sel.stop
            if stop is not None and stop < len(dataset.coords['time']) - 1:
                new_indexers['time'] = slice(time_sel.start, stop + 1, time_sel.step)
        elif isinstance(time_sel, int):
            # For single index, convert to slice including next
            if time_sel < len(dataset.coords['time']) - 1:
                new_indexers['time'] = slice(time_sel, time_sel + 2)
        elif isinstance(time_sel, (list, np.ndarray)):
            # For list/array of indices, add next indices
            extended_indices = list(time_sel)
            max_idx = len(dataset.coords['time']) - 1
            for idx in time_sel:
                if isinstance(idx, int) and idx < max_idx and (idx + 1) not in extended_indices:
                    extended_indices.append(idx + 1)
            new_indexers['time'] = sorted(extended_indices)

        return new_indexers

    def resample(self, time, method: str = 'mean', **kwargs) -> 'FlowSystem':
        """
        Resample time dimension like dataset.resample().

        Args:
            time: Resampling frequency (e.g., '1H', '1D')
            method: Resampling method ('mean', 'sum', 'max', 'min', 'first', 'last')
            **kwargs: Additional arguments passed to xarray.resample()
        """
        if not self._connected_and_transformed:
            self.connect_and_transform()

        dataset = self.to_dataset()
        resampler = dataset.resample(time=time, **kwargs)

        # Apply the specified method
        if hasattr(resampler, method):
            resampled_dataset = getattr(resampler, method)()
        else:
            raise ValueError(f'Unsupported resampling method: {method}')

        # Preserve NaN pattern at the last timestep
        resampled_dataset = self._preserve_nan_pattern(resampled_dataset, dataset)

        return self.__class__.from_dataset(resampled_dataset)
