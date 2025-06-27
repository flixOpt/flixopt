"""
This module contains the FlowSystem class, which is used to collect instances of many other classes by the end User.
"""

import json
import logging
import pathlib
import warnings
from io import StringIO
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from rich.console import Console
from rich.pretty import Pretty

from . import io as fx_io
from .core import ConversionError, DataConverter, TemporalData, TemporalDataUser, TimeSeriesData
from .effects import Effect, EffectCollection, ScalarEffects, ScalarEffectsUser, TemporalEffects, TemporalEffectsUser
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

        # Remove timesteps, as it's directly stored in dataset index
        reference_structure.pop('timesteps', None)

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
            logger.warning('FlowSystem is not connected_and_transformed. Connecting and transforming data now.')
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

        # Create FlowSystem instance with constructor parameters
        flow_system = cls(
            timesteps=ds.indexes['time'],
            hours_of_last_timestep=reference_structure.get('hours_of_last_timestep'),
            hours_of_previous_timesteps=reference_structure.get('hours_of_previous_timesteps'),
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
        data: Optional[TemporalDataUser],
    ) -> Optional[TemporalData]:
        """
        Fit data to model coordinate system (currently time, but extensible).

        Args:
            name: Name of the data
            data: Data to fit to model coordinates

        Returns:
            xr.DataArray aligned to model coordinate system
        """
        if data is None:
            return None

        if isinstance(data, TimeSeriesData):
            try:
                data.name = name  # Set name of previous object!
                return TimeSeriesData(
                    DataConverter.to_dataarray(data, timesteps=self.timesteps),
                    aggregation_group=data.aggregation_group, aggregation_weight=data.aggregation_weight
                ).rename(name)
            except ConversionError as e:
                logger.critical(f'Could not convert time series data "{name}" to DataArray: {e}. \n'
                                f'Take care to use the correct (time) index.')
        else:
            return DataConverter.to_dataarray(data, timesteps=self.timesteps).rename(name)

    def fit_effects_to_model_coords(
        self,
        label_prefix: Optional[str],
        effect_values: Optional[TemporalEffectsUser],
        label_suffix: Optional[str] = None,
    ) -> Optional[TemporalEffects]:
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

    def __getitem__(self, item) -> Element:
        """Get element by exact label with helpful error messages."""
        if item in self.all_elements:
            return self.all_elements[item]

        # Provide helpful error with suggestions
        from difflib import get_close_matches

        suggestions = get_close_matches(item, self.all_elements.keys(), n=3, cutoff=0.6)

        if suggestions:
            suggestion_str = ', '.join(f"'{s}'" for s in suggestions)
            raise KeyError(f"Element '{item}' not found. Did you mean: {suggestion_str}?")
        else:
            raise KeyError(f"Element '{item}' not found in FlowSystem")

    def __contains__(self, item: str) -> bool:
        """Check if element exists in the FlowSystem."""
        return item in self.all_elements

    def __iter__(self):
        """Iterate over element labels."""
        return iter(self.all_elements.keys())

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

    def sel(self, time: Optional[Union[str, slice, List[str], pd.Timestamp, pd.DatetimeIndex]] = None) -> 'FlowSystem':
        """
        Select a subset of the flowsystem by the time coordinate.

        Args:
            time: Time selection (e.g., slice('2023-01-01', '2023-12-31'), '2023-06-15', or list of times)

        Returns:
            FlowSystem: New FlowSystem with selected data
        """
        if not self._connected_and_transformed:
            self.connect_and_transform()

        # Build indexers dict from non-None parameters
        indexers = {}
        if time is not None:
            indexers['time'] = time

        if not indexers:
            return self.copy()  # Return a copy when no selection

        selected_dataset = self.to_dataset().sel(**indexers)
        return self.__class__.from_dataset(selected_dataset)

    def isel(self, time: Optional[Union[int, slice, List[int]]] = None) -> 'FlowSystem':
        """
        Select a subset of the flowsystem by integer indices.

        Args:
            time: Time selection by integer index (e.g., slice(0, 100), 50, or [0, 5, 10])

        Returns:
            FlowSystem: New FlowSystem with selected data
        """
        if not self._connected_and_transformed:
            self.connect_and_transform()

        # Build indexers dict from non-None parameters
        indexers = {}
        if time is not None:
            indexers['time'] = time

        if not indexers:
            return self.copy()  # Return a copy when no selection

        selected_dataset = self.to_dataset().isel(**indexers)
        return self.__class__.from_dataset(selected_dataset)

    def resample(
        self,
        time: str,
        method: Literal['mean', 'sum', 'max', 'min', 'first', 'last', 'std', 'var', 'median', 'count'] = 'mean',
        **kwargs: Any
    ) -> 'FlowSystem':
        """
        Create a resampled FlowSystem by resampling data along the time dimension (like xr.Dataset.resample()).

        Args:
            time: Resampling frequency (e.g., '3h', '2D', '1M')
            method: Resampling method. Recommended: 'mean', 'first', 'last', 'max', 'min'
            **kwargs: Additional arguments passed to xarray.resample()

        Returns:
            FlowSystem: New FlowSystem with resampled data
        """
        if not self._connected_and_transformed:
            self.connect_and_transform()

        dataset = self.to_dataset()
        resampler = dataset.resample(time=time, **kwargs)

        if hasattr(resampler, method):
            resampled_dataset = getattr(resampler, method)()
        else:
            available_methods = ['mean', 'sum', 'max', 'min', 'first', 'last', 'std', 'var', 'median', 'count']
            raise ValueError(f'Unsupported resampling method: {method}. Available: {available_methods}')

        return self.__class__.from_dataset(resampled_dataset)
