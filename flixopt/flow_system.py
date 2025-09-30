"""
This module contains the FlowSystem class, which is used to collect instances of many other classes by the end User.
"""

from __future__ import annotations

import json
import logging
import warnings
from io import StringIO
from typing import TYPE_CHECKING, Literal

import pandas as pd
from rich.console import Console
from rich.pretty import Pretty

from . import io as fx_io
from .core import NumericData, TimeSeries, TimeSeriesCollection, TimeSeriesData
from .effects import Effect, EffectCollection, EffectTimeSeries, EffectValuesDict, EffectValuesUser
from .elements import Bus, Component, Flow
from .structure import CLASS_REGISTRY, Element, SystemModel

if TYPE_CHECKING:
    import pathlib

    import numpy as np
    import pyvis
    import xarray as xr

logger = logging.getLogger('flixopt')


class FlowSystem:
    """
    A FlowSystem organizes the high level Elements (Components, Buses & Effects).

    This is the main container class that users work with to build and manage their System.

    Args:
        timesteps: The timesteps of the model.
        hours_of_last_timestep: The duration of the last time step. Uses the last time interval if not specified
        hours_of_previous_timesteps: The duration of previous timesteps.
            If None, the first time increment of time_series is used.
            This is needed to calculate previous durations (for example consecutive_on_hours).
            If you use an array, take care that its long enough to cover all previous values!

    Notes:
        - Creates an empty registry for components and buses, an empty EffectCollection, and a placeholder for a SystemModel.
        - The instance starts disconnected (self._connected == False) and will be connected automatically when trying to solve a calculation.
    """

    def __init__(
        self,
        timesteps: pd.DatetimeIndex,
        hours_of_last_timestep: float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
    ):
        """
        Initialize a FlowSystem that manages components, buses, effects, and their time-series.

        Parameters:
            timesteps: DatetimeIndex defining the primary timesteps for the system's TimeSeriesCollection.
            hours_of_last_timestep: Duration (in hours) of the final timestep; if None, inferred from timesteps or defaults in TimeSeriesCollection.
            hours_of_previous_timesteps: Scalar or array-like durations (in hours) for the preceding timesteps; used to configure non-uniform timestep lengths.

        Notes:
            Creates an empty registry for components and buses, an empty EffectCollection, and a placeholder for a SystemModel.
            The instance starts disconnected (self._connected == False) and with no active network visualization app.
            This can also be triggered manually with `_connect_network()`.
        """
        self.time_series_collection = TimeSeriesCollection(
            timesteps=timesteps,
            hours_of_last_timestep=hours_of_last_timestep,
            hours_of_previous_timesteps=hours_of_previous_timesteps,
        )

        # defaults:
        self.components: dict[str, Component] = {}
        self.buses: dict[str, Bus] = {}
        self.effects: EffectCollection = EffectCollection()
        self.model: SystemModel | None = None

        self._connected = False

        self._network_app = None

    @classmethod
    def from_dataset(cls, ds: xr.Dataset):
        timesteps_extra = pd.DatetimeIndex(ds.attrs['timesteps_extra'], name='time')
        hours_of_last_timestep = TimeSeriesCollection.calculate_hours_per_timestep(timesteps_extra).isel(time=-1).item()

        flow_system = FlowSystem(
            timesteps=timesteps_extra[:-1],
            hours_of_last_timestep=hours_of_last_timestep,
            hours_of_previous_timesteps=ds.attrs['hours_of_previous_timesteps'],
        )

        structure = fx_io.insert_dataarray({key: ds.attrs[key] for key in ['components', 'buses', 'effects']}, ds)
        flow_system.add_elements(
            *[Bus.from_dict(bus) for bus in structure['buses'].values()]
            + [Effect.from_dict(effect) for effect in structure['effects'].values()]
            + [CLASS_REGISTRY[comp['__class__']].from_dict(comp) for comp in structure['components'].values()]
        )
        return flow_system

    @classmethod
    def from_dict(cls, data: dict) -> FlowSystem:
        """
        Load a FlowSystem from a dictionary.

        Args:
            data: Dictionary containing the FlowSystem data.
        """
        timesteps_extra = pd.DatetimeIndex(data['timesteps_extra'], name='time')
        hours_of_last_timestep = TimeSeriesCollection.calculate_hours_per_timestep(timesteps_extra).isel(time=-1).item()

        flow_system = FlowSystem(
            timesteps=timesteps_extra[:-1],
            hours_of_last_timestep=hours_of_last_timestep,
            hours_of_previous_timesteps=data['hours_of_previous_timesteps'],
        )

        flow_system.add_elements(*[Bus.from_dict(bus) for bus in data['buses'].values()])

        flow_system.add_elements(*[Effect.from_dict(effect) for effect in data['effects'].values()])

        flow_system.add_elements(
            *[CLASS_REGISTRY[comp['__class__']].from_dict(comp) for comp in data['components'].values()]
        )

        flow_system.transform_data()

        return flow_system

    @classmethod
    def from_netcdf(cls, path: str | pathlib.Path):
        """
        Load a FlowSystem from a netcdf file
        """
        return cls.from_dataset(fx_io.load_dataset_from_netcdf(path))

    def add_elements(self, *elements: Element) -> None:
        """
        Add Components(Storages, Boilers, Heatpumps, ...), Buses or Effects to the FlowSystem

        Args:
            *elements: childs of  Element like Boiler, HeatPump, Bus,...
                modeling Elements
        """
        if self._connected:
            warnings.warn(
                'You are adding elements to an already connected FlowSystem. This is not recommended (But it works).',
                stacklevel=2,
            )
            self._connected = False
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

    def to_json(self, path: str | pathlib.Path):
        """
        Saves the flow system to a json file.
        This not meant to be reloaded and recreate the object,
        but rather used to document or compare the flow_system to others.

        Args:
            path: The path to the json file.
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.as_dict('stats'), f, indent=4, ensure_ascii=False)

    def as_dict(self, data_mode: Literal['data', 'name', 'stats'] = 'data') -> dict:
        """Convert the object to a dictionary representation."""
        data = {
            'components': {
                comp.label: comp.to_dict()
                for comp in sorted(self.components.values(), key=lambda component: component.label.upper())
            },
            'buses': {
                bus.label: bus.to_dict() for bus in sorted(self.buses.values(), key=lambda bus: bus.label.upper())
            },
            'effects': {
                effect.label: effect.to_dict()
                for effect in sorted(self.effects, key=lambda effect: effect.label.upper())
            },
            'timesteps_extra': [date.isoformat() for date in self.time_series_collection.timesteps_extra],
            'hours_of_previous_timesteps': self.time_series_collection.hours_of_previous_timesteps,
        }
        if data_mode == 'data':
            return fx_io.replace_timeseries(data, 'data')
        elif data_mode == 'stats':
            return fx_io.remove_none_and_empty(fx_io.replace_timeseries(data, data_mode))
        return fx_io.replace_timeseries(data, data_mode)

    def as_dataset(self, constants_in_dataset: bool = False) -> xr.Dataset:
        """
        Convert the FlowSystem to a xarray Dataset.

        Args:
            constants_in_dataset: If True, constants are included as Dataset variables.
        """
        ds = self.time_series_collection.to_dataset(include_constants=constants_in_dataset)
        ds.attrs = self.as_dict(data_mode='name')
        return ds

    def to_netcdf(self, path: str | pathlib.Path, compression: int = 0, constants_in_dataset: bool = True) -> None:

        """
        Saves the FlowSystem to a netCDF file.

        Args:
            path: The path to the netCDF file.
            compression: The compression level to use when saving the file.
            constants_in_dataset: If True, constants are included as Dataset variables.
        """
        ds = self.as_dataset(constants_in_dataset=constants_in_dataset)
        fx_io.save_dataset_to_netcdf(ds, path, compression=compression, engine='h5netcdf')
        logger.info(f'Saved FlowSystem to {path}')

    def plot_network(
        self,
        path: bool | str | pathlib.Path = 'flow_system.html',
        controls: bool
        | list[
            Literal['nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer']
        ] = True,
        show: bool = False,
    ) -> pyvis.network.Network | None:
        """
        Visualizes the network structure of a FlowSystem using PyVis, saving it as an interactive HTML file.

        Args:
            path: Path to save the HTML visualization.
                - `False`: Visualization is created but not saved.
                - `str` or `Path`: Specifies file path (default: 'flow_system.html').
            controls: UI controls to add to the visualization.
                - `True`: Enables all available controls.
                - `List`: Specify controls, e.g., ['nodes', 'layout'].
                - Options: 'nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer'.
            show: Whether to open the visualization in the web browser.

        Returns:
        - 'pyvis.network.Network' | None: The `Network` instance representing the visualization, or `None` if `pyvis` is not installed.

        Examples:
            >>> flow_system.plot_network()
            >>> flow_system.plot_network(show=False)
            >>> flow_system.plot_network(path='output/custom_network.html', controls=['nodes', 'layout'])

        Notes:
        - This function requires `pyvis`. If not installed, the function prints a warning and returns `None`.
        - Nodes are styled based on type (e.g., circles for buses, boxes for components) and annotated with node information.
        """
        from . import plotting

        node_infos, edge_infos = self.network_infos()
        return plotting.plot_network(node_infos, edge_infos, path, controls, show)

    def start_network_app(self):
        """Visualizes the network structure of a FlowSystem using Dash, Cytoscape, and networkx.
        Requires optional dependencies: dash, dash-cytoscape, dash-daq, networkx, flask, werkzeug.
        """
        from .network_app import DASH_CYTOSCAPE_AVAILABLE, VISUALIZATION_ERROR, flow_graph, shownetwork

        warnings.warn(
            'The network visualization is still experimental and might change in the future.',
            stacklevel=2,
            category=UserWarning,
        )

        if not DASH_CYTOSCAPE_AVAILABLE:
            raise ImportError(
                f'Network visualization requires optional dependencies. '
                f'Install with: `pip install flixopt[network_viz]`, `pip install flixopt[full]` '
                f'or: `pip install dash dash-cytoscape dash-daq networkx werkzeug`. '
                f'Original error: {VISUALIZATION_ERROR}'
            )

        if not self._connected:
            self._connect_network()

        if self._network_app is not None:
            logger.warning('The network app is already running. Restarting it.')
            self.stop_network_app()

        self._network_app = shownetwork(flow_graph(self))

    def stop_network_app(self):
        """Stop the network visualization server."""
        from .network_app import DASH_CYTOSCAPE_AVAILABLE, VISUALIZATION_ERROR

        if not DASH_CYTOSCAPE_AVAILABLE:
            raise ImportError(
                f'Network visualization requires optional dependencies. '
                f'Install with: `pip install flixopt[network_viz]`, `pip install flixopt[full]` '
                f'or: `pip install dash dash-cytoscape dash-daq networkx werkzeug`. '
                f'Original error: {VISUALIZATION_ERROR}'
            )

        if self._network_app is None:
            logger.warning('No network app is currently running. Cant stop it')
            return

        try:
            logger.info('Stopping network visualization server...')
            self._network_app.server_instance.shutdown()
            logger.info('Network visualization stopped.')
        except Exception as e:
            logger.error(f'Failed to stop the network visualization app: {e}')
        finally:
            self._network_app = None

    def network_infos(self) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
        if not self._connected:
            self._connect_network()
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

    def transform_data(self):
        if not self._connected:
            self._connect_network()
        for element in self.all_elements.values():
            element.transform_data(self)

    def create_time_series(
        self,
        name: str,
        data: NumericData | TimeSeriesData | TimeSeries | None,
        needs_extra_timestep: bool = False,
    ) -> TimeSeries | None:
        """
        Tries to create a TimeSeries from NumericData Data and adds it to the time_series_collection
        If the data already is a TimeSeries, nothing happens and the TimeSeries gets reset and returned
        If the data is a TimeSeriesData, it is converted to a TimeSeries, and the aggregation weights are applied.
        If the data is None, nothing happens.
        """

        if data is None:
            return None
        elif isinstance(data, TimeSeries):
            data.restore_data()
            if data in self.time_series_collection:
                return data
            return self.time_series_collection.create_time_series(
                data=data.active_data, name=name, needs_extra_timestep=needs_extra_timestep
            )
        return self.time_series_collection.create_time_series(
            data=data, name=name, needs_extra_timestep=needs_extra_timestep
        )

    def create_effect_time_series(
        self,
        label_prefix: str | None,
        effect_values: EffectValuesUser,
        label_suffix: str | None = None,
    ) -> EffectTimeSeries | None:
        """
        Transform EffectValues to EffectTimeSeries.
        Creates a TimeSeries for each key in the nested_values dictionary, using the value as the data.

        The resulting label of the TimeSeries is the label of the parent_element,
        followed by the label of the Effect in the nested_values and the label_suffix.
        If the key in the EffectValues is None, the alias 'Standard_Effect' is used
        """
        effect_values_dict: EffectValuesDict | None = self.effects.create_effect_values_dict(effect_values)
        if effect_values_dict is None:
            return None

        return {
            effect: self.create_time_series('|'.join(filter(None, [label_prefix, effect, label_suffix])), value)
            for effect, value in effect_values_dict.items()
        }

    def create_model(self) -> SystemModel:
        if not self._connected:
            raise RuntimeError('FlowSystem is not connected. Call FlowSystem.connect() first.')
        self.model = SystemModel(self)
        return self.model

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
                    warnings.warn(
                        f'The Bus {flow._bus_object.label} was added to the FlowSystem from {flow.label_full}.'
                        f'This is deprecated and will be removed in the future. '
                        f'Please pass the Bus.label to the Flow and the Bus to the FlowSystem instead.',
                        DeprecationWarning,
                        stacklevel=1,
                    )
                    self._add_buses(flow._bus_object)

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
        self._connected = True

    def __repr__(self):
        return f'<{self.__class__.__name__} with {len(self.components)} components and {len(self.effects)} effects>'

    def __str__(self):
        with StringIO() as output_buffer:
            console = Console(file=output_buffer, width=1000)  # Adjust width as needed
            console.print(Pretty(self.as_dict('stats'), expand_all=True, indent_guides=True))
            value = output_buffer.getvalue()
        return value

    @property
    def flows(self) -> dict[str, Flow]:
        set_of_flows = {flow for comp in self.components.values() for flow in comp.inputs + comp.outputs}
        return {flow.label_full: flow for flow in set_of_flows}

    @property
    def all_elements(self) -> dict[str, Element]:
        return {**self.components, **self.effects.effects, **self.flows, **self.buses}
