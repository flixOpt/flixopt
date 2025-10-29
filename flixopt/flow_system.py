"""
This module contains the FlowSystem class, which is used to collect instances of many other classes by the end User.
"""

from __future__ import annotations

import logging
import warnings
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import pandas as pd
import xarray as xr

from . import io as fx_io
from .config import CONFIG
from .core import (
    ConversionError,
    DataConverter,
    FlowSystemDimensions,
    PeriodicData,
    PeriodicDataUser,
    TemporalData,
    TemporalDataUser,
    TimeSeriesData,
)
from .effects import (
    Effect,
    EffectCollection,
    PeriodicEffects,
    PeriodicEffectsUser,
    TemporalEffects,
    TemporalEffectsUser,
)
from .elements import Bus, Component, Flow
from .structure import CompositeContainerMixin, Element, ElementContainer, FlowSystemModel, Interface

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Collection

    import pyvis

logger = logging.getLogger('flixopt')


class FlowSystem(Interface, CompositeContainerMixin[Element]):
    """
    A FlowSystem organizes the high level Elements (Components, Buses & Effects).

    This is the main container class that users work with to build and manage their System.

    Args:
        timesteps: The timesteps of the model.
        periods: The periods of the model.
        scenarios: The scenarios of the model.
        hours_of_last_timestep: The duration of the last time step. Uses the last time interval if not specified
        hours_of_previous_timesteps: The duration of previous timesteps.
            If None, the first time increment of time_series is used.
            This is needed to calculate previous durations (for example consecutive_on_hours).
            If you use an array, take care that its long enough to cover all previous values!
        weights: The weights of each period and scenario. If None, all scenarios have the same weight (normalized to 1).
            Its recommended to normalize the weights to sum up to 1.
        scenario_independent_sizes: Controls whether investment sizes are equalized across scenarios.
            - True: All sizes are shared/equalized across scenarios
            - False: All sizes are optimized separately per scenario
            - list[str]: Only specified components (by label_full) are equalized across scenarios
        scenario_independent_flow_rates: Controls whether flow rates are equalized across scenarios.
            - True: All flow rates are shared/equalized across scenarios
            - False: All flow rates are optimized separately per scenario
            - list[str]: Only specified flows (by label_full) are equalized across scenarios

    Notes:
        - Creates an empty registry for components and buses, an empty EffectCollection, and a placeholder for a SystemModel.
        - The instance starts disconnected (self._connected_and_transformed == False) and will be
        connected_and_transformed automatically when trying to solve a calculation.
    """

    def __init__(
        self,
        timesteps: pd.DatetimeIndex,
        periods: pd.Index | None = None,
        scenarios: pd.Index | None = None,
        hours_of_last_timestep: float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
        weights: PeriodicDataUser | None = None,
        scenario_independent_sizes: bool | list[str] = True,
        scenario_independent_flow_rates: bool | list[str] = False,
    ):
        self.timesteps = self._validate_timesteps(timesteps)
        self.timesteps_extra = self._create_timesteps_with_extra(self.timesteps, hours_of_last_timestep)
        self.hours_of_previous_timesteps = self._calculate_hours_of_previous_timesteps(
            self.timesteps, hours_of_previous_timesteps
        )

        self.periods = None if periods is None else self._validate_periods(periods)
        self.scenarios = None if scenarios is None else self._validate_scenarios(scenarios)

        self.weights = weights

        hours_per_timestep = self.calculate_hours_per_timestep(self.timesteps_extra)

        self.hours_of_last_timestep = hours_per_timestep[-1].item()

        self.hours_per_timestep = self.fit_to_model_coords('hours_per_timestep', hours_per_timestep)

        # Element collections
        self.components: ElementContainer[Component] = ElementContainer(element_type_name='components')
        self.buses: ElementContainer[Bus] = ElementContainer(element_type_name='buses')
        self.effects: EffectCollection = EffectCollection()
        self.model: FlowSystemModel | None = None

        self._connected_and_transformed = False
        self._used_in_calculation = False

        self._network_app = None
        self._flows_cache: ElementContainer[Flow] | None = None

        # Use properties to validate and store scenario dimension settings
        self.scenario_independent_sizes = scenario_independent_sizes
        self.scenario_independent_flow_rates = scenario_independent_flow_rates

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
    def _validate_scenarios(scenarios: pd.Index) -> pd.Index:
        """
        Validate and prepare scenario index.

        Args:
            scenarios: The scenario index to validate
        """
        if not isinstance(scenarios, pd.Index) or len(scenarios) == 0:
            raise ConversionError('Scenarios must be a non-empty Index')

        if scenarios.name != 'scenario':
            scenarios = scenarios.rename('scenario')

        return scenarios

    @staticmethod
    def _validate_periods(periods: pd.Index) -> pd.Index:
        """
        Validate and prepare period index.

        Args:
            periods: The period index to validate
        """
        if not isinstance(periods, pd.Index) or len(periods) == 0:
            raise ConversionError(f'Periods must be a non-empty Index. Got {periods}')

        if not (
            periods.dtype.kind == 'i'  # integer dtype
            and periods.is_monotonic_increasing  # rising
            and periods.is_unique
        ):
            raise ConversionError(f'Periods must be a monotonically increasing and unique Index. Got {periods}')

        if periods.name != 'period':
            periods = periods.rename('period')

        return periods

    @staticmethod
    def _create_timesteps_with_extra(
        timesteps: pd.DatetimeIndex, hours_of_last_timestep: float | None
    ) -> pd.DatetimeIndex:
        """Create timesteps with an extra step at the end."""
        if hours_of_last_timestep is None:
            hours_of_last_timestep = (timesteps[-1] - timesteps[-2]) / pd.Timedelta(hours=1)

        last_date = pd.DatetimeIndex([timesteps[-1] + pd.Timedelta(hours=hours_of_last_timestep)], name='time')
        return pd.DatetimeIndex(timesteps.append(last_date), name='time')

    @staticmethod
    def calculate_hours_per_timestep(timesteps_extra: pd.DatetimeIndex) -> xr.DataArray:
        """Calculate duration of each timestep as a 1D DataArray."""
        hours_per_step = np.diff(timesteps_extra) / pd.Timedelta(hours=1)
        return xr.DataArray(
            hours_per_step, coords={'time': timesteps_extra[:-1]}, dims='time', name='hours_per_timestep'
        )

    @staticmethod
    def _calculate_hours_of_previous_timesteps(
        timesteps: pd.DatetimeIndex, hours_of_previous_timesteps: float | np.ndarray | None
    ) -> float | np.ndarray:
        """Calculate duration of regular timesteps."""
        if hours_of_previous_timesteps is not None:
            return hours_of_previous_timesteps
        # Calculate from the first interval
        first_interval = timesteps[1] - timesteps[0]
        return first_interval.total_seconds() / 3600  # Convert to hours

    def _create_reference_structure(self) -> tuple[dict, dict[str, xr.DataArray]]:
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
        for effect in self.effects.values():
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
        if not self.connected_and_transformed:
            logger.warning('FlowSystem is not connected_and_transformed. Connecting and transforming data now.')
            self.connect_and_transform()

        return super().to_dataset()

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> FlowSystem:
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

        # Create arrays dictionary from dataset variables
        arrays_dict = {name: array for name, array in ds.data_vars.items()}

        # Create FlowSystem instance with constructor parameters
        flow_system = cls(
            timesteps=ds.indexes['time'],
            periods=ds.indexes.get('period'),
            scenarios=ds.indexes.get('scenario'),
            weights=cls._resolve_dataarray_reference(reference_structure['weights'], arrays_dict)
            if 'weights' in reference_structure
            else None,
            hours_of_last_timestep=reference_structure.get('hours_of_last_timestep'),
            hours_of_previous_timesteps=reference_structure.get('hours_of_previous_timesteps'),
            scenario_independent_sizes=reference_structure.get('scenario_independent_sizes', True),
            scenario_independent_flow_rates=reference_structure.get('scenario_independent_flow_rates', False),
        )

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

    def to_netcdf(self, path: str | pathlib.Path, compression: int = 0):
        """
        Save the FlowSystem to a NetCDF file.
        Ensures FlowSystem is connected before saving.

        Args:
            path: The path to the netCDF file.
            compression: The compression level to use when saving the file.
        """
        if not self.connected_and_transformed:
            logger.warning('FlowSystem is not connected. Calling connect_and_transform() now.')
            self.connect_and_transform()

        super().to_netcdf(path, compression)
        logger.info(f'Saved FlowSystem to {path}')

    def get_structure(self, clean: bool = False, stats: bool = False) -> dict:
        """
        Get FlowSystem structure.
        Ensures FlowSystem is connected before getting structure.

        Args:
            clean: If True, remove None and empty dicts and lists.
            stats: If True, replace DataArray references with statistics
        """
        if not self.connected_and_transformed:
            logger.warning('FlowSystem is not connected. Calling connect_and_transform() now.')
            self.connect_and_transform()

        return super().get_structure(clean, stats)

    def to_json(self, path: str | pathlib.Path):
        """
        Save the flow system to a JSON file.
        Ensures FlowSystem is connected before saving.

        Args:
            path: The path to the JSON file.
        """
        if not self.connected_and_transformed:
            logger.warning(
                'FlowSystem needs to be connected and transformed before saving to JSON. Calling connect_and_transform() now.'
            )
            self.connect_and_transform()

        super().to_json(path)

    def fit_to_model_coords(
        self,
        name: str,
        data: TemporalDataUser | PeriodicDataUser | None,
        dims: Collection[FlowSystemDimensions] | None = None,
    ) -> TemporalData | PeriodicData | None:
        """
        Fit data to model coordinate system (currently time, but extensible).

        Args:
            name: Name of the data
            data: Data to fit to model coordinates
            dims: Collection of dimension names to use for fitting. If None, all dimensions are used.

        Returns:
            xr.DataArray aligned to model coordinate system. If data is None, returns None.
        """
        if data is None:
            return None

        coords = self.coords

        if dims is not None:
            coords = {k: coords[k] for k in dims if k in coords}

        # Rest of your method stays the same, just pass coords
        if isinstance(data, TimeSeriesData):
            try:
                data.name = name  # Set name of previous object!
                return data.fit_to_coords(coords)
            except ConversionError as e:
                raise ConversionError(
                    f'Could not convert time series data "{name}" to DataArray:\n{data}\nOriginal Error: {e}'
                ) from e

        try:
            return DataConverter.to_dataarray(data, coords=coords).rename(name)
        except ConversionError as e:
            raise ConversionError(f'Could not convert data "{name}" to DataArray:\n{data}\nOriginal Error: {e}') from e

    def fit_effects_to_model_coords(
        self,
        label_prefix: str | None,
        effect_values: TemporalEffectsUser | PeriodicEffectsUser | None,
        label_suffix: str | None = None,
        dims: Collection[FlowSystemDimensions] | None = None,
        delimiter: str = '|',
    ) -> TemporalEffects | PeriodicEffects | None:
        """
        Transform EffectValues from the user to Internal Datatypes aligned with model coordinates.
        """
        if effect_values is None:
            return None

        effect_values_dict = self.effects.create_effect_values_dict(effect_values)

        return {
            effect: self.fit_to_model_coords(
                str(delimiter).join(filter(None, [label_prefix, effect, label_suffix])),
                value,
                dims=dims,
            )
            for effect, value in effect_values_dict.items()
        }

    def connect_and_transform(self):
        """Transform data for all elements using the new simplified approach."""
        if self.connected_and_transformed:
            logger.debug('FlowSystem already connected and transformed')
            return

        self.weights = self.fit_to_model_coords('weights', self.weights, dims=['period', 'scenario'])

        self._connect_network()
        for element in chain(self.components.values(), self.effects.values(), self.buses.values()):
            element.transform_data(self)
        self._connected_and_transformed = True

    def add_elements(self, *elements: Element) -> None:
        """
        Add Components(Storages, Boilers, Heatpumps, ...), Buses or Effects to the FlowSystem

        Args:
            *elements: childs of  Element like Boiler, HeatPump, Bus,...
                modeling Elements
        """
        if self.connected_and_transformed:
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

    def create_model(self, normalize_weights: bool = True) -> FlowSystemModel:
        """
        Create a linopy model from the FlowSystem.

        Args:
            normalize_weights: Whether to automatically normalize the weights (periods and scenarios) to sum up to 1 when solving.
        """
        if not self.connected_and_transformed:
            raise RuntimeError(
                'FlowSystem is not connected_and_transformed. Call FlowSystem.connect_and_transform() first.'
            )
        self.model = FlowSystemModel(self, normalize_weights)
        return self.model

    def plot_network(
        self,
        path: bool | str | pathlib.Path = 'flow_system.html',
        controls: bool
        | list[
            Literal['nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer']
        ] = True,
        show: bool | None = None,
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
        return plotting.plot_network(
            node_infos, edge_infos, path, controls, show if show is not None else CONFIG.Plotting.default_show
        )

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

        if not self._connected_and_transformed:
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
            logger.warning("No network app is currently running. Can't stop it")
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
        if not self.connected_and_transformed:
            self.connect_and_transform()
        nodes = {
            node.label_full: {
                'label': node.label,
                'class': 'Bus' if isinstance(node, Bus) else 'Component',
                'infos': node.__str__(),
            }
            for node in chain(self.components.values(), self.buses.values())
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
        # check if name is already used:
        if element.label_full in self:
            raise ValueError(f'Label of Element {element.label_full} already used in another element!')

    def _add_effects(self, *args: Effect) -> None:
        self.effects.add_effects(*args)

    def _add_components(self, *components: Component) -> None:
        for new_component in list(components):
            logger.info(f'Registered new Component: {new_component.label_full}')
            self._check_if_element_is_unique(new_component)  # check if already exists:
            self.components.add(new_component)  # Add to existing components
            self._flows_cache = None  # Invalidate flows cache

    def _add_buses(self, *buses: Bus):
        for new_bus in list(buses):
            logger.info(f'Registered new Bus: {new_bus.label_full}')
            self._check_if_element_is_unique(new_bus)  # check if already exists:
            self.buses.add(new_bus)  # Add to existing buses
            self._flows_cache = None  # Invalidate flows cache

    def _connect_network(self):
        """Connects the network of components and buses. Can be rerun without changes if no elements were added"""
        for component in self.components.values():
            for flow in component.inputs + component.outputs:
                flow.component = component.label_full
                flow.is_input_in_component = True if flow in component.inputs else False

                # Add Bus if not already added (deprecated)
                if flow._bus_object is not None and flow._bus_object.label_full not in self.buses:
                    warnings.warn(
                        f'The Bus {flow._bus_object.label_full} was added to the FlowSystem from {flow.label_full}.'
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

    def __repr__(self) -> str:
        """Return a detailed string representation showing all containers."""
        r = fx_io.format_title_with_underline('FlowSystem', '=')

        # Timestep info
        time_period = f'{self.timesteps[0].date()} to {self.timesteps[-1].date()}'
        freq_str = str(self.timesteps.freq).replace('<', '').replace('>', '') if self.timesteps.freq else 'irregular'
        r += f'Timesteps: {len(self.timesteps)} ({freq_str}) [{time_period}]\n'

        # Add periods if present
        if self.periods is not None:
            period_names = ', '.join(str(p) for p in self.periods[:3])
            if len(self.periods) > 3:
                period_names += f' ... (+{len(self.periods) - 3} more)'
            r += f'Periods: {len(self.periods)} ({period_names})\n'
        else:
            r += 'Periods: None\n'

        # Add scenarios if present
        if self.scenarios is not None:
            scenario_names = ', '.join(str(s) for s in self.scenarios[:3])
            if len(self.scenarios) > 3:
                scenario_names += f' ... (+{len(self.scenarios) - 3} more)'
            r += f'Scenarios: {len(self.scenarios)} ({scenario_names})\n'
        else:
            r += 'Scenarios: None\n'

        # Add status
        status = '✓' if self.connected_and_transformed else '⚠'
        r += f'Status: {status}\n'

        # Add grouped container view
        r += '\n' + self._format_grouped_containers()

        return r

    def __eq__(self, other: FlowSystem):
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

    def _get_container_groups(self) -> dict[str, dict]:
        """Return ordered container groups for CompositeContainerMixin."""
        return {
            'Components': dict(self.components),
            'Buses': dict(self.buses),
            'Effects': dict(self.effects),
            'Flows': dict(self.flows),
        }

    @property
    def flows(self) -> ElementContainer[Flow]:
        if self._flows_cache is None:
            flows = [f for c in self.components.values() for f in c.inputs + c.outputs]
            # Deduplicate by id and sort for reproducibility
            flows = sorted({id(f): f for f in flows}.values(), key=lambda f: f.label_full.lower())
            self._flows_cache = ElementContainer({f.label_full: f for f in flows}, element_type_name='flows')
        return self._flows_cache

    @property
    def all_elements(self) -> dict[str, Element]:
        """
        Get all elements as a dictionary.

        .. deprecated:: 3.2.0
            Use dict-like interface instead: `flow_system['element']`, `'element' in flow_system`,
            `flow_system.keys()`, `flow_system.values()`, or `flow_system.items()`.
            This property will be removed in v4.0.0.

        Returns:
            Dictionary mapping element labels to element objects.
        """
        warnings.warn(
            "The 'all_elements' property is deprecated. Use dict-like interface instead: "
            "flow_system['element'], 'element' in flow_system, flow_system.keys(), "
            'flow_system.values(), or flow_system.items(). '
            'This property will be removed in v4.0.0.',
            DeprecationWarning,
            stacklevel=2,
        )
        return {**self.components, **self.effects, **self.flows, **self.buses}

    @property
    def coords(self) -> dict[FlowSystemDimensions, pd.Index]:
        active_coords = {'time': self.timesteps}
        if self.periods is not None:
            active_coords['period'] = self.periods
        if self.scenarios is not None:
            active_coords['scenario'] = self.scenarios
        return active_coords

    @property
    def used_in_calculation(self) -> bool:
        return self._used_in_calculation

    def _validate_scenario_parameter(self, value: bool | list[str], param_name: str, element_type: str) -> None:
        """
        Validate scenario parameter value.

        Args:
            value: The value to validate
            param_name: Name of the parameter (for error messages)
            element_type: Type of elements expected in list (e.g., 'component label_full', 'flow label_full')

        Raises:
            TypeError: If value is not bool or list[str]
            ValueError: If list contains non-string elements
        """
        if isinstance(value, bool):
            return  # Valid
        elif isinstance(value, list):
            if not all(isinstance(item, str) for item in value):
                raise ValueError(f'{param_name} list must contain only strings ({element_type} values)')
        else:
            raise TypeError(f'{param_name} must be bool or list[str], got {type(value).__name__}')

    @property
    def scenario_independent_sizes(self) -> bool | list[str]:
        """
        Controls whether investment sizes are equalized across scenarios.

        Returns:
            bool or list[str]: Configuration for scenario-independent sizing
        """
        return self._scenario_independent_sizes

    @scenario_independent_sizes.setter
    def scenario_independent_sizes(self, value: bool | list[str]) -> None:
        """
        Set whether investment sizes should be equalized across scenarios.

        Args:
            value: True (all equalized), False (all vary), or list of component label_full strings to equalize

        Raises:
            TypeError: If value is not bool or list[str]
            ValueError: If list contains non-string elements
        """
        self._validate_scenario_parameter(value, 'scenario_independent_sizes', 'Element.label_full')
        self._scenario_independent_sizes = value

    @property
    def scenario_independent_flow_rates(self) -> bool | list[str]:
        """
        Controls whether flow rates are equalized across scenarios.

        Returns:
            bool or list[str]: Configuration for scenario-independent flow rates
        """
        return self._scenario_independent_flow_rates

    @scenario_independent_flow_rates.setter
    def scenario_independent_flow_rates(self, value: bool | list[str]) -> None:
        """
        Set whether flow rates should be equalized across scenarios.

        Args:
            value: True (all equalized), False (all vary), or list of flow label_full strings to equalize

        Raises:
            TypeError: If value is not bool or list[str]
            ValueError: If list contains non-string elements
        """
        self._validate_scenario_parameter(value, 'scenario_independent_flow_rates', 'Flow.label_full')
        self._scenario_independent_flow_rates = value

    def sel(
        self,
        time: str | slice | list[str] | pd.Timestamp | pd.DatetimeIndex | None = None,
        period: int | slice | list[int] | pd.Index | None = None,
        scenario: str | slice | list[str] | pd.Index | None = None,
    ) -> FlowSystem:
        """
        Select a subset of the flowsystem by the time coordinate.

        Args:
            time: Time selection (e.g., slice('2023-01-01', '2023-12-31'), '2023-06-15', or list of times)
            period: Period selection (e.g., slice(2023, 2024), or list of periods)
            scenario: Scenario selection (e.g., slice('scenario1', 'scenario2'), or list of scenarios)

        Returns:
            FlowSystem: New FlowSystem with selected data
        """
        if not self.connected_and_transformed:
            self.connect_and_transform()

        ds = self.to_dataset()

        # Build indexers dict from non-None parameters
        indexers = {}
        if time is not None:
            indexers['time'] = time
        if period is not None:
            indexers['period'] = period
        if scenario is not None:
            indexers['scenario'] = scenario

        if not indexers:
            return self.copy()  # Return a copy when no selection

        selected_dataset = ds.sel(**indexers)
        return self.__class__.from_dataset(selected_dataset)

    def isel(
        self,
        time: int | slice | list[int] | None = None,
        period: int | slice | list[int] | None = None,
        scenario: int | slice | list[int] | None = None,
    ) -> FlowSystem:
        """
        Select a subset of the flowsystem by integer indices.

        Args:
            time: Time selection by integer index (e.g., slice(0, 100), 50, or [0, 5, 10])
            period: Period selection by integer index (e.g., slice(0, 100), 50, or [0, 5, 10])
            scenario: Scenario selection by integer index (e.g., slice(0, 3), 50, or [0, 5, 10])

        Returns:
            FlowSystem: New FlowSystem with selected data
        """
        if not self.connected_and_transformed:
            self.connect_and_transform()

        ds = self.to_dataset()

        # Build indexers dict from non-None parameters
        indexers = {}
        if time is not None:
            indexers['time'] = time
        if period is not None:
            indexers['period'] = period
        if scenario is not None:
            indexers['scenario'] = scenario

        if not indexers:
            return self.copy()  # Return a copy when no selection

        selected_dataset = ds.isel(**indexers)
        return self.__class__.from_dataset(selected_dataset)

    def resample(
        self,
        time: str,
        method: Literal['mean', 'sum', 'max', 'min', 'first', 'last', 'std', 'var', 'median', 'count'] = 'mean',
        **kwargs: Any,
    ) -> FlowSystem:
        """
        Create a resampled FlowSystem by resampling data along the time dimension (like xr.Dataset.resample()).
        Only resamples data variables that have a time dimension.

        Args:
            time: Resampling frequency (e.g., '3h', '2D', '1M')
            method: Resampling method. Recommended: 'mean', 'first', 'last', 'max', 'min'
            **kwargs: Additional arguments passed to xarray.resample()

        Returns:
            FlowSystem: New FlowSystem with resampled data
        """
        if not self.connected_and_transformed:
            self.connect_and_transform()

        dataset = self.to_dataset()

        # Separate variables with and without time dimension
        time_vars = {}
        non_time_vars = {}

        for var_name, var in dataset.data_vars.items():
            if 'time' in var.dims:
                time_vars[var_name] = var
            else:
                non_time_vars[var_name] = var

        # Only resample variables that have time dimension
        time_dataset = dataset[list(time_vars.keys())]
        resampler = time_dataset.resample(time=time, **kwargs)

        if hasattr(resampler, method):
            resampled_time_data = getattr(resampler, method)()
        else:
            available_methods = ['mean', 'sum', 'max', 'min', 'first', 'last', 'std', 'var', 'median', 'count']
            raise ValueError(f'Unsupported resampling method: {method}. Available: {available_methods}')

        # Combine resampled time variables with non-time variables
        if non_time_vars:
            non_time_dataset = dataset[list(non_time_vars.keys())]
            resampled_dataset = xr.merge([resampled_time_data, non_time_dataset])
        else:
            resampled_dataset = resampled_time_data

        return self.__class__.from_dataset(resampled_dataset)

    @property
    def connected_and_transformed(self) -> bool:
        return self._connected_and_transformed
