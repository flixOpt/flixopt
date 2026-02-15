"""
Topology accessor for FlowSystem.

This module provides the TopologyAccessor class that enables the
`flow_system.topology` pattern for network structure inspection and visualization.
"""

from __future__ import annotations

import logging
import pathlib
import warnings
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import plotly.graph_objects as go
import xarray as xr

from .color_processing import ColorType, hex_to_rgba, process_colors
from .config import CONFIG, DEPRECATION_REMOVAL_VERSION
from .flow_system_status import FlowSystemStatus
from .plot_result import PlotResult

if TYPE_CHECKING:
    import pyvis

    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


def _plot_network(
    node_infos: dict,
    edge_infos: dict,
    path: str | pathlib.Path | None = None,
    controls: bool
    | list[
        Literal['nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer']
    ] = True,
    show: bool = False,
) -> pyvis.network.Network | None:
    """Visualize network structure using PyVis.

    Args:
        node_infos: Dictionary of node information.
        edge_infos: Dictionary of edge information.
        path: Path to save HTML visualization.
        controls: UI controls to add. True for all, or list of specific controls.
        show: Whether to open in browser.

    Returns:
        Network instance, or None if pyvis not installed.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        logger.critical("Plotting the flow system network was not possible. Please install pyvis: 'pip install pyvis'")
        return None

    net = Network(directed=True, height='100%' if controls is False else '800px', font_color='white')

    for node_id, node in node_infos.items():
        net.add_node(
            node_id,
            label=node['label'],
            shape={'Bus': 'circle', 'Component': 'box'}[node['class']],
            color={'Bus': '#393E46', 'Component': '#00ADB5'}[node['class']],
            title=node['infos'].replace(')', '\n)'),
            font={'size': 14},
        )

    for edge in edge_infos.values():
        # Use carrier color if available, otherwise default gray
        edge_color = edge.get('carrier_color', '#222831') or '#222831'
        net.add_edge(
            edge['start'],
            edge['end'],
            label=edge['label'],
            title=edge['infos'].replace(')', '\n)'),
            font={'color': '#4D4D4D', 'size': 14},
            color=edge_color,
        )

    net.barnes_hut(central_gravity=0.8, spring_length=50, spring_strength=0.05, gravity=-10000)

    if controls:
        net.show_buttons(filter_=controls)
    if not show and not path:
        return net
    elif path:
        path = pathlib.Path(path) if isinstance(path, str) else path
        net.write_html(path.as_posix())
    elif show:
        path = pathlib.Path('network.html')
        net.write_html(path.as_posix())

    if show:
        try:
            import webbrowser

            worked = webbrowser.open(f'file://{path.resolve()}', 2)
            if not worked:
                logger.error(f'Showing the network in the Browser went wrong. Open it manually. Its saved under {path}')
        except Exception as e:
            logger.error(
                f'Showing the network in the Browser went wrong. Open it manually. Its saved under {path}: {e}'
            )

    return net


class TopologyAccessor:
    """
    Accessor for network topology inspection and visualization on FlowSystem.

    This class provides the topology API for FlowSystem, accessible via
    `flow_system.topology`. It offers methods to inspect the network structure
    and visualize it.

    Examples:
        Visualize the network:

        >>> flow_system.topology.plot()
        >>> flow_system.topology.plot(path='my_network.html', show=True)

        Interactive visualization:

        >>> flow_system.topology.start_app()
        >>> # ... interact with the visualization ...
        >>> flow_system.topology.stop_app()

        Get network structure info:

        >>> nodes, edges = flow_system.topology.infos()
    """

    def __init__(self, flow_system: FlowSystem) -> None:
        """
        Initialize the accessor with a reference to the FlowSystem.

        Args:
            flow_system: The FlowSystem to inspect.
        """
        self._fs = flow_system

        # Cached color mappings (lazily initialized, invalidated by set_*_color methods)
        self._carrier_colors: dict[str, str] | None = None
        self._component_colors: dict[str, str] | None = None
        self._flow_colors: dict[str, str] | None = None
        self._bus_colors: dict[str, str] | None = None

    @property
    def carrier_colors(self) -> dict[str, str]:
        """Cached mapping of carrier name to hex color.

        Returns:
            Dict mapping carrier names (lowercase) to hex color strings.
            Only carriers with a color defined are included.

        Examples:
            >>> fs.topology.carrier_colors
            {'electricity': '#FECB52', 'heat': '#D62728', 'gas': '#1F77B4'}
        """
        if self._carrier_colors is None:
            self._carrier_colors = {name: carrier.color for name, carrier in self._fs.carriers.items() if carrier.color}
        return self._carrier_colors

    @property
    def component_colors(self) -> dict[str, str]:
        """Cached mapping of component id to hex color.

        Returns:
            Dict mapping component ids to hex color strings.
            Only components with a color defined are included.

        Examples:
            >>> fs.topology.component_colors
            {'Boiler': '#1f77b4', 'CHP': '#ff7f0e', 'HeatPump': '#2ca02c'}
        """
        if self._component_colors is None:
            self._component_colors = {label: comp.color for label, comp in self._fs.components.items() if comp.color}
        return self._component_colors

    @property
    def flow_colors(self) -> dict[str, str]:
        """Cached mapping of flow id to hex color (from parent component).

        Flow colors are derived from their parent component's color.

        Returns:
            Dict mapping flow ids (e.g., 'Boiler(Q_th)') to hex color strings.
            Only flows whose parent component has a color defined are included.

        Examples:
            >>> fs.topology.flow_colors
            {'Boiler(Q_th)': '#1f77b4', 'Boiler(Q_fu)': '#1f77b4', 'CHP(Q_th)': '#ff7f0e'}
        """
        if self._flow_colors is None:
            component_colors = self.component_colors
            self._flow_colors = {}
            for flow in self._fs.flows.values():
                if flow.component in component_colors:
                    self._flow_colors[flow.id] = component_colors[flow.component]
        return self._flow_colors

    @property
    def bus_colors(self) -> dict[str, str]:
        """Cached mapping of bus id to hex color (from carrier).

        Bus colors are derived from their associated carrier's color.

        Returns:
            Dict mapping bus ids to hex color strings.
            Only buses with a carrier that has a color defined are included.

        Examples:
            >>> fs.topology.bus_colors
            {'ElectricityBus': '#FECB52', 'HeatBus': '#D62728'}
        """
        if self._bus_colors is None:
            carrier_colors = self.carrier_colors
            self._bus_colors = {}
            for label, bus in self._fs.buses.items():
                if bus.carrier:
                    color = carrier_colors.get(bus.carrier.lower())
                    if color:
                        self._bus_colors[label] = color
        return self._bus_colors

    @cached_property
    def carrier_units(self) -> dict[str, str]:
        """Mapping of carrier name to unit string.

        Returns:
            Dict mapping carrier names (lowercase) to unit strings.
            Carriers without a unit defined return an empty string.

        Examples:
            >>> fs.topology.carrier_units
            {'electricity': 'kW', 'heat': 'kW', 'gas': 'kW'}
        """
        return {name: carrier.unit or '' for name, carrier in self._fs.carriers.items()}

    @cached_property
    def effect_units(self) -> dict[str, str]:
        """Mapping of effect id to unit string.

        Returns:
            Dict mapping effect ids to unit strings.
            Effects without a unit defined return an empty string.

        Examples:
            >>> fs.topology.effect_units
            {'costs': '€', 'CO2': 'kg'}
        """
        return {effect.id: effect.unit or '' for effect in self._fs.effects.values()}

    @cached_property
    def flows(self) -> xr.DataArray:
        """DataArray with 'flow' dimension and metadata coordinates.

        Coordinates on the 'flow' dimension:
            - component: Parent component id
            - bus: Connected bus id
            - carrier: Carrier name (lowercase)
            - unit: Unit string from carrier
            - is_input: Whether the flow is an input to its component

        Useful for filtering and groupby operations on flow data.

        Examples:
            Filter flow_rates by carrier:

            >>> topo = flow_system.topology
            >>> heat_flows = topo.flows.sel(flow=(topo.flows.carrier == 'heat')).flow.values
            >>> flow_system.stats.flow_rates.sel(flow=heat_flows)

            Filter by component:

            >>> boiler_flows = topo.flows.sel(flow=(topo.flows.component == 'Boiler')).flow.values
            >>> flow_system.stats.flow_rates.sel(flow=boiler_flows)

            Filter by bus:

            >>> bus_flows = topo.flows.sel(flow=(topo.flows.bus == 'HeatBus')).flow.values

            Get only input flows (into components):

            >>> inputs = topo.flows.sel(flow=topo.flows.is_input).flow.values

            Combine filters (heat carrier inputs only):

            >>> mask = (topo.flows.carrier == 'heat') & topo.flows.is_input
            >>> topo.flows.sel(flow=mask).flow.values

            GroupBy component (assign coord from topology, then group):

            >>> rates = flow_system.stats.flow_rates
            >>> rates = rates.assign_coords(component=topo.flows.component.sel(flow=rates.flow))
            >>> rates.groupby('component').sum()
        """
        flow_labels = []
        components = []
        buses = []
        carriers = []
        units = []
        is_inputs = []

        carrier_units = self.carrier_units
        for flow in self._fs.flows.values():
            flow_labels.append(flow.id)
            components.append(flow.component)
            buses.append(flow.bus)
            bus_obj = self._fs.buses.get(flow.bus)
            carrier = bus_obj.carrier.lower() if bus_obj and bus_obj.carrier else ''
            carriers.append(carrier)
            units.append(carrier_units.get(carrier, ''))
            is_inputs.append(flow.is_input_in_component)

        return xr.DataArray(
            data=np.ones(len(flow_labels)),  # Placeholder values
            dims=['flow'],
            coords={
                'flow': flow_labels,
                'component': ('flow', components),
                'bus': ('flow', buses),
                'carrier': ('flow', carriers),
                'unit': ('flow', units),
                'is_input': ('flow', is_inputs),
            },
        )

    def _invalidate_color_caches(self) -> None:
        """Reset all color caches so they are rebuilt on next access."""
        self._carrier_colors = None
        self._component_colors = None
        self._flow_colors = None
        self._bus_colors = None

    def set_component_color(self, label: str, color: str) -> None:
        """Set the color for a single component.

        Args:
            label: Component label.
            color: Color string (hex like '#FF0000', named like 'red', etc.).

        Raises:
            KeyError: If component with given label doesn't exist.

        Examples:
            >>> flow_system.topology.set_component_color('Boiler', '#D35400')
            >>> flow_system.topology.set_component_color('CHP', 'darkred')
        """
        if label not in self._fs.components:
            raise KeyError(f"Component '{label}' not found. Available: {list(self._fs.components.keys())}")
        self._fs.components[label].color = color
        self._invalidate_color_caches()

    def set_component_colors(
        self,
        colors: dict[str, str | list[str]] | str,
        overwrite: bool = True,
    ) -> dict[str, str]:
        """Set colors for multiple components at once.

        Args:
            colors: Color configuration:
                - ``str``: Colorscale name for all components (e.g., ``'turbo'``)
                - ``dict``: Component-to-color mapping (``{'Boiler': 'red'}``) or
                  colorscale-to-components (``{'Blues': ['Wind1', 'Wind2']}``)
            overwrite: If False, skip components that already have colors.

        Returns:
            Mapping of colors that were actually assigned.

        Examples:
            >>> flow_system.topology.set_component_colors('turbo')
            >>> flow_system.topology.set_component_colors({'Boiler': 'red', 'CHP': '#0000FF'})
            >>> flow_system.topology.set_component_colors({'Blues': ['Wind1', 'Wind2']})
            >>> flow_system.topology.set_component_colors('turbo', overwrite=False)
        """
        components = self._fs.components

        # Normalize to {label: color} mapping
        if isinstance(colors, str):
            color_map = process_colors(colors, list(components.keys()))
        else:
            color_map = {}
            for key, value in colors.items():
                if isinstance(value, list):
                    # Colorscale -> component list
                    missing = [c for c in value if c not in components]
                    if missing:
                        raise KeyError(f'Components not found: {missing}')
                    color_map.update(process_colors(key, value))
                else:
                    # Direct assignment
                    if key not in components:
                        raise KeyError(f"Component '{key}' not found")
                    color_map[key] = value

        # Apply colors (respecting overwrite flag)
        result = {}
        for label, color in color_map.items():
            if overwrite or components[label].color is None:
                components[label].color = color
                result[label] = color

        self._invalidate_color_caches()
        return result

    def set_carrier_color(self, carrier: str, color: str) -> None:
        """Set the color for a carrier.

        This affects bus colors derived from this carrier.

        Args:
            carrier: Carrier name (case-insensitive).
            color: Color string (hex like '#FF0000', named like 'red', etc.).

        Examples:
            >>> flow_system.topology.set_carrier_color('electricity', '#FECB52')
            >>> flow_system.topology.set_carrier_color('heat', 'firebrick')
        """
        carrier_obj = self._fs.get_carrier(carrier)
        if carrier_obj is None:
            raise KeyError(f"Carrier '{carrier}' not found.")
        carrier_obj.color = color
        self._invalidate_color_caches()

    def infos(self) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
        """
        Get network topology information as dictionaries.

        Returns node and edge information suitable for visualization or analysis.

        Returns:
            Tuple of (nodes_dict, edges_dict) where:
                - nodes_dict maps node ids to their properties (label, class, infos)
                - edges_dict maps edge ids to their properties (label, start, end, infos)

        Examples:
            >>> nodes, edges = flow_system.topology.infos()
            >>> print(nodes.keys())  # All component and bus labels
            >>> print(edges.keys())  # All flow labels
        """
        from .elements import Bus

        if not self._fs.connected_and_transformed:
            self._fs.connect_and_transform()

        nodes = {
            node.id: {
                'label': node.id,
                'class': 'Bus' if isinstance(node, Bus) else 'Component',
                'infos': node.__str__(),
            }
            for node in chain(self._fs.components.values(), self._fs.buses.values())
        }

        # Use cached colors for efficient lookup
        flow_carriers = self._fs.flow_carriers
        carrier_colors = self.carrier_colors

        edges = {}
        for flow in self._fs.flows.values():
            carrier_name = flow_carriers.get(flow.id)
            edges[flow.id] = {
                'label': flow.flow_id,
                'start': flow.bus if flow.is_input_in_component else flow.component,
                'end': flow.component if flow.is_input_in_component else flow.bus,
                'infos': flow.__str__(),
                'carrier_color': carrier_colors.get(carrier_name) if carrier_name else None,
            }

        return nodes, edges

    def plot(
        self,
        colors: ColorType | None = None,
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """
        Visualize the network structure as a Sankey diagram using Plotly.

        Creates a Sankey diagram showing the topology of the flow system,
        with buses and components as nodes, and flows as links between them.
        All links have equal width since no solution data is used.

        Args:
            colors: Color specification for nodes (buses).
                - `None`: Uses default color palette based on buses.
                - `str`: Plotly colorscale name (e.g., 'Viridis', 'Blues').
                - `list`: List of colors to cycle through.
                - `dict`: Maps bus labels to specific colors.
                Links inherit colors from their connected bus.
            show: Whether to display the figure in the browser.
                - `None`: Uses default from CONFIG.Plotting.default_show.
            **plotly_kwargs: Additional arguments passed to Plotly layout.

        Returns:
            PlotResult containing the Sankey diagram figure and topology data
            (source, target, value for each link).

        Examples:
            >>> flow_system.topology.plot()
            >>> flow_system.topology.plot(show=True)
            >>> flow_system.topology.plot(colors='Viridis')
            >>> flow_system.topology.plot(colors={'ElectricityBus': 'gold', 'HeatBus': 'red'})

        Notes:
            This visualization shows the network structure without optimization results.
            For visualizations that include flow values, use `flow_system.stats.plot.sankey.flows()`
            after running an optimization.

            Hover over nodes and links to see detailed element information.

        See Also:
            - `plot_legacy()`: Previous PyVis-based network visualization.
            - `statistics.plot.sankey.flows()`: Sankey with actual flow values from optimization.
        """
        if not self._fs.connected_and_transformed:
            self._fs.connect_and_transform()

        # Build nodes and links from topology
        nodes: set[str] = set()
        links: dict[str, list] = {
            'source': [],
            'target': [],
            'value': [],
            'label': [],
            'customdata': [],  # For hover text
            'color': [],  # Carrier-based colors
        }

        # Collect node hover info (format repr for HTML display)
        node_hover: dict[str, str] = {}
        for comp in self._fs.components.values():
            node_hover[comp.id] = repr(comp).replace('\n', '<br>')
        for bus in self._fs.buses.values():
            node_hover[bus.id] = repr(bus).replace('\n', '<br>')

        # Use cached colors for efficient lookup
        flow_carriers = self._fs.flow_carriers
        carrier_colors = self.carrier_colors

        for flow in self._fs.flows.values():
            bus_label = flow.bus
            comp_label = flow.component

            if flow.is_input_in_component:
                source = bus_label
                target = comp_label
            else:
                source = comp_label
                target = bus_label

            nodes.add(source)
            nodes.add(target)
            links['source'].append(source)
            links['target'].append(target)
            links['value'].append(1)  # Equal width for all links (no solution data)
            links['label'].append(flow.id)
            links['customdata'].append(repr(flow).replace('\n', '<br>'))  # Flow repr for hover

            # Get carrier color for this flow (subtle/semi-transparent) using cached colors
            carrier_name = flow_carriers.get(flow.id)
            color = carrier_colors.get(carrier_name) if carrier_name else None
            links['color'].append(hex_to_rgba(color, alpha=0.4) if color else hex_to_rgba('', alpha=0.4))

        # Create figure
        node_list = list(nodes)
        node_indices = {n: i for i, n in enumerate(node_list)}

        # Get colors for buses and components using cached colors
        bus_colors_cached = self.bus_colors
        component_colors_cached = self.component_colors

        # If user provided colors, process them for buses
        if colors is not None:
            bus_labels = [bus.id for bus in self._fs.buses.values()]
            bus_color_map = process_colors(colors, bus_labels)
        else:
            bus_color_map = bus_colors_cached

        # Assign colors to nodes: buses get their color, components get their color or neutral gray
        node_colors = []
        for node in node_list:
            if node in bus_color_map:
                node_colors.append(bus_color_map[node])
            elif node in component_colors_cached:
                node_colors.append(component_colors_cached[node])
            else:
                # Fallback - use a neutral gray
                node_colors.append('#808080')

        # Build hover text for nodes
        node_customdata = [node_hover.get(node, node) for node in node_list]

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color='black', width=0.5),
                        label=node_list,
                        color=node_colors,
                        customdata=node_customdata,
                        hovertemplate='%{customdata}<extra></extra>',
                    ),
                    link=dict(
                        source=[node_indices[s] for s in links['source']],
                        target=[node_indices[t] for t in links['target']],
                        value=links['value'],
                        label=links['label'],
                        customdata=links['customdata'],
                        hovertemplate='%{customdata}<extra></extra>',
                        color=links['color'],  # Carrier-based colors
                    ),
                )
            ]
        )
        title = plotly_kwargs.pop('title', 'Flow System Topology')
        fig.update_layout(title=title, **plotly_kwargs)

        # Build xarray DataArray with topology data
        data = xr.DataArray(
            data=links['value'],
            dims=['link'],
            coords={
                'link': links['label'],
                'source': ('link', links['source']),
                'target': ('link', links['target']),
            },
        )
        result = PlotResult(data=data, figure=fig)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            result.show()

        return result

    def plot_legacy(
        self,
        path: bool | str | pathlib.Path = 'flow_system.html',
        controls: bool
        | list[
            Literal['nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer']
        ] = True,
        show: bool | None = None,
    ) -> pyvis.network.Network | None:
        """
        Visualize the network structure using PyVis, saving it as an interactive HTML file.

        .. deprecated::
            Use `plot()` instead for the new Plotly-based Sankey visualization.
            This method is kept for backwards compatibility.

        Args:
            path: Path to save the HTML visualization.
                - `False`: Visualization is created but not saved.
                - `str` or `Path`: Specifies file path (default: 'flow_system.html').
            controls: UI controls to add to the visualization.
                - `True`: Enables all available controls.
                - `List`: Specify controls, e.g., ['nodes', 'layout'].
                - Options: 'nodes', 'edges', 'layout', 'interaction', 'manipulation',
                  'physics', 'selection', 'renderer'.
            show: Whether to open the visualization in the web browser.

        Returns:
            The `pyvis.network.Network` instance representing the visualization,
            or `None` if `pyvis` is not installed.

        Examples:
            >>> flow_system.topology.plot_legacy()
            >>> flow_system.topology.plot_legacy(show=False)
            >>> flow_system.topology.plot_legacy(path='output/network.html', controls=['nodes', 'layout'])

        Notes:
            This function requires `pyvis`. If not installed, the function prints
            a warning and returns `None`.
            Nodes are styled based on type (circles for buses, boxes for components)
            and annotated with node information.
        """
        warnings.warn(
            f'This method is deprecated and will be removed in v{DEPRECATION_REMOVAL_VERSION}. '
            'Use flow_system.topology.plot() instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        node_infos, edge_infos = self.infos()
        # Normalize path=False to None for _plot_network compatibility
        normalized_path = None if path is False else path
        return _plot_network(
            node_infos,
            edge_infos,
            normalized_path,
            controls,
            show if show is not None else CONFIG.Plotting.default_show,
        )

    def start_app(self) -> None:
        """
        Start an interactive network visualization using Dash and Cytoscape.

        Launches a web-based interactive visualization server that allows
        exploring the network structure dynamically.

        Raises:
            ImportError: If required dependencies are not installed.

        Examples:
            >>> flow_system.topology.start_app()
            >>> # ... interact with the visualization in browser ...
            >>> flow_system.topology.stop_app()

        Notes:
            Requires optional dependencies: dash, dash-cytoscape, dash-daq,
            networkx, flask, werkzeug.
            Install with: `pip install flixopt[network_viz]` or `pip install flixopt[full]`
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

        if self._fs.status < FlowSystemStatus.CONNECTED:
            self._fs._connect_network()

        if self._fs._network_app is not None:
            logger.warning('The network app is already running. Restarting it.')
            self.stop_app()

        self._fs._network_app = shownetwork(flow_graph(self._fs))

    def stop_app(self) -> None:
        """
        Stop the interactive network visualization server.

        Examples:
            >>> flow_system.topology.stop_app()
        """
        if self._fs._network_app is None:
            logger.warning("No network app is currently running. Can't stop it")
            return

        try:
            logger.info('Stopping network visualization server...')
            self._fs._network_app.server_instance.shutdown()
            logger.info('Network visualization stopped.')
        except Exception as e:
            logger.error(f'Failed to stop the network visualization app: {e}')
        finally:
            self._fs._network_app = None
