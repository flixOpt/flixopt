"""
Topology accessor for FlowSystem.

This module provides the TopologyAccessor class that enables the
`flow_system.topology` pattern for network structure inspection and visualization.
"""

from __future__ import annotations

import logging
import warnings
from itertools import chain
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pathlib

    import pyvis

    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


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

    def infos(self) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
        """
        Get network topology information as dictionaries.

        Returns node and edge information suitable for visualization or analysis.

        Returns:
            Tuple of (nodes_dict, edges_dict) where:
                - nodes_dict maps node labels to their properties (label, class, infos)
                - edges_dict maps edge labels to their properties (label, start, end, infos)

        Examples:
            >>> nodes, edges = flow_system.topology.infos()
            >>> print(nodes.keys())  # All component and bus labels
            >>> print(edges.keys())  # All flow labels
        """
        from .elements import Bus

        if not self._fs.connected_and_transformed:
            self._fs.connect_and_transform()

        nodes = {
            node.label_full: {
                'label': node.label,
                'class': 'Bus' if isinstance(node, Bus) else 'Component',
                'infos': node.__str__(),
            }
            for node in chain(self._fs.components.values(), self._fs.buses.values())
        }

        edges = {
            flow.label_full: {
                'label': flow.label,
                'start': flow.bus if flow.is_input_in_component else flow.component,
                'end': flow.component if flow.is_input_in_component else flow.bus,
                'infos': flow.__str__(),
            }
            for flow in self._fs.flows.values()
        }

        return nodes, edges

    def plot(
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
            >>> flow_system.topology.plot()
            >>> flow_system.topology.plot(show=False)
            >>> flow_system.topology.plot(path='output/network.html', controls=['nodes', 'layout'])

        Notes:
            This function requires `pyvis`. If not installed, the function prints
            a warning and returns `None`.
            Nodes are styled based on type (circles for buses, boxes for components)
            and annotated with node information.
        """
        from . import plotting
        from .config import CONFIG

        node_infos, edge_infos = self.infos()
        return plotting.plot_network(
            node_infos, edge_infos, path, controls, show if show is not None else CONFIG.Plotting.default_show
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

        if not self._fs._connected_and_transformed:
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
        from .network_app import DASH_CYTOSCAPE_AVAILABLE, VISUALIZATION_ERROR

        if not DASH_CYTOSCAPE_AVAILABLE:
            raise ImportError(
                f'Network visualization requires optional dependencies. '
                f'Install with: `pip install flixopt[network_viz]`, `pip install flixopt[full]` '
                f'or: `pip install dash dash-cytoscape dash-daq networkx werkzeug`. '
                f'Original error: {VISUALIZATION_ERROR}'
            )

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
