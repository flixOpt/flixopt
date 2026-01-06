"""Tests for the TopologyAccessor class."""

import tempfile
from pathlib import Path

import plotly.graph_objects as go
import pytest

import flixopt as fx


@pytest.fixture
def flow_system(simple_flow_system):
    """Get a simple flow system for testing."""
    if isinstance(simple_flow_system, fx.FlowSystem):
        return simple_flow_system
    return simple_flow_system[0]


class TestTopologyInfos:
    """Tests for topology.infos() method."""

    def test_infos_returns_tuple(self, flow_system):
        """Test that infos() returns a tuple of two dicts."""
        result = flow_system.topology.infos()
        assert isinstance(result, tuple)
        assert len(result) == 2
        nodes, edges = result
        assert isinstance(nodes, dict)
        assert isinstance(edges, dict)

    def test_infos_nodes_have_correct_structure(self, flow_system):
        """Test that nodes have label, class, and infos keys."""
        nodes, _ = flow_system.topology.infos()
        for node_data in nodes.values():
            assert 'label' in node_data
            assert 'class' in node_data
            assert 'infos' in node_data
            assert node_data['class'] in ('Bus', 'Component')

    def test_infos_edges_have_correct_structure(self, flow_system):
        """Test that edges have label, start, end, and infos keys."""
        _, edges = flow_system.topology.infos()
        for edge_data in edges.values():
            assert 'label' in edge_data
            assert 'start' in edge_data
            assert 'end' in edge_data
            assert 'infos' in edge_data

    def test_infos_contains_all_elements(self, flow_system):
        """Test that infos contains all components, buses, and flows."""
        nodes, edges = flow_system.topology.infos()

        # Check components
        for comp in flow_system.components.values():
            assert comp.label in nodes

        # Check buses
        for bus in flow_system.buses.values():
            assert bus.label in nodes

        # Check flows
        for flow in flow_system.flows.values():
            assert flow.label_full in edges


class TestTopologyPlot:
    """Tests for topology.plot() method (Sankey-based)."""

    def test_plot_returns_plotly_figure(self, flow_system):
        """Test that plot() returns a PlotResult with Plotly Figure."""
        result = flow_system.topology.plot(show=False)
        assert hasattr(result, 'figure')
        assert isinstance(result.figure, go.Figure)

    def test_plot_contains_sankey_trace(self, flow_system):
        """Test that the figure contains a Sankey trace."""
        result = flow_system.topology.plot(show=False)
        assert len(result.figure.data) == 1
        assert isinstance(result.figure.data[0], go.Sankey)

    def test_plot_has_correct_title(self, flow_system):
        """Test that the figure has the correct title."""
        result = flow_system.topology.plot(show=False)
        assert result.figure.layout.title.text == 'Flow System Topology'

    def test_plot_with_custom_title(self, flow_system):
        """Test that custom title can be passed via plotly_kwargs."""
        result = flow_system.topology.plot(show=False, title='Custom Title')
        assert result.figure.layout.title.text == 'Custom Title'

    def test_plot_contains_all_nodes(self, flow_system):
        """Test that the Sankey contains all buses and components as nodes."""
        result = flow_system.topology.plot(show=False)
        sankey = result.figure.data[0]
        node_labels = set(sankey.node.label)

        # All buses should be in nodes
        for bus in flow_system.buses.values():
            assert bus.label in node_labels

        # All components should be in nodes
        for comp in flow_system.components.values():
            assert comp.label in node_labels

    def test_plot_contains_all_flows_as_links(self, flow_system):
        """Test that all flows are represented as links."""
        result = flow_system.topology.plot(show=False)
        sankey = result.figure.data[0]
        link_labels = set(sankey.link.label)

        # All flows should be represented as links
        for flow in flow_system.flows.values():
            assert flow.label_full in link_labels

    def test_plot_with_colors(self, flow_system):
        """Test that colors parameter is accepted."""
        # Should not raise
        flow_system.topology.plot(colors='Viridis', show=False)
        flow_system.topology.plot(colors=['red', 'blue', 'green'], show=False)


class TestTopologyPlotLegacy:
    """Tests for topology.plot_legacy() method (PyVis-based)."""

    def test_plot_legacy_returns_network_or_none(self, flow_system):
        """Test that plot_legacy() returns a pyvis Network or None."""
        try:
            import pyvis

            result = flow_system.topology.plot_legacy(path=False, show=False)
            assert result is None or isinstance(result, pyvis.network.Network)
        except ImportError:
            # pyvis not installed, should return None
            result = flow_system.topology.plot_legacy(path=False, show=False)
            assert result is None

    def test_plot_legacy_creates_html_file(self, flow_system):
        """Test that plot_legacy() creates an HTML file when path is specified."""
        pytest.importorskip('pyvis')

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / 'network.html'
            flow_system.topology.plot_legacy(path=str(html_path), show=False)
            assert html_path.exists()
            content = html_path.read_text()
            assert '<html>' in content.lower() or '<!doctype' in content.lower()

    def test_plot_legacy_with_controls_list(self, flow_system):
        """Test that plot_legacy() accepts a list of controls."""
        pytest.importorskip('pyvis')

        # Should not raise
        flow_system.topology.plot_legacy(path=False, controls=['nodes', 'layout'], show=False)
