"""Tests for the TopologyAccessor class."""

import tempfile
from pathlib import Path

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
    """Tests for topology.plot() method."""

    def test_plot_returns_network_or_none(self, flow_system):
        """Test that plot() returns a pyvis Network or None."""
        try:
            import pyvis

            result = flow_system.topology.plot(path=False, show=False)
            assert result is None or isinstance(result, pyvis.network.Network)
        except ImportError:
            # pyvis not installed, should return None
            result = flow_system.topology.plot(path=False, show=False)
            assert result is None

    def test_plot_creates_html_file(self, flow_system):
        """Test that plot() creates an HTML file when path is specified."""
        pytest.importorskip('pyvis')

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / 'network.html'
            flow_system.topology.plot(path=str(html_path), show=False)
            assert html_path.exists()
            content = html_path.read_text()
            assert '<html>' in content.lower() or '<!doctype' in content.lower()

    def test_plot_with_controls_list(self, flow_system):
        """Test that plot() accepts a list of controls."""
        pytest.importorskip('pyvis')

        # Should not raise
        flow_system.topology.plot(path=False, controls=['nodes', 'layout'], show=False)


class TestDeprecatedMethods:
    """Tests for deprecated FlowSystem methods that delegate to topology."""

    def test_network_infos_deprecation_warning(self, flow_system):
        """Test that network_infos() raises a DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match='topology.infos'):
            flow_system.network_infos()

    def test_plot_network_deprecation_warning(self, flow_system):
        """Test that plot_network() raises a DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match='topology.plot'):
            flow_system.plot_network(path=False, show=False)

    def test_deprecated_methods_return_same_results(self, flow_system):
        """Test that deprecated methods return the same results as topology accessor."""
        import warnings

        # Get results from new API
        new_nodes, new_edges = flow_system.topology.infos()

        # Get results from deprecated API (suppress warning)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            old_nodes, old_edges = flow_system.network_infos()

        assert new_nodes == old_nodes
        assert new_edges == old_edges
