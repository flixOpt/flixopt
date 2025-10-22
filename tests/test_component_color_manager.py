"""Tests for ComponentColorManager functionality."""

import numpy as np
import pytest
import xarray as xr

from flixopt.plotting import ComponentColorManager, resolve_colors


class TestBasicFunctionality:
    """Test basic ComponentColorManager functionality."""

    def test_initialization_default(self):
        """Test default initialization."""
        components = ['Solar_PV', 'Wind_Onshore', 'Coal_Plant']
        manager = ComponentColorManager(components)

        assert len(manager.components) == 3
        assert manager.default_colorscale == 'plotly'
        assert 'Solar_PV' in manager.components

    def test_sorted_components(self):
        """Test that components are sorted for stability."""
        components = ['C_Component', 'A_Component', 'B_Component']
        manager = ComponentColorManager(components)

        # Components should be sorted
        assert manager.components == ['A_Component', 'B_Component', 'C_Component']

    def test_default_color_assignment(self):
        """Test that components get default colors on initialization."""
        components = ['Comp1', 'Comp2', 'Comp3']
        manager = ComponentColorManager(components)

        # Each component should have a color
        for comp in components:
            color = manager.get_color(comp)
            assert color is not None
            assert isinstance(color, str)

    def test_empty_initialization(self):
        """Test initialization without components."""
        manager = ComponentColorManager()
        assert len(manager.components) == 0


class TestConfigureAPI:
    """Test the configure() method with various inputs."""

    def test_configure_direct_colors(self):
        """Test direct color assignment (component → color)."""
        manager = ComponentColorManager()
        manager.configure({'Boiler1': '#FF0000', 'CHP': 'darkred', 'Storage': 'green'})

        assert manager.get_color('Boiler1') == '#FF0000'
        assert manager.get_color('CHP') == 'darkred'
        assert manager.get_color('Storage') == 'green'

    def test_configure_grouped_colors(self):
        """Test grouped color assignment (colorscale → list of components)."""
        manager = ComponentColorManager()
        manager.configure(
            {
                'oranges': ['Solar1', 'Solar2'],
                'blues': ['Wind1', 'Wind2'],
            }
        )

        # All should have colors
        assert manager.get_color('Solar1') is not None
        assert manager.get_color('Solar2') is not None
        assert manager.get_color('Wind1') is not None
        assert manager.get_color('Wind2') is not None

        # Solar components should have different shades
        assert manager.get_color('Solar1') != manager.get_color('Solar2')

        # Wind components should have different shades
        assert manager.get_color('Wind1') != manager.get_color('Wind2')

    def test_configure_mixed(self):
        """Test mixed direct and grouped colors."""
        manager = ComponentColorManager()
        manager.configure(
            {
                'Boiler1': '#FF0000',
                'oranges': ['Solar1', 'Solar2'],
                'blues': ['Wind1', 'Wind2'],
            }
        )

        # Direct color
        assert manager.get_color('Boiler1') == '#FF0000'

        # Grouped colors
        assert manager.get_color('Solar1') is not None
        assert manager.get_color('Wind1') is not None

    def test_configure_updates_components_list(self):
        """Test that configure() adds components to the list."""
        manager = ComponentColorManager()
        assert len(manager.components) == 0

        manager.configure({'Boiler1': '#FF0000', 'CHP': 'red'})

        assert len(manager.components) == 2
        assert 'Boiler1' in manager.components
        assert 'CHP' in manager.components


class TestColorFamilies:
    """Test color family functionality."""

    def test_default_families(self):
        """Test that default families are available."""
        manager = ComponentColorManager([])

        assert 'blues' in manager.color_families
        assert 'oranges' in manager.color_families
        assert 'greens' in manager.color_families
        assert 'reds' in manager.color_families


class TestColorStability:
    """Test color stability across different datasets."""

    def test_same_component_same_color(self):
        """Test that same component always gets same color."""
        manager = ComponentColorManager()
        manager.configure(
            {
                'oranges': ['Solar_PV'],
                'blues': ['Wind_Onshore'],
            }
        )

        # Get colors multiple times
        color1 = manager.get_color('Solar_PV')
        color2 = manager.get_color('Solar_PV')
        color3 = manager.get_color('Solar_PV')

        assert color1 == color2 == color3

    def test_color_stability_with_different_datasets(self):
        """Test that colors remain stable across different variable subsets."""
        manager = ComponentColorManager()
        manager.configure(
            {
                'oranges': ['Solar_PV'],
                'blues': ['Wind_Onshore'],
                'greys': ['Coal_Plant'],
                'reds': ['Gas_Plant'],
            }
        )

        # Dataset 1: Only Solar and Wind
        dataset1 = xr.Dataset(
            {
                'Solar_PV(Bus)|flow_rate': (['time'], np.random.rand(10)),
                'Wind_Onshore(Bus)|flow_rate': (['time'], np.random.rand(10)),
            },
            coords={'time': np.arange(10)},
        )

        # Dataset 2: All components
        dataset2 = xr.Dataset(
            {
                'Solar_PV(Bus)|flow_rate': (['time'], np.random.rand(10)),
                'Wind_Onshore(Bus)|flow_rate': (['time'], np.random.rand(10)),
                'Coal_Plant(Bus)|flow_rate': (['time'], np.random.rand(10)),
                'Gas_Plant(Bus)|flow_rate': (['time'], np.random.rand(10)),
            },
            coords={'time': np.arange(10)},
        )

        colors1 = resolve_colors(dataset1, manager, engine='plotly')
        colors2 = resolve_colors(dataset2, manager, engine='plotly')

        # Solar_PV and Wind_Onshore should have same colors in both datasets
        assert colors1['Solar_PV(Bus)|flow_rate'] == colors2['Solar_PV(Bus)|flow_rate']
        assert colors1['Wind_Onshore(Bus)|flow_rate'] == colors2['Wind_Onshore(Bus)|flow_rate']


class TestVariableExtraction:
    """Test variable to component extraction."""

    def test_extract_component_with_parentheses(self):
        """Test extracting component from variable with parentheses."""
        manager = ComponentColorManager([])

        variable = 'Solar_PV(ElectricityBus)|flow_rate'
        component = manager.extract_component(variable)

        assert component == 'Solar_PV'

    def test_extract_component_with_pipe(self):
        """Test extracting component from variable with pipe."""
        manager = ComponentColorManager([])

        variable = 'Solar_PV|investment'
        component = manager.extract_component(variable)

        assert component == 'Solar_PV'

    def test_extract_component_no_separators(self):
        """Test extracting component from variable without separators."""
        manager = ComponentColorManager([])

        variable = 'SimpleComponent'
        component = manager.extract_component(variable)

        assert component == 'SimpleComponent'


class TestVariableColorResolution:
    """Test getting colors for variables."""

    def test_get_variable_color(self):
        """Test getting color for a single variable."""
        manager = ComponentColorManager()
        manager.configure({'oranges': ['Solar_PV']})

        variable = 'Solar_PV(Bus)|flow_rate'
        color = manager.get_variable_color(variable)

        assert color is not None
        assert isinstance(color, str)

    def test_get_variable_colors_multiple(self):
        """Test getting colors for multiple variables."""
        manager = ComponentColorManager()
        manager.configure(
            {
                'oranges': ['Solar_PV'],
                'blues': ['Wind_Onshore'],
                'greys': ['Coal_Plant'],
            }
        )

        variables = ['Solar_PV(Bus)|flow_rate', 'Wind_Onshore(Bus)|flow_rate', 'Coal_Plant(Bus)|flow_rate']

        colors = manager.get_variable_colors(variables)

        assert len(colors) == 3
        assert all(var in colors for var in variables)
        assert all(isinstance(color, str) for color in colors.values())

    def test_variable_extraction_in_color_resolution(self):
        """Test that variable names are properly extracted to component names."""
        manager = ComponentColorManager()
        manager.configure({'Solar_PV': '#FF0000'})

        # Variable format with flow
        variable_color = manager.get_variable_color('Solar_PV(Bus)|flow_rate')
        component_color = manager.get_color('Solar_PV')

        # Should be the same color
        assert variable_color == component_color


class TestIntegrationWithResolveColors:
    """Test integration with resolve_colors function."""

    def test_resolve_colors_with_manager(self):
        """Test resolve_colors with ComponentColorManager."""
        manager = ComponentColorManager()
        manager.configure(
            {
                'oranges': ['Solar_PV'],
                'blues': ['Wind_Onshore'],
            }
        )

        dataset = xr.Dataset(
            {
                'Solar_PV(Bus)|flow_rate': (['time'], np.random.rand(10)),
                'Wind_Onshore(Bus)|flow_rate': (['time'], np.random.rand(10)),
            },
            coords={'time': np.arange(10)},
        )

        colors = resolve_colors(dataset, manager, engine='plotly')

        assert len(colors) == 2
        assert 'Solar_PV(Bus)|flow_rate' in colors
        assert 'Wind_Onshore(Bus)|flow_rate' in colors

    def test_resolve_colors_with_dict(self):
        """Test that resolve_colors still works with dict."""
        dataset = xr.Dataset(
            {'var1': (['time'], np.random.rand(10)), 'var2': (['time'], np.random.rand(10))},
            coords={'time': np.arange(10)},
        )

        color_dict = {'var1': '#FF0000', 'var2': '#00FF00'}
        colors = resolve_colors(dataset, color_dict, engine='plotly')

        assert colors == color_dict


class TestMethodChaining:
    """Test method chaining."""

    def test_configure_returns_self(self):
        """Test that configure() returns self for chaining."""
        manager = ComponentColorManager()
        result = manager.configure({'Boiler': 'red'})

        assert result is manager

    def test_chaining_with_initialization(self):
        """Test method chaining with initialization."""
        # Test chaining configure() after __init__
        manager = ComponentColorManager(components=['Solar_PV', 'Wind_Onshore'])
        manager.configure({'oranges': ['Solar_PV']})

        assert len(manager.components) == 2
        assert manager.get_color('Solar_PV') is not None


class TestUnknownComponents:
    """Test behavior with unknown components."""

    def test_get_color_unknown_component(self):
        """Test that unknown components get a default grey color."""
        manager = ComponentColorManager()
        manager.configure({'Boiler': 'red'})

        # Unknown component
        color = manager.get_color('UnknownComponent')

        # Should return grey default
        assert color == '#808080'

    def test_get_variable_color_unknown_component(self):
        """Test that unknown components in variables get default color."""
        manager = ComponentColorManager()
        manager.configure({'Boiler': 'red'})

        # Unknown component
        color = manager.get_variable_color('UnknownComponent(Bus)|flow')

        # Should return grey default
        assert color == '#808080'


class TestColorCaching:
    """Test that variable color caching works."""

    def test_cache_is_used(self):
        """Test that cache is used for repeated variable lookups."""
        manager = ComponentColorManager()
        manager.configure({'Solar_PV': '#FF0000'})

        # First call populates cache
        color1 = manager.get_variable_color('Solar_PV(Bus)|flow_rate')

        # Second call should hit cache
        color2 = manager.get_variable_color('Solar_PV(Bus)|flow_rate')

        assert color1 == color2
        assert 'Solar_PV(Bus)|flow_rate' in manager._variable_cache

    def test_cache_cleared_on_configure(self):
        """Test that cache is cleared when colors are reconfigured."""
        manager = ComponentColorManager()
        manager.configure({'Solar_PV': '#FF0000'})

        # Populate cache
        manager.get_variable_color('Solar_PV(Bus)|flow_rate')
        assert len(manager._variable_cache) > 0

        # Reconfigure
        manager.configure({'Solar_PV': '#00FF00'})

        # Cache should be cleared
        assert len(manager._variable_cache) == 0
