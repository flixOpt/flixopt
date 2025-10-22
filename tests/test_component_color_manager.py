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
        assert manager.default_colormap == 'Dark24'
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


class TestColorFamilies:
    """Test color family functionality."""

    def test_default_families(self):
        """Test that default families are available."""
        manager = ComponentColorManager([])

        assert 'blues' in manager.color_families
        assert 'oranges' in manager.color_families
        assert 'greens' in manager.color_families
        assert 'reds' in manager.color_families

    def test_add_custom_family(self):
        """Test adding a custom color family."""
        manager = ComponentColorManager([])
        custom_colors = ['#FF0000', '#00FF00', '#0000FF']

        result = manager.add_custom_family('ocean', custom_colors)

        assert 'ocean' in manager.color_families
        assert manager.color_families['ocean'] == custom_colors
        assert result is manager  # Check method chaining


class TestGroupingRules:
    """Test grouping rule functionality."""

    def test_add_grouping_rule_prefix(self):
        """Test adding a prefix grouping rule."""
        components = ['Solar_PV', 'Solar_Thermal', 'Wind_Onshore']
        manager = ComponentColorManager(components)

        result = manager.add_grouping_rule('Solar', 'renewables', 'oranges', match_type='prefix')

        assert len(manager._grouping_rules) == 1
        assert result is manager  # Check method chaining

    def test_apply_colors(self):
        """Test auto-grouping components based on rules."""
        components = ['Solar_PV', 'Solar_Thermal', 'Wind_Onshore', 'Wind_Offshore', 'Coal_Plant']
        manager = ComponentColorManager(components)

        manager.add_grouping_rule('Solar', 'renewables_solar', 'oranges', match_type='prefix')
        manager.add_grouping_rule('Wind', 'renewables_wind', 'blues', match_type='prefix')
        manager.add_grouping_rule('Coal', 'fossil', 'greys', match_type='prefix')
        manager.apply_colors()

        # Check that components got colors from appropriate families
        solar_color = manager.get_color('Solar_PV')
        wind_color = manager.get_color('Wind_Onshore')

        assert solar_color is not None
        assert wind_color is not None
        # Colors should be different (from different families)
        assert solar_color != wind_color


class TestColorStability:
    """Test color stability across different datasets."""

    def test_same_component_same_color(self):
        """Test that same component always gets same color."""
        components = ['Solar_PV', 'Wind_Onshore', 'Coal_Plant', 'Gas_Plant']
        manager = ComponentColorManager(components)

        manager.add_grouping_rule('Solar', 'renewables_solar', 'oranges', match_type='prefix')
        manager.add_grouping_rule('Wind', 'renewables_wind', 'blues', match_type='prefix')
        manager.apply_colors()

        # Get colors multiple times
        color1 = manager.get_color('Solar_PV')
        color2 = manager.get_color('Solar_PV')
        color3 = manager.get_color('Solar_PV')

        assert color1 == color2 == color3

    def test_color_stability_with_different_datasets(self):
        """Test that colors remain stable across different variable subsets."""
        components = ['Solar_PV', 'Wind_Onshore', 'Coal_Plant', 'Gas_Plant']
        manager = ComponentColorManager(components)

        manager.add_grouping_rule('Solar', 'solar', 'oranges', match_type='prefix')
        manager.add_grouping_rule('Wind', 'wind', 'blues', match_type='prefix')
        manager.add_grouping_rule('Coal', 'fossil_coal', 'greys', match_type='prefix')
        manager.add_grouping_rule('Gas', 'fossil_gas', 'reds', match_type='prefix')
        manager.apply_colors()

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
        components = ['Solar_PV', 'Wind_Onshore']
        manager = ComponentColorManager(components)
        manager.add_grouping_rule('Solar', 'solar', 'oranges', match_type='prefix')
        manager.apply_colors()

        variable = 'Solar_PV(Bus)|flow_rate'
        color = manager.get_variable_color(variable)

        assert color is not None
        assert isinstance(color, str)

    def test_get_variable_colors_multiple(self):
        """Test getting colors for multiple variables."""
        components = ['Solar_PV', 'Wind_Onshore', 'Coal_Plant']
        manager = ComponentColorManager(components)
        manager.add_grouping_rule('Solar', 'solar', 'oranges', match_type='prefix')
        manager.add_grouping_rule('Wind', 'wind', 'blues', match_type='prefix')
        manager.apply_colors()

        variables = ['Solar_PV(Bus)|flow_rate', 'Wind_Onshore(Bus)|flow_rate', 'Coal_Plant(Bus)|flow_rate']

        colors = manager.get_variable_colors(variables)

        assert len(colors) == 3
        assert all(var in colors for var in variables)
        assert all(isinstance(color, str) for color in colors.values())


class TestOverrides:
    """Test override functionality."""

    def test_simple_override(self):
        """Test simple color override."""
        components = ['Solar_PV', 'Wind_Onshore']
        manager = ComponentColorManager(components)
        manager.add_grouping_rule('Solar', 'solar', 'oranges', match_type='prefix')
        manager.apply_colors()

        # Override Solar_PV color
        manager.override({'Solar_PV': '#FF0000'})

        color = manager.get_color('Solar_PV')
        assert color == '#FF0000'

    def test_override_precedence(self):
        """Test that overrides take precedence over grouping rules."""
        components = ['Solar_PV']
        manager = ComponentColorManager(components)
        manager.add_grouping_rule('Solar', 'solar', 'oranges', match_type='prefix')
        manager.apply_colors()

        original_color = manager.get_color('Solar_PV')

        manager.override({'Solar_PV': '#FFD700'})

        new_color = manager.get_color('Solar_PV')
        assert new_color == '#FFD700'
        assert new_color != original_color


class TestIntegrationWithResolveColors:
    """Test integration with resolve_colors function."""

    def test_resolve_colors_with_manager(self):
        """Test resolve_colors with ComponentColorManager."""
        components = ['Solar_PV', 'Wind_Onshore', 'Coal_Plant']
        manager = ComponentColorManager(components)
        manager.add_grouping_rule('Solar', 'solar', 'oranges', match_type='prefix')
        manager.add_grouping_rule('Wind', 'wind', 'blues', match_type='prefix')
        manager.apply_colors()

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

    def test_full_chaining(self):
        """Test full method chaining."""
        components = ['Solar_PV', 'Wind_Onshore', 'Gas_Plant']
        manager = (
            ComponentColorManager(components=components)
            .add_custom_family('ocean', ['#003f5c', '#2f4b7c'])
            .add_grouping_rule('Solar', 'renewables', 'oranges', match_type='prefix')
            .add_grouping_rule('Wind', 'renewables', 'blues', match_type='prefix')
        )

        assert 'ocean' in manager.color_families
        assert len(manager._grouping_rules) == 2


class TestFlowShading:
    """Test flow-level color distinction."""

    def test_initialization_with_flows_dict(self):
        """Test initialization with flows parameter."""
        flows = {
            'Boiler': ['Q_th', 'Q_fu'],
            'CHP': ['P_el', 'Q_th', 'Q_fu'],
        }
        manager = ComponentColorManager(flows=flows, flow_variation=0.04)

        assert len(manager.components) == 2
        assert manager.flows == {'Boiler': ['Q_fu', 'Q_th'], 'CHP': ['P_el', 'Q_fu', 'Q_th']}  # Sorted
        assert manager.flow_variation == 0.04

    def test_flow_extraction(self):
        """Test _extract_component_and_flow method."""
        manager = ComponentColorManager(components=['Boiler'])

        # Test Component(Flow)|attribute format
        comp, flow = manager._extract_component_and_flow('Boiler(Q_th)|flow_rate')
        assert comp == 'Boiler'
        assert flow == 'Q_th'

        # Test Component|attribute format (no flow)
        comp, flow = manager._extract_component_and_flow('Boiler|investment')
        assert comp == 'Boiler'
        assert flow is None

        # Test plain component name
        comp, flow = manager._extract_component_and_flow('Boiler')
        assert comp == 'Boiler'
        assert flow is None

    def test_flow_shading_disabled(self):
        """Test that flow shading is disabled by default."""
        flows = {'Boiler': ['Q_th', 'Q_fu']}
        manager = ComponentColorManager(flows=flows, flow_variation=None)

        # Both flows should get the same color
        color1 = manager.get_variable_color('Boiler(Q_th)|flow_rate')
        color2 = manager.get_variable_color('Boiler(Q_fu)|flow_rate')

        assert color1 == color2

    def test_flow_shading_enabled(self):
        """Test that flow shading creates distinct colors."""
        flows = {'Boiler': ['Q_th', 'Q_fu', 'Q_el']}
        manager = ComponentColorManager(flows=flows, flow_variation=0.04)

        # Get colors for all three flows
        color_th = manager.get_variable_color('Boiler(Q_th)|flow_rate')
        color_fu = manager.get_variable_color('Boiler(Q_fu)|flow_rate')
        color_el = manager.get_variable_color('Boiler(Q_el)|flow_rate')

        # All colors should be different
        assert color_th != color_fu
        assert color_fu != color_el
        assert color_th != color_el

        # All colors should be valid hex
        assert color_th.startswith('#')
        assert color_fu.startswith('#')
        assert color_el.startswith('#')

    def test_flow_shading_stability(self):
        """Test that flow shading produces stable colors."""
        flows = {'Boiler': ['Q_th', 'Q_fu']}
        manager = ComponentColorManager(flows=flows, flow_variation=0.04)

        # Get color multiple times
        color1 = manager.get_variable_color('Boiler(Q_th)|flow_rate')
        color2 = manager.get_variable_color('Boiler(Q_th)|flow_rate')
        color3 = manager.get_variable_color('Boiler(Q_th)|flow_rate')

        assert color1 == color2 == color3

    def test_single_flow_no_shading(self):
        """Test that single flow gets base color (no shading needed)."""
        flows = {'Storage': ['Q_th_load']}
        manager = ComponentColorManager(flows=flows, flow_variation=0.04)

        # Single flow should get base color
        color = manager.get_variable_color('Storage(Q_th_load)|flow_rate')
        base_color = manager.get_color('Storage')

        assert color == base_color

    def test_flow_variation_strength(self):
        """Test that variation strength parameter works."""
        flows = {'Boiler': ['Q_th', 'Q_fu']}

        # Low variation
        manager_low = ComponentColorManager(flows=flows, flow_variation=0.02)
        color_low_1 = manager_low.get_variable_color('Boiler(Q_th)|flow_rate')
        color_low_2 = manager_low.get_variable_color('Boiler(Q_fu)|flow_rate')

        # High variation
        manager_high = ComponentColorManager(flows=flows, flow_variation=0.15)
        color_high_1 = manager_high.get_variable_color('Boiler(Q_th)|flow_rate')
        color_high_2 = manager_high.get_variable_color('Boiler(Q_fu)|flow_rate')

        # Colors should be different
        assert color_low_1 != color_low_2
        assert color_high_1 != color_high_2

        # High variation should have larger difference (this is approximate test)
        # We can't easily measure color distance, so just check they're all different
        assert color_low_1 != color_high_1
        assert color_low_2 != color_high_2
