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
        assert manager.default_colormap == 'viridis'
        assert 'Solar_PV' in manager.components
        assert 'Wind_Onshore' in manager.components
        assert 'Coal_Plant' in manager.components

    def test_initialization_custom_colormap(self):
        """Test initialization with custom default colormap."""
        components = ['Comp1', 'Comp2']
        manager = ComponentColorManager(components, default_colormap='viridis')

        assert manager.default_colormap == 'viridis'

    def test_sorted_components(self):
        """Test that components are sorted for stability."""
        components = ['C_Component', 'A_Component', 'B_Component']
        manager = ComponentColorManager(components)

        # Components should be sorted
        assert manager.components == ['A_Component', 'B_Component', 'C_Component']

    def test_duplicate_components_removed(self):
        """Test that duplicate components are removed."""
        components = ['Comp1', 'Comp2', 'Comp1', 'Comp3', 'Comp2']
        manager = ComponentColorManager(components)

        assert len(manager.components) == 3
        assert manager.components == ['Comp1', 'Comp2', 'Comp3']

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

    def test_add_custom_family_replaces_existing(self):
        """Test that adding a family with existing name replaces it."""
        manager = ComponentColorManager([])
        original = ['#FF0000']
        replacement = ['#00FF00', '#0000FF']

        manager.add_custom_family('test', original)
        manager.add_custom_family('test', replacement)

        assert manager.color_families['test'] == replacement


class TestGroupingRules:
    """Test grouping rule functionality."""

    def test_add_grouping_rule_prefix(self):
        """Test adding a prefix grouping rule."""
        components = ['Solar_PV', 'Solar_Thermal', 'Wind_Onshore']
        manager = ComponentColorManager(components)

        result = manager.add_grouping_rule('Solar', 'renewables', 'oranges', match_type='prefix')

        assert len(manager._grouping_rules) == 1
        assert result is manager  # Check method chaining

    def test_add_grouping_rule_suffix(self):
        """Test adding a suffix grouping rule."""
        components = ['PV_Solar', 'Thermal_Solar', 'Onshore_Wind']
        manager = ComponentColorManager(components)

        manager.add_grouping_rule('_Solar', 'solar_tech', 'oranges', match_type='suffix')

        assert manager._grouping_rules[0]['match_type'] == 'suffix'

    def test_add_grouping_rule_contains(self):
        """Test adding a contains grouping rule."""
        components = ['BigSolarPV', 'SmallSolarThermal', 'WindTurbine']
        manager = ComponentColorManager(components)

        manager.add_grouping_rule('Solar', 'solar_tech', 'oranges', match_type='contains')

        assert manager._grouping_rules[0]['match_type'] == 'contains'

    def test_add_grouping_rule_glob(self):
        """Test adding a glob grouping rule."""
        components = ['Solar_PV_1', 'Solar_PV_2', 'Wind_1']
        manager = ComponentColorManager(components)

        manager.add_grouping_rule('Solar_*', 'solar_tech', 'oranges', match_type='glob')

        assert manager._grouping_rules[0]['match_type'] == 'glob'

    def test_add_grouping_rule_regex(self):
        """Test adding a regex grouping rule."""
        components = ['Solar_01', 'Solar_02', 'Wind_01']
        manager = ComponentColorManager(components)

        manager.add_grouping_rule(r'Solar_\d+', 'solar_tech', 'oranges', match_type='regex')

        assert manager._grouping_rules[0]['match_type'] == 'regex'

    def test_auto_group_components(self):
        """Test auto-grouping components based on rules."""
        components = ['Solar_PV', 'Solar_Thermal', 'Wind_Onshore', 'Wind_Offshore', 'Coal_Plant']
        manager = ComponentColorManager(components)

        manager.add_grouping_rule('Solar', 'renewables_solar', 'oranges', match_type='prefix')
        manager.add_grouping_rule('Wind', 'renewables_wind', 'blues', match_type='prefix')
        manager.add_grouping_rule('Coal', 'fossil', 'greys', match_type='prefix')
        manager.auto_group_components()

        # Check that components got colors from appropriate families
        solar_color = manager.get_color('Solar_PV')
        wind_color = manager.get_color('Wind_Onshore')

        assert solar_color is not None
        assert wind_color is not None
        # Colors should be different (from different families)
        assert solar_color != wind_color

    def test_first_match_wins(self):
        """Test that first matching rule wins."""
        components = ['Solar_Wind_Hybrid']
        manager = ComponentColorManager(components)

        manager.add_grouping_rule('Solar', 'solar', 'oranges', match_type='prefix')
        manager.add_grouping_rule('Wind', 'wind', 'blues', match_type='contains')
        manager.auto_group_components()

        # Should match 'Solar' rule first (prefix match)
        color = manager.get_color('Solar_Wind_Hybrid')
        # Should be from oranges family (first rule)
        assert 'rgb' in color.lower()


class TestColorStability:
    """Test color stability across different datasets."""

    def test_same_component_same_color(self):
        """Test that same component always gets same color."""
        components = ['Solar_PV', 'Wind_Onshore', 'Coal_Plant', 'Gas_Plant']
        manager = ComponentColorManager(components)

        manager.add_grouping_rule('Solar', 'renewables_solar', 'oranges', match_type='prefix')
        manager.add_grouping_rule('Wind', 'renewables_wind', 'blues', match_type='prefix')
        manager.auto_group_components()

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
        manager.auto_group_components()

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

        variable = 'Wind_Turbine|investment_size'
        component = manager.extract_component(variable)

        assert component == 'Wind_Turbine'

    def test_extract_component_with_both(self):
        """Test extracting component from variable with both separators."""
        manager = ComponentColorManager([])

        variable = 'Gas_Plant(HeatBus)|flow_rate'
        component = manager.extract_component(variable)

        assert component == 'Gas_Plant'

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
        manager.auto_group_components()

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
        manager.auto_group_components()

        variables = ['Solar_PV(Bus)|flow_rate', 'Wind_Onshore(Bus)|flow_rate', 'Coal_Plant(Bus)|flow_rate']

        colors = manager.get_variable_colors(variables)

        assert len(colors) == 3
        assert all(var in colors for var in variables)
        assert all(isinstance(color, str) for color in colors.values())

    def test_get_variable_color_unknown_component(self):
        """Test getting color for variable with unknown component."""
        components = ['Solar_PV']
        manager = ComponentColorManager(components)

        variable = 'Unknown_Component(Bus)|flow_rate'
        color = manager.get_variable_color(variable)

        # Should still return a color (from default colormap or fallback)
        assert color is not None


class TestOverrides:
    """Test override functionality."""

    def test_simple_override(self):
        """Test simple color override."""
        components = ['Solar_PV', 'Wind_Onshore']
        manager = ComponentColorManager(components)
        manager.add_grouping_rule('Solar', 'solar', 'oranges', match_type='prefix')
        manager.auto_group_components()

        # Override Solar_PV color
        manager.override({'Solar_PV': '#FF0000'})

        color = manager.get_color('Solar_PV')
        assert color == '#FF0000'

    def test_multiple_overrides(self):
        """Test multiple overrides."""
        components = ['Solar_PV', 'Wind_Onshore', 'Coal_Plant']
        manager = ComponentColorManager(components)
        manager.auto_group_components()

        manager.override({'Solar_PV': '#FF0000', 'Wind_Onshore': '#00FF00'})

        assert manager.get_color('Solar_PV') == '#FF0000'
        assert manager.get_color('Wind_Onshore') == '#00FF00'

    def test_override_precedence(self):
        """Test that overrides take precedence over grouping rules."""
        components = ['Solar_PV']
        manager = ComponentColorManager(components)
        manager.add_grouping_rule('Solar', 'solar', 'oranges', match_type='prefix')
        manager.auto_group_components()

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
        manager.auto_group_components()

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

    def test_resolve_colors_with_colormap_name(self):
        """Test that resolve_colors still works with colormap name."""
        dataset = xr.Dataset(
            {'var1': (['time'], np.random.rand(10)), 'var2': (['time'], np.random.rand(10))},
            coords={'time': np.arange(10)},
        )

        colors = resolve_colors(dataset, 'viridis', engine='plotly')

        assert len(colors) == 2
        assert 'var1' in colors
        assert 'var2' in colors


class TestToDictMethod:
    """Test to_dict method."""

    def test_to_dict_returns_all_colors(self):
        """Test that to_dict returns colors for all components."""
        components = ['Comp1', 'Comp2', 'Comp3']
        manager = ComponentColorManager(components)

        color_dict = manager.to_dict()

        assert len(color_dict) == 3
        assert all(comp in color_dict for comp in components)

    def test_to_dict_with_grouping(self):
        """Test to_dict with grouping applied."""
        components = ['Solar_PV', 'Solar_Thermal', 'Wind_Onshore']
        manager = ComponentColorManager(components)
        manager.add_grouping_rule('Solar', 'solar', 'oranges', match_type='prefix')
        manager.auto_group_components()

        color_dict = manager.to_dict()

        assert len(color_dict) == 3
        assert 'Solar_PV' in color_dict
        assert 'Solar_Thermal' in color_dict
        assert 'Wind_Onshore' in color_dict


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_components_list(self):
        """Test with empty components list."""
        manager = ComponentColorManager([])

        assert manager.components == []
        assert manager.to_dict() == {}

    def test_get_color_for_missing_component(self):
        """Test getting color for component not in list."""
        components = ['Comp1']
        manager = ComponentColorManager(components)

        # Should return a color (fallback behavior)
        color = manager.get_color('MissingComponent')
        assert color is not None

    def test_invalid_match_type(self):
        """Test that invalid match type raises error."""
        manager = ComponentColorManager([])

        with pytest.raises(ValueError, match='match_type must be one of'):
            manager.add_grouping_rule('test', 'group', 'blues', match_type='invalid')


class TestStringRepresentation:
    """Test __repr__ and __str__ methods."""

    def test_repr_simple(self):
        """Test __repr__ with simple manager."""
        components = ['Comp1', 'Comp2', 'Comp3']
        manager = ComponentColorManager(components)

        repr_str = repr(manager)

        assert 'ComponentColorManager' in repr_str
        assert 'components=3' in repr_str
        assert 'rules=0' in repr_str
        assert 'overrides=0' in repr_str
        assert "default_colormap='viridis'" in repr_str

    def test_repr_with_rules_and_overrides(self):
        """Test __repr__ with rules and overrides."""
        manager = ComponentColorManager(['Solar_PV', 'Wind_Onshore'])
        manager.add_grouping_rule('Solar', 'solar', 'oranges', match_type='prefix')
        manager.add_grouping_rule('Wind', 'wind', 'blues', match_type='prefix')
        manager.override({'Solar_PV': '#FF0000'})

        repr_str = repr(manager)

        assert 'components=2' in repr_str
        assert 'rules=2' in repr_str
        assert 'overrides=1' in repr_str

    def test_str_simple(self):
        """Test __str__ with simple manager."""
        components = ['Comp1', 'Comp2', 'Comp3']
        manager = ComponentColorManager(components)

        str_output = str(manager)

        assert 'ComponentColorManager' in str_output
        assert 'Components: 3' in str_output
        assert 'Comp1' in str_output
        assert 'Comp2' in str_output
        assert 'Comp3' in str_output
        assert 'Grouping rules: 0' in str_output
        assert 'Overrides: 0' in str_output
        assert 'Default colormap: viridis' in str_output

    def test_str_with_many_components(self):
        """Test __str__ with many components (truncation)."""
        components = [f'Comp{i}' for i in range(10)]
        manager = ComponentColorManager(components)

        str_output = str(manager)

        assert 'Components: 10' in str_output
        assert '... (5 more)' in str_output

    def test_str_with_grouping_rules(self):
        """Test __str__ with grouping rules."""
        manager = ComponentColorManager(['Solar_PV', 'Wind_Onshore'])
        manager.add_grouping_rule('Solar', 'solar', 'oranges', match_type='prefix')
        manager.add_grouping_rule('Wind', 'wind', 'blues', match_type='suffix')

        str_output = str(manager)

        assert 'Grouping rules: 2' in str_output
        assert "prefix('Solar')" in str_output
        assert "suffix('Wind')" in str_output
        assert 'oranges' in str_output
        assert 'blues' in str_output

    def test_str_with_many_rules(self):
        """Test __str__ with many rules (truncation)."""
        manager = ComponentColorManager([])
        for i in range(5):
            manager.add_grouping_rule(f'Pattern{i}', f'group{i}', 'blues', match_type='prefix')

        str_output = str(manager)

        assert 'Grouping rules: 5' in str_output
        assert '... and 2 more' in str_output

    def test_str_with_overrides(self):
        """Test __str__ with overrides."""
        manager = ComponentColorManager(['Comp1', 'Comp2'])
        manager.override({'Comp1': '#FF0000', 'Comp2': '#00FF00'})

        str_output = str(manager)

        assert 'Overrides: 2' in str_output
        assert 'Comp1: #FF0000' in str_output
        assert 'Comp2: #00FF00' in str_output

    def test_str_with_many_overrides(self):
        """Test __str__ with many overrides (truncation)."""
        manager = ComponentColorManager([])
        overrides = {f'Comp{i}': f'#FF{i:04X}' for i in range(5)}
        manager.override(overrides)

        str_output = str(manager)

        assert 'Overrides: 5' in str_output
        assert '... and 2 more' in str_output

    def test_str_empty_manager(self):
        """Test __str__ with empty manager."""
        manager = ComponentColorManager([])

        str_output = str(manager)

        assert 'Components: 0' in str_output
        assert 'Grouping rules: 0' in str_output
        assert 'Overrides: 0' in str_output


class TestMethodChaining:
    """Test method chaining."""

    def test_chaining_add_custom_family(self):
        """Test that add_custom_family returns self for chaining."""
        manager = ComponentColorManager([])
        result = manager.add_custom_family('ocean', ['#003f5c'])
        assert result is manager

    def test_chaining_add_grouping_rule(self):
        """Test that add_grouping_rule returns self for chaining."""
        manager = ComponentColorManager([])
        result = manager.add_grouping_rule('Solar', 'solar', 'oranges')
        assert result is manager

    def test_full_chaining(self):
        """Test full method chaining."""
        components = ['Solar_PV', 'Wind_Onshore', 'Gas_Plant']
        manager = (
            ComponentColorManager(components)
            .add_custom_family('ocean', ['#003f5c', '#2f4b7c'])
            .add_grouping_rule('Solar', 'renewables', 'oranges', match_type='prefix')
            .add_grouping_rule('Wind', 'renewables', 'blues', match_type='prefix')
        )

        assert 'ocean' in manager.color_families
        assert len(manager._grouping_rules) == 2
