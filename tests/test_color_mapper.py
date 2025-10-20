"""Tests for XarrayColorMapper functionality."""

import numpy as np
import pytest
import xarray as xr

from flixopt.plotting import XarrayColorMapper


class TestBasicFunctionality:
    """Test basic XarrayColorMapper functionality."""

    def test_initialization_default(self):
        """Test default initialization."""
        mapper = XarrayColorMapper()
        assert len(mapper.get_families()) == 14  # Default families
        assert 'blues' in mapper.get_families()
        assert mapper.sort_within_groups is True

    def test_initialization_custom_families(self):
        """Test initialization with custom color families."""
        custom_families = {
            'custom1': ['#FF0000', '#00FF00', '#0000FF'],
            'custom2': ['#FFFF00', '#FF00FF', '#00FFFF'],
        }
        mapper = XarrayColorMapper(color_families=custom_families, sort_within_groups=False)
        assert len(mapper.get_families()) == 2
        assert 'custom1' in mapper.get_families()
        assert mapper.sort_within_groups is False

    def test_add_custom_family(self):
        """Test adding a custom color family."""
        mapper = XarrayColorMapper()
        mapper.add_custom_family('ocean', ['#003f5c', '#2f4b7c', '#665191'])
        assert 'ocean' in mapper.get_families()
        assert len(mapper.get_families()['ocean']) == 3


class TestPatternMatching:
    """Test pattern matching functionality."""

    def test_prefix_matching(self):
        """Test prefix pattern matching."""
        mapper = XarrayColorMapper()
        mapper.add_rule('Product_A', 'blues', 'prefix')
        mapper.add_rule('Product_B', 'greens', 'prefix')

        categories = ['Product_A1', 'Product_A2', 'Product_B1', 'Product_B2', 'Other']
        groups = mapper._group_categories(categories)

        assert 'blues' in groups
        assert 'greens' in groups
        assert '_unmatched' in groups
        assert 'Product_A1' in groups['blues']
        assert 'Product_B1' in groups['greens']
        assert 'Other' in groups['_unmatched']

    def test_suffix_matching(self):
        """Test suffix pattern matching."""
        mapper = XarrayColorMapper()
        mapper.add_rule('_test', 'blues', 'suffix')
        mapper.add_rule('_prod', 'greens', 'suffix')

        categories = ['system_test', 'system_prod', 'development']
        groups = mapper._group_categories(categories)

        assert 'system_test' in groups['blues']
        assert 'system_prod' in groups['greens']
        assert 'development' in groups['_unmatched']

    def test_contains_matching(self):
        """Test contains pattern matching."""
        mapper = XarrayColorMapper()
        mapper.add_rule('renewable', 'greens', 'contains')
        mapper.add_rule('fossil', 'reds', 'contains')

        categories = ['renewable_wind', 'fossil_gas', 'renewable_solar', 'battery']
        groups = mapper._group_categories(categories)

        assert 'renewable_wind' in groups['greens']
        assert 'fossil_gas' in groups['reds']
        assert 'battery' in groups['_unmatched']

    def test_glob_matching(self):
        """Test glob pattern matching."""
        mapper = XarrayColorMapper()
        mapper.add_rule('Product_A*', 'blues', 'glob')
        mapper.add_rule('*_test', 'greens', 'glob')

        categories = ['Product_A1', 'Product_A2', 'system_test', 'Other']
        groups = mapper._group_categories(categories)

        assert 'Product_A1' in groups['blues']
        assert 'system_test' in groups['greens']
        assert 'Other' in groups['_unmatched']

    def test_regex_matching(self):
        """Test regex pattern matching."""
        mapper = XarrayColorMapper()
        mapper.add_rule(r'^exp_[AB]\d+$', 'blues', 'regex')

        categories = ['exp_A1', 'exp_B2', 'exp_C1', 'test']
        groups = mapper._group_categories(categories)

        assert 'exp_A1' in groups['blues']
        assert 'exp_B2' in groups['blues']
        assert 'exp_C1' in groups['_unmatched']

    def test_invalid_regex(self):
        """Test that invalid regex raises error."""
        mapper = XarrayColorMapper()
        mapper.add_rule('[invalid', 'blues', 'regex')

        with pytest.raises(ValueError, match='Invalid regex pattern'):
            mapper._match_rule('test', mapper.rules[0])


class TestColorMapping:
    """Test color mapping creation."""

    def test_create_color_map_with_list(self):
        """Test creating color map from a list."""
        mapper = XarrayColorMapper()
        mapper.add_rule('A', 'blues', 'prefix')
        mapper.add_rule('B', 'greens', 'prefix')

        categories = ['A1', 'A2', 'B1', 'B2']
        color_map = mapper.create_color_map(categories)

        assert len(color_map) == 4
        assert all(key in color_map for key in categories)
        # A items should have blue colors, B items should have green colors
        # (We can't assert exact colors as they come from plotly, but we can check they exist)

    def test_create_color_map_with_numpy_array(self):
        """Test creating color map from numpy array."""
        mapper = XarrayColorMapper()
        mapper.add_rule('Product', 'blues', 'prefix')

        categories = np.array(['Product_A', 'Product_B', 'Product_C'])
        color_map = mapper.create_color_map(categories)

        assert len(color_map) == 3
        assert 'Product_A' in color_map

    def test_create_color_map_with_xarray(self):
        """Test creating color map from xarray DataArray."""
        mapper = XarrayColorMapper()
        mapper.add_rule('Product', 'blues', 'prefix')

        da = xr.DataArray([1, 2, 3], coords={'product': ['Product_A', 'Product_B', 'Product_C']}, dims=['product'])
        color_map = mapper.create_color_map(da.coords['product'])

        assert len(color_map) == 3
        assert 'Product_A' in color_map

    def test_sorting_within_groups(self):
        """Test that sorting within groups works correctly."""
        mapper = XarrayColorMapper(sort_within_groups=True)
        mapper.add_rule('Product', 'blues', 'prefix')

        categories = ['Product_C', 'Product_A', 'Product_B']
        color_map = mapper.create_color_map(categories)

        # With sorting, the order should be alphabetical
        keys = list(color_map.keys())
        assert keys == ['Product_A', 'Product_B', 'Product_C']

    def test_no_sorting_within_groups(self):
        """Test that disabling sorting preserves order."""
        mapper = XarrayColorMapper(sort_within_groups=False)
        mapper.add_rule('Product', 'blues', 'prefix')

        categories = ['Product_C', 'Product_A', 'Product_B']
        color_map = mapper.create_color_map(categories, sort_within_groups=False)

        # Without sorting, order should match rules order, then input order within group
        keys = list(color_map.keys())
        assert keys == ['Product_C', 'Product_A', 'Product_B']


class TestOverrides:
    """Test override functionality."""

    def test_override_simple(self):
        """Test simple override."""
        mapper = XarrayColorMapper()
        mapper.add_rule('Product', 'blues', 'prefix')
        mapper.add_override({'Product_A': '#FF0000'})

        categories = ['Product_A', 'Product_B']
        color_map = mapper.create_color_map(categories)

        assert color_map['Product_A'] == '#FF0000'
        assert color_map['Product_B'] != '#FF0000'  # Should use blues

    def test_override_multiple(self):
        """Test multiple overrides."""
        mapper = XarrayColorMapper()
        mapper.add_rule('Product', 'blues', 'prefix')
        mapper.add_override({'Product_A': '#FF0000', 'Product_B': '#00FF00'})

        categories = ['Product_A', 'Product_B', 'Product_C']
        color_map = mapper.create_color_map(categories)

        assert color_map['Product_A'] == '#FF0000'
        assert color_map['Product_B'] == '#00FF00'
        # Product_C should use the rule

    def test_override_precedence(self):
        """Test that overrides take precedence over rules."""
        mapper = XarrayColorMapper()
        mapper.add_rule('Special', 'blues', 'prefix')
        mapper.add_override({'Special_Case': '#FFD700'})

        categories = ['Special_Case', 'Special_Normal']
        color_map = mapper.create_color_map(categories)

        # Override should take precedence
        assert color_map['Special_Case'] == '#FFD700'


class TestXarrayIntegration:
    """Test integration with xarray DataArrays."""

    def test_apply_to_dataarray(self):
        """Test applying mapper to a DataArray."""
        mapper = XarrayColorMapper()
        mapper.add_rule('Product_A', 'blues', 'prefix')
        mapper.add_rule('Product_B', 'greens', 'prefix')

        da = xr.DataArray(
            np.random.rand(5, 4),
            coords={'time': range(5), 'product': ['Product_A1', 'Product_A2', 'Product_B1', 'Product_B2']},
            dims=['time', 'product'],
        )

        color_map = mapper.apply_to_dataarray(da, 'product')

        assert len(color_map) == 4
        assert all(prod in color_map for prod in da.product.values)

    def test_apply_to_dataarray_missing_coord(self):
        """Test that applying to missing coordinate raises error."""
        mapper = XarrayColorMapper()
        da = xr.DataArray(np.random.rand(5), coords={'time': range(5)}, dims=['time'])

        with pytest.raises(ValueError, match="Coordinate 'product' not found"):
            mapper.apply_to_dataarray(da, 'product')

    def test_reorder_coordinate(self):
        """Test reordering coordinates."""
        mapper = XarrayColorMapper()
        mapper.add_rule('A', 'blues', 'prefix')
        mapper.add_rule('B', 'greens', 'prefix')

        da = xr.DataArray(
            np.random.rand(4),
            coords={'product': ['B2', 'A1', 'B1', 'A2']},
            dims=['product'],
        )

        da_reordered = mapper.reorder_coordinate(da, 'product')

        # With sorting, items are grouped by family (order of first occurrence in input),
        # then sorted within each group
        # B items are encountered first, so greens group comes first
        expected_order = ['B1', 'B2', 'A1', 'A2']
        assert [str(v) for v in da_reordered.product.values] == expected_order

    def test_reorder_coordinate_preserves_data(self):
        """Test that reordering preserves data values."""
        mapper = XarrayColorMapper()
        mapper.add_rule('A', 'blues', 'prefix')

        original_data = np.array([10, 20, 30, 40])
        da = xr.DataArray(original_data, coords={'product': ['A4', 'A1', 'A3', 'A2']}, dims=['product'])

        da_reordered = mapper.reorder_coordinate(da, 'product')

        # Check that the data is correctly reordered with the coordinates
        assert da_reordered.sel(product='A1').values == 20
        assert da_reordered.sel(product='A2').values == 40
        assert da_reordered.sel(product='A3').values == 30
        assert da_reordered.sel(product='A4').values == 10


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_categories(self):
        """Test with empty categories list."""
        mapper = XarrayColorMapper()
        color_map = mapper.create_color_map([])
        assert color_map == {}

    def test_duplicate_categories(self):
        """Test that duplicates are handled correctly."""
        mapper = XarrayColorMapper()
        mapper.add_rule('Product', 'blues', 'prefix')

        # Duplicates should be removed
        categories = ['Product_A', 'Product_B', 'Product_A', 'Product_B']
        color_map = mapper.create_color_map(categories)

        assert len(color_map) == 2
        assert 'Product_A' in color_map
        assert 'Product_B' in color_map

    def test_invalid_match_type(self):
        """Test that invalid match type raises error."""
        mapper = XarrayColorMapper()
        with pytest.raises(ValueError, match='match_type must be one of'):
            mapper.add_rule('Product', 'blues', 'invalid_type')

    def test_first_match_wins(self):
        """Test that first matching rule wins."""
        mapper = XarrayColorMapper()
        mapper.add_rule('Product', 'blues', 'prefix')
        mapper.add_rule('Product_A', 'greens', 'prefix')  # More specific rule added second

        categories = ['Product_A1', 'Product_B1']
        groups = mapper._group_categories(categories)

        # Both should match the first rule (Product) since it's added first
        assert 'Product_A1' in groups['blues']
        assert 'Product_B1' in groups['blues']

    def test_more_items_than_colors(self):
        """Test behavior when there are more items than colors in a family."""
        mapper = XarrayColorMapper()
        mapper.add_rule('Item', 'blues', 'prefix')

        # Create many items (more than the 5 colors in blues family)
        categories = [f'Item_{i}' for i in range(10)]
        color_map = mapper.create_color_map(categories)

        # Should cycle through colors
        assert len(color_map) == 10
        # First and 6th item should have the same color (cycling)
        assert color_map['Item_0'] == color_map['Item_7']


class TestInspectionMethods:
    """Test inspection methods."""

    def test_get_rules(self):
        """Test getting rules."""
        mapper = XarrayColorMapper()
        mapper.add_rule('Product_A', 'blues', 'prefix')
        mapper.add_rule('Product_B', 'greens', 'suffix')

        rules = mapper.get_rules()
        assert len(rules) == 2
        assert rules[0]['pattern'] == 'Product_A'
        assert rules[0]['family'] == 'blues'
        assert rules[0]['match_type'] == 'prefix'

    def test_get_overrides(self):
        """Test getting overrides."""
        mapper = XarrayColorMapper()
        mapper.add_override({'Special': '#FFD700', 'Other': '#FF0000'})

        overrides = mapper.get_overrides()
        assert len(overrides) == 2
        assert overrides['Special'] == '#FFD700'

    def test_get_families(self):
        """Test getting color families."""
        mapper = XarrayColorMapper()
        families = mapper.get_families()

        assert 'blues' in families
        assert 'greens' in families
        assert len(families['blues']) == 7  # Blues[1:8] has 5 colors


class TestMethodChaining:
    """Test method chaining."""

    def test_chaining_add_rule(self):
        """Test that add_rule returns self for chaining."""
        mapper = XarrayColorMapper()
        result = mapper.add_rule('Product', 'blues', 'prefix')
        assert result is mapper

    def test_chaining_add_override(self):
        """Test that add_override returns self for chaining."""
        mapper = XarrayColorMapper()
        result = mapper.add_override({'Special': '#FFD700'})
        assert result is mapper

    def test_chaining_add_custom_family(self):
        """Test that add_custom_family returns self for chaining."""
        mapper = XarrayColorMapper()
        result = mapper.add_custom_family('custom', ['#FF0000'])
        assert result is mapper

    def test_full_chaining(self):
        """Test full method chaining."""
        mapper = (
            XarrayColorMapper()
            .add_custom_family('ocean', ['#003f5c', '#2f4b7c'])
            .add_rule('Product_A', 'blues', 'prefix')
            .add_rule('Product_B', 'greens', 'prefix')
            .add_override({'Special': '#FFD700'})
        )

        assert len(mapper.get_rules()) == 2
        assert len(mapper.get_overrides()) == 1
        assert 'ocean' in mapper.get_families()


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_energy_system_components(self):
        """Test color mapping for energy system components."""
        mapper = (
            XarrayColorMapper()
            .add_rule('Solar', 'oranges', 'prefix')
            .add_rule('Wind', 'blues', 'prefix')
            .add_rule('Gas', 'reds', 'prefix')
            .add_rule('Battery', 'greens', 'prefix')
            .add_override({'Grid_Import': '#808080'})
        )

        components = ['Solar_PV', 'Wind_Turbine', 'Gas_Turbine', 'Battery_Storage', 'Grid_Import']
        da = xr.DataArray(
            np.random.rand(24, len(components)),
            coords={'time': range(24), 'component': components},
            dims=['time', 'component'],
        )

        color_map = mapper.apply_to_dataarray(da, 'component')

        assert color_map['Grid_Import'] == '#808080'  # Override
        assert all(comp in color_map for comp in components)

    def test_scenario_analysis(self):
        """Test color mapping for scenario analysis."""
        mapper = (
            XarrayColorMapper()
            .add_rule('baseline*', 'greys', 'glob')
            .add_rule('renewable_high*', 'greens', 'glob')
            .add_rule('renewable_low*', 'teals', 'glob')
            .add_rule('fossil*', 'reds', 'glob')
        )

        scenarios = [
            'baseline_2030',
            'baseline_2050',
            'renewable_high_2030',
            'renewable_low_2050',
            'fossil_phase_out_2040',
        ]

        color_map = mapper.create_color_map(scenarios)
        assert len(color_map) == 5

    def test_product_tiers(self):
        """Test color mapping for product tiers."""
        mapper = (
            XarrayColorMapper()
            .add_rule('Premium_', 'purples', 'prefix')
            .add_rule('Standard_', 'blues', 'prefix')
            .add_rule('Budget_', 'greens', 'prefix')
        )

        products = ['Premium_A', 'Premium_B', 'Standard_A', 'Standard_B', 'Budget_A', 'Budget_B']
        da = xr.DataArray(
            np.random.rand(10, 6), coords={'time': range(10), 'product': products}, dims=['time', 'product']
        )

        da_reordered = mapper.reorder_coordinate(da, 'product')
        mapper.apply_to_dataarray(da_reordered, 'product')

        # Check grouping: all Premium together, then Standard, then Budget
        reordered_products = list(da_reordered.product.values)
        premium_indices = [i for i, p in enumerate(reordered_products) if p.startswith('Premium_')]
        standard_indices = [i for i, p in enumerate(reordered_products) if p.startswith('Standard_')]
        budget_indices = [i for i, p in enumerate(reordered_products) if p.startswith('Budget_')]

        # Check that groups are contiguous
        assert premium_indices == list(range(min(premium_indices), max(premium_indices) + 1))
        assert standard_indices == list(range(min(standard_indices), max(standard_indices) + 1))
        assert budget_indices == list(range(min(budget_indices), max(budget_indices) + 1))
