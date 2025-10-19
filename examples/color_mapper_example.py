"""
XarrayColorMapper Usage Example
================================

This example demonstrates how to use the XarrayColorMapper for creating
meaningful, visually-grouped color schemes in your plots. Instead of random
colors, it assigns color families (like shades of blue, green, red) based on
coordinate value patterns, making plots more intuitive.

INTEGRATION WITH CALCULATIONRESULTS:
The XarrayColorMapper integrates seamlessly with flixopt's CalculationResults
plotting methods. Use results.create_color_mapper() to get started, then pass
the generated color maps directly to any plotting method via the `colors` parameter.
"""

import numpy as np
import plotly.express as px
import xarray as xr

from flixopt.plotting import XarrayColorMapper, with_plotly

print('=' * 70)
print('XarrayColorMapper - Comprehensive Usage Example')
print('=' * 70)
print()

# ============================================================================
# Example 1: Basic Pattern-Based Coloring
# ============================================================================
print('Example 1: Basic Pattern-Based Coloring')
print('-' * 70)

# Create sample data with product categories
np.random.seed(42)
time_coords = np.arange(0, 24)
products = ['Premium_A', 'Premium_B', 'Standard_A', 'Standard_B', 'Budget_A', 'Budget_B']

data = xr.DataArray(
    np.random.rand(24, 6) * 100 + np.array([100, 90, 70, 65, 40, 35]),  # Different base values
    coords={'time': time_coords, 'product': products},
    dims=['time', 'product'],
    name='sales',
)

# Setup color mapper with rules based on product tiers
mapper = (
    XarrayColorMapper()
    .add_rule('Premium_', 'purples', 'prefix')  # Premium products get purple shades
    .add_rule('Standard_', 'blues', 'prefix')  # Standard products get blue shades
    .add_rule('Budget_', 'greens', 'prefix')
)  # Budget products get green shades

# Reorder products by tier for better visual grouping
data_reordered = mapper.reorder_coordinate(data, 'product')
print('Products reordered by tier:', data_reordered.product.values)

# Get color mapping
color_map = mapper.apply_to_dataarray(data_reordered, 'product')
print('\nColor assignments:')
for product, color in color_map.items():
    print(f'  {product}: {color}')

print('\n✓ Data prepared with pattern-based color grouping')
print()

# ============================================================================
# Example 2: Using with Plotly Express
# ============================================================================
print('Example 2: Integration with Plotly Express')
print('-' * 70)

# Convert to DataFrame for plotting
df = data_reordered.to_dataframe(name='sales').reset_index()

# Create a line plot with the custom color mapping
fig = px.line(
    df,
    x='time',
    y='sales',
    color='product',
    color_discrete_map=color_map,  # Use our custom color mapping
    title='Product Sales Over Time (Grouped by Tier)',
    labels={'time': 'Hour of Day', 'sales': 'Sales ($)', 'product': 'Product'},
    markers=True,
)

print('✓ Plotly figure created with grouped colors')
print('  - Premium products: Purple shades')
print('  - Standard products: Blue shades')
print('  - Budget products: Green shades')
print()

# To save or show the plot:
# fig.write_html('product_sales.html')
# fig.show()

# ============================================================================
# Example 3: Using with flixopt's with_plotly
# ============================================================================
print('Example 3: Integration with flixopt.plotting.with_plotly')
print('-' * 70)

# The color_map can be passed directly to with_plotly
fig2 = with_plotly(
    data_reordered,
    mode='area',
    colors=color_map,  # Pass the color mapping directly
    title='Product Sales - Stacked Area Chart',
    ylabel='Sales ($)',
    xlabel='Hour of Day',
)

print('✓ Created stacked area chart with grouped colors using with_plotly()')
print()

# ============================================================================
# Example 4: Advanced Pattern Matching
# ============================================================================
print('Example 4: Advanced Pattern Matching (Glob and Regex)')
print('-' * 70)

# Create scenario data with complex naming
scenarios = [
    'baseline_2020',
    'baseline_2030',
    'baseline_2050',
    'renewable_high_2030',
    'renewable_high_2050',
    'renewable_low_2030',
    'renewable_low_2050',
    'fossil_phase_out_2030',
    'fossil_phase_out_2050',
]

scenario_data = xr.DataArray(
    np.random.rand(10, len(scenarios)),
    coords={'time': np.arange(10), 'scenario': scenarios},
    dims=['time', 'scenario'],
    name='emissions',
)

# Setup mapper with different pattern types
scenario_mapper = (
    XarrayColorMapper()
    .add_rule('baseline*', 'greys', 'glob')  # Glob pattern for baseline scenarios
    .add_rule('renewable_high*', 'greens', 'glob')  # High renewable scenarios
    .add_rule('renewable_low*', 'teals', 'glob')  # Low renewable scenarios
    .add_rule('fossil*', 'reds', 'glob')
)  # Fossil phase-out scenarios

# Apply coloring
scenario_data_reordered = scenario_mapper.reorder_coordinate(scenario_data, 'scenario')
scenario_colors = scenario_mapper.apply_to_dataarray(scenario_data_reordered, 'scenario')

print('Scenarios grouped by type:')
print('  Reordered scenarios:', list(scenario_data_reordered.scenario.values))
print('\nColor assignments:')
for scenario, color in scenario_colors.items():
    print(f'  {scenario}: {color}')

print('\n✓ Complex pattern matching with glob patterns')
print()

# ============================================================================
# Example 5: Using Overrides for Special Cases
# ============================================================================
print('Example 5: Using Overrides for Special Cases')
print('-' * 70)

# Create component data
components = ['Solar_PV', 'Wind_Turbine', 'Gas_Turbine', 'Battery_Storage', 'Grid_Import']

component_data = xr.DataArray(
    np.random.rand(20, len(components)),
    coords={'time': np.arange(20), 'component': components},
    dims=['time', 'component'],
    name='power',
)

# Setup mapper with rules and special overrides
component_mapper = (
    XarrayColorMapper()
    .add_rule('Solar', 'oranges', 'prefix')  # Solar components
    .add_rule('Wind', 'blues', 'prefix')  # Wind components
    .add_rule('Gas', 'reds', 'prefix')  # Gas components
    .add_rule('Battery', 'greens', 'prefix')  # Battery components
    .add_override(
        {  # Special cases override the rules
            'Grid_Import': '#808080'  # Grey for grid import
        }
    )
)

component_colors = component_mapper.apply_to_dataarray(component_data, 'component')

print('Component color assignments:')
for component, color in component_colors.items():
    override_marker = ' [OVERRIDE]' if component == 'Grid_Import' else ''
    print(f'  {component}: {color}{override_marker}')

print('\n✓ Rules with override for special cases')
print()

# ============================================================================
# Example 6: Custom Color Families
# ============================================================================
print('Example 6: Creating Custom Color Families')
print('-' * 70)

# Create custom color families for specific use cases
custom_mapper = XarrayColorMapper()
custom_mapper.add_custom_family('ocean', ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087'])
custom_mapper.add_custom_family('sunset', ['#ffa600', '#ff7c43', '#f95d6a', '#d45087', '#a05195'])

custom_mapper.add_rule('ocean_', 'ocean', 'prefix')
custom_mapper.add_rule('land_', 'sunset', 'prefix')

zones = ['ocean_shallow', 'ocean_deep', 'ocean_reef', 'land_forest', 'land_desert', 'land_urban']
zone_data = xr.DataArray(
    np.random.rand(10, len(zones)),
    coords={'time': np.arange(10), 'zone': zones},
    dims=['time', 'zone'],
    name='temperature',
)

zone_colors = custom_mapper.apply_to_dataarray(zone_data, 'zone')

print('Custom color families:')
print('  Available families:', list(custom_mapper.get_families().keys()))
print('\nZone color assignments:')
for zone, color in zone_colors.items():
    print(f'  {zone}: {color}')

print('\n✓ Custom color families defined and applied')
print()

# ============================================================================
# Example 7: Integration with CalculationResults (Pattern)
# ============================================================================
print('Example 7: Integration with CalculationResults')
print('-' * 70)

print("""
USAGE PATTERN WITH CALCULATIONRESULTS:

# METHOD 1: Automatic usage (recommended)
# Create and configure mapper - it's automatically used by all plots
mapper = results.create_color_mapper()
mapper.add_rule('Solar', 'oranges', 'prefix')
mapper.add_rule('Wind', 'blues', 'prefix')
mapper.add_rule('Gas', 'reds', 'prefix')
mapper.add_rule('Battery', 'greens', 'prefix')

# All plotting methods automatically use the mapper (colors='auto' is default)
results['ElectricityBus'].plot_node_balance()  # Automatically uses mapper!
results['Battery'].plot_charge_state()          # Also uses mapper!

# METHOD 2: Manual color map generation
# Get data and generate colors manually for full control
data = results['ElectricityBus'].node_balance()
data_reordered = mapper.reorder_coordinate(data, 'variable')
colors = mapper.apply_to_dataarray(data_reordered, 'variable')

# Pass colors explicitly to plotting functions
results['ElectricityBus'].plot_node_balance(colors=colors)
fig = plotting.with_plotly(data_reordered, colors=colors)

# METHOD 3: Direct assignment of existing mapper
my_mapper = XarrayColorMapper()
my_mapper.add_rule('Renewable', 'greens', 'prefix')
results.color_mapper = my_mapper  # Direct assignment works too!
""")

print('✓ Pattern demonstrated for CalculationResults integration')
print()

# ============================================================================
# Summary
# ============================================================================
print('=' * 70)
print('Summary: XarrayColorMapper Key Benefits')
print('=' * 70)
print()
print('1. Visual Grouping: Similar items get similar colors automatically')
print('2. Pattern Matching: Flexible matching with prefix, suffix, glob, regex')
print('3. Coordinate Reordering: Group similar items together in plots')
print('4. Override Support: Handle special cases with explicit colors')
print('5. Custom Families: Define your own color schemes')
print('6. Easy Integration: Works seamlessly with Plotly and flixopt plotting')
print('7. CalculationResults Support: Convenience method for quick setup')
print()
print('=' * 70)
print('All examples completed successfully!')
print('=' * 70)
