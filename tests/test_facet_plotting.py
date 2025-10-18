"""
Example script demonstrating facet plotting and animation functionality.

This script shows how to use the new facet_by and animate_by parameters
to create multidimensional plots with scenarios and periods.
"""

import numpy as np
import pandas as pd
import xarray as xr

from flixopt import plotting

# Create synthetic multidimensional data for demonstration
# Dimensions: time, scenario, period
print('Creating synthetic multidimensional data...')

# Time dimension
time = pd.date_range('2024-01-01', periods=24 * 7, freq='h', name='time')

# Scenario dimension
scenarios = ['base', 'high_demand', 'renewable_focus', 'storage_heavy']

# Period dimension (e.g., different years or investment periods)
periods = [2024, 2030, 2040]

# Create sample data
np.random.seed(42)

# Create variables that will be plotted
variables = ['Solar', 'Wind', 'Gas', 'Battery_discharge', 'Battery_charge']

data_vars = {}
for var in variables:
    # Create different patterns for each variable
    base_pattern = np.sin(np.arange(len(time)) * 2 * np.pi / 24) * 50 + 100

    # Add scenario and period variations
    data = np.zeros((len(time), len(scenarios), len(periods)))

    for s_idx, _ in enumerate(scenarios):
        for p_idx, period in enumerate(periods):
            # Add scenario-specific variation
            scenario_factor = 1.0 + s_idx * 0.3
            # Add period-specific growth
            period_factor = 1.0 + (period - 2024) / 20 * 0.5
            # Add some randomness
            noise = np.random.normal(0, 10, len(time))

            data[:, s_idx, p_idx] = base_pattern * scenario_factor * period_factor + noise

            # Make battery charge negative for visualization
            if 'charge' in var.lower():
                data[:, s_idx, p_idx] = -np.abs(data[:, s_idx, p_idx])

    data_vars[var] = (['time', 'scenario', 'period'], data)

# Create xarray Dataset
ds = xr.Dataset(
    data_vars,
    coords={
        'time': time,
        'scenario': scenarios,
        'period': periods,
    },
)

print(f'Dataset shape: {ds.dims}')
print(f'Variables: {list(ds.data_vars)}')
print(f'Coordinates: {list(ds.coords)}')
print()

# ============================================================================
# Example 1: Simple faceting by scenario
# ============================================================================
print('=' * 70)
print('Example 1: Faceting by scenario (4 subplots)')
print('=' * 70)

# Filter to just one period for simplicity
ds_filtered = ds.sel(period=2024)

try:
    fig1 = plotting.with_plotly(
        ds_filtered,
        facet_by='scenario',
        mode='area',
        colors='portland',
        title='Energy Mix by Scenario (2024)',
        ylabel='Power (MW)',
        xlabel='Time',
        facet_cols=2,  # 2x2 grid
    )
    fig1.write_html('/tmp/facet_example_1_scenarios.html')
    print('✓ Created: /tmp/facet_example_1_scenarios.html')
    print('  4 subplots showing different scenarios')
    fig1.show()
except Exception as e:
    print(f'✗ Error in Example 1: {e}')
    import traceback

    traceback.print_exc()

print()

# ============================================================================
# Example 2: Animation by period
# ============================================================================
print('=' * 70)
print('Example 2: Animation by period')
print('=' * 70)

# Filter to just one scenario
ds_filtered2 = ds.sel(scenario='base')

try:
    fig2 = plotting.with_plotly(
        ds_filtered2,
        animate_by='period',
        mode='area',
        colors='viridis',
        title='Energy Mix Evolution Over Time (Base Scenario)',
        ylabel='Power (MW)',
        xlabel='Time',
    )
    fig2.write_html('/tmp/facet_example_2_animation.html')
    print('✓ Created: /tmp/facet_example_2_animation.html')
    print('  Animation cycling through periods: 2024, 2030, 2040')
except Exception as e:
    print(f'✗ Error in Example 2: {e}')
    import traceback

    traceback.print_exc()

print()

# ============================================================================
# Example 3: Combined faceting and animation
# ============================================================================
print('=' * 70)
print('Example 3: Facet by scenario AND animate by period')
print('=' * 70)

try:
    fig3 = plotting.with_plotly(
        ds,
        facet_by='scenario',
        animate_by='period',
        mode='stacked_bar',
        colors='portland',
        title='Energy Mix: Scenarios vs. Periods',
        ylabel='Power (MW)',
        xlabel='Time',
        facet_cols=2,
        # height_per_row now auto-sizes intelligently!
    )
    fig3.write_html('/tmp/facet_example_3_combined.html')
    print('✓ Created: /tmp/facet_example_3_combined.html')
    print('  4 subplots (scenarios) with animation through 3 periods')
    print('  Using intelligent auto-sizing (2 rows = 900px)')
except Exception as e:
    print(f'✗ Error in Example 3: {e}')
    import traceback

    traceback.print_exc()

print()

# ============================================================================
# Example 4: 2D faceting (scenario x period grid)
# ============================================================================
print('=' * 70)
print('Example 4: 2D faceting (scenario x period)')
print('=' * 70)

# Take just one week of data for clearer visualization
ds_week = ds.isel(time=slice(0, 24 * 7))

try:
    fig4 = plotting.with_plotly(
        ds_week,
        facet_by=['scenario', 'period'],
        mode='line',
        colors='viridis',
        title='Energy Mix: Full Grid (Scenario x Period)',
        ylabel='Power (MW)',
        xlabel='Time (one week)',
        facet_cols=3,  # 3 columns for 3 periods
    )
    fig4.write_html('/tmp/facet_example_4_2d_grid.html')
    print('✓ Created: /tmp/facet_example_4_2d_grid.html')
    print('  12 subplots (4 scenarios × 3 periods)')
except Exception as e:
    print(f'✗ Error in Example 4: {e}')
    import traceback

    traceback.print_exc()

print()

# ============================================================================
# Example 5: Area mode with positive AND negative values (faceted)
# ============================================================================
print('=' * 70)
print('Example 5: Area mode with positive AND negative values')
print('=' * 70)

# Create data with both positive and negative values for testing
print('Creating data with charging (negative) and discharging (positive)...')

try:
    fig5 = plotting.with_plotly(
        ds.sel(period=2024),
        facet_by='scenario',
        mode='area',
        colors='portland',
        title='Energy Balance with Charging/Discharging (Area Mode)',
        ylabel='Power (MW)',
        xlabel='Time',
        facet_cols=2,
    )
    fig5.write_html('/tmp/facet_example_5_area_pos_neg.html')
    print('✓ Created: /tmp/facet_example_5_area_pos_neg.html')
    print('  Area plot with both positive and negative values')
    print('  Negative values (battery charge) should stack downwards')
    print('  Positive values should stack upwards')
except Exception as e:
    print(f'✗ Error in Example 5: {e}')
    import traceback

    traceback.print_exc()

print()
print('=' * 70)
print('All examples completed!')
print('=' * 70)
print()
print('Next steps for testing with real flixopt data:')
print('1. Load your CalculationResults with scenario/period dimensions')
print("2. Use results['Component'].plot_node_balance(facet_by='scenario')")
print("3. Try animate_by='period' for time evolution visualization")
print("4. Combine both: facet_by='scenario', animate_by='period'")
