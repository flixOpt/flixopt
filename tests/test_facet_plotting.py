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

# ============================================================================
# Example 6: Stacked bar mode with animation
# ============================================================================
print('=' * 70)
print('Example 6: Stacked bar mode with animation')
print('=' * 70)

# Use hourly data for a few days for clearer stacked bars
ds_daily = ds.isel(time=slice(0, 24 * 3))  # 3 days

try:
    fig6 = plotting.with_plotly(
        ds_daily.sel(scenario='base'),
        animate_by='period',
        mode='stacked_bar',
        colors='portland',
        title='Daily Energy Profile Evolution (Stacked Bars)',
        ylabel='Power (MW)',
        xlabel='Time',
    )
    fig6.write_html('/tmp/facet_example_6_stacked_bar_anim.html')
    print('✓ Created: /tmp/facet_example_6_stacked_bar_anim.html')
    print('  Stacked bar chart with period animation')
except Exception as e:
    print(f'✗ Error in Example 6: {e}')
    import traceback

    traceback.print_exc()

print()

# ============================================================================
# Example 7: Large facet grid (test auto-sizing)
# ============================================================================
print('=' * 70)
print('Example 7: Large facet grid with auto-sizing')
print('=' * 70)

try:
    # Create more scenarios for a bigger grid
    extended_scenarios = scenarios + ['distributed', 'centralized']
    ds_extended = ds.copy()

    # Add new scenario data
    for var in variables:
        # Get existing data
        existing_data = ds[var].values

        # Create new scenarios with different patterns
        new_data = np.zeros((len(time), 2, len(periods)))
        for p_idx in range(len(periods)):
            new_data[:, 0, p_idx] = existing_data[:, 0, p_idx] * 0.8  # distributed
            new_data[:, 1, p_idx] = existing_data[:, 1, p_idx] * 1.2  # centralized

        # Combine old and new
        combined_data = np.concatenate([existing_data, new_data], axis=1)
        ds_extended[var] = (['time', 'scenario', 'period'], combined_data)

    ds_extended = ds_extended.assign_coords(scenario=extended_scenarios)

    fig7 = plotting.with_plotly(
        ds_extended.sel(period=2030),
        facet_by='scenario',
        mode='area',
        colors='viridis',
        title='Large Grid: 6 Scenarios Comparison',
        ylabel='Power (MW)',
        xlabel='Time',
        facet_cols=3,  # 3 columns, 2 rows
    )
    fig7.write_html('/tmp/facet_example_7_large_grid.html')
    print('✓ Created: /tmp/facet_example_7_large_grid.html')
    print('  6 subplots (2x3 grid) with auto-sizing')
except Exception as e:
    print(f'✗ Error in Example 7: {e}')
    import traceback

    traceback.print_exc()

print()

# ============================================================================
# Example 8: Line mode with faceting (for clearer trend comparison)
# ============================================================================
print('=' * 70)
print('Example 8: Line mode with faceting')
print('=' * 70)

# Take shorter time window for clearer line plots
ds_short = ds.isel(time=slice(0, 48))  # 2 days

try:
    fig8 = plotting.with_plotly(
        ds_short.sel(period=2024),
        facet_by='scenario',
        mode='line',
        colors='tab10',
        title='48-Hour Energy Generation Profiles',
        ylabel='Power (MW)',
        xlabel='Time',
        facet_cols=2,
    )
    fig8.write_html('/tmp/facet_example_8_line_facets.html')
    print('✓ Created: /tmp/facet_example_8_line_facets.html')
    print('  Line plots for comparing detailed trends across scenarios')
except Exception as e:
    print(f'✗ Error in Example 8: {e}')
    import traceback

    traceback.print_exc()

print()

# ============================================================================
# Example 9: Single variable across scenarios (using select parameter)
# ============================================================================
print('=' * 70)
print('Example 9: Single variable faceted by scenario')
print('=' * 70)

try:
    # Select only Solar data
    ds_solar_only = ds[['Solar']]

    fig9 = plotting.with_plotly(
        ds_solar_only.sel(period=2030),
        facet_by='scenario',
        mode='area',
        colors='YlOrRd',
        title='Solar Generation Across Scenarios (2030)',
        ylabel='Solar Power (MW)',
        xlabel='Time',
        facet_cols=4,  # Single row
    )
    fig9.write_html('/tmp/facet_example_9_single_var.html')
    print('✓ Created: /tmp/facet_example_9_single_var.html')
    print('  Single variable (Solar) across 4 scenarios')
except Exception as e:
    print(f'✗ Error in Example 9: {e}')
    import traceback

    traceback.print_exc()

print()

# ============================================================================
# Example 10: Comparison plot - Different color schemes
# ============================================================================
print('=' * 70)
print('Example 10: Testing different color schemes')
print('=' * 70)

color_schemes = ['portland', 'viridis', 'plasma', 'turbo']
ds_sample = ds.isel(time=slice(0, 72)).sel(period=2024)  # 3 days

for i, color_scheme in enumerate(color_schemes):
    try:
        scenario_to_plot = scenarios[i % len(scenarios)]
        fig = plotting.with_plotly(
            ds_sample.sel(scenario=scenario_to_plot),
            mode='area',
            colors=color_scheme,
            title=f'Color Scheme: {color_scheme.upper()} ({scenario_to_plot})',
            ylabel='Power (MW)',
            xlabel='Time',
        )
        fig.write_html(f'/tmp/facet_example_10_{color_scheme}.html')
        print(f'✓ Created: /tmp/facet_example_10_{color_scheme}.html')
    except Exception as e:
        print(f'✗ Error with {color_scheme}: {e}')

print()

# ============================================================================
# Example 11: Mixed positive/negative with 2D faceting
# ============================================================================
print('=' * 70)
print('Example 11: 2D faceting with positive/negative values')
print('=' * 70)

# Create subset with just 2 scenarios and 2 periods for clearer visualization
ds_mixed = ds.sel(scenario=['base', 'high_demand'], period=[2024, 2040])
ds_mixed_short = ds_mixed.isel(time=slice(0, 48))

try:
    fig11 = plotting.with_plotly(
        ds_mixed_short,
        facet_by=['scenario', 'period'],
        mode='area',
        colors='portland',
        title='Energy Balance Grid: Scenarios × Periods',
        ylabel='Power (MW)',
        xlabel='Time (48h)',
        facet_cols=2,
    )
    fig11.write_html('/tmp/facet_example_11_2d_mixed.html')
    print('✓ Created: /tmp/facet_example_11_2d_mixed.html')
    print('  2x2 grid showing charging/discharging across scenarios and periods')
except Exception as e:
    print(f'✗ Error in Example 11: {e}')
    import traceback

    traceback.print_exc()

print()

# ============================================================================
# Example 12: Animation with custom frame duration
# ============================================================================
print('=' * 70)
print('Example 12: Animation settings test')
print('=' * 70)

try:
    fig12 = plotting.with_plotly(
        ds.sel(scenario='renewable_focus'),
        animate_by='period',
        mode='stacked_bar',
        colors='greens',
        title='Renewable Focus Scenario: Temporal Evolution',
        ylabel='Power (MW)',
        xlabel='Time',
    )
    # Adjust animation speed (if the API supports it)
    if hasattr(fig12, 'layout') and hasattr(fig12.layout, 'updatemenus'):
        for menu in fig12.layout.updatemenus:
            if 'buttons' in menu:
                for button in menu.buttons:
                    if 'args' in button and len(button.args) > 1:
                        if isinstance(button.args[1], dict) and 'frame' in button.args[1]:
                            button.args[1]['frame']['duration'] = 1000  # 1 second per frame

    fig12.write_html('/tmp/facet_example_12_animation_settings.html')
    print('✓ Created: /tmp/facet_example_12_animation_settings.html')
    print('  Animation with custom frame duration settings')
except Exception as e:
    print(f'✗ Error in Example 12: {e}')
    import traceback

    traceback.print_exc()

print()

# ============================================================================
# Example 13: Edge case - Single facet value (should work like normal plot)
# ============================================================================
print('=' * 70)
print('Example 13: Edge case - faceting with single value')
print('=' * 70)

try:
    ds_single = ds.sel(scenario='base', period=2024)

    fig13 = plotting.with_plotly(
        ds_single,
        mode='area',
        colors='portland',
        title='Single Plot (No Real Faceting)',
        ylabel='Power (MW)',
        xlabel='Time',
    )
    fig13.write_html('/tmp/facet_example_13_single_facet.html')
    print('✓ Created: /tmp/facet_example_13_single_facet.html')
    print('  Should create normal plot when no facet dimension exists')
except Exception as e:
    print(f'✗ Error in Example 13: {e}')
    import traceback

    traceback.print_exc()

# ============================================================================
# Example 14: Real flixOpt integration - plot_charge_state with faceting
# ============================================================================
print('=' * 70)
print('Example 14: plot_charge_state() with facet_by and animate_by')
print('=' * 70)

try:
    from datetime import datetime

    import flixopt as fx

    # Create a simple flow system with storage for each scenario and period
    print('Building flow system with storage component...')

    # Time steps for a short period
    time_steps = pd.date_range('2024-01-01', periods=48, freq='h', name='time')

    # Create flow system with scenario and period dimensions
    flow_system = fx.FlowSystem(time_steps, scenarios=scenarios, periods=periods, time_unit='h')

    # Create buses
    electricity_bus = fx.Bus('Electricity', 'Electricity')

    # Create effects (costs)
    costs = fx.Effect('costs', '€', 'Costs', is_standard=True, is_objective=True)

    # Create source (power plant) - using xr.DataArray for multi-dimensional inputs
    generation_profile = xr.DataArray(
        np.random.uniform(50, 150, (len(time_steps), len(scenarios), len(periods))),
        dims=['time', 'scenario', 'period'],
        coords={'time': time_steps, 'scenario': scenarios, 'period': periods},
    )

    power_plant = fx.Source(
        'PowerPlant',
        fx.Flow(
            'PowerGeneration',
            bus=electricity_bus,
            size=200,
            relative_maximum=generation_profile / 200,  # Normalized profile
            effects_per_flow_hour={costs: 30},
        ),
    )

    # Create demand - also multi-dimensional
    demand_profile = xr.DataArray(
        np.random.uniform(60, 140, (len(time_steps), len(scenarios), len(periods))),
        dims=['time', 'scenario', 'period'],
        coords={'time': time_steps, 'scenario': scenarios, 'period': periods},
    )

    demand = fx.Sink(
        'Demand',
        fx.Flow('PowerDemand', bus=electricity_bus, size=demand_profile),
    )

    # Create storage with multi-dimensional capacity
    storage_capacity = xr.DataArray(
        [[100, 120, 150], [120, 150, 180], [110, 130, 160], [90, 110, 140]],
        dims=['scenario', 'period'],
        coords={'scenario': scenarios, 'period': periods},
    )

    battery = fx.Storage(
        'Battery',
        charging=fx.Flow(
            'Charging',
            bus=electricity_bus,
            size=50,
            effects_per_flow_hour={costs: 5},  # Small charging cost
        ),
        discharging=fx.Flow(
            'Discharging',
            bus=electricity_bus,
            size=50,
            effects_per_flow_hour={costs: 0},
        ),
        capacity_in_flow_hours=storage_capacity,
        initial_charge_state=0.5,  # Start at 50%
        eta_charge=0.95,
        eta_discharge=0.95,
        relative_loss_per_hour=0.001,  # 0.1% loss per hour
    )

    # Add all elements to the flow system
    flow_system.add_elements(electricity_bus, costs, power_plant, demand, battery)

    print('Running calculation...')
    calculation = fx.FullCalculation(
        'FacetPlotTest',
        flow_system,
        'highs',
    )

    # Solve the system
    calculation.solve(save=False)

    print('✓ Calculation successful!')
    print()

    # Now demonstrate plot_charge_state with faceting
    print('Creating faceted charge state plots...')

    # Example 14a: Facet by scenario
    print('  a) Faceting by scenario...')
    fig14a = calculation.results['Battery'].plot_charge_state(
        facet_by='scenario',
        mode='area',
        colors='blues',
        select={'period': 2024},
        save='/tmp/facet_example_14a_charge_state_scenarios.html',
        show=False,
    )
    print('     ✓ Created: /tmp/facet_example_14a_charge_state_scenarios.html')

    # Example 14b: Animate by period
    print('  b) Animating by period...')
    fig14b = calculation.results['Battery'].plot_charge_state(
        animate_by='period',
        mode='area',
        colors='greens',
        select={'scenario': 'base'},
        save='/tmp/facet_example_14b_charge_state_animation.html',
        show=False,
    )
    print('     ✓ Created: /tmp/facet_example_14b_charge_state_animation.html')

    # Example 14c: Combined faceting and animation
    print('  c) Faceting by scenario AND animating by period...')
    fig14c = calculation.results['Battery'].plot_charge_state(
        facet_by='scenario',
        animate_by='period',
        mode='area',
        colors='portland',
        facet_cols=2,
        save='/tmp/facet_example_14c_charge_state_combined.html',
        show=False,
    )
    print('     ✓ Created: /tmp/facet_example_14c_charge_state_combined.html')
    print('     4 subplots (scenarios) × 3 frames (periods)')

    # Example 14d: 2D faceting (scenario x period)
    print('  d) 2D faceting (scenario × period grid)...')
    fig14d = calculation.results['Battery'].plot_charge_state(
        facet_by=['scenario', 'period'],
        mode='line',
        colors='viridis',
        facet_cols=3,
        save='/tmp/facet_example_14d_charge_state_2d.html',
        show=False,
    )
    print('     ✓ Created: /tmp/facet_example_14d_charge_state_2d.html')
    print('     12 subplots (4 scenarios × 3 periods)')

    print()
    print('✓ All plot_charge_state examples completed successfully!')

except ImportError as e:
    print(f'✗ Skipping Example 14: flixopt not fully available ({e})')
    print('  This example requires a full flixopt installation')
except Exception as e:
    print(f'✗ Error in Example 14: {e}')
    import traceback

    traceback.print_exc()

print()
print('=' * 70)
print('All examples completed!')
print('=' * 70)
print()
print('Summary of examples:')
print('  1. Simple faceting by scenario (4 subplots)')
print('  2. Animation by period (3 frames)')
print('  3. Combined faceting + animation (4 subplots × 3 frames)')
print('  4. 2D faceting (12 subplots in grid)')
print('  5. Area mode with pos/neg values')
print('  6. Stacked bar mode with animation')
print('  7. Large grid (6 scenarios)')
print('  8. Line mode with faceting')
print('  9. Single variable across scenarios')
print(' 10. Different color schemes comparison')
print(' 11. 2D faceting with mixed values')
print(' 12. Animation with custom settings')
print(' 13. Edge case - single facet value')
print(' 14. Real flixOpt integration:')
print('     a) plot_charge_state with faceting by scenario')
print('     b) plot_charge_state with animation by period')
print('     c) plot_charge_state with combined faceting + animation')
print('     d) plot_charge_state with 2D faceting (scenario × period)')
print()
print('Next steps for testing with real flixopt data:')
print('1. Load your CalculationResults with scenario/period dimensions')
print("2. Use results['Component'].plot_node_balance(facet_by='scenario')")
print("3. Try animate_by='period' for time evolution visualization")
print("4. Combine both: facet_by='scenario', animate_by='period'")
