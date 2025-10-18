"""
Test script demonstrating how to overlay a line plot on top of area/bar plots.

This pattern is used in plot_charge_state() where:
- Flows (charging/discharging) are plotted as area/stacked_bar
- Charge state is overlaid as a line on the same plot

The key technique: Create two separate figures with the same faceting/animation,
then add the line traces to the area/bar figure.
"""

import copy

import numpy as np
import pandas as pd
import xarray as xr

from flixopt import plotting

# List to store all generated figures
all_figures = []

print('=' * 70)
print('Creating synthetic data for overlay demonstration')
print('=' * 70)

# Time dimension
time = pd.date_range('2024-01-01', periods=24 * 7, freq='h', name='time')

# Scenario and period dimensions
scenarios = ['base', 'high_demand', 'low_cost']
periods = [2024, 2030, 2040]

# Seed for reproducibility
np.random.seed(42)

# Create flow variables (generation, consumption, storage flows)
variables = {
    'Generation': np.random.uniform(50, 150, (len(time), len(scenarios), len(periods))),
    'Consumption': -np.random.uniform(40, 120, (len(time), len(scenarios), len(periods))),
    'Storage_in': -np.random.uniform(0, 30, (len(time), len(scenarios), len(periods))),
    'Storage_out': np.random.uniform(0, 30, (len(time), len(scenarios), len(periods))),
}

# Create dataset with flows
flow_ds = xr.Dataset(
    {name: (['time', 'scenario', 'period'], data) for name, data in variables.items()},
    coords={'time': time, 'scenario': scenarios, 'period': periods},
)

# Create a separate charge state variable (cumulative state)
# This should be plotted as a line on a secondary y-axis or overlaid
charge_state_data = np.zeros((len(time), len(scenarios), len(periods)))

for s_idx in range(len(scenarios)):
    for p_idx in range(len(periods)):
        # Oscillating charge state - vary by scenario and period
        base = 50 + s_idx * 15 + p_idx * 10  # Different base for each scenario/period
        oscillation = (20 - s_idx * 5) * np.sin(np.arange(len(time)) * 2 * np.pi / 24)
        trend = (10 + p_idx * 5) * np.sin(np.arange(len(time)) * 2 * np.pi / (24 * 7))  # Weekly trend
        charge_state_data[:, s_idx, p_idx] = np.clip(base + oscillation + trend, 10, 90)

charge_state_da = xr.DataArray(
    charge_state_data,
    dims=['time', 'scenario', 'period'],
    coords={'time': time, 'scenario': scenarios, 'period': periods},
    name='ChargeState',
)

print(f'Flow dataset: {dict(flow_ds.sizes)}')
print(f'Variables: {list(flow_ds.data_vars.keys())}')
print(f'Charge state: {dict(charge_state_da.sizes)}')
print()

# ============================================================================
# Example 1: Simple overlay - single scenario/period
# ============================================================================
print('=' * 70)
print('Example 1: Simple overlay (no faceting)')
print('=' * 70)

# Select single scenario and period
flow_single = flow_ds.sel(scenario='base', period=2024)
charge_single = charge_state_da.sel(scenario='base', period=2024)

# Step 1: Plot flows as area chart
fig1 = plotting.with_plotly(
    flow_single,
    mode='area',
    colors='portland',
    title='Energy Flows with Charge State Overlay',
    ylabel='Power (MW) / Charge State (%)',
    xlabel='Time',
)

# Step 2: Convert charge_state DataArray to Dataset and plot as line
charge_state_ds = charge_single.to_dataset(name='ChargeState')
charge_fig = plotting.with_plotly(
    charge_state_ds,
    mode='line',
    colors='black',  # Different color for the line
    title='',
    ylabel='',
    xlabel='',
)

# Step 3: Add the line trace to the area figure
for trace in charge_fig.data:
    trace_copy = copy.deepcopy(trace)
    trace_copy.line.width = 3  # Make line more prominent
    trace_copy.line.shape = 'linear'  # Straight line (not stepped like flows)
    trace_copy.line.dash = 'dash'  # Optional: make it dashed
    trace_copy.showlegend = False  # Avoid duplicate legend entries
    fig1.add_trace(trace_copy)

fig1.write_html('/tmp/overlay_example_1_simple.html')
all_figures.append(('Example 1: Simple overlay', fig1))
print('✓ Created: /tmp/overlay_example_1_simple.html')
print('  Area plot with overlaid line (charge state)')
print()

# ============================================================================
# Example 2: Overlay with faceting by scenario
# ============================================================================
print('=' * 70)
print('Example 2: Overlay with faceting by scenario')
print('=' * 70)

# Select single period, keep all scenarios
flow_scenarios = flow_ds.sel(period=2024)
charge_scenarios = charge_state_da.sel(period=2024)

facet_by = 'scenario'
facet_cols = 3

# Step 1: Plot flows as stacked bars
fig2 = plotting.with_plotly(
    flow_scenarios,
    facet_by=facet_by,
    mode='stacked_bar',
    colors='viridis',
    title='Energy Flows with Charge State - Faceted by Scenario',
    ylabel='Power (MW) / Charge State (%)',
    xlabel='Time',
    facet_cols=facet_cols,
)

# Step 2: Plot charge_state as lines with same faceting
charge_state_ds = charge_scenarios.to_dataset(name='ChargeState')
charge_fig = plotting.with_plotly(
    charge_state_ds,
    facet_by=facet_by,
    mode='line',
    colors='Reds',
    title='',
    facet_cols=facet_cols,
)

# Step 3: Add line traces to the main figure
# This preserves subplot assignments
for trace in charge_fig.data:
    trace_copy = copy.deepcopy(trace)
    trace_copy.line.width = 2.5
    trace_copy.line.shape = 'linear'  # Straight line for charge state
    trace_copy.showlegend = False  # Avoid duplicate legend entries
    fig2.add_trace(trace_copy)

fig2.write_html('/tmp/overlay_example_2_faceted.html')
all_figures.append(('Example 2: Overlay with faceting', fig2))
print('✓ Created: /tmp/overlay_example_2_faceted.html')
print('  3 subplots (scenarios) with charge state lines')
print()

# ============================================================================
# Example 3: Overlay with animation
# ============================================================================
print('=' * 70)
print('Example 3: Overlay with animation by period')
print('=' * 70)

# Select single scenario, keep all periods
flow_periods = flow_ds.sel(scenario='base')
charge_periods = charge_state_da.sel(scenario='base')

animate_by = 'period'

# Step 1: Plot flows as area with animation
fig3 = plotting.with_plotly(
    flow_periods,
    animate_by=animate_by,
    mode='area',
    colors='portland',
    title='Energy Flows with Animation - Base Scenario',
    ylabel='Power (MW) / Charge State (%)',
    xlabel='Time',
)

# Step 2: Plot charge_state as line with same animation
charge_state_ds = charge_periods.to_dataset(name='ChargeState')
charge_fig = plotting.with_plotly(
    charge_state_ds,
    animate_by=animate_by,
    mode='line',
    colors='black',
    title='',
)

# Step 3: Add charge_state traces to main figure
for trace in charge_fig.data:
    trace_copy = copy.deepcopy(trace)
    trace_copy.line.width = 3
    trace_copy.line.shape = 'linear'  # Straight line for charge state
    trace_copy.line.dash = 'dot'
    trace_copy.showlegend = False  # Avoid duplicate legend entries
    fig3.add_trace(trace_copy)

# Step 4: Add charge_state to animation frames
if hasattr(charge_fig, 'frames') and charge_fig.frames:
    if not hasattr(fig3, 'frames') or not fig3.frames:
        fig3.frames = []
    # Add charge_state traces to each frame
    for i, frame in enumerate(charge_fig.frames):
        if i < len(fig3.frames):
            for trace in frame.data:
                trace_copy = copy.deepcopy(trace)
                trace_copy.line.width = 3
                trace_copy.line.shape = 'linear'  # Straight line for charge state
                trace_copy.line.dash = 'dot'
                trace_copy.showlegend = False  # Avoid duplicate legend entries
                fig3.frames[i].data = fig3.frames[i].data + (trace_copy,)

fig3.write_html('/tmp/overlay_example_3_animated.html')
all_figures.append(('Example 3: Overlay with animation', fig3))
print('✓ Created: /tmp/overlay_example_3_animated.html')
print('  Animation through 3 periods with charge state line')
print()

# ============================================================================
# Example 4: Overlay with faceting AND animation
# ============================================================================
print('=' * 70)
print('Example 4: Overlay with faceting AND animation')
print('=' * 70)

# Use full dataset
flow_full = flow_ds
charge_full = charge_state_da

facet_by = 'scenario'
animate_by = 'period'
facet_cols = 3

# Step 1: Plot flows with faceting and animation
fig4 = plotting.with_plotly(
    flow_full,
    facet_by=facet_by,
    animate_by=animate_by,
    mode='area',
    colors='viridis',
    title='Complete: Faceting + Animation + Overlay',
    ylabel='Power (MW) / Charge State (%)',
    xlabel='Time',
    facet_cols=facet_cols,
)

# Step 2: Plot charge_state with same faceting and animation
charge_state_ds = charge_full.to_dataset(name='ChargeState')
charge_fig = plotting.with_plotly(
    charge_state_ds,
    facet_by=facet_by,
    animate_by=animate_by,
    mode='line',
    colors='Oranges',
    title='',
    facet_cols=facet_cols,
)

# Step 3: Add line traces to base figure
for trace in charge_fig.data:
    trace_copy = copy.deepcopy(trace)
    trace_copy.line.width = 2.5
    trace_copy.line.shape = 'linear'  # Straight line for charge state
    trace_copy.showlegend = False  # Avoid duplicate legend entries
    fig4.add_trace(trace_copy)

# Step 4: Add to animation frames
if hasattr(charge_fig, 'frames') and charge_fig.frames:
    if not hasattr(fig4, 'frames') or not fig4.frames:
        fig4.frames = []
    for i, frame in enumerate(charge_fig.frames):
        if i < len(fig4.frames):
            for trace in frame.data:
                trace_copy = copy.deepcopy(trace)
                trace_copy.line.width = 2.5
                trace_copy.line.shape = 'linear'  # Straight line for charge state
                trace_copy.showlegend = False  # Avoid duplicate legend entries
                fig4.frames[i].data = fig4.frames[i].data + (trace_copy,)

fig4.write_html('/tmp/overlay_example_4_combined.html')
all_figures.append(('Example 4: Complete overlay', fig4))
print('✓ Created: /tmp/overlay_example_4_combined.html')
print('  3 subplots (scenarios) × 3 frames (periods) with charge state')
print()

# ============================================================================
# Example 5: 2D faceting with overlay
# ============================================================================
print('=' * 70)
print('Example 5: 2D faceting (scenario × period) with overlay')
print('=' * 70)

# Use shorter time window for clearer visualization
flow_short = flow_ds.isel(time=slice(0, 48))
charge_short = charge_state_da.isel(time=slice(0, 48))

facet_by = ['scenario', 'period']
facet_cols = 3

# Step 1: Plot flows as line (for clearer 2D grid)
fig5 = plotting.with_plotly(
    flow_short,
    facet_by=facet_by,
    mode='line',
    colors='tab10',
    title='2D Faceting with Charge State Overlay (48h)',
    ylabel='Power (MW) / Charge State (%)',
    xlabel='Time',
    facet_cols=facet_cols,
)

# Step 2: Plot charge_state with same 2D faceting
charge_state_ds = charge_short.to_dataset(name='ChargeState')
charge_fig = plotting.with_plotly(
    charge_state_ds,
    facet_by=facet_by,
    mode='line',
    colors='black',
    title='',
    facet_cols=facet_cols,
)

# Step 3: Add charge state as thick dashed line
for trace in charge_fig.data:
    trace_copy = copy.deepcopy(trace)
    trace_copy.line.width = 3
    trace_copy.line.shape = 'linear'  # Straight line for charge state
    trace_copy.line.dash = 'dashdot'
    trace_copy.showlegend = False  # Avoid duplicate legend entries
    fig5.add_trace(trace_copy)

fig5.write_html('/tmp/overlay_example_5_2d_faceting.html')
all_figures.append(('Example 5: 2D faceting with overlay', fig5))
print('✓ Created: /tmp/overlay_example_5_2d_faceting.html')
print('  9 subplots (3 scenarios × 3 periods) with charge state')
print()

# ============================================================================
# Summary
# ============================================================================
print('=' * 70)
print('All examples completed!')
print('=' * 70)
print()
print('Summary of overlay technique:')
print('  1. Plot main data (flows) with desired mode (area/stacked_bar)')
print('  2. Convert overlay data to Dataset: overlay_ds = da.to_dataset(name="Name")')
print('  3. Plot overlay with mode="line" using SAME facet_by/animate_by')
print('  4. Add traces with customization:')
print('     for trace in overlay_fig.data:')
print('         trace.line.width = 2  # Make prominent')
print('         trace.line.shape = "linear"  # Smooth line (not stepped)')
print('         main_fig.add_trace(trace)')
print('  5. Add to frames: for i, frame in enumerate(overlay_fig.frames): ...')
print()
print('Key insight: Both figures must use identical faceting/animation parameters')
print('             to ensure traces are assigned to correct subplots/frames')
print()
print(f'Generated {len(all_figures)} figures total')
print()
print('To show all figures:')
print('>>> for name, fig in all_figures:')
print('>>>     print(name)')
print('>>>     fig.show()')
print()

# Optional: Uncomment to show all figures in browser at the end
# for name, fig in all_figures:
#     print(f'Showing: {name}')
#     fig.show()
