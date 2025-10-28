# Plotting and Statistics API

The plotting and statistics API provides a clean, accessor-based interface for visualizing and analyzing optimization results. The API follows a gradual expansion philosophy: start with essential functionality and add methods as needed.

## Overview

flixopt provides two main accessors for working with results:

- **`.plot`** - Access plotting methods for creating visualizations
- **`.statistics`** - Access statistical calculations with built-in plotting

Both accessors are available on `CalculationResults` and individual node results (`ComponentResults`/`BusResults`).

## Architecture

```python
CalculationResults
├── .plot         # System-level plotting
│   └── heatmap()
└── .statistics   # System-level statistics
    └── flow_summary()

ComponentResults / BusResults
├── .plot         # Node-level plotting
│   ├── node_balance()
│   ├── node_balance_pie()
│   └── charge_state()  # ComponentResults only
└── .statistics   # Node-level statistics
    └── flow_hours()
```

## Basic Usage

### Loading Results

```python
from flixopt.results import CalculationResults

# Load optimization results
results = CalculationResults.from_file('results', 'optimization')
```

### System-Level Plotting

```python
# Heatmap visualization
plotter = results.plot.heatmap('Boiler(Gas)|flow_rate')
fig = plotter.heatmap(reshape_time=('D', 'h'))
fig.show()

# Save to file
fig = plotter.heatmap(save='heatmap.html', show=False)
```

### Node-Level Plotting

```python
# Node balance plot
plotter = results['Boiler'].plot.node_balance()
fig = plotter.bar(mode='stacked')
fig.show()

# Pie chart for flow distribution
plotter = results['ElectricityBus'].plot.node_balance_pie()
fig = plotter.pie(lower_percentage_group=5.0)

# Storage charge state
plotter = results['Battery'].plot.charge_state()
fig = plotter.area()
```

### Statistics with Plotting

```python
# Get statistics plotter
plotter = results.statistics.flow_summary()

# Access raw data
data = plotter.data  # Returns xarray.Dataset
print(data)

# Create visualization
fig = plotter.plot.bar()
fig.show()

# Node-level statistics
plotter = results['Boiler'].statistics.flow_hours()
data = plotter.data
fig = plotter.plot.bar()
```

## Plotter Classes

All plotting methods return plotter objects that provide multiple visualization options:

### NodeBalancePlotter

Visualizes flow balances at nodes (components or buses).

**Methods:**
- `bar(mode='stacked'|'grouped')` - Bar chart
- `line()` - Line chart
- `area()` - Area chart
- `plot(mode=...)` - Generic plotting method

**Parameters:**
- `colors` - Color mapping dictionary
- `title` - Plot title
- `ylabel` / `xlabel` - Axis labels
- `facet_by` - Create subplots by dimension
- `animate_by` - Create animation by dimension
- `save` - Save to file
- `show` - Display plot

**Example:**
```python
plotter = results['Boiler'].plot.node_balance(unit_type='flow_rate')

# Stacked bar chart
fig = plotter.bar(
    mode='stacked',
    colors={'Q_th': '#FF6B6B', 'Q_fu': '#4ECDC4'},
    title='Boiler Flow Balance',
    ylabel='Flow Rate (MW)',
    save='boiler_balance.html'
)

# With faceting by scenario
fig = plotter.area(
    facet_by='scenario',
    facet_cols=2
)

# With animation by period
fig = plotter.line(
    animate_by='period'
)
```

### PieChartPlotter

Creates pie/donut charts for flow distribution.

**Methods:**
- `pie()` - Standard pie chart
- `donut(hole=0.4)` - Donut chart

**Parameters:**
- `lower_percentage_group` - Group small slices (default: 5%)
- `colors` - Color mapping
- `text_info` - Text to show on slices
- `hole` - Size of center hole (0=pie, >0=donut)

**Example:**
```python
plotter = results['ElectricityBus'].plot.node_balance_pie()

# Standard pie chart
fig = plotter.pie(
    lower_percentage_group=3.0,
    text_info='percent+label'
)

# Donut chart
fig = plotter.donut(
    hole=0.4,
    colors={'solar': '#FDB462', 'wind': '#80B1D3'}
)
```

### ChargeStatePlotter

Visualizes storage operation with charge state overlay.

**Methods:**
- `area()` - Area chart with charge state line
- `bar()` - Bar chart with charge state line
- `line()` - Line chart with charge state line
- `overlay(mode=..., overlay_color=...)` - Custom overlay

**Example:**
```python
plotter = results['Battery'].plot.charge_state()

# Area chart with charge state
fig = plotter.area(
    overlay_color='black',
    title='Battery Operation'
)

# Bar chart version
fig = plotter.bar()
```

### HeatmapPlotter

Creates heatmap visualizations for time series data.

**Methods:**
- `heatmap()` - Standard heatmap
- `imshow()` - Alias for heatmap

**Parameters:**
- `reshape_time` - Reshape time axis (e.g., ('D', 'h') for days vs hours)
- `colors` - Color scheme
- `facet_by` - Create subplots
- `animate_by` - Create animation

**Example:**
```python
plotter = results.plot.heatmap('Boiler(Gas)|flow_rate')

# Daily pattern heatmap
fig = plotter.heatmap(
    reshape_time=('D', 'h'),
    title='Boiler Operation Pattern'
)

# Auto-detect time reshaping
fig = plotter.heatmap(reshape_time='auto')
```

### StatisticPlotter

Returned by statistics methods, provides plotting for calculated statistics.

**Methods:**
- `bar()` - Bar chart
- `line()` - Line chart
- `area()` - Area chart
- `scatter()` - Scatter plot

**Properties:**
- `data` - Access raw xarray.Dataset

**Example:**
```python
plotter = results.statistics.flow_summary(
    components=['Boiler', 'HeatPump'],
    aggregate_time=True
)

# Get raw data
data = plotter.data
print(data)

# Visualize
fig = plotter.plot.bar(
    title='Total Flow Summary',
    ylabel='Total Flow (MWh)'
)
```

## Advanced Features

### Data Selection

Apply selections before plotting:

```python
# Select specific scenarios/periods
plotter = results['Boiler'].plot.node_balance(
    select={'scenario': 'base', 'period': 'winter'}
)
fig = plotter.bar()

# Select time range
plotter = results.statistics.flow_summary(
    components=['Boiler'],
    aggregate_time=False  # Keep time dimension
)
# Then select time range when plotting
```

### Faceting and Animation

Create subplots or animations:

```python
# Faceting (subplots)
fig = plotter.bar(
    facet_by='scenario',
    facet_cols=2,
    shared_yaxes=True
)

# Animation
fig = plotter.area(
    animate_by='period'
)

# Both
fig = plotter.line(
    facet_by='scenario',
    animate_by='time'
)
```

### Color Customization

```python
# Using color mapping
colors = {
    'solar': '#FDB462',
    'wind': '#80B1D3',
    'gas': '#FB8072'
}

fig = plotter.bar(colors=colors)

# Using color scheme
fig = plotter.heatmap(colors='viridis')
```

### Saving and Exporting

```python
# Save to file
fig = plotter.bar(
    save='output.html',
    show=False  # Don't display, just save
)

# Save with specific path
import pathlib
fig = plotter.bar(
    save=pathlib.Path('results/plots/boiler_balance.html')
)

# PNG export (for matplotlib backend)
fig = plotter.bar(
    engine='matplotlib',
    save='plot.png',
    dpi=300
)
```

## Statistics API

### CalculationResults Statistics

**Available Methods:**
- `flow_summary()` - Flow rate summary across components

**Example:**
```python
# Total flows per component
plotter = results.statistics.flow_summary(
    aggregate_time=True,
    aggregate_scenarios=True
)
fig = plotter.plot.bar()

# Time series for specific components
plotter = results.statistics.flow_summary(
    components=['Boiler', 'CHP'],
    aggregate_time=False
)
fig = plotter.plot.line()
```

### Node Statistics

**Available Methods:**
- `flow_hours()` - Total flow hours for node

**Example:**
```python
# Flow hours for a component
plotter = results['Boiler'].statistics.flow_hours()

# Get raw data
data = plotter.data
total_fuel = float(data['Q_fu'].values)

# Visualize
fig = plotter.plot.bar(
    title=f'Boiler Flow Hours (Total Fuel: {total_fuel:.1f} MWh)'
)
```

## Data Access

All plotters provide access to raw data:

```python
# Get xarray Dataset
plotter = results.statistics.flow_summary()
data = plotter.data

# Work with data directly
import xarray as xr
print(data)
print(data.dims)
print(data.coords)

# Extract specific values
boiler_flow = float(data['Boiler(Gas)|Q_fu'].sum())

# Save data
data.to_netcdf('flow_summary.nc')
```

## Backward Compatibility

Original plotting methods still work:

```python
# Original API (still supported)
results['Boiler'].plot_node_balance(save=True)
results['Bus'].plot_node_balance_pie()
results['Storage'].plot_charge_state()
results.plot_heatmap('Variable')

# New API (recommended)
results['Boiler'].plot.node_balance().bar(save=True)
results['Bus'].plot.node_balance_pie().pie()
results['Storage'].plot.charge_state().area()
results.plot.heatmap('Variable').heatmap()
```

## Gradual Expansion

The API is designed for gradual expansion. New statistics methods can be added as needed:

```python
# Future expansion examples (not yet implemented)
results.statistics.energy_balance()
results.statistics.component_effects()
results['Boiler'].statistics.capacity_utilization()
results['Battery'].statistics.storage_cycles()
```

When you need additional functionality, you can:
1. Request it in the GitHub repository
2. Implement it following the documented pattern
3. Use the DataTransformer to build custom visualizations

## See Also

- [DataTransformer Guide](./data-transformer.md) - Data transformation utilities
- [Advanced Plotting](./advanced-plotting.md) - Custom plotting patterns
- [Getting Started](../getting-started.md) - Basic flixopt tutorial
