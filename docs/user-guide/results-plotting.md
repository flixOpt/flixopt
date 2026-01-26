# Plotting Results

After solving an optimization, flixOpt provides a plotting API to visualize and analyze your results. The API is designed to be intuitive and chainable, giving you quick access to common plots while still allowing customization.

!!! tip "Related Guides"
    - [Color Management](colors.md) - Configure colors for components and carriers
    - [Plotly Customization](plotly-customization.md) - Advanced figure customization
    - [Plotting Custom Data](recipes/plotting-custom-data.md) - Plot arbitrary xarray data with the `.plotly` accessor

## Quick Start

All plotting is accessed through the `statistics.plot` accessor:

```python
flow_system.optimize(fx.solvers.HighsSolver())

flow_system.statistics.plot.balance('ElectricityBus')
flow_system.statistics.plot.sankey.flows()
flow_system.statistics.plot.heatmap('Boiler(Q_th)|flow_rate')
```

## PlotResult: Data + Figure

Every plot method returns a [`PlotResult`][flixopt.plot_result.PlotResult] containing:

- **`data`**: An xarray Dataset with the prepared data
- **`figure`**: A Plotly Figure object

```python
result = flow_system.statistics.plot.balance('Bus')

# Access the data
result.data.to_dataframe()
result.data.to_netcdf('balance_data.nc')

# Access the figure
result.figure.update_layout(title='Custom Title')
result.figure.show()
```

### Method Chaining

All `PlotResult` methods return `self`, enabling fluent chaining:

```python
flow_system.statistics.plot.balance('Bus') \
    .update(title='Custom Title', height=600) \
    .to_html('plot.html') \
    .show()
```

| Method | Description |
|--------|-------------|
| `.show()` | Display the figure |
| `.update(**kwargs)` | Update figure layout |
| `.update_traces(**kwargs)` | Update trace properties |
| `.to_html(path)` | Save as interactive HTML |
| `.to_image(path)` | Save as static image (png, svg, pdf) |
| `.to_csv(path)` | Export data to CSV |
| `.to_netcdf(path)` | Export data to netCDF |

## Available Plot Methods

### Balance Plot

Plot the energy/material balance at a Bus or Component:

```python
flow_system.statistics.plot.balance('ElectricityBus')
flow_system.statistics.plot.balance('Boiler', mode='area')
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `node` | str | Label of the Bus or Component |
| `mode` | `'bar'`, `'line'`, `'area'` | Visual style (default: `'bar'`) |
| `unit` | `'flow_rate'`, `'flow_hours'` | Power (kW) or energy (kWh) |
| `include` / `exclude` | str or list | Filter flows by substring |
| `aggregate` | `'sum'`, `'mean'`, `'max'`, `'min'` | Aggregate over time |
| `select` | dict | xarray-style data selection |

### Storage Plot

Visualize storage components with charge state and flow balance:

```python
flow_system.statistics.plot.storage('Battery')
flow_system.statistics.plot.storage('ThermalStorage', mode='line')
```

### Heatmap

Create heatmaps of time series data with automatic time reshaping:

```python
flow_system.statistics.plot.heatmap('Boiler(Q_th)|flow_rate')
flow_system.statistics.plot.heatmap(['CHP|on', 'Boiler|on'], facet_col='variable')
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `variables` | str or list | Variable name(s) to plot |
| `reshape` | tuple | Time pattern: `('D', 'h')` days×hours, `('W', 'D')` weeks×days |
| `colorscale` | str | Plotly colorscale name |

### Flows Plot

Plot flow rates filtered by nodes or components:

```python
flow_system.statistics.plot.flows(component='Boiler')
flow_system.statistics.plot.flows(start='ElectricityBus')
flow_system.statistics.plot.flows(unit='flow_hours', aggregate='sum')
```

### Compare Plot

Compare multiple elements side-by-side:

```python
flow_system.statistics.plot.compare(['Boiler', 'CHP', 'HeatPump'], variable='flow_rate')
flow_system.statistics.plot.compare(['Battery1', 'Battery2'], variable='charge_state')
```

### Sankey Diagram

Visualize energy/material flows as a Sankey diagram:

```python
flow_system.statistics.plot.sankey.flows()           # Energy amounts
flow_system.statistics.plot.sankey.sizes()           # Investment sizes
flow_system.statistics.plot.sankey.peak_flow()       # Maximum rates
flow_system.statistics.plot.sankey.effects()         # Cost/emission breakdown
```

Filter with `select`:

```python
flow_system.statistics.plot.sankey.flows(select={'bus': 'HeatBus'})
flow_system.statistics.plot.sankey.effects(select={'effect': 'costs'})
```

### Effects Plot

Plot cost, emissions, or other effect breakdowns:

```python
flow_system.statistics.plot.effects()                          # Total by component
flow_system.statistics.plot.effects(effect='costs')            # Just costs
flow_system.statistics.plot.effects(by='contributor')          # By individual flows
flow_system.statistics.plot.effects(aspect='temporal')         # Over time
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `aspect` | `'total'`, `'temporal'`, `'periodic'` | Which aspect to plot |
| `effect` | str or None | Specific effect (e.g., `'costs'`) or all |
| `by` | `'component'`, `'contributor'`, `'time'` | Grouping dimension |

### Variable Plot

Plot the same variable type across multiple elements:

```python
flow_system.statistics.plot.variable('on')              # All binary states
flow_system.statistics.plot.variable('flow_rate', include='Boiler')
flow_system.statistics.plot.variable('charge_state')    # All storage states
```

### Duration Curve

Plot load duration curves (sorted time series):

```python
flow_system.statistics.plot.duration_curve('Boiler(Q_th)')
flow_system.statistics.plot.duration_curve(['CHP(Q_th)', 'HeatPump(Q_th)'])
flow_system.statistics.plot.duration_curve('Demand(in)', normalize=True)
```

## Common Parameters

### Data Selection

Use xarray-style selection to filter data:

```python
# Single value
flow_system.statistics.plot.balance('Bus', select={'scenario': 'base'})

# Time slice
flow_system.statistics.plot.balance('Bus', select={'time': slice('2024-01', '2024-06')})

# Combined
flow_system.statistics.plot.balance('Bus', select={
    'scenario': 'base',
    'time': slice('2024-01-01', '2024-01-07')
})
```

### Faceting and Animation

Control how multi-dimensional data is displayed:

```python
flow_system.statistics.plot.balance('Bus', facet_col='scenario')
flow_system.statistics.plot.balance('Bus', animate_by='period')
```

### Include/Exclude Filtering

Filter flows using substring matching:

```python
flow_system.statistics.plot.balance('Bus', include='Q_th')
flow_system.statistics.plot.balance('Bus', exclude=['Gas', 'Grid'])
```

### Colors

Override colors using a dictionary:

```python
flow_system.statistics.plot.balance('Bus', colors={
    'Boiler(Q_th)': '#ff6b6b',
    'CHP(Q_th)': '#4ecdc4',
})
```

See [Color Management](colors.md) for configuring colors system-wide.

## Examples

### Analyzing a Bus Balance

```python
# Quick overview
flow_system.statistics.plot.balance('ElectricityBus')

# Detailed analysis with exports
result = flow_system.statistics.plot.balance(
    'ElectricityBus',
    mode='area',
    unit='flow_hours',
    select={'time': slice('2024-06-01', '2024-06-07')},
    show=False
)

# Export data
result.to_netcdf('electricity_balance.nc')
result.to_csv('electricity_balance.csv')

# Customize and display
result.update(
    title='Electricity Balance - First Week of June',
    yaxis_title='Energy [kWh]'
).show()
```

### Creating a Report

```python
plots = {
    'balance': flow_system.statistics.plot.balance('HeatBus', show=False),
    'storage': flow_system.statistics.plot.storage('ThermalStorage', show=False),
    'sankey': flow_system.statistics.plot.sankey.flows(show=False),
    'costs': flow_system.statistics.plot.effects(effect='costs', show=False),
}

for name, plot in plots.items():
    plot.to_html(f'report_{name}.html')
```

### Working with xarray Data

```python
result = flow_system.statistics.plot.balance('Bus', show=False)
ds = result.data

# Use xarray operations
ds.mean(dim='time')
ds.sel(time='2024-06')
ds.to_dataframe()
```
