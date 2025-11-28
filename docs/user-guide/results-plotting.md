# Plotting Results

After solving an optimization, FlixOpt provides a powerful plotting API to visualize and analyze your results. The API is designed to be intuitive and chainable, giving you quick access to common plots while still allowing deep customization.

## The Plot Accessor

All plotting is accessed through the `.plot` accessor on your results:

```python
results = optimization.results

# System-level plots
results.plot.balance('ElectricityBus')
results.plot.sankey()

# Element-level plots
results['Boiler'].plot.balance()
results['Battery'].plot.storage()
```

## PlotResult: Data + Figure

Every plot method returns a [`PlotResult`][flixopt.plot_accessors.PlotResult] object containing both:

- **`data`**: An xarray Dataset with the prepared data
- **`figure`**: A Plotly Figure object

This gives you full access to export data, customize the figure, or use the data for your own visualizations:

```python
result = results.plot.balance('Bus')

# Access the xarray data
print(result.data)
result.data.to_dataframe()  # Convert to pandas DataFrame
result.data.to_netcdf('balance_data.nc')  # Export as netCDF

# Access and modify the figure
result.figure.update_layout(title='Custom Title')
result.figure.show()
```

### Method Chaining

All `PlotResult` methods return `self`, enabling fluent chaining:

```python
results.plot.balance('Bus') \
    .update(title='Custom Title', height=600) \
    .update_traces(opacity=0.8) \
    .to_csv('data.csv') \
    .to_html('plot.html') \
    .show()
```

Available methods:

| Method | Description |
|--------|-------------|
| `.show()` | Display the figure |
| `.update(**kwargs)` | Update figure layout (passes to `fig.update_layout()`) |
| `.update_traces(**kwargs)` | Update traces (passes to `fig.update_traces()`) |
| `.to_html(path)` | Save as interactive HTML |
| `.to_image(path)` | Save as static image (png, svg, pdf) |
| `.to_csv(path)` | Export data to CSV (converts xarray to DataFrame) |
| `.to_netcdf(path)` | Export data to netCDF (native xarray format) |

## Available Plot Methods

### Balance Plot

Plot the energy/material balance at a node (Bus or Component), showing inputs and outputs:

```python
results.plot.balance('ElectricityBus')
results.plot.balance('Boiler', mode='area')
results['HeatBus'].plot.balance()
```

**Key parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `node` | str | Label of the Bus or Component |
| `mode` | `'bar'`, `'line'`, `'area'` | Visual style (default: `'bar'`) |
| `unit` | `'flow_rate'`, `'flow_hours'` | Power (kW) or energy (kWh) |
| `include` | str or list | Only include flows containing these substrings |
| `exclude` | str or list | Exclude flows containing these substrings |
| `aggregate` | `'sum'`, `'mean'`, `'max'`, `'min'` | Aggregate over time |
| `select` | dict | xarray-style data selection |

### Storage Plot

Visualize storage components with charge state and flow balance:

```python
results.plot.storage('Battery')
results['ThermalStorage'].plot.storage(mode='line')
```

**Key parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `component` | str | Storage component label |
| `mode` | `'bar'`, `'line'`, `'area'` | Visual style |

### Heatmap

Create heatmaps of time series data, with automatic time reshaping:

```python
results.plot.heatmap('Boiler(Q_th)|flow_rate')
results.plot.heatmap(['CHP|on', 'Boiler|on'], facet_col='variable')
```

**Key parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `variables` | str or list | Variable name(s) to plot |
| `reshape` | tuple | Time reshaping pattern, e.g., `('D', 'h')` for days × hours |
| `colorscale` | str | Plotly colorscale name |

Common reshape patterns:

- `('D', 'h')`: Days × Hours (default)
- `('W', 'D')`: Weeks × Days
- `('MS', 'D')`: Months × Days

### Flows Plot

Plot flow rates filtered by nodes or components:

```python
results.plot.flows(component='Boiler')
results.plot.flows(start='ElectricityBus')
results.plot.flows(unit='flow_hours', aggregate='sum')
```

**Key parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `start` | str or list | Filter by source node(s) |
| `end` | str or list | Filter by destination node(s) |
| `component` | str or list | Filter by parent component(s) |
| `unit` | `'flow_rate'`, `'flow_hours'` | Power or energy |
| `aggregate` | str | Time aggregation |

### Compare Plot

Compare multiple elements side-by-side:

```python
results.plot.compare(['Boiler', 'CHP', 'HeatPump'], variable='flow_rate')
results.plot.compare(['Battery1', 'Battery2'], variable='charge_state')
```

**Key parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `elements` | list | Element labels to compare |
| `variable` | str | Variable suffix to compare |
| `mode` | `'overlay'`, `'facet'` | Same axes or subplots |

### Sankey Diagram

Visualize energy/material flows as a Sankey diagram:

```python
results.plot.sankey()
results.plot.sankey(timestep=100)
results.plot.sankey(aggregate='mean')
```

**Key parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `timestep` | int or str | Specific timestep, or None for aggregation |
| `aggregate` | `'sum'`, `'mean'` | Aggregation method when timestep is None |

### Effects Plot

Plot cost, emissions, or other effect breakdowns:

```python
results.plot.effects('total', by='component')
results.plot.effects('total', mode='pie')
results.plot.effects('temporal', by='time')
```

**Key parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `effect` | str | Effect name (e.g., `'total'`, `'temporal'`, `'periodic'`) |
| `by` | `'component'`, `'time'` | Grouping dimension |
| `mode` | `'bar'`, `'pie'`, `'treemap'` | Chart type |

## Common Parameters

Most plot methods share these parameters:

### Data Selection

Use xarray-style selection to filter data before plotting:

```python
# Single value
results.plot.balance('Bus', select={'scenario': 'base'})

# Multiple values
results.plot.balance('Bus', select={'scenario': ['base', 'high_demand']})

# Time slices
results.plot.balance('Bus', select={'time': slice('2024-01', '2024-06')})

# Combined
results.plot.balance('Bus', select={
    'scenario': 'base',
    'time': slice('2024-01-01', '2024-01-07')
})
```

### Faceting and Animation

Control how multi-dimensional data is displayed:

```python
# Facet by scenario
results.plot.balance('Bus', facet_col='scenario')

# Animate by period
results.plot.balance('Bus', animate_by='period')

# Both
results.plot.balance('Bus', facet_col='scenario', animate_by='period')
```

!!! note
    Facet and animation dimensions are automatically ignored if not present in the data. Defaults are `facet_col='scenario'` and `animate_by='period'` for balance plots.

### Include/Exclude Filtering

Filter flows using simple substring matching:

```python
# Only show flows containing 'Q_th'
results.plot.balance('Bus', include='Q_th')

# Exclude flows containing 'Gas' or 'Grid'
results.plot.balance('Bus', exclude=['Gas', 'Grid'])

# Combine include and exclude
results.plot.balance('Bus', include='Boiler', exclude='auxiliary')
```

### Colors

Override colors using a dictionary:

```python
results.plot.balance('Bus', colors={
    'Boiler(Q_th)|flow_rate': '#ff6b6b',
    'CHP(Q_th)|flow_rate': '#4ecdc4',
})
```

Global colors can be set on the Results object and will be used across all plots.

### Display Control

Control whether plots are shown automatically:

```python
# Don't show (useful in scripts)
result = results.plot.balance('Bus', show=False)

# Show later
result.show()
```

The default behavior is controlled by `CONFIG.Plotting.default_show`.

## Element-Level Plotting

Access plots directly from element results for convenience:

```python
# These are equivalent:
results.plot.balance('Boiler')
results['Boiler'].plot.balance()

# Storage plotting (only for storage components)
results['Battery'].plot.storage()

# Element heatmap
results['Boiler'].plot.heatmap('on')
```

The element-level accessor automatically passes the element label to the corresponding system-level method.

## Complete Examples

### Analyzing a Bus Balance

```python
# Quick overview
results.plot.balance('ElectricityBus')

# Detailed analysis with exports
result = results.plot.balance(
    'ElectricityBus',
    mode='area',
    unit='flow_hours',
    select={'time': slice('2024-06-01', '2024-06-07')},
    show=False
)

# Access xarray data for further analysis
print(result.data)  # xarray Dataset
df = result.data.to_dataframe()  # Convert to pandas

# Export data
result.to_netcdf('electricity_balance.nc')  # Native xarray format
result.to_csv('electricity_balance.csv')  # As CSV

# Customize and display
result.update(
    title='Electricity Balance - First Week of June',
    yaxis_title='Energy [kWh]'
).show()
```

### Comparing Storage Units

```python
# Compare charge states
results.plot.compare(
    ['Battery1', 'Battery2', 'ThermalStorage'],
    variable='charge_state',
    mode='overlay'
).update(title='Storage Comparison')
```

### Creating a Report

```python
# Generate multiple plots for a report
plots = {
    'balance': results.plot.balance('HeatBus', show=False),
    'storage': results.plot.storage('ThermalStorage', show=False),
    'sankey': results.plot.sankey(show=False),
    'costs': results.plot.effects('total', mode='pie', show=False),
}

# Export all
for name, plot in plots.items():
    plot.to_html(f'report_{name}.html')
    plot.to_netcdf(f'report_{name}.nc')  # xarray native format
```

### Working with xarray Data

The `.data` attribute returns xarray objects, giving you full access to xarray's powerful data manipulation capabilities:

```python
result = results.plot.balance('Bus', show=False)

# Access the xarray Dataset
ds = result.data

# Use xarray operations
ds.mean(dim='time')  # Average over time
ds.sel(time='2024-06')  # Select specific time
ds.to_dataframe()  # Convert to pandas

# Export options
ds.to_netcdf('data.nc')  # Native xarray format
ds.to_zarr('data.zarr')  # Zarr format for large datasets
```
