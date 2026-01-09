# Plotting Results

After solving an optimization, flixOpt provides a powerful plotting API to visualize and analyze your results. The API is designed to be intuitive and chainable, giving you quick access to common plots while still allowing deep customization.

!!! tip "Plotting Custom Data"
    For plotting arbitrary xarray data (not just flixopt results), see the [Custom Data Plotting](recipes/plotting-custom-data.md) guide which covers the `.plotly` accessor.

## The Plot Accessor

All plotting is accessed through the `statistics.plot` accessor on your FlowSystem:

```python
# Run optimization
flow_system.optimize(fx.solvers.HighsSolver())

# Access plotting via statistics
flow_system.statistics.plot.balance('ElectricityBus')
flow_system.statistics.plot.sankey.flows()
flow_system.statistics.plot.heatmap('Boiler(Q_th)|flow_rate')
```

## PlotResult: Data + Figure

Every plot method returns a [`PlotResult`][flixopt.plot_result.PlotResult] object containing both:

- **`data`**: An xarray Dataset with the prepared data
- **`figure`**: A Plotly Figure object

This gives you full access to export data, customize the figure, or use the data for your own visualizations:

```python
result = flow_system.statistics.plot.balance('Bus')

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
flow_system.statistics.plot.balance('Bus') \
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
flow_system.statistics.plot.balance('ElectricityBus')
flow_system.statistics.plot.balance('Boiler', mode='area')
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
flow_system.statistics.plot.storage('Battery')
flow_system.statistics.plot.storage('ThermalStorage', mode='line')
```

**Key parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `component` | str | Storage component label |
| `mode` | `'bar'`, `'line'`, `'area'` | Visual style |

### Heatmap

Create heatmaps of time series data, with automatic time reshaping:

```python
flow_system.statistics.plot.heatmap('Boiler(Q_th)|flow_rate')
flow_system.statistics.plot.heatmap(['CHP|on', 'Boiler|on'], facet_col='variable')
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
flow_system.statistics.plot.flows(component='Boiler')
flow_system.statistics.plot.flows(start='ElectricityBus')
flow_system.statistics.plot.flows(unit='flow_hours', aggregate='sum')
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
flow_system.statistics.plot.compare(['Boiler', 'CHP', 'HeatPump'], variable='flow_rate')
flow_system.statistics.plot.compare(['Battery1', 'Battery2'], variable='charge_state')
```

**Key parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `elements` | list | Element labels to compare |
| `variable` | str | Variable suffix to compare |
| `mode` | `'overlay'`, `'facet'` | Same axes or subplots |

### Sankey Diagram

Visualize energy/material flows as a Sankey diagram. Access via the `sankey` accessor:

```python
# Energy flow amounts (default)
flow_system.statistics.plot.sankey.flows()
flow_system.statistics.plot.sankey.flows(select={'time': '2023-01-01 12:00'})  # specific time
flow_system.statistics.plot.sankey.flows(aggregate='mean')  # mean instead of sum

# Investment sizes/capacities
flow_system.statistics.plot.sankey.sizes()

# Peak flow rates
flow_system.statistics.plot.sankey.peak_flow()

# Effect contributions (costs, CO2, etc.)
flow_system.statistics.plot.sankey.effects()
flow_system.statistics.plot.sankey.effects(select={'effect': 'costs'})
```

**Available methods:**

| Method | Description |
|--------|-------------|
| `sankey.flows()` | Energy/material flow amounts |
| `sankey.sizes()` | Investment sizes/capacities |
| `sankey.peak_flow()` | Maximum flow rates |
| `sankey.effects()` | Component contributions to effects |

**Select options for filtering:**

```python
# Filter by bus or component
flow_system.statistics.plot.sankey.flows(select={'bus': 'HeatBus'})
flow_system.statistics.plot.sankey.flows(select={'component': ['Boiler', 'CHP']})

# Filter effects by name
flow_system.statistics.plot.sankey.effects(select={'effect': 'costs'})
flow_system.statistics.plot.sankey.effects(select={'effect': ['costs', 'CO2']})
```

### Effects Plot

Plot cost, emissions, or other effect breakdowns. Effects can be grouped by component, individual contributor (flows), or time.

```python
flow_system.statistics.plot.effects()  # Total of all effects by component
flow_system.statistics.plot.effects(effect='costs')  # Just costs
flow_system.statistics.plot.effects(by='contributor')  # By individual flows/components
flow_system.statistics.plot.effects(aspect='temporal', by='time')  # Over time
```

**Key parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `aspect` | `'total'`, `'temporal'`, `'periodic'` | Which aspect to plot (default: `'total'`) |
| `effect` | str or None | Specific effect to plot (e.g., `'costs'`, `'CO2'`). If None, plots all. |
| `by` | `'component'`, `'contributor'`, `'time'` | Grouping dimension (default: `'component'`) |
| `select` | dict | xarray-style data selection |
| `colors` | dict | Color overrides for categories |
| `facet_col` | str | Dimension for column facets (default: `'scenario'`) |
| `facet_row` | str | Dimension for row facets (default: `'period'`) |

**Grouping options:**

- **`by='component'`**: Groups effects by parent component (e.g., all flows from a Boiler are summed together)
- **`by='contributor'`**: Shows individual contributors - flows and components that directly contribute to effects
- **`by='time'`**: Shows effects over time (only valid for `aspect='temporal'`)

!!! note "Contributors vs Components"
    Contributors include not just flows, but also components that directly contribute to effects (e.g., via `effects_per_active_hour`). The system automatically detects all contributors from the optimization solution.

### Variable Plot

Plot the same variable type across multiple elements for comparison:

```python
flow_system.statistics.plot.variable('on')  # All binary operation states
flow_system.statistics.plot.variable('flow_rate', include='Boiler')
flow_system.statistics.plot.variable('charge_state')  # All storage charge states
```

**Key parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | str | Variable suffix to match (e.g., `'on'`, `'flow_rate'`) |
| `include` | str or list | Only include elements containing these substrings |
| `exclude` | str or list | Exclude elements containing these substrings |
| `aggregate` | str | Time aggregation method |
| `mode` | `'line'`, `'bar'`, `'area'` | Visual style |

### Duration Curve

Plot load duration curves (sorted time series) to understand utilization patterns:

```python
flow_system.statistics.plot.duration_curve('Boiler(Q_th)')
flow_system.statistics.plot.duration_curve(['CHP(Q_th)', 'HeatPump(Q_th)'])
flow_system.statistics.plot.duration_curve('Demand(in)', normalize=True)
```

**Key parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `variables` | str or list | Variable name(s) to plot |
| `normalize` | bool | Normalize x-axis to 0-100% (default: False) |
| `mode` | `'line'`, `'area'` | Visual style |

## Common Parameters

Most plot methods share these parameters:

### Data Selection

Use xarray-style selection to filter data before plotting:

```python
# Single value
flow_system.statistics.plot.balance('Bus', select={'scenario': 'base'})

# Multiple values
flow_system.statistics.plot.balance('Bus', select={'scenario': ['base', 'high_demand']})

# Time slices
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
# Facet by scenario
flow_system.statistics.plot.balance('Bus', facet_col='scenario')

# Animate by period
flow_system.statistics.plot.balance('Bus', animate_by='period')

# Both
flow_system.statistics.plot.balance('Bus', facet_col='scenario', animate_by='period')
```

!!! note
    Facet and animation dimensions are automatically ignored if not present in the data. Defaults are `facet_col='scenario'` and `animate_by='period'` for balance plots.

### Include/Exclude Filtering

Filter flows using simple substring matching:

```python
# Only show flows containing 'Q_th'
flow_system.statistics.plot.balance('Bus', include='Q_th')

# Exclude flows containing 'Gas' or 'Grid'
flow_system.statistics.plot.balance('Bus', exclude=['Gas', 'Grid'])

# Combine include and exclude
flow_system.statistics.plot.balance('Bus', include='Boiler', exclude='auxiliary')
```

### Colors

Override colors using a dictionary:

```python
flow_system.statistics.plot.balance('Bus', colors={
    'Boiler(Q_th)': '#ff6b6b',
    'CHP(Q_th)': '#4ecdc4',
})
```

## Color Management

flixOpt provides centralized color management through the `flow_system.colors` accessor and carriers. This ensures consistent colors across all visualizations.

### Carriers

[`Carriers`][flixopt.carrier.Carrier] define energy or material types with associated colors. Built-in carriers are available in `CONFIG.Carriers`:

| Carrier | Color | Description |
|---------|-------|-------------|
| `electricity` | <span class="color-swatch" style="background:#FECB52"></span>`#FECB52` | Yellow - lightning/energy |
| `heat` | <span class="color-swatch" style="background:#D62728"></span>`#D62728` | Red - warmth/fire |
| `gas` | <span class="color-swatch" style="background:#1F77B4"></span>`#1F77B4` | Blue - natural gas |
| `hydrogen` | <span class="color-swatch" style="background:#9467BD"></span>`#9467BD` | Purple - clean/future |
| `fuel` | <span class="color-swatch" style="background:#8C564B"></span>`#8C564B` | Brown - fossil/oil |
| `biomass` | <span class="color-swatch" style="background:#2CA02C"></span>`#2CA02C` | Green - organic/renewable |

Colors are from the D3/Plotly palettes for professional consistency.

Assign carriers to buses for automatic coloring:

```python
# Buses use carrier colors automatically
heat_bus = fx.Bus('HeatNetwork', carrier='heat')
elec_bus = fx.Bus('Grid', carrier='electricity')

# Plots automatically use carrier colors for bus-related elements
flow_system.statistics.plot.sankey.flows()  # Buses colored by carrier
```

### Custom Carriers

Register custom carriers on your FlowSystem:

```python
# Create a custom carrier
biogas = fx.Carrier('biogas', color='#228B22', unit='kW', description='Biogas fuel')
hydrogen = fx.Carrier('hydrogen', color='#00CED1', unit='kg/h')

# Register with FlowSystem (overrides CONFIG.Carriers defaults)
flow_system.add_carrier(biogas)
flow_system.add_carrier(hydrogen)

# Access registered carriers
flow_system.carriers  # CarrierContainer with locally registered carriers
flow_system.get_carrier('biogas')  # Returns Carrier object
```

### Color Accessor

The `flow_system.colors` accessor provides centralized color configuration:

```python
# Configure colors for components
flow_system.colors.setup({
    'Boiler': '#D35400',
    'CHP': '#8E44AD',
    'HeatPump': '#27AE60',
})

# Or set individual colors
flow_system.colors.set_component_color('Boiler', '#D35400')
flow_system.colors.set_carrier_color('biogas', '#228B22')

# Load from file
flow_system.colors.setup('colors.json')  # or .yaml
```

### Context-Aware Coloring

Plot colors are automatically resolved based on context:

- **Bus balance plots**: Colors based on the connected component
- **Component balance plots**: Colors based on the connected bus/carrier
- **Sankey diagrams**: Buses use carrier colors, components use configured colors

```python
# Plotting a bus balance → flows colored by their parent component
flow_system.statistics.plot.balance('ElectricityBus')

# Plotting a component balance → flows colored by their connected bus/carrier
flow_system.statistics.plot.balance('CHP')
```

### Color Resolution Priority

Colors are resolved in this order:

1. **Explicit colors** passed to plot methods (always override)
2. **Component/bus colors** set via `flow_system.colors.setup()`
3. **Element `meta_data['color']`** if present
4. **Carrier colors** from FlowSystem or CONFIG.Carriers
5. **Default colorscale** (controlled by `CONFIG.Plotting.default_qualitative_colorscale`)

### Persistence

Color configurations are automatically saved with the FlowSystem:

```python
# Colors are persisted
flow_system.to_netcdf('my_system.nc')

# And restored
loaded = fx.FlowSystem.from_netcdf('my_system.nc')
loaded.colors  # Configuration restored
```

### Display Control

Control whether plots are shown automatically:

```python
# Don't show (useful in scripts)
result = flow_system.statistics.plot.balance('Bus', show=False)

# Show later
result.show()
```

The default behavior is controlled by `CONFIG.Plotting.default_show`.

## Complete Examples

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
flow_system.statistics.plot.compare(
    ['Battery1', 'Battery2', 'ThermalStorage'],
    variable='charge_state',
    mode='overlay'
).update(title='Storage Comparison')
```

### Creating a Report

```python
# Generate multiple plots for a report
plots = {
    'balance': flow_system.statistics.plot.balance('HeatBus', show=False),
    'storage': flow_system.statistics.plot.storage('ThermalStorage', show=False),
    'sankey': flow_system.statistics.plot.sankey.flows(show=False),
    'costs': flow_system.statistics.plot.effects(effect='costs', show=False),
}

# Export all
for name, plot in plots.items():
    plot.to_html(f'report_{name}.html')
    plot.to_netcdf(f'report_{name}.nc')  # xarray native format
```

### Working with xarray Data

The `.data` attribute returns xarray objects, giving you full access to xarray's powerful data manipulation capabilities:

```python
result = flow_system.statistics.plot.balance('Bus', show=False)

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
