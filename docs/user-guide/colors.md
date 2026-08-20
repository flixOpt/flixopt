# Color Management

flixOpt provides centralized color management to ensure consistent colors across all visualizations.

## Carriers

[`Carriers`][flixopt.carrier.Carrier] define energy or material types with associated colors. Built-in carriers are available in `CONFIG.Carriers`:

| Carrier | Color | Hex |
|---------|-------|-----|
| `electricity` | <span style="background:#FECB52; padding:0 8px; border-radius:3px;">&nbsp;</span> | `#FECB52` |
| `heat` | <span style="background:#D62728; padding:0 8px; border-radius:3px;">&nbsp;</span> | `#D62728` |
| `gas` | <span style="background:#1F77B4; padding:0 8px; border-radius:3px;">&nbsp;</span> | `#1F77B4` |
| `hydrogen` | <span style="background:#9467BD; padding:0 8px; border-radius:3px;">&nbsp;</span> | `#9467BD` |
| `fuel` | <span style="background:#8C564B; padding:0 8px; border-radius:3px;">&nbsp;</span> | `#8C564B` |
| `biomass` | <span style="background:#2CA02C; padding:0 8px; border-radius:3px;">&nbsp;</span> | `#2CA02C` |

Assign carriers to buses for automatic coloring:

```python
heat_bus = fx.Bus('HeatNetwork', carrier='heat')
elec_bus = fx.Bus('Grid', carrier='electricity')

# Plots automatically use carrier colors
flow_system.statistics.plot.sankey.flows()
```

## Custom Carriers

Register custom carriers on your FlowSystem:

```python
biogas = fx.Carrier('biogas', color='#228B22', unit='kW', description='Biogas fuel')

flow_system.add_carrier(biogas)
```

## Setting Component Colors

### At Construction

```python
boiler = fx.LinearConverter('Boiler', ..., color='#D35400')
storage = fx.Storage('Battery', ..., color='green')
```

### Via Topology Accessor

```python
# Single component
flow_system.topology.set_component_color('Boiler', '#D35400')

# Multiple components
flow_system.topology.set_component_colors({
    'Boiler': '#D35400',
    'CHP': '#8E44AD',
    'HeatPump': '#27AE60',
})

# Apply a colorscale to all components
flow_system.topology.set_component_colors('turbo')

# Apply colorscales to groups
flow_system.topology.set_component_colors({
    'Oranges': ['Solar1', 'Solar2', 'Solar3'],
    'Blues': ['Wind1', 'Wind2'],
})
```

### Carrier Colors

```python
flow_system.topology.set_carrier_color('electricity', '#FECB52')
```

## Context-Aware Coloring

Plot colors are automatically resolved based on context:

- **Bus balance plots**: Flows colored by their parent component
- **Component balance plots**: Flows colored by their connected bus/carrier
- **Sankey diagrams**: Buses use carrier colors, components use configured colors

```python
# Plotting a bus → flows colored by component
flow_system.statistics.plot.balance('ElectricityBus')

# Plotting a component → flows colored by carrier
flow_system.statistics.plot.balance('CHP')
```

## Color Resolution Priority

Colors are resolved in this order:

1. **Explicit colors** passed to plot methods (always override)
2. **Component colors** set via topology or at construction
3. **Carrier colors** for buses
4. **Default colorscale** (`CONFIG.Plotting.default_qualitative_colorscale`)

## Persistence

Colors are automatically saved and restored with the FlowSystem:

```python
# Colors are persisted
flow_system.to_netcdf('my_system.nc')

# And restored
loaded = fx.FlowSystem.from_netcdf('my_system.nc')
loaded.topology.component_colors  # Colors preserved
```

## Accessing Colors Programmatically

The `topology` accessor provides cached dictionaries:

```python
flow_system.topology.carrier_colors   # {'electricity': '#FECB52', ...}
flow_system.topology.component_colors # {'Boiler': '#1f77b4', ...}
flow_system.topology.bus_colors       # {'ElecBus': '#FECB52', ...}
```

You can also inspect individual components:

```python
for comp in flow_system.components.values():
    print(f"{comp.label}: {comp.color}")
```

## Auto-Assignment

Components without explicit colors are automatically assigned colors when you call `optimize()` or `connect_and_transform()`. The colors come from `CONFIG.Plotting.default_qualitative_colorscale` (default: `'plotly'`).
