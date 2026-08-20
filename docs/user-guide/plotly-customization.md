# Plotly Customization

flixOpt's plotting is built on [Plotly Express](https://plotly.com/python/plotly-express/). This page covers flixopt-specific customization. For general Plotly knowledge, see:

- [Plotly Express documentation](https://plotly.com/python/plotly-express/)
- [xarray-plotly package](https://github.com/lgabs/xarray-plotly) - the `.plotly` accessor used internally

## flixOpt's Theme

flixOpt registers a `'flixopt'` Plotly template on import, but doesn't activate it by default:

```python
import plotly.io as pio
import flixopt as fx

# Template is registered but not active
'flixopt' in pio.templates  # True
pio.templates.default       # Still 'plotly'

# Activate manually
fx.CONFIG.use_theme()       # Sets 'plotly_white+flixopt'

# Or via presets (recommended)
fx.CONFIG.notebook()        # Activates theme + notebook settings
```

## Default Slot Assignments

flixOpt pre-assigns Plotly slots to provide sensible defaults:

| Plot Type | Defaults |
|-----------|----------|
| Balance (bar) | `x='time'`, `color='variable'` |
| Flows (line) | `x='time'`, `color='variable'` |
| Comparison | `facet_col='case'` or `line_dash='case'` |

Override any default by passing the parameter explicitly:

```python
# Use animation instead of faceting
flow_system.statistics.plot.balance('Heat', animation_frame='case', facet_col=None)
```

## Common Customizations

### Update Layout

```python
result = flow_system.statistics.plot.balance('Heat')
result.figure.update_layout(
    title='Custom Title',
    xaxis_title='Time',
    yaxis_title='Power [kW]',
    height=500,
)
```

### Update Traces

```python
result.figure.update_traces(opacity=0.8, line_width=2)

# Or target specific traces
for trace in result.figure.data:
    if 'Boiler' in trace.name:
        trace.line.width = 3
```

### Combine Figures

```python
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

balance = flow_system.statistics.plot.balance('Heat', show=False)
storage = flow_system.statistics.plot.storage('Tank', show=False)

for trace in balance.figure.data:
    fig.add_trace(trace, row=1, col=1)
for trace in storage.figure.data:
    fig.add_trace(trace, row=2, col=1)

fig.show()
```

## Plotly Express Parameters

These work with most flixOpt plot methods:

| Parameter | Description |
|-----------|-------------|
| `title` | Plot title |
| `height`, `width` | Figure dimensions |
| `facet_col_wrap` | Max columns before wrapping |
| `color_discrete_map` | Dict mapping labels to colors |
| `template` | Plotly template name |

```python
flow_system.statistics.plot.balance(
    'Heat',
    title='Heat Balance',
    height=400,
    color_discrete_map={'Boiler(Q_th)': 'red'},
)
```

## Display Control

```python
# Don't show automatically
result = flow_system.statistics.plot.balance('Bus', show=False)

# Show later
result.show()
```

The default is controlled by `CONFIG.Plotting.default_show`.
