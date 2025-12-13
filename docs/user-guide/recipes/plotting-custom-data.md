# Plotting Custom Data

The plot accessor (`flow_system.statistics.plot`) is designed for visualizing optimization results using element labels. If you want to create faceted plots with your own custom data (not from a FlowSystem), you can use Plotly Express directly with xarray data.

## Faceted Plots with Custom xarray Data

The key is converting your xarray Dataset to a long-form DataFrame that Plotly Express expects:

```python
import xarray as xr
import pandas as pd
import plotly.express as px

# Your custom xarray Dataset
my_data = xr.Dataset({
    'Solar': (['time', 'scenario'], solar_values),
    'Wind': (['time', 'scenario'], wind_values),
    'Demand': (['time', 'scenario'], demand_values),
}, coords={
    'time': timestamps,
    'scenario': ['Base', 'High RE', 'Low Demand']
})

# Convert to long-form DataFrame for Plotly Express
df = (
    my_data
    .to_dataframe()
    .reset_index()
    .melt(
        id_vars=['time', 'scenario'],  # Keep as columns
        var_name='variable',
        value_name='value'
    )
)

# Faceted stacked bar chart
fig = px.bar(
    df,
    x='time',
    y='value',
    color='variable',
    facet_col='scenario',
    barmode='relative',
    title='Energy Balance by Scenario'
)
fig.show()

# Faceted line plot
fig = px.line(
    df,
    x='time',
    y='value',
    color='variable',
    facet_col='scenario'
)
fig.show()

# Faceted area chart
fig = px.area(
    df,
    x='time',
    y='value',
    color='variable',
    facet_col='scenario'
)
fig.show()
```

## Common Plotly Express Faceting Options

| Parameter | Description |
|-----------|-------------|
| `facet_col` | Dimension for column subplots |
| `facet_row` | Dimension for row subplots |
| `animation_frame` | Dimension for animation slider |
| `facet_col_wrap` | Number of columns before wrapping |

```python
# Row and column facets
fig = px.line(df, x='time', y='value', color='variable',
              facet_col='scenario', facet_row='region')

# Animation over time periods
fig = px.bar(df, x='variable', y='value', color='variable',
             animation_frame='period', barmode='group')

# Wrap columns
fig = px.line(df, x='time', y='value', color='variable',
              facet_col='scenario', facet_col_wrap=2)
```

## Heatmaps with Custom Data

For heatmaps, you can pass 2D arrays directly to `px.imshow`:

```python
import plotly.express as px

# 2D data (e.g., days × hours)
heatmap_data = my_data['Solar'].sel(scenario='Base').values.reshape(365, 24)

fig = px.imshow(
    heatmap_data,
    labels={'x': 'Hour', 'y': 'Day', 'color': 'Power [kW]'},
    aspect='auto',
    color_continuous_scale='portland'
)
fig.show()

# Faceted heatmaps using subplots
from plotly.subplots import make_subplots
import plotly.graph_objects as go

scenarios = ['Base', 'High RE']
fig = make_subplots(rows=1, cols=len(scenarios), subplot_titles=scenarios)

for i, scenario in enumerate(scenarios, 1):
    data = my_data['Solar'].sel(scenario=scenario).values.reshape(365, 24)
    fig.add_trace(go.Heatmap(z=data, colorscale='portland'), row=1, col=i)

fig.update_layout(title='Solar Output by Scenario')
fig.show()
```

This approach gives you full control over your visualizations while leveraging Plotly's powerful faceting capabilities.
