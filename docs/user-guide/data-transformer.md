# DataTransformer Guide

The `DataTransformer` class provides utilities for converting xarray data into formats suitable for plotting, following industry-standard patterns for data visualization.

## Overview

Modern plotting libraries like Plotly Express work best with "tidy" (long-format) data where:
- Each row represents one observation
- Each column represents one variable
- Dimensions become columns

The `DataTransformer` handles the conversion from xarray's multi-dimensional arrays to these plotting-friendly formats.

## Why DataTransformer?

**Problem:** Plotly Express doesn't directly handle xarray's multidimensional data.

```python
# xarray has multiple dimensions
data = xr.DataArray(
    [[100, 150, 200], [50, 75, 100]],
    dims=['generator', 'time'],
    coords={'generator': ['solar', 'wind'], 'time': [0, 1, 2]}
)

# This won't work:
px.line(data)  # ❌ Error!
```

**Solution:** Transform to tidy format first.

```python
from flixopt.plotting_accessor import DataTransformer

# Convert to tidy DataFrame
df = DataTransformer.to_tidy_dataframe(data, value_name='generation')

# Now Plotly Express works:
fig = px.line(df, x='time', y='generation', color='generator')  # ✓ Works!
```

## Core Methods

### to_tidy_dataframe()

Convert xarray to tidy (long-format) DataFrame.

**Signature:**
```python
DataTransformer.to_tidy_dataframe(
    data: xr.DataArray | xr.Dataset,
    value_name: str = 'value',
    reset_index: bool = True
) -> pd.DataFrame
```

**Parameters:**
- `data`: xarray DataArray or Dataset to convert
- `value_name`: Name for the value column (for DataArrays)
- `reset_index`: If True, dimensions become regular columns

**Returns:** pandas DataFrame in tidy format

**Example:**
```python
import xarray as xr
import plotly.express as px
from flixopt.plotting_accessor import DataTransformer

# Multi-dimensional data
data = xr.DataArray(
    [[100, 150, 200], [50, 75, 100]],
    dims=['generator', 'time'],
    coords={
        'generator': ['solar', 'wind'],
        'time': [0, 1, 2]
    }
)

# Convert to tidy format
df = DataTransformer.to_tidy_dataframe(data, value_name='generation')
print(df)
#   generator  time  generation
# 0     solar     0         100
# 1     solar     1         150
# 2     solar     2         200
# 3      wind     0          50
# 4      wind     1          75
# 5      wind     2         100

# Now plot with Plotly Express
fig = px.line(df, x='time', y='generation', color='generator')
fig.show()
```

**With Dataset:**
```python
# Dataset with multiple variables
ds = xr.Dataset({
    'generation': generation_data,
    'capacity': capacity_data
})

# Convert to tidy format (each variable becomes a column)
df = DataTransformer.to_tidy_dataframe(ds)
print(df)
#   generator  time  generation  capacity
# 0     solar     0         100       200
# 1     solar     1         150       200
# 2      wind     0          50       100
# 3      wind     1          75       100
```

### to_wide_dataframe()

Convert xarray to wide (matrix-like) format.

**Signature:**
```python
DataTransformer.to_wide_dataframe(
    data: xr.DataArray,
    index_dim: str | None = None,
    columns_dim: str | None = None
) -> pd.DataFrame
```

**Parameters:**
- `data`: xarray DataArray to convert (must have 2 dimensions)
- `index_dim`: Dimension for index (usually 'time')
- `columns_dim`: Dimension for columns (usually categories)

**Returns:** pandas DataFrame in wide format

**Example:**
```python
# Convert time series to wide format
data = xr.DataArray(
    [[100, 50], [150, 75], [200, 100]],
    dims=['time', 'generator'],
    coords={
        'time': [0, 1, 2],
        'generator': ['solar', 'wind']
    }
)

df = DataTransformer.to_wide_dataframe(
    data,
    index_dim='time',
    columns_dim='generator'
)
print(df)
# generator  solar  wind
# time
# 0            100    50
# 1            150    75
# 2            200   100

# Useful for stacked area charts
fig = px.area(df)
```

### aggregate_dimension()

Reduce data along a dimension using aggregation.

**Signature:**
```python
DataTransformer.aggregate_dimension(
    data: xr.DataArray | xr.Dataset,
    dim: str,
    method: Literal['sum', 'mean', 'max', 'min', 'std', 'median'] = 'sum'
) -> xr.DataArray | xr.Dataset
```

**Parameters:**
- `data`: xarray data to aggregate
- `dim`: Dimension to aggregate along
- `method`: Aggregation method

**Returns:** Aggregated xarray with dimension removed

**Example:**
```python
# Sum over time to get totals per generator
data = xr.DataArray(
    [[100, 150, 200], [50, 75, 100]],
    dims=['generator', 'time'],
    coords={
        'generator': ['solar', 'wind'],
        'time': [0, 1, 2]
    }
)

total = DataTransformer.aggregate_dimension(data, dim='time', method='sum')
print(total)
# <xarray.DataArray (generator: 2)>
# array([450, 225])
# Coordinates:
#   * generator  (generator) <U5 'solar' 'wind'

# Convert to tidy for bar chart
df = DataTransformer.to_tidy_dataframe(total, value_name='total')
fig = px.bar(df, x='generator', y='total')
```

**Available Methods:**
- `'sum'` - Sum values along dimension
- `'mean'` - Average values
- `'max'` - Maximum value
- `'min'` - Minimum value
- `'std'` - Standard deviation
- `'median'` - Median value

### select_subset()

Select specific values or ranges from dimensions.

**Signature:**
```python
DataTransformer.select_subset(
    data: xr.DataArray | xr.Dataset,
    **selectors
) -> xr.DataArray | xr.Dataset
```

**Parameters:**
- `data`: xarray data to select from
- `**selectors`: Dimension names and values to select

**Returns:** Selected subset of data

**Example:**
```python
# Select single generator
solar = DataTransformer.select_subset(data, generator='solar')
print(solar)
# <xarray.DataArray (time: 3)>
# array([100, 150, 200])

# Select time range
morning = DataTransformer.select_subset(
    data,
    time=slice(0, 1)
)

# Multiple selectors
subset = DataTransformer.select_subset(
    data,
    generator='solar',
    time=slice(0, 1)
)
```

### melt_dataset()

Convert Dataset to ultra-tidy format with variable names as a column.

**Signature:**
```python
DataTransformer.melt_dataset(
    data: xr.Dataset,
    id_vars: list[str] | None = None,
    value_vars: list[str] | None = None,
    var_name: str = 'variable',
    value_name: str = 'value'
) -> pd.DataFrame
```

**Parameters:**
- `data`: xarray Dataset to melt
- `id_vars`: Dimension names to keep as identifier columns
- `value_vars`: Data variable names to melt (None = all)
- `var_name`: Name for variable column
- `value_name`: Name for value column

**Returns:** Melted pandas DataFrame

**Example:**
```python
# Compare multiple variables in one plot
ds = xr.Dataset({
    'generation': xr.DataArray([100, 150, 200], dims=['time']),
    'demand': xr.DataArray([120, 130, 180], dims=['time']),
    'storage': xr.DataArray([10, 15, 5], dims=['time'])
})

df = DataTransformer.melt_dataset(ds)
print(df)
#    time    variable  value
# 0     0  generation    100
# 1     1  generation    150
# 2     2  generation    200
# 3     0      demand    120
# 4     1      demand    130
# 5     2      demand    180
# 6     0     storage     10
# 7     1     storage     15
# 8     2     storage      5

# Plot all variables together
fig = px.line(df, x='time', y='value', color='variable')
```

## Usage in Plotters

All plotter classes have direct access to DataTransformer methods via helper functions:

```python
class CustomPlotter(InteractivePlotter):
    def my_custom_plot(self):
        # Get data from parent
        data = self._get_dataset()

        # Use transformation helpers
        df = self._to_tidy(value_name='generation')
        aggregated = self._aggregate(dim='time', method='sum')
        subset = self._select(generator='solar')
        wide_df = self._to_wide(index_dim='time', columns_dim='generator')
        melted = self._melt(var_name='metric', value_name='value')

        # Create custom visualization
        fig = px.scatter(df, x='time', y='generation', color='generator')
        return fig
```

## Common Patterns

### Pattern 1: Aggregate → Tidy → Plot

```python
# Sum over time, convert to tidy, plot
total = DataTransformer.aggregate_dimension(data, dim='time', method='sum')
df = DataTransformer.to_tidy_dataframe(total, value_name='total_generation')
fig = px.bar(df, x='generator', y='total_generation')
```

### Pattern 2: Select → Aggregate → Plot

```python
# Select subset, aggregate, plot
solar = DataTransformer.select_subset(data, generator='solar')
avg = DataTransformer.aggregate_dimension(solar, dim='time', method='mean')
# Now plot avg...
```

### Pattern 3: Multi-variable Comparison

```python
# Melt dataset, plot all variables
df = DataTransformer.melt_dataset(dataset)
fig = px.line(df, x='time', y='value', color='variable', facet_row='generator')
```

### Pattern 4: Time Series with Aggregation

```python
# Keep time, aggregate other dimensions
data_2d = DataTransformer.aggregate_dimension(data_3d, dim='scenario', method='mean')
df = DataTransformer.to_tidy_dataframe(data_2d, value_name='flow')
fig = px.line(df, x='time', y='flow', color='generator')
```

### Pattern 5: Heatmap Preparation

```python
# Convert to wide format for heatmap
df_wide = DataTransformer.to_wide_dataframe(
    data,
    index_dim='day',
    columns_dim='hour'
)
fig = px.imshow(df_wide)
```

## Performance Tips

1. **Aggregate early**: Reduce data size before converting to DataFrame
   ```python
   # Good: Aggregate first, then convert
   total = DataTransformer.aggregate_dimension(data, dim='time')
   df = DataTransformer.to_tidy_dataframe(total)

   # Less efficient: Convert all data, then aggregate
   df = DataTransformer.to_tidy_dataframe(data)
   df_total = df.groupby('generator').sum()
   ```

2. **Select before processing**: Filter data early
   ```python
   # Good: Select first
   subset = DataTransformer.select_subset(data, time=slice(0, 100))
   df = DataTransformer.to_tidy_dataframe(subset)

   # Less efficient: Convert all, then filter
   df = DataTransformer.to_tidy_dataframe(data)
   df_subset = df[df['time'] <= 100]
   ```

3. **Use appropriate format**: Choose the right format for your visualization
   - Tidy format → Most Plotly Express plots
   - Wide format → Heatmaps, some area charts
   - Melted format → Comparing multiple variables

## Integration with Results

DataTransformer is integrated throughout the results system:

```python
# In statistics methods
plotter = results.statistics.flow_summary()
# Internally uses DataTransformer for data preparation

# In plot accessors
node_plotter = results['Boiler'].plot.node_balance()
# Uses DataTransformer helpers for visualization

# Direct usage
from flixopt.plotting_accessor import DataTransformer
data = results['Boiler'].solution['Q_th|flow_rate']
df = DataTransformer.to_tidy_dataframe(data, value_name='thermal_power')
```

## Best Practices

1. **Use tidy format as default** - It's the most flexible for Plotly Express
2. **Name value columns meaningfully** - Use descriptive names instead of 'value'
3. **Aggregate before plotting large datasets** - Improves performance
4. **Select subsets when appropriate** - Don't plot more data than needed
5. **Chain transformations logically** - select → aggregate → convert → plot

## Example Workflow

Complete workflow from xarray to interactive plot:

```python
import xarray as xr
import plotly.express as px
from flixopt.plotting_accessor import DataTransformer

# 1. Start with multi-dimensional xarray data
data = xr.DataArray(
    [...],  # Your optimization results
    dims=['generator', 'time', 'scenario'],
    coords={...}
)

# 2. Select relevant subset
data_base = DataTransformer.select_subset(data, scenario='base')

# 3. Aggregate if needed
data_total = DataTransformer.aggregate_dimension(data_base, dim='time', method='sum')

# 4. Convert to tidy format
df = DataTransformer.to_tidy_dataframe(data_total, value_name='total_generation')

# 5. Create visualization
fig = px.bar(
    df,
    x='generator',
    y='total_generation',
    title='Total Generation by Generator (Base Scenario)',
    labels={'total_generation': 'Generation (MWh)'}
)
fig.show()
```

## See Also

- [Plotting and Statistics API](./plotting-and-statistics.md) - Main plotting guide
- [Advanced Plotting Patterns](./advanced-plotting.md) - Custom visualization patterns
- [xarray Documentation](https://docs.xarray.dev/) - Learn more about xarray
- [Plotly Express Documentation](https://plotly.com/python/plotly-express/) - Plotly plotting guide
