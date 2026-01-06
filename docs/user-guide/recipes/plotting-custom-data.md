# Plotting Custom Data

While the plot accessor (`flow_system.statistics.plot`) is designed for optimization results, you often need to plot custom xarray data. The `.fxplot` accessor provides the same convenience for any `xr.Dataset` or `xr.DataArray`.

## Quick Example

```python
import flixopt as fx
import xarray as xr

ds = xr.Dataset({
    'Solar': (['time'], solar_values),
    'Wind': (['time'], wind_values),
})

# Plot directly - no conversion needed!
ds.fxplot.line(title='Energy Generation')
ds.fxplot.stacked_bar(title='Stacked Generation')
```

## Full Documentation

For comprehensive documentation with interactive examples, see the [Custom Data Plotting](../../notebooks/fxplot_accessor_demo.ipynb) notebook which covers:

- All available plot methods (line, bar, stacked_bar, area, scatter, heatmap, pie)
- Automatic x-axis selection and faceting
- Custom colors and axis labels
- Duration curves with `.fxstats.to_duration_curve()`
- Configuration options
- Combining with xarray operations
