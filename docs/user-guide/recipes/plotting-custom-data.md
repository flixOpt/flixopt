# Plotting Custom Data

While the plot accessor (`flow_system.statistics.plot`) is designed for optimization results, you often need to plot custom xarray data. The `.plotly` accessor provides the same convenience for any `xr.Dataset` or `xr.DataArray`.

!!! note "Accessor Registration"
    The `.plotly` and `.fxstats` accessors are automatically registered when you import flixopt.
    Just `import flixopt` and they become available on all xarray objects.

## Quick Example

```python
import flixopt as fx  # Registers .plotly and .fxstats accessors
import xarray as xr

ds = xr.Dataset({
    'Solar': (['time'], solar_values),
    'Wind': (['time'], wind_values),
})

# Plot directly - no conversion needed!
ds.plotly.line(title='Energy Generation')
ds.plotly.bar(title='Stacked Generation')
```

## Full Documentation

The `.plotly` accessor is provided by the [xarray_plotly](https://github.com/FBumann/xarray_plotly) package. See the [full documentation](https://fbumann.github.io/xarray_plotly/) for:

- All available plot methods (line, bar, area, scatter, imshow, pie, box)
- Automatic dimension assignment
- Custom colors and styling
- Combining with xarray operations

For duration curves, use `.fxstats.to_duration_curve()` before plotting.
