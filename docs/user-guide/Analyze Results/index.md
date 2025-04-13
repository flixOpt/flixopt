# Results

The results of the optimization are stored in the `results` attribute of the [`Calculation`][flixopt.calculation.Calculation] object.
Depending on the type of calculation, the results are stored in different formats. For both [`FullCalculation`][flixopt.calculation.FullCalculation] and [`AggregatedCalculation`][flixopt.calculation.AggregatedCalculation], the results are stored in a [`CalculationResults`][flixopt.results.CalculationResults] object.
This object can be saved to a file and reloaded later. The used flow system is also stored in the results in the form of a xarray.Dataset. A proper FlowSystem can be reconstructed from the dataset using the [`FlowSystem.from_dataset`][flixopt.flow_system.FlowSystem.from_dataset] method.

## General Data handling

The results object provides a dictionary-like access to the results of the calculation. You can access any result by subscripting the object with the result's label.
The solution of the optimization is stored in the `solution` attribute of the results object as an [xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html).
This dataset contains all solutions for the variables of the optimization.
As the dataset can become a bit large, it is recommended to pass it to the [`fx.results.filter_dataset()`][flixopt.results.filter_dataset] function to select only the variables of interest.
There you can filter out variables that dont have a certain dimension, select a subset of the timesteps or **drop trailing nan values**.
### Dataset handling
Further, here are some of the most commonly used methods to process a dataset:

- `solution.sel(time=slice('2020-01-01', '2020-01-10'))`: Select a subset of the solution by time
- `solution.isel(time=0)`: Select a subset of the solution by time (by index)
- `solution.sum('time')`: Sum the solution over all timesteps (take care that you might need to multiply by the timesteps_per_hour to get the actual flow_hours)
- `solution.to_dataframe()`: Convert the solution to a pandas.DataFrame (leads to Multiindexes)
- `solution.to_pandas()`: Convert the solution to a pandas.DataFrame or pandas.Series, depending on the number of dimensions
- `solution.resample('D').sum()`: Resample the solution to daily timesteps and sum the values

For more information on how to use xarray, please refer to the [xarray documentation](https://docs.xarray.dev/en/stable/).

Instead
```python
results: fx.CalculationResults
da: xarray.Dataarray = results['Boiler(Q_th)|flow_rate']
```

From there you have all the functionality of an [xarray.DataArray](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html).
Here are some of the most commonly used methods:

- `da[2]` or `da[2:5]`: Select a subset of the data by index (time)
- `da.loc(time=slice('2020-01-01', '2020-01-10'))`: Select a subset of the data by time
- `da.sel(time='2020-01-01')`: Select a subset of the data by time (single timestamp)
- `da.sel(time=slice('2020-01-01', '2020-01-10'))`: Select a subset of the data by time
- `da.isel(time=0)`: Select a subset of the data by time (by index)
- `da.isel(time=range(0,5))`: Select a subset of the data by time (by index)
- `da.plot()`: Plot the data
- `da.to_dataframe()`: Convert the data to a pandas.DataFrame
- `da.to_netcdf()`: Save the data to a netcdf file

### Syntax

```python
result = calculation_results[result_label]

Where:
- `calculation_results` is a [`CalculationResults`][flixopt.results.CalculationResults] instance
- `result_label` is a string representing the label of the result
- The returned `result` is a pandas.DataFrame
## Accessing Component Results

The [`CalculationResults`][flixopt.results.CalculationResults] object provides dictionary-like access to individual component results. You can access any component's results by subscripting the object with the component's label.

### Syntax

```python
component_result = calculation_results[component_label]
```

Where:
- `calculation_results` is a [`CalculationResults`][flixopt.results.CalculationResults] instance
- `component_label` is a string representing the label of the component
- The returned `component_result` is a [`ComponentResults`][flixopt.results.ComponentResults] object

- The same goes for buses and effects, with corresponding return types.

### Example

```python
boiler_results = calculation_results['Boiler']

# You can also access nested components in hierarchical models
chp_turbine_results = calculation_results['CHP.Turbine']
```

### Return Value

The subscript operation returns a [`ComponentResult`][flixopt.results.ComponentResult] object that contains:
- Time series data for all component variables
- Metadata about the component's operation
- Component-specific performance metrics

### Additional Methods

You can chain this with other methods to extract specific information:

```python
# Get the power output time series of a generator
generator_power = calculation_results['Generator'].get_variable('power_output')

# Get the efficiency of a boiler
boiler_efficiency = calculation_results['Boiler'].get_metric('efficiency')
```

### Error Handling

If the component label doesn't exist, a `KeyError` is raised:

```python
try:
    missing_component = calculation_results['NonExistentComponent']
except KeyError as e:
    print(f"Component not found: {e}")
```

### See Also
- [`CalculationResults.get_component()`][flixopt.results.CalculationResults.get_component] - Alternative method for accessing components
- [`ComponentResult`][flixopt.results.ComponentResult] - Documentation for the returned component result object

## Extracting Results for specific Scenarios

If the calculation was run with scenarios, the results can be filtered by scenario using the `scenario` keyword argument.




`calculation.results['Storage'].node_balance().isel(scenario=0, drop=True).to_pandas()`