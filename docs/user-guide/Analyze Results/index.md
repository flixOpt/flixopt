# Working with FlixOpt Results

The results of an optimization are stored in the `results` attribute of a [`Calculation`][flixopt.calculation.Calculation] object. This documentation provides a comprehensive guide to working with these results effectively.

## Result Objects

Depending on the calculation type, the results are stored in different formats:

- For [`FullCalculation`][flixopt.calculation.FullCalculation] and [`AggregatedCalculation`][flixopt.calculation.AggregatedCalculation]: Results are stored in a [`CalculationResults`][flixopt.results.CalculationResults] object
- For [`SegmentedCalculation`][flixopt.calculation.SegmentedCalculation]: Results are stored in a [`SegmentedCalculationResults`][flixopt.results.SegmentedCalculationResults] object

These result objects can be saved to files and reloaded later for further analysis.

## Accessing Results

The results object provides dictionary-like access to components, buses, and effects:

```python
# Get results for a specific component
boiler_results = calculation_results['Boiler']

# Get results for a specific bus
electricity_bus_results = calculation_results['ElectricityBus']

# Get results for a specific effect
costs_results = calculation_results['Costs']
```

Each of these results is an instance of [`ComponentResults`][flixopt.results.ComponentResults], [`BusResults`][flixopt.results.BusResults], or [`EffectResults`][flixopt.results.EffectResults] respectively, providing specialized methods for each type.

## Working with the Solution Dataset

The core of the results is the `solution` attribute, which is an [xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) containing all variables from the optimization.

```python
# Access the complete solution dataset
solution = calculation_results.solution

# Access a specific variable directly
flow_rate = calculation_results.solution['Boiler(Q_th)|flow_rate']

# Or more conveniently through the component
flow_rate = calculation_results['Boiler'].solution['Boiler(Q_th)|flow_rate']
```

### Filtering the Solution

The solution dataset can become large with many variables. Use the [`filter_solution`][flixopt.results.CalculationResults.filter_solution] method to select only variables of interest:

```python
# Get only time-dependent variables
time_vars = calculation_results.filter_solution(variable_dims='time')

# Get only scalar variables
scalar_vars = calculation_results.filter_solution(variable_dims='scalar')

# Filter for a specific component and time range
boiler_jan_2022 = calculation_results.filter_solution(
    element='Boiler',
    timesteps=pd.date_range('2020-01-01 00:00', '2020-01-01 04:00', freq='H')
)

```

### Common xarray Operations

The solution dataset supports all xarray functionality:

```python
solution: xarray.DataSet = calculation_results.solution
# Select a time range
solution_jan = solution.sel(time=slice('2020-01-01', '2020-01-31'))
solution_3_steps = solution.isel(time=slice(0, 3))

solution_time_x = solution.sel(time='2020-01-01')

# Sum over time dimension
total_by_var = solution.sum('time')

# Resample to daily values
daily_sums = solution.resample(time='D').sum()

# Convert to pandas DataFrame
solution_df = solution.to_dataframe() # Or solution.to_pandas() to not convert without a multiindex
```

## Component Results

The [`ComponentResults`][flixopt.results.ComponentResults] class provides specialized methods for analyzing component behavior:

```python
# Get a component's results
storage = calculation_results['Storage']

# Plot the node balance (inputs and outputs) of a component
storage.plot_node_balance(save='storage_balance.html')

# For storage components, plot the charge state
storage.plot_charge_state(show=True)

# Get the node balance as a dataset
balance = storage.node_balance(negate_inputs=True)
```

### Working with Storage Results

Storage components have additional methods:

```python
# Check if a component is a storage
is_storage = calculation_results['Battery'].is_storage

# Get the charge state of a storage
charge_state = calculation_results['Battery'].charge_state

# Get node balance including charge state
balance_with_charge = calculation_results['Battery'].node_balance_with_charge_state()
```

## Bus Results

The [`BusResults`][flixopt.results.BusResults] class provides methods for analyzing energy or material flows through buses:

```python
# Get a bus's results
heat_bus = calculation_results['Fernwärme']

# Plot the node balance of a bus
heat_bus.plot_node_balance(show=True)

# Show a pie chart of flows through the bus
heat_bus.plot_node_balance_pie(lower_percentage_group=2, show=True)
```

## Effect Results

The [`EffectResults`][flixopt.results.EffectResults] class helps analyze effects like costs or emissions:

```python
# Get an effect's results
costs = calculation_results['Costs']

# Get the shares of an effect from a specific element
boiler_costs = costs.get_shares_from('Boiler')
```

## Working with Scenarios

If your calculation included scenarios, you can access scenario-specific results:

```python
# Select specific scenario when plotting
storage.plot_charge_state(scenario='high_demand')

# Filter node balance for a specific scenario
balance = storage.node_balance()
scenario_balance = balance.sel(scenario='high_demand')

# View results for a single variable as a DataFrame, with columns represensting scenarios
df_flow_rate = solution['Storage(Q_th)|flow_rate'].to_pandas()
```

If plotting without specifiing a scenario, the first scenario is used.

## Visualization

The results objects provide several visualization methods:

```python
# Plot a heatmap of a variable
calculation_results.plot_heatmap(
    variable_name='Boiler(Q_th)|flow_rate',
    heatmap_timeframes='D',
    heatmap_timesteps_per_frame='h',
    show=True
)

# Plot the network graph
calculation_results.plot_network(show=True)
```

## Saving and Loading Results

Results can be saved to files and loaded later:

```python
# Save results to files
calculation_results.to_file(folder='results', compression=5)

# Load results from files
loaded_results = fx.results.CalculationResults.from_file('results', 'optimization_run')
```

## Converting to Other Formats

Results can be converted to various formats for further analysis:

```python
# Convert to pandas DataFrame
df = calculation_results['Boiler'].node_balance().to_dataframe()

# Save as CSV
df.to_csv('boiler_results.csv')

# Convert flow rates to flow_hours (kW to kWh) - Multiply rate by duration
flow_hours = calculation_results['Boiler'].node_balance(mode='flow_hours')
```

## Tips for Working with Large Datasets

- Use `filter_solution()` to limit the variables you're working with
- Select only the time range you need with `sel(time=slice(start, end))`
- Consider using `isel()` instead of `sel()` for faster indexing by position
- For aggregated views, use `resample()` to reduce the data size