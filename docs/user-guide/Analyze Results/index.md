# Results

The results of the optimization are stored in the `results` attribute of the [`Calculation`][flixopt.calculation.Calculation] object.
Depending on the type of calculation, the results are stored in different formats. For both [`FullCalculation`][flixopt.calculation.FullCalculation] and [`AggregatedCalculation`][flixopt.calculation.AggregatedCalculation], the results are stored in a [`CalculationResults`][flixopt.results.CalculationResults] object.
THis object can be saved to a file and reloaded later. The used flow system is also stored in the results in the form of a xarray.Dataset. A proper FlowSystem can be reconstructed from the dataset using the [`FlowSystem.from_dataset`][flixopt.flow_system.FlowSystem.from_dataset] method.

## Extracting Results for specific Elements

In most cases, one wants to extract the results for a specific element, such as a bus or a component.
