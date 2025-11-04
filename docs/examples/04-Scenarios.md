# Scenarios Example
**Note:** This example relies on time series data. You can find it in the `examples` folder of the FlixOpt repository.

This example demonstrates how to model energy systems with multiple scenarios and periods, including:

- **Multiple scenarios**: Define different demand scenarios (e.g., "Base Case" and "High Demand") with associated probabilities
- **Multiple periods**: Model different time periods (e.g., years 2020, 2021, 2022) with varying parameters
- **Scenario weights**: Assign probabilities to scenarios for stochastic optimization
- **Period-specific parameters**: Use different values across periods (e.g., escalating gas prices, varying storage losses)

```python
{! ../examples/04_Scenarios/scenario_example.py !}
```
