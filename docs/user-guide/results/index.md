# Analyzing Results

!!! note "Under Development"
    This section is being expanded with detailed tutorials.

Learn how to work with optimization results:

- Accessing solution data
- Plotting flows and states
- Exporting to various formats
- Comparing scenarios and periods

## Accessing Results

After running an optimization, access results directly from the FlowSystem:

```python
# Run optimization
flow_system.optimize(fx.solvers.HighsSolver())

# Access the full solution dataset
solution = flow_system.solution
print(solution['Boiler(Q_th)|flow_rate'])

# Access component-specific solutions
boiler = flow_system.components['Boiler']
print(boiler.solution)

# Access flow solutions
flow = flow_system.flows['Boiler(Q_th)']
print(flow.solution)
```

## Saving and Loading

Save the FlowSystem (including solution) for later analysis:

```python
# Save to NetCDF
flow_system.to_netcdf('results/my_system.nc')

# Load later
loaded_fs = fx.FlowSystem.from_netcdf('results/my_system.nc')
print(loaded_fs.solution)
```

## Getting Started

For now, see:

- **[Examples](../../examples/index.md)** - Result analysis patterns in working code
