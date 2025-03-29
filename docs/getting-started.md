# Getting Started with FlixOpt

This guide will help you install FlixOpt, understand its basic concepts, and run your first optimization model.

## Installation

### Basic Installation

Install FlixOpt directly into your environment using pip:

```bash
pip install flixopt
```

This provides the core functionality with the HiGHS solver included.

### Full Installation

For all features including interactive network visualizations and time series aggregation:

```bash
pip install "flixopt[full]""
```

## Basic Workflow

Working with FlixOpt follows a general pattern:

1. **Create a [`FlowSystem`][flixopt.flow_system.FlowSystem]** with a time series
2. **Define [`Effects`][flixopt.effects.Effect]** (costs, emissions, etc.)
3. **Define [`Buses`][flixopt.elements.Bus]** as connection points in your system
4. **Add [`Components`][flixopt.components]** like converters, storage, sources/sinks with their Flows
5. **Run [`Calculations`][flixopt.calculation]** to optimize your system
6. **Analyze [`Results`][flixopt.results]** using built-in or external visualization tools

## Next Steps

Now that you've installed FlixOpt and understand the basic workflow, you can:

- Learn about the [core concepts of FlixOpt](user-guide/index.md)
- Explore some [examples](examples/)
- Check the [API reference](api-reference/index.md) for detailed documentation
