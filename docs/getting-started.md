# Getting Started with flixOpt

This guide will help you install flixOpt, understand its basic concepts, and run your first optimization model.

## Installation

### Basic Installation

Install flixOpt directly into your environment using pip:

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

Working with flixOpt follows a general pattern:

1. **Create a [`FlowSystem`][flixOpt.flow_system.FlowSystem]** with a time series
2. **Define [`Effects`][flixOpt.effects.Effect]** (costs, emissions, etc.)
3. **Define [`Buses`][flixOpt.elements.Bus]** as connection points in your system
4. **Add [`Components`][flixOpt.components]** like converters, storage, sources/sinks with their Flows
5. **Run [`Calculations`][flixOpt.calculation]** to optimize your system
6. **Analyze [`Results`][flixOpt.results]** using built-in or external visualization tools

## Next Steps

Now that you've installed flixOpt and understand the basic workflow, you can:

- Learn about the [core concepts of flixOpt](concepts-and-math/index.md)
- Explore some [examples](examples/index.md)
- Check the [API reference](api-reference/index.md) for detailed documentation
