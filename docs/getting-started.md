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
pip install "flixopt[full]"
```

## Logging

FlixOpt uses Python's standard logging module with optional colored output via [colorlog](https://github.com/borntyping/python-colorlog). Logging is silent by default but can be easily configured.

```python
from flixopt import CONFIG

# Enable colored console logging
CONFIG.Logging.enable_console('INFO')

# Or use a preset configuration for exploring
CONFIG.exploring()
```

For advanced logging configuration, you can use Python's standard logging module directly:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

For more details on logging configuration, see the [`CONFIG.Logging`][flixopt.config.CONFIG.Logging] documentation.

## Basic Workflow

Working with FlixOpt follows a general pattern:

1. **Create a [`FlowSystem`][flixopt.flow_system.FlowSystem]** with a time series
2. **Define [`Effects`][flixopt.effects.Effect]** (costs, emissions, etc.)
3. **Define [`Buses`][flixopt.elements.Bus]** as connection points in your system
4. **Add [`Components`][flixopt.components]** like converters, storage, sources/sinks with their Flows
5. **Run [`Optimizations`][flixopt.optimization]** to optimize your system
6. **Analyze [`Results`][flixopt.results]** using built-in or external visualization tools

## Next Steps

Now that you've installed FlixOpt and understand the basic workflow, you can:

- Learn about the [core concepts of flixopt](user-guide/core-concepts.md)
- Explore some [examples](examples/index.md)
- Check the [API reference](api-reference/index.md) for detailed documentation
