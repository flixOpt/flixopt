# Installation

This guide covers installing flixOpt and its dependencies.


## Basic Installation

Install flixOpt directly into your environment using pip:

```bash
pip install flixopt
```

This provides the core functionality with the HiGHS solver included.

## Full Installation

For all features including interactive network visualizations and time series aggregation:

```bash
pip install "flixopt[full]"
```

## Development Installation

If you want to contribute to flixOpt or work with the latest development version:

```bash
git clone https://github.com/flixOpt/flixopt.git
cd flixopt
pip install -e ".[full,dev,docs]"
```

## Solver Installation

### HiGHS (Included)

The HiGHS solver is included with flixOpt and works out of the box. No additional installation is required.

### Gurobi (Optional)

For academic use, Gurobi offers free licenses:

1. Register for an academic license at [gurobi.com](https://www.gurobi.com/academia/)
2. Install Gurobi:
   ```bash
   pip install gurobipy
   ```
3. Activate your license following Gurobi's instructions

## Verification

Verify your installation by running:

```python
import flixopt
print(flixopt.__version__)
```

## Logging Configuration

flixOpt uses Python's standard logging module with optional colored output via [colorlog](https://github.com/borntyping/python-colorlog). Logging is silent by default but can be easily configured:

```python
from flixopt import CONFIG

# Enable colored console logging
CONFIG.Logging.enable_console('INFO')

# Or use a preset configuration for exploring
CONFIG.exploring()
```

Since flixOpt uses Python's standard logging, you can also configure it directly:

```python
import logging

# Get the flixopt logger and configure it
logger = logging.getLogger('flixopt')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())
```

For more details on logging configuration, see the [`CONFIG.Logging`][flixopt.config.CONFIG.Logging] documentation.

## Next Steps

- Follow the [Quick Start](quick-start.md) guide
- Explore the [Minimal Example](../notebooks/01-quickstart.ipynb)
- Read about [Core Concepts](../user-guide/core-concepts.md)
