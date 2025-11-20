# Installation

This guide covers installing flixOpt and its dependencies.

## Requirements

- Python 3.9 or higher
- pip package manager

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
pip install -e ".[full,dev]"
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

### CPLEX (Optional)

IBM CPLEX is available with academic licenses:

1. Download from [IBM Academic Initiative](https://www.ibm.com/academic/)
2. Install following IBM's instructions
3. Install the Python API

### GLPK (Optional)

The GNU Linear Programming Kit can be installed via:

```bash
pip install pyglpk
```

## Verification

Verify your installation by running:

```python
import flixopt
print(flixopt.__version__)
```

## Logging Configuration

flixOpt uses [loguru](https://loguru.readthedocs.io/) for logging. Logging is silent by default but can be easily configured:

```python
from flixopt import CONFIG

# Enable console logging
CONFIG.Logging.console = True
CONFIG.Logging.level = 'INFO'
CONFIG.apply()

# Or use a preset configuration for exploring
CONFIG.exploring()
```

For more details on logging configuration, see the [`CONFIG.Logging`][flixopt.config.CONFIG.Logging] documentation.

## Next Steps

- Follow the [Quick Start](quick-start.md) guide
- Explore the [Minimal Example](../examples/00-Minimal Example.md)
- Read about [Core Concepts](../user-guide/core-concepts.md)
