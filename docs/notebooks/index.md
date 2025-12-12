# Examples

Learn flixopt through practical examples organized by topic. Each notebook includes a real-world user story and progressively builds your understanding.

## Getting Started

| Notebook | Description |
|----------|-------------|
| [01-Quickstart](01-quickstart.ipynb) | Minimal working example - heat a workshop with a gas boiler |
| [02-Heat System](02-heat-system.ipynb) | District heating with thermal storage and time-varying prices |

## Investment & Planning

| Notebook | Description |
|----------|-------------|
| [03-Investment Optimization](03-investment-optimization.ipynb) | Size a solar heating system - let the optimizer decide equipment sizes |
| [04-Operational Constraints](04-operational-constraints.ipynb) | Industrial boiler with startup costs, minimum uptime, and load constraints |

## Advanced Modeling

| Notebook | Description |
|----------|-------------|
| [05-Multi-Carrier Systems](05-multi-carrier-system.ipynb) | Hospital with CHP producing both electricity and heat |
| [06-Piecewise Efficiency](06-piecewise-efficiency.ipynb) | Heat pump with temperature-dependent COP and part-load curves |

## Scenarios & Scaling

| Notebook | Description |
|----------|-------------|
| [07-Scenarios and Periods](07-scenarios-and-periods.ipynb) | Multi-year planning with uncertain demand scenarios |
| [08-Large-Scale Optimization](08-large-scale-optimization.ipynb) | Speed up large problems with resampling and two-stage optimization |

## Results & Visualization

| Notebook | Description |
|----------|-------------|
| [09-Plotting and Data Access](09-plotting-and-data-access.ipynb) | Access optimization results and create visualizations |

## Key Concepts by Notebook

| Concept | Introduced In |
|---------|---------------|
| `FlowSystem`, `Bus`, `Flow` | 01-Quickstart |
| `Storage`, time-varying prices | 02-Heat System |
| `InvestParameters`, optimal sizing | 03-Investment |
| `StatusParameters`, startup costs | 04-Operational |
| Multi-carrier, CHP | 05-Multi-Carrier |
| `Piecewise`, variable efficiency | 06-Piecewise |
| Periods, scenarios, weights | 07-Scenarios |
| `transform.resample()`, `fix_sizes()` | 08-Large-Scale |
| `statistics`, `topology`, plotting | 09-Plotting |
