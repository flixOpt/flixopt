# Examples

Learn flixopt through practical examples organized by topic. Each notebook includes a real-world user story and progressively builds your understanding.

## Basics

| Notebook | Description |
|----------|-------------|
| [01-Quickstart](01-quickstart.ipynb) | Minimal working example - heat a workshop with a gas boiler |
| [02-Heat System](02-heat-system.ipynb) | District heating with thermal storage and time-varying prices |

## Investment

| Notebook | Description |
|----------|-------------|
| [03-Sizing](03-investment-optimization.ipynb) | Size a solar heating system - let the optimizer decide equipment sizes |
| [04-Constraints](04-operational-constraints.ipynb) | Industrial boiler with startup costs, minimum uptime, and load constraints |

## Advanced

| Notebook | Description |
|----------|-------------|
| [05-Multi-Carrier](05-multi-carrier-system.ipynb) | Hospital with CHP producing both electricity and heat |
| [10-Transmission](10-transmission.ipynb) | Connect sites with pipelines or cables, including losses and bidirectional flow |

## Non-Linear Modeling

| Notebook | Description |
|----------|-------------|
| [06a-Time-Varying](06a-time-varying-parameters.ipynb) | Heat pump with temperature-dependent COP |
| [06b-Piecewise Conversion](06b-piecewise-conversion.ipynb) | Gas engine with load-dependent efficiency curves |
| [06c-Piecewise Effects](06c-piecewise-effects.ipynb) | Economies of scale in investment costs |

## Scaling

| Notebook | Description |
|----------|-------------|
| [07-Scenarios](07-scenarios-and-periods.ipynb) | Multi-year planning with uncertain demand scenarios |
| [08-Large-Scale](08-large-scale-optimization.ipynb) | Speed up large problems with resampling and two-stage optimization |

## Results

| Notebook | Description |
|----------|-------------|
| [09-Plotting](09-plotting-and-data-access.ipynb) | Access optimization results and create visualizations |

## Key Concepts

| Concept | Introduced In |
|---------|---------------|
| `FlowSystem`, `Bus`, `Flow` | Quickstart |
| `Storage`, time-varying prices | Heat System |
| `InvestParameters`, optimal sizing | Sizing |
| `StatusParameters`, startup costs | Constraints |
| Multi-carrier, CHP | Multi-Carrier |
| `Transmission`, losses, bidirectional | Transmission |
| Time-varying `conversion_factors` | Time-Varying Parameters |
| `PiecewiseConversion`, part-load efficiency | Piecewise Conversion |
| `PiecewiseEffects`, economies of scale | Piecewise Effects |
| Periods, scenarios, weights | Scenarios |
| `transform.resample()`, `fix_sizes()` | Large-Scale |
| `statistics`, `topology`, plotting | Plotting |
