# Benchmarks

Performance benchmarking suite for flixopt.

## Quick Start

```bash
just quick myrun                              # Run benchmarks → JSON
just table benchmarks/results/myrun_*.json    # View as table
```

## Workflow

### Single Run

```bash
# Quick smoke test (~2 min)
just quick myrun

# Full benchmark (~15-20 min)
just all myrun

# Specific models/phases
just run myrun --model simple district --phase build memory --sizes 24 168 720
```

### Compare Commits (sweep)

```bash
# Preview
just sweep-dry HEAD~5..HEAD --all --quick

# Run sweep
just sweep HEAD~5..HEAD --all --quick

# View results
just table ~/.cache/flixopt-benchmarks/*.json
```

## Commands

| Command | Description |
|---------|-------------|
| `just quick <name>` | Quick smoke test (all models, small sizes) |
| `just all <name>` | Full benchmark (all models, all sizes) |
| `just run <name> [args]` | Run specific models/phases/sizes |
| `just sweep <revs> [args]` | Benchmark across git commits |
| `just sweep-dry <revs> [args]` | Preview sweep without running |
| `just table <files>` | Show results as markdown table |
| `just csv <files>` | Show results as CSV |
| `just list` | Show available models and phases |
| `just latest` | Show recent result files |
| `just clean` | Remove results |

## Timing Estimates

| Command | Est. Time |
|---------|-----------|
| `just quick name` | ~2 min |
| `just all name` | ~15-20 min |
| `just sweep HEAD~5..HEAD --all --quick` | ~2 min/commit |

## Models

| Model | Description |
|-------|-------------|
| `simple` | Simple Heat System (1 boiler + 1 storage) |
| `district` | District Heating (CHP + boiler + heat pump + storage) |
| `synthetic_xl` | Synthetic XL (~50 converters × 10 buses) |

## Phases

| Phase | Measures |
|-------|----------|
| `build` | Time for `fs.build_model()` |
| `io` | Time for `to_dataset()` + `from_dataset()` |
| `lp_write` | Time for LP file writing |
| `memory` | Peak memory during model build |

## Results

- **JSON files**: `benchmarks/results/<name>_<model>_<phase>.json`
- **Sweep cache**: `~/.cache/flixopt-benchmarks/` (survives git checkout)

## Adding Models

Create `benchmarks/models/<name>.py`:

```python
LABEL = 'My Model'
SIZES = [24, 168, 720, 2190, 8760]
QUICK_SIZES = [24, 168]

def build(size: int) -> fx.FlowSystem:
    ...
```

## Adding Phases

Create `benchmarks/runners/<name>.py`:

```python
def run(model_module, size: int, iterations: int) -> dict:
    ...
    return {
        'median': float,
        'mean': float,
        'std': float,
        'min': float,
        'max': float,
        'q25': float,
        'q75': float,
        'times': list[float],
        'unit': str,  # optional, e.g., 'MB'
    }
```
