# Benchmarks

Performance benchmarking suite for flixopt.

## Quick Start

```bash
# Run quick smoke test (~2 min)
just quick myrun

# View results
just plot benchmarks/results/myrun_*.json
```

## Workflow

### Single Run (current code)

```bash
# Quick smoke test - all models, small sizes, 3 iterations
just quick myrun

# Full benchmark - all models, all sizes, 10 iterations
just all myrun

# Specific models/phases
just run myrun --model simple district --phase build memory --sizes 24 168

# View results
just plot benchmarks/results/myrun_*.json
```

### Compare Commits (sweep)

```bash
# Preview what will run (no execution)
just sweep-dry HEAD~5..HEAD --all --quick

# Run sweep across commits
just sweep HEAD~5..HEAD --all --quick

# Results auto-plotted after sweep, or manually:
just plot ~/.cache/flixopt-benchmarks/*.json

# Generate markdown table
just table ~/.cache/flixopt-benchmarks/*.json
```

## Timing Estimates

| Command | Models | Sizes | Iterations | Est. Time |
|---------|--------|-------|------------|-----------|
| `just quick name` | 3 | 24, 168 | 3 | ~2 min |
| `just all name` | 3 | 24→8760 | 10 | ~15-20 min |
| `just run name --model simple --phase build` | 1 | 24→8760 | 10 | ~1-2 min |
| `just sweep HEAD~5..HEAD --all --quick` | 3 | 24, 168 | 5 | ~2 min/commit |

## Available Commands

```bash
just list          # Show available models and phases
just latest        # Show recent result files
just clean         # Remove results (keeps cache)
just clean-all     # Remove results and cache
```

## Models

| Model | Description | Components |
|-------|-------------|------------|
| `simple` | Simple Heat System | 1 boiler + 1 storage |
| `district` | District Heating | CHP + boiler + heat pump + storage |
| `synthetic_xl` | Synthetic XL (~50 converters) | 50 converters × 10 buses |

## Phases (Runners)

| Phase | Measures |
|-------|----------|
| `build` | Time for `fs.build_model()` |
| `io` | Time for `to_dataset()` + `from_dataset()` round-trip |
| `lp_write` | Time for LP file writing |
| `memory` | Peak memory during model build |

## Plot Layout

The unified plot shows:
- **Rows**: Phases (build, io, lp_write, memory)
- **Columns**: Models (Simple, District, Synthetic XL)
- **X-axis**: Timesteps (log scale)
- **Y-axis**: Value (ms or MB)
- **Color**: Runs/commits (light → dark blue for old → new)

## Results Storage

| Location | Purpose |
|----------|---------|
| `benchmarks/results/` | Normal runs, plots, tables |
| `~/.cache/flixopt-benchmarks/` | Sweep results (survives git checkout) |

## Adding New Models

Create `benchmarks/models/<name>.py`:

```python
LABEL = 'My Model'
SIZES = [24, 168, 720, 2190, 8760]
QUICK_SIZES = [24, 168]

def build(size: int) -> fx.FlowSystem:
    """Build a FlowSystem with *size* timesteps."""
    ...
```

## Adding New Phases

Create `benchmarks/runners/<name>.py`:

```python
def run(model_module, size: int, iterations: int) -> dict:
    """Run benchmark and return results."""
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
        'unit': str,  # optional, e.g., 'MB' for memory
    }
```
