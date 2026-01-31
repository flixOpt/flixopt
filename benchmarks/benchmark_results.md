# Benchmark Results: Model Build Performance

Benchmarked `build_model()` across commits since `302413c4` on branch `feature/element-data-classes`.

**Date:** 2026-01-31

## XL System (2000h, 300 converters, 50 storages)

| Commit | Description | Build (ms) | Speedup vs base |
|--------|-------------|------------|-----------------|
| `302413c4` | Base | **10,711** | 1.00x |
| `7dd56dde` | Summary of changes | **13,620** | 0.79x (regression) |
| `f38f828f` | sparse groupby in conversion | **7,101** | 1.51x |
| `2a94130f` | sparse groupby in piecewise_conversion | **4,428** | 2.42x |
| `805bcc56` | xr.concat → numpy pre-alloc | **3,468** | 3.09x |
| `82e69989` | fix build_effects_array signature | **3,733** | 2.87x |
| `9c2d3d3b` | Add sparse_weighted_sum | **3,317** | 3.23x |

## Complex System (72h, piecewise)

| Commit | Description | Build (ms) | Speedup vs base |
|--------|-------------|------------|-----------------|
| `302413c4` | Base | **678** | 1.00x |
| `7dd56dde` | Summary of changes | **1,156** | 0.59x (regression) |
| `f38f828f` | sparse groupby in conversion | **925** | 0.73x |
| `2a94130f` | sparse groupby in piecewise_conversion | **801** | 0.85x |
| `805bcc56` | xr.concat → numpy pre-alloc | **1,194** | 0.57x |
| `82e69989` | fix build_effects_array signature | **915** | 0.74x |
| `9c2d3d3b` | Add sparse_weighted_sum | **947** | 0.72x |

## Key Takeaways

- **XL system: 3.23x overall speedup** — from 10.7s down to 3.3s. The biggest gains came from sparse groupby in piecewise conversion (`2a94130f`, jumping from 1.51x to 2.42x) and numpy pre-allocation replacing `xr.concat` (`805bcc56`, reaching 3.09x).

- **Complex system: regression overall** — went from 678ms to 947ms (0.72x). The Complex system is small enough that the overhead of the new sparse/numpy approaches may outweigh their benefits. The initial commit `7dd56dde` introduced a significant regression that was never fully recovered.

## How to Run Benchmarks Across Commits

To benchmark `build_model()` across a range of commits, use the following approach:

```bash
# 1. Stash any uncommitted changes
git stash --include-untracked

# 2. Loop over commits and run the benchmark at each one
for SHA in 302413c4 7dd56dde f38f828f 2a94130f 805bcc56 82e69989 9c2d3d3b; do
    echo "=== $SHA ==="
    git checkout "$SHA" --force 2>/dev/null
    python benchmarks/benchmark_model_build.py --system complex --iterations 3
done

# 3. Restore your branch and stash
git checkout feature/element-data-classes --force
git stash pop
```

To run specific system types:

```bash
# Single system
python benchmarks/benchmark_model_build.py --system complex
python benchmarks/benchmark_model_build.py --system synthetic --converters 300 --timesteps 2000

# All systems
python benchmarks/benchmark_model_build.py --all

# Custom iterations
python benchmarks/benchmark_model_build.py --all --iterations 5
```

Available `--system` options: `complex`, `district`, `multiperiod`, `seasonal`, `synthetic`.
For `synthetic`, use `--converters`, `--timesteps`, and `--periods` to configure the system size.
