# Benchmark Results: Model Build Performance

Benchmarked `build_model()` and LP file write across commits on branch `feature/element-data-classes`, starting from the main branch divergence point.

**Date:** 2026-01-31

## XL System (2000h, 300 converters, 50 storages)

| Commit | Description | Build (ms) | Build speedup | Write LP (ms) | Write speedup |
|--------|-------------|------------|---------------|---------------|---------------|
| `42f593e7` | **main branch (base)** | **113,360** | 1.00x | **44,815** | 1.00x |
| `302413c4` | Summary of changes | **7,718** | 14.69x | **15,369** | 2.92x |
| `7dd56dde` | Summary of changes | **9,572** | 11.84x | **15,780** | 2.84x |
| `f38f828f` | sparse groupby in conversion | **3,649** | 31.07x | **10,370** | 4.32x |
| `2a94130f` | sparse groupby in piecewise_conversion | **2,323** | 48.80x | **9,584** | 4.68x |
| `805bcc56` | xr.concat → numpy pre-alloc | **2,075** | 54.63x | **10,825** | 4.14x |
| `82e69989` | fix build_effects_array signature | **2,333** | 48.59x | **10,331** | 4.34x |
| `9c2d3d3b` | Add sparse_weighted_sum | **1,638** | 69.21x | **9,427** | 4.75x |
| `8277d5d3` | Add sparse_weighted_sum (2) | **2,785** | 40.70x | **9,129** | 4.91x |
| `c67a6a7e` | Clean up, revert piecewise | **2,616** | 43.33x | **9,574** | 4.68x |
| `52a581fe` | Improve piecewise | **1,743** | 65.04x | **9,763** | 4.59x |
| `8c8eb5c9` | Pre-combine xarray coeffs in storage | **1,676** | 67.64x | **8,868** | 5.05x |

## Complex System (72h, piecewise)

| Commit | Description | Build (ms) | Build speedup | Write LP (ms) | Write speedup |
|--------|-------------|------------|---------------|---------------|---------------|
| `42f593e7` | **main branch (base)** | **1,003** | 1.00x | **417** | 1.00x |
| `302413c4` | Summary of changes | **533** | 1.88x | **129** | 3.23x |
| `7dd56dde` | Summary of changes | **430** | 2.33x | **103** | 4.05x |
| `f38f828f` | sparse groupby in conversion | **452** | 2.22x | **136** | 3.07x |
| `2a94130f` | sparse groupby in piecewise_conversion | **440** | 2.28x | **112** | 3.72x |
| `805bcc56` | xr.concat → numpy pre-alloc | **475** | 2.11x | **132** | 3.16x |
| `82e69989` | fix build_effects_array signature | **391** | 2.57x | **99** | 4.21x |
| `9c2d3d3b` | Add sparse_weighted_sum | **404** | 2.48x | **96** | 4.34x |
| `8277d5d3` | Add sparse_weighted_sum (2) | **416** | 2.41x | **98** | 4.26x |
| `c67a6a7e` | Clean up, revert piecewise | **453** | 2.21x | **108** | 3.86x |
| `52a581fe` | Improve piecewise | **426** | 2.35x | **105** | 3.97x |
| `8c8eb5c9` | Pre-combine xarray coeffs in storage | **383** | 2.62x | **100** | 4.17x |

LP file size: 528.28 MB (XL, branch) vs 503.88 MB (XL, main), 0.21 MB (Complex) — unchanged.

## Key Takeaways

- **XL system: 67.6x build speedup** — from 113.4s down to 1.7s. LP write improved 5.1x (44.8s → 8.9s). The bulk of the gain came from the initial refactoring (`302413c4`, 14.7x), with sparse groupby and weighted sum optimizations adding further large improvements.

- **Complex system: 2.62x build speedup** — from 1,003ms down to 383ms. LP write improved 4.2x (417ms → 100ms). Gains are more modest since this system is small (72 timesteps, 14 flows) and dominated by per-operation linopy/xarray overhead.

## How to Run Benchmarks Across Commits

To benchmark `build_model()` across a range of commits, use the following approach:

```bash
# 1. Stash any uncommitted changes
git stash --include-untracked

# 2. Loop over commits and run the benchmark at each one
for SHA in 302413c4 7dd56dde f38f828f 2a94130f 805bcc56 82e69989 9c2d3d3b 8277d5d3 c67a6a7e 52a581fe 8c8eb5c9; do
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
