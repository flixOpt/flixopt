# Model Building Performance: Scaling Analysis

Comparing `main` branch vs `feature/element-data-classes` (batched approach).

## Executive Summary

The batched approach provides **7-32x speedup** depending on model size, with the benefit growing as models get larger.

| Dimension | Speedup Range | Key Insight |
|-----------|---------------|-------------|
| Converters | 3.6x → 24x | Speedup grows linearly with count |
| Effects | 7x → 32x | Speedup grows dramatically with effect count |
| Periods | 10x → 12x | Consistent across period counts |
| Timesteps | 8x → 12x | Consistent across time horizons |
| Storages | 9x → 19x | Speedup grows with count |

## Scaling by Number of Converters

Base config: 720 timesteps, 1 period, 2 effects, 5 storages

| Converters | Main (ms) | Main Vars | Feature (ms) | Feature Vars | Speedup |
|------------|-----------|-----------|--------------|--------------|---------|
| 10 | 1,189 | 168 | 322 | 15 | **3.6x** |
| 20 | 2,305 | 248 | 329 | 15 | **7.0x** |
| 50 | 3,196 | 488 | 351 | 15 | **9.1x** |
| 100 | 6,230 | 888 | 479 | 15 | **13.0x** |
| 200 | 12,806 | 1,688 | 533 | 15 | **24.0x** |

**Key finding:** Main branch scales O(n) with converters (168→1688 vars), while feature branch stays constant (15 vars). Build time on main grows ~11x for 20x more converters, while feature grows only ~1.7x.

## Scaling by Number of Effects

Base config: 720 timesteps, 1 period, 50 converters (102 flows), **each flow contributes to ALL effects**

| Effects | Main (ms) | Feature (ms) | Speedup |
|---------|-----------|--------------|---------|
| 1 | 2,912 | 399 | **7.2x** |
| 2 | 3,785 | 269 | **14.0x** |
| 5 | 8,335 | 327 | **25.4x** |
| 10 | 12,533 | 454 | **27.6x** |
| 15 | 15,892 | 583 | **27.2x** |
| 20 | 21,708 | 678 | **32.0x** |

**Key finding:** Feature branch scales **dramatically better** with effects:
- Main: 2,912 → 21,708ms (**7.5x growth** for 20x effects)
- Feature: 399 → 678ms (**1.7x growth** for 20x effects)

The speedup grows from **7x to 32x** as effects increase. The batched approach handles effect share constraints in O(1) instead of O(n_effects × n_flows).

## Scaling by Number of Periods

Base config: 720 timesteps, 2 effects, 50 converters, 5 storages

| Periods | Main (ms) | Feature (ms) | Speedup |
|---------|-----------|--------------|---------|
| 1 | 4,215 | 358 | **11.7x** |
| 2 | 6,179 | 506 | **12.2x** |
| 5 | 5,233 | 507 | **10.3x** |
| 10 | 5,749 | 487 | **11.8x** |

**Key finding:** Speedup remains consistent (~10-12x) across different period counts. Both branches handle multi-period efficiently.

## Scaling by Number of Timesteps

Base config: 1 period, 2 effects, 50 converters, 5 storages

| Timesteps | Main (ms) | Feature (ms) | Speedup |
|-----------|-----------|--------------|---------|
| 168 (1 week) | 3,118 | 347 | **8.9x** |
| 720 (1 month) | 3,101 | 371 | **8.3x** |
| 2000 (~3 months) | 4,679 | 394 | **11.8x** |

**Key finding:** Build time is relatively insensitive to timestep count for both branches. The constraint matrices scale with timesteps, but variable/constraint creation overhead dominates.

## Scaling by Number of Storages

Base config: 720 timesteps, 1 period, 2 effects, 50 converters

| Storages | Main (ms) | Main Vars | Feature (ms) | Feature Vars | Speedup |
|----------|-----------|-----------|--------------|--------------|---------|
| 0 | 2,909 | 418 | 222 | 9 | **13.1x** |
| 5 | 3,221 | 488 | 372 | 15 | **8.6x** |
| 10 | 3,738 | 558 | 378 | 15 | **9.8x** |
| 20 | 4,933 | 698 | 389 | 15 | **12.6x** |
| 50 | 8,117 | 1,118 | 420 | 15 | **19.3x** |

**Key finding:** Similar pattern to converters - main scales O(n) with storages while feature stays constant.

## Why the Batched Approach is Faster

### Old Approach (Main Branch)
- Creates one linopy Variable per flow/storage element
- Each variable creation has ~1ms overhead
- 200 converters × 2 flows = 400 variables = 400ms just for creation
- Constraints created per-element in loops

### New Approach (Feature Branch)
- Creates one batched Variable with element dimension
- Single variable creation regardless of element count
- `flow|rate` variable contains ALL flows in one DataArray
- Constraints use vectorized xarray operations with masks

### Variable Count Comparison

| Model Size | Main Vars | Feature Vars | Reduction |
|------------|-----------|--------------|-----------|
| 10 converters | 168 | 15 | 11x |
| 50 converters | 488 | 15 | 33x |
| 200 converters | 1,688 | 15 | 113x |

## Recommendations

1. **For large models (>50 converters):** Expect 10-25x speedup
2. **For multi-period models:** Expect consistent ~10-12x speedup
3. **For many effects:** Speedup grows dramatically (7x → 32x for 1→20 effects)
4. **Variable count is constant:** Model introspection tools may need updates

---
*Benchmark run on feature/element-data-classes branch*
*Base comparison: main branch*
