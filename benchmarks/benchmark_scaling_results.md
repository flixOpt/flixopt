# Model Building Performance: Scaling Analysis

Comparing `main` branch vs `feature/element-data-classes` (batched approach).

## Executive Summary

The batched approach provides **6-24x speedup** depending on model size, with the benefit growing as models get larger.

| Dimension | Speedup Range | Key Insight |
|-----------|---------------|-------------|
| Converters | 3.6x → 24x | Speedup grows linearly with count |
| Effects | 14x → 23x | Speedup grows with effect count (many contributors) |
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

Base config: 720 timesteps, 1 period, 100 converters (200 contributing flows), 0 storages

| Effects | Main (ms) | Feature (ms) | Speedup | Main growth | Feature growth |
|---------|-----------|--------------|---------|-------------|----------------|
| 1 | 5,721 | 396 | **14.4x** | - | - |
| 2 | 6,094 | 463 | **13.1x** | 1.06x | 1.17x |
| 3 | 6,562 | 388 | **16.9x** | 1.08x | 0.84x |
| 5 | 9,081 | 428 | **21.2x** | 1.38x | 1.10x |
| 7 | 8,495 | 409 | **20.7x** | 0.94x | 0.96x |
| 10 | 11,256 | 483 | **23.3x** | 1.32x | 1.18x |

**Key finding:** With many contributing flows, feature branch scales **better** than main:
- Main: 5,721 → 11,256ms (1.97x growth for 10x effects)
- Feature: 396 → 483ms (1.22x growth for 10x effects)

The speedup actually **increases** from 14x to 23x as effects increase, because the batched approach handles effect share constraints more efficiently.

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
3. **For many effects with many contributors:** Speedup increases (14x → 23x)
4. **Variable count is constant:** Model introspection tools may need updates

---
*Benchmark run on feature/element-data-classes branch*
*Base comparison: main branch*
