# Design Document: Cluster Architecture for flixopt

## Executive Summary

This document defines the architecture for cluster representation in flixopt using **true `(cluster, time)` dimensions**.

### Key Decision: True Dimensions (Option B)

**Chosen Approach:**
```python
# Clustered data structure:
data.dims = ('cluster', 'time', 'period', 'scenario')
data.shape = (9, 24, ...)  # 9 clusters × 24 timesteps each
```

**Why True Dimensions?**
1. **Temporal constraints just work** - `x[:, 1:] - x[:, :-1]` naturally stays within clusters
2. **No boundary masking** - StorageModel, StatusModel constraints are clean and vectorized
3. **Variable segment durations supported** - `timestep_duration[cluster, time]` handles different segment lengths
4. **Plotting trivial** - existing `facet_col='cluster'` works automatically

### Document Scope

1. Current architecture analysis (Part 1)
2. Architectural options and recommendation (Part 2)
3. Impact on Features - StatusModel, StorageModel, etc. (Part 3)
4. Plotting improvements (Part 4)
5. Variable segment durations (Part 5)
6. Implementation roadmap (Part 6)

---

## Part 1: Current Architecture Analysis

### 1.1 Time Dimension Structure

**Current Implementation:**
```
time: (n_clusters × timesteps_per_cluster,)  # Flat, e.g., (864,) for 9 clusters × 96 timesteps
```

**Key Properties:**
- `cluster_weight`: Shape `(time,)` with repeated values per cluster
- `timestep_duration`: Shape `(time,)` or scalar
- `aggregation_weight = timestep_duration × cluster_weight`

**Cluster Tracking:**
- `cluster_start_positions`: Array of indices where each cluster begins
- `ClusterStructure`: Stores cluster_order, occurrences, n_clusters, timesteps_per_cluster

### 1.2 Features Affected by Time Structure

| Feature | Time Usage | Clustering Impact |
|---------|-----------|-------------------|
| **StatusModel** | `aggregation_weight` for active hours, `timestep_duration` for effects | Must sum correctly across clusters |
| **InvestmentModel** | Periodic (no time dim) | Unaffected by time structure |
| **PiecewiseModel** | Per-timestep lambda variables | Must preserve cluster structure |
| **ShareAllocationModel** | Uses `cluster_weight` explicitly | Directly depends on weight structure |
| **StorageModel** | Charge balance across time | Needs cluster boundary handling |
| **InterclusterStorageModel** | SOC_boundary linking | Uses cluster indices extensively |

### 1.3 Current Plotting Structure

**StatisticsPlotAccessor Methods:**
- `balance()`: Node flow visualization
- `storage()`: Dual-axis charge/discharge + SOC
- `heatmap()`: 2D time reshaping (days × hours)
- `duration_curve()`: Sorted load profiles
- `effects()`: Cost/emission breakdown

**Clustering in Plots:**
- `ClusterStructure.plot()`: Shows cluster assignments
- Cluster weight applied when aggregating (`cluster_weight.sum('time')`)
- No visual separation between clusters in time series plots

---

## Part 2: Architectural Options

### 2.1 Option A: Enhanced Flat with Indexers

Keep flat `time` dimension, add xarray indexer properties.

**Pros:** Supports variable-length clusters
**Cons:** Every temporal constraint needs explicit boundary masking

**NOT RECOMMENDED** - see Option B.

### 2.2 Option B: True (cluster, time) Dimensions (RECOMMENDED)

Reshape time to 2D when clustering is active:

```python
# ═══════════════════════════════════════════════════════════════
# DIMENSION STRUCTURE
# ═══════════════════════════════════════════════════════════════

# Non-clustered:
data.dims = ('time', 'period', 'scenario')
data.shape = (8760, ...)  # Full year hourly

# Clustered:
data.dims = ('cluster', 'time', 'period', 'scenario')
data.shape = (9, 24, ...)  # 9 clusters × 24 timesteps each

# Varying segment durations supported:
timestep_duration.dims = ('cluster', 'time')
timestep_duration.shape = (9, 24)  # Different durations per segment per cluster
```

**Key Benefits - Temporal Constraints Just Work!**

```python
# ═══════════════════════════════════════════════════════════════
# STORAGE: Charge balance naturally within clusters
# ═══════════════════════════════════════════════════════════════
# charge_state shape: (cluster, time+1, period, scenario) - extra timestep for boundaries
charge_state = ...  # (9, 25, ...)

# Balance constraint - NO MASKING NEEDED!
lhs = charge_state[:, 1:] - charge_state[:, :-1] * (1 - loss) - charge + discharge
# Shape: (cluster, time, period, scenario) = (9, 24, ...)

# Delta per cluster (for inter-cluster linking):
delta_soc = charge_state[:, -1] - charge_state[:, 0]  # Shape: (cluster, ...)

# ═══════════════════════════════════════════════════════════════
# STATUS: Uptime/downtime constraints stay within clusters
# ═══════════════════════════════════════════════════════════════
status = ...  # (cluster, time, ...)

# State transitions - naturally per cluster!
activate = status[:, 1:] - status[:, :-1]  # No boundary issues!

# min_uptime constraint - works correctly, can't span clusters

# ═══════════════════════════════════════════════════════════════
# INTER-CLUSTER OPERATIONS
# ═══════════════════════════════════════════════════════════════
# Select first/last timestep of each cluster:
at_start = data.isel(time=0)   # Shape: (cluster, period, scenario)
at_end = data.isel(time=-1)    # Shape: (cluster, period, scenario)

# Compute per-cluster statistics:
mean_per_cluster = data.mean(dim='time')
max_per_cluster = data.max(dim='time')
```

**Varying Segment Durations (Future Segmentation):**

```python
# Same NUMBER of segments per cluster, different DURATIONS:
timestep_duration = xr.DataArray(
    [
        [2, 2, 1, 1, 2, 4],  # Cluster 0: segments sum to 12h
        [1, 3, 2, 2, 2, 2],  # Cluster 1: segments sum to 12h
        ...
    ],
    dims=['cluster', 'time'],
    coords={'cluster': range(9), 'time': range(6)}
)

# aggregation_weight still works:
aggregation_weight = timestep_duration * cluster_weight  # (cluster, time) * (cluster,)
```

**Pros:**
- Temporal constraints naturally stay within clusters - NO MASKING!
- StatusModel uptime/downtime just works
- Storage balance is clean
- Much less code, fewer bugs
- Supports varying segment durations (same count, different lengths)

**Cons:**
- More upfront refactoring
- All code paths need to handle `(cluster, time)` vs `(time,)` based on `is_clustered`

### 2.3 Recommendation: Option B (True Dimensions)

Given:
- Uniform timestep COUNT per cluster (tsam default)
- Variable segment DURATIONS supported via `timestep_duration[cluster, time]`
- Much cleaner constraint handling

**Option B is the recommended choice.**

---

## Part 3: Impact on Features

### 3.1 StatusModel Impact - SOLVED BY TRUE DIMENSIONS

**With `(cluster, time)` dimensions, temporal constraints naturally stay within clusters!**

```python
status.dims = ('cluster', 'time', 'period', 'scenario')
status.shape = (9, 24, ...)

# State transitions - per cluster, no boundary issues!
activate = status[:, 1:] - status[:, :-1]

# min_uptime constraint operates within each cluster's time dimension
# Cannot accidentally span cluster boundaries
```

**What works automatically:**
- ✅ `min_uptime`, `min_downtime` - constraints stay within clusters
- ✅ `initial_status` - applies to each cluster's first timestep
- ✅ State transitions - naturally per cluster
- ✅ `active_hours` - uses `aggregation_weight` correctly
- ✅ `effects_per_startup` - counted per cluster, weighted by `cluster_weight`

**Optional Enhancement - cluster_mode for special cases:**
```python
class StatusParameters:
    # NEW: How to handle cluster boundaries (default: independent)
    cluster_mode: Literal['independent', 'cyclic'] = 'independent'
```

| Mode | Behavior |
|------|----------|
| `independent` | Each cluster starts fresh (default, most common) |
| `cyclic` | `status[:, 0] == status[:, -1]` - status returns to start |

### 3.2 StorageModel Impact - SIMPLIFIED

**With `(cluster, time)` dimensions, storage constraints become trivial:**

```python
charge_state.dims = ('cluster', 'time_extra', 'period', 'scenario')
charge_state.shape = (9, 25, ...)  # 24 timesteps + 1 boundary per cluster

# ═══════════════════════════════════════════════════════════════
# Charge balance - NO MASKING!
# ═══════════════════════════════════════════════════════════════
lhs = (
    charge_state[:, 1:] -
    charge_state[:, :-1] * (1 - loss_rate) -
    charge * eta_charge +
    discharge / eta_discharge
)
self.add_constraints(lhs == 0, name='charge_balance')  # Clean!

# ═══════════════════════════════════════════════════════════════
# Delta SOC per cluster (for inter-cluster linking)
# ═══════════════════════════════════════════════════════════════
delta_soc = charge_state[:, -1] - charge_state[:, 0]  # Shape: (cluster, ...)

# ═══════════════════════════════════════════════════════════════
# Cluster start constraint (relative SOC starts at 0)
# ═══════════════════════════════════════════════════════════════
self.add_constraints(charge_state[:, 0] == 0, name='cluster_start')

# ═══════════════════════════════════════════════════════════════
# Cyclic constraint (optional)
# ═══════════════════════════════════════════════════════════════
self.add_constraints(charge_state[:, 0] == charge_state[:, -1], name='cyclic')
```

**InterclusterStorageModel also simplified** - SOC_boundary linking uses clean slicing.

### 3.3 ShareAllocationModel Impact

**Current Code (features.py:624):**
```python
self._eq_total.lhs -= (self.total_per_timestep * self._model.cluster_weight).sum(dim='time')
```

**With Enhanced Helpers:**
No changes needed - `cluster_weight` structure preserved.

### 3.4 PiecewiseModel Impact

**Current Code:** Creates lambda variables per timestep.

**With Enhanced Helpers:**
No changes needed - operates on flat time dimension.

### 3.5 Summary: Models with True (cluster, time) Dimensions

| Model | Cross-Timestep Constraints | With True Dims | Action Needed |
|-------|---------------------------|----------------|---------------|
| **StorageModel** | `cs[t] - cs[t-1]` | ✅ Just works | Simplify code |
| **StatusModel** | min_uptime, min_downtime | ✅ Just works | Optional cluster_mode |
| **consecutive_duration_tracking** | State machine | ✅ Just works | No changes |
| **state_transition_bounds** | `activate[t] - status[t-1]` | ✅ Just works | No changes |
| **PiecewiseModel** | Per-timestep only | ✅ Just works | No changes |
| **ShareAllocationModel** | Sum with cluster_weight | ✅ Just works | No changes |
| **InvestmentModel** | No time dimension | ✅ Just works | No changes |

**Key Insight:** With true `(cluster, time)` dimensions, `x[:, 1:] - x[:, :-1]` naturally stays within clusters!

---

## Part 4: Plotting Improvements

### 4.1 Key Benefit of True Dimensions: Minimal Plotting Changes

With true `(cluster, time)` dimensions, plotting becomes trivial because:
1. Data already has the right shape - no reshaping needed
2. Existing `facet_col='cluster'` parameter just works
3. Only minimal changes needed: auto-add cluster separators in combined views

### 4.2 Proposed Approach: Leverage Existing Infrastructure

#### 4.2.1 Use Existing facet_col Parameter

**No new plot methods needed!** The existing infrastructure handles `cluster` dimension:

```python
# ═══════════════════════════════════════════════════════════════
# EXISTING API - works automatically with (cluster, time) dims!
# ═══════════════════════════════════════════════════════════════
fs.statistics.plot.storage('Battery', facet_col='cluster')  # One subplot per cluster
fs.statistics.plot.balance('Heat', facet_col='cluster')     # One subplot per cluster
fs.statistics.plot.flows(..., facet_col='cluster')          # Same pattern

# Combine with other dimensions
fs.statistics.plot.balance('Heat', facet_col='cluster', facet_row='scenario')
```

#### 4.2.2 Auto-Add Cluster Separators (Small Change)

For combined views (no faceting), add visual separators:

```python
def _create_base_plot(self, data, **kwargs):
    """Base plot creation - add cluster separators if combined view."""
    fig = ...  # existing logic

    # Auto-add cluster separators if clustered and showing combined time
    if self._fs.is_clustered and 'cluster' not in kwargs.get('facet_col', ''):
        # Add subtle vertical lines between clusters
        for cluster_idx in range(1, self._fs.clustering.n_clusters):
            x_pos = cluster_idx * self._fs.clustering.timesteps_per_cluster
            fig.add_vline(x=x_pos, line_dash='dot', opacity=0.3, line_color='gray')

    return fig
```

#### 4.2.3 Per-Cluster Statistics (Natural with True Dims)

With `(cluster, time)` dimensions, aggregation is trivial:

```python
# Mean per cluster - just use xarray
mean_per_cluster = data.mean(dim='time')  # Shape: (cluster, ...)
max_per_cluster = data.max(dim='time')

# Can plot directly
fs.statistics.plot.bar(data.mean('time'), x='cluster', title='Mean by Cluster')
```

#### 4.2.4 Heatmap (Already Correct Shape)

With true dimensions, heatmaps work directly:

```python
# Data already has (cluster, time) shape - heatmap just works!
def cluster_heatmap(self, variable):
    data = self._get_variable(variable)

    # With (cluster, time) dims, no reshaping needed!
    return self._plot_heatmap(
        data,  # Already (cluster, time, ...)
        x='time',
        y='cluster',
        colorbar_title=variable
    )
```

### 4.3 Summary: Plotting Changes Required

| Change | Scope | Complexity |
|--------|-------|------------|
| Auto cluster separators in base plot | ~10 lines in `_create_base_plot` | Low |
| Ensure facet_col='cluster' works | Should work already | None |
| Heatmap with cluster dim | Works automatically | None |
| No new plot methods needed | - | - |

---

## Part 5: Variable Segment Durations (Future)

### 5.1 Clarification: Variable Durations, NOT Variable Counts

With true `(cluster, time)` dimensions:
- **Same number** of timesteps per cluster (required for rectangular array)
- **Different durations** per timestep within each cluster (via `timestep_duration`)

This is exactly what tsam segmentation provides and what we need.

### 5.2 TSAM Segmentation Features

TSAM supports intra-period segmentation:
```python
tsam.TimeSeriesAggregation(
    segmentation=True,       # Enable subdivision
    noSegments=6,            # Segments per typical period
    segmentRepresentationMethod='meanRepresentation'
)
```

**What TSAM provides:**
- Uniform segment count across all typical periods ✅
- Various representation methods (mean, medoid, distribution)
- Different segment durations per cluster ✅

### 5.3 Implementation with True Dimensions

With `(cluster, time)` dimensions, variable segment durations are trivial:

```python
# ═══════════════════════════════════════════════════════════════
# DIMENSION STRUCTURE
# ═══════════════════════════════════════════════════════════════
data.dims = ('cluster', 'time', 'period', 'scenario')
data.shape = (9, 6, ...)  # 9 clusters × 6 segments each

# ═══════════════════════════════════════════════════════════════
# VARIABLE SEGMENT DURATIONS - just a 2D array!
# ═══════════════════════════════════════════════════════════════
timestep_duration = xr.DataArray(
    [
        [2, 2, 4, 4, 6, 6],  # Cluster 0: short-short-long pattern
        [1, 1, 4, 8, 4, 6],  # Cluster 1: different pattern
        [3, 3, 3, 3, 6, 6],  # Cluster 2: uniform start, longer end
        ...
    ],
    dims=['cluster', 'time'],
    coords={'cluster': range(9), 'time': range(6)}
)

# ═══════════════════════════════════════════════════════════════
# AGGREGATION WEIGHT - combines duration and cluster weight
# ═══════════════════════════════════════════════════════════════
# cluster_weight shape: (cluster,) - how many days each cluster represents
# timestep_duration shape: (cluster, time) - duration of each segment
aggregation_weight = timestep_duration * cluster_weight  # Broadcasting!

# ═══════════════════════════════════════════════════════════════
# ALL EXISTING CONSTRAINTS JUST WORK
# ═══════════════════════════════════════════════════════════════
# Storage balance: uses aggregation_weight correctly
# StatusModel: active_hours weighted by aggregation_weight
# Cost calculations: weighted by aggregation_weight
```

### 5.4 No Complex Infrastructure Needed

With true dimensions, segmentation requires **no special infrastructure**:

| Aspect | With True Dims |
|--------|----------------|
| Different segment durations | Just set `timestep_duration[cluster, time]` |
| Constraint generation | No changes - already works |
| Cost calculations | No changes - uses `aggregation_weight` |
| Plotting | No changes - `cluster` dim exists |

**The only addition needed:** Update `transform_accessor.cluster()` to accept tsam segmentation parameters and construct the 2D `timestep_duration` array.

---

## Part 6: Implementation Roadmap

### Phase 1: Core Dimension Refactoring (PRIORITY)

**Goal:** Introduce true `(cluster, time)` dimensions throughout the codebase.

**Tasks:**
1. Update `FlowSystem` to support `(cluster, time)` dimension structure when clustered
2. Add `is_clustered` property to `FlowSystem`
3. Update `Clustering` class with:
   - `n_clusters: int` property
   - `timesteps_per_cluster: int` property
   - Coordinate accessors for cluster dimension
4. Update `cluster_weight` to have shape `(cluster,)` instead of `(time,)`
5. Update `timestep_duration` to have shape `(cluster, time)` when clustered
6. Update `aggregation_weight` computation to broadcast correctly

**Files:**
- `flixopt/flow_system.py` - Core dimension handling
- `flixopt/clustering/base.py` - Updated Clustering class

**Key Changes:**
```python
# FlowSystem property updates:
@property
def is_clustered(self) -> bool:
    return self.clustering is not None

@property
def cluster_weight(self) -> xr.DataArray:
    if not self.is_clustered:
        return xr.DataArray(1.0)
    # Shape: (cluster,) - one weight per cluster
    return xr.DataArray(
        self.clustering.cluster_occurrences,
        dims=['cluster'],
        coords={'cluster': range(self.clustering.n_clusters)}
    )

@property
def timestep_duration(self) -> xr.DataArray:
    if not self.is_clustered:
        return self._timestep_duration  # Shape: (time,) or scalar
    # Shape: (cluster, time) when clustered
    return self._timestep_duration  # Already 2D from clustering

@property
def aggregation_weight(self) -> xr.DataArray:
    return self.timestep_duration * self.cluster_weight  # Broadcasting handles shapes
```

### Phase 2: Update Variable/Constraint Creation

**Goal:** All variables and constraints use `(cluster, time)` dimensions when clustered.

**Tasks:**
1. Update `create_variable` to use `(cluster, time, period, scenario)` dims when clustered
2. Update constraint generation in all models
3. Verify linopy handles multi-dimensional constraint arrays correctly
4. Add tests for both clustered and non-clustered paths

**Files:**
- `flixopt/core.py` - Variable creation
- `flixopt/components.py` - StorageModel, other component models
- `flixopt/features.py` - StatusModel, other feature models

**Key Pattern:**
```python
# Dimension-aware variable creation:
def _get_time_dims(self) -> list[str]:
    if self.flow_system.is_clustered:
        return ['cluster', 'time']
    return ['time']

def _get_time_coords(self) -> dict:
    if self.flow_system.is_clustered:
        return {
            'cluster': range(self.flow_system.clustering.n_clusters),
            'time': range(self.flow_system.clustering.timesteps_per_cluster)
        }
    return {'time': self.flow_system.time_coords}
```

### Phase 3: Simplify StorageModel and InterclusterStorageModel

**Goal:** Leverage true dimensions for clean constraint generation.

**Tasks:**
1. Simplify `StorageModel.charge_balance` - no boundary masking needed
2. Simplify delta SOC calculation: `charge_state[:, -1] - charge_state[:, 0]`
3. Simplify `InterclusterStorageModel` linking constraints
4. Update `intercluster_helpers.py` utilities

**Files:**
- `flixopt/components.py` - StorageModel, InterclusterStorageModel
- `flixopt/clustering/intercluster_helpers.py` - Simplified helpers

**Before/After:**
```python
# BEFORE (flat time with masking):
start_positions = clustering.cluster_start_positions
end_positions = start_positions[1:] - 1
mask = _build_boundary_mask(...)
balance = charge_state.isel(time=slice(1, None)).where(~mask) - ...

# AFTER (true dimensions):
# charge_state shape: (cluster, time+1, ...)
balance = (
    charge_state[:, 1:] -
    charge_state[:, :-1] * (1 - loss_rate) -
    charge * eta_charge +
    discharge / eta_discharge
)
# No masking needed - constraints naturally stay within clusters!
```

### Phase 4: Update transform_accessor.cluster()

**Goal:** Produce true `(cluster, time)` shaped data.

**Tasks:**
1. Update `cluster()` to reshape time series to `(cluster, time)`
2. Generate proper coordinates for cluster dimension
3. Update `expand_solution()` to handle reverse transformation
4. Handle SOC_boundary expansion for inter-cluster storage

**Files:**
- `flixopt/transform_accessor.py` - cluster() and expand_solution()

**Key Implementation:**
```python
def cluster(self, n_clusters, cluster_duration, ...):
    """Create clustered FlowSystem with (cluster, time) dimensions."""
    ...
    # Reshape all time series: (flat_time,) → (cluster, time)
    for key, ts in time_series.items():
        reshaped = ts.values.reshape(n_clusters, timesteps_per_cluster)
        new_ts = xr.DataArray(
            reshaped,
            dims=['cluster', 'time'],
            coords={'cluster': range(n_clusters), 'time': range(timesteps_per_cluster)}
        )
        clustered_time_series[key] = new_ts
    ...

def expand_solution(self):
    """Expand clustered solution back to original timeline."""
    expanded = {}
    for var_name, var_data in self.solution.items():
        if 'cluster' in var_data.dims:
            # Expand using cluster_order to map back to original periods
            expanded[var_name] = self._expand_clustered_data(var_data)
        else:
            expanded[var_name] = var_data
    return xr.Dataset(expanded)
```

### Phase 5: Plotting Integration

**Goal:** Minimal changes - leverage existing infrastructure.

**Tasks:**
1. Ensure `facet_col='cluster'` works with existing plot methods
2. Add auto cluster separators in combined time series views
3. Test heatmaps with `(cluster, time)` data

**Files:**
- `flixopt/statistics_accessor.py` - Minor update to base plot method

**Implementation:**
```python
# In _create_base_plot or similar:
def _add_cluster_separators(self, fig):
    """Add subtle separators between clusters in combined view."""
    if self._fs.is_clustered:
        for cluster_idx in range(1, self._fs.clustering.n_clusters):
            x_pos = cluster_idx * self._fs.clustering.timesteps_per_cluster
            fig.add_vline(x=x_pos, line_dash='dot', opacity=0.3)
```

### Phase Summary

| Phase | Goal | Complexity | StatusModel Fix? |
|-------|------|------------|------------------|
| 1 | Core dimension refactoring | High | N/A (prep work) |
| 2 | Variable/constraint creation | Medium | ✅ Automatic |
| 3 | StorageModel simplification | Medium | N/A |
| 4 | transform_accessor updates | Medium | N/A |
| 5 | Plotting integration | Low | N/A |

**Key Insight:** With true `(cluster, time)` dimensions, StatusModel and other temporal constraints **just work** without any special handling. The dimension structure naturally prevents constraints from spanning cluster boundaries.

---

## Part 7: Testing Strategy

### 7.1 Unit Tests

```python
# Test cluster helpers
def test_cluster_labels_uniform():
    """Verify cluster_labels for uniform cluster lengths."""

def test_cluster_slices_variable():
    """Verify cluster_slices for variable cluster lengths."""

def test_boundaries_vary_by_period():
    """Verify boundary dispatch for different periods."""
```

### 7.2 Integration Tests

```python
# Test storage with different cluster modes
def test_storage_intercluster_with_helpers():
    """Verify intercluster storage using new helpers."""

def test_storage_variable_boundaries():
    """Verify storage with period-varying boundaries."""
```

### 7.3 Plotting Tests

```python
# Test new plot methods
def test_storage_by_cluster_facets():
    """Verify faceted cluster view."""

def test_cluster_heatmap():
    """Verify cluster heatmap rendering."""
```

---

## Part 8: Decisions (Resolved)

| Question | Decision |
|----------|----------|
| **Naming** | Use `cluster` as the dimension/coordinate name |
| **Indexer return type** | Always return proper multi-dimensional xarray DataArrays |
| **Segmentation** | tsam uniform segments only (sufficient for current needs) |
| **Backwards compatibility** | Not a concern - this is not released yet |

---

## Appendix A: File Reference

| File | Purpose |
|------|---------|
| `flixopt/clustering/base.py` | ClusterStructure, Clustering classes |
| `flixopt/clustering/intercluster_helpers.py` | SOC boundary utilities |
| `flixopt/flow_system.py` | FlowSystem with is_clustered property |
| `flixopt/transform_accessor.py` | cluster() method, solution expansion |
| `flixopt/components.py` | StorageModel, InterclusterStorageModel |
| `flixopt/features.py` | StatusModel, ShareAllocationModel |
| `flixopt/statistics_accessor.py` | Plotting methods |
| `flixopt/plotting.py` | Plot utilities |

## Appendix B: Code Examples

### B.1 Working with True (cluster, time) Dimensions

```python
# ═══════════════════════════════════════════════════════════════
# DIMENSION STRUCTURE
# ═══════════════════════════════════════════════════════════════
# Non-clustered:
flow_rate.dims   # ('time', 'period', 'scenario')
flow_rate.shape  # (8760, ...)

# Clustered:
flow_rate.dims   # ('cluster', 'time', 'period', 'scenario')
flow_rate.shape  # (9, 24, ...)  # 9 clusters × 24 timesteps

# ═══════════════════════════════════════════════════════════════
# NATURAL CLUSTER BOUNDARY OPERATIONS
# ═══════════════════════════════════════════════════════════════
# First/last timestep of each cluster - just use isel!
flow_at_start = flow_rate.isel(time=0)   # Shape: (cluster, period, scenario)
flow_at_end = flow_rate.isel(time=-1)    # Shape: (cluster, period, scenario)

# Delta per cluster - trivial!
delta_per_cluster = flow_rate.isel(time=-1) - flow_rate.isel(time=0)

# ═══════════════════════════════════════════════════════════════
# TEMPORAL CONSTRAINTS - JUST WORK!
# ═══════════════════════════════════════════════════════════════
# Storage balance - naturally stays within clusters
balance = charge_state[:, 1:] - charge_state[:, :-1]  # No masking needed!

# Status transitions - naturally per cluster
activate = status[:, 1:] - status[:, :-1]  # No boundary issues!

# ═══════════════════════════════════════════════════════════════
# PER-CLUSTER AGGREGATION - use xarray directly
# ═══════════════════════════════════════════════════════════════
mean_per_cluster = flow_rate.mean(dim='time')   # Shape: (cluster, ...)
max_per_cluster = flow_rate.max(dim='time')
total_per_cluster = (flow_rate * timestep_duration).sum(dim='time')

# ═══════════════════════════════════════════════════════════════
# SELECT SPECIFIC WITHIN-CLUSTER TIMESTEP
# ═══════════════════════════════════════════════════════════════
# Peak hour (hour 18) from each cluster
peak_values = flow_rate.isel(time=18)  # Shape: (cluster, ...)

# Multiple timesteps
morning_values = flow_rate.isel(time=slice(6, 12))  # Hours 6-11 from each cluster
```

### B.2 Storage Constraints with True Dimensions

```python
# ═══════════════════════════════════════════════════════════════
# charge_state has one extra timestep per cluster for boundaries
# ═══════════════════════════════════════════════════════════════
# charge_state.dims = ('cluster', 'time_cs', 'period', 'scenario')
# charge_state.shape = (9, 25, ...)  # 24 timesteps + 1 boundary

# Charge balance - vectorized, no loops!
lhs = (
    charge_state[:, 1:] -                          # SOC at end of timestep
    charge_state[:, :-1] * (1 - loss_rate) -       # SOC at start, with loss
    charge * eta_charge +                          # Charging adds energy
    discharge / eta_discharge                      # Discharging removes energy
)
model.add_constraints(lhs == 0, name='charge_balance')

# Delta SOC per cluster (for inter-cluster linking)
delta_soc = charge_state[:, -1] - charge_state[:, 0]  # Shape: (cluster, ...)

# Cluster start constraint (relative SOC starts at 0 within each cluster)
model.add_constraints(charge_state[:, 0] == 0, name='cluster_start')

# Cyclic constraint (optional)
model.add_constraints(
    charge_state[:, 0] == charge_state[:, -1],
    name='cyclic'
)
```

### B.3 Cluster Plotting (Uses Existing API!)

```python
# ═══════════════════════════════════════════════════════════════
# FACET BY CLUSTER - uses existing facet_col parameter
# ═══════════════════════════════════════════════════════════════
fs.statistics.plot.storage('Battery', facet_col='cluster')
fs.statistics.plot.balance('Heat', facet_col='cluster')
fs.statistics.plot.flows(..., facet_col='cluster')

# ═══════════════════════════════════════════════════════════════
# REGULAR PLOTS - auto-add cluster separators when clustered
# ═══════════════════════════════════════════════════════════════
fs.statistics.plot.storage('Battery')  # separators added automatically

# ═══════════════════════════════════════════════════════════════
# COMBINE WITH OTHER FACETS
# ═══════════════════════════════════════════════════════════════
fs.statistics.plot.balance('Heat', facet_col='cluster', facet_row='scenario')
```

### B.4 Check Clustering Status and Access Properties

```python
if flow_system.is_clustered:
    clustering = flow_system.clustering
    print(f"Clustered: {clustering.n_clusters} clusters × {clustering.timesteps_per_cluster} timesteps")

    # Dimension information
    print(f"Data shape: (cluster={clustering.n_clusters}, time={clustering.timesteps_per_cluster})")

    # Cluster weights (how many original periods each cluster represents)
    print(f"Cluster weights: {flow_system.cluster_weight.values}")

    # Aggregation weight (cluster_weight × timestep_duration)
    print(f"Aggregation weight shape: {flow_system.aggregation_weight.shape}")
else:
    print("Not clustered - full time resolution")
```

### B.5 Variable Segment Durations (Future)

```python
# ═══════════════════════════════════════════════════════════════
# timestep_duration varies per (cluster, time)
# ═══════════════════════════════════════════════════════════════
timestep_duration = xr.DataArray(
    [
        [2, 2, 4, 4, 6, 6],  # Cluster 0: short-short-long pattern
        [1, 1, 4, 8, 4, 6],  # Cluster 1: different pattern
        ...
    ],
    dims=['cluster', 'time'],
    coords={'cluster': range(9), 'time': range(6)}
)

# aggregation_weight combines duration and cluster importance
aggregation_weight = timestep_duration * cluster_weight
# Shape: (cluster, time) × (cluster,) → (cluster, time) via broadcasting

# All constraints use aggregation_weight for proper weighting
total_cost = (cost_per_timestep * aggregation_weight).sum(dim=['cluster', 'time'])
```
