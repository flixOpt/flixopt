# Design Document: Cluster Architecture for flixopt

## Executive Summary

This document explores architectural options for improving cluster representation in flixopt, addressing:
1. Enhanced cluster helpers for the current flat time structure
2. Impact on StatusModel and other Features
3. Improved UX for cluster visualization and plotting
4. Future support for variable segmentation per cluster/period/scenario

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

### 2.1 Option A: Enhanced Flat with xarray-based Indexers (Recommended)

Keep flat `time` dimension but add **xarray-based indexer properties** that work seamlessly with `.isel()`:

```python
class Clustering:
    # ═══════════════════════════════════════════════════════════════
    # CORE INDEXER PROPERTIES (xarray DataArrays)
    # ═══════════════════════════════════════════════════════════════

    @property
    def cluster_start(self) -> xr.DataArray:
        """Time indices of cluster starts.

        Shape: (cluster,)
        Values: [0, 96, 192, ...] for 96 timesteps per cluster

        Usage:
            # Select start of each cluster (broadcasts across period/scenario)
            data.isel(time=clustering.cluster_start)

            # Shift by 1 for "second timestep of each cluster"
            data.isel(time=clustering.cluster_start + 1)
        """

    @property
    def cluster_end(self) -> xr.DataArray:
        """Time indices of cluster ends (last timestep, inclusive).

        Shape: (cluster,)
        Values: [95, 191, 287, ...] for 96 timesteps per cluster

        Usage:
            # Select end of each cluster
            data.isel(time=clustering.cluster_end)

            # Compute delta (end - start) for each cluster
            delta = data.isel(time=clustering.cluster_end) - data.isel(time=clustering.cluster_start)
        """

    @property
    def within_cluster_time(self) -> xr.DataArray:
        """Within-cluster time index for each timestep.

        Shape: (time,)
        Values: [0, 1, 2, ..., 95, 0, 1, 2, ..., 95, ...]  # repeating pattern

        Usage:
            # Select all timesteps at position 12 within their cluster
            mask = clustering.within_cluster_time == 12
            data.where(mask, drop=True)
        """

    @property
    def cluster(self) -> xr.DataArray:
        """Cluster ID for each timestep.

        Shape: (time,)
        Values: [0, 0, ..., 0, 1, 1, ..., 1, ...]  # cluster assignment

        Usage:
            # Group by cluster
            data.groupby(clustering.cluster).mean()
        """

    # ═══════════════════════════════════════════════════════════════
    # CONVENIENCE PROPERTIES
    # ═══════════════════════════════════════════════════════════════

    @property
    def n_clusters(self) -> int:
        """Number of clusters."""

    @property
    def timesteps_per_cluster(self) -> int:
        """Timesteps in each cluster (uniform)."""

    @property
    def cluster_coords(self) -> xr.DataArray:
        """Cluster coordinate values: [0, 1, 2, ..., n_clusters-1]"""
```

**Key Design Principle: Indexers are xarray DataArrays**

This enables powerful, dimension-preserving operations:

```python
# ═══════════════════════════════════════════════════════════════
# EXAMPLE: Select start of each cluster (works across all dims!)
# ═══════════════════════════════════════════════════════════════
charge_state = ...  # shape: (time, period, scenario) e.g., (864, 2, 3)

# Get cluster starts - returns shape (cluster, period, scenario)
cs_at_starts = charge_state.isel(time=clustering.cluster_start)
# Result shape: (9, 2, 3) for 9 clusters

# ═══════════════════════════════════════════════════════════════
# EXAMPLE: Compute delta per cluster
# ═══════════════════════════════════════════════════════════════
delta = (
    charge_state.isel(time=clustering.cluster_end) -
    charge_state.isel(time=clustering.cluster_start)
)
# Result shape: (cluster, period, scenario) = (9, 2, 3)

# ═══════════════════════════════════════════════════════════════
# EXAMPLE: Shift indexer for charge_state (has extra timestep!)
# ═══════════════════════════════════════════════════════════════
# charge_state has shape (time+1,) due to extra boundary timestep
# Need to shift indices by cluster position
cs_at_ends = charge_state.isel(time=clustering.cluster_end + 1)  # +1 for boundary

# ═══════════════════════════════════════════════════════════════
# EXAMPLE: Select specific within-cluster position
# ═══════════════════════════════════════════════════════════════
# Get all values at hour 12 within each cluster
hour_12_mask = clustering.within_cluster_time == 12
peak_values = data.where(hour_12_mask, drop=True)
```

**Pros:**
- Pure xarray - no numpy/dict gymnastics
- Dimension-preserving: indexers broadcast across period/scenario automatically
- Easy adjustments: `cluster_start + 1`, `cluster_end - 1`
- Works with linopy variables directly
- Clean, intuitive API

**Cons:**
- tsam uniform segments only (sufficient per user requirement)

### 2.2 Option B: True (cluster, time) Dimensions

Reshape time to 2D when clustering is active:

```python
# Clustered mode
data.dims = ('cluster', 'time', 'period', 'scenario')
data.shape = (9, 96, ...)  # 9 clusters × 96 timesteps each
```

**Pros:**
- Clean, intuitive structure
- Natural indexing: `data[:, -1] - data[:, 0]` for delta
- No boundary masking needed

**Cons:**
- Requires uniform cluster lengths
- Different boundaries per period/scenario very complex
- Major refactoring across codebase

### 2.3 Option C: Padded Rectangular with Masks

Use `(cluster, max_time)` with NaN padding for shorter clusters:

```python
data.shape = (9, 96, ...)  # Pad shorter clusters
valid_mask.shape = (9, 96)  # True where data is valid
```

**Pros:**
- Clean cluster dimension
- Supports variable lengths

**Cons:**
- Wasted memory/computation
- Complex masking in all operations
- linopy constraints need `.where(mask)`

### 2.4 Recommendation: Option A (Enhanced Flat)

Given the requirements for:
- Variable-length clusters (future segmentation)
- Different boundaries per period/scenario
- Minimal breaking changes

**Option A is the most practical choice.**

---

## Part 3: Impact on Features

### 3.1 StatusModel Impact - CRITICAL ISSUE

**Problem:** StatusModel has temporal constraints that span timesteps:

| Constraint | Current Behavior | Problem with Clustering |
|------------|------------------|------------------------|
| `min_uptime=4` | Must stay on 4 consecutive hours | Spans cluster boundaries incorrectly |
| `min_downtime=2` | Must stay off 2 consecutive hours | Same issue |
| `initial_status` | Status before t=0 | Undefined at each cluster start |
| `effects_per_startup` | Cost per on→off transition | Counted per cluster, not per original period |

**Example of the bug:**
```
Cluster 0: [t=0...t=95]  - component turns ON at t=90
Cluster 1: [t=96...t=191] - different typical day!

With min_uptime=8:
- Current: Constraint forces component to stay on t=90→t=97 (spans into cluster 1)
- Reality: Cluster 1 is a DIFFERENT day, constraint makes no sense
```

**Options for StatusModel with Clustering:**

| Mode | Description | Use Case |
|------|-------------|----------|
| **independent** | Each cluster has independent status constraints | Most common - typical days are independent |
| **cyclic** | Status at cluster end = status at cluster start | Repeating patterns |
| **ignore_temporal** | Disable min_uptime/downtime when clustered | Simple approximation |

**Recommended Approach:**
```python
class StatusParameters:
    # Existing
    min_uptime: float | None = None
    min_downtime: float | None = None
    initial_status: bool | None = None

    # NEW: How to handle with clustering
    cluster_mode: Literal['independent', 'cyclic', 'ignore_temporal'] = 'independent'
```

**Implementation for `cluster_mode='independent'`:**
```python
# In StatusModel, when adding consecutive duration constraints:
if clustering is not None and self.cluster_mode == 'independent':
    # Mask out constraints at cluster boundaries
    # Each cluster is treated independently
    for constraint in [uptime_constraint, downtime_constraint]:
        # Don't enforce across cluster boundaries
        mask = np.ones(n_timesteps, dtype=bool)
        mask[clustering.cluster_start.values[1:]] = False  # Break at cluster starts
        constraint = constraint.where(mask)
```

**What works correctly already:**
- `active_hours` tracking uses `aggregation_weight` → ✅ correct
- `effects_per_active_hour` uses `timestep_duration` → ✅ correct
- Total startup count (if properly weighted) → needs `cluster_weight`

### 3.2 StorageModel Impact

**Current Code (components.py):**
- Uses `cluster_start_positions` for boundary masking
- InterclusterStorageModel has complex index calculations

**With Enhanced Helpers:**
```python
# Before: Manual index calculation
start_positions = clustering.cluster_start_positions
end_positions = start_positions[1:] - 1

# After: Clean helper usage
clustering = self.flow_system.clustering
delta_soc = clustering.compute_delta_per_cluster(self.charge_state)
```

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

### 3.5 Summary: Models Requiring Cluster-Awareness

| Model | Has Cross-Timestep Constraints | Clustering Impact | Action Needed |
|-------|-------------------------------|-------------------|---------------|
| **StorageModel** | charge_state[t] depends on charge_state[t-1] | ✅ Already handled | InterclusterStorageModel exists |
| **StatusModel** | min_uptime, min_downtime, initial_status | ❌ **BUG** | Add cluster_mode parameter |
| **consecutive_duration_tracking** | State machine for uptime/downtime | ❌ **BUG** | Break at cluster boundaries |
| **state_transition_bounds** | activate[t] depends on status[t-1] | ⚠️ Partial | May span boundaries incorrectly |
| **PiecewiseModel** | Per-timestep only | ✅ OK | No changes needed |
| **ShareAllocationModel** | Uses cluster_weight | ✅ OK | No changes needed |
| **InvestmentModel** | No time dimension | ✅ OK | No changes needed |

**Key Insight:** Any constraint of the form `x[t] - x[t-1]` or `x[t:t+n]` needs cluster boundary handling.

---

## Part 4: Plotting Improvements

### 4.1 Current UX Issues

1. **No visual cluster separation**: Time series plots show continuous lines
2. **Cluster identity hidden**: Hard to see which timesteps belong to which cluster
3. **SOC continuity misleading**: Storage plots suggest continuous operation

### 4.2 Proposed Improvements

#### 4.2.1 Cluster-Separated Time Series

Add visual separators between clusters:

```python
def plot_with_cluster_separation(self, data, **kwargs):
    """Plot time series with vertical lines between clusters."""
    fig = self._create_base_plot(data, **kwargs)

    if self._fs.is_clustered:
        for start_idx in self._fs.clustering.cluster_start_indices()[1:]:
            fig.add_vline(x=data.time[start_idx], line_dash='dash', opacity=0.3)

    return fig
```

#### 4.2.2 Faceted Cluster View

Display each cluster as a separate subplot:

```python
def storage_by_cluster(self, storage_label, **kwargs):
    """Plot storage operation with one subplot per cluster."""
    data = self._get_storage_data(storage_label)

    if not self._fs.is_clustered:
        return self.storage(storage_label, **kwargs)

    # Reshape to (cluster, within_cluster_time)
    clustering = self._fs.clustering
    facet_data = []
    for cluster_id, cluster_slice in clustering.cluster_slices().items():
        cluster_data = data.isel(time=cluster_slice)
        cluster_data = cluster_data.assign_coords(
            cluster=cluster_id,
            within_time=range(len(cluster_slice))
        )
        facet_data.append(cluster_data)

    combined = xr.concat(facet_data, dim='cluster')
    return self._plot_faceted(combined, facet_col='cluster', **kwargs)
```

#### 4.2.3 Cluster Summary Statistics

Add aggregate views per cluster:

```python
def cluster_summary(self, variable, statistic='mean'):
    """Show per-cluster statistics as bar chart."""
    data = self._get_variable(variable)
    clustering = self._fs.clustering

    summaries = []
    for cluster_id, cluster_slice in clustering.cluster_slices().items():
        cluster_data = data.isel(time=cluster_slice)
        if statistic == 'mean':
            val = cluster_data.mean('time')
        elif statistic == 'max':
            val = cluster_data.max('time')
        elif statistic == 'min':
            val = cluster_data.min('time')
        summaries.append(val.assign_coords(cluster=cluster_id))

    return self._plot_bar(xr.concat(summaries, dim='cluster'))
```

#### 4.2.4 Inter-Cluster SOC Visualization

Show SOC_boundary values for intercluster storage:

```python
def intercluster_soc(self, storage_label):
    """Plot SOC boundaries across original timeline."""
    storage = self._get_component(storage_label)
    if not hasattr(storage.submodel, 'SOC_boundary'):
        raise ValueError("Storage not in intercluster mode")

    soc_boundary = storage.submodel.SOC_boundary.solution
    cluster_order = self._fs.clustering.cluster_order

    # Plot SOC at each original period boundary
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=range(len(soc_boundary)),
        y=soc_boundary.values,
        mode='lines+markers',
        name='SOC Boundary'
    ))
    fig.update_layout(
        xaxis_title='Original Period',
        yaxis_title='State of Charge',
        title=f'{storage_label} Inter-Cluster SOC'
    )
    return PlotResult(fig)
```

### 4.3 Heatmap Enhancements

Current heatmap reshapes time to (days, hours). For clustered data:

```python
def cluster_heatmap(self, variable):
    """Heatmap with clusters on y-axis, within-cluster time on x-axis."""
    data = self._get_variable(variable)
    clustering = self._fs.clustering

    # Reshape: (total_time,) -> (n_clusters, timesteps_per_cluster)
    reshaped = data.values.reshape(
        clustering.n_clusters,
        clustering.timesteps_per_cluster
    )

    return self._plot_heatmap(
        reshaped,
        x_label='Within-Cluster Time',
        y_label='Cluster',
        colorbar_title=variable
    )
```

---

## Part 5: Variable Segmentation Architecture

### 5.1 Segmentation Types

| Type | Description | Complexity |
|------|-------------|------------|
| **Uniform segments** | All clusters have same structure | Current implementation |
| **Variable per cluster** | Cluster 1: 24 steps, Cluster 2: 48 steps | Medium |
| **Variable per period** | Period 1 clusters differ from Period 2 | High |
| **Variable per scenario** | Scenario A differs from Scenario B | High |
| **Full variability** | Different per (cluster, period, scenario) | Very High |

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
- Uniform segment count across all typical periods
- Various representation methods (mean, medoid, distribution)
- Segment duration = `timesteps_per_cluster / noSegments`

**What TSAM does NOT provide:**
- Variable segment lengths within a period
- Different segment counts per cluster

### 5.3 Implementing Variable Segmentation

#### 5.3.1 Data Structures

```python
@dataclass
class SegmentStructure:
    """Structure for variable-length segments within clusters."""

    # Shape: (cluster,) - number of segments in each cluster
    n_segments_per_cluster: xr.DataArray

    # Shape: (cluster, max_segments) - duration of each segment (NaN if not used)
    segment_durations: xr.DataArray

    # Shape: (cluster, max_segments) - start index within cluster
    segment_start_indices: xr.DataArray

    # For period/scenario variation, add those dims to all arrays

    def get_segment_slice(self, cluster, segment, period=None, scenario=None) -> slice:
        """Get time slice for a specific segment."""
        ...
```

#### 5.3.2 Enhanced ClusterStructure

```python
class ClusterStructure:
    # Existing
    cluster_order: xr.DataArray
    cluster_occurrences: xr.DataArray
    n_clusters: int
    timesteps_per_cluster: int | xr.DataArray  # Allow variable

    # New for segmentation
    segmentation: SegmentStructure | None

    # Period/scenario awareness
    _boundaries_by_slice: dict[tuple, BoundaryInfo]  # (period, scenario) -> info

    @property
    def has_variable_boundaries(self) -> bool:
        """True if boundaries differ across periods/scenarios."""
        return len(self._boundaries_by_slice) > 1

    def get_boundaries(self, period=None, scenario=None) -> BoundaryInfo:
        """Get cluster boundaries for specific period/scenario."""
        key = (period, scenario)
        if key in self._boundaries_by_slice:
            return self._boundaries_by_slice[key]
        return self._default_boundaries
```

#### 5.3.3 Integration with FlowSystem

```python
class FlowSystem:
    @property
    def is_clustered(self) -> bool:
        return self.clustering is not None

    @property
    def has_segmentation(self) -> bool:
        return self.is_clustered and self.clustering.segmentation is not None

    @property
    def has_variable_cluster_lengths(self) -> bool:
        """True if clusters have different numbers of timesteps."""
        if not self.is_clustered:
            return False
        tpc = self.clustering.timesteps_per_cluster
        if isinstance(tpc, int):
            return False
        return len(np.unique(tpc)) > 1
```

### 5.4 Constraint Generation with Variable Segments

When segment lengths vary, constraint generation must loop or use advanced indexing:

```python
def _add_charge_state_constraints(self):
    clustering = self.flow_system.clustering

    if not clustering.has_variable_boundaries:
        # Vectorized path - all clusters have same structure
        self._add_charge_state_vectorized()
    else:
        # Loop path - boundaries vary
        for period in self.flow_system.periods or [None]:
            for scenario in self.flow_system.scenarios or [None]:
                self._add_charge_state_for_slice(period, scenario)

def _add_charge_state_for_slice(self, period, scenario):
    """Add constraints for specific period/scenario slice."""
    boundaries = self.clustering.get_boundaries(period, scenario)

    for cluster_id in range(boundaries.n_clusters):
        slc = boundaries.cluster_slices[cluster_id]
        cs_cluster = self.charge_state.isel(time=slc)

        if period is not None:
            cs_cluster = cs_cluster.sel(period=period)
        if scenario is not None:
            cs_cluster = cs_cluster.sel(scenario=scenario)

        # Add constraints for this cluster
        self._add_balance_for_cluster(cs_cluster, cluster_id, period, scenario)
```

---

## Part 6: Implementation Roadmap (Focused)

### Phase 1: xarray-based Indexers (PRIORITY)

**Goal:** Add xarray-based cluster indexer properties to `Clustering`.

**Tasks:**
1. Add `cluster_start` property → `xr.DataArray` with dims `(cluster,)`
2. Add `cluster_end` property → `xr.DataArray` with dims `(cluster,)`
3. Add `cluster` property → `xr.DataArray` with dims `(time,)` for cluster labels
4. Add `within_cluster_time` property → `xr.DataArray` with dims `(time,)`
5. Add convenience: `n_clusters`, `timesteps_per_cluster`, `cluster_coords`
6. Add `is_clustered` property to `FlowSystem`

**Files:**
- `flixopt/clustering/base.py` - Add indexer properties to `Clustering`
- `flixopt/flow_system.py` - Add `is_clustered` convenience property

**Example Implementation:**
```python
@property
def cluster_start(self) -> xr.DataArray:
    """Time indices where each cluster starts."""
    indices = np.arange(0, self.n_clusters * self.timesteps_per_cluster, self.timesteps_per_cluster)
    return xr.DataArray(indices, dims=['cluster'], coords={'cluster': np.arange(self.n_clusters)})

@property
def cluster_end(self) -> xr.DataArray:
    """Time indices where each cluster ends (inclusive)."""
    return self.cluster_start + self.timesteps_per_cluster - 1
```

### Phase 2: Refactor InterclusterStorageModel

**Goal:** Use new xarray indexers in `InterclusterStorageModel`.

**Tasks:**
1. Replace manual index calculations with `clustering.cluster_start`, `clustering.cluster_end`
2. Simplify `_compute_delta_soc()` using indexer arithmetic
3. Simplify `_add_cluster_start_constraints()` using indexers
4. Handle charge_state offset (extra timestep) cleanly

**Files:**
- `flixopt/components.py` - Refactor `InterclusterStorageModel`

**Before/After Example:**
```python
# BEFORE: Manual calculation
start_positions = clustering.cluster_start_positions
end_positions = start_positions[1:] - 1
delta = charge_state.isel(time=end_indices) - charge_state.isel(time=start_indices)

# AFTER: xarray indexers
# Note: charge_state has +1 timesteps, so shift accordingly
delta = (
    self.charge_state.isel(time=clustering.cluster_end + 1) -
    self.charge_state.isel(time=clustering.cluster_start)
)
```

### Phase 3: expand_solution() with Offset Handling

**Goal:** Proper solution expansion for variables with different time structures.

**Tasks:**
1. Update `expand_solution()` to detect variable type (regular vs charge_state)
2. Add offset handling for intercluster charge_state expansion
3. Map SOC_boundary values to original timeline correctly
4. Test with all storage cluster_modes

**Files:**
- `flixopt/transform_accessor.py` - Update `expand_solution()`
- `flixopt/clustering/base.py` - Add expansion helpers if needed

**Key Insight:**
```python
def expand_solution():
    for var_name, var_data in solution.items():
        if 'charge_state' in var_name and is_intercluster:
            # Special handling: map SOC_boundary to original period boundaries
            expanded = _expand_intercluster_soc(var_data)
        else:
            # Normal expansion using timestep_mapping
            expanded = result.expand_data(var_data)
```

### Phase 4: Fix StatusModel & Temporal Constraints

**Goal:** Handle cross-timestep constraints correctly with clustering.

**Tasks:**
1. Add `cluster_mode` parameter to `StatusParameters` (default: `'independent'`)
2. Update `consecutive_duration_tracking` to break at cluster boundaries
3. Update `state_transition_bounds` to handle cluster boundaries
4. Add warning/error if incompatible constraints used with clustering

**Implementation:**
```python
# In modeling.py consecutive_duration_tracking:
def consecutive_duration_tracking(..., clustering=None):
    ...
    if clustering is not None:
        # Don't track duration across cluster boundaries
        # Reset tracking at start of each cluster
        reset_mask = np.zeros(n_timesteps, dtype=bool)
        reset_mask[clustering.cluster_start.values] = True
        # Modify constraints to reset at cluster starts
```

**Files:**
- `flixopt/interface.py` - Add `cluster_mode` to `StatusParameters`
- `flixopt/modeling.py` - Update `consecutive_duration_tracking`, `state_transition_bounds`
- `flixopt/features.py` - Update `StatusModel` to use cluster_mode

### Phase 5: Cluster-Aware Plotting (Minimal Code)

**Goal:** Leverage existing plot infrastructure - no new methods needed!

**Key Insight:** Add `cluster` as a coordinate to solution data, then existing faceting works:

```python
# EXISTING API - no new methods needed!
fs.statistics.plot.storage('Battery', facet_col='cluster')  # faceted by cluster
fs.statistics.plot.balance('Heat', facet_col='cluster')     # works automatically
fs.statistics.plot.flows(..., facet_col='cluster')          # same pattern
```

**Tasks:**
1. Add `cluster` coordinate to solution Dataset when clustered
2. Auto-add cluster separator lines in time series plots (if clustered)
3. Ensure `facet_col='cluster'` works with existing plot methods

**Implementation:**
```python
# In expand_solution() or statistics accessor:
if flow_system.is_clustered:
    # Add cluster coordinate to all time-dimensioned variables
    solution = solution.assign_coords(cluster=('time', clustering.cluster.values))

# In plot methods (minimal change):
def _create_base_plot(self, data, **kwargs):
    fig = ...  # existing logic

    # Auto-add cluster separators if clustered
    if self._fs.is_clustered and 'time' in data.dims:
        for idx in self._fs.clustering.cluster_start.values[1:]:
            fig.add_vline(x=idx, line_dash='dot', opacity=0.3)

    return fig
```

**Files:**
- `flixopt/transform_accessor.py` - Add cluster coord in expand_solution()
- `flixopt/statistics_accessor.py` - Add separator lines (small change to base plot)

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

### B.1 Using xarray Indexers

```python
clustering = flow_system.clustering

# ═══════════════════════════════════════════════════════════════
# Select values at cluster boundaries
# ═══════════════════════════════════════════════════════════════
flow_at_starts = flow_rate.isel(time=clustering.cluster_start)
flow_at_ends = flow_rate.isel(time=clustering.cluster_end)

# ═══════════════════════════════════════════════════════════════
# Compute delta per cluster (e.g., for storage charge change)
# ═══════════════════════════════════════════════════════════════
delta = data.isel(time=clustering.cluster_end) - data.isel(time=clustering.cluster_start)
# Result has dims: (cluster, period, scenario) if those exist

# ═══════════════════════════════════════════════════════════════
# Handle charge_state (has extra timestep at end of each cluster)
# ═══════════════════════════════════════════════════════════════
# charge_state shape: (time + n_clusters,) due to boundary timesteps
cs_at_cluster_start = charge_state.isel(time=clustering.cluster_start)
cs_at_cluster_end = charge_state.isel(time=clustering.cluster_end + 1)  # +1 for boundary

# ═══════════════════════════════════════════════════════════════
# Group operations by cluster
# ═══════════════════════════════════════════════════════════════
mean_per_cluster = data.groupby(clustering.cluster).mean()
max_per_cluster = data.groupby(clustering.cluster).max()

# ═══════════════════════════════════════════════════════════════
# Select specific within-cluster timestep
# ═══════════════════════════════════════════════════════════════
# Get all peak hours (e.g., hour 18) from each cluster
peak_mask = clustering.within_cluster_time == 18
peak_values = data.where(peak_mask, drop=True)
```

### B.2 Cluster Plotting (Uses Existing API!)

```python
# ═══════════════════════════════════════════════════════════════
# Facet by cluster - uses existing facet_col parameter
# ═══════════════════════════════════════════════════════════════
fs.statistics.plot.storage('Battery', facet_col='cluster')
fs.statistics.plot.balance('Heat', facet_col='cluster')
fs.statistics.plot.flows(..., facet_col='cluster')

# ═══════════════════════════════════════════════════════════════
# Regular plots auto-add cluster separator lines when clustered
# ═══════════════════════════════════════════════════════════════
fs.statistics.plot.storage('Battery')  # separators added automatically

# ═══════════════════════════════════════════════════════════════
# Combine with other facets
# ═══════════════════════════════════════════════════════════════
fs.statistics.plot.balance('Heat', facet_col='cluster', facet_row='scenario')
```

### B.3 Check Clustering Status

```python
if flow_system.is_clustered:
    clustering = flow_system.clustering
    print(f"Clustered: {clustering.n_clusters} clusters × {clustering.timesteps_per_cluster} timesteps")

    # Access indexers
    print(f"Cluster starts: {clustering.cluster_start.values}")
    print(f"Cluster ends: {clustering.cluster_end.values}")
else:
    print("Not clustered - full time resolution")
```
