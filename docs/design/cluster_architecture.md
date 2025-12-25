The enhanced# Design Document: Cluster Architecture for flixopt

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

### 2.1 Option A: Enhanced Flat with Cluster Helpers (Recommended)

Keep flat `time` dimension but add rich helper infrastructure:

```python
class Clustering:
    # Core properties
    cluster_labels: xr.DataArray      # (time,) or (time, period, scenario)
    timesteps_per_cluster: xr.DataArray  # (cluster,) or (cluster, period, scenario)

    # Index helpers (period/scenario-aware)
    def cluster_start_indices(self, period=None, scenario=None) -> np.ndarray
    def cluster_end_indices(self, period=None, scenario=None) -> np.ndarray
    def cluster_slices(self, period=None, scenario=None) -> dict[int, slice]

    # Data access helpers
    def get_cluster_data(self, data, cluster_id, period=None, scenario=None)
    def iter_clusters(self, data, period=None, scenario=None)
    def get_cluster_boundaries(self, data, period=None, scenario=None)
    def compute_delta_per_cluster(self, data, period=None, scenario=None)

    # Boundary variability
    boundaries_vary: bool
    boundaries_vary_by_period: bool
    boundaries_vary_by_scenario: bool
```

**Pros:**
- Supports variable-length clusters
- Supports different boundaries per period/scenario
- Minimal breaking changes
- linopy-compatible

**Cons:**
- Less intuitive than true `(cluster, time)` shape
- Requires helper methods for clean code

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

### 3.1 StatusModel Impact

**Current Code (features.py:200-211):**
```python
# Active hours tracking
tracked_expression=(self.status * self._model.aggregation_weight).sum('time')
```

**With Enhanced Helpers:**
No changes needed - `aggregation_weight` already handles clustering correctly.

**Potential Enhancement:**
Could add per-cluster status summaries for visualization:
```python
@property
def status_per_cluster(self) -> xr.DataArray:
    """Active hours per cluster."""
    clustering = self.flow_system.clustering
    if clustering is None:
        return None
    # Use helpers to compute per-cluster active time
    return clustering.aggregate_per_cluster(
        self.status * self._model.timestep_duration
    )
```

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

## Part 6: Implementation Roadmap

### Phase 1: Core Helpers (Minimal Change)

**Goal:** Add cluster helpers without changing existing behavior.

**Tasks:**
1. Add `is_clustered`, `n_clusters` to FlowSystem
2. Add `cluster_labels`, `cluster_slices`, index methods to Clustering
3. Add `boundaries_vary` flag infrastructure
4. Refactor InterclusterStorageModel to use helpers

**Files:**
- `flixopt/clustering/base.py`
- `flixopt/flow_system.py`
- `flixopt/components.py`

### Phase 2: Plotting Improvements

**Goal:** Better cluster visualization UX.

**Tasks:**
1. Add cluster separator lines to time series plots
2. Implement `storage_by_cluster()` faceted view
3. Add `cluster_summary()` statistics
4. Implement `cluster_heatmap()`
5. Add `intercluster_soc()` for inter-cluster storage

**Files:**
- `flixopt/statistics_accessor.py`
- `flixopt/plotting.py`

### Phase 3: Period/Scenario-Aware Helpers

**Goal:** Support different cluster boundaries per period/scenario.

**Tasks:**
1. Extend helper methods with period/scenario parameters
2. Add `_get_boundaries(period, scenario)` dispatch
3. Update constraint generation to loop when needed
4. Update tests for varying boundaries

**Files:**
- `flixopt/clustering/base.py`
- `flixopt/components.py`
- `flixopt/features.py` (if needed)

### Phase 4: Segmentation Infrastructure

**Goal:** Prepare for tsam segmentation support.

**Tasks:**
1. Define `SegmentStructure` dataclass
2. Integrate with `ClusterStructure`
3. Update `transform_accessor.cluster()` to accept segmentation params
4. Update constraint generation for segments

**Files:**
- `flixopt/clustering/base.py`
- `flixopt/transform_accessor.py`
- `flixopt/components.py`

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

## Part 8: Open Questions

1. **Naming**: Should the coordinate be `cluster` or `cluster_idx`?
2. **Default behavior**: When boundaries vary, should helpers require period/scenario or auto-detect?
3. **Segmentation granularity**: Support arbitrary segments or only tsam's uniform segments?
4. **Backwards compatibility**: Keep old `cluster_start_positions` property or deprecate?

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

### B.1 Using Enhanced Helpers

```python
# Get cluster boundaries
clustering = flow_system.clustering
starts, ends = clustering.get_cluster_boundaries(charge_state)

# Compute delta SOC per cluster
delta_soc = clustering.compute_delta_per_cluster(charge_state)

# Iterate over clusters
for cluster_id, cluster_data in clustering.iter_clusters(flow_rate):
    process(cluster_data)
```

### B.2 Faceted Storage Plot

```python
# Plot storage with cluster facets
fs.statistics.plot.storage_by_cluster('Battery')

# Plot cluster summary
fs.statistics.plot.cluster_summary('HeatDemand|Q_th', statistic='max')
```

### B.3 Variable Boundaries

```python
# Check if boundaries vary
if clustering.boundaries_vary:
    for period in fs.periods:
        slices = clustering.cluster_slices(period=period)
        # Process per-period
else:
    slices = clustering.cluster_slices()
    # Single set of slices
```
