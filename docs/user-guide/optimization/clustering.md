# Time-Series Clustering

Time-series clustering reduces large optimization problems by aggregating timesteps into representative **typical periods**. This enables fast investment optimization while preserving key system dynamics.

## When to Use Clustering

Use clustering when:

- Optimizing over a **full year** or longer
- **Investment sizing** is the primary goal (not detailed dispatch)
- You need **faster solve times** and can accept approximation
- The system has **repeating patterns** (daily, weekly, seasonal)

**Skip clustering** for:

- Short optimization horizons (days to weeks)
- Dispatch-only problems without investments
- Systems requiring exact temporal sequences

## Two-Stage Workflow

The recommended approach: cluster for fast sizing, then validate at full resolution.

```python
import flixopt as fx
from tsam import ExtremeConfig

# Load or create your FlowSystem
flow_system = fx.FlowSystem(timesteps)
flow_system.add_elements(...)

# Stage 1: Cluster and optimize (fast)
fs_clustered = flow_system.transform.cluster(
    n_clusters=12,
    cluster_duration='1D',
    extremes=ExtremeConfig(method='new_cluster', max_value=['HeatDemand(Q)|fixed_relative_profile']),
)
fs_clustered.optimize(fx.solvers.HighsSolver())

# Stage 2: Expand back to full resolution
fs_expanded = fs_clustered.transform.expand()

# Access full-resolution results
charge_state = fs_expanded.solution['Storage|charge_state']
flow_rates = fs_expanded.solution['Boiler(Q_th)|flow_rate']
```

## Clustering Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `n_clusters` | Number of typical periods | `12` (typical days for a year) |
| `cluster_duration` | Duration of each cluster | `'1D'`, `'24h'`, or `24` (hours) |
| `weights` | Clustering weights per time series | `{'demand': 2.0, 'solar': 1.0}` |
| `cluster` | tsam `ClusterConfig` for clustering options | `ClusterConfig(method='k_medoids')` |
| `extremes` | tsam `ExtremeConfig` for peak preservation | `ExtremeConfig(method='new_cluster', max_value=[...])` |
| `predef_cluster_assignments` | Manual cluster assignments | Array of cluster indices |

### Peak Selection with ExtremeConfig

Use `ExtremeConfig` to ensure extreme conditions are represented:

```python
from tsam import ExtremeConfig

# Ensure the peak demand day is included
fs_clustered = flow_system.transform.cluster(
    n_clusters=8,
    cluster_duration='1D',
    extremes=ExtremeConfig(
        method='new_cluster',  # Create new cluster for extremes
        max_value=['HeatDemand(Q)|fixed_relative_profile'],  # Capture peak demand
    ),
)
```

Without peak selection, the clustering algorithm might average out extreme days, leading to undersized equipment.

**ExtremeConfig options:**

| Field | Description |
|-------|-------------|
| `method` | How extremes are handled: `'new_cluster'`, `'append'`, `'replace_cluster_center'` |
| `max_value` | Time series where maximum values should be preserved |
| `min_value` | Time series where minimum values should be preserved |
| `max_period` | Time series where period with maximum sum should be preserved |
| `min_period` | Time series where period with minimum sum should be preserved |

### Advanced Clustering Options with ClusterConfig

Fine-tune the clustering algorithm with `ClusterConfig`:

```python
from tsam import ClusterConfig, ExtremeConfig

fs_clustered = flow_system.transform.cluster(
    n_clusters=8,
    cluster_duration='1D',
    cluster=ClusterConfig(
        method='hierarchical',  # Clustering algorithm
        representation='medoid',  # Use actual periods, not averages
    ),
    extremes=ExtremeConfig(method='new_cluster', max_value=['demand']),
)
```

**Available clustering algorithms** (`ClusterConfig.method`):

| Method | Description |
|--------|-------------|
| `'hierarchical'` | Produces consistent hierarchical groupings (default) |
| `'k_means'` | Fast, good for most cases |
| `'k_medoids'` | Uses actual periods as representatives |
| `'k_maxoids'` | Maximizes representativeness |
| `'averaging'` | Simple averaging of similar periods |

**Representation methods** (`ClusterConfig.representation`):

| Method | Description |
|--------|-------------|
| `'medoid'` | Use actual periods as representatives (default) |
| `'mean'` | Average of all periods in cluster |
| `'distribution'` | Preserve value distribution (duration curves) |

For additional tsam parameters, pass them as keyword arguments:

```python
# Pass any tsam.aggregate() parameter
fs_clustered = flow_system.transform.cluster(
    n_clusters=8,
    cluster_duration='1D',
    normalize_column_means=True,  # Normalize all time series to same mean
    preserve_column_means=True,  # Rescale results to match original means
)
```

### Clustering Quality Metrics

After clustering, access quality metrics to evaluate the aggregation accuracy:

```python
fs_clustered = flow_system.transform.cluster(n_clusters=8, cluster_duration='1D')

# Access clustering metrics (xr.Dataset)
metrics = fs_clustered.clustering.metrics
print(metrics)  # Shows RMSE, MAE, etc. per time series

# Access specific metric
rmse = metrics['RMSE']  # xr.DataArray with dims [time_series, period?, scenario?]
```

## Storage Modes

Storage behavior during clustering is controlled via the `cluster_mode` parameter:

```python
storage = fx.Storage(
    'SeasonalPit',
    capacity_in_flow_hours=5000,
    cluster_mode='intercluster_cyclic',  # Default
    ...
)
```

### Available Modes

| Mode | Behavior | Best For |
|------|----------|----------|
| `'intercluster_cyclic'` | Links storage across clusters + yearly cycling | Seasonal storage (pit, underground) |
| `'intercluster'` | Links storage across clusters, free start/end | Multi-year optimization |
| `'cyclic'` | Each cluster independent, but start = end | Daily storage (battery, hot water tank) |
| `'independent'` | Each cluster fully independent | Quick estimates, debugging |

### How Inter-Cluster Linking Works

For `'intercluster'` and `'intercluster_cyclic'` modes, the optimizer tracks:

1. **`SOC_boundary`**: Absolute state-of-charge at the start of each original period
2. **`charge_state`**: Relative change (ΔE) within each typical period

During expansion, these combine with self-discharge decay:

```text
actual_SOC(t) = SOC_boundary[period] × (1 - loss)^t + ΔE(t)
```

This enables accurate modeling of seasonal storage that charges in summer and discharges in winter.

### Choosing the Right Mode

```python
# Seasonal pit storage - needs yearly linking
pit_storage = fx.Storage(
    'SeasonalPit',
    cluster_mode='intercluster_cyclic',
    capacity_in_flow_hours=10000,
    relative_loss_per_hour=0.0001,
    ...
)

# Daily hot water tank - only needs daily cycling
tank = fx.Storage(
    'HotWaterTank',
    cluster_mode='cyclic',
    capacity_in_flow_hours=50,
    ...
)

# Battery with quick estimate
battery = fx.Storage(
    'Battery',
    cluster_mode='independent',  # Fastest, ignores long-term effects
    ...
)
```

## Multi-Dimensional Support

Clustering works with periods and scenarios:

```python
# FlowSystem with multiple periods and scenarios
flow_system = fx.FlowSystem(
    timesteps,
    periods=pd.Index([2025, 2030, 2035], name='period'),
    scenarios=pd.Index(['low', 'base', 'high'], name='scenario'),
)

# Cluster - dimensions are preserved
fs_clustered = flow_system.transform.cluster(
    n_clusters=8,
    cluster_duration='1D',
)

# Solution has all dimensions
# Dims: (time, cluster, period, scenario)
flow_rate = fs_clustered.solution['Boiler(Q_th)|flow_rate']
```

## Expanding Solutions

After optimization, expand results back to full resolution:

```python
fs_expanded = fs_clustered.transform.expand()

# Full timesteps are restored
print(f"Original: {len(flow_system.timesteps)} timesteps")
print(f"Clustered: {len(fs_clustered.timesteps)} timesteps")
print(f"Expanded: {len(fs_expanded.timesteps)} timesteps")

# Storage charge state correctly reconstructed
charge_state = fs_expanded.solution['Storage|charge_state']
```

The expansion:

1. Maps each original timestep to its assigned cluster
2. For storage with inter-cluster linking, combines `SOC_boundary` with within-cluster `charge_state`
3. Applies self-discharge decay factors

## Performance Tips

### Cluster Count Selection

| Time Horizon   | Cluster Duration | Suggested n_clusters |
|----------------|------------------|---------------------|
| 1 year         | 1 day | 8-16 |
| 1 year         | 1 week | 4-8 |
| Multiple years | 1 day | 12-24 |

### Speed vs Accuracy Trade-off

```python
# Fast (less accurate) - for quick estimates
fs_fast = flow_system.transform.cluster(n_clusters=4, cluster_duration='1D')

# Balanced - typical production use
fs_balanced = flow_system.transform.cluster(n_clusters=12, cluster_duration='1D')

# Accurate (slower) - for final results
fs_accurate = flow_system.transform.cluster(n_clusters=24, cluster_duration='1D')
```

## See Also

- [Storage Component](../mathematical-notation/elements/Storage.md) - Storage mathematical formulation
- [Notebooks: Clustering](../../notebooks/08c-clustering.ipynb) - Interactive examples
- [Notebooks: Storage Modes](../../notebooks/08c2-clustering-storage-modes.ipynb) - Storage mode comparison
