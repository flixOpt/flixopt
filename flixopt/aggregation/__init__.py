"""
Time Series Aggregation Module for flixopt.

This module provides an abstraction layer for time series aggregation that
supports multiple backends while maintaining proper handling of multi-dimensional
data (period, scenario dimensions).

Available backends:
- TSAMBackend: Uses tsam package for k-means clustering into typical periods
- ManualBackend: Accepts user-provided mapping/weights for external aggregation

Key classes:
- AggregationResult: Universal result container from any aggregation backend
- ClusterStructure: Hierarchical structure info for storage inter-period linking
- Aggregator: Protocol that all backends implement

Example usage:

    # Using TSAM backend
    from flixopt.aggregation import TSAMBackend

    backend = TSAMBackend(cluster_duration='1D', n_segments=4)
    result = backend.aggregate(data, n_representatives=8)

    # Using manual/external aggregation (PyPSA-style)
    from flixopt.aggregation import ManualBackend
    import xarray as xr

    backend = ManualBackend(
        timestep_mapping=xr.DataArray(my_mapping, dims=['original_time']),
        representative_weights=xr.DataArray(my_weights, dims=['time']),
    )
    result = backend.aggregate(data)

    # Or via transform accessor
    fs_aggregated = fs.transform.aggregate(method='tsam', n_representatives=8)
    fs_aggregated = fs.transform.set_aggregation(my_mapping, my_weights)
"""

from .base import (
    AggregationInfo,
    AggregationResult,
    Aggregator,
    ClusterStructure,
    create_cluster_structure_from_mapping,
)
from .manual import (
    ManualBackend,
    create_manual_backend_from_labels,
    create_manual_backend_from_selection,
)

# Lazy import for InterClusterLinking to avoid circular imports
# It depends on structure.Submodel which has complex import dependencies
InterClusterLinking = None


def _get_inter_cluster_linking():
    """Get InterClusterLinking class with lazy import."""
    global InterClusterLinking
    if InterClusterLinking is None:
        from .storage_linking import InterClusterLinking as _InterClusterLinking

        InterClusterLinking = _InterClusterLinking
    return InterClusterLinking


# Conditional imports based on package availability
_BACKENDS = {'manual': ManualBackend}

try:
    from .tsam_backend import TSAMBackend, plot_aggregation

    _BACKENDS['tsam'] = TSAMBackend
except ImportError:
    # tsam not installed - TSAMBackend not available
    TSAMBackend = None
    plot_aggregation = None


def get_backend(name: str):
    """Get aggregation backend by name.

    Args:
        name: Backend name ('tsam', 'manual').

    Returns:
        Backend class.

    Raises:
        ValueError: If backend is not available.
    """
    if name not in _BACKENDS:
        available = list(_BACKENDS.keys())
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")

    backend_class = _BACKENDS[name]
    if backend_class is None:
        raise ImportError(
            f"Backend '{name}' is not available. Install required dependencies (e.g., 'pip install tsam' for TSAM)."
        )

    return backend_class


def list_backends() -> list[str]:
    """List available aggregation backends.

    Returns:
        List of backend names that are currently available.
    """
    return [name for name, cls in _BACKENDS.items() if cls is not None]


__all__ = [
    # Core classes
    'AggregationResult',
    'AggregationInfo',
    'ClusterStructure',
    'Aggregator',
    'InterClusterLinking',
    # Backends
    'TSAMBackend',
    'ManualBackend',
    # Utilities
    'create_cluster_structure_from_mapping',
    'create_manual_backend_from_labels',
    'create_manual_backend_from_selection',
    'plot_aggregation',
    'get_backend',
    'list_backends',
]
