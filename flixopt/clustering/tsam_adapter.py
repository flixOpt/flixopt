"""Adapter to convert tsam v3 results to xarray for flixopt.

This module provides conversion functions from tsam's AggregationResult
to xarray DataArrays/Datasets for flixopt's IO and clustering systems.

With the simplified Clustering structure, we only need:
- cluster_assignments (from result.cluster_assignments)
- cluster_weights (from result.cluster_weights)
- metrics (from result.accuracy)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from tsam import AggregationResult


def cluster_assignments_to_dataarray(result: AggregationResult) -> xr.DataArray:
    """Convert cluster_assignments to xr.DataArray.

    Args:
        result: tsam AggregationResult.

    Returns:
        DataArray with dims [original_cluster], values are cluster IDs (0 to n_clusters-1).
    """
    return xr.DataArray(
        np.array(result.cluster_assignments),
        dims=['original_cluster'],
        name='cluster_assignments',
    )


def cluster_weights_to_dataarray(result: AggregationResult) -> xr.DataArray:
    """Convert cluster_weights dict to xr.DataArray.

    Args:
        result: tsam AggregationResult.

    Returns:
        DataArray with dims [cluster], values are occurrence counts.
    """
    n_clusters = result.n_clusters
    weights = np.array([result.cluster_weights.get(c, 0) for c in range(n_clusters)])
    return xr.DataArray(weights, dims=['cluster'], name='cluster_weights')


def accuracy_to_dataset(result: AggregationResult) -> xr.Dataset:
    """Convert AccuracyMetrics to xarray Dataset.

    Args:
        result: tsam AggregationResult with accuracy property.

    Returns:
        Dataset with RMSE, MAE etc. as DataArrays with dims [time_series].
        Returns empty Dataset if no accuracy metrics available.
    """
    try:
        accuracy = result.accuracy
        data_vars = {}

        # tsam v3: accuracy.rmse and accuracy.mae are pandas Series
        for attr in ['rmse', 'mae', 'rmse_duration']:
            if hasattr(accuracy, attr):
                values = getattr(accuracy, attr)
                if isinstance(values, pd.Series):
                    data_vars[attr.upper()] = xr.DataArray(
                        values.values,
                        dims=['time_series'],
                        coords={'time_series': list(values.index)},
                    )
                elif isinstance(values, dict):
                    time_series = list(values.keys())
                    if time_series:
                        data_vars[attr.upper()] = xr.DataArray(
                            [values.get(ts, np.nan) for ts in time_series],
                            dims=['time_series'],
                            coords={'time_series': time_series},
                        )

        return xr.Dataset(data_vars) if data_vars else xr.Dataset()
    except Exception:
        pass
    return xr.Dataset()


def combine_results_multidim(
    results: dict[tuple, AggregationResult],
    periods: list,
    scenarios: list,
) -> dict[str, xr.DataArray]:
    """Combine per-(period, scenario) results into multi-dimensional DataArrays.

    Args:
        results: Dict mapping (period, scenario) tuples to AggregationResults.
        periods: List of period labels ([None] if no periods).
        scenarios: List of scenario labels ([None] if no scenarios).

    Returns:
        Dict with 'cluster_assignments' and 'cluster_weights' as DataArrays.
    """
    has_periods = periods != [None]
    has_scenarios = scenarios != [None]
    first_key = (periods[0], scenarios[0])

    # Simple case: no period/scenario dimensions
    if not has_periods and not has_scenarios:
        result = results[first_key]
        return {
            'cluster_assignments': cluster_assignments_to_dataarray(result),
            'cluster_weights': cluster_weights_to_dataarray(result),
        }

    # Multi-dimensional: build slices and concatenate
    slices = {
        'cluster_assignments': {},
        'cluster_weights': {},
    }

    for p in periods:
        for s in scenarios:
            key = (p, s)
            result = results[key]
            slices['cluster_assignments'][key] = cluster_assignments_to_dataarray(result)
            slices['cluster_weights'][key] = cluster_weights_to_dataarray(result)

    # Combine using xr.concat
    combined = {}
    for name, slice_dict in slices.items():
        combined[name] = _concat_slices(slice_dict, periods, scenarios, name)

    return combined


def _concat_slices(
    slices: dict[tuple, xr.DataArray],
    periods: list,
    scenarios: list,
    name: str,
) -> xr.DataArray:
    """Concatenate per-(period, scenario) slices into multi-dimensional DataArray."""
    has_periods = periods != [None]
    has_scenarios = scenarios != [None]

    if has_periods and has_scenarios:
        period_arrays = []
        for p in periods:
            scenario_arrays = [slices[(p, s)] for s in scenarios]
            period_arrays.append(xr.concat(scenario_arrays, dim=pd.Index(scenarios, name='scenario')))
        result = xr.concat(period_arrays, dim=pd.Index(periods, name='period'))
    elif has_periods:
        result = xr.concat([slices[(p, None)] for p in periods], dim=pd.Index(periods, name='period'))
    else:
        result = xr.concat([slices[(None, s)] for s in scenarios], dim=pd.Index(scenarios, name='scenario'))

    # Get base dimension from the first slice
    first_slice = next(iter(slices.values()))
    base_dim = first_slice.dims[0]
    return result.transpose(base_dim, ...).rename(name)


def combine_metrics_multidim(
    results: dict[tuple, AggregationResult],
    periods: list,
    scenarios: list,
) -> xr.Dataset:
    """Combine accuracy metrics from per-(period, scenario) results.

    Args:
        results: Dict mapping (period, scenario) tuples to AggregationResults.
        periods: List of period labels ([None] if no periods).
        scenarios: List of scenario labels ([None] if no scenarios).

    Returns:
        xr.Dataset with metrics having [time_series, period?, scenario?] dims.
    """
    has_periods = periods != [None]
    has_scenarios = scenarios != [None]

    # Get metrics from each slice
    metrics_slices: dict[tuple, xr.Dataset] = {}
    for key, result in results.items():
        metrics_slices[key] = accuracy_to_dataset(result)

    # If all empty, return empty
    if all(ds.sizes == {} for ds in metrics_slices.values()):
        return xr.Dataset()

    # Get sample to find variable names
    sample = next((ds for ds in metrics_slices.values() if ds.sizes), None)
    if sample is None:
        return xr.Dataset()

    # If single slice (no multi-dim), return directly
    if not has_periods and not has_scenarios:
        return metrics_slices[(periods[0], scenarios[0])]

    # Combine slices
    data_vars = {}
    for var_name in sample.data_vars:
        slices = {}
        for (p, s), ds in metrics_slices.items():
            if var_name in ds:
                slices[(p, s)] = ds[var_name]
            else:
                # Fill with NaN if missing
                slices[(p, s)] = sample[var_name].copy()
                slices[(p, s)].values[:] = np.nan

        data_vars[var_name] = _concat_slices(slices, periods, scenarios, var_name)

    return xr.Dataset(data_vars)
