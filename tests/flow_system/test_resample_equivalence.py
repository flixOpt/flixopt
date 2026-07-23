"""
Tests to ensure the dimension grouping optimization in _resample_by_dimension_groups
is equivalent to naive Dataset resampling.

These tests verify that the optimization (grouping variables by dimensions before
resampling) produces identical results to simply calling Dataset.resample() directly.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt.transform_accessor import TransformAccessor


def naive_dataset_resample(dataset: xr.Dataset, freq: str, method: str) -> xr.Dataset:
    """
    Naive resampling: simply call Dataset.resample().method() directly.

    This is the straightforward approach without dimension grouping optimization.
    """
    return getattr(dataset.resample(time=freq), method)()


def create_dataset_with_mixed_dimensions(n_timesteps=48, seed=42):
    """
    Create a dataset with variables having different dimension structures.

    This mimics realistic data with:
    - Variables with only time dimension
    - Variables with time + one other dimension
    - Variables with time + multiple dimensions
    """
    np.random.seed(seed)
    timesteps = pd.date_range('2020-01-01', periods=n_timesteps, freq='h')

    ds = xr.Dataset(
        coords={
            'time': timesteps,
            'component': ['comp1', 'comp2'],
            'bus': ['bus1', 'bus2'],
            'scenario': ['base', 'alt'],
        }
    )

    # Variable with only time dimension
    ds['total_demand'] = xr.DataArray(
        np.random.randn(n_timesteps),
        dims=['time'],
        coords={'time': ds.time},
    )

    # Variable with time + component
    ds['component_flow'] = xr.DataArray(
        np.random.randn(n_timesteps, 2),
        dims=['time', 'component'],
        coords={'time': ds.time, 'component': ds.component},
    )

    # Variable with time + bus
    ds['bus_balance'] = xr.DataArray(
        np.random.randn(n_timesteps, 2),
        dims=['time', 'bus'],
        coords={'time': ds.time, 'bus': ds.bus},
    )

    # Variable with time + component + bus
    ds['flow_on_bus'] = xr.DataArray(
        np.random.randn(n_timesteps, 2, 2),
        dims=['time', 'component', 'bus'],
        coords={'time': ds.time, 'component': ds.component, 'bus': ds.bus},
    )

    # Variable with time + scenario
    ds['scenario_demand'] = xr.DataArray(
        np.random.randn(n_timesteps, 2),
        dims=['time', 'scenario'],
        coords={'time': ds.time, 'scenario': ds.scenario},
    )

    # Variable with time + component + scenario
    ds['component_scenario_flow'] = xr.DataArray(
        np.random.randn(n_timesteps, 2, 2),
        dims=['time', 'component', 'scenario'],
        coords={'time': ds.time, 'component': ds.component, 'scenario': ds.scenario},
    )

    return ds


@pytest.mark.parametrize('method', ['mean', 'sum', 'max', 'min', 'first', 'last'])
@pytest.mark.parametrize('freq', ['2h', '4h', '1D'])
def test_resample_equivalence_mixed_dimensions(method, freq):
    """
    Test that _resample_by_dimension_groups produces same results as naive resampling.

    Uses a dataset with variables having different dimension structures.
    """
    ds = create_dataset_with_mixed_dimensions(n_timesteps=100)

    # Method 1: Optimized approach (with dimension grouping)
    result_optimized = TransformAccessor._resample_by_dimension_groups(ds, freq, method)

    # Method 2: Naive approach (direct Dataset resampling)
    result_naive = naive_dataset_resample(ds, freq, method)

    # Compare results
    xr.testing.assert_allclose(result_optimized, result_naive)


@pytest.mark.parametrize('method', ['mean', 'sum', 'max', 'min', 'first', 'last', 'std', 'var', 'median'])
def test_resample_equivalence_single_dimension(method):
    """
    Test with variables having only time dimension.
    """
    timesteps = pd.date_range('2020-01-01', periods=48, freq='h')

    ds = xr.Dataset(coords={'time': timesteps})
    ds['var1'] = xr.DataArray(np.random.randn(48), dims=['time'], coords={'time': ds.time})
    ds['var2'] = xr.DataArray(np.random.randn(48) * 10, dims=['time'], coords={'time': ds.time})
    ds['var3'] = xr.DataArray(np.random.randn(48) / 5, dims=['time'], coords={'time': ds.time})

    # Optimized approach
    result_optimized = TransformAccessor._resample_by_dimension_groups(ds, '2h', method)

    # Naive approach
    result_naive = naive_dataset_resample(ds, '2h', method)

    # Compare results
    xr.testing.assert_allclose(result_optimized, result_naive)


def test_resample_equivalence_empty_dataset():
    """
    Test with an empty dataset (edge case).
    """
    timesteps = pd.date_range('2020-01-01', periods=48, freq='h')
    ds = xr.Dataset(coords={'time': timesteps})

    # Both should handle empty dataset gracefully
    result_optimized = TransformAccessor._resample_by_dimension_groups(ds, '2h', 'mean')
    result_naive = naive_dataset_resample(ds, '2h', 'mean')

    xr.testing.assert_allclose(result_optimized, result_naive)


def test_resample_equivalence_single_variable():
    """
    Test with a single variable.
    """
    timesteps = pd.date_range('2020-01-01', periods=48, freq='h')
    ds = xr.Dataset(coords={'time': timesteps})
    ds['single_var'] = xr.DataArray(np.random.randn(48), dims=['time'], coords={'time': ds.time})

    # Test multiple methods
    for method in ['mean', 'sum', 'max', 'min']:
        result_optimized = TransformAccessor._resample_by_dimension_groups(ds, '3h', method)
        result_naive = naive_dataset_resample(ds, '3h', method)

        xr.testing.assert_allclose(result_optimized, result_naive)


def test_resample_equivalence_with_nans():
    """
    Test with NaN values to ensure they're handled consistently.
    """
    timesteps = pd.date_range('2020-01-01', periods=48, freq='h')

    ds = xr.Dataset(coords={'time': timesteps, 'component': ['a', 'b']})

    # Create variable with some NaN values
    data = np.random.randn(48, 2)
    data[5:10, 0] = np.nan
    data[20:25, 1] = np.nan

    ds['var_with_nans'] = xr.DataArray(
        data, dims=['time', 'component'], coords={'time': ds.time, 'component': ds.component}
    )

    # Test with methods that handle NaNs
    for method in ['mean', 'sum', 'max', 'min', 'first', 'last']:
        result_optimized = TransformAccessor._resample_by_dimension_groups(ds, '2h', method)
        result_naive = naive_dataset_resample(ds, '2h', method)

        xr.testing.assert_allclose(result_optimized, result_naive)


def test_resample_equivalence_different_dimension_orders():
    """
    Test that dimension order doesn't affect the equivalence.
    """
    timesteps = pd.date_range('2020-01-01', periods=48, freq='h')

    ds = xr.Dataset(
        coords={
            'time': timesteps,
            'x': ['x1', 'x2'],
            'y': ['y1', 'y2'],
        }
    )

    # Variable with time first
    ds['var_time_first'] = xr.DataArray(
        np.random.randn(48, 2, 2),
        dims=['time', 'x', 'y'],
        coords={'time': ds.time, 'x': ds.x, 'y': ds.y},
    )

    # Variable with time in middle
    ds['var_time_middle'] = xr.DataArray(
        np.random.randn(2, 48, 2),
        dims=['x', 'time', 'y'],
        coords={'x': ds.x, 'time': ds.time, 'y': ds.y},
    )

    # Variable with time last
    ds['var_time_last'] = xr.DataArray(
        np.random.randn(2, 2, 48),
        dims=['x', 'y', 'time'],
        coords={'x': ds.x, 'y': ds.y, 'time': ds.time},
    )

    for method in ['mean', 'sum', 'max', 'min']:
        result_optimized = TransformAccessor._resample_by_dimension_groups(ds, '2h', method)
        result_naive = naive_dataset_resample(ds, '2h', method)

        xr.testing.assert_allclose(result_optimized, result_naive)


def test_resample_equivalence_multiple_variables_same_dims():
    """
    Test with multiple variables sharing the same dimensions.

    This is the key optimization case - variables with same dims should be
    grouped and resampled together.
    """
    timesteps = pd.date_range('2020-01-01', periods=48, freq='h')

    ds = xr.Dataset(coords={'time': timesteps, 'location': ['A', 'B', 'C']})

    # Multiple variables with same dimensions (time, location)
    for i in range(3):
        ds[f'var_{i}'] = xr.DataArray(
            np.random.randn(48, 3),
            dims=['time', 'location'],
            coords={'time': ds.time, 'location': ds.location},
        )

    for method in ['mean', 'sum', 'max', 'min']:
        result_optimized = TransformAccessor._resample_by_dimension_groups(ds, '2h', method)
        result_naive = naive_dataset_resample(ds, '2h', method)

        xr.testing.assert_allclose(result_optimized, result_naive)


def test_resample_equivalence_large_dataset():
    """
    Test with a larger, more realistic dataset.
    """
    timesteps = pd.date_range('2020-01-01', periods=168, freq='h')  # One week

    ds = xr.Dataset(
        coords={
            'time': timesteps,
            'component': [f'comp_{i}' for i in range(5)],
            'bus': [f'bus_{i}' for i in range(3)],
        }
    )

    # Various variable types
    ds['simple_var'] = xr.DataArray(np.random.randn(168), dims=['time'], coords={'time': ds.time})
    ds['component_var'] = xr.DataArray(
        np.random.randn(168, 5), dims=['time', 'component'], coords={'time': ds.time, 'component': ds.component}
    )
    ds['bus_var'] = xr.DataArray(np.random.randn(168, 3), dims=['time', 'bus'], coords={'time': ds.time, 'bus': ds.bus})
    ds['complex_var'] = xr.DataArray(
        np.random.randn(168, 5, 3),
        dims=['time', 'component', 'bus'],
        coords={'time': ds.time, 'component': ds.component, 'bus': ds.bus},
    )

    # Test with a subset of methods (to keep test time reasonable)
    for method in ['mean', 'sum', 'first']:
        result_optimized = TransformAccessor._resample_by_dimension_groups(ds, '1D', method)
        result_naive = naive_dataset_resample(ds, '1D', method)

        xr.testing.assert_allclose(result_optimized, result_naive)


def test_resample_equivalence_with_kwargs():
    """
    Test that kwargs are properly forwarded to resample().

    Verifies that additional arguments like label and closed are correctly
    passed through the optimization path.
    """
    timesteps = pd.date_range('2020-01-01', periods=48, freq='h')
    ds = xr.Dataset(coords={'time': timesteps})
    ds['var'] = xr.DataArray(np.random.randn(48), dims=['time'], coords={'time': ds.time})

    kwargs = {'label': 'right', 'closed': 'right'}
    result_optimized = TransformAccessor._resample_by_dimension_groups(ds, '2h', 'mean', **kwargs)
    result_naive = ds.resample(time='2h', **kwargs).mean()

    xr.testing.assert_allclose(result_optimized, result_naive)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
