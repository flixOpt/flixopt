#!/usr/bin/env python3
"""
Benchmark comparing h5netcdf vs netcdf4 for file I/O with compression.
Tests with large xarray datasets (300 variables, 80,000 timesteps).
"""

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import xarray as xr
import yaml


def create_dataset(time_steps, n_vars):
    """Create a dataset with specified dimensions. Each variable is 1D (time only)."""
    # Create coordinate array
    time_coords = np.arange(time_steps)

    # Create dataset with n_vars variables, each 1D
    np.random.seed(42)
    data_vars = {}

    for i in range(n_vars):
        # Generate random 1D data (time series)
        data = np.random.randn(time_steps).astype(np.float32)
        var_name = f'var_{i:03d}'
        data_vars[var_name] = (['time'], data)

    ds = xr.Dataset(
        data_vars,
        coords={
            'time': time_coords,
        },
    )

    ds.attrs['description'] = f'Test dataset: {n_vars} vars, {time_steps} timesteps'

    return ds


def benchmark_write(ds, filepath, engine, compression_level=4):
    """Benchmark write performance."""
    encoding = {var: {'zlib': True, 'complevel': compression_level} for var in ds.data_vars}

    start = time.perf_counter()
    ds.to_netcdf(filepath, engine=engine, encoding=encoding)
    elapsed = time.perf_counter() - start

    file_size = os.path.getsize(filepath)

    return elapsed, file_size


def benchmark_read(filepath, engine):
    """Benchmark read performance."""
    start = time.perf_counter()
    ds = xr.open_dataset(filepath, engine=engine)
    ds.load()
    ds.close()
    elapsed = time.perf_counter() - start

    return elapsed


def run_benchmark():
    """Run the complete benchmark for multiple dataset sizes."""

    # Define different dataset configurations to test
    # (time_steps, n_vars, name)
    dataset_configs = [
        (100, 20, 'tiny'),
        (1000, 50, 'small'),
        (5000, 100, 'medium'),
        (10000, 200, 'large'),
        # (20000, 300, 'xlarge'),
        # (80000, 300, 'xxlarge'),
    ]

    # Test compression levels
    compression_levels = [0, 4, 9]
    engines = ['h5netcdf', 'netcdf4']

    results = {'configurations': [], 'benchmarks': {}}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for time_steps, n_vars, config_name in dataset_configs:
            ds = create_dataset(time_steps, n_vars)

            # Calculate uncompressed size
            uncompressed_size = sum(ds[var].nbytes for var in ds.data_vars)

            config_info = {
                'name': config_name,
                'time_steps': time_steps,
                'num_variables': n_vars,
                'dimensions': dict(ds.dims),
                'uncompressed_size_mb': round(uncompressed_size / (1024 * 1024), 2),
            }
            results['configurations'].append(config_info)

            results['benchmarks'][config_name] = {}

            for comp_level in compression_levels:
                comp_key = f'compression_{comp_level}'
                results['benchmarks'][config_name][comp_key] = {}

                for engine in engines:
                    filepath = tmpdir / f'test_{config_name}_{engine}_comp{comp_level}.nc'

                    try:
                        # Write benchmark
                        write_time, file_size = benchmark_write(ds, filepath, engine, comp_level)

                        # Read benchmark
                        read_time = benchmark_read(filepath, engine)

                        results['benchmarks'][config_name][comp_key][engine] = {
                            'write_time_seconds': round(write_time, 3),
                            'read_time_seconds': round(read_time, 3),
                            'file_size_mb': round(file_size / (1024 * 1024), 2),
                            'compression_ratio': round(uncompressed_size / file_size, 2),
                            'write_throughput_mbps': round((uncompressed_size / (1024 * 1024)) / write_time, 2)
                            if write_time > 0
                            else 0,
                            'read_throughput_mbps': round((uncompressed_size / (1024 * 1024)) / read_time, 2)
                            if read_time > 0
                            else 0,
                        }

                    except Exception as e:
                        results['benchmarks'][config_name][comp_key][engine] = {'error': str(e)}

                    # Clean up
                    if filepath.exists():
                        filepath.unlink()

            # Clean up dataset
            del ds

    return results


def main():
    results = run_benchmark()

    # Write results to YAML
    output_file = 'netcdf_benchmark_results.yaml'

    with open(output_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)


if __name__ == '__main__':
    main()
