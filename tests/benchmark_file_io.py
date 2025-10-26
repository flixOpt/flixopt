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
import plotly.graph_objects as go
import xarray as xr
import yaml
from plotly.subplots import make_subplots


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


def create_plots(results):
    """Create interactive Plotly visualizations of benchmark results."""

    # Extract data for plotting
    config_names = [c['name'] for c in results['configurations']]

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            'Write Time Comparison',
            'Read Time Comparison',
            'File Size with Compression',
            'Throughput Comparison',
        ),
        specs=[[{'secondary_y': False}, {'secondary_y': False}], [{'secondary_y': False}, {'secondary_y': False}]],
    )

    colors = {'h5netcdf': '#1f77b4', 'netcdf4': '#ff7f0e'}

    compression_levels = [0, 4, 9]
    engines = ['h5netcdf', 'netcdf4']

    # Plot 1: Write times (compression level 4)
    for engine in engines:
        write_times = []
        for config_name in config_names:
            try:
                time_val = results['benchmarks'][config_name]['compression_4'][engine]['write_time_seconds']
                write_times.append(time_val)
            except KeyError:
                write_times.append(None)

        fig.add_trace(
            go.Scatter(
                x=config_names,
                y=write_times,
                name=engine,
                mode='lines+markers',
                line=dict(color=colors[engine]),
                legendgroup=engine,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Plot 2: Read times (compression level 4)
    for engine in engines:
        read_times = []
        for config_name in config_names:
            try:
                time_val = results['benchmarks'][config_name]['compression_4'][engine]['read_time_seconds']
                read_times.append(time_val)
            except KeyError:
                read_times.append(None)

        fig.add_trace(
            go.Scatter(
                x=config_names,
                y=read_times,
                name=engine,
                mode='lines+markers',
                line=dict(color=colors[engine]),
                legendgroup=engine,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Plot 3: File sizes for different compression levels
    for comp_level in compression_levels:
        for engine in engines:
            file_sizes = []
            for config_name in config_names:
                try:
                    size_val = results['benchmarks'][config_name][f'compression_{comp_level}'][engine]['file_size_mb']
                    file_sizes.append(size_val)
                except KeyError:
                    file_sizes.append(None)

            fig.add_trace(
                go.Scatter(
                    x=config_names,
                    y=file_sizes,
                    name=f'{engine} (comp={comp_level})',
                    mode='lines+markers',
                    line=dict(color=colors[engine], dash=['solid', 'dash', 'dot'][comp_level // 4]),
                    legendgroup=f'{engine}_comp',
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

    # Plot 4: Write throughput (compression level 4)
    for engine in engines:
        throughputs = []
        for config_name in config_names:
            try:
                tp_val = results['benchmarks'][config_name]['compression_4'][engine]['write_throughput_mbps']
                throughputs.append(tp_val)
            except KeyError:
                throughputs.append(None)

        fig.add_trace(
            go.Scatter(
                x=config_names,
                y=throughputs,
                name=engine,
                mode='lines+markers',
                line=dict(color=colors[engine]),
                legendgroup=engine,
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    # Update axes labels
    fig.update_xaxes(title_text='Dataset Size', row=1, col=1)
    fig.update_xaxes(title_text='Dataset Size', row=1, col=2)
    fig.update_xaxes(title_text='Dataset Size', row=2, col=1)
    fig.update_xaxes(title_text='Dataset Size', row=2, col=2)

    fig.update_yaxes(title_text='Time (seconds)', row=1, col=1)
    fig.update_yaxes(title_text='Time (seconds)', row=1, col=2)
    fig.update_yaxes(title_text='File Size (MB)', row=2, col=1)
    fig.update_yaxes(title_text='Throughput (MB/s)', row=2, col=2)

    # Update layout
    fig.update_layout(
        height=800, title_text='NetCDF Engine Benchmark: h5netcdf vs netcdf4', showlegend=True, hovermode='x unified'
    )

    return fig


def run_benchmark():
    """Run the complete benchmark for multiple dataset sizes."""

    # Define different dataset configurations to test
    # (time_steps, n_vars, name)
    dataset_configs = [
        (100, 20, 'tiny'),
        (1000, 50, 'small'),
        (5000, 100, 'medium'),
        (10000, 200, 'large'),
        (20000, 300, 'xlarge'),
        (80000, 300, 'xxlarge'),
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

    # Create and save plots
    fig = create_plots(results)
    fig.write_html('netcdf_benchmark_plots.html')


if __name__ == '__main__':
    main()
