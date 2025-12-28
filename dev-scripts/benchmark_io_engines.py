#!/usr/bin/env python3
"""
Benchmark comparison of netcdf4 vs h5netcdf IO engines.

Tests read/write performance across different dataset sizes to validate
the switch from netcdf4 to h5netcdf for flixopt's NetCDF IO operations.

Results stored as xarray Dataset.

Usage:
    uv run python dev-scripts/benchmark_io_engines.py --sizes 100 500 1000 5000 8760 --repeats 5
    uv run python dev-scripts/benchmark_io_engines.py --plot dev-scripts/benchmark_io_engines.html
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Callable


def build_dataset(n_timesteps: int, n_variables: int = 20) -> xr.Dataset:
    """
    Build a dataset similar to flixopt FlowSystem data.

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps (e.g., 8760 for 1 year hourly).
    n_variables : int
        Number of data variables to include.

    Returns
    -------
    xr.Dataset
        Dataset with realistic structure.
    """
    rng = np.random.default_rng(42)

    coords = {
        'time': np.arange(n_timesteps),
        'component': [f'Component_{i}' for i in range(5)],
    }

    data_vars = {}

    # Add 1D time series variables (most common in flixopt)
    for i in range(n_variables // 2):
        data_vars[f'flow_{i}'] = xr.DataArray(
            rng.random(n_timesteps) * 100,
            dims=['time'],
            coords={'time': coords['time']},
            attrs={'units': 'kW', 'description': f'Flow variable {i}'},
        )

    # Add 2D variables (time x component)
    for i in range(n_variables // 4):
        data_vars[f'state_{i}'] = xr.DataArray(
            rng.random((n_timesteps, 5)) * 50,
            dims=['time', 'component'],
            coords=coords,
            attrs={'units': 'kWh', 'description': f'State variable {i}'},
        )

    # Add scalar variables
    for i in range(n_variables // 4):
        data_vars[f'param_{i}'] = xr.DataArray(
            rng.random() * 1000,
            attrs={'units': 'EUR', 'description': f'Parameter {i}'},
        )

    ds = xr.Dataset(
        data_vars,
        attrs={
            'title': 'Benchmark Dataset',
            'n_timesteps': n_timesteps,
            'n_variables': n_variables,
        },
    )

    return ds


def time_function(func: Callable[[], Any], repeats: int, warmup: int = 1) -> np.ndarray:
    """Time a function over multiple iterations."""
    for _ in range(warmup):
        func()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)

    return np.array(times)


def run_io_benchmark(
    timestep_sizes: list[int],
    n_variables: int = 20,
    repeats: int = 5,
    compression: int = 5,
) -> xr.Dataset:
    """
    Run IO benchmark comparing netcdf4 and h5netcdf engines.

    Parameters
    ----------
    timestep_sizes : list[int]
        List of timestep counts to test.
    n_variables : int
        Number of variables per dataset.
    repeats : int
        Number of timing repetitions.
    compression : int
        Compression level (0-9).

    Returns
    -------
    xr.Dataset
        Dataset with timing results.
    """
    engines = ['netcdf4', 'h5netcdf']
    operations = ['write', 'read']

    results: dict[tuple[str, str, int], np.ndarray] = {}
    file_sizes: dict[tuple[str, int], int] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for n_timesteps in timestep_sizes:
            print(f'\nBuilding dataset with {n_timesteps} timesteps, {n_variables} variables...')
            ds = build_dataset(n_timesteps, n_variables)

            total_elements = sum(np.prod(var.shape) if var.shape else 1 for var in ds.data_vars.values())
            print(f'  Total elements: {total_elements:,}')

            for engine in engines:
                filepath = Path(tmpdir) / f'benchmark_{engine}_{n_timesteps}.nc'

                # Benchmark write
                encoding = (
                    None
                    if compression == 0
                    else {var: {'zlib': True, 'complevel': compression} for var in ds.data_vars}
                )

                def write_func(ds=ds, fp=filepath, eng=engine, enc=encoding):
                    ds.to_netcdf(fp, engine=eng, encoding=enc)

                print(f'  Benchmarking {engine} write...')
                try:
                    write_times = time_function(write_func, repeats)
                    results[('write', engine, n_timesteps)] = write_times

                    # Record file size after write
                    file_sizes[(engine, n_timesteps)] = filepath.stat().st_size
                except Exception as e:
                    print(f'    ERROR: {e}')
                    results[('write', engine, n_timesteps)] = np.full(repeats, np.nan)
                    file_sizes[(engine, n_timesteps)] = 0

                # Benchmark read
                def read_func(fp=filepath, eng=engine):
                    return xr.load_dataset(fp, engine=eng)

                print(f'  Benchmarking {engine} read...')
                try:
                    read_times = time_function(read_func, repeats)
                    results[('read', engine, n_timesteps)] = read_times
                except Exception as e:
                    print(f'    ERROR: {e}')
                    results[('read', engine, n_timesteps)] = np.full(repeats, np.nan)

    # Build xarray Dataset
    data_vars = {}

    for op in operations:
        for engine in engines:
            var_name = f'{op}_{engine}'
            data = np.array([results[(op, engine, n)] for n in timestep_sizes])
            data_vars[var_name] = xr.DataArray(
                data * 1000,  # Convert to milliseconds
                dims=['n_timesteps', 'repeat'],
                coords={
                    'n_timesteps': timestep_sizes,
                    'repeat': range(repeats),
                },
                attrs={
                    'units': 'ms',
                    'description': f'{op.capitalize()} operation',
                    'engine': engine,
                },
            )

    # Add file sizes
    for engine in engines:
        sizes = [file_sizes.get((engine, n), 0) for n in timestep_sizes]
        data_vars[f'file_size_{engine}'] = xr.DataArray(
            np.array(sizes) / 1024,  # Convert to KB
            dims=['n_timesteps'],
            coords={'n_timesteps': timestep_sizes},
            attrs={'units': 'KB', 'engine': engine},
        )

    # Add speedup/slowdown calculations (h5netcdf relative to netcdf4)
    for op in operations:
        netcdf4_median = data_vars[f'{op}_netcdf4'].median(dim='repeat')
        h5netcdf_median = data_vars[f'{op}_h5netcdf'].median(dim='repeat')
        # Ratio > 1 means h5netcdf is slower, < 1 means h5netcdf is faster
        data_vars[f'{op}_ratio'] = h5netcdf_median / netcdf4_median
        data_vars[f'{op}_ratio'].attrs = {
            'description': f'h5netcdf/netcdf4 ratio for {op} (<1 = h5netcdf faster)',
        }

    ds_results = xr.Dataset(data_vars)
    ds_results.attrs['n_variables'] = n_variables
    ds_results.attrs['repeats'] = repeats
    ds_results.attrs['compression'] = compression

    return ds_results


OPERATIONS = ['write', 'read']
OP_LABELS = {
    'write': 'Write',
    'read': 'Read',
}


def print_results(ds: xr.Dataset) -> None:
    """Print benchmark results in a formatted table."""
    n_timesteps_values = ds.coords['n_timesteps'].values

    print('\n' + '=' * 90)
    print('IO ENGINE BENCHMARK: netcdf4 vs h5netcdf')
    print('=' * 90)

    for op in OPERATIONS:
        print(f'\n{"-" * 90}')
        print(f'{OP_LABELS[op]} Performance')
        print('-' * 90)
        print(f'{"Timesteps":>12s} {"netcdf4 (ms)":>15s} {"h5netcdf (ms)":>15s} {"Ratio":>10s} {"Winner":>15s}')
        print('-' * 90)

        netcdf4 = ds[f'{op}_netcdf4'].median(dim='repeat')
        h5netcdf = ds[f'{op}_h5netcdf'].median(dim='repeat')
        ratio = ds[f'{op}_ratio']

        for i, n in enumerate(n_timesteps_values):
            nc4_val = float(netcdf4[i])
            h5_val = float(h5netcdf[i])
            r = float(ratio[i])

            if r < 1:
                winner = f'h5netcdf {1 / r:.1f}x faster'
            elif r > 1:
                winner = f'netcdf4 {r:.1f}x faster'
            else:
                winner = 'tie'

            print(f'{n:>12d} {nc4_val:>15.2f} {h5_val:>15.2f} {r:>10.2f} {winner:>15s}')

    # File size comparison
    print(f'\n{"-" * 90}')
    print('File Size Comparison')
    print('-' * 90)
    print(f'{"Timesteps":>12s} {"netcdf4 (KB)":>15s} {"h5netcdf (KB)":>15s} {"Difference":>15s}')
    print('-' * 90)

    for i, n in enumerate(n_timesteps_values):
        nc4_size = float(ds['file_size_netcdf4'][i])
        h5_size = float(ds['file_size_h5netcdf'][i])
        diff_pct = (h5_size - nc4_size) / nc4_size * 100 if nc4_size > 0 else 0
        print(f'{n:>12d} {nc4_size:>15.1f} {h5_size:>15.1f} {diff_pct:>+14.1f}%')

    # Summary
    print('\n' + '=' * 90)
    print('SUMMARY (at largest dataset size)')
    print('=' * 90)

    max_idx = len(n_timesteps_values) - 1
    for op in OPERATIONS:
        ratio = float(ds[f'{op}_ratio'][max_idx])
        if ratio < 1:
            print(f'  {OP_LABELS[op]}: h5netcdf is {1 / ratio:.1f}x FASTER')
        else:
            print(f'  {OP_LABELS[op]}: netcdf4 is {ratio:.1f}x faster')


def plot_results(ds: xr.Dataset, output_path: str = 'benchmark_io_engines.html') -> None:
    """Create comparison plots."""
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_timesteps_values = ds.coords['n_timesteps'].values

    # Build dataframe for plotting
    rows = []
    for op in OPERATIONS:
        for engine in ['netcdf4', 'h5netcdf']:
            for n in n_timesteps_values:
                median_time = float(ds[f'{op}_{engine}'].median(dim='repeat').sel(n_timesteps=n))
                rows.append(
                    {
                        'operation': OP_LABELS[op],
                        'engine': engine,
                        'n_timesteps': int(n),
                        'time_ms': median_time,
                    }
                )
    df = pd.DataFrame(rows)

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=('Write Performance', 'Read Performance', 'Performance Ratio', 'File Size'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}], [{'type': 'scatter'}, {'type': 'bar'}]],
    )

    colors = {'netcdf4': '#EF553B', 'h5netcdf': '#636EFA'}

    # Write performance
    for engine in ['netcdf4', 'h5netcdf']:
        df_engine = df[(df['operation'] == 'Write') & (df['engine'] == engine)]
        fig.add_trace(
            go.Scatter(
                x=df_engine['n_timesteps'],
                y=df_engine['time_ms'],
                mode='lines+markers',
                name=f'{engine} (write)',
                line=dict(color=colors[engine]),
                legendgroup=engine,
            ),
            row=1,
            col=1,
        )

    # Read performance
    for engine in ['netcdf4', 'h5netcdf']:
        df_engine = df[(df['operation'] == 'Read') & (df['engine'] == engine)]
        fig.add_trace(
            go.Scatter(
                x=df_engine['n_timesteps'],
                y=df_engine['time_ms'],
                mode='lines+markers',
                name=f'{engine} (read)',
                line=dict(color=colors[engine], dash='dash'),
                legendgroup=engine,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Ratio plot
    for op in OPERATIONS:
        ratio = ds[f'{op}_ratio'].values
        fig.add_trace(
            go.Scatter(
                x=list(n_timesteps_values),
                y=list(ratio),
                mode='lines+markers',
                name=f'{OP_LABELS[op]} ratio',
            ),
            row=2,
            col=1,
        )

    # Add reference line at ratio=1
    fig.add_hline(y=1, line_dash='dash', line_color='gray', row=2, col=1)
    fig.add_annotation(
        x=n_timesteps_values[-1],
        y=1,
        text='Equal performance',
        showarrow=False,
        row=2,
        col=1,
        yshift=10,
    )

    # File size comparison (bar chart at largest size)
    max_n = n_timesteps_values[-1]
    sizes = [
        float(ds['file_size_netcdf4'].sel(n_timesteps=max_n)),
        float(ds['file_size_h5netcdf'].sel(n_timesteps=max_n)),
    ]
    fig.add_trace(
        go.Bar(
            x=['netcdf4', 'h5netcdf'], y=sizes, marker_color=[colors['netcdf4'], colors['h5netcdf']], showlegend=False
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_xaxes(title_text='Timesteps', row=1, col=1)
    fig.update_xaxes(title_text='Timesteps', row=1, col=2)
    fig.update_xaxes(title_text='Timesteps', row=2, col=1)
    fig.update_xaxes(title_text='Engine', row=2, col=2)

    fig.update_yaxes(title_text='Time (ms)', row=1, col=1)
    fig.update_yaxes(title_text='Time (ms)', row=1, col=2)
    fig.update_yaxes(title_text='h5netcdf / netcdf4', row=2, col=1)
    fig.update_yaxes(title_text='File Size (KB)', row=2, col=2)

    fig.update_layout(
        height=700,
        width=1000,
        title_text='NetCDF IO Engine Benchmark: netcdf4 vs h5netcdf',
        showlegend=True,
    )

    fig.write_html(output_path)
    print(f'\nPlot saved to {output_path}')


def main() -> xr.Dataset:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        default=[100, 500, 1000, 2000, 5000, 8760],
        help='Timestep sizes to test (default: 100 500 1000 2000 5000 8760)',
    )
    parser.add_argument(
        '--n-variables',
        type=int,
        default=20,
        help='Number of variables per dataset (default: 20)',
    )
    parser.add_argument(
        '--repeats',
        type=int,
        default=5,
        help='Number of repetitions (default: 5)',
    )
    parser.add_argument(
        '--compression',
        type=int,
        default=5,
        help='Compression level 0-9 (default: 5)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output NetCDF file for results',
    )
    parser.add_argument(
        '--plot',
        type=str,
        default=None,
        help='Output HTML plot file',
    )
    args = parser.parse_args()

    print('Running IO engine benchmark...')
    print(f'  Timestep sizes: {args.sizes}')
    print(f'  Variables: {args.n_variables}')
    print(f'  Repeats: {args.repeats}')
    print(f'  Compression: {args.compression}')

    ds = run_io_benchmark(
        timestep_sizes=args.sizes,
        n_variables=args.n_variables,
        repeats=args.repeats,
        compression=args.compression,
    )

    print_results(ds)

    if args.output:
        ds.to_netcdf(args.output)
        print(f'\nResults saved to {args.output}')

    if args.plot:
        plot_results(ds, args.plot)

    return ds


if __name__ == '__main__':
    ds = main()
