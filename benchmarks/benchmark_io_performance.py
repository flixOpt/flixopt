"""Benchmark script for FlowSystem IO performance.

Tests to_dataset() and from_dataset() performance with large FlowSystems.
Run this to compare performance before/after optimizations.

Usage:
    python benchmarks/benchmark_io_performance.py
"""

import tempfile
import time
from typing import NamedTuple

import numpy as np
import pandas as pd

import flixopt as fx


class BenchmarkResult(NamedTuple):
    """Results from a benchmark run."""

    name: str
    mean_ms: float
    std_ms: float
    iterations: int


def create_large_flow_system(
    n_timesteps: int = 2190,
    n_periods: int = 12,
    n_components: int = 125,
) -> fx.FlowSystem:
    """Create a large FlowSystem for benchmarking.

    Args:
        n_timesteps: Number of timesteps (default 2190 = ~1 year at 4h resolution).
        n_periods: Number of periods (default 12).
        n_components: Number of sink/source pairs (default 125).

    Returns:
        Configured FlowSystem.
    """
    timesteps = pd.date_range('2024-01-01', periods=n_timesteps, freq='4h')
    periods = pd.Index([2028 + i * 2 for i in range(n_periods)], name='period')

    fs = fx.FlowSystem(timesteps=timesteps, periods=periods)
    fs.add_elements(fx.Effect('Cost', '€', is_objective=True))

    n_buses = 10
    buses = [fx.Bus(f'Bus_{i}') for i in range(n_buses)]
    fs.add_elements(*buses)

    # Create demand profile with daily pattern
    base_demand = 100 + 50 * np.sin(2 * np.pi * np.arange(n_timesteps) / 24)

    for i in range(n_components):
        bus = buses[i % n_buses]
        # Add noise to create unique profiles
        profile = base_demand + np.random.normal(0, 10, n_timesteps)
        profile = np.clip(profile / profile.max(), 0.1, 1.0)

        fs.add_elements(
            fx.Sink(
                f'D_{i}',
                inputs=[fx.Flow(f'Q_{i}', bus=bus.label, size=100, fixed_relative_profile=profile)],
            )
        )
        fs.add_elements(
            fx.Source(
                f'S_{i}',
                outputs=[fx.Flow(f'P_{i}', bus=bus.label, size=500, effects_per_flow_hour={'Cost': 20 + i})],
            )
        )

    return fs


def benchmark_function(func, iterations: int = 5, warmup: int = 1) -> BenchmarkResult:
    """Benchmark a function with multiple iterations.

    Args:
        func: Function to benchmark (callable with no arguments).
        iterations: Number of timed iterations.
        warmup: Number of warmup iterations (not timed).

    Returns:
        BenchmarkResult with timing statistics.
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return BenchmarkResult(
        name=func.__name__ if hasattr(func, '__name__') else str(func),
        mean_ms=np.mean(times) * 1000,
        std_ms=np.std(times) * 1000,
        iterations=iterations,
    )


def run_io_benchmarks(
    n_timesteps: int = 2190,
    n_periods: int = 12,
    n_components: int = 125,
    iterations: int = 5,
) -> dict[str, BenchmarkResult]:
    """Run IO performance benchmarks.

    Args:
        n_timesteps: Number of timesteps for the FlowSystem.
        n_periods: Number of periods.
        n_components: Number of components (sink/source pairs).
        iterations: Number of benchmark iterations.

    Returns:
        Dictionary mapping benchmark names to results.
    """
    print('=' * 70)
    print('FlowSystem IO Performance Benchmark')
    print('=' * 70)
    print('\nConfiguration:')
    print(f'  Timesteps: {n_timesteps}')
    print(f'  Periods: {n_periods}')
    print(f'  Components: {n_components}')
    print(f'  Iterations: {iterations}')

    # Create FlowSystem
    print('\n1. Creating FlowSystem...')
    fs = create_large_flow_system(n_timesteps, n_periods, n_components)
    print(f'   Components: {len(fs.components)}')

    # Create dataset
    print('\n2. Creating dataset...')
    ds = fs.to_dataset()
    print(f'   Variables: {len(ds.data_vars)}')
    print(f'   Size: {ds.nbytes / 1e6:.1f} MB')

    results = {}

    # Benchmark to_dataset
    print('\n3. Benchmarking to_dataset()...')
    result = benchmark_function(lambda: fs.to_dataset(), iterations=iterations)
    results['to_dataset'] = result
    print(f'   Mean: {result.mean_ms:.1f}ms (std: {result.std_ms:.1f}ms)')

    # Benchmark from_dataset
    print('\n4. Benchmarking from_dataset()...')
    result = benchmark_function(lambda: fx.FlowSystem.from_dataset(ds), iterations=iterations)
    results['from_dataset'] = result
    print(f'   Mean: {result.mean_ms:.1f}ms (std: {result.std_ms:.1f}ms)')

    # Benchmark NetCDF round-trip
    print('\n5. Benchmarking NetCDF round-trip...')
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
        tmp_path = f.name

    def netcdf_roundtrip():
        fs.to_netcdf(tmp_path, overwrite=True)
        return fx.FlowSystem.from_netcdf(tmp_path)

    result = benchmark_function(netcdf_roundtrip, iterations=iterations)
    results['netcdf_roundtrip'] = result
    print(f'   Mean: {result.mean_ms:.1f}ms (std: {result.std_ms:.1f}ms)')

    # Verify restoration
    print('\n6. Verification...')
    fs_restored = fx.FlowSystem.from_dataset(ds)
    print(f'   Components restored: {len(fs_restored.components)}')
    print(f'   Timesteps restored: {len(fs_restored.timesteps)}')
    print(f'   Periods restored: {len(fs_restored.periods)}')

    # Summary
    print('\n' + '=' * 70)
    print('Summary')
    print('=' * 70)
    for name, res in results.items():
        print(f'  {name}: {res.mean_ms:.1f}ms (+/- {res.std_ms:.1f}ms)')

    total_ms = results['to_dataset'].mean_ms + results['from_dataset'].mean_ms
    print(f'\n  Total (to + from): {total_ms:.1f}ms')

    return results


if __name__ == '__main__':
    run_io_benchmarks()
