"""Benchmark script for FlowSystem IO performance.

Tests to_dataset() and from_dataset() performance with large FlowSystems.
Run this to compare performance before/after optimizations.

Usage:
    python benchmarks/benchmark_io_performance.py
"""

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
        Configured FlowSystem ready for optimization.
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

    for i in range(n_components // 2):
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
    n_clusters: int = 8,
    iterations: int = 5,
) -> dict[str, BenchmarkResult]:
    """Run IO performance benchmarks.

    Args:
        n_timesteps: Number of timesteps for the FlowSystem.
        n_periods: Number of periods.
        n_components: Number of components (sink/source pairs).
        n_clusters: Number of clusters for aggregation.
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
    print(f'  Clusters: {n_clusters}')
    print(f'  Iterations: {iterations}')

    # Create and prepare FlowSystem
    print('\n1. Creating FlowSystem...')
    fs = create_large_flow_system(n_timesteps, n_periods, n_components)
    print(f'   Components: {len(fs.components)}')

    print('\n2. Clustering and solving...')
    fs_clustered = fs.transform.cluster(n_clusters=n_clusters, cluster_duration='1D')
    fs_clustered.optimize(fx.solvers.GurobiSolver())

    print('\n3. Expanding...')
    fs_expanded = fs_clustered.transform.expand()
    print(f'   Expanded timesteps: {len(fs_expanded.timesteps)}')

    # Create dataset with solution
    print('\n4. Creating dataset...')
    ds = fs_expanded.to_dataset(include_solution=True)
    print(f'   Variables: {len(ds.data_vars)}')
    print(f'   Size: {ds.nbytes / 1e6:.1f} MB')

    results = {}

    # Benchmark to_dataset
    print('\n5. Benchmarking to_dataset()...')
    result = benchmark_function(lambda: fs_expanded.to_dataset(include_solution=True), iterations=iterations)
    results['to_dataset'] = result
    print(f'   Mean: {result.mean_ms:.1f}ms (std: {result.std_ms:.1f}ms)')

    # Benchmark from_dataset
    print('\n6. Benchmarking from_dataset()...')
    result = benchmark_function(lambda: fx.FlowSystem.from_dataset(ds), iterations=iterations)
    results['from_dataset'] = result
    print(f'   Mean: {result.mean_ms:.1f}ms (std: {result.std_ms:.1f}ms)')

    # Verify restoration
    print('\n7. Verification...')
    fs_restored = fx.FlowSystem.from_dataset(ds)
    print(f'   Components restored: {len(fs_restored.components)}')
    print(f'   Timesteps restored: {len(fs_restored.timesteps)}')
    print(f'   Has solution: {fs_restored.solution is not None}')
    if fs_restored.solution is not None:
        print(f'   Solution variables: {len(fs_restored.solution.data_vars)}')

    # Summary
    print('\n' + '=' * 70)
    print('Summary')
    print('=' * 70)
    for name, res in results.items():
        print(f'  {name}: {res.mean_ms:.1f}ms (+/- {res.std_ms:.1f}ms)')

    return results


if __name__ == '__main__':
    run_io_benchmarks()
