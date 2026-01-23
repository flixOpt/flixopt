"""Benchmark script for model build and LP file I/O performance.

Tests build_model() and LP file writing with large FlowSystems.

Usage:
    python benchmarks/benchmark_model_build.py
"""

import os
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
    file_size_mb: float | None = None


def create_flow_system(
    n_timesteps: int = 168,
    n_periods: int | None = None,
    n_components: int = 50,
) -> fx.FlowSystem:
    """Create a FlowSystem for benchmarking.

    Args:
        n_timesteps: Number of timesteps.
        n_periods: Number of periods (None for no periods).
        n_components: Number of sink/source pairs.

    Returns:
        Configured FlowSystem.
    """
    timesteps = pd.date_range('2024-01-01', periods=n_timesteps, freq='h')
    periods = pd.Index([2028 + i * 2 for i in range(n_periods)], name='period') if n_periods else None

    fs = fx.FlowSystem(timesteps=timesteps, periods=periods)
    fs.add_elements(fx.Effect('Cost', '€', is_objective=True))

    n_buses = 5
    buses = [fx.Bus(f'Bus_{i}') for i in range(n_buses)]
    fs.add_elements(*buses)

    # Create demand profile
    base_demand = 100 + 50 * np.sin(2 * np.pi * np.arange(n_timesteps) / 24)

    for i in range(n_components):
        bus = buses[i % n_buses]
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
    """Benchmark a function with multiple iterations."""
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


def run_model_benchmarks(
    n_timesteps: int = 168,
    n_periods: int | None = None,
    n_components: int = 50,
    iterations: int = 3,
) -> dict[str, BenchmarkResult]:
    """Run model build and LP file benchmarks."""
    print('=' * 70)
    print('Model Build & LP File Benchmark')
    print('=' * 70)
    print('\nConfiguration:')
    print(f'  Timesteps: {n_timesteps}')
    print(f'  Periods: {n_periods or "None"}')
    print(f'  Components: {n_components}')
    print(f'  Iterations: {iterations}')

    results = {}

    # Create FlowSystem
    print('\n1. Creating FlowSystem...')
    fs = create_flow_system(n_timesteps, n_periods, n_components)
    print(f'   Components: {len(fs.components)}')
    print(f'   Flows: {len(fs.flows)}')

    # Benchmark build_model
    print('\n2. Benchmarking build_model()...')

    def build_model():
        # Need fresh FlowSystem each time since build_model modifies it
        fs_fresh = create_flow_system(n_timesteps, n_periods, n_components)
        fs_fresh.build_model()
        return fs_fresh

    result = benchmark_function(build_model, iterations=iterations, warmup=1)
    results['build_model'] = result
    print(f'   Mean: {result.mean_ms:.1f}ms (std: {result.std_ms:.1f}ms)')

    # Build model once for LP file benchmarks
    print('\n3. Building model for LP benchmarks...')
    fs.build_model()
    model = fs.model

    print(f'   Variables: {len(model.variables)}')
    print(f'   Constraints: {len(model.constraints)}')

    # Benchmark LP file write
    print('\n4. Benchmarking LP file write...')
    with tempfile.TemporaryDirectory() as tmpdir:
        lp_path = os.path.join(tmpdir, 'model.lp')

        def write_lp():
            model.to_file(lp_path)

        result = benchmark_function(write_lp, iterations=iterations, warmup=1)
        file_size_mb = os.path.getsize(lp_path) / 1e6

        results['write_lp'] = BenchmarkResult(
            name='write_lp',
            mean_ms=result.mean_ms,
            std_ms=result.std_ms,
            iterations=result.iterations,
            file_size_mb=file_size_mb,
        )
        print(f'   Mean: {result.mean_ms:.1f}ms (std: {result.std_ms:.1f}ms)')
        print(f'   File size: {file_size_mb:.2f} MB')

    # Summary
    print('\n' + '=' * 70)
    print('Summary')
    print('=' * 70)
    print(f'\n  {"Operation":<20} {"Mean":>12} {"Std":>12} {"File Size":>12}')
    print(f'  {"-" * 20} {"-" * 12} {"-" * 12} {"-" * 12}')

    for key, res in results.items():
        size_str = f'{res.file_size_mb:.2f} MB' if res.file_size_mb else '-'
        print(f'  {key:<20} {res.mean_ms:>9.1f}ms {res.std_ms:>9.1f}ms {size_str:>12}')

    return results


def run_scaling_benchmark():
    """Run benchmarks with different system sizes."""
    print('\n' + '=' * 70)
    print('Scaling Benchmark')
    print('=' * 70)

    configs = [
        # (n_timesteps, n_periods, n_components)
        (24, None, 10),
        (168, None, 10),
        (168, None, 50),
        (168, None, 100),
        (168, 3, 50),
        (720, None, 50),
    ]

    print(f'\n  {"Config":<30} {"build_model":>15} {"write_lp":>15} {"LP Size":>12}')
    print(f'  {"-" * 30} {"-" * 15} {"-" * 15} {"-" * 12}')

    for n_ts, n_per, n_comp in configs:
        results = run_model_benchmarks(n_ts, n_per, n_comp, iterations=3)

        per_str = f', {n_per}p' if n_per else ''
        config = f'{n_ts}ts, {n_comp}c{per_str}'

        build_ms = results['build_model'].mean_ms
        lp_ms = results['write_lp'].mean_ms
        lp_size = results['write_lp'].file_size_mb

        print(f'  {config:<30} {build_ms:>12.1f}ms {lp_ms:>12.1f}ms {lp_size:>9.2f} MB')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark model build and LP file I/O')
    parser.add_argument('--timesteps', '-t', type=int, default=168, help='Number of timesteps')
    parser.add_argument('--periods', '-p', type=int, default=None, help='Number of periods')
    parser.add_argument('--components', '-c', type=int, default=50, help='Number of components')
    parser.add_argument('--iterations', '-i', type=int, default=3, help='Benchmark iterations')
    parser.add_argument('--scaling', '-s', action='store_true', help='Run scaling benchmark')
    args = parser.parse_args()

    if args.scaling:
        run_scaling_benchmark()
    else:
        run_model_benchmarks(args.timesteps, args.periods, args.components, args.iterations)
