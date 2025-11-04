"""
Compact benchmark to identify bottlenecks in FlowSystem sel()/resample() operations.

Key features:
- Decorator-based registry pattern for easy addition of new benchmark operations
- Clean pandas-based output formatting
- Automatic analysis and reporting
"""

import timeit
from typing import Callable, Dict
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import xarray as xr
import flixopt as fx


# ============================================================================
# Registry Pattern for Benchmark Operations
# ============================================================================

@dataclass
class BenchmarkOp:
    """A single benchmark operation with its metadata."""
    name: str
    func: Callable
    needs_dataset: bool = False
    enabled: bool = True


@dataclass
class Registry:
    """
    Registry for benchmark operations using decorator pattern.

    Usage:
        @registry.add('operation_name', needs_dataset=True)
        def my_benchmark(flow_system, dataset=None):
            return lambda: do_something()
    """
    ops: Dict[str, BenchmarkOp] = field(default_factory=dict)

    def add(self, name: str, needs_dataset: bool = False):
        """Decorator to register a benchmark operation."""
        def decorator(func: Callable) -> Callable:
            self.ops[name] = BenchmarkOp(name, func, needs_dataset)
            return func
        return decorator

    def enabled_ops(self) -> Dict[str, BenchmarkOp]:
        """Return all enabled operations."""
        return {k: v for k, v in self.ops.items() if v.enabled}

    def enable_only(self, names: list[str]):
        """Enable only specified operations."""
        for name, op in self.ops.items():
            op.enabled = name in names


# Global registry
registry = Registry()


# ============================================================================
# Benchmark Operation Definitions
# ============================================================================

@registry.add('to_dataset')
def bench_to_dataset(flow_system, dataset=None):
    """Convert FlowSystem to xarray Dataset."""
    return lambda: flow_system.to_dataset()


@registry.add('from_dataset', needs_dataset=True)
def bench_from_dataset(flow_system, dataset):
    """Convert xarray Dataset back to FlowSystem."""
    return lambda: fx.FlowSystem.from_dataset(dataset)


@registry.add('sel', needs_dataset=True)
def bench_sel(flow_system, dataset):
    """Select half the timesteps."""
    sel_slice = slice('2020-01-01', dataset.indexes['time'][len(dataset.indexes['time'])//2])
    return lambda: fx.FlowSystem._dataset_sel(dataset, time=sel_slice)


@registry.add('resample', needs_dataset=True)
def bench_resample(flow_system, dataset):
    """Resample dataset to 4-hour frequency."""
    return lambda: fx.FlowSystem._dataset_resample(dataset, '4h')


@registry.add('coarsen', needs_dataset=True)
def bench_coarsen(flow_system, dataset):
    """Xarray coarsen (reference implementation)."""
    return lambda: dataset.coarsen(time=4).mean()


# Add your own operations here:
# @registry.add('my_operation', needs_dataset=True)
# def bench_my_operation(flow_system, dataset):
#     return lambda: do_something(dataset)


# ============================================================================
# FlowSystem Creation & Benchmark Execution
# ============================================================================

def create_flow_system(n_timesteps: int, n_components: int):
    """Create a FlowSystem with specified size."""
    fx.CONFIG.silent()
    timesteps = pd.date_range('2020-01-01', periods=n_timesteps, freq='h')
    flow_system = fx.FlowSystem(timesteps)

    demand = 50 + 20 * np.sin(2 * np.pi * np.arange(n_timesteps) / n_timesteps)

    flow_system.add_elements(
        fx.Bus('Heat'),
        fx.Bus('Gas'),
        fx.Effect('Costs', '€', 'Cost', is_standard=True, is_objective=True),
    )

    # Add N boilers
    for i in range(n_components):
        flow_system.add_elements(
            fx.linear_converters.Boiler(
                f'Boiler_{i}',
                eta=np.random.random(len(timesteps))/2 + 0.5,
                Q_th=fx.Flow(label=f'Heat_{i}', bus='Heat', size=50 + i * 5),
                Q_fu=fx.Flow(label=f'Gas_{i}', bus='Gas'),
            )
        )

    flow_system.add_elements(
        fx.Sink('Sink', inputs=[fx.Flow(label='Demand', bus='Heat', size=1,
                                        fixed_relative_profile=demand)]),
        fx.Source('Source', outputs=[fx.Flow(label='Gas', bus='Gas', size=1000,
                                             effects_per_flow_hour=0.04)]),
    )

    flow_system.connect_and_transform()
    return flow_system


def run_benchmarks(flow_system, n_runs: int) -> Dict[str, float]:
    """Run all enabled benchmark operations and return timings."""
    results = {}
    enabled = registry.enabled_ops()

    # Get dataset once if needed
    dataset = None
    if any(op.needs_dataset for op in enabled.values()):
        dataset = flow_system.to_dataset()

    # Run each operation
    for name, op in enabled.items():
        func = op.func(flow_system, dataset)
        results[name] = timeit.timeit(func, number=n_runs) / n_runs

    return results


def run_all_configs(config: dict) -> pd.DataFrame:
    """Run benchmark for all configurations and return DataFrame."""
    print("Starting benchmark...")
    print(f"Timesteps: {config['timestep_sizes']}")
    print(f"Components: {config['component_counts']}")
    print(f"Runs: {config['n_runs']}")
    print(f"Operations: {', '.join(registry.enabled_ops().keys())}\n")

    rows = []
    total = len(config['timestep_sizes']) * len(config['component_counts'])
    current = 0

    for n_ts in config['timestep_sizes']:
        for n_comp in config['component_counts']:
            current += 1
            print(f"[{current}/{total}] {n_ts} timesteps, {n_comp} components...")

            flow_system = create_flow_system(n_ts, n_comp)
            results = run_benchmarks(flow_system, config['n_runs'])

            rows.append({'timesteps': n_ts, 'components': n_comp, **results})

    df = pd.DataFrame(rows)

    # Add total column
    op_cols = list(registry.enabled_ops().keys())
    df['total'] = df[op_cols].sum(axis=1)

    return df


# ============================================================================
# Analysis & Reporting
# ============================================================================

def analyze_and_report(df: pd.DataFrame):
    """Print all analysis tables."""
    op_cols = [col for col in df.columns if col not in ['timesteps', 'components', 'total']]

    # 1. Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY: Average timing across all configurations")
    print("=" * 100)

    summary = df[op_cols + ['total']].mean().to_frame(name='time_s')
    summary['time_ms'] = summary['time_s'] * 1000
    summary['pct'] = (summary['time_s'] / summary.loc['total', 'time_s'] * 100).fillna(0)
    summary = summary.drop('total').sort_values('time_s', ascending=False)
    summary = pd.concat([summary, df[['total']].mean().to_frame(name='time_s')])

    print(summary.to_string(float_format='%.4f'))

    # 2. Scaling by timesteps
    print("\n" + "=" * 100)
    print("SCALING BY TIMESTEPS")
    print("=" * 100)

    by_ts = df.groupby('timesteps')[op_cols].mean()
    by_ts['bottleneck'] = by_ts.idxmax(axis=1)
    print(by_ts.to_string(float_format='%.4f'))

    # 3. Scaling by components
    print("\n" + "=" * 100)
    print("SCALING BY COMPONENTS")
    print("=" * 100)

    by_comp = df.groupby('components')[op_cols].mean()
    by_comp['bottleneck'] = by_comp.idxmax(axis=1)
    print(by_comp.to_string(float_format='%.4f'))

    # 4. Detailed results
    print("\n" + "=" * 100)
    print("DETAILED RESULTS")
    print("=" * 100)
    print(df.to_string(index=False, float_format='%.4f'))


def save_results(df: pd.DataFrame, filename: str = 'benchmark_results'):
    """Save results as CSV and NetCDF."""
    # CSV
    df.to_csv(f'{filename}.csv', index=False, float_format='%.6f')
    print(f"\n✓ Saved {filename}.csv")

    # NetCDF (if multiple configs)
    timesteps = sorted(df['timesteps'].unique())
    components = sorted(df['components'].unique())

    if len(timesteps) > 1 or len(components) > 1:
        op_cols = [col for col in df.columns if col not in ['timesteps', 'components', 'total']]

        ds = xr.Dataset(coords={'timesteps': timesteps, 'components': components})
        for op in op_cols:
            pivoted = df.pivot(index='timesteps', columns='components', values=op)
            ds[op] = (('timesteps', 'components'), pivoted.values)

        ds.to_netcdf(f'{filename}.nc')
        print(f"✓ Saved {filename}.nc")


# ============================================================================
# Main Execution
# ============================================================================

CONFIG = {
    'timestep_sizes': [100, 5000],
    'component_counts': [10],
    'n_runs': 1,
}


if __name__ == '__main__':
    # Optional: Customize operations
    # registry.enable_only(['to_dataset', 'from_dataset', 'sel'])

    # Run benchmark
    results_df = run_all_configs(CONFIG)

    # Analyze and display
    analyze_and_report(results_df)

    # Save results
    save_results(results_df)