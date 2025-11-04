"""
General-purpose benchmarking framework with decorator-based registry pattern.

Key features:
- Decorator-based registry for easy addition of benchmark operations
- Flexible setup phase that provides arbitrary context to benchmarks
- Clean pandas-based output formatting
- Automatic analysis and reporting
- Reusable for any benchmarking scenario

Usage:
    1. Create a setup function that returns a dict of objects needed for benchmarks
    2. Decorate benchmark functions with @registry.add('name')
    3. Benchmark functions receive **context from setup and return a callable
    4. Configure and run benchmarks
"""

import timeit
from typing import Callable, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import xarray as xr
import flixopt as fx


# ============================================================================
# Generic Registry Pattern for Benchmarking
# ============================================================================

@dataclass
class BenchmarkOp:
    """A single benchmark operation with its metadata."""
    name: str
    func: Callable
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Registry:
    """
    Registry for benchmark operations using decorator pattern.

    Usage:
        @registry.add('operation_name', category='transform')
        def my_benchmark(**context):
            obj = context['my_object']
            return lambda: obj.do_something()
    """
    ops: Dict[str, BenchmarkOp] = field(default_factory=dict)

    def add(self, name: str, **metadata):
        """Decorator to register a benchmark operation with optional metadata."""
        def decorator(func: Callable) -> Callable:
            self.ops[name] = BenchmarkOp(name, func, metadata=metadata)
            return func
        return decorator

    def enabled_ops(self) -> Dict[str, BenchmarkOp]:
        """Return all enabled operations."""
        return {k: v for k, v in self.ops.items() if v.enabled}

    def enable_only(self, names: list[str]):
        """Enable only specified operations."""
        for name, op in self.ops.items():
            op.enabled = name in names

    def disable(self, names: list[str]):
        """Disable specified operations."""
        for name in names:
            if name in self.ops:
                self.ops[name].enabled = False


# Global registry
registry = Registry()


# ============================================================================
# Benchmark Operation Definitions (FlowSystem Example)
# ============================================================================

@registry.add('to_dataset', category='conversion')
def bench_to_dataset(**context):
    """Convert FlowSystem to xarray Dataset."""
    flow_system = context['flow_system']
    return lambda: flow_system.to_dataset()


@registry.add('from_dataset', category='conversion')
def bench_from_dataset(**context):
    """Convert xarray Dataset back to FlowSystem."""
    dataset = context['dataset']
    return lambda: fx.FlowSystem.from_dataset(dataset)


@registry.add('sel', category='transform')
def bench_sel(**context):
    """Select half the timesteps."""
    dataset = context['dataset']
    sel_slice = slice('2020-01-01', dataset.indexes['time'][len(dataset.indexes['time'])//2])
    return lambda: fx.FlowSystem._dataset_sel(dataset, time=sel_slice)


@registry.add('resample', category='transform')
def bench_resample(**context):
    """Resample dataset to 4-hour frequency."""
    dataset = context['dataset']
    return lambda: fx.FlowSystem._dataset_resample(dataset, '4h')


@registry.add('coarsen', category='reference')
def bench_coarsen(**context):
    """Xarray coarsen (reference implementation)."""
    dataset = context['dataset']
    return lambda: dataset.coarsen(time=4).mean()


# Add your own operations here:
# @registry.add('my_operation', category='custom')
# def bench_my_operation(**context):
#     my_obj = context['my_object']
#     param = context.get('optional_param', 'default')
#     return lambda: my_obj.do_something(param)


# ============================================================================
# Generic Benchmark Execution Framework
# ============================================================================

def run_benchmarks(setup_func: Callable, n_runs: int) -> Dict[str, float]:
    """
    Run all enabled benchmark operations and return timings.

    Args:
        setup_func: Function that returns a dict of context for benchmarks
        n_runs: Number of times to run each benchmark

    Returns:
        Dict mapping operation name to average time in seconds
    """
    results = {}
    enabled = registry.enabled_ops()

    # Setup phase: prepare context for benchmarks
    context = setup_func()

    # Run each operation
    for name, op in enabled.items():
        func = op.func(**context)
        results[name] = timeit.timeit(func, number=n_runs) / n_runs

    return results


# ============================================================================
# FlowSystem-specific Setup (Example)
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


def setup_flow_system_context(n_timesteps: int, n_components: int) -> Dict[str, Any]:
    """
    Setup function for FlowSystem benchmarks.

    Returns a dict with all objects needed by the benchmark operations.
    """
    flow_system = create_flow_system(n_timesteps, n_components)
    dataset = flow_system.to_dataset()

    return {
        'flow_system': flow_system,
        'dataset': dataset,
        'n_timesteps': n_timesteps,
        'n_components': n_components,
    }


def run_all_configs(config: dict, setup_func_factory: Callable) -> pd.DataFrame:
    """
    Run benchmark for all configurations and return DataFrame.

    Args:
        config: Dict with 'param_grid' (dict of param_name -> list of values),
                'n_runs' (int), and optionally 'param_display_names' (dict)
        setup_func_factory: Function that takes **params and returns a setup function

    Returns:
        DataFrame with results for all parameter combinations
    """
    param_grid = config['param_grid']
    n_runs = config['n_runs']
    param_display = config.get('param_display_names', {})

    print("Starting benchmark...")
    print(f"Parameters: {param_grid}")
    print(f"Runs: {n_runs}")
    print(f"Operations: {', '.join(registry.enabled_ops().keys())}\n")

    rows = []

    # Generate all combinations of parameters
    import itertools
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    total = len(combinations)

    for idx, values in enumerate(combinations, 1):
        params = dict(zip(param_names, values))
        display_params = ', '.join(f"{param_display.get(k, k)}={v}" for k, v in params.items())
        print(f"[{idx}/{total}] {display_params}...")

        setup_func = setup_func_factory(**params)
        results = run_benchmarks(setup_func, n_runs)

        rows.append({**params, **results})

    df = pd.DataFrame(rows)

    # Add total column
    op_cols = list(registry.enabled_ops().keys())
    df['total'] = df[op_cols].sum(axis=1)

    return df


# ============================================================================
# Analysis & Reporting
# ============================================================================

def analyze_and_report(df: pd.DataFrame, param_cols: list[str] = None):
    """
    Print all analysis tables.

    Args:
        df: DataFrame with benchmark results
        param_cols: List of parameter column names (auto-detected if None)
    """
    # Auto-detect parameter columns
    if param_cols is None:
        op_cols = list(registry.enabled_ops().keys())
        param_cols = [col for col in df.columns if col not in op_cols + ['total']]

    op_cols = [col for col in df.columns if col not in param_cols + ['total']]

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

    # 2. Scaling by each parameter
    for param_col in param_cols:
        print("\n" + "=" * 100)
        print(f"SCALING BY {param_col.upper()}")
        print("=" * 100)

        by_param = df.groupby(param_col)[op_cols].mean()
        by_param['bottleneck'] = by_param.idxmax(axis=1)
        print(by_param.to_string(float_format='%.4f'))

    # 3. Detailed results
    print("\n" + "=" * 100)
    print("DETAILED RESULTS")
    print("=" * 100)
    print(df.to_string(index=False, float_format='%.4f'))


def save_results(df: pd.DataFrame, filename: str = 'benchmark_results', param_cols: list[str] = None):
    """
    Save results as CSV and optionally as NetCDF.

    Args:
        df: DataFrame with benchmark results
        filename: Base filename for output files
        param_cols: List of parameter column names (auto-detected if None)
    """
    # CSV - always save
    df.to_csv(f'{filename}.csv', index=False, float_format='%.6f')
    print(f"\n✓ Saved {filename}.csv")

    # Auto-detect parameter columns
    if param_cols is None:
        op_cols = list(registry.enabled_ops().keys())
        param_cols = [col for col in df.columns if col not in op_cols + ['total']]

    op_cols = [col for col in df.columns if col not in param_cols + ['total']]

    # NetCDF - only if we have exactly 2 parameters (for 2D grid)
    if len(param_cols) == 2:
        param1, param2 = param_cols
        values1 = sorted(df[param1].unique())
        values2 = sorted(df[param2].unique())

        if len(values1) > 1 or len(values2) > 1:
            ds = xr.Dataset(coords={param1: values1, param2: values2})
            for op in op_cols:
                pivoted = df.pivot(index=param1, columns=param2, values=op)
                ds[op] = ((param1, param2), pivoted.values)

            ds.to_netcdf(f'{filename}.nc')
            print(f"✓ Saved {filename}.nc")


# ============================================================================
# Main Execution (FlowSystem Example)
# ============================================================================

# Configuration for FlowSystem benchmarks
CONFIG = {
    'param_grid': {
        'n_timesteps': [100, 5000],
        'n_components': [10],
    },
    'param_display_names': {
        'n_timesteps': 'timesteps',
        'n_components': 'components',
    },
    'n_runs': 1,
}


def flow_system_setup_factory(n_timesteps: int, n_components: int) -> Callable:
    """Factory that creates setup functions with fixed parameters."""
    return lambda: setup_flow_system_context(n_timesteps, n_components)


if __name__ == '__main__':
    # Optional: Customize which operations to run
    # registry.enable_only(['to_dataset', 'from_dataset', 'sel'])
    # registry.disable(['coarsen'])

    # Run benchmark
    results_df = run_all_configs(CONFIG, flow_system_setup_factory)

    # Analyze and display
    analyze_and_report(results_df)

    # Save results
    save_results(results_df)


# ============================================================================
# Example: How to create your own benchmark
# ============================================================================
"""
# 1. Define your setup function
def setup_my_objects(size: int, complexity: int) -> Dict[str, Any]:
    obj = create_my_object(size, complexity)
    return {
        'my_object': obj,
        'size': size,
        'complexity': complexity,
    }

# 2. Register benchmark operations
@registry.add('operation_1', category='basic')
def bench_op1(**context):
    obj = context['my_object']
    return lambda: obj.method1()

@registry.add('operation_2', category='advanced')
def bench_op2(**context):
    obj = context['my_object']
    param = context['size']
    return lambda: obj.method2(param)

# 3. Create factory and config
def my_setup_factory(size: int, complexity: int) -> Callable:
    return lambda: setup_my_objects(size, complexity)

my_config = {
    'param_grid': {
        'size': [100, 1000, 10000],
        'complexity': [1, 5, 10],
    },
    'n_runs': 10,
}

# 4. Run benchmark
results = run_all_configs(my_config, my_setup_factory)
analyze_and_report(results)
save_results(results, 'my_benchmark_results')
"""