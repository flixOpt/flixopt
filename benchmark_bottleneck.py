"""
Compact benchmark to identify bottlenecks in FlowSystem sel()/resample() operations.
Measures to_dataset(), from_dataset(), and actual dataset operations separately.
"""

import timeit
import numpy as np
import pandas as pd
import xarray as xr
import flixopt as fx

# Configuration
CONFIG = {
    'timestep_sizes': [100, 500, 1000, 3000, 8760],#, 2000, 5000, 8760],  # Number of timesteps to test
    'component_counts': [1, 5, 10, 20, 50],#, 20, 50],  # Number of boilers to test
    'n_runs': 1,  # Number of timing iterations for each configuration
}


def create_flow_system(n_timesteps, n_components):
    """Create a FlowSystem with specified size."""
    fx.CONFIG.silent()

    timesteps = pd.date_range('2020-01-01', periods=n_timesteps, freq='h')
    flow_system = fx.FlowSystem(timesteps)

    # Create demand profile
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
        fx.Sink('Sink', inputs=[fx.Flow(label='Demand', bus='Heat', size=1, fixed_relative_profile=demand)]),
        fx.Source('Source', outputs=[fx.Flow(label='Gas', bus='Gas', size=1000, effects_per_flow_hour=0.04)]),
    )

    flow_system.connect_and_transform()
    return flow_system


def benchmark_operations(flow_system, n_runs):
    """
    Benchmark individual operations to identify bottlenecks.

    Measures only individual steps:
    1. to_dataset() - FlowSystem to xarray conversion
    2. from_dataset() - xarray to FlowSystem conversion
    3. sel() - xarray selection operation
    4. resample() - optimized resampling using _resample_by_dimension_groups()
    """
    results = {}

    # Get dataset once for operations that need it
    ds = flow_system.to_dataset()

    # 1. Benchmark to_dataset()
    results['to_dataset'] = timeit.timeit(
        lambda: flow_system.to_dataset(),
        number=n_runs
    ) / n_runs

    # 2. Benchmark from_dataset()
    results['from_dataset'] = timeit.timeit(
        lambda: fx.FlowSystem.from_dataset(ds),
        number=n_runs
    ) / n_runs

    # 3. Benchmark dataset.sel() (select half the timesteps)
    sel_slice = slice('2020-01-01', ds.indexes['time'][len(ds.indexes['time'])//2])
    results['sel'] = timeit.timeit(
        lambda: ds.sel(time=sel_slice),
        number=n_runs
    ) / n_runs

    # 4. Benchmark optimized resample (using _resample_by_dimension_groups)
    def optimized_resample():
        time_var_names = [v for v in ds.data_vars if 'time' in ds[v].dims]
        non_time_var_names = [v for v in ds.data_vars if v not in time_var_names]
        time_dataset = ds[time_var_names]
        resampled_time_dataset = flow_system._resample_by_dimension_groups(time_dataset, '2h', 'mean')
        if non_time_var_names:
            non_time_dataset = ds[non_time_var_names]
            return xr.merge([resampled_time_dataset, non_time_dataset])
        return resampled_time_dataset

    results['resample'] = timeit.timeit(optimized_resample, number=n_runs) / n_runs

    return results


def run_benchmark(config):
    """Run benchmark for all configurations and return results as xarray Dataset."""
    print("Starting bottleneck benchmark...")
    print(f"Timestep sizes: {config['timestep_sizes']}")
    print(f"Component counts: {config['component_counts']}")
    print(f"Runs per configuration: {config['n_runs']}")
    print("\nMeasuring individual operations: to_dataset, from_dataset, sel, resample")
    print()

    # Storage for results - only 4 operations now
    operations = ['to_dataset', 'from_dataset', 'sel', 'resample']
    data_arrays = {op: [] for op in operations}

    total_configs = len(config['timestep_sizes']) * len(config['component_counts'])
    current = 0

    for n_timesteps in config['timestep_sizes']:
        for n_components in config['component_counts']:
            current += 1
            print(f"[{current}/{total_configs}] Testing: {n_timesteps} timesteps, {n_components} components...")

            # Create FlowSystem
            flow_system = create_flow_system(n_timesteps, n_components)

            # Benchmark operations
            results = benchmark_operations(flow_system, config['n_runs'])

            # Store results
            for op_name in operations:
                data_arrays[op_name].append(results[op_name])

    # Convert to xarray Dataset
    coords = {
        'timesteps': config['timestep_sizes'],
        'components': config['component_counts'],
    }

    # Reshape data for xarray
    shape = (len(config['timestep_sizes']), len(config['component_counts']))
    dataset_vars = {}

    for op_name in operations:
        dataset_vars[op_name] = (
            ['timesteps', 'components'],
            np.array(data_arrays[op_name]).reshape(shape)
        )

    ds = xr.Dataset(dataset_vars, coords=coords)
    ds.attrs['n_runs'] = config['n_runs']
    ds.attrs['description'] = 'Bottleneck analysis of individual FlowSystem operations'
    ds.attrs['operations'] = ', '.join(operations)

    return ds


def analyze_results(ds):
    """Analyze and print benchmark results."""
    print("\n" + "=" * 80)
    print("BOTTLENECK ANALYSIS - Individual Operation Timings")
    print("=" * 80)

    # Overall statistics
    print("\nOverall averages (across all configurations):")
    print("-" * 80)
    for var in ['to_dataset', 'from_dataset', 'sel', 'resample']:
        avg_time = float(ds[var].mean())
        print(f"  {var:15s}: {avg_time:.4f}s ({avg_time*1000:.1f}ms)")

    # Find the bottleneck
    print("\n" + "=" * 80)
    print("Bottleneck identification:")
    print("-" * 80)
    means = {var: float(ds[var].mean()) for var in ['to_dataset', 'from_dataset', 'sel', 'resample']}
    total = sum(means.values())
    sorted_ops = sorted(means.items(), key=lambda x: x[1], reverse=True)

    for i, (op, time) in enumerate(sorted_ops, 1):
        pct = (time / total) * 100
        print(f"  {i}. {op:15s}: {time:.4f}s ({pct:5.1f}% of total)")

    print(f"\nTotal time for all operations: {total:.4f}s")

    # Scaling analysis by timesteps
    print("\n" + "=" * 80)
    print("Scaling by timesteps (averaged over components):")
    print("=" * 80)
    print(f"{'Timesteps':>10} {'to_dataset':>12} {'from_dataset':>13} {'sel':>10} {'resample':>12} {'Bottleneck':>15}")
    print("-" * 80)
    for n_ts in ds.coords['timesteps'].values:
        subset = ds.sel(timesteps=n_ts)
        to_ds = float(subset['to_dataset'].mean())
        from_ds = float(subset['from_dataset'].mean())
        sel = float(subset['sel'].mean())
        resample = float(subset['resample'].mean())
        times = {'to_dataset': to_ds, 'from_dataset': from_ds, 'sel': sel, 'resample': resample}
        bottleneck = max(times.items(), key=lambda x: x[1])[0]
        print(f"{n_ts:10d} {to_ds:12.4f}s {from_ds:13.4f}s {sel:10.4f}s {resample:12.4f}s {bottleneck:>15s}")

    # Scaling analysis by components
    print("\n" + "=" * 80)
    print("Scaling by components (averaged over timesteps):")
    print("=" * 80)
    print(f"{'Components':>11} {'to_dataset':>12} {'from_dataset':>13} {'sel':>10} {'resample':>12} {'Bottleneck':>15}")
    print("-" * 80)
    for n_comp in ds.coords['components'].values:
        subset = ds.sel(components=n_comp)
        to_ds = float(subset['to_dataset'].mean())
        from_ds = float(subset['from_dataset'].mean())
        sel = float(subset['sel'].mean())
        resample = float(subset['resample'].mean())
        times = {'to_dataset': to_ds, 'from_dataset': from_ds, 'sel': sel, 'resample': resample}
        bottleneck = max(times.items(), key=lambda x: x[1])[0]
        print(f"{n_comp:11d} {to_ds:12.4f}s {from_ds:13.4f}s {sel:10.4f}s {resample:12.4f}s {bottleneck:>15s}")


if __name__ == '__main__':
    # Run benchmark
    results = run_benchmark(CONFIG)

    # Save results
    results.to_netcdf('benchmark_results.nc')
    print("\n✓ Results saved to benchmark_results.nc")

    # Analyze and display
    analyze_results(results)

    # Also save as CSV for easy viewing
    df = results.to_dataframe()
    df.to_csv('benchmark_results.csv')
    print("✓ Results saved to benchmark_results.csv")
