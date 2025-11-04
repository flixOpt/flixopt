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
    'timestep_sizes': [100, 500, 1000, 2000, 5000, 8760],  # Number of timesteps to test
    'component_counts': [1, 5, 10, 20, 50],  # Number of boilers to test
    'n_runs': 3,  # Number of timing iterations for each configuration
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
    """Benchmark individual operations and return timing results."""
    results = {}

    # 1. Benchmark to_dataset()
    results['to_dataset'] = timeit.timeit(
        lambda: flow_system.to_dataset(),
        number=n_runs
    ) / n_runs

    # Get dataset once for subsequent operations
    ds = flow_system.to_dataset()

    # 2. Benchmark from_dataset()
    results['from_dataset'] = timeit.timeit(
        lambda: fx.FlowSystem.from_dataset(ds),
        number=n_runs
    ) / n_runs

    # 3. Benchmark dataset.sel()
    sel_slice = slice('2020-01-01', ds.indexes['time'][len(ds.indexes['time'])//2])
    results['dataset_sel'] = timeit.timeit(
        lambda: ds.sel(time=sel_slice),
        number=n_runs
    ) / n_runs

    # 4. Benchmark dataset.resample()
    results['dataset_resample'] = timeit.timeit(
        lambda: ds.resample(time='2h').mean(),
        number=n_runs
    ) / n_runs

    # 5. Benchmark sel + resample on dataset
    results['dataset_sel_resample'] = timeit.timeit(
        lambda: ds.sel(time=sel_slice).resample(time='2h').mean(),
        number=n_runs
    ) / n_runs

    # 6. Benchmark OLD FlowSystem.sel().resample() (double conversion)
    def old_approach():
        ds1 = flow_system.to_dataset()
        ds1_sel = ds1.sel(time=sel_slice)
        fs_sel = fx.FlowSystem.from_dataset(ds1_sel)
        return fs_sel.resample('2h', method='mean')

    results['old_sel_resample'] = timeit.timeit(old_approach, number=n_runs) / n_runs

    # 7. Benchmark NEW FlowSystem.sel_and_resample() (single conversion)
    results['new_sel_and_resample'] = timeit.timeit(
        lambda: flow_system.sel_and_resample(time=sel_slice, freq='2h', method='mean'),
        number=n_runs
    ) / n_runs

    return results


def run_benchmark(config):
    """Run benchmark for all configurations and return results as xarray Dataset."""
    print("Starting benchmark...")
    print(f"Timestep sizes: {config['timestep_sizes']}")
    print(f"Component counts: {config['component_counts']}")
    print(f"Runs per configuration: {config['n_runs']}")
    print()

    # Storage for results
    data_arrays = {}
    operation_names = []

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
            for op_name, time_value in results.items():
                if op_name not in data_arrays:
                    data_arrays[op_name] = []
                    operation_names.append(op_name)
                data_arrays[op_name].append(time_value)

    # Convert to xarray Dataset
    coords = {
        'timesteps': config['timestep_sizes'],
        'components': config['component_counts'],
    }

    # Reshape data for xarray
    shape = (len(config['timestep_sizes']), len(config['component_counts']))
    dataset_vars = {}

    for op_name, values in data_arrays.items():
        dataset_vars[op_name] = (
            ['timesteps', 'components'],
            np.array(values).reshape(shape)
        )

    ds = xr.Dataset(dataset_vars, coords=coords)
    ds.attrs['n_runs'] = config['n_runs']
    ds.attrs['description'] = 'Benchmark of FlowSystem sel/resample operations'

    return ds


def analyze_results(ds):
    """Analyze and print benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Calculate conversion overhead
    conversion_overhead = ds['to_dataset'] + ds['from_dataset']
    dataset_operation = ds['dataset_sel_resample']

    # OLD approach breakdown
    old_total = ds['old_sel_resample']
    old_conversion_pct = (2 * conversion_overhead / old_total * 100)
    old_operation_pct = (dataset_operation / old_total * 100)

    # NEW approach breakdown
    new_total = ds['new_sel_and_resample']
    new_conversion_pct = (conversion_overhead / new_total * 100)
    new_operation_pct = (dataset_operation / new_total * 100)

    # Speedup
    speedup = old_total / new_total

    print("\nBottleneck Analysis (averaged across all configurations):")
    print("-" * 80)

    print(f"\nOLD approach (sel().resample()):")
    print(f"  Total time:           {float(old_total.mean()):.4f}s")
    print(f"  Conversion overhead:  {float((2 * conversion_overhead).mean()):.4f}s ({float(old_conversion_pct.mean()):.1f}%)")
    print(f"  Dataset operations:   {float(dataset_operation.mean()):.4f}s ({float(old_operation_pct.mean()):.1f}%)")

    print(f"\nNEW approach (sel_and_resample()):")
    print(f"  Total time:           {float(new_total.mean()):.4f}s")
    print(f"  Conversion overhead:  {float(conversion_overhead.mean()):.4f}s ({float(new_conversion_pct.mean()):.1f}%)")
    print(f"  Dataset operations:   {float(dataset_operation.mean()):.4f}s ({float(new_operation_pct.mean()):.1f}%)")

    print(f"\nPerformance improvement:")
    print(f"  Average speedup:      {float(speedup.mean()):.2f}x")
    print(f"  Time saved:           {float((old_total - new_total).mean() * 1000):.1f}ms per operation")

    print("\n" + "=" * 80)
    print("Detailed breakdown (mean times in seconds):")
    print("=" * 80)
    for var in ds.data_vars:
        print(f"  {var:25s}: {float(ds[var].mean()):.4f}s")

    print("\n" + "=" * 80)
    print("How FlowSystem size affects bottleneck:")
    print("=" * 80)

    # Show scaling by timesteps (averaged over components)
    print("\nScaling by timesteps (averaged over components):")
    print(f"{'Timesteps':>10} {'to_dataset':>12} {'from_dataset':>13} {'dataset_op':>12} {'speedup':>10}")
    print("-" * 60)
    for n_ts in ds.coords['timesteps'].values:
        subset = ds.sel(timesteps=n_ts)
        to_ds = float(subset['to_dataset'].mean())
        from_ds = float(subset['from_dataset'].mean())
        ds_op = float(subset['dataset_sel_resample'].mean())
        sp = float(subset['old_sel_resample'].mean() / subset['new_sel_and_resample'].mean())
        print(f"{n_ts:10d} {to_ds:12.4f}s {from_ds:13.4f}s {ds_op:12.4f}s {sp:10.2f}x")

    # Show scaling by components (averaged over timesteps)
    print("\nScaling by components (averaged over timesteps):")
    print(f"{'Components':>11} {'to_dataset':>12} {'from_dataset':>13} {'dataset_op':>12} {'speedup':>10}")
    print("-" * 60)
    for n_comp in ds.coords['components'].values:
        subset = ds.sel(components=n_comp)
        to_ds = float(subset['to_dataset'].mean())
        from_ds = float(subset['from_dataset'].mean())
        ds_op = float(subset['dataset_sel_resample'].mean())
        sp = float(subset['old_sel_resample'].mean() / subset['new_sel_and_resample'].mean())
        print(f"{n_comp:11d} {to_ds:12.4f}s {from_ds:13.4f}s {ds_op:12.4f}s {sp:10.2f}x")


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
