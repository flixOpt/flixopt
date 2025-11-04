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
    'timestep_sizes': [100, 5000, 8760],  # Number of timesteps to test
    'component_counts': [10],  # Number of boilers to test
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
    """
    Benchmark individual operations to identify bottlenecks.
    Returns a dict with operation names as keys and timing results as values.
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
        lambda: fx.FlowSystem._dataset_sel(ds, time=sel_slice),
        number=n_runs
    ) / n_runs

    # 4. Benchmark resample
    results['resample'] = timeit.timeit(
        lambda: fx.FlowSystem._dataset_resample(ds, '4h'),
        number=n_runs
    ) / n_runs

    # 5. Benchmark coarsen (reference)
    results['coarsen'] = timeit.timeit(
        lambda: ds.coarsen(time=4).mean(),
        number=n_runs
    ) / n_runs

    return results


def run_benchmark(config):
    """Run benchmark for all configurations and return results as DataFrame."""
    print("Starting bottleneck benchmark...")
    print(f"Timestep sizes: {config['timestep_sizes']}")
    print(f"Component counts: {config['component_counts']}")
    print(f"Runs per configuration: {config['n_runs']}\n")

    # Collect all results
    rows = []

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

            # Create row with all metadata and results
            row = {
                'timesteps': n_timesteps,
                'components': n_components,
                **results  # Unpack all operation timings
            }
            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Add derived columns
    df['total'] = df[['to_dataset', 'from_dataset', 'sel', 'resample', 'coarsen']].sum(axis=1)

    return df


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "=" * 100)
    print("SUMMARY: Average timing across all configurations")
    print("=" * 100)

    # Calculate means for operation columns
    operation_cols = ['to_dataset', 'from_dataset', 'sel', 'resample', 'coarsen', 'total']
    summary = df[operation_cols].mean().to_frame(name='avg_time_s')
    summary['avg_time_ms'] = summary['avg_time_s'] * 1000
    summary['pct_of_total'] = (summary['avg_time_s'] / summary.loc['total', 'avg_time_s'] * 100).fillna(0)

    # Sort by time (excluding total)
    summary_sorted = summary.drop('total').sort_values('avg_time_s', ascending=False)
    summary_sorted = pd.concat([summary_sorted, summary.loc[['total']]])

    print(summary_sorted.to_string(float_format=lambda x: f'{x:.4f}'))


def print_by_timesteps(df):
    """Print results grouped by timesteps."""
    print("\n" + "=" * 100)
    print("SCALING BY TIMESTEPS (averaged over components)")
    print("=" * 100)

    operation_cols = ['to_dataset', 'from_dataset', 'sel', 'resample', 'coarsen']

    # Group by timesteps and calculate mean
    grouped = df.groupby('timesteps')[operation_cols].mean()

    # Add bottleneck column
    grouped['bottleneck'] = grouped.idxmax(axis=1)

    print(grouped.to_string(float_format=lambda x: f'{x:.4f}'))


def print_by_components(df):
    """Print results grouped by components."""
    print("\n" + "=" * 100)
    print("SCALING BY COMPONENTS (averaged over timesteps)")
    print("=" * 100)

    operation_cols = ['to_dataset', 'from_dataset', 'sel', 'resample', 'coarsen']

    # Group by components and calculate mean
    grouped = df.groupby('components')[operation_cols].mean()

    # Add bottleneck column
    grouped['bottleneck'] = grouped.idxmax(axis=1)

    print(grouped.to_string(float_format=lambda x: f'{x:.4f}'))


def print_detailed_results(df):
    """Print full detailed results table."""
    print("\n" + "=" * 100)
    print("DETAILED RESULTS (all configurations)")
    print("=" * 100)

    # Select columns to display
    display_cols = ['timesteps', 'components', 'to_dataset', 'from_dataset',
                    'sel', 'resample', 'coarsen', 'total']

    print(df[display_cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))


def save_results(df, base_filename='benchmark_results'):
    """Save results to multiple formats."""
    # Save as CSV
    csv_file = f'{base_filename}.csv'
    df.to_csv(csv_file, index=False, float_format='%.6f')
    print(f"\n✓ Results saved to {csv_file}")

    # Save as NetCDF (via xarray)
    # Pivot to create a proper multidimensional structure
    operation_cols = ['to_dataset', 'from_dataset', 'sel', 'resample', 'coarsen']

    # If we have multiple timesteps and components, create proper coordinates
    timesteps = sorted(df['timesteps'].unique())
    components = sorted(df['components'].unique())

    if len(timesteps) > 1 or len(components) > 1:
        # Create xarray dataset
        ds = xr.Dataset(
            coords={
                'timesteps': timesteps,
                'components': components,
            }
        )

        # Add each operation as a data variable
        for op in operation_cols:
            # Pivot the dataframe
            pivoted = df.pivot(index='timesteps', columns='components', values=op)
            ds[op] = (('timesteps', 'components'), pivoted.values)

        nc_file = f'{base_filename}.nc'
        ds.to_netcdf(nc_file)
        print(f"✓ Results saved to {nc_file}")


if __name__ == '__main__':
    # Run benchmark
    results_df = run_benchmark(CONFIG)

    # Print all analyses
    print_summary(results_df)
    print_by_timesteps(results_df)
    print_by_components(results_df)
    print_detailed_results(results_df)

    # Save results
    save_results(results_df)