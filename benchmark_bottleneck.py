"""

Compact benchmark to identify bottlenecks in FlowSystem sel()/resample() operations.

Measures to_dataset(), from_dataset(), and actual dataset operations separately.



This script uses a decorator-based pattern to easily add new benchmark operations

without modifying the core benchmarking logic.

"""



import timeit

from typing import Callable, Dict, Optional, Any

from dataclasses import dataclass

import numpy as np

import pandas as pd

import xarray as xr

import flixopt as fx





# ============================================================================

# Benchmark Operation Registry Pattern

# ============================================================================



@dataclass

class BenchmarkOperation:

    """Represents a single benchmark operation."""

    name: str

    func: Callable

    description: str = ""

    requires_dataset: bool = False

    enabled: bool = True





class BenchmarkRegistry:

    """Registry for benchmark operations using decorator pattern."""



    def __init__(self):

        self._operations: Dict[str, BenchmarkOperation] = {}



    def register(

        self,

        name: str,

        description: str = "",

        requires_dataset: bool = False,

        enabled: bool = True

    ):

        """Decorator to register a benchmark operation.



        Args:

            name: Unique identifier for the operation

            description: Human-readable description

            requires_dataset: Whether the operation needs a pre-computed dataset

            enabled: Whether the operation is enabled by default



        Usage:

            @registry.register('my_operation', description='Does something')

            def my_operation(flow_system, dataset=None):

                return lambda: some_operation()

        """

        def decorator(func: Callable) -> Callable:

            operation = BenchmarkOperation(

                name=name,

                func=func,

                description=description,

                requires_dataset=requires_dataset,

                enabled=enabled

            )

            self._operations[name] = operation

            return func

        return decorator



    def get_operation(self, name: str) -> Optional[BenchmarkOperation]:

        """Get a registered operation by name."""

        return self._operations.get(name)



    def get_enabled_operations(self) -> Dict[str, BenchmarkOperation]:

        """Get all enabled operations."""

        return {name: op for name, op in self._operations.items() if op.enabled}



    def list_operations(self) -> list[str]:

        """List all registered operation names."""

        return list(self._operations.keys())



    def enable_operation(self, name: str):

        """Enable a specific operation."""

        if name in self._operations:

            self._operations[name].enabled = True



    def disable_operation(self, name: str):

        """Disable a specific operation."""

        if name in self._operations:

            self._operations[name].enabled = False



    def enable_only(self, names: list[str]):

        """Enable only the specified operations, disable all others."""

        for name, op in self._operations.items():

            op.enabled = name in names





# Global registry instance

registry = BenchmarkRegistry()





# ============================================================================

# Benchmark Operation Definitions

# ============================================================================



@registry.register(

    'to_dataset',

    description='Convert FlowSystem to xarray Dataset',

    requires_dataset=False

)

def benchmark_to_dataset(flow_system, dataset=None):

    """Benchmark to_dataset() conversion."""

    return lambda: flow_system.to_dataset()





@registry.register(

    'from_dataset',

    description='Convert xarray Dataset back to FlowSystem',

    requires_dataset=True

)

def benchmark_from_dataset(flow_system, dataset):

    """Benchmark from_dataset() conversion."""

    return lambda: fx.FlowSystem.from_dataset(dataset)





@registry.register(

    'sel',

    description='Select half the timesteps using dataset.sel()',

    requires_dataset=True

)

def benchmark_sel(flow_system, dataset):

    """Benchmark dataset selection operation."""

    sel_slice = slice('2020-01-01', dataset.indexes['time'][len(dataset.indexes['time'])//2])

    return lambda: fx.FlowSystem._dataset_sel(dataset, time=sel_slice)





@registry.register(

    'resample',

    description='Resample dataset to 4-hour frequency',

    requires_dataset=True

)

def benchmark_resample(flow_system, dataset):

    """Benchmark dataset resampling operation."""

    return lambda: fx.FlowSystem._dataset_resample(dataset, '4h')





@registry.register(

    'coarsen',

    description='Coarsen dataset using xarray (reference implementation)',

    requires_dataset=True

)

def benchmark_coarsen(flow_system, dataset):

    """Benchmark xarray coarsen operation as reference."""

    return lambda: dataset.coarsen(time=4).mean()





# ============================================================================

# Example: Adding custom operations (commented out)

# ============================================================================



# @registry.register(

#     'sel_first_10',

#     description='Select first 10 timesteps',

#     requires_dataset=True

# )

# def benchmark_sel_first_10(flow_system, dataset):

#     """Select just the first 10 timesteps."""

#     return lambda: dataset.isel(time=slice(0, 10))





# @registry.register(

#     'aggregate_hourly',

#     description='Aggregate to hourly mean',

#     requires_dataset=True

# )

# def benchmark_aggregate(flow_system, dataset):

#     """Aggregate data to hourly means."""

#     return lambda: dataset.resample(time='1h').mean()





# ============================================================================

# Benchmark Runner

# ============================================================================



class BenchmarkRunner:

    """Runs benchmark operations using the registry pattern."""



    def __init__(self, registry: BenchmarkRegistry):

        self.registry = registry



    def run_operations(self, flow_system, n_runs: int = 1) -> Dict[str, float]:

        """

        Run all enabled benchmark operations.



        Args:

            flow_system: The FlowSystem to benchmark

            n_runs: Number of timing iterations for each operation



        Returns:

            Dict mapping operation names to average execution times

        """

        results = {}



        # Get dataset once for operations that need it

        dataset = None

        enabled_ops = self.registry.get_enabled_operations()



        if any(op.requires_dataset for op in enabled_ops.values()):

            dataset = flow_system.to_dataset()



        # Run each enabled operation

        for name, operation in enabled_ops.items():

            # Get the callable from the operation function

            callable_func = operation.func(flow_system, dataset)



            # Time the operation

            elapsed_time = timeit.timeit(callable_func, number=n_runs) / n_runs

            results[name] = elapsed_time



        return results





# ============================================================================

# FlowSystem Creation

# ============================================================================



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





# ============================================================================

# Main Benchmark Loop

# ============================================================================



def run_benchmark(config, runner: BenchmarkRunner):

    """Run benchmark for all configurations and return results as DataFrame."""

    print("Starting bottleneck benchmark...")

    print(f"Timestep sizes: {config['timestep_sizes']}")

    print(f"Component counts: {config['component_counts']}")

    print(f"Runs per configuration: {config['n_runs']}")

    print(f"Enabled operations: {', '.join(runner.registry.get_enabled_operations().keys())}\n")



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



            # Benchmark operations using the runner

            results = runner.run_operations(flow_system, config['n_runs'])



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

    operation_cols = list(runner.registry.get_enabled_operations().keys())

    df['total'] = df[operation_cols].sum(axis=1)



    return df





# ============================================================================

# Analysis & Reporting Functions

# ============================================================================



def print_summary(df, operation_cols):

    """Print summary statistics."""

    print("\n" + "=" * 100)

    print("SUMMARY: Average timing across all configurations")

    print("=" * 100)



    # Calculate means for operation columns

    all_cols = operation_cols + ['total']

    summary = df[all_cols].mean().to_frame(name='avg_time_s')

    summary['avg_time_ms'] = summary['avg_time_s'] * 1000

    summary['pct_of_total'] = (summary['avg_time_s'] / summary.loc['total', 'avg_time_s'] * 100).fillna(0)



    # Sort by time (excluding total)

    summary_sorted = summary.drop('total').sort_values('avg_time_s', ascending=False)

    summary_sorted = pd.concat([summary_sorted, summary.loc[['total']]])



    print(summary_sorted.to_string(float_format=lambda x: f'{x:.4f}'))





def print_by_timesteps(df, operation_cols):

    """Print results grouped by timesteps."""

    print("\n" + "=" * 100)

    print("SCALING BY TIMESTEPS (averaged over components)")

    print("=" * 100)



    # Group by timesteps and calculate mean

    grouped = df.groupby('timesteps')[operation_cols].mean()



    # Add bottleneck column

    grouped['bottleneck'] = grouped.idxmax(axis=1)



    print(grouped.to_string(float_format=lambda x: f'{x:.4f}'))





def print_by_components(df, operation_cols):

    """Print results grouped by components."""

    print("\n" + "=" * 100)

    print("SCALING BY COMPONENTS (averaged over timesteps)")

    print("=" * 100)



    # Group by components and calculate mean

    grouped = df.groupby('components')[operation_cols].mean()



    # Add bottleneck column

    grouped['bottleneck'] = grouped.idxmax(axis=1)



    print(grouped.to_string(float_format=lambda x: f'{x:.4f}'))





def print_detailed_results(df, operation_cols):

    """Print full detailed results table."""

    print("\n" + "=" * 100)

    print("DETAILED RESULTS (all configurations)")

    print("=" * 100)



    # Select columns to display

    display_cols = ['timesteps', 'components'] + operation_cols + ['total']



    print(df[display_cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))





def save_results(df, operation_cols, base_filename='benchmark_results'):

    """Save results to multiple formats."""

    # Save as CSV

    csv_file = f'{base_filename}.csv'

    df.to_csv(csv_file, index=False, float_format='%.6f')

    print(f"\n✓ Results saved to {csv_file}")



    # Save as NetCDF (via xarray)

    # Pivot to create a proper multidimensional structure

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





# ============================================================================

# Configuration & Main Execution

# ============================================================================



# Configuration

CONFIG = {

    'timestep_sizes': [100, 5000],  # Number of timesteps to test

    'component_counts': [10],       # Number of boilers to test

    'n_runs': 1,                    # Number of timing iterations for each configuration

}





if __name__ == '__main__':

    # Optional: Customize which operations to run

    # registry.enable_only(['to_dataset', 'from_dataset', 'sel'])  # Run only specific ops

    # registry.disable_operation('coarsen')  # Disable a specific operation



    # Create runner with the global registry

    runner = BenchmarkRunner(registry)



    # Run benchmark

    results_df = run_benchmark(CONFIG, runner)



    # Get the list of operation columns from enabled operations

    operation_cols = list(registry.get_enabled_operations().keys())



    # Print all analyses

    print_summary(results_df, operation_cols)

    print_by_timesteps(results_df, operation_cols)

    print_by_components(results_df, operation_cols)

    print_detailed_results(results_df, operation_cols)



    # Save results

    save_results(results_df, operation_cols)

