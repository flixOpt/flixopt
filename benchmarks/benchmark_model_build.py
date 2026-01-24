"""Benchmark script for FlixOpt performance.

Tests various operations: build_model(), LP file write, connect(), transform.

Usage:
    python benchmarks/benchmark_model_build.py              # Run default benchmarks
    python benchmarks/benchmark_model_build.py --all        # Run all system types
    python benchmarks/benchmark_model_build.py --system complex  # Run specific system
"""

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import flixopt as fx


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    n_timesteps: int = 0
    n_periods: int = 0
    n_scenarios: int = 0
    n_components: int = 0
    n_flows: int = 0
    n_vars: int = 0
    n_cons: int = 0
    # Timings (ms)
    connect_ms: float = 0.0
    build_ms: float = 0.0
    write_lp_ms: float = 0.0
    transform_ms: float = 0.0
    # File size
    lp_size_mb: float = 0.0


def _time_it(func, iterations: int = 3, warmup: int = 1) -> tuple[float, float]:
    """Time a function, return (mean_ms, std_ms)."""
    for _ in range(warmup):
        func()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)

    return np.mean(times) * 1000, np.std(times) * 1000


def benchmark_system(create_func, iterations: int = 3) -> BenchmarkResult:
    """Run full benchmark suite for a FlowSystem creator function."""
    result = BenchmarkResult(name=create_func.__name__)

    # Create system and get basic info
    fs = create_func()
    result.n_timesteps = len(fs.timesteps)
    result.n_periods = len(fs.periods) if fs.periods is not None else 0
    result.n_scenarios = len(fs.scenarios) if fs.scenarios is not None else 0
    result.n_components = len(fs.components)
    result.n_flows = len(fs.flows)

    # Benchmark connect (if not already connected)
    def do_connect():
        fs_fresh = create_func()
        fs_fresh.connect_and_transform()

    result.connect_ms, _ = _time_it(do_connect, iterations=iterations)

    # Benchmark build_model
    def do_build():
        fs_fresh = create_func()
        fs_fresh.build_model()
        return fs_fresh

    build_times = []
    for _ in range(iterations):
        fs_fresh = create_func()
        start = time.perf_counter()
        fs_fresh.build_model()
        build_times.append(time.perf_counter() - start)
        result.n_vars = len(fs_fresh.model.variables)
        result.n_cons = len(fs_fresh.model.constraints)

    result.build_ms = np.mean(build_times) * 1000

    # Benchmark LP file write (suppress progress bars)
    import io
    import sys

    fs.build_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        lp_path = os.path.join(tmpdir, 'model.lp')

        def do_write_lp():
            # Suppress linopy progress bars during timing
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                fs.model.to_file(lp_path)
            finally:
                sys.stderr = old_stderr

        result.write_lp_ms, _ = _time_it(do_write_lp, iterations=iterations)
        result.lp_size_mb = os.path.getsize(lp_path) / 1e6

    # Benchmark transform operations (if applicable)
    if result.n_timesteps >= 168:  # Only if enough timesteps for meaningful transform

        def do_transform():
            fs_fresh = create_func()
            # Chain some common transforms
            fs_fresh.transform.sel(
                time=slice(fs_fresh.timesteps[0], fs_fresh.timesteps[min(167, len(fs_fresh.timesteps) - 1)])
            )

        result.transform_ms, _ = _time_it(do_transform, iterations=iterations)

    return result


# =============================================================================
# Example Systems from Notebooks
# =============================================================================


def _get_notebook_data_dir() -> Path:
    """Get the notebook data directory."""
    return Path(__file__).parent.parent / 'docs' / 'notebooks' / 'data'


def load_district_heating() -> fx.FlowSystem:
    """Load district heating system from notebook data."""
    path = _get_notebook_data_dir() / 'district_heating_system.nc4'
    if not path.exists():
        raise FileNotFoundError(f'Run docs/notebooks/data/generate_example_systems.py first: {path}')
    return fx.FlowSystem.from_netcdf(path)


def load_complex_system() -> fx.FlowSystem:
    """Load complex multi-carrier system from notebook data."""
    path = _get_notebook_data_dir() / 'complex_system.nc4'
    if not path.exists():
        raise FileNotFoundError(f'Run docs/notebooks/data/generate_example_systems.py first: {path}')
    return fx.FlowSystem.from_netcdf(path)


def load_multiperiod_system() -> fx.FlowSystem:
    """Load multiperiod system from notebook data."""
    path = _get_notebook_data_dir() / 'multiperiod_system.nc4'
    if not path.exists():
        raise FileNotFoundError(f'Run docs/notebooks/data/generate_example_systems.py first: {path}')
    return fx.FlowSystem.from_netcdf(path)


def load_seasonal_storage() -> fx.FlowSystem:
    """Load seasonal storage system (8760h) from notebook data."""
    path = _get_notebook_data_dir() / 'seasonal_storage_system.nc4'
    if not path.exists():
        raise FileNotFoundError(f'Run docs/notebooks/data/generate_example_systems.py first: {path}')
    return fx.FlowSystem.from_netcdf(path)


# =============================================================================
# Synthetic Systems for Stress Testing
# =============================================================================


def create_large_system(
    n_timesteps: int = 720,
    n_periods: int | None = 2,
    n_scenarios: int | None = None,
    n_converters: int = 20,
    n_storages: int = 5,
    with_status: bool = True,
    with_investment: bool = True,
    with_piecewise: bool = True,
) -> fx.FlowSystem:
    """Create a large synthetic FlowSystem for stress testing.

    Features:
    - Multiple buses (electricity, heat, gas)
    - Multiple effects (costs, CO2)
    - Converters with optional status, investment, piecewise
    - Storages with optional investment
    - Demands and supplies

    Args:
        n_timesteps: Number of timesteps per period.
        n_periods: Number of periods (None for single period).
        n_scenarios: Number of scenarios (None for no scenarios).
        n_converters: Number of converter components.
        n_storages: Number of storage components.
        with_status: Include status variables/constraints.
        with_investment: Include investment variables/constraints.
        with_piecewise: Include piecewise conversion (on some converters).

    Returns:
        Configured FlowSystem.
    """
    timesteps = pd.date_range('2024-01-01', periods=n_timesteps, freq='h')
    periods = pd.Index([2030 + i * 5 for i in range(n_periods)], name='period') if n_periods else None
    scenarios = pd.Index([f'S{i}' for i in range(n_scenarios)], name='scenario') if n_scenarios else None
    scenario_weights = np.ones(n_scenarios) / n_scenarios if n_scenarios else None

    fs = fx.FlowSystem(
        timesteps=timesteps,
        periods=periods,
        scenarios=scenarios,
        scenario_weights=scenario_weights,
    )

    # Effects
    fs.add_elements(
        fx.Effect('costs', '€', 'Total Costs', is_standard=True, is_objective=True),
        fx.Effect('CO2', 'kg', 'CO2 Emissions'),
    )

    # Buses
    fs.add_elements(
        fx.Bus('Electricity'),
        fx.Bus('Heat'),
        fx.Bus('Gas'),
    )

    # Demand profiles (sinusoidal + noise)
    base_profile = 50 + 30 * np.sin(2 * np.pi * np.arange(n_timesteps) / 24)
    heat_profile = base_profile + np.random.normal(0, 5, n_timesteps)
    heat_profile = np.clip(heat_profile / heat_profile.max(), 0.2, 1.0)

    elec_profile = base_profile * 0.5 + np.random.normal(0, 3, n_timesteps)
    elec_profile = np.clip(elec_profile / elec_profile.max(), 0.1, 1.0)

    # Price profiles
    gas_price = 30 + 5 * np.sin(2 * np.pi * np.arange(n_timesteps) / (24 * 7))  # Weekly variation
    elec_price = 50 + 20 * np.sin(2 * np.pi * np.arange(n_timesteps) / 24)  # Daily variation

    # Gas supply
    fs.add_elements(
        fx.Source(
            'GasGrid',
            outputs=[fx.Flow('Gas', bus='Gas', size=5000, effects_per_flow_hour={'costs': gas_price, 'CO2': 0.2})],
        )
    )

    # Electricity grid (buy/sell)
    fs.add_elements(
        fx.Source(
            'ElecBuy',
            outputs=[
                fx.Flow('El', bus='Electricity', size=2000, effects_per_flow_hour={'costs': elec_price, 'CO2': 0.4})
            ],
        ),
        fx.Sink(
            'ElecSell',
            inputs=[fx.Flow('El', bus='Electricity', size=1000, effects_per_flow_hour={'costs': -elec_price * 0.8})],
        ),
    )

    # Demands
    fs.add_elements(
        fx.Sink('HeatDemand', inputs=[fx.Flow('Heat', bus='Heat', size=1, fixed_relative_profile=heat_profile)]),
        fx.Sink('ElecDemand', inputs=[fx.Flow('El', bus='Electricity', size=1, fixed_relative_profile=elec_profile)]),
    )

    # Converters (CHPs and Boilers)
    for i in range(n_converters):
        is_chp = i % 3 != 0  # 2/3 are CHPs, 1/3 are boilers
        use_piecewise = with_piecewise and i % 5 == 0  # Every 5th gets piecewise

        size_param = (
            fx.InvestParameters(
                minimum_size=50,
                maximum_size=200,
                effects_of_investment_per_size={'costs': 100},
                linked_periods=True if n_periods else None,
            )
            if with_investment
            else 150
        )

        status_param = fx.StatusParameters(effects_per_startup={'costs': 500}) if with_status else None

        if is_chp:
            # CHP unit
            if use_piecewise:
                fs.add_elements(
                    fx.LinearConverter(
                        f'CHP_{i}',
                        inputs=[fx.Flow('Gas', bus='Gas', size=300)],
                        outputs=[
                            fx.Flow('El', bus='Electricity', size=100),
                            fx.Flow('Heat', bus='Heat', size=size_param, status_parameters=status_param),
                        ],
                        piecewise_conversion=fx.PiecewiseConversion(
                            {
                                'Gas': fx.Piecewise([fx.Piece(start=100, end=200), fx.Piece(start=200, end=300)]),
                                'El': fx.Piecewise([fx.Piece(start=30, end=70), fx.Piece(start=70, end=100)]),
                                'Heat': fx.Piecewise([fx.Piece(start=50, end=100), fx.Piece(start=100, end=150)]),
                            }
                        ),
                    )
                )
            else:
                fs.add_elements(
                    fx.linear_converters.CHP(
                        f'CHP_{i}',
                        thermal_efficiency=0.50,
                        electrical_efficiency=0.35,
                        thermal_flow=fx.Flow('Heat', bus='Heat', size=size_param, status_parameters=status_param),
                        electrical_flow=fx.Flow('El', bus='Electricity', size=100),
                        fuel_flow=fx.Flow('Gas', bus='Gas'),
                    )
                )
        else:
            # Boiler
            fs.add_elements(
                fx.linear_converters.Boiler(
                    f'Boiler_{i}',
                    thermal_efficiency=0.90,
                    thermal_flow=fx.Flow(
                        'Heat',
                        bus='Heat',
                        size=size_param,
                        relative_minimum=0.2,
                        status_parameters=status_param,
                    ),
                    fuel_flow=fx.Flow('Gas', bus='Gas'),
                )
            )

    # Storages
    for i in range(n_storages):
        capacity_param = (
            fx.InvestParameters(
                minimum_size=0,
                maximum_size=1000,
                effects_of_investment_per_size={'costs': 10},
            )
            if with_investment
            else 500
        )

        fs.add_elements(
            fx.Storage(
                f'Storage_{i}',
                capacity_in_flow_hours=capacity_param,
                initial_charge_state=0,
                eta_charge=0.95,
                eta_discharge=0.95,
                relative_loss_per_hour=0.001,
                charging=fx.Flow('Charge', bus='Heat', size=100),
                discharging=fx.Flow('Discharge', bus='Heat', size=100),
            )
        )

    return fs


# =============================================================================
# Benchmark Runners
# =============================================================================


def run_single_benchmark(name: str, create_func, iterations: int = 3, verbose: bool = True) -> BenchmarkResult:
    """Run full benchmark for a single system."""
    if verbose:
        print(f'  {name}...', end=' ', flush=True)

    result = benchmark_system(create_func, iterations=iterations)
    result.name = name

    if verbose:
        print(f'{result.build_ms:.0f}ms')

    return result


def results_to_dataframe(results: list[BenchmarkResult]) -> pd.DataFrame:
    """Convert benchmark results to a formatted DataFrame."""
    data = []
    for r in results:
        data.append(
            {
                'System': r.name,
                'Timesteps': r.n_timesteps,
                'Periods': r.n_periods,
                'Scenarios': r.n_scenarios,
                'Components': r.n_components,
                'Flows': r.n_flows,
                'Variables': r.n_vars,
                'Constraints': r.n_cons,
                'Connect (ms)': round(r.connect_ms, 1),
                'Build (ms)': round(r.build_ms, 1),
                'Write LP (ms)': round(r.write_lp_ms, 1),
                'Transform (ms)': round(r.transform_ms, 1),
                'LP Size (MB)': round(r.lp_size_mb, 2),
            }
        )
    return pd.DataFrame(data)


def run_all_benchmarks(iterations: int = 3) -> pd.DataFrame:
    """Run benchmarks on all available systems and return DataFrame."""
    print('=' * 70)
    print('FlixOpt Performance Benchmarks')
    print('=' * 70)

    results = []

    # Notebook systems (if available)
    notebook_systems = [
        ('Complex (72h, piecewise)', load_complex_system),
        ('District Heating (744h)', load_district_heating),
        ('Multiperiod (336h×3p×2s)', load_multiperiod_system),
    ]

    print('\nNotebook Example Systems:')
    for name, loader in notebook_systems:
        try:
            results.append(run_single_benchmark(name, loader, iterations))
        except FileNotFoundError:
            print(f'  {name}... SKIPPED (run generate_example_systems.py first)')

    # Synthetic stress-test systems
    print('\nSynthetic Stress-Test Systems:')

    synthetic_systems = [
        (
            'Small (168h, basic)',
            lambda: create_large_system(
                n_timesteps=168,
                n_periods=None,
                n_converters=10,
                n_storages=2,
                with_status=False,
                with_investment=False,
                with_piecewise=False,
            ),
        ),
        (
            'Medium (720h, all features)',
            lambda: create_large_system(
                n_timesteps=720,
                n_periods=None,
                n_converters=20,
                n_storages=5,
                with_status=True,
                with_investment=True,
                with_piecewise=True,
            ),
        ),
        (
            'Large (720h, 50 conv)',
            lambda: create_large_system(
                n_timesteps=720,
                n_periods=None,
                n_converters=50,
                n_storages=10,
                with_status=True,
                with_investment=True,
                with_piecewise=True,
            ),
        ),
        (
            'Multiperiod (720h×3p)',
            lambda: create_large_system(
                n_timesteps=720,
                n_periods=3,
                n_converters=20,
                n_storages=5,
                with_status=True,
                with_investment=True,
                with_piecewise=True,
            ),
        ),
        (
            'Full Year (8760h)',
            lambda: create_large_system(
                n_timesteps=8760,
                n_periods=None,
                n_converters=10,
                n_storages=3,
                with_status=False,
                with_investment=True,
                with_piecewise=False,
            ),
        ),
    ]

    for name, creator in synthetic_systems:
        try:
            results.append(run_single_benchmark(name, creator, iterations))
        except Exception as e:
            print(f'  {name}... ERROR ({e})')

    # Convert to DataFrame and display
    df = results_to_dataframe(results)

    print('\n' + '=' * 70)
    print('Results')
    print('=' * 70)

    # Display timing columns
    timing_cols = ['System', 'Connect (ms)', 'Build (ms)', 'Write LP (ms)', 'LP Size (MB)']
    print('\nTiming Results:')
    print(df[timing_cols].to_string(index=False))

    # Display size columns
    size_cols = ['System', 'Timesteps', 'Components', 'Flows', 'Variables', 'Constraints']
    print('\nModel Size:')
    print(df[size_cols].to_string(index=False))

    return df


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark FlixOpt performance')
    parser.add_argument('--all', '-a', action='store_true', help='Run all benchmarks')
    parser.add_argument(
        '--system',
        '-s',
        choices=['complex', 'district', 'multiperiod', 'seasonal', 'synthetic'],
        help='Run specific system benchmark',
    )
    parser.add_argument('--iterations', '-i', type=int, default=3, help='Number of iterations')
    parser.add_argument('--converters', '-c', type=int, default=20, help='Number of converters (synthetic)')
    parser.add_argument('--timesteps', '-t', type=int, default=720, help='Number of timesteps (synthetic)')
    parser.add_argument('--periods', '-p', type=int, default=None, help='Number of periods (synthetic)')
    args = parser.parse_args()

    if args.all:
        df = run_all_benchmarks(args.iterations)
        return df
    elif args.system:
        loaders = {
            'complex': ('Complex System', load_complex_system),
            'district': ('District Heating', load_district_heating),
            'multiperiod': ('Multiperiod', load_multiperiod_system),
            'seasonal': ('Seasonal Storage (8760h)', load_seasonal_storage),
            'synthetic': (
                'Synthetic',
                lambda: create_large_system(
                    n_timesteps=args.timesteps, n_periods=args.periods, n_converters=args.converters
                ),
            ),
        }
        name, loader = loaders[args.system]
        result = run_single_benchmark(name, loader, args.iterations, verbose=False)
        df = results_to_dataframe([result])
        print(df.to_string(index=False))
        return df
    else:
        # Default: run all benchmarks
        df = run_all_benchmarks(args.iterations)
        return df


if __name__ == '__main__':
    main()
