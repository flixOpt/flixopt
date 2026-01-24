"""Benchmark script for model build performance.

Tests build_model() with various FlowSystem configurations to measure performance.

Usage:
    python benchmarks/benchmark_model_build.py              # Run default benchmarks
    python benchmarks/benchmark_model_build.py --all        # Run all system types
    python benchmarks/benchmark_model_build.py --system complex  # Run specific system
"""

import time
from pathlib import Path
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
    n_vars: int = 0
    n_cons: int = 0


def benchmark_build(create_func, iterations: int = 3, warmup: int = 1) -> BenchmarkResult:
    """Benchmark build_model() for a FlowSystem creator function."""
    # Warmup
    for _ in range(warmup):
        fs = create_func()
        fs.build_model()

    # Timed runs
    times = []
    n_vars = n_cons = 0
    for _ in range(iterations):
        fs = create_func()
        start = time.perf_counter()
        fs.build_model()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        n_vars = len(fs.model.variables)
        n_cons = len(fs.model.constraints)

    return BenchmarkResult(
        name=create_func.__name__,
        mean_ms=np.mean(times) * 1000,
        std_ms=np.std(times) * 1000,
        iterations=iterations,
        n_vars=n_vars,
        n_cons=n_cons,
    )


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


def run_single_benchmark(name: str, create_func, iterations: int = 3) -> BenchmarkResult:
    """Run benchmark for a single system."""
    print(f'\n{name}:')

    # Get system info
    fs = create_func()
    print(
        f'  Timesteps: {len(fs.timesteps)}, Periods: {len(fs.periods) if fs.periods is not None else 0}, '
        f'Scenarios: {len(fs.scenarios) if fs.scenarios is not None else 0}'
    )
    print(f'  Components: {len(fs.components)}, Flows: {len(fs.flows)}')

    # Benchmark
    result = benchmark_build(create_func, iterations=iterations)
    print(f'  Build: {result.mean_ms:.1f}ms (±{result.std_ms:.1f}ms)')
    print(f'  Variables: {result.n_vars}, Constraints: {result.n_cons}')

    return result


def run_all_benchmarks(iterations: int = 3):
    """Run benchmarks on all available systems."""
    print('=' * 70)
    print('FlixOpt Model Build Benchmarks')
    print('=' * 70)

    results = {}

    # Notebook systems (if available)
    notebook_systems = [
        ('Complex System (72h)', load_complex_system),
        ('District Heating (744h)', load_district_heating),
        ('Multiperiod (336h x 3 periods x 2 scenarios)', load_multiperiod_system),
    ]

    print('\n--- Notebook Example Systems ---')
    for name, loader in notebook_systems:
        try:
            results[name] = run_single_benchmark(name, loader, iterations)
        except FileNotFoundError as e:
            print(f'\n{name}: SKIPPED ({e})')

    # Synthetic stress-test systems
    print('\n--- Synthetic Stress-Test Systems ---')

    synthetic_systems = [
        (
            'Small (168h, 10 conv, no features)',
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
            'Medium (720h, 20 conv, all features)',
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
            'Large (720h, 50 conv, all features)',
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
            'Multiperiod (720h x 3 periods, 20 conv)',
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
            'Full Year (8760h, 10 conv, basic)',
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
            results[name] = run_single_benchmark(name, creator, iterations)
        except Exception as e:
            print(f'\n{name}: ERROR ({e})')

    # Summary table
    print('\n' + '=' * 70)
    print('Summary')
    print('=' * 70)
    print(f'\n  {"System":<45} {"Build (ms)":>12} {"Vars":>8} {"Cons":>8}')
    print(f'  {"-" * 45} {"-" * 12} {"-" * 8} {"-" * 8}')

    for name, res in results.items():
        print(f'  {name:<45} {res.mean_ms:>9.1f}ms {res.n_vars:>8} {res.n_cons:>8}')

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark FlixOpt model build performance')
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
        run_all_benchmarks(args.iterations)
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
        run_single_benchmark(name, loader, args.iterations)
    else:
        # Default: run a quick benchmark with synthetic system
        print('Running default benchmark (use --all for comprehensive benchmarks)')
        run_single_benchmark(
            'Default (720h, 20 converters)',
            lambda: create_large_system(n_timesteps=720, n_converters=20),
            iterations=args.iterations,
        )


if __name__ == '__main__':
    main()
