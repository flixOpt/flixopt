"""Generate example FlowSystem files for the plotting notebook.

This script creates three FlowSystems of varying complexity:
1. simple_system - Basic heat system (boiler + storage + sink)
2. complex_system - Multi-carrier with multiple effects and piecewise efficiency
3. multiperiod_system - System with periods and scenarios

Run this script to regenerate the example data files.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import flixopt as fx

# Output directory (same as this script)
OUTPUT_DIR = Path(__file__).parent


def create_simple_system() -> fx.FlowSystem:
    """Create a simple heat system with boiler, storage, and demand.

    Components:
    - Gas boiler (150 kW)
    - Thermal storage (500 kWh)
    - Office heat demand

    One week, hourly resolution.
    """
    # One week, hourly
    timesteps = pd.date_range('2024-01-15', periods=168, freq='h')

    # Create demand pattern
    hours = np.arange(168)
    hour_of_day = hours % 24
    day_of_week = (hours // 24) % 7

    base_demand = np.where((hour_of_day >= 7) & (hour_of_day <= 18), 80, 30)
    weekend_factor = np.where(day_of_week >= 5, 0.5, 1.0)

    np.random.seed(42)
    heat_demand = base_demand * weekend_factor + np.random.normal(0, 5, len(hours))
    heat_demand = np.clip(heat_demand, 20, 100)

    # Time-varying gas price
    gas_price = np.where((hour_of_day >= 6) & (hour_of_day <= 22), 0.08, 0.05)

    fs = fx.FlowSystem(timesteps)
    fs.add_elements(
        fx.Bus('Gas', carrier='gas'),
        fx.Bus('Heat', carrier='heat'),
        fx.Effect('costs', '€', 'Operating Costs', is_standard=True, is_objective=True),
        fx.Source('GasGrid', outputs=[fx.Flow('Gas', bus='Gas', size=500, effects_per_flow_hour=gas_price)]),
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.92,
            thermal_flow=fx.Flow('Heat', bus='Heat', size=150),
            fuel_flow=fx.Flow('Gas', bus='Gas'),
        ),
        fx.Storage(
            'ThermalStorage',
            capacity_in_flow_hours=500,
            initial_charge_state=250,
            minimal_final_charge_state=200,
            eta_charge=0.98,
            eta_discharge=0.98,
            relative_loss_per_hour=0.005,
            charging=fx.Flow('Charge', bus='Heat', size=100),
            discharging=fx.Flow('Discharge', bus='Heat', size=100),
        ),
        fx.Sink('Office', inputs=[fx.Flow('Heat', bus='Heat', size=1, fixed_relative_profile=heat_demand)]),
    )
    return fs


def create_complex_system() -> fx.FlowSystem:
    """Create a complex multi-carrier system with multiple effects.

    Components:
    - Gas grid (with CO2 emissions)
    - Electricity grid (with time-varying price and CO2)
    - CHP with piecewise efficiency
    - Heat pump
    - Gas boiler (backup)
    - Thermal storage
    - Heat demand

    Effects: costs (objective), CO2

    Three days, hourly resolution.
    """
    timesteps = pd.date_range('2024-06-01', periods=72, freq='h')
    hours = np.arange(72)
    hour_of_day = hours % 24

    # Demand profiles
    np.random.seed(123)
    heat_demand = 50 + 30 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi / 2) + np.random.normal(0, 5, 72)
    heat_demand = np.clip(heat_demand, 20, 100)

    electricity_demand = 20 + 15 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 3, 72)
    electricity_demand = np.clip(electricity_demand, 10, 50)

    # Price profiles
    electricity_price = np.where((hour_of_day >= 8) & (hour_of_day <= 20), 0.25, 0.12)
    gas_price = 0.06

    # CO2 factors (kg/kWh)
    electricity_co2 = np.where((hour_of_day >= 8) & (hour_of_day <= 20), 0.4, 0.3)  # Higher during peak
    gas_co2 = 0.2

    fs = fx.FlowSystem(timesteps)
    fs.add_elements(
        # Buses
        fx.Bus('Gas', carrier='gas'),
        fx.Bus('Electricity', carrier='electricity'),
        fx.Bus('Heat', carrier='heat'),
        # Effects
        fx.Effect('costs', '€', 'Total Costs', is_standard=True, is_objective=True),
        fx.Effect('CO2', 'kg', 'CO2 Emissions'),
        # Gas supply
        fx.Source(
            'GasGrid',
            outputs=[fx.Flow('Gas', bus='Gas', size=300, effects_per_flow_hour={'costs': gas_price, 'CO2': gas_co2})],
        ),
        # Electricity grid (import and export)
        fx.Source(
            'ElectricityImport',
            outputs=[
                fx.Flow(
                    'El',
                    bus='Electricity',
                    size=100,
                    effects_per_flow_hour={'costs': electricity_price, 'CO2': electricity_co2},
                )
            ],
        ),
        fx.Sink(
            'ElectricityExport',
            inputs=[
                fx.Flow('El', bus='Electricity', size=50, effects_per_flow_hour={'costs': -electricity_price * 0.8})
            ],
        ),
        # CHP with piecewise efficiency (efficiency varies with load)
        fx.LinearConverter(
            'CHP',
            inputs=[fx.Flow('Gas', bus='Gas')],
            outputs=[fx.Flow('El', bus='Electricity'), fx.Flow('Heat', bus='Heat')],
            piecewise_conversion=fx.PiecewiseConversion(
                {
                    'Gas': fx.Piecewise(
                        [
                            fx.Piece(start=80, end=160),  # Part load
                            fx.Piece(start=160, end=200),  # Full load
                        ]
                    ),
                    'El': fx.Piecewise(
                        [
                            fx.Piece(start=25, end=60),  # ~31-38% electrical efficiency
                            fx.Piece(start=60, end=80),  # ~38-40% electrical efficiency
                        ]
                    ),
                    'Heat': fx.Piecewise(
                        [
                            fx.Piece(start=35, end=70),  # ~44% thermal efficiency
                            fx.Piece(start=70, end=85),  # ~43% thermal efficiency
                        ]
                    ),
                }
            ),
            status_parameters=fx.StatusParameters(effects_per_active_hour={'costs': 2}),
        ),
        # Heat pump
        fx.linear_converters.HeatPump(
            'HeatPump',
            thermal_flow=fx.Flow('Heat', bus='Heat', size=40),
            electrical_flow=fx.Flow('El', bus='Electricity'),
            cop=3.5,
        ),
        # Backup boiler
        fx.linear_converters.Boiler(
            'BackupBoiler',
            thermal_flow=fx.Flow('Heat', bus='Heat', size=80),
            fuel_flow=fx.Flow('Gas', bus='Gas'),
            thermal_efficiency=0.90,
        ),
        # Thermal storage
        fx.Storage(
            'HeatStorage',
            capacity_in_flow_hours=200,
            initial_charge_state=100,
            eta_charge=0.95,
            eta_discharge=0.95,
            charging=fx.Flow('Charge', bus='Heat', size=50),
            discharging=fx.Flow('Discharge', bus='Heat', size=50),
        ),
        # Demands
        fx.Sink('HeatDemand', inputs=[fx.Flow('Heat', bus='Heat', size=1, fixed_relative_profile=heat_demand)]),
        fx.Sink(
            'ElDemand', inputs=[fx.Flow('El', bus='Electricity', size=1, fixed_relative_profile=electricity_demand)]
        ),
    )
    return fs


def create_multiperiod_system() -> fx.FlowSystem:
    """Create a system with multiple periods and scenarios.

    Same structure as simple system but with:
    - 3 planning periods (years 2024, 2025, 2026)
    - 2 scenarios (high demand, low demand)

    Each period: 48 hours (2 days representative)
    """
    timesteps = pd.date_range('2024-01-01', periods=48, freq='h')
    hour_of_day = np.arange(48) % 24

    # Period definitions (years)
    periods = pd.Index([2024, 2025, 2026], name='period')

    # Scenario definitions
    scenarios = pd.Index(['high_demand', 'low_demand'], name='scenario')
    scenario_weights = np.array([0.3, 0.7])

    # Base demand pattern (hourly)
    base_pattern = np.where((hour_of_day >= 7) & (hour_of_day <= 18), 80.0, 35.0)

    # Scenario-specific scaling
    np.random.seed(42)
    high_demand = base_pattern * 1.2 + np.random.normal(0, 5, 48)
    low_demand = base_pattern * 0.85 + np.random.normal(0, 3, 48)

    # Create DataFrame with scenario columns
    heat_demand = pd.DataFrame(
        {
            'high_demand': np.clip(high_demand, 20, 120),
            'low_demand': np.clip(low_demand, 15, 90),
        },
        index=timesteps,
    )

    # Gas price varies by period (rising costs)
    gas_prices = np.array([0.06, 0.08, 0.10])  # Per period

    fs = fx.FlowSystem(
        timesteps,
        periods=periods,
        scenarios=scenarios,
        scenario_weights=scenario_weights,
    )

    fs.add_elements(
        fx.Bus('Gas', carrier='gas'),
        fx.Bus('Heat', carrier='heat'),
        fx.Effect('costs', '€', 'Operating Costs', is_standard=True, is_objective=True),
        fx.Source('GasGrid', outputs=[fx.Flow('Gas', bus='Gas', size=500, effects_per_flow_hour=gas_prices)]),
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.92,
            thermal_flow=fx.Flow('Heat', bus='Heat', size=200),
            fuel_flow=fx.Flow('Gas', bus='Gas'),
        ),
        fx.Storage(
            'ThermalStorage',
            capacity_in_flow_hours=300,
            initial_charge_state=150,
            eta_charge=0.98,
            eta_discharge=0.98,
            charging=fx.Flow('Charge', bus='Heat', size=80),
            discharging=fx.Flow('Discharge', bus='Heat', size=80),
        ),
        fx.Sink('Building', inputs=[fx.Flow('Heat', bus='Heat', size=1, fixed_relative_profile=heat_demand)]),
    )
    return fs


def main():
    """Generate all example systems and save to netCDF."""
    solver = fx.solvers.HighsSolver(log_to_console=False)

    systems = [
        ('simple_system', create_simple_system),
        ('complex_system', create_complex_system),
        ('multiperiod_system', create_multiperiod_system),
    ]

    for name, create_func in systems:
        print(f'Creating {name}...')
        fs = create_func()

        print('  Optimizing...')
        fs.optimize(solver)

        output_path = OUTPUT_DIR / f'{name}.nc4'
        print(f'  Saving to {output_path}...')
        fs.to_netcdf(output_path, overwrite=True)

        print(f'  Done. Objective: {fs.solution["objective"].item():.2f}')
        print()

    print('All systems generated successfully!')


if __name__ == '__main__':
    main()
