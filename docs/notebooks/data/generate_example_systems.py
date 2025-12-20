"""Generate example FlowSystem files for notebooks.

This script creates FlowSystems of varying complexity:
1. simple_system - Basic heat system (boiler + storage + sink)
2. complex_system - Multi-carrier with multiple effects and piecewise efficiency
3. multiperiod_system - System with periods and scenarios
4. district_heating_system - Real-world district heating data with investments (1 month)
5. operational_system - Real-world district heating for operational planning (2 weeks, no investments)
6. seasonal_storage_system - Solar thermal + seasonal pit storage (full year, 8760h)

Run this script to regenerate the example data files.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import flixopt as fx

# Output directory (same as this script)
try:
    OUTPUT_DIR = Path(__file__).parent
    DATA_DIR = Path(__file__).parent  # Zeitreihen2020.csv is in the same directory
except NameError:
    # Running in notebook context (e.g., mkdocs-jupyter)
    OUTPUT_DIR = Path('docs/notebooks/data')
    DATA_DIR = Path('docs/notebooks/data')


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
    fs.add_carriers(
        fx.Carrier('gas', '#3498db', 'kW'),
        fx.Carrier('heat', '#e74c3c', 'kW'),
    )
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
    fs.add_carriers(
        fx.Carrier('gas', '#3498db', 'kW'),
        fx.Carrier('electricity', '#f1c40f', 'kW'),
        fx.Carrier('heat', '#e74c3c', 'kW'),
    )
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
            inputs=[fx.Flow('Gas', bus='Gas', size=200)],
            outputs=[fx.Flow('El', bus='Electricity', size=80), fx.Flow('Heat', bus='Heat', size=85)],
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
        # Heat pump (with investment)
        fx.linear_converters.HeatPump(
            'HeatPump',
            thermal_flow=fx.Flow(
                'Heat',
                bus='Heat',
                size=fx.InvestParameters(
                    effects_of_investment={'costs': 500},
                    effects_of_investment_per_size={'costs': 100},
                    maximum_size=60,
                ),
            ),
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
        # Thermal storage (with investment)
        fx.Storage(
            'HeatStorage',
            capacity_in_flow_hours=fx.InvestParameters(
                effects_of_investment={'costs': 200},
                effects_of_investment_per_size={'costs': 10},
                maximum_size=300,
            ),
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


def create_district_heating_system() -> fx.FlowSystem:
    """Create a district heating system using real-world data.

    Based on Zeitreihen2020.csv data:
    - One month of data at 15-minute resolution
    - CHP, boiler, storage, and grid connections
    - Investment optimization for sizing

    Used by: 08a-aggregation, 08b-rolling-horizon, 08c-clustering notebooks
    """
    # Load real data
    data_path = DATA_DIR / 'Zeitreihen2020.csv'
    data = pd.read_csv(data_path, index_col=0, parse_dates=True).sort_index()
    data = data['2020-01-01':'2020-01-31 23:45:00']  # One month
    data.index.name = 'time'

    timesteps = data.index
    electricity_demand = data['P_Netz/MW'].to_numpy()
    heat_demand = data['Q_Netz/MW'].to_numpy()
    electricity_price = data['Strompr.€/MWh'].to_numpy()
    gas_price = data['Gaspr.€/MWh'].to_numpy()

    fs = fx.FlowSystem(timesteps)
    fs.add_elements(
        # Buses
        fx.Bus('Electricity'),
        fx.Bus('Heat'),
        fx.Bus('Gas'),
        fx.Bus('Coal'),
        # Effects
        fx.Effect('costs', '€', 'Total Costs', is_standard=True, is_objective=True),
        fx.Effect('CO2', 'kg', 'CO2 Emissions'),
        # CHP unit with investment
        fx.linear_converters.CHP(
            'CHP',
            thermal_efficiency=0.58,
            electrical_efficiency=0.22,
            electrical_flow=fx.Flow('P_el', bus='Electricity', size=200),
            thermal_flow=fx.Flow(
                'Q_th',
                bus='Heat',
                size=fx.InvestParameters(
                    minimum_size=100,
                    maximum_size=300,
                    effects_of_investment_per_size={'costs': 10},
                ),
                relative_minimum=0.3,
                status_parameters=fx.StatusParameters(),
            ),
            fuel_flow=fx.Flow('Q_fu', bus='Coal'),
        ),
        # Gas Boiler with investment
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.85,
            thermal_flow=fx.Flow(
                'Q_th',
                bus='Heat',
                size=fx.InvestParameters(
                    minimum_size=0,
                    maximum_size=150,
                    effects_of_investment_per_size={'costs': 5},
                ),
                relative_minimum=0.1,
                status_parameters=fx.StatusParameters(),
            ),
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
        ),
        # Thermal Storage with investment
        fx.Storage(
            'Storage',
            capacity_in_flow_hours=fx.InvestParameters(
                minimum_size=0,
                maximum_size=1000,
                effects_of_investment_per_size={'costs': 0.5},
            ),
            initial_charge_state=0,
            eta_charge=1,
            eta_discharge=1,
            relative_loss_per_hour=0.001,
            charging=fx.Flow('Charge', size=137, bus='Heat'),
            discharging=fx.Flow('Discharge', size=158, bus='Heat'),
        ),
        # Fuel sources
        fx.Source(
            'GasGrid',
            outputs=[fx.Flow('Q_Gas', bus='Gas', size=1000, effects_per_flow_hour={'costs': gas_price, 'CO2': 0.3})],
        ),
        fx.Source(
            'CoalSupply',
            outputs=[fx.Flow('Q_Coal', bus='Coal', size=1000, effects_per_flow_hour={'costs': 4.6, 'CO2': 0.3})],
        ),
        # Electricity grid
        fx.Source(
            'GridBuy',
            outputs=[
                fx.Flow(
                    'P_el',
                    bus='Electricity',
                    size=1000,
                    effects_per_flow_hour={'costs': electricity_price + 0.5, 'CO2': 0.3},
                )
            ],
        ),
        fx.Sink(
            'GridSell',
            inputs=[fx.Flow('P_el', bus='Electricity', size=1000, effects_per_flow_hour=-(electricity_price - 0.5))],
        ),
        # Demands
        fx.Sink('HeatDemand', inputs=[fx.Flow('Q_th', bus='Heat', size=1, fixed_relative_profile=heat_demand)]),
        fx.Sink(
            'ElecDemand', inputs=[fx.Flow('P_el', bus='Electricity', size=1, fixed_relative_profile=electricity_demand)]
        ),
    )
    return fs


def create_operational_system() -> fx.FlowSystem:
    """Create an operational district heating system (no investments).

    Based on Zeitreihen2020.csv data (two weeks):
    - CHP with startup costs
    - Boiler with startup costs
    - Storage with fixed capacity
    - No investment parameters (for rolling horizon optimization)

    Used by: 08b-rolling-horizon notebook
    """
    # Load real data
    data_path = DATA_DIR / 'Zeitreihen2020.csv'
    data = pd.read_csv(data_path, index_col=0, parse_dates=True).sort_index()
    data = data['2020-01-01':'2020-01-14 23:45:00']  # Two weeks
    data.index.name = 'time'

    timesteps = data.index
    electricity_demand = data['P_Netz/MW'].to_numpy()
    heat_demand = data['Q_Netz/MW'].to_numpy()
    electricity_price = data['Strompr.€/MWh'].to_numpy()
    gas_price = data['Gaspr.€/MWh'].to_numpy()

    fs = fx.FlowSystem(timesteps)
    fs.add_elements(
        fx.Bus('Electricity'),
        fx.Bus('Heat'),
        fx.Bus('Gas'),
        fx.Bus('Coal'),
        fx.Effect('costs', '€', 'Total Costs', is_standard=True, is_objective=True),
        fx.Effect('CO2', 'kg', 'CO2 Emissions'),
        # CHP with startup costs
        fx.linear_converters.CHP(
            'CHP',
            thermal_efficiency=0.58,
            electrical_efficiency=0.22,
            status_parameters=fx.StatusParameters(effects_per_startup=24000),
            electrical_flow=fx.Flow('P_el', bus='Electricity', size=200),
            thermal_flow=fx.Flow('Q_th', bus='Heat', size=200),
            fuel_flow=fx.Flow('Q_fu', bus='Coal', size=288, relative_minimum=87 / 288, previous_flow_rate=100),
        ),
        # Boiler with startup costs
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.85,
            thermal_flow=fx.Flow('Q_th', bus='Heat'),
            fuel_flow=fx.Flow(
                'Q_fu',
                bus='Gas',
                size=95,
                relative_minimum=12 / 95,
                previous_flow_rate=20,
                status_parameters=fx.StatusParameters(effects_per_startup=1000),
            ),
        ),
        # Storage with fixed capacity
        fx.Storage(
            'Storage',
            capacity_in_flow_hours=684,
            initial_charge_state=137,
            minimal_final_charge_state=137,
            maximal_final_charge_state=158,
            eta_charge=1,
            eta_discharge=1,
            relative_loss_per_hour=0.001,
            prevent_simultaneous_charge_and_discharge=True,
            charging=fx.Flow('Charge', size=137, bus='Heat'),
            discharging=fx.Flow('Discharge', size=158, bus='Heat'),
        ),
        fx.Source(
            'GasGrid',
            outputs=[fx.Flow('Q_Gas', bus='Gas', size=1000, effects_per_flow_hour={'costs': gas_price, 'CO2': 0.3})],
        ),
        fx.Source(
            'CoalSupply',
            outputs=[fx.Flow('Q_Coal', bus='Coal', size=1000, effects_per_flow_hour={'costs': 4.6, 'CO2': 0.3})],
        ),
        fx.Source(
            'GridBuy',
            outputs=[
                fx.Flow(
                    'P_el',
                    bus='Electricity',
                    size=1000,
                    effects_per_flow_hour={'costs': electricity_price + 0.5, 'CO2': 0.3},
                )
            ],
        ),
        fx.Sink(
            'GridSell',
            inputs=[fx.Flow('P_el', bus='Electricity', size=1000, effects_per_flow_hour=-(electricity_price - 0.5))],
        ),
        fx.Sink('HeatDemand', inputs=[fx.Flow('Q_th', bus='Heat', size=1, fixed_relative_profile=heat_demand)]),
        fx.Sink(
            'ElecDemand', inputs=[fx.Flow('P_el', bus='Electricity', size=1, fixed_relative_profile=electricity_demand)]
        ),
    )
    return fs


def create_seasonal_storage_system() -> fx.FlowSystem:
    """Create a district heating system with solar thermal and seasonal storage.

    Demonstrates seasonal storage value with:
    - Full year at hourly resolution (8760 timesteps)
    - Solar thermal: high in summer, low in winter
    - Heat demand: high in winter, low in summer
    - Large seasonal pit storage (bridges seasons)
    - Gas boiler backup

    This system clearly shows the value of inter-cluster storage linking:
    - Summer: excess solar heat stored in pit
    - Winter: stored heat reduces gas consumption

    Used by: 08c-clustering, 08c2-clustering-storage-modes notebooks
    """
    # Full year, hourly
    timesteps = pd.date_range('2024-01-01', periods=8760, freq='h')
    hours = np.arange(8760)
    hour_of_day = hours % 24
    day_of_year = hours // 24

    np.random.seed(42)

    # --- Solar irradiance profile ---
    # Seasonal variation: peaks in summer (day ~180), low in winter
    seasonal_solar = 0.5 + 0.5 * np.cos(2 * np.pi * (day_of_year - 172) / 365)  # Peak around June 21

    # Daily variation: peaks at noon
    daily_solar = np.maximum(0, np.cos(2 * np.pi * (hour_of_day - 12) / 24))

    # Combine and scale (MW of solar thermal potential per MW installed)
    solar_profile = seasonal_solar * daily_solar
    solar_profile = solar_profile * (0.8 + 0.2 * np.random.random(8760))  # Add some variation
    solar_profile = np.clip(solar_profile, 0, 1)

    # --- Heat demand profile ---
    # Seasonal: high in winter, low in summer
    seasonal_demand = 0.6 + 0.4 * np.cos(2 * np.pi * day_of_year / 365)  # Peak Jan 1

    # Daily: higher during day, lower at night
    daily_demand = 0.7 + 0.3 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)

    # Combine and scale to ~5 MW peak
    heat_demand = 5 * seasonal_demand * daily_demand
    heat_demand = heat_demand * (0.9 + 0.2 * np.random.random(8760))  # Add variation
    heat_demand = np.clip(heat_demand, 0.5, 6)  # MW

    # --- Gas price (slight seasonal variation) ---
    gas_price = 40 + 10 * np.cos(2 * np.pi * day_of_year / 365)  # €/MWh, higher in winter

    fs = fx.FlowSystem(timesteps)
    fs.add_carriers(
        fx.Carrier('gas', '#3498db', 'MW'),
        fx.Carrier('heat', '#e74c3c', 'MW'),
    )
    fs.add_elements(
        # Buses
        fx.Bus('Gas', carrier='gas'),
        fx.Bus('Heat', carrier='heat'),
        # Effects
        fx.Effect('costs', '€', 'Total Costs', is_standard=True, is_objective=True),
        fx.Effect('CO2', 'kg', 'CO2 Emissions'),
        # Solar thermal collector (investment) - profile includes 70% collector efficiency
        # Costs annualized for single-year analysis
        fx.Source(
            'SolarThermal',
            outputs=[
                fx.Flow(
                    'Q_th',
                    bus='Heat',
                    size=fx.InvestParameters(
                        minimum_size=0,
                        maximum_size=20,  # MW peak
                        effects_of_investment_per_size={'costs': 15000},  # €/MW (annualized)
                    ),
                    fixed_relative_profile=solar_profile * 0.7,  # 70% collector efficiency
                )
            ],
        ),
        # Gas boiler (backup)
        fx.linear_converters.Boiler(
            'GasBoiler',
            thermal_efficiency=0.90,
            thermal_flow=fx.Flow(
                'Q_th',
                bus='Heat',
                size=fx.InvestParameters(
                    minimum_size=0,
                    maximum_size=8,  # MW
                    effects_of_investment_per_size={'costs': 20000},  # €/MW (annualized)
                ),
            ),
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
        ),
        # Gas supply (higher price makes solar+storage more attractive)
        fx.Source(
            'GasGrid',
            outputs=[
                fx.Flow(
                    'Q_gas',
                    bus='Gas',
                    size=20,
                    effects_per_flow_hour={'costs': gas_price * 1.5, 'CO2': 0.2},  # €/MWh
                )
            ],
        ),
        # Seasonal pit storage (large capacity for seasonal shifting)
        fx.Storage(
            'SeasonalStorage',
            capacity_in_flow_hours=fx.InvestParameters(
                minimum_size=0,
                maximum_size=5000,  # MWh - large for seasonal storage
                effects_of_investment_per_size={'costs': 20},  # €/MWh (pit storage is cheap)
            ),
            initial_charge_state='equals_final',  # Yearly cyclic
            eta_charge=0.95,
            eta_discharge=0.95,
            relative_loss_per_hour=0.0001,  # Very low losses for pit storage
            charging=fx.Flow(
                'Charge',
                bus='Heat',
                size=fx.InvestParameters(maximum_size=10, effects_of_investment_per_size={'costs': 5000}),
            ),
            discharging=fx.Flow(
                'Discharge',
                bus='Heat',
                size=fx.InvestParameters(maximum_size=10, effects_of_investment_per_size={'costs': 5000}),
            ),
        ),
        # Heat demand
        fx.Sink(
            'HeatDemand',
            inputs=[fx.Flow('Q_th', bus='Heat', size=1, fixed_relative_profile=heat_demand)],
        ),
    )
    return fs


def create_multiperiod_system() -> fx.FlowSystem:
    """Create a system with multiple periods and scenarios.

    Same structure as simple system but with:
    - 3 planning periods (years 2024, 2025, 2026)
    - 2 scenarios (high demand, low demand)

    Each period: 336 hours (2 weeks) - suitable for clustering demonstrations.
    Use transform.sisel() to select subsets if needed.
    """
    n_hours = 336  # 2 weeks
    timesteps = pd.date_range('2024-01-01', periods=n_hours, freq='h')
    hour_of_day = np.arange(n_hours) % 24
    day_of_week = (np.arange(n_hours) // 24) % 7

    # Period definitions (years)
    periods = pd.Index([2024, 2025, 2026], name='period')

    # Scenario definitions
    scenarios = pd.Index(['high_demand', 'low_demand'], name='scenario')
    scenario_weights = np.array([0.3, 0.7])

    # Base demand pattern (hourly) with daily and weekly variation
    base_pattern = np.where((hour_of_day >= 7) & (hour_of_day <= 18), 80.0, 35.0)
    weekend_factor = np.where(day_of_week >= 5, 0.6, 1.0)
    base_pattern = base_pattern * weekend_factor

    # Scenario-specific scaling
    np.random.seed(42)
    high_demand = base_pattern * 1.3 + np.random.normal(0, 8, n_hours)
    low_demand = base_pattern * 0.8 + np.random.normal(0, 5, n_hours)

    # Create DataFrame with scenario columns
    heat_demand = pd.DataFrame(
        {
            'high_demand': np.clip(high_demand, 20, 150),
            'low_demand': np.clip(low_demand, 15, 100),
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
    fs.add_carriers(
        fx.Carrier('gas', '#3498db', 'kW'),
        fx.Carrier('heat', '#e74c3c', 'kW'),
    )
    fs.add_elements(
        fx.Bus('Gas', carrier='gas'),
        fx.Bus('Heat', carrier='heat'),
        fx.Effect('costs', '€', 'Operating Costs', is_standard=True, is_objective=True),
        fx.Source('GasGrid', outputs=[fx.Flow('Gas', bus='Gas', size=500, effects_per_flow_hour=gas_prices)]),
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.92,
            thermal_flow=fx.Flow(
                'Heat',
                bus='Heat',
                size=fx.InvestParameters(
                    effects_of_investment={'costs': 1000},
                    effects_of_investment_per_size={'costs': 50},
                    maximum_size=250,
                ),
            ),
            fuel_flow=fx.Flow('Gas', bus='Gas'),
        ),
        fx.Storage(
            'ThermalStorage',
            capacity_in_flow_hours=fx.InvestParameters(
                effects_of_investment={'costs': 500},
                effects_of_investment_per_size={'costs': 15},
                maximum_size=400,
            ),
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
        ('district_heating_system', create_district_heating_system),
        ('operational_system', create_operational_system),
        ('seasonal_storage_system', create_seasonal_storage_system),
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
