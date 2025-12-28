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

try:
    from .generate_realistic_profiles import (
        ElectricityLoadGenerator,
        GasPriceGenerator,
        ThermalLoadGenerator,
        load_electricity_prices,
        load_weather,
    )
except ImportError:
    from generate_realistic_profiles import (
        ElectricityLoadGenerator,
        GasPriceGenerator,
        ThermalLoadGenerator,
        load_electricity_prices,
        load_weather,
    )

import flixopt as fx

# Output directory (same as this script)
try:
    OUTPUT_DIR = Path(__file__).parent
    DATA_DIR = Path(__file__).parent  # Zeitreihen2020.csv is in the same directory
except NameError:
    # Running in notebook context (e.g., mkdocs-jupyter)
    OUTPUT_DIR = Path('docs/notebooks/data')
    DATA_DIR = Path('docs/notebooks/data')

# Load shared data
_weather = load_weather()
_elec_prices = load_electricity_prices()
_elec_prices.index = _elec_prices.index.tz_localize(None)  # Remove timezone for compatibility


def create_simple_system() -> fx.FlowSystem:
    """Create a simple heat system with boiler, storage, and demand.

    Components:
    - Gas boiler (150 kW)
    - Thermal storage (500 kWh)
    - Office heat demand (BDEW profile)

    One week (January 2020), hourly resolution.
    Uses realistic BDEW heat demand and seasonal gas prices.
    """
    # One week, hourly (January 2020 for realistic data)
    timesteps = pd.date_range('2020-01-15', periods=168, freq='h')
    temp = _weather.loc[timesteps, 'temperature_C'].values

    # BDEW office heat demand profile (scaled to fit 150 kW boiler)
    thermal_gen = ThermalLoadGenerator()
    heat_demand = thermal_gen.generate(timesteps, temp, 'office', annual_demand_kwh=15_000)

    # Seasonal gas price
    gas_gen = GasPriceGenerator()
    gas_price = gas_gen.generate(timesteps) / 1000  # EUR/kWh

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
    - Heat demand (BDEW retail profile)
    - Electricity demand (BDEW commercial profile)

    Effects: costs (objective), CO2

    Three days (June 2020), hourly resolution.
    Uses realistic BDEW profiles and OPSD electricity prices.
    """
    timesteps = pd.date_range('2020-06-01', periods=72, freq='h')
    temp = _weather.loc[timesteps, 'temperature_C'].values

    # BDEW demand profiles (scaled to fit component sizes)
    thermal_gen = ThermalLoadGenerator()
    heat_demand = thermal_gen.generate(timesteps, temp, 'retail', annual_demand_kwh=2_000)

    elec_gen = ElectricityLoadGenerator()
    electricity_demand = elec_gen.generate(timesteps, 'commercial', annual_demand_kwh=50_000)

    # Real electricity prices (OPSD) and seasonal gas prices
    electricity_price = _elec_prices.reindex(timesteps, method='ffill').values / 1000  # EUR/kWh
    gas_gen = GasPriceGenerator()
    gas_price = gas_gen.generate(timesteps) / 1000  # EUR/kWh

    # CO2 factors (kg/kWh) - higher during peak hours
    hour_of_day = timesteps.hour.values
    electricity_co2 = np.where((hour_of_day >= 8) & (hour_of_day <= 20), 0.4, 0.3)
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
    """Create a district heating system with BDEW profiles.

    Uses realistic German data:
    - One month (January 2020), hourly resolution
    - BDEW industrial heat profile
    - BDEW commercial electricity profile
    - OPSD electricity prices
    - Seasonal gas prices
    - CHP, boiler, storage, and grid connections
    - Investment optimization for sizing

    Used by: 08a-aggregation, 08c-clustering, 08e-clustering-internals notebooks
    """
    # One month, hourly
    timesteps = pd.date_range('2020-01-01', '2020-01-31 23:00:00', freq='h')
    temp = _weather.loc[timesteps, 'temperature_C'].values

    # BDEW profiles (MW scale for district heating)
    thermal_gen = ThermalLoadGenerator()
    heat_demand = thermal_gen.generate(timesteps, temp, 'industrial', annual_demand_kwh=15_000_000) / 1000  # MW

    elec_gen = ElectricityLoadGenerator()
    electricity_demand = elec_gen.generate(timesteps, 'commercial', annual_demand_kwh=5_000_000) / 1000  # MW

    # Prices
    electricity_price = _elec_prices.reindex(timesteps, method='ffill').values  # EUR/MWh
    gas_gen = GasPriceGenerator()
    gas_price = gas_gen.generate(timesteps)  # EUR/MWh

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

    Uses realistic German data (two weeks, January 2020):
    - BDEW industrial heat profile
    - BDEW commercial electricity profile
    - OPSD electricity prices
    - Seasonal gas prices
    - CHP with startup costs
    - Boiler with startup costs
    - Storage with fixed capacity
    - No investment parameters (for rolling horizon optimization)

    Used by: 08b-rolling-horizon notebook
    """
    # Two weeks, hourly
    timesteps = pd.date_range('2020-01-01', '2020-01-14 23:00:00', freq='h')
    temp = _weather.loc[timesteps, 'temperature_C'].values

    # BDEW profiles (MW scale)
    thermal_gen = ThermalLoadGenerator()
    heat_demand = thermal_gen.generate(timesteps, temp, 'industrial', annual_demand_kwh=15_000_000) / 1000  # MW

    elec_gen = ElectricityLoadGenerator()
    electricity_demand = elec_gen.generate(timesteps, 'commercial', annual_demand_kwh=5_000_000) / 1000  # MW

    # Prices
    electricity_price = _elec_prices.reindex(timesteps, method='ffill').values  # EUR/MWh
    gas_gen = GasPriceGenerator()
    gas_price = gas_gen.generate(timesteps)  # EUR/MWh

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
    - Solar thermal from PVGIS irradiance data
    - Heat demand from BDEW industrial profile
    - Large seasonal pit storage (bridges seasons)
    - Gas boiler backup

    This system clearly shows the value of inter-cluster storage linking:
    - Summer: excess solar heat stored in pit
    - Winter: stored heat reduces gas consumption

    Uses realistic PVGIS solar irradiance and BDEW heat profiles.
    Used by: 08c-clustering, 08c2-clustering-storage-modes notebooks
    """
    # Full year, hourly (use non-leap year to match TMY data which has 8760 hours)
    timesteps = pd.date_range('2019-01-01', periods=8760, freq='h')
    # Map to 2020 weather data (TMY has 8760 hours, no Feb 29)
    temp = _weather['temperature_C'].values
    ghi = _weather['ghi_W_m2'].values

    # --- Solar thermal profile from PVGIS irradiance ---
    # Normalize GHI to 0-1 range and apply collector efficiency
    solar_profile = ghi / 1000  # Normalized (1000 W/m² = 1.0)
    solar_profile = np.clip(solar_profile, 0, 1)

    # --- Heat demand from BDEW industrial profile ---
    # Scale to MW (district heating scale)
    # Use 2019 year for demandlib (non-leap year)
    thermal_gen = ThermalLoadGenerator(year=2019)
    heat_demand_kw = thermal_gen.generate(timesteps, temp, 'industrial', annual_demand_kwh=20_000_000)
    heat_demand = heat_demand_kw / 1000  # Convert to MW

    # --- Gas price with seasonal variation ---
    gas_gen = GasPriceGenerator()
    gas_price = gas_gen.generate(timesteps)  # EUR/MWh

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
                maximum_size=50000,  # MWh - large for seasonal storage
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

    Uses BDEW residential heat profile as base, scaled for scenarios.
    """
    n_hours = 336  # 2 weeks
    timesteps = pd.date_range('2020-01-01', periods=n_hours, freq='h')
    temp = _weather.loc[timesteps, 'temperature_C'].values

    # Period definitions (years)
    periods = pd.Index([2024, 2025, 2026], name='period')

    # Scenario definitions
    scenarios = pd.Index(['high_demand', 'low_demand'], name='scenario')
    scenario_weights = np.array([0.3, 0.7])

    # BDEW residential heat profile as base (scaled to fit 250 kW boiler with scenarios)
    thermal_gen = ThermalLoadGenerator()
    base_demand = thermal_gen.generate(timesteps, temp, 'residential', annual_demand_kwh=30_000)

    # Scenario-specific scaling
    high_demand = base_demand * 1.3
    low_demand = base_demand * 0.7

    # Create DataFrame with scenario columns
    heat_demand = pd.DataFrame(
        {
            'high_demand': high_demand,
            'low_demand': low_demand,
        },
        index=timesteps,
    )

    # Gas price varies by period (rising costs, based on seasonal price)
    gas_gen = GasPriceGenerator()
    base_gas = gas_gen.generate(timesteps).mean() / 1000  # Average EUR/kWh
    gas_prices = np.array([base_gas, base_gas * 1.2, base_gas * 1.5])  # Rising costs per period

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

        output_path = OUTPUT_DIR / f'{name}.nc4'
        print(f'  Saving to {output_path}...')
        fs.to_netcdf(output_path, overwrite=True)

    print('All systems generated successfully!')


if __name__ == '__main__':
    main()
