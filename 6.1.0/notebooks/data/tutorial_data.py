"""Generate tutorial data for notebooks 01-07.

These functions return data (timesteps, profiles, prices) rather than full FlowSystems,
so notebooks can demonstrate building systems step by step.

Usage:
    from data.tutorial_data import get_quickstart_data, get_heat_system_data, ...
"""

import numpy as np
import pandas as pd
import xarray as xr


def get_quickstart_data() -> dict:
    """Data for 01-quickstart: minimal 4-hour example.

    Returns:
        dict with: timesteps, heat_demand (xr.DataArray)
    """
    timesteps = pd.date_range('2024-01-15 08:00', periods=4, freq='h')
    heat_demand = xr.DataArray(
        [30, 50, 45, 25],
        dims=['time'],
        coords={'time': timesteps},
        name='Heat Demand [kW]',
    )
    return {
        'timesteps': timesteps,
        'heat_demand': heat_demand,
    }


def get_heat_system_data() -> dict:
    """Data for 02-heat-system: one week with storage.

    Returns:
        dict with: timesteps, heat_demand, gas_price (arrays)
    """
    timesteps = pd.date_range('2024-01-15', periods=168, freq='h')
    hours = np.arange(168)
    hour_of_day = hours % 24
    day_of_week = (hours // 24) % 7

    # Office heat demand pattern
    base_demand = np.where((hour_of_day >= 7) & (hour_of_day <= 18), 80, 30)
    weekend_factor = np.where(day_of_week >= 5, 0.5, 1.0)
    np.random.seed(42)
    heat_demand = base_demand * weekend_factor + np.random.normal(0, 5, len(timesteps))
    heat_demand = np.clip(heat_demand, 20, 100)

    # Time-of-use gas prices
    gas_price = np.where((hour_of_day >= 6) & (hour_of_day <= 22), 0.08, 0.05)

    return {
        'timesteps': timesteps,
        'heat_demand': heat_demand,
        'gas_price': gas_price,
    }


def get_investment_data() -> dict:
    """Data for 03-investment-optimization: solar pool heating.

    Returns:
        dict with: timesteps, solar_profile, pool_demand, costs
    """
    timesteps = pd.date_range('2024-07-15', periods=168, freq='h')
    hours = np.arange(168)
    hour_of_day = hours % 24

    # Solar profile
    solar_profile = np.maximum(0, np.sin((hour_of_day - 6) * np.pi / 12)) * 0.8
    solar_profile = np.where((hour_of_day >= 6) & (hour_of_day <= 20), solar_profile, 0)
    np.random.seed(42)
    solar_profile = solar_profile * np.random.uniform(0.6, 1.0, len(timesteps))

    # Pool demand
    pool_demand = np.where((hour_of_day >= 8) & (hour_of_day <= 22), 150, 50)

    return {
        'timesteps': timesteps,
        'solar_profile': solar_profile,
        'pool_demand': pool_demand,
        'gas_price': 0.12,
        'solar_cost_per_kw_week': 20 / 52,
        'tank_cost_per_kwh_week': 1.5 / 52,
    }


def get_constraints_data() -> dict:
    """Data for 04-operational-constraints: factory steam demand.

    Returns:
        dict with: timesteps, steam_demand
    """
    timesteps = pd.date_range('2024-03-11', periods=72, freq='h')
    hours = np.arange(72)
    hour_of_day = hours % 24

    # Shift-based demand
    steam_demand = np.select(
        [
            (hour_of_day >= 6) & (hour_of_day < 14),
            (hour_of_day >= 14) & (hour_of_day < 22),
        ],
        [400, 350],
        default=80,
    ).astype(float)

    np.random.seed(123)
    steam_demand = steam_demand + np.random.normal(0, 20, len(steam_demand))
    steam_demand = np.clip(steam_demand, 50, 450)

    return {
        'timesteps': timesteps,
        'steam_demand': steam_demand,
    }


def get_multicarrier_data() -> dict:
    """Data for 05-multi-carrier-system: hospital CHP.

    Returns:
        dict with: timesteps, electricity_demand, heat_demand, prices
    """
    timesteps = pd.date_range('2024-02-05', periods=168, freq='h')
    hours = np.arange(168)
    hour_of_day = hours % 24

    # Electricity demand
    elec_base = 150
    elec_daily = 100 * np.sin((hour_of_day - 6) * np.pi / 12)
    elec_daily = np.maximum(0, elec_daily)
    electricity_demand = elec_base + elec_daily

    # Heat demand
    heat_pattern = np.select(
        [
            (hour_of_day >= 5) & (hour_of_day < 9),
            (hour_of_day >= 9) & (hour_of_day < 17),
            (hour_of_day >= 17) & (hour_of_day < 22),
        ],
        [350, 250, 300],
        default=200,
    ).astype(float)

    np.random.seed(456)
    electricity_demand += np.random.normal(0, 15, len(timesteps))
    heat_demand = heat_pattern + np.random.normal(0, 20, len(timesteps))
    electricity_demand = np.clip(electricity_demand, 100, 300)
    heat_demand = np.clip(heat_demand, 150, 400)

    # Prices
    elec_buy_price = np.where((hour_of_day >= 7) & (hour_of_day <= 21), 0.35, 0.20)

    return {
        'timesteps': timesteps,
        'electricity_demand': electricity_demand,
        'heat_demand': heat_demand,
        'elec_buy_price': elec_buy_price,
        'elec_sell_price': 0.12,
        'gas_price': 0.05,
    }


def get_time_varying_data() -> dict:
    """Data for 06a-time-varying-parameters: heat pump with variable COP.

    Returns:
        dict with: timesteps, outdoor_temp, heat_demand, cop
    """
    timesteps = pd.date_range('2024-01-22', periods=168, freq='h')
    hours = np.arange(168)
    hour_of_day = hours % 24

    # Outdoor temperature
    temp_base = 2
    temp_amplitude = 5
    outdoor_temp = temp_base + temp_amplitude * np.sin((hour_of_day - 6) * np.pi / 12)
    np.random.seed(789)
    outdoor_temp = outdoor_temp + np.repeat(np.random.uniform(-3, 3, 7), 24)

    # Heat demand (inversely related to temperature)
    heat_demand = 200 - 8 * outdoor_temp
    heat_demand = np.clip(heat_demand, 100, 300)

    # COP calculation
    t_supply = 45 + 273.15
    t_source = outdoor_temp + 273.15
    carnot_cop = t_supply / (t_supply - t_source)
    cop = np.clip(0.45 * carnot_cop, 2.0, 5.0)

    return {
        'timesteps': timesteps,
        'outdoor_temp': outdoor_temp,
        'heat_demand': heat_demand,
        'cop': cop,
    }


def get_scenarios_data() -> dict:
    """Data for 07-scenarios-and-periods: multi-year planning.

    Returns:
        dict with: timesteps, periods, scenarios, weights, heat_demand (DataFrame), prices
    """
    timesteps = pd.date_range('2024-01-15', periods=168, freq='h')
    periods = pd.Index([2024, 2025, 2026], name='period')
    scenarios = pd.Index(['Mild Winter', 'Harsh Winter'], name='scenario')
    scenario_weights = np.array([0.6, 0.4])

    hours = np.arange(168)
    hour_of_day = hours % 24

    # Base pattern
    daily_pattern = np.select(
        [
            (hour_of_day >= 6) & (hour_of_day < 9),
            (hour_of_day >= 9) & (hour_of_day < 17),
            (hour_of_day >= 17) & (hour_of_day < 22),
        ],
        [180, 120, 160],
        default=100,
    ).astype(float)

    np.random.seed(42)
    noise = np.random.normal(0, 10, len(timesteps))

    mild_demand = np.clip(daily_pattern * 0.8 + noise, 60, 200)
    harsh_demand = np.clip(daily_pattern * 1.3 + noise * 1.5, 100, 280)

    heat_demand = pd.DataFrame(
        {'Mild Winter': mild_demand, 'Harsh Winter': harsh_demand},
        index=timesteps,
    )

    return {
        'timesteps': timesteps,
        'periods': periods,
        'scenarios': scenarios,
        'scenario_weights': scenario_weights,
        'heat_demand': heat_demand,
        'gas_prices': np.array([0.06, 0.08, 0.10]),
        'elec_prices': np.array([0.28, 0.34, 0.43]),
    }
