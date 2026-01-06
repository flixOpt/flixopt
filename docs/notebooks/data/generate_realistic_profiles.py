"""Generate realistic German energy profiles for flixOpt examples.

This module provides functions to create realistic time series data for:
- Thermal load profiles (BDEW standard load profiles via demandlib)
- Electricity load profiles (BDEW standard load profiles via demandlib)
- Solar generation profiles (via pvlib)
- Energy prices (bundled OPSD data)
- Weather data (bundled PVGIS TMY data for Dresden)

Example:
    >>> from generate_realistic_profiles import load_weather, ThermalLoadGenerator
    >>> weather = load_weather()
    >>> thermal = ThermalLoadGenerator()
    >>> heat_demand = thermal.generate(weather.index, weather['temperature_C'], 'residential', 50000)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import pvlib
from demandlib import bdew

# Reset warnings to default after imports. Some dependencies (demandlib, pvlib)
# may configure warnings during import. This ensures consistent warning behavior
# when this module is used in different contexts (scripts, notebooks, tests).
warnings.resetwarnings()

# Data directory
DATA_DIR = Path(__file__).parent / 'raw'


# === Data Loading ===


def load_weather() -> pd.DataFrame:
    """Load PVGIS TMY weather data for Dresden.

    Returns
    -------
    pd.DataFrame
        Hourly weather data with columns:
        - temperature_C: Ambient temperature (°C)
        - ghi_W_m2: Global horizontal irradiance (W/m²)
        - dni_W_m2: Direct normal irradiance (W/m²)
        - dhi_W_m2: Diffuse horizontal irradiance (W/m²)
        - wind_speed_m_s: Wind speed at 10m (m/s)
    """
    return pd.read_csv(DATA_DIR / 'tmy_dresden.csv', parse_dates=['time'], index_col='time')


def load_electricity_prices() -> pd.Series:
    """Load German day-ahead electricity prices (2020).

    Returns
    -------
    pd.Series
        Hourly electricity prices in EUR/MWh
    """
    df = pd.read_csv(DATA_DIR / 'electricity_prices_de_2020.csv', parse_dates=['time'], index_col='time')
    return df['price_eur_mwh']


# === Profile Generators ===


class ThermalLoadGenerator:
    """Generate thermal load profiles using BDEW standard load profiles.

    Uses demandlib to create realistic heat demand profiles based on
    German BDEW (Bundesverband der Energie- und Wasserwirtschaft) standards.
    """

    BUILDING_TYPES = {
        'residential': {'shlp_type': 'EFH', 'building_class': 5},  # Single-family house
        'residential_multi': {'shlp_type': 'MFH', 'building_class': 5},  # Multi-family
        'office': {'shlp_type': 'GKO', 'building_class': 0},  # Commercial office
        'retail': {'shlp_type': 'GHA', 'building_class': 0},  # Retail/shops
        'industrial': {'shlp_type': 'GMK', 'building_class': 0},  # Industrial
    }

    def __init__(self, year: int = 2020):
        self.year = year
        self.holidays = holidays.Germany(years=year)

    def generate(
        self,
        timesteps: pd.DatetimeIndex,
        temperature: np.ndarray | pd.Series,
        building_type: str = 'residential',
        annual_demand_kwh: float = 20000,
    ) -> np.ndarray:
        """Generate thermal load profile.

        Parameters
        ----------
        timesteps
            Time index for the profile
        temperature
            Ambient temperature in Celsius (same length as timesteps)
        building_type
            One of: 'residential', 'residential_multi', 'office', 'retail', 'industrial'
        annual_demand_kwh
            Total annual heat demand in kWh

        Returns
        -------
        np.ndarray
            Heat demand profile in kW
        """
        params = self.BUILDING_TYPES[building_type]
        temp_series = pd.Series(temperature, index=timesteps)

        profile = bdew.HeatBuilding(
            timesteps,
            holidays=self.holidays,
            temperature=temp_series,
            shlp_type=params['shlp_type'],
            building_class=params['building_class'],
            wind_class=0,
            annual_heat_demand=annual_demand_kwh,
            name=building_type,
        )
        return profile.get_bdew_profile().values


class ElectricityLoadGenerator:
    """Generate electricity load profiles using BDEW standard load profiles."""

    CONSUMER_TYPES = {
        'household': 'h0',
        'commercial': 'g0',
        'commercial_office': 'g1',
        'commercial_retail': 'g4',
        'agricultural': 'l0',
    }

    def __init__(self, year: int = 2020):
        self.year = year
        self.holidays = holidays.Germany(years=year)

    def generate(
        self,
        timesteps: pd.DatetimeIndex,
        consumer_type: str = 'household',
        annual_demand_kwh: float = 4000,
    ) -> np.ndarray:
        """Generate electricity load profile.

        Parameters
        ----------
        timesteps
            Time index for the profile
        consumer_type
            One of: 'household', 'commercial', 'commercial_office', 'commercial_retail', 'agricultural'
        annual_demand_kwh
            Total annual electricity demand in kWh

        Returns
        -------
        np.ndarray
            Electricity demand profile in kW
        """
        slp_type = self.CONSUMER_TYPES[consumer_type]
        e_slp = bdew.ElecSlp(self.year, holidays=self.holidays)
        profile = e_slp.get_scaled_power_profiles({slp_type: annual_demand_kwh})
        # Resample to hourly and align with requested timesteps
        profile_hourly = profile[slp_type].resample('h').mean()
        return profile_hourly.reindex(timesteps, method='ffill').values


class SolarGenerator:
    """Generate solar irradiance and PV generation profiles using pvlib.

    Uses Dresden location (51.05°N, 13.74°E) as default.
    """

    def __init__(self, latitude: float = 51.05, longitude: float = 13.74):
        self.location = pvlib.location.Location(latitude, longitude, 'Europe/Berlin', 120, 'Dresden')

    def generate_pv_profile(
        self,
        timesteps: pd.DatetimeIndex,
        weather: pd.DataFrame,
        surface_tilt: float = 35,
        surface_azimuth: float = 180,  # South-facing
        capacity_kw: float = 1.0,
    ) -> np.ndarray:
        """Generate PV power output profile.

        Parameters
        ----------
        timesteps
            Time index for the profile
        weather
            Weather data with 'ghi_W_m2', 'dni_W_m2', 'dhi_W_m2', 'temperature_C'
        surface_tilt
            Panel tilt angle in degrees (0=horizontal, 90=vertical)
        surface_azimuth
            Panel azimuth in degrees (180=south, 90=east, 270=west)
        capacity_kw
            Installed PV capacity in kW

        Returns
        -------
        np.ndarray
            PV power output in kW
        """
        # Ensure weather is aligned with timesteps
        weather = weather.reindex(timesteps, method='ffill')

        # Get solar position
        solar_position = self.location.get_solarposition(timesteps)

        # Calculate plane-of-array irradiance
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            solar_zenith=solar_position['apparent_zenith'],
            solar_azimuth=solar_position['azimuth'],
            dni=weather['dni_W_m2'],
            ghi=weather['ghi_W_m2'],
            dhi=weather['dhi_W_m2'],
        )

        # Simple efficiency model: ~15% module efficiency, ~85% system efficiency
        system_efficiency = 0.15 * 0.85
        pv_output = poa['poa_global'] * system_efficiency * capacity_kw / 1000

        return np.clip(pv_output.fillna(0).values, 0, capacity_kw)


class GasPriceGenerator:
    """Generate synthetic gas price profiles with seasonal variation."""

    def generate(
        self,
        timesteps: pd.DatetimeIndex,
        base_price: float = 35,
        winter_premium: float = 10,
    ) -> np.ndarray:
        """Generate gas price profile.

        Parameters
        ----------
        timesteps
            Time index for the profile
        base_price
            Base gas price in EUR/MWh
        winter_premium
            Additional winter price in EUR/MWh

        Returns
        -------
        np.ndarray
            Gas prices in EUR/MWh
        """
        day_of_year = timesteps.dayofyear.values
        # Peak in mid-January (day 15), trough in mid-July
        seasonal = winter_premium * np.cos(2 * np.pi * (day_of_year - 15) / 365)
        return base_price + seasonal
