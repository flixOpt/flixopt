# Bundled Data Sources

## Weather Data (TMY)

**File:** `tmy_dresden.csv`
**Location:** Dresden, Germany (51.05°N, 13.74°E)
**Source:** PVGIS - Photovoltaic Geographical Information System
**Provider:** European Commission Joint Research Centre
**License:** Free for any use
**URL:** https://re.jrc.ec.europa.eu/pvg_tools/en/

**Columns:**
- `temperature_C`: 2m air temperature (°C)
- `ghi_W_m2`: Global horizontal irradiance (W/m²)
- `dni_W_m2`: Direct normal irradiance (W/m²)
- `dhi_W_m2`: Diffuse horizontal irradiance (W/m²)
- `wind_speed_m_s`: Wind speed at 10m (m/s)
- `relative_humidity_percent`: Relative humidity (%)

## Electricity Prices

**File:** `electricity_prices_de_2020.csv`
**Coverage:** Germany, Jan-Sep 2020, hourly
**Source:** Open Power System Data
**License:** Open Database License (ODbL)
**URL:** https://data.open-power-system-data.org/time_series/

**Attribution required:** "Data from Open Power System Data. https://open-power-system-data.org"

**Columns:**
- `price_eur_mwh`: Day-ahead electricity price (EUR/MWh)
