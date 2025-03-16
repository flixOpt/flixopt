# Linear Converters API Reference

The `linear_converters` module provides pre-defined specialized converters that simplify the usage of the `LinearConverter` class. Common energy system elements like boilers, heat pumps, and CHPs are predefined.
For more advanced LinearConverters, refer to the [LinearConverter API Reference](linear-converter.md).

::: flixOpt.linear_converters.Boiler

::: flixOpt.linear_converters.Power2Heat

::: flixOpt.linear_converters.HeatPump

::: flixOpt.linear_converters.HeatPumpWithSource

::: flixOpt.linear_converters.CoolingTower

::: flixOpt.linear_converters.CHP


## Examples

### Creating a Boiler

```python
import flixOpt as fo

# Create buses
fuel_bus = fo.Bus("Fuel")
heat_bus = fo.Bus("Heat")

# Create flows
fuel_flow = fo.Flow("fuel", fuel_bus)
heat_flow = fo.Flow("heat", heat_bus)

# Create a boiler with 90% efficiency
boiler = fo.linear_converters.Boiler(
    label="Boiler",
    eta=0.9,        # 90% thermal efficiency
    Q_fu=fuel_flow, # Fuel input flow
    Q_th=heat_flow  # Thermal output flow
)
```

### Creating a Heat Pump

```python
import flixOpt as fo

# Create buses
electricity_bus = fo.Bus("Electricity")
heat_bus = fo.Bus("Heat")

# Create flows
power_flow = fo.Flow("power", electricity_bus)
heat_flow = fo.Flow("heat", heat_bus)

# Create a heat pump with COP of 3
heat_pump = fo.linear_converters.HeatPump(
    label="HeatPump",
    COP=3.0,        # Coefficient of Performance
    P_el=power_flow, # Electrical input flow
    Q_th=heat_flow   # Thermal output flow
)
```

### Creating a CHP Unit

```python
import flixOpt as fo

# Create buses
fuel_bus = fo.Bus("Fuel")
electricity_bus = fo.Bus("Electricity")
heat_bus = fo.Bus("Heat")

# Create flows
fuel_flow = fo.Flow("fuel", fuel_bus)
power_flow = fo.Flow("power", electricity_bus)
heat_flow = fo.Flow("heat", heat_bus)

# Create a CHP unit
chp = fo.linear_converters.CHP(
    label="CHP_Unit",
    eta_th=0.45,     # 45% thermal efficiency
    eta_el=0.35,     # 35% electrical efficiency
    Q_fu=fuel_flow,  # Fuel input flow
    P_el=power_flow, # Electrical output flow
    Q_th=heat_flow   # Thermal output flow
)
```

### Creating a Heat Pump with Source

```python
import flixOpt as fo

# Create buses
electricity_bus = fo.Bus("Electricity")
heat_source_bus = fo.Bus("HeatSource")
heat_output_bus = fo.Bus("Heat")

# Create flows
power_flow = fo.Flow("power", electricity_bus)
source_flow = fo.Flow("source", heat_source_bus)
heat_flow = fo.Flow("heat", heat_output_bus)

# Create a heat pump with source
hp_with_source = fo.linear_converters.HeatPumpWithSource(
    label="HeatPump",
    COP=3.5,           # Coefficient of Performance
    P_el=power_flow,   # Electrical input flow
    Q_ab=source_flow,  # Heat source input flow
    Q_th=heat_flow     # Thermal output flow
)
```

These pre-defined components simplify the process of building energy system models by providing specialized implementations of common energy converters.
