"""District heating system: CHP + boiler + heat pump + storage + grids."""

from __future__ import annotations

import numpy as np
import pandas as pd

import flixopt as fx

LABEL = 'District Heating'
SIZES = [24, 168, 720, 2190, 8760]
QUICK_SIZES = [24, 168]


def build(size: int) -> fx.FlowSystem:
    """Build a district heating system with *size* timesteps."""
    timesteps = pd.date_range('2020-01-01', periods=size, freq='h')
    fs = fx.FlowSystem(timesteps)

    # Buses
    heat_bus = fx.Bus('Heat')
    gas_bus = fx.Bus('Gas')
    elec_bus = fx.Bus('Electricity')
    fs.add_elements(heat_bus, gas_bus, elec_bus)

    # Effects
    fs.add_elements(
        fx.Effect('costs', '€', is_objective=True, is_standard=True),
        fx.Effect('CO2', 'kg'),
    )

    # Profiles
    hours = np.arange(size)
    heat_demand = 200 + 100 * np.sin(2 * np.pi * hours / 24) + 50 * np.sin(2 * np.pi * hours / (24 * 365))
    elec_price = 0.12 + 0.05 * np.sin(2 * np.pi * hours / 24)

    # Sources
    fs.add_elements(
        fx.Source(
            'GasTariff',
            outputs=[
                fx.Flow('Gas', bus='Gas', size=2000, effects_per_flow_hour={'costs': 0.04, 'CO2': 0.2}),
            ],
        ),
        fx.Source(
            'ElecGrid',
            outputs=[
                fx.Flow('Elec', bus='Electricity', size=1000, effects_per_flow_hour={'costs': elec_price, 'CO2': 0.4}),
            ],
        ),
    )

    # Sinks
    fs.add_elements(
        fx.Sink(
            'HeatDemand',
            inputs=[
                fx.Flow('Heat', bus='Heat', size=1, fixed_relative_profile=heat_demand),
            ],
        ),
        fx.Sink(
            'ElecExport',
            inputs=[
                fx.Flow('Elec', bus='Electricity', size=500, effects_per_flow_hour={'costs': -0.08}),
            ],
        ),
    )

    # Converters
    fs.add_elements(
        fx.linear_converters.CHP(
            'CHP',
            thermal_efficiency=0.45,
            electrical_efficiency=0.35,
            fuel_flow=fx.Flow('Gas', bus='Gas'),
            electrical_flow=fx.Flow('Elec', bus='Electricity'),
            thermal_flow=fx.Flow('Heat', bus='Heat', size=300),
        ),
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.9,
            fuel_flow=fx.Flow('Gas', bus='Gas'),
            thermal_flow=fx.Flow('Heat', bus='Heat', size=400),
        ),
        fx.linear_converters.HeatPump(
            'HeatPump',
            cop=3.5,
            electrical_flow=fx.Flow('Elec', bus='Electricity', size=100),
            thermal_flow=fx.Flow('Heat', bus='Heat', size=350),
        ),
    )

    # Storage
    fs.add_elements(
        fx.Storage(
            'ThermalStorage',
            charging=fx.Flow('Charge', bus='Heat', size=100),
            discharging=fx.Flow('Discharge', bus='Heat', size=100),
            capacity_in_flow_hours=50,
            initial_charge_state=25,
            eta_charge=0.95,
            eta_discharge=0.95,
            relative_loss_per_hour=0.005,
            prevent_simultaneous_charge_and_discharge=True,
        ),
    )

    return fs
