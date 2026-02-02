"""Simple heat system: boiler + storage + sink."""

from __future__ import annotations

import numpy as np
import pandas as pd

import flixopt as fx

LABEL = 'Simple Heat System'
SIZES = [24, 168, 720, 2190, 8760]
QUICK_SIZES = [24, 168]


def build(size: int) -> fx.FlowSystem:
    """Build a simple heat system with *size* timesteps."""
    timesteps = pd.date_range('2020-01-01', periods=size, freq='h')
    fs = fx.FlowSystem(timesteps)

    # Buses
    heat_bus = fx.Bus('Heat')
    gas_bus = fx.Bus('Gas')
    fs.add_elements(heat_bus, gas_bus)

    # Effects
    fs.add_elements(fx.Effect('costs', '€', is_objective=True, is_standard=True))

    # Demand profile (sinusoidal daily pattern)
    hours = np.arange(size)
    demand = 80 + 40 * np.sin(2 * np.pi * hours / 24)

    # Components
    fs.add_elements(
        fx.Source(
            'GasTariff',
            outputs=[
                fx.Flow('Gas', bus='Gas', size=1000, effects_per_flow_hour={'costs': 0.04}),
            ],
        ),
        fx.Sink(
            'HeatDemand',
            inputs=[
                fx.Flow('Heat', bus='Heat', size=1, fixed_relative_profile=demand),
            ],
        ),
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.9,
            fuel_flow=fx.Flow('Gas', bus='Gas'),
            thermal_flow=fx.Flow('Heat', bus='Heat', size=200),
        ),
        fx.Storage(
            'ThermalStorage',
            charging=fx.Flow('Charge', bus='Heat', size=50),
            discharging=fx.Flow('Discharge', bus='Heat', size=50),
            capacity_in_flow_hours=10,
            initial_charge_state=5,
            eta_charge=0.95,
            eta_discharge=0.95,
            relative_loss_per_hour=0.001,
            prevent_simultaneous_charge_and_discharge=True,
        ),
    )
    return fs
