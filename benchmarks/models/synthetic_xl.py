"""Synthetic XL model: many converters to stress-test component scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd

import flixopt as fx

LABEL = 'Synthetic XL (~50 converters)'
SIZES = [24, 168, 720, 2190]
QUICK_SIZES = [24, 168]

N_CONVERTERS = 50
N_BUSES = 10


def build(size: int) -> fx.FlowSystem:
    """Build a synthetic system with many converters and *size* timesteps."""
    timesteps = pd.date_range('2020-01-01', periods=size, freq='h')
    fs = fx.FlowSystem(timesteps)

    # Effects
    fs.add_elements(fx.Effect('costs', '€', is_objective=True, is_standard=True))

    # Buses
    buses = [fx.Bus(f'Bus_{i}') for i in range(N_BUSES)]
    fs.add_elements(*buses)

    hours = np.arange(size)
    base_demand = 100 + 50 * np.sin(2 * np.pi * hours / 24)

    # Sources and sinks for each bus
    for i, bus in enumerate(buses):
        profile = base_demand + 10 * np.sin(2 * np.pi * hours / (24 * (i + 1)))
        fs.add_elements(
            fx.Source(
                f'Src_{i}',
                outputs=[
                    fx.Flow(f'Out_{i}', bus=bus.label, size=5000, effects_per_flow_hour={'costs': 0.03 + 0.01 * i}),
                ],
            ),
            fx.Sink(
                f'Sink_{i}',
                inputs=[
                    fx.Flow(f'In_{i}', bus=bus.label, size=1, fixed_relative_profile=profile),
                ],
            ),
        )

    # Many converters connecting different buses
    rng = np.random.default_rng(42)
    for c in range(N_CONVERTERS):
        in_bus = buses[c % N_BUSES]
        out_bus = buses[(c + 1) % N_BUSES]
        eff = 0.7 + 0.25 * rng.random()

        fs.add_elements(
            fx.LinearConverter(
                f'Conv_{c}',
                inputs=[fx.Flow('In', bus=in_bus.label)],
                outputs=[fx.Flow('Out', bus=out_bus.label, size=200)],
                conversion_factors=[{'In': eff, 'Out': 1}],
            ),
        )

    return fs
