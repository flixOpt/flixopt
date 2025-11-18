"""
This script shows how to use the flixopt framework to model a super minimalistic energy system in the most concise way possible.
THis can also be used to create proposals for new features, bug reports etc
"""

import numpy as np
import pandas as pd

import flixopt as fx

if __name__ == '__main__':
    fx.CONFIG.silent()
    flow_system = fx.FlowSystem(pd.date_range('2020-01-01', periods=3, freq='h'))

    flow_system.add_elements(
        fx.Bus('Heat'),
        fx.Bus('Gas'),
        fx.Effect('Costs', '€', 'Cost', is_standard=True, is_objective=True),
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.5,
            thermal_flow=fx.Flow(label='Heat', bus='Heat', size=50),
            fuel_flow=fx.Flow(label='Gas', bus='Gas'),
        ),
        fx.Sink(
            'Sink',
            inputs=[fx.Flow(label='Demand', bus='Heat', size=1, fixed_relative_profile=np.array([30, 0, 20]))],
        ),
        fx.Source(
            'Source',
            outputs=[fx.Flow(label='Gas', bus='Gas', size=1000, effects_per_flow_hour=0.04)],
        ),
    )

    calculation = fx.FullCalculation('Simulation1', flow_system).do_modeling().solve(fx.solvers.HighsSolver(0.01, 60))
    calculation.results['Heat'].plot_node_balance()
