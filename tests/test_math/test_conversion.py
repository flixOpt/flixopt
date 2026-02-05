"""Mathematical correctness tests for conversion & efficiency."""

import numpy as np
from numpy.testing import assert_allclose

import flixopt as fx

from .conftest import make_flow_system


class TestConversionEfficiency:
    def test_boiler_efficiency(self, optimize):
        """Proves: Boiler applies Q_fu = Q_th / eta to compute fuel consumption.

        Sensitivity: If eta were ignored (treated as 1.0), cost would be 40 instead of 50.
        """
        fs = make_flow_system(3)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([10, 20, 10])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=0.8,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat'),
            ),
        )
        fs = optimize(fs)
        # fuel = (10+20+10)/0.8 = 50, cost@1€/kWh = 50
        assert_allclose(fs.solution['costs'].item(), 50.0, rtol=1e-5)

    def test_variable_efficiency(self, optimize):
        """Proves: Boiler accepts a time-varying efficiency array and applies it per timestep.

        Sensitivity: If a scalar mean (0.75) were used, cost=26.67. If only the first
        value (0.5) were broadcast, cost=40. Only per-timestep application yields 30.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([10, 10])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=np.array([0.5, 1.0]),
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat'),
            ),
        )
        fs = optimize(fs)
        # fuel = 10/0.5 + 10/1.0 = 30
        assert_allclose(fs.solution['costs'].item(), 30.0, rtol=1e-5)

    def test_chp_dual_output(self, optimize):
        """Proves: CHP conversion factors for both thermal and electrical output are correct.
        fuel = Q_th / eta_th, P_el = fuel * eta_el. Revenue from P_el reduces total cost.

        Sensitivity: If electrical output were zero (eta_el broken), cost=200 instead of 40.
        If eta_th were wrong (e.g. 1.0), fuel=100 and cost changes to −60.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Elec'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'HeatDemand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([50, 50])),
                ],
            ),
            fx.Sink(
                'ElecGrid',
                inputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour=-2),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.linear_converters.CHP(
                'CHP',
                thermal_efficiency=0.5,
                electrical_efficiency=0.4,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat'),
                electrical_flow=fx.Flow('elec', bus='Elec'),
            ),
        )
        fs = optimize(fs)
        # Per timestep: fuel = 50/0.5 = 100, elec = 100*0.4 = 40
        # Per timestep cost = 100*1 - 40*2 = 20, total = 2*20 = 40
        assert_allclose(fs.solution['costs'].item(), 40.0, rtol=1e-5)
