"""Mathematical correctness tests for status (on/off) variables."""

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

import flixopt as fx

from .conftest import make_flow_system, solve


class TestStatusVariables:
    def test_startup_cost(self):
        """Proves: effects_per_startup adds a fixed cost each time the unit transitions to on.

        Demand pattern [0,10,0,10,0] forces 2 start-up events.

        Sensitivity: Without startup costs, objective=40 (fuel only).
        With 100€/startup × 2 startups, objective=240.
        """
        fs = make_flow_system(5)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([0, 10, 0, 10, 0])),
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
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=100,
                    status_parameters=fx.StatusParameters(effects_per_startup=100),
                ),
            ),
        )
        solve(fs)
        # fuel = (10+10)/0.5 = 40, startups = 2, cost = 40 + 200 = 240
        assert_allclose(fs.solution['costs'].item(), 240.0, rtol=1e-5)

    def test_active_hours_max(self):
        """Proves: active_hours_max limits the total number of on-hours for a unit.

        Cheap boiler (eta=1.0) limited to 1 hour; expensive backup (eta=0.5).
        Optimizer assigns the single cheap hour to the highest-demand timestep (t=1, 20kW).

        Sensitivity: Without the limit, cheap boiler runs all 3 hours → cost=40.
        With limit=1, forced to use expensive backup for 2 hours → cost=60.
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
                'CheapBoiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=100,
                    status_parameters=fx.StatusParameters(active_hours_max=1),
                ),
            ),
            fx.linear_converters.Boiler(
                'ExpensiveBoiler',
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat', size=100),
            ),
        )
        solve(fs)
        # CheapBoiler runs at t=1 (biggest demand): cost = 20*1 = 20
        # ExpensiveBoiler covers t=0 and t=2: cost = (10+10)/0.5 = 40
        # Total = 60
        assert_allclose(fs.solution['costs'].item(), 60.0, rtol=1e-5)

    def test_min_uptime_forces_operation(self):
        """Proves: min_uptime forces a unit to stay on for at least N consecutive hours
        once started, even if cheaper to turn off earlier.

        Cheap boiler (eta=0.5) with min_uptime=2 and max_uptime=2 → must run in
        blocks of exactly 2 hours. Expensive backup (eta=0.2).
        demand = [5, 10, 20, 18, 12]. Optimal: boiler on t=0,1 and t=3,4; backup at t=2.

        Sensitivity: Without min_uptime (but with max_uptime=2), the boiler could
        run at t=2 and t=3 (highest demand) and let backup cover the rest, yielding
        a different cost and status pattern. The constraint forces status=[1,1,0,1,1].
        """
        fs = fx.FlowSystem(pd.date_range('2020-01-01', periods=5, freq='h'))
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([5, 10, 20, 18, 12])),
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
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=100,
                    previous_flow_rate=0,
                    status_parameters=fx.StatusParameters(min_uptime=2, max_uptime=2),
                ),
            ),
            fx.linear_converters.Boiler(
                'Backup',
                thermal_efficiency=0.2,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat', size=100),
            ),
        )
        solve(fs)
        # Boiler on t=0,1 (block of 2) and t=3,4 (block of 2). Off at t=2 → backup.
        # Boiler fuel: (5+10+18+12)/0.5 = 90. Backup fuel: 20/0.2 = 100. Total = 190.
        assert_allclose(fs.solution['costs'].item(), 190.0, rtol=1e-5)
        assert_allclose(
            fs.solution['Boiler(heat)|status'].values[:-1],
            [1, 1, 0, 1, 1],
            atol=1e-5,
        )

    def test_min_downtime_prevents_restart(self):
        """Proves: min_downtime prevents a unit from restarting before N consecutive
        off-hours have elapsed.

        Cheap boiler (eta=1.0, min_downtime=3) was on before the horizon
        (previous_flow_rate=20). demand = [20, 0, 20, 0]. Boiler serves t=0,
        turns off at t=1. Must stay off for t=1,2,3 → cannot serve t=2.
        Expensive backup (eta=0.5) covers t=2.

        Sensitivity: Without min_downtime, boiler restarts at t=2 → cost=40.
        With min_downtime=3, backup needed at t=2 → cost=60.
        """
        fs = make_flow_system(4)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([20, 0, 20, 0])),
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
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=100,
                    previous_flow_rate=20,
                    status_parameters=fx.StatusParameters(min_downtime=3),
                ),
            ),
            fx.linear_converters.Boiler(
                'Backup',
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat', size=100),
            ),
        )
        solve(fs)
        # t=0: Boiler on (fuel=20). Turns off at t=1.
        # min_downtime=3: must stay off t=1,2,3. Can't restart at t=2.
        # Backup covers t=2: fuel = 20/0.5 = 40.
        # Without min_downtime: boiler at t=2 (fuel=20), total=40 vs 60.
        assert_allclose(fs.solution['costs'].item(), 60.0, rtol=1e-5)
        # Verify boiler off at t=2 (where demand exists but can't restart)
        assert_allclose(fs.solution['Boiler(heat)|status'].values[2], 0.0, atol=1e-5)
