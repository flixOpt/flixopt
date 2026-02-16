"""Mathematical correctness tests for flow constraints."""

import numpy as np
from numpy.testing import assert_allclose

import flixopt as fx

from .conftest import make_flow_system


class TestFlowConstraints:
    def test_relative_minimum(self, optimize):
        """Proves: relative_minimum enforces a minimum flow rate as a fraction of size
        when the unit is active (status=1).

        Boiler (size=100, relative_minimum=0.4). When on, must produce at least 40 kW.
        Demand=[30,30]. Since 30 < 40, boiler must produce 40 and excess is absorbed.

        Sensitivity: Without relative_minimum, boiler produces exactly 30 each timestep
        → cost=60. With relative_minimum=0.4, must produce 40 → cost=80.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat', imbalance_penalty_per_flow_hour=0),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow(bus='Heat', flow_id='heat', size=1, fixed_relative_profile=np.array([30, 30])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow(bus='Gas', flow_id='gas', effects_per_flow_hour=1),
                ],
            ),
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow(bus='Gas', flow_id='fuel'),
                thermal_flow=fx.Flow(bus='Heat', flow_id='heat', size=100, relative_minimum=0.4),
            ),
        )
        fs = optimize(fs)
        # Must produce at least 40 (relative_minimum=0.4 × size=100)
        # cost = 2 × 40 = 80 (vs 60 without the constraint)
        assert_allclose(fs.solution['costs'].item(), 80.0, rtol=1e-5)
        # Verify flow rate is at least 40
        flow = fs.solution['Boiler(heat)|flow_rate'].values[:-1]
        assert all(f >= 40.0 - 1e-5 for f in flow), f'Flow below relative_minimum: {flow}'

    def test_relative_maximum(self, optimize):
        """Proves: relative_maximum limits the maximum flow rate as a fraction of size.

        Source (size=100, relative_maximum=0.5). Max output = 50 kW.
        Demand=[60,60]. Can only get 50 from CheapSrc, rest from ExpensiveSrc.

        Sensitivity: Without relative_maximum, CheapSrc covers all 60 → cost=120.
        With relative_maximum=0.5, CheapSrc capped at 50 (2×50×1=100),
        ExpensiveSrc covers 10 each timestep (2×10×5=100) → total cost=200.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow(bus='Heat', flow_id='heat', size=1, fixed_relative_profile=np.array([60, 60])),
                ],
            ),
            fx.Source(
                'CheapSrc',
                outputs=[
                    fx.Flow(bus='Heat', flow_id='heat', size=100, relative_maximum=0.5, effects_per_flow_hour=1),
                ],
            ),
            fx.Source(
                'ExpensiveSrc',
                outputs=[
                    fx.Flow(bus='Heat', flow_id='heat', effects_per_flow_hour=5),
                ],
            ),
        )
        fs = optimize(fs)
        # CheapSrc capped at 50 (relative_maximum=0.5 × size=100): 2 × 50 × 1 = 100
        # ExpensiveSrc covers remaining 10 each timestep: 2 × 10 × 5 = 100
        # Total = 200
        assert_allclose(fs.solution['costs'].item(), 200.0, rtol=1e-5)
        # Verify CheapSrc flow rate is at most 50
        flow = fs.solution['CheapSrc(heat)|flow_rate'].values[:-1]
        assert all(f <= 50.0 + 1e-5 for f in flow), f'Flow above relative_maximum: {flow}'

    def test_flow_hours_max(self, optimize):
        """Proves: flow_hours_max limits the total cumulative flow-hours per period.

        CheapSrc (flow_hours_max=30). Total allowed = 30 kWh over horizon.
        Demand=[20,20,20] (total=60). Must split between cheap and expensive.

        Sensitivity: Without flow_hours_max, all from CheapSrc → cost=60.
        With flow_hours_max=30, CheapSrc limited to 30, ExpensiveSrc covers 30 → cost=180.
        """
        fs = make_flow_system(3)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow(bus='Heat', flow_id='heat', size=1, fixed_relative_profile=np.array([20, 20, 20])),
                ],
            ),
            fx.Source(
                'CheapSrc',
                outputs=[
                    fx.Flow(bus='Heat', flow_id='heat', flow_hours_max=30, effects_per_flow_hour=1),
                ],
            ),
            fx.Source(
                'ExpensiveSrc',
                outputs=[
                    fx.Flow(bus='Heat', flow_id='heat', effects_per_flow_hour=5),
                ],
            ),
        )
        fs = optimize(fs)
        # CheapSrc limited to 30 kWh total: 30 × 1 = 30
        # ExpensiveSrc covers remaining 30: 30 × 5 = 150
        # Total = 180
        assert_allclose(fs.solution['costs'].item(), 180.0, rtol=1e-5)
        # Verify total flow hours from CheapSrc
        total_flow = fs.solution['CheapSrc(heat)|flow_rate'].values[:-1].sum()
        assert_allclose(total_flow, 30.0, rtol=1e-5)

    def test_flow_hours_min(self, optimize):
        """Proves: flow_hours_min forces a minimum total cumulative flow-hours per period.

        ExpensiveSrc (flow_hours_min=40). Must produce at least 40 kWh total.
        Demand=[30,30] (total=60). CheapSrc is preferred but ExpensiveSrc must hit 40.

        Sensitivity: Without flow_hours_min, all from CheapSrc → cost=60.
        With flow_hours_min=40, ExpensiveSrc forced to produce 40 → cost=220.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),  # Strict balance (no imbalance penalty = must balance)
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow(bus='Heat', flow_id='heat', size=1, fixed_relative_profile=np.array([30, 30])),
                ],
            ),
            fx.Source(
                'CheapSrc',
                outputs=[
                    fx.Flow(bus='Heat', flow_id='heat', effects_per_flow_hour=1),
                ],
            ),
            fx.Source(
                'ExpensiveSrc',
                outputs=[
                    fx.Flow(bus='Heat', flow_id='heat', flow_hours_min=40, effects_per_flow_hour=5),
                ],
            ),
        )
        fs = optimize(fs)
        # ExpensiveSrc must produce at least 40 kWh: 40 × 5 = 200
        # CheapSrc covers remaining 20 of demand: 20 × 1 = 20
        # Total = 220
        assert_allclose(fs.solution['costs'].item(), 220.0, rtol=1e-5)
        # Verify ExpensiveSrc total is at least 40
        total_exp = fs.solution['ExpensiveSrc(heat)|flow_rate'].values[:-1].sum()
        assert total_exp >= 40.0 - 1e-5, f'ExpensiveSrc total below minimum: {total_exp}'

    def test_load_factor_max(self, optimize):
        """Proves: load_factor_max limits utilization to (flow_hours) / (size × total_hours).

        CheapSrc (size=50, load_factor_max=0.5). Over 2 hours, max flow_hours = 50 × 2 × 0.5 = 50.
        Demand=[40,40] (total=80). CheapSrc capped at 50 total.

        Sensitivity: Without load_factor_max, CheapSrc covers 80 → cost=80.
        With load_factor_max=0.5, CheapSrc limited to 50, ExpensiveSrc covers 30 → cost=200.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow(bus='Heat', flow_id='heat', size=1, fixed_relative_profile=np.array([40, 40])),
                ],
            ),
            fx.Source(
                'CheapSrc',
                outputs=[
                    fx.Flow(bus='Heat', flow_id='heat', size=50, load_factor_max=0.5, effects_per_flow_hour=1),
                ],
            ),
            fx.Source(
                'ExpensiveSrc',
                outputs=[
                    fx.Flow(bus='Heat', flow_id='heat', effects_per_flow_hour=5),
                ],
            ),
        )
        fs = optimize(fs)
        # load_factor_max=0.5 means max flow_hours = 50 × 2 × 0.5 = 50
        # CheapSrc: 50 × 1 = 50
        # ExpensiveSrc: 30 × 5 = 150
        # Total = 200
        assert_allclose(fs.solution['costs'].item(), 200.0, rtol=1e-5)

    def test_load_factor_min(self, optimize):
        """Proves: load_factor_min forces minimum utilization (flow_hours) / (size × total_hours).

        ExpensiveSrc (size=100, load_factor_min=0.3). Over 2 hours, min flow_hours = 100 × 2 × 0.3 = 60.
        Demand=[30,30] (total=60). ExpensiveSrc must produce at least 60.

        Sensitivity: Without load_factor_min, all from CheapSrc → cost=60.
        With load_factor_min=0.3, ExpensiveSrc forced to produce 60 → cost=300.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat', imbalance_penalty_per_flow_hour=0),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow(bus='Heat', flow_id='heat', size=1, fixed_relative_profile=np.array([30, 30])),
                ],
            ),
            fx.Source(
                'CheapSrc',
                outputs=[
                    fx.Flow(bus='Heat', flow_id='heat', effects_per_flow_hour=1),
                ],
            ),
            fx.Source(
                'ExpensiveSrc',
                outputs=[
                    fx.Flow(bus='Heat', flow_id='heat', size=100, load_factor_min=0.3, effects_per_flow_hour=5),
                ],
            ),
        )
        fs = optimize(fs)
        # load_factor_min=0.3 means min flow_hours = 100 × 2 × 0.3 = 60
        # ExpensiveSrc must produce 60: 60 × 5 = 300
        # CheapSrc can produce 0 (demand covered by ExpensiveSrc excess)
        # Total = 300
        assert_allclose(fs.solution['costs'].item(), 300.0, rtol=1e-5)
        # Verify ExpensiveSrc total is at least 60
        total_exp = fs.solution['ExpensiveSrc(heat)|flow_rate'].values[:-1].sum()
        assert total_exp >= 60.0 - 1e-5, f'ExpensiveSrc total below load_factor_min: {total_exp}'
