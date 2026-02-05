"""Mathematical correctness tests for bus balance & dispatch."""

import numpy as np
from numpy.testing import assert_allclose

import flixopt as fx

from .conftest import make_flow_system


class TestBusBalance:
    def test_merit_order_dispatch(self, solve):
        """Proves: Bus balance forces total supply = demand, and the optimizer
        dispatches sources in merit order (cheapest first, up to capacity).

        Src1: 1€/kWh, max 20. Src2: 2€/kWh, max 20. Demand=30 per timestep.
        Optimal: Src1=20, Src2=10.

        Sensitivity: If bus balance allowed oversupply, Src2 could be zero → cost=40.
        If merit order were wrong (Src2 first), cost=100. Only correct bus balance
        with merit order yields cost=80 and the exact flow split [20,10].
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat', imbalance_penalty_per_flow_hour=None),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([30, 30])),
                ],
            ),
            fx.Source(
                'Src1',
                outputs=[
                    fx.Flow('heat', bus='Heat', effects_per_flow_hour=1, size=20),
                ],
            ),
            fx.Source(
                'Src2',
                outputs=[
                    fx.Flow('heat', bus='Heat', effects_per_flow_hour=2, size=20),
                ],
            ),
        )
        solve(fs)
        # Src1 at max 20 @1€, Src2 covers remaining 10 @2€
        # cost = 2*(20*1 + 10*2) = 80
        assert_allclose(fs.solution['costs'].item(), 80.0, rtol=1e-5)
        # Verify individual flows to confirm dispatch split
        src1 = fs.solution['Src1(heat)|flow_rate'].values[:-1]
        src2 = fs.solution['Src2(heat)|flow_rate'].values[:-1]
        assert_allclose(src1, [20, 20], rtol=1e-5)
        assert_allclose(src2, [10, 10], rtol=1e-5)

    def test_imbalance_penalty(self, solve):
        """Proves: imbalance_penalty_per_flow_hour creates a 'Penalty' effect that
        charges for any mismatch between supply and demand on a bus.

        Source fixed at 20, demand=10 → 10 excess per timestep, penalty=100€/kWh.

        Sensitivity: Without the penalty mechanism, objective=40 (fuel only).
        With penalty, objective=2040 (fuel 40 + penalty 2000). The penalty is
        tracked in a separate 'Penalty' effect, not in 'costs'.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat', imbalance_penalty_per_flow_hour=100),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([10, 10])),
                ],
            ),
            fx.Source(
                'Src',
                outputs=[
                    fx.Flow(
                        'heat', bus='Heat', size=1, fixed_relative_profile=np.array([20, 20]), effects_per_flow_hour=1
                    ),
                ],
            ),
        )
        solve(fs)
        # Each timestep: source=20, demand=10, excess=10
        # fuel = 2*20*1 = 40, penalty = 2*10*100 = 2000
        # Penalty goes to separate 'Penalty' effect, not 'costs'
        assert_allclose(fs.solution['costs'].item(), 40.0, rtol=1e-5)
        assert_allclose(fs.solution['Penalty'].item(), 2000.0, rtol=1e-5)
        assert_allclose(fs.solution['objective'].item(), 2040.0, rtol=1e-5)

    def test_prevent_simultaneous_flow_rates(self, solve):
        """Proves: prevent_simultaneous_flow_rates on a Source prevents multiple outputs
        from being active at the same time, forcing sequential operation.

        Source with 2 outputs to 2 buses. Both buses have demand=10 each timestep.
        Output1: 1€/kWh, Output2: 1€/kWh. Without exclusion, both active → cost=40.
        With exclusion, only one output per timestep → must use expensive backup (5€/kWh)
        for the other bus.

        Sensitivity: Without prevent_simultaneous, cost=40. With it, cost=2*(10+50)=120.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat1'),
            fx.Bus('Heat2'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand1',
                inputs=[
                    fx.Flow('heat', bus='Heat1', size=1, fixed_relative_profile=np.array([10, 10])),
                ],
            ),
            fx.Sink(
                'Demand2',
                inputs=[
                    fx.Flow('heat', bus='Heat2', size=1, fixed_relative_profile=np.array([10, 10])),
                ],
            ),
            fx.Source(
                'DualSrc',
                outputs=[
                    fx.Flow('heat1', bus='Heat1', effects_per_flow_hour=1, size=100),
                    fx.Flow('heat2', bus='Heat2', effects_per_flow_hour=1, size=100),
                ],
                prevent_simultaneous_flow_rates=True,
            ),
            fx.Source(
                'Backup1',
                outputs=[
                    fx.Flow('heat', bus='Heat1', effects_per_flow_hour=5),
                ],
            ),
            fx.Source(
                'Backup2',
                outputs=[
                    fx.Flow('heat', bus='Heat2', effects_per_flow_hour=5),
                ],
            ),
        )
        solve(fs)
        # Each timestep: DualSrc serves one bus @1€, backup serves other @5€
        # cost per ts = 10*1 + 10*5 = 60, total = 120
        assert_allclose(fs.solution['costs'].item(), 120.0, rtol=1e-5)
