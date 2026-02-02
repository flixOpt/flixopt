"""Mathematical correctness tests for effects & objective."""

import numpy as np
from numpy.testing import assert_allclose

import flixopt as fx

from .conftest import make_flow_system, solve


class TestEffects:
    def test_effects_per_flow_hour(self):
        """Proves: effects_per_flow_hour correctly accumulates flow × rate for each
        named effect independently.

        Source has costs=2€/kWh and CO2=0.5kg/kWh. Total flow=30.

        Sensitivity: If effects_per_flow_hour were ignored, both effects=0. If only
        one effect were applied, the other would be wrong. Both values (60€, 15kg)
        are uniquely determined by the rates and total flow.
        """
        fs = make_flow_system(2)
        co2 = fx.Effect('CO2', 'kg')
        costs = fx.Effect('costs', '€', is_standard=True, is_objective=True)
        fs.add_elements(
            fx.Bus('Heat'),
            costs,
            co2,
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([10, 20])),
                ],
            ),
            fx.Source(
                'HeatSrc',
                outputs=[
                    fx.Flow('heat', bus='Heat', effects_per_flow_hour={'costs': 2, 'CO2': 0.5}),
                ],
            ),
        )
        solve(fs)
        # costs = (10+20)*2 = 60, CO2 = (10+20)*0.5 = 15
        assert_allclose(fs.solution['costs'].item(), 60.0, rtol=1e-5)
        assert_allclose(fs.solution['CO2'].item(), 15.0, rtol=1e-5)

    def test_share_from_temporal(self):
        """Proves: share_from_temporal correctly adds a weighted fraction of one effect's
        temporal sum into another effect's total.

        costs has share_from_temporal={'CO2': 0.5}. Direct costs=20, CO2=200.
        Shared portion: 0.5 × 200 = 100. Total costs = 20 + 100 = 120.

        Sensitivity: Without the share mechanism, costs=20 (6× less). The 120
        value is impossible without share_from_temporal working.
        """
        fs = make_flow_system(2)
        co2 = fx.Effect('CO2', 'kg')
        costs = fx.Effect('costs', '€', is_standard=True, is_objective=True, share_from_temporal={'CO2': 0.5})
        fs.add_elements(
            fx.Bus('Heat'),
            costs,
            co2,
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([10, 10])),
                ],
            ),
            fx.Source(
                'HeatSrc',
                outputs=[
                    fx.Flow('heat', bus='Heat', effects_per_flow_hour={'costs': 1, 'CO2': 10}),
                ],
            ),
        )
        solve(fs)
        # direct costs = 20*1 = 20, CO2 = 20*10 = 200
        # costs += 0.5 * CO2_temporal = 0.5 * 200 = 100
        # total costs = 20 + 100 = 120
        assert_allclose(fs.solution['costs'].item(), 120.0, rtol=1e-5)
        assert_allclose(fs.solution['CO2'].item(), 200.0, rtol=1e-5)

    def test_effect_maximum_total(self):
        """Proves: maximum_total on an effect constrains the optimizer to respect an
        upper bound on cumulative effect, forcing suboptimal dispatch.

        CO2 capped at 15kg. Dirty source: 1€+1kgCO2/kWh. Clean source: 10€+0kgCO2/kWh.
        Demand=20. Optimizer must split: 15 from Dirty + 5 from Clean.

        Sensitivity: Without the CO2 cap, all 20 from Dirty → cost=20 instead of 65.
        The 3.25× cost increase proves the constraint is binding.
        """
        fs = make_flow_system(2)
        co2 = fx.Effect('CO2', 'kg', maximum_total=15)
        costs = fx.Effect('costs', '€', is_standard=True, is_objective=True)
        fs.add_elements(
            fx.Bus('Heat'),
            costs,
            co2,
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([10, 10])),
                ],
            ),
            fx.Source(
                'Dirty',
                outputs=[
                    fx.Flow('heat', bus='Heat', effects_per_flow_hour={'costs': 1, 'CO2': 1}),
                ],
            ),
            fx.Source(
                'Clean',
                outputs=[
                    fx.Flow('heat', bus='Heat', effects_per_flow_hour={'costs': 10, 'CO2': 0}),
                ],
            ),
        )
        solve(fs)
        # Without CO2 limit: all from Dirty = 20€
        # With CO2 max=15: 15 from Dirty (15€), 5 from Clean (50€) → total 65€
        assert_allclose(fs.solution['costs'].item(), 65.0, rtol=1e-5)
        assert_allclose(fs.solution['CO2'].item(), 15.0, rtol=1e-5)
