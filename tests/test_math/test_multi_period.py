"""Mathematical correctness tests for multi-period optimization.

Tests verify that period weights, over-period constraints, and linked
investments work correctly across multiple planning periods.
"""

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose

import flixopt as fx

from .conftest import _SOLVER, make_flow_system, make_multi_period_flow_system


class TestMultiPeriod:
    def test_period_weights_affect_objective(self, optimize):
        """Proves: period weights scale per-period costs in the objective.

        3 ts, periods=[2020, 2025], weight_of_last_period=5.
        Weights = [5, 5] (2025-2020=5, last=5).
        Grid @1€, Demand=[10, 10, 10]. Per-period cost=30. Objective = 5*30 + 5*30 = 300.

        Sensitivity: If weights were [1, 1], objective=60.
        With weights [5, 5], objective=300.
        """
        fs = make_multi_period_flow_system(n_timesteps=3, periods=[2020, 2025], weight_of_last_period=5)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[fx.Flow('elec', bus='Elec', effects_per_flow_hour=1)],
            ),
        )
        fs = optimize(fs)
        # Per-period cost = 30. Weights = [5, 5]. Objective = 300.
        assert_allclose(fs.solution['objective'].item(), 300.0, rtol=1e-5)

    def test_flow_hours_max_over_periods(self, optimize):
        """Proves: flow_hours_max_over_periods caps the weighted total flow-hours
        across all periods.

        3 ts, periods=[2020, 2025], weight_of_last_period=5. Weights=[5, 5].
        DirtySource @1€ with flow_hours_max_over_periods=50.
        CleanSource @10€. Demand=[10, 10, 10] per period.
        Without constraint, all dirty → objective=300. With cap, forced to use clean.

        Sensitivity: Without constraint, objective=300.
        With constraint, objective > 300.
        """
        fs = make_multi_period_flow_system(n_timesteps=3, periods=[2020, 2025], weight_of_last_period=5)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                ],
            ),
            fx.Source(
                'DirtySource',
                outputs=[
                    fx.Flow(
                        'elec',
                        bus='Elec',
                        effects_per_flow_hour=1,
                        flow_hours_max_over_periods=50,
                    ),
                ],
            ),
            fx.Source(
                'CleanSource',
                outputs=[fx.Flow('elec', bus='Elec', effects_per_flow_hour=10)],
            ),
        )
        fs = optimize(fs)
        # Constrained: weighted dirty flow_hours <= 50. Objective > 300.
        assert fs.solution['objective'].item() > 300.0 + 1e-5

    def test_flow_hours_min_over_periods(self, optimize):
        """Proves: flow_hours_min_over_periods forces a minimum weighted total
        of flow-hours across all periods.

        3 ts, periods=[2020, 2025], weight_of_last_period=5. Weights=[5, 5].
        ExpensiveSource @10€ with flow_hours_min_over_periods=100.
        CheapSource @1€. Demand=[10, 10, 10] per period.
        Forces min production from expensive source.

        Sensitivity: Without constraint, all cheap → objective=300.
        With constraint, must use expensive → objective > 300.
        """
        fs = make_multi_period_flow_system(n_timesteps=3, periods=[2020, 2025], weight_of_last_period=5)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                ],
            ),
            fx.Source(
                'CheapSource',
                outputs=[fx.Flow('elec', bus='Elec', effects_per_flow_hour=1)],
            ),
            fx.Source(
                'ExpensiveSource',
                outputs=[
                    fx.Flow(
                        'elec',
                        bus='Elec',
                        effects_per_flow_hour=10,
                        flow_hours_min_over_periods=100,
                    ),
                ],
            ),
        )
        fs = optimize(fs)
        # Forced to use expensive source. Objective > 300.
        assert fs.solution['objective'].item() > 300.0 + 1e-5

    def test_effect_maximum_over_periods(self, optimize):
        """Proves: Effect.maximum_over_periods caps the weighted total of an effect
        across all periods.

        CO2 effect with maximum_over_periods=50. DirtySource emits CO2=1 per kWh.
        3 ts, 2 periods. Caps total dirty across periods.

        Sensitivity: Without CO2 cap, all dirty → objective=300.
        With cap, forced to use clean → objective > 300.
        """
        fs = make_multi_period_flow_system(n_timesteps=3, periods=[2020, 2025], weight_of_last_period=5)
        co2 = fx.Effect('CO2', 'kg', maximum_over_periods=50)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            co2,
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                ],
            ),
            fx.Source(
                'DirtySource',
                outputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour={'costs': 1, 'CO2': 1}),
                ],
            ),
            fx.Source(
                'CleanSource',
                outputs=[fx.Flow('elec', bus='Elec', effects_per_flow_hour=10)],
            ),
        )
        fs = optimize(fs)
        # CO2 cap forces use of clean source. Objective > 300.
        assert fs.solution['objective'].item() > 300.0 + 1e-5

    def test_effect_minimum_over_periods(self, optimize):
        """Proves: Effect.minimum_over_periods forces a minimum weighted total of
        an effect across all periods.

        CO2 effect with minimum_over_periods=100. DirtySource emits CO2=1/kWh @1€.
        CheapSource @1€ no CO2. 3 ts. Bus has imbalance_penalty=0.
        Must produce enough dirty to meet min CO2 across periods.

        Sensitivity: Without constraint, cheapest split → objective=60.
        With min CO2=100, must overproduce dirty → objective > 60.
        """
        fs = make_multi_period_flow_system(n_timesteps=3, periods=[2020, 2025], weight_of_last_period=5)
        co2 = fx.Effect('CO2', 'kg', minimum_over_periods=100)
        fs.add_elements(
            fx.Bus('Elec', imbalance_penalty_per_flow_hour=0),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            co2,
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([2, 2, 2])),
                ],
            ),
            fx.Source(
                'DirtySource',
                outputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour={'costs': 1, 'CO2': 1}),
                ],
            ),
            fx.Source(
                'CheapSource',
                outputs=[fx.Flow('elec', bus='Elec', effects_per_flow_hour=1)],
            ),
        )
        fs = optimize(fs)
        # Must overproduce to meet min CO2. Objective > 60.
        assert fs.solution['objective'].item() > 60.0 + 1e-5

    def test_invest_linked_periods(self, optimize):
        """Proves: InvestParameters.linked_periods forces equal investment sizes
        across linked periods.

        periods=[2020, 2025], weight_of_last_period=5.
        Source with invest, linked_periods=(2020, 2025) → sizes must match.

        Structural check: invested sizes are equal across linked periods.
        """
        fs = make_multi_period_flow_system(
            n_timesteps=3,
            periods=[2020, 2025],
            weight_of_last_period=5,
        )
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow(
                        'elec',
                        bus='Elec',
                        size=fx.InvestParameters(
                            maximum_size=100,
                            effects_of_investment_per_size=1,
                            linked_periods=(2020, 2025),
                        ),
                        effects_per_flow_hour=1,
                    ),
                ],
            ),
        )
        fs = optimize(fs)
        # Verify sizes are equal for linked periods 2020 and 2025
        size = fs.solution['Grid(elec)|size']
        if 'period' in size.dims:
            size_2020 = size.sel(period=2020).item()
            size_2025 = size.sel(period=2025).item()
            assert_allclose(size_2020, size_2025, rtol=1e-5)

    def test_effect_period_weights(self, optimize):
        """Proves: Effect.period_weights overrides default period weights.

        periods=[2020, 2025], weight_of_last_period=5. Default weights=[5, 5].
        Effect 'costs' with period_weights=[1, 10].
        Grid @1€, Demand=[10, 10, 10]. Per-period cost=30.
        Objective = 1*30 + 10*30 = 330 (default weights would give 300).

        Sensitivity: With default weights [5, 5], objective=300.
        With custom [1, 10], objective=330.
        """
        fs = make_multi_period_flow_system(n_timesteps=3, periods=[2020, 2025], weight_of_last_period=5)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect(
                'costs',
                '€',
                is_standard=True,
                is_objective=True,
                period_weights=xr.DataArray([1, 10], dims='period', coords={'period': [2020, 2025]}),
            ),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[fx.Flow('elec', bus='Elec', effects_per_flow_hour=1)],
            ),
        )
        fs = optimize(fs)
        # Custom period_weights=[1, 10]. Per-period cost=30.
        # Objective = 1*30 + 10*30 = 330.
        assert_allclose(fs.solution['objective'].item(), 330.0, rtol=1e-5)

    def test_storage_relative_minimum_final_charge_state_scalar(self, optimize):
        """Proves: scalar relative_minimum_final_charge_state works in multi-period.

        Regression test for the scalar branch fix in _relative_charge_state_bounds.
        Uses 3 timesteps (not 2) to avoid ambiguity with 2 periods.

        3 ts, periods=[2020, 2025], weight_of_last_period=5. Weights=[5, 5].
        Storage: capacity=100, initial=50, relative_minimum_final_charge_state=0.5.
        Grid @[1, 1, 100], Demand=[0, 0, 80].
        Per-period: charge 50 @t0+t1 (cost=50), discharge 50 @t2, grid 30 @100=3000.
        Per-period cost=3050. Objective = 5*3050 + 5*3050 = 30500.
        """
        fs = make_multi_period_flow_system(n_timesteps=3, periods=[2020, 2025], weight_of_last_period=5)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([0, 0, 80])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour=np.array([1, 1, 100])),
                ],
            ),
            fx.Storage(
                'Battery',
                charging=fx.Flow('charge', bus='Elec', size=200),
                discharging=fx.Flow('discharge', bus='Elec', size=200),
                capacity_in_flow_hours=100,
                initial_charge_state=50,
                relative_minimum_final_charge_state=0.5,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['objective'].item(), 30500.0, rtol=1e-5)

    def test_storage_relative_maximum_final_charge_state_scalar(self, optimize):
        """Proves: scalar relative_maximum_final_charge_state works in multi-period.

        Regression test for the scalar branch fix in _relative_charge_state_bounds.
        Uses 3 timesteps (not 2) to avoid ambiguity with 2 periods.

        3 ts, periods=[2020, 2025], weight_of_last_period=5. Weights=[5, 5].
        Storage: capacity=100, initial=80, relative_maximum_final_charge_state=0.2.
        Demand=[50, 0, 0], Grid @[100, 1, 1], imbalance_penalty=5.
        Per-period: discharge 50 for demand @t0 (SOC=30), discharge 10 excess @t1
        (penalty=50, SOC=20). Objective per period=50.
        Total objective = 5*50 + 5*50 = 500.
        """
        fs = make_multi_period_flow_system(n_timesteps=3, periods=[2020, 2025], weight_of_last_period=5)
        fs.add_elements(
            fx.Bus('Elec', imbalance_penalty_per_flow_hour=5),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([50, 0, 0])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour=np.array([100, 1, 1])),
                ],
            ),
            fx.Storage(
                'Battery',
                charging=fx.Flow('charge', bus='Elec', size=200),
                discharging=fx.Flow('discharge', bus='Elec', size=200),
                capacity_in_flow_hours=100,
                initial_charge_state=80,
                relative_maximum_final_charge_state=0.2,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['objective'].item(), 500.0, rtol=1e-5)

    def test_fix_sizes_preserves_per_period_sizes(self, optimize):
        """Proves: transform.fix_sizes() preserves per-period investment sizes
        in multi-period models (two-stage sizing -> dispatch workflow).

        3 ts, periods=[2020, 2025], weight_of_last_period=5. Weights=[5, 5].
        Demand peaks at 50 (2020) and 80 (2025), so optimal sizes differ per period.
        Boiler invest: 10 fixed + 1 per size. Fuel @1.
        Per-period costs: 2020: (10+50) + 80 = 140; 2025: (10+80) + 110 = 200.
        Objective = 5*140 + 5*200 = 1700.

        Stage 2 (fixed sizes) must reproduce the same sizes and objective.

        Sensitivity: Before the fix, fix_sizes() collapsed sizes via .item(),
        raising 'ValueError: can only convert an array of size 1 to a Python
        scalar' on any multi-period model. If per-period sizes were collapsed
        to a single value instead, stage-2 sizes or objective would differ.
        """
        fs = make_multi_period_flow_system(n_timesteps=3, periods=[2020, 2025], weight_of_last_period=5)
        demand = xr.DataArray(
            np.array([[10, 50, 20], [10, 80, 20]], dtype=float),
            coords={'period': [2020, 2025], 'time': fs.timesteps},
            dims=['period', 'time'],
        )
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=demand),
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
                    size=fx.InvestParameters(
                        maximum_size=200,
                        effects_of_investment=10,
                        effects_of_investment_per_size=1,
                    ),
                ),
            ),
        )
        # Stage 1: sizing
        fs = optimize(fs)
        assert_allclose(fs.solution['Boiler(heat)|size'].values, [50.0, 80.0], rtol=1e-5)
        assert_allclose(fs.solution['objective'].item(), 1700.0, rtol=1e-5)

        # Stage 2: fix sizes and dispatch
        fs_dispatch = fs.transform.fix_sizes()
        fs_dispatch.optimize(_SOLVER)
        assert_allclose(fs_dispatch.solution['Boiler(heat)|size'].values, [50.0, 80.0], rtol=1e-5)
        assert_allclose(fs_dispatch.solution['objective'].item(), 1700.0, rtol=1e-5)

    def test_fix_sizes_no_invest_reproduces_objective(self, optimize):
        """Proves: transform.fix_sizes() does not charge investment for a size of 0.

        Single period, 3 ts. Demand=[10, 50, 20] (sum 80). A DirectHeat source @2€
        competes with a Boiler whose investment costs a prohibitive 100000€ fixed.
        Optimal is to NOT invest and serve demand directly: objective = 80*2 = 160.

        Stage 2 must reproduce size 0 AND objective 160.

        Sensitivity: fix_sizes() used to force mandatory=True unconditionally. With
        a fixed size of 0 that still charges the flat effects_of_investment (100000),
        so the dispatch objective jumped to 100160 instead of 160.
        """
        fs = make_flow_system(n_timesteps=3)
        demand = xr.DataArray(np.array([10, 50, 20], dtype=float), coords={'time': fs.timesteps}, dims=['time'])
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink('Demand', inputs=[fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=demand)]),
            fx.Source('DirectHeat', outputs=[fx.Flow('h', bus='Heat', effects_per_flow_hour=2)]),
            fx.Source('GasSrc', outputs=[fx.Flow('gas', bus='Gas', effects_per_flow_hour=1)]),
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=fx.InvestParameters(
                        maximum_size=200,
                        effects_of_investment=100000,
                        effects_of_investment_per_size=1,
                    ),
                ),
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['Boiler(heat)|size'].item(), 0.0, atol=1e-6)
        assert_allclose(fs.solution['objective'].item(), 160.0, rtol=1e-5)

        fs_dispatch = fs.transform.fix_sizes()
        fs_dispatch.optimize(_SOLVER)
        assert_allclose(fs_dispatch.solution['Boiler(heat)|size'].item(), 0.0, atol=1e-6)
        assert_allclose(fs_dispatch.solution['objective'].item(), 160.0, rtol=1e-5)

    def test_fix_sizes_mixed_period_invest_reproduces_objective(self, optimize):
        """Proves: transform.fix_sizes() charges investment per period, not globally.

        periods=[2020, 2021], weight_of_last_period=1 -> weights [1, 1]. 3 ts each.
        Demand is 0 in 2020 and [10, 90, 10] in 2021. The Boiler (10000€ fixed
        invest + 1 per size) is only built in 2021 (size 90); 2020 stays at size 0.
        Per-period cost: 2020: 0; 2021: 10000 + 90 + 110 = 10200. Objective = 10200.

        Stage 2 must reproduce sizes [0, 90] AND objective 10200.

        Sensitivity: with the old unconditional mandatory=True, the 2020 period
        (size 0) was still charged the 10000€ fixed investment, inflating the
        objective to 20200. A scalar mandatory flag cannot express "invest in 2021
        but not 2020"; keeping it optional lets the invested binary gate the cost.
        """
        fs = make_multi_period_flow_system(n_timesteps=3, periods=[2020, 2021], weight_of_last_period=1)
        demand = xr.DataArray(
            np.array([[0, 0, 0], [10, 90, 10]], dtype=float),
            coords={'period': [2020, 2021], 'time': fs.timesteps},
            dims=['period', 'time'],
        )
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink('Demand', inputs=[fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=demand)]),
            fx.Source('GasSrc', outputs=[fx.Flow('gas', bus='Gas', effects_per_flow_hour=1)]),
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=fx.InvestParameters(
                        maximum_size=200,
                        effects_of_investment=10000,
                        effects_of_investment_per_size=1,
                    ),
                ),
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['Boiler(heat)|size'].values, [0.0, 90.0], atol=1e-6)
        assert_allclose(fs.solution['objective'].item(), 10200.0, rtol=1e-5)

        fs_dispatch = fs.transform.fix_sizes()
        fs_dispatch.optimize(_SOLVER)
        assert_allclose(fs_dispatch.solution['Boiler(heat)|size'].values, [0.0, 90.0], atol=1e-6)
        assert_allclose(fs_dispatch.solution['objective'].item(), 10200.0, rtol=1e-5)
