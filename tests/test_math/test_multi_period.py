"""Mathematical correctness tests for multi-period optimization.

Tests verify that period weights, over-period constraints, and linked
investments work correctly across multiple planning periods.
"""

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose

import flixopt as fx

from .conftest import make_multi_period_flow_system


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
        fs.add(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[
                    fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                ],
            ),
            fx.Port(
                'Grid',
                imports=[fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=1)],
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
        fs.add(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[
                    fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                ],
            ),
            fx.Port(
                'DirtySource',
                imports=[
                    fx.Flow(
                        bus='Elec',
                        flow_id='elec',
                        effects_per_flow_hour=1,
                        flow_hours_max_over_periods=50,
                    ),
                ],
            ),
            fx.Port(
                'CleanSource',
                imports=[fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=10)],
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
        fs.add(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[
                    fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                ],
            ),
            fx.Port(
                'CheapSource',
                imports=[fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=1)],
            ),
            fx.Port(
                'ExpensiveSource',
                imports=[
                    fx.Flow(
                        bus='Elec',
                        flow_id='elec',
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
        fs.add(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            co2,
            fx.Port(
                'Demand',
                exports=[
                    fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                ],
            ),
            fx.Port(
                'DirtySource',
                imports=[
                    fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour={'costs': 1, 'CO2': 1}),
                ],
            ),
            fx.Port(
                'CleanSource',
                imports=[fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=10)],
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
        fs.add(
            fx.Bus('Elec', imbalance_penalty_per_flow_hour=0),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            co2,
            fx.Port(
                'Demand',
                exports=[
                    fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([2, 2, 2])),
                ],
            ),
            fx.Port(
                'DirtySource',
                imports=[
                    fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour={'costs': 1, 'CO2': 1}),
                ],
            ),
            fx.Port(
                'CheapSource',
                imports=[fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=1)],
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
        fs.add(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[
                    fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                ],
            ),
            fx.Port(
                'Grid',
                imports=[
                    fx.Flow(
                        bus='Elec',
                        flow_id='elec',
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
        fs.add(
            fx.Bus('Elec'),
            fx.Effect(
                'costs',
                '€',
                is_standard=True,
                is_objective=True,
                period_weights=xr.DataArray([1, 10], dims='period', coords={'period': [2020, 2025]}),
            ),
            fx.Port(
                'Demand',
                exports=[
                    fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                ],
            ),
            fx.Port(
                'Grid',
                imports=[fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=1)],
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
        fs.add(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[
                    fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([0, 0, 80])),
                ],
            ),
            fx.Port(
                'Grid',
                imports=[
                    fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=np.array([1, 1, 100])),
                ],
            ),
            fx.Storage(
                'Battery',
                charging=fx.Flow(bus='Elec', size=200),
                discharging=fx.Flow(bus='Elec', size=200),
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
        fs.add(
            fx.Bus('Elec', imbalance_penalty_per_flow_hour=5),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[
                    fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([50, 0, 0])),
                ],
            ),
            fx.Port(
                'Grid',
                imports=[
                    fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=np.array([100, 1, 1])),
                ],
            ),
            fx.Storage(
                'Battery',
                charging=fx.Flow(bus='Elec', size=200),
                discharging=fx.Flow(bus='Elec', size=200),
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
