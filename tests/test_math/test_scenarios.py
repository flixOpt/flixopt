"""Mathematical correctness tests for scenario optimization.

Tests verify that scenario weights, scenario-independent sizes, and
scenario-independent flow rates work correctly.
"""

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose

import flixopt as fx

from .conftest import make_scenario_flow_system


def _scenario_demand(fs, low_values, high_values):
    """Create a scenario-dependent demand profile aligned with FlowSystem timesteps."""
    return xr.DataArray(
        [low_values, high_values],
        dims=['scenario', 'time'],
        coords={'scenario': ['low', 'high'], 'time': fs.timesteps},
    )


class TestScenarios:
    def test_scenario_weights_affect_objective(self, optimize):
        """Proves: scenario weights correctly weight per-scenario costs.

        2 ts, scenarios=['low', 'high'], weights=[0.3, 0.7] (normalized).
        Demand: low=[10, 10], high=[30, 30]. Grid @1€.
        Per-scenario costs: low=20, high=60.
        Objective = 0.3*20 + 0.7*60 = 48.

        Sensitivity: With equal weights [0.5, 0.5], objective=40.
        """
        fs = make_scenario_flow_system(
            n_timesteps=2,
            scenarios=['low', 'high'],
            scenario_weights=[0.3, 0.7],
        )
        demand = _scenario_demand(fs, [10, 10], [30, 30])
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=demand)],
            ),
            fx.Source(
                'Grid',
                outputs=[fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=1)],
            ),
        )
        fs = optimize(fs)
        # low: 20, high: 60. Weighted: 0.3*20 + 0.7*60 = 48.
        assert_allclose(fs.solution['objective'].item(), 48.0, rtol=1e-5)

    def test_scenario_independent_sizes(self, optimize):
        """Proves: scenario_independent_sizes=True forces the same invested size
        across all scenarios.

        2 ts, scenarios=['low', 'high'], weights=[0.5, 0.5].
        Demand: low=[10, 10], high=[30, 30]. Grid with InvestParameters.
        With independent sizes (default): size must be the same across scenarios.

        The invested size must be the same across both scenarios.
        """
        fs = make_scenario_flow_system(
            n_timesteps=2,
            scenarios=['low', 'high'],
            scenario_weights=[0.5, 0.5],
        )
        demand = _scenario_demand(fs, [10, 10], [30, 30])
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=demand)],
            ),
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow(
                        bus='Elec',
                        flow_id='elec',
                        size=fx.InvestParameters(maximum_size=100, effects_of_investment_per_size=1),
                        effects_per_flow_hour=1,
                    ),
                ],
            ),
        )
        fs = optimize(fs)
        # With scenario_independent_sizes=True (default), size is the same
        size = fs.solution['Grid(elec)|size']
        if 'scenario' in size.dims:
            size_low = size.sel(scenario='low').item()
            size_high = size.sel(scenario='high').item()
            assert_allclose(size_low, size_high, rtol=1e-5)

    def test_scenario_independent_flow_rates(self, optimize):
        """Proves: scenario_independent_flow_rates forces identical flow rates
        across scenarios for specified flows, even when demands differ.

        2 ts, scenarios=['low', 'high'], weights=[0.5, 0.5].
        scenario_independent_flow_rates=['Grid(elec)'] (only Grid, not Demand).
        Demand: low=[10, 10], high=[30, 30]. Grid @1€.
        Grid rate must match across scenarios → rate=30 (max of demands).
        Low scenario excess absorbed by Dump sink (free).

        Sensitivity: Without constraint, rates vary → objective = 0.5*20 + 0.5*60 = 40.
        With constraint, Grid=30 in both → objective = 0.5*60 + 0.5*60 = 60.
        """
        fs = make_scenario_flow_system(
            n_timesteps=2,
            scenarios=['low', 'high'],
            scenario_weights=[0.5, 0.5],
        )
        fs.scenario_independent_flow_rates = ['Grid(elec)']
        demand = _scenario_demand(fs, [10, 10], [30, 30])
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=demand)],
            ),
            fx.Sink(
                'Dump',
                inputs=[fx.Flow(bus='Elec', flow_id='elec')],
            ),
            fx.Source(
                'Grid',
                outputs=[fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=1)],
            ),
        )
        fs = optimize(fs)
        # With independent flow rates on Grid, must produce 30 in both scenarios.
        # Objective = 0.5*60 + 0.5*60 = 60.
        assert_allclose(fs.solution['objective'].item(), 60.0, rtol=1e-5)

    def test_storage_relative_minimum_final_charge_state_scalar(self, optimize):
        """Proves: scalar relative_minimum_final_charge_state works with scenarios.

        Regression test for the scalar branch fix in _relative_charge_state_bounds.
        Uses 3 timesteps (not 2) to avoid ambiguity with 2 scenarios.

        3 ts, scenarios=['low', 'high'], weights=[0.5, 0.5].
        Storage: capacity=100, initial=50, relative_minimum_final_charge_state=0.5.
        Grid @[1, 1, 100], Demand=[0, 0, 80] (same in both scenarios).
        Per-scenario: charge 50 @t0+t1 (cost=50), discharge 50 @t2, grid 30 @100=3000.
        Per-scenario cost=3050. Objective = 0.5*3050 + 0.5*3050 = 3050.
        """
        fs = make_scenario_flow_system(
            n_timesteps=3,
            scenarios=['low', 'high'],
            scenario_weights=[0.5, 0.5],
        )
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([0, 0, 80])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[
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
        assert_allclose(fs.solution['objective'].item(), 3050.0, rtol=1e-5)

    def test_storage_relative_maximum_final_charge_state_scalar(self, optimize):
        """Proves: scalar relative_maximum_final_charge_state works with scenarios.

        Regression test for the scalar branch fix in _relative_charge_state_bounds.
        Uses 3 timesteps (not 2) to avoid ambiguity with 2 scenarios.

        3 ts, scenarios=['low', 'high'], weights=[0.5, 0.5].
        Storage: capacity=100, initial=80, relative_maximum_final_charge_state=0.2.
        Demand=[50, 0, 0], Grid @[100, 1, 1], imbalance_penalty=5.
        Per-scenario: discharge 50 for demand @t0, discharge 10 excess @t1 (penalty=50).
        Objective = 0.5*50 + 0.5*50 = 50.
        """
        fs = make_scenario_flow_system(
            n_timesteps=3,
            scenarios=['low', 'high'],
            scenario_weights=[0.5, 0.5],
        )
        fs.add_elements(
            fx.Bus('Elec', imbalance_penalty_per_flow_hour=5),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([50, 0, 0])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[
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
        assert_allclose(fs.solution['objective'].item(), 50.0, rtol=1e-5)
