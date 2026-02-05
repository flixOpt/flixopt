"""Mathematical correctness tests for scenario optimization.

Tests verify that scenario weights, scenario-independent sizes, and
scenario-independent flow rates work correctly.
"""

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
                inputs=[fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=demand)],
            ),
            fx.Source(
                'Grid',
                outputs=[fx.Flow('elec', bus='Elec', effects_per_flow_hour=1)],
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
                inputs=[fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=demand)],
            ),
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow(
                        'elec',
                        bus='Elec',
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
                inputs=[fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=demand)],
            ),
            fx.Sink(
                'Dump',
                inputs=[fx.Flow('elec', bus='Elec')],
            ),
            fx.Source(
                'Grid',
                outputs=[fx.Flow('elec', bus='Elec', effects_per_flow_hour=1)],
            ),
        )
        fs = optimize(fs)
        # With independent flow rates on Grid, must produce 30 in both scenarios.
        # Objective = 0.5*60 + 0.5*60 = 60.
        assert_allclose(fs.solution['objective'].item(), 60.0, rtol=1e-5)
