"""Mathematical correctness tests for Flow investment decisions.

Tests for InvestParameters applied to Flows, including sizing optimization,
optional investments, minimum/fixed sizes, and piecewise investment costs.
"""

import numpy as np
from numpy.testing import assert_allclose

import flixopt as fx

from .conftest import make_flow_system


class TestFlowInvest:
    def test_invest_size_optimized(self, optimize):
        """Proves: InvestParameters correctly sizes the unit to match peak demand
        when there is a per-size investment cost.

        Sensitivity: If sizing were broken (e.g. forced to max=200), invest cost
        would be 10+200=210, total=290 instead of 140. If sized to 0, infeasible.
        Only size=50 (peak demand) minimizes the sum of invest + fuel cost.
        """
        fs = make_flow_system(3)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([10, 50, 20])),
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
        fs = optimize(fs)
        # size = 50 (peak), invest cost = 10 + 50*1 = 60, fuel = 80
        # total = 140
        assert_allclose(fs.solution['Boiler(heat)|size'].item(), 50.0, rtol=1e-5)
        assert_allclose(fs.solution['costs'].item(), 140.0, rtol=1e-5)

    def test_invest_optional_not_built(self, optimize):
        """Proves: Optional investment is correctly skipped when the fixed investment
        cost outweighs operational savings.

        InvestBoiler has eta=1.0 (efficient) but 99999€ fixed invest cost.
        CheapBoiler has eta=0.5 (inefficient) but no invest cost.

        Sensitivity: If investment cost were ignored (free invest), InvestBoiler
        would be built and used → fuel=20 instead of 40. The cost difference (40
        vs 20) proves the investment mechanism is working.
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
                'InvestBoiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=fx.InvestParameters(
                        maximum_size=100,
                        effects_of_investment=99999,
                    ),
                ),
            ),
            fx.linear_converters.Boiler(
                'CheapBoiler',
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat', size=100),
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['InvestBoiler(heat)|invested'].item(), 0.0, atol=1e-5)
        # All demand served by CheapBoiler: fuel = 20/0.5 = 40
        # If invest were free, InvestBoiler would run: fuel = 20/1.0 = 20 (different!)
        assert_allclose(fs.solution['costs'].item(), 40.0, rtol=1e-5)

    def test_invest_minimum_size(self, optimize):
        """Proves: InvestParameters.minimum_size forces the invested capacity to be
        at least the specified value, even when demand is much smaller.

        Demand peak=10, minimum_size=100, cost_per_size=1 → must invest 100.

        Sensitivity: Without minimum_size, optimal invest=10 → cost=10+20=30.
        With minimum_size=100, invest cost=100 → cost=120. The 4× cost difference
        proves the constraint is active.
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
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=fx.InvestParameters(
                        minimum_size=100,
                        maximum_size=200,
                        mandatory=True,
                        effects_of_investment_per_size=1,
                    ),
                ),
            ),
        )
        fs = optimize(fs)
        # Must invest at least 100, cost_per_size=1 → invest=100
        assert_allclose(fs.solution['Boiler(heat)|size'].item(), 100.0, rtol=1e-5)
        # fuel=20, invest=100 → total=120
        assert_allclose(fs.solution['costs'].item(), 120.0, rtol=1e-5)

    def test_invest_fixed_size(self, optimize):
        """Proves: fixed_size creates a binary invest-or-not decision at exactly the
        specified capacity — no continuous sizing.

        FixedBoiler: fixed_size=80, invest_cost=10€, eta=1.0.
        Backup: eta=0.5, no invest. Demand=[30,30], gas=1€/kWh.

        Sensitivity: Without fixed_size (free continuous sizing), optimal size=30,
        invest=10, fuel=60, total=70. With fixed_size=80, invest=10, fuel=60,
        total=70 (same invest cost but size=80 not 30). The key assertion is that
        invested size is exactly 80, not 30.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([30, 30])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.linear_converters.Boiler(
                'FixedBoiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=fx.InvestParameters(
                        fixed_size=80,
                        effects_of_investment=10,
                    ),
                ),
            ),
            fx.linear_converters.Boiler(
                'Backup',
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat', size=100),
            ),
        )
        fs = optimize(fs)
        # FixedBoiler invested (10€ < savings from eta=1.0 vs 0.5)
        # size must be exactly 80 (not optimized to 30)
        assert_allclose(fs.solution['FixedBoiler(heat)|size'].item(), 80.0, rtol=1e-5)
        assert_allclose(fs.solution['FixedBoiler(heat)|invested'].item(), 1.0, atol=1e-5)
        # fuel=60 (all from FixedBoiler @eta=1), invest=10, total=70
        assert_allclose(fs.solution['costs'].item(), 70.0, rtol=1e-5)

    def test_piecewise_invest_cost(self, optimize):
        """Proves: piecewise_effects_of_investment applies non-linear investment costs
        where the cost-per-size changes across size segments (economies of scale).

        Segment 1: size 0→50, cost 0→100 (2€/kW).
        Segment 2: size 50→200, cost 100→250 (1€/kW, cheaper per unit).
        Demand peak=80. Optimal size=80, in segment 2.
        Invest cost = 100 + (80-50)×(250-100)/(200-50) = 100 + 30 = 130.

        Sensitivity: If linear cost at 2€/kW throughout, invest=160 → total=240.
        With piecewise (economies of scale), invest=130 → total=210.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([80, 80])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=0.5),
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
                        piecewise_effects_of_investment=fx.PiecewiseEffects(
                            piecewise_origin=fx.Piecewise([fx.Piece(0, 50), fx.Piece(50, 200)]),
                            piecewise_shares={
                                'costs': fx.Piecewise([fx.Piece(0, 100), fx.Piece(100, 250)]),
                            },
                        ),
                    ),
                ),
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['Boiler(heat)|size'].item(), 80.0, rtol=1e-5)
        # invest = 100 + 30/150*150 = 100 + 30 = 130. fuel = 160*0.5 = 80. total = 210.
        assert_allclose(fs.solution['costs'].item(), 210.0, rtol=1e-5)

    def test_invest_mandatory_forces_investment(self, optimize):
        """Proves: mandatory=True forces investment even when it's not economical.

        ExpensiveBoiler: mandatory=True, fixed invest=1000€, per_size=1€/kW, eta=1.0.
        CheapBoiler: no invest, eta=0.5. Demand=[10,10].

        Without mandatory, CheapBoiler covers all: fuel=40, total=40.
        With mandatory=True, ExpensiveBoiler must be built: invest=1000+10, fuel=20, total=1030.

        Sensitivity: If mandatory were ignored, optimizer would skip the expensive
        investment → cost=40 instead of 1030. The 25× cost difference proves
        mandatory is enforced.
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
                'ExpensiveBoiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=fx.InvestParameters(
                        minimum_size=10,
                        maximum_size=100,
                        mandatory=True,
                        effects_of_investment=1000,
                        effects_of_investment_per_size=1,
                    ),
                ),
            ),
            fx.linear_converters.Boiler(
                'CheapBoiler',
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat', size=100),
            ),
        )
        fs = optimize(fs)
        # mandatory=True forces ExpensiveBoiler to be built, size=10 (minimum needed)
        # Note: with mandatory=True, there's no 'invested' binary - it's always invested
        assert_allclose(fs.solution['ExpensiveBoiler(heat)|size'].item(), 10.0, rtol=1e-5)
        # invest=1000+10*1=1010, fuel from ExpensiveBoiler=20 (eta=1.0), total=1030
        assert_allclose(fs.solution['costs'].item(), 1030.0, rtol=1e-5)

    def test_invest_not_mandatory_skips_when_uneconomical(self, optimize):
        """Proves: mandatory=False (default) allows optimizer to skip investment
        when it's not economical.

        ExpensiveBoiler: mandatory=False, invest_cost=1000€, eta=1.0.
        CheapBoiler: no invest, eta=0.5. Demand=[10,10].

        With mandatory=False, optimizer skips expensive investment.
        CheapBoiler covers all: fuel=40, total=40.

        Sensitivity: This is the complement to test_invest_mandatory_forces_investment.
        cost=40 here vs cost=1020 with mandatory=True proves the flag works.
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
                'ExpensiveBoiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=fx.InvestParameters(
                        minimum_size=10,
                        maximum_size=100,
                        mandatory=False,
                        effects_of_investment=1000,
                    ),
                ),
            ),
            fx.linear_converters.Boiler(
                'CheapBoiler',
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat', size=100),
            ),
        )
        fs = optimize(fs)
        # mandatory=False allows skipping uneconomical investment
        assert_allclose(fs.solution['ExpensiveBoiler(heat)|invested'].item(), 0.0, atol=1e-5)
        # CheapBoiler covers all: fuel = 20/0.5 = 40
        assert_allclose(fs.solution['costs'].item(), 40.0, rtol=1e-5)

    def test_invest_effects_of_retirement(self, optimize):
        """Proves: effects_of_retirement adds a cost when NOT investing.

        Boiler with effects_of_retirement=500€. If not built, incur 500€ penalty.
        Backup available. Demand=[10,10].

        Case: invest_cost=100 + fuel=20 = 120 < retirement=500 + backup_fuel=40 = 540.
        Optimizer builds the boiler to avoid retirement cost.

        Sensitivity: Without effects_of_retirement, backup is cheaper (fuel=40 vs 120).
        With retirement=500, investing becomes cheaper. Cost difference proves feature.
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
                'NewBoiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=fx.InvestParameters(
                        minimum_size=10,
                        maximum_size=100,
                        effects_of_investment=100,
                        effects_of_retirement=500,  # Penalty if NOT investing
                    ),
                ),
            ),
            fx.linear_converters.Boiler(
                'Backup',
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat', size=100),
            ),
        )
        fs = optimize(fs)
        # Building NewBoiler: invest=100, fuel=20, total=120
        # Not building: retirement=500, backup_fuel=40, total=540
        # Optimizer chooses to build (120 < 540)
        assert_allclose(fs.solution['NewBoiler(heat)|invested'].item(), 1.0, atol=1e-5)
        assert_allclose(fs.solution['costs'].item(), 120.0, rtol=1e-5)

    def test_invest_retirement_triggers_when_not_investing(self, optimize):
        """Proves: effects_of_retirement is incurred when investment is skipped.

        Boiler with invest_cost=1000, effects_of_retirement=50.
        Backup available at eta=0.5. Demand=[10,10].

        Case: invest_cost=1000 + fuel=20 = 1020 > retirement=50 + backup_fuel=40 = 90.
        Optimizer skips investment, pays retirement cost.

        Sensitivity: Without effects_of_retirement, cost=40. With it, cost=90.
        The 50€ difference proves retirement cost is applied.
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
                'ExpensiveBoiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=fx.InvestParameters(
                        minimum_size=10,
                        maximum_size=100,
                        effects_of_investment=1000,
                        effects_of_retirement=50,  # Small penalty for not investing
                    ),
                ),
            ),
            fx.linear_converters.Boiler(
                'Backup',
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat', size=100),
            ),
        )
        fs = optimize(fs)
        # Not building: retirement=50, backup_fuel=40, total=90
        # Building: invest=1000, fuel=20, total=1020
        # Optimizer skips investment (90 < 1020)
        assert_allclose(fs.solution['ExpensiveBoiler(heat)|invested'].item(), 0.0, atol=1e-5)
        assert_allclose(fs.solution['costs'].item(), 90.0, rtol=1e-5)


class TestFlowInvestWithStatus:
    """Tests for combined InvestParameters and StatusParameters on the same Flow."""

    def test_invest_with_startup_cost(self, optimize):
        """Proves: InvestParameters and StatusParameters work together correctly.

        Boiler with investment sizing AND startup costs.
        Demand=[0,20,0,20]. Two startup events if boiler is used.

        Sensitivity: Without startup_cost, cost = invest + fuel.
        With startup_cost=50 × 2, cost increases by 100.
        """
        fs = make_flow_system(4)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([0, 20, 0, 20])),
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
                        maximum_size=100,
                        effects_of_investment=10,
                        effects_of_investment_per_size=1,
                    ),
                    status_parameters=fx.StatusParameters(effects_per_startup=50),
                ),
            ),
        )
        fs = optimize(fs)
        # size=20 (peak), invest=10+20=30, fuel=40, 2 startups=100
        # total = 30 + 40 + 100 = 170
        assert_allclose(fs.solution['Boiler(heat)|size'].item(), 20.0, rtol=1e-5)
        assert_allclose(fs.solution['costs'].item(), 170.0, rtol=1e-5)

    def test_invest_with_min_uptime(self, optimize):
        """Proves: Invested unit respects min_uptime constraint.

        InvestBoiler with sizing AND min_uptime=2. Once started, must stay on 2 hours.
        Backup available but expensive. Demand=[20,10,20].

        Without min_uptime, InvestBoiler could freely turn on/off.
        With min_uptime=2, once started it must stay on for 2 hours.

        Sensitivity: The cost changes due to min_uptime forcing operation patterns.
        """
        fs = make_flow_system(3)
        fs.add_elements(
            fx.Bus('Heat'),  # Strict balance (demand must be met)
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([20, 10, 20])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.linear_converters.Boiler(
                'InvestBoiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    relative_minimum=0.1,
                    size=fx.InvestParameters(
                        maximum_size=100,
                        effects_of_investment_per_size=1,
                    ),
                    status_parameters=fx.StatusParameters(min_uptime=2),
                ),
            ),
            fx.linear_converters.Boiler(
                'Backup',
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat', size=100),
            ),
        )
        fs = optimize(fs)
        # InvestBoiler is built (cheaper fuel @eta=1.0 vs Backup @eta=0.5)
        # size=20 (peak demand), invest=20
        # min_uptime=2: runs continuously t=0,1,2
        # fuel = 20 + 10 + 20 = 50
        # total = 20 (invest) + 50 (fuel) = 70
        assert_allclose(fs.solution['InvestBoiler(heat)|size'].item(), 20.0, rtol=1e-5)
        assert_allclose(fs.solution['costs'].item(), 70.0, rtol=1e-5)
        # Verify InvestBoiler runs all 3 hours due to min_uptime
        status = fs.solution['InvestBoiler(heat)|status'].values[:-1]
        assert_allclose(status, [1, 1, 1], atol=1e-5)

    def test_invest_with_active_hours_max(self, optimize):
        """Proves: Invested unit respects active_hours_max constraint.

        InvestBoiler (eta=1.0) with active_hours_max=2. Backup (eta=0.5).
        Demand=[10,10,10,10]. InvestBoiler can only run 2 of 4 hours.

        Sensitivity: Without limit, InvestBoiler runs all 4 hours → fuel=40.
        With active_hours_max=2, InvestBoiler runs 2 hours, backup runs 2 → cost higher.
        """
        fs = make_flow_system(4)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10, 10])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.linear_converters.Boiler(
                'InvestBoiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=fx.InvestParameters(
                        maximum_size=100,
                        effects_of_investment_per_size=0.1,
                    ),
                    status_parameters=fx.StatusParameters(active_hours_max=2),
                ),
            ),
            fx.linear_converters.Boiler(
                'Backup',
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat', size=100),
            ),
        )
        fs = optimize(fs)
        # InvestBoiler: 2 hours @ eta=1.0 → fuel=20
        # Backup: 2 hours @ eta=0.5 → fuel=40
        # invest = 10*0.1 = 1
        # total = 1 + 20 + 40 = 61
        assert_allclose(fs.solution['costs'].item(), 61.0, rtol=1e-5)
        # Verify InvestBoiler only runs 2 hours
        status = fs.solution['InvestBoiler(heat)|status'].values[:-1]
        assert_allclose(sum(status), 2.0, atol=1e-5)
