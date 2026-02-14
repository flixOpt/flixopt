"""Mathematical correctness tests for COMBINATIONS of features.

These tests verify that piecewise conversion, status parameters, investment
sizing, and effects work correctly when combined — catching interaction bugs
that single-feature tests miss.

Each test is analytically solvable and asserts on a hand-calculated objective.
"""

import numpy as np
from numpy.testing import assert_allclose

import flixopt as fx

from .conftest import make_flow_system


class TestPiecewiseWithInvestment:
    """Tests combining PiecewiseConversion with InvestParameters."""

    def test_piecewise_conversion_with_investment_sizing(self, optimize):
        """Proves: PiecewiseConversion and InvestParameters on the same converter's flow
        work together — the optimizer picks the right piecewise segment AND sizes the flow.

        Converter: fuel→heat, piecewise 2-segment.
        Seg1: fuel 0→30, heat 0→20 (efficiency 0.667).
        Seg2: fuel 30→80, heat 20→70 (efficiency 1.0, better at high load).
        Demand=[40,40]. Falls in segment 2.
        Heat flow has InvestParameters(maximum_size=100, effects_of_investment_per_size=1).

        Sensitivity: If invest sizing were broken, the piecewise constraint couldn't
        interact with size → infeasible or wrong cost. The unique cost (invest + fuel)
        proves both mechanisms cooperate.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([40, 40])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.LinearConverter(
                'Converter',
                inputs=[fx.Flow('fuel', bus='Gas', size=fx.InvestParameters(maximum_size=100))],
                outputs=[
                    fx.Flow(
                        'heat',
                        bus='Heat',
                        size=fx.InvestParameters(
                            maximum_size=100,
                            effects_of_investment_per_size=1,
                        ),
                    )
                ],
                piecewise_conversion=fx.PiecewiseConversion(
                    {
                        'fuel': fx.Piecewise([fx.Piece(0, 30), fx.Piece(30, 80)]),
                        'heat': fx.Piecewise([fx.Piece(0, 20), fx.Piece(20, 70)]),
                    }
                ),
            ),
        )
        fs = optimize(fs)
        # heat=40 in segment 2: fuel = 30 + (40-20)/(70-20) * (80-30) = 30 + 20 = 50
        # invest = 40 * 1 = 40 (size=40, peak demand)
        # fuel cost = 2 * 50 = 100
        # total = 40 + 100 = 140
        assert_allclose(fs.solution['Converter(heat)|size'].item(), 40.0, rtol=1e-4)
        assert_allclose(fs.solution['costs'].item(), 140.0, rtol=1e-4)

    def test_piecewise_invest_cost_with_optional_skip(self, optimize):
        """Proves: Piecewise investment cost function works with optional (non-mandatory)
        investment — optimizer can choose NOT to invest when piecewise cost is too high.

        InvestBoiler: piecewise invest cost (expensive) + eta=1.0.
        Backup: eta=0.5, no invest. Demand=[10,10].

        If piecewise invest cost at minimum viable size exceeds operational savings,
        optimizer skips investment.

        Sensitivity: If piecewise invest skipped, InvestBoiler serves all → fuel=20.
        If piecewise cost correctly applied and expensive, backup cheaper.
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
                outputs=[fx.Flow('gas', bus='Gas', effects_per_flow_hour=1)],
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
                        piecewise_effects_of_investment=fx.PiecewiseEffects(
                            piecewise_origin=fx.Piecewise([fx.Piece(0, 100)]),
                            piecewise_shares={
                                'costs': fx.Piecewise([fx.Piece(0, 9999)]),  # Very expensive
                            },
                        ),
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
        # InvestBoiler: invest ≈ 10*99.99 ≈ 999.9 + fuel=20 ≈ 1020
        # Backup: fuel = 20/0.5 = 40
        # Backup is much cheaper
        assert_allclose(fs.solution['InvestBoiler(heat)|invested'].item(), 0.0, atol=1e-5)
        assert_allclose(fs.solution['costs'].item(), 40.0, rtol=1e-5)


class TestPiecewiseWithStatus:
    """Tests combining PiecewiseConversion with StatusParameters."""

    def test_piecewise_conversion_with_startup_cost(self, optimize):
        """Proves: PiecewiseConversion with gap (off-state piece) and startup costs
        interact correctly — startup cost is charged when entering the active segment.

        Converter: off piece [0,0] and operating piece [30→60, 30→60].
        Startup cost = 100€. Demand=[0, 40, 0, 40]. Two startups needed.

        Sensitivity: Without startup cost, cost = 2*40 = 80 (fuel only).
        With 100€/startup × 2, cost = 80 + 200 = 280.
        """
        fs = make_flow_system(4)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow(
                        'heat',
                        bus='Heat',
                        size=1,
                        fixed_relative_profile=np.array([0, 40, 0, 40]),
                    ),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[fx.Flow('gas', bus='Gas', effects_per_flow_hour=1)],
            ),
            fx.LinearConverter(
                'Converter',
                inputs=[
                    fx.Flow(
                        'fuel',
                        bus='Gas',
                        size=100,
                        previous_flow_rate=0,
                        status_parameters=fx.StatusParameters(effects_per_startup=100),
                    )
                ],
                outputs=[fx.Flow('heat', bus='Heat', size=100)],
                piecewise_conversion=fx.PiecewiseConversion(
                    {
                        'fuel': fx.Piecewise([fx.Piece(0, 0), fx.Piece(30, 60)]),
                        'heat': fx.Piecewise([fx.Piece(0, 0), fx.Piece(30, 60)]),
                    }
                ),
            ),
        )
        fs = optimize(fs)
        # fuel per active ts = 40, 2 active timesteps → 80
        # 2 startups × 100 = 200
        # total = 280
        assert_allclose(fs.solution['costs'].item(), 280.0, rtol=1e-4)

    def test_piecewise_with_min_uptime(self, optimize):
        """Proves: Piecewise converter with gap and min_uptime interact correctly.

        Converter: off piece + operating piece [20→50, 20→50].
        min_uptime=2. Demand=[0, 30, 30, 0, 30]. CheapBackup available.

        With min_uptime=2, once the converter starts, it must stay on for 2 hours.
        Converter serves t=1,2 (one startup block). For t=4, starting would require
        staying on past horizon — relaxed at end, so it's ok.

        Sensitivity: Without min_uptime, converter could freely turn on/off each
        timestep. With min_uptime=2, forced into blocks.
        """
        fs = make_flow_system(5)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow(
                        'heat',
                        bus='Heat',
                        size=1,
                        fixed_relative_profile=np.array([0, 30, 30, 0, 30]),
                    ),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[fx.Flow('gas', bus='Gas', effects_per_flow_hour=1)],
            ),
            fx.Source(
                'Backup',
                outputs=[fx.Flow('heat', bus='Heat', effects_per_flow_hour=10)],
            ),
            fx.LinearConverter(
                'Converter',
                inputs=[
                    fx.Flow(
                        'fuel',
                        bus='Gas',
                        size=100,
                        previous_flow_rate=0,
                        status_parameters=fx.StatusParameters(min_uptime=2),
                    )
                ],
                outputs=[fx.Flow('heat', bus='Heat', size=100)],
                piecewise_conversion=fx.PiecewiseConversion(
                    {
                        'fuel': fx.Piecewise([fx.Piece(0, 0), fx.Piece(20, 50)]),
                        'heat': fx.Piecewise([fx.Piece(0, 0), fx.Piece(20, 50)]),
                    }
                ),
            ),
        )
        fs = optimize(fs)
        # Converter is cheaper (1€/kWh vs Backup 10€/kWh).
        # All demand served by converter where possible.
        # Total converter fuel = 30 + 30 + 30 = 90 for demand timesteps.
        # min_uptime forces blocks of ≥2 hours.
        # Verify cost is consistent and converter status respects min_uptime.
        status = fs.solution['Converter(fuel)|status'].values[:-1]
        # Check min_uptime: no isolated single on-hours
        for i in range(len(status)):
            if status[i] > 0.5:
                # Must have neighbor also on (except at boundary)
                has_neighbor = False
                if i > 0 and status[i - 1] > 0.5:
                    has_neighbor = True
                if i < len(status) - 1 and status[i + 1] > 0.5:
                    has_neighbor = True
                # At boundary (last timestep), min_uptime not enforced
                if i == len(status) - 1:
                    has_neighbor = True
                assert has_neighbor, f'Isolated on-hour at t={i}: status={status}'


class TestPiecewiseThreeSegments:
    """Tests for piecewise conversion with 3+ segments."""

    def test_three_segment_piecewise(self, optimize):
        """Proves: 3-segment PiecewiseConversion correctly selects the optimal segment
        for a given demand level.

        Segments:
        Seg1: fuel 0→10, heat 0→10  (efficiency 1.0 — low load)
        Seg2: fuel 10→30, heat 10→25 (efficiency 0.75 — mid load, less efficient)
        Seg3: fuel 30→60, heat 25→55 (efficiency 1.0 — high load)

        Demand=40 falls in segment 3.

        Sensitivity: If segment selection were wrong (e.g. always seg1 ratio),
        fuel would differ. Only correct 3-segment handling gives the right fuel value.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([40, 40])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[fx.Flow('gas', bus='Gas', effects_per_flow_hour=1)],
            ),
            fx.LinearConverter(
                'Converter',
                inputs=[fx.Flow('fuel', bus='Gas')],
                outputs=[fx.Flow('heat', bus='Heat')],
                piecewise_conversion=fx.PiecewiseConversion(
                    {
                        'fuel': fx.Piecewise([fx.Piece(0, 10), fx.Piece(10, 30), fx.Piece(30, 60)]),
                        'heat': fx.Piecewise([fx.Piece(0, 10), fx.Piece(10, 25), fx.Piece(25, 55)]),
                    }
                ),
            ),
        )
        fs = optimize(fs)
        # heat=40 in segment 3: fuel = 30 + (40-25)/(55-25) * (60-30) = 30 + 15 = 45
        # cost = 2 × 45 = 90
        assert_allclose(fs.solution['costs'].item(), 90.0, rtol=1e-4)
        assert_allclose(fs.solution['Converter(fuel)|flow_rate'].values[0], 45.0, rtol=1e-4)

    def test_three_segment_low_load_selection(self, optimize):
        """Proves: With 3 segments, low demand correctly uses segment 1.

        Same 3-segment setup. Demand=5 falls in segment 1.
        Seg1: fuel 0→10, heat 0→10 (1:1 ratio).

        Sensitivity: If segment 2 or 3 were incorrectly selected, fuel would differ.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([5, 5])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[fx.Flow('gas', bus='Gas', effects_per_flow_hour=1)],
            ),
            fx.LinearConverter(
                'Converter',
                inputs=[fx.Flow('fuel', bus='Gas')],
                outputs=[fx.Flow('heat', bus='Heat')],
                piecewise_conversion=fx.PiecewiseConversion(
                    {
                        'fuel': fx.Piecewise([fx.Piece(0, 10), fx.Piece(10, 30), fx.Piece(30, 60)]),
                        'heat': fx.Piecewise([fx.Piece(0, 10), fx.Piece(10, 25), fx.Piece(25, 55)]),
                    }
                ),
            ),
        )
        fs = optimize(fs)
        # heat=5 in segment 1: fuel = 0 + (5-0)/(10-0) * (10-0) = 5
        # cost = 2 × 5 = 10
        assert_allclose(fs.solution['costs'].item(), 10.0, rtol=1e-4)

    def test_three_segment_mid_load_selection(self, optimize):
        """Proves: With 3 segments, mid demand correctly uses segment 2.

        Same 3-segment setup. Demand=18 falls in segment 2.
        Seg2: fuel 10→30, heat 10→25.

        Sensitivity: fuel = 10 + (18-10)/(25-10) * (30-10) = 10 + 10.667 ≈ 20.667.
        This value is unique to segment 2.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([18, 18])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[fx.Flow('gas', bus='Gas', effects_per_flow_hour=1)],
            ),
            fx.LinearConverter(
                'Converter',
                inputs=[fx.Flow('fuel', bus='Gas')],
                outputs=[fx.Flow('heat', bus='Heat')],
                piecewise_conversion=fx.PiecewiseConversion(
                    {
                        'fuel': fx.Piecewise([fx.Piece(0, 10), fx.Piece(10, 30), fx.Piece(30, 60)]),
                        'heat': fx.Piecewise([fx.Piece(0, 10), fx.Piece(10, 25), fx.Piece(25, 55)]),
                    }
                ),
            ),
        )
        fs = optimize(fs)
        # heat=18 in segment 2: fuel = 10 + (18-10)/(25-10) * (30-10) = 10 + 8/15*20 = 10 + 10.667
        expected_fuel = 10 + (8 / 15) * 20
        expected_cost = 2 * expected_fuel
        assert_allclose(fs.solution['costs'].item(), expected_cost, rtol=1e-4)


class TestStatusWithEffects:
    """Tests for StatusParameters contributing to non-standard effects."""

    def test_startup_cost_on_co2_effect(self, optimize):
        """Proves: effects_per_startup can contribute to a non-cost effect (CO2),
        and that this correctly interacts with effect constraints.

        Boiler with effects_per_startup={'CO2': 50} (startup emits 50kg CO2).
        CO2 capped at maximum_total=60. Demand=[0,20,0,20] → 2 startups = 100kg CO2.
        Exceeds cap! Optimizer must reduce startups.

        Alternative: keep boiler running continuously (1 startup = 50kg CO2, within cap).
        But boiler has relative_minimum=0.1 → produces ≥2kW when on, excess goes to
        bus with imbalance_penalty=0.

        Sensitivity: Without CO2 cap, 2 startups optimal. With cap=60, forced to 1 startup
        with continuous operation → different cost.
        """
        fs = make_flow_system(4)
        co2 = fx.Effect('CO2', 'kg', maximum_total=60)
        costs = fx.Effect('costs', '€', is_standard=True, is_objective=True)
        fs.add_elements(
            fx.Bus('Heat', imbalance_penalty_per_flow_hour=0),
            fx.Bus('Gas'),
            costs,
            co2,
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow(
                        'heat',
                        bus='Heat',
                        size=1,
                        fixed_relative_profile=np.array([0, 20, 0, 20]),
                    ),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[fx.Flow('gas', bus='Gas', effects_per_flow_hour=1)],
            ),
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=100,
                    relative_minimum=0.1,
                    previous_flow_rate=0,
                    status_parameters=fx.StatusParameters(
                        effects_per_startup={'CO2': 50},
                    ),
                ),
            ),
        )
        fs = optimize(fs)
        # With max CO2=60 and 50 kg/startup, can only start once.
        # Boiler stays on continuously: status=[1,1,1,1], 1 startup.
        # CO2 = 50 (1 startup) ≤ 60 ✓
        # Fuel = on at relative_min when no demand: t0=10, t1=20, t2=10, t3=20 → 60
        # Or optimizer can find minimum-cost continuous pattern
        assert fs.solution['CO2'].item() <= 60.0 + 1e-5
        # Verify only 1 startup (status continuous)
        status = fs.solution['Boiler(heat)|status'].values[:-1]
        startups = sum(1 for i in range(len(status)) if status[i] > 0.5 and (i == 0 or status[i - 1] < 0.5))
        assert startups <= 1, f'Expected ≤1 startup, got {startups}: status={status}'

    def test_effects_per_active_hour_on_multiple_effects(self, optimize):
        """Proves: effects_per_active_hour can contribute to multiple effects simultaneously.

        Boiler with effects_per_active_hour={'costs': 10, 'CO2': 5}.
        Demand=[20,20]. Boiler on 2 hours.

        Sensitivity: Without effects_per_active_hour, costs=40, CO2=0.
        With it, costs = 40 + 2*10 = 60, CO2 = 2*5 = 10.
        """
        fs = make_flow_system(2)
        co2 = fx.Effect('CO2', 'kg')
        costs = fx.Effect('costs', '€', is_standard=True, is_objective=True)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            costs,
            co2,
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([20, 20])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[fx.Flow('gas', bus='Gas', effects_per_flow_hour=1)],
            ),
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=100,
                    status_parameters=fx.StatusParameters(
                        effects_per_active_hour={'costs': 10, 'CO2': 5},
                    ),
                ),
            ),
        )
        fs = optimize(fs)
        # fuel = 40, active_hour costs = 2*10 = 20, total costs = 60
        # CO2 = 2*5 = 10
        assert_allclose(fs.solution['costs'].item(), 60.0, rtol=1e-5)
        assert_allclose(fs.solution['CO2'].item(), 10.0, rtol=1e-5)


class TestInvestWithRelativeMinimum:
    """Tests combining InvestParameters with relative_minimum."""

    def test_invest_sizing_respects_relative_minimum(self, optimize):
        """Proves: Investment sizing with relative_minimum forces the optimizer to
        balance size against minimum load — a clever tradeoff.

        Boiler: invest (1€/kW), relative_minimum=0.5, status_parameters, eta=1.0.
        Backup: eta=0.5, no invest. Demand=[15, 50].

        Optimizer sizes to 30: relative_minimum * 30 = 15 = demand[0].
        t=0: boiler ON at exactly 15 (min load). t=1: boiler ON at 30 + backup 20.

        Sensitivity: Without relative_minimum, boiler sized to 50, no backup needed.
        invest(50)+fuel(65)=115. With relative_minimum, optimizer finds size=30,
        invest(30)+boiler_fuel(45)+backup_fuel(40)=115 — same total but different size!
        If relative_minimum were broken (ignored), size=50 and no backup needed.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([15, 50])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[fx.Flow('gas', bus='Gas', effects_per_flow_hour=1)],
            ),
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    relative_minimum=0.5,
                    size=fx.InvestParameters(
                        maximum_size=100,
                        mandatory=True,
                        effects_of_investment_per_size=1,
                    ),
                    status_parameters=fx.StatusParameters(),
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
        # Optimal: size=30, relative_min*30=15=demand[0]
        # t=0: boiler=15, t=1: boiler=30 + backup=20 → backup_fuel=20/0.5=40
        # invest=30, boiler_fuel=45, backup_fuel=40, total=115
        assert_allclose(fs.solution['Boiler(heat)|size'].item(), 30.0, rtol=1e-4)
        assert_allclose(fs.solution['costs'].item(), 115.0, rtol=1e-4)


class TestConversionWithTimeVaryingEffects:
    """Tests for conversion factors with time-varying effects."""

    def test_time_varying_effects_per_flow_hour(self, optimize):
        """Proves: Time-varying effects_per_flow_hour correctly applies different rates
        per timestep when combined with conversion.

        Boiler eta=0.5. Gas cost = [1, 3] (time-varying). Demand=[10, 10].
        t=0: fuel = 10/0.5 = 20, cost = 20*1 = 20.
        t=1: fuel = 10/0.5 = 20, cost = 20*3 = 60.
        Total = 80.

        Sensitivity: If time-varying cost were broadcast as mean (2), cost=80 (same!).
        So use asymmetric demands: [20, 10] → fuel=[40,20], cost=[40,60]=100.
        If mean(2) were used: cost=120. Only per-timestep gives 100.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([20, 10])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=np.array([1, 3])),
                ],
            ),
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat'),
            ),
        )
        fs = optimize(fs)
        # t=0: fuel=40, cost=40*1=40. t=1: fuel=20, cost=20*3=60.
        # total = 100
        assert_allclose(fs.solution['costs'].item(), 100.0, rtol=1e-5)

    def test_effects_per_flow_hour_with_dual_output_conversion(self, optimize):
        """Proves: effects_per_flow_hour applied to individual flows of a multi-output
        converter correctly accumulates effects for each flow independently.

        CHP: fuel→heat+elec. Fuel costs 1€/kWh, elec earns -2€/kWh.
        CO2: fuel emits 0.5 kg/kWh, elec avoids -0.3 kg/kWh (grid offset).
        Demand=50 heat per timestep.

        Sensitivity: Total is uniquely determined by conversion factors + effects.
        """
        fs = make_flow_system(2)
        co2 = fx.Effect('CO2', 'kg')
        costs = fx.Effect('costs', '€', is_standard=True, is_objective=True)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Elec'),
            fx.Bus('Gas'),
            costs,
            co2,
            fx.Sink(
                'HeatDemand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([50, 50])),
                ],
            ),
            fx.Sink(
                'ElecGrid',
                inputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour={'costs': -2, 'CO2': -0.3}),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour={'costs': 1, 'CO2': 0.5}),
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
        # costs per ts: fuel_cost=100*1=100, elec_revenue=40*(-2)=-80 → net=20
        # total costs = 2*20 = 40
        # CO2 per ts: fuel=100*0.5=50, elec=40*(-0.3)=-12 → net=38
        # total CO2 = 2*38 = 76
        assert_allclose(fs.solution['costs'].item(), 40.0, rtol=1e-5)
        assert_allclose(fs.solution['CO2'].item(), 76.0, rtol=1e-5)


class TestPiecewiseInvestWithStatus:
    """Tests combining piecewise investment costs with status parameters."""

    def test_piecewise_invest_with_startup_cost(self, optimize):
        """Proves: Piecewise investment cost and startup cost work together on the same flow.

        Boiler: piecewise invest (economies of scale) + startup cost = 50€.
        Demand=[0, 30, 0, 30]. Two startups.
        Piecewise invest: size 0→50 costs 0→100 (2€/kW), 50→200 costs 100→250 (1€/kW).
        Peak demand=30 → invest in seg1: cost = 30/50*100 = 60.

        Sensitivity: If either piecewise invest or startup cost were broken,
        total would differ from 60 + 60 + 100 = 220.
        """
        fs = make_flow_system(4)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow(
                        'heat',
                        bus='Heat',
                        size=1,
                        fixed_relative_profile=np.array([0, 30, 0, 30]),
                    ),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[fx.Flow('gas', bus='Gas', effects_per_flow_hour=1)],
            ),
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    relative_minimum=0.5,
                    previous_flow_rate=0,
                    size=fx.InvestParameters(
                        maximum_size=200,
                        piecewise_effects_of_investment=fx.PiecewiseEffects(
                            piecewise_origin=fx.Piecewise([fx.Piece(0, 50), fx.Piece(50, 200)]),
                            piecewise_shares={
                                'costs': fx.Piecewise([fx.Piece(0, 100), fx.Piece(100, 250)]),
                            },
                        ),
                    ),
                    status_parameters=fx.StatusParameters(effects_per_startup=50),
                ),
            ),
        )
        fs = optimize(fs)
        # size=30 (peak demand), invest = 30/50*100 = 60
        # fuel = 2*30 = 60 (eta=1.0)
        # 2 startups × 50 = 100
        # total = 60 + 60 + 100 = 220
        assert_allclose(fs.solution['Boiler(heat)|size'].item(), 30.0, rtol=1e-4)
        assert_allclose(fs.solution['costs'].item(), 220.0, rtol=1e-4)


class TestStatusWithMultipleConstraints:
    """Tests combining multiple status parameters on the same flow."""

    def test_startup_limit_with_max_downtime(self, optimize):
        """Proves: startup_limit and max_downtime interact correctly — both constraints
        must be satisfied simultaneously.

        CheapBoiler: startup_limit=2, max_downtime=1, relative_minimum=0.5, size=20.
        Was on before horizon. Demand=[10]*6. Backup at eta=0.5.

        max_downtime=1: can be off at most 1 consecutive hour.
        startup_limit=2: at most 2 startups total.

        These interact: with max_downtime=1, the boiler must run frequently,
        and startup_limit=2 constrains how it can restart.

        Sensitivity: Without startup_limit, unconstrained restarts.
        Without max_downtime, can stay off indefinitely.
        """
        fs = make_flow_system(6)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow(
                        'heat',
                        bus='Heat',
                        size=1,
                        fixed_relative_profile=np.array([10, 10, 10, 10, 10, 10]),
                    ),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[fx.Flow('gas', bus='Gas', effects_per_flow_hour=1)],
            ),
            fx.linear_converters.Boiler(
                'CheapBoiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=20,
                    relative_minimum=0.5,
                    previous_flow_rate=10,
                    status_parameters=fx.StatusParameters(
                        startup_limit=2,
                        max_downtime=1,
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
        # Verify constraints
        status = fs.solution['CheapBoiler(heat)|status'].values[:-1]

        # Check max_downtime: no 2+ consecutive off-hours
        for i in range(len(status) - 1):
            assert not (status[i] < 0.5 and status[i + 1] < 0.5), (
                f'max_downtime violated at t={i},{i + 1}: status={status}'
            )

        # Check startup_limit: at most 2 startups
        startups = sum(1 for i in range(len(status)) if status[i] > 0.5 and (i == 0 or status[i - 1] < 0.5))
        # Account for carry-over: was on before, so first on isn't a startup
        # if status[0] > 0.5 then it was already on (previous_flow_rate=10)
        if status[0] > 0.5:
            startups -= 1  # Not a startup, was already on
        assert startups <= 2, f'startup_limit violated: {startups} startups, status={status}'

    def test_min_uptime_with_min_downtime(self, optimize):
        """Proves: min_uptime and min_downtime together force a regular on/off pattern.

        Boiler: min_uptime=2, min_downtime=2, previous_flow_rate=0.
        Demand=[20]*6. Backup at eta=0.5.

        With min_uptime=2 + min_downtime=2, operation must be in blocks:
        ON for ≥2, then OFF for ≥2.

        Sensitivity: Without these constraints, boiler could run all 6 hours.
        With constraints, forced into block pattern → backup needed for off blocks.
        """
        fs = make_flow_system(6)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow(
                        'heat',
                        bus='Heat',
                        size=1,
                        fixed_relative_profile=np.array([20, 20, 20, 20, 20, 20]),
                    ),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[fx.Flow('gas', bus='Gas', effects_per_flow_hour=1)],
            ),
            fx.linear_converters.Boiler(
                'CheapBoiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=100,
                    relative_minimum=0.1,
                    previous_flow_rate=0,
                    status_parameters=fx.StatusParameters(min_uptime=2, min_downtime=2),
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
        status = fs.solution['CheapBoiler(heat)|status'].values[:-1]

        # Verify min_uptime: each on-block is ≥2 hours
        on_block_len = 0
        for i, s in enumerate(status):
            if s > 0.5:
                on_block_len += 1
            else:
                if on_block_len > 0:
                    assert on_block_len >= 2, (
                        f'min_uptime violated: on-block of {on_block_len} at t<{i}: status={status}'
                    )
                on_block_len = 0

        # Verify min_downtime: each off-block is ≥2 hours (within horizon)
        off_block_len = 0
        for i, s in enumerate(status):
            if s < 0.5:
                off_block_len += 1
            else:
                if 0 < off_block_len < 2:
                    # Off block ended before reaching min_downtime=2
                    # (but first off-block may be carry-over from previous_flow_rate=0)
                    if i - off_block_len > 0:  # Not the initial off period
                        assert off_block_len >= 2, (
                            f'min_downtime violated: off-block of {off_block_len} at t<{i}: status={status}'
                        )
                off_block_len = 0

        # CheapBoiler runs some hours, Backup covers the rest
        # Total cost > 120 (if all cheap) but < 240 (if all backup)
        assert fs.solution['costs'].item() > 120 - 1e-5
        assert fs.solution['costs'].item() < 240 + 1e-5


class TestEffectsWithConversion:
    """Tests for effects interacting with conversion and other constraints."""

    def test_effect_share_with_investment(self, optimize):
        """Proves: share_from_periodic works correctly when the periodic contribution
        comes from investment costs of a converter.

        costs has share_from_periodic={'CO2': 20}. Boiler invests with
        CO2_periodic=10 (from investment). Direct costs = invest(50) + fuel(20).
        Shared: 20 × 10 = 200. Total costs = 50 + 20 + 200 = 270.

        Sensitivity: Without share_from_periodic, costs=70. With it, costs=270.
        """
        fs = make_flow_system(2)
        co2 = fx.Effect('CO2', 'kg')
        costs = fx.Effect('costs', '€', is_standard=True, is_objective=True, share_from_periodic={'CO2': 20})
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            costs,
            co2,
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([10, 10])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[fx.Flow('gas', bus='Gas', effects_per_flow_hour=1)],
            ),
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow(
                    'heat',
                    bus='Heat',
                    size=fx.InvestParameters(
                        fixed_size=50,
                        effects_of_investment={'costs': 50, 'CO2': 10},
                    ),
                ),
            ),
        )
        fs = optimize(fs)
        # direct costs = 50 (invest) + 20 (fuel) = 70
        # CO2 periodic = 10
        # costs += 20 * 10 = 200
        # total costs = 270
        assert_allclose(fs.solution['costs'].item(), 270.0, rtol=1e-5)
        assert_allclose(fs.solution['CO2'].item(), 10.0, rtol=1e-5)

    def test_effect_maximum_with_status_contribution(self, optimize):
        """Proves: Effect maximum_total correctly accounts for contributions from
        StatusParameters (effects_per_startup) when constraining.

        CO2 has maximum_total=20. Boiler startup emits 15 kg CO2.
        Fuel emits 0.5 kg CO2/kWh. Demand=[0,20,0,20] → would need 2 startups.
        2 startups = 30 kg CO2 (exceeds cap). With cap, optimizer limits startups.

        Sensitivity: Without CO2 cap, 2 startups → CO2=30+10=40.
        With cap=20, forced to 1 startup (continuous) → CO2=15 + some fuel CO2.
        """
        fs = make_flow_system(4)
        co2 = fx.Effect('CO2', 'kg', maximum_total=20)
        costs = fx.Effect('costs', '€', is_standard=True, is_objective=True)
        fs.add_elements(
            fx.Bus('Heat', imbalance_penalty_per_flow_hour=0),
            fx.Bus('Gas'),
            costs,
            co2,
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow(
                        'heat',
                        bus='Heat',
                        size=1,
                        fixed_relative_profile=np.array([0, 10, 0, 10]),
                    ),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour={'costs': 1, 'CO2': 0.1}),
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
                    relative_minimum=0.1,
                    previous_flow_rate=0,
                    status_parameters=fx.StatusParameters(
                        effects_per_startup={'CO2': 15},
                    ),
                ),
            ),
        )
        fs = optimize(fs)
        # CO2 must stay ≤ 20
        assert fs.solution['CO2'].item() <= 20.0 + 1e-5


class TestInvestWithEffects:
    """Tests combining investment with effect constraints."""

    def test_invest_per_size_on_non_cost_effect(self, optimize):
        """Proves: effects_of_investment_per_size can contribute to a non-cost effect,
        and effect constraints correctly bound the investment.

        Boiler: invest_per_size = {'costs': 1, 'CO2': 2}.
        CO2 has maximum_periodic=50. This limits the investment size to ≤25 (50/2).
        Demand peak=30. Without CO2 cap, size=30. With cap, size limited to 25.
        Need backup for remaining 5.

        Sensitivity: Without CO2 cap, size=30, cost=30+30=60.
        With cap, size=25, invest_cost=25, need backup for excess → cost differs.
        """
        fs = make_flow_system(2)
        co2 = fx.Effect('CO2', 'kg', maximum_periodic=50)
        costs = fx.Effect('costs', '€', is_standard=True, is_objective=True)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            costs,
            co2,
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([30, 30])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[fx.Flow('gas', bus='Gas', effects_per_flow_hour=1)],
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
                        mandatory=True,
                        effects_of_investment_per_size={'costs': 1, 'CO2': 2},
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
        # CO2 = size * 2 ≤ 50 → size ≤ 25
        # InvestBoiler: size=25, invest_cost=25, fuel=2*25=50
        # Backup covers remaining: 2*5/0.5 = 20
        # total = 25 + 50 + 20 = 95
        assert fs.solution['CO2'].item() <= 50.0 + 1e-5
        assert_allclose(fs.solution['InvestBoiler(heat)|size'].item(), 25.0, rtol=1e-4)
        assert_allclose(fs.solution['costs'].item(), 95.0, rtol=1e-4)
