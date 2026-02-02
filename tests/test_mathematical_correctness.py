"""
End-to-end mathematical correctness tests for flixopt.

Each test builds a tiny, analytically solvable optimization model and asserts
that the objective (or key solution variables) match a hand-calculated value.
This catches regressions in formulations without relying on recorded baselines.
"""

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

import flixopt as fx

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fs(n_timesteps: int = 3) -> tuple[fx.FlowSystem, pd.DatetimeIndex]:
    ts = pd.date_range('2020-01-01', periods=n_timesteps, freq='h')
    return fx.FlowSystem(ts), ts


def _solve(fs: fx.FlowSystem) -> fx.FlowSystem:
    fs.optimize(fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=60, log_to_console=False))
    return fs


# ===========================================================================
# Category 1: Conversion & Efficiency
# ===========================================================================


class TestConversionEfficiency:
    def test_boiler_efficiency(self):
        """Q_fu = Q_th / eta  →  fuel cost = sum(demand) / eta."""
        fs, _ = _make_fs(3)
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
                'Boiler',
                thermal_efficiency=0.8,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat'),
            ),
        )
        _solve(fs)
        # fuel = (10+20+10)/0.8 = 50, cost@1€/kWh = 50
        assert_allclose(fs.solution['costs'].item(), 50.0, rtol=1e-5)

    def test_variable_efficiency(self):
        """Time-varying eta: cost = sum(demand_t / eta_t)."""
        fs, _ = _make_fs(2)
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
                thermal_efficiency=np.array([0.5, 1.0]),
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat'),
            ),
        )
        _solve(fs)
        # fuel = 10/0.5 + 10/1.0 = 30
        assert_allclose(fs.solution['costs'].item(), 30.0, rtol=1e-5)

    def test_chp_dual_output(self):
        """CHP: fuel = Q_th / eta_th, P_el = fuel * eta_el.
        Revenue from selling electricity reduces total cost."""
        fs, _ = _make_fs(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Elec'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            # Heat demand of 50 each timestep
            fx.Sink(
                'HeatDemand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([50, 50])),
                ],
            ),
            # Electricity sold: sink with revenue on its input flow
            fx.Sink(
                'ElecGrid',
                inputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour=-2),
                ],
            ),
            # Gas at 1€/kWh
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
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
        _solve(fs)
        # Per timestep: fuel = 50/0.5 = 100, elec = 100*0.4 = 40
        # Per timestep cost = 100*1 - 40*2 = 20, total = 2*20 = 40
        assert_allclose(fs.solution['costs'].item(), 40.0, rtol=1e-5)


# ===========================================================================
# Category 2: Storage
# ===========================================================================


class TestStorage:
    def test_storage_shift_saves_money(self):
        """Buy cheap at t=1, discharge at t=2 to avoid expensive purchase."""
        fs, _ = _make_fs(3)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            # Demand only at t=2
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([0, 0, 20])),
                ],
            ),
            # Electricity price varies: [10, 1, 10]
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour=np.array([10, 1, 10])),
                ],
            ),
            fx.Storage(
                'Battery',
                charging=fx.Flow('charge', bus='Elec', size=100),
                discharging=fx.Flow('discharge', bus='Elec', size=100),
                capacity_in_flow_hours=100,
                initial_charge_state=0,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        _solve(fs)
        # Optimal: buy 20 at t=1 @1€ = 20€  (not 20@10€ = 200€)
        assert_allclose(fs.solution['costs'].item(), 20.0, rtol=1e-5)

    def test_storage_losses(self):
        """relative_loss_per_hour reduces available energy."""
        fs, _ = _make_fs(2)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            # Must serve 90 at t=1
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([0, 90])),
                ],
            ),
            # Cheap source only at t=0
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour=np.array([1, 1000])),
                ],
            ),
            fx.Storage(
                'Battery',
                charging=fx.Flow('charge', bus='Elec', size=200),
                discharging=fx.Flow('discharge', bus='Elec', size=200),
                capacity_in_flow_hours=200,
                initial_charge_state=0,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0.1,
            ),
        )
        _solve(fs)
        # Must charge 100 at t=0: after 1h loss = 100*(1-0.1) = 90 available
        # cost = 100 * 1 = 100
        assert_allclose(fs.solution['costs'].item(), 100.0, rtol=1e-5)

    def test_storage_eta_charge_discharge(self):
        """Round-trip efficiency: available = charged * eta_charge * eta_discharge."""
        fs, _ = _make_fs(2)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            # Must serve 72 at t=1
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([0, 72])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour=np.array([1, 1000])),
                ],
            ),
            fx.Storage(
                'Battery',
                charging=fx.Flow('charge', bus='Elec', size=200),
                discharging=fx.Flow('discharge', bus='Elec', size=200),
                capacity_in_flow_hours=200,
                initial_charge_state=0,
                eta_charge=0.9,
                eta_discharge=0.8,
                relative_loss_per_hour=0,
            ),
        )
        _solve(fs)
        # Need 72 out → discharge = 72, stored needed = 72/0.8 = 90
        # charge needed = 90/0.9 = 100 → cost = 100*1 = 100
        assert_allclose(fs.solution['costs'].item(), 100.0, rtol=1e-5)

    def test_storage_soc_bounds(self):
        """relative_maximum_charge_state limits usable capacity.

        Storage has 100 kWh capacity but max SOC = 0.5, so only 50 kWh usable.
        With demand of 60 at t=1, storage can only provide 50 from cheap t=0;
        remaining 10 must come from the expensive source at t=1.
        If SOC bounds were ignored, all 60 could be stored cheaply.
        """
        fs, _ = _make_fs(2)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([0, 60])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour=np.array([1, 100])),
                ],
            ),
            fx.Storage(
                'Battery',
                charging=fx.Flow('charge', bus='Elec', size=200),
                discharging=fx.Flow('discharge', bus='Elec', size=200),
                capacity_in_flow_hours=100,
                initial_charge_state=0,
                relative_maximum_charge_state=0.5,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        _solve(fs)
        # Can store max 50 at t=0 @1€ = 50€. Remaining 10 at t=1 @100€ = 1000€.
        # Total = 1050. Without SOC limit: 60@1€ = 60€ (different!)
        assert_allclose(fs.solution['costs'].item(), 1050.0, rtol=1e-5)


# ===========================================================================
# Category 3: Status (On/Off) Variables
# ===========================================================================


class TestStatusVariables:
    def test_startup_cost(self):
        """effects_per_startup adds cost each time the unit starts."""
        fs, _ = _make_fs(5)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            # demand = [0, 10, 0, 10, 0] → 2 startups
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
        _solve(fs)
        # fuel = (10+10)/0.5 = 40, startups = 2, cost = 40 + 200 = 240
        assert_allclose(fs.solution['costs'].item(), 240.0, rtol=1e-5)

    def test_active_hours_max(self):
        """active_hours_max limits how many timesteps a unit can run."""
        fs, _ = _make_fs(3)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            # demand = [10, 20, 10]
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
            # Cheap boiler, limited to 1 hour
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
            # Expensive backup
            fx.linear_converters.Boiler(
                'ExpensiveBoiler',
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat', size=100),
            ),
        )
        _solve(fs)
        # CheapBoiler runs at t=1 (biggest demand): cost = 20*1 = 20
        # ExpensiveBoiler covers t=0 and t=2: cost = (10+10)/0.5 = 40
        # Total = 60
        assert_allclose(fs.solution['costs'].item(), 60.0, rtol=1e-5)

    def test_min_uptime_forces_operation(self):
        """min_uptime=2 keeps cheap boiler on for 2 consecutive hours.

        demand = [5, 10, 20, 18, 12], same as test_functional.
        Cheap boiler (eta=0.5) with min_uptime=2 and max_uptime=2.
        Expensive backup (eta=0.2).
        The cheap boiler must run in blocks of exactly 2 timesteps.
        Optimal: on at t=0,1 and t=3,4. Off at t=2 → backup covers t=2.

        Without min_uptime, cheap boiler could run only at the 3 cheapest slots.
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
        _solve(fs)
        # Boiler on t=0,1 (block of 2) and t=3,4 (block of 2). Off at t=2 → backup.
        # Boiler fuel: (5+10+18+12)/0.5 = 90. Backup fuel: 20/0.2 = 100. Total = 190.
        assert_allclose(fs.solution['costs'].item(), 190.0, rtol=1e-5)
        assert_allclose(
            fs.solution['Boiler(heat)|status'].values[:-1],
            [1, 1, 0, 1, 1],
            atol=1e-5,
        )

    def test_min_downtime_prevents_restart(self):
        """min_downtime prevents restarting too quickly.

        demand = [20, 0, 20, 0], min_downtime=3, previous_flow_rate=20 (was on).
        Boiler on at t=0 (continuing), turns off at t=1. Must stay off for 3h
        (t=1,2,3). Cannot restart at t=2 → backup covers t=2.
        Without min_downtime, boiler would restart at t=2 (cheaper).
        """
        fs, _ = _make_fs(4)
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
            # Cheap boiler with min_downtime=3
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
            # Expensive backup
            fx.linear_converters.Boiler(
                'Backup',
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat', size=100),
            ),
        )
        _solve(fs)
        # t=0: Boiler on (fuel=20). Turns off at t=1.
        # min_downtime=3: must stay off t=1,2,3. Can't restart at t=2.
        # Backup covers t=2: fuel = 20/0.5 = 40.
        # Without min_downtime: boiler at t=2 (fuel=20), total=40 vs 60.
        assert_allclose(fs.solution['costs'].item(), 60.0, rtol=1e-5)
        # Verify boiler off at t=2 (where demand exists but can't restart)
        assert_allclose(fs.solution['Boiler(heat)|status'].values[2], 0.0, atol=1e-5)


# ===========================================================================
# Category 4: Piecewise Linearization
# ===========================================================================


class TestPiecewise:
    def test_piecewise_selects_cheap_segment(self):
        """Optimizer picks the segment with lower fuel cost.

        A LinearConverter with 2-segment piecewise conversion:
        - Segment 1 (low load):  fuel 10→30, heat 5→15  (ratio 2:1)
        - Segment 2 (high load): fuel 30→100, heat 15→60 (ratio 70/45 ≈ 1.56:1)
        High-load segment is more efficient. With demand=45 (within segment 2),
        the optimizer should use segment 2.
        fuel at heat=45: linear interp in seg2: 30 + (45-15)/(60-15)*(100-30) = 30 + 30/45*70 ≈ 76.67
        """
        fs, _ = _make_fs(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([45, 45])),
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
                inputs=[fx.Flow('fuel', bus='Gas')],
                outputs=[fx.Flow('heat', bus='Heat')],
                piecewise_conversion=fx.PiecewiseConversion(
                    {
                        'fuel': fx.Piecewise([fx.Piece(10, 30), fx.Piece(30, 100)]),
                        'heat': fx.Piecewise([fx.Piece(5, 15), fx.Piece(15, 60)]),
                    }
                ),
            ),
        )
        _solve(fs)
        # heat=45 in segment 2: fuel = 30 + (45-15)/(60-15) * (100-30) = 30 + 46.667 = 76.667
        # cost per timestep = 76.667, total = 2 * 76.667 ≈ 153.333
        assert_allclose(fs.solution['costs'].item(), 2 * (30 + 30 / 45 * 70), rtol=1e-4)

    def test_piecewise_conversion_at_breakpoint(self):
        """Flow ratios match segment definition at a breakpoint.

        Force operation exactly at the boundary between segments.
        Demand = 15 = end of segment 1 = start of segment 2.
        fuel at heat=15: segment 1 end → fuel=30, segment 2 start → fuel=30.
        Both agree: fuel=30.
        """
        fs, _ = _make_fs(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([15, 15])),
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
                inputs=[fx.Flow('fuel', bus='Gas')],
                outputs=[fx.Flow('heat', bus='Heat')],
                piecewise_conversion=fx.PiecewiseConversion(
                    {
                        'fuel': fx.Piecewise([fx.Piece(10, 30), fx.Piece(30, 100)]),
                        'heat': fx.Piecewise([fx.Piece(5, 15), fx.Piece(15, 60)]),
                    }
                ),
            ),
        )
        _solve(fs)
        # At breakpoint: fuel = 30 per timestep, total = 60
        assert_allclose(fs.solution['costs'].item(), 60.0, rtol=1e-5)
        # Verify fuel flow rate
        assert_allclose(fs.solution['Converter(fuel)|flow_rate'].values[0], 30.0, rtol=1e-5)


# ===========================================================================
# Category 5: Investment
# ===========================================================================


class TestInvestment:
    def test_invest_size_optimized(self):
        """Optimal investment size = peak demand."""
        fs, _ = _make_fs(3)
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
        _solve(fs)
        # size = 50 (peak), invest cost = 10 + 50*1 = 60, fuel = 80
        # total = 140
        assert_allclose(fs.solution['Boiler(heat)|size'].item(), 50.0, rtol=1e-5)
        assert_allclose(fs.solution['costs'].item(), 140.0, rtol=1e-5)

    def test_invest_optional_not_built(self):
        """Optional invest skipped when investment cost outweighs fuel savings.

        The invest boiler has better efficiency (1.0 vs 0.5) but high fixed
        investment cost (99999). If the investment mechanism were broken and
        allowed free investment, the optimizer would use the invest boiler
        (fuel=20) instead of the cheap boiler (fuel=40), changing the objective.
        """
        fs, _ = _make_fs(2)
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
            # High-efficiency boiler with prohibitive investment cost
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
            # Low-efficiency boiler always available (no invest needed)
            fx.linear_converters.Boiler(
                'CheapBoiler',
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat', size=100),
            ),
        )
        _solve(fs)
        assert_allclose(fs.solution['InvestBoiler(heat)|invested'].item(), 0.0, atol=1e-5)
        # All demand served by CheapBoiler: fuel = 20/0.5 = 40
        # If invest were free, InvestBoiler would run: fuel = 20/1.0 = 20 (different!)
        assert_allclose(fs.solution['costs'].item(), 40.0, rtol=1e-5)

    def test_invest_minimum_size(self):
        """minimum_size forces oversized investment."""
        fs, _ = _make_fs(2)
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
        _solve(fs)
        # Must invest at least 100, cost_per_size=1 → invest=100
        assert_allclose(fs.solution['Boiler(heat)|size'].item(), 100.0, rtol=1e-5)
        # fuel=20, invest=100 → total=120
        assert_allclose(fs.solution['costs'].item(), 120.0, rtol=1e-5)


# ===========================================================================
# Category 5: Effects & Objective
# ===========================================================================


class TestEffects:
    def test_effects_per_flow_hour(self):
        """effects_per_flow_hour accumulates correctly for multiple effects."""
        fs, _ = _make_fs(2)
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
        _solve(fs)
        # costs = (10+20)*2 = 60, CO2 = (10+20)*0.5 = 15
        assert_allclose(fs.solution['costs'].item(), 60.0, rtol=1e-5)
        assert_allclose(fs.solution['CO2'].item(), 15.0, rtol=1e-5)

    def test_share_from_temporal(self):
        """share_from_temporal adds a fraction of one effect to another."""
        fs, _ = _make_fs(2)
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
        _solve(fs)
        # direct costs = 20*1 = 20, CO2 = 20*10 = 200
        # costs += 0.5 * CO2_temporal = 0.5 * 200 = 100
        # total costs = 20 + 100 = 120
        assert_allclose(fs.solution['costs'].item(), 120.0, rtol=1e-5)
        assert_allclose(fs.solution['CO2'].item(), 200.0, rtol=1e-5)

    def test_effect_maximum_total(self):
        """maximum_total on an effect forces suboptimal dispatch."""
        fs, _ = _make_fs(2)
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
            # Cheap but high CO2
            fx.Source(
                'Dirty',
                outputs=[
                    fx.Flow('heat', bus='Heat', effects_per_flow_hour={'costs': 1, 'CO2': 1}),
                ],
            ),
            # Expensive but no CO2
            fx.Source(
                'Clean',
                outputs=[
                    fx.Flow('heat', bus='Heat', effects_per_flow_hour={'costs': 10, 'CO2': 0}),
                ],
            ),
        )
        _solve(fs)
        # Without CO2 limit: all from Dirty = 20€
        # With CO2 max=15: 15 from Dirty (15€), 5 from Clean (50€) → total 65€
        assert_allclose(fs.solution['costs'].item(), 65.0, rtol=1e-5)
        assert_allclose(fs.solution['CO2'].item(), 15.0, rtol=1e-5)


# ===========================================================================
# Category 6: Bus Balance
# ===========================================================================


class TestBusBalance:
    def test_merit_order_dispatch(self):
        """Cheap source is maxed out before expensive source is used.

        With no imbalance allowed, the bus balance constraint forces
        total supply = demand. The cost structure (1 vs 2 €/kWh) and
        capacity limit (20) on Src1 uniquely determine the dispatch split.
        If bus balance were broken, feasibility or cost would change.
        """
        fs, _ = _make_fs(2)
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
        _solve(fs)
        # Src1 at max 20 @1€, Src2 covers remaining 10 @2€
        # cost = 2*(20*1 + 10*2) = 80
        assert_allclose(fs.solution['costs'].item(), 80.0, rtol=1e-5)
        # Verify individual flows to confirm dispatch split
        src1 = fs.solution['Src1(heat)|flow_rate'].values[:-1]
        src2 = fs.solution['Src2(heat)|flow_rate'].values[:-1]
        assert_allclose(src1, [20, 20], rtol=1e-5)
        assert_allclose(src2, [10, 10], rtol=1e-5)

    def test_imbalance_penalty(self):
        """Excess supply is penalized via imbalance_penalty_per_flow_hour."""
        fs, _ = _make_fs(2)
        fs.add_elements(
            fx.Bus('Heat', imbalance_penalty_per_flow_hour=100),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            # Demand = 10 each timestep
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([10, 10])),
                ],
            ),
            # Source forced to produce exactly 20 each timestep
            fx.Source(
                'Src',
                outputs=[
                    fx.Flow(
                        'heat', bus='Heat', size=1, fixed_relative_profile=np.array([20, 20]), effects_per_flow_hour=1
                    ),
                ],
            ),
        )
        _solve(fs)
        # Each timestep: source=20, demand=10, excess=10
        # fuel = 2*20*1 = 40, penalty = 2*10*100 = 2000
        # Penalty goes to separate 'Penalty' effect, not 'costs'
        assert_allclose(fs.solution['costs'].item(), 40.0, rtol=1e-5)
        assert_allclose(fs.solution['Penalty'].item(), 2000.0, rtol=1e-5)
        assert_allclose(fs.solution['objective'].item(), 2040.0, rtol=1e-5)
