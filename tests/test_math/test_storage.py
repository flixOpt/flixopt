"""Mathematical correctness tests for storage."""

import numpy as np
from numpy.testing import assert_allclose

import flixopt as fx
from flixopt import InvestParameters

from .conftest import make_flow_system


class TestStorage:
    def test_storage_shift_saves_money(self, optimize):
        """Proves: Storage enables temporal arbitrage — charge cheap, discharge when expensive.

        Sensitivity: Without storage, demand at t=2 must be bought at 10€/kWh → cost=200.
        With working storage, buy at t=1 for 1€/kWh → cost=20. A 10× difference.
        """
        fs = make_flow_system(3)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([0, 0, 20])),
                ],
            ),
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
        fs = optimize(fs)
        # Optimal: buy 20 at t=1 @1€ = 20€  (not 20@10€ = 200€)
        assert_allclose(fs.solution['costs'].item(), 20.0, rtol=1e-5)

    def test_storage_losses(self, optimize):
        """Proves: relative_loss_per_hour correctly reduces stored energy over time.

        Sensitivity: If losses were ignored (0%), only 90 would be charged → cost=90.
        With 10% loss, must charge 100 to have 90 after 1h → cost=100.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([0, 90])),
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
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0.1,
            ),
        )
        fs = optimize(fs)
        # Must charge 100 at t=0: after 1h loss = 100*(1-0.1) = 90 available
        # cost = 100 * 1 = 100
        assert_allclose(fs.solution['costs'].item(), 100.0, rtol=1e-5)

    def test_storage_eta_charge_discharge(self, optimize):
        """Proves: eta_charge and eta_discharge are both applied to the energy flow.
        Stored = charged * eta_charge; discharged = stored * eta_discharge.

        Sensitivity: If eta_charge broken (1.0), cost=90. If eta_discharge broken (1.0),
        cost=80. If both broken, cost=72. Only both correct yields cost=100.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
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
        fs = optimize(fs)
        # Need 72 out → discharge = 72, stored needed = 72/0.8 = 90
        # charge needed = 90/0.9 = 100 → cost = 100*1 = 100
        assert_allclose(fs.solution['costs'].item(), 100.0, rtol=1e-5)

    def test_storage_soc_bounds(self, optimize):
        """Proves: relative_maximum_charge_state caps how much energy can be stored.

        Storage has 100 kWh capacity but max SOC = 0.5 → only 50 kWh usable.
        Demand of 60 at t=1: storage provides 50 from cheap t=0, remaining 10
        from the expensive source at t=1.

        Sensitivity: If SOC bound were ignored, all 60 stored cheaply → cost=60.
        With the bound enforced, cost=1050 (50×1 + 10×100).
        """
        fs = make_flow_system(2)
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
        fs = optimize(fs)
        # Can store max 50 at t=0 @1€ = 50€. Remaining 10 at t=1 @100€ = 1000€.
        # Total = 1050. Without SOC limit: 60@1€ = 60€ (different!)
        assert_allclose(fs.solution['costs'].item(), 1050.0, rtol=1e-5)

    def test_storage_cyclic_charge_state(self, optimize):
        """Proves: initial_charge_state='equals_final' forces the storage to end at the
        same level it started, preventing free energy extraction.

        Price=[1,100]. Demand=[0,50]. Without cyclic constraint, storage starts full
        (initial=50) and discharges for free. With cyclic, must recharge what was used.

        Sensitivity: Without cyclic, initial_charge_state=50 gives 50 free energy.
        With cyclic, must buy 50 at some point to replenish → cost=50.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([0, 50])),
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
                initial_charge_state='equals_final',
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        # Charge 50 at t=0 @1€, discharge 50 at t=1. Final = initial (cyclic).
        # cost = 50*1 = 50
        assert_allclose(fs.solution['costs'].item(), 50.0, rtol=1e-5)

    def test_storage_minimal_final_charge_state(self, optimize):
        """Proves: minimal_final_charge_state forces the storage to retain at least the
        specified absolute energy at the end, even when discharging would be profitable.

        Storage capacity=100, initial=0, minimal_final=60. Price=[1,100].
        Demand=[0,20]. Must charge ≥80 at t=0 (20 for demand + 60 for final).

        Sensitivity: Without final constraint, charge only 20 → cost=20.
        With minimal_final=60, charge 80 → cost=80.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([0, 20])),
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
                minimal_final_charge_state=60,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        # Charge 80 at t=0 @1€, discharge 20 at t=1. Final SOC=60. cost=80.
        assert_allclose(fs.solution['costs'].item(), 80.0, rtol=1e-5)

    def test_storage_invest_capacity(self, optimize):
        """Proves: InvestParameters on capacity_in_flow_hours correctly sizes the storage.
        The optimizer balances investment cost against operational savings.

        invest_per_size=1€/kWh. Price=[1,10]. Demand=[0,50]. Storage saves 9€/kWh
        shifted but costs 1€/kWh invested. Net saving=8€/kWh → invest all 50.

        Sensitivity: If invest cost were 100€/kWh (>9 saving), no storage built → cost=500.
        At 1€/kWh, storage built → cost=50*1 (buy) + 50*1 (invest) = 100.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([0, 50])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour=np.array([1, 10])),
                ],
            ),
            fx.Storage(
                'Battery',
                charging=fx.Flow('charge', bus='Elec', size=200),
                discharging=fx.Flow('discharge', bus='Elec', size=200),
                capacity_in_flow_hours=fx.InvestParameters(
                    maximum_size=200,
                    effects_of_investment_per_size=1,
                ),
                initial_charge_state=0,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        # Invest 50 kWh @1€/kWh = 50€. Buy 50 at t=0 @1€ = 50€. Total = 100€.
        # Without storage: buy 50 at t=1 @10€ = 500€.
        assert_allclose(fs.solution['Battery|size'].item(), 50.0, rtol=1e-5)
        assert_allclose(fs.solution['costs'].item(), 100.0, rtol=1e-5)

    def test_prevent_simultaneous_charge_and_discharge(self, optimize):
        """Proves: prevent_simultaneous_charge_and_discharge=True prevents the storage
        from charging and discharging in the same timestep.

        Without this constraint, a storage with eta_charge=0.9, eta_discharge=0.9
        and a generous imbalance penalty could exploit simultaneous charge/discharge
        to game the bus balance. With the constraint, charge and discharge flows
        are mutually exclusive per timestep.

        Setup: Source at 1€/kWh, demand=10 at every timestep. Storage with
        prevent_simultaneous=True. Verify that at no timestep both charge>0 and
        discharge>0.

        Sensitivity: This is a structural constraint. If broken, the optimizer
        could charge and discharge simultaneously, which is physically nonsensical.
        """
        fs = make_flow_system(3)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([10, 20, 10])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour=np.array([1, 10, 1])),
                ],
            ),
            fx.Storage(
                'Battery',
                charging=fx.Flow('charge', bus='Elec', size=100),
                discharging=fx.Flow('discharge', bus='Elec', size=100),
                capacity_in_flow_hours=100,
                initial_charge_state=0,
                eta_charge=0.9,
                eta_discharge=0.9,
                relative_loss_per_hour=0,
                prevent_simultaneous_charge_and_discharge=True,
            ),
        )
        fs = optimize(fs)
        charge = fs.solution['Battery(charge)|flow_rate'].values[:-1]
        discharge = fs.solution['Battery(discharge)|flow_rate'].values[:-1]
        # At no timestep should both be > 0
        for t in range(len(charge)):
            assert not (charge[t] > 1e-5 and discharge[t] > 1e-5), (
                f'Simultaneous charge/discharge at t={t}: charge={charge[t]}, discharge={discharge[t]}'
            )

    def test_storage_relative_minimum_charge_state(self, optimize):
        """Proves: relative_minimum_charge_state enforces a minimum SOC at all times.

        Storage capacity=100, initial=50, relative_minimum_charge_state=0.3.
        Grid prices=[1,100,1]. Demand=[0,80,0].
        SOC must stay >= 30 at all times. SOC starts at 50.
        @t0: charge 50 more → SOC=100. @t1: discharge 70 → SOC=30 (exactly min).
        Grid covers remaining 10 @t1 at price 100.

        Sensitivity: Without min SOC, discharge all 100 → no grid → cost=50.
        With min SOC=0.3, max discharge=70 → grid covers 10 @100€ → cost=1050.
        """
        fs = make_flow_system(3)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([0, 80, 0])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour=np.array([1, 100, 1])),
                ],
            ),
            fx.Storage(
                'Battery',
                charging=fx.Flow('charge', bus='Elec', size=200),
                discharging=fx.Flow('discharge', bus='Elec', size=200),
                capacity_in_flow_hours=100,
                initial_charge_state=50,
                relative_minimum_charge_state=0.3,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        # @t0: charge 50 → SOC=100. Cost=50*1=50.
        # @t1: discharge 70 → SOC=30 (min). Grid covers 10 @100=1000. Cost=1050.
        # Total = 1050. Without min SOC: charge 30 @t0 → SOC=80, discharge 80 @t1 → cost=30.
        assert_allclose(fs.solution['costs'].item(), 1050.0, rtol=1e-5)

    def test_storage_maximal_final_charge_state(self, optimize):
        """Proves: maximal_final_charge_state caps the storage level at the end,
        forcing discharge even when not needed by demand.

        Storage capacity=100, initial=80, maximal_final_charge_state=20.
        Demand=[50, 0]. Grid @[100, 1]. imbalance_penalty=5 to absorb excess.
        Without max final: discharge 50 @t0, final=30. objective=0 (no grid, no penalty).
        With max final=20: discharge 60, excess 10 penalized @5. objective=50.

        Sensitivity: Without max final, objective=0. With max final=20, objective=50.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Elec', imbalance_penalty_per_flow_hour=5),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([50, 0])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour=np.array([100, 1])),
                ],
            ),
            fx.Storage(
                'Battery',
                charging=fx.Flow('charge', bus='Elec', size=200),
                discharging=fx.Flow('discharge', bus='Elec', size=200),
                capacity_in_flow_hours=100,
                initial_charge_state=80,
                maximal_final_charge_state=20,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        # Discharge 60, excess 10 penalized @5 → penalty=50. Objective=50.
        assert_allclose(fs.solution['objective'].item(), 50.0, rtol=1e-5)

    def test_storage_relative_minimum_final_charge_state(self, optimize):
        """Proves: relative_minimum_final_charge_state forces a minimum final SOC
        as a fraction of capacity.

        Storage capacity=100, initial=50. Demand=[0, 80]. Grid @[1, 100].
        relative_minimum_charge_state=0 (time-varying), relative_min_final=0.5.
        Without final constraint: charge 30 @t0 (cost=30), SOC=80, discharge 80 @t1.
        With relative_min_final=0.5: final SOC >= 50. @t0 charge 50 → SOC=100.
        @t1 discharge 50, grid covers 30 @100€.

        Sensitivity: Without constraint, cost=30. With min final=0.5, cost=3050.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([0, 80])),
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
                initial_charge_state=50,
                relative_minimum_charge_state=np.array([0, 0]),
                relative_minimum_final_charge_state=0.5,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        # @t0: charge 50 → SOC=100. Cost=50.
        # @t1: discharge 50 → SOC=50 (min final). Grid covers 30 @100€=3000€.
        # Total = 3050. Without min final: charge 30 @1€ → discharge 80 → cost=30.
        assert_allclose(fs.solution['costs'].item(), 3050.0, rtol=1e-5)

    def test_storage_relative_maximum_final_charge_state(self, optimize):
        """Proves: relative_maximum_final_charge_state caps the storage at end
        as a fraction of capacity. Same logic as maximal_final but relative.

        Storage capacity=100, initial=80, relative_maximum_final_charge_state=0.2.
        Equivalent to maximal_final_charge_state=20.
        Demand=[50, 0]. Grid @[100, 1]. imbalance_penalty=5.
        relative_maximum_charge_state=1.0 (time-varying) for proper final override.

        Sensitivity: Without max final, discharge 50 → final=30. objective=0.
        With relative_max_final=0.2 (=20 abs), must discharge 60 → excess 10 * 5€ = 50€.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Elec', imbalance_penalty_per_flow_hour=5),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([50, 0])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour=np.array([100, 1])),
                ],
            ),
            fx.Storage(
                'Battery',
                charging=fx.Flow('charge', bus='Elec', size=200),
                discharging=fx.Flow('discharge', bus='Elec', size=200),
                capacity_in_flow_hours=100,
                initial_charge_state=80,
                relative_maximum_charge_state=np.array([1.0, 1.0]),
                relative_maximum_final_charge_state=0.2,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        # Discharge 60, excess 10 penalized @5 → penalty=50. Objective=50.
        assert_allclose(fs.solution['objective'].item(), 50.0, rtol=1e-5)

    def test_storage_relative_minimum_final_charge_state_scalar(self, optimize):
        """Proves: relative_minimum_final_charge_state works when relative_minimum_charge_state
        is a scalar (default=0, no time dimension).

        Same scenario as test_storage_relative_minimum_final_charge_state but using
        scalar defaults instead of arrays — this was previously a bug where the scalar
        branch ignored the final override entirely.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([0, 80])),
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
                initial_charge_state=50,
                relative_minimum_final_charge_state=0.5,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['costs'].item(), 3050.0, rtol=1e-5)

    def test_storage_relative_maximum_final_charge_state_scalar(self, optimize):
        """Proves: relative_maximum_final_charge_state works when relative_maximum_charge_state
        is a scalar (default=1, no time dimension).

        Same scenario as test_storage_relative_maximum_final_charge_state but using
        scalar defaults instead of arrays — this was previously a bug where the scalar
        branch ignored the final override entirely.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Elec', imbalance_penalty_per_flow_hour=5),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([50, 0])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour=np.array([100, 1])),
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
        assert_allclose(fs.solution['objective'].item(), 50.0, rtol=1e-5)

    def test_storage_balanced_invest(self, optimize):
        """Proves: balanced=True forces charge and discharge invest sizes to be equal.

        Storage with InvestParameters on charge and discharge flows.
        Grid prices=[1, 100, 100]. Demand=[0, 80, 80].
        Without balanced, discharge_size could be 80 (minimum needed), charge_size=160.
        With balanced, both sizes must equal → invest size = 160.

        Sensitivity: Without balanced, invest=80+160=240, ops=160.
        With balanced, invest=160+160=320, ops=160.
        """
        fs = make_flow_system(3)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([0, 80, 80])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour=np.array([1, 100, 100])),
                ],
            ),
            fx.Storage(
                'Battery',
                charging=fx.Flow(
                    'charge',
                    bus='Elec',
                    size=InvestParameters(maximum_size=200, effects_of_investment_per_size=1),
                ),
                discharging=fx.Flow(
                    'discharge',
                    bus='Elec',
                    size=InvestParameters(maximum_size=200, effects_of_investment_per_size=1),
                ),
                capacity_in_flow_hours=200,
                initial_charge_state=0,
                balanced=True,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        # With balanced: charge_size = discharge_size = 160.
        # Charge 160 @t0 @1€ = 160€. Discharge 80 @t1, 80 @t2. Invest 160+160=320€.
        # But wait — we need to think about this more carefully.
        # @t0: charge 160 (max rate). @t1: discharge 80. @t2: discharge 80. SOC: 0→160→80→0.
        # Invest: charge_size=160 @1€ = 160€. discharge_size=160 @1€ = 160€. Total invest=320€.
        # Ops: 160 @1€ = 160€. Total = 480€.
        # Without balanced: charge_size=160, discharge_size=80 → invest 240, ops 160 → 400€.
        charge_size = fs.solution['Battery(charge)|size'].item()
        discharge_size = fs.solution['Battery(discharge)|size'].item()
        assert_allclose(charge_size, discharge_size, rtol=1e-5)
        # With balanced, total cost is higher than without
        assert fs.solution['costs'].item() > 400.0 - 1e-5
