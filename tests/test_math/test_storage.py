"""Mathematical correctness tests for storage."""

import numpy as np
from numpy.testing import assert_allclose

import flixopt as fx

from .conftest import make_flow_system


class TestStorage:
    def test_storage_shift_saves_money(self, solve):
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
        solve(fs)
        # Optimal: buy 20 at t=1 @1€ = 20€  (not 20@10€ = 200€)
        assert_allclose(fs.solution['costs'].item(), 20.0, rtol=1e-5)

    def test_storage_losses(self, solve):
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
        solve(fs)
        # Must charge 100 at t=0: after 1h loss = 100*(1-0.1) = 90 available
        # cost = 100 * 1 = 100
        assert_allclose(fs.solution['costs'].item(), 100.0, rtol=1e-5)

    def test_storage_eta_charge_discharge(self, solve):
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
        solve(fs)
        # Need 72 out → discharge = 72, stored needed = 72/0.8 = 90
        # charge needed = 90/0.9 = 100 → cost = 100*1 = 100
        assert_allclose(fs.solution['costs'].item(), 100.0, rtol=1e-5)

    def test_storage_soc_bounds(self, solve):
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
        solve(fs)
        # Can store max 50 at t=0 @1€ = 50€. Remaining 10 at t=1 @100€ = 1000€.
        # Total = 1050. Without SOC limit: 60@1€ = 60€ (different!)
        assert_allclose(fs.solution['costs'].item(), 1050.0, rtol=1e-5)

    def test_storage_cyclic_charge_state(self, solve):
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
        solve(fs)
        # Charge 50 at t=0 @1€, discharge 50 at t=1. Final = initial (cyclic).
        # cost = 50*1 = 50
        assert_allclose(fs.solution['costs'].item(), 50.0, rtol=1e-5)

    def test_storage_minimal_final_charge_state(self, solve):
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
        solve(fs)
        # Charge 80 at t=0 @1€, discharge 20 at t=1. Final SOC=60. cost=80.
        assert_allclose(fs.solution['costs'].item(), 80.0, rtol=1e-5)

    def test_storage_invest_capacity(self, solve):
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
        solve(fs)
        # Invest 50 kWh @1€/kWh = 50€. Buy 50 at t=0 @1€ = 50€. Total = 100€.
        # Without storage: buy 50 at t=1 @10€ = 500€.
        assert_allclose(fs.solution['Battery|size'].item(), 50.0, rtol=1e-5)
        assert_allclose(fs.solution['costs'].item(), 100.0, rtol=1e-5)

    def test_prevent_simultaneous_charge_and_discharge(self, solve):
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
        solve(fs)
        charge = fs.solution['Battery(charge)|flow_rate'].values[:-1]
        discharge = fs.solution['Battery(discharge)|flow_rate'].values[:-1]
        # At no timestep should both be > 0
        for t in range(len(charge)):
            assert not (charge[t] > 1e-5 and discharge[t] > 1e-5), (
                f'Simultaneous charge/discharge at t={t}: charge={charge[t]}, discharge={discharge[t]}'
            )
