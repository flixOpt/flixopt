"""Regression tests for cluster → optimize → expand numerical equivalence.

For a sinusoidal demand around a constant mean, storage flattens the dispatch
in the clustered solve, and expansion repeats those flat values per cluster.
The expected post-expansion totals are therefore derivable from the fixture
parameters (mean demand, boiler efficiency, gas cost). Computing them
analytically keeps the assertions tight without hardcoding magic numbers.
"""

import numpy as np
import pandas as pd
import pytest

import flixopt as fx

tsam = pytest.importorskip('tsam')

# Fixture parameters — single source of truth for derived reference values
N_HOURS = 192  # 8 days
MEAN_DEMAND = 15.0  # demand = sin(...) * 10 + MEAN_DEMAND, sin term averages to 0
BOILER_ETA = 0.9
GAS_PRICE = 0.05


@pytest.fixture
def system_with_storage():
    """System with storage (tests charge_state) and effects (tests segment totals)."""
    ts = pd.date_range('2020-01-01', periods=N_HOURS, freq='h')
    demand = np.sin(np.linspace(0, 16 * np.pi, N_HOURS)) * 10 + MEAN_DEMAND

    fs = fx.FlowSystem(ts)
    fs.add_elements(
        fx.Bus('Heat'),
        fx.Bus('Gas'),
        fx.Effect('costs', '€', is_standard=True, is_objective=True),
        fx.Sink('D', inputs=[fx.Flow('Q', bus='Heat', fixed_relative_profile=demand, size=1)]),
        fx.Source('G', outputs=[fx.Flow('Gas', bus='Gas', effects_per_flow_hour=GAS_PRICE)]),
        fx.linear_converters.Boiler(
            'B',
            thermal_efficiency=BOILER_ETA,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow('Q_th', bus='Heat'),
        ),
        fx.Storage(
            'S',
            capacity_in_flow_hours=50,
            initial_charge_state=0.5,
            charging=fx.Flow('in', bus='Heat', size=10),
            discharging=fx.Flow('out', bus='Heat', size=10),
        ),
    )
    return fs


# Derived expected totals — when storage flattens dispatch to the mean
EXPECTED_HEAT_SUM = MEAN_DEMAND * N_HOURS  # boiler thermal output = demand
EXPECTED_GAS_SUM = EXPECTED_HEAT_SUM / BOILER_ETA
EXPECTED_COSTS = EXPECTED_GAS_SUM * GAS_PRICE


class TestNonSegmentedExpansion:
    """Test that non-segmented cluster → expand produces correct values."""

    def test_expanded_objective_matches(self, system_with_storage, solver_fixture):
        fs_c = system_with_storage.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        assert fs_e.solution['objective'].item() == pytest.approx(EXPECTED_COSTS, rel=1e-6)

    def test_expanded_flow_rates(self, system_with_storage, solver_fixture):
        fs_c = system_with_storage.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        sol = fs_e.solution
        assert float(np.nansum(sol['flow|rate'].sel(flow='B(Q_th)').values)) == pytest.approx(
            EXPECTED_HEAT_SUM, rel=1e-6
        )
        assert float(np.nansum(sol['flow|rate'].sel(flow='D(Q)').values)) == pytest.approx(EXPECTED_HEAT_SUM, rel=1e-6)
        assert float(np.nansum(sol['flow|rate'].sel(flow='G(Gas)').values)) == pytest.approx(EXPECTED_GAS_SUM, rel=1e-6)

    def test_expanded_costs(self, system_with_storage, solver_fixture):
        fs_c = system_with_storage.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        sol = fs_e.solution
        assert float(np.nansum(sol['effect|per_timestep'].sel(effect='costs').values)) == pytest.approx(
            EXPECTED_COSTS, rel=1e-6
        )
        assert float(
            np.nansum(sol['share|temporal'].sel(contributor='G(Gas)', effect='costs').values)
        ) == pytest.approx(EXPECTED_COSTS, rel=1e-6)

    def test_expanded_storage(self, system_with_storage, solver_fixture):
        fs_c = system_with_storage.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        sol = fs_e.solution
        # The inter-cluster SOC level is unpinned, so its absolute value is degenerate
        # and solver-dependent — an all-zero SOC is a valid optimum. Assert the
        # determinate invariants instead (see issue #733).
        cs = sol['intercluster_storage|charge_state'].sel(intercluster_storage='S').values
        assert np.all(np.isfinite(cs))
        assert float(np.nanmin(cs)) >= -1e-5
        # Net discharge should be ~0 (balanced storage)
        assert float(
            np.nansum(sol['intercluster_storage|netto_discharge'].sel(intercluster_storage='S').values)
        ) == pytest.approx(0, abs=1e-4)

    def test_expanded_shapes(self, system_with_storage, solver_fixture):
        fs_c = system_with_storage.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        sol = fs_e.solution
        # 192 original timesteps + 1 extra boundary = 193
        for name in sol.data_vars:
            if 'time' in sol[name].dims:
                assert sol[name].sizes['time'] == N_HOURS + 1, f'{name} has wrong time size'


class TestSegmentedExpansion:
    """Test that segmented cluster → expand produces correct values."""

    def test_expanded_objective_matches(self, system_with_storage, solver_fixture):
        fs_c = system_with_storage.transform.cluster(
            n_clusters=2, cluster_duration='1D', segments=tsam.SegmentConfig(n_segments=6)
        )
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        assert fs_e.solution['objective'].item() == pytest.approx(EXPECTED_COSTS, rel=1e-6)

    def test_expanded_flow_rates(self, system_with_storage, solver_fixture):
        fs_c = system_with_storage.transform.cluster(
            n_clusters=2, cluster_duration='1D', segments=tsam.SegmentConfig(n_segments=6)
        )
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        sol = fs_e.solution
        assert float(np.nansum(sol['flow|rate'].sel(flow='B(Q_th)').values)) == pytest.approx(
            EXPECTED_HEAT_SUM, rel=1e-6
        )
        assert float(np.nansum(sol['flow|rate'].sel(flow='D(Q)').values)) == pytest.approx(EXPECTED_HEAT_SUM, rel=1e-6)
        assert float(np.nansum(sol['flow|rate'].sel(flow='G(Gas)').values)) == pytest.approx(EXPECTED_GAS_SUM, rel=1e-6)

    def test_expanded_costs(self, system_with_storage, solver_fixture):
        fs_c = system_with_storage.transform.cluster(
            n_clusters=2, cluster_duration='1D', segments=tsam.SegmentConfig(n_segments=6)
        )
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        sol = fs_e.solution
        assert float(np.nansum(sol['effect|per_timestep'].sel(effect='costs').values)) == pytest.approx(
            EXPECTED_COSTS, rel=1e-6
        )
        assert float(
            np.nansum(sol['share|temporal'].sel(contributor='G(Gas)', effect='costs').values)
        ) == pytest.approx(EXPECTED_COSTS, rel=1e-6)

    def test_expanded_shapes(self, system_with_storage, solver_fixture):
        fs_c = system_with_storage.transform.cluster(
            n_clusters=2, cluster_duration='1D', segments=tsam.SegmentConfig(n_segments=6)
        )
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        sol = fs_e.solution
        for name in sol.data_vars:
            if 'time' in sol[name].dims:
                assert sol[name].sizes['time'] == N_HOURS + 1, f'{name} has wrong time size'

    def test_no_nans_in_expanded_flow_rates(self, system_with_storage, solver_fixture):
        """Segmented expansion must ffill — no NaNs in flow rates (except extra boundary)."""
        fs_c = system_with_storage.transform.cluster(
            n_clusters=2, cluster_duration='1D', segments=tsam.SegmentConfig(n_segments=6)
        )
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        sol = fs_e.solution
        for flow in ['B(Q_th)', 'D(Q)', 'G(Gas)']:
            # Exclude last timestep (extra boundary, may be NaN for non-state variables)
            name = flow
            vals = sol['flow|rate'].sel(flow=flow).isel(time=slice(None, -1))
            assert not vals.isnull().any(), f'{name} has NaN values after expansion'
