"""Regression tests for cluster → optimize → expand numerical equivalence.

These tests verify that the expanded solution values match known reference
values, catching any changes in the clustering/expansion pipeline.
"""

import numpy as np
import pandas as pd
import pytest

import flixopt as fx

tsam = pytest.importorskip('tsam')


@pytest.fixture
def system_with_storage():
    """System with storage (tests charge_state) and effects (tests segment totals)."""
    ts = pd.date_range('2020-01-01', periods=192, freq='h')  # 8 days
    demand = np.sin(np.linspace(0, 16 * np.pi, 192)) * 10 + 15

    fs = fx.FlowSystem(ts)
    fs.add_elements(
        fx.Bus('Heat'),
        fx.Bus('Gas'),
        fx.Effect('costs', '€', is_standard=True, is_objective=True),
        fx.Sink('D', inputs=[fx.Flow('Q', bus='Heat', fixed_relative_profile=demand, size=1)]),
        fx.Source('G', outputs=[fx.Flow('Gas', bus='Gas', effects_per_flow_hour=0.05)]),
        fx.linear_converters.Boiler(
            'B',
            thermal_efficiency=0.9,
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


class TestNonSegmentedExpansion:
    """Test that non-segmented cluster → expand produces correct values."""

    def test_expanded_objective_matches(self, system_with_storage, solver_fixture):
        fs_c = system_with_storage.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        assert fs_e.solution['objective'].item() == pytest.approx(160.0, abs=1e-6)

    def test_expanded_flow_rates(self, system_with_storage, solver_fixture):
        fs_c = system_with_storage.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        sol = fs_e.solution
        assert float(np.nansum(sol['B(Q_th)|flow_rate'].values)) == pytest.approx(2880.0, abs=1e-6)
        assert float(np.nansum(sol['D(Q)|flow_rate'].values)) == pytest.approx(2880.0, abs=1e-6)
        assert float(np.nansum(sol['G(Gas)|flow_rate'].values)) == pytest.approx(3200.0, abs=1e-6)

    def test_expanded_costs(self, system_with_storage, solver_fixture):
        fs_c = system_with_storage.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        sol = fs_e.solution
        assert float(np.nansum(sol['costs(temporal)|per_timestep'].values)) == pytest.approx(160.0, abs=1e-6)
        assert float(np.nansum(sol['G(Gas)->costs(temporal)'].values)) == pytest.approx(160.0, abs=1e-6)

    def test_expanded_storage(self, system_with_storage, solver_fixture):
        fs_c = system_with_storage.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        sol = fs_e.solution
        # Storage dispatch varies by solver — check charge_state is non-trivial
        assert float(np.nansum(sol['S|charge_state'].values)) > 0
        # Net discharge should be ~0 (balanced storage)
        assert float(np.nansum(sol['S|netto_discharge'].values)) == pytest.approx(0, abs=1e-4)

    def test_expanded_shapes(self, system_with_storage, solver_fixture):
        fs_c = system_with_storage.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        sol = fs_e.solution
        # 192 original timesteps + 1 extra boundary = 193
        for name in sol.data_vars:
            if 'time' in sol[name].dims:
                assert sol[name].sizes['time'] == 193, f'{name} has wrong time size'


class TestSegmentedExpansion:
    """Test that segmented cluster → expand produces correct values."""

    def test_expanded_objective_matches(self, system_with_storage, solver_fixture):
        fs_c = system_with_storage.transform.cluster(
            n_clusters=2, cluster_duration='1D', segments=tsam.SegmentConfig(n_segments=6)
        )
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        assert fs_e.solution['objective'].item() == pytest.approx(160.0, abs=1e-6)

    def test_expanded_flow_rates(self, system_with_storage, solver_fixture):
        fs_c = system_with_storage.transform.cluster(
            n_clusters=2, cluster_duration='1D', segments=tsam.SegmentConfig(n_segments=6)
        )
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        sol = fs_e.solution
        assert float(np.nansum(sol['B(Q_th)|flow_rate'].values)) == pytest.approx(2880.0, abs=1e-6)
        assert float(np.nansum(sol['D(Q)|flow_rate'].values)) == pytest.approx(2880.0, abs=1e-6)
        assert float(np.nansum(sol['G(Gas)|flow_rate'].values)) == pytest.approx(3200.0, abs=1e-6)

    def test_expanded_costs(self, system_with_storage, solver_fixture):
        fs_c = system_with_storage.transform.cluster(
            n_clusters=2, cluster_duration='1D', segments=tsam.SegmentConfig(n_segments=6)
        )
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        sol = fs_e.solution
        assert float(np.nansum(sol['costs(temporal)|per_timestep'].values)) == pytest.approx(160.0, abs=1e-6)
        assert float(np.nansum(sol['G(Gas)->costs(temporal)'].values)) == pytest.approx(160.0, abs=1e-6)

    def test_expanded_shapes(self, system_with_storage, solver_fixture):
        fs_c = system_with_storage.transform.cluster(
            n_clusters=2, cluster_duration='1D', segments=tsam.SegmentConfig(n_segments=6)
        )
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        sol = fs_e.solution
        for name in sol.data_vars:
            if 'time' in sol[name].dims:
                assert sol[name].sizes['time'] == 193, f'{name} has wrong time size'

    def test_no_nans_in_expanded_flow_rates(self, system_with_storage, solver_fixture):
        """Segmented expansion must ffill — no NaNs in flow rates (except extra boundary)."""
        fs_c = system_with_storage.transform.cluster(
            n_clusters=2, cluster_duration='1D', segments=tsam.SegmentConfig(n_segments=6)
        )
        fs_c.optimize(solver_fixture)
        fs_e = fs_c.transform.expand()

        sol = fs_e.solution
        for name in ['B(Q_th)|flow_rate', 'D(Q)|flow_rate', 'G(Gas)|flow_rate']:
            # Exclude last timestep (extra boundary, may be NaN for non-state variables)
            vals = sol[name].isel(time=slice(None, -1))
            assert not vals.isnull().any(), f'{name} has NaN values after expansion'
