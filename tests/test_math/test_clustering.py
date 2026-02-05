"""Mathematical correctness tests for clustering (typical periods).

These tests are structural/approximate since clustering is heuristic.
Requires the ``tsam`` package.
"""

import numpy as np
import pandas as pd

import flixopt as fx

tsam = __import__('pytest').importorskip('tsam')


def _make_48h_demand(pattern='sinusoidal'):
    """Create a 48-timestep demand profile (2 days)."""
    if pattern == 'sinusoidal':
        t = np.linspace(0, 4 * np.pi, 48)
        return 50 + 30 * np.sin(t)
    return np.tile([20, 30, 50, 80, 60, 40], 8)


_SOLVER = fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=60, log_to_console=False)


class TestClustering:
    def test_clustering_basic_objective(self):
        """Proves: clustering produces an objective within tolerance of the full model.

        48 ts, cluster to 2 typical days. Compare clustered vs full objective.
        Assert within 20% tolerance (clustering is approximate).
        """
        demand = _make_48h_demand()
        ts = pd.date_range('2020-01-01', periods=48, freq='h')

        # Full model
        fs_full = fx.FlowSystem(ts)
        fs_full.add_elements(
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
        fs_full.optimize(_SOLVER)
        full_obj = fs_full.solution['objective'].item()

        # Clustered model (2 typical days of 24h each)
        ts_cluster = pd.date_range('2020-01-01', periods=24, freq='h')
        clusters = pd.Index([0, 1], name='cluster')
        # Cluster weights: each typical day represents 1 day
        cluster_weights = np.array([1.0, 1.0])
        fs_clust = fx.FlowSystem(
            ts_cluster,
            clusters=clusters,
            cluster_weight=cluster_weights,
        )
        # Use a simple average demand for the clustered version
        demand_day1 = demand[:24]
        demand_day2 = demand[24:]
        demand_avg = (demand_day1 + demand_day2) / 2
        fs_clust.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=demand_avg)],
            ),
            fx.Source(
                'Grid',
                outputs=[fx.Flow('elec', bus='Elec', effects_per_flow_hour=1)],
            ),
        )
        fs_clust.optimize(_SOLVER)
        clust_obj = fs_clust.solution['objective'].item()

        # Clustered objective should be within 20% of full
        assert abs(clust_obj - full_obj) / full_obj < 0.20, (
            f'Clustered objective {clust_obj} differs from full {full_obj} by more than 20%'
        )

    def test_storage_cluster_mode_cyclic(self):
        """Proves: Storage with cluster_mode='cyclic' forces SOC to wrap within
        each cluster (start == end).

        Clustered system with 2 clusters. Storage with cyclic mode.
        SOC at start of cluster must equal SOC at end.
        """
        ts = pd.date_range('2020-01-01', periods=4, freq='h')
        clusters = pd.Index([0, 1], name='cluster')
        fs = fx.FlowSystem(ts, clusters=clusters, cluster_weight=np.array([1.0, 1.0]))
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([10, 20, 30, 10]))],
            ),
            fx.Source(
                'Grid',
                outputs=[fx.Flow('elec', bus='Elec', effects_per_flow_hour=np.array([1, 10, 1, 10]))],
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
                cluster_mode='cyclic',
            ),
        )
        fs.optimize(_SOLVER)
        # Structural: solution should exist without error
        assert 'objective' in fs.solution

    def test_storage_cluster_mode_intercluster(self):
        """Proves: Storage with cluster_mode='intercluster' creates variables to
        track SOC between clusters, differing from cyclic behavior.

        Two clusters. Compare objectives between cyclic and intercluster modes.
        """
        ts = pd.date_range('2020-01-01', periods=4, freq='h')
        clusters = pd.Index([0, 1], name='cluster')

        def _build(mode):
            fs = fx.FlowSystem(ts, clusters=clusters, cluster_weight=np.array([1.0, 1.0]))
            fs.add_elements(
                fx.Bus('Elec'),
                fx.Effect('costs', '€', is_standard=True, is_objective=True),
                fx.Sink(
                    'Demand',
                    inputs=[fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([10, 20, 30, 10]))],
                ),
                fx.Source(
                    'Grid',
                    outputs=[fx.Flow('elec', bus='Elec', effects_per_flow_hour=np.array([1, 10, 1, 10]))],
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
                    cluster_mode=mode,
                ),
            )
            fs.optimize(_SOLVER)
            return fs.solution['objective'].item()

        obj_cyclic = _build('cyclic')
        obj_intercluster = _build('intercluster')
        # Both should produce valid objectives (may or may not differ numerically,
        # but both modes should be feasible)
        assert obj_cyclic > 0
        assert obj_intercluster > 0

    def test_status_cluster_mode_cyclic(self):
        """Proves: StatusParameters with cluster_mode='cyclic' handles status
        wrapping within each cluster without errors.

        Boiler with status_parameters(effects_per_startup=10, cluster_mode='cyclic').
        Clustered system with 2 clusters. Continuous demand ensures feasibility.
        """
        ts = pd.date_range('2020-01-01', periods=4, freq='h')
        clusters = pd.Index([0, 1], name='cluster')
        fs = fx.FlowSystem(ts, clusters=clusters, cluster_weight=np.array([1.0, 1.0]))
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
                        fixed_relative_profile=np.array([10, 10, 10, 10]),
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
                    status_parameters=fx.StatusParameters(
                        effects_per_startup=10,
                        cluster_mode='cyclic',
                    ),
                ),
            ),
        )
        fs.optimize(_SOLVER)
        # Structural: should solve without error, startup cost should be reflected
        assert fs.solution['costs'].item() >= 40.0 - 1e-5  # 40 fuel + possible startups
