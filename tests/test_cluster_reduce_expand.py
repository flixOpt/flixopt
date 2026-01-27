"""Tests for cluster() and expand() functionality."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import flixopt as fx


def create_simple_system(timesteps: pd.DatetimeIndex) -> fx.FlowSystem:
    """Create a simple FlowSystem for testing clustering."""
    # Create varying demand - different for each day to test clustering
    hours = len(timesteps)
    demand = np.sin(np.linspace(0, 4 * np.pi, hours)) * 10 + 15  # Oscillating demand

    flow_system = fx.FlowSystem(timesteps)
    flow_system.add_elements(
        fx.Bus('Heat'),
        fx.Bus('Gas'),
        fx.Effect('costs', '€', is_standard=True, is_objective=True),
        fx.Sink('HeatDemand', inputs=[fx.Flow('Q', bus='Heat', fixed_relative_profile=demand, size=1)]),
        fx.Source('GasSource', outputs=[fx.Flow('Gas', bus='Gas', effects_per_flow_hour=0.05)]),
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.9,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow('Q_th', bus='Heat'),
        ),
    )
    return flow_system


@pytest.fixture
def timesteps_2_days():
    """48 hour timesteps (2 days)."""
    return pd.date_range('2020-01-01', periods=48, freq='h')


@pytest.fixture
def timesteps_8_days():
    """192 hour timesteps (8 days) - more realistic for clustering."""
    return pd.date_range('2020-01-01', periods=192, freq='h')


def test_cluster_creates_reduced_timesteps(timesteps_8_days):
    """Test that cluster creates a FlowSystem with fewer timesteps."""
    fs = create_simple_system(timesteps_8_days)

    # Reduce to 2 typical clusters (days)
    fs_reduced = fs.transform.cluster(
        n_clusters=2,
        cluster_duration='1D',
    )

    # Clustered FlowSystem has 2D structure: (cluster, time)
    # - timesteps: within-cluster time (24 hours)
    # - clusters: cluster indices (2 clusters)
    # Total effective timesteps = 2 * 24 = 48
    assert len(fs_reduced.timesteps) == 24  # Within-cluster time
    assert len(fs_reduced.clusters) == 2  # Number of clusters
    assert len(fs_reduced.timesteps) * len(fs_reduced.clusters) == 48  # Total
    assert hasattr(fs_reduced, 'clustering')
    assert fs_reduced.clustering.n_clusters == 2


def test_expand_restores_full_timesteps(solver_fixture, timesteps_8_days):
    """Test that expand restores full timestep count."""
    fs = create_simple_system(timesteps_8_days)

    # Reduce to 2 typical clusters
    fs_reduced = fs.transform.cluster(
        n_clusters=2,
        cluster_duration='1D',
    )

    # Optimize
    fs_reduced.optimize(solver_fixture)
    assert fs_reduced.solution is not None
    # Clustered: 24 within-cluster timesteps, 2 clusters
    assert len(fs_reduced.timesteps) == 24
    assert len(fs_reduced.clusters) == 2

    # Expand back to full
    fs_expanded = fs_reduced.transform.expand()

    # Should have original timestep count (flat, no clusters)
    assert len(fs_expanded.timesteps) == 192
    assert fs_expanded.clusters is None  # Expanded FlowSystem has no cluster dimension
    assert fs_expanded.solution is not None


def test_expand_preserves_solution_variables(solver_fixture, timesteps_8_days):
    """Test that expand keeps all solution variables."""
    fs = create_simple_system(timesteps_8_days)

    fs_reduced = fs.transform.cluster(
        n_clusters=2,
        cluster_duration='1D',
    )
    fs_reduced.optimize(solver_fixture)

    reduced_vars = set(fs_reduced.solution.data_vars)

    fs_expanded = fs_reduced.transform.expand()
    expanded_vars = set(fs_expanded.solution.data_vars)

    # Should have all the same variables
    assert reduced_vars == expanded_vars


def test_expand_maps_values_correctly(solver_fixture, timesteps_8_days):
    """Test that expand correctly maps typical cluster values to all segments."""
    fs = create_simple_system(timesteps_8_days)

    fs_reduced = fs.transform.cluster(
        n_clusters=2,
        cluster_duration='1D',
    )
    fs_reduced.optimize(solver_fixture)

    # Get cluster_assignments to know mapping
    info = fs_reduced.clustering
    cluster_assignments = info.cluster_assignments.values
    timesteps_per_cluster = info.timesteps_per_cluster  # 24

    reduced_flow = fs_reduced.solution['Boiler(Q_th)|flow_rate'].values

    fs_expanded = fs_reduced.transform.expand()
    expanded_flow = fs_expanded.solution['Boiler(Q_th)|flow_rate'].values

    # Check that values are correctly mapped
    # For each original segment, values should match the corresponding typical cluster
    for orig_segment_idx, cluster_id in enumerate(cluster_assignments):
        orig_start = orig_segment_idx * timesteps_per_cluster
        orig_end = orig_start + timesteps_per_cluster

        # Values in the expanded solution for this original segment
        # should match the reduced solution for the corresponding typical cluster
        # With 2D cluster structure, use cluster_id to index the cluster dimension
        # Note: solution may have extra timesteps (timesteps_extra), so slice to timesteps_per_cluster
        if reduced_flow.ndim == 2:
            # 2D structure: (cluster, time) - exclude extra timestep if present
            expected = reduced_flow[cluster_id, :timesteps_per_cluster]
        else:
            # Flat structure: (time,)
            typical_start = cluster_id * timesteps_per_cluster
            typical_end = typical_start + timesteps_per_cluster
            expected = reduced_flow[typical_start:typical_end]
        actual = expanded_flow[orig_start:orig_end]

        assert_allclose(actual, expected, rtol=1e-10)


def test_expand_enables_statistics_accessor(solver_fixture, timesteps_8_days):
    """Test that statistics accessor works on expanded FlowSystem."""
    fs = create_simple_system(timesteps_8_days)

    fs_reduced = fs.transform.cluster(
        n_clusters=2,
        cluster_duration='1D',
    )
    fs_reduced.optimize(solver_fixture)

    fs_expanded = fs_reduced.transform.expand()

    # These should work without errors
    flow_rates = fs_expanded.statistics.flow_rates
    assert 'Boiler(Q_th)' in flow_rates
    assert len(flow_rates['Boiler(Q_th)'].coords['time']) == 193  # 192 + 1 extra timestep

    flow_hours = fs_expanded.statistics.flow_hours
    assert 'Boiler(Q_th)' in flow_hours


def test_expand_statistics_match_clustered(solver_fixture, timesteps_8_days):
    """Test that total_effects match between clustered and expanded FlowSystem."""
    fs = create_simple_system(timesteps_8_days)

    fs_reduced = fs.transform.cluster(
        n_clusters=2,
        cluster_duration='1D',
    )
    fs_reduced.optimize(solver_fixture)

    fs_expanded = fs_reduced.transform.expand()

    # Total effects should match between clustered and expanded
    reduced_total = fs_reduced.statistics.total_effects['costs'].sum('contributor').item()
    expanded_total = fs_expanded.statistics.total_effects['costs'].sum('contributor').item()

    assert_allclose(reduced_total, expanded_total, rtol=1e-6)

    # Flow hours should also match (need to sum over time with proper weighting)
    # With 2D cluster structure, sum over both cluster and time dimensions
    reduced_fh = fs_reduced.statistics.flow_hours['Boiler(Q_th)'] * fs_reduced.cluster_weight
    reduced_flow_hours = reduced_fh.sum().item()  # Sum over all dimensions
    # Expanded FlowSystem has no cluster_weight (implicitly 1.0 for all timesteps)
    expanded_flow_hours = fs_expanded.statistics.flow_hours['Boiler(Q_th)'].sum().item()

    assert_allclose(reduced_flow_hours, expanded_flow_hours, rtol=1e-6)


def test_expand_withoutclustering_raises(solver_fixture, timesteps_2_days):
    """Test that expand raises error if not a reduced FlowSystem."""
    fs = create_simple_system(timesteps_2_days)
    fs.optimize(solver_fixture)

    with pytest.raises(ValueError, match='cluster'):
        fs.transform.expand()


def test_expand_without_solution_raises(timesteps_8_days):
    """Test that expand raises error if no solution."""
    fs = create_simple_system(timesteps_8_days)

    fs_reduced = fs.transform.cluster(
        n_clusters=2,
        cluster_duration='1D',
    )
    # Don't optimize - no solution

    with pytest.raises(ValueError, match='no solution'):
        fs_reduced.transform.expand()


# ==================== Multi-dimensional Tests ====================


def create_system_with_scenarios(timesteps: pd.DatetimeIndex, scenarios: pd.Index) -> fx.FlowSystem:
    """Create a FlowSystem with scenarios for testing."""
    hours = len(timesteps)

    # Create different demand profiles per scenario
    demands = {}
    for i, scenario in enumerate(scenarios):
        # Different pattern per scenario
        base_demand = np.sin(np.linspace(0, 4 * np.pi, hours)) * 10 + 15
        demands[scenario] = base_demand * (1 + 0.2 * i)  # Scale differently per scenario

    # Create DataFrame with scenarios as columns
    demand_df = pd.DataFrame(demands, index=timesteps)

    flow_system = fx.FlowSystem(timesteps, scenarios=scenarios)
    flow_system.add_elements(
        fx.Bus('Heat'),
        fx.Bus('Gas'),
        fx.Effect('costs', '€', is_standard=True, is_objective=True),
        fx.Sink(
            'HeatDemand',
            inputs=[fx.Flow('Q', bus='Heat', fixed_relative_profile=demand_df, size=1)],
        ),
        fx.Source('GasSource', outputs=[fx.Flow('Gas', bus='Gas', effects_per_flow_hour=0.05)]),
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.9,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow('Q_th', bus='Heat'),
        ),
    )
    return flow_system


@pytest.fixture
def scenarios_2():
    """Two scenarios for testing."""
    return pd.Index(['base', 'high'], name='scenario')


def test_cluster_with_scenarios(timesteps_8_days, scenarios_2):
    """Test that cluster handles scenarios correctly."""
    fs = create_system_with_scenarios(timesteps_8_days, scenarios_2)

    # Verify scenarios are set up correctly
    assert fs.scenarios is not None
    assert len(fs.scenarios) == 2

    # Reduce to 2 typical clusters
    fs_reduced = fs.transform.cluster(
        n_clusters=2,
        cluster_duration='1D',
    )

    # Clustered: 24 within-cluster timesteps, 2 clusters
    # Total effective timesteps = 2 * 24 = 48
    assert len(fs_reduced.timesteps) == 24
    assert len(fs_reduced.clusters) == 2
    assert len(fs_reduced.timesteps) * len(fs_reduced.clusters) == 48

    # Should have aggregation info with cluster structure
    info = fs_reduced.clustering
    assert info is not None
    assert info.n_clusters == 2
    # Clustered FlowSystem preserves scenarios
    assert fs_reduced.scenarios is not None
    assert len(fs_reduced.scenarios) == 2


def test_cluster_and_expand_with_scenarios(solver_fixture, timesteps_8_days, scenarios_2):
    """Test full cluster -> optimize -> expand cycle with scenarios."""
    fs = create_system_with_scenarios(timesteps_8_days, scenarios_2)

    # Reduce
    fs_reduced = fs.transform.cluster(
        n_clusters=2,
        cluster_duration='1D',
    )

    # Optimize
    fs_reduced.optimize(solver_fixture)
    assert fs_reduced.solution is not None

    # Expand
    fs_expanded = fs_reduced.transform.expand()

    # Should have original timesteps
    assert len(fs_expanded.timesteps) == 192

    # Solution should have scenario dimension
    flow_var = 'Boiler(Q_th)|flow_rate'
    assert flow_var in fs_expanded.solution
    assert 'scenario' in fs_expanded.solution[flow_var].dims
    assert len(fs_expanded.solution[flow_var].coords['time']) == 193  # 192 + 1 extra timestep


def test_expand_maps_scenarios_independently(solver_fixture, timesteps_8_days, scenarios_2):
    """Test that expand correctly maps scenarios in multi-scenario systems."""
    fs = create_system_with_scenarios(timesteps_8_days, scenarios_2)

    fs_reduced = fs.transform.cluster(
        n_clusters=2,
        cluster_duration='1D',
    )
    fs_reduced.optimize(solver_fixture)

    info = fs_reduced.clustering
    timesteps_per_cluster = info.timesteps_per_cluster  # 24

    reduced_flow = fs_reduced.solution['Boiler(Q_th)|flow_rate']
    fs_expanded = fs_reduced.transform.expand()
    expanded_flow = fs_expanded.solution['Boiler(Q_th)|flow_rate']

    # Check mapping for each scenario using its own cluster_assignments
    for scenario in scenarios_2:
        # Get the cluster_assignments for THIS scenario
        cluster_assignments = info.cluster_assignments.sel(scenario=scenario).values

        reduced_scenario = reduced_flow.sel(scenario=scenario).values
        expanded_scenario = expanded_flow.sel(scenario=scenario).values

        # Verify mapping is correct for this scenario using its own cluster_assignments
        for orig_segment_idx, cluster_id in enumerate(cluster_assignments):
            orig_start = orig_segment_idx * timesteps_per_cluster
            orig_end = orig_start + timesteps_per_cluster

            # With 2D cluster structure, use cluster_id to index the cluster dimension
            # Note: solution may have extra timesteps (timesteps_extra), so slice to timesteps_per_cluster
            if reduced_scenario.ndim == 2:
                # 2D structure: (cluster, time) - exclude extra timestep if present
                expected = reduced_scenario[cluster_id, :timesteps_per_cluster]
            else:
                # Flat structure: (time,)
                typical_start = cluster_id * timesteps_per_cluster
                typical_end = typical_start + timesteps_per_cluster
                expected = reduced_scenario[typical_start:typical_end]
            actual = expanded_scenario[orig_start:orig_end]

            assert_allclose(actual, expected, rtol=1e-10, err_msg=f'Mismatch for scenario {scenario}')


# ==================== Storage Clustering Tests ====================


def create_system_with_storage(
    timesteps: pd.DatetimeIndex,
    cluster_mode: str = 'intercluster_cyclic',
    relative_loss_per_hour: float = 0.0,
) -> fx.FlowSystem:
    """Create a FlowSystem with storage for testing clustering.

    Args:
        timesteps: DatetimeIndex for the simulation.
        cluster_mode: Storage cluster mode ('independent', 'cyclic', 'intercluster', 'intercluster_cyclic').
        relative_loss_per_hour: Self-discharge rate per hour (0.0 = no loss).
    """
    # Create demand pattern: high during day (hours 8-18), low at night
    hour_of_day = np.array([t.hour for t in timesteps])
    demand = np.where((hour_of_day >= 8) & (hour_of_day < 18), 20, 5)

    flow_system = fx.FlowSystem(timesteps)
    flow_system.add_elements(
        fx.Bus('Elec'),
        fx.Effect('costs', '€', is_standard=True, is_objective=True),
        fx.Source('Grid', outputs=[fx.Flow('P', bus='Elec', size=100, effects_per_flow_hour=0.1)]),
        fx.Sink('Load', inputs=[fx.Flow('P', bus='Elec', fixed_relative_profile=demand, size=1)]),
        fx.Storage(
            'Battery',
            charging=fx.Flow('charge', bus='Elec', size=30),
            discharging=fx.Flow('discharge', bus='Elec', size=30),
            capacity_in_flow_hours=100,
            relative_loss_per_hour=relative_loss_per_hour,
            cluster_mode=cluster_mode,
        ),
    )
    return flow_system


class TestStorageClusterModes:
    """Tests for different storage cluster_mode options."""

    def test_storage_cluster_mode_independent(self, solver_fixture, timesteps_8_days):
        """Storage with cluster_mode='independent' - each cluster starts fresh."""
        fs = create_system_with_storage(timesteps_8_days, cluster_mode='independent')
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Should have charge_state in solution
        assert 'Battery|charge_state' in fs_clustered.solution

        # Independent mode should NOT have SOC_boundary
        assert 'Battery|SOC_boundary' not in fs_clustered.solution

        # Verify solution is valid (no errors)
        assert fs_clustered.solution is not None

    def test_storage_cluster_mode_cyclic(self, solver_fixture, timesteps_8_days):
        """Storage with cluster_mode='cyclic' - start equals end per cluster."""
        fs = create_system_with_storage(timesteps_8_days, cluster_mode='cyclic')
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Should have charge_state in solution
        assert 'Battery|charge_state' in fs_clustered.solution

        # Cyclic mode should NOT have SOC_boundary (only intercluster modes do)
        assert 'Battery|SOC_boundary' not in fs_clustered.solution

    def test_storage_cluster_mode_intercluster(self, solver_fixture, timesteps_8_days):
        """Storage with cluster_mode='intercluster' - SOC links across clusters."""
        fs = create_system_with_storage(timesteps_8_days, cluster_mode='intercluster')
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Intercluster mode SHOULD have SOC_boundary
        assert 'Battery|SOC_boundary' in fs_clustered.solution

        soc_boundary = fs_clustered.solution['Battery|SOC_boundary']
        assert 'cluster_boundary' in soc_boundary.dims

        # Number of boundaries = n_original_clusters + 1
        n_original_clusters = fs_clustered.clustering.n_original_clusters
        assert soc_boundary.sizes['cluster_boundary'] == n_original_clusters + 1

    def test_storage_cluster_mode_intercluster_cyclic(self, solver_fixture, timesteps_8_days):
        """Storage with cluster_mode='intercluster_cyclic' - linked with yearly cycling."""
        fs = create_system_with_storage(timesteps_8_days, cluster_mode='intercluster_cyclic')
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Intercluster_cyclic mode SHOULD have SOC_boundary
        assert 'Battery|SOC_boundary' in fs_clustered.solution

        soc_boundary = fs_clustered.solution['Battery|SOC_boundary']
        assert 'cluster_boundary' in soc_boundary.dims

        # First and last SOC_boundary values should be equal (cyclic constraint)
        first_soc = soc_boundary.isel(cluster_boundary=0).item()
        last_soc = soc_boundary.isel(cluster_boundary=-1).item()
        assert_allclose(first_soc, last_soc, rtol=1e-6)


class TestInterclusterStorageLinking:
    """Tests for inter-cluster storage linking and SOC_boundary behavior."""

    def test_intercluster_storage_has_soc_boundary(self, solver_fixture, timesteps_8_days):
        """Verify intercluster storage creates SOC_boundary variable."""
        fs = create_system_with_storage(timesteps_8_days, cluster_mode='intercluster_cyclic')
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Verify SOC_boundary exists in solution
        assert 'Battery|SOC_boundary' in fs_clustered.solution
        soc_boundary = fs_clustered.solution['Battery|SOC_boundary']
        assert 'cluster_boundary' in soc_boundary.dims

    def test_expand_combines_soc_boundary_with_charge_state(self, solver_fixture, timesteps_8_days):
        """Expanded charge_state should be non-negative (combined with SOC_boundary)."""
        fs = create_system_with_storage(timesteps_8_days, cluster_mode='intercluster_cyclic')
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Note: Before expansion, charge_state represents ΔE (relative to period start)
        # which can be negative. After expansion, it becomes absolute SOC.

        # After expansion: charge_state should be non-negative (absolute SOC)
        fs_expanded = fs_clustered.transform.expand()
        cs_after = fs_expanded.solution['Battery|charge_state']

        # All values should be >= 0 (with small tolerance for numerical issues)
        assert (cs_after >= -0.01).all(), f'Negative charge_state found: min={float(cs_after.min())}'

    def test_storage_self_discharge_decay_in_expansion(self, solver_fixture, timesteps_8_days):
        """Verify self-discharge decay factor applied correctly during expansion."""
        # Use significant self-discharge to make decay visible
        fs = create_system_with_storage(
            timesteps_8_days,
            cluster_mode='intercluster_cyclic',
            relative_loss_per_hour=0.01,  # 1% per hour
        )
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Expand solution
        fs_expanded = fs_clustered.transform.expand()
        cs_expanded = fs_expanded.solution['Battery|charge_state']

        # With self-discharge, SOC should decay over time within each period
        # The expanded solution should still be non-negative
        assert (cs_expanded >= -0.01).all()

    def test_expanded_charge_state_matches_manual_calculation(self, solver_fixture, timesteps_8_days):
        """Verify expanded charge_state = SOC_boundary * decay + delta_E formula."""
        loss_rate = 0.01  # 1% per hour
        fs = create_system_with_storage(
            timesteps_8_days,
            cluster_mode='intercluster_cyclic',
            relative_loss_per_hour=loss_rate,
        )
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Get values needed for manual calculation
        soc_boundary = fs_clustered.solution['Battery|SOC_boundary']
        cs_clustered = fs_clustered.solution['Battery|charge_state']
        clustering = fs_clustered.clustering
        cluster_assignments = clustering.cluster_assignments.values
        timesteps_per_cluster = clustering.timesteps_per_cluster

        fs_expanded = fs_clustered.transform.expand()
        cs_expanded = fs_expanded.solution['Battery|charge_state']

        # Manual verification for first few timesteps of first period
        p = 0  # First period
        cluster = int(cluster_assignments[p])
        soc_b = soc_boundary.isel(cluster_boundary=p).item()

        for t in [0, 5, 12, 23]:
            global_t = p * timesteps_per_cluster + t
            delta_e = cs_clustered.isel(cluster=cluster, time=t).item()
            decay = (1 - loss_rate) ** t
            expected = soc_b * decay + delta_e
            expected_clipped = max(0.0, expected)
            actual = cs_expanded.isel(time=global_t).item()

            assert_allclose(
                actual,
                expected_clipped,
                rtol=0.01,
                err_msg=f'Mismatch at period {p}, time {t}: expected {expected_clipped}, got {actual}',
            )


# ==================== Multi-Period Clustering Tests ====================


def create_system_with_periods(timesteps: pd.DatetimeIndex, periods: pd.Index) -> fx.FlowSystem:
    """Create a FlowSystem with periods for testing multi-period clustering."""
    hours = len(timesteps)
    # Create demand pattern that varies by day to ensure multiple clusters
    hour_of_day = np.array([t.hour for t in timesteps])
    day_of_year = np.arange(hours) // 24
    # Add day-based variation: odd days have higher demand
    base_demand = np.where((hour_of_day >= 8) & (hour_of_day < 18), 20, 8)
    demand = base_demand * (1 + 0.3 * (day_of_year % 2))  # 30% higher on odd days

    flow_system = fx.FlowSystem(timesteps, periods=periods)
    flow_system.add_elements(
        fx.Bus('Heat'),
        fx.Bus('Gas'),
        fx.Effect('costs', '€', is_standard=True, is_objective=True),
        fx.Sink('HeatDemand', inputs=[fx.Flow('Q', bus='Heat', fixed_relative_profile=demand, size=1)]),
        fx.Source('GasSource', outputs=[fx.Flow('Gas', bus='Gas', effects_per_flow_hour=0.05)]),
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.9,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow('Q_th', bus='Heat'),
        ),
    )
    return flow_system


def create_system_with_periods_and_scenarios(
    timesteps: pd.DatetimeIndex, periods: pd.Index, scenarios: pd.Index
) -> fx.FlowSystem:
    """Create a FlowSystem with both periods and scenarios."""
    hours = len(timesteps)

    # Create demand that varies by scenario AND by day (for clustering)
    hour_of_day = np.array([t.hour for t in timesteps])
    day_of_year = np.arange(hours) // 24
    base_demand = np.where((hour_of_day >= 8) & (hour_of_day < 18), 20, 8)
    # Add day variation for clustering
    base_demand = base_demand * (1 + 0.3 * (day_of_year % 2))

    # Create demand array with explicit scenario dimension using xarray
    # Shape: (time, scenario)
    demand_data = np.column_stack([base_demand * (1 + 0.2 * i) for i in range(len(scenarios))])
    demand_da = xr.DataArray(
        demand_data,
        dims=['time', 'scenario'],
        coords={'time': timesteps, 'scenario': scenarios},
    )

    flow_system = fx.FlowSystem(timesteps, periods=periods, scenarios=scenarios)
    flow_system.add_elements(
        fx.Bus('Heat'),
        fx.Bus('Gas'),
        fx.Effect('costs', '€', is_standard=True, is_objective=True),
        fx.Sink(
            'HeatDemand',
            inputs=[fx.Flow('Q', bus='Heat', fixed_relative_profile=demand_da, size=1)],
        ),
        fx.Source('GasSource', outputs=[fx.Flow('Gas', bus='Gas', effects_per_flow_hour=0.05)]),
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.9,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow('Q_th', bus='Heat'),
        ),
    )
    return flow_system


@pytest.fixture
def periods_2():
    """Two periods for testing."""
    return pd.Index([2020, 2021], name='period')


class TestMultiPeriodClustering:
    """Tests for clustering with periods dimension."""

    def test_cluster_with_periods(self, timesteps_8_days, periods_2):
        """Test clustering with periods dimension."""
        fs = create_system_with_periods(timesteps_8_days, periods_2)

        # Verify periods are set up correctly
        assert fs.periods is not None
        assert len(fs.periods) == 2

        # Cluster
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        # Should have period dimension preserved
        assert fs_clustered.periods is not None
        assert len(fs_clustered.periods) == 2

        # Clustered: 24 within-cluster timesteps, 2 clusters
        assert len(fs_clustered.timesteps) == 24
        assert len(fs_clustered.clusters) == 2

    def test_cluster_with_periods_optimizes(self, solver_fixture, timesteps_8_days, periods_2):
        """Test that clustering with periods can be optimized."""
        fs = create_system_with_periods(timesteps_8_days, periods_2)
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Should have solution with period dimension
        assert fs_clustered.solution is not None
        flow_var = 'Boiler(Q_th)|flow_rate'
        assert flow_var in fs_clustered.solution
        assert 'period' in fs_clustered.solution[flow_var].dims

    def test_expand_with_periods(self, solver_fixture, timesteps_8_days, periods_2):
        """Verify expansion handles period dimension correctly."""
        fs = create_system_with_periods(timesteps_8_days, periods_2)
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Expand
        fs_expanded = fs_clustered.transform.expand()

        # Should have original timesteps and periods
        assert len(fs_expanded.timesteps) == 192
        assert fs_expanded.periods is not None
        assert len(fs_expanded.periods) == 2

        # Solution should have period dimension
        flow_var = 'Boiler(Q_th)|flow_rate'
        assert 'period' in fs_expanded.solution[flow_var].dims
        assert len(fs_expanded.solution[flow_var].coords['time']) == 193  # 192 + 1 extra timestep

    def test_cluster_with_periods_and_scenarios(self, solver_fixture, timesteps_8_days, periods_2, scenarios_2):
        """Clustering should work with both periods and scenarios."""
        fs = create_system_with_periods_and_scenarios(timesteps_8_days, periods_2, scenarios_2)

        # Verify setup
        assert fs.periods is not None
        assert fs.scenarios is not None
        assert len(fs.periods) == 2
        assert len(fs.scenarios) == 2

        # Cluster and optimize
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Verify dimensions
        flow_var = 'Boiler(Q_th)|flow_rate'
        assert 'period' in fs_clustered.solution[flow_var].dims
        assert 'scenario' in fs_clustered.solution[flow_var].dims
        assert 'cluster' in fs_clustered.solution[flow_var].dims

        # Expand and verify
        fs_expanded = fs_clustered.transform.expand()
        assert 'period' in fs_expanded.solution[flow_var].dims
        assert 'scenario' in fs_expanded.solution[flow_var].dims
        assert len(fs_expanded.solution[flow_var].coords['time']) == 193  # 192 + 1 extra timestep


# ==================== Peak Selection Tests ====================


def create_system_with_peak_demand(timesteps: pd.DatetimeIndex) -> fx.FlowSystem:
    """Create a FlowSystem with clearly identifiable peak demand days."""
    hours = len(timesteps)

    # Create demand with distinct patterns to ensure multiple clusters
    # Days 0,1: low demand (base pattern)
    # Days 2,3: medium demand (higher pattern)
    # Days 4,5,6: normal demand (moderate pattern)
    # Day 7: extreme peak (very high)
    day = np.arange(hours) // 24
    hour_of_day = np.arange(hours) % 24

    # Base pattern varies by day group
    base = np.where((hour_of_day >= 8) & (hour_of_day < 18), 15, 5)

    demand = np.where(
        (day == 7) & (hour_of_day >= 10) & (hour_of_day < 14),
        50,  # Extreme peak on day 7
        np.where(
            day <= 1,
            base * 0.7,  # Low days
            np.where(day <= 3, base * 1.3, base),  # Medium days vs normal
        ),
    )

    flow_system = fx.FlowSystem(timesteps)
    flow_system.add_elements(
        fx.Bus('Heat'),
        fx.Bus('Gas'),
        fx.Effect('costs', '€', is_standard=True, is_objective=True),
        fx.Sink('HeatDemand', inputs=[fx.Flow('Q', bus='Heat', fixed_relative_profile=demand, size=1)]),
        fx.Source('GasSource', outputs=[fx.Flow('Gas', bus='Gas', effects_per_flow_hour=0.05)]),
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.9,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow('Q_th', bus='Heat'),
        ),
    )
    return flow_system


class TestPeakSelection:
    """Tests for extremes config with max_value and min_value parameters."""

    def test_extremes_max_value_parameter_accepted(self, timesteps_8_days):
        """Verify extremes max_value parameter is accepted."""
        from tsam import ExtremeConfig

        fs = create_system_with_peak_demand(timesteps_8_days)

        # Should not raise an error
        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            extremes=ExtremeConfig(method='new_cluster', max_value=['HeatDemand(Q)|fixed_relative_profile']),
        )

        assert fs_clustered is not None
        assert len(fs_clustered.clusters) == 2

    def test_extremes_min_value_parameter_accepted(self, timesteps_8_days):
        """Verify extremes min_value parameter is accepted."""
        from tsam import ExtremeConfig

        fs = create_system_with_peak_demand(timesteps_8_days)

        # Should not raise an error
        # Note: tsam requires n_clusters >= 3 when using min_value to avoid index error
        fs_clustered = fs.transform.cluster(
            n_clusters=3,
            cluster_duration='1D',
            extremes=ExtremeConfig(method='new_cluster', min_value=['HeatDemand(Q)|fixed_relative_profile']),
        )

        assert fs_clustered is not None
        assert len(fs_clustered.clusters) == 3

    def test_extremes_captures_extreme_demand_day(self, solver_fixture, timesteps_8_days):
        """Verify extremes config captures day with maximum demand."""
        from tsam import ExtremeConfig

        fs = create_system_with_peak_demand(timesteps_8_days)

        # Cluster WITH extremes config
        fs_with_peaks = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            extremes=ExtremeConfig(method='new_cluster', max_value=['HeatDemand(Q)|fixed_relative_profile']),
        )
        fs_with_peaks.optimize(solver_fixture)

        # The peak day (day 7 with demand=50) should be captured
        # Check that the clustered solution can handle the peak demand
        flow_rates = fs_with_peaks.solution['Boiler(Q_th)|flow_rate']

        # At least one cluster should have flow rate >= 50 (the peak)
        max_flow = float(flow_rates.max())
        assert max_flow >= 49, f'Peak demand not captured: max_flow={max_flow}'

    def test_clustering_without_extremes_may_miss_peaks(self, solver_fixture, timesteps_8_days):
        """Show that without extremes config, extreme days might be averaged out."""
        fs = create_system_with_peak_demand(timesteps_8_days)

        # Cluster WITHOUT extremes config (may or may not capture peak)
        fs_no_peaks = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            # No extremes config
        )
        fs_no_peaks.optimize(solver_fixture)

        # This test just verifies the clustering works
        # The peak may or may not be captured depending on clustering algorithm
        assert fs_no_peaks.solution is not None

    def test_extremes_new_cluster_increases_n_clusters(self, solver_fixture, timesteps_8_days):
        """Test that method='new_cluster' can increase n_clusters when extreme periods are detected.

        Note: tsam's new_cluster method may or may not add clusters depending on whether
        the extreme period is already captured by an existing cluster. The assertion
        checks that at least the requested n_clusters is maintained.
        """
        from tsam import ExtremeConfig

        fs = create_system_with_peak_demand(timesteps_8_days)

        # Cluster with extremes as new clusters
        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            extremes=ExtremeConfig(
                method='new_cluster',
                max_value=['HeatDemand(Q)|fixed_relative_profile'],
            ),
        )

        # n_clusters should be >= 2 (may be higher if extreme periods are added as new clusters)
        assert fs_clustered.clustering.n_clusters >= 2

        # Verify optimization works with the actual cluster count
        fs_clustered.optimize(solver_fixture)
        assert fs_clustered.solution is not None

        # Verify expansion works
        fs_expanded = fs_clustered.transform.expand()
        assert len(fs_expanded.timesteps) == 192

        # The sum of cluster occurrences should equal n_original_clusters (8 days)
        assert int(fs_clustered.clustering.cluster_occurrences.sum()) == 8

    def test_extremes_new_cluster_rejected_for_multi_period(self, timesteps_8_days, periods_2):
        """Test that method='new_cluster' is rejected for multi-period systems."""
        from tsam import ExtremeConfig

        fs = create_system_with_periods(timesteps_8_days, periods_2)

        with pytest.raises(ValueError, match='method="new_cluster" is not supported'):
            fs.transform.cluster(
                n_clusters=2,
                cluster_duration='1D',
                extremes=ExtremeConfig(
                    method='new_cluster',
                    max_value=['HeatDemand(Q)|fixed_relative_profile'],
                ),
            )

    def test_extremes_replace_works_for_multi_period(self, solver_fixture, timesteps_8_days, periods_2):
        """Test that method='replace' works correctly for multi-period systems."""
        from tsam import ExtremeConfig

        fs = create_system_with_periods(timesteps_8_days, periods_2)

        # method='replace' should work - it maintains the requested n_clusters
        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            extremes=ExtremeConfig(
                method='replace',
                max_value=['HeatDemand(Q)|fixed_relative_profile'],
            ),
        )

        assert fs_clustered.clustering.n_clusters == 2
        fs_clustered.optimize(solver_fixture)
        assert fs_clustered.solution is not None

    def test_extremes_append_with_segments(self, solver_fixture, timesteps_8_days):
        """Test that method='append' works correctly with segmentation."""
        from tsam import ExtremeConfig, SegmentConfig

        fs = create_system_with_peak_demand(timesteps_8_days)

        # Cluster with BOTH extremes AND segments
        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            extremes=ExtremeConfig(
                method='append',
                max_value=['HeatDemand(Q)|fixed_relative_profile'],
            ),
            segments=SegmentConfig(n_segments=6),
        )

        # n_clusters should be >= 2 (extreme periods add clusters)
        n_clusters = fs_clustered.clustering.n_clusters
        assert n_clusters >= 2

        # n_representatives = n_clusters * n_segments
        assert fs_clustered.clustering.n_representatives == n_clusters * 6

        # Verify optimization works
        fs_clustered.optimize(solver_fixture)
        assert fs_clustered.solution is not None

        # Verify expansion works
        fs_expanded = fs_clustered.transform.expand()
        assert len(fs_expanded.timesteps) == 192

        # The sum of cluster occurrences should equal n_original_clusters (8 days)
        assert int(fs_clustered.clustering.cluster_occurrences.sum()) == 8


# ==================== Data Vars Parameter Tests ====================


class TestDataVarsParameter:
    """Tests for data_vars parameter in cluster() method."""

    def test_cluster_with_data_vars_subset(self, timesteps_8_days):
        """Test clustering with a subset of variables."""
        # Create system with multiple time-varying data
        hours = len(timesteps_8_days)
        demand = np.sin(np.linspace(0, 4 * np.pi, hours)) * 10 + 15
        price = np.cos(np.linspace(0, 4 * np.pi, hours)) * 0.02 + 0.05  # Different pattern

        fs = fx.FlowSystem(timesteps_8_days)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink('HeatDemand', inputs=[fx.Flow('Q', bus='Heat', fixed_relative_profile=demand, size=1)]),
            fx.Source('GasSource', outputs=[fx.Flow('Gas', bus='Gas', effects_per_flow_hour=price)]),
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=0.9,
                fuel_flow=fx.Flow('Q_fu', bus='Gas'),
                thermal_flow=fx.Flow('Q_th', bus='Heat'),
            ),
        )

        # Cluster based only on demand profile (not price)
        fs_reduced = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            data_vars=['HeatDemand(Q)|fixed_relative_profile'],
        )

        # Should have clustered structure
        assert len(fs_reduced.timesteps) == 24
        assert len(fs_reduced.clusters) == 2

    def test_data_vars_validation_error(self, timesteps_8_days):
        """Test that invalid data_vars raises ValueError."""
        fs = create_simple_system(timesteps_8_days)

        with pytest.raises(ValueError, match='data_vars not found'):
            fs.transform.cluster(
                n_clusters=2,
                cluster_duration='1D',
                data_vars=['NonExistentVariable'],
            )

    def test_data_vars_preserves_all_flowsystem_data(self, timesteps_8_days):
        """Test that clustering with data_vars preserves all FlowSystem variables."""
        # Create system with multiple time-varying data
        hours = len(timesteps_8_days)
        demand = np.sin(np.linspace(0, 4 * np.pi, hours)) * 10 + 15
        price = np.cos(np.linspace(0, 4 * np.pi, hours)) * 0.02 + 0.05

        fs = fx.FlowSystem(timesteps_8_days)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink('HeatDemand', inputs=[fx.Flow('Q', bus='Heat', fixed_relative_profile=demand, size=1)]),
            fx.Source('GasSource', outputs=[fx.Flow('Gas', bus='Gas', effects_per_flow_hour=price)]),
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=0.9,
                fuel_flow=fx.Flow('Q_fu', bus='Gas'),
                thermal_flow=fx.Flow('Q_th', bus='Heat'),
            ),
        )

        # Cluster based only on demand profile
        fs_reduced = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            data_vars=['HeatDemand(Q)|fixed_relative_profile'],
        )

        # Both demand and price should be preserved in the reduced FlowSystem
        ds = fs_reduced.to_dataset()
        assert 'HeatDemand(Q)|fixed_relative_profile' in ds.data_vars
        assert 'GasSource(Gas)|costs|per_flow_hour' in ds.data_vars

    def test_data_vars_optimization_works(self, solver_fixture, timesteps_8_days):
        """Test that FlowSystem clustered with data_vars can be optimized."""
        hours = len(timesteps_8_days)
        demand = np.sin(np.linspace(0, 4 * np.pi, hours)) * 10 + 15
        price = np.cos(np.linspace(0, 4 * np.pi, hours)) * 0.02 + 0.05

        fs = fx.FlowSystem(timesteps_8_days)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink('HeatDemand', inputs=[fx.Flow('Q', bus='Heat', fixed_relative_profile=demand, size=1)]),
            fx.Source('GasSource', outputs=[fx.Flow('Gas', bus='Gas', effects_per_flow_hour=price)]),
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=0.9,
                fuel_flow=fx.Flow('Q_fu', bus='Gas'),
                thermal_flow=fx.Flow('Q_th', bus='Heat'),
            ),
        )

        fs_reduced = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            data_vars=['HeatDemand(Q)|fixed_relative_profile'],
        )

        # Should optimize successfully
        fs_reduced.optimize(solver_fixture)
        assert fs_reduced.solution is not None
        assert 'Boiler(Q_th)|flow_rate' in fs_reduced.solution

    def test_data_vars_with_multiple_variables(self, timesteps_8_days):
        """Test clustering with multiple selected variables."""
        hours = len(timesteps_8_days)
        demand = np.sin(np.linspace(0, 4 * np.pi, hours)) * 10 + 15
        price = np.cos(np.linspace(0, 4 * np.pi, hours)) * 0.02 + 0.05

        fs = fx.FlowSystem(timesteps_8_days)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink('HeatDemand', inputs=[fx.Flow('Q', bus='Heat', fixed_relative_profile=demand, size=1)]),
            fx.Source('GasSource', outputs=[fx.Flow('Gas', bus='Gas', effects_per_flow_hour=price)]),
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=0.9,
                fuel_flow=fx.Flow('Q_fu', bus='Gas'),
                thermal_flow=fx.Flow('Q_th', bus='Heat'),
            ),
        )

        # Cluster based on both demand and price
        fs_reduced = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            data_vars=[
                'HeatDemand(Q)|fixed_relative_profile',
                'GasSource(Gas)|costs|per_flow_hour',
            ],
        )

        assert len(fs_reduced.timesteps) == 24
        assert len(fs_reduced.clusters) == 2


# ==================== Segmentation Tests ====================


class TestSegmentation:
    """Tests for intra-period segmentation (variable timestep durations within clusters)."""

    def test_segment_config_creates_segmented_system(self, timesteps_8_days):
        """Test that SegmentConfig creates a segmented FlowSystem."""
        from tsam import SegmentConfig

        fs = create_simple_system(timesteps_8_days)

        # Cluster with 6 segments per day (instead of 24 hourly timesteps)
        fs_segmented = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=6),
        )

        # Verify segmentation properties
        assert fs_segmented.clustering.is_segmented is True
        assert fs_segmented.clustering.n_segments == 6
        assert fs_segmented.clustering.timesteps_per_cluster == 24  # Original period length

        # Time dimension should have n_segments entries (not timesteps_per_cluster)
        assert len(fs_segmented.timesteps) == 6  # 6 segments

        # Verify RangeIndex for segmented time
        assert isinstance(fs_segmented.timesteps, pd.RangeIndex)

    def test_segmented_system_has_variable_timestep_durations(self, timesteps_8_days):
        """Test that segmented systems have variable timestep durations."""
        from tsam import SegmentConfig

        fs = create_simple_system(timesteps_8_days)

        fs_segmented = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=6),
        )

        # Timestep duration should be a DataArray with cluster dimension
        timestep_duration = fs_segmented.timestep_duration
        assert 'cluster' in timestep_duration.dims
        assert 'time' in timestep_duration.dims

        # Sum of durations per cluster should equal original period length (24 hours)
        for cluster in fs_segmented.clusters:
            cluster_duration_sum = timestep_duration.sel(cluster=cluster).sum().item()
            assert_allclose(cluster_duration_sum, 24.0, rtol=1e-6)

    def test_segmented_system_optimizes(self, solver_fixture, timesteps_8_days):
        """Test that segmented systems can be optimized."""
        from tsam import SegmentConfig

        fs = create_simple_system(timesteps_8_days)

        fs_segmented = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=6),
        )

        # Optimize
        fs_segmented.optimize(solver_fixture)

        # Should have solution
        assert fs_segmented.solution is not None
        assert 'objective' in fs_segmented.solution

        # Flow rates should have (cluster, time) structure with 6 time points
        flow_var = 'Boiler(Q_th)|flow_rate'
        assert flow_var in fs_segmented.solution
        # time dimension has n_segments + 1 (for previous_flow_rate pattern)
        assert fs_segmented.solution[flow_var].sizes['time'] == 7  # 6 + 1

    def test_segmented_expand_restores_original_timesteps(self, solver_fixture, timesteps_8_days):
        """Test that expand() restores the original timestep count for segmented systems."""
        from tsam import SegmentConfig

        fs = create_simple_system(timesteps_8_days)

        # Cluster with segments
        fs_segmented = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=6),
        )

        # Optimize and expand
        fs_segmented.optimize(solver_fixture)
        fs_expanded = fs_segmented.transform.expand()

        # Should have original timesteps restored
        assert len(fs_expanded.timesteps) == 192  # 8 days * 24 hours
        assert fs_expanded.clusters is None  # No cluster dimension after expansion

        # Should have DatetimeIndex after expansion (not RangeIndex)
        assert isinstance(fs_expanded.timesteps, pd.DatetimeIndex)

    def test_segmented_expand_preserves_objective(self, solver_fixture, timesteps_8_days):
        """Test that expand() preserves the objective value for segmented systems."""
        from tsam import SegmentConfig

        fs = create_simple_system(timesteps_8_days)

        fs_segmented = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=6),
        )

        fs_segmented.optimize(solver_fixture)
        segmented_objective = fs_segmented.solution['objective'].item()

        fs_expanded = fs_segmented.transform.expand()
        expanded_objective = fs_expanded.solution['objective'].item()

        # Objectives should be equal (expand preserves solution)
        assert_allclose(segmented_objective, expanded_objective, rtol=1e-6)

    def test_segmented_expand_has_correct_flow_rates(self, solver_fixture, timesteps_8_days):
        """Test that expanded flow rates have correct timestep count."""
        from tsam import SegmentConfig

        fs = create_simple_system(timesteps_8_days)

        fs_segmented = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=6),
        )

        fs_segmented.optimize(solver_fixture)
        fs_expanded = fs_segmented.transform.expand()

        # Check flow rates dimension
        flow_var = 'Boiler(Q_th)|flow_rate'
        flow_rates = fs_expanded.solution[flow_var]

        # Should have original time dimension
        assert flow_rates.sizes['time'] == 193  # 192 + 1 (previous_flow_rate)

    def test_segmented_statistics_after_expand(self, solver_fixture, timesteps_8_days):
        """Test that statistics accessor works after expanding segmented system."""
        from tsam import SegmentConfig

        fs = create_simple_system(timesteps_8_days)

        fs_segmented = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=6),
        )

        fs_segmented.optimize(solver_fixture)
        fs_expanded = fs_segmented.transform.expand()

        # Statistics should work
        stats = fs_expanded.statistics
        assert hasattr(stats, 'flow_rates')
        assert hasattr(stats, 'total_effects')

        # Flow rates should have correct dimensions
        flow_rates = stats.flow_rates
        assert 'time' in flow_rates.dims

    def test_segmented_timestep_mapping_uses_segment_assignments(self, timesteps_8_days):
        """Test that timestep_mapping correctly maps original timesteps to segments."""
        from tsam import SegmentConfig

        fs = create_simple_system(timesteps_8_days)

        fs_segmented = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=6),
        )

        mapping = fs_segmented.clustering.timestep_mapping

        # Mapping should have original timestep count
        assert len(mapping.values) == 192

        # Each mapped value should be in valid range: [0, n_clusters * n_segments)
        max_valid_idx = 2 * 6 - 1  # n_clusters * n_segments - 1
        assert mapping.min().item() >= 0
        assert mapping.max().item() <= max_valid_idx

    @pytest.mark.parametrize('freq', ['1h', '2h'])
    def test_segmented_total_effects_match_solution(self, solver_fixture, freq):
        """Test that total_effects matches solution Cost after expand with segmentation.

        This is a regression test for the bug where expansion_divisor was computed
        incorrectly for segmented systems, causing total_effects to not match the
        solution's objective value.
        """
        from tsam import SegmentConfig

        # Create system with specified timestep frequency
        n_timesteps = 72 if freq == '1h' else 36  # 3 days worth
        timesteps = pd.date_range('2024-01-01', periods=n_timesteps, freq=freq)
        fs = fx.FlowSystem(timesteps=timesteps)

        # Minimal components: effect + source + sink with varying demand
        fs.add_elements(fx.Effect('Cost', unit='EUR', is_objective=True))
        fs.add_elements(fx.Bus('Heat'))
        fs.add_elements(
            fx.Source(
                'Boiler',
                outputs=[fx.Flow('Q', bus='Heat', size=100, effects_per_flow_hour={'Cost': 50})],
            )
        )
        demand_profile = np.tile([0.5, 1], n_timesteps // 2)
        fs.add_elements(
            fx.Sink('Demand', inputs=[fx.Flow('Q', bus='Heat', size=50, fixed_relative_profile=demand_profile)])
        )

        # Cluster with segments -> solve -> expand
        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=4),
        )
        fs_clustered.optimize(solver_fixture)
        fs_expanded = fs_clustered.transform.expand()

        # Validate: total_effects must match solution objective
        computed = fs_expanded.statistics.total_effects['Cost'].sum('contributor')
        expected = fs_expanded.solution['Cost']
        assert np.allclose(computed.values, expected.values, rtol=1e-5), (
            f'total_effects mismatch: computed={float(computed):.2f}, expected={float(expected):.2f}'
        )


class TestSegmentationWithStorage:
    """Tests for segmentation combined with storage components."""

    def test_segmented_storage_optimizes(self, solver_fixture, timesteps_8_days):
        """Test that segmented systems with storage can be optimized."""
        from tsam import SegmentConfig

        fs = create_system_with_storage(timesteps_8_days, cluster_mode='cyclic')

        fs_segmented = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=6),
        )

        fs_segmented.optimize(solver_fixture)

        # Should have solution with charge_state
        assert fs_segmented.solution is not None
        assert 'Battery|charge_state' in fs_segmented.solution

    def test_segmented_storage_expand(self, solver_fixture, timesteps_8_days):
        """Test that segmented storage systems can be expanded."""
        from tsam import SegmentConfig

        fs = create_system_with_storage(timesteps_8_days, cluster_mode='cyclic')

        fs_segmented = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=6),
        )

        fs_segmented.optimize(solver_fixture)
        fs_expanded = fs_segmented.transform.expand()

        # Charge state should be expanded to original timesteps
        charge_state = fs_expanded.solution['Battery|charge_state']
        # charge_state has time dimension = n_original_timesteps + 1
        assert charge_state.sizes['time'] == 193


class TestSegmentationWithPeriods:
    """Tests for segmentation combined with multi-period systems."""

    def test_segmented_with_periods(self, solver_fixture, timesteps_8_days, periods_2):
        """Test segmentation with multiple periods."""
        from tsam import SegmentConfig

        fs = create_system_with_periods(timesteps_8_days, periods_2)

        fs_segmented = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=6),
        )

        # Verify structure
        assert fs_segmented.clustering.is_segmented is True
        assert fs_segmented.periods is not None
        assert len(fs_segmented.periods) == 2

        # Optimize
        fs_segmented.optimize(solver_fixture)
        assert fs_segmented.solution is not None

    def test_segmented_with_periods_expand(self, solver_fixture, timesteps_8_days, periods_2):
        """Test expansion of segmented multi-period systems."""
        from tsam import SegmentConfig

        fs = create_system_with_periods(timesteps_8_days, periods_2)

        fs_segmented = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=6),
        )

        fs_segmented.optimize(solver_fixture)
        fs_expanded = fs_segmented.transform.expand()

        # Should have original timesteps and periods preserved
        assert len(fs_expanded.timesteps) == 192
        assert fs_expanded.periods is not None
        assert len(fs_expanded.periods) == 2

        # Solution should have period dimension
        flow_var = 'Boiler(Q_th)|flow_rate'
        assert 'period' in fs_expanded.solution[flow_var].dims

    def test_segmented_different_clustering_per_period(self, solver_fixture, timesteps_8_days, periods_2):
        """Test that different periods can have different cluster assignments."""
        from tsam import SegmentConfig

        fs = create_system_with_periods(timesteps_8_days, periods_2)

        fs_segmented = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=6),
        )

        # Verify cluster_assignments has period dimension
        cluster_assignments = fs_segmented.clustering.cluster_assignments
        assert 'period' in cluster_assignments.dims

        # Each period should have independent cluster assignments
        # (may or may not be different depending on data)
        assert cluster_assignments.sizes['period'] == 2

        fs_segmented.optimize(solver_fixture)
        fs_expanded = fs_segmented.transform.expand()

        # Expanded solution should preserve period dimension
        flow_var = 'Boiler(Q_th)|flow_rate'
        assert 'period' in fs_expanded.solution[flow_var].dims
        assert fs_expanded.solution[flow_var].sizes['period'] == 2

    def test_segmented_expand_maps_correctly_per_period(self, solver_fixture, timesteps_8_days, periods_2):
        """Test that expand maps values correctly for each period independently."""
        from tsam import SegmentConfig

        fs = create_system_with_periods(timesteps_8_days, periods_2)

        fs_segmented = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=6),
        )

        fs_segmented.optimize(solver_fixture)

        # Get the timestep_mapping which should be multi-dimensional
        mapping = fs_segmented.clustering.timestep_mapping

        # Mapping should have period dimension
        assert 'period' in mapping.dims
        assert mapping.sizes['period'] == 2

        # Expand and verify each period has correct number of timesteps
        fs_expanded = fs_segmented.transform.expand()
        flow_var = 'Boiler(Q_th)|flow_rate'
        flow_rates = fs_expanded.solution[flow_var]

        # Each period should have the original time dimension
        # time = 193 (192 + 1 for previous_flow_rate pattern)
        assert flow_rates.sizes['time'] == 193
        assert flow_rates.sizes['period'] == 2


class TestSegmentationIO:
    """Tests for IO round-trip of segmented systems."""

    def test_segmented_roundtrip(self, solver_fixture, timesteps_8_days, tmp_path):
        """Test that segmented systems survive IO round-trip."""
        from tsam import SegmentConfig

        fs = create_simple_system(timesteps_8_days)

        fs_segmented = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=6),
        )

        fs_segmented.optimize(solver_fixture)

        # Save and load
        path = tmp_path / 'segmented.nc4'
        fs_segmented.to_netcdf(path)
        fs_loaded = fx.FlowSystem.from_netcdf(path)

        # Verify segmentation preserved
        assert fs_loaded.clustering.is_segmented is True
        assert fs_loaded.clustering.n_segments == 6

        # Verify solution preserved
        assert_allclose(
            fs_loaded.solution['objective'].item(),
            fs_segmented.solution['objective'].item(),
            rtol=1e-6,
        )

    def test_segmented_expand_after_load(self, solver_fixture, timesteps_8_days, tmp_path):
        """Test that expand works after loading segmented system."""
        from tsam import SegmentConfig

        fs = create_simple_system(timesteps_8_days)

        fs_segmented = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=6),
        )

        fs_segmented.optimize(solver_fixture)

        # Save, load, and expand
        path = tmp_path / 'segmented.nc4'
        fs_segmented.to_netcdf(path)
        fs_loaded = fx.FlowSystem.from_netcdf(path)
        fs_expanded = fs_loaded.transform.expand()

        # Should have original timesteps
        assert len(fs_expanded.timesteps) == 192

        # Objective should be preserved
        assert_allclose(
            fs_expanded.solution['objective'].item(),
            fs_segmented.solution['objective'].item(),
            rtol=1e-6,
        )


class TestStartupShutdownExpansion:
    """Tests for correct expansion of startup/shutdown binary events."""

    def test_startup_shutdown_first_timestep_only(self, solver_fixture, timesteps_8_days):
        """Test that startup/shutdown events are placed at first timestep of each segment only."""
        from tsam import SegmentConfig

        # Create system with on/off behavior
        fs = fx.FlowSystem(timesteps=timesteps_8_days)
        fs.add_elements(fx.Effect('Cost', unit='EUR', is_objective=True))
        fs.add_elements(fx.Bus('Heat'))

        # Source with minimum active time (forces on/off tracking)
        fs.add_elements(
            fx.Source(
                'Boiler',
                outputs=[
                    fx.Flow(
                        'Q',
                        bus='Heat',
                        size=100,
                        status_parameters=fx.StatusParameters(effects_per_startup={'Cost': 10}),
                        effects_per_flow_hour={'Cost': 50},
                    )
                ],
            )
        )

        # Variable demand that forces startups
        demand_pattern = np.array([0.8] * 12 + [0.0] * 12)  # On/off pattern per day (0-1 range)
        demand_profile = np.tile(demand_pattern, 8)
        fs.add_elements(
            fx.Sink('Demand', inputs=[fx.Flow('Q', bus='Heat', size=50, fixed_relative_profile=demand_profile)])
        )

        # Cluster with segments
        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=6),
        )

        fs_clustered.optimize(solver_fixture)

        # Check if startup variable exists
        startup_var = 'Boiler(Q)|startup'
        if startup_var not in fs_clustered.solution:
            pytest.skip('Startup variable not in solution (solver may not have triggered any startups)')

        # Expand and check startup placement
        fs_expanded = fs_clustered.transform.expand()

        startup_expanded = fs_expanded.solution[startup_var]

        # In expanded form, startup should be sparse: mostly zeros with 1s only at segment boundaries
        # The total count should match the clustered solution (after weighting)
        startup_clustered = fs_clustered.solution[startup_var]

        # Get cluster weights for proper comparison
        cluster_weight = fs_clustered.to_dataset()['cluster_weight']

        # For expanded: just sum all startups
        total_expanded = float(startup_expanded.sum())

        # For clustered: sum with weights
        total_clustered = float((startup_clustered * cluster_weight).sum())

        # They should match (startup events are preserved, just relocated to first timestep)
        assert_allclose(total_expanded, total_clustered, rtol=1e-5)

        # Verify sparsity: most timesteps should be 0
        n_timesteps = startup_expanded.sizes['time']
        n_nonzero = int((startup_expanded > 0.5).sum())  # Binary, so 0.5 threshold
        assert n_nonzero < n_timesteps * 0.2, f'Expected sparse startups, but got {n_nonzero}/{n_timesteps} non-zero'

    def test_startup_timing_preserved_non_segmented(self, solver_fixture, timesteps_8_days):
        """Test that startup timing within cluster is preserved for non-segmented systems."""
        # Create system with on/off behavior
        fs = fx.FlowSystem(timesteps=timesteps_8_days)
        fs.add_elements(fx.Effect('Cost', unit='EUR', is_objective=True))
        fs.add_elements(fx.Bus('Heat'))

        fs.add_elements(
            fx.Source(
                'Boiler',
                outputs=[
                    fx.Flow(
                        'Q',
                        bus='Heat',
                        size=100,
                        status_parameters=fx.StatusParameters(effects_per_startup={'Cost': 10}),
                        effects_per_flow_hour={'Cost': 50},
                    )
                ],
            )
        )

        demand_pattern = np.array([0.8] * 12 + [0.0] * 12)  # On/off pattern per day (0-1 range)
        demand_profile = np.tile(demand_pattern, 8)
        fs.add_elements(
            fx.Sink('Demand', inputs=[fx.Flow('Q', bus='Heat', size=50, fixed_relative_profile=demand_profile)])
        )

        # Cluster WITHOUT segments
        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
        )

        fs_clustered.optimize(solver_fixture)

        startup_var = 'Boiler(Q)|startup'
        if startup_var not in fs_clustered.solution:
            pytest.skip('Startup variable not in solution')

        fs_expanded = fs_clustered.transform.expand()

        # For non-segmented systems, timing within cluster should be preserved
        # The expanded startup should match the clustered values at corresponding positions
        startup_clustered = fs_clustered.solution[startup_var]
        startup_expanded = fs_expanded.solution[startup_var]

        # Get cluster assignments to verify mapping
        cluster_assignments = fs_clustered.clustering.cluster_assignments.values
        timesteps_per_cluster = 24

        # Check that expanded values match clustered values at correct positions
        for orig_day in range(8):
            cluster_id = cluster_assignments[orig_day]
            for hour in range(timesteps_per_cluster):
                orig_idx = orig_day * timesteps_per_cluster + hour
                clustered_val = float(startup_clustered.isel(cluster=cluster_id, time=hour))
                expanded_val = float(startup_expanded.isel(time=orig_idx))
                assert abs(clustered_val - expanded_val) < 1e-6, (
                    f'Mismatch at day {orig_day}, hour {hour}: clustered={clustered_val}, expanded={expanded_val}'
                )


class TestCombineSlices:
    """Tests for the combine_slices utility function."""

    def test_single_dim(self):
        """Test combining slices with a single extra dimension."""
        from flixopt.clustering.base import combine_slices

        slices = {
            ('A',): np.array([1.0, 2.0, 3.0]),
            ('B',): np.array([4.0, 5.0, 6.0]),
        }
        result = combine_slices(
            slices,
            extra_dims=['x'],
            dim_coords={'x': ['A', 'B']},
            output_dim='time',
            output_coord=[0, 1, 2],
        )

        assert result.dims == ('time', 'x')
        assert result.shape == (3, 2)
        assert result.sel(x='A').values.tolist() == [1.0, 2.0, 3.0]
        assert result.sel(x='B').values.tolist() == [4.0, 5.0, 6.0]

    def test_two_dims(self):
        """Test combining slices with two extra dimensions."""
        from flixopt.clustering.base import combine_slices

        slices = {
            ('P1', 'base'): np.array([1.0, 2.0]),
            ('P1', 'high'): np.array([3.0, 4.0]),
            ('P2', 'base'): np.array([5.0, 6.0]),
            ('P2', 'high'): np.array([7.0, 8.0]),
        }
        result = combine_slices(
            slices,
            extra_dims=['period', 'scenario'],
            dim_coords={'period': ['P1', 'P2'], 'scenario': ['base', 'high']},
            output_dim='time',
            output_coord=[0, 1],
        )

        assert result.dims == ('time', 'period', 'scenario')
        assert result.shape == (2, 2, 2)
        assert result.sel(period='P1', scenario='base').values.tolist() == [1.0, 2.0]
        assert result.sel(period='P2', scenario='high').values.tolist() == [7.0, 8.0]

    def test_attrs_propagation(self):
        """Test that attrs are propagated to the result."""
        from flixopt.clustering.base import combine_slices

        slices = {('A',): np.array([1.0, 2.0])}
        result = combine_slices(
            slices,
            extra_dims=['x'],
            dim_coords={'x': ['A']},
            output_dim='time',
            output_coord=[0, 1],
            attrs={'units': 'kW', 'description': 'power'},
        )

        assert result.attrs['units'] == 'kW'
        assert result.attrs['description'] == 'power'

    def test_datetime_coords(self):
        """Test with pandas DatetimeIndex as output coordinates."""
        from flixopt.clustering.base import combine_slices

        time_index = pd.date_range('2020-01-01', periods=3, freq='h')
        slices = {('A',): np.array([1.0, 2.0, 3.0])}
        result = combine_slices(
            slices,
            extra_dims=['x'],
            dim_coords={'x': ['A']},
            output_dim='time',
            output_coord=time_index,
        )

        assert result.dims == ('time', 'x')
        assert len(result.coords['time']) == 3
        assert result.coords['time'][0].values == time_index[0]
