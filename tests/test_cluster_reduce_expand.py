"""Tests for cluster() and expand_solution() functionality."""

import numpy as np
import pandas as pd
import pytest
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
    assert fs_reduced.clustering.result.cluster_structure.n_clusters == 2


def test_expand_solution_restores_full_timesteps(solver_fixture, timesteps_8_days):
    """Test that expand_solution restores full timestep count."""
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
    fs_expanded = fs_reduced.transform.expand_solution()

    # Should have original timestep count (flat, no clusters)
    assert len(fs_expanded.timesteps) == 192
    assert fs_expanded.clusters is None  # Expanded FlowSystem has no cluster dimension
    assert fs_expanded.solution is not None


def test_expand_solution_preserves_solution_variables(solver_fixture, timesteps_8_days):
    """Test that expand_solution keeps all solution variables."""
    fs = create_simple_system(timesteps_8_days)

    fs_reduced = fs.transform.cluster(
        n_clusters=2,
        cluster_duration='1D',
    )
    fs_reduced.optimize(solver_fixture)

    reduced_vars = set(fs_reduced.solution.data_vars)

    fs_expanded = fs_reduced.transform.expand_solution()
    expanded_vars = set(fs_expanded.solution.data_vars)

    # Should have all the same variables
    assert reduced_vars == expanded_vars


def test_expand_solution_maps_values_correctly(solver_fixture, timesteps_8_days):
    """Test that expand_solution correctly maps typical cluster values to all segments."""
    fs = create_simple_system(timesteps_8_days)

    fs_reduced = fs.transform.cluster(
        n_clusters=2,
        cluster_duration='1D',
    )
    fs_reduced.optimize(solver_fixture)

    # Get cluster_order to know mapping
    info = fs_reduced.clustering
    cluster_order = info.result.cluster_structure.cluster_order.values
    timesteps_per_cluster = info.result.cluster_structure.timesteps_per_cluster  # 24

    reduced_flow = fs_reduced.solution['Boiler(Q_th)|flow_rate'].values

    fs_expanded = fs_reduced.transform.expand_solution()
    expanded_flow = fs_expanded.solution['Boiler(Q_th)|flow_rate'].values

    # Check that values are correctly mapped
    # For each original segment, values should match the corresponding typical cluster
    for orig_segment_idx, cluster_id in enumerate(cluster_order):
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


def test_expand_solution_enables_statistics_accessor(solver_fixture, timesteps_8_days):
    """Test that statistics accessor works on expanded FlowSystem."""
    fs = create_simple_system(timesteps_8_days)

    fs_reduced = fs.transform.cluster(
        n_clusters=2,
        cluster_duration='1D',
    )
    fs_reduced.optimize(solver_fixture)

    fs_expanded = fs_reduced.transform.expand_solution()

    # These should work without errors
    flow_rates = fs_expanded.statistics.flow_rates
    assert 'Boiler(Q_th)' in flow_rates
    assert len(flow_rates['Boiler(Q_th)'].coords['time']) == 192

    flow_hours = fs_expanded.statistics.flow_hours
    assert 'Boiler(Q_th)' in flow_hours


def test_expand_solution_statistics_match_clustered(solver_fixture, timesteps_8_days):
    """Test that total_effects match between clustered and expanded FlowSystem."""
    fs = create_simple_system(timesteps_8_days)

    fs_reduced = fs.transform.cluster(
        n_clusters=2,
        cluster_duration='1D',
    )
    fs_reduced.optimize(solver_fixture)

    fs_expanded = fs_reduced.transform.expand_solution()

    # Total effects should match between clustered and expanded
    reduced_total = fs_reduced.statistics.total_effects['costs'].sum('contributor').item()
    expanded_total = fs_expanded.statistics.total_effects['costs'].sum('contributor').item()

    assert_allclose(reduced_total, expanded_total, rtol=1e-6)

    # Flow hours should also match (need to sum over time with proper weighting)
    # With 2D cluster structure, sum over both cluster and time dimensions
    reduced_fh = fs_reduced.statistics.flow_hours['Boiler(Q_th)'] * fs_reduced.cluster_weight
    reduced_flow_hours = reduced_fh.sum().item()  # Sum over all dimensions
    expanded_fh = fs_expanded.statistics.flow_hours['Boiler(Q_th)'] * fs_expanded.cluster_weight
    expanded_flow_hours = expanded_fh.sum().item()

    assert_allclose(reduced_flow_hours, expanded_flow_hours, rtol=1e-6)


def test_expand_solution_withoutclustering_raises(solver_fixture, timesteps_2_days):
    """Test that expand_solution raises error if not a reduced FlowSystem."""
    fs = create_simple_system(timesteps_2_days)
    fs.optimize(solver_fixture)

    with pytest.raises(ValueError, match='cluster'):
        fs.transform.expand_solution()


def test_expand_solution_without_solution_raises(timesteps_8_days):
    """Test that expand_solution raises error if no solution."""
    fs = create_simple_system(timesteps_8_days)

    fs_reduced = fs.transform.cluster(
        n_clusters=2,
        cluster_duration='1D',
    )
    # Don't optimize - no solution

    with pytest.raises(ValueError, match='no solution'):
        fs_reduced.transform.expand_solution()


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
    assert info.result.cluster_structure is not None
    assert info.result.cluster_structure.n_clusters == 2
    # Clustered FlowSystem preserves scenarios
    assert fs_reduced.scenarios is not None
    assert len(fs_reduced.scenarios) == 2


def test_cluster_and_expand_with_scenarios(solver_fixture, timesteps_8_days, scenarios_2):
    """Test full cluster -> optimize -> expand_solution cycle with scenarios."""
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
    fs_expanded = fs_reduced.transform.expand_solution()

    # Should have original timesteps
    assert len(fs_expanded.timesteps) == 192

    # Solution should have scenario dimension
    flow_var = 'Boiler(Q_th)|flow_rate'
    assert flow_var in fs_expanded.solution
    assert 'scenario' in fs_expanded.solution[flow_var].dims
    assert len(fs_expanded.solution[flow_var].coords['time']) == 192


def test_expand_solution_maps_scenarios_independently(solver_fixture, timesteps_8_days, scenarios_2):
    """Test that expand_solution correctly maps scenarios in multi-scenario systems."""
    fs = create_system_with_scenarios(timesteps_8_days, scenarios_2)

    fs_reduced = fs.transform.cluster(
        n_clusters=2,
        cluster_duration='1D',
    )
    fs_reduced.optimize(solver_fixture)

    info = fs_reduced.clustering
    cluster_structure = info.result.cluster_structure
    timesteps_per_cluster = cluster_structure.timesteps_per_cluster  # 24

    reduced_flow = fs_reduced.solution['Boiler(Q_th)|flow_rate']
    fs_expanded = fs_reduced.transform.expand_solution()
    expanded_flow = fs_expanded.solution['Boiler(Q_th)|flow_rate']

    # Check mapping for each scenario using its own cluster_order
    for scenario in scenarios_2:
        # Get the cluster_order for THIS scenario
        cluster_order = cluster_structure.get_cluster_order_for_slice(scenario=scenario)

        reduced_scenario = reduced_flow.sel(scenario=scenario).values
        expanded_scenario = expanded_flow.sel(scenario=scenario).values

        # Verify mapping is correct for this scenario using its own cluster_order
        for orig_segment_idx, cluster_id in enumerate(cluster_order):
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

        # Number of boundaries = n_original_periods + 1
        n_original_periods = fs_clustered.clustering.result.cluster_structure.n_original_periods
        assert soc_boundary.sizes['cluster_boundary'] == n_original_periods + 1

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

    def test_expand_solution_combines_soc_boundary_with_charge_state(self, solver_fixture, timesteps_8_days):
        """Expanded charge_state should be non-negative (combined with SOC_boundary)."""
        fs = create_system_with_storage(timesteps_8_days, cluster_mode='intercluster_cyclic')
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Note: Before expansion, charge_state represents ΔE (relative to period start)
        # which can be negative. After expansion, it becomes absolute SOC.

        # After expansion: charge_state should be non-negative (absolute SOC)
        fs_expanded = fs_clustered.transform.expand_solution()
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
        fs_expanded = fs_clustered.transform.expand_solution()
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
        cluster_structure = fs_clustered.clustering.result.cluster_structure
        cluster_order = cluster_structure.cluster_order.values
        timesteps_per_cluster = cluster_structure.timesteps_per_cluster

        fs_expanded = fs_clustered.transform.expand_solution()
        cs_expanded = fs_expanded.solution['Battery|charge_state']

        # Manual verification for first few timesteps of first period
        p = 0  # First period
        cluster = int(cluster_order[p])
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
    import xarray as xr

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

    def test_expand_solution_with_periods(self, solver_fixture, timesteps_8_days, periods_2):
        """Verify expansion handles period dimension correctly."""
        fs = create_system_with_periods(timesteps_8_days, periods_2)
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Expand
        fs_expanded = fs_clustered.transform.expand_solution()

        # Should have original timesteps and periods
        assert len(fs_expanded.timesteps) == 192
        assert fs_expanded.periods is not None
        assert len(fs_expanded.periods) == 2

        # Solution should have period dimension
        flow_var = 'Boiler(Q_th)|flow_rate'
        assert 'period' in fs_expanded.solution[flow_var].dims
        assert len(fs_expanded.solution[flow_var].coords['time']) == 192

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
        fs_expanded = fs_clustered.transform.expand_solution()
        assert 'period' in fs_expanded.solution[flow_var].dims
        assert 'scenario' in fs_expanded.solution[flow_var].dims
        assert len(fs_expanded.solution[flow_var].coords['time']) == 192


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
    """Tests for time_series_for_high_peaks and time_series_for_low_peaks parameters."""

    def test_time_series_for_high_peaks_parameter_accepted(self, timesteps_8_days):
        """Verify time_series_for_high_peaks parameter is accepted."""
        fs = create_system_with_peak_demand(timesteps_8_days)

        # Should not raise an error
        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            time_series_for_high_peaks=['HeatDemand(Q)|fixed_relative_profile'],
        )

        assert fs_clustered is not None
        assert len(fs_clustered.clusters) == 2

    def test_time_series_for_low_peaks_parameter_accepted(self, timesteps_8_days):
        """Verify time_series_for_low_peaks parameter is accepted."""
        fs = create_system_with_peak_demand(timesteps_8_days)

        # Should not raise an error
        # Note: tsam requires n_clusters >= 3 when using low_peaks to avoid index error
        fs_clustered = fs.transform.cluster(
            n_clusters=3,
            cluster_duration='1D',
            time_series_for_low_peaks=['HeatDemand(Q)|fixed_relative_profile'],
        )

        assert fs_clustered is not None
        assert len(fs_clustered.clusters) == 3

    def test_high_peaks_captures_extreme_demand_day(self, solver_fixture, timesteps_8_days):
        """Verify high peak selection captures day with maximum demand."""
        fs = create_system_with_peak_demand(timesteps_8_days)

        # Cluster WITH high peak selection
        fs_with_peaks = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            time_series_for_high_peaks=['HeatDemand(Q)|fixed_relative_profile'],
        )
        fs_with_peaks.optimize(solver_fixture)

        # The peak day (day 7 with demand=50) should be captured
        # Check that the clustered solution can handle the peak demand
        flow_rates = fs_with_peaks.solution['Boiler(Q_th)|flow_rate']

        # At least one cluster should have flow rate >= 50 (the peak)
        max_flow = float(flow_rates.max())
        assert max_flow >= 49, f'Peak demand not captured: max_flow={max_flow}'

    def test_clustering_without_peaks_may_miss_extremes(self, solver_fixture, timesteps_8_days):
        """Show that without peak selection, extreme days might be averaged out."""
        fs = create_system_with_peak_demand(timesteps_8_days)

        # Cluster WITHOUT high peak selection (may or may not capture peak)
        fs_no_peaks = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            # No time_series_for_high_peaks
        )
        fs_no_peaks.optimize(solver_fixture)

        # This test just verifies the clustering works
        # The peak may or may not be captured depending on clustering algorithm
        assert fs_no_peaks.solution is not None
