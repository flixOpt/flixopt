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

    # Should have 2 * 24 = 48 timesteps instead of 192
    assert len(fs_reduced.timesteps) == 48
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
    assert len(fs_reduced.timesteps) == 48

    # Expand back to full
    fs_expanded = fs_reduced.transform.expand_solution()

    # Should have original timestep count
    assert len(fs_expanded.timesteps) == 192
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

        typical_start = cluster_id * timesteps_per_cluster
        typical_end = typical_start + timesteps_per_cluster

        # Values in the expanded solution for this original segment
        # should match the reduced solution for the corresponding typical cluster
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
    reduced_flow_hours = (
        (fs_reduced.statistics.flow_hours['Boiler(Q_th)'] * fs_reduced.cluster_weight).sum('time').item()
    )
    expanded_flow_hours = (
        (fs_expanded.statistics.flow_hours['Boiler(Q_th)'] * fs_expanded.cluster_weight).sum('time').item()
    )

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

    # Should have 2 * 24 = 48 timesteps
    assert len(fs_reduced.timesteps) == 48

    # Should have aggregation info with cluster structure
    info = fs_reduced.clustering
    assert info is not None
    assert info.result.cluster_structure is not None
    assert info.result.cluster_structure.n_clusters == 2
    # Original FlowSystem had scenarios
    assert info.original_flow_system.scenarios is not None


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

            typical_start = cluster_id * timesteps_per_cluster
            typical_end = typical_start + timesteps_per_cluster

            expected = reduced_scenario[typical_start:typical_end]
            actual = expanded_scenario[orig_start:orig_end]

            assert_allclose(actual, expected, rtol=1e-10, err_msg=f'Mismatch for scenario {scenario}')
