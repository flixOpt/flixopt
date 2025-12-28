"""Tests for clustering serialization and deserialization."""

import numpy as np
import pandas as pd
import pytest

import flixopt as fx


@pytest.fixture
def simple_system_24h():
    """Create a simple flow system with 24 hourly timesteps."""
    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')

    fs = fx.FlowSystem(timesteps)
    fs.add_elements(
        fx.Bus('heat'),
        fx.Effect('costs', unit='EUR', description='costs', is_objective=True, is_standard=True),
    )
    fs.add_elements(
        fx.Sink('demand', inputs=[fx.Flow('in', bus='heat', fixed_relative_profile=np.ones(24), size=10)]),
        fx.Source('source', outputs=[fx.Flow('out', bus='heat', size=50, effects_per_flow_hour={'costs': 0.05})]),
    )
    return fs


@pytest.fixture
def simple_system_8_days():
    """Create a simple flow system with 8 days of hourly timesteps."""
    timesteps = pd.date_range('2023-01-01', periods=8 * 24, freq='h')

    # Create varying demand profile with different patterns for different days
    # 4 "weekdays" with high demand, 4 "weekend" days with low demand
    hourly_pattern = np.sin(np.linspace(0, 2 * np.pi, 24)) * 0.5 + 0.5
    weekday_profile = hourly_pattern * 1.5  # Higher demand
    weekend_profile = hourly_pattern * 0.5  # Lower demand
    demand_profile = np.concatenate(
        [
            weekday_profile,
            weekday_profile,
            weekday_profile,
            weekday_profile,
            weekend_profile,
            weekend_profile,
            weekend_profile,
            weekend_profile,
        ]
    )

    fs = fx.FlowSystem(timesteps)
    fs.add_elements(
        fx.Bus('heat'),
        fx.Effect('costs', unit='EUR', description='costs', is_objective=True, is_standard=True),
    )
    fs.add_elements(
        fx.Sink('demand', inputs=[fx.Flow('in', bus='heat', fixed_relative_profile=demand_profile, size=10)]),
        fx.Source('source', outputs=[fx.Flow('out', bus='heat', size=50, effects_per_flow_hour={'costs': 0.05})]),
    )
    return fs


class TestClusteringRoundtrip:
    """Test that clustering survives dataset roundtrip."""

    def test_clustering_to_dataset_has_clustering_attrs(self, simple_system_8_days):
        """Clustered FlowSystem dataset should have clustering info."""
        fs = simple_system_8_days
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        ds = fs_clustered.to_dataset(include_solution=False)

        # Check that clustering attrs are present
        assert 'clustering' in ds.attrs

        # Check that clustering arrays are present with prefix
        clustering_vars = [name for name in ds.data_vars if name.startswith('clustering|')]
        assert len(clustering_vars) > 0

    def test_clustering_roundtrip_preserves_clustering_object(self, simple_system_8_days):
        """Clustering object should be restored after roundtrip."""
        fs = simple_system_8_days
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        # Roundtrip
        ds = fs_clustered.to_dataset(include_solution=False)
        fs_restored = fx.FlowSystem.from_dataset(ds)

        # Clustering should be restored
        assert fs_restored.clustering is not None
        assert fs_restored.clustering.backend_name == 'tsam'

    def test_clustering_roundtrip_preserves_n_clusters(self, simple_system_8_days):
        """Number of clusters should be preserved after roundtrip."""
        fs = simple_system_8_days
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        ds = fs_clustered.to_dataset(include_solution=False)
        fs_restored = fx.FlowSystem.from_dataset(ds)

        assert fs_restored.clustering.n_clusters == 2

    def test_clustering_roundtrip_preserves_timesteps_per_cluster(self, simple_system_8_days):
        """Timesteps per cluster should be preserved after roundtrip."""
        fs = simple_system_8_days
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        ds = fs_clustered.to_dataset(include_solution=False)
        fs_restored = fx.FlowSystem.from_dataset(ds)

        assert fs_restored.clustering.timesteps_per_cluster == 24

    def test_clustering_roundtrip_preserves_original_timesteps(self, simple_system_8_days):
        """Original timesteps should be preserved after roundtrip."""
        fs = simple_system_8_days
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        original_timesteps = fs_clustered.clustering.original_timesteps

        ds = fs_clustered.to_dataset(include_solution=False)
        fs_restored = fx.FlowSystem.from_dataset(ds)

        pd.testing.assert_index_equal(fs_restored.clustering.original_timesteps, original_timesteps)

    def test_clustering_roundtrip_preserves_timestep_mapping(self, simple_system_8_days):
        """Timestep mapping should be preserved after roundtrip."""
        fs = simple_system_8_days
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        original_mapping = fs_clustered.clustering.timestep_mapping.values.copy()

        ds = fs_clustered.to_dataset(include_solution=False)
        fs_restored = fx.FlowSystem.from_dataset(ds)

        np.testing.assert_array_equal(fs_restored.clustering.timestep_mapping.values, original_mapping)


class TestClusteringWithSolutionRoundtrip:
    """Test that clustering with solution survives roundtrip."""

    def test_expand_solution_after_roundtrip(self, simple_system_8_days, solver_fixture):
        """expand_solution should work after loading from dataset."""
        fs = simple_system_8_days
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        # Solve
        fs_clustered.optimize(solver_fixture)

        # Roundtrip
        ds = fs_clustered.to_dataset(include_solution=True)
        fs_restored = fx.FlowSystem.from_dataset(ds)

        # expand_solution should work
        fs_expanded = fs_restored.transform.expand_solution()

        # Check expanded FlowSystem has correct number of timesteps
        assert len(fs_expanded.timesteps) == 8 * 24

    def test_expand_solution_after_netcdf_roundtrip(self, simple_system_8_days, tmp_path, solver_fixture):
        """expand_solution should work after loading from NetCDF file."""
        fs = simple_system_8_days
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        # Solve
        fs_clustered.optimize(solver_fixture)

        # Save to NetCDF
        nc_path = tmp_path / 'clustered.nc'
        fs_clustered.to_netcdf(nc_path)

        # Load from NetCDF
        fs_restored = fx.FlowSystem.from_netcdf(nc_path)

        # expand_solution should work
        fs_expanded = fs_restored.transform.expand_solution()

        # Check expanded FlowSystem has correct number of timesteps
        assert len(fs_expanded.timesteps) == 8 * 24


class TestClusteringDerivedProperties:
    """Test derived properties on Clustering object."""

    def test_original_timesteps_property(self, simple_system_8_days):
        """original_timesteps property should return correct DatetimeIndex."""
        fs = simple_system_8_days
        original_timesteps = fs.timesteps

        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        # Check values are equal (name attribute may differ)
        pd.testing.assert_index_equal(
            fs_clustered.clustering.original_timesteps,
            original_timesteps,
            check_names=False,
        )

    def test_simple_system_has_no_periods_or_scenarios(self, simple_system_8_days):
        """Clustered simple system should preserve that it has no periods/scenarios."""
        fs = simple_system_8_days
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        # FlowSystem without periods/scenarios should remain so after clustering
        assert fs_clustered.periods is None
        assert fs_clustered.scenarios is None


class TestClusteringWithScenarios:
    """Test clustering IO with scenarios."""

    @pytest.fixture
    def system_with_scenarios(self):
        """Create a flow system with scenarios."""
        timesteps = pd.date_range('2023-01-01', periods=4 * 24, freq='h')
        scenarios = pd.Index(['Low', 'High'], name='scenario')

        # Create varying demand profile for clustering
        demand_profile = np.tile(np.sin(np.linspace(0, 2 * np.pi, 24)) * 0.5 + 0.5, 4)

        fs = fx.FlowSystem(timesteps, scenarios=scenarios)
        fs.add_elements(
            fx.Bus('heat'),
            fx.Effect('costs', unit='EUR', description='costs', is_objective=True, is_standard=True),
        )
        fs.add_elements(
            fx.Sink('demand', inputs=[fx.Flow('in', bus='heat', fixed_relative_profile=demand_profile, size=10)]),
            fx.Source('source', outputs=[fx.Flow('out', bus='heat', size=50, effects_per_flow_hour={'costs': 0.05})]),
        )
        return fs

    def test_clustering_roundtrip_preserves_scenarios(self, system_with_scenarios):
        """Scenarios should be preserved after clustering and roundtrip."""
        fs = system_with_scenarios
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        ds = fs_clustered.to_dataset(include_solution=False)
        fs_restored = fx.FlowSystem.from_dataset(ds)

        # Scenarios should be preserved in the FlowSystem itself
        pd.testing.assert_index_equal(
            fs_restored.scenarios,
            pd.Index(['Low', 'High'], name='scenario'),
            check_names=False,
        )
