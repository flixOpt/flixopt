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


class TestClusteringJsonExport:
    """Test that clustering can be exported to JSON."""

    def test_clustering_json_export_unsolved(self, simple_system_8_days, tmp_path):
        """Unsolved clustered FlowSystem should export to JSON without error."""
        fs = simple_system_8_days
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        # Save to JSON should work
        json_path = tmp_path / 'clustered.json'
        fs_clustered.to_json(json_path)

        # File should exist and be valid JSON
        assert json_path.exists()
        import json

        with open(json_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_clustering_json_export_solved(self, simple_system_8_days, tmp_path, solver_fixture):
        """Solved clustered FlowSystem should export to JSON without error."""
        fs = simple_system_8_days
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Save to JSON should work
        json_path = tmp_path / 'clustered_solved.json'
        fs_clustered.to_json(json_path)

        # File should exist
        assert json_path.exists()


class TestExpandedFlowSystemIO:
    """Test that expanded FlowSystems can be saved and loaded."""

    def test_expanded_flowsystem_to_dataset(self, simple_system_8_days, solver_fixture):
        """Expanded FlowSystem should be convertible to dataset."""
        fs = simple_system_8_days
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        fs_expanded = fs_clustered.transform.expand_solution()

        # Should be able to convert to dataset
        ds = fs_expanded.to_dataset(include_solution=True)

        # Should have correct timesteps
        assert len(ds.coords['time']) == 8 * 24

        # Should NOT have clustering info (it was expanded)
        assert fs_expanded.clustering is None

    def test_expanded_flowsystem_netcdf_roundtrip(self, simple_system_8_days, tmp_path, solver_fixture):
        """Expanded FlowSystem should roundtrip through NetCDF."""
        fs = simple_system_8_days
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        fs_expanded = fs_clustered.transform.expand_solution()

        # Save to NetCDF
        nc_path = tmp_path / 'expanded.nc'
        fs_expanded.to_netcdf(nc_path)

        # Load from NetCDF
        fs_restored = fx.FlowSystem.from_netcdf(nc_path)

        # Should have correct timesteps
        assert len(fs_restored.timesteps) == 8 * 24

        # Solution should be preserved
        assert fs_restored.solution is not None

    def test_expanded_flowsystem_json_export(self, simple_system_8_days, tmp_path, solver_fixture):
        """Expanded FlowSystem should export to JSON without error."""
        fs = simple_system_8_days
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        fs_expanded = fs_clustered.transform.expand_solution()

        # Save to JSON should work
        json_path = tmp_path / 'expanded.json'
        fs_expanded.to_json(json_path)

        # File should exist
        assert json_path.exists()


class TestClusteringWithPeriodsIO:
    """Test clustering IO with periods."""

    @pytest.fixture
    def system_with_periods(self):
        """Create a flow system with periods."""
        timesteps = pd.date_range('2023-01-01', periods=4 * 24, freq='h')
        periods = pd.Index([2023, 2024], name='period')

        demand_profile = np.tile(np.sin(np.linspace(0, 2 * np.pi, 24)) * 0.5 + 0.5, 4)

        fs = fx.FlowSystem(timesteps, periods=periods)
        fs.add_elements(
            fx.Bus('heat'),
            fx.Effect('costs', unit='EUR', description='costs', is_objective=True, is_standard=True),
        )
        fs.add_elements(
            fx.Sink('demand', inputs=[fx.Flow('in', bus='heat', fixed_relative_profile=demand_profile, size=10)]),
            fx.Source('source', outputs=[fx.Flow('out', bus='heat', size=50, effects_per_flow_hour={'costs': 0.05})]),
        )
        return fs

    def test_clustering_with_periods_netcdf_roundtrip(self, system_with_periods, tmp_path, solver_fixture):
        """Clustered FlowSystem with periods should roundtrip through NetCDF."""
        fs = system_with_periods
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Save to NetCDF
        nc_path = tmp_path / 'clustered_periods.nc'
        fs_clustered.to_netcdf(nc_path)

        # Load from NetCDF
        fs_restored = fx.FlowSystem.from_netcdf(nc_path)

        # Clustering should be preserved
        assert fs_restored.clustering is not None
        assert fs_restored.clustering.n_clusters == 2

        # Periods should be preserved
        pd.testing.assert_index_equal(fs_restored.periods, pd.Index([2023, 2024], name='period'), check_names=False)

        # expand_solution should work
        fs_expanded = fs_restored.transform.expand_solution()
        assert len(fs_expanded.timesteps) == 4 * 24


class TestClusterWeightRoundtrip:
    """Test that cluster_weight is properly preserved."""

    def test_cluster_weight_in_dataset(self, simple_system_8_days):
        """cluster_weight should be present in dataset."""
        fs = simple_system_8_days
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        ds = fs_clustered.to_dataset(include_solution=False)

        # cluster_weight should be in data_vars
        assert 'cluster_weight' in ds.data_vars

    def test_cluster_weight_roundtrip(self, simple_system_8_days):
        """cluster_weight should be preserved after roundtrip."""
        fs = simple_system_8_days
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        original_weight = fs_clustered.cluster_weight.values.copy()

        ds = fs_clustered.to_dataset(include_solution=False)
        fs_restored = fx.FlowSystem.from_dataset(ds)

        np.testing.assert_array_equal(fs_restored.cluster_weight.values, original_weight)

    def test_cluster_weight_sums_to_original_clusters(self, simple_system_8_days):
        """cluster_weight should sum to number of original clusters."""
        fs = simple_system_8_days
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        # 8 days clustered -> weights should sum to 8
        assert fs_clustered.cluster_weight.sum() == 8

        # After roundtrip
        ds = fs_clustered.to_dataset(include_solution=False)
        fs_restored = fx.FlowSystem.from_dataset(ds)
        assert fs_restored.cluster_weight.sum() == 8


class TestInterclusterStorageIO:
    """Test IO for intercluster storage mode."""

    @pytest.fixture
    def system_with_intercluster_storage(self):
        """Create system with intercluster storage."""
        timesteps = pd.date_range('2023-01-01', periods=4 * 24, freq='h')

        # Varying demand to make storage useful
        demand_profile = np.tile(np.sin(np.linspace(0, 2 * np.pi, 24)) * 0.5 + 0.5, 4)

        fs = fx.FlowSystem(timesteps)
        fs.add_elements(
            fx.Bus('heat'),
            fx.Effect('costs', unit='EUR', description='costs', is_objective=True, is_standard=True),
        )
        fs.add_elements(
            fx.Sink('demand', inputs=[fx.Flow('in', bus='heat', fixed_relative_profile=demand_profile, size=10)]),
            fx.Source('source', outputs=[fx.Flow('out', bus='heat', size=50, effects_per_flow_hour={'costs': 0.1})]),
            fx.Storage(
                'storage',
                charging=fx.Flow('in', bus='heat', size=20),
                discharging=fx.Flow('out', bus='heat', size=20),
                capacity_in_flow_hours=100,
                cluster_mode='intercluster',  # Key: intercluster mode
            ),
        )
        return fs

    def test_intercluster_storage_solution_roundtrip(self, system_with_intercluster_storage, solver_fixture):
        """Intercluster storage solution should roundtrip correctly."""
        fs = system_with_intercluster_storage
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Solution should have SOC_boundary variable
        assert 'storage|SOC_boundary' in fs_clustered.solution

        # Roundtrip
        ds = fs_clustered.to_dataset(include_solution=True)
        fs_restored = fx.FlowSystem.from_dataset(ds)

        # SOC_boundary should be preserved
        assert 'storage|SOC_boundary' in fs_restored.solution

        # expand_solution should work
        fs_expanded = fs_restored.transform.expand_solution()

        # After expansion, SOC_boundary is combined into charge_state
        assert 'storage|SOC_boundary' not in fs_expanded.solution
        assert 'storage|charge_state' in fs_expanded.solution

    def test_intercluster_storage_netcdf_roundtrip(self, system_with_intercluster_storage, tmp_path, solver_fixture):
        """Intercluster storage solution should roundtrip through NetCDF."""
        fs = system_with_intercluster_storage
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Save to NetCDF
        nc_path = tmp_path / 'intercluster.nc'
        fs_clustered.to_netcdf(nc_path)

        # Load from NetCDF
        fs_restored = fx.FlowSystem.from_netcdf(nc_path)

        # expand_solution should produce valid charge_state
        fs_expanded = fs_restored.transform.expand_solution()
        charge_state = fs_expanded.solution['storage|charge_state']

        # Charge state should be non-negative (after combining with SOC_boundary)
        assert (charge_state >= -1e-6).all()


class TestClusteringEdgeCases:
    """Test edge cases in clustering IO."""

    def test_single_cluster_roundtrip(self, simple_system_8_days):
        """Single cluster should work correctly."""
        fs = simple_system_8_days
        # 8 days with 1 cluster = all days map to same cluster
        fs_clustered = fs.transform.cluster(n_clusters=1, cluster_duration='1D')

        ds = fs_clustered.to_dataset(include_solution=False)
        fs_restored = fx.FlowSystem.from_dataset(ds)

        assert fs_restored.clustering.n_clusters == 1
        assert fs_restored.cluster_weight.sum() == 8  # All 8 days in one cluster

    def test_max_clusters_roundtrip(self, simple_system_8_days):
        """Maximum clusters (one per day) should work correctly."""
        fs = simple_system_8_days
        # 8 days with 8 clusters = each day is its own cluster
        fs_clustered = fs.transform.cluster(n_clusters=8, cluster_duration='1D')

        ds = fs_clustered.to_dataset(include_solution=False)
        fs_restored = fx.FlowSystem.from_dataset(ds)

        assert fs_restored.clustering.n_clusters == 8
        # Each cluster represents 1 day
        np.testing.assert_array_equal(fs_restored.cluster_weight.values, np.ones(8))

    def test_clustering_preserves_component_labels(self, simple_system_8_days, solver_fixture):
        """Component labels should be preserved through clustering and expansion."""
        fs = simple_system_8_days
        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Roundtrip
        ds = fs_clustered.to_dataset(include_solution=True)
        fs_restored = fx.FlowSystem.from_dataset(ds)

        # Expand
        fs_expanded = fs_restored.transform.expand_solution()

        # Component labels should be preserved
        assert 'demand' in fs_expanded.components
        assert 'source' in fs_expanded.components


class TestSegmentationIO:
    """Tests for segmentation serialization and deserialization."""

    def test_segmentation_netcdf_roundtrip(self, simple_system_8_days, solver_fixture, tmp_path):
        """Test that segmented FlowSystem can be saved and loaded from netCDF."""
        fs = simple_system_8_days
        fs_segmented = fs.transform.cluster(n_clusters=2, cluster_duration='1D', n_segments=6)
        fs_segmented.optimize(solver_fixture)

        # Save to netCDF
        path = tmp_path / 'segmented.nc'
        fs_segmented.to_netcdf(path)

        # Load back
        fs_loaded = fx.FlowSystem.from_netcdf(path)

        # Verify segmentation is preserved
        assert fs_loaded.is_segmented is True
        assert isinstance(fs_loaded.timesteps, pd.RangeIndex)
        assert len(fs_loaded.timesteps) == 6  # n_segments
        assert fs_loaded.clustering is not None
        assert fs_loaded.clustering.result.cluster_structure.is_segmented is True
        assert fs_loaded.clustering.result.cluster_structure.n_segments == 6
        assert fs_loaded.clustering.result.cluster_structure.segment_timestep_counts is not None

    def test_segmentation_expand_after_roundtrip(self, simple_system_8_days, solver_fixture, tmp_path):
        """Test that expand_solution works after netCDF roundtrip for segmented systems."""
        fs = simple_system_8_days
        fs_segmented = fs.transform.cluster(n_clusters=2, cluster_duration='1D', n_segments=6)
        fs_segmented.optimize(solver_fixture)

        # Save and load
        path = tmp_path / 'segmented.nc'
        fs_segmented.to_netcdf(path)
        fs_loaded = fx.FlowSystem.from_netcdf(path)

        # Expand solution
        fs_expanded = fs_loaded.transform.expand_solution()

        # Verify expansion
        assert isinstance(fs_expanded.timesteps, pd.DatetimeIndex)
        assert len(fs_expanded.timesteps) == 8 * 24  # Original timesteps
        assert fs_expanded.solution is not None

    def test_segmentation_dataset_roundtrip(self, simple_system_8_days, solver_fixture):
        """Test that segmented FlowSystem can roundtrip through Dataset."""
        fs = simple_system_8_days
        fs_segmented = fs.transform.cluster(n_clusters=2, cluster_duration='1D', n_segments=4)
        fs_segmented.optimize(solver_fixture)

        # To dataset and back
        ds = fs_segmented.to_dataset(include_solution=True)
        fs_restored = fx.FlowSystem.from_dataset(ds)

        # Verify
        assert fs_restored.is_segmented is True
        assert fs_restored.clustering.result.cluster_structure.n_segments == 4
        segment_counts = fs_restored.clustering.result.cluster_structure.segment_timestep_counts
        assert segment_counts is not None
        # Sum of segment counts per cluster should equal 24 (timesteps per cluster)
        for c in range(2):
            assert int(segment_counts.sel(cluster=c).sum().values) == 24

    def test_segmentation_with_periods_scenarios_roundtrip(self, solver_fixture, tmp_path):
        """Test segmentation with periods and scenarios survives IO roundtrip."""
        # Create system with periods and scenarios
        timesteps = pd.date_range('2023-01-01', periods=8 * 24, freq='h')
        periods = pd.Index([2020, 2021], name='period')
        scenarios = pd.Index(['low', 'high'], name='scenario')
        demand = np.sin(np.linspace(0, 4 * np.pi, 8 * 24)) * 10 + 15

        fs = fx.FlowSystem(timesteps, periods=periods, scenarios=scenarios)
        fs.add_elements(
            fx.Bus('heat'),
            fx.Effect('costs', unit='EUR', is_objective=True, is_standard=True),
            fx.Sink('demand', inputs=[fx.Flow('in', bus='heat', fixed_relative_profile=demand, size=10)]),
            fx.Source('source', outputs=[fx.Flow('out', bus='heat', size=50, effects_per_flow_hour={'costs': 0.05})]),
        )

        # Cluster with segmentation
        fs_segmented = fs.transform.cluster(n_clusters=2, cluster_duration='1D', n_segments=6)
        fs_segmented.optimize(solver_fixture)

        # Verify multi-dimensional timestep_duration
        assert fs_segmented.timestep_duration is not None
        assert 'period' in fs_segmented.timestep_duration.dims
        assert 'scenario' in fs_segmented.timestep_duration.dims

        # Save and load
        path = tmp_path / 'segmented_multi.nc'
        fs_segmented.to_netcdf(path)
        fs_loaded = fx.FlowSystem.from_netcdf(path)

        # Verify everything is preserved
        assert fs_loaded.is_segmented is True
        assert fs_loaded.timestep_duration is not None
        assert fs_loaded.timestep_duration.shape == fs_segmented.timestep_duration.shape
        assert list(fs_loaded.periods) == list(fs_segmented.periods)
        assert list(fs_loaded.scenarios) == list(fs_segmented.scenarios)

        # Expand should work
        fs_expanded = fs_loaded.transform.expand_solution()
        assert len(fs_expanded.timesteps) == 8 * 24
        assert fs_expanded.solution is not None
