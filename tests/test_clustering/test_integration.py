"""Integration tests for flixopt.aggregation module with FlowSystem."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt import FlowSystem


class TestWeights:
    """Tests for FlowSystem.weights dict property."""

    def test_weights_is_dict(self):
        """Test weights returns a dict."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))
        weights = fs.weights

        assert isinstance(weights, dict)
        assert 'time' in weights

    def test_time_weight(self):
        """Test weights['time'] returns timestep_duration."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))
        weights = fs.weights

        # For hourly data, timestep_duration is 1.0
        assert float(weights['time'].mean()) == 1.0

    def test_cluster_not_in_weights_when_non_clustered(self):
        """Test weights doesn't have 'cluster' key for non-clustered systems."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))
        weights = fs.weights

        # Non-clustered: 'cluster' not in weights
        assert 'cluster' not in weights

    def test_temporal_dims_non_clustered(self):
        """Test temporal_dims is ['time'] for non-clustered systems."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))

        assert fs.temporal_dims == ['time']

    def test_temporal_weight(self):
        """Test temporal_weight returns time * cluster."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))

        expected = fs.weights['time'] * fs.weights.get('cluster', 1.0)
        xr.testing.assert_equal(fs.temporal_weight, expected)

    def test_sum_temporal(self):
        """Test sum_temporal applies full temporal weighting (time * cluster) and sums."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=3, freq='h'))

        # Input is a rate (e.g., flow_rate in MW)
        data = xr.DataArray([10.0, 20.0, 30.0], dims=['time'], coords={'time': fs.timesteps})

        result = fs.sum_temporal(data)

        # For hourly non-clustered: temporal = time * cluster = 1.0 * 1.0 = 1.0
        # result = sum(data * temporal) = sum(data) = 60
        assert float(result.values) == 60.0


class TestFlowSystemDimsIndexesWeights:
    """Tests for FlowSystem.dims, .indexes, .weights properties."""

    def test_dims_property(self):
        """Test that FlowSystem.dims returns active dimension names."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))

        dims = fs.dims
        assert dims == ['time']

    def test_indexes_property(self):
        """Test that FlowSystem.indexes returns active indexes."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))

        indexes = fs.indexes
        assert isinstance(indexes, dict)
        assert 'time' in indexes
        assert len(indexes['time']) == 24

    def test_weights_keys_match_dims(self):
        """Test that weights.keys() is subset of dims (only 'time' for simple case)."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))

        # For non-clustered, weights only has 'time'
        assert set(fs.weights.keys()) == {'time'}

    def test_temporal_weight_calculation(self):
        """Test that temporal_weight = timestep_duration * cluster_weight."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))

        expected = fs.timestep_duration * 1.0  # cluster is 1.0 for non-clustered

        np.testing.assert_array_almost_equal(fs.temporal_weight.values, expected.values)

    def test_weights_with_cluster_weight(self):
        """Test weights property includes cluster_weight when provided."""
        # Create FlowSystem with custom cluster_weight
        timesteps = pd.date_range('2024-01-01', periods=24, freq='h')
        cluster_weight = xr.DataArray(
            np.array([2.0] * 12 + [1.0] * 12),
            dims=['time'],
            coords={'time': timesteps},
        )

        fs = FlowSystem(timesteps=timesteps, cluster_weight=cluster_weight)

        weights = fs.weights

        # cluster weight should be in weights (FlowSystem has cluster_weight set)
        # But note: 'cluster' only appears in weights if clusters dimension exists
        # Since we didn't set clusters, 'cluster' won't be in weights
        # The cluster_weight is applied via temporal_weight
        assert 'cluster' not in weights  # No cluster dimension

        # temporal_weight = timestep_duration * cluster_weight
        # timestep_duration is 1h for all
        expected = 1.0 * cluster_weight
        np.testing.assert_array_almost_equal(fs.temporal_weight.values, expected.values)


class TestClusteringData:
    """Tests for FlowSystem.transform.clustering_data method."""

    def test_clustering_data_method_exists(self):
        """Test that transform.clustering_data method exists."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=48, freq='h'))

        assert hasattr(fs.transform, 'clustering_data')
        assert callable(fs.transform.clustering_data)

    def test_clustering_data_returns_dataset(self):
        """Test that clustering_data returns an xr.Dataset."""
        from flixopt import Bus, Flow, Sink, Source

        n_hours = 48
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=n_hours, freq='h'))

        # Add components with time-varying data
        demand_data = np.sin(np.linspace(0, 4 * np.pi, n_hours)) + 2
        bus = Bus('electricity')
        source = Source('grid', outputs=[Flow(bus='electricity', flow_id='grid_in', size=100)])
        sink = Sink(
            'demand',
            inputs=[Flow(bus='electricity', flow_id='demand_out', size=100, fixed_relative_profile=demand_data)],
        )
        fs.add_elements(source, sink, bus)

        clustering_data = fs.transform.clustering_data()

        assert isinstance(clustering_data, xr.Dataset)

    def test_clustering_data_contains_only_time_varying(self):
        """Test that clustering_data returns only time-varying data."""
        from flixopt import Bus, Flow, Sink, Source

        n_hours = 48
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=n_hours, freq='h'))

        # Add components with time-varying and constant data
        demand_data = np.sin(np.linspace(0, 4 * np.pi, n_hours)) + 2
        bus = Bus('electricity')
        source = Source('grid', outputs=[Flow(bus='electricity', flow_id='grid_in', size=100)])
        sink = Sink(
            'demand',
            inputs=[Flow(bus='electricity', flow_id='demand_out', size=100, fixed_relative_profile=demand_data)],
        )
        fs.add_elements(source, sink, bus)

        clustering_data = fs.transform.clustering_data()

        # Should contain the demand profile
        assert 'demand(demand_out)|fixed_relative_profile' in clustering_data.data_vars

        # All arrays should have 'time' dimension
        for var in clustering_data.data_vars:
            assert 'time' in clustering_data[var].dims

    def test_clustering_data_with_periods(self):
        """Test clustering_data with multi-period system."""
        from flixopt import Bus, Effect, Flow, Sink, Source

        n_hours = 48
        periods = pd.Index([2024, 2030], name='period')
        fs = FlowSystem(
            timesteps=pd.date_range('2024-01-01', periods=n_hours, freq='h'),
            periods=periods,
        )

        # Add components
        demand_data = xr.DataArray(
            np.random.rand(n_hours, 2),
            dims=['time', 'period'],
            coords={'time': fs.timesteps, 'period': periods},
        )
        bus = Bus('electricity')
        effect = Effect('costs', '€', is_objective=True)
        source = Source('grid', outputs=[Flow(bus='electricity', flow_id='grid_in', size=100)])
        sink = Sink(
            'demand',
            inputs=[Flow(bus='electricity', flow_id='demand_out', size=100, fixed_relative_profile=demand_data)],
        )
        fs.add_elements(source, sink, bus, effect)

        # Get data for specific period
        data_2024 = fs.transform.clustering_data(period=2024)

        # Should not have period dimension (it was selected)
        assert 'period' not in data_2024.dims

        # Get data for all periods
        data_all = fs.transform.clustering_data()
        assert 'period' in data_all.dims


class TestClusterMethod:
    """Tests for FlowSystem.transform.cluster method."""

    def test_cluster_method_exists(self):
        """Test that transform.cluster method exists."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=48, freq='h'))

        assert hasattr(fs.transform, 'cluster')
        assert callable(fs.transform.cluster)

    def test_cluster_reduces_timesteps(self):
        """Test that cluster reduces timesteps."""
        # This test requires tsam to be installed
        pytest.importorskip('tsam')
        from flixopt import Bus, Flow, Sink, Source
        from flixopt.core import TimeSeriesData

        # Create FlowSystem with 7 days of data (168 hours)
        n_hours = 168  # 7 days
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=n_hours, freq='h'))

        # Add some basic components with time series data
        demand_data = np.sin(np.linspace(0, 14 * np.pi, n_hours)) + 2  # Varying demand over 7 days
        bus = Bus('electricity')
        # Bus label is passed as string to Flow
        grid_flow = Flow(bus='electricity', flow_id='grid_in', size=100)
        demand_flow = Flow(
            bus='electricity', flow_id='demand_out', size=100, fixed_relative_profile=TimeSeriesData(demand_data / 100)
        )
        source = Source('grid', outputs=[grid_flow])
        sink = Sink('demand', inputs=[demand_flow])
        fs.add_elements(source, sink, bus)

        # Reduce 7 days to 2 representative days
        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
        )

        # Clustered FlowSystem has 2D structure: (cluster, time)
        # - timesteps: within-cluster time (24 hours)
        # - clusters: cluster indices (2 clusters)
        # Total effective timesteps = 2 * 24 = 48
        assert len(fs_clustered.timesteps) == 24  # Within-cluster time
        assert len(fs_clustered.clusters) == 2  # Number of clusters
        assert len(fs_clustered.timesteps) * len(fs_clustered.clusters) == 48


class TestClusterAdvancedOptions:
    """Tests for advanced clustering options."""

    @pytest.fixture
    def basic_flow_system(self):
        """Create a basic FlowSystem for testing."""
        pytest.importorskip('tsam')
        from flixopt import Bus, Flow, Sink, Source
        from flixopt.core import TimeSeriesData

        n_hours = 168  # 7 days
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=n_hours, freq='h'))

        demand_data = np.sin(np.linspace(0, 14 * np.pi, n_hours)) + 2
        bus = Bus('electricity')
        grid_flow = Flow(bus='electricity', flow_id='grid_in', size=100)
        demand_flow = Flow(
            bus='electricity', flow_id='demand_out', size=100, fixed_relative_profile=TimeSeriesData(demand_data / 100)
        )
        source = Source('grid', outputs=[grid_flow])
        sink = Sink('demand', inputs=[demand_flow])
        fs.add_elements(source, sink, bus)
        return fs

    def test_cluster_config_parameter(self, basic_flow_system):
        """Test that cluster config parameter works."""
        from tsam import ClusterConfig

        fs_clustered = basic_flow_system.transform.cluster(
            n_clusters=2, cluster_duration='1D', cluster=ClusterConfig(method='hierarchical')
        )
        assert len(fs_clustered.clusters) == 2

    def test_hierarchical_is_deterministic(self, basic_flow_system):
        """Test that hierarchical clustering (default) produces deterministic results."""
        fs1 = basic_flow_system.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs2 = basic_flow_system.transform.cluster(n_clusters=2, cluster_duration='1D')

        # Hierarchical clustering should produce identical cluster orders
        xr.testing.assert_equal(fs1.clustering.cluster_assignments, fs2.clustering.cluster_assignments)

    def test_metrics_available(self, basic_flow_system):
        """Test that clustering metrics are available after clustering."""
        fs_clustered = basic_flow_system.transform.cluster(n_clusters=2, cluster_duration='1D')

        assert fs_clustered.clustering.metrics is not None
        assert isinstance(fs_clustered.clustering.metrics, xr.Dataset)
        assert 'time_series' in fs_clustered.clustering.metrics.dims
        assert len(fs_clustered.clustering.metrics.data_vars) > 0

    def test_representation_method_parameter(self, basic_flow_system):
        """Test that representation method via ClusterConfig works."""
        from tsam import ClusterConfig

        fs_clustered = basic_flow_system.transform.cluster(
            n_clusters=2, cluster_duration='1D', cluster=ClusterConfig(representation='medoid')
        )
        assert len(fs_clustered.clusters) == 2

    def test_preserve_column_means_parameter(self, basic_flow_system):
        """Test that preserve_column_means parameter works via tsam_kwargs."""
        fs_clustered = basic_flow_system.transform.cluster(
            n_clusters=2, cluster_duration='1D', preserve_column_means=False
        )
        assert len(fs_clustered.clusters) == 2

    def test_tsam_kwargs_passthrough(self, basic_flow_system):
        """Test that additional kwargs are passed to tsam."""
        # preserve_column_means is a valid tsam.aggregate() parameter
        fs_clustered = basic_flow_system.transform.cluster(
            n_clusters=2, cluster_duration='1D', preserve_column_means=False
        )
        assert len(fs_clustered.clusters) == 2

    def test_metrics_with_periods(self):
        """Test that metrics have period dimension for multi-period FlowSystems."""
        pytest.importorskip('tsam')
        from flixopt import Bus, Flow, Sink, Source
        from flixopt.core import TimeSeriesData

        n_hours = 168  # 7 days
        fs = FlowSystem(
            timesteps=pd.date_range('2024-01-01', periods=n_hours, freq='h'),
            periods=pd.Index([2025, 2030], name='period'),
        )

        demand_data = np.sin(np.linspace(0, 14 * np.pi, n_hours)) + 2
        bus = Bus('electricity')
        grid_flow = Flow(bus='electricity', flow_id='grid_in', size=100)
        demand_flow = Flow(
            bus='electricity', flow_id='demand_out', size=100, fixed_relative_profile=TimeSeriesData(demand_data / 100)
        )
        source = Source('grid', outputs=[grid_flow])
        sink = Sink('demand', inputs=[demand_flow])
        fs.add_elements(source, sink, bus)

        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        # Metrics should have period dimension
        assert fs_clustered.clustering.metrics is not None
        assert 'period' in fs_clustered.clustering.metrics.dims
        assert len(fs_clustered.clustering.metrics.period) == 2


class TestClusteringModuleImports:
    """Tests for flixopt.clustering module imports."""

    def test_import_from_flixopt(self):
        """Test that clustering module can be imported from flixopt."""
        from flixopt import clustering

        assert hasattr(clustering, 'Clustering')
