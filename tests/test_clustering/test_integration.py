"""Integration tests for flixopt.aggregation module with FlowSystem."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt import FlowSystem, Weights


class TestWeights:
    """Tests for Weights class."""

    def test_creation_via_flow_system(self):
        """Test Weights creation via FlowSystem.weights property."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))
        weights = fs.weights

        assert isinstance(weights, Weights)
        assert 'time' in weights.time.dims

    def test_time_property(self):
        """Test Weights.time returns timestep_duration."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))
        weights = fs.weights

        # For hourly data, timestep_duration is 1.0
        assert float(weights.time.mean()) == 1.0

    def test_cluster_property_non_clustered(self):
        """Test Weights.cluster returns 1.0 for non-clustered systems."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))
        weights = fs.weights

        assert weights.cluster == 1.0

    def test_temporal_dims_non_clustered(self):
        """Test temporal_dims is ['time'] for non-clustered systems."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))
        weights = fs.weights

        assert weights.temporal_dims == ['time']

    def test_temporal_property(self):
        """Test Weights.temporal returns time * cluster."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))
        weights = fs.weights

        expected = weights.time * weights.cluster
        xr.testing.assert_equal(weights.temporal, expected)

    def test_sum_temporal(self):
        """Test sum_temporal applies full temporal weighting (time * cluster) and sums."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=3, freq='h'))
        weights = fs.weights

        # Input is a rate (e.g., flow_rate in MW)
        data = xr.DataArray([10.0, 20.0, 30.0], dims=['time'], coords={'time': fs.timesteps})

        result = weights.sum_temporal(data)

        # For hourly non-clustered: temporal = time * cluster = 1.0 * 1.0 = 1.0
        # result = sum(data * temporal) = sum(data) = 60
        assert float(result.values) == 60.0


class TestFlowSystemWeightsProperty:
    """Tests for FlowSystem.weights property."""

    def test_weights_property_exists(self):
        """Test that FlowSystem has weights property."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))

        weights = fs.weights
        assert isinstance(weights, Weights)

    def test_weights_temporal_calculation(self):
        """Test that weights.temporal = timestep_duration * cluster_weight."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))

        weights = fs.weights
        expected = fs.timestep_duration * 1.0  # cluster is 1.0 for non-clustered

        np.testing.assert_array_almost_equal(weights.temporal.values, expected.values)

    def test_weights_with_cluster_weight(self):
        """Test weights property includes cluster_weight."""
        # Create FlowSystem with custom cluster_weight
        timesteps = pd.date_range('2024-01-01', periods=24, freq='h')
        cluster_weight = xr.DataArray(
            np.array([2.0] * 12 + [1.0] * 12),
            dims=['time'],
            coords={'time': timesteps},
        )

        fs = FlowSystem(timesteps=timesteps, cluster_weight=cluster_weight)

        weights = fs.weights

        # cluster property should return the cluster_weight
        xr.testing.assert_equal(weights.cluster, cluster_weight)

        # temporal = timestep_duration * cluster_weight
        # timestep_duration is 1h for all
        expected = 1.0 * cluster_weight
        np.testing.assert_array_almost_equal(weights.temporal.values, expected.values)


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
        grid_flow = Flow('grid_in', bus='electricity', size=100)
        demand_flow = Flow(
            'demand_out', bus='electricity', size=100, fixed_relative_profile=TimeSeriesData(demand_data / 100)
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


class TestClusteringModuleImports:
    """Tests for flixopt.clustering module imports."""

    def test_import_from_flixopt(self):
        """Test that clustering module can be imported from flixopt."""
        from flixopt import clustering

        assert hasattr(clustering, 'ClusterResult')
        assert hasattr(clustering, 'ClusterStructure')
        assert hasattr(clustering, 'Clustering')

    def test_create_cluster_structure_from_mapping_available(self):
        """Test that create_cluster_structure_from_mapping is available."""
        from flixopt.clustering import create_cluster_structure_from_mapping

        assert callable(create_cluster_structure_from_mapping)
