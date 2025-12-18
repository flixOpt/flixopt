"""Integration tests for flixopt.aggregation module with FlowSystem."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt import FlowSystem, TimeSeriesWeights


class TestTimeSeriesWeights:
    """Tests for TimeSeriesWeights class."""

    def test_creation(self):
        """Test TimeSeriesWeights creation."""
        temporal = xr.DataArray([1.0, 1.0, 1.0], dims=['time'])
        weights = TimeSeriesWeights(temporal=temporal)

        assert 'time' in weights.temporal.dims
        assert float(weights.temporal.sum().values) == 3.0

    def test_invalid_no_time_dim(self):
        """Test error when temporal has no time dimension."""
        temporal = xr.DataArray([1.0, 1.0], dims=['other'])

        with pytest.raises(ValueError, match='time'):
            TimeSeriesWeights(temporal=temporal)

    def test_sum_over_time(self):
        """Test sum_over_time convenience method."""
        temporal = xr.DataArray([2.0, 3.0, 1.0], dims=['time'], coords={'time': [0, 1, 2]})
        weights = TimeSeriesWeights(temporal=temporal)

        data = xr.DataArray([10.0, 20.0, 30.0], dims=['time'], coords={'time': [0, 1, 2]})
        result = weights.sum_over_time(data)

        # 10*2 + 20*3 + 30*1 = 20 + 60 + 30 = 110
        assert float(result.values) == 110.0

    def test_effective_objective(self):
        """Test effective_objective property."""
        temporal = xr.DataArray([1.0, 1.0], dims=['time'])
        objective = xr.DataArray([2.0, 2.0], dims=['time'])

        # Without override
        weights1 = TimeSeriesWeights(temporal=temporal)
        assert np.array_equal(weights1.effective_objective.values, temporal.values)

        # With override
        weights2 = TimeSeriesWeights(temporal=temporal, objective=objective)
        assert np.array_equal(weights2.effective_objective.values, objective.values)


class TestFlowSystemWeightsProperty:
    """Tests for FlowSystem.weights property."""

    def test_weights_property_exists(self):
        """Test that FlowSystem has weights property."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))

        weights = fs.weights
        assert isinstance(weights, TimeSeriesWeights)

    def test_weights_temporal_equals_aggregation_weight(self):
        """Test that weights.temporal equals aggregation_weight."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))

        weights = fs.weights
        aggregation_weight = fs.aggregation_weight

        np.testing.assert_array_almost_equal(weights.temporal.values, aggregation_weight.values)

    def test_weights_with_cluster_weight(self):
        """Test weights property includes cluster_weight."""
        # Create FlowSystem with custom cluster_weight
        timesteps = pd.date_range('2024-01-01', periods=24, freq='h')
        cluster_weight = np.array([2.0] * 12 + [1.0] * 12)  # First 12h weighted 2x

        fs = FlowSystem(timesteps=timesteps, cluster_weight=cluster_weight)

        weights = fs.weights

        # temporal = timestep_duration * cluster_weight
        # timestep_duration is 1h for all, so temporal = cluster_weight
        expected = 1.0 * cluster_weight
        np.testing.assert_array_almost_equal(weights.temporal.values, expected)


class TestAggregateMethod:
    """Tests for FlowSystem.transform.aggregate method."""

    def test_aggregate_method_exists(self):
        """Test that transform.aggregate method exists."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=48, freq='h'))

        assert hasattr(fs.transform, 'aggregate')
        assert callable(fs.transform.aggregate)

    def test_aggregate_tsam_delegates_to_cluster_reduce(self):
        """Test that aggregate with method='tsam' works."""
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

        # Should delegate to cluster_reduce - reduce 7 days to 2 representative days
        fs_agg = fs.transform.aggregate(
            method='tsam',
            n_representatives=2,
            reduce=True,
            cluster_duration='1D',
        )

        # Check that timesteps were reduced (from 168 hours to 48 hours = 2 days x 24 hours)
        assert len(fs_agg.timesteps) < len(fs.timesteps)
        assert len(fs_agg.timesteps) == 48  # 2 representative days x 24 hours


class TestSetAggregationMethod:
    """Tests for FlowSystem.transform.set_aggregation method."""

    def test_set_aggregation_method_exists(self):
        """Test that transform.set_aggregation method exists."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))

        assert hasattr(fs.transform, 'set_aggregation')
        assert callable(fs.transform.set_aggregation)

    def test_set_aggregation_raises_not_implemented(self):
        """Test that set_aggregation raises NotImplementedError for now."""
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=24, freq='h'))

        mapping = xr.DataArray(np.arange(24) % 4, dims=['original_time'])
        weights = xr.DataArray([6.0, 6.0, 6.0, 6.0], dims=['time'])

        # For now, should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            fs.transform.set_aggregation(
                timestep_mapping=mapping,
                weights=weights,
                reduce=True,
            )


class TestAggregationModuleImports:
    """Tests for flixopt.aggregation module imports."""

    def test_import_from_flixopt(self):
        """Test that aggregation module can be imported from flixopt."""
        from flixopt import aggregation

        assert hasattr(aggregation, 'AggregationResult')
        assert hasattr(aggregation, 'ClusterStructure')
        assert hasattr(aggregation, 'Aggregator')
        assert hasattr(aggregation, 'TSAMBackend')
        assert hasattr(aggregation, 'ManualBackend')

    def test_list_backends(self):
        """Test list_backends function."""
        from flixopt.aggregation import list_backends

        backends = list_backends()
        assert 'manual' in backends
        # 'tsam' may or may not be available depending on installation

    def test_get_backend(self):
        """Test get_backend function."""
        from flixopt.aggregation import ManualBackend, get_backend

        backend_cls = get_backend('manual')
        assert backend_cls is ManualBackend

    def test_get_backend_invalid(self):
        """Test get_backend raises for invalid backend."""
        from flixopt.aggregation import get_backend

        with pytest.raises(ValueError, match='Unknown backend'):
            get_backend('nonexistent')
