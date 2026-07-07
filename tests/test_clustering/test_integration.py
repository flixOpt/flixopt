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


class TestClusterInputs:
    """Tests for FlowSystem.transform.cluster_inputs — variable discovery helper."""

    def _two_var_system(self, n_hours: int = 168):
        from flixopt import Bus, Effect, Flow, Sink, Source
        from flixopt.core import TimeSeriesData

        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=n_hours, freq='h'))
        varying = np.sin(np.linspace(0, 14 * np.pi, n_hours)) + 2
        constant = np.full(n_hours, 0.8)
        bus = Bus('electricity')
        fs.add_elements(
            Effect('costs', '€', is_standard=True, is_objective=True),
            Source('grid', outputs=[Flow('grid_in', bus='electricity', size=100)]),
            Sink(
                'demand',
                inputs=[
                    Flow(
                        'demand_out', bus='electricity', size=100, fixed_relative_profile=TimeSeriesData(varying / 100)
                    )
                ],
            ),
            Sink(
                'constant_load',
                inputs=[
                    Flow('constant_out', bus='electricity', size=50, fixed_relative_profile=TimeSeriesData(constant))
                ],
            ),
            bus,
        )
        return fs

    def test_returns_only_time_dim_vars(self):
        """cluster_inputs() returns every Dataset variable with a `time` dim."""
        fs = self._two_var_system()

        ds_time_vars = fs.transform.cluster_inputs()
        for var in ds_time_vars.data_vars:
            assert 'time' in ds_time_vars[var].dims, f'{var} should have a time dim'

    def test_includes_constants(self):
        """Constant time-series columns are included (they are passed to tsam too)."""
        fs = self._two_var_system()

        ds_time_vars = fs.transform.cluster_inputs()
        names = set(ds_time_vars.data_vars)
        assert 'demand(demand_out)|fixed_relative_profile' in names
        assert 'constant_load(constant_out)|fixed_relative_profile' in names

    def test_documented_weight_zero_pattern(self):
        """User pattern: enumerate columns and zero-weight everything except one.

        Regression test for the v6 migration recipe — users discovering clustering
        inputs via cluster_inputs() and feeding the names back into
        ClusterConfig(weights={...}) should not raise.
        """
        pytest.importorskip('tsam')
        from tsam import ClusterConfig

        fs = self._two_var_system()

        # Enumerate columns the way users are expected to
        ds_time_vars = fs.transform.cluster_inputs()
        target = 'demand(demand_out)|fixed_relative_profile'
        assert target in ds_time_vars.data_vars

        weights = {target: 1}
        weights.update({v: 0 for v in ds_time_vars.data_vars if v != target})

        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D', cluster=ClusterConfig(weights=weights))
        assert fs_clustered.clustering.n_clusters == 2

    def test_matches_what_cluster_sees(self):
        """The set of columns from cluster_inputs() == what cluster() passes to tsam.

        If this drifts, the recommended user pattern stops working — users would
        pass weights for variables tsam doesn't see, or miss ones it does.
        """
        fs = self._two_var_system()

        # cluster_inputs is documented as the source of truth
        discovered = set(fs.transform.cluster_inputs().data_vars)

        # Reproduce the cluster() filter inline
        ds = fs.to_dataset(include_solution=False)
        actual = {name for name in ds.data_vars if 'time' in ds[name].dims}

        assert discovered == actual


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

    def test_cluster_with_constant_columns(self):
        """Constant time-series columns must not break clustering.

        The pre-refactor path called ``drop_constant_arrays`` to avoid feeding
        zero-variance columns into tsam. The new tsam_xarray-backed path skips
        that filter, so this test guards against any future regression where a
        constant column makes tsam_xarray crash, normalize-divide-by-zero, or
        silently drop the column from the reduced FlowSystem.
        """
        pytest.importorskip('tsam')
        from flixopt import Bus, Flow, Sink, Source
        from flixopt.core import TimeSeriesData

        n_hours = 168
        fs = FlowSystem(timesteps=pd.date_range('2024-01-01', periods=n_hours, freq='h'))

        varying = np.sin(np.linspace(0, 14 * np.pi, n_hours)) + 2
        constant = np.full(n_hours, 0.8)

        bus = Bus('electricity')
        grid_flow = Flow('grid_in', bus='electricity', size=100)
        demand_flow = Flow(
            'demand_out', bus='electricity', size=100, fixed_relative_profile=TimeSeriesData(varying / 100)
        )
        constant_flow = Flow(
            'constant_out', bus='electricity', size=50, fixed_relative_profile=TimeSeriesData(constant)
        )
        fs.add_elements(
            Source('grid', outputs=[grid_flow]),
            Sink('demand', inputs=[demand_flow]),
            Sink('constant_load', inputs=[constant_flow]),
            bus,
        )

        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        # Reduced FlowSystem keeps the constant column
        ds = fs_clustered.to_dataset(include_solution=False)
        assert 'constant_load(constant_out)|fixed_relative_profile' in ds.data_vars

        # And the constant column stays constant after clustering
        constant_da = ds['constant_load(constant_out)|fixed_relative_profile']
        np.testing.assert_allclose(constant_da.values, 0.8)


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
        grid_flow = Flow('grid_in', bus='electricity', size=100)
        demand_flow = Flow(
            'demand_out', bus='electricity', size=100, fixed_relative_profile=TimeSeriesData(demand_data / 100)
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

    def test_unknown_weight_keys_raise(self, basic_flow_system):
        """Test that unknown keys in ClusterConfig.weights raise ValueError.

        tsam_xarray validates weight keys and raises ValueError for unknown coords.
        """
        from tsam import ClusterConfig

        # Get actual clustering column names
        ds = basic_flow_system.to_dataset(include_solution=False)
        real_columns = [n for n in ds.data_vars if 'time' in ds[n].dims]

        # Build weights with real keys + extra bogus keys
        weights = {col: 1.0 for col in real_columns}
        weights['nonexistent_variable'] = 0.5
        weights['another_missing_col'] = 0.3

        with pytest.raises(ValueError, match='unknown'):
            basic_flow_system.transform.cluster(
                n_clusters=2,
                cluster_duration='1D',
                cluster=ClusterConfig(weights=weights),
            )

    def test_unknown_weight_keys_raise_multiperiod(self):
        """Test that unknown weight keys raise ValueError in multi-period clustering."""
        pytest.importorskip('tsam')
        from tsam import ClusterConfig

        from flixopt import Bus, Flow, Sink, Source
        from flixopt.core import TimeSeriesData

        n_hours = 168  # 7 days
        fs = FlowSystem(
            timesteps=pd.date_range('2024-01-01', periods=n_hours, freq='h'),
            periods=pd.Index([2025, 2030], name='period'),
        )

        demand_data = np.sin(np.linspace(0, 14 * np.pi, n_hours)) + 2
        bus = Bus('electricity')
        grid_flow = Flow('grid_in', bus='electricity', size=100)
        demand_flow = Flow(
            'demand_out',
            bus='electricity',
            size=100,
            fixed_relative_profile=TimeSeriesData(demand_data / 100),
        )
        source = Source('grid', outputs=[grid_flow])
        sink = Sink('demand', inputs=[demand_flow])
        fs.add_elements(source, sink, bus)

        ds = fs.to_dataset(include_solution=False)
        weights = {n: 1.0 for n in ds.data_vars if 'time' in ds[n].dims}
        weights['nonexistent_period_var'] = 0.7

        with pytest.raises(ValueError, match='unknown'):
            fs.transform.cluster(
                n_clusters=2,
                cluster_duration='1D',
                cluster=ClusterConfig(weights=weights),
            )

    def test_valid_weight_keys_multiperiod(self):
        """Test that valid weight keys work in multi-period clustering.

        Each period is clustered independently; weights for valid columns
        must be filtered per slice so no extra keys leak through to tsam.
        """
        pytest.importorskip('tsam')
        from tsam import ClusterConfig

        from flixopt import Bus, Flow, Sink, Source
        from flixopt.core import TimeSeriesData

        n_hours = 168  # 7 days
        fs = FlowSystem(
            timesteps=pd.date_range('2024-01-01', periods=n_hours, freq='h'),
            periods=pd.Index([2025, 2030], name='period'),
        )

        demand_data = np.sin(np.linspace(0, 14 * np.pi, n_hours)) + 2
        bus = Bus('electricity')
        grid_flow = Flow('grid_in', bus='electricity', size=100)
        demand_flow = Flow(
            'demand_out',
            bus='electricity',
            size=100,
            fixed_relative_profile=TimeSeriesData(demand_data / 100),
        )
        source = Source('grid', outputs=[grid_flow])
        sink = Sink('demand', inputs=[demand_flow])
        fs.add_elements(source, sink, bus)

        ds = fs.to_dataset(include_solution=False)
        weights = {n: 1.0 for n in ds.data_vars if 'time' in ds[n].dims}

        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            cluster=ClusterConfig(weights=weights),
        )
        assert len(fs_clustered.clusters) == 2


class TestClusteringModuleImports:
    """Tests for flixopt.clustering module imports."""

    def test_import_from_flixopt(self):
        """Test that clustering module can be imported from flixopt."""
        from flixopt import clustering

        assert hasattr(clustering, 'Clustering')
