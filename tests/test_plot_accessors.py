"""Tests for the new plot accessor API."""

import plotly.graph_objects as go
import pytest
import xarray as xr

import flixopt as fx
from flixopt.plot_accessors import PlotResult

from .conftest import create_optimization_and_solve


@pytest.fixture
def results(simple_flow_system):
    """Create results from a solved optimization."""
    optimization = create_optimization_and_solve(
        simple_flow_system, fx.solvers.HighsSolver(0.01, 30), 'test_plot_accessors'
    )
    return optimization.results


class TestPlotResult:
    """Tests for PlotResult class."""

    def test_plot_result_attributes(self):
        """Test that PlotResult has data and figure attributes."""
        ds = xr.Dataset({'a': ('x', [1, 2, 3])})
        fig = go.Figure()
        result = PlotResult(data=ds, figure=fig)

        assert isinstance(result.data, xr.Dataset)
        assert isinstance(result.figure, go.Figure)

    def test_update_returns_self(self):
        """Test that update() returns self for chaining."""
        result = PlotResult(data=xr.Dataset(), figure=go.Figure())
        returned = result.update(title='Test')
        assert returned is result

    def test_update_traces_returns_self(self):
        """Test that update_traces() returns self for chaining."""
        result = PlotResult(data=xr.Dataset(), figure=go.Figure())
        returned = result.update_traces()
        assert returned is result

    def test_to_csv(self, tmp_path):
        """Test that to_csv() exports data correctly."""
        ds = xr.Dataset({'a': ('x', [1, 2, 3]), 'b': ('x', [4, 5, 6])})
        result = PlotResult(data=ds, figure=go.Figure())

        csv_path = tmp_path / 'test.csv'
        returned = result.to_csv(csv_path)

        assert returned is result
        assert csv_path.exists()

    def test_to_netcdf(self, tmp_path):
        """Test that to_netcdf() exports data correctly."""
        ds = xr.Dataset({'a': ('x', [1, 2, 3])})
        result = PlotResult(data=ds, figure=go.Figure())

        nc_path = tmp_path / 'test.nc'
        returned = result.to_netcdf(nc_path)

        assert returned is result
        assert nc_path.exists()

        # Verify contents
        loaded = xr.open_dataset(nc_path)
        xr.testing.assert_equal(loaded, ds)

    def test_to_html(self, tmp_path):
        """Test that to_html() exports figure correctly."""
        result = PlotResult(data=xr.Dataset(), figure=go.Figure())

        html_path = tmp_path / 'test.html'
        returned = result.to_html(html_path)

        assert returned is result
        assert html_path.exists()


class TestPlotAccessorBalance:
    """Tests for PlotAccessor.balance()."""

    def test_balance_returns_plot_result(self, results):
        """Test that balance() returns a PlotResult."""
        result = results.plot.balance('Boiler', show=False)
        assert isinstance(result, PlotResult)
        assert isinstance(result.data, xr.Dataset)
        assert isinstance(result.figure, go.Figure)

    def test_balance_data_has_expected_variables(self, results):
        """Test that balance data has expected structure."""
        result = results.plot.balance('Boiler', show=False)
        # Data should be an xarray Dataset with flow variables
        assert len(result.data.data_vars) > 0

    def test_balance_with_include_filter(self, results):
        """Test balance with include filter."""
        result = results.plot.balance('Boiler', include='Q_th', show=False)
        assert isinstance(result, PlotResult)
        # All variables should contain 'Q_th'
        for var in result.data.data_vars:
            assert 'Q_th' in var

    def test_balance_with_exclude_filter(self, results):
        """Test balance with exclude filter."""
        result = results.plot.balance('Boiler', exclude='Gas', show=False)
        assert isinstance(result, PlotResult)
        # No variables should contain 'Gas'
        for var in result.data.data_vars:
            assert 'Gas' not in var

    def test_balance_with_flow_hours(self, results):
        """Test balance with flow_hours unit."""
        result = results.plot.balance('Boiler', unit='flow_hours', show=False)
        assert isinstance(result, PlotResult)
        # Variable names should contain 'flow_hours' instead of 'flow_rate'
        for var in result.data.data_vars:
            assert 'flow_hours' in var or 'flow_rate' not in var

    def test_balance_with_aggregation(self, results):
        """Test balance with time aggregation."""
        result = results.plot.balance('Boiler', aggregate='sum', show=False)
        assert isinstance(result, PlotResult)
        # After aggregation, time dimension should not be present
        assert 'time' not in result.data.dims

    def test_balance_with_unit_flow_hours(self, results):
        """Test balance with flow_hours unit."""
        result = results.plot.balance('Boiler', unit='flow_hours', show=False)
        assert isinstance(result, PlotResult)


class TestPlotAccessorHeatmap:
    """Tests for PlotAccessor.heatmap()."""

    def test_heatmap_single_variable(self, results):
        """Test heatmap with single variable."""
        # Find a variable name
        var_names = list(results.solution.data_vars)
        time_vars = [v for v in var_names if 'time' in results.solution[v].dims]
        if time_vars:
            # Heatmap requires sufficient data for reshaping - test with reshape=None
            # to skip the time reshaping for short time series
            result = results.plot.heatmap(time_vars[0], reshape=None, show=False)
            assert isinstance(result, PlotResult)
            assert isinstance(result.data, xr.Dataset)

    def test_heatmap_multiple_variables(self, results):
        """Test heatmap with multiple variables."""
        var_names = list(results.solution.data_vars)
        time_vars = [v for v in var_names if 'time' in results.solution[v].dims][:2]
        if len(time_vars) >= 2:
            # Multi-variable heatmap with faceting by variable
            # Note: This requires proper time reshaping for the heatmap to work
            # For short time series, we skip this test
            import pytest

            pytest.skip('Multi-variable heatmap requires longer time series for proper reshaping')


class TestPlotAccessorStorage:
    """Tests for PlotAccessor.storage()."""

    def test_storage_returns_plot_result(self, results):
        """Test that storage() returns a PlotResult for storage components."""
        # Find storage component
        storage_comps = results.storages
        if storage_comps:
            storage_label = storage_comps[0].label
            result = results.plot.storage(storage_label, show=False)
            assert isinstance(result, PlotResult)
            assert isinstance(result.data, xr.Dataset)

    def test_storage_raises_for_non_storage(self, results):
        """Test that storage() raises ValueError for non-storage components."""
        with pytest.raises(ValueError, match='not a storage'):
            results.plot.storage('Boiler', show=False)


class TestPlotAccessorFlows:
    """Tests for PlotAccessor.flows()."""

    def test_flows_returns_plot_result(self, results):
        """Test that flows() returns a PlotResult."""
        result = results.plot.flows(show=False)
        assert isinstance(result, PlotResult)
        assert isinstance(result.data, xr.Dataset)

    def test_flows_with_component_filter(self, results):
        """Test flows with component filter."""
        result = results.plot.flows(component='Boiler', show=False)
        assert isinstance(result, PlotResult)

    def test_flows_with_flow_hours(self, results):
        """Test flows with flow_hours unit."""
        result = results.plot.flows(unit='flow_hours', show=False)
        assert isinstance(result, PlotResult)


class TestPlotAccessorCompare:
    """Tests for PlotAccessor.compare()."""

    def test_compare_returns_plot_result(self, results):
        """Test that compare() returns a PlotResult."""
        # Get actual component names from results
        component_names = list(results.components.keys())[:2]
        if len(component_names) >= 2:
            result = results.plot.compare(component_names, variable='flow_rate', show=False)
            assert isinstance(result, PlotResult)
            assert isinstance(result.data, xr.Dataset)


class TestPlotAccessorSankey:
    """Tests for PlotAccessor.sankey()."""

    def test_sankey_returns_plot_result(self, results):
        """Test that sankey() returns a PlotResult."""
        result = results.plot.sankey(show=False)
        assert isinstance(result, PlotResult)
        assert isinstance(result.data, xr.Dataset)

    def test_sankey_data_has_expected_coords(self, results):
        """Test that sankey data has expected coordinates."""
        result = results.plot.sankey(show=False)
        assert 'source' in result.data.coords
        assert 'target' in result.data.coords
        assert 'value' in result.data.data_vars


class TestPlotAccessorSize:
    """Tests for PlotAccessor.size()."""

    def test_size_returns_plot_result(self, results):
        """Test that size() returns a PlotResult."""
        result = results.plot.size(show=False)
        assert isinstance(result, PlotResult)
        assert isinstance(result.data, xr.Dataset)

    def test_size_with_component_filter(self, results):
        """Test size with component filter."""
        result = results.plot.size(component='Boiler', show=False)
        assert isinstance(result, PlotResult)
        # All variables should be from Boiler
        for var in result.data.data_vars:
            assert 'Boiler' in var


class TestPlotAccessorEffects:
    """Tests for PlotAccessor.effects()."""

    def test_effects_returns_plot_result(self, results):
        """Test that effects() returns a PlotResult."""
        # Default: aspect='total', all effects
        result = results.plot.effects(show=False)
        assert isinstance(result, PlotResult)
        assert isinstance(result.data, xr.Dataset)

    def test_effects_with_aspect(self, results):
        """Test effects with different aspects."""
        for aspect in ['total', 'temporal', 'periodic']:
            result = results.plot.effects(aspect=aspect, show=False)
            assert isinstance(result, PlotResult)

    def test_effects_with_specific_effect(self, results):
        """Test effects filtering to a specific effect."""
        # Get available effects
        effects_ds = results.effects_per_component
        available_effects = effects_ds['total'].coords['effect'].values.tolist()
        if available_effects:
            result = results.plot.effects(effect=available_effects[0], show=False)
            assert isinstance(result, PlotResult)

    def test_effects_by_component(self, results):
        """Test effects grouped by component."""
        result = results.plot.effects(by='component', show=False)
        assert isinstance(result, PlotResult)

    def test_effects_by_time(self, results):
        """Test effects grouped by time."""
        result = results.plot.effects(aspect='temporal', by='time', show=False)
        assert isinstance(result, PlotResult)


class TestElementPlotAccessor:
    """Tests for ElementPlotAccessor."""

    def test_element_balance(self, results):
        """Test element-level balance plot."""
        result = results['Boiler'].plot.balance(show=False)
        assert isinstance(result, PlotResult)
        assert isinstance(result.data, xr.Dataset)

    def test_element_heatmap(self, results):
        """Test element-level heatmap plot."""
        # Find a time-series variable for Boiler
        boiler_results = results['Boiler']
        time_vars = [v for v in boiler_results.solution.data_vars if 'time' in boiler_results.solution[v].dims]
        if time_vars:
            result = boiler_results.plot.heatmap(time_vars[0].split('|')[-1], show=False)
            assert isinstance(result, PlotResult)

    def test_element_storage(self, results):
        """Test element-level storage plot."""
        storage_comps = results.storages
        if storage_comps:
            storage = storage_comps[0]
            result = storage.plot.storage(show=False)
            assert isinstance(result, PlotResult)

    def test_element_storage_raises_for_non_storage(self, results):
        """Test that storage() raises for non-storage components."""
        with pytest.raises(ValueError, match='not a storage'):
            results['Boiler'].plot.storage(show=False)


class TestPlotAccessorVariable:
    """Tests for PlotAccessor.variable()."""

    def test_variable_returns_plot_result(self, results):
        """Test that variable() returns a PlotResult."""
        result = results.plot.variable('flow_rate', show=False)
        assert isinstance(result, PlotResult)
        assert isinstance(result.data, xr.Dataset)

    def test_variable_with_include_filter(self, results):
        """Test variable with include filter."""
        result = results.plot.variable('flow_rate', include='Boiler', show=False)
        assert isinstance(result, PlotResult)
        # All variables should be from Boiler
        for var in result.data.data_vars:
            assert 'Boiler' in var

    def test_variable_with_exclude_filter(self, results):
        """Test variable with exclude filter."""
        result = results.plot.variable('flow_rate', exclude='Boiler', show=False)
        assert isinstance(result, PlotResult)
        # No variables should be from Boiler
        for var in result.data.data_vars:
            assert 'Boiler' not in var

    def test_variable_with_aggregation(self, results):
        """Test variable with time aggregation."""
        result = results.plot.variable('flow_rate', aggregate='sum', show=False)
        assert isinstance(result, PlotResult)
        # After aggregation, time dimension should not be present
        assert 'time' not in result.data.dims


class TestPlotAccessorDurationCurve:
    """Tests for PlotAccessor.duration_curve()."""

    def test_duration_curve_returns_plot_result(self, results):
        """Test that duration_curve() returns a PlotResult."""
        # Find a time-series variable
        var_names = list(results.solution.data_vars)
        time_vars = [v for v in var_names if 'time' in results.solution[v].dims]
        if time_vars:
            result = results.plot.duration_curve(time_vars[0], show=False)
            assert isinstance(result, PlotResult)
            assert isinstance(result.data, xr.Dataset)

    def test_duration_curve_has_duration_dimension(self, results):
        """Test that duration curve data has duration dimension."""
        var_names = list(results.solution.data_vars)
        time_vars = [v for v in var_names if 'time' in results.solution[v].dims]
        if time_vars:
            result = results.plot.duration_curve(time_vars[0], show=False)
            # Should have duration dimension (not time)
            assert 'time' not in result.data.dims
            assert 'duration' in result.data.dims or 'duration_pct' in result.data.dims

    def test_duration_curve_normalized(self, results):
        """Test duration curve with normalized x-axis."""
        var_names = list(results.solution.data_vars)
        time_vars = [v for v in var_names if 'time' in results.solution[v].dims]
        if time_vars:
            result = results.plot.duration_curve(time_vars[0], normalize=True, show=False)
            assert isinstance(result, PlotResult)
            assert 'duration_pct' in result.data.dims

    def test_duration_curve_multiple_variables(self, results):
        """Test duration curve with multiple variables."""
        var_names = list(results.solution.data_vars)
        time_vars = [v for v in var_names if 'time' in results.solution[v].dims][:2]
        if len(time_vars) >= 2:
            result = results.plot.duration_curve(time_vars, show=False)
            assert isinstance(result, PlotResult)
            assert len(result.data.data_vars) == 2

    def test_duration_curve_sort_by(self, results):
        """Test duration curve with sort_by parameter."""
        import numpy as np

        var_names = list(results.solution.data_vars)
        time_vars = [v for v in var_names if 'time' in results.solution[v].dims][:2]
        if len(time_vars) >= 2:
            # Sort all variables by the first one
            result = results.plot.duration_curve(time_vars, sort_by=time_vars[0], show=False)
            assert isinstance(result, PlotResult)
            # The first variable should still be sorted descending (ignoring nan values)
            first_var_data = result.data[time_vars[0]].values
            # Filter out nan values for the comparison
            non_nan_data = first_var_data[~np.isnan(first_var_data)]
            assert all(non_nan_data[i] >= non_nan_data[i + 1] for i in range(len(non_nan_data) - 1))


class TestChaining:
    """Tests for method chaining."""

    def test_update_chain(self, results):
        """Test chaining update methods."""
        result = results.plot.balance('Boiler', show=False).update(title='Custom Title').update_traces()
        assert isinstance(result, PlotResult)
        assert result.figure.layout.title.text == 'Custom Title'

    def test_export_chain(self, results, tmp_path):
        """Test chaining export methods."""
        csv_path = tmp_path / 'data.csv'
        html_path = tmp_path / 'plot.html'

        result = results.plot.balance('Boiler', show=False).to_csv(csv_path).to_html(html_path)

        assert isinstance(result, PlotResult)
        assert csv_path.exists()
        assert html_path.exists()
