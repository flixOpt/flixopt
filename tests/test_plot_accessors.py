"""Tests for the new plot accessor API."""

import pandas as pd
import plotly.graph_objects as go
import pytest

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
        df = pd.DataFrame({'a': [1, 2, 3]})
        fig = go.Figure()
        result = PlotResult(data=df, figure=fig)

        assert isinstance(result.data, pd.DataFrame)
        assert isinstance(result.figure, go.Figure)

    def test_update_returns_self(self):
        """Test that update() returns self for chaining."""
        result = PlotResult(data=pd.DataFrame(), figure=go.Figure())
        returned = result.update(title='Test')
        assert returned is result

    def test_update_traces_returns_self(self):
        """Test that update_traces() returns self for chaining."""
        result = PlotResult(data=pd.DataFrame(), figure=go.Figure())
        returned = result.update_traces()
        assert returned is result

    def test_to_csv(self, tmp_path):
        """Test that to_csv() exports data correctly."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = PlotResult(data=df, figure=go.Figure())

        csv_path = tmp_path / 'test.csv'
        returned = result.to_csv(csv_path, index=False)

        assert returned is result
        assert csv_path.exists()

        # Verify contents
        loaded = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(loaded, df)

    def test_to_html(self, tmp_path):
        """Test that to_html() exports figure correctly."""
        result = PlotResult(data=pd.DataFrame(), figure=go.Figure())

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
        assert isinstance(result.data, pd.DataFrame)
        assert isinstance(result.figure, go.Figure)

    def test_balance_data_has_expected_columns(self, results):
        """Test that balance data has expected columns."""
        result = results.plot.balance('Boiler', show=False)
        assert 'flow' in result.data.columns
        assert 'value' in result.data.columns

    def test_balance_with_include_filter(self, results):
        """Test balance with include filter."""
        result = results.plot.balance('Boiler', include='Q_th', show=False)
        assert isinstance(result, PlotResult)
        # All flows should contain 'Q_th'
        for flow in result.data['flow'].unique():
            assert 'Q_th' in flow

    def test_balance_with_exclude_filter(self, results):
        """Test balance with exclude filter."""
        result = results.plot.balance('Boiler', exclude='Gas', show=False)
        assert isinstance(result, PlotResult)
        # No flows should contain 'Gas'
        for flow in result.data['flow'].unique():
            assert 'Gas' not in flow

    def test_balance_with_flow_hours(self, results):
        """Test balance with flow_hours unit."""
        result = results.plot.balance('Boiler', unit='flow_hours', show=False)
        assert isinstance(result, PlotResult)
        # Flow names should contain 'flow_hours' instead of 'flow_rate'
        flows = result.data['flow'].unique()
        for flow in flows:
            assert 'flow_hours' in flow or 'flow_rate' not in flow

    def test_balance_with_aggregation(self, results):
        """Test balance with time aggregation."""
        result = results.plot.balance('Boiler', aggregate='sum', show=False)
        assert isinstance(result, PlotResult)
        # After aggregation, time dimension should not be present
        # (or data should be much smaller)

    def test_balance_mode_options(self, results):
        """Test balance with different modes."""
        for mode in ['bar', 'line', 'area']:
            result = results.plot.balance('Boiler', mode=mode, show=False)
            assert isinstance(result, PlotResult)


class TestPlotAccessorHeatmap:
    """Tests for PlotAccessor.heatmap()."""

    def test_heatmap_single_variable(self, results):
        """Test heatmap with single variable."""
        # Find a variable name
        var_names = list(results.solution.data_vars)
        time_vars = [v for v in var_names if 'time' in results.solution[v].dims]
        if time_vars:
            result = results.plot.heatmap(time_vars[0], show=False)
            assert isinstance(result, PlotResult)

    def test_heatmap_multiple_variables(self, results):
        """Test heatmap with multiple variables."""
        var_names = list(results.solution.data_vars)
        time_vars = [v for v in var_names if 'time' in results.solution[v].dims][:2]
        if len(time_vars) >= 2:
            result = results.plot.heatmap(time_vars, show=False)
            assert isinstance(result, PlotResult)


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
        result = results.plot.compare(['Boiler', 'CHP'], variable='flow_rate', show=False)
        assert isinstance(result, PlotResult)


class TestPlotAccessorSankey:
    """Tests for PlotAccessor.sankey()."""

    def test_sankey_returns_plot_result(self, results):
        """Test that sankey() returns a PlotResult."""
        result = results.plot.sankey(show=False)
        assert isinstance(result, PlotResult)

    def test_sankey_data_has_expected_columns(self, results):
        """Test that sankey data has expected columns."""
        result = results.plot.sankey(show=False)
        assert 'source' in result.data.columns
        assert 'target' in result.data.columns
        assert 'value' in result.data.columns


class TestPlotAccessorEffects:
    """Tests for PlotAccessor.effects()."""

    def test_effects_returns_plot_result(self, results):
        """Test that effects() returns a PlotResult."""
        result = results.plot.effects('cost', show=False)
        assert isinstance(result, PlotResult)

    def test_effects_by_component(self, results):
        """Test effects grouped by component."""
        result = results.plot.effects('cost', by='component', show=False)
        assert isinstance(result, PlotResult)

    def test_effects_mode_options(self, results):
        """Test effects with different modes."""
        for mode in ['bar', 'pie']:
            result = results.plot.effects('cost', mode=mode, show=False)
            assert isinstance(result, PlotResult)


class TestElementPlotAccessor:
    """Tests for ElementPlotAccessor."""

    def test_element_balance(self, results):
        """Test element-level balance plot."""
        result = results['Boiler'].plot.balance(show=False)
        assert isinstance(result, PlotResult)

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

        result = results.plot.balance('Boiler', show=False).to_csv(csv_path, index=False).to_html(html_path)

        assert isinstance(result, PlotResult)
        assert csv_path.exists()
        assert html_path.exists()
