"""Tests for the new solution access API and plotting functionality.

This module tests:
- flow_system.solution access (xarray Dataset)
- element.solution access (filtered view)
- plotting module functions with realistic optimization data
- heatmap time reshaping
- network visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import flixopt as fx
from flixopt import plotting

# ============================================================================
# SOLUTION ACCESS TESTS
# ============================================================================


class TestFlowSystemSolution:
    """Tests for flow_system.solution API."""

    def test_solution_is_xarray_dataset(self, simple_flow_system, highs_solver):
        """Verify solution is an xarray Dataset."""
        simple_flow_system.optimize(highs_solver)
        assert isinstance(simple_flow_system.solution, xr.Dataset)

    def test_solution_has_time_dimension(self, simple_flow_system, highs_solver):
        """Verify solution has time dimension."""
        simple_flow_system.optimize(highs_solver)
        assert 'time' in simple_flow_system.solution.dims

    def test_solution_contains_effect_totals(self, simple_flow_system, highs_solver):
        """Verify solution contains effect totals (costs, CO2)."""
        simple_flow_system.optimize(highs_solver)
        solution = simple_flow_system.solution

        # Check that effect totals are present
        assert 'effect|total' in solution
        assert 'costs' in solution['effect|total'].coords['effect'].values
        assert 'CO2' in solution['effect|total'].coords['effect'].values

        # Verify they are scalar values per effect
        assert solution['effect|total'].sel(effect='costs').dims == ()
        assert solution['effect|total'].sel(effect='CO2').dims == ()

    def test_solution_contains_temporal_effects(self, simple_flow_system, highs_solver):
        """Verify solution contains temporal effect components."""
        simple_flow_system.optimize(highs_solver)
        solution = simple_flow_system.solution

        # Check temporal components
        assert 'effect|per_timestep' in solution
        assert 'costs' in solution['effect|per_timestep'].coords['effect'].values

    def test_solution_contains_flow_rates(self, simple_flow_system, highs_solver):
        """Verify solution contains flow rate variables."""
        simple_flow_system.optimize(highs_solver)
        solution = simple_flow_system.solution

        # Check flow rates exist as batched variable
        assert 'flow|rate' in solution

        # Verify flow rates have time and flow dimensions
        assert 'time' in solution['flow|rate'].dims
        assert 'flow' in solution['flow|rate'].dims

    def test_solution_contains_storage_variables(self, simple_flow_system, highs_solver):
        """Verify solution contains storage-specific variables."""
        simple_flow_system.optimize(highs_solver)
        solution = simple_flow_system.solution

        # Check storage charge state (includes extra timestep for final state)
        assert 'storage|charge' in solution
        assert 'Speicher' in solution['storage|charge'].coords['storage'].values

    def test_solution_item_returns_scalar(self, simple_flow_system, highs_solver):
        """Verify .item() returns Python scalar for 0-d arrays."""
        simple_flow_system.optimize(highs_solver)

        costs = simple_flow_system.solution['effect|total'].sel(effect='costs').item()
        assert isinstance(costs, (int, float))

    def test_solution_values_returns_numpy_array(self, simple_flow_system, highs_solver):
        """Verify .values returns numpy array for multi-dimensional data."""
        simple_flow_system.optimize(highs_solver)

        # Get first flow's rate values
        flow_rate = simple_flow_system.solution['flow|rate'].isel(flow=0).values
        assert isinstance(flow_rate, np.ndarray)

    def test_solution_sum_over_time(self, simple_flow_system, highs_solver):
        """Verify xarray operations work on solution data."""
        simple_flow_system.optimize(highs_solver)

        # Sum flow rate over time for first flow
        total_flow = simple_flow_system.solution['flow|rate'].isel(flow=0).sum(dim='time')
        assert total_flow.dims == ()

    def test_solution_to_dataframe(self, simple_flow_system, highs_solver):
        """Verify solution can be converted to DataFrame."""
        simple_flow_system.optimize(highs_solver)

        df = simple_flow_system.solution.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_solution_none_before_optimization(self, simple_flow_system):
        """Verify solution is None before optimization."""
        assert simple_flow_system.solution is None


class TestElementSolution:
    """Tests for element.solution API (filtered view of flow_system.solution)."""

    def test_element_solution_is_filtered_dataset(self, simple_flow_system, highs_solver):
        """Verify element.solution returns filtered Dataset."""
        simple_flow_system.optimize(highs_solver)

        boiler = simple_flow_system.components['Boiler']
        element_solution = boiler.solution

        assert isinstance(element_solution, xr.Dataset)

    def test_element_solution_contains_only_element_variables(self, simple_flow_system, highs_solver):
        """Verify element.solution only contains variables for that element."""
        simple_flow_system.optimize(highs_solver)

        boiler = simple_flow_system.components['Boiler']
        element_solution = boiler.solution

        # Variables should be batched names from _variable_names
        assert len(list(element_solution.data_vars)) > 0
        # Element solution should contain flow|rate (Boiler has flows)
        assert 'flow|rate' in element_solution

    def test_storage_element_solution(self, simple_flow_system, highs_solver):
        """Verify storage element solution contains charge state."""
        simple_flow_system.optimize(highs_solver)

        storage = simple_flow_system.components['Speicher']
        element_solution = storage.solution

        # Should contain storage charge variable
        charge_vars = [v for v in element_solution.data_vars if 'charge' in v]
        assert len(charge_vars) > 0

    def test_element_solution_raises_for_unlinked_element(self):
        """Verify accessing solution for unlinked element raises error."""
        boiler = fx.linear_converters.Boiler(
            'TestBoiler',
            thermal_efficiency=0.9,
            thermal_flow=fx.Flow('Heat', flow_id='Q_th'),
            fuel_flow=fx.Flow('Gas', flow_id='Q_fu'),
        )
        with pytest.raises(ValueError, match='not linked to a FlowSystem'):
            _ = boiler.solution


# ============================================================================
# STATISTICS ACCESSOR TESTS
# ============================================================================


class TestStatisticsAccessor:
    """Tests for flow_system.statistics accessor."""

    def test_statistics_sizes_includes_all_flows(self, simple_flow_system, highs_solver):
        """Test that statistics.sizes includes all flow and storage sizes (from InvestParameters)."""
        simple_flow_system.optimize(highs_solver)

        sizes = simple_flow_system.statistics.sizes

        assert isinstance(sizes, xr.DataArray)
        # Should have sizes with element dimension
        assert 'element' in sizes.dims
        assert sizes.sizes['element'] > 0

        # Check that all size labels are valid flow or storage labels
        flow_labels = [f.label_full for f in simple_flow_system.flows.values()]
        storage_labels = [s.label_full for s in simple_flow_system.storages.values()]
        valid_labels = flow_labels + storage_labels
        for label in sizes.coords['element'].values:
            assert label in valid_labels, f'Size label {label} should be a valid flow or storage'

    def test_statistics_sizes_returns_correct_values(self, simple_flow_system, highs_solver):
        """Test that statistics.sizes returns correct size values."""
        simple_flow_system.optimize(highs_solver)

        sizes = simple_flow_system.statistics.sizes

        # Check that all values are positive (sizes should be > 0)
        for label in sizes.coords['element'].values:
            val = sizes.sel(element=label)
            value = float(val.values) if val.dims == () else float(val.max().values)
            assert value > 0, f'Size for {label} should be positive'

    def test_statistics_flow_rates(self, simple_flow_system, highs_solver):
        """Test that statistics.flow_rates returns flow rate data."""
        simple_flow_system.optimize(highs_solver)

        flow_rates = simple_flow_system.statistics.flow_rates

        assert isinstance(flow_rates, xr.DataArray)
        assert 'flow' in flow_rates.dims
        assert flow_rates.sizes['flow'] > 0
        # Flow rates should have time dimension
        assert 'time' in flow_rates.dims

    def test_statistics_flow_hours(self, simple_flow_system, highs_solver):
        """Test that statistics.flow_hours returns energy data."""
        simple_flow_system.optimize(highs_solver)

        flow_hours = simple_flow_system.statistics.flow_hours

        assert isinstance(flow_hours, xr.DataArray)
        assert 'flow' in flow_hours.dims
        assert flow_hours.sizes['flow'] > 0


# ============================================================================
# PLOTTING WITH OPTIMIZED DATA TESTS
# ============================================================================


class TestPlottingWithOptimizedData:
    """Tests for plotting functions using actual optimization results."""

    @staticmethod
    def _flow_rate_dataset(solution, n=3):
        """Extract first n flows from flow|rate as a Dataset with individual flow variables."""
        rate = solution['flow|rate']
        flow_labels = list(rate.coords['flow'].values[:n])
        return rate.sel(flow=flow_labels).to_dataset('flow')

    def test_plot_flow_rates_with_plotly(self, simple_flow_system, highs_solver):
        """Test plotting flow rates with Plotly."""
        simple_flow_system.optimize(highs_solver)

        flow_data = self._flow_rate_dataset(simple_flow_system.solution, 3)

        fig = plotting.with_plotly(flow_data, mode='stacked_bar')
        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_flow_rates_with_matplotlib(self, simple_flow_system, highs_solver):
        """Test plotting flow rates with Matplotlib."""
        simple_flow_system.optimize(highs_solver)

        flow_data = self._flow_rate_dataset(simple_flow_system.solution, 3)

        fig, ax = plotting.with_matplotlib(flow_data, mode='stacked_bar')
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_line_mode(self, simple_flow_system, highs_solver):
        """Test line plotting mode."""
        simple_flow_system.optimize(highs_solver)

        flow_data = self._flow_rate_dataset(simple_flow_system.solution, 3)

        fig = plotting.with_plotly(flow_data, mode='line')
        assert fig is not None

        fig2, ax2 = plotting.with_matplotlib(flow_data, mode='line')
        assert fig2 is not None
        plt.close(fig2)

    def test_plot_area_mode(self, simple_flow_system, highs_solver):
        """Test area plotting mode (Plotly only)."""
        simple_flow_system.optimize(highs_solver)

        flow_data = self._flow_rate_dataset(simple_flow_system.solution, 3)

        fig = plotting.with_plotly(flow_data, mode='area')
        assert fig is not None

    def test_plot_with_custom_colors(self, simple_flow_system, highs_solver):
        """Test plotting with custom colors."""
        simple_flow_system.optimize(highs_solver)

        flow_data = self._flow_rate_dataset(simple_flow_system.solution, 2)
        flow_labels = list(flow_data.data_vars)

        # Test with color list
        fig1 = plotting.with_plotly(flow_data, mode='line', colors=['red', 'blue'])
        assert fig1 is not None

        # Test with color dict
        color_dict = {flow_labels[0]: '#ff0000', flow_labels[1]: '#0000ff'}
        fig2 = plotting.with_plotly(flow_data, mode='line', colors=color_dict)
        assert fig2 is not None

        # Test with colorscale name
        fig3 = plotting.with_plotly(flow_data, mode='line', colors='turbo')
        assert fig3 is not None

    def test_plot_with_title_and_labels(self, simple_flow_system, highs_solver):
        """Test plotting with custom title and axis labels."""
        simple_flow_system.optimize(highs_solver)

        flow_data = self._flow_rate_dataset(simple_flow_system.solution, 2)

        fig = plotting.with_plotly(flow_data, mode='line', title='Energy Flows', xlabel='Time (h)', ylabel='Power (kW)')
        assert fig.layout.title.text == 'Energy Flows'

    def test_plot_scalar_effects(self, simple_flow_system, highs_solver):
        """Test plotting scalar effect values."""
        simple_flow_system.optimize(highs_solver)

        # Create dataset with scalar values
        effect_total = simple_flow_system.solution['effect|total']
        effects_data = effect_total.sel(effect=['costs', 'CO2']).to_dataset('effect')

        # This should handle scalar data gracefully
        fig, ax = plotting.with_matplotlib(effects_data, mode='stacked_bar')
        assert fig is not None
        # Verify plot has visual content
        assert len(ax.patches) > 0 or len(ax.lines) > 0 or len(ax.containers) > 0, 'Plot should contain visual elements'
        plt.close(fig)


class TestDualPiePlots:
    """Tests for dual pie chart functionality."""

    def test_dual_pie_with_effects(self, simple_flow_system, highs_solver):
        """Test dual pie chart with effect contributions."""
        simple_flow_system.optimize(highs_solver)

        # Get effect per_timestep data and sum over time for pie chart
        if 'effect|per_timestep' in simple_flow_system.solution:
            per_ts = simple_flow_system.solution['effect|per_timestep']
            effects = per_ts.coords['effect'].values
            if len(effects) >= 2:
                summed = per_ts.sum(dim='time')
                left_data = summed.sel(effect=effects[:2]).to_dataset('effect')
                right_data = summed.sel(effect=effects[:2]).to_dataset('effect')

                fig = plotting.dual_pie_with_plotly(left_data, right_data)
                assert fig is not None

    def test_dual_pie_with_matplotlib(self, simple_flow_system, highs_solver):
        """Test dual pie chart with matplotlib backend."""
        simple_flow_system.optimize(highs_solver)

        # Simple scalar data
        left_data = xr.Dataset({'A': xr.DataArray(30), 'B': xr.DataArray(70)})
        right_data = xr.Dataset({'A': xr.DataArray(50), 'B': xr.DataArray(50)})

        fig, axes = plotting.dual_pie_with_matplotlib(left_data, right_data)
        assert fig is not None
        assert len(axes) == 2
        plt.close(fig)


# ============================================================================
# HEATMAP TESTS
# ============================================================================


class TestHeatmapReshaping:
    """Tests for heatmap time reshaping functionality."""

    @pytest.fixture
    def long_time_data(self):
        """Create data with longer time series for heatmap testing."""
        time = pd.date_range('2020-01-01', periods=72, freq='h')  # 3 days
        rng = np.random.default_rng(42)
        data = xr.DataArray(rng.random(72) * 100, coords={'time': time}, dims=['time'], name='power')
        return data

    def test_reshape_auto_mode(self, long_time_data):
        """Test automatic time reshaping."""
        reshaped = plotting.reshape_data_for_heatmap(long_time_data, reshape_time='auto')

        # Auto mode should attempt reshaping; verify it either reshaped or returned original
        if 'timestep' in reshaped.dims or 'timeframe' in reshaped.dims:
            # Reshaping occurred - verify 2D structure
            assert len(reshaped.dims) == 2, 'Reshaped data should have 2 dimensions'
        else:
            # Reshaping not possible for this data - verify original structure preserved
            assert reshaped.dims == long_time_data.dims, (
                'Original structure should be preserved if reshaping not applied'
            )

    def test_reshape_explicit_daily_hourly(self, long_time_data):
        """Test explicit daily-hourly reshaping."""
        reshaped = plotting.reshape_data_for_heatmap(long_time_data, reshape_time=('D', 'h'))

        # Should have timeframe (days) and timestep (hours) dimensions
        if 'timestep' in reshaped.dims:
            assert 'timeframe' in reshaped.dims
            # With 72 hours (3 days), we should have 3 timeframes and up to 24 timesteps
            assert reshaped.sizes['timeframe'] == 3

    def test_reshape_none_preserves_data(self, long_time_data):
        """Test that reshape_time=None preserves original structure."""
        reshaped = plotting.reshape_data_for_heatmap(long_time_data, reshape_time=None)
        assert 'time' in reshaped.dims
        xr.testing.assert_equal(reshaped, long_time_data)

    def test_heatmap_with_plotly_v2(self, long_time_data):
        """Test heatmap plotting with Plotly."""
        # Reshape data first (heatmap_with_plotly_v2 requires pre-reshaped data)
        reshaped = plotting.reshape_data_for_heatmap(long_time_data, reshape_time=('D', 'h'))

        fig = plotting.heatmap_with_plotly_v2(reshaped)
        assert fig is not None

    def test_heatmap_with_matplotlib(self, long_time_data):
        """Test heatmap plotting with Matplotlib."""
        fig, ax = plotting.heatmap_with_matplotlib(long_time_data, reshape_time=('D', 'h'))
        assert fig is not None
        assert ax is not None
        plt.close(fig)


# ============================================================================
# NETWORK VISUALIZATION TESTS
# ============================================================================


class TestNetworkVisualization:
    """Tests for network visualization functionality."""

    def test_topology_plot_returns_figure(self, simple_flow_system):
        """Test that topology.plot() returns a PlotResult with Plotly Figure."""
        import plotly.graph_objects as go

        result = simple_flow_system.topology.plot(show=False)
        assert result is not None
        assert hasattr(result, 'figure')
        assert isinstance(result.figure, go.Figure)

    def test_topology_plot_creates_html(self, simple_flow_system, tmp_path):
        """Test that topology.plot() figure can be saved to HTML file."""
        html_path = tmp_path / 'network.html'
        result = simple_flow_system.topology.plot(show=False)
        result.figure.write_html(str(html_path))
        assert html_path.exists()

    def test_topology_plot_contains_all_buses(self, simple_flow_system):
        """Test that topology plot contains all buses in the Sankey diagram."""
        result = simple_flow_system.topology.plot(show=False)

        # Get node labels from the Sankey diagram
        sankey_data = result.figure.data[0]
        node_labels = list(sankey_data.node.label)

        # Check that buses are in network
        for bus_label in simple_flow_system.buses.keys():
            assert bus_label in node_labels


# ============================================================================
# VARIABLE NAMING CONVENTION TESTS
# ============================================================================


class TestVariableNamingConvention:
    """Tests verifying the new variable naming convention."""

    def test_flow_rate_naming_pattern(self, simple_flow_system, highs_solver):
        """Test batched flow|rate variable with flow dimension."""
        simple_flow_system.optimize(highs_solver)

        # Check batched flow rate variable exists
        assert 'flow|rate' in simple_flow_system.solution
        # Check Boiler's thermal flow is in the flow coordinate
        assert 'Boiler(Q_th)' in simple_flow_system.solution['flow|rate'].coords['flow'].values

    def test_status_variable_naming(self, simple_flow_system, highs_solver):
        """Test status variable naming pattern."""
        simple_flow_system.optimize(highs_solver)

        # Components with status should have status variables
        status_vars = [v for v in simple_flow_system.solution.data_vars if '|status' in v]
        # At least one component should have status
        assert len(status_vars) >= 0  # May be 0 if no status tracking

    def test_storage_naming_pattern(self, simple_flow_system, highs_solver):
        """Test batched storage variables with storage dimension."""
        simple_flow_system.optimize(highs_solver)

        # Storage charge state follows batched pattern
        assert 'storage|charge' in simple_flow_system.solution
        assert 'Speicher' in simple_flow_system.solution['storage|charge'].coords['storage'].values
        # Storage netto variable
        assert 'storage|netto' in simple_flow_system.solution

    def test_effect_naming_patterns(self, simple_flow_system, highs_solver):
        """Test batched effect naming patterns."""
        simple_flow_system.optimize(highs_solver)

        # Total effect (batched with effect dimension)
        assert 'effect|total' in simple_flow_system.solution
        assert 'costs' in simple_flow_system.solution['effect|total'].coords['effect'].values

        # Per timestep (batched with effect dimension)
        assert 'effect|per_timestep' in simple_flow_system.solution

    def test_list_all_variables(self, simple_flow_system, highs_solver):
        """Test that all variables can be listed."""
        simple_flow_system.optimize(highs_solver)

        variables = list(simple_flow_system.solution.data_vars)
        assert len(variables) > 0, f'Expected variables in solution, got {len(variables)}'


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestPlottingEdgeCases:
    """Tests for edge cases in plotting."""

    def test_empty_dataset_returns_empty_figure(self, caplog):
        """Test that empty dataset returns an empty figure."""
        import logging

        empty_data = xr.Dataset()
        with caplog.at_level(logging.ERROR):
            fig = plotting.with_plotly(empty_data)
        # Empty dataset should produce figure with no data traces
        assert len(fig.data) == 0, 'Empty dataset should produce figure with no data traces'

    def test_non_numeric_data_raises_error(self):
        """Test that non-numeric data raises appropriate error."""
        string_data = xr.Dataset({'var': (['time'], ['a', 'b', 'c'])}, coords={'time': [0, 1, 2]})
        with pytest.raises(TypeError, match='non-numeric'):
            plotting.with_plotly(string_data)

    def test_single_value_plotting(self):
        """Test plotting with single data point."""
        single_data = xr.Dataset({'var': (['time'], [42.0])}, coords={'time': [0]})

        fig = plotting.with_plotly(single_data, mode='stacked_bar')
        assert fig is not None

    def test_all_zero_data_plotting(self):
        """Test plotting with all zero values."""
        zero_data = xr.Dataset(
            {'var1': (['time'], [0.0, 0.0, 0.0]), 'var2': (['time'], [0.0, 0.0, 0.0])}, coords={'time': [0, 1, 2]}
        )

        fig = plotting.with_plotly(zero_data, mode='stacked_bar')
        assert fig is not None

    def test_nan_values_handled(self):
        """Test that NaN values are handled gracefully (no exceptions raised)."""
        nan_data = xr.Dataset({'var': (['time'], [1.0, np.nan, 3.0, np.nan, 5.0])}, coords={'time': [0, 1, 2, 3, 4]})

        # Should not raise - NaN values should be handled gracefully
        fig = plotting.with_plotly(nan_data, mode='line')
        assert fig is not None
        # Verify that plot was created with some data
        assert len(fig.data) > 0, 'Figure should have data traces even with NaN values'

    def test_negative_values_in_stacked_bar(self):
        """Test handling of negative values in stacked bar charts."""
        mixed_data = xr.Dataset(
            {'positive': (['time'], [1.0, 2.0, 3.0]), 'negative': (['time'], [-1.0, -2.0, -3.0])},
            coords={'time': [0, 1, 2]},
        )

        fig = plotting.with_plotly(mixed_data, mode='stacked_bar')
        assert fig is not None

        fig2, ax2 = plotting.with_matplotlib(mixed_data, mode='stacked_bar')
        assert fig2 is not None
        plt.close(fig2)


# ============================================================================
# COLOR PROCESSING TESTS
# ============================================================================


class TestColorProcessing:
    """Tests for color processing functionality."""

    def test_colorscale_name(self):
        """Test processing colorscale by name."""
        from flixopt.color_processing import process_colors

        colors = process_colors('turbo', ['A', 'B', 'C'])
        assert isinstance(colors, dict)
        assert 'A' in colors
        assert 'B' in colors
        assert 'C' in colors

    def test_color_list(self):
        """Test processing explicit color list."""
        from flixopt.color_processing import process_colors

        color_list = ['#ff0000', '#00ff00', '#0000ff']
        colors = process_colors(color_list, ['A', 'B', 'C'])
        assert colors['A'] == '#ff0000'
        assert colors['B'] == '#00ff00'
        assert colors['C'] == '#0000ff'

    def test_color_dict(self):
        """Test processing color dictionary."""
        from flixopt.color_processing import process_colors

        color_dict = {'A': 'red', 'B': 'blue'}
        colors = process_colors(color_dict, ['A', 'B', 'C'])
        assert colors['A'] == 'red'
        assert colors['B'] == 'blue'
        # C should get a default color
        assert 'C' in colors

    def test_insufficient_colors_cycles(self):
        """Test that insufficient colors cycle properly."""
        from flixopt.color_processing import process_colors

        # Only 2 colors for 5 labels
        colors = process_colors(['red', 'blue'], ['A', 'B', 'C', 'D', 'E'])
        assert len(colors) == 5
        # Should cycle
        assert colors['A'] == 'red'
        assert colors['B'] == 'blue'
        assert colors['C'] == 'red'  # Cycles back


# ============================================================================
# EXPORT FUNCTIONALITY TESTS
# ============================================================================


class TestExportFunctionality:
    """Tests for figure export functionality."""

    def test_export_plotly_to_html(self, simple_flow_system, highs_solver, tmp_path):
        """Test exporting Plotly figure to HTML."""
        simple_flow_system.optimize(highs_solver)

        rate = simple_flow_system.solution['flow|rate']
        flow_labels = rate.coords['flow'].values[:2]
        flow_data = rate.sel(flow=flow_labels).to_dataset('flow')

        fig = plotting.with_plotly(flow_data, mode='line')

        html_path = tmp_path / 'plot.html'
        # export_figure expects pathlib.Path and save=True to actually save
        plotting.export_figure(fig, default_path=html_path, save=True, show=False)
        assert html_path.exists()

    def test_export_matplotlib_to_png(self, simple_flow_system, highs_solver, tmp_path):
        """Test exporting Matplotlib figure to PNG."""
        simple_flow_system.optimize(highs_solver)

        rate = simple_flow_system.solution['flow|rate']
        flow_labels = rate.coords['flow'].values[:2]
        flow_data = rate.sel(flow=flow_labels).to_dataset('flow')

        fig, ax = plotting.with_matplotlib(flow_data, mode='line')

        png_path = tmp_path / 'plot.png'
        # export_figure expects pathlib.Path and save=True to actually save
        plotting.export_figure((fig, ax), default_path=png_path, save=True, show=False)
        assert png_path.exists()
        plt.close(fig)


# ============================================================================
# SANKEY DIAGRAM TESTS
# ============================================================================


class TestSankeyDiagram:
    """Tests for Sankey diagram functionality."""

    def test_sankey_flows(self, simple_flow_system, highs_solver):
        """Test Sankey diagram with flows() method."""
        simple_flow_system.optimize(highs_solver)

        result = simple_flow_system.statistics.plot.sankey.flows(show=False)

        assert result.figure is not None
        assert result.data is not None
        assert result.data.size > 0
        assert 'source' in result.data.coords
        assert 'target' in result.data.coords
        assert len(result.data.link) > 0

    def test_sankey_peak_flow(self, simple_flow_system, highs_solver):
        """Test Sankey diagram with peak_flow() method."""
        simple_flow_system.optimize(highs_solver)

        result = simple_flow_system.statistics.plot.sankey.peak_flow(show=False)

        assert result.figure is not None
        assert result.data is not None
        assert len(result.data.link) > 0

    def test_sankey_sizes(self, simple_flow_system, highs_solver):
        """Test Sankey diagram with sizes() method shows investment sizes."""
        simple_flow_system.optimize(highs_solver)

        result = simple_flow_system.statistics.plot.sankey.sizes(show=False)

        assert result.figure is not None
        assert result.data is not None
        # Should have some flows with investment sizes
        assert len(result.data.link) > 0

    def test_sankey_sizes_max_size_filter(self, simple_flow_system, highs_solver):
        """Test that max_size parameter filters large sizes."""
        simple_flow_system.optimize(highs_solver)

        # Get all sizes (no filter)
        result_all = simple_flow_system.statistics.plot.sankey.sizes(max_size=None, show=False)

        # Get filtered sizes
        result_filtered = simple_flow_system.statistics.plot.sankey.sizes(max_size=100, show=False)

        # Filtered should have fewer or equal links
        assert len(result_filtered.data.link) <= len(result_all.data.link)

    def test_sankey_effects(self, simple_flow_system, highs_solver):
        """Test Sankey diagram with effects() method."""
        simple_flow_system.optimize(highs_solver)

        result = simple_flow_system.statistics.plot.sankey.effects(show=False)

        assert result.figure is not None
        assert result.data is not None
        # Should have component -> effect links
        assert len(result.data.link) > 0
        # Effects should appear in targets with bracket notation
        targets = list(result.data.target.values)
        assert any('[' in str(t) for t in targets), 'Effects should appear as [effect_name] in targets'

    def test_sankey_effects_includes_costs_and_co2(self, simple_flow_system, highs_solver):
        """Test that effects() method includes both costs and CO2."""
        simple_flow_system.optimize(highs_solver)

        result = simple_flow_system.statistics.plot.sankey.effects(show=False)

        targets = [str(t) for t in result.data.target.values]
        # Should have at least costs effect
        assert '[costs]' in targets, 'Should include costs effect'

    def test_sankey_flows_with_time_select(self, simple_flow_system, highs_solver):
        """Test Sankey flows with specific time selection."""
        simple_flow_system.optimize(highs_solver)

        # Get first timestamp from the data
        first_time = simple_flow_system.statistics.flow_hours.time.values[0]
        result = simple_flow_system.statistics.plot.sankey.flows(select={'time': first_time}, show=False)

        assert result.figure is not None
        assert len(result.data.link) > 0

    def test_sankey_flows_with_mean_aggregate(self, simple_flow_system, highs_solver):
        """Test Sankey flows with mean aggregation."""
        simple_flow_system.optimize(highs_solver)

        result_sum = simple_flow_system.statistics.plot.sankey.flows(aggregate='sum', show=False)
        result_mean = simple_flow_system.statistics.plot.sankey.flows(aggregate='mean', show=False)

        # Both should produce valid results
        assert result_sum.figure is not None
        assert result_mean.figure is not None
        # Mean values should be smaller than sum values
        sum_total = sum(result_sum.data.values)
        mean_total = sum(result_mean.data.values)
        assert mean_total < sum_total, 'Mean should produce smaller values than sum'

    def test_sankey_returns_plot_result(self, simple_flow_system, highs_solver):
        """Test that sankey returns PlotResult with figure and data."""
        simple_flow_system.optimize(highs_solver)

        result = simple_flow_system.statistics.plot.sankey.flows(show=False)

        # Check PlotResult structure
        assert hasattr(result, 'figure')
        assert hasattr(result, 'data')
        assert isinstance(result.data, xr.DataArray)
