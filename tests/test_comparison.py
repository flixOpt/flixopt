"""Tests for the Comparison class.

Tests:
- Basic comparison creation
- Statistics concatenation with different topologies
- Plot methods
- Error handling
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import flixopt as fx

# ============================================================================
# FIXTURES
# ============================================================================


_TIMESTEPS = pd.date_range('2020-01-01', periods=24, freq='h', name='time')


def _build_base_flow_system():
    """Factory: base flow system with boiler and storage."""
    fs = fx.FlowSystem(_TIMESTEPS, name='Base')
    fs.add_elements(
        fx.Effect('costs', '€', 'Costs', is_standard=True, is_objective=True),
        fx.Effect('CO2', 'kg', 'CO2 Emissions'),
        fx.Bus('Electricity'),
        fx.Bus('Heat'),
        fx.Bus('Gas'),
    )
    fs.add_elements(
        fx.Source(
            'Grid',
            outputs=[fx.Flow('P_el', bus='Electricity', size=100, effects_per_flow_hour={'costs': 0.3})],
        ),
        fx.Source(
            'GasSupply',
            outputs=[fx.Flow('Q_gas', bus='Gas', size=200, effects_per_flow_hour={'costs': 0.05, 'CO2': 0.2})],
        ),
        fx.Sink(
            'HeatDemand',
            inputs=[fx.Flow('Q_demand', bus='Heat', size=50, fixed_relative_profile=0.6)],
        ),
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.9,
            thermal_flow=fx.Flow('Q_th', bus='Heat', size=60),
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
        ),
        fx.Storage(
            'ThermalStorage',
            charging=fx.Flow('Q_charge', bus='Heat', size=20),
            discharging=fx.Flow('Q_discharge', bus='Heat', size=20),
            capacity_in_flow_hours=40,
            initial_charge_state=0.5,
        ),
    )
    return fs


def _build_flow_system_with_chp():
    """Factory: flow system with additional CHP component."""
    fs = fx.FlowSystem(_TIMESTEPS, name='WithCHP')
    fs.add_elements(
        fx.Effect('costs', '€', 'Costs', is_standard=True, is_objective=True),
        fx.Effect('CO2', 'kg', 'CO2 Emissions'),
        fx.Bus('Electricity'),
        fx.Bus('Heat'),
        fx.Bus('Gas'),
    )
    fs.add_elements(
        fx.Source(
            'Grid',
            outputs=[fx.Flow('P_el', bus='Electricity', size=100, effects_per_flow_hour={'costs': 0.3})],
        ),
        fx.Source(
            'GasSupply',
            outputs=[fx.Flow('Q_gas', bus='Gas', size=200, effects_per_flow_hour={'costs': 0.05, 'CO2': 0.2})],
        ),
        fx.Sink(
            'HeatDemand',
            inputs=[fx.Flow('Q_demand', bus='Heat', size=50, fixed_relative_profile=0.6)],
        ),
        fx.Sink(
            'ElectricitySink',
            inputs=[fx.Flow('P_sink', bus='Electricity', size=100)],
        ),
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.9,
            thermal_flow=fx.Flow('Q_th', bus='Heat', size=60),
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
        ),
        fx.linear_converters.CHP(
            'CHP',
            thermal_efficiency=0.5,
            electrical_efficiency=0.3,
            thermal_flow=fx.Flow('Q_th_chp', bus='Heat', size=30),
            electrical_flow=fx.Flow('P_el_chp', bus='Electricity', size=18),
            fuel_flow=fx.Flow('Q_fu_chp', bus='Gas'),
        ),
        fx.Storage(
            'ThermalStorage',
            charging=fx.Flow('Q_charge', bus='Heat', size=20),
            discharging=fx.Flow('Q_discharge', bus='Heat', size=20),
            capacity_in_flow_hours=40,
            initial_charge_state=0.5,
        ),
    )
    return fs


@pytest.fixture
def base_flow_system():
    """Unoptimized base flow system (function-scoped for tests needing fresh instance)."""
    return _build_base_flow_system()


@pytest.fixture(scope='module')
def optimized_base():
    """Optimized base flow system (module-scoped, solved once)."""
    fs = _build_base_flow_system()
    solver = fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=60)
    fs.optimize(solver)
    return fs


@pytest.fixture(scope='module')
def optimized_with_chp():
    """Optimized flow system with CHP (module-scoped, solved once)."""
    fs = _build_flow_system_with_chp()
    solver = fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=60)
    fs.optimize(solver)
    return fs


# ============================================================================
# BASIC COMPARISON TESTS
# ============================================================================


class TestComparisonCreation:
    """Tests for Comparison class creation and validation."""

    def test_comparison_requires_two_systems(self, optimized_base):
        """Comparison requires at least 2 FlowSystems."""
        with pytest.raises(ValueError, match='at least 2'):
            fx.Comparison([optimized_base])

    def test_comparison_creation_with_names(self, optimized_base, optimized_with_chp):
        """Comparison can be created with custom names."""
        comp = fx.Comparison([optimized_base, optimized_with_chp], names=['base', 'chp'])
        assert comp.names == ['base', 'chp']

    def test_comparison_uses_flowsystem_names(self, optimized_base, optimized_with_chp):
        """Comparison uses FlowSystem.name by default."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        assert comp.names == ['Base', 'WithCHP']

    def test_comparison_rejects_duplicate_names(self, optimized_base, optimized_with_chp):
        """Comparison rejects duplicate case names."""
        with pytest.raises(ValueError, match='unique'):
            fx.Comparison([optimized_base, optimized_with_chp], names=['same', 'same'])

    def test_comparison_rejects_unoptimized_system(self, base_flow_system, optimized_with_chp):
        """Comparison rejects FlowSystems without solutions when accessing solution."""
        comp = fx.Comparison([base_flow_system, optimized_with_chp])
        # Accessing solution triggers validation
        with pytest.raises(RuntimeError, match='no solution'):
            _ = comp.solution


# ============================================================================
# CONTAINER PROTOCOL TESTS
# ============================================================================


class TestComparisonContainerProtocol:
    """Tests for Comparison container protocol (__len__, __getitem__, __iter__, __contains__)."""

    def test_len(self, optimized_base, optimized_with_chp):
        """len() returns number of cases."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        assert len(comp) == 2

    def test_getitem_by_index(self, optimized_base, optimized_with_chp):
        """Indexing by int returns FlowSystem."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        assert comp[0] is optimized_base
        assert comp[1] is optimized_with_chp
        assert comp[-1] is optimized_with_chp

    def test_getitem_by_name(self, optimized_base, optimized_with_chp):
        """Indexing by name returns FlowSystem."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        assert comp['Base'] is optimized_base
        assert comp['WithCHP'] is optimized_with_chp

    def test_getitem_invalid_name_raises(self, optimized_base, optimized_with_chp):
        """Indexing by invalid name raises KeyError."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        with pytest.raises(KeyError, match='not found'):
            _ = comp['NonexistentCase']

    def test_getitem_invalid_index_raises(self, optimized_base, optimized_with_chp):
        """Indexing by invalid index raises IndexError."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        with pytest.raises(IndexError):
            _ = comp[99]

    def test_iter(self, optimized_base, optimized_with_chp):
        """Iteration yields (name, FlowSystem) pairs."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        items = list(comp)
        assert items == [('Base', optimized_base), ('WithCHP', optimized_with_chp)]

    def test_contains(self, optimized_base, optimized_with_chp):
        """'in' operator checks for case name."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        assert 'Base' in comp
        assert 'WithCHP' in comp
        assert 'NonexistentCase' not in comp

    def test_flow_systems_property(self, optimized_base, optimized_with_chp):
        """flow_systems returns dict mapping name to FlowSystem."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        fs_dict = comp.flow_systems
        assert isinstance(fs_dict, dict)
        assert fs_dict['Base'] is optimized_base
        assert fs_dict['WithCHP'] is optimized_with_chp

    def test_is_optimized_true(self, optimized_base, optimized_with_chp):
        """is_optimized returns True when all systems optimized."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        assert comp.is_optimized is True

    def test_is_optimized_false(self, base_flow_system, optimized_with_chp):
        """is_optimized returns False when some systems not optimized."""
        comp = fx.Comparison([base_flow_system, optimized_with_chp])
        assert comp.is_optimized is False

    def test_dims_returns_shared_dimensions(self, optimized_base, optimized_with_chp):
        """dims returns dimensions shared across all systems."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        dims = comp.dims
        assert 'time' in dims
        assert dims['time'] == 25  # 24 intervals + 1 boundary point

    def test_repr_contains_case_names(self, optimized_base, optimized_with_chp):
        """__repr__ includes case names."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        repr_str = repr(comp)
        assert 'Base' in repr_str
        assert 'WithCHP' in repr_str

    def test_repr_shows_optimization_status(self, base_flow_system, optimized_with_chp):
        """__repr__ shows optimization status."""
        comp = fx.Comparison([base_flow_system, optimized_with_chp])
        repr_str = repr(comp)
        # Should show different status symbols for optimized vs not
        assert '✓' in repr_str  # optimized_with_chp
        assert '○' in repr_str  # base_flow_system (not optimized)


# ============================================================================
# SOLUTION AND STATISTICS TESTS
# ============================================================================


class TestComparisonSolution:
    """Tests for Comparison.solution property."""

    def test_solution_has_case_dimension(self, optimized_base, optimized_with_chp):
        """Combined solution has 'case' dimension."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        assert 'case' in comp.solution.dims

    def test_solution_contains_all_variables(self, optimized_base, optimized_with_chp):
        """Combined solution contains variables from both systems."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        solution = comp.solution

        # Variables from base system
        assert 'Boiler(Q_th)' in solution['flow|rate'].coords['flow'].values

        # Variables only in CHP system should also be present
        assert 'CHP(Q_th_chp)' in solution['flow|rate'].coords['flow'].values

    def test_solution_fills_missing_with_nan(self, optimized_base, optimized_with_chp):
        """Variables not in all systems are filled with NaN."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])

        # CHP variable should be NaN for base system
        chp_flow = comp.solution['flow|rate'].sel(flow='CHP(Q_th_chp)')
        base_values = chp_flow.sel(case='Base')
        assert np.all(np.isnan(base_values.values))

        # CHP variable should have real values for WithCHP system
        chp_values = chp_flow.sel(case='WithCHP')
        assert not np.all(np.isnan(chp_values.values))


class TestComparisonStatistics:
    """Tests for Comparison.statistics property."""

    def test_statistics_flow_rates_has_case_dimension(self, optimized_base, optimized_with_chp):
        """Combined flow_rates has 'case' dimension."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        assert 'case' in comp.statistics.flow_rates.dims

    def test_statistics_contains_all_flows(self, optimized_base, optimized_with_chp):
        """Combined statistics contains flows from both systems."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        flow_rates = comp.statistics.flow_rates

        flow_names = list(str(f) for f in flow_rates.coords['flow'].values)
        # Common flows
        assert 'Boiler(Q_th)' in flow_names

        # CHP-only flows
        assert 'CHP(Q_th_chp)' in flow_names

    def test_statistics_colors_merged(self, optimized_base, optimized_with_chp):
        """Component colors are merged from all systems."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        colors = comp.statistics.component_colors

        assert 'Boiler' in colors
        assert 'CHP' in colors


# ============================================================================
# PLOT METHOD TESTS
# ============================================================================


class TestComparisonPlotMethods:
    """Tests for Comparison.statistics.plot methods."""

    def test_balance_returns_plot_result(self, optimized_base, optimized_with_chp):
        """balance() returns PlotResult with data and figure."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        result = comp.statistics.plot.balance('Heat', show=False)

        assert hasattr(result, 'data')
        assert hasattr(result, 'figure')
        assert isinstance(result.data, xr.DataArray)

    def test_balance_includes_all_flows(self, optimized_base, optimized_with_chp):
        """balance() includes flows from both systems (with non-zero values)."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        result = comp.statistics.plot.balance('Heat', show=False)

        # Should include flows that have non-zero values in at least one system
        # Note: CHP is not used (all zeros) in this test, so it's correctly filtered out
        # The Boiler flow is present in both systems
        assert 'Boiler(Q_th)' in result.data.coords['flow'].values

    def test_balance_data_has_case_dimension(self, optimized_base, optimized_with_chp):
        """balance() data has 'case' dimension."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        result = comp.statistics.plot.balance('Heat', show=False)

        assert 'case' in result.data.dims

    def test_carrier_balance(self, optimized_base, optimized_with_chp):
        """carrier_balance() works without error (even with no carriers defined)."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        # carrier_balance requires buses to have carrier attribute set
        # With no carriers defined, it should return empty result without error
        with pytest.warns(UserWarning, match='No buses found with carrier'):
            result = comp.statistics.plot.carrier_balance('heat', show=False)

        # Just check it runs without error and returns PlotResult
        assert hasattr(result, 'data')
        assert hasattr(result, 'figure')

    def test_flows(self, optimized_base, optimized_with_chp):
        """flows() works correctly."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        result = comp.statistics.plot.flows(show=False)

        assert 'case' in result.data.dims

    def test_sizes(self, optimized_base, optimized_with_chp):
        """sizes() works correctly (may be empty if no investment variables)."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        result = comp.statistics.plot.sizes(show=False)

        # May be empty if no investment variables in the test systems
        if 'element' in result.data.dims:
            assert 'case' in result.data.dims

    def test_effects(self, optimized_base, optimized_with_chp):
        """effects() works correctly."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        result = comp.statistics.plot.effects(show=False)

        assert 'case' in result.data.dims

    def test_charge_states(self, optimized_base, optimized_with_chp):
        """charge_states() works correctly."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        result = comp.statistics.plot.charge_states(show=False)

        assert 'case' in result.data.dims

    def test_duration_curve(self, optimized_base, optimized_with_chp):
        """duration_curve() works correctly."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        result = comp.statistics.plot.duration_curve('Boiler(Q_th)', show=False)

        assert 'case' in result.data.dims

    def test_storage(self, optimized_base, optimized_with_chp):
        """storage() works correctly (may be empty if no storage in test systems)."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        result = comp.statistics.plot.storage('ThermalStorage', show=False)

        # May be empty if ThermalStorage not in test systems
        if 'flow' in result.data.dims:
            assert 'case' in result.data.dims

    def test_heatmap(self, optimized_base, optimized_with_chp):
        """heatmap() works correctly."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        result = comp.statistics.plot.heatmap('Boiler(Q_th)', show=False)

        assert 'case' in result.data.dims


class TestComparisonPlotKwargs:
    """Tests for kwargs handling in plot methods."""

    def test_data_kwargs_passed_through(self, optimized_base, optimized_with_chp):
        """Data kwargs (like 'unit') are passed to underlying method."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])

        # flow_hours should change the data
        result_rate = comp.statistics.plot.balance('Heat', unit='flow_rate', show=False)
        result_hours = comp.statistics.plot.balance('Heat', unit='flow_hours', show=False)

        # Values should be different (hours = rate * time)
        # Just check they both work without error
        assert result_rate.data is not None
        assert result_hours.data is not None

    def test_plotly_kwargs_passed_through(self, optimized_base, optimized_with_chp):
        """Plotly kwargs are passed to figure creation."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        result = comp.statistics.plot.balance('Heat', show=False, height=600)

        # Check height was applied
        assert result.figure.layout.height == 600


# ============================================================================
# DIFF METHOD TESTS
# ============================================================================


class TestComparisonDiff:
    """Tests for Comparison.diff() method."""

    def test_diff_returns_dataset(self, optimized_base, optimized_with_chp):
        """diff() returns an xarray Dataset."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        diff = comp.diff()

        assert isinstance(diff, xr.Dataset)
        assert 'case' in diff.dims

    def test_diff_reference_by_index(self, optimized_base, optimized_with_chp):
        """diff() accepts reference by index."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        diff = comp.diff(reference=0)

        assert isinstance(diff, xr.Dataset)
        assert 'case' in diff.dims

    def test_diff_reference_by_name(self, optimized_base, optimized_with_chp):
        """diff() accepts reference by name."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])
        diff = comp.diff(reference='Base')

        assert diff is not None


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestComparisonErrors:
    """Tests for error handling."""

    def test_balance_unknown_node_returns_empty(self, optimized_base, optimized_with_chp):
        """balance() with unknown node returns empty result."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])

        # This should not raise because at least one system might have it
        # But if no system has it, it returns empty with a warning
        with pytest.warns(UserWarning, match='not found in buses or components'):
            result = comp.statistics.plot.balance('NonexistentBus', show=False)
        assert result.data.dims == ()

    def test_diff_invalid_reference_raises(self, optimized_base, optimized_with_chp):
        """diff() with invalid reference raises ValueError."""
        comp = fx.Comparison([optimized_base, optimized_with_chp])

        with pytest.raises(ValueError, match='not found'):
            comp.diff(reference='NonexistentCase')
