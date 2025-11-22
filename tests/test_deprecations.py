"""Comprehensive pytest-based test for all deprecation warnings with v5.0.0 removal message."""

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import flixopt as fx
from flixopt.config import DEPRECATION_REMOVAL_VERSION, change_logging_level
from flixopt.linear_converters import CHP, Boiler, HeatPump, HeatPumpWithSource, Power2Heat
from flixopt.results import plot_heatmap


# === Parameter deprecations (via _handle_deprecated_kwarg) ===
@pytest.mark.parametrize(
    'name,factory',
    [
        ("Source 'source'", lambda: fx.Source('s1', source=fx.Flow('out1', bus='bus', size=10))),
        ("Sink 'sink'", lambda: fx.Sink('sink1', sink=fx.Flow('in2', bus='bus', size=10))),
        ("InvestParameters 'fix_effects'", lambda: fx.InvestParameters(minimum_size=10, fix_effects={'costs': 100})),
        (
            "InvestParameters 'specific_effects'",
            lambda: fx.InvestParameters(minimum_size=10, specific_effects={'costs': 10}),
        ),
        (
            "InvestParameters 'divest_effects'",
            lambda: fx.InvestParameters(minimum_size=10, divest_effects={'costs': 50}),
        ),
        (
            "InvestParameters 'piecewise_effects'",
            lambda: fx.InvestParameters(minimum_size=10, piecewise_effects=[]),
        ),
        ("InvestParameters 'optional'", lambda: fx.InvestParameters(minimum_size=10, optional=True)),
        ("OnOffParameters 'on_hours_total_min'", lambda: fx.OnOffParameters(on_hours_total_min=10)),
        ("OnOffParameters 'on_hours_total_max'", lambda: fx.OnOffParameters(on_hours_total_max=20)),
        ("OnOffParameters 'switch_on_total_max'", lambda: fx.OnOffParameters(switch_on_total_max=5)),
        ("Flow 'flow_hours_total_min'", lambda: fx.Flow('f1', bus='bus', size=10, flow_hours_total_min=5)),
        ("Flow 'flow_hours_total_max'", lambda: fx.Flow('f2', bus='bus', size=10, flow_hours_total_max=20)),
        (
            "Flow 'flow_hours_per_period_min'",
            lambda: fx.Flow('f3', bus='bus', size=10, flow_hours_per_period_min=5),
        ),
        (
            "Flow 'flow_hours_per_period_max'",
            lambda: fx.Flow('f4', bus='bus', size=10, flow_hours_per_period_max=20),
        ),
        ("Flow 'total_flow_hours_min'", lambda: fx.Flow('f5', bus='bus', size=10, total_flow_hours_min=5)),
        ("Flow 'total_flow_hours_max'", lambda: fx.Flow('f6', bus='bus', size=10, total_flow_hours_max=20)),
        (
            "Effect 'minimum_operation'",
            lambda: fx.Effect('e1', unit='€', description='test', minimum_operation=100),
        ),
        (
            "Effect 'maximum_operation'",
            lambda: fx.Effect('e2', unit='€', description='test', maximum_operation=200),
        ),
        ("Effect 'minimum_invest'", lambda: fx.Effect('e3', unit='€', description='test', minimum_invest=50)),
        ("Effect 'maximum_invest'", lambda: fx.Effect('e4', unit='€', description='test', maximum_invest=150)),
        (
            "Effect 'minimum_operation_per_hour'",
            lambda: fx.Effect('e5', unit='€', description='test', minimum_operation_per_hour=10),
        ),
        (
            "Effect 'maximum_operation_per_hour'",
            lambda: fx.Effect('e6', unit='€', description='test', maximum_operation_per_hour=30),
        ),
        # Linear converters
        (
            "Boiler 'Q_fu'",
            lambda: Boiler(
                'b1', Q_fu=fx.Flow('f1', 'bus', 10), thermal_flow=fx.Flow('h1', 'bus', 9), thermal_efficiency=0.9
            ),
        ),
        (
            "Boiler 'Q_th'",
            lambda: Boiler(
                'b2', fuel_flow=fx.Flow('f2', 'bus', 10), Q_th=fx.Flow('h2', 'bus', 9), thermal_efficiency=0.9
            ),
        ),
        (
            "Boiler 'eta'",
            lambda: Boiler('b3', fuel_flow=fx.Flow('f3', 'bus', 10), thermal_flow=fx.Flow('h3', 'bus', 9), eta=0.9),
        ),
        (
            "Power2Heat 'P_el'",
            lambda: Power2Heat(
                'p1', P_el=fx.Flow('e1', 'bus', 10), thermal_flow=fx.Flow('h4', 'bus', 9), thermal_efficiency=0.9
            ),
        ),
        (
            "Power2Heat 'Q_th'",
            lambda: Power2Heat(
                'p2', electrical_flow=fx.Flow('e2', 'bus', 10), Q_th=fx.Flow('h5', 'bus', 9), thermal_efficiency=0.9
            ),
        ),
        (
            "Power2Heat 'eta'",
            lambda: Power2Heat(
                'p3', electrical_flow=fx.Flow('e3', 'bus', 10), thermal_flow=fx.Flow('h6', 'bus', 9), eta=0.9
            ),
        ),
        (
            "HeatPump 'P_el'",
            lambda: HeatPump('hp1', P_el=fx.Flow('e4', 'bus', 10), thermal_flow=fx.Flow('h7', 'bus', 30), cop=3.0),
        ),
        (
            "HeatPump 'Q_th'",
            lambda: HeatPump('hp2', electrical_flow=fx.Flow('e5', 'bus', 10), Q_th=fx.Flow('h8', 'bus', 30), cop=3.0),
        ),
        (
            "HeatPump 'COP'",
            lambda: HeatPump(
                'hp3', electrical_flow=fx.Flow('e6', 'bus', 10), thermal_flow=fx.Flow('h9', 'bus', 30), COP=3.0
            ),
        ),
        (
            "CHP 'Q_fu'",
            lambda: CHP(
                'chp1',
                Q_fu=fx.Flow('f4', 'bus', 100),
                electrical_flow=fx.Flow('e7', 'bus', 30),
                thermal_flow=fx.Flow('h10', 'bus', 60),
                thermal_efficiency=0.6,
                electrical_efficiency=0.3,
            ),
        ),
        (
            "CHP 'P_el'",
            lambda: CHP(
                'chp2',
                fuel_flow=fx.Flow('f5', 'bus', 100),
                P_el=fx.Flow('e8', 'bus', 30),
                thermal_flow=fx.Flow('h11', 'bus', 60),
                thermal_efficiency=0.6,
                electrical_efficiency=0.3,
            ),
        ),
        (
            "CHP 'Q_th'",
            lambda: CHP(
                'chp3',
                fuel_flow=fx.Flow('f6', 'bus', 100),
                electrical_flow=fx.Flow('e9', 'bus', 30),
                Q_th=fx.Flow('h12', 'bus', 60),
                thermal_efficiency=0.6,
                electrical_efficiency=0.3,
            ),
        ),
        (
            "CHP 'eta_th'",
            lambda: CHP(
                'chp4',
                fuel_flow=fx.Flow('f7', 'bus', 100),
                electrical_flow=fx.Flow('e10', 'bus', 30),
                thermal_flow=fx.Flow('h13', 'bus', 60),
                eta_th=0.6,
                electrical_efficiency=0.3,
            ),
        ),
        (
            "CHP 'eta_el'",
            lambda: CHP(
                'chp5',
                fuel_flow=fx.Flow('f8', 'bus', 100),
                electrical_flow=fx.Flow('e11', 'bus', 30),
                thermal_flow=fx.Flow('h14', 'bus', 60),
                thermal_efficiency=0.6,
                eta_el=0.3,
            ),
        ),
        (
            "HeatPumpWithSource 'COP'",
            lambda: HeatPumpWithSource(
                'hps1',
                electrical_flow=fx.Flow('e12', 'bus', 10),
                heat_source_flow=fx.Flow('hs1', 'bus', 20),
                thermal_flow=fx.Flow('h15', 'bus', 30),
                COP=3.0,
            ),
        ),
        (
            "HeatPumpWithSource 'P_el'",
            lambda: HeatPumpWithSource(
                'hps2',
                P_el=fx.Flow('e13', 'bus', 10),
                heat_source_flow=fx.Flow('hs2', 'bus', 20),
                thermal_flow=fx.Flow('h16', 'bus', 30),
                cop=3.0,
            ),
        ),
        (
            "HeatPumpWithSource 'Q_ab'",
            lambda: HeatPumpWithSource(
                'hps3',
                electrical_flow=fx.Flow('e14', 'bus', 10),
                Q_ab=fx.Flow('hs3', 'bus', 20),
                thermal_flow=fx.Flow('h17', 'bus', 30),
                cop=3.0,
            ),
        ),
        (
            "HeatPumpWithSource 'Q_th'",
            lambda: HeatPumpWithSource(
                'hps4',
                electrical_flow=fx.Flow('e15', 'bus', 10),
                heat_source_flow=fx.Flow('hs4', 'bus', 20),
                Q_th=fx.Flow('h18', 'bus', 30),
                cop=3.0,
            ),
        ),
        # TimeSeriesData parameters
        ("TimeSeriesData 'agg_group'", lambda: fx.TimeSeriesData([1, 2, 3], agg_group=1)),
        ("TimeSeriesData 'agg_weight'", lambda: fx.TimeSeriesData([1, 2, 3], agg_weight=2.5)),
        # Storage parameter
        (
            "Storage 'initial_charge_state=lastValueOfSim'",
            lambda: fx.Storage(
                'stor1',
                charging=fx.Flow('charge', 'bus', 10),
                discharging=fx.Flow('discharge', 'bus', 10),
                capacity_in_flow_hours=10,
                initial_charge_state='lastValueOfSim',
            ),
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else '',
)
def test_parameter_deprecations(name, factory):
    """Test all parameter deprecations include removal version message."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        factory()
        assert len(w) > 0, f'No warning raised for {name}'
        assert f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}' in str(w[0].message), (
            f'Missing removal version in {name}'
        )


# === Property deprecations ===
@pytest.fixture(scope='module')
def deprecated_instances():
    """Create instances for property testing."""
    return {
        'data': fx.TimeSeriesData([1, 2, 3], aggregation_group=1),
        'boiler': Boiler(
            'b_prop', fuel_flow=fx.Flow('f_p', 'bus', 10), thermal_flow=fx.Flow('h_p', 'bus', 9), thermal_efficiency=0.9
        ),
        'invest_with_effects': fx.InvestParameters(
            minimum_size=10,
            maximum_size=100,
            mandatory=False,
            effects_of_investment={'costs': 100},
            effects_of_investment_per_size={'costs': 10},
            effects_of_retirement={'costs': 50},
            piecewise_effects_of_investment=None,
        ),
        'invest': fx.InvestParameters(minimum_size=10, maximum_size=100, mandatory=False),
        'onoff': fx.OnOffParameters(
            on_hours_min=5,
            on_hours_max=10,
            switch_on_max=3,
        ),
        'flow': fx.Flow('f_prop', bus='bus', size=10, flow_hours_min=5, flow_hours_max=20),
        'chp': CHP(
            'chp_prop',
            fuel_flow=fx.Flow('f_chp', 'bus', 100),
            electrical_flow=fx.Flow('e_chp', 'bus', 30),
            thermal_flow=fx.Flow('h_chp', 'bus', 60),
            thermal_efficiency=0.6,
            electrical_efficiency=0.3,
        ),
        'hp': HeatPump(
            'hp_prop', electrical_flow=fx.Flow('e_hp', 'bus', 10), thermal_flow=fx.Flow('h_hp', 'bus', 30), cop=3.0
        ),
        'hps': HeatPumpWithSource(
            'hps_prop',
            electrical_flow=fx.Flow('e_hps', 'bus', 10),
            heat_source_flow=fx.Flow('hs_hps', 'bus', 20),
            thermal_flow=fx.Flow('h_hps', 'bus', 30),
            cop=3.0,
        ),
        'source': fx.Source('source_prop', outputs=[fx.Flow('out', 'bus', 10)]),
        'sink': fx.Sink('sink_prop', inputs=[fx.Flow('in', 'bus', 10)]),
        'storage': fx.Storage(
            'storage_prop',
            charging=fx.Flow('charge', 'bus', 10),
            discharging=fx.Flow('discharge', 'bus', 10),
            capacity_in_flow_hours=10,
        ),
        'effect': fx.Effect('effect_prop', unit='€', description='test'),
    }


@pytest.mark.parametrize(
    'name,accessor',
    [
        # TimeSeriesData properties
        ('TimeSeriesData.agg_group', lambda objs: objs['data'].agg_group),
        ('TimeSeriesData.agg_weight', lambda objs: objs['data'].agg_weight),
        # InvestParameters properties
        ('InvestParameters.optional', lambda objs: objs['invest'].optional),
        ('InvestParameters.fix_effects', lambda objs: objs['invest_with_effects'].fix_effects),
        ('InvestParameters.specific_effects', lambda objs: objs['invest_with_effects'].specific_effects),
        ('InvestParameters.divest_effects', lambda objs: objs['invest_with_effects'].divest_effects),
        ('InvestParameters.piecewise_effects', lambda objs: objs['invest_with_effects'].piecewise_effects),
        # OnOffParameters properties
        ('OnOffParameters.on_hours_total_min', lambda objs: objs['onoff'].on_hours_total_min),
        ('OnOffParameters.on_hours_total_max', lambda objs: objs['onoff'].on_hours_total_max),
        ('OnOffParameters.switch_on_total_max', lambda objs: objs['onoff'].switch_on_total_max),
        # Flow properties
        ('Flow.flow_hours_total_min', lambda objs: objs['flow'].flow_hours_total_min),
        ('Flow.flow_hours_total_max', lambda objs: objs['flow'].flow_hours_total_max),
        # Boiler properties
        ('Boiler.eta', lambda objs: objs['boiler'].eta),
        ('Boiler.Q_fu', lambda objs: objs['boiler'].Q_fu),
        ('Boiler.Q_th', lambda objs: objs['boiler'].Q_th),
        # CHP properties
        ('CHP.eta_th', lambda objs: objs['chp'].eta_th),
        ('CHP.eta_el', lambda objs: objs['chp'].eta_el),
        ('CHP.Q_fu', lambda objs: objs['chp'].Q_fu),
        ('CHP.P_el', lambda objs: objs['chp'].P_el),
        ('CHP.Q_th', lambda objs: objs['chp'].Q_th),
        # HeatPump properties
        ('HeatPump.COP', lambda objs: objs['hp'].COP),
        ('HeatPump.P_el', lambda objs: objs['hp'].P_el),
        ('HeatPump.Q_th', lambda objs: objs['hp'].Q_th),
        # HeatPumpWithSource properties
        ('HeatPumpWithSource.COP', lambda objs: objs['hps'].COP),
        ('HeatPumpWithSource.P_el', lambda objs: objs['hps'].P_el),
        ('HeatPumpWithSource.Q_ab', lambda objs: objs['hps'].Q_ab),
        ('HeatPumpWithSource.Q_th', lambda objs: objs['hps'].Q_th),
        # Source properties
        ('Source.source', lambda objs: objs['source'].source),
        # Sink properties
        ('Sink.sink', lambda objs: objs['sink'].sink),
        # Effect property getters
        ('Effect.minimum_total_per_period (getter)', lambda objs: objs['effect'].minimum_total_per_period),
        ('Effect.maximum_total_per_period (getter)', lambda objs: objs['effect'].maximum_total_per_period),
    ],
    ids=lambda x: x if isinstance(x, str) else '',
)
def test_property_deprecations(name, accessor, deprecated_instances):
    """Test all property deprecations include removal version message."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        accessor(deprecated_instances)
        assert len(w) > 0, f'No warning raised for {name}'
        assert f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}' in str(w[0].message), (
            f'Missing removal version in {name}'
        )


# === Property setter deprecations ===
@pytest.mark.parametrize(
    'name,setter',
    [
        # InvestParameters setter
        ('InvestParameters.optional (setter)', lambda: setattr(fx.InvestParameters(minimum_size=10), 'optional', True)),
        # OnOffParameters setters
        (
            'OnOffParameters.on_hours_total_min (setter)',
            lambda: setattr(fx.OnOffParameters(), 'on_hours_total_min', 10),
        ),
        (
            'OnOffParameters.on_hours_total_max (setter)',
            lambda: setattr(fx.OnOffParameters(), 'on_hours_total_max', 20),
        ),
        (
            'OnOffParameters.switch_on_total_max (setter)',
            lambda: setattr(fx.OnOffParameters(), 'switch_on_total_max', 5),
        ),
        # Flow setters
        ('Flow.flow_hours_total_min (setter)', lambda: setattr(fx.Flow('f', 'bus', 10), 'flow_hours_total_min', 5)),
        ('Flow.flow_hours_total_max (setter)', lambda: setattr(fx.Flow('f', 'bus', 10), 'flow_hours_total_max', 20)),
        # Effect setters
        ('Effect.minimum_operation (setter)', lambda: setattr(fx.Effect('e', '€', 'test'), 'minimum_operation', 100)),
        ('Effect.maximum_operation (setter)', lambda: setattr(fx.Effect('e', '€', 'test'), 'maximum_operation', 200)),
        ('Effect.minimum_invest (setter)', lambda: setattr(fx.Effect('e', '€', 'test'), 'minimum_invest', 50)),
        ('Effect.maximum_invest (setter)', lambda: setattr(fx.Effect('e', '€', 'test'), 'maximum_invest', 150)),
        (
            'Effect.minimum_operation_per_hour (setter)',
            lambda: setattr(fx.Effect('e', '€', 'test'), 'minimum_operation_per_hour', 10),
        ),
        (
            'Effect.maximum_operation_per_hour (setter)',
            lambda: setattr(fx.Effect('e', '€', 'test'), 'maximum_operation_per_hour', 30),
        ),
        (
            'Effect.minimum_total_per_period (setter)',
            lambda: setattr(fx.Effect('e', '€', 'test'), 'minimum_total_per_period', 100),
        ),
        (
            'Effect.maximum_total_per_period (setter)',
            lambda: setattr(fx.Effect('e', '€', 'test'), 'maximum_total_per_period', 200),
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else '',
)
def test_property_setter_deprecations(name, setter):
    """Test all property setter deprecations include removal version message."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        setter()
        assert len(w) > 0, f'No warning raised for {name}'
        assert f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}' in str(w[0].message), (
            f'Missing removal version in {name}'
        )


# === FlowSystem-specific deprecations ===
def test_flowsystem_all_elements_property():
    """Test FlowSystem.all_elements property deprecation."""
    fs = fx.FlowSystem(timesteps=pd.date_range('2020-01-01', periods=10, freq='h'))
    fs.add_elements(fx.Source('s1', outputs=[fx.Flow('out', 'bus', 10)]))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        _ = fs.all_elements
        assert len(w) > 0, 'No warning raised for FlowSystem.all_elements'
        assert f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}' in str(w[0].message)


def test_flowsystem_weights_getter():
    """Test FlowSystem.weights getter deprecation."""
    fs = fx.FlowSystem(
        timesteps=pd.date_range('2020-01-01', periods=10, freq='h'), scenarios=pd.Index(['A', 'B'], name='scenario')
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        _ = fs.weights
        assert len(w) > 0, 'No warning raised for FlowSystem.weights getter'
        assert f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}' in str(w[0].message)


def test_flowsystem_weights_setter():
    """Test FlowSystem.weights setter deprecation."""
    fs = fx.FlowSystem(
        timesteps=pd.date_range('2020-01-01', periods=10, freq='h'),
        scenarios=pd.Index(['A', 'B', 'C'], name='scenario'),
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        fs.weights = [1, 2, 3]
        assert len(w) > 0, 'No warning raised for FlowSystem.weights setter'
        assert f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}' in str(w[0].message)


# === Calculation deprecations ===
def test_calculation_active_timesteps_parameter():
    """Test Calculation active_timesteps parameter deprecation."""
    fs = fx.FlowSystem(timesteps=pd.date_range('2020-01-01', periods=10, freq='h'))
    fs.add_elements(fx.Source('s1', outputs=[fx.Flow('out', 'bus', 10)]))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        _ = fx.calculation.Calculation(
            'test', fs, 'costs', active_timesteps=pd.date_range('2020-01-01', periods=5, freq='h')
        )
        assert len(w) > 0, 'No warning raised for Calculation active_timesteps parameter'
        assert f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}' in str(w[0].message)


def test_calculation_active_timesteps_property():
    """Test Calculation.active_timesteps property deprecation."""
    fs = fx.FlowSystem(timesteps=pd.date_range('2020-01-01', periods=10, freq='h'))
    fs.add_elements(fx.Source('s1', outputs=[fx.Flow('out', 'bus', 10)]))
    calc = fx.calculation.Calculation('test', fs, 'costs')

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        _ = calc.active_timesteps
        assert len(w) > 0, 'No warning raised for Calculation.active_timesteps property'
        assert f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}' in str(w[0].message)


# === Config function deprecations ===
def test_change_logging_level_function():
    """Test change_logging_level() function deprecation."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        change_logging_level('INFO')
        assert len(w) > 0, 'No warning raised for change_logging_level()'
        assert f'will be removed in version {DEPRECATION_REMOVAL_VERSION}' in str(w[0].message)


# === Results-related deprecations ===
@pytest.fixture
def simple_results():
    """Create a simple calculation results object for testing."""
    # Create a minimal flow system
    fs = fx.FlowSystem(timesteps=pd.date_range('2020-01-01', periods=5, freq='h'))
    source = fx.Source('source1', outputs=[fx.Flow('out', 'bus1', size=10)])
    sink = fx.Sink('sink1', inputs=[fx.Flow('in', 'bus1', size=10)])
    fs.add_elements(source, sink)

    # Create and solve calculation
    calc = fx.calculation.Calculation('test', fs, 'costs')
    calc.do_modeling()
    solver = fx.solvers.HighsSolver(mip_gap=0.01, time_limit_seconds=30)
    calc.solve(solver)

    return calc.results


def test_results_flow_system_parameter(simple_results):
    """Test CalculationResults flow_system parameter deprecation."""
    # Get the flow_system_data from existing results
    fs_data = simple_results.flow_system_data

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        # Create new results with deprecated parameter
        from flixopt.results import CalculationResults

        _ = CalculationResults(
            solution=simple_results.solution,
            flow_system=fs_data,  # deprecated parameter
            results_path=None,
        )
        assert len(w) > 0, 'No warning raised for flow_system parameter'
        assert f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}' in str(w[0].message)


def test_results_plot_node_balance_indexer(simple_results):
    """Test ComponentResults.plot_node_balance indexer parameter deprecation."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        simple_results['source1'].plot_node_balance(indexer={'time': slice(0, 2)}, show=False, save=False)
        assert len(w) > 0, 'No warning raised for plot_node_balance indexer parameter'
        assert f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}' in str(w[0].message)


def test_results_plot_io_area_indexer(simple_results):
    """Test ComponentResults.plot_io_area indexer parameter deprecation."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        simple_results['source1'].plot_io_area(indexer={'time': slice(0, 2)}, show=False, save=False)
        assert len(w) > 0, 'No warning raised for plot_io_area indexer parameter'
        assert f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}' in str(w[0].message)


def test_results_io_balance_indexer(simple_results):
    """Test ComponentResults.io_balance indexer parameter deprecation."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        _ = simple_results['source1'].io_balance(indexer={'time': slice(0, 2)})
        assert len(w) > 0, 'No warning raised for io_balance indexer parameter'
        assert f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}' in str(w[0].message)


def test_plot_heatmap_function_heatmap_params():
    """Test plot_heatmap function heatmap_timeframes/heatmap_timesteps_per_frame parameters."""
    # Create simple test data - 7 days * 24 hours = 168 hours
    data = xr.DataArray(
        np.random.rand(168),
        coords={'time': pd.date_range('2020-01-01', periods=168, freq='h')},
        dims=['time'],
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        plot_heatmap(
            data,
            name='test',
            heatmap_timeframes='D',  # Days
            heatmap_timesteps_per_frame='h',  # Hours
            show=False,
            save=False,
        )
        assert len(w) > 0, 'No warning raised for heatmap_timeframes/heatmap_timesteps_per_frame'
        assert f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}' in str(w[0].message)


def test_plot_heatmap_function_color_map():
    """Test plot_heatmap function color_map parameter."""
    data = xr.DataArray(
        np.random.rand(24),
        coords={'time': pd.date_range('2020-01-01', periods=24, freq='h')},
        dims=['time'],
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        plot_heatmap(data, name='test', color_map='viridis', show=False, save=False)
        assert len(w) > 0, 'No warning raised for color_map parameter'
        assert f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}' in str(w[0].message)


def test_plot_heatmap_function_indexer():
    """Test plot_heatmap function indexer parameter."""
    time_index = pd.date_range('2020-01-01', periods=24, freq='h')
    data = xr.DataArray(
        np.random.rand(24),
        coords={'time': time_index},
        dims=['time'],
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        # Use actual datetime values for slicing
        plot_heatmap(data, name='test', indexer={'time': slice(time_index[0], time_index[10])}, show=False, save=False)
        assert len(w) > 0, 'No warning raised for indexer parameter'
        assert f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}' in str(w[0].message)
