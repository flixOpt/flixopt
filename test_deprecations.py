"""Comprehensive pytest-based test for all deprecation warnings with v5.0.0 removal message."""

import warnings

import pandas as pd
import pytest

import flixopt as fx
from flixopt.linear_converters import CHP, Boiler, HeatPump, HeatPumpWithSource, Power2Heat


@pytest.fixture(scope='module')
def setup():
    """Setup test environment."""
    ts = pd.date_range('2023-01-01', periods=24, freq='h')
    fs = fx.FlowSystem(ts)
    bus = fx.Bus('bus')
    fs.add_elements(bus)
    return {'ts': ts, 'fs': fs, 'bus': bus}


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
    ],
    ids=lambda x: x if isinstance(x, str) else '',
)
def test_parameter_deprecations(name, factory):
    """Test all parameter deprecations include v5.0.0 removal message."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        factory()
        assert len(w) > 0, f'No warning raised for {name}'
        assert 'Will be removed in v5.0.0' in str(w[0].message), f'Missing removal version in {name}'


# === Property deprecations ===
@pytest.fixture(scope='module')
def deprecated_instances():
    """Create instances for property testing."""
    return {
        'data': fx.TimeSeriesData([1, 2, 3], aggregation_group=1),
        'boiler': Boiler(
            'b_prop', fuel_flow=fx.Flow('f_p', 'bus', 10), thermal_flow=fx.Flow('h_p', 'bus', 9), thermal_efficiency=0.9
        ),
        'invest': fx.InvestParameters(minimum_size=10, maximum_size=100, mandatory=False),
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
    }


@pytest.mark.parametrize(
    'name,accessor',
    [
        ('TimeSeriesData.agg_group', lambda objs: objs['data'].agg_group),
        ('TimeSeriesData.agg_weight', lambda objs: objs['data'].agg_weight),
        ('InvestParameters.optional', lambda objs: objs['invest'].optional),
        ('Boiler.eta', lambda objs: objs['boiler'].eta),
        ('Boiler.Q_fu', lambda objs: objs['boiler'].Q_fu),
        ('Boiler.Q_th', lambda objs: objs['boiler'].Q_th),
        ('CHP.eta_th', lambda objs: objs['chp'].eta_th),
        ('CHP.eta_el', lambda objs: objs['chp'].eta_el),
        ('CHP.Q_fu', lambda objs: objs['chp'].Q_fu),
        ('CHP.P_el', lambda objs: objs['chp'].P_el),
        ('CHP.Q_th', lambda objs: objs['chp'].Q_th),
        ('HeatPump.COP', lambda objs: objs['hp'].COP),
        ('HeatPump.P_el', lambda objs: objs['hp'].P_el),
        ('HeatPump.Q_th', lambda objs: objs['hp'].Q_th),
        ('HeatPumpWithSource.COP', lambda objs: objs['hps'].COP),
        ('HeatPumpWithSource.P_el', lambda objs: objs['hps'].P_el),
        ('HeatPumpWithSource.Q_ab', lambda objs: objs['hps'].Q_ab),
        ('HeatPumpWithSource.Q_th', lambda objs: objs['hps'].Q_th),
    ],
    ids=lambda x: x if isinstance(x, str) else '',
)
def test_property_deprecations(name, accessor, deprecated_instances):
    """Test all property deprecations include v5.0.0 removal message."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        accessor(deprecated_instances)
        assert len(w) > 0, f'No warning raised for {name}'
        assert 'Will be removed in v5.0.0' in str(w[0].message), f'Missing removal version in {name}'
