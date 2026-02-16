"""Integration tests for FlowSystem.resample() - verifies correct data resampling and structure preservation."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import flixopt as fx


@pytest.fixture
def simple_fs():
    """Simple FlowSystem with basic components."""
    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
    fs = fx.FlowSystem(timesteps)
    fs.add_elements(
        fx.Bus('heat'), fx.Effect('costs', unit='€', description='costs', is_objective=True, is_standard=True)
    )
    fs.add_elements(
        fx.Sink(
            'demand',
            inputs=[fx.Flow(bus='heat', flow_id='in', fixed_relative_profile=np.linspace(10, 20, 24), size=1)],
        ),
        fx.Source(
            'source', outputs=[fx.Flow(bus='heat', flow_id='out', size=50, effects_per_flow_hour={'costs': 0.05})]
        ),
    )
    return fs


@pytest.fixture
def complex_fs():
    """FlowSystem with complex elements (storage, piecewise, invest)."""
    timesteps = pd.date_range('2023-01-01', periods=48, freq='h')
    fs = fx.FlowSystem(timesteps)

    fs.add_elements(
        fx.Bus('heat'),
        fx.Bus('elec'),
        fx.Effect('costs', unit='€', description='costs', is_objective=True, is_standard=True),
    )

    # Storage
    fs.add_elements(
        fx.Storage(
            'battery',
            charging=fx.Flow(bus='elec', flow_id='charge', size=10),
            discharging=fx.Flow(bus='elec', flow_id='discharge', size=10),
            capacity_in_flow_hours=fx.InvestParameters(fixed_size=100),
        )
    )

    # Piecewise converter
    converter = fx.linear_converters.Boiler(
        'boiler', thermal_efficiency=0.9, fuel_flow=fx.Flow(bus='elec', flow_id='gas'), thermal_flow=fx.Flow(bus='heat')
    )
    converter.thermal_flow.size = 100
    fs.add_elements(converter)

    # Component with investment
    fs.add_elements(
        fx.Source(
            'pv',
            outputs=[
                fx.Flow(
                    bus='elec',
                    flow_id='gen',
                    size=fx.InvestParameters(maximum_size=1000, effects_of_investment_per_size={'costs': 100}),
                )
            ],
        )
    )

    return fs


# === Basic Functionality ===


@pytest.mark.parametrize('freq,method', [('2h', 'mean'), ('4h', 'sum'), ('6h', 'first')])
def test_basic_resample(simple_fs, freq, method):
    """Test basic resampling preserves structure."""
    fs_r = simple_fs.resample(freq, method=method)
    assert len(fs_r.components) == len(simple_fs.components)
    assert len(fs_r.buses) == len(simple_fs.buses)
    assert len(fs_r.timesteps) < len(simple_fs.timesteps)


@pytest.mark.parametrize(
    'method,expected',
    [
        ('mean', [15.0, 35.0]),
        ('sum', [30.0, 70.0]),
        ('first', [10.0, 30.0]),
        ('last', [20.0, 40.0]),
    ],
)
def test_resample_methods(method, expected):
    """Test different resampling methods."""
    ts = pd.date_range('2023-01-01', periods=4, freq='h')
    fs = fx.FlowSystem(ts)
    fs.add_elements(fx.Bus('b'), fx.Effect('costs', unit='€', description='costs', is_objective=True, is_standard=True))
    fs.add_elements(
        fx.Sink(
            's',
            inputs=[fx.Flow(bus='b', flow_id='in', fixed_relative_profile=np.array([10.0, 20.0, 30.0, 40.0]), size=1)],
        )
    )

    fs_r = fs.resample('2h', method=method)
    assert_allclose(fs_r.flows['s(in)'].fixed_relative_profile.values, expected, rtol=1e-10)


def test_structure_preserved(simple_fs):
    """Test all structural elements preserved."""
    fs_r = simple_fs.resample('2h', method='mean')
    assert set(simple_fs.components.keys()) == set(fs_r.components.keys())
    assert set(simple_fs.buses.keys()) == set(fs_r.buses.keys())
    assert set(simple_fs.effects.keys()) == set(fs_r.effects.keys())

    # Flow connections preserved
    for label in simple_fs.flows.keys():
        assert simple_fs.flows[label].bus == fs_r.flows[label].bus
        assert simple_fs.flows[label].component == fs_r.flows[label].component


def test_time_metadata_updated(simple_fs):
    """Test time metadata correctly updated."""
    fs_r = simple_fs.resample('3h', method='mean')
    assert len(fs_r.timesteps) == 8
    assert_allclose(fs_r.timestep_duration.values, 3.0)
    assert fs_r.hours_of_last_timestep == 3.0


# === Advanced Dimensions ===


@pytest.mark.parametrize(
    'dim_name,dim_value',
    [
        ('periods', pd.Index([2023, 2024], name='period')),
        ('scenarios', pd.Index(['base', 'high'], name='scenario')),
    ],
)
def test_with_dimensions(simple_fs, dim_name, dim_value):
    """Test resampling preserves period/scenario dimensions."""
    fs = fx.FlowSystem(simple_fs.timesteps, **{dim_name: dim_value})
    fs.add_elements(fx.Bus('h'), fx.Effect('costs', unit='€', description='costs', is_objective=True, is_standard=True))
    fs.add_elements(fx.Sink('d', inputs=[fx.Flow(bus='h', flow_id='in', fixed_relative_profile=np.ones(24), size=1)]))

    fs_r = fs.resample('2h', method='mean')
    assert getattr(fs_r, dim_name) is not None
    pd.testing.assert_index_equal(getattr(fs_r, dim_name), dim_value)


# === Complex Elements ===


def test_storage_resample(complex_fs):
    """Test storage component resampling."""
    fs_r = complex_fs.resample('4h', method='mean')
    assert 'battery' in fs_r.components
    storage = fs_r.components['battery']
    assert storage.charging.label == 'charge'
    assert storage.discharging.label == 'discharge'


def test_converter_resample(complex_fs):
    """Test converter component resampling."""
    fs_r = complex_fs.resample('4h', method='mean')
    assert 'boiler' in fs_r.components
    boiler = fs_r.components['boiler']
    assert hasattr(boiler, 'thermal_efficiency')


def test_invest_resample(complex_fs):
    """Test investment parameters preserved."""
    fs_r = complex_fs.resample('4h', method='mean')
    pv_flow = fs_r.flows['pv(gen)']
    assert isinstance(pv_flow.size, fx.InvestParameters)
    assert pv_flow.size.maximum_size == 1000


# === Modeling Integration ===


@pytest.mark.parametrize('with_dim', [None, 'periods', 'scenarios'])
def test_modeling(with_dim):
    """Test resampled FlowSystem can be modeled."""
    ts = pd.date_range('2023-01-01', periods=48, freq='h')
    kwargs = {}
    if with_dim == 'periods':
        kwargs['periods'] = pd.Index([2023, 2024], name='period')
    elif with_dim == 'scenarios':
        kwargs['scenarios'] = pd.Index(['base', 'high'], name='scenario')

    fs = fx.FlowSystem(ts, **kwargs)
    fs.add_elements(fx.Bus('h'), fx.Effect('costs', unit='€', description='costs', is_objective=True, is_standard=True))
    fs.add_elements(
        fx.Sink('d', inputs=[fx.Flow(bus='h', flow_id='in', fixed_relative_profile=np.linspace(10, 30, 48), size=1)]),
        fx.Source('s', outputs=[fx.Flow(bus='h', flow_id='out', size=100, effects_per_flow_hour={'costs': 0.05})]),
    )

    fs_r = fs.resample('4h', method='mean')
    fs_r.build_model()

    assert fs_r.model is not None
    assert len(fs_r.model.variables) > 0


def test_model_structure_preserved():
    """Test model structure (var/constraint types) preserved."""
    ts = pd.date_range('2023-01-01', periods=48, freq='h')
    fs = fx.FlowSystem(ts)
    fs.add_elements(fx.Bus('h'), fx.Effect('costs', unit='€', description='costs', is_objective=True, is_standard=True))
    fs.add_elements(
        fx.Sink('d', inputs=[fx.Flow(bus='h', flow_id='in', fixed_relative_profile=np.linspace(10, 30, 48), size=1)]),
        fx.Source('s', outputs=[fx.Flow(bus='h', flow_id='out', size=100, effects_per_flow_hour={'costs': 0.05})]),
    )

    fs.build_model()

    fs_r = fs.resample('4h', method='mean')
    fs_r.build_model()

    # Same number of variable/constraint types
    assert len(fs.model.variables) == len(fs_r.model.variables)
    assert len(fs.model.constraints) == len(fs_r.model.constraints)

    # Same names
    assert set(fs.model.variables.labels.data_vars.keys()) == set(fs_r.model.variables.labels.data_vars.keys())
    assert set(fs.model.constraints.labels.data_vars.keys()) == set(fs_r.model.constraints.labels.data_vars.keys())


# === Advanced Features ===


def test_dataset_roundtrip(simple_fs):
    """Test dataset serialization."""
    fs_r = simple_fs.resample('2h', method='mean')
    assert fx.FlowSystem.from_dataset(fs_r.to_dataset()) == fs_r


def test_dataset_chaining(simple_fs):
    """Test power user pattern."""
    ds = simple_fs.to_dataset()
    ds = fx.FlowSystem._dataset_sel(ds, time='2023-01-01')
    ds = fx.FlowSystem._dataset_resample(ds, freq='2h', method='mean')
    fs_result = fx.FlowSystem.from_dataset(ds)

    fs_simple = simple_fs.sel(time='2023-01-01').resample('2h', method='mean')
    assert fs_result == fs_simple


@pytest.mark.parametrize('freq,exp_len', [('2h', 84), ('6h', 28), ('1D', 7)])
def test_frequencies(freq, exp_len):
    """Test various frequencies."""
    ts = pd.date_range('2023-01-01', periods=168, freq='h')
    fs = fx.FlowSystem(ts)
    fs.add_elements(fx.Bus('b'), fx.Effect('costs', unit='€', description='costs', is_objective=True, is_standard=True))
    fs.add_elements(fx.Sink('s', inputs=[fx.Flow(bus='b', flow_id='in', fixed_relative_profile=np.ones(168), size=1)]))

    assert len(fs.resample(freq, method='mean').timesteps) == exp_len


def test_irregular_timesteps_error():
    """Test that resampling irregular timesteps to finer resolution raises error without fill_gaps."""
    ts = pd.DatetimeIndex(['2023-01-01 00:00', '2023-01-01 01:00', '2023-01-01 03:00'], name='time')
    fs = fx.FlowSystem(ts)
    fs.add_elements(fx.Bus('b'), fx.Effect('costs', unit='€', description='costs', is_objective=True, is_standard=True))
    fs.add_elements(fx.Sink('s', inputs=[fx.Flow(bus='b', flow_id='in', fixed_relative_profile=np.ones(3), size=1)]))

    with pytest.raises(ValueError, match='Resampling created gaps'):
        fs.transform.resample('1h', method='mean')


def test_irregular_timesteps_with_fill_gaps():
    """Test that resampling irregular timesteps works with explicit fill_gaps strategy."""
    ts = pd.DatetimeIndex(['2023-01-01 00:00', '2023-01-01 01:00', '2023-01-01 03:00'], name='time')
    fs = fx.FlowSystem(ts)
    fs.add_elements(fx.Bus('b'), fx.Effect('costs', unit='€', description='costs', is_objective=True, is_standard=True))
    fs.add_elements(
        fx.Sink('s', inputs=[fx.Flow(bus='b', flow_id='in', fixed_relative_profile=np.array([1.0, 2.0, 4.0]), size=1)])
    )

    # Test with ffill
    fs_r = fs.transform.resample('1h', method='mean', fill_gaps='ffill')
    assert len(fs_r.timesteps) == 4
    # Gap at 02:00 should be filled with previous value (2.0)
    assert_allclose(fs_r.flows['s(in)'].fixed_relative_profile.values, [1.0, 2.0, 2.0, 4.0])


if __name__ == '__main__':
    pytest.main(['-v', __file__])
