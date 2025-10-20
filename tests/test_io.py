import pytest

import flixopt as fx
from flixopt.io import CalculationResultsPaths

from .conftest import (
    assert_almost_equal_numeric,
    flow_system_base,
    flow_system_long,
    flow_system_segments_of_flows_2,
    simple_flow_system,
    simple_flow_system_scenarios,
)


@pytest.fixture(
    params=[
        flow_system_base,
        simple_flow_system_scenarios,
        flow_system_segments_of_flows_2,
        simple_flow_system,
        flow_system_long,
    ]
)
def flow_system(request):
    fs = request.getfixturevalue(request.param.__name__)
    if isinstance(fs, fx.FlowSystem):
        return fs
    else:
        return fs[0]


@pytest.mark.slow
def test_flow_system_file_io(flow_system, highs_solver):
    calculation_0 = fx.FullCalculation('IO', flow_system=flow_system)
    calculation_0.do_modeling()
    calculation_0.solve(highs_solver)
    calculation_0.flow_system.plot_network()

    calculation_0.results.to_file()
    paths = CalculationResultsPaths(calculation_0.folder, calculation_0.name)
    flow_system_1 = fx.FlowSystem.from_netcdf(paths.flow_system)

    calculation_1 = fx.FullCalculation('Loaded_IO', flow_system=flow_system_1)
    calculation_1.do_modeling()
    calculation_1.solve(highs_solver)
    calculation_1.flow_system.plot_network()

    assert_almost_equal_numeric(
        calculation_0.results.model.objective.value,
        calculation_1.results.model.objective.value,
        'objective of loaded flow_system doesnt match the original',
    )

    assert_almost_equal_numeric(
        calculation_0.results.solution['costs'].values,
        calculation_1.results.solution['costs'].values,
        'costs doesnt match expected value',
    )


def test_flow_system_io(flow_system):
    flow_system.to_json('fs.json')

    ds = flow_system.to_dataset()
    new_fs = fx.FlowSystem.from_dataset(ds)

    assert flow_system == new_fs

    print(flow_system)
    flow_system.__repr__()
    flow_system.__str__()


def test_prevent_simultaneous_flows_single_group_roundtrip():
    """Test that single constraint group serializes and deserializes correctly."""
    import pandas as pd

    timesteps = pd.date_range('2020-01-01', periods=3, freq='h')
    fs = fx.FlowSystem(timesteps=timesteps)

    # Add buses
    fs.add_elements(
        fx.Bus('test'),
        fx.Bus('output_bus'),
    )

    # Create flows and converter with single constraint group
    flow1 = fx.Flow('flow1', bus='test', size=100)
    flow2 = fx.Flow('flow2', bus='test', size=100)
    flow3 = fx.Flow('flow3', bus='test', size=100)
    output = fx.Flow('output', bus='output_bus', size=200)

    conv = fx.LinearConverter(
        'test_conv',
        inputs=[flow1, flow2, flow3],
        outputs=[output],
        conversion_factors=[{'flow1': 1, 'output': 0.9}],
        prevent_simultaneous_flows=['flow1', 'flow2', 'flow3'],  # Single group (string labels)
    )

    fs.add_elements(conv)

    # Serialize and deserialize
    ds = fs.to_dataset()
    new_fs = fx.FlowSystem.from_dataset(ds)

    # Verify prevent_simultaneous_flows is preserved
    new_conv = new_fs.components['test_conv']
    assert new_conv.prevent_simultaneous_flows == [['flow1', 'flow2', 'flow3']]
    assert new_conv.prevent_simultaneous_flows == conv.prevent_simultaneous_flows


def test_prevent_simultaneous_flows_multiple_groups_roundtrip():
    """Test that multiple constraint groups serialize and deserialize correctly."""
    import pandas as pd

    timesteps = pd.date_range('2020-01-01', periods=3, freq='h')
    fs = fx.FlowSystem(timesteps=timesteps)

    # Add buses
    fs.add_elements(
        fx.Bus('fuel'),
        fx.Bus('cooling'),
        fx.Bus('steam'),
    )

    # Create flows for different constraint groups
    coal = fx.Flow('coal', bus='fuel', size=100)
    gas = fx.Flow('gas', bus='fuel', size=100)
    biomass = fx.Flow('biomass', bus='fuel', size=100)

    water_cooling = fx.Flow('water_cooling', bus='cooling', size=50)
    air_cooling = fx.Flow('air_cooling', bus='cooling', size=50)

    steam = fx.Flow('steam', bus='steam', size=200)

    # Create converter with two independent constraint groups
    conv = fx.LinearConverter(
        'power_plant',
        inputs=[coal, gas, biomass, water_cooling, air_cooling],
        outputs=[steam],
        conversion_factors=[{'coal': 1, 'steam': 0.8}],
        prevent_simultaneous_flows=[
            ['coal', 'gas', 'biomass'],  # Group 0: at most 1 fuel
            ['water_cooling', 'air_cooling'],  # Group 1: at most 1 cooling method
        ],
    )

    fs.add_elements(conv)

    # Serialize and deserialize
    ds = fs.to_dataset()
    new_fs = fx.FlowSystem.from_dataset(ds)

    # Verify prevent_simultaneous_flows is preserved with correct structure
    new_conv = new_fs.components['power_plant']
    expected = [['coal', 'gas', 'biomass'], ['water_cooling', 'air_cooling']]
    assert new_conv.prevent_simultaneous_flows == expected
    assert new_conv.prevent_simultaneous_flows == conv.prevent_simultaneous_flows


if __name__ == '__main__':
    pytest.main(['-v', '--disable-warnings'])
