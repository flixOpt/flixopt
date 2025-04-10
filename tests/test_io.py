from typing import Dict, List, Optional, Union

import pytest

import flixopt as fx
from flixopt.io import CalculationResultsPaths

from .conftest import (
    assert_almost_equal_numeric,
    flow_system_base,
    flow_system_long,
    flow_system_segments_of_flows_2,
    simple_flow_system,
)


@pytest.fixture(params=[flow_system_base, flow_system_segments_of_flows_2, simple_flow_system, flow_system_long])
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

    calculation_0.results.to_file()
    paths = CalculationResultsPaths(calculation_0.folder, calculation_0.name)
    flow_system_1 = fx.FlowSystem.from_netcdf(paths.flow_system)

    calculation_1 = fx.FullCalculation('Loaded_IO', flow_system=flow_system_1)
    calculation_1.do_modeling()
    calculation_1.solve(highs_solver)

    assert_almost_equal_numeric(
        calculation_0.results.model.objective.value,
        calculation_1.results.model.objective.value,
        'objective of loaded flow_system doesnt match the original',
    )

    assert_almost_equal_numeric(
        calculation_0.results.solution['costs|total'].values,
        calculation_1.results.solution['costs|total'].values,
        'costs doesnt match expected value',
    )


def test_flow_system_io(flow_system):
    di = flow_system.as_dict()
    _ = fx.FlowSystem.from_dict(di)

    ds = flow_system.as_dataset()
    _ = fx.FlowSystem.from_dataset(ds)

    print(flow_system)
    flow_system.__repr__()
    flow_system.__str__()


if __name__ == '__main__':
    pytest.main(['-v', '--disable-warnings'])
