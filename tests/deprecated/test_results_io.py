"""Tests for deprecated Results I/O functionality - ported from feature/v5.

This module contains the original test_flow_system_file_io test from feature/v5
that uses the deprecated Optimization/Results API. This test will be removed in v6.0.0.

For new tests, use FlowSystem.solution.to_netcdf() instead.
"""

import uuid

import pytest

import flixopt as fx
from flixopt.io import ResultsPaths

from ..conftest import (
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
def test_flow_system_file_io(flow_system, highs_solver, request):
    # Use UUID to ensure unique names across parallel test workers
    unique_id = uuid.uuid4().hex[:12]
    worker_id = getattr(request.config, 'workerinput', {}).get('workerid', 'main')
    test_id = f'{worker_id}-{unique_id}'

    calculation_0 = fx.Optimization(f'IO-{test_id}', flow_system=flow_system)
    calculation_0.do_modeling()
    calculation_0.solve(highs_solver)
    calculation_0.flow_system.plot_network()

    calculation_0.results.to_file()
    paths = ResultsPaths(calculation_0.folder, calculation_0.name)
    flow_system_1 = fx.FlowSystem.from_netcdf(paths.flow_system)

    calculation_1 = fx.Optimization(f'Loaded_IO-{test_id}', flow_system=flow_system_1)
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
