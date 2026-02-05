"""Shared helpers for mathematical correctness tests.

Each test in this directory builds a tiny, analytically solvable optimization
model and asserts that the objective (or key solution variables) match a
hand-calculated value. This catches regressions in formulations without
relying on recorded baselines.

The ``optimize`` fixture is parametrized so every test runs three times,
each verifying a different pipeline:

``solve``
    Baseline correctness check.
``save->reload->solve``
    Proves the FlowSystem definition survives IO.
``solve->save->reload``
    Proves the solution data survives IO.
"""

import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

import flixopt as fx

# Enable legacy solution access for backward compatibility in tests
fx.CONFIG.Legacy.solution_access = True

_SOLVER = fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=60, log_to_console=False)


def make_flow_system(n_timesteps: int = 3) -> fx.FlowSystem:
    """Create a minimal FlowSystem with the given number of hourly timesteps."""
    ts = pd.date_range('2020-01-01', periods=n_timesteps, freq='h')
    return fx.FlowSystem(ts)


def make_multi_period_flow_system(
    n_timesteps: int = 3,
    periods=None,
    weight_of_last_period=None,
) -> fx.FlowSystem:
    """Create a FlowSystem with multi-period support."""
    ts = pd.date_range('2020-01-01', periods=n_timesteps, freq='h')
    if periods is None:
        periods = [2020, 2025]
    return fx.FlowSystem(
        ts,
        periods=pd.Index(periods, name='period'),
        weight_of_last_period=weight_of_last_period,
    )


def make_scenario_flow_system(
    n_timesteps: int = 3,
    scenarios=None,
    scenario_weights=None,
) -> fx.FlowSystem:
    """Create a FlowSystem with scenario support."""
    ts = pd.date_range('2020-01-01', periods=n_timesteps, freq='h')
    if scenarios is None:
        scenarios = ['low', 'high']
    if scenario_weights is not None and not isinstance(scenario_weights, np.ndarray):
        scenario_weights = np.array(scenario_weights)
    return fx.FlowSystem(
        ts,
        scenarios=pd.Index(scenarios, name='scenario'),
        scenario_weights=scenario_weights,
    )


def _netcdf_roundtrip(fs: fx.FlowSystem) -> fx.FlowSystem:
    """Save to NetCDF and reload."""
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / 'flow_system.nc'
        fs.to_netcdf(path)
        return fx.FlowSystem.from_netcdf(path)


@pytest.fixture(
    params=[
        'solve',
        'save->reload->solve',
        'solve->save->reload',
    ]
)
def optimize(request):
    """Callable fixture that optimizes a FlowSystem and returns it."""

    def _optimize(fs: fx.FlowSystem) -> fx.FlowSystem:
        if request.param == 'save->reload->solve':
            fs = _netcdf_roundtrip(fs)
        fs.optimize(_SOLVER)
        if request.param == 'solve->save->reload':
            fs = _netcdf_roundtrip(fs)
        return fs

    return _optimize
