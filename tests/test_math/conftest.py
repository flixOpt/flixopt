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

import pandas as pd
import pytest

import flixopt as fx

_SOLVER = fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=60, log_to_console=False)


def make_flow_system(n_timesteps: int = 3) -> fx.FlowSystem:
    """Create a minimal FlowSystem with the given number of hourly timesteps."""
    ts = pd.date_range('2020-01-01', periods=n_timesteps, freq='h')
    return fx.FlowSystem(ts)


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
