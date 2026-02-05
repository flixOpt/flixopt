"""Shared helpers for mathematical correctness tests.

Each test in this directory builds a tiny, analytically solvable optimization
model and asserts that the objective (or key solution variables) match a
hand-calculated value. This catches regressions in formulations without
relying on recorded baselines.

The ``optimize`` fixture is parametrized so every test runs twice: once
directly, and once after a dataset round-trip (serialize then deserialize)
to verify IO preservation.
"""

import pandas as pd
import pytest

import flixopt as fx


def make_flow_system(n_timesteps: int = 3) -> fx.FlowSystem:
    """Create a minimal FlowSystem with the given number of hourly timesteps."""
    ts = pd.date_range('2020-01-01', periods=n_timesteps, freq='h')
    return fx.FlowSystem(ts)


@pytest.fixture(params=['direct', 'io_roundtrip'])
def optimize(request):
    """Callable fixture that optimizes a FlowSystem and returns it.

    ``direct``       -- optimize as-is.
    ``io_roundtrip`` -- serialize to Dataset, deserialize, then optimize.
    """

    def _optimize(fs: fx.FlowSystem) -> fx.FlowSystem:
        if request.param == 'io_roundtrip':
            ds = fs.to_dataset()
            fs = fx.FlowSystem.from_dataset(ds)
        fs.optimize(fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=60, log_to_console=False))
        return fs

    return _optimize
