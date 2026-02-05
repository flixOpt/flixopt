"""Shared helpers for mathematical correctness tests.

Each test in this directory builds a tiny, analytically solvable optimization
model and asserts that the objective (or key solution variables) match a
hand-calculated value. This catches regressions in formulations without
relying on recorded baselines.

The ``solve`` fixture is parametrized so every test runs twice: once solving
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


def _optimize(fs: fx.FlowSystem) -> fx.FlowSystem:
    """Run HiGHS (exact, silent) and return the same object."""
    fs.optimize(fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=60, log_to_console=False))
    return fs


@pytest.fixture(params=['direct', 'io_roundtrip'])
def solve(request):
    """Callable fixture that optimizes a FlowSystem.

    ``direct``       -- solve as-is.
    ``io_roundtrip`` -- serialize to Dataset, deserialize, solve, then patch
                        the result back onto the original object so callers'
                        references stay valid.
    """

    def _solve(fs: fx.FlowSystem) -> fx.FlowSystem:
        if request.param == 'io_roundtrip':
            ds = fs.to_dataset()
            fs_restored = fx.FlowSystem.from_dataset(ds)
            _optimize(fs_restored)
            fs.__dict__ = fs_restored.__dict__
            return fs
        return _optimize(fs)

    return _solve
