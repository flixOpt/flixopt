"""Shared helpers for mathematical correctness tests.

Each test in this directory builds a tiny, analytically solvable optimization
model and asserts that the objective (or key solution variables) match a
hand-calculated value. This catches regressions in formulations without
relying on recorded baselines.
"""

import pandas as pd

import flixopt as fx


def make_flow_system(n_timesteps: int = 3) -> fx.FlowSystem:
    """Create a minimal FlowSystem with the given number of hourly timesteps."""
    ts = pd.date_range('2020-01-01', periods=n_timesteps, freq='h')
    return fx.FlowSystem(ts)


def solve(fs: fx.FlowSystem) -> fx.FlowSystem:
    """Optimize a FlowSystem with HiGHS (exact, silent)."""
    fs.optimize(fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=60, log_to_console=False))
    return fs
