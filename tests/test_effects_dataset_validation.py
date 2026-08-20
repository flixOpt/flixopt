"""Regression tests for the effect-total consistency check in ``_create_effects_dataset``.

The check compares ``ds[effect].sum(...)`` against ``solution[label]``. When those two
arrays carry the same dimensions in a different order (e.g. a clustered system expanded
back produces ``(scenario, period)`` where the computed total is ``(period, scenario)``),
comparing the raw ``.values`` with ``np.allclose`` used to raise a ``ValueError`` on the
shape mismatch -- turning a soft warning into a hard crash. The comparison must instead be
dimension-aware.
"""

import numpy as np
import pandas as pd
import pytest

import flixopt as fx


@pytest.fixture
def solved_multiperiod_scenario_system():
    """A solved system with 3 periods x 2 scenarios (different sizes on purpose)."""
    timesteps = pd.date_range('2024-01-01', periods=24, freq='h')
    periods = pd.Index([2024, 2025, 2026], name='period')
    scenarios = pd.Index(['high', 'low'], name='scenario')

    fs = fx.FlowSystem(timesteps, periods=periods, scenarios=scenarios)
    fs.add_elements(
        fx.Bus('heat'),
        fx.Effect('costs', unit='EUR', description='costs', is_objective=True, is_standard=True),
        fx.Sink('demand', inputs=[fx.Flow('in', bus='heat', fixed_relative_profile=np.ones(24), size=10)]),
        fx.Source('source', outputs=[fx.Flow('out', bus='heat', size=50, effects_per_flow_hour={'costs': 0.05})]),
    )
    fs.optimize(fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=60, log_to_console=False))
    return fs


@pytest.mark.parametrize('mode', ['temporal', 'periodic', 'total'])
def test_effects_dataset_tolerates_transposed_solution_dims(solved_multiperiod_scenario_system, mode):
    """Transposing an effect total in the solution must not crash the validation."""
    fs = solved_multiperiod_scenario_system
    suffix = {'temporal': '(temporal)|per_timestep', 'periodic': '(periodic)', 'total': ''}[mode]
    label = f'costs{suffix}'
    assert label in fs.solution

    found = fs.solution[label]
    transposable = [d for d in found.dims if d != 'time']
    assert len(transposable) >= 2  # need >=2 non-time dims to reorder into a shape mismatch

    # Force the opposite dim order the bug is about.
    reordered = [d for d in reversed(found.dims)]
    fs._solution[label] = found.transpose(*reordered)

    ds = fs.stats._create_effects_dataset(mode)
    assert 'costs' in ds
