"""Round-trip regression matrix for clustered FlowSystems.

Derived from property-based fuzzing (save -> load -> expand) that exercised random
combinations of periods, scenarios, storage (all initial-charge modes), converters and
optimize on/off. These curated cases pin the representative combinations so the
save/load/expand path stays intact across dimension layouts.
"""

import numpy as np
import pandas as pd
import pytest

import flixopt as fx

SOLVER = fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=60, log_to_console=False)


def _build_system(n_days, periods, scenarios, storage, storage_init, converter):
    hours = n_days * 24
    timesteps = pd.date_range('2024-01-01', periods=hours, freq='h')
    period_idx = pd.Index(periods, name='period') if periods else None
    scenario_idx = pd.Index(scenarios, name='scenario') if scenarios else None

    daily = 0.5 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, 24))
    profile = np.clip(np.tile(daily, n_days) * 0.9, 0.01, None)

    extra = {'weight_of_last_period': 1.0} if period_idx is not None else {}
    fs = fx.FlowSystem(timesteps, periods=period_idx, scenarios=scenario_idx, **extra)
    elements = [
        fx.Bus('heat'),
        fx.Effect('costs', 'EUR', 'costs', is_objective=True, is_standard=True),
        fx.Sink('demand', inputs=[fx.Flow('in', bus='heat', fixed_relative_profile=profile, size=10)]),
        fx.Source('grid', outputs=[fx.Flow('g', bus='heat', size=1000, effects_per_flow_hour={'costs': 0.1})]),
    ]
    if converter:
        elements += [
            fx.Bus('gas'),
            fx.Source('gas_src', outputs=[fx.Flow('gs', bus='gas', size=1000, effects_per_flow_hour={'costs': 0.05})]),
            fx.LinearConverter(
                'boiler',
                inputs=[fx.Flow('bi', bus='gas', size=1000)],
                outputs=[fx.Flow('bo', bus='heat', size=1000)],
                conversion_factors=[{'bi': 1.0, 'bo': 0.9}],
            ),
        ]
    if storage:
        init = {'equals_final': 'equals_final', 'none': None, 'value': 2.0}[storage_init]
        elements.append(
            fx.Storage(
                'battery',
                charging=fx.Flow('sc', bus='heat', size=10),
                discharging=fx.Flow('sd', bus='heat', size=10),
                capacity_in_flow_hours=20,
                initial_charge_state=init,
            )
        )
    fs.add_elements(*elements)
    return fs


# (label, n_days, periods, scenarios, storage, storage_init, converter, n_clusters)
CONFIGS = [
    ('plain', 3, None, None, False, 'none', False, 2),
    ('periods', 4, [2020, 2025], None, False, 'none', False, 2),
    ('scenarios', 4, None, ['a', 'b'], False, 'none', False, 2),
    ('periods+scenarios', 3, [2020, 2025, 2030], ['a', 'b'], False, 'none', False, 2),
    ('storage-cyclic', 4, None, None, True, 'equals_final', False, 2),
    ('storage-free', 4, None, None, True, 'none', False, 2),
    ('storage-value', 4, None, None, True, 'value', False, 3),
    ('converter', 3, None, None, False, 'none', True, 2),
    ('storage+conv+scen', 4, None, ['a', 'b'], True, 'equals_final', True, 2),
    ('storage+periods', 4, [2020, 2025], None, True, 'value', False, 2),
    ('k1', 3, None, None, True, 'equals_final', False, 1),
    ('k-max', 3, None, None, False, 'none', False, 3),
]


@pytest.mark.parametrize('cfg', CONFIGS, ids=[c[0] for c in CONFIGS])
def test_clustered_roundtrip_and_expand(cfg, tmp_path):
    label, n_days, periods, scenarios, storage, storage_init, converter, n_clusters = cfg
    fs = _build_system(n_days, periods, scenarios, storage, storage_init, converter)
    clustered = fs.transform.cluster(n_clusters=n_clusters, cluster_duration='1D')
    clustered.optimize(SOLVER)

    costs_before = float(clustered.solution['costs'].sum().item())

    path = tmp_path / f'{label}.nc4'
    clustered.to_netcdf(path)
    reloaded = fx.FlowSystem.from_netcdf(path)

    assert reloaded.clustering is not None
    assert reloaded.clustering.n_clusters == n_clusters
    assert len(reloaded.timesteps) == len(clustered.timesteps)

    costs_after = float(reloaded.solution['costs'].sum().item())
    assert np.isclose(costs_before, costs_after, rtol=1e-6, atol=1e-4)

    # effect-total validation must not crash on any dimension layout
    reloaded.stats._create_effects_dataset('total')

    expanded = reloaded.transform.expand()
    assert len(expanded.timesteps) == n_days * 24


@pytest.mark.parametrize('cfg', CONFIGS, ids=[c[0] for c in CONFIGS])
def test_clustered_structure_only_roundtrip(cfg, tmp_path):
    """Round-trip of an unsolved (no-solution) clustered system must also reload."""
    label, n_days, periods, scenarios, storage, storage_init, converter, n_clusters = cfg
    fs = _build_system(n_days, periods, scenarios, storage, storage_init, converter)
    clustered = fs.transform.cluster(n_clusters=n_clusters, cluster_duration='1D')

    path = tmp_path / f'{label}_structure.nc4'
    clustered.to_netcdf(path)
    reloaded = fx.FlowSystem.from_netcdf(path)

    assert reloaded.clustering is not None
    assert reloaded.clustering.n_clusters == n_clusters
    assert len(reloaded.timesteps) == len(clustered.timesteps)
