"""Tests for flixopt.tables — table-based I/O for FlowSystem."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

import flixopt as fx
from flixopt.tables import from_dir, from_tables, to_dir, to_tables

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_simple_tables() -> dict[str, pl.DataFrame]:
    """Build minimal tables for a simple boiler + heat-load system."""
    timesteps = pl.DataFrame({
        'time': pd.date_range('2020-01-01', periods=5, freq='h'),
        'dt': [1.0] * 5,
    })

    buses = pl.DataFrame({
        'bus': ['Gas', 'Heat'],
        'carrier': [None, None],
        'imbalance_penalty': [None, None],
    })

    effects = pl.DataFrame({
        'effect': ['costs'],
        'unit': ['€'],
        'is_standard': [True],
        'is_objective': [True],
    })

    flows = pl.DataFrame({
        'flow': ['Boiler(fuel)', 'Boiler(heat)', 'GasTariff(gas)', 'HeatLoad(demand)'],
        'bus': ['Gas', 'Heat', 'Gas', 'Heat'],
        'component': ['Boiler', 'Boiler', 'GasTariff', 'HeatLoad'],
        'direction': ['in', 'out', 'out', 'in'],
        'size': [200.0, 200.0, 1000.0, 1.0],
        'rel_min': [0.0, 0.0, 0.0, 0.0],
        'rel_max': [1.0, 1.0, 1.0, 1.0],
    })

    # Note: conversion equation is sum(coeff * rate * sign) = 0
    # where sign = -1 for inputs, +1 for outputs.
    # So {fuel: 0.9, heat: 1} → 0.9*fuel*(-1) + 1*heat*(+1) = 0 → heat = 0.9*fuel
    converters = pl.DataFrame({
        'converter': ['Boiler', 'Boiler'],
        'eq_idx': [0, 0],
        'flow': ['Boiler(fuel)', 'Boiler(heat)'],
        'value': [0.9, 1.0],
    })

    flow_effects = pl.DataFrame({
        'flow': ['GasTariff(gas)'],
        'effect': ['costs'],
        'value': [0.04],
    })

    demand_profile = np.array([30.0, 0.0, 90.0, 110.0, 20.0])
    flow_profiles = pl.DataFrame({
        'flow': ['HeatLoad(demand)'] * 5,
        'time': pd.date_range('2020-01-01', periods=5, freq='h'),
        'fixed_profile': demand_profile.tolist(),
    })

    return {
        'timesteps': timesteps,
        'buses': buses,
        'effects': effects,
        'flows': flows,
        'converters': converters,
        'flow_effects': flow_effects,
        'flow_profiles': flow_profiles,
    }


# ---------------------------------------------------------------------------
# Phase 1 — Core
# ---------------------------------------------------------------------------


class TestFromTablesBasic:
    """Test from_tables() with simple table definitions."""

    def test_returns_flow_system(self):
        tables = _build_simple_tables()
        fs = from_tables(tables)
        assert type(fs).__name__ == 'FlowSystem'

    def test_timesteps_parsed(self):
        tables = _build_simple_tables()
        fs = from_tables(tables)
        assert len(fs.model_coords.timesteps) == 5

    def test_buses_created(self):
        tables = _build_simple_tables()
        fs = from_tables(tables)
        assert 'Gas' in fs.buses
        assert 'Heat' in fs.buses

    def test_effects_created(self):
        tables = _build_simple_tables()
        fs = from_tables(tables)
        assert 'costs' in fs.effects
        assert fs.effects['costs'].is_objective is True
        assert fs.effects['costs'].is_standard is True

    def test_components_created(self):
        tables = _build_simple_tables()
        fs = from_tables(tables)
        assert 'Boiler' in fs.components
        assert 'GasTariff' in fs.components
        assert 'HeatLoad' in fs.components

    def test_component_types_inferred(self):
        tables = _build_simple_tables()
        fs = from_tables(tables)
        assert type(fs.components['Boiler']).__name__ == 'LinearConverter'
        assert type(fs.components['GasTariff']).__name__ == 'Source'
        assert type(fs.components['HeatLoad']).__name__ == 'Sink'

    def test_flow_effects_applied(self):
        tables = _build_simple_tables()
        fs = from_tables(tables)
        gas_flow = fs.components['GasTariff'].outputs[0]
        assert 'costs' in gas_flow.effects_per_flow_hour
        assert gas_flow.effects_per_flow_hour['costs'] == 0.04

    def test_conversion_factors_applied(self):
        tables = _build_simple_tables()
        fs = from_tables(tables)
        boiler = fs.components['Boiler']
        assert len(boiler.conversion_factors) == 1
        assert 'fuel' in boiler.conversion_factors[0]

    def test_solves(self, highs_solver):
        tables = _build_simple_tables()
        fs = from_tables(tables)
        fs.optimize(highs_solver)
        assert fs.solution is not None


class TestToTables:
    """Test to_tables() produces valid table dicts."""

    def test_required_tables_present(self):
        tables = _build_simple_tables()
        fs = from_tables(tables)
        result = to_tables(fs)
        assert 'timesteps' in result
        assert 'buses' in result
        assert 'effects' in result
        assert 'flows' in result

    def test_timesteps_roundtrip(self):
        tables = _build_simple_tables()
        fs = from_tables(tables)
        result = to_tables(fs)
        assert len(result['timesteps']) == 5

    def test_buses_roundtrip(self):
        tables = _build_simple_tables()
        fs = from_tables(tables)
        result = to_tables(fs)
        bus_labels = set(result['buses']['bus'].to_list())
        assert bus_labels == {'Gas', 'Heat'}

    def test_effects_roundtrip(self):
        tables = _build_simple_tables()
        fs = from_tables(tables)
        result = to_tables(fs)
        effect_labels = set(result['effects']['effect'].to_list())
        assert 'costs' in effect_labels

    def test_flows_roundtrip(self):
        tables = _build_simple_tables()
        fs = from_tables(tables)
        result = to_tables(fs)
        flow_labels = set(result['flows']['flow'].to_list())
        assert 'Boiler(fuel)' in flow_labels
        assert 'Boiler(heat)' in flow_labels

    def test_converters_roundtrip(self):
        tables = _build_simple_tables()
        fs = from_tables(tables)
        result = to_tables(fs)
        assert 'converters' in result
        assert len(result['converters']) >= 2  # 2 flows in one equation

    def test_flow_effects_roundtrip(self):
        tables = _build_simple_tables()
        fs = from_tables(tables)
        result = to_tables(fs)
        assert 'flow_effects' in result
        fe = result['flow_effects']
        gas_effects = fe.filter(pl.col('flow') == 'GasTariff(gas)')
        assert len(gas_effects) >= 1


class TestRoundtrip:
    """Verify objects → tables → objects roundtrip produces equivalent models."""

    def test_simple_roundtrip_solves(self, highs_solver):
        """Build from tables, convert back, rebuild, compare objectives."""
        tables = _build_simple_tables()
        fs1 = from_tables(tables)
        fs1.optimize(highs_solver)
        obj1 = float(fs1.solution['effect|total'].sel(effect='costs').values)

        tables2 = to_tables(fs1)
        fs2 = from_tables(tables2)
        fs2.optimize(highs_solver)
        obj2 = float(fs2.solution['effect|total'].sel(effect='costs').values)

        np.testing.assert_allclose(obj1, obj2, rtol=1e-4)

    def test_object_api_roundtrip(self, highs_solver):
        """Build via object API → to_tables → from_tables → solve → compare."""
        timesteps = pd.date_range('2020-01-01', periods=5, freq='h', name='time')
        fs1 = fx.FlowSystem(timesteps)
        fs1.add_elements(
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Bus('Gas'),
            fx.Bus('Heat'),
            fx.Source('GasTariff', outputs=[
                fx.Flow('gas', bus='Gas', size=1000, effects_per_flow_hour={'costs': 0.04}),
            ]),
            fx.Sink('HeatLoad', inputs=[
                fx.Flow('demand', bus='Heat', size=1,
                        fixed_relative_profile=np.array([30.0, 0.0, 90.0, 110.0, 20.0])),
            ]),
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=0.9,
                thermal_flow=fx.Flow('heat', bus='Heat', size=200),
                fuel_flow=fx.Flow('fuel', bus='Gas'),
            ),
        )
        fs1.optimize(highs_solver)
        obj1 = float(fs1.solution['effect|total'].sel(effect='costs').values)

        tables = to_tables(fs1)
        fs2 = from_tables(tables)
        fs2.optimize(highs_solver)
        obj2 = float(fs2.solution['effect|total'].sel(effect='costs').values)

        np.testing.assert_allclose(obj1, obj2, rtol=1e-4)


class TestDirectConstruction:
    """Build a model directly from DataFrames and verify it solves correctly."""

    def test_source_sink_only(self, highs_solver):
        """Simplest possible model: source → bus → sink."""
        tables = {
            'timesteps': pl.DataFrame({
                'time': pd.date_range('2020-01-01', periods=3, freq='h'),
                'dt': [1.0, 1.0, 1.0],
            }),
            'buses': pl.DataFrame({'bus': ['Elec'], 'carrier': [None], 'imbalance_penalty': [None]}),
            'effects': pl.DataFrame({
                'effect': ['costs'],
                'unit': ['€'],
                'is_standard': [True],
                'is_objective': [True],
            }),
            'flows': pl.DataFrame({
                'flow': ['Grid(power)', 'Load(demand)'],
                'bus': ['Elec', 'Elec'],
                'component': ['Grid', 'Load'],
                'direction': ['out', 'in'],
                'size': [100.0, 1.0],
                'rel_min': [0.0, 0.0],
                'rel_max': [1.0, 1.0],
            }),
            'flow_effects': pl.DataFrame({
                'flow': ['Grid(power)'],
                'effect': ['costs'],
                'value': [0.10],
            }),
            'flow_profiles': pl.DataFrame({
                'flow': ['Load(demand)'] * 3,
                'time': pd.date_range('2020-01-01', periods=3, freq='h'),
                'fixed_profile': [10.0, 20.0, 15.0],
            }),
        }
        fs = from_tables(tables)
        fs.optimize(highs_solver)

        # Total cost = sum(demand * price) = (10 + 20 + 15) * 0.10 = 4.5
        total_cost = float(fs.solution['effect|total'].sel(effect='costs').values)
        np.testing.assert_allclose(total_cost, 4.5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Phase 2 — Investment + Status + Effect Bounds
# ---------------------------------------------------------------------------


class TestInvestments:
    """Test investment table parsing."""

    def test_invest_from_tables(self, highs_solver):
        tables = _build_simple_tables()
        # Add an investment on the boiler heat flow
        tables['investments'] = pl.DataFrame({
            'element': ['Boiler(heat)'],
            'element_type': ['flow'],
            'fixed_size': [None],
            'minimum_size': [10.0],
            'maximum_size': [200.0],
            'mandatory': [False],
            'linked_periods': [None],
        })
        tables['investment_effects'] = pl.DataFrame({
            'element': ['Boiler(heat)'],
            'effect': ['costs'],
            'per_size': [5.0],
            'on_invest': [None],
            'on_retire': [None],
        })
        # Remove fixed size from flows table
        flows = tables['flows'].to_dicts()
        for row in flows:
            if row['flow'] == 'Boiler(heat)':
                row['size'] = None
        tables['flows'] = pl.DataFrame(flows)

        fs = from_tables(tables)
        boiler = fs.components['Boiler']
        heat_flow = None
        for f in boiler.outputs.values():
            if f.label == 'heat':
                heat_flow = f
                break
        assert heat_flow is not None
        assert type(heat_flow.size).__name__ == 'InvestParameters'
        assert heat_flow.size.minimum_size == 10.0
        assert heat_flow.size.maximum_size == 200.0

        fs.optimize(highs_solver)
        assert fs.solution is not None


class TestStatus:
    """Test status table parsing."""

    def test_status_from_tables(self, highs_solver):
        tables = _build_simple_tables()
        tables['status'] = pl.DataFrame({
            'element': ['Boiler(heat)'],
            'element_type': ['flow'],
            'active_hours_min': [None],
            'active_hours_max': [None],
            'startup_limit': [None],
            'min_uptime': [None],
            'max_uptime': [None],
            'min_downtime': [None],
            'max_downtime': [None],
            'force_startup_tracking': [False],
            'cluster_mode': ['relaxed'],
        })
        tables['status_effects'] = pl.DataFrame({
            'element': ['Boiler(heat)'],
            'effect': ['costs'],
            'per_startup': [10.0],
            'per_active_hour': [None],
        })

        fs = from_tables(tables)
        heat_flow = None
        for f in fs.components['Boiler'].outputs.values():
            if f.label == 'heat':
                heat_flow = f
                break
        assert heat_flow is not None
        assert heat_flow.status_parameters is not None
        assert heat_flow.status_parameters.effects_per_startup == {'costs': 10.0}


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class TestStorage:
    """Test storage table parsing and roundtrip."""

    def _build_storage_tables(self) -> dict[str, pl.DataFrame]:
        """Build tables with a simple storage system."""
        timesteps = pl.DataFrame({
            'time': pd.date_range('2020-01-01', periods=5, freq='h'),
            'dt': [1.0] * 5,
        })
        buses = pl.DataFrame({
            'bus': ['Heat', 'Gas'],
            'carrier': [None, None],
            'imbalance_penalty': [None, None],
        })
        effects = pl.DataFrame({
            'effect': ['costs'],
            'unit': ['€'],
            'is_standard': [True],
            'is_objective': [True],
        })
        flows = pl.DataFrame({
            'flow': [
                'Battery(charge)', 'Battery(discharge)',
                'GasTariff(gas)', 'HeatLoad(demand)',
                'Boiler(fuel)', 'Boiler(heat)',
            ],
            'bus': ['Heat', 'Heat', 'Gas', 'Heat', 'Gas', 'Heat'],
            'component': ['Battery', 'Battery', 'GasTariff', 'HeatLoad', 'Boiler', 'Boiler'],
            'direction': ['in', 'out', 'out', 'in', 'in', 'out'],
            'size': [50.0, 50.0, 1000.0, 1.0, 200.0, 100.0],
            'rel_min': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'rel_max': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        })
        storages = pl.DataFrame({
            'storage': ['Battery'],
            'charge_flow': ['Battery(charge)'],
            'discharge_flow': ['Battery(discharge)'],
            'capacity': [100.0],
            'eta_charge': [0.95],
            'eta_discharge': [0.95],
            'relative_loss_per_hour': [0.0],
            'initial_charge_state': ['0.0'],
            'rel_min_charge_state': [0.0],
            'rel_max_charge_state': [1.0],
            'prevent_simultaneous': [True],
            'balanced': [False],
            'cluster_mode': ['intercluster_cyclic'],
        })
        converters = pl.DataFrame({
            'converter': ['Boiler', 'Boiler'],
            'eq_idx': [0, 0],
            'flow': ['Boiler(fuel)', 'Boiler(heat)'],
            'value': [0.9, 1.0],
        })
        flow_effects = pl.DataFrame({
            'flow': ['GasTariff(gas)'],
            'effect': ['costs'],
            'value': [0.04],
        })
        flow_profiles = pl.DataFrame({
            'flow': ['HeatLoad(demand)'] * 5,
            'time': pd.date_range('2020-01-01', periods=5, freq='h'),
            'fixed_profile': [30.0, 0.0, 90.0, 110.0, 20.0],
        })
        return {
            'timesteps': timesteps,
            'buses': buses,
            'effects': effects,
            'flows': flows,
            'storages': storages,
            'converters': converters,
            'flow_effects': flow_effects,
            'flow_profiles': flow_profiles,
        }

    def test_storage_created(self):
        tables = self._build_storage_tables()
        fs = from_tables(tables)
        assert 'Battery' in fs.components
        storage = fs.components['Battery']
        assert type(storage).__name__ == 'Storage'
        assert storage.eta_charge == 0.95
        assert storage.capacity_in_flow_hours == 100.0

    def test_storage_solves(self, highs_solver):
        tables = self._build_storage_tables()
        fs = from_tables(tables)
        fs.optimize(highs_solver)
        assert fs.solution is not None

    def test_storage_roundtrip(self, highs_solver):
        tables = self._build_storage_tables()
        fs1 = from_tables(tables)
        fs1.optimize(highs_solver)
        obj1 = float(fs1.solution['effect|total'].sel(effect='costs').values)

        tables2 = to_tables(fs1)
        assert 'storages' in tables2
        fs2 = from_tables(tables2)
        fs2.optimize(highs_solver)
        obj2 = float(fs2.solution['effect|total'].sel(effect='costs').values)

        np.testing.assert_allclose(obj1, obj2, rtol=1e-4)


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


class TestCsvIO:
    """Test from_dir / to_dir CSV roundtrip."""

    def test_csv_roundtrip(self, highs_solver, tmp_path):
        tables = _build_simple_tables()
        fs1 = from_tables(tables)
        fs1.optimize(highs_solver)
        obj1 = float(fs1.solution['effect|total'].sel(effect='costs').values)

        # Write to CSV
        to_dir(fs1, tmp_path / 'model')

        # Read back
        fs2 = from_dir(tmp_path / 'model')
        fs2.optimize(highs_solver)
        obj2 = float(fs2.solution['effect|total'].sel(effect='costs').values)

        np.testing.assert_allclose(obj1, obj2, rtol=1e-4)

    def test_dir_not_found(self):
        with pytest.raises(FileNotFoundError):
            from_dir('/nonexistent/path')


# ---------------------------------------------------------------------------
# Time-varying effects
# ---------------------------------------------------------------------------


class TestTimeVaryingEffects:
    """Test time-varying flow effects."""

    def test_time_varying_flow_effects(self, highs_solver):
        """Build model with time-varying electricity price."""
        prices = [0.05, 0.10, 0.15, 0.10, 0.05]
        tables = {
            'timesteps': pl.DataFrame({
                'time': pd.date_range('2020-01-01', periods=5, freq='h'),
                'dt': [1.0] * 5,
            }),
            'buses': pl.DataFrame({'bus': ['Elec'], 'carrier': [None], 'imbalance_penalty': [None]}),
            'effects': pl.DataFrame({
                'effect': ['costs'], 'unit': ['€'],
                'is_standard': [True], 'is_objective': [True],
            }),
            'flows': pl.DataFrame({
                'flow': ['Grid(power)', 'Load(demand)'],
                'bus': ['Elec', 'Elec'],
                'component': ['Grid', 'Load'],
                'direction': ['out', 'in'],
                'size': [100.0, 1.0],
                'rel_min': [0.0, 0.0],
                'rel_max': [1.0, 1.0],
            }),
            'flow_effects': pl.DataFrame({
                'flow': ['Grid(power)'] * 5,
                'effect': ['costs'] * 5,
                'time': pd.date_range('2020-01-01', periods=5, freq='h'),
                'value': prices,
            }),
            'flow_profiles': pl.DataFrame({
                'flow': ['Load(demand)'] * 5,
                'time': pd.date_range('2020-01-01', periods=5, freq='h'),
                'fixed_profile': [10.0, 20.0, 15.0, 10.0, 5.0],
            }),
        }
        fs = from_tables(tables)
        fs.optimize(highs_solver)

        expected_cost = sum(d * p for d, p in zip([10, 20, 15, 10, 5], prices, strict=True))
        actual_cost = float(fs.solution['effect|total'].sel(effect='costs').values)
        np.testing.assert_allclose(actual_cost, expected_cost, rtol=1e-4)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    """Test error handling."""

    def test_missing_required_table(self):
        with pytest.raises(KeyError, match='timesteps'):
            from_tables({})

    def test_missing_buses(self):
        tables = _build_simple_tables()
        del tables['buses']
        with pytest.raises(KeyError, match='buses'):
            from_tables(tables)

    def test_missing_effects(self):
        tables = _build_simple_tables()
        del tables['effects']
        with pytest.raises(KeyError, match='effects'):
            from_tables(tables)

    def test_missing_flows(self):
        tables = _build_simple_tables()
        del tables['flows']
        with pytest.raises(KeyError, match='flows'):
            from_tables(tables)
