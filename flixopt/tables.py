"""Table-based I/O for FlowSystem model definitions.

Converts between ``dict[str, polars.DataFrame]`` and :class:`FlowSystem`,
enabling CSV/DataFrame-driven model construction without Python object code.

Conversion chain (existing code unchanged)::

    DataFrames/CSVs  <-->  tables.py  <-->  (Flow, Bus, ...) objects  -->  FlowSystem

Install the optional dependency with::

    pip install flixopt[tables]

Public API:
    - :func:`from_tables` — Build a FlowSystem from a dict of polars DataFrames.
    - :func:`to_tables` — Extract all model data as a dict of polars DataFrames.
    - :func:`from_dir` — Load CSVs from a directory and call :func:`from_tables`.
    - :func:`to_dir` — Call :func:`to_tables` and write CSVs to a directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


def _import_polars() -> Any:
    """Import polars with a helpful error message if not installed."""
    try:
        import polars as pl
    except ImportError:
        raise ImportError(
            'The `polars` package is required for table-based I/O. Install it with: pip install flixopt[tables]'
        ) from None
    return pl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def from_tables(
    tables: dict[str, pl.DataFrame],
    *,
    weight_of_last_period: float | None = None,
    hours_of_last_timestep: float | None = None,
    hours_of_previous_timesteps: float | np.ndarray | None = None,
    scenario_independent_sizes: bool | list[str] = True,
    scenario_independent_flow_rates: bool | list[str] = False,
    name: str | None = None,
) -> Any:
    """Build a :class:`FlowSystem` from a dict of polars DataFrames.

    Args:
        tables: Mapping of table name to polars DataFrame. Required tables:
            ``timesteps``, ``buses``, ``effects``, ``flows``.
        weight_of_last_period: Passed through to :class:`FlowSystem`.
        hours_of_last_timestep: Passed through to :class:`FlowSystem`.
        hours_of_previous_timesteps: Passed through to :class:`FlowSystem`.
        scenario_independent_sizes: Passed through to :class:`FlowSystem`.
        scenario_independent_flow_rates: Passed through to :class:`FlowSystem`.
        name: Optional name for the FlowSystem.

    Returns:
        A fully constructed :class:`FlowSystem` ready for optimization.
    """
    pl = _import_polars()
    from .components import LinearConverter, Sink, Source, SourceAndSink, Storage, Transmission
    from .effects import Effect
    from .elements import Bus, Flow
    from .flow_system import FlowSystem
    from .interface import InvestParameters, Piece, Piecewise, PiecewiseConversion, PiecewiseEffects, StatusParameters

    # ------------------------------------------------------------------
    # 1. Parse timesteps
    # ------------------------------------------------------------------
    ts_df = _require_table(tables, 'timesteps')
    timesteps, timestep_duration = _parse_timesteps(ts_df)

    # ------------------------------------------------------------------
    # 2. Parse periods / scenarios (Phase 3)
    # ------------------------------------------------------------------
    periods = None
    scenario_weights = None
    scenarios = None

    if 'periods' in tables and len(tables['periods']) > 0:
        periods_df = tables['periods']
        period_values = periods_df['period'].to_list()
        periods = pd.Index(period_values, name='period')
        if 'weight' in periods_df.columns:
            weights = periods_df['weight'].to_list()
            if not all(w is None for w in weights):
                # Use weight of last period from explicit column
                last_w = weights[-1]
                if last_w is not None and weight_of_last_period is None:
                    weight_of_last_period = float(last_w)

    if 'scenarios' in tables and len(tables['scenarios']) > 0:
        scenarios_df = tables['scenarios']
        scenario_values = scenarios_df['scenario'].to_list()
        scenarios = pd.Index(scenario_values, name='scenario')
        if 'weight' in scenarios_df.columns:
            scenario_weights = np.array(scenarios_df['weight'].to_list(), dtype=float)

    # ------------------------------------------------------------------
    # 3. Parse buses
    # ------------------------------------------------------------------
    buses_df = _require_table(tables, 'buses')
    bus_objects: dict[str, Bus] = {}
    for row in buses_df.iter_rows(named=True):
        bus_kwargs: dict[str, Any] = {'label': row['bus']}
        if row.get('carrier') is not None:
            bus_kwargs['carrier'] = row['carrier']
        if row.get('imbalance_penalty') is not None:
            bus_kwargs['imbalance_penalty_per_flow_hour'] = row['imbalance_penalty']
        bus_objects[row['bus']] = Bus(**bus_kwargs)

    # Handle bus_timeseries for time-varying imbalance_penalty
    if 'bus_timeseries' in tables and len(tables['bus_timeseries']) > 0:
        bus_ts_df = tables['bus_timeseries']
        for bus_label in bus_ts_df['bus'].unique().to_list():
            sub = bus_ts_df.filter(pl.col('bus') == bus_label)
            if 'imbalance_penalty' in sub.columns:
                ts_data = _rows_to_numeric(sub, 'imbalance_penalty', timesteps, periods, scenarios)
                bus_objects[bus_label].imbalance_penalty_per_flow_hour = ts_data

    # ------------------------------------------------------------------
    # 4. Parse effects + effect_bounds + effect_shares
    # ------------------------------------------------------------------
    effects_df = _require_table(tables, 'effects')
    effect_objects: dict[str, Effect] = {}
    for row in effects_df.iter_rows(named=True):
        effect_kwargs: dict[str, Any] = {
            'label': row['effect'],
            'unit': row['unit'],
        }
        if row.get('is_standard'):
            effect_kwargs['is_standard'] = True
        if row.get('is_objective'):
            effect_kwargs['is_objective'] = True
        effect_objects[row['effect']] = Effect(**effect_kwargs)

    # effect_bounds (Phase 2)
    if 'effect_bounds' in tables and len(tables['effect_bounds']) > 0:
        _apply_effect_bounds(tables['effect_bounds'], effect_objects, timesteps, periods, scenarios, pl)

    # effect_shares (Phase 3)
    if 'effect_shares' in tables and len(tables['effect_shares']) > 0:
        _apply_effect_shares(tables['effect_shares'], effect_objects, timesteps, periods, scenarios, pl)

    # ------------------------------------------------------------------
    # 5. Parse investments table (Phase 2)
    # ------------------------------------------------------------------
    invest_map: dict[str, dict[str, Any]] = {}  # element_label -> invest kwargs
    if 'investments' in tables and len(tables['investments']) > 0:
        invest_map = _parse_investments(tables['investments'], pl, periods, scenarios)

    # Parse investment_effects (Phase 2)
    invest_effects_map: dict[str, dict[str, Any]] = {}
    if 'investment_effects' in tables and len(tables['investment_effects']) > 0:
        invest_effects_map = _parse_investment_effects(tables['investment_effects'], pl, periods, scenarios)

    # Parse piecewise_investment_effects (Phase 3)
    piecewise_invest_map: dict[str, PiecewiseEffects] = {}
    if 'piecewise_investment_effects' in tables and len(tables['piecewise_investment_effects']) > 0:
        piecewise_invest_map = _parse_piecewise_investment_effects(
            tables['piecewise_investment_effects'], pl, timesteps, periods, scenarios
        )

    # ------------------------------------------------------------------
    # 6. Parse status table (Phase 2)
    # ------------------------------------------------------------------
    status_map: dict[str, StatusParameters] = {}
    if 'status' in tables and len(tables['status']) > 0:
        status_map = _parse_status(tables['status'], tables.get('status_effects'), pl, timesteps, periods, scenarios)

    # ------------------------------------------------------------------
    # 7. Parse flows + flow_profiles + flow_effects
    # ------------------------------------------------------------------
    flows_df = _require_table(tables, 'flows')
    flow_objects: dict[str, Flow] = {}  # keyed by flow label (qualified: "comp(flow)")
    flow_component_map: dict[str, str] = {}  # flow_label -> component_label
    flow_direction_map: dict[str, str] = {}  # flow_label -> "in" or "out"

    # Group flow_profiles by flow label
    flow_profiles: dict[str, dict[str, Any]] = {}
    if 'flow_profiles' in tables and len(tables['flow_profiles']) > 0:
        fp_df = tables['flow_profiles']
        for flow_label in fp_df['flow'].unique().to_list():
            sub = fp_df.filter(pl.col('flow') == flow_label)
            flow_profiles[flow_label] = {}
            if 'fixed_profile' in sub.columns and sub['fixed_profile'].drop_nulls().len() > 0:
                flow_profiles[flow_label]['fixed_relative_profile'] = _rows_to_numeric(
                    sub, 'fixed_profile', timesteps, periods, scenarios
                )
            if 'rel_min' in sub.columns and sub['rel_min'].drop_nulls().len() > 0:
                flow_profiles[flow_label]['relative_minimum'] = _rows_to_numeric(
                    sub, 'rel_min', timesteps, periods, scenarios
                )
            if 'rel_max' in sub.columns and sub['rel_max'].drop_nulls().len() > 0:
                flow_profiles[flow_label]['relative_maximum'] = _rows_to_numeric(
                    sub, 'rel_max', timesteps, periods, scenarios
                )

    # Group flow_effects by flow label
    flow_effects: dict[str, dict[str, Any]] = {}  # flow_label -> {effect_label: numeric}
    if 'flow_effects' in tables and len(tables['flow_effects']) > 0:
        fe_df = tables['flow_effects']
        for flow_label in fe_df['flow'].unique().to_list():
            flow_sub = fe_df.filter(pl.col('flow') == flow_label)
            effects_dict: dict[str, Any] = {}
            for effect_label in flow_sub['effect'].unique().to_list():
                effect_sub = flow_sub.filter(pl.col('effect') == effect_label)
                if 'time' in effect_sub.columns and effect_sub['time'].drop_nulls().len() > 0:
                    effects_dict[effect_label] = _rows_to_numeric(effect_sub, 'value', timesteps, periods, scenarios)
                else:
                    effects_dict[effect_label] = effect_sub['value'].to_list()[0]
            flow_effects[flow_label] = effects_dict

    # Build Flow objects — handle period/scenario varying rows
    for flow_label in flows_df['flow'].unique().to_list():
        sub = flows_df.filter(pl.col('flow') == flow_label)
        row = sub.row(0, named=True)

        component_label = row['component']
        direction = row['direction']
        flow_component_map[flow_label] = component_label
        flow_direction_map[flow_label] = direction

        # Extract the short flow label from "component(short_label)"
        short_label = _extract_short_label(flow_label, component_label)

        flow_kwargs: dict[str, Any] = {
            'label': short_label,
            'bus': row['bus'],
        }

        # Size — may be overridden by investments table
        if flow_label in invest_map:
            inv_kw = dict(invest_map[flow_label])
            if flow_label in invest_effects_map:
                inv_kw.update(invest_effects_map[flow_label])
            if flow_label in piecewise_invest_map:
                inv_kw['piecewise_effects_of_investment'] = piecewise_invest_map[flow_label]
            flow_kwargs['size'] = InvestParameters(**inv_kw)
        elif row.get('size') is not None:
            if _has_varying_rows(sub, 'size'):
                flow_kwargs['size'] = _ps_column_to_numeric(sub, 'size', periods, scenarios)
            else:
                flow_kwargs['size'] = row['size']

        # Scalar flow params
        if row.get('rel_min') is not None and row['rel_min'] != 0.0:
            flow_kwargs['relative_minimum'] = row['rel_min']
        if row.get('rel_max') is not None and row['rel_max'] != 1.0:
            flow_kwargs['relative_maximum'] = row['rel_max']
        if row.get('previous_flow_rate') is not None:
            pfr_val = row['previous_flow_rate']
            if isinstance(pfr_val, str) and ',' in pfr_val:
                flow_kwargs['previous_flow_rate'] = [float(v) for v in pfr_val.split(',')]
            else:
                flow_kwargs['previous_flow_rate'] = pfr_val

        # PS-dimensioned params
        for col, kwarg in [
            ('flow_hours_min', 'flow_hours_min'),
            ('flow_hours_max', 'flow_hours_max'),
            ('load_factor_min', 'load_factor_min'),
            ('load_factor_max', 'load_factor_max'),
        ]:
            if col in sub.columns and sub[col].drop_nulls().len() > 0:
                if _has_varying_rows(sub, col):
                    flow_kwargs[kwarg] = _ps_column_to_numeric(sub, col, periods, scenarios)
                else:
                    flow_kwargs[kwarg] = row[col]

        # S-dimensioned params
        for col, kwarg in [
            ('flow_hours_min_over_periods', 'flow_hours_min_over_periods'),
            ('flow_hours_max_over_periods', 'flow_hours_max_over_periods'),
        ]:
            if col in sub.columns and sub[col].drop_nulls().len() > 0:
                if 'scenario' in sub.columns and sub['scenario'].drop_nulls().len() > 0:
                    flow_kwargs[kwarg] = _s_column_to_numeric(sub, col, scenarios)
                else:
                    flow_kwargs[kwarg] = row[col]

        # Override with flow_profiles
        if flow_label in flow_profiles:
            flow_kwargs.update(flow_profiles[flow_label])

        # Set effects_per_flow_hour
        if flow_label in flow_effects:
            flow_kwargs['effects_per_flow_hour'] = flow_effects[flow_label]

        # Status parameters (Phase 2)
        # Check both flow label and "flow:flow_label" in status_map
        status_key_flow = f'flow:{flow_label}'
        if status_key_flow in status_map:
            flow_kwargs['status_parameters'] = status_map[status_key_flow]

        flow_objects[flow_label] = Flow(**flow_kwargs)

    # ------------------------------------------------------------------
    # 8. Parse converters
    # ------------------------------------------------------------------
    converter_components: set[str] = set()
    converter_factors: dict[str, list[dict[str, Any]]] = {}  # comp -> list of {flow: coeff}

    if 'converters' in tables and len(tables['converters']) > 0:
        conv_df = tables['converters']
        for comp_label in conv_df['converter'].unique().to_list():
            converter_components.add(comp_label)
            comp_sub = conv_df.filter(pl.col('converter') == comp_label)
            equations: dict[int, dict[str, Any]] = {}
            for eq_idx in sorted(comp_sub['eq_idx'].unique().to_list()):
                eq_sub = comp_sub.filter(pl.col('eq_idx') == eq_idx)
                eq_dict: dict[str, Any] = {}
                for flow_label in eq_sub['flow'].unique().to_list():
                    flow_sub = eq_sub.filter(pl.col('flow') == flow_label)
                    short = _extract_short_label(flow_label, comp_label)
                    if 'time' in flow_sub.columns and flow_sub['time'].drop_nulls().len() > 0:
                        eq_dict[short] = _rows_to_numeric(flow_sub, 'value', timesteps, periods, scenarios)
                    else:
                        eq_dict[short] = flow_sub['value'].to_list()[0]
                equations[eq_idx] = eq_dict
            converter_factors[comp_label] = [equations[i] for i in sorted(equations)]

    # ------------------------------------------------------------------
    # 9. Parse piecewise_conversions (Phase 3)
    # ------------------------------------------------------------------
    piecewise_conv_components: set[str] = set()
    piecewise_conv_map: dict[str, PiecewiseConversion] = {}

    if 'piecewise_conversions' in tables and len(tables['piecewise_conversions']) > 0:
        pw_df = tables['piecewise_conversions']
        for comp_label in pw_df['converter'].unique().to_list():
            piecewise_conv_components.add(comp_label)
            comp_sub = pw_df.filter(pl.col('converter') == comp_label)
            piecewises: dict[str, Piecewise] = {}
            for flow_label in comp_sub['flow'].unique().to_list():
                flow_sub = comp_sub.filter(pl.col('flow') == flow_label)
                short = _extract_short_label(flow_label, comp_label)
                pieces = []
                for piece_idx in sorted(flow_sub['piece_idx'].unique().to_list()):
                    piece_sub = flow_sub.filter(pl.col('piece_idx') == piece_idx)
                    if 'time' in piece_sub.columns and piece_sub['time'].drop_nulls().len() > 0:
                        start = _rows_to_numeric(piece_sub, 'start', timesteps, periods, scenarios)
                        end = _rows_to_numeric(piece_sub, 'end', timesteps, periods, scenarios)
                    else:
                        start = piece_sub['start'].to_list()[0]
                        end = piece_sub['end'].to_list()[0]
                    pieces.append(Piece(start=start, end=end))
                piecewises[short] = Piecewise(pieces)
            piecewise_conv_map[comp_label] = PiecewiseConversion(piecewises)

    # ------------------------------------------------------------------
    # 10. Parse storages
    # ------------------------------------------------------------------
    storage_components: set[str] = set()
    storage_params: dict[str, dict[str, Any]] = {}

    if 'storages' in tables and len(tables['storages']) > 0:
        stor_df = tables['storages']
        for storage_label in stor_df['storage'].unique().to_list():
            storage_components.add(storage_label)
            sub = stor_df.filter(pl.col('storage') == storage_label)
            row = sub.row(0, named=True)
            params: dict[str, Any] = {
                'charge_flow': row['charge_flow'],
                'discharge_flow': row['discharge_flow'],
            }

            # capacity — may be overridden by investments table
            if storage_label in invest_map:
                inv_kw = dict(invest_map[storage_label])
                if storage_label in invest_effects_map:
                    inv_kw.update(invest_effects_map[storage_label])
                if storage_label in piecewise_invest_map:
                    inv_kw['piecewise_effects_of_investment'] = piecewise_invest_map[storage_label]
                params['capacity'] = InvestParameters(**inv_kw)
            elif 'capacity' in sub.columns and sub['capacity'].drop_nulls().len() > 0:
                if _has_varying_rows(sub, 'capacity'):
                    params['capacity'] = _ps_column_to_numeric(sub, 'capacity', periods, scenarios)
                else:
                    params['capacity'] = row['capacity']

            # Scalar params with defaults
            for col, default in [
                ('eta_charge', 1.0),
                ('eta_discharge', 1.0),
                ('relative_loss_per_hour', 0.0),
                ('rel_min_charge_state', 0.0),
                ('rel_max_charge_state', 1.0),
            ]:
                if col in sub.columns and row.get(col) is not None and row[col] != default:
                    params[col] = row[col]

            # initial_charge_state
            if 'initial_charge_state' in sub.columns and row.get('initial_charge_state') is not None:
                val = row['initial_charge_state']
                if isinstance(val, str) and val == 'equals_final':
                    params['initial_charge_state'] = 'equals_final'
                else:
                    params['initial_charge_state'] = float(val)

            # PS-dimensioned params
            for col in [
                'minimal_final_charge_state',
                'maximal_final_charge_state',
                'rel_min_final_charge_state',
                'rel_max_final_charge_state',
            ]:
                if col in sub.columns and sub[col].drop_nulls().len() > 0:
                    if _has_varying_rows(sub, col):
                        params[col] = _ps_column_to_numeric(sub, col, periods, scenarios)
                    else:
                        params[col] = row[col]

            # Boolean params
            for col, default in [
                ('prevent_simultaneous', True),
                ('balanced', False),
            ]:
                if col in sub.columns and row.get(col) is not None and row[col] != default:
                    params[col] = row[col]

            # cluster_mode
            if 'cluster_mode' in sub.columns and row.get('cluster_mode') is not None:
                params['cluster_mode'] = row['cluster_mode']

            storage_params[storage_label] = params

    # Handle storage_timeseries
    if 'storage_timeseries' in tables and len(tables['storage_timeseries']) > 0:
        sts_df = tables['storage_timeseries']
        for storage_label in sts_df['storage'].unique().to_list():
            if storage_label not in storage_params:
                continue
            sub = sts_df.filter(pl.col('storage') == storage_label)
            for col in [
                'eta_charge',
                'eta_discharge',
                'relative_loss_per_hour',
                'rel_min_charge_state',
                'rel_max_charge_state',
            ]:
                if col in sub.columns and sub[col].drop_nulls().len() > 0:
                    storage_params[storage_label][col] = _rows_to_numeric(sub, col, timesteps, periods, scenarios)

    # ------------------------------------------------------------------
    # 11. Parse transmissions (Phase 3)
    # ------------------------------------------------------------------
    transmission_components: set[str] = set()
    transmission_params: dict[str, dict[str, Any]] = {}

    if 'transmissions' in tables and len(tables['transmissions']) > 0:
        trans_df = tables['transmissions']
        for trans_label in trans_df['transmission'].unique().to_list():
            transmission_components.add(trans_label)
            sub = trans_df.filter(pl.col('transmission') == trans_label)
            row = sub.row(0, named=True)
            params = {
                'in1_flow': row['in1_flow'],
                'out1_flow': row['out1_flow'],
            }
            if row.get('in2_flow') is not None:
                params['in2_flow'] = row['in2_flow']
            if row.get('out2_flow') is not None:
                params['out2_flow'] = row['out2_flow']

            # Losses — can be time-varying
            for col in ['relative_losses', 'absolute_losses']:
                if col in sub.columns and sub[col].drop_nulls().len() > 0:
                    if _has_varying_rows(sub, col):
                        params[col] = _ps_column_to_numeric(sub, col, periods, scenarios)
                    else:
                        params[col] = row[col]

            for col, default in [
                ('prevent_simultaneous', True),
                ('balanced', False),
            ]:
                if col in sub.columns and row.get(col) is not None and row[col] != default:
                    params[col] = row[col]

            transmission_params[trans_label] = params

    # Handle transmission_timeseries
    if 'transmission_timeseries' in tables and len(tables['transmission_timeseries']) > 0:
        tts_df = tables['transmission_timeseries']
        for trans_label in tts_df['transmission'].unique().to_list():
            if trans_label not in transmission_params:
                continue
            sub = tts_df.filter(pl.col('transmission') == trans_label)
            for col in ['relative_losses', 'absolute_losses']:
                if col in sub.columns and sub[col].drop_nulls().len() > 0:
                    transmission_params[trans_label][col] = _rows_to_numeric(sub, col, timesteps, periods, scenarios)

    # ------------------------------------------------------------------
    # 11b. Parse component-level attributes (prevent_simultaneous_flow_rates)
    # ------------------------------------------------------------------
    prevent_simultaneous_labels: set[str] = set()
    if 'components' in tables and len(tables['components']) > 0:
        comp_attr_df = tables['components']
        for row_dict in comp_attr_df.iter_rows(named=True):
            if row_dict.get('prevent_simultaneous_flow_rates'):
                prevent_simultaneous_labels.add(row_dict['component'])

    # ------------------------------------------------------------------
    # 12. Group flows by component and build component objects
    # ------------------------------------------------------------------
    # Collect which components are which type
    all_converter_labels = converter_components | piecewise_conv_components
    all_storage_labels = storage_components
    all_transmission_labels = transmission_components

    # Group flows by component
    comp_flows: dict[str, dict[str, list[Flow]]] = {}  # comp -> {"in": [...], "out": [...]}
    for flow_label, comp_label in flow_component_map.items():
        direction = flow_direction_map[flow_label]
        if comp_label not in comp_flows:
            comp_flows[comp_label] = {'in': [], 'out': []}
        comp_flows[comp_label][direction].append(flow_objects[flow_label])

    component_objects = []
    for comp_label, flow_groups in comp_flows.items():
        inputs = flow_groups.get('in', [])
        outputs = flow_groups.get('out', [])

        if comp_label in all_converter_labels:
            conv_kwargs: dict[str, Any] = {
                'label': comp_label,
                'inputs': inputs,
                'outputs': outputs,
            }
            if comp_label in converter_factors:
                conv_kwargs['conversion_factors'] = converter_factors[comp_label]
            if comp_label in piecewise_conv_map:
                conv_kwargs['piecewise_conversion'] = piecewise_conv_map[comp_label]
            # Component-level status
            status_key_comp = f'component:{comp_label}'
            if status_key_comp in status_map:
                conv_kwargs['status_parameters'] = status_map[status_key_comp]
            component_objects.append(LinearConverter(**conv_kwargs))

        elif comp_label in all_storage_labels:
            sp = storage_params[comp_label]
            # Look up flow objects by their label
            charge_flow_label = sp['charge_flow']
            discharge_flow_label = sp['discharge_flow']
            charge_flow = flow_objects[charge_flow_label]
            discharge_flow = flow_objects[discharge_flow_label]

            stor_kwargs: dict[str, Any] = {
                'label': comp_label,
                'charging': charge_flow,
                'discharging': discharge_flow,
            }
            if 'capacity' in sp:
                stor_kwargs['capacity_in_flow_hours'] = sp['capacity']
            for col, kwarg in [
                ('eta_charge', 'eta_charge'),
                ('eta_discharge', 'eta_discharge'),
                ('relative_loss_per_hour', 'relative_loss_per_hour'),
                ('rel_min_charge_state', 'relative_minimum_charge_state'),
                ('rel_max_charge_state', 'relative_maximum_charge_state'),
                ('initial_charge_state', 'initial_charge_state'),
                ('minimal_final_charge_state', 'minimal_final_charge_state'),
                ('maximal_final_charge_state', 'maximal_final_charge_state'),
                ('rel_min_final_charge_state', 'relative_minimum_final_charge_state'),
                ('rel_max_final_charge_state', 'relative_maximum_final_charge_state'),
                ('prevent_simultaneous', 'prevent_simultaneous_charge_and_discharge'),
                ('balanced', 'balanced'),
                ('cluster_mode', 'cluster_mode'),
            ]:
                if col in sp:
                    stor_kwargs[kwarg] = sp[col]

            component_objects.append(Storage(**stor_kwargs))

        elif comp_label in all_transmission_labels:
            tp = transmission_params[comp_label]
            trans_kwargs: dict[str, Any] = {
                'label': comp_label,
                'in1': flow_objects[tp['in1_flow']],
                'out1': flow_objects[tp['out1_flow']],
            }
            if 'in2_flow' in tp:
                trans_kwargs['in2'] = flow_objects[tp['in2_flow']]
            if 'out2_flow' in tp:
                trans_kwargs['out2'] = flow_objects[tp['out2_flow']]
            if 'relative_losses' in tp:
                trans_kwargs['relative_losses'] = tp['relative_losses']
            if 'absolute_losses' in tp:
                trans_kwargs['absolute_losses'] = tp['absolute_losses']
            for col, kwarg, default in [
                ('prevent_simultaneous', 'prevent_simultaneous_flows_in_both_directions', True),
                ('balanced', 'balanced', False),
            ]:
                if col in tp and tp[col] != default:
                    trans_kwargs[kwarg] = tp[col]
            # Component-level status
            status_key_comp = f'component:{comp_label}'
            if status_key_comp in status_map:
                trans_kwargs['status_parameters'] = status_map[status_key_comp]

            component_objects.append(Transmission(**trans_kwargs))

        else:
            # Infer Source / Sink / SourceAndSink
            comp_kwargs: dict[str, Any] = {'label': comp_label}
            if comp_label in prevent_simultaneous_labels:
                comp_kwargs['prevent_simultaneous_flow_rates'] = True
            if inputs and outputs:
                comp_kwargs['inputs'] = inputs
                comp_kwargs['outputs'] = outputs
                component_objects.append(SourceAndSink(**comp_kwargs))
            elif outputs and not inputs:
                comp_kwargs['outputs'] = outputs
                component_objects.append(Source(**comp_kwargs))
            elif inputs and not outputs:
                comp_kwargs['inputs'] = inputs
                component_objects.append(Sink(**comp_kwargs))
            else:
                raise ValueError(f'Component "{comp_label}" has no flows.')

    # ------------------------------------------------------------------
    # 13. Build FlowSystem
    # ------------------------------------------------------------------
    fs = FlowSystem(
        timesteps=timesteps,
        periods=periods,
        scenarios=scenarios,
        scenario_weights=scenario_weights,
        weight_of_last_period=weight_of_last_period,
        hours_of_last_timestep=hours_of_last_timestep,
        hours_of_previous_timesteps=hours_of_previous_timesteps,
        scenario_independent_sizes=scenario_independent_sizes,
        scenario_independent_flow_rates=scenario_independent_flow_rates,
        timestep_duration=timestep_duration,
        name=name,
    )

    all_elements = list(effect_objects.values()) + list(bus_objects.values()) + component_objects
    fs.add_elements(*all_elements)
    return fs


def to_tables(flow_system: Any) -> dict[str, Any]:
    """Extract all model data from a FlowSystem as a dict of polars DataFrames.

    Args:
        flow_system: A :class:`FlowSystem` instance.

    Returns:
        Dict mapping table name to polars DataFrame.
    """
    pl = _import_polars()
    from .components import LinearConverter, Sink, Source, SourceAndSink, Storage, Transmission
    from .interface import InvestParameters

    tables: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # timesteps
    # ------------------------------------------------------------------
    mc = flow_system.model_coords
    ts_index = mc.timesteps
    dt_array = mc.timestep_duration

    if isinstance(ts_index, pd.DatetimeIndex):
        time_col = ts_index.tolist()
    else:
        time_col = ts_index.tolist()

    # Extract dt values
    if dt_array is not None:
        if isinstance(dt_array, xr.DataArray):
            dt_vals = dt_array.values.tolist()
        else:
            dt_vals = list(dt_array)
    else:
        dt_vals = [1.0] * len(ts_index)

    tables['timesteps'] = pl.DataFrame({'time': time_col, 'dt': dt_vals})

    # ------------------------------------------------------------------
    # periods
    # ------------------------------------------------------------------
    if mc.periods is not None:
        period_data: dict[str, list] = {'period': mc.periods.tolist()}
        tables['periods'] = pl.DataFrame(period_data)

    # ------------------------------------------------------------------
    # scenarios
    # ------------------------------------------------------------------
    if mc.scenarios is not None:
        scenario_data: dict[str, list] = {'scenario': mc.scenarios.tolist()}
        if hasattr(mc, 'scenario_weights') and mc.scenario_weights is not None:
            if isinstance(mc.scenario_weights, xr.DataArray):
                scenario_data['weight'] = mc.scenario_weights.values.tolist()
            elif isinstance(mc.scenario_weights, np.ndarray):
                scenario_data['weight'] = mc.scenario_weights.tolist()
        tables['scenarios'] = pl.DataFrame(scenario_data)

    # ------------------------------------------------------------------
    # buses
    # ------------------------------------------------------------------
    bus_rows = []
    bus_ts_rows = []
    for bus in flow_system.buses.values():
        row: dict[str, Any] = {'bus': bus.label}
        if hasattr(bus, 'carrier') and bus.carrier is not None:
            row['carrier'] = bus.carrier if isinstance(bus.carrier, str) else bus.carrier.label
        else:
            row['carrier'] = None
        penalty = getattr(bus, 'imbalance_penalty_per_flow_hour', None)
        if penalty is not None and not isinstance(penalty, (int, float, np.integer, np.floating)):
            # Time-varying — put in bus_timeseries
            row['imbalance_penalty'] = None
            bus_ts_rows.extend(
                _numeric_to_rows(
                    bus.label, 'bus', penalty, ts_index, mc.periods, mc.scenarios, value_col='imbalance_penalty'
                )
            )
        else:
            row['imbalance_penalty'] = float(penalty) if penalty is not None else None
        bus_rows.append(row)

    tables['buses'] = pl.DataFrame(bus_rows)
    if bus_ts_rows:
        tables['bus_timeseries'] = pl.DataFrame(bus_ts_rows)

    # ------------------------------------------------------------------
    # effects
    # ------------------------------------------------------------------
    effect_rows = []
    effect_bound_rows = []
    effect_share_rows = []
    for effect in flow_system.effects.values():
        effect_rows.append(
            {
                'effect': effect.label,
                'unit': effect.unit,
                'is_standard': effect.is_standard,
                'is_objective': effect.is_objective,
            }
        )
        # Effect bounds
        bound_result = _extract_effect_bounds(effect, mc.periods, mc.scenarios, ts_index)
        if bound_result:
            if isinstance(bound_result, list):
                effect_bound_rows.extend(bound_result)
            else:
                effect_bound_rows.append(bound_result)
        # Effect shares
        share_rows = _extract_effect_shares(effect, ts_index, mc.periods, mc.scenarios)
        effect_share_rows.extend(share_rows)

    tables['effects'] = pl.DataFrame(effect_rows)
    if effect_bound_rows:
        tables['effect_bounds'] = pl.DataFrame(effect_bound_rows)
    if effect_share_rows:
        tables['effect_shares'] = pl.DataFrame(effect_share_rows)

    # ------------------------------------------------------------------
    # flows, flow_profiles, flow_effects, converters, storages, etc.
    # ------------------------------------------------------------------
    flow_rows = []
    flow_profile_rows = []
    flow_effect_rows = []
    component_rows = []
    converter_rows = []
    piecewise_conv_rows = []
    storage_rows = []
    storage_ts_rows = []
    transmission_rows = []
    investment_rows = []
    investment_effect_rows = []
    piecewise_invest_rows = []
    status_rows = []
    status_effect_rows = []

    standard_effect_label = next((e.label for e in flow_system.effects.values() if e.is_standard), None)

    for component in flow_system.components.values():
        # Determine direction for each flow
        for flow in component.inputs.values():
            _add_flow_rows(
                flow,
                component.label,
                'in',
                flow_rows,
                flow_profile_rows,
                flow_effect_rows,
                investment_rows,
                investment_effect_rows,
                piecewise_invest_rows,
                status_rows,
                status_effect_rows,
                ts_index,
                mc.periods,
                mc.scenarios,
                standard_effect_label,
            )
        for flow in component.outputs.values():
            _add_flow_rows(
                flow,
                component.label,
                'out',
                flow_rows,
                flow_profile_rows,
                flow_effect_rows,
                investment_rows,
                investment_effect_rows,
                piecewise_invest_rows,
                status_rows,
                status_effect_rows,
                ts_index,
                mc.periods,
                mc.scenarios,
                standard_effect_label,
            )

        # Converter-specific
        if isinstance(component, LinearConverter):
            if component.conversion_factors:
                for eq_idx, eq in enumerate(component.conversion_factors):
                    for flow_label, coeff in eq.items():
                        qualified = f'{component.label}({flow_label})'
                        if _is_scalar(coeff):
                            converter_rows.append(
                                {
                                    'converter': component.label,
                                    'eq_idx': eq_idx,
                                    'flow': qualified,
                                    'value': float(coeff),
                                }
                            )
                        else:
                            converter_rows.extend(
                                _numeric_to_value_rows(
                                    coeff,
                                    ts_index,
                                    mc.periods,
                                    mc.scenarios,
                                    extra={'converter': component.label, 'eq_idx': eq_idx, 'flow': qualified},
                                )
                            )

            if component.piecewise_conversion is not None:
                for flow_label, piecewise in component.piecewise_conversion.piecewises.items():
                    qualified = f'{component.label}({flow_label})'
                    for piece_idx, piece in enumerate(piecewise):
                        if _is_scalar(piece.start) and _is_scalar(piece.end):
                            piecewise_conv_rows.append(
                                {
                                    'converter': component.label,
                                    'piece_idx': piece_idx,
                                    'flow': qualified,
                                    'start': float(piece.start),
                                    'end': float(piece.end),
                                }
                            )
                        else:
                            piecewise_conv_rows.extend(
                                _piece_to_rows(
                                    piece,
                                    ts_index,
                                    mc.periods,
                                    mc.scenarios,
                                    extra={'converter': component.label, 'piece_idx': piece_idx, 'flow': qualified},
                                )
                            )

            # Component-level status
            if component.status_parameters is not None:
                _add_status_rows(
                    component.label,
                    'component',
                    component.status_parameters,
                    status_rows,
                    status_effect_rows,
                    ts_index,
                    mc.periods,
                    mc.scenarios,
                    standard_effect_label,
                )

        elif isinstance(component, Storage):
            stor_row: dict[str, Any] = {
                'storage': component.label,
                'charge_flow': component.charging.label_full,
                'discharge_flow': component.discharging.label_full,
            }

            # capacity
            if isinstance(component.capacity_in_flow_hours, InvestParameters):
                _add_invest_rows(
                    component.label,
                    'storage',
                    component.capacity_in_flow_hours,
                    investment_rows,
                    investment_effect_rows,
                    piecewise_invest_rows,
                    mc.periods,
                    mc.scenarios,
                    standard_effect_label,
                )
                stor_row['capacity'] = None
            else:
                cap = component.capacity_in_flow_hours
                stor_row['capacity'] = _scalar_or_none(cap)

            # Scalar params
            for attr, col, default in [
                ('eta_charge', 'eta_charge', 1.0),
                ('eta_discharge', 'eta_discharge', 1.0),
                ('relative_loss_per_hour', 'relative_loss_per_hour', 0.0),
                ('relative_minimum_charge_state', 'rel_min_charge_state', 0.0),
                ('relative_maximum_charge_state', 'rel_max_charge_state', 1.0),
            ]:
                val = getattr(component, attr)
                if _is_scalar(val):
                    stor_row[col] = float(val)
                else:
                    stor_row[col] = default
                    # Time-varying → storage_timeseries
                    storage_ts_rows.extend(
                        _numeric_to_rows(
                            component.label, 'storage', val, ts_index, mc.periods, mc.scenarios, value_col=col
                        )
                    )

            # initial_charge_state
            ics = component.initial_charge_state
            if isinstance(ics, str):
                stor_row['initial_charge_state'] = ics
            elif ics is not None:
                stor_row['initial_charge_state'] = _scalar_or_none(ics)
            else:
                stor_row['initial_charge_state'] = None

            # PS params
            for attr, col in [
                ('minimal_final_charge_state', 'minimal_final_charge_state'),
                ('maximal_final_charge_state', 'maximal_final_charge_state'),
                ('relative_minimum_final_charge_state', 'rel_min_final_charge_state'),
                ('relative_maximum_final_charge_state', 'rel_max_final_charge_state'),
            ]:
                val = getattr(component, attr, None)
                stor_row[col] = _scalar_or_none(val) if val is not None else None

            stor_row['prevent_simultaneous'] = component.prevent_simultaneous_charge_and_discharge
            stor_row['balanced'] = component.balanced
            stor_row['cluster_mode'] = component.cluster_mode

            storage_rows.append(stor_row)

        elif isinstance(component, Transmission):
            trans_row: dict[str, Any] = {
                'transmission': component.label,
                'in1_flow': component.in1.label_full,
                'out1_flow': component.out1.label_full,
                'in2_flow': component.in2.label_full if component.in2 else None,
                'out2_flow': component.out2.label_full if component.out2 else None,
            }
            for attr, col in [
                ('relative_losses', 'relative_losses'),
                ('absolute_losses', 'absolute_losses'),
            ]:
                val = getattr(component, attr, None)
                trans_row[col] = _scalar_or_none(val) if val is not None else None

            trans_row['prevent_simultaneous'] = bool(component.prevent_simultaneous_flows)
            trans_row['balanced'] = component.balanced
            transmission_rows.append(trans_row)

            # Component-level status
            if component.status_parameters is not None:
                _add_status_rows(
                    component.label,
                    'component',
                    component.status_parameters,
                    status_rows,
                    status_effect_rows,
                    ts_index,
                    mc.periods,
                    mc.scenarios,
                    standard_effect_label,
                )

        elif isinstance(component, (Source, Sink, SourceAndSink)):
            if getattr(component, 'prevent_simultaneous_flow_rates', False):
                component_rows.append(
                    {
                        'component': component.label,
                        'prevent_simultaneous_flow_rates': True,
                    }
                )

    tables['flows'] = (
        pl.DataFrame(flow_rows)
        if flow_rows
        else pl.DataFrame(schema={'flow': pl.Utf8, 'bus': pl.Utf8, 'component': pl.Utf8, 'direction': pl.Utf8})
    )
    if flow_profile_rows:
        tables['flow_profiles'] = pl.DataFrame(flow_profile_rows)
    if flow_effect_rows:
        tables['flow_effects'] = pl.DataFrame(flow_effect_rows)
    if component_rows:
        tables['components'] = pl.DataFrame(component_rows)
    if converter_rows:
        tables['converters'] = pl.DataFrame(converter_rows)
    if piecewise_conv_rows:
        tables['piecewise_conversions'] = pl.DataFrame(piecewise_conv_rows)
    if storage_rows:
        tables['storages'] = pl.DataFrame(storage_rows)
    if storage_ts_rows:
        tables['storage_timeseries'] = pl.DataFrame(storage_ts_rows)
    if transmission_rows:
        tables['transmissions'] = pl.DataFrame(transmission_rows)
    if investment_rows:
        tables['investments'] = pl.DataFrame(investment_rows)
    if investment_effect_rows:
        tables['investment_effects'] = pl.DataFrame(investment_effect_rows)
    if piecewise_invest_rows:
        tables['piecewise_investment_effects'] = pl.DataFrame(piecewise_invest_rows)
    if status_rows:
        tables['status'] = pl.DataFrame(status_rows)
    if status_effect_rows:
        tables['status_effects'] = pl.DataFrame(status_effect_rows)

    return tables


def from_dir(path: str | Path, **kwargs: Any) -> Any:
    """Load CSVs from a directory and build a FlowSystem.

    Each ``*.csv`` file in the directory becomes a table (filename without extension as key).

    Args:
        path: Directory containing CSV files.
        **kwargs: Additional keyword arguments passed to :func:`from_tables`.

    Returns:
        A :class:`FlowSystem` instance.
    """
    pl = _import_polars()
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f'Directory not found: {path}')

    tables: dict[str, Any] = {}
    for csv_file in sorted(path.glob('*.csv')):
        table_name = csv_file.stem
        tables[table_name] = pl.read_csv(csv_file, try_parse_dates=True)

    return from_tables(tables, **kwargs)


def to_dir(flow_system: Any, path: str | Path) -> None:
    """Write FlowSystem model data as CSVs to a directory.

    Args:
        flow_system: A :class:`FlowSystem` instance.
        path: Target directory (created if it doesn't exist).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    tables = to_tables(flow_system)
    for name, df in tables.items():
        df.write_csv(path / f'{name}.csv')


# ---------------------------------------------------------------------------
# Internal helpers — from_tables direction
# ---------------------------------------------------------------------------


def _require_table(tables: dict[str, Any], name: str) -> Any:
    """Get a required table or raise with a helpful message."""
    if name not in tables:
        raise KeyError(f'Required table "{name}" not found. Available tables: {sorted(tables.keys())}')
    return tables[name]


def _parse_timesteps(ts_df: Any) -> tuple[pd.DatetimeIndex | pd.RangeIndex, xr.DataArray | None]:
    """Parse the timesteps table into a pandas index and optional duration array."""
    time_col = ts_df['time']

    # Try to interpret as datetime
    import polars as pl

    if time_col.dtype == pl.Datetime or time_col.dtype == pl.Date:
        time_values = time_col.to_list()
        timesteps = pd.DatetimeIndex(time_values, name='time')
    elif time_col.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        time_values = time_col.to_list()
        timesteps = pd.RangeIndex(start=time_values[0], stop=time_values[-1] + 1, name='time')
        if len(timesteps) != len(time_values):
            # Non-contiguous integers — use as-is
            timesteps = pd.Index(time_values, name='time')
    elif time_col.dtype == pl.Utf8:
        # Try parsing as datetime strings
        try:
            time_values = pd.to_datetime(time_col.to_list())
            timesteps = pd.DatetimeIndex(time_values, name='time')
        except (ValueError, TypeError):
            timesteps = pd.Index(time_col.to_list(), name='time')
    else:
        timesteps = pd.Index(time_col.to_list(), name='time')

    # Parse dt column
    timestep_duration = None
    if 'dt' in ts_df.columns:
        dt_values = np.array(ts_df['dt'].to_list(), dtype=float)
        timestep_duration = xr.DataArray(dt_values, dims=['time'], coords={'time': timesteps})

    return timesteps, timestep_duration


def _rows_to_numeric(
    df: Any,
    value_col: str,
    timesteps: pd.Index,
    periods: pd.Index | None,
    scenarios: pd.Index | None,
) -> Any:
    """Convert long-format rows to a numeric value suitable for FlowSystem parameters.

    Returns a scalar, numpy array, or xarray DataArray depending on dimensionality.
    """
    has_time = 'time' in df.columns and df['time'].drop_nulls().len() > 0
    has_period = 'period' in df.columns and df['period'].drop_nulls().len() > 0
    has_scenario = 'scenario' in df.columns and df['scenario'].drop_nulls().len() > 0

    values = df[value_col].to_list()

    if not has_time and not has_period and not has_scenario:
        # Scalar
        return values[0] if len(values) == 1 else np.array(values, dtype=float)

    if has_time and not has_period and not has_scenario:
        # 1D time series
        return np.array(values, dtype=float)

    # Multi-dimensional — build xarray DataArray
    coords: dict[str, Any] = {}
    dims: list[str] = []

    if has_time:
        dims.append('time')
        coords['time'] = timesteps
    if has_period:
        dims.append('period')
        coords['period'] = periods
    if has_scenario:
        dims.append('scenario')
        coords['scenario'] = scenarios

    # Build from the rows
    if len(dims) == 1:
        return np.array(values, dtype=float)

    # Multi-dim: pivot the data
    # Create index columns for xarray alignment
    shape = tuple(len(coords[d]) for d in dims)
    result = np.full(shape, np.nan)

    # Map index values to positions
    dim_maps = {}
    for d in dims:
        if d == 'time':
            idx_values = df['time'].to_list()
            if isinstance(timesteps, pd.DatetimeIndex):
                idx_values = pd.DatetimeIndex(idx_values)
            coord_list = coords[d].tolist()
            dim_maps[d] = {v: i for i, v in enumerate(coord_list)}
        else:
            coord_list = coords[d].tolist()
            dim_maps[d] = {v: i for i, v in enumerate(coord_list)}

    # Fill array
    row_dicts = df.select([*[d for d in dims if d in df.columns], value_col]).to_dicts()
    for row in row_dicts:
        indices = tuple(dim_maps[d][row[d]] for d in dims)
        result[indices] = row[value_col]

    return xr.DataArray(result, dims=dims, coords=coords)


def _ps_column_to_numeric(
    df: Any,
    col: str,
    periods: pd.Index | None,
    scenarios: pd.Index | None,
) -> Any:
    """Convert a column that may vary by period/scenario to a numeric value."""
    has_period = 'period' in df.columns and df['period'].drop_nulls().len() > 0
    has_scenario = 'scenario' in df.columns and df['scenario'].drop_nulls().len() > 0

    values = df[col].drop_nulls().to_list()
    if not values:
        return None

    if not has_period and not has_scenario:
        return values[0] if len(values) == 1 else np.array(values, dtype=float)

    coords: dict[str, Any] = {}
    dims: list[str] = []
    if has_period and periods is not None:
        dims.append('period')
        coords['period'] = periods
    if has_scenario and scenarios is not None:
        dims.append('scenario')
        coords['scenario'] = scenarios

    if not dims:
        return values[0] if len(values) == 1 else np.array(values, dtype=float)

    shape = tuple(len(coords[d]) for d in dims)
    result = np.full(shape, np.nan)

    dim_maps = {}
    for d in dims:
        coord_list = coords[d].tolist()
        dim_maps[d] = {v: i for i, v in enumerate(coord_list)}

    available_cols = [d for d in dims if d in df.columns]
    row_dicts = df.select([*available_cols, col]).to_dicts()
    for row in row_dicts:
        if row[col] is None:
            continue
        indices = tuple(dim_maps[d][row[d]] for d in dims if d in row)
        result[indices] = row[col]

    return xr.DataArray(result, dims=dims, coords=coords)


def _s_column_to_numeric(
    df: Any,
    col: str,
    scenarios: pd.Index | None,
) -> Any:
    """Convert a column that may vary by scenario to a numeric value."""
    if scenarios is None or 'scenario' not in df.columns:
        values = df[col].drop_nulls().to_list()
        return values[0] if len(values) == 1 else np.array(values, dtype=float)

    coords = {'scenario': scenarios}
    result = np.full(len(scenarios), np.nan)
    scenario_map = {v: i for i, v in enumerate(scenarios.tolist())}

    for row in df.select(['scenario', col]).to_dicts():
        if row[col] is not None and row['scenario'] is not None:
            result[scenario_map[row['scenario']]] = row[col]

    return xr.DataArray(result, dims=['scenario'], coords=coords)


def _has_varying_rows(df: Any, col: str) -> bool:
    """Check if a column has varying values across period/scenario rows."""
    if df.height <= 1:
        return False
    has_period = 'period' in df.columns and df['period'].drop_nulls().len() > 0
    has_scenario = 'scenario' in df.columns and df['scenario'].drop_nulls().len() > 0
    return has_period or has_scenario


def _extract_short_label(flow_label: str, component_label: str) -> str:
    """Extract short flow label from qualified 'component(flow)' format."""
    prefix = f'{component_label}('
    if flow_label.startswith(prefix) and flow_label.endswith(')'):
        return flow_label[len(prefix) : -1]
    return flow_label


def _parse_investments(
    inv_df: Any,
    pl: Any,
    periods: pd.Index | None,
    scenarios: pd.Index | None,
) -> dict[str, dict[str, Any]]:
    """Parse the investments table into a dict of kwargs per element."""
    result: dict[str, dict[str, Any]] = {}
    for element_label in inv_df['element'].unique().to_list():
        sub = inv_df.filter(pl.col('element') == element_label)
        row = sub.row(0, named=True)
        kwargs: dict[str, Any] = {}

        if 'fixed_size' in sub.columns and row.get('fixed_size') is not None:
            if _has_varying_rows(sub, 'fixed_size'):
                kwargs['fixed_size'] = _ps_column_to_numeric(sub, 'fixed_size', periods, scenarios)
            else:
                kwargs['fixed_size'] = row['fixed_size']
        else:
            if 'minimum_size' in sub.columns and row.get('minimum_size') is not None:
                if _has_varying_rows(sub, 'minimum_size'):
                    kwargs['minimum_size'] = _ps_column_to_numeric(sub, 'minimum_size', periods, scenarios)
                else:
                    kwargs['minimum_size'] = row['minimum_size']
            if 'maximum_size' in sub.columns and row.get('maximum_size') is not None:
                if _has_varying_rows(sub, 'maximum_size'):
                    kwargs['maximum_size'] = _ps_column_to_numeric(sub, 'maximum_size', periods, scenarios)
                else:
                    kwargs['maximum_size'] = row['maximum_size']

        if 'mandatory' in sub.columns and row.get('mandatory') is not None:
            kwargs['mandatory'] = bool(row['mandatory'])

        if 'linked_periods' in sub.columns and row.get('linked_periods') is not None:
            lp_str = row['linked_periods']
            if ':' in str(lp_str):
                parts = str(lp_str).split(':')
                kwargs['linked_periods'] = (int(parts[0]), int(parts[1]))

        result[element_label] = kwargs
    return result


def _parse_investment_effects(
    ie_df: Any,
    pl: Any,
    periods: pd.Index | None,
    scenarios: pd.Index | None,
) -> dict[str, dict[str, Any]]:
    """Parse investment_effects table into kwargs per element."""
    result: dict[str, dict[str, Any]] = {}
    for element_label in ie_df['element'].unique().to_list():
        sub = ie_df.filter(pl.col('element') == element_label)
        per_size: dict[str, Any] = {}
        on_invest: dict[str, Any] = {}
        on_retire: dict[str, Any] = {}

        for effect_label in sub['effect'].unique().to_list():
            effect_sub = sub.filter(pl.col('effect') == effect_label)
            row = effect_sub.row(0, named=True)
            if 'per_size' in effect_sub.columns and row.get('per_size') is not None:
                if _has_varying_rows(effect_sub, 'per_size'):
                    per_size[effect_label] = _ps_column_to_numeric(effect_sub, 'per_size', periods, scenarios)
                else:
                    per_size[effect_label] = row['per_size']
            if 'on_invest' in effect_sub.columns and row.get('on_invest') is not None:
                if _has_varying_rows(effect_sub, 'on_invest'):
                    on_invest[effect_label] = _ps_column_to_numeric(effect_sub, 'on_invest', periods, scenarios)
                else:
                    on_invest[effect_label] = row['on_invest']
            if 'on_retire' in effect_sub.columns and row.get('on_retire') is not None:
                if _has_varying_rows(effect_sub, 'on_retire'):
                    on_retire[effect_label] = _ps_column_to_numeric(effect_sub, 'on_retire', periods, scenarios)
                else:
                    on_retire[effect_label] = row['on_retire']

        kwargs: dict[str, Any] = {}
        if per_size:
            kwargs['effects_of_investment_per_size'] = per_size
        if on_invest:
            kwargs['effects_of_investment'] = on_invest
        if on_retire:
            kwargs['effects_of_retirement'] = on_retire
        result[element_label] = kwargs
    return result


def _parse_piecewise_investment_effects(
    pie_df: Any,
    pl: Any,
    timesteps: pd.Index,
    periods: pd.Index | None,
    scenarios: pd.Index | None,
) -> dict[str, Any]:
    """Parse piecewise_investment_effects table."""
    from .interface import Piece, Piecewise, PiecewiseEffects

    result: dict[str, Any] = {}
    for element_label in pie_df['element'].unique().to_list():
        sub = pie_df.filter(pl.col('element') == element_label)
        # Group by type (origin vs effect labels)
        types = sub['type'].unique().to_list()

        piecewises: dict[str, Piecewise] = {}
        for type_label in types:
            type_sub = sub.filter(pl.col('type') == type_label)
            pieces = []
            for piece_idx in sorted(type_sub['piece_idx'].unique().to_list()):
                piece_sub = type_sub.filter(pl.col('piece_idx') == piece_idx)
                row = piece_sub.row(0, named=True)
                pieces.append(Piece(start=row['start'], end=row['end']))
            piecewises[type_label] = Piecewise(pieces)

        origin = piecewises.pop('__origin__')
        result[element_label] = PiecewiseEffects(piecewise_origin=origin, piecewise_shares=piecewises)
    return result


def _parse_status(
    status_df: Any,
    status_effects_df: Any | None,
    pl: Any,
    timesteps: pd.Index,
    periods: pd.Index | None,
    scenarios: pd.Index | None,
) -> dict[str, Any]:
    """Parse status and status_effects tables into StatusParameters keyed by 'type:label'."""
    from .interface import StatusParameters

    # Parse status_effects first
    se_map: dict[str, dict[str, dict[str, Any]]] = {}  # element_label -> {effect_type -> {effect: value}}
    if status_effects_df is not None and len(status_effects_df) > 0:
        for element_label in status_effects_df['element'].unique().to_list():
            sub = status_effects_df.filter(pl.col('element') == element_label)
            per_startup: dict[str, Any] = {}
            per_active_hour: dict[str, Any] = {}
            for effect_label in sub['effect'].unique().to_list():
                effect_sub = sub.filter(pl.col('effect') == effect_label)
                if 'per_startup' in effect_sub.columns and effect_sub['per_startup'].drop_nulls().len() > 0:
                    if 'time' in effect_sub.columns and effect_sub['time'].drop_nulls().len() > 0:
                        per_startup[effect_label] = _rows_to_numeric(
                            effect_sub, 'per_startup', timesteps, periods, scenarios
                        )
                    else:
                        per_startup[effect_label] = effect_sub['per_startup'].drop_nulls().to_list()[0]
                if 'per_active_hour' in effect_sub.columns and effect_sub['per_active_hour'].drop_nulls().len() > 0:
                    if 'time' in effect_sub.columns and effect_sub['time'].drop_nulls().len() > 0:
                        per_active_hour[effect_label] = _rows_to_numeric(
                            effect_sub, 'per_active_hour', timesteps, periods, scenarios
                        )
                    else:
                        per_active_hour[effect_label] = effect_sub['per_active_hour'].drop_nulls().to_list()[0]
            se_map[element_label] = {'per_startup': per_startup, 'per_active_hour': per_active_hour}

    result: dict[str, StatusParameters] = {}
    for element_label in status_df['element'].unique().to_list():
        sub = status_df.filter(pl.col('element') == element_label)
        row = sub.row(0, named=True)
        element_type = row['element_type']

        sp_kwargs: dict[str, Any] = {}

        for col, kwarg in [
            ('active_hours_min', 'active_hours_min'),
            ('active_hours_max', 'active_hours_max'),
            ('startup_limit', 'startup_limit'),
        ]:
            if col in sub.columns and row.get(col) is not None:
                if _has_varying_rows(sub, col):
                    sp_kwargs[kwarg] = _ps_column_to_numeric(sub, col, periods, scenarios)
                else:
                    sp_kwargs[kwarg] = row[col]

        for col, kwarg in [
            ('min_uptime', 'min_uptime'),
            ('max_uptime', 'max_uptime'),
            ('min_downtime', 'min_downtime'),
            ('max_downtime', 'max_downtime'),
        ]:
            if col in sub.columns and row.get(col) is not None:
                sp_kwargs[kwarg] = row[col]

        if 'force_startup_tracking' in sub.columns and row.get('force_startup_tracking'):
            sp_kwargs['force_startup_tracking'] = True

        if 'cluster_mode' in sub.columns and row.get('cluster_mode') is not None:
            sp_kwargs['cluster_mode'] = row['cluster_mode']

        # Merge effects
        if element_label in se_map:
            se = se_map[element_label]
            if se['per_startup']:
                sp_kwargs['effects_per_startup'] = se['per_startup']
            if se['per_active_hour']:
                sp_kwargs['effects_per_active_hour'] = se['per_active_hour']

        key = f'{element_type}:{element_label}'
        result[key] = StatusParameters(**sp_kwargs)

    return result


def _apply_effect_bounds(
    eb_df: Any,
    effect_objects: dict[str, Any],
    timesteps: pd.Index,
    periods: pd.Index | None,
    scenarios: pd.Index | None,
    pl: Any,
) -> None:
    """Apply effect_bounds table values to Effect objects."""
    col_to_attr = {
        'min_temporal': 'minimum_temporal',
        'max_temporal': 'maximum_temporal',
        'min_periodic': 'minimum_periodic',
        'max_periodic': 'maximum_periodic',
        'min_total': 'minimum_total',
        'max_total': 'maximum_total',
        'min_over_periods': 'minimum_over_periods',
        'max_over_periods': 'maximum_over_periods',
        'period_weight': 'period_weights',
        'min_per_hour': 'minimum_per_hour',
        'max_per_hour': 'maximum_per_hour',
    }

    for effect_label in eb_df['effect'].unique().to_list():
        if effect_label not in effect_objects:
            continue
        sub = eb_df.filter(pl.col('effect') == effect_label)
        effect = effect_objects[effect_label]

        for col, attr in col_to_attr.items():
            if col not in sub.columns or sub[col].drop_nulls().len() == 0:
                continue
            if _has_varying_rows(sub, col):
                setattr(effect, attr, _ps_column_to_numeric(sub, col, periods, scenarios))
            else:
                row = sub.row(0, named=True)
                if row.get(col) is not None:
                    setattr(effect, attr, row[col])


def _apply_effect_shares(
    es_df: Any,
    effect_objects: dict[str, Any],
    timesteps: pd.Index,
    periods: pd.Index | None,
    scenarios: pd.Index | None,
    pl: Any,
) -> None:
    """Apply effect_shares table values to Effect objects."""
    for target_label in es_df['target_effect'].unique().to_list():
        if target_label not in effect_objects:
            continue
        target_sub = es_df.filter(pl.col('target_effect') == target_label)
        effect = effect_objects[target_label]

        temporal_shares: dict[str, Any] = {}
        periodic_shares: dict[str, Any] = {}

        for source_label in target_sub['source_effect'].unique().to_list():
            source_sub = target_sub.filter(pl.col('source_effect') == source_label)
            for share_type in source_sub['share_type'].unique().to_list():
                type_sub = source_sub.filter(pl.col('share_type') == share_type)
                if 'time' in type_sub.columns and type_sub['time'].drop_nulls().len() > 0:
                    value = _rows_to_numeric(type_sub, 'value', timesteps, periods, scenarios)
                else:
                    value = type_sub['value'].to_list()[0]

                if share_type == 'temporal':
                    temporal_shares[source_label] = value
                elif share_type == 'periodic':
                    periodic_shares[source_label] = value

        if temporal_shares:
            effect.share_from_temporal = temporal_shares
        if periodic_shares:
            effect.share_from_periodic = periodic_shares


# ---------------------------------------------------------------------------
# Internal helpers — to_tables direction
# ---------------------------------------------------------------------------


def _to_float(val: Any) -> float | None:
    """Safely convert a value to float, handling xr.DataArray scalars."""
    if val is None:
        return None
    if isinstance(val, xr.DataArray):
        return float(val.item())
    if isinstance(val, np.ndarray) and val.ndim == 0:
        return float(val.item())
    return float(val)


def _is_scalar(val: Any) -> bool:
    """Check if a value is a scalar (not array-like)."""
    if val is None:
        return True
    if isinstance(val, (int, float, np.integer, np.floating)):
        return True
    if isinstance(val, xr.DataArray):
        return val.ndim == 0 or val.size == 1
    if isinstance(val, np.ndarray):
        return val.ndim == 0 or val.size == 1
    if isinstance(val, (pd.Series, pd.DataFrame)):
        return False
    return True


def _scalar_or_none(val: Any) -> float | None:
    """Convert a value to a float scalar or None."""
    if val is None:
        return None
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    if isinstance(val, xr.DataArray):
        if val.ndim == 0 or val.size == 1:
            return float(val.item())
        return None
    if isinstance(val, np.ndarray):
        if val.ndim == 0 or val.size == 1:
            return float(val.item())
        return None
    return float(val)


def _numeric_to_rows(
    element_label: str,
    element_col: str,
    value: Any,
    timesteps: pd.Index,
    periods: pd.Index | None,
    scenarios: pd.Index | None,
    value_col: str = 'value',
) -> list[dict[str, Any]]:
    """Convert a numeric value to long-format row dicts for a timeseries table."""
    rows = []

    if isinstance(value, xr.DataArray):
        # Determine which dims are present
        da = value
        has_time = 'time' in da.dims
        has_period = 'period' in da.dims
        has_scenario = 'scenario' in da.dims

        if has_time and not has_period and not has_scenario:
            for i, t in enumerate(timesteps):
                row: dict[str, Any] = {element_col: element_label, 'time': t}
                row[value_col] = float(da.values[i])
                rows.append(row)
        elif has_time:
            # Iterate over all coords
            for idx in np.ndindex(da.shape):
                row = {element_col: element_label}
                for dim_idx, dim_name in enumerate(da.dims):
                    coord_val = da.coords[dim_name].values[idx[dim_idx]]
                    if dim_name == 'time':
                        row['time'] = _convert_timestamp(coord_val)
                    else:
                        row[dim_name] = _convert_coord(coord_val)
                row[value_col] = float(da.values[idx])
                rows.append(row)
        else:
            # No time dim — scalar or PS
            for idx in np.ndindex(da.shape):
                row = {element_col: element_label}
                for dim_idx, dim_name in enumerate(da.dims):
                    coord_val = da.coords[dim_name].values[idx[dim_idx]]
                    row[dim_name] = _convert_coord(coord_val)
                row[value_col] = float(da.values[idx])
                rows.append(row)
    elif isinstance(value, np.ndarray):
        if value.ndim == 1 and len(value) == len(timesteps):
            for i, t in enumerate(timesteps):
                rows.append({element_col: element_label, 'time': t, value_col: float(value[i])})
        else:
            # Flat array
            for i, t in enumerate(timesteps):
                if i < len(value):
                    rows.append({element_col: element_label, 'time': t, value_col: float(value[i])})
    else:
        rows.append({element_col: element_label, value_col: float(value)})

    return rows


def _numeric_to_value_rows(
    value: Any,
    timesteps: pd.Index,
    periods: pd.Index | None,
    scenarios: pd.Index | None,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert a numeric to rows with 'time', 'value', and optional extra columns."""
    rows = []
    base = dict(extra) if extra else {}

    if isinstance(value, xr.DataArray):
        for idx in np.ndindex(value.shape):
            row = dict(base)
            for dim_idx, dim_name in enumerate(value.dims):
                coord_val = value.coords[dim_name].values[idx[dim_idx]]
                if dim_name == 'time':
                    row['time'] = _convert_timestamp(coord_val)
                else:
                    row[dim_name] = _convert_coord(coord_val)
            row['value'] = float(value.values[idx])
            rows.append(row)
    elif isinstance(value, np.ndarray):
        for i, t in enumerate(timesteps):
            row = dict(base)
            row['time'] = t
            row['value'] = float(value[i]) if i < len(value) else 0.0
            rows.append(row)
    else:
        row = dict(base)
        row['value'] = float(value)
        rows.append(row)

    return rows


def _piece_to_rows(
    piece: Any,
    timesteps: pd.Index,
    periods: pd.Index | None,
    scenarios: pd.Index | None,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert a Piece with time-varying start/end to rows."""
    start_rows = _numeric_to_value_rows(piece.start, timesteps, periods, scenarios, extra)
    end_rows = _numeric_to_value_rows(piece.end, timesteps, periods, scenarios, extra)

    # Merge start/end into single rows
    rows = []
    for s_row, e_row in zip(start_rows, end_rows, strict=True):
        row = dict(s_row)
        row['start'] = row.pop('value')
        row['end'] = e_row['value']
        rows.append(row)
    return rows


def _convert_timestamp(val: Any) -> Any:
    """Convert a numpy datetime64 or similar to a Python-native type."""
    if isinstance(val, (np.datetime64, pd.Timestamp)):
        return pd.Timestamp(val).to_pydatetime()
    return val


def _convert_coord(val: Any) -> Any:
    """Convert a numpy scalar to a Python-native type."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.str_):
        return str(val)
    return val


def _add_flow_rows(
    flow: Any,
    component_label: str,
    direction: str,
    flow_rows: list[dict[str, Any]],
    flow_profile_rows: list[dict[str, Any]],
    flow_effect_rows: list[dict[str, Any]],
    investment_rows: list[dict[str, Any]],
    investment_effect_rows: list[dict[str, Any]],
    piecewise_invest_rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    status_effect_rows: list[dict[str, Any]],
    timesteps: pd.Index,
    periods: pd.Index | None,
    scenarios: pd.Index | None,
    standard_effect_label: str | None = None,
) -> None:
    """Add rows for a single Flow to the various output tables."""
    from .interface import InvestParameters

    flow_label = flow.label_full  # "component(short_label)"

    row: dict[str, Any] = {
        'flow': flow_label,
        'bus': flow.bus if isinstance(flow.bus, str) else flow.bus.label,
        'component': component_label,
        'direction': direction,
    }

    # Size
    if isinstance(flow.size, InvestParameters):
        row['size'] = None
        _add_invest_rows(
            flow_label,
            'flow',
            flow.size,
            investment_rows,
            investment_effect_rows,
            piecewise_invest_rows,
            periods,
            scenarios,
            standard_effect_label,
        )
    else:
        row['size'] = _scalar_or_none(flow.size)

    # Scalar defaults
    rel_min = flow.relative_minimum
    rel_max = flow.relative_maximum
    row['rel_min'] = float(rel_min) if _is_scalar(rel_min) else 0.0
    row['rel_max'] = float(rel_max) if _is_scalar(rel_max) else 1.0

    # PS params
    for attr, col in [
        ('flow_hours_min', 'flow_hours_min'),
        ('flow_hours_max', 'flow_hours_max'),
        ('load_factor_min', 'load_factor_min'),
        ('load_factor_max', 'load_factor_max'),
    ]:
        val = getattr(flow, attr, None)
        row[col] = _scalar_or_none(val)

    for attr, col in [
        ('flow_hours_min_over_periods', 'flow_hours_min_over_periods'),
        ('flow_hours_max_over_periods', 'flow_hours_max_over_periods'),
    ]:
        val = getattr(flow, attr, None)
        row[col] = _scalar_or_none(val)

    pfr = flow.previous_flow_rate
    if isinstance(pfr, (int, float)):
        row['previous_flow_rate'] = pfr
    elif pfr is not None:
        # Array-like: serialize as comma-separated string
        row['previous_flow_rate'] = ','.join(str(float(v)) for v in pfr)
    else:
        row['previous_flow_rate'] = None

    flow_rows.append(row)

    # Flow profiles (time-varying rel_min, rel_max, fixed_relative_profile)
    profile_data: dict[str, Any] = {}
    if flow.fixed_relative_profile is not None and not _is_scalar(flow.fixed_relative_profile):
        profile_data['fixed_profile'] = flow.fixed_relative_profile
    elif flow.fixed_relative_profile is not None and _is_scalar(flow.fixed_relative_profile):
        # Scalar profile goes into the profiles table too if it's set
        profile_data['fixed_profile'] = flow.fixed_relative_profile

    if not _is_scalar(rel_min):
        profile_data['rel_min'] = rel_min
    if not _is_scalar(rel_max):
        profile_data['rel_max'] = rel_max

    if profile_data:
        _add_profile_rows(flow_label, profile_data, flow_profile_rows, timesteps, periods, scenarios)

    # Flow effects
    effects = flow.effects_per_flow_hour
    if effects is not None and not isinstance(effects, dict):
        effects = {standard_effect_label: effects}
    if effects:
        for effect_label, effect_value in effects.items():
            if _is_scalar(effect_value):
                flow_effect_rows.append(
                    {
                        'flow': flow_label,
                        'effect': effect_label,
                        'value': float(effect_value),
                    }
                )
            else:
                for ts_row in _numeric_to_value_rows(
                    effect_value, timesteps, periods, scenarios, extra={'flow': flow_label, 'effect': effect_label}
                ):
                    flow_effect_rows.append(ts_row)

    # Status parameters
    if flow.status_parameters is not None:
        _add_status_rows(
            flow_label,
            'flow',
            flow.status_parameters,
            status_rows,
            status_effect_rows,
            timesteps,
            periods,
            scenarios,
            standard_effect_label,
        )


def _add_profile_rows(
    flow_label: str,
    profile_data: dict[str, Any],
    flow_profile_rows: list[dict[str, Any]],
    timesteps: pd.Index,
    periods: pd.Index | None,
    scenarios: pd.Index | None,
) -> None:
    """Add flow_profiles rows for time-varying profiles."""
    # Determine the most complex value to establish the time index
    has_ts = any(not _is_scalar(v) for v in profile_data.values())

    if not has_ts:
        # All scalars — single row without time
        row: dict[str, Any] = {'flow': flow_label}
        for col, val in profile_data.items():
            row[col] = float(val) if val is not None else None
        flow_profile_rows.append(row)
        return

    # Build time-indexed rows
    # Find the value with most dimensions to determine structure
    primary_val = next((v for v in profile_data.values() if not _is_scalar(v)), None)
    if primary_val is None:
        return

    if isinstance(primary_val, xr.DataArray):
        for idx in np.ndindex(primary_val.shape):
            row = {'flow': flow_label}
            for dim_idx, dim_name in enumerate(primary_val.dims):
                coord_val = primary_val.coords[dim_name].values[idx[dim_idx]]
                if dim_name == 'time':
                    row['time'] = _convert_timestamp(coord_val)
                else:
                    row[dim_name] = _convert_coord(coord_val)
            for col, val in profile_data.items():
                if isinstance(val, xr.DataArray):
                    if val.ndim == 0 or val.size == 1:
                        row[col] = float(val.item())
                    else:
                        row[col] = float(val.values[idx])
                elif _is_scalar(val):
                    row[col] = _to_float(val)
            flow_profile_rows.append(row)
    elif isinstance(primary_val, np.ndarray):
        for i, t in enumerate(timesteps):
            row = {'flow': flow_label, 'time': t}
            for col, val in profile_data.items():
                if isinstance(val, np.ndarray):
                    row[col] = float(val[i]) if i < len(val) else None
                elif isinstance(val, xr.DataArray):
                    if val.ndim == 0 or val.size == 1:
                        row[col] = float(val.item())
                    else:
                        row[col] = float(val.values[i]) if i < val.size else None
                elif _is_scalar(val):
                    row[col] = _to_float(val)
            flow_profile_rows.append(row)


def _add_invest_rows(
    element_label: str,
    element_type: str,
    invest: Any,
    investment_rows: list[dict[str, Any]],
    investment_effect_rows: list[dict[str, Any]],
    piecewise_invest_rows: list[dict[str, Any]],
    periods: pd.Index | None,
    scenarios: pd.Index | None,
    standard_effect_label: str | None = None,
) -> None:
    """Add rows for InvestParameters to the investments and investment_effects tables."""
    row: dict[str, Any] = {
        'element': element_label,
        'element_type': element_type,
    }

    row['fixed_size'] = _scalar_or_none(invest.fixed_size)
    row['minimum_size'] = _scalar_or_none(invest.minimum_size)
    row['maximum_size'] = _scalar_or_none(invest.maximum_size)
    row['mandatory'] = invest.mandatory

    if invest.linked_periods is not None:
        if isinstance(invest.linked_periods, tuple):
            row['linked_periods'] = f'{invest.linked_periods[0]}:{invest.linked_periods[1]}'
        else:
            row['linked_periods'] = str(invest.linked_periods)
    else:
        row['linked_periods'] = None

    investment_rows.append(row)

    # Investment effects — consolidate into one row per (element, effect)
    merged_ie: dict[str, dict[str, Any]] = {}  # effect_label -> row dict
    for attr, col in [
        ('effects_of_investment_per_size', 'per_size'),
        ('effects_of_investment', 'on_invest'),
        ('effects_of_retirement', 'on_retire'),
    ]:
        effect_dict = getattr(invest, attr, None)
        if effect_dict:
            if not isinstance(effect_dict, dict):
                effect_dict = {standard_effect_label: effect_dict}
            for effect_label, effect_value in effect_dict.items():
                if effect_label not in merged_ie:
                    merged_ie[effect_label] = {
                        'element': element_label,
                        'effect': effect_label,
                    }
                merged_ie[effect_label][col] = _scalar_or_none(effect_value)
    investment_effect_rows.extend(merged_ie.values())

    # Piecewise effects
    if invest.piecewise_effects_of_investment is not None:
        pw = invest.piecewise_effects_of_investment
        # Origin pieces
        for piece_idx, piece in enumerate(pw.piecewise_origin):
            piecewise_invest_rows.append(
                {
                    'element': element_label,
                    'piece_idx': piece_idx,
                    'type': '__origin__',
                    'start': float(piece.start),
                    'end': float(piece.end),
                }
            )
        # Share pieces per effect
        for effect_label, piecewise in pw.piecewise_shares.items():
            for piece_idx, piece in enumerate(piecewise):
                piecewise_invest_rows.append(
                    {
                        'element': element_label,
                        'piece_idx': piece_idx,
                        'type': effect_label,
                        'start': float(piece.start),
                        'end': float(piece.end),
                    }
                )


def _add_status_rows(
    element_label: str,
    element_type: str,
    status: Any,
    status_rows: list[dict[str, Any]],
    status_effect_rows: list[dict[str, Any]],
    timesteps: pd.Index,
    periods: pd.Index | None,
    scenarios: pd.Index | None,
    standard_effect_label: str | None = None,
) -> None:
    """Add rows for StatusParameters to status and status_effects tables."""
    row: dict[str, Any] = {
        'element': element_label,
        'element_type': element_type,
    }

    for attr, col in [
        ('active_hours_min', 'active_hours_min'),
        ('active_hours_max', 'active_hours_max'),
        ('startup_limit', 'startup_limit'),
        ('min_uptime', 'min_uptime'),
        ('max_uptime', 'max_uptime'),
        ('min_downtime', 'min_downtime'),
        ('max_downtime', 'max_downtime'),
    ]:
        val = getattr(status, attr, None)
        row[col] = _scalar_or_none(val)

    row['force_startup_tracking'] = status.force_startup_tracking
    row['cluster_mode'] = status.cluster_mode

    status_rows.append(row)

    # Status effects
    for attr, col in [
        ('effects_per_startup', 'per_startup'),
        ('effects_per_active_hour', 'per_active_hour'),
    ]:
        effect_dict = getattr(status, attr, None)
        if effect_dict:
            if isinstance(effect_dict, dict):
                for effect_label, effect_value in effect_dict.items():
                    if _is_scalar(effect_value):
                        status_effect_rows.append(
                            {
                                'element': element_label,
                                'effect': effect_label,
                                col: float(effect_value),
                            }
                        )
                    else:
                        for ts_row in _numeric_to_value_rows(
                            effect_value,
                            timesteps,
                            periods,
                            scenarios,
                            extra={'element': element_label, 'effect': effect_label},
                        ):
                            ts_row[col] = ts_row.pop('value')
                            status_effect_rows.append(ts_row)
            elif _is_scalar(effect_dict):
                # Numeric value for the standard effect
                status_effect_rows.append(
                    {
                        'element': element_label,
                        'effect': standard_effect_label,
                        col: float(effect_dict),
                    }
                )


def _extract_effect_bounds(
    effect: Any,
    periods: pd.Index | None,
    scenarios: pd.Index | None,
    timesteps: pd.Index,
) -> dict[str, Any] | None:
    """Extract effect bounds as a row dict, or None if no bounds set."""
    attr_map = {
        'minimum_temporal': 'min_temporal',
        'maximum_temporal': 'max_temporal',
        'minimum_periodic': 'min_periodic',
        'maximum_periodic': 'max_periodic',
        'minimum_total': 'min_total',
        'maximum_total': 'max_total',
        'minimum_over_periods': 'min_over_periods',
        'maximum_over_periods': 'max_over_periods',
        'period_weights': 'period_weight',
        'minimum_per_hour': 'min_per_hour',
        'maximum_per_hour': 'max_per_hour',
    }

    row: dict[str, Any] = {'effect': effect.label}
    has_any = False
    period_weight_da = None
    for attr, col in attr_map.items():
        val = getattr(effect, attr, None)
        if val is not None:
            if col == 'period_weight' and isinstance(val, xr.DataArray) and val.size > 1:
                # period_weights is period-indexed; handle as multi-row expansion below
                period_weight_da = val
                row[col] = None
                has_any = True
            else:
                row[col] = _scalar_or_none(val)
                if row[col] is not None:
                    has_any = True
                else:
                    row[col] = None
        else:
            row[col] = None

    if not has_any:
        return None

    if period_weight_da is not None:
        # Expand into one row per period
        rows = []
        for p_idx, p_val in enumerate(period_weight_da.coords['period'].values):
            r = dict(row)
            r['period'] = _convert_coord(p_val)
            r['period_weight'] = float(period_weight_da.values[p_idx])
            rows.append(r)
        return rows

    return row


def _extract_effect_shares(
    effect: Any,
    timesteps: pd.Index,
    periods: pd.Index | None,
    scenarios: pd.Index | None,
) -> list[dict[str, Any]]:
    """Extract effect share rows from an Effect object."""
    rows = []

    for attr, share_type in [
        ('share_from_temporal', 'temporal'),
        ('share_from_periodic', 'periodic'),
    ]:
        share_dict = getattr(effect, attr, None)
        if share_dict and isinstance(share_dict, dict):
            for source_label, value in share_dict.items():
                if _is_scalar(value):
                    rows.append(
                        {
                            'target_effect': effect.label,
                            'source_effect': source_label,
                            'share_type': share_type,
                            'value': float(value),
                        }
                    )
                else:
                    for ts_row in _numeric_to_value_rows(
                        value,
                        timesteps,
                        periods,
                        scenarios,
                        extra={
                            'target_effect': effect.label,
                            'source_effect': source_label,
                            'share_type': share_type,
                        },
                    ):
                        rows.append(ts_row)

    return rows
