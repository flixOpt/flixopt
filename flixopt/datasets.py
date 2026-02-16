"""Dataset builders for FlowSystem elements.

Functions that eagerly build xr.Dataset containers from element lists,
replacing lazy cached_property getters with a single upfront computation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from .core import align_effects_to_coords, align_to_coords
from .features import fast_isnull, stack_along_dim
from .interface import InvestParameters

if TYPE_CHECKING:
    from .elements import Flow
    from .interface import StatusParameters

logger = logging.getLogger('flixopt')

# Canonical dimension ordering for all arrays
_CANONICAL_ORDER = ['flow', 'cluster', 'time', 'period', 'scenario']


def _ensure_canonical_order(arr: xr.DataArray) -> xr.DataArray:
    """Ensure array has canonical dimension order and coord dict order."""
    actual_dims = [d for d in _CANONICAL_ORDER if d in arr.dims]
    for d in arr.dims:
        if d not in actual_dims:
            actual_dims.append(d)

    if list(arr.dims) != actual_dims:
        arr = arr.transpose(*actual_dims)

    if list(arr.coords.keys()) != list(arr.dims):
        ordered_coords = {d: arr.coords[d] for d in arr.dims}
        arr = xr.DataArray(arr.values, dims=arr.dims, coords=ordered_coords)

    return arr


def build_flows_dataset(
    flows: list[Flow],
    coords: dict[str, pd.Index],
    effect_ids: list[str],
    timestep_duration: xr.DataArray | float | None = None,
    normalize_effects: Any = None,
) -> xr.Dataset:
    """Build an xr.Dataset containing all numeric flow data.

    Args:
        flows: List of all Flow elements.
        coords: Model coordinate indexes (time, period, scenario).
        effect_ids: List of effect IDs for building effect arrays.
        timestep_duration: Duration per timestep (for previous duration computation).
        normalize_effects: Callable to normalize raw effect values.

    Returns:
        Dataset with all flow parameters as variables, boolean masks, and attrs.
    """
    from .batched import build_effects_array

    if not flows:
        return xr.Dataset()

    flow_ids = [f.id for f in flows]
    ids_index = pd.Index(flow_ids)
    dim = 'flow'

    def _align(flow, attr, dims=None):
        raw = getattr(flow, attr)
        return align_to_coords(raw, coords, name=f'{flow.id}|{attr}', dims=dims)

    def _model_coords(dims=None):
        if dims is None:
            dims = ['time', 'period', 'scenario']
        return {d: coords[d] for d in dims if d in coords}

    def _batched_parameter(ids, attr, dims):
        if not ids:
            return None
        by_id = {f.id: f for f in flows}
        values = [align_to_coords(getattr(by_id[fid], attr), coords, name=f'{fid}|{attr}', dims=dims) for fid in ids]
        arr = stack_along_dim(values, dim, ids, _model_coords(dims))
        return _ensure_canonical_order(arr)

    ds = xr.Dataset()

    # === Boolean masks ===
    def _mask(condition):
        return xr.DataArray([condition(f) for f in flows], dims=[dim], coords={dim: ids_index})

    ds['has_status'] = _mask(lambda f: f.status_parameters is not None)
    ds['has_investment'] = _mask(lambda f: isinstance(f.size, InvestParameters))
    ds['has_optional_investment'] = _mask(lambda f: isinstance(f.size, InvestParameters) and not f.size.mandatory)
    ds['has_mandatory_investment'] = _mask(lambda f: isinstance(f.size, InvestParameters) and f.size.mandatory)
    ds['has_fixed_size'] = _mask(lambda f: f.size is not None and not isinstance(f.size, InvestParameters))
    ds['has_size'] = _mask(lambda f: f.size is not None)
    ds['has_effects'] = _mask(lambda f: f.effects_per_flow_hour is not None)
    ds['has_flow_hours_min'] = _mask(lambda f: f.flow_hours_min is not None)
    ds['has_flow_hours_max'] = _mask(lambda f: f.flow_hours_max is not None)
    ds['has_load_factor_min'] = _mask(lambda f: f.load_factor_min is not None)
    ds['has_load_factor_max'] = _mask(lambda f: f.load_factor_max is not None)

    # Status tracking masks (inline StatusData logic)
    status_params = {f.id: f.status_parameters for f in flows if f.status_parameters is not None}

    def _status_mask(condition):
        mask = np.zeros(len(flow_ids), dtype=bool)
        for i, fid in enumerate(flow_ids):
            if fid in status_params:
                mask[i] = condition(status_params[fid])
        return xr.DataArray(mask, dims=[dim], coords={dim: ids_index})

    ds['has_startup_tracking'] = _status_mask(
        lambda p: (
            p.effects_per_startup
            or p.min_uptime is not None
            or p.max_uptime is not None
            or p.startup_limit is not None
            or p.force_startup_tracking
        )
    )
    ds['has_uptime_tracking'] = _status_mask(lambda p: p.min_uptime is not None or p.max_uptime is not None)
    ds['has_downtime_tracking'] = _status_mask(lambda p: p.min_downtime is not None or p.max_downtime is not None)
    ds['has_startup_limit'] = _status_mask(lambda p: p.startup_limit is not None)

    # === Relative bounds ===
    rel_min_values = [_align(f, 'relative_minimum') for f in flows]
    ds['relative_minimum'] = _ensure_canonical_order(
        stack_along_dim(rel_min_values, dim, flow_ids, _model_coords(None))
    )

    rel_max_values = [_align(f, 'relative_maximum') for f in flows]
    ds['relative_maximum'] = _ensure_canonical_order(
        stack_along_dim(rel_max_values, dim, flow_ids, _model_coords(None))
    )

    # Fixed relative profile
    fixed_values = [
        _align(f, 'fixed_relative_profile') if f.fixed_relative_profile is not None else np.nan for f in flows
    ]
    ds['fixed_relative_profile'] = _ensure_canonical_order(
        stack_along_dim(fixed_values, dim, flow_ids, _model_coords(None))
    )

    # Effective relative bounds
    fixed = ds['fixed_relative_profile']
    ds['effective_relative_minimum'] = ds['relative_minimum'].where(fast_isnull(fixed), fixed)
    ds['effective_relative_maximum'] = ds['relative_maximum'].where(fast_isnull(fixed), fixed)

    # === Size arrays ===
    fixed_size_values = []
    eff_size_lower_values = []
    eff_size_upper_values = []
    for f in flows:
        if f.size is None:
            fixed_size_values.append(np.nan)
            eff_size_lower_values.append(np.nan)
            eff_size_upper_values.append(np.nan)
        elif isinstance(f.size, InvestParameters):
            fixed_size_values.append(np.nan)
            eff_size_lower_values.append(f.size.minimum_or_fixed_size)
            eff_size_upper_values.append(f.size.maximum_or_fixed_size)
        else:
            aligned = _align(f, 'size', ['period', 'scenario'])
            fixed_size_values.append(aligned)
            eff_size_lower_values.append(aligned)
            eff_size_upper_values.append(aligned)

    ds['fixed_size'] = _ensure_canonical_order(
        stack_along_dim(fixed_size_values, dim, flow_ids, _model_coords(['period', 'scenario']))
    )
    ds['effective_size_lower'] = _ensure_canonical_order(
        stack_along_dim(eff_size_lower_values, dim, flow_ids, _model_coords(['period', 'scenario']))
    )
    ds['effective_size_upper'] = _ensure_canonical_order(
        stack_along_dim(eff_size_upper_values, dim, flow_ids, _model_coords(['period', 'scenario']))
    )

    # === Investment size bounds (all flows, NaN for non-investment) ===
    invest_ids = [f.id for f in flows if isinstance(f.size, InvestParameters)]
    if invest_ids:
        invest_params = {f.id: f.size for f in flows if isinstance(f.size, InvestParameters)}

        inv_min_values = [
            invest_params[fid].minimum_or_fixed_size if invest_params[fid].mandatory else 0.0 for fid in invest_ids
        ]
        inv_min = stack_along_dim(inv_min_values, dim, invest_ids)

        inv_max_values = [invest_params[fid].maximum_or_fixed_size for fid in invest_ids]
        inv_max = stack_along_dim(inv_max_values, dim, invest_ids)

        ds['size_minimum_all'] = _ensure_canonical_order(inv_min.reindex({dim: ids_index}))
        ds['size_maximum_all'] = _ensure_canonical_order(inv_max.reindex({dim: ids_index}))
    else:
        nan_arr = xr.DataArray(np.nan, dims=[dim], coords={dim: ids_index})
        ds['size_minimum_all'] = nan_arr
        ds['size_maximum_all'] = nan_arr

    # === Flow hours / load factor bounds (subset arrays) ===
    fh_min_ids = [f.id for f in flows if f.flow_hours_min is not None]
    fh = _batched_parameter(fh_min_ids, 'flow_hours_min', ['period', 'scenario'])
    if fh is not None:
        ds['flow_hours_minimum'] = fh

    fh_max_ids = [f.id for f in flows if f.flow_hours_max is not None]
    fh = _batched_parameter(fh_max_ids, 'flow_hours_max', ['period', 'scenario'])
    if fh is not None:
        ds['flow_hours_maximum'] = fh

    fh_op_min_ids = [f.id for f in flows if f.flow_hours_min_over_periods is not None]
    fh = _batched_parameter(fh_op_min_ids, 'flow_hours_min_over_periods', ['scenario'])
    if fh is not None:
        ds['flow_hours_minimum_over_periods'] = fh

    fh_op_max_ids = [f.id for f in flows if f.flow_hours_max_over_periods is not None]
    fh = _batched_parameter(fh_op_max_ids, 'flow_hours_max_over_periods', ['scenario'])
    if fh is not None:
        ds['flow_hours_maximum_over_periods'] = fh

    lf_min_ids = [f.id for f in flows if f.load_factor_min is not None]
    lf = _batched_parameter(lf_min_ids, 'load_factor_min', ['period', 'scenario'])
    if lf is not None:
        ds['load_factor_minimum'] = lf

    lf_max_ids = [f.id for f in flows if f.load_factor_max is not None]
    lf = _batched_parameter(lf_max_ids, 'load_factor_max', ['period', 'scenario'])
    if lf is not None:
        ds['load_factor_maximum'] = lf

    # === Effects per flow hour ===
    with_effects = [f.id for f in flows if f.effects_per_flow_hour is not None]
    if with_effects and effect_ids:
        norm = normalize_effects or (lambda x: x)
        by_id = {f.id: f for f in flows}
        dicts = {}
        for fid in with_effects:
            raw = by_id[fid].effects_per_flow_hour
            normalized = norm(raw) or {}
            aligned = align_effects_to_coords(normalized, coords, prefix=fid, suffix='per_flow_hour')
            dicts[fid] = aligned or {}
        arr = build_effects_array(dicts, effect_ids, dim)
        if arr is not None:
            ds['effects_per_flow_hour'] = arr

    # Note: linked_periods is NOT computed here — it's handled directly via
    # InvestParameters in InvestmentBuilder.add_linked_periods_constraints()

    # === Investment effects (delegated to InvestmentData patterns) ===
    if invest_ids:
        invest_params_dict = {f.id: f.size for f in flows if isinstance(f.size, InvestParameters)}
        _build_investment_effects(ds, invest_params_dict, dim, effect_ids, coords, normalize_effects)

    # === Status effects and bounds ===
    if status_params:
        _build_status_data(ds, status_params, dim, effect_ids, timestep_duration, flows, coords, normalize_effects)

    return ds


def _build_investment_effects(
    ds: xr.Dataset,
    invest_params: dict[str, InvestParameters],
    dim: str,
    effect_ids: list[str],
    coords: dict[str, pd.Index],
    normalize_effects: Any,
) -> None:
    """Add investment-related effect arrays to the dataset."""
    from .batched import InvestmentData

    inv = InvestmentData(
        params=invest_params,
        dim_name=dim,
        effect_ids=effect_ids,
        coords=coords,
        normalize_effects=normalize_effects,
    )

    # Effects per size
    if inv.effects_per_size is not None:
        ds['invest_effects_per_size'] = inv.effects_per_size

    # Effects of investment (optional)
    if inv.effects_of_investment is not None:
        ds['invest_effects_of_investment'] = inv.effects_of_investment

    # Effects of retirement (optional)
    if inv.effects_of_retirement is not None:
        ds['invest_effects_of_retirement'] = inv.effects_of_retirement

    # Mandatory investment effects
    if inv.effects_of_investment_mandatory is not None:
        ds['invest_effects_of_investment_mandatory'] = inv.effects_of_investment_mandatory

    # Constant retirement effects
    if inv.effects_of_retirement_constant is not None:
        ds['invest_effects_of_retirement_constant'] = inv.effects_of_retirement_constant

    # Optional investment size bounds
    if inv.optional_size_minimum is not None:
        ds['optional_investment_size_minimum'] = inv.optional_size_minimum
    if inv.optional_size_maximum is not None:
        ds['optional_investment_size_maximum'] = inv.optional_size_maximum

    # Piecewise effects
    if inv.piecewise_element_ids:
        ds.attrs['piecewise_element_ids'] = inv.piecewise_element_ids
        ds.attrs['piecewise_max_segments'] = inv.piecewise_max_segments
        ds.attrs['piecewise_effect_names'] = inv.piecewise_effect_names
        if inv.piecewise_segment_mask is not None:
            ds['piecewise_segment_mask'] = inv.piecewise_segment_mask
        if inv.piecewise_origin_starts is not None:
            ds['piecewise_origin_starts'] = inv.piecewise_origin_starts
        if inv.piecewise_origin_ends is not None:
            ds['piecewise_origin_ends'] = inv.piecewise_origin_ends
        if inv.piecewise_effect_starts is not None:
            ds['piecewise_effect_starts'] = inv.piecewise_effect_starts
        if inv.piecewise_effect_ends is not None:
            ds['piecewise_effect_ends'] = inv.piecewise_effect_ends


def _build_status_data(
    ds: xr.Dataset,
    status_params: dict[str, StatusParameters],
    dim: str,
    effect_ids: list[str],
    timestep_duration: xr.DataArray | float | None,
    flows: list[Flow],
    coords: dict[str, pd.Index],
    normalize_effects: Any,
) -> None:
    """Add status-related arrays to the dataset."""
    from .batched import StatusData

    # Build previous_states for duration computation
    from .config import CONFIG
    from .modeling import ModelingUtilitiesAbstract

    previous_states = {}
    for f in flows:
        if f.previous_flow_rate is not None:
            previous_states[f.id] = ModelingUtilitiesAbstract.to_binary(
                values=xr.DataArray(
                    [f.previous_flow_rate] if np.isscalar(f.previous_flow_rate) else f.previous_flow_rate,
                    dims='time',
                ),
                epsilon=CONFIG.Modeling.epsilon,
                dims='time',
            )

    sd = StatusData(
        params=status_params,
        dim_name=dim,
        effect_ids=effect_ids,
        timestep_duration=timestep_duration,
        previous_states=previous_states,
        coords=coords,
        normalize_effects=normalize_effects,
    )

    # Effects
    if sd.effects_per_active_hour is not None:
        ds['effects_per_active_hour'] = sd.effects_per_active_hour
    if sd.effects_per_startup is not None:
        ds['effects_per_startup'] = sd.effects_per_startup

    # Duration bounds
    if sd.min_uptime is not None:
        ds['min_uptime'] = sd.min_uptime
    if sd.max_uptime is not None:
        ds['max_uptime'] = sd.max_uptime
    if sd.min_downtime is not None:
        ds['min_downtime'] = sd.min_downtime
    if sd.max_downtime is not None:
        ds['max_downtime'] = sd.max_downtime
    if sd.startup_limit is not None:
        ds['startup_limit'] = sd.startup_limit
    if sd.previous_uptime is not None:
        ds['previous_uptime'] = sd.previous_uptime
    if sd.previous_downtime is not None:
        ds['previous_downtime'] = sd.previous_downtime
