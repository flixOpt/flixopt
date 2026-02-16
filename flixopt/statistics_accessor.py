"""Statistics accessor for FlowSystem.

This module provides a user-friendly API for analyzing optimization results
directly from a FlowSystem.

Structure:
    - `.stats` - Data/metrics access (cached xarray Datasets)
    - `.stats.plot` - Plotting methods using the statistics data

Example:
    >>> flow_system.optimize(solver)
    >>> # Data access
    >>> flow_system.stats.flow_rates
    >>> flow_system.stats.flow_hours
    >>> # Plotting
    >>> flow_system.stats.plot.balance('ElectricityBus')
    >>> flow_system.stats.plot.heatmap('Boiler|on')
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from xarray_plotly.figures import add_secondary_y, update_traces

from .color_processing import ColorType, hex_to_rgba, process_colors
from .config import CONFIG
from .plot_result import PlotResult
from .structure import EffectVarName, FlowVarName, StorageVarName

if TYPE_CHECKING:
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')

# Type aliases
SelectType = dict[str, Any]
"""xarray-style selection dict: {'time': slice(...), 'scenario': 'base'}"""

FilterType = str | list[str]
"""For include/exclude filtering: exact label(s) to match, e.g., 'Boiler(Q_th)' or ['Boiler(Q_th)', 'CHP(Q_th)']"""


# Sankey select types with Literal keys for IDE autocomplete
FlowSankeySelect = dict[Literal['flow', 'bus', 'component', 'carrier', 'time', 'period', 'scenario'], Any]
"""Select options for flow-based sankey: flow, bus, component, carrier, time, period, scenario."""

EffectsSankeySelect = dict[Literal['effect', 'component', 'contributor', 'period', 'scenario'], Any]
"""Select options for effects sankey: effect, component, contributor, period, scenario."""


# Default slot assignments for plotting methods
# Use None for slots that should be blocked (prevent auto-assignment)
_SLOT_DEFAULTS: dict[str, dict[str, str | None]] = {
    'balance': {'x': 'time', 'color': 'flow', 'pattern_shape': None},
    'carrier_balance': {'x': 'time', 'color': 'component', 'pattern_shape': None},
    'flows': {'x': 'time', 'color': 'flow', 'symbol': None},
    'charge_states': {'x': 'time', 'color': 'storage', 'symbol': None},
    'storage': {'x': 'time', 'color': 'flow', 'pattern_shape': None},
    'storage_line': {'x': 'time', 'color': None, 'line_dash': None, 'symbol': None},
    'sizes': {'x': 'element', 'color': 'element'},
    'duration_curve': {'color': 'variable', 'symbol': None},  # x is computed dynamically
    'effects': {},  # x is computed dynamically
    'heatmap': {},
}


def _apply_slot_defaults(plotly_kwargs: dict, method: str) -> None:
    """Apply default slot assignments for a plotting method."""
    defaults = _SLOT_DEFAULTS.get(method, {})
    for slot, value in defaults.items():
        plotly_kwargs.setdefault(slot, value)


def _reshape_time_for_heatmap(
    data: xr.DataArray,
    reshape: tuple[str, str],
    fill: Literal['ffill', 'bfill'] | None = 'ffill',
) -> xr.DataArray:
    """Reshape time dimension into 2D (timeframe × timestep) for heatmap display.

    Args:
        data: DataArray with 'time' dimension.
        reshape: Tuple of (outer_freq, inner_freq), e.g. ('D', 'h') for days × hours.
        fill: Method to fill missing values after resampling.

    Returns:
        DataArray with 'time' replaced by 'timestep' and 'timeframe' dimensions.
    """
    if 'time' not in data.dims:
        return data

    timeframes, timesteps_per_frame = reshape

    # Define formats for different combinations
    formats = {
        ('YS', 'W'): ('%Y', '%W'),
        ('YS', 'D'): ('%Y', '%j'),
        ('YS', 'h'): ('%Y', '%j %H:00'),
        ('MS', 'D'): ('%Y-%m', '%d'),
        ('MS', 'h'): ('%Y-%m', '%d %H:00'),
        ('W', 'D'): ('%Y-w%W', '%w_%A'),
        ('W', 'h'): ('%Y-w%W', '%w_%A %H:00'),
        ('D', 'h'): ('%Y-%m-%d', '%H:00'),
        ('D', '15min'): ('%Y-%m-%d', '%H:%M'),
        ('h', '15min'): ('%Y-%m-%d %H:00', '%M'),
        ('h', 'min'): ('%Y-%m-%d %H:00', '%M'),
    }

    format_pair = (timeframes, timesteps_per_frame)
    if format_pair not in formats:
        raise ValueError(f'{format_pair} is not a valid format. Choose from {list(formats.keys())}')
    period_format, step_format = formats[format_pair]

    # Resample along time dimension
    resampled = data.resample(time=timesteps_per_frame).mean()

    # Apply fill if specified
    if fill == 'ffill':
        resampled = resampled.ffill(dim='time')
    elif fill == 'bfill':
        resampled = resampled.bfill(dim='time')

    # Create period and step labels
    time_values = pd.to_datetime(resampled.coords['time'].values)
    period_labels = time_values.strftime(period_format)
    step_labels = time_values.strftime(step_format)

    # Handle special case for weekly day format
    if '%w_%A' in step_format:
        step_labels = pd.Series(step_labels).replace('0_Sunday', '7_Sunday').values

    # Add period and step as coordinates
    resampled = resampled.assign_coords({'timeframe': ('time', period_labels), 'timestep': ('time', step_labels)})

    # Convert to multi-index and unstack
    resampled = resampled.set_index(time=['timeframe', 'timestep'])
    result = resampled.unstack('time')

    # Reorder: timestep, timeframe, then other dimensions
    other_dims = [d for d in result.dims if d not in ['timestep', 'timeframe']]
    return result.transpose('timestep', 'timeframe', *other_dims)


def _apply_unified_hover(fig: go.Figure, unit: str = '', decimals: int = 1) -> None:
    """Apply unified hover mode with clean formatting to any Plotly figure.

    Sets up 'x unified' hovermode with spike lines and formats hover labels
    as '<b>name</b>: value unit'.

    Works with any plot type (area, bar, line, scatter).

    Args:
        fig: Plotly Figure to style.
        unit: Unit string to append (e.g., 'kW', 'MWh'). Empty for no unit.
        decimals: Number of decimal places for values.
    """
    unit_suffix = f' {unit}' if unit else ''
    hover_template = f'<b>%{{fullData.name}}</b>: %{{y:.{decimals}f}}{unit_suffix}<extra></extra>'

    # Apply to all traces (main + animation frames) using xarray_plotly helper
    update_traces(fig, hovertemplate=hover_template)

    # Layout settings for unified hover
    fig.update_layout(hovermode='x unified')
    # Apply spike settings to all x-axes (for faceted plots with xaxis, xaxis2, xaxis3, etc.)
    fig.update_xaxes(showspikes=True, spikecolor='gray', spikethickness=1)


# --- Helper functions ---


def _prepare_for_heatmap(
    da: xr.DataArray,
    reshape: tuple[str, str] | Literal['auto'] | None,
) -> xr.DataArray:
    """Prepare DataArray for heatmap: determine axes, reshape if needed, transpose/squeeze.

    Args:
        da: DataArray to prepare for heatmap display.
        reshape: Time reshape frequencies as (outer, inner), 'auto' to auto-detect,
            or None to disable reshaping.
    """

    def finalize(da: xr.DataArray, heatmap_dims: list[str]) -> xr.DataArray:
        """Transpose, squeeze, and clear name if needed."""
        other = [d for d in da.dims if d not in heatmap_dims]
        da = da.transpose(*[d for d in heatmap_dims if d in da.dims], *other)
        for dim in [d for d in da.dims if d not in heatmap_dims and da.sizes[d] == 1]:
            da = da.squeeze(dim, drop=True)
        return da.rename('') if da.sizes.get('variable', 1) > 1 else da

    def fallback_dims() -> list[str]:
        """Default dims: (variable, time) if multi-var, else first 2 dims with size > 1."""
        if da.sizes.get('variable', 1) > 1:
            return ['variable', 'time']
        dims = [d for d in da.dims if da.sizes[d] > 1][:2]
        return dims if len(dims) >= 2 else list(da.dims)[:2]

    def can_auto_reshape() -> bool:
        """Check if data is suitable for auto-reshaping (not too many non-time dims)."""
        non_time_dims = [d for d in da.dims if d not in ('time', 'timestep', 'timeframe') and da.sizes[d] > 1]
        # Allow reshape if we have at most 1 other dimension (can facet on it)
        # Or if it's just variable dimension
        return len(non_time_dims) <= 1

    is_clustered = 'cluster' in da.dims and da.sizes['cluster'] > 1
    has_time = 'time' in da.dims

    # Clustered: use (time, cluster) as natural 2D
    if is_clustered and reshape in (None, 'auto'):
        return finalize(da, ['time', 'cluster'])

    # Apply auto-reshape: try ('D', 'h') by default if appropriate
    if reshape == 'auto' and has_time and can_auto_reshape():
        try:
            return finalize(_reshape_time_for_heatmap(da, ('D', 'h')), ['timestep', 'timeframe'])
        except (ValueError, KeyError):
            # Fall through to default dims if reshape fails
            pass

    # Apply explicit reshape if specified
    if reshape and reshape != 'auto' and has_time:
        return finalize(_reshape_time_for_heatmap(da, reshape), ['timestep', 'timeframe'])

    return finalize(da, fallback_dims())


def _filter_by_labels(
    names: list[str],
    include: FilterType | None,
    exclude: FilterType | None,
) -> list[str]:
    """Filter names using exact string matching.

    Args:
        names: List of names to filter.
        include: Only keep names that exactly match one of these labels.
        exclude: Remove names that exactly match one of these labels.

    Returns:
        Filtered list of names.
    """
    result = names.copy()
    if include is not None:
        include_set = {include} if isinstance(include, str) else set(include)
        result = [n for n in result if n in include_set]
    if exclude is not None:
        exclude_set = {exclude} if isinstance(exclude, str) else set(exclude)
        result = [n for n in result if n not in exclude_set]
    return result


def _apply_selection(
    data: xr.Dataset | xr.DataArray, select: SelectType | None, drop: bool = True
) -> xr.Dataset | xr.DataArray:
    """Apply xarray-style selection to dataset or dataarray.

    Args:
        data: Dataset or DataArray to select from.
        select: xarray-style selection dict.
        drop: If True (default), drop dimensions that become scalar after selection.
            This prevents auto-faceting when selecting a single value.
    """
    if select is None:
        return data
    valid_select = {k: v for k, v in select.items() if k in data.dims or k in data.coords}
    if valid_select:
        data = data.sel(valid_select, drop=drop)
    return data


def _sort_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Sort dataset variables alphabetically for consistent plotting order."""
    sorted_vars = sorted(ds.data_vars)
    return ds[sorted_vars]


def _sort_dataarray(da: xr.DataArray, dim: str) -> xr.DataArray:
    """Sort DataArray along a dimension alphabetically for consistent plotting order."""
    if dim not in da.dims:
        return da
    sorted_idx = sorted(da.coords[dim].values)
    return da.sel({dim: sorted_idx})


def _filter_small_variables(ds: xr.Dataset, threshold: float | None) -> xr.Dataset:
    """Remove variables where max absolute value is below threshold.

    Useful for filtering out solver noise or non-invested components.

    Args:
        ds: Dataset to filter.
        threshold: Minimum max absolute value to keep. If None, no filtering.

    Returns:
        Filtered dataset.
    """
    if threshold is None or not ds.data_vars:
        return ds
    max_vals = abs(ds).max()  # Single computation for all variables
    keep = [v for v in ds.data_vars if float(max_vals.variables[v].values) >= threshold]
    return ds[keep] if keep else ds


def _filter_small_dataarray(da: xr.DataArray, dim: str, threshold: float | None) -> xr.DataArray:
    """Remove entries along a dimension where max absolute value is below threshold.

    Args:
        da: DataArray to filter.
        dim: Dimension to filter along.
        threshold: Minimum max absolute value to keep. If None, no filtering.

    Returns:
        Filtered DataArray.
    """
    if threshold is None or dim not in da.dims:
        return da
    other_dims = [d for d in da.dims if d != dim]
    if other_dims:
        max_vals = abs(da).max(other_dims)
    else:
        max_vals = abs(da)
    keep = max_vals >= threshold
    return da.sel({dim: keep})


def _filter_by_carrier(ds: xr.Dataset, carrier: str | list[str] | None) -> xr.Dataset:
    """Filter dataset variables by carrier attribute.

    Args:
        ds: Dataset with variables that have 'carrier' attributes.
        carrier: Carrier name(s) to keep. None means no filtering.

    Returns:
        Dataset containing only variables matching the carrier(s).
    """
    if carrier is None:
        return ds

    carriers = [carrier] if isinstance(carrier, str) else carrier
    carriers = [c.lower() for c in carriers]

    matching_vars = [var for var in ds.data_vars if ds[var].attrs.get('carrier', '').lower() in carriers]
    return ds[matching_vars] if matching_vars else xr.Dataset()


def _dataset_to_long_df(ds: xr.Dataset, value_name: str = 'value', var_name: str = 'variable') -> pd.DataFrame:
    """Convert xarray Dataset to long-form DataFrame for plotly express."""
    if not ds.data_vars:
        return pd.DataFrame()
    if all(ds[var].ndim == 0 for var in ds.data_vars):
        rows = [{var_name: var, value_name: float(ds[var].values)} for var in ds.data_vars]
        return pd.DataFrame(rows)
    df = ds.to_dataframe().reset_index()
    # Only use coordinates that are actually present as columns after reset_index
    coord_cols = [c for c in ds.coords.keys() if c in df.columns]
    return df.melt(id_vars=coord_cols, var_name=var_name, value_name=value_name)


def _build_color_kwargs(colors: ColorType | None, labels: list[str]) -> dict[str, Any]:
    """Build color kwargs for plotly based on color type (no smart defaults).

    Args:
        colors: Dict (color_discrete_map), list (color_discrete_sequence),
               or string (colorscale name to convert to dict).
        labels: Variable labels for creating dict from colorscale name.

    Returns:
        Dict with either 'color_discrete_map' or 'color_discrete_sequence'.
    """
    if colors is None:
        return {}
    if isinstance(colors, dict):
        return {'color_discrete_map': colors}
    if isinstance(colors, list):
        return {'color_discrete_sequence': colors}
    if isinstance(colors, str):
        return {'color_discrete_map': process_colors(colors, labels)}
    return {}


def _merge_color_kwargs(
    colors: ColorType | None,
    labels: list[str],
    smart_defaults: dict[str, str],
) -> dict[str, Any]:
    """Build color kwargs, merging user colors with smart defaults.

    Args:
        colors: User-provided colors (dict, list, str colorscale, or None).
        labels: Variable labels (used for colorscale conversion).
        smart_defaults: Pre-computed smart default color map.

    Returns:
        Dict with 'color_discrete_map' or 'color_discrete_sequence'.

    Behavior:
        - None: Use smart_defaults
        - dict: Merge with smart_defaults (user overrides win)
        - list: Use as color_discrete_sequence (no smart defaults)
        - str: Convert colorscale to map (no smart defaults)
    """
    if colors is None:
        return {'color_discrete_map': smart_defaults}

    if isinstance(colors, dict):
        merged = smart_defaults.copy()
        merged.update(colors)  # User overrides win
        return {'color_discrete_map': merged}

    if isinstance(colors, list):
        return {'color_discrete_sequence': colors}

    if isinstance(colors, str):
        return {'color_discrete_map': process_colors(colors, labels)}

    return {'color_discrete_map': smart_defaults}


# --- Statistics Accessor (data only) ---


class StatisticsAccessor:
    """Statistics accessor for FlowSystem. Access via ``flow_system.stats``.

    This accessor provides cached data properties for optimization results.
    Use ``.plot`` for visualization methods.

    Data Properties:
        ``flow_rates`` : xr.DataArray
            Flow rates for all flows (dims: flow, time).
        ``flow_hours`` : xr.DataArray
            Flow hours (energy) for all flows (dims: flow, time).
        ``sizes`` : xr.DataArray
            Sizes for all flows and storages (dim: element).
        ``charge_states`` : xr.DataArray
            Charge states for all storage components (dims: storage, time).
        ``temporal_effects`` : xr.DataArray
            Temporal effects per contributor per timestep (dims: effect, contributor, time).
        ``periodic_effects`` : xr.DataArray
            Periodic (investment) effects per contributor (dims: effect, contributor).
        ``total_effects`` : xr.DataArray
            Total effects (temporal + periodic) per contributor (dims: effect, contributor).
        ``effect_share_factors`` : dict
            Conversion factors between effects.

    Examples:
        >>> flow_system.optimize(solver)
        >>> flow_system.stats.flow_rates  # Get data
        >>> flow_system.stats.plot.balance('Bus')  # Plot
    """

    def __init__(self, flow_system: FlowSystem) -> None:
        self._fs = flow_system
        # Plotting accessor (lazy)
        self._plot: StatisticsPlotAccessor | None = None

    def _require_solution(self) -> xr.Dataset:
        """Get solution, raising if not available."""
        if self._fs.solution is None:
            raise RuntimeError('FlowSystem has no solution. Run optimize() or solve() first.')
        return self._fs.solution

    @property
    def carrier_colors(self) -> dict[str, str]:
        """Cached mapping of carrier name to color.

        Delegates to topology accessor for centralized color caching.

        Returns:
            Dict mapping carrier names (lowercase) to hex color strings.
        """
        return self._fs.topology.carrier_colors

    @property
    def component_colors(self) -> dict[str, str]:
        """Cached mapping of component label to color.

        Delegates to topology accessor for centralized color caching.

        Returns:
            Dict mapping component labels to hex color strings.
        """
        return self._fs.topology.component_colors

    @property
    def flow_colors(self) -> dict[str, str]:
        """Cached mapping of flow label_full to color (from parent component).

        Delegates to topology accessor for centralized color caching.

        Returns:
            Dict mapping flow labels (e.g., 'Boiler(Q_th)') to hex color strings.
        """
        return self._fs.topology.flow_colors

    @property
    def bus_colors(self) -> dict[str, str]:
        """Cached mapping of bus label to color (from carrier).

        Delegates to topology accessor for centralized color caching.

        Returns:
            Dict mapping bus labels to hex color strings.
        """
        return self._fs.topology.bus_colors

    @property
    def carrier_units(self) -> dict[str, str]:
        """Cached mapping of carrier name to unit string.

        Delegates to topology accessor for centralized unit caching.

        Returns:
            Dict mapping carrier names (lowercase) to unit strings.
        """
        return self._fs.topology.carrier_units

    @property
    def effect_units(self) -> dict[str, str]:
        """Cached mapping of effect label to unit string.

        Delegates to topology accessor for centralized unit caching.

        Returns:
            Dict mapping effect labels to unit strings.
        """
        return self._fs.topology.effect_units

    @property
    def plot(self) -> StatisticsPlotAccessor:
        """Access plotting methods for statistics.

        Returns:
            A StatisticsPlotAccessor instance.

        Examples:
            >>> flow_system.stats.plot.balance('ElectricityBus')
            >>> flow_system.stats.plot.heatmap('Boiler|on')
        """
        if self._plot is None:
            self._plot = StatisticsPlotAccessor(self)
        return self._plot

    @cached_property
    def flow_rates(self) -> xr.DataArray:
        """All flow rates as a DataArray with 'flow' dimension."""
        self._require_solution()
        return self._fs.solution[FlowVarName.RATE]

    @cached_property
    def flow_hours(self) -> xr.DataArray:
        """All flow hours (energy) as a DataArray with 'flow' dimension."""
        return self.flow_rates * self._fs.timestep_duration

    @cached_property
    def flow_sizes(self) -> xr.DataArray:
        """Flow sizes as a DataArray with 'flow' dimension."""
        self._require_solution()
        return self._fs.solution[FlowVarName.SIZE].dropna('flow', how='all')

    @cached_property
    def storage_sizes(self) -> xr.DataArray:
        """Storage capacity sizes as a DataArray with 'storage' dimension."""
        self._require_solution()
        return self._fs.solution[StorageVarName.SIZE].dropna('storage', how='all')

    @cached_property
    def sizes(self) -> xr.DataArray:
        """All investment sizes (flows and storage capacities) as a DataArray with 'element' dim."""
        return xr.concat(
            [self.flow_sizes.rename(flow='element'), self.storage_sizes.rename(storage='element')],
            dim='element',
        )

    @cached_property
    def charge_states(self) -> xr.DataArray:
        """All storage charge states as a DataArray with 'storage' dimension."""
        self._require_solution()
        return self._fs.solution[StorageVarName.CHARGE]

    @cached_property
    def effect_share_factors(self) -> dict[str, dict]:
        """Effect share factors for temporal and periodic modes.

        Returns:
            Dict with 'temporal' and 'periodic' keys, each containing
            conversion factors between effects.
        """
        self._require_solution()
        factors = self._fs.effects.calculate_effect_share_factors()
        return {'temporal': factors[0], 'periodic': factors[1]}

    @cached_property
    def temporal_effects(self) -> xr.DataArray:
        """Temporal effects per contributor per timestep.

        Returns a DataArray with dimensions [effect, time, contributor]
        (plus period/scenario if present).

        Coordinates:
            - contributor: Individual contributor labels
            - component: Parent component label for groupby operations
            - component_type: Component type (e.g., 'Boiler', 'Source', 'Sink')

        Examples:
            >>> # Get costs per contributor per timestep
            >>> statistics.temporal_effects.sel(effect='costs')
            >>> # Sum over all contributors to get total costs per timestep
            >>> statistics.temporal_effects.sel(effect='costs').sum('contributor')
            >>> # Group by component
            >>> statistics.temporal_effects.sel(effect='costs').groupby('component').sum()

        Returns:
            xr.DataArray with effect, contributor, and time dimensions.
        """
        self._require_solution()
        da = self._create_effects_array('temporal')
        dim_order = ['effect', 'time', 'period', 'scenario', 'contributor']
        return da.transpose(*dim_order, missing_dims='ignore')

    @cached_property
    def periodic_effects(self) -> xr.DataArray:
        """Periodic (investment) effects per contributor.

        Returns a DataArray with dimensions [effect, contributor]
        (plus period/scenario if present).

        Coordinates:
            - contributor: Individual contributor labels
            - component: Parent component label for groupby operations
            - component_type: Component type (e.g., 'Boiler', 'Source', 'Sink')

        Examples:
            >>> # Get investment costs per contributor
            >>> statistics.periodic_effects.sel(effect='costs')
            >>> # Sum over all contributors to get total investment costs
            >>> statistics.periodic_effects.sel(effect='costs').sum('contributor')
            >>> # Group by component
            >>> statistics.periodic_effects.sel(effect='costs').groupby('component').sum()

        Returns:
            xr.DataArray with effect and contributor dimensions.
        """
        self._require_solution()
        da = self._create_effects_array('periodic')
        dim_order = ['effect', 'period', 'scenario', 'contributor']
        return da.transpose(*dim_order, missing_dims='ignore')

    @cached_property
    def total_effects(self) -> xr.DataArray:
        """Total effects (temporal + periodic) per contributor.

        Returns a DataArray with dimensions [effect, contributor]
        (plus period/scenario if present).

        Coordinates:
            - contributor: Individual contributor labels
            - component: Parent component label for groupby operations
            - component_type: Component type (e.g., 'Boiler', 'Source', 'Sink')

        Examples:
            >>> # Get total costs per contributor
            >>> statistics.total_effects.sel(effect='costs')
            >>> # Sum over all contributors to get total system costs
            >>> statistics.total_effects.sel(effect='costs').sum('contributor')
            >>> # Group by component
            >>> statistics.total_effects.sel(effect='costs').groupby('component').sum()
            >>> # Group by component type
            >>> statistics.total_effects.sel(effect='costs').groupby('component_type').sum()

        Returns:
            xr.DataArray with effect and contributor dimensions.
        """
        self._require_solution()
        da = self._create_effects_array('total')
        dim_order = ['effect', 'period', 'scenario', 'contributor']
        return da.transpose(*dim_order, missing_dims='ignore')

    def get_effect_shares(
        self,
        element: str,
        effect: str,
        mode: Literal['temporal', 'periodic'] | None = None,
        include_flows: bool = False,
    ) -> xr.Dataset:
        """Retrieve individual effect shares for a specific element and effect.

        Args:
            element: The element identifier (component or flow label).
            effect: The effect identifier.
            mode: 'temporal', 'periodic', or None for both.
            include_flows: Whether to include effects from flows connected to this element.

        Returns:
            xr.Dataset containing the requested effect shares.

        Raises:
            ValueError: If the effect is not available or mode is invalid.
        """
        self._require_solution()

        if effect not in self._fs.effects:
            raise ValueError(f'Effect {effect} is not available.')

        if mode is None:
            return xr.merge(
                [
                    self.get_effect_shares(
                        element=element, effect=effect, mode='temporal', include_flows=include_flows
                    ),
                    self.get_effect_shares(
                        element=element, effect=effect, mode='periodic', include_flows=include_flows
                    ),
                ]
            )

        if mode not in ['temporal', 'periodic']:
            raise ValueError(f'Mode {mode} is not available. Choose between "temporal" and "periodic".')

        ds = xr.Dataset()
        label = f'{element}->{effect}({mode})'
        if label in self._fs.solution:
            ds = xr.Dataset({label: self._fs.solution[label]})

        if include_flows:
            if element not in self._fs.components:
                raise ValueError(f'Only use Components when retrieving Effects including flows. Got {element}')
            comp = self._fs.components[element]
            flows = [flow.split('|')[0] for flow in comp.flows]
            return xr.merge(
                [ds]
                + [
                    self.get_effect_shares(element=flow, effect=effect, mode=mode, include_flows=False)
                    for flow in flows
                ]
            )

        return ds

    def _create_template_for_mode(self, mode: Literal['temporal', 'periodic', 'total']) -> xr.DataArray:
        """Create a template DataArray with the correct dimensions for a given mode."""
        coords = {}
        if mode == 'temporal':
            # Use solution's time coordinates if available (handles expanded solutions with extra timestep)
            solution = self._fs.solution
            if solution is not None and 'time' in solution.dims:
                coords['time'] = solution.coords['time'].values
            else:
                coords['time'] = self._fs.timesteps
        if self._fs.periods is not None:
            coords['period'] = self._fs.periods
        if self._fs.scenarios is not None:
            coords['scenario'] = self._fs.scenarios

        if coords:
            shape = tuple(len(coords[dim]) for dim in coords)
            return xr.DataArray(np.full(shape, np.nan, dtype=float), coords=coords, dims=list(coords.keys()))
        else:
            return xr.DataArray(np.nan)

    def _create_effects_array(self, mode: Literal['temporal', 'periodic', 'total']) -> xr.DataArray:
        """Create DataArray containing effect totals for all contributors.

        Returns a DataArray with dimensions (effect, contributor, ...) where ...
        depends on mode (time for temporal, nothing for periodic/total).

        Uses batched share|temporal and share|periodic DataArrays from the solution.
        Excludes effect-to-effect shares which are intermediate conversions.
        Provides component and component_type coordinates for flexible groupby operations.
        """
        solution = self._fs.solution
        template = self._create_template_for_mode(mode)
        effect_labels = set(self._fs.effects.keys())

        # Determine modes to process
        modes_to_process = ['temporal', 'periodic'] if mode == 'total' else [mode]
        # Detect contributors from combined share variables (share|temporal, share|periodic)
        detected_contributors: set[str] = set()
        for current_mode in modes_to_process:
            share_name = f'share|{current_mode}'
            if share_name not in solution:
                continue
            share_da = solution[share_name]
            for c in share_da.coords['contributor'].values:
                base_name = str(c).split('(')[0] if '(' in str(c) else str(c)
                if base_name not in effect_labels:
                    detected_contributors.add(str(c))

        contributors = sorted(detected_contributors)

        if not contributors:
            return xr.DataArray()

        # Build metadata for each contributor
        def get_parent_component(contributor: str) -> str:
            if contributor in self._fs.flows:
                return self._fs.flows[contributor].component
            elif contributor in self._fs.components:
                return contributor
            return contributor

        def get_contributor_type(contributor: str) -> str:
            if contributor in self._fs.flows:
                parent = self._fs.flows[contributor].component
                return type(self._fs.components[parent]).__name__
            elif contributor in self._fs.components:
                return type(self._fs.components[contributor]).__name__
            elif contributor in self._fs.buses:
                return type(self._fs.buses[contributor]).__name__
            return 'Unknown'

        parents = [get_parent_component(c) for c in contributors]
        contributor_types = [get_contributor_type(c) for c in contributors]

        effect_arrays = []

        for effect in self._fs.effects:
            contributor_arrays = []

            for contributor in contributors:
                share_total: xr.DataArray | None = None

                for current_mode in modes_to_process:
                    # Get conversion factors: which source effects contribute to this target effect
                    conversion_factors = {
                        key[0]: value
                        for key, value in self.effect_share_factors[current_mode].items()
                        if key[1] == effect
                    }
                    conversion_factors[effect] = 1  # Direct contribution

                    for source_effect, factor in conversion_factors.items():
                        share_name = f'share|{current_mode}'
                        if share_name not in solution:
                            continue
                        share_da = solution[share_name]
                        if source_effect not in share_da.coords['effect'].values:
                            continue
                        if contributor not in share_da.coords['contributor'].values:
                            continue
                        da = share_da.sel(effect=source_effect, contributor=contributor, drop=True).fillna(0) * factor
                        # For total mode, sum temporal over time (apply cluster_weight for proper weighting)
                        if mode == 'total' and current_mode == 'temporal' and 'time' in da.dims:
                            weighted = da * self._fs.weights.get('cluster', 1.0)
                            temporal_dims = [d for d in weighted.dims if d not in ('period', 'scenario')]
                            da = weighted.sum(temporal_dims)
                        if share_total is None:
                            share_total = da
                        else:
                            share_total = share_total + da

                # If no share found, use NaN template
                if share_total is None:
                    share_total = xr.full_like(template, np.nan, dtype=float)

                contributor_arrays.append(share_total.expand_dims(contributor=[contributor]))

            # Concatenate all contributors for this effect
            effect_da = xr.concat(contributor_arrays, dim='contributor', coords='minimal', join='outer')
            effect_arrays.append(effect_da.expand_dims(effect=[effect]))

        # Concatenate all effects
        result = xr.concat(effect_arrays, dim='effect', coords='minimal', join='outer')

        # Add unit coordinate for effect dimension
        effect_units = [self.effect_units.get(e, '') for e in self._fs.effects]
        result = result.assign_coords(
            effect_unit=('effect', effect_units),
            component=('contributor', parents),
            component_type=('contributor', contributor_types),
        )

        # Validation: check totals match solution
        effect_var_map = {
            'temporal': EffectVarName.PER_TIMESTEP,
            'periodic': EffectVarName.PERIODIC,
            'total': EffectVarName.TOTAL,
        }
        effect_var_name = effect_var_map[mode]
        if effect_var_name in solution:
            for effect in self._fs.effects:
                if effect in solution[effect_var_name].coords.get('effect', xr.DataArray([])).values:
                    computed = result.sel(effect=effect).sum('contributor')
                    found = solution[effect_var_name].sel(effect=effect)
                    if not np.allclose(computed.fillna(0).values, found.fillna(0).values, equal_nan=True):
                        logger.critical(
                            f'Results for {effect}({mode}) in effects_array doesnt match {effect_var_name}\n{computed=}\n, {found=}'
                        )

        return result


# --- Sankey Plot Accessor ---


class SankeyPlotAccessor:
    """Sankey diagram accessor. Access via ``flow_system.stats.plot.sankey``.

    Provides typed methods for different sankey diagram types.

    Examples:
        >>> fs.stats.plot.sankey.flows(select={'bus': 'HeatBus'})
        >>> fs.stats.plot.sankey.effects(select={'effect': 'costs'})
        >>> fs.stats.plot.sankey.sizes(select={'component': 'Boiler'})
    """

    def __init__(self, plot_accessor: StatisticsPlotAccessor) -> None:
        self._plot = plot_accessor
        self._stats = plot_accessor._stats
        self._fs = plot_accessor._fs

    def _extract_flow_filters(
        self, select: FlowSankeySelect | None
    ) -> tuple[SelectType | None, list[str] | None, list[str] | None, list[str] | None, list[str] | None]:
        """Extract special filters from select dict.

        Returns:
            Tuple of (xarray_select, flow_filter, bus_filter, component_filter, carrier_filter).
        """
        if select is None:
            return None, None, None, None, None

        select = dict(select)  # Copy to avoid mutating original
        flow_filter = select.pop('flow', None)
        bus_filter = select.pop('bus', None)
        component_filter = select.pop('component', None)
        carrier_filter = select.pop('carrier', None)

        # Normalize to lists
        if isinstance(flow_filter, str):
            flow_filter = [flow_filter]
        if isinstance(bus_filter, str):
            bus_filter = [bus_filter]
        if isinstance(component_filter, str):
            component_filter = [component_filter]
        if isinstance(carrier_filter, str):
            carrier_filter = [carrier_filter]

        return select if select else None, flow_filter, bus_filter, component_filter, carrier_filter

    def _build_flow_links(
        self,
        ds: xr.Dataset,
        flow_filter: list[str] | None = None,
        bus_filter: list[str] | None = None,
        component_filter: list[str] | None = None,
        carrier_filter: list[str] | None = None,
        min_value: float = 1e-6,
    ) -> tuple[set[str], dict[str, list]]:
        """Build Sankey nodes and links from flow data."""
        nodes: set[str] = set()
        links: dict[str, list] = {'source': [], 'target': [], 'value': [], 'label': [], 'carrier': []}

        # Normalize carrier filter to lowercase
        if carrier_filter is not None:
            carrier_filter = [c.lower() for c in carrier_filter]

        # Extract all topology metadata as plain dicts for fast lookup
        topo = self._fs.topology.flows
        flow_labels = topo.coords['flow'].values
        topo_bus = dict(zip(flow_labels, topo.coords['bus'].values, strict=False))
        topo_comp = dict(zip(flow_labels, topo.coords['component'].values, strict=False))
        topo_carrier = dict(zip(flow_labels, topo.coords['carrier'].values, strict=False))
        topo_is_input = dict(zip(flow_labels, topo.coords['is_input'].values, strict=False))

        for label in flow_labels:
            if label not in ds:
                continue

            # Apply filters
            if flow_filter is not None and label not in flow_filter:
                continue
            bus_label = str(topo_bus[label])
            comp_label = str(topo_comp[label])
            carrier_name = str(topo_carrier[label])
            if bus_filter is not None and bus_label not in bus_filter:
                continue
            if carrier_filter is not None and carrier_name.lower() not in carrier_filter:
                continue
            if component_filter is not None and comp_label not in component_filter:
                continue

            value = float(ds[label].values)
            if abs(value) < min_value:
                continue

            if topo_is_input[label]:
                source, target = bus_label, comp_label
            else:
                source, target = comp_label, bus_label

            nodes.add(source)
            nodes.add(target)
            links['source'].append(source)
            links['target'].append(target)
            links['value'].append(abs(value))
            links['label'].append(label)
            links['carrier'].append(carrier_name)

        return nodes, links

    def _create_figure(
        self,
        nodes: set[str],
        links: dict[str, list],
        colors: ColorType | None,
        title: str,
        **plotly_kwargs: Any,
    ) -> go.Figure:
        """Create Plotly Sankey figure."""
        node_list = list(nodes)
        node_indices = {n: i for i, n in enumerate(node_list)}

        # Build node colors: buses use carrier colors, components use process_colors
        node_colors = self._get_node_colors(node_list, colors)

        # Build link colors from carrier colors (subtle/semi-transparent)
        link_colors = self._get_link_colors(links.get('carrier', []))

        link_dict: dict[str, Any] = dict(
            source=[node_indices[s] for s in links['source']],
            target=[node_indices[t] for t in links['target']],
            value=links['value'],
            label=links['label'],
        )
        if link_colors:
            link_dict['color'] = link_colors

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15, thickness=20, line=dict(color='black', width=0.5), label=node_list, color=node_colors
                    ),
                    link=link_dict,
                )
            ]
        )
        fig.update_layout(title=title, **plotly_kwargs)
        return fig

    def _get_node_colors(self, node_list: list[str], colors: ColorType | None) -> list[str]:
        """Get colors for nodes: buses use bus_colors, components use component_colors."""
        # Get cached colors
        bus_colors = self._stats.bus_colors
        component_colors = self._stats.component_colors

        # Get fallback colors for nodes without explicit colors
        uncolored = [n for n in node_list if n not in bus_colors and n not in component_colors]
        fallback_colors = process_colors(colors, uncolored) if uncolored else {}

        node_colors = []
        for node in node_list:
            if node in bus_colors:
                node_colors.append(bus_colors[node])
            elif node in component_colors:
                node_colors.append(component_colors[node])
            else:
                node_colors.append(fallback_colors[node])

        return node_colors

    def _get_link_colors(self, carriers: list[str | None]) -> list[str]:
        """Get subtle/semi-transparent colors for links based on their carriers."""
        if not carriers:
            return []

        # Use cached carrier colors for efficiency
        carrier_colors = self._stats.carrier_colors

        link_colors = []
        for carrier_name in carriers:
            hex_color = carrier_colors.get(carrier_name.lower()) if carrier_name else None
            link_colors.append(hex_to_rgba(hex_color, alpha=0.4) if hex_color else hex_to_rgba('', alpha=0.4))

        return link_colors

    def _finalize(self, fig: go.Figure, links: dict[str, list], show: bool | None) -> PlotResult:
        """Create PlotResult and optionally show figure."""
        coords: dict[str, Any] = {
            'link': range(len(links['value'])),
            'source': ('link', links['source']),
            'target': ('link', links['target']),
            'label': ('link', links['label']),
        }
        # Add carrier if present
        if 'carrier' in links:
            coords['carrier'] = ('link', links['carrier'])

        sankey_da = xr.DataArray(links['value'], dims=['link'], coords=coords)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=sankey_da, figure=fig)

    def flows(
        self,
        *,
        aggregate: Literal['sum', 'mean'] = 'sum',
        select: FlowSankeySelect | None = None,
        colors: ColorType | None = None,
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot Sankey diagram of energy/material flow amounts.

        Args:
            aggregate: How to aggregate over time ('sum' or 'mean').
            select: Filter options:
                - flow: filter by flow label (e.g., 'Boiler|Q_th')
                - bus: filter by bus label (e.g., 'HeatBus')
                - component: filter by component label (e.g., 'Boiler')
                - time: select specific time (e.g., 100 or '2023-01-01')
                - period, scenario: xarray dimension selection
            colors: Color specification for nodes.
            show: Whether to display the figure.
            **plotly_kwargs: Additional arguments passed to Plotly layout.

        Returns:
            PlotResult with Sankey flow data and figure.
        """
        self._stats._require_solution()
        xr_select, flow_filter, bus_filter, component_filter, carrier_filter = self._extract_flow_filters(select)

        da = self._stats.flow_hours.copy()

        # Apply period/scenario weights
        if 'period' in da.dims and self._fs.period_weights is not None:
            da = da * self._fs.period_weights
        if 'scenario' in da.dims and self._fs.scenario_weights is not None:
            weights = self._fs.scenario_weights / self._fs.scenario_weights.sum()
            da = da * weights

        da = _apply_selection(da, xr_select)

        # Aggregate remaining dimensions
        if 'time' in da.dims:
            da = getattr(da, aggregate)(dim='time')
        for dim in ['period', 'scenario']:
            if dim in da.dims:
                da = da.sum(dim=dim)

        # Convert to Dataset for _build_flow_links
        ds = da.to_dataset('flow')
        nodes, links = self._build_flow_links(ds, flow_filter, bus_filter, component_filter, carrier_filter)
        fig = self._create_figure(nodes, links, colors, 'Energy Flow', **plotly_kwargs)
        return self._finalize(fig, links, show)

    def sizes(
        self,
        *,
        select: FlowSankeySelect | None = None,
        max_size: float | None = None,
        colors: ColorType | None = None,
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot Sankey diagram of investment sizes/capacities.

        Args:
            select: Filter options:
                - flow: filter by flow label (e.g., 'Boiler|Q_th')
                - bus: filter by bus label (e.g., 'HeatBus')
                - component: filter by component label (e.g., 'Boiler')
                - period, scenario: xarray dimension selection
            max_size: Filter flows with sizes exceeding this value.
            colors: Color specification for nodes.
            show: Whether to display the figure.
            **plotly_kwargs: Additional arguments passed to Plotly layout.

        Returns:
            PlotResult with Sankey size data and figure.
        """
        self._stats._require_solution()
        xr_select, flow_filter, bus_filter, component_filter, carrier_filter = self._extract_flow_filters(select)

        # Use flow_sizes (DataArray with 'flow' dim) for Sankey - storage sizes not applicable
        da = self._stats.flow_sizes.copy()
        da = _apply_selection(da, xr_select)

        # Collapse remaining dimensions
        for dim in ['period', 'scenario']:
            if dim in da.dims:
                da = da.max(dim=dim)

        # Convert to Dataset for _build_flow_links
        ds = da.to_dataset('flow')

        # Apply max_size filter
        if max_size is not None and ds.data_vars:
            valid_labels = [lbl for lbl in ds.data_vars if float(ds[lbl].max()) < max_size]
            ds = ds[valid_labels]

        nodes, links = self._build_flow_links(ds, flow_filter, bus_filter, component_filter, carrier_filter)
        fig = self._create_figure(nodes, links, colors, 'Investment Sizes (Capacities)', **plotly_kwargs)
        return self._finalize(fig, links, show)

    def peak_flow(
        self,
        *,
        select: FlowSankeySelect | None = None,
        colors: ColorType | None = None,
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot Sankey diagram of peak (maximum) flow rates.

        Args:
            select: Filter options:
                - flow: filter by flow label (e.g., 'Boiler|Q_th')
                - bus: filter by bus label (e.g., 'HeatBus')
                - component: filter by component label (e.g., 'Boiler')
                - time, period, scenario: xarray dimension selection
            colors: Color specification for nodes.
            show: Whether to display the figure.
            **plotly_kwargs: Additional arguments passed to Plotly layout.

        Returns:
            PlotResult with Sankey peak flow data and figure.
        """
        self._stats._require_solution()
        xr_select, flow_filter, bus_filter, component_filter, carrier_filter = self._extract_flow_filters(select)

        da = self._stats.flow_rates.copy()
        da = _apply_selection(da, xr_select)

        # Take max over all dimensions
        for dim in ['time', 'period', 'scenario']:
            if dim in da.dims:
                da = da.max(dim=dim)

        # Convert to Dataset for _build_flow_links
        ds = da.to_dataset('flow')
        nodes, links = self._build_flow_links(ds, flow_filter, bus_filter, component_filter, carrier_filter)
        fig = self._create_figure(nodes, links, colors, 'Peak Flow Rates', **plotly_kwargs)
        return self._finalize(fig, links, show)

    def effects(
        self,
        *,
        select: EffectsSankeySelect | None = None,
        colors: ColorType | None = None,
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot Sankey diagram of component contributions to effects.

        Shows how each component contributes to costs, CO2, and other effects.

        Args:
            select: Filter options:
                - effect: filter which effects are shown (e.g., 'costs', ['costs', 'CO2'])
                - component: filter by component label (e.g., 'Boiler')
                - contributor: filter by contributor label (e.g., 'Boiler|Q_th')
                - period, scenario: xarray dimension selection
            colors: Color specification for nodes.
            show: Whether to display the figure.
            **plotly_kwargs: Additional arguments passed to Plotly layout.

        Returns:
            PlotResult with Sankey effects data and figure.
        """
        self._stats._require_solution()
        total_effects = self._stats.total_effects

        # Extract special filters from select
        effect_filter: list[str] | None = None
        component_filter: list[str] | None = None
        contributor_filter: list[str] | None = None
        xr_select: SelectType | None = None

        if select is not None:
            select = dict(select)  # Copy to avoid mutating
            effect_filter = select.pop('effect', None)
            component_filter = select.pop('component', None)
            contributor_filter = select.pop('contributor', None)
            xr_select = select if select else None

            # Normalize to lists
            if isinstance(effect_filter, str):
                effect_filter = [effect_filter]
            if isinstance(component_filter, str):
                component_filter = [component_filter]
            if isinstance(contributor_filter, str):
                contributor_filter = [contributor_filter]

        # Determine which effects to include
        effect_names = list(str(e) for e in total_effects.coords['effect'].values)
        if effect_filter is not None:
            effect_names = [e for e in effect_names if e in effect_filter]

        # Collect all links: component -> effect
        nodes: set[str] = set()
        links: dict[str, list] = {'source': [], 'target': [], 'value': [], 'label': []}

        for effect_name in effect_names:
            effect_data = total_effects.sel(effect=effect_name, drop=True)
            effect_data = _apply_selection(effect_data, xr_select)

            # Sum over remaining dimensions
            for dim in ['period', 'scenario']:
                if dim in effect_data.dims:
                    effect_data = effect_data.sum(dim=dim)

            contributors = effect_data.coords['contributor'].values
            components = effect_data.coords['component'].values

            for contributor, component in zip(contributors, components, strict=False):
                if component_filter is not None and component not in component_filter:
                    continue
                if contributor_filter is not None and contributor not in contributor_filter:
                    continue

                value = float(effect_data.sel(contributor=contributor).values)
                if not np.isfinite(value) or abs(value) < 1e-6:
                    continue

                source = str(component)
                target = f'[{effect_name}]'

                nodes.add(source)
                nodes.add(target)
                links['source'].append(source)
                links['target'].append(target)
                links['value'].append(abs(value))
                links['label'].append(f'{contributor} → {effect_name}: {value:.2f}')

        fig = self._create_figure(nodes, links, colors, 'Effect Contributions by Component', **plotly_kwargs)
        return self._finalize(fig, links, show)


# --- Statistics Plot Accessor ---


class StatisticsPlotAccessor:
    """Plot accessor for statistics. Access via ``flow_system.stats.plot``.

    All methods return PlotResult with both data and figure.
    """

    def __init__(self, statistics: StatisticsAccessor) -> None:
        self._stats = statistics
        self._fs = statistics._fs
        self._sankey: SankeyPlotAccessor | None = None

    def _get_unit_label(self, flow_label: str) -> str:
        """Get the unit label for a flow from topology."""
        topo_flows = self._fs.topology.flows
        if flow_label in topo_flows.coords['flow'].values:
            return str(topo_flows.sel(flow=flow_label).coords['unit'].values)
        return ''

    @property
    def sankey(self) -> SankeyPlotAccessor:
        """Access sankey diagram methods with typed select options.

        Returns:
            SankeyPlotAccessor with methods: flows(), sizes(), peak_flow(), effects()

        Examples:
            >>> fs.stats.plot.sankey.flows(select={'bus': 'HeatBus'})
            >>> fs.stats.plot.sankey.effects(select={'effect': 'costs'})
        """
        if self._sankey is None:
            self._sankey = SankeyPlotAccessor(self)
        return self._sankey

    def _get_smart_color_defaults(
        self,
        labels: list[str],
        color_by: Literal['component', 'carrier'] = 'component',
    ) -> dict[str, str]:
        """Build smart color defaults for labels.

        Args:
            labels: Variable or flow labels.
            color_by: 'component' for component colors, 'carrier' for carrier colors.

        Returns:
            Dict mapping labels to hex colors. Uncolored labels get fallback colors.
        """
        component_colors = self._stats.component_colors
        carrier_colors = self._stats.carrier_colors

        # Extract topology metadata as dicts for fast lookup
        topo = self._fs.topology.flows
        topo_carriers = dict(zip(topo.coords['flow'].values, topo.coords['carrier'].values, strict=False))
        topo_components = dict(zip(topo.coords['flow'].values, topo.coords['component'].values, strict=False))

        color_map = {}
        uncolored = []

        for label in labels:
            color = None

            if color_by == 'carrier':
                carrier_name = topo_carriers.get(label)
                color = carrier_colors.get(str(carrier_name)) if carrier_name is not None else None
            else:  # color_by == 'component'
                comp_name = topo_components.get(label)
                if comp_name is not None:
                    color = component_colors.get(str(comp_name))
                else:
                    # Extract component name from label (non-flow labels like effect contributors)
                    comp_name = label.split('(')[0].strip() if '(' in label else label
                    color = component_colors.get(comp_name)

            if color:
                color_map[label] = color
            else:
                uncolored.append(label)

        if uncolored:
            color_map.update(process_colors(None, uncolored))

        return color_map

    def _build_color_kwargs(
        self,
        colors: ColorType | None,
        labels: list[str],
        color_by: Literal['component', 'carrier'] = 'component',
    ) -> dict[str, Any]:
        """Build color kwargs with smart defaults.

        Args:
            colors: User-provided colors (dict, list, str colorscale, or None).
            labels: Variable labels for color mapping.
            color_by: 'component' for component colors, 'carrier' for carrier colors.

        Returns:
            Dict with 'color_discrete_map' or 'color_discrete_sequence'.
        """
        smart_defaults = self._get_smart_color_defaults(labels, color_by)
        return _merge_color_kwargs(colors, labels, smart_defaults)

    def _resolve_variable_names(self, variables: list[str], solution: xr.Dataset) -> list[str]:
        """Resolve flow labels to variable names with fallback.

        For each variable:
        1. First check if it exists in the dataset as-is
        2. If not found and doesn't contain '|', try adding '|flow_rate' suffix
        3. If still not found, try '|charge_state' suffix (for storages)

        Args:
            variables: List of flow labels or variable names.
            solution: The solution dataset to check variable existence.

        Returns:
            List of resolved variable names.
        """
        resolved = []
        for var in variables:
            if var in solution:
                # Variable exists as-is, use it directly
                resolved.append(var)
            elif '|' not in var:
                # Not found and no '|', try common suffixes
                flow_rate_var = f'{var}|flow_rate'
                charge_state_var = f'{var}|charge_state'
                if flow_rate_var in solution:
                    resolved.append(flow_rate_var)
                elif charge_state_var in solution:
                    resolved.append(charge_state_var)
                else:
                    # Let it fail with the original name for clear error message
                    resolved.append(var)
            else:
                # Contains '|' but not in solution - let it fail with original name
                resolved.append(var)
        return resolved

    def balance(
        self,
        node: str,
        *,
        select: SelectType | None = None,
        include: FilterType | None = None,
        exclude: FilterType | None = None,
        unit: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        colors: ColorType | None = None,
        round_decimals: int | None = 6,
        threshold: float | None = 1e-5,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot node balance (inputs vs outputs) for a Bus or Component.

        Args:
            node: Label of the Bus or Component to plot.
            select: xarray-style selection dict.
            include: Only include flows with these exact labels.
            exclude: Exclude flows with these exact labels.
            unit: 'flow_rate' (power) or 'flow_hours' (energy).
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            round_decimals: Round values to this many decimal places to avoid numerical noise
                (e.g., tiny negative values from solver precision). Set to None to disable.
            threshold: Filter out variables where max absolute value is below this.
                Useful for removing solver noise. Set to None to disable.
            show: Whether to display the plot.
            data_only: If True, skip figure creation and return only data (for performance).
            **plotly_kwargs: Additional arguments passed to the plotly accessor (e.g.,
                facet_col, facet_row, animation_frame).

        Returns:
            PlotResult with .data and .figure.
        """
        self._stats._require_solution()

        # Get the element
        if node in self._fs.buses:
            element = self._fs.buses[node]
            is_bus = True
        elif node in self._fs.components:
            element = self._fs.components[node]
            is_bus = False
        else:
            raise KeyError(f"'{node}' not found in buses or components")

        input_labels = [f.id for f in element.inputs.values()]
        output_labels = [f.id for f in element.outputs.values()]
        all_labels = input_labels + output_labels

        filtered_labels = _filter_by_labels(all_labels, include, exclude)
        if not filtered_labels:
            logger.warning(f'No flows remaining after filtering for node {node}')
            return PlotResult(data=xr.DataArray(), figure=go.Figure())

        # Get data from statistics (DataArray with 'flow' dimension)
        source_da = self._stats.flow_rates if unit == 'flow_rate' else self._stats.flow_hours
        available = [lbl for lbl in filtered_labels if lbl in source_da.coords['flow'].values]
        da = source_da.sel(flow=available)

        # Negate inputs: create sign array
        signs = xr.DataArray(
            [(-1 if lbl in input_labels else 1) for lbl in available],
            dims=['flow'],
            coords={'flow': available},
        )
        da = da * signs

        da = _apply_selection(da, select)

        # Round to avoid numerical noise (tiny negative values from solver precision)
        if round_decimals is not None:
            da = da.round(round_decimals)

        # Filter out flows below threshold
        da = _filter_small_dataarray(da, 'flow', threshold)

        # Build color kwargs: bus balance → component colors, component balance → carrier colors
        labels = list(str(f) for f in da.coords['flow'].values)
        color_by: Literal['component', 'carrier'] = 'component' if is_bus else 'carrier'
        color_kwargs = self._build_color_kwargs(colors, labels, color_by)

        # Early return for data_only mode (skip figure creation for performance)
        if data_only:
            return PlotResult(data=da, figure=go.Figure())

        # Sort for consistent plotting order
        da = _sort_dataarray(da, 'flow')

        # Get unit label from topology
        unit_label = self._get_unit_label(available[0]) if available else ''

        _apply_slot_defaults(plotly_kwargs, 'balance')
        fig = da.plotly.fast_bar(
            title=f'{node} [{unit_label}]' if unit_label else node,
            **color_kwargs,
            **plotly_kwargs,
        )
        _apply_unified_hover(fig, unit=unit_label)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=da, figure=fig)

    def carrier_balance(
        self,
        carrier: str,
        *,
        select: SelectType | None = None,
        include: FilterType | None = None,
        exclude: FilterType | None = None,
        unit: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        colors: ColorType | None = None,
        round_decimals: int | None = 6,
        threshold: float | None = 1e-5,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot carrier-level balance showing all flows of a carrier type.

        Shows production (positive) and consumption (negative) of a carrier
        across all buses of that carrier type in the system.

        Args:
            carrier: Carrier name (e.g., 'heat', 'electricity', 'gas').
            select: xarray-style selection dict.
            include: Only include flows with these exact labels.
            exclude: Exclude flows with these exact labels.
            unit: 'flow_rate' (power) or 'flow_hours' (energy).
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            round_decimals: Round values to this many decimal places to avoid numerical noise
                (e.g., tiny negative values from solver precision). Set to None to disable.
            threshold: Filter out variables where max absolute value is below this.
                Useful for removing solver noise. Set to None to disable.
            show: Whether to display the plot.
            data_only: If True, skip figure creation and return only data (for performance).
            **plotly_kwargs: Additional arguments passed to the plotly accessor (e.g.,
                facet_col, facet_row, animation_frame).

        Returns:
            PlotResult with .data and .figure.

        Examples:
            >>> fs.stats.plot.carrier_balance('heat')
            >>> fs.stats.plot.carrier_balance('electricity', unit='flow_hours')

        Notes:
            - Data is aggregated by component (not individual flows)
            - Supply (inputs to carrier buses) shown as positive
            - Demand (outputs from carrier buses) shown as negative
            - Components with both supply and demand get separate entries
              (e.g., 'Storage (supply)' and 'Storage (demand)')
        """
        self._stats._require_solution()
        carrier = carrier.lower()

        # Find all buses with this carrier
        carrier_buses = [bus for bus in self._fs.buses.values() if bus.carrier == carrier]
        if not carrier_buses:
            raise KeyError(f"No buses found with carrier '{carrier}'")

        # Collect all flows connected to these buses, grouped by component
        input_labels: list[str] = []  # Inputs to buses = production
        output_labels: list[str] = []  # Outputs from buses = consumption
        component_inputs: dict[str, list[str]] = {}  # component -> input flow labels
        component_outputs: dict[str, list[str]] = {}  # component -> output flow labels

        for bus in carrier_buses:
            for flow in bus.inputs.values():
                input_labels.append(flow.id)
                component_inputs.setdefault(flow.component, []).append(flow.id)
            for flow in bus.outputs.values():
                output_labels.append(flow.id)
                component_outputs.setdefault(flow.component, []).append(flow.id)

        all_labels = input_labels + output_labels
        filtered_labels = _filter_by_labels(all_labels, include, exclude)
        if not filtered_labels:
            logger.warning(f'No flows remaining after filtering for carrier {carrier}')
            return PlotResult(data=xr.DataArray(), figure=go.Figure())

        # Get source data (DataArray with 'flow' dimension)
        source_da = self._stats.flow_rates if unit == 'flow_rate' else self._stats.flow_hours
        available_flows = set(str(f) for f in source_da.coords['flow'].values)

        # Find components with same carrier on both sides (supply and demand)
        same_carrier_components = set(component_inputs.keys()) & set(component_outputs.keys())
        filtered_set = set(filtered_labels)

        # Aggregate by component with separate supply/demand entries
        parts: list[xr.DataArray] = []
        part_names: list[str] = []

        for comp_name, labels in component_inputs.items():
            labels = [lbl for lbl in labels if lbl in filtered_set and lbl in available_flows]
            if not labels:
                continue
            supply = source_da.sel(flow=labels).sum('flow')
            var_name = f'{comp_name} (supply)' if comp_name in same_carrier_components else comp_name
            parts.append(supply)
            part_names.append(var_name)

        for comp_name, labels in component_outputs.items():
            labels = [lbl for lbl in labels if lbl in filtered_set and lbl in available_flows]
            if not labels:
                continue
            demand = -source_da.sel(flow=labels).sum('flow')
            var_name = f'{comp_name} (demand)' if comp_name in same_carrier_components else comp_name
            parts.append(demand)
            part_names.append(var_name)

        if not parts:
            logger.warning(f'No data after aggregation for carrier {carrier}')
            return PlotResult(data=xr.DataArray(), figure=go.Figure())

        da = xr.concat(parts, dim=pd.Index(part_names, name='component'))

        da = _apply_selection(da, select)

        # Round to avoid numerical noise (tiny negative values from solver precision)
        if round_decimals is not None:
            da = da.round(round_decimals)

        # Filter out components below threshold
        da = _filter_small_dataarray(da, 'component', threshold)

        # Build color kwargs with component colors
        labels = list(str(c) for c in da.coords['component'].values)
        color_kwargs = self._build_color_kwargs(colors, labels, color_by='component')

        # Early return for data_only mode (skip figure creation for performance)
        if data_only:
            return PlotResult(data=da, figure=go.Figure())

        # Sort for consistent plotting order
        da = _sort_dataarray(da, 'component')

        # Get unit label from carrier
        unit_label = self._stats.carrier_units.get(carrier, '')

        _apply_slot_defaults(plotly_kwargs, 'carrier_balance')
        fig = da.plotly.fast_bar(
            title=f'{carrier.capitalize()} Balance [{unit_label}]' if unit_label else f'{carrier.capitalize()} Balance',
            **color_kwargs,
            **plotly_kwargs,
        )
        _apply_unified_hover(fig, unit=unit_label)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=da, figure=fig)

    def heatmap(
        self,
        variables: str | list[str],
        *,
        select: SelectType | None = None,
        reshape: tuple[str, str] | Literal['auto'] | None = ('D', 'h'),
        colors: str | list[str] | None = None,
        threshold: float | None = 1e-5,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot heatmap of time series data.

        By default, time is reshaped into days × hours for clear daily pattern visualization.
        For clustered data, the natural (cluster, time) shape is used instead.

        Multiple variables are shown as facets. If no time dimension exists, reshaping
        is skipped and data dimensions are used directly.

        Args:
            variables: Flow label(s) or variable name(s). Flow labels like 'Boiler(Q_th)'
                are automatically resolved to 'Boiler(Q_th)|flow_rate'. Full variable
                names like 'Storage|charge_state' are used as-is.
            select: xarray-style selection, e.g. {'scenario': 'Base Case'}.
            reshape: Time reshape frequencies as (outer, inner). Default ``('D', 'h')``
                reshapes into days × hours. Use None to disable reshaping and use
                data dimensions directly.
            colors: Colorscale name (str) or list of colors for heatmap coloring.
                Dicts are not supported for heatmaps (use str or list[str]).
            threshold: Filter out variables where max absolute value is below this.
                Useful for removing solver noise. Set to None to disable.
            show: Whether to display the figure.
            data_only: If True, skip figure creation and return only data (for performance).
            **plotly_kwargs: Additional arguments passed to plotly accessor (e.g.,
                facet_col, animation_frame).

        Returns:
            PlotResult with processed data and figure.
        """
        self._stats._require_solution()
        if isinstance(variables, str):
            variables = [variables]

        # Resolve variables: try flow_rates first, fall back to solution
        flow_rates = self._stats.flow_rates
        flow_labels = list(str(f) for f in flow_rates.coords['flow'].values)
        arrays = []
        for var in variables:
            if var in flow_labels:
                arrays.append(flow_rates.sel(flow=var).rename(var))
            elif var in self._fs.solution:
                arrays.append(self._fs.solution[var].rename(var))
            elif '|' not in var and f'{var}|flow_rate' in self._fs.solution:
                arrays.append(self._fs.solution[f'{var}|flow_rate'].rename(var))
            else:
                raise KeyError(f"Variable '{var}' not found in flow_rates or solution")

        da = xr.concat(arrays, dim=pd.Index([a.name for a in arrays], name='variable'))
        da = _apply_selection(da, select)

        # Filter small variables
        da = _filter_small_dataarray(da, 'variable', threshold)
        da = _sort_dataarray(da, 'variable')

        # Prepare for heatmap (reshape, transpose, squeeze)
        da = _prepare_for_heatmap(da, reshape)

        # Early return for data_only mode (skip figure creation for performance)
        if data_only:
            return PlotResult(data=da, figure=go.Figure())

        # Only pass colors if not already in plotly_kwargs (avoid duplicate arg error)
        if 'color_continuous_scale' not in plotly_kwargs:
            plotly_kwargs['color_continuous_scale'] = colors
        fig = da.plotly.imshow(**plotly_kwargs)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=da, figure=fig)

    def flows(
        self,
        *,
        start: str | list[str] | None = None,
        end: str | list[str] | None = None,
        component: str | list[str] | None = None,
        select: SelectType | None = None,
        unit: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        colors: ColorType | None = None,
        threshold: float | None = 1e-5,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot flow rates filtered by start/end nodes or component.

        Args:
            start: Filter by source node(s).
            end: Filter by destination node(s).
            component: Filter by parent component(s).
            select: xarray-style selection.
            unit: 'flow_rate' or 'flow_hours'.
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            threshold: Filter out variables where max absolute value is below this.
                Useful for removing solver noise. Set to None to disable.
            show: Whether to display.
            data_only: If True, skip figure creation and return only data (for performance).
            **plotly_kwargs: Additional arguments passed to the plotly accessor (e.g.,
                facet_col, facet_row, animation_frame).

        Returns:
            PlotResult with flow data.
        """
        self._stats._require_solution()

        source_da = self._stats.flow_rates if unit == 'flow_rate' else self._stats.flow_hours
        available_flows = set(str(f) for f in source_da.coords['flow'].values)

        # Filter by connection
        if start is not None or end is not None or component is not None:
            matching_labels = []
            starts = [start] if isinstance(start, str) else (start or [])
            ends = [end] if isinstance(end, str) else (end or [])
            components = [component] if isinstance(component, str) else (component or [])

            for flow in self._fs.flows.values():
                bus_label = flow.bus
                comp_label = flow.component

                if flow.is_input_in_component:
                    if starts and bus_label not in starts:
                        continue
                    if ends and comp_label not in ends:
                        continue
                else:
                    if starts and comp_label not in starts:
                        continue
                    if ends and bus_label not in ends:
                        continue

                if components and comp_label not in components:
                    continue
                matching_labels.append(flow.id)

            selected_flows = [lbl for lbl in matching_labels if lbl in available_flows]
            da = source_da.sel(flow=selected_flows)
        else:
            da = source_da

        da = _apply_selection(da, select)

        # Filter out flows below threshold
        da = _filter_small_dataarray(da, 'flow', threshold)

        # Early return for data_only mode (skip figure creation for performance)
        if data_only:
            return PlotResult(data=da, figure=go.Figure())

        # Sort for consistent plotting order
        da = _sort_dataarray(da, 'flow')

        # Get unit label from topology
        unit_label = ''
        if da.sizes.get('flow', 0) > 0:
            unit_label = self._get_unit_label(str(da.coords['flow'].values[0]))

        # Build color kwargs with smart defaults from component colors
        labels = list(str(f) for f in da.coords['flow'].values)
        color_kwargs = self._build_color_kwargs(colors, labels)

        _apply_slot_defaults(plotly_kwargs, 'flows')
        fig = da.plotly.line(
            title=f'Flows [{unit_label}]' if unit_label else 'Flows',
            **color_kwargs,
            **plotly_kwargs,
        )

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=da, figure=fig)

    def sizes(
        self,
        *,
        max_size: float | None = 1e6,
        select: SelectType | None = None,
        colors: ColorType | None = None,
        threshold: float | None = 1e-5,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot investment sizes (capacities) of flows.

        Args:
            max_size: Maximum size to include (filters defaults).
            select: xarray-style selection.
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            threshold: Filter out variables where max absolute value is below this.
                Useful for removing non-invested components. Set to None to disable.
            show: Whether to display.
            data_only: If True, skip figure creation and return only data (for performance).
            **plotly_kwargs: Additional arguments passed to the plotly accessor (e.g.,
                facet_col, facet_row, animation_frame).

        Returns:
            PlotResult with size data.
        """
        self._stats._require_solution()
        da = self._stats.sizes

        da = _apply_selection(da, select)

        if max_size is not None and 'element' in da.dims:
            keep = abs(da).max([d for d in da.dims if d != 'element']) < max_size
            da = da.sel(element=keep)

        # Filter out entries below threshold
        da = _filter_small_dataarray(da, 'element', threshold)

        # Early return for data_only mode (skip figure creation for performance)
        if data_only:
            return PlotResult(data=da, figure=go.Figure())

        if da.sizes.get('element', 0) == 0:
            fig = go.Figure()
        else:
            # Sort for consistent plotting order
            da = _sort_dataarray(da, 'element')
            # Build color kwargs with smart defaults from component colors
            labels = list(str(e) for e in da.coords['element'].values)
            color_kwargs = self._build_color_kwargs(colors, labels)
            _apply_slot_defaults(plotly_kwargs, 'sizes')
            fig = da.plotly.bar(
                title='Investment Sizes',
                labels={'value': 'Size'},
                **color_kwargs,
                **plotly_kwargs,
            )

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=da, figure=fig)

    def duration_curve(
        self,
        variables: str | list[str],
        *,
        select: SelectType | None = None,
        normalize: bool = False,
        colors: ColorType | None = None,
        threshold: float | None = 1e-5,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot load duration curves (sorted time series).

        Args:
            variables: Flow label(s) or variable name(s). Flow labels like 'Boiler(Q_th)'
                are looked up in flow_rates. Full variable names like 'Boiler(Q_th)|flow_rate'
                are stripped to their flow label. Other variables (e.g., 'Storage|charge_state')
                are looked up in the solution directly.
            select: xarray-style selection.
            normalize: If True, normalize x-axis to 0-100%.
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            threshold: Filter out variables where max absolute value is below this.
                Useful for removing solver noise. Set to None to disable.
            show: Whether to display.
            data_only: If True, skip figure creation and return only data (for performance).
            **plotly_kwargs: Additional arguments passed to the plotly accessor (e.g.,
                facet_col, facet_row, animation_frame).

        Returns:
            PlotResult with sorted duration curve data.
        """
        solution = self._stats._require_solution()

        if isinstance(variables, str):
            variables = [variables]

        # Normalize variable names: strip |flow_rate suffix for flow_rates lookup
        flow_rates = self._stats.flow_rates
        flow_labels = set(str(f) for f in flow_rates.coords['flow'].values)
        normalized_vars = []
        for var in variables:
            if var.endswith('|flow_rate'):
                var = var[: -len('|flow_rate')]
            normalized_vars.append(var)

        # Collect arrays, build a DataArray with 'variable' dim
        arrays = []
        for var in normalized_vars:
            if var in flow_labels:
                arrays.append(flow_rates.sel(flow=var, drop=True).rename(var))
            elif var in solution:
                arrays.append(solution[var].rename(var))
            else:
                flow_rate_var = f'{var}|flow_rate'
                if flow_rate_var in solution:
                    arrays.append(solution[flow_rate_var].rename(var))
                else:
                    raise KeyError(f"Variable '{var}' not found in flow_rates or solution")

        da = xr.concat(arrays, dim=pd.Index([a.name for a in arrays], name='variable'))
        da = _apply_selection(da, select)

        # Sort each variable's values independently (duration curve)
        sorted_arrays = []
        for var in da.coords['variable'].values:
            arr = da.sel(variable=var, drop=True)
            # Sort descending along time
            if 'time' in arr.dims:
                sorted_vals = np.flip(np.sort(arr.values, axis=arr.dims.index('time')), axis=arr.dims.index('time'))
                duration_dim = np.arange(len(arr.coords['time']))
                if normalize:
                    duration_dim = duration_dim / len(duration_dim) * 100
                arr = xr.DataArray(
                    sorted_vals,
                    dims=['duration'],
                    coords={'duration': duration_dim},
                )
            sorted_arrays.append(arr)

        result_da = xr.concat(
            sorted_arrays, dim=pd.Index(list(str(v) for v in da.coords['variable'].values), name='variable')
        )

        # Filter out variables below threshold
        result_da = _filter_small_dataarray(result_da, 'variable', threshold)

        # Early return for data_only mode (skip figure creation for performance)
        if data_only:
            return PlotResult(data=result_da, figure=go.Figure())

        # Sort for consistent plotting order
        result_da = _sort_dataarray(result_da, 'variable')

        # Get unit label from first variable's carrier
        unit_label = ''
        if normalized_vars and normalized_vars[0] in flow_labels:
            unit_label = self._get_unit_label(normalized_vars[0])

        # Build color kwargs with smart defaults from component colors
        labels = list(str(v) for v in result_da.coords['variable'].values)
        color_kwargs = self._build_color_kwargs(colors, labels)

        plotly_kwargs.setdefault('x', 'duration')
        _apply_slot_defaults(plotly_kwargs, 'duration_curve')
        fig = result_da.plotly.line(
            title=f'Duration Curve [{unit_label}]' if unit_label else 'Duration Curve',
            **color_kwargs,
            **plotly_kwargs,
        )

        x_label = 'Duration [%]' if normalize else 'Timesteps'
        fig.update_xaxes(title_text=x_label)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=result_da, figure=fig)

    def effects(
        self,
        aspect: Literal['total', 'temporal', 'periodic'] = 'total',
        *,
        effect: str | None = None,
        by: Literal['component', 'contributor', 'time'] | None = None,
        select: SelectType | None = None,
        colors: ColorType | None = None,
        threshold: float | None = 1e-5,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot effect (cost, emissions, etc.) breakdown.

        Args:
            aspect: Which aspect to plot - 'total', 'temporal', or 'periodic'.
            effect: Specific effect name to plot (e.g., 'costs', 'CO2').
                    If None, plots all effects.
            by: Group by 'component', 'contributor' (individual flows), 'time',
                or None to show aggregated totals per effect.
            select: xarray-style selection.
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            threshold: Filter out variables where max absolute value is below this.
                Useful for removing solver noise. Set to None to disable.
            show: Whether to display.
            data_only: If True, skip figure creation and return only data (for performance).
            **plotly_kwargs: Additional arguments passed to the plotly accessor (e.g.,
                facet_col, facet_row, animation_frame).

        Returns:
            PlotResult with effect breakdown data.

        Examples:
            >>> flow_system.stats.plot.effects()  # Aggregated totals per effect
            >>> flow_system.stats.plot.effects(effect='costs')  # Just costs
            >>> flow_system.stats.plot.effects(by='component')  # Breakdown by component
            >>> flow_system.stats.plot.effects(by='contributor')  # By individual flows
            >>> flow_system.stats.plot.effects(aspect='temporal', by='time')  # Over time
        """
        self._stats._require_solution()

        # Get the appropriate effects DataArray based on aspect
        effects_da: xr.DataArray | None = {
            'total': self._stats.total_effects,
            'temporal': self._stats.temporal_effects,
            'periodic': self._stats.periodic_effects,
        }.get(aspect)
        if effects_da is None:
            raise ValueError(f"Aspect '{aspect}' not valid. Choose from 'total', 'temporal', 'periodic'.")

        # Filter to specific effect(s)
        if effect is not None:
            effect_names = list(str(e) for e in effects_da.coords['effect'].values)
            if effect not in effect_names:
                raise ValueError(f"Effect '{effect}' not found. Available: {effect_names}")
            da = effects_da.sel(effect=effect, drop=True)
        else:
            da = effects_da

        # Group by component (default) unless by='contributor'
        if by != 'contributor' and 'contributor' in da.dims:
            da = da.groupby('component').sum()

        da = _apply_selection(da, select)

        has_effect_dim = 'effect' in da.dims

        # Sum over dimensions based on 'by' parameter
        if by is None:
            for dim in ['time', 'component', 'contributor']:
                if dim in da.dims:
                    da = da.sum(dim=dim)
            x_col = 'effect' if has_effect_dim else None
            color_col = 'effect' if has_effect_dim else None
        elif by == 'component':
            if 'time' in da.dims:
                da = da.sum(dim='time')
            x_col = 'component'
            color_col = 'effect' if has_effect_dim else 'component'
        elif by == 'contributor':
            if 'time' in da.dims:
                da = da.sum(dim='time')
            x_col = 'contributor'
            color_col = 'effect' if has_effect_dim else 'contributor'
        elif by == 'time':
            if 'time' not in da.dims:
                raise ValueError(f"Cannot plot by 'time' for aspect '{aspect}' - no time dimension.")
            for dim in ['component', 'contributor']:
                if dim in da.dims:
                    da = da.sum(dim=dim)
            x_col = 'time'
            color_col = 'effect' if has_effect_dim else None
        else:
            raise ValueError(f"'by' must be one of 'component', 'contributor', 'time', or None, got {by!r}")

        # Filter along the color/grouping dimension
        filter_dim = color_col or x_col
        if filter_dim and filter_dim in da.dims:
            da = _filter_small_dataarray(da, filter_dim, threshold)

        # Early return for data_only mode (skip figure creation for performance)
        if data_only:
            return PlotResult(data=da, figure=go.Figure())

        # Sort for consistent plotting order
        if filter_dim and filter_dim in da.dims:
            da = _sort_dataarray(da, filter_dim)

        # Build title
        effect_label = effect or 'Effects'
        title = f'{effect_label} ({aspect})' if by is None else f'{effect_label} ({aspect}) by {by}'

        # Allow user override of color via plotly_kwargs
        color = plotly_kwargs.pop('color', color_col)

        # Build color kwargs
        if color and color in da.dims:
            labels = list(str(v) for v in da.coords[color].values)
        elif color and color in da.coords:
            labels = list(str(v) for v in da.coords[color].values)
        else:
            labels = []
        color_kwargs = self._build_color_kwargs(colors, labels) if labels else {}

        plotly_kwargs.setdefault('x', x_col)
        _apply_slot_defaults(plotly_kwargs, 'effects')
        fig = da.plotly.bar(
            color=color,
            title=title,
            **color_kwargs,
            **plotly_kwargs,
        )
        fig.update_layout(bargap=0, bargroupgap=0)
        fig.update_traces(marker_line_width=0)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=da, figure=fig)

    def charge_states(
        self,
        storages: str | list[str] | None = None,
        *,
        select: SelectType | None = None,
        colors: ColorType | None = None,
        threshold: float | None = 1e-5,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot storage charge states over time.

        Args:
            storages: Storage label(s) to plot. If None, plots all storages.
            select: xarray-style selection.
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            threshold: Filter out variables where max absolute value is below this.
                Useful for removing non-invested storages. Set to None to disable.
            show: Whether to display.
            data_only: If True, skip figure creation and return only data (for performance).
            **plotly_kwargs: Additional arguments passed to the plotly accessor (e.g.,
                facet_col, facet_row, animation_frame).

        Returns:
            PlotResult with charge state data.
        """
        self._stats._require_solution()
        da = self._stats.charge_states

        if storages is not None:
            if isinstance(storages, str):
                storages = [storages]
            available = [s for s in storages if s in da.coords['storage'].values]
            da = da.sel(storage=available)

        da = _apply_selection(da, select)

        # Filter out storages below threshold
        da = _filter_small_dataarray(da, 'storage', threshold)

        # Early return for data_only mode (skip figure creation for performance)
        if data_only:
            return PlotResult(data=da, figure=go.Figure())

        # Sort for consistent plotting order
        da = _sort_dataarray(da, 'storage')

        # Build color kwargs with smart defaults from component colors
        labels = list(str(s) for s in da.coords['storage'].values)
        color_kwargs = self._build_color_kwargs(colors, labels)

        _apply_slot_defaults(plotly_kwargs, 'charge_states')
        fig = da.plotly.line(
            title='Storage Charge States',
            **color_kwargs,
            **plotly_kwargs,
        )
        fig.update_yaxes(title_text='Charge State')

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=da, figure=fig)

    def storage(
        self,
        storage: str,
        *,
        select: SelectType | None = None,
        unit: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        colors: ColorType | None = None,
        charge_state_color: str = 'black',
        round_decimals: int | None = 6,
        threshold: float | None = 1e-5,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot storage operation: balance and charge state in vertically stacked subplots.

        Creates two subplots sharing the x-axis:
        - Top: Charging/discharging flows as stacked bars (inputs negative, outputs positive)
        - Bottom: Charge state over time as a line

        Args:
            storage: Storage component label.
            select: xarray-style selection.
            unit: 'flow_rate' (power) or 'flow_hours' (energy).
            colors: Color specification for flow bars.
            charge_state_color: Color for the charge state line overlay.
            round_decimals: Round values to this many decimal places to avoid numerical noise
                (e.g., tiny negative values from solver precision). Set to None to disable.
            threshold: Filter out flow variables where max absolute value is below this.
                Useful for removing solver noise. Set to None to disable.
            show: Whether to display.
            data_only: If True, skip figure creation and return only data (for performance).
            **plotly_kwargs: Additional arguments passed to the plotly accessor (e.g.,
                facet_col, facet_row, animation_frame).

        Returns:
            PlotResult with combined balance and charge state data.

        Raises:
            KeyError: If storage component not found.
            ValueError: If component is not a storage.
        """
        self._stats._require_solution()

        # Get the storage component
        if storage not in self._fs.components:
            raise KeyError(f"'{storage}' not found in components")

        component = self._fs.components[storage]

        # Check if it's a storage by looking for charge_state variable
        charge_state_var = f'{storage}|charge_state'
        if charge_state_var not in self._fs.solution:
            raise ValueError(f"'{storage}' is not a storage (no charge_state variable found)")

        # Get flow data
        input_labels = [f.id for f in component.flows.values() if f.is_input_in_component]
        output_labels = [f.id for f in component.flows.values() if not f.is_input_in_component]
        all_labels = input_labels + output_labels

        source_da = self._stats.flow_rates if unit == 'flow_rate' else self._stats.flow_hours
        available_flows = set(str(f) for f in source_da.coords['flow'].values)
        available = [lbl for lbl in all_labels if lbl in available_flows]
        flow_da = source_da.sel(flow=available)

        # Negate outputs for balance view (discharging shown as negative)
        signs = xr.DataArray(
            [(-1 if lbl in output_labels else 1) for lbl in available],
            dims=['flow'],
            coords={'flow': available},
        )
        flow_da = flow_da * signs

        # Get charge state
        charge_da = self._fs.solution[charge_state_var]

        # Apply selection
        flow_da = _apply_selection(flow_da, select)
        charge_da = _apply_selection(charge_da, select)

        # Round to avoid numerical noise (tiny negative values from solver precision)
        if round_decimals is not None:
            flow_da = flow_da.round(round_decimals)

        # Filter out flow variables below threshold
        flow_da = _filter_small_dataarray(flow_da, 'flow', threshold)

        # Early return for data_only mode (skip figure creation for performance)
        if data_only:
            return PlotResult(data=flow_da, figure=go.Figure())

        # Sort for consistent plotting order
        flow_da = _sort_dataarray(flow_da, 'flow')

        # Build color kwargs with carrier colors (storage is a component, flows colored by carrier)
        labels = list(str(f) for f in flow_da.coords['flow'].values)
        color_kwargs = self._build_color_kwargs(colors, labels, color_by='carrier')

        # Get unit label from topology
        unit_label = self._get_unit_label(available[0]) if available else ''

        # Create stacked area chart for flows (styled as bar)
        _apply_slot_defaults(plotly_kwargs, 'storage')
        fig = flow_da.plotly.fast_bar(
            title=f'{storage} Operation [{unit_label}]' if unit_label else f'{storage} Operation',
            **color_kwargs,
            **plotly_kwargs,
        )
        _apply_unified_hover(fig, unit=unit_label)

        # Add charge state as line on secondary y-axis
        line_kwargs = {k: v for k, v in plotly_kwargs.items() if k not in ('pattern_shape', 'color')}
        _apply_slot_defaults(line_kwargs, 'storage_line')
        line_fig = charge_da.plotly.line(**line_kwargs)
        update_traces(
            line_fig,
            line=dict(color=charge_state_color, width=2),
            name='charge_state',
            legendgroup='charge_state',
            showlegend=False,
        )
        if line_fig.data:
            line_fig.data[0].showlegend = True
        fig = add_secondary_y(fig, line_fig, secondary_y_title='Charge State')

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=flow_da, figure=fig)
