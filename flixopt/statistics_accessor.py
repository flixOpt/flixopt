"""Statistics accessor for FlowSystem.

This module provides a user-friendly API for analyzing optimization results
directly from a FlowSystem.

Structure:
    - `.statistics` - Data/metrics access (cached xarray Datasets)
    - `.statistics.plot` - Plotting methods using the statistics data

Example:
    >>> flow_system.optimize(solver)
    >>> # Data access
    >>> flow_system.statistics.flow_rates
    >>> flow_system.statistics.flow_hours
    >>> # Plotting
    >>> flow_system.statistics.plot.balance('ElectricityBus')
    >>> flow_system.statistics.plot.heatmap('Boiler|on')
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from xarray_plotly.figures import update_traces

from .color_processing import ColorType, hex_to_rgba, process_colors
from .config import CONFIG
from .plot_result import PlotResult
from .structure import VariableCategory

if TYPE_CHECKING:
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')

# Type aliases
SelectType = dict[str, Any]
"""xarray-style selection dict: {'time': slice(...), 'scenario': 'base'}"""

FilterType = str | list[str]
"""For include/exclude filtering: 'Boiler' or ['Boiler', 'CHP']"""


# Sankey select types with Literal keys for IDE autocomplete
FlowSankeySelect = dict[Literal['flow', 'bus', 'component', 'carrier', 'time', 'period', 'scenario'], Any]
"""Select options for flow-based sankey: flow, bus, component, carrier, time, period, scenario."""

EffectsSankeySelect = dict[Literal['effect', 'component', 'contributor', 'period', 'scenario'], Any]
"""Select options for effects sankey: effect, component, contributor, period, scenario."""


# Default slot assignments for plotting methods
# Use None for slots that should be blocked (prevent auto-assignment)
_SLOT_DEFAULTS: dict[str, dict[str, str | None]] = {
    'balance': {'x': 'time', 'color': 'variable', 'pattern_shape': None},
    'carrier_balance': {'x': 'time', 'color': 'variable', 'pattern_shape': None},
    'flows': {'x': 'time', 'color': 'variable', 'symbol': None},
    'charge_states': {'x': 'time', 'color': 'variable', 'symbol': None},
    'storage': {'x': 'time', 'color': 'variable', 'pattern_shape': None},
    'sizes': {'x': 'variable', 'color': 'variable'},
    'duration_curve': {'symbol': None},  # x is computed dynamically
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


def _filter_by_pattern(
    names: list[str],
    include: FilterType | None,
    exclude: FilterType | None,
) -> list[str]:
    """Filter names using substring matching."""
    result = names.copy()
    if include is not None:
        patterns = [include] if isinstance(include, str) else include
        result = [n for n in result if any(p in n for p in patterns)]
    if exclude is not None:
        patterns = [exclude] if isinstance(exclude, str) else exclude
        result = [n for n in result if not any(p in n for p in patterns)]
    return result


def _apply_selection(ds: xr.Dataset, select: SelectType | None, drop: bool = True) -> xr.Dataset:
    """Apply xarray-style selection to dataset.

    Args:
        ds: Dataset to select from.
        select: xarray-style selection dict.
        drop: If True (default), drop dimensions that become scalar after selection.
            This prevents auto-faceting when selecting a single value.
    """
    if select is None:
        return ds
    valid_select = {k: v for k, v in select.items() if k in ds.dims or k in ds.coords}
    if valid_select:
        ds = ds.sel(valid_select, drop=drop)
    return ds


def add_line_overlay(
    fig: go.Figure,
    da: xr.DataArray,
    *,
    x: str | None = None,
    facet_col: str | None = None,
    facet_row: str | None = None,
    animation_frame: str | None = None,
    color: str | None = None,
    line_color: str = 'black',
    name: str | None = None,
    secondary_y: bool = False,
    y_title: str | None = None,
    showlegend: bool = True,
) -> None:
    """Add line traces on top of existing figure, optionally on secondary y-axis.

    This function creates line traces from a DataArray and adds them to an existing
    figure. When using secondary_y=True, it correctly handles faceted figures by
    creating matching secondary axes for each primary axis.

    Args:
        fig: Plotly figure to add traces to.
        da: DataArray to plot as lines.
        x: Dimension to use for x-axis. If None, auto-detects 'time' or first dim.
        facet_col: Dimension for column facets (must match primary figure).
        facet_row: Dimension for row facets (must match primary figure).
        animation_frame: Dimension for animation slider (must match primary figure).
        color: Dimension to color by (creates multiple lines).
        line_color: Color for lines when color is None.
        name: Legend name for the traces.
        secondary_y: If True, plot on secondary y-axis.
        y_title: Title for the y-axis (secondary if secondary_y=True).
        showlegend: Whether to show legend entries.
    """
    if da.size == 0:
        return

    # Auto-detect x dimension if not specified
    if x is None:
        x = 'time' if 'time' in da.dims else da.dims[0]

    # Build kwargs for line plot, only passing facet params if specified
    line_kwargs: dict[str, Any] = {'x': x}
    if color is not None:
        line_kwargs['color'] = color
    if facet_col is not None:
        line_kwargs['facet_col'] = facet_col
    if facet_row is not None:
        line_kwargs['facet_row'] = facet_row
    if animation_frame is not None:
        line_kwargs['animation_frame'] = animation_frame

    # Create line figure with same facets
    line_fig = da.plotly.line(**line_kwargs)

    if secondary_y:
        # Get the primary y-axes from the bar figure to create matching secondary axes
        primary_yaxes = [key for key in fig.layout if key.startswith('yaxis')]

        # For each primary y-axis, create a secondary y-axis.
        # Secondary axis numbering strategy:
        # - Primary axes are named 'yaxis', 'yaxis2', 'yaxis3', etc.
        # - We use +100 offset (yaxis101, yaxis102, ...) to avoid conflicts
        # - Each secondary axis 'overlays' its corresponding primary axis
        for i, primary_key in enumerate(sorted(primary_yaxes, key=lambda x: int(x[5:]) if x[5:] else 0)):
            primary_num = primary_key[5:] if primary_key[5:] else '1'
            secondary_num = int(primary_num) + 100
            secondary_key = f'yaxis{secondary_num}'
            secondary_anchor = f'x{primary_num}' if primary_num != '1' else 'x'

            fig.layout[secondary_key] = dict(
                overlaying=f'y{primary_num}' if primary_num != '1' else 'y',
                side='right',
                showgrid=False,
                title=y_title if i == len(primary_yaxes) - 1 else None,
                anchor=secondary_anchor,
            )

    # Add line traces with correct axis assignments
    for i, trace in enumerate(line_fig.data):
        if name is not None:
            trace.name = name
        if color is None:
            trace.line = dict(color=line_color, width=2)

        if secondary_y:
            primary_num = i + 1 if i > 0 else 1
            trace.yaxis = f'y{primary_num + 100}'

        trace.showlegend = showlegend and (i == 0)
        if name is not None:
            trace.legendgroup = name
        fig.add_trace(trace)


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
    """Build color kwargs for plotly based on color type.

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


# --- Statistics Accessor (data only) ---


class StatisticsAccessor:
    """Statistics accessor for FlowSystem. Access via ``flow_system.statistics``.

    This accessor provides cached data properties for optimization results.
    Use ``.plot`` for visualization methods.

    Data Properties:
        ``flow_rates`` : xr.Dataset
            Flow rates for all flows.
        ``flow_hours`` : xr.Dataset
            Flow hours (energy) for all flows.
        ``sizes`` : xr.Dataset
            Sizes for all flows.
        ``charge_states`` : xr.Dataset
            Charge states for all storage components.
        ``temporal_effects`` : xr.Dataset
            Temporal effects per contributor per timestep.
        ``periodic_effects`` : xr.Dataset
            Periodic (investment) effects per contributor.
        ``total_effects`` : xr.Dataset
            Total effects (temporal + periodic) per contributor.
        ``effect_share_factors`` : dict
            Conversion factors between effects.

    Examples:
        >>> flow_system.optimize(solver)
        >>> flow_system.statistics.flow_rates  # Get data
        >>> flow_system.statistics.plot.balance('Bus')  # Plot
    """

    def __init__(self, flow_system: FlowSystem) -> None:
        self._fs = flow_system
        # Cached data
        self._flow_rates: xr.Dataset | None = None
        self._flow_hours: xr.Dataset | None = None
        self._flow_sizes: xr.Dataset | None = None
        self._storage_sizes: xr.Dataset | None = None
        self._sizes: xr.Dataset | None = None
        self._charge_states: xr.Dataset | None = None
        self._effect_share_factors: dict[str, dict] | None = None
        self._temporal_effects: xr.Dataset | None = None
        self._periodic_effects: xr.Dataset | None = None
        self._total_effects: xr.Dataset | None = None
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
            >>> flow_system.statistics.plot.balance('ElectricityBus')
            >>> flow_system.statistics.plot.heatmap('Boiler|on')
        """
        if self._plot is None:
            self._plot = StatisticsPlotAccessor(self)
        return self._plot

    @property
    def flow_rates(self) -> xr.Dataset:
        """All flow rates as a Dataset with flow labels as variable names.

        Each variable has attributes:
            - 'carrier': carrier type (e.g., 'heat', 'electricity', 'gas')
            - 'unit': carrier unit (e.g., 'kW')
        """
        self._require_solution()
        if self._flow_rates is None:
            flow_rate_vars = self._fs.get_variables_by_category(VariableCategory.FLOW_RATE)
            flow_carriers = self._fs.flow_carriers  # Cached lookup
            carrier_units = self.carrier_units  # Cached lookup
            data_vars = {}
            for v in flow_rate_vars:
                flow_label = v.rsplit('|', 1)[0]  # Extract label from 'label|flow_rate'
                da = self._fs.solution[v].copy()
                # Add carrier and unit as attributes
                carrier = flow_carriers.get(flow_label)
                da.attrs['carrier'] = carrier
                da.attrs['unit'] = carrier_units.get(carrier, '') if carrier else ''
                data_vars[flow_label] = da
            self._flow_rates = xr.Dataset(data_vars)
        return self._flow_rates

    @property
    def flow_hours(self) -> xr.Dataset:
        """All flow hours (energy) as a Dataset with flow labels as variable names.

        Each variable has attributes:
            - 'carrier': carrier type (e.g., 'heat', 'electricity', 'gas')
            - 'unit': energy unit (e.g., 'kWh', 'm3/s*h')
        """
        self._require_solution()
        if self._flow_hours is None:
            hours = self._fs.timestep_duration
            flow_rates = self.flow_rates
            # Multiply and preserve/transform attributes
            data_vars = {}
            for var in flow_rates.data_vars:
                da = flow_rates[var] * hours
                da.attrs['carrier'] = flow_rates[var].attrs.get('carrier')
                # Convert power unit to energy unit (e.g., 'kW' -> 'kWh', 'm3/s' -> 'm3/s*h')
                power_unit = flow_rates[var].attrs.get('unit', '')
                da.attrs['unit'] = f'{power_unit}*h' if power_unit else ''
                data_vars[var] = da
            self._flow_hours = xr.Dataset(data_vars)
        return self._flow_hours

    @property
    def flow_sizes(self) -> xr.Dataset:
        """Flow sizes as a Dataset with flow labels as variable names."""
        self._require_solution()
        if self._flow_sizes is None:
            flow_size_vars = self._fs.get_variables_by_category(VariableCategory.FLOW_SIZE)
            self._flow_sizes = xr.Dataset({v.rsplit('|', 1)[0]: self._fs.solution[v] for v in flow_size_vars})
        return self._flow_sizes

    @property
    def storage_sizes(self) -> xr.Dataset:
        """Storage capacity sizes as a Dataset with storage labels as variable names."""
        self._require_solution()
        if self._storage_sizes is None:
            storage_size_vars = self._fs.get_variables_by_category(VariableCategory.STORAGE_SIZE)
            self._storage_sizes = xr.Dataset({v.rsplit('|', 1)[0]: self._fs.solution[v] for v in storage_size_vars})
        return self._storage_sizes

    @property
    def sizes(self) -> xr.Dataset:
        """All investment sizes (flows and storage capacities) as a Dataset."""
        if self._sizes is None:
            self._sizes = xr.merge([self.flow_sizes, self.storage_sizes])
        return self._sizes

    @property
    def charge_states(self) -> xr.Dataset:
        """All storage charge states as a Dataset with storage labels as variable names."""
        self._require_solution()
        if self._charge_states is None:
            charge_vars = self._fs.get_variables_by_category(VariableCategory.CHARGE_STATE)
            self._charge_states = xr.Dataset({v.rsplit('|', 1)[0]: self._fs.solution[v] for v in charge_vars})
        return self._charge_states

    @property
    def effect_share_factors(self) -> dict[str, dict]:
        """Effect share factors for temporal and periodic modes.

        Returns:
            Dict with 'temporal' and 'periodic' keys, each containing
            conversion factors between effects.
        """
        self._require_solution()
        if self._effect_share_factors is None:
            factors = self._fs.effects.calculate_effect_share_factors()
            self._effect_share_factors = {'temporal': factors[0], 'periodic': factors[1]}
        return self._effect_share_factors

    @property
    def temporal_effects(self) -> xr.Dataset:
        """Temporal effects per contributor per timestep.

        Returns a Dataset where each effect is a data variable with dimensions
        [time, contributor] (plus period/scenario if present).

        Coordinates:
            - contributor: Individual contributor labels
            - component: Parent component label for groupby operations
            - component_type: Component type (e.g., 'Boiler', 'Source', 'Sink')

        Examples:
            >>> # Get costs per contributor per timestep
            >>> statistics.temporal_effects['costs']
            >>> # Sum over all contributors to get total costs per timestep
            >>> statistics.temporal_effects['costs'].sum('contributor')
            >>> # Group by component
            >>> statistics.temporal_effects['costs'].groupby('component').sum()

        Returns:
            xr.Dataset with effects as variables and contributor dimension.
        """
        self._require_solution()
        if self._temporal_effects is None:
            ds = self._create_effects_dataset('temporal')
            dim_order = ['time', 'period', 'scenario', 'contributor']
            self._temporal_effects = ds.transpose(*dim_order, missing_dims='ignore')
        return self._temporal_effects

    @property
    def periodic_effects(self) -> xr.Dataset:
        """Periodic (investment) effects per contributor.

        Returns a Dataset where each effect is a data variable with dimensions
        [contributor] (plus period/scenario if present).

        Coordinates:
            - contributor: Individual contributor labels
            - component: Parent component label for groupby operations
            - component_type: Component type (e.g., 'Boiler', 'Source', 'Sink')

        Examples:
            >>> # Get investment costs per contributor
            >>> statistics.periodic_effects['costs']
            >>> # Sum over all contributors to get total investment costs
            >>> statistics.periodic_effects['costs'].sum('contributor')
            >>> # Group by component
            >>> statistics.periodic_effects['costs'].groupby('component').sum()

        Returns:
            xr.Dataset with effects as variables and contributor dimension.
        """
        self._require_solution()
        if self._periodic_effects is None:
            ds = self._create_effects_dataset('periodic')
            dim_order = ['period', 'scenario', 'contributor']
            self._periodic_effects = ds.transpose(*dim_order, missing_dims='ignore')
        return self._periodic_effects

    @property
    def total_effects(self) -> xr.Dataset:
        """Total effects (temporal + periodic) per contributor.

        Returns a Dataset where each effect is a data variable with dimensions
        [contributor] (plus period/scenario if present).

        Coordinates:
            - contributor: Individual contributor labels
            - component: Parent component label for groupby operations
            - component_type: Component type (e.g., 'Boiler', 'Source', 'Sink')

        Examples:
            >>> # Get total costs per contributor
            >>> statistics.total_effects['costs']
            >>> # Sum over all contributors to get total system costs
            >>> statistics.total_effects['costs'].sum('contributor')
            >>> # Group by component
            >>> statistics.total_effects['costs'].groupby('component').sum()
            >>> # Group by component type
            >>> statistics.total_effects['costs'].groupby('component_type').sum()

        Returns:
            xr.Dataset with effects as variables and contributor dimension.
        """
        self._require_solution()
        if self._total_effects is None:
            ds = self._create_effects_dataset('total')
            dim_order = ['period', 'scenario', 'contributor']
            self._total_effects = ds.transpose(*dim_order, missing_dims='ignore')
        return self._total_effects

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

    def _create_effects_dataset(self, mode: Literal['temporal', 'periodic', 'total']) -> xr.Dataset:
        """Create dataset containing effect totals for all contributors.

        Detects contributors (flows, components, etc.) from solution data variables.
        Excludes effect-to-effect shares which are intermediate conversions.
        Provides component and component_type coordinates for flexible groupby operations.
        """
        solution = self._fs.solution
        template = self._create_template_for_mode(mode)

        # Detect contributors from solution data variables
        # Pattern: {contributor}->{effect}(temporal) or {contributor}->{effect}(periodic)
        contributor_pattern = re.compile(r'^(.+)->(.+)\((temporal|periodic)\)$')
        effect_labels = set(self._fs.effects.keys())

        detected_contributors: set[str] = set()
        for var in solution.data_vars:
            match = contributor_pattern.match(str(var))
            if match:
                contributor = match.group(1)
                # Exclude effect-to-effect shares (e.g., costs(temporal) -> Effect1(temporal))
                base_name = contributor.split('(')[0] if '(' in contributor else contributor
                if base_name not in effect_labels:
                    detected_contributors.add(contributor)

        contributors = sorted(detected_contributors)

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

        # Determine modes to process
        modes_to_process = ['temporal', 'periodic'] if mode == 'total' else [mode]

        ds = xr.Dataset()

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
                        label = f'{contributor}->{source_effect}({current_mode})'
                        if label in solution:
                            da = solution[label] * factor
                            # For total mode, sum temporal over time (apply cluster_weight for proper weighting)
                            # Sum over all temporal dimensions (time, and cluster if present)
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
            da = xr.concat(contributor_arrays, dim='contributor', coords='minimal', join='outer').rename(effect)
            # Add unit attribute from effect definition
            da.attrs['unit'] = self.effect_units.get(effect, '')
            ds[effect] = da

        # Add groupby coordinates for contributor dimension
        ds = ds.assign_coords(
            component=('contributor', parents),
            component_type=('contributor', contributor_types),
        )

        # Validation: check totals match solution
        suffix_map = {'temporal': '(temporal)|per_timestep', 'periodic': '(periodic)', 'total': ''}
        for effect in self._fs.effects:
            label = f'{effect}{suffix_map[mode]}'
            if label in solution:
                computed = ds[effect].sum('contributor')
                found = solution[label]
                if not np.allclose(computed.fillna(0).values, found.fillna(0).values, equal_nan=True):
                    logger.critical(
                        f'Results for {effect}({mode}) in effects_dataset doesnt match {label}\n{computed=}\n, {found=}'
                    )

        return ds


# --- Sankey Plot Accessor ---


class SankeyPlotAccessor:
    """Sankey diagram accessor. Access via ``flow_system.statistics.plot.sankey``.

    Provides typed methods for different sankey diagram types.

    Examples:
        >>> fs.statistics.plot.sankey.flows(select={'bus': 'HeatBus'})
        >>> fs.statistics.plot.sankey.effects(select={'effect': 'costs'})
        >>> fs.statistics.plot.sankey.sizes(select={'component': 'Boiler'})
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

        # Use flow_rates to get carrier names from xarray attributes (already computed)
        flow_rates = self._stats.flow_rates

        for flow in self._fs.flows.values():
            label = flow.label_full
            if label not in ds:
                continue

            # Apply filters
            if flow_filter is not None and label not in flow_filter:
                continue
            bus_label = flow.bus
            comp_label = flow.component
            if bus_filter is not None and bus_label not in bus_filter:
                continue

            # Get carrier name from flow_rates xarray attribute (efficient lookup)
            carrier_name = flow_rates[label].attrs.get('carrier') if label in flow_rates else None

            if carrier_filter is not None:
                if carrier_name is None or carrier_name.lower() not in carrier_filter:
                    continue
            if component_filter is not None and comp_label not in component_filter:
                continue

            value = float(ds[label].values)
            if abs(value) < min_value:
                continue

            if flow.is_input_in_component:
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

        sankey_ds = xr.Dataset({'value': ('link', links['value'])}, coords=coords)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=sankey_ds, figure=fig)

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

        ds = self._stats.flow_hours.copy()

        # Apply period/scenario weights
        if 'period' in ds.dims and self._fs.period_weights is not None:
            ds = ds * self._fs.period_weights
        if 'scenario' in ds.dims and self._fs.scenario_weights is not None:
            weights = self._fs.scenario_weights / self._fs.scenario_weights.sum()
            ds = ds * weights

        ds = _apply_selection(ds, xr_select)

        # Aggregate remaining dimensions
        if 'time' in ds.dims:
            ds = getattr(ds, aggregate)(dim='time')
        for dim in ['period', 'scenario']:
            if dim in ds.dims:
                ds = ds.sum(dim=dim)

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

        ds = self._stats.sizes.copy()
        ds = _apply_selection(ds, xr_select)

        # Collapse remaining dimensions
        for dim in ['period', 'scenario']:
            if dim in ds.dims:
                ds = ds.max(dim=dim)

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

        ds = self._stats.flow_rates.copy()
        ds = _apply_selection(ds, xr_select)

        # Take max over all dimensions
        for dim in ['time', 'period', 'scenario']:
            if dim in ds.dims:
                ds = ds.max(dim=dim)

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
        effect_names = list(total_effects.data_vars)
        if effect_filter is not None:
            effect_names = [e for e in effect_names if e in effect_filter]

        # Collect all links: component -> effect
        nodes: set[str] = set()
        links: dict[str, list] = {'source': [], 'target': [], 'value': [], 'label': []}

        for effect_name in effect_names:
            effect_data = total_effects[effect_name]
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
    """Plot accessor for statistics. Access via ``flow_system.statistics.plot``.

    All methods return PlotResult with both data and figure.
    """

    def __init__(self, statistics: StatisticsAccessor) -> None:
        self._stats = statistics
        self._fs = statistics._fs
        self._sankey: SankeyPlotAccessor | None = None

    @property
    def sankey(self) -> SankeyPlotAccessor:
        """Access sankey diagram methods with typed select options.

        Returns:
            SankeyPlotAccessor with methods: flows(), sizes(), peak_flow(), effects()

        Examples:
            >>> fs.statistics.plot.sankey.flows(select={'bus': 'HeatBus'})
            >>> fs.statistics.plot.sankey.effects(select={'effect': 'costs'})
        """
        if self._sankey is None:
            self._sankey = SankeyPlotAccessor(self)
        return self._sankey

    def _get_color_map_for_balance(self, node: str, flow_labels: list[str]) -> dict[str, str]:
        """Build color map for balance plot.

        - Bus balance: colors from component.color (using cached component_colors)
        - Component balance: colors from flow's carrier (using cached carrier_colors)

        Raises:
            RuntimeError: If FlowSystem is not connected_and_transformed.
        """
        if not self._fs.connected_and_transformed:
            raise RuntimeError(
                'FlowSystem is not connected_and_transformed. Call FlowSystem.connect_and_transform() first.'
            )

        is_bus = node in self._fs.buses
        color_map = {}
        uncolored = []

        # Get cached colors for efficient lookup
        carrier_colors = self._stats.carrier_colors
        component_colors = self._stats.component_colors
        flow_rates = self._stats.flow_rates

        for label in flow_labels:
            if is_bus:
                # Use cached component colors
                comp_label = self._fs.flows[label].component
                color = component_colors.get(comp_label)
            else:
                # Use carrier name from xarray attribute (already computed) + cached colors
                carrier_name = flow_rates[label].attrs.get('carrier') if label in flow_rates else None
                color = carrier_colors.get(carrier_name) if carrier_name else None

            if color:
                color_map[label] = color
            else:
                uncolored.append(label)

        if uncolored:
            color_map.update(process_colors(None, uncolored))

        return color_map

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
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot node balance (inputs vs outputs) for a Bus or Component.

        Args:
            node: Label of the Bus or Component to plot.
            select: xarray-style selection dict.
            include: Only include flows containing these substrings.
            exclude: Exclude flows containing these substrings.
            unit: 'flow_rate' (power) or 'flow_hours' (energy).
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            round_decimals: Round values to this many decimal places to avoid numerical noise
                (e.g., tiny negative values from solver precision). Set to None to disable.
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
        elif node in self._fs.components:
            element = self._fs.components[node]
        else:
            raise KeyError(f"'{node}' not found in buses or components")

        input_labels = [f.label_full for f in element.inputs.values()]
        output_labels = [f.label_full for f in element.outputs.values()]
        all_labels = input_labels + output_labels

        filtered_labels = _filter_by_pattern(all_labels, include, exclude)
        if not filtered_labels:
            logger.warning(f'No flows remaining after filtering for node {node}')
            return PlotResult(data=xr.Dataset(), figure=go.Figure())

        # Get data from statistics
        if unit == 'flow_rate':
            ds = self._stats.flow_rates[[lbl for lbl in filtered_labels if lbl in self._stats.flow_rates]]
        else:
            ds = self._stats.flow_hours[[lbl for lbl in filtered_labels if lbl in self._stats.flow_hours]]

        # Negate inputs
        for label in input_labels:
            if label in ds:
                ds[label] = -ds[label]

        ds = _apply_selection(ds, select)

        # Build color kwargs - use default colors from element attributes if not specified
        if colors is None:
            color_kwargs = {'color_discrete_map': self._get_color_map_for_balance(node, list(ds.data_vars))}
        else:
            color_kwargs = _build_color_kwargs(colors, list(ds.data_vars))

        # Early return for data_only mode (skip figure creation for performance)
        if data_only:
            return PlotResult(data=ds, figure=go.Figure())

        # Round to avoid numerical noise (tiny negative values from solver precision)
        if round_decimals is not None:
            ds = ds.round(round_decimals)

        # Get unit label from first data variable's attributes
        unit_label = ''
        if ds.data_vars:
            first_var = next(iter(ds.data_vars))
            unit_label = ds[first_var].attrs.get('unit', '')

        _apply_slot_defaults(plotly_kwargs, 'balance')
        fig = ds.plotly.fast_bar(
            title=f'{node} [{unit_label}]' if unit_label else node,
            **color_kwargs,
            **plotly_kwargs,
        )
        _apply_unified_hover(fig, unit=unit_label)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=ds, figure=fig)

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
            include: Only include flows containing these substrings.
            exclude: Exclude flows containing these substrings.
            unit: 'flow_rate' (power) or 'flow_hours' (energy).
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            round_decimals: Round values to this many decimal places to avoid numerical noise
                (e.g., tiny negative values from solver precision). Set to None to disable.
            show: Whether to display the plot.
            data_only: If True, skip figure creation and return only data (for performance).
            **plotly_kwargs: Additional arguments passed to the plotly accessor (e.g.,
                facet_col, facet_row, animation_frame).

        Returns:
            PlotResult with .data and .figure.

        Examples:
            >>> fs.statistics.plot.carrier_balance('heat')
            >>> fs.statistics.plot.carrier_balance('electricity', unit='flow_hours')

        Notes:
            - Inputs to carrier buses (from sources/converters) are shown as positive
            - Outputs from carrier buses (to sinks/converters) are shown as negative
            - Internal transfers between buses of the same carrier appear on both sides
        """
        self._stats._require_solution()
        carrier = carrier.lower()

        # Find all buses with this carrier
        carrier_buses = [bus for bus in self._fs.buses.values() if bus.carrier == carrier]
        if not carrier_buses:
            raise KeyError(f"No buses found with carrier '{carrier}'")

        # Collect all flows connected to these buses
        input_labels: list[str] = []  # Inputs to buses = production
        output_labels: list[str] = []  # Outputs from buses = consumption

        for bus in carrier_buses:
            for flow in bus.inputs.values():
                input_labels.append(flow.label_full)
            for flow in bus.outputs.values():
                output_labels.append(flow.label_full)

        all_labels = input_labels + output_labels
        filtered_labels = _filter_by_pattern(all_labels, include, exclude)
        if not filtered_labels:
            logger.warning(f'No flows remaining after filtering for carrier {carrier}')
            return PlotResult(data=xr.Dataset(), figure=go.Figure())

        # Get data from statistics
        if unit == 'flow_rate':
            ds = self._stats.flow_rates[[lbl for lbl in filtered_labels if lbl in self._stats.flow_rates]]
        else:
            ds = self._stats.flow_hours[[lbl for lbl in filtered_labels if lbl in self._stats.flow_hours]]

        # Negate outputs (consumption) - opposite convention from bus balance
        for label in output_labels:
            if label in ds:
                ds[label] = -ds[label]

        ds = _apply_selection(ds, select)

        # Build color kwargs
        if colors is None:
            component_colors = self._stats.component_colors
            color_map = {}
            uncolored = []
            for label in ds.data_vars:
                flow = self._fs.flows.get(label)
                if flow:
                    color = component_colors.get(flow.component)
                    if color:
                        color_map[label] = color
                        continue
                uncolored.append(label)
            if uncolored:
                color_map.update(process_colors(None, uncolored))
            color_kwargs = {'color_discrete_map': color_map}
        else:
            color_kwargs = _build_color_kwargs(colors, list(ds.data_vars))

        # Early return for data_only mode (skip figure creation for performance)
        if data_only:
            return PlotResult(data=ds, figure=go.Figure())

        # Round to avoid numerical noise (tiny negative values from solver precision)
        if round_decimals is not None:
            ds = ds.round(round_decimals)

        # Get unit label from carrier or first data variable
        unit_label = ''
        if ds.data_vars:
            first_var = next(iter(ds.data_vars))
            unit_label = ds[first_var].attrs.get('unit', '')

        _apply_slot_defaults(plotly_kwargs, 'carrier_balance')
        fig = ds.plotly.fast_bar(
            title=f'{carrier.capitalize()} Balance [{unit_label}]' if unit_label else f'{carrier.capitalize()} Balance',
            **color_kwargs,
            **plotly_kwargs,
        )
        _apply_unified_hover(fig, unit=unit_label)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=ds, figure=fig)

    def heatmap(
        self,
        variables: str | list[str],
        *,
        select: SelectType | None = None,
        reshape: tuple[str, str] | Literal['auto'] | None = ('D', 'h'),
        colors: str | list[str] | None = None,
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
            show: Whether to display the figure.
            data_only: If True, skip figure creation and return only data (for performance).
            **plotly_kwargs: Additional arguments passed to plotly accessor (e.g.,
                facet_col, animation_frame).

        Returns:
            PlotResult with processed data and figure.
        """
        solution = self._stats._require_solution()
        if isinstance(variables, str):
            variables = [variables]

        # Resolve, select, and stack into single DataArray
        resolved = self._resolve_variable_names(variables, solution)
        ds = _apply_selection(solution[resolved], select)
        da = xr.concat([ds[v] for v in ds.data_vars], dim=pd.Index(list(ds.data_vars), name='variable'))

        # Prepare for heatmap (reshape, transpose, squeeze)
        da = _prepare_for_heatmap(da, reshape)

        # Early return for data_only mode (skip figure creation for performance)
        if data_only:
            return PlotResult(data=da.to_dataset(name='value'), figure=go.Figure())

        # Only pass colors if not already in plotly_kwargs (avoid duplicate arg error)
        if 'color_continuous_scale' not in plotly_kwargs:
            plotly_kwargs['color_continuous_scale'] = colors
        fig = da.plotly.imshow(**plotly_kwargs)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=da.to_dataset(name='value'), figure=fig)

    def flows(
        self,
        *,
        start: str | list[str] | None = None,
        end: str | list[str] | None = None,
        component: str | list[str] | None = None,
        select: SelectType | None = None,
        unit: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        colors: ColorType | None = None,
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
            show: Whether to display.
            data_only: If True, skip figure creation and return only data (for performance).
            **plotly_kwargs: Additional arguments passed to the plotly accessor (e.g.,
                facet_col, facet_row, animation_frame).

        Returns:
            PlotResult with flow data.
        """
        self._stats._require_solution()

        ds = self._stats.flow_rates if unit == 'flow_rate' else self._stats.flow_hours

        # Filter by connection
        if start is not None or end is not None or component is not None:
            matching_labels = []
            starts = [start] if isinstance(start, str) else (start or [])
            ends = [end] if isinstance(end, str) else (end or [])
            components = [component] if isinstance(component, str) else (component or [])

            for flow in self._fs.flows.values():
                # Get bus label (could be string or Bus object)
                bus_label = flow.bus
                comp_label = flow.component

                # start/end filtering based on flow direction
                if flow.is_input_in_component:
                    # Flow goes: bus -> component, so start=bus, end=component
                    if starts and bus_label not in starts:
                        continue
                    if ends and comp_label not in ends:
                        continue
                else:
                    # Flow goes: component -> bus, so start=component, end=bus
                    if starts and comp_label not in starts:
                        continue
                    if ends and bus_label not in ends:
                        continue

                if components and comp_label not in components:
                    continue
                matching_labels.append(flow.label_full)

            ds = ds[[lbl for lbl in matching_labels if lbl in ds]]

        ds = _apply_selection(ds, select)

        # Early return for data_only mode (skip figure creation for performance)
        if data_only:
            return PlotResult(data=ds, figure=go.Figure())

        # Get unit label from first data variable's attributes
        unit_label = ''
        if ds.data_vars:
            first_var = next(iter(ds.data_vars))
            unit_label = ds[first_var].attrs.get('unit', '')

        # Build color kwargs
        color_kwargs = _build_color_kwargs(colors, list(ds.data_vars))

        _apply_slot_defaults(plotly_kwargs, 'flows')
        fig = ds.plotly.line(
            title=f'Flows [{unit_label}]' if unit_label else 'Flows',
            **color_kwargs,
            **plotly_kwargs,
        )

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=ds, figure=fig)

    def sizes(
        self,
        *,
        max_size: float | None = 1e6,
        select: SelectType | None = None,
        colors: ColorType | None = None,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot investment sizes (capacities) of flows.

        Args:
            max_size: Maximum size to include (filters defaults).
            select: xarray-style selection.
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            show: Whether to display.
            data_only: If True, skip figure creation and return only data (for performance).
            **plotly_kwargs: Additional arguments passed to the plotly accessor (e.g.,
                facet_col, facet_row, animation_frame).

        Returns:
            PlotResult with size data.
        """
        self._stats._require_solution()
        ds = self._stats.sizes

        ds = _apply_selection(ds, select)

        if max_size is not None and ds.data_vars:
            valid_labels = [lbl for lbl in ds.data_vars if float(ds[lbl].max()) < max_size]
            ds = ds[valid_labels]

        # Early return for data_only mode (skip figure creation for performance)
        if data_only:
            return PlotResult(data=ds, figure=go.Figure())

        if not ds.data_vars:
            fig = go.Figure()
        else:
            # Build color kwargs
            color_kwargs = _build_color_kwargs(colors, list(ds.data_vars))
            _apply_slot_defaults(plotly_kwargs, 'sizes')
            fig = ds.plotly.bar(
                title='Investment Sizes',
                labels={'value': 'Size'},
                **color_kwargs,
                **plotly_kwargs,
            )

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=ds, figure=fig)

    def duration_curve(
        self,
        variables: str | list[str],
        *,
        select: SelectType | None = None,
        normalize: bool = False,
        colors: ColorType | None = None,
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
        normalized_vars = []
        for var in variables:
            # Strip |flow_rate suffix if present
            if var.endswith('|flow_rate'):
                var = var[: -len('|flow_rate')]
            normalized_vars.append(var)

        # Try to get from flow_rates first, fall back to solution for non-flow variables
        ds_parts = []
        for var in normalized_vars:
            if var in flow_rates:
                ds_parts.append(flow_rates[[var]])
            elif var in solution:
                ds_parts.append(solution[[var]])
            else:
                # Try with |flow_rate suffix as last resort
                flow_rate_var = f'{var}|flow_rate'
                if flow_rate_var in solution:
                    ds_parts.append(solution[[flow_rate_var]].rename({flow_rate_var: var}))
                else:
                    raise KeyError(f"Variable '{var}' not found in flow_rates or solution")

        ds = xr.merge(ds_parts)
        ds = _apply_selection(ds, select)

        result_ds = ds.fxstats.to_duration_curve(normalize=normalize)

        # Early return for data_only mode (skip figure creation for performance)
        if data_only:
            return PlotResult(data=result_ds, figure=go.Figure())

        # Get unit label from first data variable's attributes
        unit_label = ''
        if ds.data_vars:
            first_var = next(iter(ds.data_vars))
            unit_label = ds[first_var].attrs.get('unit', '')

        # Build color kwargs
        color_kwargs = _build_color_kwargs(colors, list(result_ds.data_vars))

        plotly_kwargs.setdefault('x', 'duration_pct' if normalize else 'duration')
        _apply_slot_defaults(plotly_kwargs, 'duration_curve')
        fig = result_ds.plotly.line(
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

        return PlotResult(data=result_ds, figure=fig)

    def effects(
        self,
        aspect: Literal['total', 'temporal', 'periodic'] = 'total',
        *,
        effect: str | None = None,
        by: Literal['component', 'contributor', 'time'] | None = None,
        select: SelectType | None = None,
        colors: ColorType | None = None,
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
            show: Whether to display.
            data_only: If True, skip figure creation and return only data (for performance).
            **plotly_kwargs: Additional arguments passed to the plotly accessor (e.g.,
                facet_col, facet_row, animation_frame).

        Returns:
            PlotResult with effect breakdown data.

        Examples:
            >>> flow_system.statistics.plot.effects()  # Aggregated totals per effect
            >>> flow_system.statistics.plot.effects(effect='costs')  # Just costs
            >>> flow_system.statistics.plot.effects(by='component')  # Breakdown by component
            >>> flow_system.statistics.plot.effects(by='contributor')  # By individual flows
            >>> flow_system.statistics.plot.effects(aspect='temporal', by='time')  # Over time
        """
        self._stats._require_solution()

        # Get the appropriate effects dataset based on aspect
        effects_ds = {
            'total': self._stats.total_effects,
            'temporal': self._stats.temporal_effects,
            'periodic': self._stats.periodic_effects,
        }.get(aspect)
        if effects_ds is None:
            raise ValueError(f"Aspect '{aspect}' not valid. Choose from 'total', 'temporal', 'periodic'.")

        # Filter to specific effect(s) and apply selection
        if effect is not None:
            if effect not in effects_ds:
                raise ValueError(f"Effect '{effect}' not found. Available: {list(effects_ds.data_vars)}")
            ds = effects_ds[[effect]]
        else:
            ds = effects_ds

        # Group by component (default) unless by='contributor'
        if by != 'contributor' and 'contributor' in ds.dims:
            ds = ds.groupby('component').sum()

        ds = _apply_selection(ds, select)

        # Sum over dimensions based on 'by' parameter
        if by is None:
            for dim in ['time', 'component', 'contributor']:
                if dim in ds.dims:
                    ds = ds.sum(dim=dim)
            x_col, color_col = 'variable', 'variable'
        elif by == 'component':
            if 'time' in ds.dims:
                ds = ds.sum(dim='time')
            x_col = 'component'
            color_col = 'variable' if len(ds.data_vars) > 1 else 'component'
        elif by == 'contributor':
            if 'time' in ds.dims:
                ds = ds.sum(dim='time')
            x_col = 'contributor'
            color_col = 'variable' if len(ds.data_vars) > 1 else 'contributor'
        elif by == 'time':
            if 'time' not in ds.dims:
                raise ValueError(f"Cannot plot by 'time' for aspect '{aspect}' - no time dimension.")
            for dim in ['component', 'contributor']:
                if dim in ds.dims:
                    ds = ds.sum(dim=dim)
            x_col = 'time'
            color_col = 'variable' if len(ds.data_vars) > 1 else None
        else:
            raise ValueError(f"'by' must be one of 'component', 'contributor', 'time', or None, got {by!r}")

        # Early return for data_only mode (skip figure creation for performance)
        if data_only:
            return PlotResult(data=ds, figure=go.Figure())

        # Build title
        effect_label = effect or 'Effects'
        title = f'{effect_label} ({aspect})' if by is None else f'{effect_label} ({aspect}) by {by}'

        # Allow user override of color via plotly_kwargs
        color = plotly_kwargs.pop('color', color_col)

        # Build color kwargs
        color_dim = color or color_col or 'variable'
        if color_dim in ds.coords:
            labels = list(ds.coords[color_dim].values)
        elif color_dim == 'variable':
            labels = list(ds.data_vars)
        else:
            labels = []
        color_kwargs = _build_color_kwargs(colors, labels) if labels else {}

        plotly_kwargs.setdefault('x', x_col)
        _apply_slot_defaults(plotly_kwargs, 'effects')
        fig = ds.plotly.bar(
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

        return PlotResult(data=ds, figure=fig)

    def charge_states(
        self,
        storages: str | list[str] | None = None,
        *,
        select: SelectType | None = None,
        colors: ColorType | None = None,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot storage charge states over time.

        Args:
            storages: Storage label(s) to plot. If None, plots all storages.
            select: xarray-style selection.
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            show: Whether to display.
            data_only: If True, skip figure creation and return only data (for performance).
            **plotly_kwargs: Additional arguments passed to the plotly accessor (e.g.,
                facet_col, facet_row, animation_frame).

        Returns:
            PlotResult with charge state data.
        """
        self._stats._require_solution()
        ds = self._stats.charge_states

        if storages is not None:
            if isinstance(storages, str):
                storages = [storages]
            ds = ds[[s for s in storages if s in ds]]

        ds = _apply_selection(ds, select)

        # Early return for data_only mode (skip figure creation for performance)
        if data_only:
            return PlotResult(data=ds, figure=go.Figure())

        # Build color kwargs
        color_kwargs = _build_color_kwargs(colors, list(ds.data_vars))

        _apply_slot_defaults(plotly_kwargs, 'charge_states')
        fig = ds.plotly.line(
            title='Storage Charge States',
            **color_kwargs,
            **plotly_kwargs,
        )
        fig.update_yaxes(title_text='Charge State')

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=ds, figure=fig)

    def storage(
        self,
        storage: str,
        *,
        select: SelectType | None = None,
        unit: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        colors: ColorType | None = None,
        charge_state_color: str = 'black',
        round_decimals: int | None = 6,
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
        input_labels = [f.label_full for f in component.inputs.values()]
        output_labels = [f.label_full for f in component.outputs.values()]
        all_labels = input_labels + output_labels

        if unit == 'flow_rate':
            ds = self._stats.flow_rates[[lbl for lbl in all_labels if lbl in self._stats.flow_rates]]
        else:
            ds = self._stats.flow_hours[[lbl for lbl in all_labels if lbl in self._stats.flow_hours]]

        # Negate outputs for balance view (discharging shown as negative)
        for label in output_labels:
            if label in ds:
                ds[label] = -ds[label]

        # Get charge state and add to dataset
        charge_state = self._fs.solution[charge_state_var].rename(storage)
        ds['charge_state'] = charge_state

        # Apply selection
        ds = _apply_selection(ds, select)

        # Early return for data_only mode (skip figure creation for performance)
        if data_only:
            return PlotResult(data=ds, figure=go.Figure())

        # Separate flow data from charge_state
        flow_labels = [lbl for lbl in ds.data_vars if lbl != 'charge_state']
        flow_ds = ds[flow_labels]
        charge_da = ds['charge_state']

        # Round to avoid numerical noise (tiny negative values from solver precision)
        if round_decimals is not None:
            flow_ds = flow_ds.round(round_decimals)

        # Build color kwargs - use default colors from element attributes if not specified
        if colors is None:
            color_kwargs = {'color_discrete_map': self._get_color_map_for_balance(storage, flow_labels)}
        else:
            color_kwargs = _build_color_kwargs(colors, flow_labels)

        # Get unit label from flow data
        unit_label = ''
        if flow_ds.data_vars:
            first_var = next(iter(flow_ds.data_vars))
            unit_label = flow_ds[first_var].attrs.get('unit', '')

        # Create stacked area chart for flows (styled as bar)
        _apply_slot_defaults(plotly_kwargs, 'storage')
        fig = flow_ds.plotly.fast_bar(
            title=f'{storage} Operation [{unit_label}]' if unit_label else f'{storage} Operation',
            **color_kwargs,
            **plotly_kwargs,
        )
        _apply_unified_hover(fig, unit=unit_label)

        # Add charge state as line on secondary y-axis
        # Only pass faceting kwargs that add_line_overlay accepts
        overlay_kwargs = {
            k: v for k, v in plotly_kwargs.items() if k in ('x', 'facet_col', 'facet_row', 'animation_frame')
        }
        add_line_overlay(
            fig,
            charge_da,
            line_color=charge_state_color,
            name='charge_state',
            secondary_y=True,
            y_title='Charge State',
            **overlay_kwargs,
        )

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=ds, figure=fig)
