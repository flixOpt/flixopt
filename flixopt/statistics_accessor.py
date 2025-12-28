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
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr

from .color_processing import ColorType, hex_to_rgba, process_colors
from .config import CONFIG
from .plot_result import PlotResult

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


def _heatmap_figure(
    data: xr.DataArray,
    colors: str | list[str] | None = None,
    title: str = '',
    facet_col: str | None = None,
    animation_frame: str | None = None,
    facet_col_wrap: int | None = None,
    **imshow_kwargs: Any,
) -> go.Figure:
    """Create heatmap figure using px.imshow.

    Args:
        data: DataArray with 2-4 dimensions. First two are heatmap axes.
        colors: Colorscale name (str) or list of colors. Dicts are not supported
            for heatmaps as color_continuous_scale requires a colorscale specification.
        title: Plot title.
        facet_col: Dimension for subplot columns.
        animation_frame: Dimension for animation slider.
        facet_col_wrap: Max columns before wrapping.
        **imshow_kwargs: Additional args for px.imshow.

    Returns:
        Plotly Figure.
    """
    if data.size == 0:
        return go.Figure()

    colors = colors or CONFIG.Plotting.default_sequential_colorscale
    facet_col_wrap = facet_col_wrap or CONFIG.Plotting.default_facet_cols

    imshow_args: dict[str, Any] = {
        'img': data,
        'color_continuous_scale': colors,
        'title': title,
        **imshow_kwargs,
    }

    if facet_col and facet_col in data.dims:
        imshow_args['facet_col'] = facet_col
        if facet_col_wrap < data.sizes[facet_col]:
            imshow_args['facet_col_wrap'] = facet_col_wrap

    if animation_frame and animation_frame in data.dims:
        imshow_args['animation_frame'] = animation_frame

    return px.imshow(**imshow_args)


# --- Helper functions ---


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


def _resolve_auto_facets(
    ds: xr.Dataset,
    facet_col: str | Literal['auto'] | None,
    facet_row: str | Literal['auto'] | None,
    animation_frame: str | Literal['auto'] | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Resolve 'auto' facet/animation dimensions based on available data dimensions.

    When 'auto' is specified, extra dimensions are assigned to slots based on:
    - CONFIG.Plotting.extra_dim_priority: Order of dimensions (default: cluster → period → scenario)
    - CONFIG.Plotting.dim_slot_priority: Order of slots (default: facet_col → facet_row → animation_frame)

    Args:
        ds: Dataset to check for available dimensions.
        facet_col: Dimension name, 'auto', or None.
        facet_row: Dimension name, 'auto', or None.
        animation_frame: Dimension name, 'auto', or None.

    Returns:
        Tuple of (resolved_facet_col, resolved_facet_row, resolved_animation_frame).
        Each is either a valid dimension name or None.
    """
    # Get available extra dimensions with size > 1, sorted by priority
    available = {d for d in ds.dims if ds.sizes[d] > 1}
    extra_dims = [d for d in CONFIG.Plotting.extra_dim_priority if d in available]
    used: set[str] = set()

    # Map slot names to their input values
    slots = {
        'facet_col': facet_col,
        'facet_row': facet_row,
        'animation_frame': animation_frame,
    }
    results: dict[str, str | None] = {'facet_col': None, 'facet_row': None, 'animation_frame': None}

    # First pass: resolve explicit dimensions (not 'auto' or None) to mark them as used
    for slot_name, value in slots.items():
        if value is not None and value != 'auto':
            if value in available and value not in used:
                used.add(value)
                results[slot_name] = value

    # Second pass: resolve 'auto' slots in dim_slot_priority order
    dim_iter = iter(d for d in extra_dims if d not in used)
    for slot_name in CONFIG.Plotting.dim_slot_priority:
        if slots.get(slot_name) == 'auto':
            next_dim = next(dim_iter, None)
            if next_dim:
                used.add(next_dim)
                results[slot_name] = next_dim

    return results['facet_col'], results['facet_row'], results['animation_frame']


def _resolve_facets(
    ds: xr.Dataset,
    facet_col: str | Literal['auto'] | None,
    facet_row: str | Literal['auto'] | None,
) -> tuple[str | None, str | None]:
    """Resolve facet dimensions, returning None if not present in data.

    Legacy wrapper for _resolve_auto_facets for backward compatibility.
    """
    resolved_col, resolved_row, _ = _resolve_auto_facets(ds, facet_col, facet_row, None)
    return resolved_col, resolved_row


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


def _create_stacked_bar(
    ds: xr.Dataset,
    colors: ColorType,
    title: str,
    facet_col: str | None,
    facet_row: str | None,
    animation_frame: str | None = None,
    **plotly_kwargs: Any,
) -> go.Figure:
    """Create a stacked bar chart from xarray Dataset."""
    df = _dataset_to_long_df(ds)
    if df.empty:
        return go.Figure()
    x_col = 'time' if 'time' in df.columns else df.columns[0]
    variables = df['variable'].unique().tolist()
    color_map = process_colors(colors, variables, default_colorscale=CONFIG.Plotting.default_qualitative_colorscale)
    fig = px.bar(
        df,
        x=x_col,
        y='value',
        color='variable',
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
        color_discrete_map=color_map,
        title=title,
        **plotly_kwargs,
    )
    fig.update_layout(barmode='relative', bargap=0, bargroupgap=0)
    fig.update_traces(marker_line_width=0)
    return fig


def _create_line(
    ds: xr.Dataset,
    colors: ColorType,
    title: str,
    facet_col: str | None,
    facet_row: str | None,
    animation_frame: str | None = None,
    **plotly_kwargs: Any,
) -> go.Figure:
    """Create a line chart from xarray Dataset."""
    df = _dataset_to_long_df(ds)
    if df.empty:
        return go.Figure()
    x_col = 'time' if 'time' in df.columns else df.columns[0]
    variables = df['variable'].unique().tolist()
    color_map = process_colors(colors, variables, default_colorscale=CONFIG.Plotting.default_qualitative_colorscale)
    return px.line(
        df,
        x=x_col,
        y='value',
        color='variable',
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
        color_discrete_map=color_map,
        title=title,
        **plotly_kwargs,
    )


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
            flow_rate_vars = [v for v in self._fs.solution.data_vars if v.endswith('|flow_rate')]
            flow_carriers = self._fs.flow_carriers  # Cached lookup
            carrier_units = self.carrier_units  # Cached lookup
            data_vars = {}
            for v in flow_rate_vars:
                flow_label = v.replace('|flow_rate', '')
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
            flow_labels = set(self._fs.flows.keys())
            size_vars = [
                v for v in self._fs.solution.data_vars if v.endswith('|size') and v.replace('|size', '') in flow_labels
            ]
            self._flow_sizes = xr.Dataset({v.replace('|size', ''): self._fs.solution[v] for v in size_vars})
        return self._flow_sizes

    @property
    def storage_sizes(self) -> xr.Dataset:
        """Storage capacity sizes as a Dataset with storage labels as variable names."""
        self._require_solution()
        if self._storage_sizes is None:
            storage_labels = set(self._fs.storages.keys())
            size_vars = [
                v
                for v in self._fs.solution.data_vars
                if v.endswith('|size') and v.replace('|size', '') in storage_labels
            ]
            self._storage_sizes = xr.Dataset({v.replace('|size', ''): self._fs.solution[v] for v in size_vars})
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
            charge_vars = [v for v in self._fs.solution.data_vars if v.endswith('|charge_state')]
            self._charge_states = xr.Dataset(
                {v.replace('|charge_state', ''): self._fs.solution[v] for v in charge_vars}
            )
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
            flows = [f.label_full.split('|')[0] for f in comp.inputs + comp.outputs]
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
                                weighted = da * self._fs.cluster_weight
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
        """Get colors for nodes: buses use cached bus_colors, components use process_colors."""
        # Get fallback colors from process_colors
        fallback_colors = process_colors(colors, node_list)

        # Use cached bus colors for efficiency
        bus_colors = self._stats.bus_colors

        node_colors = []
        for node in node_list:
            # Check if node is a bus with a cached color
            if node in bus_colors:
                node_colors.append(bus_colors[node])
            else:
                # Fall back to process_colors
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
            color_map.update(process_colors(CONFIG.Plotting.default_qualitative_colorscale, uncolored))

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
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        show: bool | None = None,
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
            facet_col: Dimension for column facets. 'auto' uses first available of
                cluster/period/scenario. None disables.
            facet_row: Dimension for row facets. 'auto' uses first available after facet_col.
            animation_frame: Dimension for animation slider. 'auto' uses first available
                after facets.
            show: Whether to display the plot.

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

        input_labels = [f.label_full for f in element.inputs]
        output_labels = [f.label_full for f in element.outputs]
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
        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            ds, facet_col, facet_row, animation_frame
        )

        # Build color map from Element.color attributes if no colors specified
        if colors is None:
            colors = self._get_color_map_for_balance(node, list(ds.data_vars))

        # Get unit label from first data variable's attributes
        unit_label = ''
        if ds.data_vars:
            first_var = next(iter(ds.data_vars))
            unit_label = ds[first_var].attrs.get('unit', '')

        fig = _create_stacked_bar(
            ds,
            colors=colors,
            title=f'{node} [{unit_label}]' if unit_label else node,
            facet_col=actual_facet_col,
            facet_row=actual_facet_row,
            animation_frame=actual_anim,
            **plotly_kwargs,
        )

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
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        show: bool | None = None,
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
            facet_col: Dimension for column facets. 'auto' uses first available of
                cluster/period/scenario.
            facet_row: Dimension for row facets. 'auto' uses first available after facet_col.
            animation_frame: Dimension for animation slider.
            show: Whether to display the plot.

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
            for flow in bus.inputs:
                input_labels.append(flow.label_full)
            for flow in bus.outputs:
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
        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            ds, facet_col, facet_row, animation_frame
        )

        # Use cached component colors for flows
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
                color_map.update(process_colors(CONFIG.Plotting.default_qualitative_colorscale, uncolored))
            colors = color_map

        # Get unit label from carrier or first data variable
        unit_label = ''
        if ds.data_vars:
            first_var = next(iter(ds.data_vars))
            unit_label = ds[first_var].attrs.get('unit', '')

        fig = _create_stacked_bar(
            ds,
            colors=colors,
            title=f'{carrier.capitalize()} Balance [{unit_label}]' if unit_label else f'{carrier.capitalize()} Balance',
            facet_col=actual_facet_col,
            facet_row=actual_facet_row,
            animation_frame=actual_anim,
            **plotly_kwargs,
        )

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
        reshape: tuple[str, str] | Literal['auto'] | None = 'auto',
        colors: str | list[str] | None = None,
        facet_col: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = None,
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot heatmap of time series data.

        Time is reshaped into 2D (e.g., days × hours) when possible. For clustered data,
        the natural (cluster, time) shape is used directly without reshaping.

        Multiple variables are shown as facets. If too many dimensions exist to display
        without data loss, reshaping is skipped and variables are shown on the y-axis.

        Args:
            variables: Flow label(s) or variable name(s). Flow labels like 'Boiler(Q_th)'
                are automatically resolved to 'Boiler(Q_th)|flow_rate'. Full variable
                names like 'Storage|charge_state' are used as-is.
            select: xarray-style selection, e.g. {'scenario': 'Base Case'}.
            reshape: Time reshape frequencies as (outer, inner), e.g. ('D', 'h') for
                days × hours. 'auto' uses (cluster, time) for clustered data or
                ('D', 'h') otherwise. None disables reshaping.
            colors: Colorscale name (str) or list of colors for heatmap coloring.
                Dicts are not supported for heatmaps (use str or list[str]).
            facet_col: Dimension for subplot columns. 'auto' uses first available of
                cluster/period/scenario. With multiple variables, 'variable' is used.
            animation_frame: Dimension for animation slider. 'auto' uses first available.
            show: Whether to display the figure.
            **plotly_kwargs: Additional arguments passed to px.imshow.

        Returns:
            PlotResult with processed data and figure.
        """
        solution = self._stats._require_solution()

        if isinstance(variables, str):
            variables = [variables]

        # Resolve flow labels to variable names
        resolved_variables = self._resolve_variable_names(variables, solution)

        ds = solution[resolved_variables]
        ds = _apply_selection(ds, select)

        # Stack variables into single DataArray
        variable_names = list(ds.data_vars)
        dataarrays = [ds[var] for var in variable_names]
        da = xr.concat(dataarrays, dim=pd.Index(variable_names, name='variable'))

        # Check if data is clustered (has cluster dimension with size > 1)
        is_clustered = 'cluster' in da.dims and da.sizes['cluster'] > 1

        # Determine facet and animation from available dims
        has_multiple_vars = 'variable' in da.dims and da.sizes['variable'] > 1

        if has_multiple_vars:
            actual_facet = 'variable'
            # Resolve animation using auto logic, excluding 'variable' which is used for facet
            _, _, actual_animation = _resolve_auto_facets(da.to_dataset(name='value'), None, None, animation_frame)
            if actual_animation == 'variable':
                actual_animation = None
        else:
            # Resolve facet and animation using auto logic
            actual_facet, _, actual_animation = _resolve_auto_facets(
                da.to_dataset(name='value'), facet_col, None, animation_frame
            )

        # Determine heatmap dimensions based on data structure
        if is_clustered and (reshape == 'auto' or reshape is None):
            # Clustered data: use (time, cluster) as natural 2D heatmap axes
            heatmap_dims = ['time', 'cluster']
        elif reshape and reshape != 'auto' and 'time' in da.dims:
            # Non-clustered with explicit reshape: reshape time to (day, hour) etc.
            # Extra dims will be handled via facet/animation or dropped
            da = _reshape_time_for_heatmap(da, reshape)
            heatmap_dims = ['timestep', 'timeframe']
        elif reshape == 'auto' and 'time' in da.dims and not is_clustered:
            # Auto mode for non-clustered: use default ('D', 'h') reshape
            # Extra dims will be handled via facet/animation or dropped
            da = _reshape_time_for_heatmap(da, ('D', 'h'))
            heatmap_dims = ['timestep', 'timeframe']
        elif has_multiple_vars:
            # Can't reshape but have multiple vars: use variable + time as heatmap axes
            heatmap_dims = ['variable', 'time']
            # variable is now a heatmap dim, use period/scenario for facet/animation
            actual_facet, _, actual_animation = _resolve_auto_facets(
                da.to_dataset(name='value'), facet_col, None, animation_frame
            )
        else:
            # Fallback: use first two available dimensions
            available_dims = [d for d in da.dims if da.sizes[d] > 1]
            if len(available_dims) >= 2:
                heatmap_dims = available_dims[:2]
            elif 'time' in da.dims:
                heatmap_dims = ['time']
            else:
                heatmap_dims = list(da.dims)[:1]

        # Keep only dims we need
        keep_dims = set(heatmap_dims) | {d for d in [actual_facet, actual_animation] if d is not None}
        for dim in [d for d in da.dims if d not in keep_dims]:
            da = da.isel({dim: 0}, drop=True) if da.sizes[dim] > 1 else da.squeeze(dim, drop=True)

        # Transpose to expected order
        dim_order = heatmap_dims + [d for d in [actual_facet, actual_animation] if d]
        da = da.transpose(*dim_order)

        # Clear name for multiple variables (colorbar would show first var's name)
        if has_multiple_vars:
            da = da.rename('')

        fig = _heatmap_figure(
            da,
            colors=colors,
            facet_col=actual_facet,
            animation_frame=actual_animation,
            **plotly_kwargs,
        )

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        reshaped_ds = da.to_dataset(name='value') if isinstance(da, xr.DataArray) else da
        return PlotResult(data=reshaped_ds, figure=fig)

    def flows(
        self,
        *,
        start: str | list[str] | None = None,
        end: str | list[str] | None = None,
        component: str | list[str] | None = None,
        select: SelectType | None = None,
        unit: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        colors: ColorType | None = None,
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        show: bool | None = None,
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
            facet_col: Dimension for column facets. 'auto' uses first available.
            facet_row: Dimension for row facets.
            animation_frame: Dimension for animation slider.
            show: Whether to display.

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
        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            ds, facet_col, facet_row, animation_frame
        )

        # Get unit label from first data variable's attributes
        unit_label = ''
        if ds.data_vars:
            first_var = next(iter(ds.data_vars))
            unit_label = ds[first_var].attrs.get('unit', '')

        fig = _create_line(
            ds,
            colors=colors,
            title=f'Flows [{unit_label}]' if unit_label else 'Flows',
            facet_col=actual_facet_col,
            facet_row=actual_facet_row,
            animation_frame=actual_anim,
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
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot investment sizes (capacities) of flows.

        Args:
            max_size: Maximum size to include (filters defaults).
            select: xarray-style selection.
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            facet_col: Dimension for column facets. 'auto' uses first available.
            facet_row: Dimension for row facets.
            animation_frame: Dimension for animation slider.
            show: Whether to display.

        Returns:
            PlotResult with size data.
        """
        self._stats._require_solution()
        ds = self._stats.sizes

        ds = _apply_selection(ds, select)

        if max_size is not None and ds.data_vars:
            valid_labels = [lbl for lbl in ds.data_vars if float(ds[lbl].max()) < max_size]
            ds = ds[valid_labels]

        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            ds, facet_col, facet_row, animation_frame
        )

        df = _dataset_to_long_df(ds)
        if df.empty:
            fig = go.Figure()
        else:
            variables = df['variable'].unique().tolist()
            color_map = process_colors(colors, variables)
            fig = px.bar(
                df,
                x='variable',
                y='value',
                color='variable',
                facet_col=actual_facet_col,
                facet_row=actual_facet_row,
                animation_frame=actual_anim,
                color_discrete_map=color_map,
                title='Investment Sizes',
                labels={'variable': 'Flow', 'value': 'Size'},
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
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        show: bool | None = None,
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
            facet_col: Dimension for column facets. 'auto' uses first available.
            facet_row: Dimension for row facets.
            animation_frame: Dimension for animation slider.
            show: Whether to display.

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

        if 'time' not in ds.dims:
            raise ValueError('Duration curve requires time dimension')

        def sort_descending(arr: np.ndarray) -> np.ndarray:
            return np.sort(arr)[::-1]

        result_ds = xr.apply_ufunc(
            sort_descending,
            ds,
            input_core_dims=[['time']],
            output_core_dims=[['time']],
            vectorize=True,
        )

        duration_name = 'duration_pct' if normalize else 'duration'
        result_ds = result_ds.rename({'time': duration_name})

        n_timesteps = result_ds.sizes[duration_name]
        duration_coord = np.linspace(0, 100, n_timesteps) if normalize else np.arange(n_timesteps)
        result_ds = result_ds.assign_coords({duration_name: duration_coord})

        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            result_ds, facet_col, facet_row, animation_frame
        )

        # Get unit label from first data variable's attributes
        unit_label = ''
        if ds.data_vars:
            first_var = next(iter(ds.data_vars))
            unit_label = ds[first_var].attrs.get('unit', '')

        fig = _create_line(
            result_ds,
            colors=colors,
            title=f'Duration Curve [{unit_label}]' if unit_label else 'Duration Curve',
            facet_col=actual_facet_col,
            facet_row=actual_facet_row,
            animation_frame=actual_anim,
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
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        show: bool | None = None,
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
            facet_col: Dimension for column facets. 'auto' uses first available.
            facet_row: Dimension for row facets.
            animation_frame: Dimension for animation slider.
            show: Whether to display.

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
        if aspect == 'total':
            effects_ds = self._stats.total_effects
        elif aspect == 'temporal':
            effects_ds = self._stats.temporal_effects
        elif aspect == 'periodic':
            effects_ds = self._stats.periodic_effects
        else:
            raise ValueError(f"Aspect '{aspect}' not valid. Choose from 'total', 'temporal', 'periodic'.")

        # Get available effects (data variables in the dataset)
        available_effects = list(effects_ds.data_vars)

        # Filter to specific effect if requested
        if effect is not None:
            if effect not in available_effects:
                raise ValueError(f"Effect '{effect}' not found. Available: {available_effects}")
            effects_to_plot = [effect]
        else:
            effects_to_plot = available_effects

        # Build a combined DataArray with effect dimension
        effect_arrays = []
        for eff in effects_to_plot:
            da = effects_ds[eff]
            if by == 'contributor':
                # Keep individual contributors (flows) - no groupby
                effect_arrays.append(da.expand_dims(effect=[eff]))
            else:
                # Group by component (sum over contributor within each component)
                da_grouped = da.groupby('component').sum()
                effect_arrays.append(da_grouped.expand_dims(effect=[eff]))

        combined = xr.concat(effect_arrays, dim='effect')

        # Apply selection
        combined = _apply_selection(combined.to_dataset(name='value'), select)['value']

        # Group by the specified dimension
        if by is None:
            # Aggregate totals per effect - sum over all dimensions except effect
            if 'time' in combined.dims:
                combined = combined.sum(dim='time')
            if 'component' in combined.dims:
                combined = combined.sum(dim='component')
            if 'contributor' in combined.dims:
                combined = combined.sum(dim='contributor')
            x_col = 'effect'
            color_col = 'effect'
        elif by == 'component':
            # Sum over time if present
            if 'time' in combined.dims:
                combined = combined.sum(dim='time')
            x_col = 'component'
            color_col = 'effect' if len(effects_to_plot) > 1 else 'component'
        elif by == 'contributor':
            # Sum over time if present
            if 'time' in combined.dims:
                combined = combined.sum(dim='time')
            x_col = 'contributor'
            color_col = 'effect' if len(effects_to_plot) > 1 else 'contributor'
        elif by == 'time':
            if 'time' not in combined.dims:
                raise ValueError(f"Cannot plot by 'time' for aspect '{aspect}' - no time dimension.")
            # Sum over components or contributors
            if 'component' in combined.dims:
                combined = combined.sum(dim='component')
            if 'contributor' in combined.dims:
                combined = combined.sum(dim='contributor')
            x_col = 'time'
            color_col = 'effect' if len(effects_to_plot) > 1 else None
        else:
            raise ValueError(f"'by' must be one of 'component', 'contributor', 'time', or None, got {by!r}")

        # Resolve facets
        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            combined.to_dataset(name='value'), facet_col, facet_row, animation_frame
        )

        # Convert to DataFrame for plotly express
        df = combined.to_dataframe(name='value').reset_index()

        # Build color map
        if color_col and color_col in df.columns:
            color_items = df[color_col].unique().tolist()
            color_map = process_colors(colors, color_items)
        else:
            color_map = None

        # Build title with unit if single effect
        effect_label = effect if effect else 'Effects'
        if effect and effect in effects_ds:
            unit_label = effects_ds[effect].attrs.get('unit', '')
            title = f'{effect_label} [{unit_label}]' if unit_label else effect_label
        else:
            title = effect_label
        title = f'{title} ({aspect})' if by is None else f'{title} ({aspect}) by {by}'

        fig = px.bar(
            df,
            x=x_col,
            y='value',
            color=color_col,
            color_discrete_map=color_map,
            facet_col=actual_facet_col,
            facet_row=actual_facet_row,
            animation_frame=actual_anim,
            title=title,
            **plotly_kwargs,
        )
        fig.update_layout(bargap=0, bargroupgap=0)
        fig.update_traces(marker_line_width=0)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=combined.to_dataset(name=aspect), figure=fig)

    def charge_states(
        self,
        storages: str | list[str] | None = None,
        *,
        select: SelectType | None = None,
        colors: ColorType | None = None,
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot storage charge states over time.

        Args:
            storages: Storage label(s) to plot. If None, plots all storages.
            select: xarray-style selection.
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            facet_col: Dimension for column facets. 'auto' uses first available.
            facet_row: Dimension for row facets.
            animation_frame: Dimension for animation slider.
            show: Whether to display.

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
        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            ds, facet_col, facet_row, animation_frame
        )

        fig = _create_line(
            ds,
            colors=colors,
            title='Storage Charge States',
            facet_col=actual_facet_col,
            facet_row=actual_facet_row,
            animation_frame=actual_anim,
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
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        show: bool | None = None,
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
            facet_col: Dimension for column facets. 'auto' uses first available.
            facet_row: Dimension for row facets.
            animation_frame: Dimension for animation slider.
            show: Whether to display.

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
        input_labels = [f.label_full for f in component.inputs]
        output_labels = [f.label_full for f in component.outputs]
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
        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            ds, facet_col, facet_row, animation_frame
        )

        # Build color map
        flow_labels = [lbl for lbl in ds.data_vars if lbl != 'charge_state']
        if colors is None:
            colors = self._get_color_map_for_balance(storage, flow_labels)
        color_map = process_colors(colors, flow_labels)
        color_map['charge_state'] = 'black'

        # Convert to long-form DataFrame
        df = _dataset_to_long_df(ds)

        # Create figure with facets using px.bar for flows
        flow_df = df[df['variable'] != 'charge_state']
        charge_df = df[df['variable'] == 'charge_state']

        fig = px.bar(
            flow_df,
            x='time',
            y='value',
            color='variable',
            facet_col=actual_facet_col,
            facet_row=actual_facet_row,
            animation_frame=actual_anim,
            color_discrete_map=color_map,
            title=f'{storage} Operation ({unit})',
            **plotly_kwargs,
        )
        fig.update_layout(bargap=0, bargroupgap=0)
        fig.update_traces(marker_line_width=0)

        # Add charge state as line on secondary y-axis
        if not charge_df.empty:
            # Create line figure with same facets to get matching trace structure
            line_fig = px.line(
                charge_df,
                x='time',
                y='value',
                facet_col=actual_facet_col,
                facet_row=actual_facet_row,
                animation_frame=actual_anim,
            )

            # Get the primary y-axes from the bar figure to create matching secondary axes
            # px creates axes named: yaxis, yaxis2, yaxis3, etc.
            primary_yaxes = [key for key in fig.layout if key.startswith('yaxis')]

            # For each primary y-axis, create a secondary y-axis
            for i, primary_key in enumerate(sorted(primary_yaxes, key=lambda x: int(x[5:]) if x[5:] else 0)):
                # Determine secondary axis name (y -> y2, y2 -> y3 pattern won't work)
                # Instead use a consistent offset: yaxis -> yaxis10, yaxis2 -> yaxis11, etc.
                primary_num = primary_key[5:] if primary_key[5:] else '1'
                secondary_num = int(primary_num) + 100  # Use high offset to avoid conflicts
                secondary_key = f'yaxis{secondary_num}'
                secondary_anchor = f'x{primary_num}' if primary_num != '1' else 'x'

                fig.layout[secondary_key] = dict(
                    overlaying=f'y{primary_num}' if primary_num != '1' else 'y',
                    side='right',
                    showgrid=False,
                    title='Charge State' if i == len(primary_yaxes) - 1 else None,
                    anchor=secondary_anchor,
                )

            # Add line traces with correct axis assignments
            for i, trace in enumerate(line_fig.data):
                # Map trace index to secondary y-axis
                primary_num = i + 1 if i > 0 else 1
                secondary_yaxis = f'y{primary_num + 100}'

                trace.name = 'charge_state'
                trace.line = dict(color=charge_state_color, width=2)
                trace.yaxis = secondary_yaxis
                trace.showlegend = i == 0  # Only show legend for first trace
                trace.legendgroup = 'charge_state'
                fig.add_trace(trace)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=ds, figure=fig)
