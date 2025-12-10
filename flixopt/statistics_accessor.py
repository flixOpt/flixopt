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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr

from .color_processing import ColorType, process_colors
from .config import CONFIG

if TYPE_CHECKING:
    from pathlib import Path

    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')

# Type aliases
SelectType = dict[str, Any]
"""xarray-style selection dict: {'time': slice(...), 'scenario': 'base'}"""

FilterType = str | list[str]
"""For include/exclude filtering: 'Boiler' or ['Boiler', 'CHP']"""


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


@dataclass
class PlotResult:
    """Container returned by all plot methods. Holds both data and figure.

    Attributes:
        data: Prepared xarray Dataset used for the plot.
        figure: Plotly figure object.
    """

    data: xr.Dataset
    figure: go.Figure

    def show(self) -> PlotResult:
        """Display the figure. Returns self for chaining."""
        self.figure.show()
        return self

    def update(self, **layout_kwargs: Any) -> PlotResult:
        """Update figure layout. Returns self for chaining."""
        self.figure.update_layout(**layout_kwargs)
        return self

    def update_traces(self, **trace_kwargs: Any) -> PlotResult:
        """Update figure traces. Returns self for chaining."""
        self.figure.update_traces(**trace_kwargs)
        return self

    def to_html(self, path: str | Path) -> PlotResult:
        """Save figure as interactive HTML. Returns self for chaining."""
        self.figure.write_html(str(path))
        return self

    def to_image(self, path: str | Path, **kwargs: Any) -> PlotResult:
        """Save figure as static image. Returns self for chaining."""
        self.figure.write_image(str(path), **kwargs)
        return self

    def to_csv(self, path: str | Path, **kwargs: Any) -> PlotResult:
        """Export the underlying data to CSV. Returns self for chaining."""
        self.data.to_dataframe().to_csv(path, **kwargs)
        return self

    def to_netcdf(self, path: str | Path, **kwargs: Any) -> PlotResult:
        """Export the underlying data to netCDF. Returns self for chaining."""
        self.data.to_netcdf(path, **kwargs)
        return self


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


def _apply_selection(ds: xr.Dataset, select: SelectType | None) -> xr.Dataset:
    """Apply xarray-style selection to dataset."""
    if select is None:
        return ds
    valid_select = {k: v for k, v in select.items() if k in ds.dims or k in ds.coords}
    if valid_select:
        ds = ds.sel(valid_select)
    return ds


def _resolve_facets(
    ds: xr.Dataset,
    facet_col: str | None,
    facet_row: str | None,
) -> tuple[str | None, str | None]:
    """Resolve facet dimensions, returning None if not present in data."""
    actual_facet_col = facet_col if facet_col and facet_col in ds.dims else None
    actual_facet_row = facet_row if facet_row and facet_row in ds.dims else None
    return actual_facet_col, actual_facet_row


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
        """All flow rates as a Dataset with flow labels as variable names."""
        self._require_solution()
        if self._flow_rates is None:
            flow_rate_vars = [v for v in self._fs.solution.data_vars if v.endswith('|flow_rate')]
            self._flow_rates = xr.Dataset({v.replace('|flow_rate', ''): self._fs.solution[v] for v in flow_rate_vars})
        return self._flow_rates

    @property
    def flow_hours(self) -> xr.Dataset:
        """All flow hours (energy) as a Dataset with flow labels as variable names."""
        self._require_solution()
        if self._flow_hours is None:
            hours = self._fs.hours_per_timestep
            self._flow_hours = self.flow_rates * hours
        return self._flow_hours

    @property
    def sizes(self) -> xr.Dataset:
        """All flow sizes as a Dataset with flow labels as variable names."""
        self._require_solution()
        if self._sizes is None:
            # Get flow labels to filter only flow sizes (not storage capacity sizes)
            flow_labels = set(self._fs.flows.keys())
            size_vars = [
                v for v in self._fs.solution.data_vars if v.endswith('|size') and v.replace('|size', '') in flow_labels
            ]
            self._sizes = xr.Dataset({v.replace('|size', ''): self._fs.solution[v] for v in size_vars})
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
                            # For total mode, sum temporal over time
                            if mode == 'total' and current_mode == 'temporal' and 'time' in da.dims:
                                da = da.sum('time')
                            if share_total is None:
                                share_total = da
                            else:
                                share_total = share_total + da

                # If no share found, use NaN template
                if share_total is None:
                    share_total = xr.full_like(template, np.nan, dtype=float)

                contributor_arrays.append(share_total.expand_dims(contributor=[contributor]))

            # Concatenate all contributors for this effect
            ds[effect] = xr.concat(contributor_arrays, dim='contributor', coords='minimal', join='outer').rename(effect)

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
                    logger.warning(
                        f'Results for {effect}({mode}) in effects_dataset doesnt match {label}\n{computed=}\n, {found=}'
                    )

        return ds


# --- Statistics Plot Accessor ---


class StatisticsPlotAccessor:
    """Plot accessor for statistics. Access via ``flow_system.statistics.plot``.

    All methods return PlotResult with both data and figure.
    """

    def __init__(self, statistics: StatisticsAccessor) -> None:
        self._stats = statistics
        self._fs = statistics._fs

    def _get_color_map_for_balance(self, node: str, flow_labels: list[str]) -> dict[str, str]:
        """Build color map for balance plot.

        - Bus balance: colors from component.color
        - Component balance: colors from flow's carrier

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

        for label in flow_labels:
            if is_bus:
                color = self._fs.components[self._fs.flows[label].component].color
            else:
                carrier = self._fs.get_carrier(label)  # get_carrier accepts flow labels
                color = carrier.color if carrier else None

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
        facet_col: str | None = 'period',
        facet_row: str | None = 'scenario',
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
            facet_col: Dimension for column facets.
            facet_row: Dimension for row facets.
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
        actual_facet_col, actual_facet_row = _resolve_facets(ds, facet_col, facet_row)

        # Build color map from Element.color attributes if no colors specified
        if colors is None:
            colors = self._get_color_map_for_balance(node, list(ds.data_vars))

        fig = _create_stacked_bar(
            ds,
            colors=colors,
            title=f'{node} ({unit})',
            facet_col=actual_facet_col,
            facet_row=actual_facet_row,
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
        reshape: tuple[str, str] | None = ('D', 'h'),
        colors: str | list[str] | None = None,
        facet_col: str | None = 'period',
        animation_frame: str | None = 'scenario',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot heatmap of time series data.

        Time is reshaped into 2D (e.g., days × hours) when possible. Multiple variables
        are shown as facets. If too many dimensions exist to display without data loss,
        reshaping is skipped and variables are shown on the y-axis with time on x-axis.

        Args:
            variables: Flow label(s) or variable name(s). Flow labels like 'Boiler(Q_th)'
                are automatically resolved to 'Boiler(Q_th)|flow_rate'. Full variable
                names like 'Storage|charge_state' are used as-is.
            select: xarray-style selection, e.g. {'scenario': 'Base Case'}.
            reshape: Time reshape frequencies as (outer, inner), e.g. ('D', 'h') for
                    days × hours. Set to None to disable reshaping.
            colors: Colorscale name (str) or list of colors for heatmap coloring.
                Dicts are not supported for heatmaps (use str or list[str]).
            facet_col: Dimension for subplot columns (default: 'period').
                      With multiple variables, 'variable' is used instead.
            animation_frame: Dimension for animation slider (default: 'scenario').
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

        # Determine facet and animation from available dims
        has_multiple_vars = 'variable' in da.dims and da.sizes['variable'] > 1

        if has_multiple_vars:
            actual_facet = 'variable'
            actual_animation = (
                animation_frame
                if animation_frame in da.dims
                else (facet_col if facet_col in da.dims and da.sizes.get(facet_col, 1) > 1 else None)
            )
        else:
            actual_facet = facet_col if facet_col in da.dims and da.sizes.get(facet_col, 0) > 1 else None
            actual_animation = (
                animation_frame if animation_frame in da.dims and da.sizes.get(animation_frame, 0) > 1 else None
            )

        # Count non-time dims with size > 1 (these need facet/animation slots)
        extra_dims = [d for d in da.dims if d != 'time' and da.sizes[d] > 1]
        used_slots = len([d for d in [actual_facet, actual_animation] if d])
        would_drop = len(extra_dims) > used_slots

        # Reshape time only if we wouldn't lose data (all extra dims fit in facet + animation)
        if reshape and 'time' in da.dims and not would_drop:
            da = _reshape_time_for_heatmap(da, reshape)
            heatmap_dims = ['timestep', 'timeframe']
        elif has_multiple_vars:
            # Can't reshape but have multiple vars: use variable + time as heatmap axes
            heatmap_dims = ['variable', 'time']
            # variable is now a heatmap dim, use period/scenario for facet/animation
            actual_facet = facet_col if facet_col in da.dims and da.sizes.get(facet_col, 0) > 1 else None
            actual_animation = (
                animation_frame if animation_frame in da.dims and da.sizes.get(animation_frame, 0) > 1 else None
            )
        else:
            heatmap_dims = ['time'] if 'time' in da.dims else list(da.dims)[:1]

        # Keep only dims we need
        keep_dims = set(heatmap_dims) | {actual_facet, actual_animation} - {None}
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
        facet_col: str | None = 'period',
        facet_row: str | None = 'scenario',
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
            facet_col: Dimension for column facets.
            facet_row: Dimension for row facets.
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
                comp_label = flow.component.label_full

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
        actual_facet_col, actual_facet_row = _resolve_facets(ds, facet_col, facet_row)

        fig = _create_line(
            ds,
            colors=colors,
            title=f'Flows ({unit})',
            facet_col=actual_facet_col,
            facet_row=actual_facet_row,
            **plotly_kwargs,
        )

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=ds, figure=fig)

    def _prepare_sankey_data(
        self,
        mode: Literal['flow_hours', 'sizes', 'peak_flow'],
        timestep: int | str | None,
        aggregate: Literal['sum', 'mean'],
        select: SelectType | None,
    ) -> tuple[xr.Dataset, str]:
        """Prepare data for Sankey diagram based on mode.

        Args:
            mode: What to display - flow_hours, sizes, or peak_flow.
            timestep: Specific timestep (only for flow_hours mode).
            aggregate: Aggregation method (only for flow_hours mode).
            select: xarray-style selection.

        Returns:
            Tuple of (prepared Dataset, title string).
        """
        if mode == 'sizes':
            ds = self._stats.sizes.copy()
            title = 'Investment Sizes (Capacities)'
        elif mode == 'peak_flow':
            ds = self._stats.flow_rates.copy()
            ds = _apply_selection(ds, select)
            if 'time' in ds.dims:
                ds = ds.max(dim='time')
            for dim in ['period', 'scenario']:
                if dim in ds.dims:
                    ds = ds.max(dim=dim)
            return ds, 'Peak Flow Rates'
        else:  # flow_hours
            ds = self._stats.flow_hours.copy()
            title = 'Energy Flow'

        # Apply weights for flow_hours
        if mode == 'flow_hours':
            if 'period' in ds.dims and self._fs.period_weights is not None:
                ds = ds * self._fs.period_weights
            if 'scenario' in ds.dims and self._fs.scenario_weights is not None:
                weights = self._fs.scenario_weights / self._fs.scenario_weights.sum()
                ds = ds * weights

        ds = _apply_selection(ds, select)

        # Time aggregation (only for flow_hours)
        if mode == 'flow_hours':
            if timestep is not None:
                if isinstance(timestep, int):
                    ds = ds.isel(time=timestep)
                else:
                    ds = ds.sel(time=timestep)
            elif 'time' in ds.dims:
                ds = getattr(ds, aggregate)(dim='time')

        # Collapse remaining dimensions
        for dim in ['period', 'scenario']:
            if dim in ds.dims:
                ds = ds.sum(dim=dim) if mode == 'flow_hours' else ds.max(dim=dim)

        return ds, title

    def _build_effects_sankey(
        self,
        select: SelectType | None,
        colors: ColorType | None,
        **plotly_kwargs: Any,
    ) -> tuple[go.Figure, xr.Dataset]:
        """Build Sankey diagram showing contributions from components to effects.

        Creates a Sankey with:
        - Left side: Components (grouped by type)
        - Right side: Effects (costs, CO2, etc.)
        - Links: Contributions from each component to each effect

        Args:
            select: xarray-style selection.
            colors: Color specification for nodes.
            **plotly_kwargs: Additional Plotly layout arguments.

        Returns:
            Tuple of (Plotly Figure, Dataset with link data).
        """
        total_effects = self._stats.total_effects

        # Collect all links: component -> effect
        nodes: set[str] = set()
        links: dict[str, list] = {'source': [], 'target': [], 'value': [], 'label': []}

        for effect_name in total_effects.data_vars:
            effect_data = total_effects[effect_name]
            effect_data = _apply_selection(effect_data, select)

            # Sum over any remaining dimensions
            for dim in ['period', 'scenario']:
                if dim in effect_data.dims:
                    effect_data = effect_data.sum(dim=dim)

            contributors = effect_data.coords['contributor'].values
            components = effect_data.coords['component'].values

            for contributor, component in zip(contributors, components, strict=False):
                value = float(effect_data.sel(contributor=contributor).values)
                if not np.isfinite(value) or abs(value) < 1e-6:
                    continue

                # Use component as source node, effect as target
                source = str(component)
                target = f'[{effect_name}]'  # Bracket notation to distinguish effects

                nodes.add(source)
                nodes.add(target)
                links['source'].append(source)
                links['target'].append(target)
                links['value'].append(abs(value))
                links['label'].append(f'{contributor} → {effect_name}: {value:.2f}')

        # Create figure
        node_list = list(nodes)
        node_indices = {n: i for i, n in enumerate(node_list)}

        color_map = process_colors(colors, node_list)
        node_colors = [color_map[node] for node in node_list]

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15, thickness=20, line=dict(color='black', width=0.5), label=node_list, color=node_colors
                    ),
                    link=dict(
                        source=[node_indices[s] for s in links['source']],
                        target=[node_indices[t] for t in links['target']],
                        value=links['value'],
                        label=links['label'],
                    ),
                )
            ]
        )
        fig.update_layout(title='Effect Contributions by Component', **plotly_kwargs)

        sankey_ds = xr.Dataset(
            {'value': ('link', links['value'])},
            coords={
                'link': range(len(links['value'])),
                'source': ('link', links['source']),
                'target': ('link', links['target']),
                'label': ('link', links['label']),
            },
        )

        return fig, sankey_ds

    def _build_sankey_links(
        self,
        ds: xr.Dataset,
        min_value: float = 1e-6,
    ) -> tuple[set[str], dict[str, list]]:
        """Build Sankey nodes and links from flow data.

        Args:
            ds: Dataset with flow values (one variable per flow).
            min_value: Minimum value threshold to include a link.

        Returns:
            Tuple of (nodes set, links dict with source/target/value/label).
        """
        nodes: set[str] = set()
        links: dict[str, list] = {'source': [], 'target': [], 'value': [], 'label': []}

        for flow in self._fs.flows.values():
            label = flow.label_full
            if label not in ds:
                continue
            value = float(ds[label].values)
            if abs(value) < min_value:
                continue

            # flow.bus and flow.component are already strings (bus label, component label_full)
            bus_label = flow.bus
            comp_label = flow.component

            if flow.is_input_in_component:
                source = bus_label
                target = comp_label
            else:
                source = comp_label
                target = bus_label

            nodes.add(source)
            nodes.add(target)
            links['source'].append(source)
            links['target'].append(target)
            links['value'].append(abs(value))
            links['label'].append(label)

        return nodes, links

    def _create_sankey_figure(
        self,
        nodes: set[str],
        links: dict[str, list],
        colors: ColorType | None,
        title: str,
        **plotly_kwargs: Any,
    ) -> go.Figure:
        """Create Plotly Sankey figure.

        Args:
            nodes: Set of node labels.
            links: Dict with source, target, value, label lists.
            colors: Color specification for nodes.
            title: Figure title.
            **plotly_kwargs: Additional Plotly layout arguments.

        Returns:
            Plotly Figure with Sankey diagram.
        """
        node_list = list(nodes)
        node_indices = {n: i for i, n in enumerate(node_list)}

        color_map = process_colors(colors, node_list)
        node_colors = [color_map[node] for node in node_list]

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15, thickness=20, line=dict(color='black', width=0.5), label=node_list, color=node_colors
                    ),
                    link=dict(
                        source=[node_indices[s] for s in links['source']],
                        target=[node_indices[t] for t in links['target']],
                        value=links['value'],
                        label=links['label'],
                    ),
                )
            ]
        )
        fig.update_layout(title=title, **plotly_kwargs)
        return fig

    def sankey(
        self,
        *,
        mode: Literal['flow_hours', 'sizes', 'peak_flow', 'effects'] = 'flow_hours',
        timestep: int | str | None = None,
        aggregate: Literal['sum', 'mean'] = 'sum',
        select: SelectType | None = None,
        max_size: float | None = None,
        colors: ColorType | None = None,
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot Sankey diagram of the flow system.

        Args:
            mode: What to display:
                - 'flow_hours': Energy/material amounts (default)
                - 'sizes': Investment capacities
                - 'peak_flow': Maximum flow rates
                - 'effects': Component contributions to all effects (costs, CO2, etc.)
            timestep: Specific timestep to show, or None for aggregation (flow_hours only).
            aggregate: How to aggregate if timestep is None ('sum' or 'mean', flow_hours only).
            select: xarray-style selection.
            max_size: Filter flows with sizes exceeding this value (sizes mode only).
            colors: Color specification for nodes (colorscale name, color list, or label-to-color dict).
            show: Whether to display.
            **plotly_kwargs: Additional arguments passed to Plotly layout.

        Returns:
            PlotResult with Sankey flow data and figure.

        Examples:
            >>> # Show energy flows (default)
            >>> flow_system.statistics.plot.sankey()
            >>> # Show investment sizes/capacities
            >>> flow_system.statistics.plot.sankey(mode='sizes')
            >>> # Show peak flow rates
            >>> flow_system.statistics.plot.sankey(mode='peak_flow')
            >>> # Show effect contributions (components -> effects like costs, CO2)
            >>> flow_system.statistics.plot.sankey(mode='effects')
        """
        self._stats._require_solution()

        if mode == 'effects':
            fig, sankey_ds = self._build_effects_sankey(select, colors, **plotly_kwargs)
        else:
            ds, title = self._prepare_sankey_data(mode, timestep, aggregate, select)

            # Apply max_size filter for sizes mode
            if max_size is not None and mode == 'sizes' and ds.data_vars:
                valid_labels = [lbl for lbl in ds.data_vars if float(ds[lbl].max()) < max_size]
                ds = ds[valid_labels]

            nodes, links = self._build_sankey_links(ds)
            fig = self._create_sankey_figure(nodes, links, colors, title, **plotly_kwargs)

            n_links = len(links['value'])
            sankey_ds = xr.Dataset(
                {'value': ('link', links['value'])},
                coords={
                    'link': range(n_links),
                    'source': ('link', links['source']),
                    'target': ('link', links['target']),
                    'label': ('link', links['label']),
                },
            )

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=sankey_ds, figure=fig)

    def sizes(
        self,
        *,
        max_size: float | None = 1e6,
        select: SelectType | None = None,
        colors: ColorType | None = None,
        facet_col: str | None = 'period',
        facet_row: str | None = 'scenario',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot investment sizes (capacities) of flows.

        Args:
            max_size: Maximum size to include (filters defaults).
            select: xarray-style selection.
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            facet_col: Dimension for column facets.
            facet_row: Dimension for row facets.
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

        actual_facet_col, actual_facet_row = _resolve_facets(ds, facet_col, facet_row)

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
        facet_col: str | None = 'period',
        facet_row: str | None = 'scenario',
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
            facet_col: Dimension for column facets.
            facet_row: Dimension for row facets.
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

        actual_facet_col, actual_facet_row = _resolve_facets(result_ds, facet_col, facet_row)

        fig = _create_line(
            result_ds,
            colors=colors,
            title='Duration Curve',
            facet_col=actual_facet_col,
            facet_row=actual_facet_row,
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
        by: Literal['component', 'contributor', 'time'] = 'component',
        select: SelectType | None = None,
        colors: ColorType | None = None,
        facet_col: str | None = 'period',
        facet_row: str | None = 'scenario',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot effect (cost, emissions, etc.) breakdown.

        Args:
            aspect: Which aspect to plot - 'total', 'temporal', or 'periodic'.
            effect: Specific effect name to plot (e.g., 'costs', 'CO2').
                    If None, plots all effects.
            by: Group by 'component', 'contributor' (individual flows), or 'time'.
            select: xarray-style selection.
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            facet_col: Dimension for column facets (ignored if not in data).
            facet_row: Dimension for row facets (ignored if not in data).
            show: Whether to display.

        Returns:
            PlotResult with effect breakdown data.

        Examples:
            >>> flow_system.statistics.plot.effects()  # Total of all effects by component
            >>> flow_system.statistics.plot.effects(effect='costs')  # Just costs
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
        if by == 'component':
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
            raise ValueError(f"'by' must be one of 'component', 'contributor', 'time', got {by!r}")

        # Resolve facets
        actual_facet_col, actual_facet_row = _resolve_facets(combined.to_dataset(name='value'), facet_col, facet_row)

        # Convert to DataFrame for plotly express
        df = combined.to_dataframe(name='value').reset_index()

        # Build color map
        if color_col and color_col in df.columns:
            color_items = df[color_col].unique().tolist()
            color_map = process_colors(colors, color_items)
        else:
            color_map = None

        # Build title
        effect_label = effect if effect else 'Effects'
        title = f'{effect_label} ({aspect}) by {by}'

        fig = px.bar(
            df,
            x=x_col,
            y='value',
            color=color_col,
            color_discrete_map=color_map,
            facet_col=actual_facet_col,
            facet_row=actual_facet_row,
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
