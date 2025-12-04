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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr

from . import plotting
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
    coord_cols = list(ds.coords.keys())
    return df.melt(id_vars=coord_cols, var_name=var_name, value_name=value_name)


def _create_stacked_bar(
    ds: xr.Dataset,
    colors: dict[str, str] | None,
    title: str,
    facet_col: str | None,
    facet_row: str | None,
    **plotly_kwargs: Any,
) -> go.Figure:
    """Create a stacked bar chart from xarray Dataset."""
    import plotly.express as px

    df = _dataset_to_long_df(ds)
    if df.empty:
        return go.Figure()
    x_col = 'time' if 'time' in df.columns else df.columns[0]
    variables = df['variable'].unique().tolist()
    color_map = {var: colors.get(var) for var in variables if colors and var in colors} or None
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
    colors: dict[str, str] | None,
    title: str,
    facet_col: str | None,
    facet_row: str | None,
    **plotly_kwargs: Any,
) -> go.Figure:
    """Create a line chart from xarray Dataset."""
    import plotly.express as px

    df = _dataset_to_long_df(ds)
    if df.empty:
        return go.Figure()
    x_col = 'time' if 'time' in df.columns else df.columns[0]
    variables = df['variable'].unique().tolist()
    color_map = {var: colors.get(var) for var in variables if colors and var in colors} or None
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
        ``effects_per_component`` : xr.Dataset
            Effect results aggregated by component.
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
        self._effects_per_component: xr.Dataset | None = None
        self._effect_share_factors: dict[str, dict] | None = None
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
            size_vars = [v for v in self._fs.solution.data_vars if v.endswith('|size')]
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
    def effects_per_component(self) -> xr.Dataset:
        """Effect results aggregated by component.

        Returns a dataset with:
        - 'temporal': temporal effects per component per timestep
        - 'periodic': periodic (investment) effects per component
        - 'total': sum of temporal and periodic effects per component

        Each variable has dimensions [time, period, scenario, component, effect]
        (missing dimensions are omitted).

        Returns:
            xr.Dataset with effect results aggregated by component.
        """
        self._require_solution()
        if self._effects_per_component is None:
            self._effects_per_component = xr.Dataset(
                {
                    mode: self._create_effects_dataset(mode).to_dataarray('effect', name=mode)
                    for mode in ['temporal', 'periodic', 'total']
                }
            )
            dim_order = ['time', 'period', 'scenario', 'component', 'effect']
            self._effects_per_component = self._effects_per_component.transpose(*dim_order, missing_dims='ignore')
        return self._effects_per_component

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

    def _compute_effect_total(
        self,
        element: str,
        effect: str,
        mode: Literal['temporal', 'periodic', 'total'] = 'total',
        include_flows: bool = False,
    ) -> xr.DataArray:
        """Calculate total effect for a specific element and effect.

        Computes total direct and indirect effects considering conversion factors.

        Args:
            element: The element identifier.
            effect: The effect identifier.
            mode: 'temporal', 'periodic', or 'total'.
            include_flows: Whether to include effects from flows connected to this element.

        Returns:
            xr.DataArray with total effects.
        """
        if effect not in self._fs.effects:
            raise ValueError(f'Effect {effect} is not available.')

        if mode == 'total':
            temporal = self._compute_effect_total(
                element=element, effect=effect, mode='temporal', include_flows=include_flows
            )
            periodic = self._compute_effect_total(
                element=element, effect=effect, mode='periodic', include_flows=include_flows
            )
            if periodic.isnull().all() and temporal.isnull().all():
                return xr.DataArray(np.nan)
            if temporal.isnull().all():
                return periodic.rename(f'{element}->{effect}')
            temporal = temporal.sum('time')
            if periodic.isnull().all():
                return temporal.rename(f'{element}->{effect}')
            return periodic + temporal

        total = xr.DataArray(0)
        share_exists = False

        relevant_conversion_factors = {
            key[0]: value for key, value in self.effect_share_factors[mode].items() if key[1] == effect
        }
        relevant_conversion_factors[effect] = 1  # Share to itself is 1

        for target_effect, conversion_factor in relevant_conversion_factors.items():
            label = f'{element}->{target_effect}({mode})'
            if label in self._fs.solution:
                share_exists = True
                da = self._fs.solution[label]
                total = da * conversion_factor + total

            if include_flows:
                if element not in self._fs.components:
                    raise ValueError(f'Only use Components when retrieving Effects including flows. Got {element}')
                comp = self._fs.components[element]
                flows = [f.label_full.split('|')[0] for f in comp.inputs + comp.outputs]
                for flow in flows:
                    label = f'{flow}->{target_effect}({mode})'
                    if label in self._fs.solution:
                        share_exists = True
                        da = self._fs.solution[label]
                        total = da * conversion_factor + total

        if not share_exists:
            total = xr.DataArray(np.nan)
        return total.rename(f'{element}->{effect}({mode})')

    def _create_template_for_mode(self, mode: Literal['temporal', 'periodic', 'total']) -> xr.DataArray:
        """Create a template DataArray with the correct dimensions for a given mode."""
        coords = {}
        if mode == 'temporal':
            coords['time'] = self._fs.timesteps_extra
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
        """Create dataset containing effect totals for all components (including their flows)."""
        template = self._create_template_for_mode(mode)
        ds = xr.Dataset()
        all_arrays: dict[str, list] = {}
        components_list = list(self._fs.components.keys())

        # Collect arrays for all effects and components
        for effect in self._fs.effects:
            effect_arrays = []
            for component in components_list:
                da = self._compute_effect_total(element=component, effect=effect, mode=mode, include_flows=True)
                effect_arrays.append(da)
            all_arrays[effect] = effect_arrays

        # Process all effects: expand scalar NaN arrays to match template dimensions
        for effect in self._fs.effects:
            dataarrays = all_arrays[effect]
            component_arrays = []

            for component, arr in zip(components_list, dataarrays, strict=False):
                # Expand scalar NaN arrays to match template dimensions
                if not arr.dims and np.isnan(arr.item()):
                    arr = xr.full_like(template, np.nan, dtype=float).rename(arr.name)
                component_arrays.append(arr.expand_dims(component=[component]))

            ds[effect] = xr.concat(component_arrays, dim='component', coords='minimal', join='outer').rename(effect)

        # Validation test
        suffix = {'temporal': '(temporal)|per_timestep', 'periodic': '(periodic)', 'total': ''}
        for effect in self._fs.effects:
            label = f'{effect}{suffix[mode]}'
            if label in self._fs.solution:
                computed = ds[effect].sum('component')
                found = self._fs.solution[label]
                if not np.allclose(computed.values, found.fillna(0).values):
                    logger.critical(
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

    def balance(
        self,
        node: str,
        *,
        select: SelectType | None = None,
        include: FilterType | None = None,
        exclude: FilterType | None = None,
        unit: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        colors: dict[str, str] | None = None,
        facet_col: str | None = 'scenario',
        facet_row: str | None = 'period',
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
            colors: Color overrides for flows.
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
        reshape: tuple[str, str] = ('D', 'h'),
        colorscale: str = 'viridis',
        facet_col: str | None = 'scenario',
        facet_row: str | None = 'period',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot heatmap of time series data with time reshaping.

        Args:
            variables: Variable name(s) from solution.
            select: xarray-style selection.
            reshape: How to reshape time axis - (outer, inner) frequency.
            colorscale: Plotly colorscale name.
            facet_col: Dimension for column facets.
            facet_row: Dimension for row facets.
            show: Whether to display.

        Returns:
            PlotResult with reshaped data.
        """
        solution = self._stats._require_solution()

        if isinstance(variables, str):
            variables = [variables]

        ds = solution[variables]
        ds = _apply_selection(ds, select)

        variable_names = list(ds.data_vars)
        dataarrays = [ds[var] for var in variable_names]
        da = xr.concat(dataarrays, dim=pd.Index(variable_names, name='variable'))

        actual_facet_col, actual_facet_row = _resolve_facets(da.to_dataset(name='value'), facet_col, facet_row)
        if len(variables) > 1 and actual_facet_col is None:
            actual_facet_col = 'variable'

        facet_by = [d for d in [actual_facet_col, actual_facet_row] if d] or None

        reshaped_data = plotting.reshape_data_for_heatmap(da, reshape)
        fig = plotting.heatmap_with_plotly(
            reshaped_data,
            colors=colorscale,
            facet_by=facet_by,
            reshape_time=None,
            **plotly_kwargs,
        )

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        reshaped_ds = (
            reshaped_data.to_dataset(name='value') if isinstance(reshaped_data, xr.DataArray) else reshaped_data
        )
        return PlotResult(data=reshaped_ds, figure=fig)

    def flows(
        self,
        *,
        start: str | list[str] | None = None,
        end: str | list[str] | None = None,
        component: str | list[str] | None = None,
        select: SelectType | None = None,
        unit: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        colors: dict[str, str] | None = None,
        facet_col: str | None = 'scenario',
        facet_row: str | None = 'period',
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
            colors: Color overrides.
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
                if starts and flow.bus_out.label not in starts:
                    continue
                if ends and flow.bus_in.label not in ends:
                    continue
                if components and flow.component.label not in components:
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

    def sankey(
        self,
        *,
        timestep: int | str | None = None,
        aggregate: Literal['sum', 'mean'] = 'sum',
        select: SelectType | None = None,
        colors: dict[str, str] | None = None,
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot Sankey diagram of energy/material flow hours.

        Args:
            timestep: Specific timestep to show, or None for aggregation.
            aggregate: How to aggregate if timestep is None.
            select: xarray-style selection.
            colors: Color overrides for flows/nodes.
            show: Whether to display.

        Returns:
            PlotResult with Sankey flow data.
        """
        self._stats._require_solution()

        ds = self._stats.flow_hours.copy()

        # Apply weights
        if 'period' in ds.dims and self._fs.period_weights is not None:
            ds = ds * self._fs.period_weights
        if 'scenario' in ds.dims and self._fs.scenario_weights is not None:
            weights = self._fs.scenario_weights / self._fs.scenario_weights.sum()
            ds = ds * weights

        ds = _apply_selection(ds, select)

        if timestep is not None:
            if isinstance(timestep, int):
                ds = ds.isel(time=timestep)
            else:
                ds = ds.sel(time=timestep)
        elif 'time' in ds.dims:
            ds = getattr(ds, aggregate)(dim='time')

        for dim in ['period', 'scenario']:
            if dim in ds.dims:
                ds = ds.sum(dim=dim)

        # Build Sankey
        nodes = set()
        links = {'source': [], 'target': [], 'value': [], 'label': []}

        for flow in self._fs.flows.values():
            label = flow.label_full
            if label not in ds:
                continue
            value = float(ds[label].values)
            if abs(value) < 1e-6:
                continue

            source = flow.bus_out.label if flow.bus_out else flow.component.label
            target = flow.bus_in.label if flow.bus_in else flow.component.label

            nodes.add(source)
            nodes.add(target)
            links['source'].append(source)
            links['target'].append(target)
            links['value'].append(abs(value))
            links['label'].append(label)

        node_list = list(nodes)
        node_indices = {n: i for i, n in enumerate(node_list)}

        node_colors = [colors.get(node) if colors else None for node in node_list]
        if any(node_colors):
            node_colors = [c if c else 'lightgray' for c in node_colors]
        else:
            node_colors = None

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
        fig.update_layout(title='Energy Flow Sankey', **plotly_kwargs)

        sankey_ds = xr.Dataset(
            {'value': ('link', links['value'])},
            coords={'link': links['label'], 'source': ('link', links['source']), 'target': ('link', links['target'])},
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
        colors: dict[str, str] | None = None,
        facet_col: str | None = 'scenario',
        facet_row: str | None = 'period',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot investment sizes (capacities) of flows.

        Args:
            max_size: Maximum size to include (filters defaults).
            select: xarray-style selection.
            colors: Color overrides.
            facet_col: Dimension for column facets.
            facet_row: Dimension for row facets.
            show: Whether to display.

        Returns:
            PlotResult with size data.
        """
        import plotly.express as px

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
            color_map = {var: colors.get(var) for var in variables if colors and var in colors} or None
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
        colors: dict[str, str] | None = None,
        facet_col: str | None = 'scenario',
        facet_row: str | None = 'period',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot load duration curves (sorted time series).

        Args:
            variables: Variable name(s) to plot.
            select: xarray-style selection.
            normalize: If True, normalize x-axis to 0-100%.
            colors: Color overrides.
            facet_col: Dimension for column facets.
            facet_row: Dimension for row facets.
            show: Whether to display.

        Returns:
            PlotResult with sorted duration curve data.
        """
        solution = self._stats._require_solution()

        if isinstance(variables, str):
            variables = [variables]

        ds = solution[variables]
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
