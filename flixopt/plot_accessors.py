"""Plot accessors for flixopt Results.

This module provides a user-friendly plotting API for optimization results.
All plot methods return a PlotResult object containing both the prepared
data (as an xarray Dataset) and the Plotly figure.

Example:
    >>> results = Results.from_file('results', 'optimization')
    >>> results.plot.balance('ElectricityBus')  # Quick plot
    >>> ds = results.plot.balance('Bus').data  # Get xarray data for export
    >>> results.plot.balance('Bus').update(title='Custom').show()  # Chain modifications
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

    from .results import Results, _NodeResults

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
        data: Prepared xarray Dataset used for the plot. Ready for export or custom plotting.
        figure: Plotly figure object. Can be modified with update_layout(), update_traces(), etc.

    Example:
        >>> result = results.plot.balance('Bus')
        >>> result.data.to_dataframe()  # Convert to DataFrame
        >>> result.data.to_netcdf('balance.nc')  # Export as netCDF
        >>> result.figure.update_layout(title='Custom')  # Modify figure
        >>> result.show()  # Display
    """

    data: xr.Dataset
    figure: go.Figure

    def show(self) -> PlotResult:
        """Display the figure. Returns self for chaining."""
        self.figure.show()
        return self

    def update(self, **layout_kwargs: Any) -> PlotResult:
        """Update figure layout. Returns self for chaining.

        Args:
            **layout_kwargs: Keyword arguments passed to fig.update_layout().

        Example:
            result.update(title='Custom Title', height=600).show()
        """
        self.figure.update_layout(**layout_kwargs)
        return self

    def update_traces(self, **trace_kwargs: Any) -> PlotResult:
        """Update figure traces. Returns self for chaining.

        Args:
            **trace_kwargs: Keyword arguments passed to fig.update_traces().
        """
        self.figure.update_traces(**trace_kwargs)
        return self

    def to_html(self, path: str | Path) -> PlotResult:
        """Save figure as interactive HTML. Returns self for chaining."""
        self.figure.write_html(str(path))
        return self

    def to_image(self, path: str | Path, **kwargs: Any) -> PlotResult:
        """Save figure as static image (png, svg, pdf, etc.). Returns self for chaining."""
        self.figure.write_image(str(path), **kwargs)
        return self

    def to_csv(self, path: str | Path, **kwargs: Any) -> PlotResult:
        """Export the underlying data to CSV. Returns self for chaining.

        Converts the xarray Dataset to a DataFrame before exporting.
        """
        self.data.to_dataframe().to_csv(path, **kwargs)
        return self

    def to_netcdf(self, path: str | Path, **kwargs: Any) -> PlotResult:
        """Export the underlying data to netCDF. Returns self for chaining."""
        self.data.to_netcdf(path, **kwargs)
        return self


def _filter_by_pattern(
    names: list[str],
    include: FilterType | None,
    exclude: FilterType | None,
) -> list[str]:
    """Filter names using substring matching.

    Args:
        names: List of names to filter.
        include: Only include names containing these substrings (OR logic).
        exclude: Exclude names containing these substrings.

    Returns:
        Filtered list of names.
    """
    result = names.copy()

    if include is not None:
        patterns = [include] if isinstance(include, str) else include
        result = [n for n in result if any(p in n for p in patterns)]

    if exclude is not None:
        patterns = [exclude] if isinstance(exclude, str) else exclude
        result = [n for n in result if not any(p in n for p in patterns)]

    return result


def _resolve_facet_animate(
    ds: xr.Dataset,
    facet_col: str | None,
    facet_row: str | None,
    animate_by: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Resolve facet/animate dimensions, returning None if not present in data."""
    actual_facet_col = facet_col if facet_col and facet_col in ds.dims else None
    actual_facet_row = facet_row if facet_row and facet_row in ds.dims else None
    actual_animate = animate_by if animate_by and animate_by in ds.dims else None
    return actual_facet_col, actual_facet_row, actual_animate


def _apply_selection(ds: xr.Dataset, select: SelectType | None) -> xr.Dataset:
    """Apply xarray-style selection to dataset."""
    if select is None:
        return ds

    # Filter select to only include dimensions that exist
    valid_select = {k: v for k, v in select.items() if k in ds.dims or k in ds.coords}
    if valid_select:
        ds = ds.sel(valid_select)
    return ds


def _merge_colors(
    global_colors: dict[str, str],
    override: dict[str, str] | None,
) -> dict[str, str]:
    """Merge global colors with per-plot overrides."""
    colors = global_colors.copy()
    if override:
        colors.update(override)
    return colors


def _dataset_to_long_df(ds: xr.Dataset, value_name: str = 'value', var_name: str = 'variable') -> pd.DataFrame:
    """Convert xarray Dataset to long-form DataFrame for plotly express.

    Each data variable becomes a row with its name in the 'variable' column.
    Handles scalar values (0-dimensional data) by creating single-row DataFrames.
    """
    if not ds.data_vars:
        return pd.DataFrame()

    # Check if all data variables are scalar (0-dimensional)
    if all(ds[var].ndim == 0 for var in ds.data_vars):
        # Build DataFrame manually for scalar values
        rows = []
        for var in ds.data_vars:
            rows.append({var_name: var, value_name: float(ds[var].values)})
        return pd.DataFrame(rows)

    # Convert to wide DataFrame, then melt to long form
    df = ds.to_dataframe().reset_index()
    coord_cols = list(ds.coords.keys())

    return df.melt(id_vars=coord_cols, var_name=var_name, value_name=value_name)


def _create_stacked_bar(
    ds: xr.Dataset,
    colors: dict[str, str],
    title: str,
    facet_col: str | None,
    facet_row: str | None,
    **plotly_kwargs: Any,
) -> go.Figure:
    """Create a stacked bar chart from xarray Dataset using plotly express."""
    import plotly.express as px

    df = _dataset_to_long_df(ds)
    if df.empty:
        return go.Figure()

    # Determine x-axis (time or first non-facet dimension)
    x_col = 'time' if 'time' in df.columns else df.columns[0]

    # Build color map from colors dict
    variables = df['variable'].unique().tolist()
    color_map = {var: colors.get(var, None) for var in variables}
    # Remove None values - let plotly use defaults
    color_map = {k: v for k, v in color_map.items() if v is not None} or None

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

    # Style as stacked bar
    fig.update_layout(barmode='relative', bargap=0, bargroupgap=0)
    fig.update_traces(marker_line_width=0)

    return fig


def _create_line(
    ds: xr.Dataset,
    colors: dict[str, str],
    title: str,
    facet_col: str | None,
    facet_row: str | None,
    **plotly_kwargs: Any,
) -> go.Figure:
    """Create a line chart from xarray Dataset using plotly express."""
    import plotly.express as px

    df = _dataset_to_long_df(ds)
    if df.empty:
        return go.Figure()

    # Determine x-axis (time or first dimension)
    x_col = 'time' if 'time' in df.columns else df.columns[0]

    # Build color map
    variables = df['variable'].unique().tolist()
    color_map = {var: colors.get(var, None) for var in variables}
    color_map = {k: v for k, v in color_map.items() if v is not None} or None

    fig = px.line(
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

    return fig


class PlotAccessor:
    """Plot accessor for Results. Access via results.plot.<method>()

    This accessor provides a unified interface for creating plots from
    optimization results. All methods return a PlotResult object containing
    both the prepared data and the Plotly figure.

    Example:
        >>> results.plot.balance('ElectricityBus')
        >>> results.plot.heatmap('Boiler|on')
        >>> results.plot.storage('Battery')
    """

    def __init__(self, results: Results):
        self._results = results

    @property
    def colors(self) -> dict[str, str]:
        """Global colors from Results."""
        return self._results.colors

    def balance(
        self,
        node: str,
        *,
        # Data selection (xarray-style)
        select: SelectType | None = None,
        # Flow filtering
        include: FilterType | None = None,
        exclude: FilterType | None = None,
        # Data transformation
        unit: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        aggregate: Literal['sum', 'mean', 'max', 'min'] | None = None,
        # Visual style
        colors: dict[str, str] | None = None,
        # Faceting
        facet_col: str | None = 'scenario',
        facet_row: str | None = 'period',
        # Display
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot node balance (inputs vs outputs) for a Bus or Component.

        Args:
            node: Label of the Bus or Component to plot.
            select: xarray-style selection dict. Supports:
                - Single values: {'scenario': 'base'}
                - Multiple values: {'scenario': ['base', 'high']}
                - Slices: {'time': slice('2024-01', '2024-06')}
            include: Only include flows containing these substrings (OR logic).
            exclude: Exclude flows containing these substrings.
            unit: 'flow_rate' (power, kW) or 'flow_hours' (energy, kWh).
            aggregate: Aggregate over time dimension before plotting.
            colors: Override colors (merged with global colors).
            facet_col: Dimension for column facets (ignored if not in data).
            facet_row: Dimension for row facets (ignored if not in data).
            show: Whether to display the plot. None uses CONFIG.Plotting.default_show.
            **plotly_kwargs: Passed to plotly express.

        Returns:
            PlotResult with .data (Dataset) and .figure (go.Figure).

        Examples:
            >>> results.plot.balance('ElectricityBus')
            >>> results.plot.balance('Bus', select={'time': slice('2024-01', '2024-03')})
            >>> results.plot.balance('Bus', include=['Boiler', 'CHP'], exclude=['Grid'])
            >>> ds = results.plot.balance('Bus').data  # Get data for export
        """
        # Get node results
        node_results = self._results[node]

        # Get all flow variable names
        all_flows = node_results.inputs + node_results.outputs

        # Apply include/exclude filtering
        filtered_flows = _filter_by_pattern(all_flows, include, exclude)

        if not filtered_flows:
            logger.warning(f'No flows remaining after filtering for node {node}')
            return PlotResult(data=xr.Dataset(), figure=go.Figure())

        # Determine which are inputs/outputs after filtering
        inputs = [f for f in filtered_flows if f in node_results.inputs]
        outputs = [f for f in filtered_flows if f in node_results.outputs]

        # Get the data
        ds = node_results.solution[filtered_flows]

        # Apply unit conversion
        if unit == 'flow_hours':
            ds = ds * self._results.hours_per_timestep
            ds = ds.rename_vars({var: var.replace('flow_rate', 'flow_hours') for var in ds.data_vars})
            # Update inputs/outputs lists with new names
            inputs = [i.replace('flow_rate', 'flow_hours') for i in inputs]
            outputs = [o.replace('flow_rate', 'flow_hours') for o in outputs]

        # Negate inputs (convention: inputs are negative in balance plot)
        for var in inputs:
            if var in ds:
                ds[var] = -ds[var]

        # Apply selection
        ds = _apply_selection(ds, select)

        # Apply aggregation
        if aggregate is not None:
            if 'time' in ds.dims:
                ds = getattr(ds, aggregate)(dim='time')

        # Resolve facets (ignore if dimension not present)
        actual_facet_col, actual_facet_row, _ = _resolve_facet_animate(ds, facet_col, facet_row, None)

        # Resolve colors
        merged_colors = _merge_colors(self.colors, colors)

        # Create figure
        fig = _create_stacked_bar(
            ds,
            colors=merged_colors,
            title=f'{node} ({unit})',
            facet_col=actual_facet_col,
            facet_row=actual_facet_row,
            **plotly_kwargs,
        )

        # Handle show
        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=ds, figure=fig)

    def heatmap(
        self,
        variables: str | list[str],
        *,
        # Data selection
        select: SelectType | None = None,
        # Reshaping
        reshape: tuple[str, str] = ('D', 'h'),
        # Visual style
        colorscale: str = 'viridis',
        # Faceting
        facet_col: str | None = 'scenario',
        facet_row: str | None = 'period',
        # Display
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot heatmap of time series data with time reshaping.

        Args:
            variables: Single variable name or list of variables.
            select: xarray-style selection.
            reshape: How to reshape time axis - (outer, inner) frequency.
                Common patterns:
                - ('D', 'h'): Days x Hours (default)
                - ('W', 'D'): Weeks x Days
                - ('MS', 'D'): Months x Days
            colorscale: Plotly colorscale name.
            facet_col: Dimension for column facets (ignored if not in data).
            facet_row: Dimension for row facets (ignored if not in data).
            show: Whether to display.

        Returns:
            PlotResult with reshaped data ready for heatmap.

        Examples:
            >>> results.plot.heatmap('Boiler|on')
            >>> results.plot.heatmap(['Boiler|on', 'CHP|on'], facet_col='variable')
        """
        # Normalize to list
        if isinstance(variables, str):
            variables = [variables]

        # Get the data as Dataset
        ds = self._results.solution[variables]

        # Apply selection
        ds = _apply_selection(ds, select)

        # Convert Dataset to DataArray with 'variable' dimension
        variable_names = list(ds.data_vars)
        dataarrays = [ds[var] for var in variable_names]
        # Use pd.Index to create a proper coordinate for the new dimension
        da = xr.concat(dataarrays, dim=pd.Index(variable_names, name='variable'))

        # Resolve facets (ignore if dimension not present)
        actual_facet_col, actual_facet_row, _ = _resolve_facet_animate(
            da.to_dataset(name='value'), facet_col, facet_row, None
        )

        # For multiple variables, auto-facet by variable if no facet specified
        if len(variables) > 1 and actual_facet_col is None:
            actual_facet_col = 'variable'

        # Build facet_by list
        facet_by = []
        if actual_facet_col:
            facet_by.append(actual_facet_col)
        if actual_facet_row:
            facet_by.append(actual_facet_row)
        facet_by = facet_by if facet_by else None

        # Reshape data for heatmap
        reshaped_data = plotting.reshape_data_for_heatmap(da, reshape)

        # Create heatmap figure
        fig = plotting.heatmap_with_plotly(
            reshaped_data,
            colors=colorscale,
            facet_by=facet_by,
            reshape_time=None,  # Already reshaped above
            **plotly_kwargs,
        )

        # Handle show
        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        # Convert DataArray to Dataset for consistent return type
        if isinstance(reshaped_data, xr.DataArray):
            reshaped_ds = reshaped_data.to_dataset(name='value')
        else:
            reshaped_ds = reshaped_data

        return PlotResult(data=reshaped_ds, figure=fig)

    def storage(
        self,
        component: str,
        *,
        # Data selection
        select: SelectType | None = None,
        # Visual style
        colors: dict[str, str] | None = None,
        charge_state_color: str = 'black',
        # Faceting
        facet_col: str | None = 'scenario',
        facet_row: str | None = 'period',
        # Display
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot storage component with charge state overlaid on flow balance.

        Shows charging/discharging flows as stacked bars and the charge state
        as an overlaid line.

        Args:
            component: Storage component label.
            select: xarray-style selection.
            colors: Override colors for flows.
            charge_state_color: Color for the charge state line.
            facet_col: Dimension for column facets (ignored if not in data).
            facet_row: Dimension for row facets (ignored if not in data).
            show: Whether to display.

        Returns:
            PlotResult with combined storage data (flows + charge state).
        """
        comp_results = self._results[component]

        if not hasattr(comp_results, 'is_storage') or not comp_results.is_storage:
            raise ValueError(f'{component} is not a storage component')

        # Get node balance (flows) with last timestep for proper alignment
        flows_ds = comp_results.node_balance(with_last_timestep=True).fillna(0)
        charge_state_var = f'{component}|charge_state'
        charge_state_da = comp_results.charge_state

        # Apply selection
        flows_ds = _apply_selection(flows_ds, select)
        charge_state_da = _apply_selection(charge_state_da, select)

        # Resolve facets (ignore if dimension not present)
        actual_facet_col, actual_facet_row, _ = _resolve_facet_animate(flows_ds, facet_col, facet_row, None)

        # Merge colors
        merged_colors = _merge_colors(self.colors, colors)

        # Create figure for flows (stacked bars)
        fig = _create_stacked_bar(
            flows_ds,
            colors=merged_colors,
            title=f'{component} Storage',
            facet_col=actual_facet_col,
            facet_row=actual_facet_row,
            **plotly_kwargs,
        )

        # Create figure for charge state (line overlay)
        charge_state_ds = xr.Dataset({charge_state_var: charge_state_da})
        charge_state_fig = _create_line(
            charge_state_ds,
            colors={},
            title='',
            facet_col=actual_facet_col,
            facet_row=actual_facet_row,
            **plotly_kwargs,
        )

        # Add charge state traces to the main figure
        for trace in charge_state_fig.data:
            trace.line.width = 2
            trace.line.shape = 'linear'
            trace.line.color = charge_state_color
            fig.add_trace(trace)

        # Combine data for return
        combined_ds = flows_ds.copy()
        combined_ds[charge_state_var] = charge_state_da

        # Handle show
        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=combined_ds, figure=fig)

    def flows(
        self,
        *,
        # Flow filtering
        start: str | list[str] | None = None,
        end: str | list[str] | None = None,
        component: str | list[str] | None = None,
        # Data selection
        select: SelectType | None = None,
        # Transformation
        unit: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        aggregate: Literal['sum', 'mean', 'max', 'min'] | None = None,
        # Visual style
        colors: dict[str, str] | None = None,
        # Faceting
        facet_col: str | None = 'scenario',
        facet_row: str | None = 'period',
        # Display
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
            aggregate: Aggregate over time.
            colors: Override colors.
            facet_col: Dimension for column facets (ignored if not in data).
            facet_row: Dimension for row facets (ignored if not in data).
            show: Whether to display.

        Returns:
            PlotResult with flow data.

        Examples:
            >>> results.plot.flows(start='ElectricityBus')
            >>> results.plot.flows(component='Boiler')
            >>> results.plot.flows(unit='flow_hours', aggregate='sum')
        """
        # Get flow rates using existing method
        if unit == 'flow_rate':
            da = self._results.flow_rates(start=start, end=end, component=component)
        else:
            da = self._results.flow_hours(start=start, end=end, component=component)

        # Apply selection
        if select:
            valid_select = {k: v for k, v in select.items() if k in da.dims or k in da.coords}
            if valid_select:
                da = da.sel(valid_select)

        # Apply aggregation
        if aggregate is not None:
            if 'time' in da.dims:
                da = getattr(da, aggregate)(dim='time')

        # Convert DataArray to Dataset for plotting (each flow as a variable)
        # First, unstack the flow dimension into separate variables
        flow_labels = da.coords['flow'].values.tolist()
        ds = xr.Dataset({label: da.sel(flow=label, drop=True) for label in flow_labels})

        # Resolve facets (ignore if dimension not present)
        actual_facet_col, actual_facet_row, _ = _resolve_facet_animate(ds, facet_col, facet_row, None)

        # Merge colors
        merged_colors = _merge_colors(self.colors, colors)

        # Create figure
        fig = _create_line(
            ds,
            colors=merged_colors,
            title=f'Flows ({unit})',
            facet_col=actual_facet_col,
            facet_row=actual_facet_row,
            **plotly_kwargs,
        )

        # Handle show
        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        # Return Dataset (ds has each flow as a variable)
        return PlotResult(data=ds, figure=fig)

    def compare(
        self,
        elements: list[str],
        *,
        variable: str = 'flow_rate',
        # Data selection
        select: SelectType | None = None,
        # Visual style
        mode: Literal['overlay', 'facet'] = 'overlay',
        colors: dict[str, str] | None = None,
        # Display
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Compare multiple elements side-by-side or overlaid.

        Args:
            elements: List of element labels to compare.
            variable: Which variable to compare (suffix like 'flow_rate', 'on', etc.).
            select: xarray-style selection.
            mode: 'overlay' (same axes) or 'facet' (subplots).
            colors: Override colors.
            show: Whether to display.

        Returns:
            PlotResult with comparison data.

        Examples:
            >>> results.plot.compare(['Boiler', 'CHP', 'HeatPump'], variable='on')
        """
        # Collect data from each element
        datasets = {}
        for element in elements:
            elem_results = self._results[element]
            # Find variable matching the suffix
            matching_vars = [v for v in elem_results.solution.data_vars if variable in v]
            if matching_vars:
                # Take first match, rename to element name
                var_name = matching_vars[0]
                datasets[element] = elem_results.solution[var_name].rename(element)

        if not datasets:
            logger.warning(f'No matching variables found for {variable} in elements {elements}')
            return PlotResult(data=xr.Dataset(), figure=go.Figure())

        # Merge into single dataset
        ds = xr.merge([da.to_dataset(name=name) for name, da in datasets.items()])

        # Apply selection
        ds = _apply_selection(ds, select)

        # Merge colors
        merged_colors = _merge_colors(self.colors, colors)

        # Create figure
        # For facet mode, convert Dataset to DataArray with 'element' dimension
        if mode == 'facet':
            # Stack variables into a single DataArray with 'element' dimension
            da_list = [ds[var].expand_dims(element=[var]) for var in ds.data_vars]
            stacked = xr.concat(da_list, dim='element')
            plot_data = stacked.to_dataset(name='value')
            facet_by = 'element'
        else:
            plot_data = ds
            facet_by = None

        fig = plotting.with_plotly(
            plot_data,
            mode='line',
            colors=merged_colors,
            title=f'Comparison: {variable}',
            facet_by=facet_by,
            **plotly_kwargs,
        )

        # Handle show
        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=ds, figure=fig)

    def sankey(
        self,
        *,
        # Time handling
        timestep: int | str | None = None,
        aggregate: Literal['sum', 'mean'] = 'sum',
        # Data selection
        select: SelectType | None = None,
        # Visual style
        colors: dict[str, str] | None = None,
        # Display
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot Sankey diagram of energy/material flow hours.

        Sankey diagrams show energy flows as a single diagram. When multiple
        scenarios or periods are present, they are aggregated using their
        respective weights (scenario probabilities and period durations).

        Args:
            timestep: Specific timestep to show, or None for aggregation.
            aggregate: How to aggregate if timestep is None ('sum' or 'mean').
            select: xarray-style selection to filter specific scenarios/periods
                before aggregation.
            colors: Override colors for flows/nodes.
            show: Whether to display.

        Returns:
            PlotResult with Sankey flow data.

        Examples:
            >>> results.plot.sankey()  # Weighted sum over all scenarios/periods
            >>> results.plot.sankey(timestep=100)
            >>> results.plot.sankey(select={'scenario': 'base'})  # Single scenario
        """
        # Get all flow hours (energy, not power - appropriate for Sankey)
        da = self._results.flow_hours()

        # Apply weights before selection - this way selection automatically gets correct weighted values
        flow_system = self._results.flow_system

        # Apply period weights (duration of each period)
        if 'period' in da.dims and flow_system.period_weights is not None:
            da = da * flow_system.period_weights

        # Apply scenario weights (normalized probabilities)
        if 'scenario' in da.dims and flow_system.scenario_weights is not None:
            scenario_weights = flow_system.scenario_weights
            scenario_weights = scenario_weights / scenario_weights.sum()  # Normalize
            da = da * scenario_weights

        # Apply selection
        if select:
            valid_select = {k: v for k, v in select.items() if k in da.dims or k in da.coords}
            if valid_select:
                da = da.sel(valid_select)

        # Handle timestep or aggregation over time
        if timestep is not None:
            if isinstance(timestep, int):
                da = da.isel(time=timestep)
            else:
                da = da.sel(time=timestep)
        elif 'time' in da.dims:
            da = getattr(da, aggregate)(dim='time')

        # Sum remaining dimensions (already weighted)
        if 'period' in da.dims:
            da = da.sum(dim='period')
        if 'scenario' in da.dims:
            da = da.sum(dim='scenario')

        # Get flow metadata from solution attrs
        flow_attrs = self._results.solution.attrs.get('Flows', {})

        # Build Sankey data
        nodes = set()
        links = {'source': [], 'target': [], 'value': [], 'label': []}

        for flow_label in da.coords['flow'].values:
            value = float(da.sel(flow=flow_label).values)
            if abs(value) < 1e-6:
                continue

            # Get flow metadata
            flow_info = flow_attrs.get(flow_label, {})
            source = flow_info.get('start', flow_label.split('|')[0])
            target = flow_info.get('end', 'Unknown')

            nodes.add(source)
            nodes.add(target)

            links['source'].append(source)
            links['target'].append(target)
            links['value'].append(abs(value))
            links['label'].append(flow_label)

        # Convert node names to indices
        node_list = list(nodes)
        node_indices = {n: i for i, n in enumerate(node_list)}

        # Merge colors from Results with any overrides
        merged_colors = _merge_colors(self.colors, colors)

        # Build node colors (try to match node name in colors)
        node_colors = [merged_colors.get(node) for node in node_list]
        # Only use colors if at least one node has a color, fill None with default
        if any(node_colors):
            node_colors = [c if c else 'lightgray' for c in node_colors]
        else:
            node_colors = None

        # Create Sankey figure
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color='black', width=0.5),
                        label=node_list,
                        color=node_colors,
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

        # Create Dataset with sankey link data
        sankey_ds = xr.Dataset(
            {
                'value': ('link', links['value']),
            },
            coords={
                'link': links['label'],
                'source': ('link', links['source']),
                'target': ('link', links['target']),
            },
        )

        # Handle show
        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=sankey_ds, figure=fig)

    def sizes(
        self,
        *,
        # Flow filtering
        start: str | list[str] | None = None,
        end: str | list[str] | None = None,
        component: str | list[str] | None = None,
        # Size filtering
        max_size: float | None = 1e6,
        # Data selection
        select: SelectType | None = None,
        # Visual style
        colors: dict[str, str] | None = None,
        # Faceting
        facet_col: str | None = 'scenario',
        facet_row: str | None = 'period',
        # Display
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot investment sizes (capacities) of flows.

        Shows the optimized sizes as a bar chart, useful for understanding
        investment decisions. By default, filters out very large sizes
        (> 1e6) which typically represent unbounded/default values.

        Args:
            start: Filter by source node(s).
            end: Filter by destination node(s).
            component: Filter by parent component(s).
            max_size: Maximum size to include. Sizes above this
                are excluded (default: 1e6). Set to None to include all.
            select: xarray-style selection (e.g., for scenarios).
            colors: Override colors.
            facet_col: Dimension for column facets (ignored if not in data).
            facet_row: Dimension for row facets (ignored if not in data).
            show: Whether to display.

        Returns:
            PlotResult with size data.

        Examples:
            >>> results.plot.sizes()  # All sizes (excluding defaults)
            >>> results.plot.sizes(max_size=None)  # Include all sizes
            >>> results.plot.sizes(component='Boiler')  # Specific component
        """
        import plotly.express as px

        # Get flow sizes using existing method
        da = self._results.sizes(start=start, end=end, component=component)

        # Apply selection
        if select:
            valid_select = {k: v for k, v in select.items() if k in da.dims or k in da.coords}
            if valid_select:
                da = da.sel(valid_select)

        # Filter out large default sizes
        if max_size is not None and da.size > 0:
            max_per_flow = da.max(dim=[d for d in da.dims if d != 'flow'])
            valid_flows = max_per_flow.coords['flow'].values[max_per_flow.values < max_size]
            da = da.sel(flow=valid_flows)

        # Convert to Dataset
        flow_labels = da.coords['flow'].values.tolist()
        ds = xr.Dataset({label: da.sel(flow=label, drop=True) for label in flow_labels})

        # Resolve facets
        actual_facet_col, actual_facet_row, _ = _resolve_facet_animate(ds, facet_col, facet_row, None)

        # Convert to long-form DataFrame
        df = _dataset_to_long_df(ds)
        if df.empty:
            fig = go.Figure()
        else:
            # Merge colors
            merged_colors = _merge_colors(self.colors, colors)
            variables = df['variable'].unique().tolist()
            color_map = {var: merged_colors.get(var) for var in variables}
            color_map = {k: v for k, v in color_map.items() if v is not None} or None

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

        # Handle show
        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=ds, figure=fig)

    def effects(
        self,
        aspect: Literal['total', 'temporal', 'periodic'] = 'total',
        *,
        effect: str | None = None,
        by: Literal['component', 'time'] = 'component',
        # Data selection
        select: SelectType | None = None,
        # Visual style
        colors: dict[str, str] | None = None,
        # Faceting
        facet_col: str | None = 'scenario',
        facet_row: str | None = 'period',
        # Display
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot effect (cost, emissions, etc.) breakdown.

        Args:
            aspect: Which aspect to plot - 'total', 'temporal', or 'periodic'.
            effect: Specific effect name to plot (e.g., 'costs', 'CO2').
                    If None, plots all effects.
            by: Group by 'component' or 'time'.
            select: xarray-style selection.
            colors: Override colors.
            facet_col: Dimension for column facets (ignored if not in data).
            facet_row: Dimension for row facets (ignored if not in data).
            show: Whether to display.

        Returns:
            PlotResult with effect breakdown data.

        Examples:
            >>> results.plot.effects()  # Total of all effects by component
            >>> results.plot.effects(effect='costs')  # Just costs
            >>> results.plot.effects(aspect='temporal', by='time')  # Over time
        """
        import plotly.express as px

        # Get effects per component
        effects_ds = self._results.effects_per_component

        # Select the aspect (total, temporal, periodic)
        if aspect not in effects_ds:
            available = list(effects_ds.data_vars)
            raise ValueError(f"Aspect '{aspect}' not found. Available: {available}")

        da = effects_ds[aspect]

        # Filter to specific effect if requested
        if effect is not None:
            if 'effect' not in da.dims:
                raise ValueError(f"No 'effect' dimension in data for aspect '{aspect}'")
            available_effects = da.coords['effect'].values.tolist()
            if effect not in available_effects:
                raise ValueError(f"Effect '{effect}' not found. Available: {available_effects}")
            da = da.sel(effect=effect)

        # Apply selection
        if select:
            valid_select = {k: v for k, v in select.items() if k in da.dims or k in da.coords}
            if valid_select:
                da = da.sel(valid_select)

        # Group by the specified dimension
        if by == 'component':
            # Sum over time if present
            if 'time' in da.dims:
                da = da.sum(dim='time')
            x_col = 'component'
            color_col = 'effect' if 'effect' in da.dims else 'component'
        elif by == 'time':
            # Sum over components
            if 'component' in da.dims:
                da = da.sum(dim='component')
            x_col = 'time'
            color_col = 'effect' if 'effect' in da.dims else None
        else:
            raise ValueError(f"'by' must be one of 'component', 'time', got {by!r}")

        # Resolve facets (ignore if dimension not present)
        actual_facet_col, actual_facet_row, _ = _resolve_facet_animate(da, facet_col, facet_row, None)

        # Convert to DataFrame for plotly express (required for pie/treemap)
        df = da.to_dataframe(name='value').reset_index()

        # Merge colors
        merged_colors = _merge_colors(self.colors, colors)
        color_items = df[color_col].unique().tolist() if color_col and color_col in df.columns else []
        color_map = plotting.process_colors(
            merged_colors,
            color_items,
            default_colorscale=CONFIG.Plotting.default_qualitative_colorscale,
        )

        # Build title
        effect_label = effect if effect else 'Effects'
        title = f'{effect_label} ({aspect}) by {by}'

        fig = (
            px.bar(
                df,
                x=x_col,
                y='value',
                color=color_col,
                color_discrete_map=color_map if color_col else None,
                facet_col=actual_facet_col,
                facet_row=actual_facet_row,
                title=title,
                **plotly_kwargs,
            )
            .update_layout(bargap=0, bargroupgap=0)
            .update_traces(marker_line_width=0)
        )

        # Handle show
        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        # Convert DataArray to Dataset for consistent return type
        return PlotResult(data=da.to_dataset(name=aspect), figure=fig)

    def variable(
        self,
        pattern: str,
        *,
        # Data selection
        select: SelectType | None = None,
        # Filtering
        include: FilterType | None = None,
        exclude: FilterType | None = None,
        # Transformation
        aggregate: Literal['sum', 'mean', 'max', 'min'] | None = None,
        # Visual style
        colors: dict[str, str] | None = None,
        # Faceting
        facet_col: str | None = 'scenario',
        facet_row: str | None = 'period',
        # Display
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot the same variable type across multiple elements.

        Searches all elements for variables matching the pattern and plots them
        together for easy comparison.

        Args:
            pattern: Variable suffix to match (e.g., 'on', 'flow_rate', 'charge_state').
                     Matches variables ending with this pattern.
            select: xarray-style selection.
            include: Only include elements containing these substrings.
            exclude: Exclude elements containing these substrings.
            aggregate: Aggregate over time dimension.
            colors: Override colors.
            facet_col: Dimension for column facets (ignored if not in data).
            facet_row: Dimension for row facets (ignored if not in data).
            show: Whether to display.

        Returns:
            PlotResult with matched variables as Dataset.

        Examples:
            >>> results.plot.variable('on')  # All binary operation states
            >>> results.plot.variable('flow_rate', include='Boiler')
            >>> results.plot.variable('charge_state')  # All storage charge states
        """
        # Find all matching variables across all elements
        matching_vars = {}

        for var_name in self._results.solution.data_vars:
            # Check if variable matches the pattern (ends with pattern or contains |pattern)
            if var_name.endswith(pattern) or f'|{pattern}' in var_name:
                # Extract element name (part before the |)
                element_name = var_name.split('|')[0] if '|' in var_name else var_name
                matching_vars[var_name] = element_name

        if not matching_vars:
            logger.warning(f'No variables found matching pattern: {pattern}')
            return PlotResult(data=xr.Dataset(), figure=go.Figure())

        # Apply include/exclude filtering on element names
        filtered_vars = {}
        for var_name, element_name in matching_vars.items():
            # Check include filter
            if include is not None:
                patterns = [include] if isinstance(include, str) else include
                if not any(p in element_name for p in patterns):
                    continue
            # Check exclude filter
            if exclude is not None:
                patterns = [exclude] if isinstance(exclude, str) else exclude
                if any(p in element_name for p in patterns):
                    continue
            filtered_vars[var_name] = element_name

        if not filtered_vars:
            logger.warning(f'No variables remaining after filtering for pattern: {pattern}')
            return PlotResult(data=xr.Dataset(), figure=go.Figure())

        # Build Dataset with variable names as keys to avoid collisions
        # (e.g., 'Boiler|flow_rate' and 'Boiler|flow_rate_max' would both map to 'Boiler')
        ds = xr.Dataset({var_name: self._results.solution[var_name] for var_name in filtered_vars})

        # Apply selection
        ds = _apply_selection(ds, select)

        # Apply aggregation
        if aggregate is not None and 'time' in ds.dims:
            ds = getattr(ds, aggregate)(dim='time')

        # Resolve facets (ignore if dimension not present)
        actual_facet_col, actual_facet_row, _ = _resolve_facet_animate(ds, facet_col, facet_row, None)

        # Merge colors
        merged_colors = _merge_colors(self.colors, colors)

        # Create figure
        fig = _create_line(
            ds,
            colors=merged_colors,
            title=f'{pattern} across elements',
            facet_col=actual_facet_col,
            facet_row=actual_facet_row,
            **plotly_kwargs,
        )

        # Handle show
        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=ds, figure=fig)

    def duration_curve(
        self,
        variables: str | list[str],
        *,
        # Data selection
        select: SelectType | None = None,
        # Sorting
        sort_by: str | None = None,
        # Transformation
        normalize: bool = False,
        # Visual style
        colors: dict[str, str] | None = None,
        # Faceting
        facet_col: str | None = 'scenario',
        facet_row: str | None = 'period',
        # Display
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot load duration curves (sorted time series).

        Duration curves show values sorted from highest to lowest, useful for
        understanding utilization patterns and peak demands.

        Args:
            variables: Variable name(s) to plot.
            select: xarray-style selection.
            sort_by: Variable to use for sorting order. If None, each variable
                     is sorted independently. If specified, all variables use
                     the sort order of this variable (useful for seeing correlations).
            normalize: If True, normalize x-axis to 0-100% of time.
            colors: Override colors.
            facet_col: Dimension for column facets (default: 'scenario').
            facet_row: Dimension for row facets (default: 'period').
            show: Whether to display.

        Returns:
            PlotResult with sorted duration curve data.

        Examples:
            >>> results.plot.duration_curve('Boiler(Q_th)|flow_rate')
            >>> results.plot.duration_curve(['CHP|on', 'Boiler|on'])
            >>> results.plot.duration_curve('demand', normalize=True)
            >>> # Sort all by demand to see correlations
            >>> results.plot.duration_curve(['demand', 'price', 'Boiler|on'], sort_by='demand')
        """
        # Normalize to list
        if isinstance(variables, str):
            variables = [variables]

        # Get the data
        ds = self._results.solution[variables]

        # Apply selection
        ds = _apply_selection(ds, select)

        # Check for time dimension
        if 'time' not in ds.dims:
            raise ValueError('Duration curve requires time dimension in data')

        # Identify extra dimensions (scenario, period, etc.)
        extra_dims = [d for d in ds.dims if d != 'time']

        # Resolve facet dimensions (only keep those that exist in data)
        actual_facet_col = facet_col if facet_col and facet_col in extra_dims else None
        actual_facet_row = facet_row if facet_row and facet_row in extra_dims else None

        # Dimensions to iterate over for separate duration curves
        facet_dims = [d for d in [actual_facet_col, actual_facet_row] if d is not None]
        # Dimensions to average over (not time, not faceted)
        avg_dims = [d for d in extra_dims if d not in facet_dims]

        # Average over non-faceted dimensions
        if avg_dims:
            ds = ds.mean(dim=avg_dims)

        if sort_by is not None:
            if sort_by not in ds.data_vars:
                raise ValueError(f"sort_by variable '{sort_by}' not in variables. Available: {list(ds.data_vars)}")

        # Build duration curves using xr.apply_ufunc for clean sorting along time axis
        duration_name = 'duration_pct' if normalize else 'duration'

        def sort_descending(arr: np.ndarray) -> np.ndarray:
            """Sort array in descending order."""
            return np.sort(arr)[::-1]

        def apply_sort_order(arr: np.ndarray, sort_indices: np.ndarray) -> np.ndarray:
            """Apply pre-computed sort indices to array."""
            return arr[sort_indices]

        if sort_by is not None:
            # Compute sort indices from reference variable (descending order)
            sort_indices = xr.apply_ufunc(
                lambda x: np.argsort(x)[::-1],
                ds[sort_by],
                input_core_dims=[['time']],
                output_core_dims=[['time']],
                vectorize=True,
            )
            # Apply same sort order to all variables
            result_ds = xr.apply_ufunc(
                apply_sort_order,
                ds,
                sort_indices,
                input_core_dims=[['time'], ['time']],
                output_core_dims=[['time']],
                vectorize=True,
            )
        else:
            # Sort each variable independently (descending)
            result_ds = xr.apply_ufunc(
                sort_descending,
                ds,
                input_core_dims=[['time']],
                output_core_dims=[['time']],
                vectorize=True,
            )

        # Rename time dimension to duration
        result_ds = result_ds.rename({'time': duration_name})

        # Update duration coordinate
        n_timesteps = result_ds.sizes[duration_name]
        if normalize:
            duration_coord = np.linspace(0, 100, n_timesteps)
        else:
            duration_coord = np.arange(n_timesteps)
        result_ds = result_ds.assign_coords({duration_name: duration_coord})

        # Merge colors
        merged_colors = _merge_colors(self.colors, colors)

        # Extract facet dimensions
        actual_facet_col = facet_dims[0] if len(facet_dims) > 0 else None
        actual_facet_row = facet_dims[1] if len(facet_dims) > 1 else None

        # Create figure
        fig = _create_line(
            result_ds,
            colors=merged_colors,
            title='Duration Curve',
            facet_col=actual_facet_col,
            facet_row=actual_facet_row,
            **plotly_kwargs,
        )

        # Update axis labels
        x_label = 'Duration [%]' if normalize else 'Timesteps'
        fig.update_xaxes(title_text=x_label)

        # Handle show
        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            fig.show()

        return PlotResult(data=result_ds, figure=fig)


class ElementPlotAccessor:
    """Plot accessor for individual element results (ComponentResults, BusResults).

    Access via results['ElementName'].plot.<method>()

    Example:
        >>> results['Boiler'].plot.balance()
        >>> results['Battery'].plot.storage()
    """

    def __init__(self, element_results: _NodeResults):
        self._element = element_results
        self._results = element_results._results

    def balance(self, **kwargs: Any) -> PlotResult:
        """Plot balance for this element.

        All kwargs are passed to PlotAccessor.balance().
        See PlotAccessor.balance() for full documentation.
        """
        return self._results.plot.balance(self._element.label, **kwargs)

    def heatmap(
        self,
        variable: str | list[str] | None = None,
        **kwargs: Any,
    ) -> PlotResult:
        """Plot heatmap for this element's variables.

        Args:
            variable: Variable suffix (e.g., 'on') or full name.
                      If None, uses all time-series variables.
            **kwargs: Passed to PlotAccessor.heatmap().
        """
        if variable is None:
            # Get all time-series variables for this element
            variables = [v for v in self._element.solution.data_vars if 'time' in self._element.solution[v].dims]
        elif isinstance(variable, str):
            # Check if it's a suffix or full name
            if '|' in variable:
                variables = [variable]
            else:
                # Find variables matching the suffix
                variables = [v for v in self._element.solution.data_vars if variable in v]
        else:
            variables = variable

        if not variables:
            logger.warning(f'No matching variables found for {variable} in {self._element.label}')
            return PlotResult(data=xr.Dataset(), figure=go.Figure())

        return self._results.plot.heatmap(variables, **kwargs)

    def storage(self, **kwargs: Any) -> PlotResult:
        """Plot storage state (only for storage components).

        All kwargs are passed to PlotAccessor.storage().
        See PlotAccessor.storage() for full documentation.

        Raises:
            ValueError: If this component is not a storage.
        """
        # Check if element has is_storage attribute (only ComponentResults has it)
        if not hasattr(self._element, 'is_storage') or not self._element.is_storage:
            raise ValueError(f'{self._element.label} is not a storage component')
        return self._results.plot.storage(self._element.label, **kwargs)
