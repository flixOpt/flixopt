"""Xarray accessors for plotting (``.fxplot``) and statistics (``.fxstats``)."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr

from .color_processing import ColorType, process_colors
from .config import CONFIG


def assign_slots(
    ds: xr.Dataset,
    *,
    x: str | Literal['auto'] | None = 'auto',
    color: str | Literal['auto'] | None = 'auto',
    facet_col: str | Literal['auto'] | None = 'auto',
    facet_row: str | Literal['auto'] | None = 'auto',
    animation_frame: str | Literal['auto'] | None = 'auto',
    exclude_dims: set[str] | None = None,
) -> dict[str, str | None]:
    """Assign dimensions to plot slots using CONFIG.Plotting.dim_priority.

    Dimensions are assigned in priority order to slots based on CONFIG.Plotting.slot_priority.

    Slot values:
        - 'auto': auto-assign from available dims using priority
        - None: skip this slot (not available for this plot type)
        - str: use this specific dimension

    'variable' is treated as a dimension when len(data_vars) > 1. It represents
    the data_var names column in the melted DataFrame.

    Args:
        ds: Dataset to analyze for available dimensions.
        x: X-axis dimension. 'auto' assigns first available from priority.
        color: Color grouping dimension.
        facet_col: Column faceting dimension.
        facet_row: Row faceting dimension.
        animation_frame: Animation slider dimension.
        exclude_dims: Dimensions to exclude from auto-assignment (e.g., already used for x elsewhere).

    Returns:
        Dict with keys 'x', 'color', 'facet_col', 'facet_row', 'animation_frame'
        and values being assigned dimension names (or None if slot skipped/unfilled).
    """
    # Get available dimensions with size > 1, excluding specified dims
    exclude = exclude_dims or set()
    available = {d for d in ds.dims if ds.sizes[d] > 1 and d not in exclude}
    # 'variable' is available when there are multiple data_vars (and not excluded)
    if len(ds.data_vars) > 1 and 'variable' not in exclude:
        available.add('variable')

    # Get priority-ordered list of available dims
    priority_dims = [d for d in CONFIG.Plotting.dim_priority if d in available]
    # Add any available dims not in priority list (fallback)
    priority_dims.extend(d for d in available if d not in priority_dims)

    # Slot specification
    slots = {
        'x': x,
        'color': color,
        'facet_col': facet_col,
        'facet_row': facet_row,
        'animation_frame': animation_frame,
    }
    # Slot fill order from config
    slot_order = CONFIG.Plotting.slot_priority

    results: dict[str, str | None] = {k: None for k in slot_order}
    used: set[str] = set()

    # First pass: resolve explicit dimensions (not 'auto' or None) to mark them as used
    for slot_name, value in slots.items():
        if value is not None and value != 'auto':
            used.add(value)
            results[slot_name] = value

    # Second pass: resolve 'auto' slots in config-defined fill order
    dim_iter = iter(d for d in priority_dims if d not in used)
    for slot_name in slot_order:
        if slots[slot_name] == 'auto':
            next_dim = next(dim_iter, None)
            if next_dim:
                used.add(next_dim)
                results[slot_name] = next_dim

    # Warn if any dimensions were not assigned to any slot
    unassigned = available - used
    if unassigned:
        available_slots = [k for k, v in slots.items() if v is not None]
        unavailable_slots = [k for k, v in slots.items() if v is None]
        if unavailable_slots:
            warnings.warn(
                f'Dimensions {unassigned} not assigned to any plot dimension. '
                f'Not available for this plot type: {unavailable_slots}. '
                f'Reduce dimensions before plotting (e.g., .sel(), .isel(), .mean()).',
                stacklevel=3,
            )
        else:
            warnings.warn(
                f'Dimensions {unassigned} not assigned to any plot dimension ({available_slots}). '
                f'Reduce dimensions before plotting (e.g., .sel(), .isel(), .mean()).',
                stacklevel=3,
            )

    return results


def _build_fig_kwargs(
    slots: dict[str, str | None],
    ds_sizes: dict[str, int],
    px_kwargs: dict[str, Any],
    facet_cols: int | None = None,
) -> dict[str, Any]:
    """Build plotly express kwargs from slot assignments.

    Adds facet/animation args only if slots are assigned and not overridden in px_kwargs.
    Handles facet_col_wrap based on dimension size.
    """
    facet_col_wrap = facet_cols or CONFIG.Plotting.default_facet_cols
    result: dict[str, Any] = {}

    # Add facet/animation kwargs from slots (skip if None or already in px_kwargs)
    for slot in ('color', 'facet_col', 'facet_row', 'animation_frame'):
        if slots.get(slot) and slot not in px_kwargs:
            result[slot] = slots[slot]

    # Add facet_col_wrap if facet_col is set and dimension is large enough
    if result.get('facet_col'):
        dim_size = ds_sizes.get(result['facet_col'], facet_col_wrap + 1)
        if facet_col_wrap < dim_size:
            result['facet_col_wrap'] = facet_col_wrap

    return result


def _dataset_to_long_df(ds: xr.Dataset, value_name: str = 'value', var_name: str = 'variable') -> pd.DataFrame:
    """Convert Dataset to long-form DataFrame for Plotly Express."""
    if not ds.data_vars:
        return pd.DataFrame()
    if all(ds[var].ndim == 0 for var in ds.data_vars):
        rows = [{var_name: var, value_name: float(ds[var].values)} for var in ds.data_vars]
        return pd.DataFrame(rows)
    df = ds.to_dataframe().reset_index()
    # Use dims (not just coords) as id_vars - dims without coords become integer indices
    id_cols = [c for c in ds.dims if c in df.columns]
    return df.melt(id_vars=id_cols, var_name=var_name, value_name=value_name)


@xr.register_dataset_accessor('fxplot')
class DatasetPlotAccessor:
    """Plot accessor for any xr.Dataset. Access via ``dataset.fxplot``.

    Provides convenient plotting methods that automatically handle multi-dimensional
    data through faceting and animation. All methods return a Plotly Figure.

    This accessor is globally registered when flixopt is imported and works on
    any xr.Dataset.

    Examples:
        Basic usage::

            import flixopt
            import xarray as xr

            ds = xr.Dataset({'A': (['time'], [1, 2, 3]), 'B': (['time'], [3, 2, 1])})
            ds.fxplot.stacked_bar()
            ds.fxplot.line()
            ds.fxplot.area()

        With faceting::

            ds.fxplot.stacked_bar(facet_col='scenario')
            ds.fxplot.line(facet_col='period', animation_frame='scenario')

        Heatmap::

            ds.fxplot.heatmap('temperature')
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        """Initialize the accessor with an xr.Dataset object."""
        self._ds = xarray_obj

    def bar(
        self,
        *,
        x: str | Literal['auto'] | None = 'auto',
        color: str | Literal['auto'] | None = 'auto',
        colors: ColorType | None = None,
        title: str = '',
        xlabel: str = '',
        ylabel: str = '',
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        facet_cols: int | None = None,
        exclude_dims: set[str] | None = None,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create a grouped bar chart from the dataset.

        Args:
            x: Dimension for x-axis. 'auto' uses CONFIG.Plotting.dim_priority.
            color: Dimension for color grouping. 'auto' uses 'variable' (data_var names)
                if available, otherwise uses CONFIG priority.
            colors: Color specification (colorscale name, color list, or dict mapping).
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            facet_col: Dimension for column facets. 'auto' uses CONFIG priority.
            facet_row: Dimension for row facets. 'auto' uses CONFIG priority.
            animation_frame: Dimension for animation slider.
            facet_cols: Number of columns in facet grid wrap.
            exclude_dims: Dimensions to exclude from auto-assignment.
            **px_kwargs: Additional arguments passed to plotly.express.bar.

        Returns:
            Plotly Figure.
        """
        slots = assign_slots(
            self._ds,
            x=x,
            color=color,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            exclude_dims=exclude_dims,
        )
        df = _dataset_to_long_df(self._ds)
        if df.empty:
            return go.Figure()

        color_labels = df[slots['color']].unique().tolist() if slots['color'] and slots['color'] in df.columns else []
        color_map = process_colors(colors, color_labels, CONFIG.Plotting.default_qualitative_colorscale)

        labels = {**(({slots['x']: xlabel}) if xlabel and slots['x'] else {}), **({'value': ylabel} if ylabel else {})}
        fig_kwargs = {
            'data_frame': df,
            'x': slots['x'],
            'y': 'value',
            'title': title,
            'barmode': 'group',
            'color_discrete_map': color_map,
            **({'labels': labels} if labels else {}),
            **_build_fig_kwargs(slots, dict(self._ds.sizes), px_kwargs, facet_cols),
        }
        return px.bar(**{**fig_kwargs, **px_kwargs})

    def stacked_bar(
        self,
        *,
        x: str | Literal['auto'] | None = 'auto',
        color: str | Literal['auto'] | None = 'auto',
        colors: ColorType | None = None,
        title: str = '',
        xlabel: str = '',
        ylabel: str = '',
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        facet_cols: int | None = None,
        exclude_dims: set[str] | None = None,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create a stacked bar chart from the dataset.

        Variables in the dataset become stacked segments. Positive and negative
        values are stacked separately.

        Args:
            x: Dimension for x-axis. 'auto' uses CONFIG.Plotting.dim_priority.
            color: Dimension for color grouping. 'auto' uses 'variable' (data_var names)
                if available, otherwise uses CONFIG priority.
            colors: Color specification (colorscale name, color list, or dict mapping).
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            facet_col: Dimension for column facets. 'auto' uses CONFIG priority.
            facet_row: Dimension for row facets.
            animation_frame: Dimension for animation slider.
            facet_cols: Number of columns in facet grid wrap.
            **px_kwargs: Additional arguments passed to plotly.express.bar.

        Returns:
            Plotly Figure.
        """
        slots = assign_slots(
            self._ds,
            x=x,
            color=color,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            exclude_dims=exclude_dims,
        )
        df = _dataset_to_long_df(self._ds)
        if df.empty:
            return go.Figure()

        color_labels = df[slots['color']].unique().tolist() if slots['color'] and slots['color'] in df.columns else []
        color_map = process_colors(colors, color_labels, CONFIG.Plotting.default_qualitative_colorscale)

        labels = {**(({slots['x']: xlabel}) if xlabel and slots['x'] else {}), **({'value': ylabel} if ylabel else {})}
        fig_kwargs = {
            'data_frame': df,
            'x': slots['x'],
            'y': 'value',
            'title': title,
            'color_discrete_map': color_map,
            **({'labels': labels} if labels else {}),
            **_build_fig_kwargs(slots, dict(self._ds.sizes), px_kwargs, facet_cols),
        }
        fig = px.bar(**{**fig_kwargs, **px_kwargs})
        fig.update_layout(barmode='relative', bargap=0, bargroupgap=0)
        fig.update_traces(marker_line_width=0)
        return fig

    def line(
        self,
        *,
        x: str | Literal['auto'] | None = 'auto',
        color: str | Literal['auto'] | None = 'auto',
        colors: ColorType | None = None,
        title: str = '',
        xlabel: str = '',
        ylabel: str = '',
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        facet_cols: int | None = None,
        line_shape: str | None = None,
        exclude_dims: set[str] | None = None,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create a line chart from the dataset.

        Each variable in the dataset becomes a separate line.

        Args:
            x: Dimension for x-axis. 'auto' uses CONFIG.Plotting.dim_priority.
            color: Dimension for color grouping. 'auto' uses 'variable' (data_var names)
                if available, otherwise uses CONFIG priority.
            colors: Color specification (colorscale name, color list, or dict mapping).
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            facet_col: Dimension for column facets. 'auto' uses CONFIG priority.
            facet_row: Dimension for row facets.
            animation_frame: Dimension for animation slider.
            facet_cols: Number of columns in facet grid wrap.
            line_shape: Line interpolation ('linear', 'hv', 'vh', 'hvh', 'vhv', 'spline').
                Default from CONFIG.Plotting.default_line_shape.
            **px_kwargs: Additional arguments passed to plotly.express.line.

        Returns:
            Plotly Figure.
        """
        slots = assign_slots(
            self._ds,
            x=x,
            color=color,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            exclude_dims=exclude_dims,
        )
        df = _dataset_to_long_df(self._ds)
        if df.empty:
            return go.Figure()

        color_labels = df[slots['color']].unique().tolist() if slots['color'] and slots['color'] in df.columns else []
        color_map = process_colors(colors, color_labels, CONFIG.Plotting.default_qualitative_colorscale)

        labels = {**(({slots['x']: xlabel}) if xlabel and slots['x'] else {}), **({'value': ylabel} if ylabel else {})}
        fig_kwargs = {
            'data_frame': df,
            'x': slots['x'],
            'y': 'value',
            'title': title,
            'line_shape': line_shape or CONFIG.Plotting.default_line_shape,
            'color_discrete_map': color_map,
            **({'labels': labels} if labels else {}),
            **_build_fig_kwargs(slots, dict(self._ds.sizes), px_kwargs, facet_cols),
        }
        return px.line(**{**fig_kwargs, **px_kwargs})

    def area(
        self,
        *,
        x: str | Literal['auto'] | None = 'auto',
        color: str | Literal['auto'] | None = 'auto',
        colors: ColorType | None = None,
        title: str = '',
        xlabel: str = '',
        ylabel: str = '',
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        facet_cols: int | None = None,
        line_shape: str | None = None,
        exclude_dims: set[str] | None = None,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create a stacked area chart from the dataset.

        Args:
            x: Dimension for x-axis. 'auto' uses CONFIG.Plotting.dim_priority.
            color: Dimension for color grouping. 'auto' uses 'variable' (data_var names)
                if available, otherwise uses CONFIG priority.
            colors: Color specification (colorscale name, color list, or dict mapping).
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            facet_col: Dimension for column facets. 'auto' uses CONFIG priority.
            facet_row: Dimension for row facets.
            animation_frame: Dimension for animation slider.
            facet_cols: Number of columns in facet grid wrap.
            line_shape: Line interpolation. Default from CONFIG.Plotting.default_line_shape.
            **px_kwargs: Additional arguments passed to plotly.express.area.

        Returns:
            Plotly Figure.
        """
        slots = assign_slots(
            self._ds,
            x=x,
            color=color,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            exclude_dims=exclude_dims,
        )
        df = _dataset_to_long_df(self._ds)
        if df.empty:
            return go.Figure()

        color_labels = df[slots['color']].unique().tolist() if slots['color'] and slots['color'] in df.columns else []
        color_map = process_colors(colors, color_labels, CONFIG.Plotting.default_qualitative_colorscale)

        labels = {**(({slots['x']: xlabel}) if xlabel and slots['x'] else {}), **({'value': ylabel} if ylabel else {})}
        fig_kwargs = {
            'data_frame': df,
            'x': slots['x'],
            'y': 'value',
            'title': title,
            'line_shape': line_shape or CONFIG.Plotting.default_line_shape,
            'color_discrete_map': color_map,
            **({'labels': labels} if labels else {}),
            **_build_fig_kwargs(slots, dict(self._ds.sizes), px_kwargs, facet_cols),
        }
        return px.area(**{**fig_kwargs, **px_kwargs})

    def heatmap(
        self,
        variable: str | None = None,
        *,
        colors: str | list[str] | None = None,
        title: str = '',
        facet_col: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        facet_cols: int | None = None,
        **imshow_kwargs: Any,
    ) -> go.Figure:
        """Create a heatmap visualization.

        If the dataset has multiple variables, select one with the `variable` parameter.
        If only one variable exists, it is used automatically.

        Args:
            variable: Variable name to plot. Required if dataset has multiple variables.
                If None and dataset has one variable, that variable is used.
            colors: Colorscale name or list of colors.
            title: Plot title.
            facet_col: Dimension for column facets.
            animation_frame: Dimension for animation slider.
            facet_cols: Number of columns in facet grid wrap.
            **imshow_kwargs: Additional arguments passed to plotly.express.imshow.

        Returns:
            Plotly Figure.
        """
        # Select single variable
        if variable is None:
            if len(self._ds.data_vars) == 1:
                variable = list(self._ds.data_vars)[0]
            else:
                raise ValueError(
                    f'Dataset has {len(self._ds.data_vars)} variables. '
                    f"Please specify which variable to plot with variable='name'."
                )

        da = self._ds[variable]

        if da.size == 0:
            return go.Figure()

        colors = colors or CONFIG.Plotting.default_sequential_colorscale
        facet_col_wrap = facet_cols or CONFIG.Plotting.default_facet_cols

        # Heatmap uses imshow - first 2 dims are the x/y axes of the heatmap
        # Only call assign_slots if we need to resolve 'auto' values
        if facet_col == 'auto' or animation_frame == 'auto':
            heatmap_axes = set(list(da.dims)[:2]) if len(da.dims) >= 2 else set()
            slots = assign_slots(
                self._ds,
                x=None,
                color=None,
                facet_col=facet_col,
                facet_row=None,
                animation_frame=animation_frame,
                exclude_dims=heatmap_axes,
            )
            resolved_facet = slots['facet_col']
            resolved_animation = slots['animation_frame']
        else:
            # Values already resolved (or None), use directly without re-resolving
            resolved_facet = facet_col
            resolved_animation = animation_frame

        imshow_args: dict[str, Any] = {
            'color_continuous_scale': colors,
            'title': title or variable,
        }

        if resolved_facet and resolved_facet in da.dims:
            imshow_args['facet_col'] = resolved_facet
            if facet_col_wrap < da.sizes[resolved_facet]:
                imshow_args['facet_col_wrap'] = facet_col_wrap

        if resolved_animation and resolved_animation in da.dims:
            imshow_args['animation_frame'] = resolved_animation

        # Squeeze singleton dimensions not used for faceting/animation
        # px.imshow can't handle extra singleton dims in multi-dimensional data
        dims_to_preserve = set(list(da.dims)[:2])  # First 2 dims are heatmap x/y axes
        if resolved_facet and resolved_facet in da.dims:
            dims_to_preserve.add(resolved_facet)
        if resolved_animation and resolved_animation in da.dims:
            dims_to_preserve.add(resolved_animation)
        for dim in list(da.dims):
            if dim not in dims_to_preserve and da.sizes[dim] == 1:
                da = da.squeeze(dim)
        imshow_args['img'] = da

        # Use binary_string=False to handle non-numeric coords (e.g., string labels)
        if 'binary_string' not in imshow_kwargs:
            imshow_args['binary_string'] = False

        return px.imshow(**{**imshow_args, **imshow_kwargs})

    def scatter(
        self,
        x: str,
        y: str,
        *,
        title: str = '',
        xlabel: str = '',
        ylabel: str = '',
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        facet_cols: int | None = None,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create a scatter plot from two variables in the dataset.

        Args:
            x: Variable name for x-axis.
            y: Variable name for y-axis.
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            facet_col: Dimension for column facets. 'auto' uses CONFIG priority.
            facet_row: Dimension for row facets.
            animation_frame: Dimension for animation slider.
            facet_cols: Number of columns in facet grid wrap.
            **px_kwargs: Additional arguments passed to plotly.express.scatter.

        Returns:
            Plotly Figure.
        """
        if x not in self._ds.data_vars:
            raise ValueError(f"Variable '{x}' not found in dataset. Available: {list(self._ds.data_vars)}")
        if y not in self._ds.data_vars:
            raise ValueError(f"Variable '{y}' not found in dataset. Available: {list(self._ds.data_vars)}")

        df = self._ds[[x, y]].to_dataframe().reset_index()
        if df.empty:
            return go.Figure()

        # Scatter uses explicit x/y variable names, not dimensions
        slots = assign_slots(
            self._ds, x=None, color=None, facet_col=facet_col, facet_row=facet_row, animation_frame=animation_frame
        )

        facet_col_wrap = facet_cols or CONFIG.Plotting.default_facet_cols
        fig_kwargs: dict[str, Any] = {
            'data_frame': df,
            'x': x,
            'y': y,
            'title': title,
            **px_kwargs,
        }
        if xlabel:
            fig_kwargs['labels'] = {**fig_kwargs.get('labels', {}), x: xlabel}
        if ylabel:
            fig_kwargs['labels'] = {**fig_kwargs.get('labels', {}), y: ylabel}

        # Only use facets if the column actually exists in the dataframe
        # (scatter uses wide format, so 'variable' column doesn't exist)
        if slots['facet_col'] and slots['facet_col'] in df.columns:
            fig_kwargs['facet_col'] = slots['facet_col']
            if facet_col_wrap < self._ds.sizes.get(slots['facet_col'], facet_col_wrap + 1):
                fig_kwargs['facet_col_wrap'] = facet_col_wrap
        if slots['facet_row'] and slots['facet_row'] in df.columns:
            fig_kwargs['facet_row'] = slots['facet_row']
        if slots['animation_frame'] and slots['animation_frame'] in df.columns:
            fig_kwargs['animation_frame'] = slots['animation_frame']

        return px.scatter(**fig_kwargs)

    def pie(
        self,
        *,
        colors: ColorType | None = None,
        title: str = '',
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        facet_cols: int | None = None,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create a pie chart from aggregated dataset values.

        Extra dimensions are auto-assigned to facet_col and facet_row.
        For scalar values, a single pie is shown.

        Note:
            ``px.pie()`` does not support animation_frame, so only facets are available.

        Args:
            colors: Color specification (colorscale name, color list, or dict mapping).
            title: Plot title.
            facet_col: Dimension for column facets. 'auto' uses CONFIG priority.
            facet_row: Dimension for row facets. 'auto' uses CONFIG priority.
            facet_cols: Number of columns in facet grid wrap.
            **px_kwargs: Additional arguments passed to plotly.express.pie.

        Returns:
            Plotly Figure.

        Example:
            >>> ds.sum('time').fxplot.pie()  # Sum over time, then pie chart
            >>> ds.sum('time').fxplot.pie(facet_col='scenario')  # Pie per scenario
        """
        max_ndim = max((self._ds[v].ndim for v in self._ds.data_vars), default=0)

        names = list(self._ds.data_vars)
        color_map = process_colors(colors, names, default_colorscale=CONFIG.Plotting.default_qualitative_colorscale)

        # Scalar case - single pie
        if max_ndim == 0:
            values = [float(self._ds[v].values) for v in names]
            df = pd.DataFrame({'variable': names, 'value': values})
            return px.pie(
                df,
                names='variable',
                values='value',
                title=title,
                color='variable',
                color_discrete_map=color_map,
                **px_kwargs,
            )

        # Multi-dimensional case - faceted pies (px.pie doesn't support animation_frame)
        df = _dataset_to_long_df(self._ds)
        if df.empty:
            return go.Figure()

        # Pie uses 'variable' for names and 'value' for values, no x/color/animation_frame
        slots = assign_slots(
            self._ds, x=None, color=None, facet_col=facet_col, facet_row=facet_row, animation_frame=None
        )

        facet_col_wrap = facet_cols or CONFIG.Plotting.default_facet_cols
        fig_kwargs: dict[str, Any] = {
            'data_frame': df,
            'names': 'variable',
            'values': 'value',
            'title': title,
            'color': 'variable',
            'color_discrete_map': color_map,
            **px_kwargs,
        }

        if slots['facet_col']:
            fig_kwargs['facet_col'] = slots['facet_col']
            if facet_col_wrap < self._ds.sizes.get(slots['facet_col'], facet_col_wrap + 1):
                fig_kwargs['facet_col_wrap'] = facet_col_wrap
        if slots['facet_row']:
            fig_kwargs['facet_row'] = slots['facet_row']

        return px.pie(**fig_kwargs)


@xr.register_dataset_accessor('fxstats')
class DatasetStatsAccessor:
    """Statistics/transformation accessor for any xr.Dataset. Access via ``dataset.fxstats``.

    Provides data transformation methods that return new datasets.
    Chain with ``.fxplot`` for visualization.

    Examples:
        Duration curve::

            ds.fxstats.to_duration_curve().fxplot.line()
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._ds = xarray_obj

    def to_duration_curve(self, *, normalize: bool = True) -> xr.Dataset:
        """Transform dataset to duration curve format (sorted values).

        Values are sorted in descending order along the 'time' dimension.
        The time coordinate is replaced with duration (percentage or index).

        Args:
            normalize: If True, x-axis shows percentage (0-100). If False, shows timestep index.

        Returns:
            Transformed xr.Dataset with duration coordinate instead of time.

        Example:
            >>> ds.fxstats.to_duration_curve().fxplot.line(title='Duration Curve')
        """
        import numpy as np

        if 'time' not in self._ds.dims:
            raise ValueError("Duration curve requires a 'time' dimension.")

        # Sort each variable along time dimension (descending)
        sorted_ds = self._ds.copy()
        for var in sorted_ds.data_vars:
            da = sorted_ds[var]
            time_axis = da.dims.index('time')
            # Sort along time axis (descending) - use flip for correct axis
            sorted_values = np.flip(np.sort(da.values, axis=time_axis), axis=time_axis)
            sorted_ds[var] = (da.dims, sorted_values)

        # Replace time coordinate with duration
        n_timesteps = sorted_ds.sizes['time']
        if normalize:
            duration_coord = np.linspace(0, 100, n_timesteps)
            sorted_ds = sorted_ds.assign_coords({'time': duration_coord})
            sorted_ds = sorted_ds.rename({'time': 'duration_pct'})
        else:
            duration_coord = np.arange(n_timesteps)
            sorted_ds = sorted_ds.assign_coords({'time': duration_coord})
            sorted_ds = sorted_ds.rename({'time': 'duration'})

        return sorted_ds


@xr.register_dataarray_accessor('fxplot')
class DataArrayPlotAccessor:
    """Plot accessor for any xr.DataArray. Access via ``dataarray.fxplot``.

    Provides convenient plotting methods. For bar/stacked_bar/line/area,
    the DataArray is converted to a Dataset first. For heatmap, it works
    directly with the DataArray.

    Examples:
        Basic usage::

            import flixopt
            import xarray as xr

            da = xr.DataArray([1, 2, 3], dims=['time'], name='temperature')
            da.fxplot.line()
            da.fxplot.heatmap()
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        """Initialize the accessor with an xr.DataArray object."""
        self._da = xarray_obj

    def _to_dataset(self) -> xr.Dataset:
        """Convert DataArray to Dataset for plotting."""
        name = self._da.name or 'value'
        return self._da.to_dataset(name=name)

    def bar(
        self,
        *,
        x: str | Literal['auto'] | None = 'auto',
        colors: ColorType | None = None,
        title: str = '',
        xlabel: str = '',
        ylabel: str = '',
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        facet_cols: int | None = None,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create a grouped bar chart. See DatasetPlotAccessor.bar for details."""
        return self._to_dataset().fxplot.bar(
            x=x,
            colors=colors,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            facet_cols=facet_cols,
            **px_kwargs,
        )

    def stacked_bar(
        self,
        *,
        x: str | Literal['auto'] | None = 'auto',
        colors: ColorType | None = None,
        title: str = '',
        xlabel: str = '',
        ylabel: str = '',
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        facet_cols: int | None = None,
        exclude_dims: set[str] | None = None,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create a stacked bar chart. See DatasetPlotAccessor.stacked_bar for details."""
        return self._to_dataset().fxplot.stacked_bar(
            x=x,
            colors=colors,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            facet_cols=facet_cols,
            **px_kwargs,
        )

    def line(
        self,
        *,
        x: str | Literal['auto'] | None = 'auto',
        colors: ColorType | None = None,
        title: str = '',
        xlabel: str = '',
        ylabel: str = '',
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        facet_cols: int | None = None,
        line_shape: str | None = None,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create a line chart. See DatasetPlotAccessor.line for details."""
        return self._to_dataset().fxplot.line(
            x=x,
            colors=colors,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            facet_cols=facet_cols,
            line_shape=line_shape,
            **px_kwargs,
        )

    def area(
        self,
        *,
        x: str | Literal['auto'] | None = 'auto',
        colors: ColorType | None = None,
        title: str = '',
        xlabel: str = '',
        ylabel: str = '',
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        facet_cols: int | None = None,
        line_shape: str | None = None,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create a stacked area chart. See DatasetPlotAccessor.area for details."""
        return self._to_dataset().fxplot.area(
            x=x,
            colors=colors,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            facet_cols=facet_cols,
            line_shape=line_shape,
            **px_kwargs,
        )

    def heatmap(
        self,
        *,
        colors: str | list[str] | None = None,
        title: str = '',
        facet_col: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        facet_cols: int | None = None,
        **imshow_kwargs: Any,
    ) -> go.Figure:
        """Create a heatmap visualization directly from the DataArray.

        Args:
            colors: Colorscale name or list of colors.
            title: Plot title.
            facet_col: Dimension for column facets.
            animation_frame: Dimension for animation slider.
            facet_cols: Number of columns in facet grid wrap.
            **imshow_kwargs: Additional arguments passed to plotly.express.imshow.

        Returns:
            Plotly Figure.
        """
        da = self._da

        if da.size == 0:
            return go.Figure()

        colors = colors or CONFIG.Plotting.default_sequential_colorscale
        facet_col_wrap = facet_cols or CONFIG.Plotting.default_facet_cols

        # Heatmap uses imshow - first 2 dims are the x/y axes of the heatmap
        # Only call assign_slots if we need to resolve 'auto' values
        if facet_col == 'auto' or animation_frame == 'auto':
            heatmap_axes = set(list(da.dims)[:2]) if len(da.dims) >= 2 else set()
            ds_for_resolution = da.to_dataset(name='_temp')
            slots = assign_slots(
                ds_for_resolution,
                x=None,
                color=None,
                facet_col=facet_col,
                facet_row=None,
                animation_frame=animation_frame,
                exclude_dims=heatmap_axes,
            )
            resolved_facet = slots['facet_col']
            resolved_animation = slots['animation_frame']
        else:
            # Values already resolved (or None), use directly without re-resolving
            resolved_facet = facet_col
            resolved_animation = animation_frame

        imshow_args: dict[str, Any] = {
            'color_continuous_scale': colors,
            'title': title or (da.name if da.name else ''),
        }

        if resolved_facet and resolved_facet in da.dims:
            imshow_args['facet_col'] = resolved_facet
            if facet_col_wrap < da.sizes[resolved_facet]:
                imshow_args['facet_col_wrap'] = facet_col_wrap

        if resolved_animation and resolved_animation in da.dims:
            imshow_args['animation_frame'] = resolved_animation

        # Squeeze singleton dimensions not used for faceting/animation
        # px.imshow can't handle extra singleton dims in multi-dimensional data
        dims_to_preserve = set(list(da.dims)[:2])  # First 2 dims are heatmap x/y axes
        if resolved_facet and resolved_facet in da.dims:
            dims_to_preserve.add(resolved_facet)
        if resolved_animation and resolved_animation in da.dims:
            dims_to_preserve.add(resolved_animation)
        for dim in list(da.dims):
            if dim not in dims_to_preserve and da.sizes[dim] == 1:
                da = da.squeeze(dim)
        imshow_args['img'] = da

        # Use binary_string=False to handle non-numeric coords (e.g., string labels)
        if 'binary_string' not in imshow_kwargs:
            imshow_args['binary_string'] = False

        return px.imshow(**{**imshow_args, **imshow_kwargs})
