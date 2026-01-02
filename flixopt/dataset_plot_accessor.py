"""Dataset plot accessor for xarray Datasets.

Provides convenient plotting methods for any xr.Dataset via the .fxplot accessor.
This is globally registered and available on all xr.Dataset objects when flixopt is imported.

Example:
    >>> import flixopt
    >>> import xarray as xr
    >>> ds = xr.Dataset({'temp': (['time', 'location'], data)})
    >>> ds.fxplot.line()  # Line plot of all variables
    >>> ds.fxplot.stacked_bar()  # Stacked bar chart
    >>> ds.fxplot.heatmap('temp')  # Heatmap of specific variable
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr

from .color_processing import ColorType, process_colors
from .config import CONFIG


def _get_x_dim(dims: list[str], x: str | Literal['auto'] | None = 'auto') -> str:
    """Determine the x-axis dimension from available dimensions.

    Args:
        dims: List of available dimension names.
        x: Explicit dimension name, 'auto' to use priority list, or None.

    Returns:
        Dimension name to use for x-axis. Returns 'variable' for scalar data.
    """
    if x and x != 'auto':
        return x

    # Check priority list first
    for dim in CONFIG.Plotting.x_dim_priority:
        if dim in dims:
            return dim

    # Fallback to first available dimension, or 'variable' for scalar data
    return dims[0] if dims else 'variable'


def _resolve_auto_facets(
    ds: xr.Dataset,
    facet_col: str | Literal['auto'] | None,
    facet_row: str | Literal['auto'] | None,
    animation_frame: str | Literal['auto'] | None = None,
    exclude_dims: set[str] | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Resolve 'auto' facet/animation dimensions based on available data dimensions.

    When 'auto' is specified, extra dimensions are assigned to slots based on:
    - CONFIG.Plotting.extra_dim_priority: Order of dimensions (default: cluster -> period -> scenario)
    - CONFIG.Plotting.dim_slot_priority: Order of slots (default: facet_col -> facet_row -> animation_frame)

    Args:
        ds: Dataset to check for available dimensions.
        facet_col: Dimension name, 'auto', or None.
        facet_row: Dimension name, 'auto', or None.
        animation_frame: Dimension name, 'auto', or None.
        exclude_dims: Dimensions to exclude (e.g., x-axis dimension).

    Returns:
        Tuple of (resolved_facet_col, resolved_facet_row, resolved_animation_frame).
        Each is either a valid dimension name or None.
    """
    # Get available extra dimensions with size > 1, excluding specified dims
    exclude = exclude_dims or set()
    available = {d for d in ds.dims if ds.sizes[d] > 1 and d not in exclude}
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


def _dataset_to_long_df(ds: xr.Dataset, value_name: str = 'value', var_name: str = 'variable') -> pd.DataFrame:
    """Convert xarray Dataset to long-form DataFrame for plotly express."""
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
        """Create a grouped bar chart from the dataset.

        Args:
            x: Dimension for x-axis. 'auto' uses CONFIG.Plotting.x_dim_priority.
            colors: Color specification (colorscale name, color list, or dict mapping).
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            facet_col: Dimension for column facets. 'auto' uses CONFIG priority.
            facet_row: Dimension for row facets. 'auto' uses CONFIG priority.
            animation_frame: Dimension for animation slider.
            facet_cols: Number of columns in facet grid wrap.
            **px_kwargs: Additional arguments passed to plotly.express.bar.

        Returns:
            Plotly Figure.
        """
        # Determine x-axis first, then resolve facets from remaining dims
        dims = list(self._ds.dims)
        x_col = _get_x_dim(dims, x)
        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            self._ds, facet_col, facet_row, animation_frame, exclude_dims={x_col}
        )

        df = _dataset_to_long_df(self._ds)
        if df.empty:
            return go.Figure()

        variables = df['variable'].unique().tolist()
        color_map = process_colors(colors, variables, default_colorscale=CONFIG.Plotting.default_qualitative_colorscale)

        facet_col_wrap = facet_cols or CONFIG.Plotting.default_facet_cols
        fig_kwargs: dict[str, Any] = {
            'data_frame': df,
            'x': x_col,
            'y': 'value',
            'title': title,
            'barmode': 'group',
            **px_kwargs,
        }
        # Only color by variable if it's not already on x-axis
        if x_col != 'variable':
            fig_kwargs['color'] = 'variable'
            fig_kwargs['color_discrete_map'] = color_map
        if xlabel:
            fig_kwargs['labels'] = {**fig_kwargs.get('labels', {}), x_col: xlabel}
        if ylabel:
            fig_kwargs['labels'] = {**fig_kwargs.get('labels', {}), 'value': ylabel}

        if actual_facet_col:
            fig_kwargs['facet_col'] = actual_facet_col
            if facet_col_wrap < self._ds.sizes.get(actual_facet_col, facet_col_wrap + 1):
                fig_kwargs['facet_col_wrap'] = facet_col_wrap
        if actual_facet_row:
            fig_kwargs['facet_row'] = actual_facet_row
        if actual_anim:
            fig_kwargs['animation_frame'] = actual_anim

        return px.bar(**fig_kwargs)

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
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create a stacked bar chart from the dataset.

        Variables in the dataset become stacked segments. Positive and negative
        values are stacked separately.

        Args:
            x: Dimension for x-axis. 'auto' uses CONFIG.Plotting.x_dim_priority.
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
        # Determine x-axis first, then resolve facets from remaining dims
        dims = list(self._ds.dims)
        x_col = _get_x_dim(dims, x)
        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            self._ds, facet_col, facet_row, animation_frame, exclude_dims={x_col}
        )

        df = _dataset_to_long_df(self._ds)
        if df.empty:
            return go.Figure()

        variables = df['variable'].unique().tolist()
        color_map = process_colors(colors, variables, default_colorscale=CONFIG.Plotting.default_qualitative_colorscale)

        facet_col_wrap = facet_cols or CONFIG.Plotting.default_facet_cols
        fig_kwargs: dict[str, Any] = {
            'data_frame': df,
            'x': x_col,
            'y': 'value',
            'title': title,
            **px_kwargs,
        }
        # Only color by variable if it's not already on x-axis
        if x_col != 'variable':
            fig_kwargs['color'] = 'variable'
            fig_kwargs['color_discrete_map'] = color_map
        if xlabel:
            fig_kwargs['labels'] = {**fig_kwargs.get('labels', {}), x_col: xlabel}
        if ylabel:
            fig_kwargs['labels'] = {**fig_kwargs.get('labels', {}), 'value': ylabel}

        if actual_facet_col:
            fig_kwargs['facet_col'] = actual_facet_col
            if facet_col_wrap < self._ds.sizes.get(actual_facet_col, facet_col_wrap + 1):
                fig_kwargs['facet_col_wrap'] = facet_col_wrap
        if actual_facet_row:
            fig_kwargs['facet_row'] = actual_facet_row
        if actual_anim:
            fig_kwargs['animation_frame'] = actual_anim

        fig = px.bar(**fig_kwargs)
        fig.update_layout(barmode='relative', bargap=0, bargroupgap=0)
        fig.update_traces(marker_line_width=0)
        return fig

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
        """Create a line chart from the dataset.

        Each variable in the dataset becomes a separate line.

        Args:
            x: Dimension for x-axis. 'auto' uses CONFIG.Plotting.x_dim_priority.
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
        # Determine x-axis first, then resolve facets from remaining dims
        dims = list(self._ds.dims)
        x_col = _get_x_dim(dims, x)
        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            self._ds, facet_col, facet_row, animation_frame, exclude_dims={x_col}
        )

        df = _dataset_to_long_df(self._ds)
        if df.empty:
            return go.Figure()

        variables = df['variable'].unique().tolist()
        color_map = process_colors(colors, variables, default_colorscale=CONFIG.Plotting.default_qualitative_colorscale)

        facet_col_wrap = facet_cols or CONFIG.Plotting.default_facet_cols
        fig_kwargs: dict[str, Any] = {
            'data_frame': df,
            'x': x_col,
            'y': 'value',
            'title': title,
            'line_shape': line_shape or CONFIG.Plotting.default_line_shape,
            **px_kwargs,
        }
        # Only color by variable if it's not already on x-axis
        if x_col != 'variable':
            fig_kwargs['color'] = 'variable'
            fig_kwargs['color_discrete_map'] = color_map
        if xlabel:
            fig_kwargs['labels'] = {**fig_kwargs.get('labels', {}), x_col: xlabel}
        if ylabel:
            fig_kwargs['labels'] = {**fig_kwargs.get('labels', {}), 'value': ylabel}

        if actual_facet_col:
            fig_kwargs['facet_col'] = actual_facet_col
            if facet_col_wrap < self._ds.sizes.get(actual_facet_col, facet_col_wrap + 1):
                fig_kwargs['facet_col_wrap'] = facet_col_wrap
        if actual_facet_row:
            fig_kwargs['facet_row'] = actual_facet_row
        if actual_anim:
            fig_kwargs['animation_frame'] = actual_anim

        return px.line(**fig_kwargs)

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
        """Create a stacked area chart from the dataset.

        Args:
            x: Dimension for x-axis. 'auto' uses CONFIG.Plotting.x_dim_priority.
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
        # Determine x-axis first, then resolve facets from remaining dims
        dims = list(self._ds.dims)
        x_col = _get_x_dim(dims, x)
        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            self._ds, facet_col, facet_row, animation_frame, exclude_dims={x_col}
        )

        df = _dataset_to_long_df(self._ds)
        if df.empty:
            return go.Figure()

        variables = df['variable'].unique().tolist()
        color_map = process_colors(colors, variables, default_colorscale=CONFIG.Plotting.default_qualitative_colorscale)

        facet_col_wrap = facet_cols or CONFIG.Plotting.default_facet_cols
        fig_kwargs: dict[str, Any] = {
            'data_frame': df,
            'x': x_col,
            'y': 'value',
            'title': title,
            'line_shape': line_shape or CONFIG.Plotting.default_line_shape,
            **px_kwargs,
        }
        # Only color by variable if it's not already on x-axis
        if x_col != 'variable':
            fig_kwargs['color'] = 'variable'
            fig_kwargs['color_discrete_map'] = color_map
        if xlabel:
            fig_kwargs['labels'] = {**fig_kwargs.get('labels', {}), x_col: xlabel}
        if ylabel:
            fig_kwargs['labels'] = {**fig_kwargs.get('labels', {}), 'value': ylabel}

        if actual_facet_col:
            fig_kwargs['facet_col'] = actual_facet_col
            if facet_col_wrap < self._ds.sizes.get(actual_facet_col, facet_col_wrap + 1):
                fig_kwargs['facet_col_wrap'] = facet_col_wrap
        if actual_facet_row:
            fig_kwargs['facet_row'] = actual_facet_row
        if actual_anim:
            fig_kwargs['animation_frame'] = actual_anim

        return px.area(**fig_kwargs)

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

        actual_facet_col, _, actual_anim = _resolve_auto_facets(self._ds, facet_col, None, animation_frame)

        imshow_args: dict[str, Any] = {
            'img': da,
            'color_continuous_scale': colors,
            'title': title or variable,
            **imshow_kwargs,
        }

        if actual_facet_col and actual_facet_col in da.dims:
            imshow_args['facet_col'] = actual_facet_col
            if facet_col_wrap < da.sizes[actual_facet_col]:
                imshow_args['facet_col_wrap'] = facet_col_wrap

        if actual_anim and actual_anim in da.dims:
            imshow_args['animation_frame'] = actual_anim

        return px.imshow(**imshow_args)

    def scatter(
        self,
        x: str,
        y: str,
        *,
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
        """Create a scatter plot from two variables in the dataset.

        Args:
            x: Variable name for x-axis.
            y: Variable name for y-axis.
            colors: Color specification (colorscale name, color list, or dict mapping).
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

        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            self._ds, facet_col, facet_row, animation_frame
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

        if actual_facet_col:
            fig_kwargs['facet_col'] = actual_facet_col
            if facet_col_wrap < self._ds.sizes.get(actual_facet_col, facet_col_wrap + 1):
                fig_kwargs['facet_col_wrap'] = facet_col_wrap
        if actual_facet_row:
            fig_kwargs['facet_row'] = actual_facet_row
        if actual_anim:
            fig_kwargs['animation_frame'] = actual_anim

        return px.scatter(**fig_kwargs)

    def pie(
        self,
        *,
        colors: ColorType | None = None,
        title: str = '',
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        facet_cols: int | None = None,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create a pie chart from aggregated dataset values.

        Extra dimensions are auto-assigned to facet_col, facet_row, and animation_frame.
        For scalar values, a single pie is shown.

        Args:
            colors: Color specification (colorscale name, color list, or dict mapping).
            title: Plot title.
            facet_col: Dimension for column facets. 'auto' uses CONFIG priority.
            facet_row: Dimension for row facets. 'auto' uses CONFIG priority.
            animation_frame: Dimension for animation slider. 'auto' uses CONFIG priority.
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

        # Multi-dimensional case - faceted/animated pies
        df = _dataset_to_long_df(self._ds)
        if df.empty:
            return go.Figure()

        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            self._ds, facet_col, facet_row, animation_frame
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

        if actual_facet_col:
            fig_kwargs['facet_col'] = actual_facet_col
            if facet_col_wrap < self._ds.sizes.get(actual_facet_col, facet_col_wrap + 1):
                fig_kwargs['facet_col_wrap'] = facet_col_wrap
        if actual_facet_row:
            fig_kwargs['facet_row'] = actual_facet_row
        if actual_anim:
            fig_kwargs['animation_frame'] = actual_anim

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
            # Sort along time axis (descending)
            sorted_values = np.sort(da.values, axis=da.dims.index('time'))[::-1]
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

        # Use Dataset for facet resolution
        ds_for_resolution = da.to_dataset(name='_temp')
        actual_facet_col, _, actual_anim = _resolve_auto_facets(ds_for_resolution, facet_col, None, animation_frame)

        imshow_args: dict[str, Any] = {
            'img': da,
            'color_continuous_scale': colors,
            'title': title or (da.name if da.name else ''),
            **imshow_kwargs,
        }

        if actual_facet_col and actual_facet_col in da.dims:
            imshow_args['facet_col'] = actual_facet_col
            if facet_col_wrap < da.sizes[actual_facet_col]:
                imshow_args['facet_col_wrap'] = facet_col_wrap

        if actual_anim and actual_anim in da.dims:
            imshow_args['animation_frame'] = actual_anim

        return px.imshow(**imshow_args)
