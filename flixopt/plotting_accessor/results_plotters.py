"""Plotter classes for results visualization in flixopt.

This module provides specialized plotter classes for visualizing CalculationResults,
ComponentResults, and BusResults. These plotters integrate with the .plot accessor
pattern and provide both convenient methods and flexible parameter handling.

Architecture:
- ResultsPlotterBase: Base class with common functionality
- NodeBalancePlotter: For node balance visualization
- PieChartPlotter: For pie chart visualization
- ChargeStatePlotter: For storage charge state visualization
- HeatmapPlotter: For heatmap visualization

All plotters support both Plotly and Matplotlib backends and integrate with
flixopt's color management and export capabilities.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, Literal

import xarray as xr

from .. import plotting
from .data_transformer import DataTransformer

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go


class ResultsPlotterBase:
    """Base class for all results plotters.

    Provides common functionality for data access, color management,
    export handling, and integration with flixopt's plotting infrastructure.

    Args:
        data: xarray Dataset or DataArray to visualize
        parent: Parent results object (ComponentResults, BusResults, or CalculationResults)
        name: Name for plot titles and file naming
        folder: Default save folder for plots

    Attributes:
        data: The data to visualize
        parent: Parent results object
        name: Name for titles/files
        folder: Default save folder
    """

    def __init__(
        self,
        data: xr.Dataset | xr.DataArray,
        parent: Any,
        name: str,
        folder: pathlib.Path | None = None,
    ):
        """Initialize the plotter with data and configuration."""
        self.data = data
        self.parent = parent
        self.name = name
        self.folder = folder or pathlib.Path('.')
        self._transformer = DataTransformer()

    def _get_colors(self) -> dict[str, str] | None:
        """Get color mapping from parent results object."""
        if hasattr(self.parent, 'colors'):
            return self.parent.colors
        if hasattr(self.parent, '_calculation_results') and hasattr(self.parent._calculation_results, 'colors'):
            return self.parent._calculation_results.colors
        return None

    def _export_figure(
        self,
        figure_like: go.Figure | tuple[plt.Figure, plt.Axes],
        title: str,
        save: bool | pathlib.Path = False,
        show: bool | None = None,
        dpi: int | None = None,
        default_filetype: str = '.html',
    ) -> go.Figure | tuple[plt.Figure, plt.Axes]:
        """Export figure using flixopt's export infrastructure.

        Args:
            figure_like: Figure to export
            title: Title for file naming
            save: Whether/where to save
            show: Whether to display
            dpi: DPI for raster exports
            default_filetype: Default file extension

        Returns:
            The figure object
        """
        return plotting.export_figure(
            figure_like=figure_like,
            default_path=self.folder / title,
            default_filetype=default_filetype,
            user_path=None if isinstance(save, bool) else pathlib.Path(save),
            show=show,
            save=True if save else False,
            dpi=dpi,
        )

    def _to_tidy(self, data: xr.Dataset | xr.DataArray | None = None, value_name: str = 'value'):
        """Convert data to tidy/long format DataFrame.

        This is the recommended format for Plotly Express.

        Args:
            data: Data to convert. If None, uses self.data.
            value_name: Name for value column (for DataArrays)

        Returns:
            Tidy DataFrame ready for Plotly Express
        """
        if data is None:
            data = self.data
        return self._transformer.to_tidy_dataframe(data, value_name=value_name)

    def _to_wide(self, data: xr.DataArray | None = None, index_dim: str | None = None, columns_dim: str | None = None):
        """Convert data to wide format DataFrame.

        Args:
            data: DataArray to convert. If None, uses self.data.
            index_dim: Dimension for index (usually 'time')
            columns_dim: Dimension for columns (usually category)

        Returns:
            Wide DataFrame
        """
        if data is None:
            data = self.data
        return self._transformer.to_wide_dataframe(data, index_dim=index_dim, columns_dim=columns_dim)

    def _aggregate(
        self,
        data: xr.DataArray | xr.Dataset | None = None,
        dim: str = 'time',
        method: Literal['sum', 'mean', 'max', 'min', 'std', 'median'] = 'sum',
    ):
        """Aggregate data along a dimension.

        Args:
            data: Data to aggregate. If None, uses self.data.
            dim: Dimension to aggregate along
            method: Aggregation method

        Returns:
            Aggregated data
        """
        if data is None:
            data = self.data
        return self._transformer.aggregate_dimension(data, dim=dim, method=method)

    def _select(self, data: xr.DataArray | xr.Dataset | None = None, **selectors: Any):
        """Select subset of data.

        Args:
            data: Data to select from. If None, uses self.data.
            **selectors: Dimension selectors

        Returns:
            Selected subset
        """
        if data is None:
            data = self.data
        return self._transformer.select_subset(data, **selectors)

    def _melt(self, data: xr.Dataset | None = None, var_name: str = 'variable', value_name: str = 'value'):
        """Convert Dataset to ultra-tidy format.

        Args:
            data: Dataset to melt. If None, uses self.data.
            var_name: Name for variable column
            value_name: Name for value column

        Returns:
            Melted DataFrame
        """
        if data is None:
            data = self.data
        return self._transformer.melt_dataset(data, var_name=var_name, value_name=value_name)


class NodeBalancePlotter(ResultsPlotterBase):
    """Plotter for node balance visualization.

    Provides methods for visualizing flow balances at nodes (components or buses)
    with support for different chart types, faceting, and animation.

    Examples:
        >>> # Via accessor
        >>> plotter = results['Boiler'].plot.node_balance()
        >>> fig = plotter.bar()
        >>> fig = plotter.area(facet_by='scenario')
        >>> fig = plotter.line(animate_by='period')
    """

    def __init__(
        self,
        data: xr.Dataset,
        parent: Any,
        name: str,
        folder: pathlib.Path | None = None,
        unit_type: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
    ):
        """Initialize node balance plotter.

        Args:
            data: Node balance dataset
            parent: Parent results object
            name: Node name
            folder: Save folder
            unit_type: Whether data is 'flow_rate' or 'flow_hours'
        """
        super().__init__(data, parent, name, folder)
        self.unit_type = unit_type

    def _make_title(self, suffix: str = '') -> str:
        """Create title for plot."""
        unit_label = 'flow rates' if self.unit_type == 'flow_rate' else 'flow hours'
        return f'{self.name} ({unit_label}){suffix}'

    def _plot_with_engine(
        self,
        mode: Literal['stacked_bar', 'grouped_bar', 'line', 'area'],
        engine: plotting.PlottingEngine = 'plotly',
        colors: plotting.ColorType | None = None,
        title: str | None = None,
        ylabel: str = '',
        xlabel: str = 'Time in h',
        facet_by: str | list[str] | None = None,
        animate_by: str | None = None,
        facet_cols: int | None = None,
        save: bool | pathlib.Path = False,
        show: bool | None = None,
        dpi: int | None = None,
        **plot_kwargs: Any,
    ) -> go.Figure | tuple[plt.Figure, plt.Axes]:
        """Internal plotting method that handles both engines.

        Args:
            mode: Plot mode (stacked_bar, grouped_bar, line, area)
            engine: Plotting engine (plotly or matplotlib)
            colors: Color specification
            title: Plot title
            ylabel: Y-axis label
            xlabel: X-axis label
            facet_by: Dimension(s) for faceting
            animate_by: Dimension for animation
            facet_cols: Number of facet columns
            save: Whether/where to save
            show: Whether to display
            dpi: DPI for exports
            **plot_kwargs: Additional plotting parameters

        Returns:
            Figure object
        """
        if engine not in {'plotly', 'matplotlib'}:
            raise ValueError(f'Engine "{engine}" not supported. Use "plotly" or "matplotlib"')

        # Use parent colors if not specified
        if colors is None:
            colors = self._get_colors()

        # Generate title if not provided
        if title is None:
            title = self._make_title()

        # Plot with appropriate engine
        if engine == 'plotly':
            figure_like = plotting.with_plotly(
                self.data,
                mode=mode,
                colors=colors,
                title=title,
                ylabel=ylabel,
                xlabel=xlabel,
                facet_by=facet_by,
                animate_by=animate_by,
                facet_cols=facet_cols,
                **plot_kwargs,
            )
            default_filetype = '.html'
        else:  # matplotlib
            # Check for extra dimensions
            extra_dims = [d for d in self.data.dims if d != 'time']
            if extra_dims:
                raise ValueError(
                    f'Matplotlib engine only supports a single time axis, but found extra dimensions: {extra_dims}. '
                    f'Use select={{...}} to reduce dimensions or switch to engine="plotly" for faceting/animation.'
                )
            figure_like = plotting.with_matplotlib(
                self.data,
                mode=mode if mode != 'grouped_bar' else 'stacked_bar',  # matplotlib doesn't support grouped_bar
                colors=colors,
                title=title,
                ylabel=ylabel,
                xlabel=xlabel,
                **plot_kwargs,
            )
            default_filetype = '.png'

        return self._export_figure(figure_like, title, save=save, show=show, dpi=dpi, default_filetype=default_filetype)

    def bar(
        self,
        mode: Literal['stacked', 'grouped'] = 'stacked',
        engine: plotting.PlottingEngine = 'plotly',
        **kwargs,
    ) -> go.Figure | tuple[plt.Figure, plt.Axes]:
        """Create a bar chart.

        Args:
            mode: 'stacked' or 'grouped'
            engine: 'plotly' or 'matplotlib'
            **kwargs: Additional parameters for _plot_with_engine()

        Returns:
            Figure object
        """
        plot_mode = 'stacked_bar' if mode == 'stacked' else 'grouped_bar'
        return self._plot_with_engine(mode=plot_mode, engine=engine, **kwargs)

    def line(
        self,
        engine: plotting.PlottingEngine = 'plotly',
        **kwargs,
    ) -> go.Figure | tuple[plt.Figure, plt.Axes]:
        """Create a line chart.

        Args:
            engine: 'plotly' or 'matplotlib'
            **kwargs: Additional parameters for _plot_with_engine()

        Returns:
            Figure object
        """
        return self._plot_with_engine(mode='line', engine=engine, **kwargs)

    def area(
        self,
        engine: plotting.PlottingEngine = 'plotly',
        **kwargs,
    ) -> go.Figure | tuple[plt.Figure, plt.Axes]:
        """Create an area chart.

        Args:
            engine: 'plotly' or 'matplotlib'
            **kwargs: Additional parameters for _plot_with_engine()

        Returns:
            Figure object
        """
        return self._plot_with_engine(mode='area', engine=engine, **kwargs)

    def plot(
        self,
        mode: Literal['stacked_bar', 'grouped_bar', 'line', 'area'] = 'stacked_bar',
        **kwargs,
    ) -> go.Figure | tuple[plt.Figure, plt.Axes]:
        """Create a plot with specified mode.

        Args:
            mode: Plot mode
            **kwargs: Additional parameters for _plot_with_engine()

        Returns:
            Figure object
        """
        return self._plot_with_engine(mode=mode, **kwargs)


class PieChartPlotter(ResultsPlotterBase):
    """Plotter for pie chart visualization.

    Creates dual pie charts showing input and output flow distributions.

    Examples:
        >>> # Via accessor
        >>> plotter = results['Bus'].plot.node_balance_pie()
        >>> fig = plotter.pie()
        >>> fig = plotter.donut(hole=0.4)
    """

    def __init__(
        self,
        data_left: xr.Dataset,
        data_right: xr.Dataset,
        parent: Any,
        name: str,
        folder: pathlib.Path | None = None,
    ):
        """Initialize pie chart plotter.

        Args:
            data_left: Left pie chart data (usually inputs)
            data_right: Right pie chart data (usually outputs)
            parent: Parent results object
            name: Node name
            folder: Save folder
        """
        super().__init__(data_left, parent, name, folder)
        self.data_right = data_right

    def _make_title(self, suffix: str = '') -> str:
        """Create title for plot."""
        return f'{self.name} (total flow hours){suffix}'

    def pie(
        self,
        lower_percentage_group: float = 5.0,
        colors: plotting.ColorType | None = None,
        text_info: str = 'percent+label+value',
        text_position: str = 'inside',
        hole: float = 0.0,
        engine: plotting.PlottingEngine = 'plotly',
        title: str | None = None,
        save: bool | pathlib.Path = False,
        show: bool | None = None,
        dpi: int | None = None,
        **kwargs,
    ) -> go.Figure | tuple[plt.Figure, list[plt.Axes]]:
        """Create dual pie charts.

        Args:
            lower_percentage_group: Threshold for grouping small slices
            colors: Color specification
            text_info: Text to display on slices
            text_position: Position of text
            hole: Size of center hole (0=pie, >0=donut)
            engine: Plotting engine
            title: Plot title
            save: Whether/where to save
            show: Whether to display
            dpi: DPI for exports
            **kwargs: Additional parameters

        Returns:
            Figure object
        """
        if engine not in {'plotly', 'matplotlib'}:
            raise ValueError(f'Engine "{engine}" not supported. Use "plotly" or "matplotlib"')

        # Use parent colors if not specified
        if colors is None:
            colors = self._get_colors()

        # Generate title if not provided
        if title is None:
            title = self._make_title()

        # Plot with appropriate engine
        if engine == 'plotly':
            figure_like = plotting.dual_pie_with_plotly(
                data_left=self.data,
                data_right=self.data_right,
                colors=colors,
                title=title,
                text_info=text_info,
                text_position=text_position,
                hole=hole,
                subtitles=('Inputs', 'Outputs'),
                legend_title='Flows',
                lower_percentage_group=lower_percentage_group,
                **kwargs,
            )
            default_filetype = '.html'
        else:  # matplotlib
            figure_like = plotting.dual_pie_with_matplotlib(
                data_left=self.data,
                data_right=self.data_right,
                colors=colors,
                title=title,
                hole=hole,
                subtitles=('Inputs', 'Outputs'),
                legend_title='Flows',
                lower_percentage_group=lower_percentage_group,
                **kwargs,
            )
            default_filetype = '.png'

        return self._export_figure(figure_like, title, save=save, show=show, dpi=dpi, default_filetype=default_filetype)

    def donut(self, hole: float = 0.4, **kwargs) -> go.Figure | tuple[plt.Figure, list[plt.Axes]]:
        """Create dual donut charts (pie charts with center hole).

        Args:
            hole: Size of center hole (default 0.4)
            **kwargs: Additional parameters for pie()

        Returns:
            Figure object
        """
        return self.pie(hole=hole, **kwargs)


class ChargeStatePlotter(ResultsPlotterBase):
    """Plotter for storage charge state visualization.

    Creates plots showing storage flows and charge state together,
    with charge state as an overlay line.

    Examples:
        >>> # Via accessor
        >>> plotter = results['Storage'].plot.charge_state()
        >>> fig = plotter.area()
        >>> fig = plotter.overlay(overlay_color='red')
    """

    def __init__(
        self,
        flow_data: xr.Dataset,
        charge_state_data: xr.DataArray,
        parent: Any,
        name: str,
        folder: pathlib.Path | None = None,
        charge_state_var_name: str = 'charge_state',
    ):
        """Initialize charge state plotter.

        Args:
            flow_data: Flow balance dataset
            charge_state_data: Charge state DataArray
            parent: Parent results object
            name: Storage name
            folder: Save folder
            charge_state_var_name: Variable name for charge state
        """
        super().__init__(flow_data, parent, name, folder)
        self.charge_state_data = charge_state_data
        self.charge_state_var_name = charge_state_var_name

    def _make_title(self, suffix: str = '') -> str:
        """Create title for plot."""
        return f'Operation Balance of {self.name}{suffix}'

    def _create_overlay(
        self,
        mode: Literal['area', 'stacked_bar', 'line'],
        overlay_color: str = 'black',
        engine: plotting.PlottingEngine = 'plotly',
        colors: plotting.ColorType | None = None,
        title: str | None = None,
        facet_by: str | list[str] | None = None,
        animate_by: str | None = None,
        facet_cols: int | None = None,
        save: bool | pathlib.Path = False,
        show: bool | None = None,
        dpi: int | None = None,
        **plot_kwargs: Any,
    ) -> go.Figure | tuple[plt.Figure, plt.Axes]:
        """Create flow plot with charge state overlay.

        Args:
            mode: Plot mode for flows
            overlay_color: Color for charge state line
            engine: Plotting engine
            colors: Color specification
            title: Plot title
            facet_by: Dimension(s) for faceting
            animate_by: Dimension for animation
            facet_cols: Number of facet columns
            save: Whether/where to save
            show: Whether to display
            dpi: DPI for exports
            **plot_kwargs: Additional parameters

        Returns:
            Figure object
        """
        if engine not in {'plotly', 'matplotlib'}:
            raise ValueError(f'Engine "{engine}" not supported. Use "plotly" or "matplotlib"')

        # Use parent colors if not specified
        if colors is None:
            colors = self._get_colors()

        # Generate title if not provided
        if title is None:
            title = self._make_title()

        if engine == 'plotly':
            # Plot flows
            figure_like = plotting.with_plotly(
                self.data,
                mode=mode,
                colors=colors,
                title=title,
                xlabel='Time in h',
                facet_by=facet_by,
                animate_by=animate_by,
                facet_cols=facet_cols,
                **plot_kwargs,
            )

            # Prepare charge state as Dataset
            charge_state_ds = xr.Dataset({self.charge_state_var_name: self.charge_state_data})

            # Plot charge state as line
            charge_state_fig = plotting.with_plotly(
                charge_state_ds,
                mode='line',
                colors=colors,
                title='',
                xlabel='Time in h',
                facet_by=facet_by,
                animate_by=animate_by,
                facet_cols=facet_cols,
                **plot_kwargs,
            )

            # Add charge state traces to main figure
            for trace in charge_state_fig.data:
                trace.line.width = 2
                trace.line.shape = 'linear'
                trace.line.color = overlay_color
                figure_like.add_trace(trace)

            # Handle animation frames
            if hasattr(charge_state_fig, 'frames') and charge_state_fig.frames:
                for i, frame in enumerate(charge_state_fig.frames):
                    if i < len(figure_like.frames):
                        for trace in frame.data:
                            trace.line.width = 2
                            trace.line.shape = 'linear'
                            trace.line.color = overlay_color
                            figure_like.frames[i].data = figure_like.frames[i].data + (trace,)

            default_filetype = '.html'

        else:  # matplotlib
            # Check for extra dimensions
            extra_dims = [d for d in self.data.dims if d != 'time']
            if extra_dims:
                raise ValueError(
                    f'Matplotlib engine only supports a single time axis, but found extra dimensions: {extra_dims}. '
                    f'Use select={{...}} to reduce dimensions or switch to engine="plotly" for faceting/animation.'
                )

            # Plot flows
            fig, ax = plotting.with_matplotlib(
                self.data,
                mode=mode if mode != 'grouped_bar' else 'stacked_bar',
                colors=colors,
                title=title,
                xlabel='Time in h',
                **plot_kwargs,
            )

            # Add charge state as line overlay
            charge_state_df = self.charge_state_data.to_dataframe()
            ax.plot(
                charge_state_df.index,
                charge_state_df.values.flatten(),
                label=self.charge_state_var_name,
                linewidth=2,
                color=overlay_color,
            )

            # Recreate legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                labels,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                ncol=5,
                frameon=False,
            )
            fig.tight_layout()

            figure_like = fig, ax
            default_filetype = '.png'

        return self._export_figure(figure_like, title, save=save, show=show, dpi=dpi, default_filetype=default_filetype)

    def overlay(
        self,
        mode: Literal['area', 'stacked_bar', 'line'] = 'area',
        overlay_color: str = 'black',
        **kwargs,
    ) -> go.Figure | tuple[plt.Figure, plt.Axes]:
        """Create plot with charge state overlay.

        Args:
            mode: Plot mode for flows
            overlay_color: Color for charge state line
            **kwargs: Additional parameters for _create_overlay()

        Returns:
            Figure object
        """
        return self._create_overlay(mode=mode, overlay_color=overlay_color, **kwargs)

    def area(self, **kwargs) -> go.Figure | tuple[plt.Figure, plt.Axes]:
        """Create area chart with charge state overlay.

        Args:
            **kwargs: Additional parameters for _create_overlay()

        Returns:
            Figure object
        """
        return self._create_overlay(mode='area', **kwargs)

    def bar(self, **kwargs) -> go.Figure | tuple[plt.Figure, plt.Axes]:
        """Create bar chart with charge state overlay.

        Args:
            **kwargs: Additional parameters for _create_overlay()

        Returns:
            Figure object
        """
        return self._create_overlay(mode='stacked_bar', **kwargs)

    def line(self, **kwargs) -> go.Figure | tuple[plt.Figure, plt.Axes]:
        """Create line chart with charge state overlay.

        Args:
            **kwargs: Additional parameters for _create_overlay()

        Returns:
            Figure object
        """
        return self._create_overlay(mode='line', **kwargs)


class HeatmapPlotter(ResultsPlotterBase):
    """Plotter for heatmap visualization.

    Creates heatmap visualizations with support for time reshaping,
    multi-variable plots, faceting, and animation.

    Examples:
        >>> # Via accessor
        >>> plotter = results.plot.heatmap('Variable')
        >>> fig = plotter.heatmap()
        >>> fig = plotter.heatmap(reshape_time=('D', 'h'))
    """

    def __init__(
        self,
        data: xr.DataArray,
        parent: Any,
        name: str,
        folder: pathlib.Path | None = None,
    ):
        """Initialize heatmap plotter.

        Args:
            data: Data to visualize
            parent: Parent results object
            name: Variable name
            folder: Save folder
        """
        super().__init__(data, parent, name, folder)

    def heatmap(
        self,
        reshape_time: tuple[Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'], Literal['W', 'D', 'h', '15min', 'min']]
        | Literal['auto']
        | None = 'auto',
        fill: Literal['ffill', 'bfill'] | None = 'ffill',
        colors: plotting.ColorType | None = None,
        engine: plotting.PlottingEngine = 'plotly',
        title: str | None = None,
        facet_by: str | list[str] | None = None,
        animate_by: str | None = None,
        facet_cols: int | None = None,
        save: bool | pathlib.Path = False,
        show: bool | None = None,
        dpi: int | None = None,
        **plot_kwargs: Any,
    ) -> go.Figure | tuple[plt.Figure, plt.Axes]:
        """Create heatmap visualization.

        Args:
            reshape_time: Time reshaping configuration
            fill: Fill method for missing values
            colors: Color specification
            engine: Plotting engine
            title: Plot title
            facet_by: Dimension(s) for faceting
            animate_by: Dimension for animation
            facet_cols: Number of facet columns
            save: Whether/where to save
            show: Whether to display
            dpi: DPI for exports
            **plot_kwargs: Additional parameters

        Returns:
            Figure object
        """
        if engine not in {'plotly', 'matplotlib'}:
            raise ValueError(f'Engine "{engine}" not supported. Use "plotly" or "matplotlib"')

        # Use parent colors if not specified
        if colors is None:
            colors = self._get_colors()

        # Generate title if not provided
        if title is None:
            title_name = self.data.name if self.data.name else self.name
            if isinstance(reshape_time, tuple):
                timeframes, timesteps_per_frame = reshape_time
                title = f'{title_name} ({timeframes} vs {timesteps_per_frame})'
            else:
                title = title_name

        # Plot with appropriate engine
        if engine == 'plotly':
            figure_like = plotting.heatmap_with_plotly(
                data=self.data,
                colors=colors,
                title=title,
                facet_by=facet_by,
                animate_by=animate_by,
                facet_cols=facet_cols,
                reshape_time=reshape_time,
                fill=fill,
                **plot_kwargs,
            )
            default_filetype = '.html'
        else:  # matplotlib
            # Matplotlib has more restrictions on dimensions
            if facet_by or animate_by:
                raise ValueError(
                    'Matplotlib heatmaps do not support faceting or animation. '
                    'Switch to engine="plotly" or remove facet_by/animate_by parameters.'
                )
            figure_like = plotting.heatmap_with_matplotlib(
                data=self.data,
                colors=colors,
                title=title,
                reshape_time=reshape_time,
                fill=fill,
                **plot_kwargs,
            )
            default_filetype = '.png'

        return self._export_figure(figure_like, title, save=save, show=show, dpi=dpi, default_filetype=default_filetype)

    def imshow(self, **kwargs) -> go.Figure | tuple[plt.Figure, plt.Axes]:
        """Alias for heatmap() method.

        Args:
            **kwargs: Parameters for heatmap()

        Returns:
            Figure object
        """
        return self.heatmap(**kwargs)
