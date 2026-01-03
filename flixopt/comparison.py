"""Compare multiple FlowSystems side-by-side."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import xarray as xr

from .config import CONFIG
from .plot_result import PlotResult

if TYPE_CHECKING:
    from .flow_system import FlowSystem

__all__ = ['Comparison']

# Type aliases (matching statistics_accessor.py)
SelectType = dict[str, Any]
FilterType = str | list[str]
ColorType = str | list[str] | dict[str, str] | None


class Comparison:
    """Compare multiple FlowSystems side-by-side.

    Combines solutions and statistics from multiple FlowSystems into unified
    xarray Datasets with a 'case' dimension. The existing plotting infrastructure
    automatically handles faceting by the 'case' dimension.

    All FlowSystems must have matching dimensions (time, period, scenario, etc.).
    Use `flow_system.transform.sel()` to align dimensions before comparing.

    Args:
        flow_systems: List of FlowSystems to compare. All must be optimized
            and have matching dimensions.
        names: Optional names for each case. If None, uses FlowSystem.name.

    Raises:
        ValueError: If FlowSystems have mismatched dimensions.
        RuntimeError: If any FlowSystem has no solution.

    Examples:
        ```python
        # Compare two systems (uses FlowSystem.name by default)
        comp = fx.Comparison([fs_base, fs_modified])

        # Or with custom names
        comp = fx.Comparison([fs_base, fs_modified], names=['baseline', 'modified'])

        # Side-by-side plots (auto-facets by 'case')
        comp.statistics.plot.balance('Heat')
        comp.statistics.flow_rates.fxplot.line()

        # Access combined data
        comp.solution  # xr.Dataset with 'case' dimension
        comp.statistics.flow_rates  # xr.Dataset with 'case' dimension

        # Compute differences relative to first case
        comp.diff()  # Returns xr.Dataset of differences
        comp.diff('baseline')  # Or specify reference by name

        # For systems with different dimensions, align first:
        fs_both = ...  # Has scenario dimension
        fs_mild = fs_both.transform.sel(scenario='Mild')  # Select one scenario
        fs_other = ...  # Also select to match
        comp = fx.Comparison([fs_mild, fs_other])  # Now dimensions match
        ```
    """

    def __init__(self, flow_systems: list[FlowSystem], names: list[str] | None = None) -> None:
        if len(flow_systems) < 2:
            raise ValueError('Comparison requires at least 2 FlowSystems')

        self._systems = flow_systems
        self._names = names or [fs.name for fs in flow_systems]

        if len(self._names) != len(self._systems):
            raise ValueError(
                f'Number of names ({len(self._names)}) must match number of FlowSystems ({len(self._systems)})'
            )

        if len(set(self._names)) != len(self._names):
            raise ValueError(f'Case names must be unique, got: {self._names}')

        # Validate all FlowSystems have solutions
        for fs in flow_systems:
            if fs.solution is None:
                raise RuntimeError(f"FlowSystem '{fs.name}' has no solution. Run optimize() first.")

        # Validate matching dimensions across all FlowSystems
        self._validate_matching_dimensions()

        # Caches
        self._solution: xr.Dataset | None = None
        self._statistics: ComparisonStatistics | None = None

    def _validate_matching_dimensions(self) -> None:
        """Validate that all FlowSystems have matching dimensions."""
        reference = self._systems[0]
        ref_dims = set(reference.solution.dims)
        ref_name = self._names[0]

        for fs, name in zip(self._systems[1:], self._names[1:], strict=True):
            fs_dims = set(fs.solution.dims)
            if fs_dims != ref_dims:
                missing = ref_dims - fs_dims
                extra = fs_dims - ref_dims
                msg_parts = [f"Dimension mismatch between '{ref_name}' and '{name}'."]
                if missing:
                    msg_parts.append(f"Missing in '{name}': {missing}.")
                if extra:
                    msg_parts.append(f"Extra in '{name}': {extra}.")
                msg_parts.append('Use .transform.sel() to align dimensions before comparing.')
                raise ValueError(' '.join(msg_parts))

    @property
    def names(self) -> list[str]:
        """Case names for each FlowSystem."""
        return self._names

    @property
    def solution(self) -> xr.Dataset:
        """Combined solution Dataset with 'case' dimension."""
        if self._solution is None:
            datasets = []
            for fs, name in zip(self._systems, self._names, strict=True):
                ds = fs.solution.expand_dims(case=[name])
                datasets.append(ds)
            self._solution = xr.concat(datasets, dim='case', join='outer', fill_value=float('nan'))
        return self._solution

    @property
    def statistics(self) -> ComparisonStatistics:
        """Combined statistics accessor with 'case' dimension."""
        if self._statistics is None:
            self._statistics = ComparisonStatistics(self)
        return self._statistics

    def diff(self, reference: str | int = 0) -> xr.Dataset:
        """Compute differences relative to a reference case.

        Args:
            reference: Reference case name or index (default: 0, first case).

        Returns:
            Dataset with differences (each case minus reference).
        """
        if isinstance(reference, str):
            if reference not in self._names:
                raise ValueError(f"Reference '{reference}' not found. Available: {self._names}")
            ref_idx = self._names.index(reference)
        else:
            ref_idx = reference

        ref_data = self.solution.isel(case=ref_idx)
        return self.solution - ref_data


class ComparisonStatistics:
    """Combined statistics accessor for comparing FlowSystems.

    Mirrors StatisticsAccessor properties, concatenating data with a 'case' dimension.
    Access via ``Comparison.statistics``.
    """

    def __init__(self, comparison: Comparison) -> None:
        self._comp = comparison
        # Caches for dataset properties
        self._flow_rates: xr.Dataset | None = None
        self._flow_hours: xr.Dataset | None = None
        self._flow_sizes: xr.Dataset | None = None
        self._storage_sizes: xr.Dataset | None = None
        self._sizes: xr.Dataset | None = None
        self._charge_states: xr.Dataset | None = None
        self._temporal_effects: xr.Dataset | None = None
        self._periodic_effects: xr.Dataset | None = None
        self._total_effects: xr.Dataset | None = None
        # Caches for dict properties
        self._carrier_colors: dict[str, str] | None = None
        self._component_colors: dict[str, str] | None = None
        self._bus_colors: dict[str, str] | None = None
        self._carrier_units: dict[str, str] | None = None
        self._effect_units: dict[str, str] | None = None
        # Plot accessor
        self._plot: ComparisonStatisticsPlot | None = None

    def _concat_property(self, prop_name: str) -> xr.Dataset:
        """Concatenate a statistics property across all cases."""
        datasets = []
        for fs, name in zip(self._comp._systems, self._comp._names, strict=True):
            ds = getattr(fs.statistics, prop_name)
            datasets.append(ds.expand_dims(case=[name]))
        return xr.concat(datasets, dim='case', join='outer', fill_value=float('nan'))

    def _merge_dict_property(self, prop_name: str) -> dict[str, str]:
        """Merge a dict property from all cases (later cases override)."""
        result: dict[str, str] = {}
        for fs in self._comp._systems:
            result.update(getattr(fs.statistics, prop_name))
        return result

    @property
    def flow_rates(self) -> xr.Dataset:
        """Combined flow rates with 'case' dimension."""
        if self._flow_rates is None:
            self._flow_rates = self._concat_property('flow_rates')
        return self._flow_rates

    @property
    def flow_hours(self) -> xr.Dataset:
        """Combined flow hours (energy) with 'case' dimension."""
        if self._flow_hours is None:
            self._flow_hours = self._concat_property('flow_hours')
        return self._flow_hours

    @property
    def flow_sizes(self) -> xr.Dataset:
        """Combined flow investment sizes with 'case' dimension."""
        if self._flow_sizes is None:
            self._flow_sizes = self._concat_property('flow_sizes')
        return self._flow_sizes

    @property
    def storage_sizes(self) -> xr.Dataset:
        """Combined storage capacity sizes with 'case' dimension."""
        if self._storage_sizes is None:
            self._storage_sizes = self._concat_property('storage_sizes')
        return self._storage_sizes

    @property
    def sizes(self) -> xr.Dataset:
        """Combined sizes (flow + storage) with 'case' dimension."""
        if self._sizes is None:
            self._sizes = self._concat_property('sizes')
        return self._sizes

    @property
    def charge_states(self) -> xr.Dataset:
        """Combined storage charge states with 'case' dimension."""
        if self._charge_states is None:
            self._charge_states = self._concat_property('charge_states')
        return self._charge_states

    @property
    def temporal_effects(self) -> xr.Dataset:
        """Combined temporal effects with 'case' dimension."""
        if self._temporal_effects is None:
            self._temporal_effects = self._concat_property('temporal_effects')
        return self._temporal_effects

    @property
    def periodic_effects(self) -> xr.Dataset:
        """Combined periodic effects with 'case' dimension."""
        if self._periodic_effects is None:
            self._periodic_effects = self._concat_property('periodic_effects')
        return self._periodic_effects

    @property
    def total_effects(self) -> xr.Dataset:
        """Combined total effects with 'case' dimension."""
        if self._total_effects is None:
            self._total_effects = self._concat_property('total_effects')
        return self._total_effects

    @property
    def carrier_colors(self) -> dict[str, str]:
        """Merged carrier colors from all cases."""
        if self._carrier_colors is None:
            self._carrier_colors = self._merge_dict_property('carrier_colors')
        return self._carrier_colors

    @property
    def component_colors(self) -> dict[str, str]:
        """Merged component colors from all cases."""
        if self._component_colors is None:
            self._component_colors = self._merge_dict_property('component_colors')
        return self._component_colors

    @property
    def bus_colors(self) -> dict[str, str]:
        """Merged bus colors from all cases."""
        if self._bus_colors is None:
            self._bus_colors = self._merge_dict_property('bus_colors')
        return self._bus_colors

    @property
    def carrier_units(self) -> dict[str, str]:
        """Merged carrier units from all cases."""
        if self._carrier_units is None:
            self._carrier_units = self._merge_dict_property('carrier_units')
        return self._carrier_units

    @property
    def effect_units(self) -> dict[str, str]:
        """Merged effect units from all cases."""
        if self._effect_units is None:
            self._effect_units = self._merge_dict_property('effect_units')
        return self._effect_units

    @property
    def plot(self) -> ComparisonStatisticsPlot:
        """Access plot methods for comparison statistics."""
        if self._plot is None:
            self._plot = ComparisonStatisticsPlot(self)
        return self._plot


class ComparisonStatisticsPlot:
    """Plot accessor for comparison statistics.

    Mirrors StatisticsPlotAccessor methods, operating on combined data
    from multiple FlowSystems. The 'case' dimension is automatically
    used for faceting.
    """

    def __init__(self, statistics: ComparisonStatistics) -> None:
        self._stats = statistics
        self._comp = statistics._comp

    def _concat_plot_data(self, method_name: str, *args, **kwargs) -> xr.Dataset:
        """Call a plot method on each system and concatenate the resulting data.

        This ensures all data variables from all systems are included,
        even if topologies differ between systems.
        """
        # Disable show for individual calls, we'll handle it after combining
        kwargs['show'] = False
        datasets = []
        for fs, name in zip(self._comp._systems, self._comp._names, strict=True):
            try:
                plot_method = getattr(fs.statistics.plot, method_name)
                result = plot_method(*args, **kwargs)
                ds = result.data.expand_dims(case=[name])
                datasets.append(ds)
            except (KeyError, ValueError):
                # Node/element might not exist in this system - skip it
                continue

        if not datasets:
            return xr.Dataset()

        return xr.concat(datasets, dim='case', join='outer', fill_value=float('nan'))

    def balance(
        self,
        node: str,
        *,
        select: SelectType | None = None,
        include: FilterType | None = None,
        exclude: FilterType | None = None,
        unit: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        colors: ColorType = None,
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot node balance comparison across cases.

        See StatisticsPlotAccessor.balance for full documentation.
        The 'case' dimension is automatically used for faceting.
        """
        from .statistics_accessor import _resolve_auto_facets

        # Get combined data from all systems
        ds = self._concat_plot_data('balance', node, select=select, include=include, exclude=exclude, unit=unit)

        if not ds.data_vars:
            import plotly.graph_objects as go

            return PlotResult(data=xr.Dataset(), figure=go.Figure())

        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            ds, facet_col, facet_row, animation_frame
        )

        # Get unit label
        unit_label = ''
        if ds.data_vars:
            first_var = next(iter(ds.data_vars))
            unit_label = ds[first_var].attrs.get('unit', '')

        fig = ds.fxplot.stacked_bar(
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
        colors: ColorType = None,
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot carrier balance comparison across cases.

        See StatisticsPlotAccessor.carrier_balance for full documentation.
        """
        from .statistics_accessor import _resolve_auto_facets

        # Get combined data from all systems
        ds = self._concat_plot_data(
            'carrier_balance', carrier, select=select, include=include, exclude=exclude, unit=unit
        )

        if not ds.data_vars:
            import plotly.graph_objects as go

            return PlotResult(data=xr.Dataset(), figure=go.Figure())

        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            ds, facet_col, facet_row, animation_frame
        )

        unit_label = ''
        if ds.data_vars:
            first_var = next(iter(ds.data_vars))
            unit_label = ds[first_var].attrs.get('unit', '')

        fig = ds.fxplot.stacked_bar(
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

    def flows(
        self,
        *,
        start: str | list[str] | None = None,
        end: str | list[str] | None = None,
        component: str | list[str] | None = None,
        select: SelectType | None = None,
        unit: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        colors: ColorType = None,
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot flows comparison across cases.

        See StatisticsPlotAccessor.flows for full documentation.
        """
        from .statistics_accessor import _resolve_auto_facets

        # Get combined data from all systems
        ds = self._concat_plot_data('flows', start=start, end=end, component=component, select=select, unit=unit)

        if not ds.data_vars:
            import plotly.graph_objects as go

            return PlotResult(data=xr.Dataset(), figure=go.Figure())

        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            ds, facet_col, facet_row, animation_frame
        )

        unit_label = ''
        if ds.data_vars:
            first_var = next(iter(ds.data_vars))
            unit_label = ds[first_var].attrs.get('unit', '')

        fig = ds.fxplot.line(
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
        colors: ColorType = None,
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot investment sizes comparison across cases.

        See StatisticsPlotAccessor.sizes for full documentation.
        """
        import plotly.express as px

        from .color_processing import process_colors
        from .statistics_accessor import _dataset_to_long_df, _resolve_auto_facets

        # Get combined data from all systems
        ds = self._concat_plot_data('sizes', max_size=max_size, select=select)

        if not ds.data_vars:
            import plotly.graph_objects as go

            return PlotResult(data=xr.Dataset(), figure=go.Figure())

        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            ds, facet_col, facet_row, animation_frame
        )

        df = _dataset_to_long_df(ds)
        if df.empty:
            import plotly.graph_objects as go

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
        colors: ColorType = None,
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot duration curves comparison across cases.

        See StatisticsPlotAccessor.duration_curve for full documentation.
        """
        from .statistics_accessor import _resolve_auto_facets

        # Get combined data from all systems
        ds = self._concat_plot_data('duration_curve', variables, select=select, normalize=normalize)

        if not ds.data_vars:
            import plotly.graph_objects as go

            return PlotResult(data=xr.Dataset(), figure=go.Figure())

        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            ds, facet_col, facet_row, animation_frame
        )

        unit_label = ''
        if ds.data_vars:
            first_var = next(iter(ds.data_vars))
            unit_label = ds[first_var].attrs.get('unit', '')

        fig = ds.fxplot.line(
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

        return PlotResult(data=ds, figure=fig)

    def effects(
        self,
        aspect: Literal['total', 'temporal', 'periodic'] = 'total',
        *,
        effect: str | None = None,
        by: Literal['component', 'contributor', 'time'] | None = None,
        select: SelectType | None = None,
        colors: ColorType = None,
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot effects comparison across cases.

        See StatisticsPlotAccessor.effects for full documentation.
        """
        import plotly.express as px

        from .color_processing import process_colors
        from .statistics_accessor import _resolve_auto_facets

        # Get combined data from all systems
        ds = self._concat_plot_data('effects', aspect, effect=effect, by=by, select=select)

        if not ds.data_vars:
            import plotly.graph_objects as go

            return PlotResult(data=xr.Dataset(), figure=go.Figure())

        # The underlying effects method returns a Dataset with a single data var named after aspect
        # Convert back to DataArray for processing
        combined = ds[aspect] if aspect in ds else next(iter(ds.data_vars))
        if isinstance(combined, xr.Dataset):
            combined = combined[next(iter(combined.data_vars))]

        # Determine x_col and color_col based on dimensions
        if by is None:
            x_col = 'effect'
            color_col = 'effect'
        elif by == 'component':
            x_col = 'component'
            color_col = 'effect' if 'effect' in combined.dims and combined.sizes.get('effect', 1) > 1 else 'component'
        elif by == 'contributor':
            x_col = 'contributor'
            color_col = 'effect' if 'effect' in combined.dims and combined.sizes.get('effect', 1) > 1 else 'contributor'
        elif by == 'time':
            x_col = 'time'
            color_col = 'effect' if 'effect' in combined.dims and combined.sizes.get('effect', 1) > 1 else None
        else:
            x_col = 'effect'
            color_col = 'effect'

        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            combined.to_dataset(name='value'), facet_col, facet_row, animation_frame
        )

        df = combined.to_dataframe(name='value').reset_index()

        if color_col and color_col in df.columns:
            color_items = df[color_col].unique().tolist()
            color_map = process_colors(colors, color_items)
        else:
            color_map = None

        effect_label = effect if effect else 'Effects'
        title = f'{effect_label} ({aspect})' if by is None else f'{effect_label} ({aspect}) by {by}'

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

        return PlotResult(data=ds, figure=fig)

    def charge_states(
        self,
        storages: str | list[str] | None = None,
        *,
        select: SelectType | None = None,
        colors: ColorType = None,
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot charge states comparison across cases.

        See StatisticsPlotAccessor.charge_states for full documentation.
        """
        from .statistics_accessor import _resolve_auto_facets

        # Get combined data from all systems
        ds = self._concat_plot_data('charge_states', storages, select=select)

        if not ds.data_vars:
            import plotly.graph_objects as go

            return PlotResult(data=xr.Dataset(), figure=go.Figure())

        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            ds, facet_col, facet_row, animation_frame
        )

        fig = ds.fxplot.line(
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
        """Plot heatmap comparison across cases.

        See StatisticsPlotAccessor.heatmap for full documentation.
        """
        from .statistics_accessor import _resolve_auto_facets

        # Get combined data from all systems
        ds = self._concat_plot_data('heatmap', variables, select=select, reshape=reshape)

        if not ds.data_vars:
            import plotly.graph_objects as go

            return PlotResult(data=xr.Dataset(), figure=go.Figure())

        # Convert to DataArray for heatmap plotting
        if len(ds.data_vars) == 1:
            da = ds[next(iter(ds.data_vars))]
        else:
            import pandas as pd

            variable_names = list(ds.data_vars)
            dataarrays = [ds[var] for var in variable_names]
            da = xr.concat(dataarrays, dim=pd.Index(variable_names, name='variable'))

        actual_facet_col, _, actual_animation = _resolve_auto_facets(
            da.to_dataset(name='value'), facet_col, None, animation_frame
        )

        fig = da.fxplot.heatmap(
            colors=colors,
            facet_col=actual_facet_col,
            animation_frame=actual_animation,
            **plotly_kwargs,
        )

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
        colors: ColorType = None,
        charge_state_color: str = 'black',
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot storage operation comparison across cases.

        See StatisticsPlotAccessor.storage for full documentation.
        """
        from .statistics_accessor import _resolve_auto_facets

        # Get combined data from all systems
        ds = self._concat_plot_data('storage', storage, select=select, unit=unit, charge_state_color=charge_state_color)

        if not ds.data_vars:
            import plotly.graph_objects as go

            return PlotResult(data=xr.Dataset(), figure=go.Figure())

        actual_facet_col, actual_facet_row, actual_anim = _resolve_auto_facets(
            ds, facet_col, facet_row, animation_frame
        )

        # Create stacked bar for flows
        fig = ds.fxplot.stacked_bar(
            colors=colors,
            title=f'{storage} Operation',
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
