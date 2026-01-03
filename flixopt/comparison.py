"""Compare multiple FlowSystems side-by-side."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

    # Core dimensions that must match across FlowSystems
    _CORE_DIMS = {'time', 'cluster', 'period', 'scenario'}

    def _validate_matching_dimensions(self) -> None:
        """Validate that all FlowSystems have matching core dimensions.

        Only validates core dimensions (time, period, scenario). Auxiliary
        dimensions like 'cluster_boundary' are ignored as they don't affect
        the comparison logic.
        """
        reference = self._systems[0]
        ref_core_dims = set(reference.solution.dims) & self._CORE_DIMS
        ref_name = self._names[0]

        for fs, name in zip(self._systems[1:], self._names[1:], strict=True):
            fs_core_dims = set(fs.solution.dims) & self._CORE_DIMS
            if fs_core_dims != ref_core_dims:
                missing = ref_core_dims - fs_core_dims
                extra = fs_core_dims - ref_core_dims
                msg_parts = [f"Core dimension mismatch between '{ref_name}' and '{name}'."]
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

    Wraps StatisticsPlotAccessor methods, combining data from all FlowSystems
    with a 'case' dimension for faceting.
    """

    def __init__(self, statistics: ComparisonStatistics) -> None:
        self._stats = statistics
        self._comp = statistics._comp

    def _combine_data(self, method_name: str, *args, **kwargs) -> tuple[xr.Dataset, str]:
        """Call plot method on each system and combine data. Returns (combined_data, title)."""
        datasets = []
        title = ''
        kwargs['show'] = False

        for fs, case_name in zip(self._comp._systems, self._comp._names, strict=True):
            try:
                result = getattr(fs.statistics.plot, method_name)(*args, **kwargs)
                datasets.append(result.data.expand_dims(case=[case_name]))
                title = result.figure.layout.title.text or title
            except (KeyError, ValueError):
                continue

        if not datasets:
            return xr.Dataset(), ''

        return xr.concat(datasets, dim='case', join='outer', fill_value=float('nan')), title

    def _resolve_facets(self, ds: xr.Dataset, facet_col='auto', facet_row='auto', animation_frame='auto'):
        """Resolve auto facets."""
        from .statistics_accessor import _resolve_auto_facets

        return _resolve_auto_facets(ds, facet_col, facet_row, animation_frame)

    def _finalize(self, ds: xr.Dataset, fig, show: bool | None) -> PlotResult:
        """Handle show and return PlotResult."""
        import plotly.graph_objects as go

        if show is None:
            show = CONFIG.Plotting.default_show
        if show and fig:
            fig.show()
        return PlotResult(data=ds, figure=fig or go.Figure())

    def balance(
        self,
        node: str,
        *,
        colors=None,
        facet_col='auto',
        facet_row='auto',
        animation_frame='auto',
        show: bool | None = None,
        **kwargs,
    ) -> PlotResult:
        """Plot node balance comparison. See StatisticsPlotAccessor.balance."""
        ds, title = self._combine_data('balance', node, **kwargs)
        if not ds.data_vars:
            return self._finalize(ds, None, show)
        col, row, anim = self._resolve_facets(ds, facet_col, facet_row, animation_frame)
        fig = ds.fxplot.stacked_bar(colors=colors, title=title, facet_col=col, facet_row=row, animation_frame=anim)
        return self._finalize(ds, fig, show)

    def carrier_balance(
        self,
        carrier: str,
        *,
        colors=None,
        facet_col='auto',
        facet_row='auto',
        animation_frame='auto',
        show: bool | None = None,
        **kwargs,
    ) -> PlotResult:
        """Plot carrier balance comparison. See StatisticsPlotAccessor.carrier_balance."""
        ds, title = self._combine_data('carrier_balance', carrier, **kwargs)
        if not ds.data_vars:
            return self._finalize(ds, None, show)
        col, row, anim = self._resolve_facets(ds, facet_col, facet_row, animation_frame)
        fig = ds.fxplot.stacked_bar(colors=colors, title=title, facet_col=col, facet_row=row, animation_frame=anim)
        return self._finalize(ds, fig, show)

    def flows(
        self,
        *,
        colors=None,
        facet_col='auto',
        facet_row='auto',
        animation_frame='auto',
        show: bool | None = None,
        **kwargs,
    ) -> PlotResult:
        """Plot flows comparison. See StatisticsPlotAccessor.flows."""
        ds, title = self._combine_data('flows', **kwargs)
        if not ds.data_vars:
            return self._finalize(ds, None, show)
        col, row, anim = self._resolve_facets(ds, facet_col, facet_row, animation_frame)
        fig = ds.fxplot.line(colors=colors, title=title, facet_col=col, facet_row=row, animation_frame=anim)
        return self._finalize(ds, fig, show)

    def storage(
        self,
        storage: str,
        *,
        colors=None,
        facet_col='auto',
        facet_row='auto',
        animation_frame='auto',
        show: bool | None = None,
        **kwargs,
    ) -> PlotResult:
        """Plot storage operation comparison. See StatisticsPlotAccessor.storage."""
        ds, title = self._combine_data('storage', storage, **kwargs)
        if not ds.data_vars:
            return self._finalize(ds, None, show)
        col, row, anim = self._resolve_facets(ds, facet_col, facet_row, animation_frame)
        fig = ds.fxplot.stacked_bar(colors=colors, title=title, facet_col=col, facet_row=row, animation_frame=anim)
        return self._finalize(ds, fig, show)

    def charge_states(
        self,
        storages=None,
        *,
        colors=None,
        facet_col='auto',
        facet_row='auto',
        animation_frame='auto',
        show: bool | None = None,
        **kwargs,
    ) -> PlotResult:
        """Plot charge states comparison. See StatisticsPlotAccessor.charge_states."""
        ds, title = self._combine_data('charge_states', storages, **kwargs)
        if not ds.data_vars:
            return self._finalize(ds, None, show)
        col, row, anim = self._resolve_facets(ds, facet_col, facet_row, animation_frame)
        fig = ds.fxplot.line(colors=colors, title=title, facet_col=col, facet_row=row, animation_frame=anim)
        fig.update_yaxes(title_text='Charge State')
        return self._finalize(ds, fig, show)

    def duration_curve(
        self,
        variables,
        *,
        normalize: bool = False,
        colors=None,
        facet_col='auto',
        facet_row='auto',
        animation_frame='auto',
        show: bool | None = None,
        **kwargs,
    ) -> PlotResult:
        """Plot duration curves comparison. See StatisticsPlotAccessor.duration_curve."""
        ds, title = self._combine_data('duration_curve', variables, normalize=normalize, **kwargs)
        if not ds.data_vars:
            return self._finalize(ds, None, show)
        col, row, anim = self._resolve_facets(ds, facet_col, facet_row, animation_frame)
        fig = ds.fxplot.line(colors=colors, title=title, facet_col=col, facet_row=row, animation_frame=anim)
        fig.update_xaxes(title_text='Duration [%]' if normalize else 'Timesteps')
        return self._finalize(ds, fig, show)

    def sizes(
        self,
        *,
        colors=None,
        facet_col='auto',
        facet_row='auto',
        animation_frame='auto',
        show: bool | None = None,
        **kwargs,
    ) -> PlotResult:
        """Plot investment sizes comparison. See StatisticsPlotAccessor.sizes."""
        import plotly.express as px

        from .color_processing import process_colors
        from .statistics_accessor import _dataset_to_long_df

        ds, title = self._combine_data('sizes', **kwargs)
        if not ds.data_vars:
            return self._finalize(ds, None, show)
        col, row, anim = self._resolve_facets(ds, facet_col, facet_row, animation_frame)
        df = _dataset_to_long_df(ds)
        color_map = process_colors(colors, df['variable'].unique().tolist()) if not df.empty else None
        fig = px.bar(
            df,
            x='variable',
            y='value',
            color='variable',
            title=title,
            facet_col=col,
            facet_row=row,
            animation_frame=anim,
            color_discrete_map=color_map,
        )
        return self._finalize(ds, fig, show)

    def effects(
        self,
        aspect='total',
        *,
        colors=None,
        facet_col='auto',
        facet_row='auto',
        animation_frame='auto',
        show: bool | None = None,
        **kwargs,
    ) -> PlotResult:
        """Plot effects comparison. See StatisticsPlotAccessor.effects."""
        import plotly.express as px

        from .color_processing import process_colors

        ds, title = self._combine_data('effects', aspect, **kwargs)
        if not ds.data_vars:
            return self._finalize(ds, None, show)
        col, row, anim = self._resolve_facets(ds, facet_col, facet_row, animation_frame)

        # Get the data array and convert to dataframe
        da = ds[aspect] if aspect in ds else ds[next(iter(ds.data_vars))]
        df = da.to_dataframe(name='value').reset_index()

        by = kwargs.get('by')
        x_col = by if by else 'effect'
        color_col = x_col if x_col in df.columns else None
        color_map = process_colors(colors, df[color_col].unique().tolist()) if color_col else None

        fig = px.bar(
            df,
            x=x_col,
            y='value',
            color=color_col,
            title=title,
            facet_col=col,
            facet_row=row,
            animation_frame=anim,
            color_discrete_map=color_map,
        )
        fig.update_layout(bargap=0, bargroupgap=0)
        fig.update_traces(marker_line_width=0)
        return self._finalize(ds, fig, show)

    def heatmap(
        self, variables, *, colors=None, facet_col='auto', animation_frame='auto', show: bool | None = None, **kwargs
    ) -> PlotResult:
        """Plot heatmap comparison. See StatisticsPlotAccessor.heatmap."""
        ds, _ = self._combine_data('heatmap', variables, **kwargs)
        if not ds.data_vars:
            return self._finalize(ds, None, show)
        col, _, anim = self._resolve_facets(ds, facet_col, None, animation_frame)
        da = ds[next(iter(ds.data_vars))]
        fig = da.fxplot.heatmap(colors=colors, facet_col=col, animation_frame=anim)
        return self._finalize(ds, fig, show)
