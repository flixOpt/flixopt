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

    Args:
        flow_systems: List of FlowSystems to compare.
        names: Optional names for each case. If None, uses FlowSystem.name.

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

        # Caches
        self._solution: xr.Dataset | None = None
        self._statistics: ComparisonStatistics | None = None

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
                if fs.solution is None:
                    raise RuntimeError(f"FlowSystem '{fs.name}' has no solution. Run optimize() first.")
                ds = fs.solution.expand_dims(case=[name])
                datasets.append(ds)
            self._solution = xr.concat(datasets, dim='case')
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
        return xr.concat(datasets, dim='case')

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

    def _get_first_stats_plot(self):
        """Get StatisticsPlotAccessor from first FlowSystem for delegation."""
        return self._comp._systems[0].statistics.plot

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
        from .statistics_accessor import _apply_selection, _filter_by_pattern, _resolve_auto_facets

        # Get flow labels from first system (assumes same topology)
        fs = self._comp._systems[0]
        if node in fs.buses:
            element = fs.buses[node]
        elif node in fs.components:
            element = fs.components[node]
        else:
            raise KeyError(f"'{node}' not found in buses or components")

        input_labels = [f.label_full for f in element.inputs]
        output_labels = [f.label_full for f in element.outputs]
        all_labels = input_labels + output_labels
        filtered_labels = _filter_by_pattern(all_labels, include, exclude)

        if not filtered_labels:
            import plotly.graph_objects as go

            return PlotResult(data=xr.Dataset(), figure=go.Figure())

        # Get combined data
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
        from .statistics_accessor import _apply_selection, _filter_by_pattern, _resolve_auto_facets

        carrier = carrier.lower()
        fs = self._comp._systems[0]

        carrier_buses = [bus for bus in fs.buses.values() if bus.carrier == carrier]
        if not carrier_buses:
            raise KeyError(f"No buses found with carrier '{carrier}'")

        input_labels: list[str] = []
        output_labels: list[str] = []
        for bus in carrier_buses:
            for flow in bus.inputs:
                input_labels.append(flow.label_full)
            for flow in bus.outputs:
                output_labels.append(flow.label_full)

        all_labels = input_labels + output_labels
        filtered_labels = _filter_by_pattern(all_labels, include, exclude)

        if not filtered_labels:
            import plotly.graph_objects as go

            return PlotResult(data=xr.Dataset(), figure=go.Figure())

        if unit == 'flow_rate':
            ds = self._stats.flow_rates[[lbl for lbl in filtered_labels if lbl in self._stats.flow_rates]]
        else:
            ds = self._stats.flow_hours[[lbl for lbl in filtered_labels if lbl in self._stats.flow_hours]]

        for label in output_labels:
            if label in ds:
                ds[label] = -ds[label]

        ds = _apply_selection(ds, select)
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
        from .statistics_accessor import _apply_selection, _resolve_auto_facets

        ds = self._stats.flow_rates if unit == 'flow_rate' else self._stats.flow_hours
        fs = self._comp._systems[0]

        if start is not None or end is not None or component is not None:
            matching_labels = []
            starts = [start] if isinstance(start, str) else (start or [])
            ends = [end] if isinstance(end, str) else (end or [])
            components = [component] if isinstance(component, str) else (component or [])

            for flow in fs.flows.values():
                bus_label = flow.bus
                comp_label = flow.component

                if flow.is_input_in_component:
                    if starts and bus_label not in starts:
                        continue
                    if ends and comp_label not in ends:
                        continue
                else:
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
        from .statistics_accessor import _apply_selection, _dataset_to_long_df, _resolve_auto_facets

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
        import numpy as np

        from .statistics_accessor import _apply_selection, _resolve_auto_facets

        if isinstance(variables, str):
            variables = [variables]

        flow_rates = self._stats.flow_rates
        solution = self._comp.solution

        normalized_vars = []
        for var in variables:
            if var.endswith('|flow_rate'):
                var = var[: -len('|flow_rate')]
            normalized_vars.append(var)

        ds_parts = []
        for var in normalized_vars:
            if var in flow_rates:
                ds_parts.append(flow_rates[[var]])
            elif var in solution:
                ds_parts.append(solution[[var]])
            else:
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

        unit_label = ''
        if ds.data_vars:
            first_var = next(iter(ds.data_vars))
            unit_label = ds[first_var].attrs.get('unit', '')

        fig = result_ds.fxplot.line(
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
        from .statistics_accessor import _apply_selection, _resolve_auto_facets

        if aspect == 'total':
            effects_ds = self._stats.total_effects
        elif aspect == 'temporal':
            effects_ds = self._stats.temporal_effects
        elif aspect == 'periodic':
            effects_ds = self._stats.periodic_effects
        else:
            raise ValueError(f"Aspect '{aspect}' not valid. Choose from 'total', 'temporal', 'periodic'.")

        available_effects = list(effects_ds.data_vars)

        if effect is not None:
            if effect not in available_effects:
                raise ValueError(f"Effect '{effect}' not found. Available: {available_effects}")
            effects_to_plot = [effect]
        else:
            effects_to_plot = available_effects

        effect_arrays = []
        for eff in effects_to_plot:
            da = effects_ds[eff]
            if by == 'contributor':
                effect_arrays.append(da.expand_dims(effect=[eff]))
            else:
                da_grouped = da.groupby('component').sum()
                effect_arrays.append(da_grouped.expand_dims(effect=[eff]))

        combined = xr.concat(effect_arrays, dim='effect')
        combined = _apply_selection(combined.to_dataset(name='value'), select)['value']

        if by is None:
            if 'time' in combined.dims:
                combined = combined.sum(dim='time')
            if 'component' in combined.dims:
                combined = combined.sum(dim='component')
            if 'contributor' in combined.dims:
                combined = combined.sum(dim='contributor')
            x_col = 'effect'
            color_col = 'effect'
        elif by == 'component':
            if 'time' in combined.dims:
                combined = combined.sum(dim='time')
            x_col = 'component'
            color_col = 'effect' if len(effects_to_plot) > 1 else 'component'
        elif by == 'contributor':
            if 'time' in combined.dims:
                combined = combined.sum(dim='time')
            x_col = 'contributor'
            color_col = 'effect' if len(effects_to_plot) > 1 else 'contributor'
        elif by == 'time':
            if 'time' not in combined.dims:
                raise ValueError(f"Cannot plot by 'time' for aspect '{aspect}' - no time dimension.")
            if 'component' in combined.dims:
                combined = combined.sum(dim='component')
            if 'contributor' in combined.dims:
                combined = combined.sum(dim='contributor')
            x_col = 'time'
            color_col = 'effect' if len(effects_to_plot) > 1 else None
        else:
            raise ValueError(f"'by' must be one of 'component', 'contributor', 'time', or None, got {by!r}")

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
        from .statistics_accessor import _apply_selection, _resolve_auto_facets

        ds = self._stats.charge_states

        if storages is not None:
            if isinstance(storages, str):
                storages = [storages]
            ds = ds[[s for s in storages if s in ds]]

        ds = _apply_selection(ds, select)
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
        import pandas as pd

        from .statistics_accessor import _apply_selection, _reshape_time_for_heatmap, _resolve_auto_facets

        solution = self._comp.solution

        if isinstance(variables, str):
            variables = [variables]

        # Resolve flow labels
        resolved_variables = []
        for var in variables:
            if var in solution:
                resolved_variables.append(var)
            elif '|' not in var:
                flow_rate_var = f'{var}|flow_rate'
                charge_state_var = f'{var}|charge_state'
                if flow_rate_var in solution:
                    resolved_variables.append(flow_rate_var)
                elif charge_state_var in solution:
                    resolved_variables.append(charge_state_var)
                else:
                    resolved_variables.append(var)
            else:
                resolved_variables.append(var)

        ds = solution[resolved_variables]
        ds = _apply_selection(ds, select)

        variable_names = list(ds.data_vars)
        dataarrays = [ds[var] for var in variable_names]
        da = xr.concat(dataarrays, dim=pd.Index(variable_names, name='variable'))

        is_clustered = 'cluster' in da.dims and da.sizes['cluster'] > 1
        has_multiple_vars = 'variable' in da.dims and da.sizes['variable'] > 1

        if has_multiple_vars:
            actual_facet = 'variable'
            _, _, actual_animation = _resolve_auto_facets(da.to_dataset(name='value'), None, None, animation_frame)
            if actual_animation == 'variable':
                actual_animation = None
        else:
            actual_facet, _, actual_animation = _resolve_auto_facets(
                da.to_dataset(name='value'), facet_col, None, animation_frame
            )

        if is_clustered and (reshape == 'auto' or reshape is None):
            heatmap_dims = ['time', 'cluster']
        elif reshape and reshape != 'auto' and 'time' in da.dims:
            da = _reshape_time_for_heatmap(da, reshape)
            heatmap_dims = ['timestep', 'timeframe']
        elif reshape == 'auto' and 'time' in da.dims and not is_clustered:
            da = _reshape_time_for_heatmap(da, ('D', 'h'))
            heatmap_dims = ['timestep', 'timeframe']
        elif has_multiple_vars:
            heatmap_dims = ['variable', 'time']
            actual_facet, _, actual_animation = _resolve_auto_facets(
                da.to_dataset(name='value'), facet_col, None, animation_frame
            )
        else:
            available_dims = [d for d in da.dims if da.sizes[d] > 1]
            if len(available_dims) >= 2:
                heatmap_dims = available_dims[:2]
            elif 'time' in da.dims:
                heatmap_dims = ['time']
            else:
                heatmap_dims = list(da.dims)[:1]

        keep_dims = set(heatmap_dims) | {d for d in [actual_facet, actual_animation] if d is not None}
        for dim in [d for d in da.dims if d not in keep_dims]:
            da = da.isel({dim: 0}, drop=True) if da.sizes[dim] > 1 else da.squeeze(dim, drop=True)

        dim_order = heatmap_dims + [d for d in [actual_facet, actual_animation] if d]
        da = da.transpose(*dim_order)

        if has_multiple_vars:
            da = da.rename('')

        fig = da.fxplot.heatmap(
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
        # Delegate to first system's plot method for now (complex subplot logic)
        # This is a simplification - for full support would need to reimplement
        from .statistics_accessor import _apply_selection, _resolve_auto_facets

        fs = self._comp._systems[0]
        if storage not in fs.components:
            raise KeyError(f"Storage '{storage}' not found in components")

        from .components import Storage as StorageComponent

        comp = fs.components[storage]
        if not isinstance(comp, StorageComponent):
            raise ValueError(f"'{storage}' is not a Storage component")

        # Get combined data
        input_labels = [f.label_full for f in comp.inputs]
        output_labels = [f.label_full for f in comp.outputs]
        all_labels = input_labels + output_labels

        if unit == 'flow_rate':
            ds = self._stats.flow_rates[[lbl for lbl in all_labels if lbl in self._stats.flow_rates]]
        else:
            ds = self._stats.flow_hours[[lbl for lbl in all_labels if lbl in self._stats.flow_hours]]

        for label in input_labels:
            if label in ds:
                ds[label] = -ds[label]

        charge_ds = self._stats.charge_states[[storage]] if storage in self._stats.charge_states else None

        ds = _apply_selection(ds, select)
        if charge_ds is not None:
            charge_ds = _apply_selection(charge_ds, select)

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

        # Combine data
        if charge_ds is not None:
            combined_ds = xr.merge([ds, charge_ds])
        else:
            combined_ds = ds

        return PlotResult(data=combined_ds, figure=fig)
