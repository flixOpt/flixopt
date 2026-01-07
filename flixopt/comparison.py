"""Compare multiple FlowSystems side-by-side."""

from __future__ import annotations

import warnings
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

    Combines solutions, statistics, and inputs from multiple FlowSystems into
    unified xarray Datasets with a 'case' dimension. The existing plotting
    infrastructure automatically handles faceting by the 'case' dimension.

    For comparing solutions/statistics, all FlowSystems must be optimized and
    have matching dimensions. For comparing inputs only, optimization is not
    required.

    Args:
        flow_systems: List of FlowSystems to compare.
        names: Optional names for each case. If None, uses FlowSystem.name.

    Raises:
        ValueError: If case names are not unique.
        RuntimeError: If accessing solution/statistics without optimized FlowSystems.

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
        self._names = names or [fs.name or f'System {i}' for i, fs in enumerate(flow_systems)]

        if len(self._names) != len(self._systems):
            raise ValueError(
                f'Number of names ({len(self._names)}) must match number of FlowSystems ({len(self._systems)})'
            )

        if len(set(self._names)) != len(self._names):
            raise ValueError(f'Case names must be unique, got: {self._names}')

        # Caches
        self._solution: xr.Dataset | None = None
        self._statistics: ComparisonStatistics | None = None
        self._inputs: xr.Dataset | None = None

    # Core dimensions that must match across FlowSystems
    # Note: 'cluster' and 'cluster_boundary' are auxiliary dimensions from clustering
    _CORE_DIMS = {'time', 'period', 'scenario'}

    def _warn_mismatched_dimensions(self, datasets: list[xr.Dataset]) -> None:
        """Warn if datasets have mismatched dimensions or coordinates.

        xarray handles mismatches gracefully with join='outer', but this may
        introduce NaN values for non-overlapping coordinates.
        """
        ref_ds = datasets[0]
        ref_core_dims = set(ref_ds.dims) & self._CORE_DIMS
        ref_name = self._names[0]

        for ds, name in zip(datasets[1:], self._names[1:], strict=True):
            ds_core_dims = set(ds.dims) & self._CORE_DIMS
            if ds_core_dims != ref_core_dims:
                missing = ref_core_dims - ds_core_dims
                extra = ds_core_dims - ref_core_dims
                msg_parts = [f"Dimension mismatch between '{ref_name}' and '{name}'."]
                if missing:
                    msg_parts.append(f'Missing: {missing}.')
                if extra:
                    msg_parts.append(f'Extra: {extra}.')
                msg_parts.append('This may introduce NaN values.')
                warnings.warn(' '.join(msg_parts), stacklevel=4)

            # Check coordinate alignment
            for dim in ref_core_dims & ds_core_dims:
                ref_coords = ref_ds.coords[dim].values
                ds_coords = ds.coords[dim].values
                if len(ref_coords) != len(ds_coords) or not (ref_coords == ds_coords).all():
                    warnings.warn(
                        f"Coordinates differ for '{dim}' between '{ref_name}' and '{name}'. "
                        f'This may introduce NaN values.',
                        stacklevel=4,
                    )

    @property
    def names(self) -> list[str]:
        """Case names for each FlowSystem."""
        return self._names

    def _require_solutions(self) -> None:
        """Validate all FlowSystems have solutions."""
        for fs in self._systems:
            if fs.solution is None:
                raise RuntimeError(f"FlowSystem '{fs.name}' has no solution. Run optimize() first.")

    @property
    def solution(self) -> xr.Dataset:
        """Combined solution Dataset with 'case' dimension."""
        if self._solution is None:
            self._require_solutions()
            datasets = [fs.solution for fs in self._systems]
            self._warn_mismatched_dimensions(datasets)
            self._solution = xr.concat(
                [ds.expand_dims(case=[name]) for ds, name in zip(datasets, self._names, strict=True)],
                dim='case',
                join='outer',
                fill_value=float('nan'),
            )
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
            n_cases = len(self._names)
            if not (-n_cases <= ref_idx < n_cases):
                raise IndexError(f'Reference index {ref_idx} out of range for {n_cases} cases.')

        ref_data = self.solution.isel(case=ref_idx)
        return self.solution - ref_data

    @property
    def inputs(self) -> xr.Dataset:
        """Combined input data Dataset with 'case' dimension.

        Concatenates input parameters from all FlowSystems. Each FlowSystem's
        ``.inputs`` Dataset is combined with a 'case' dimension.

        Returns:
            xr.Dataset with all input parameters. Variable naming follows
            the pattern ``{element.label_full}|{parameter_name}``.

        Examples:
            ```python
            comp = fx.Comparison([fs1, fs2], names=['Base', 'Modified'])
            comp.inputs  # All inputs with 'case' dimension
            comp.inputs['Boiler(Q_th)|relative_minimum']  # Specific parameter
            ```
        """
        if self._inputs is None:
            datasets = [fs.to_dataset(include_solution=False) for fs in self._systems]
            self._warn_mismatched_dimensions(datasets)
            self._inputs = xr.concat(
                [ds.expand_dims(case=[name]) for ds, name in zip(datasets, self._names, strict=True)],
                dim='case',
                join='outer',
                fill_value=float('nan'),
            )
        return self._inputs


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
            try:
                ds = getattr(fs.statistics, prop_name)
                datasets.append(ds.expand_dims(case=[name]))
            except RuntimeError as e:
                warnings.warn(f"Skipping case '{name}': {e}", stacklevel=3)
                continue
        if not datasets:
            return xr.Dataset()
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
        # Use data_only=True to skip figure creation for performance
        kwargs = {**kwargs, 'show': False, 'data_only': True}

        for fs, case_name in zip(self._comp._systems, self._comp._names, strict=True):
            try:
                result = getattr(fs.statistics.plot, method_name)(*args, **kwargs)
                datasets.append(result.data.expand_dims(case=[case_name]))
            except (KeyError, ValueError) as e:
                warnings.warn(
                    f"Skipping case '{case_name}' in {method_name}: {e}",
                    stacklevel=3,
                )
                continue

        if not datasets:
            return xr.Dataset(), ''

        return xr.concat(datasets, dim='case', join='outer', fill_value=float('nan')), title

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
        select: SelectType | None = None,
        include: FilterType | None = None,
        exclude: FilterType | None = None,
        unit: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        colors: ColorType | None = None,
        facet_col: str | Literal['auto'] | None = 'auto',
        facet_row: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs,
    ) -> PlotResult:
        """Plot node balance comparison. See StatisticsPlotAccessor.balance."""
        ds, title = self._combine_data('balance', node, select=select, include=include, exclude=exclude, unit=unit)
        if not ds.data_vars:
            return self._finalize(ds, None, show)
        fig = ds.fxplot.stacked_bar(
            colors=colors,
            title=title,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **plotly_kwargs,
        )
        return self._finalize(ds, fig, show)

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
        **plotly_kwargs,
    ) -> PlotResult:
        """Plot carrier balance comparison. See StatisticsPlotAccessor.carrier_balance."""
        ds, title = self._combine_data(
            'carrier_balance', carrier, select=select, include=include, exclude=exclude, unit=unit
        )
        if not ds.data_vars:
            return self._finalize(ds, None, show)
        fig = ds.fxplot.stacked_bar(
            colors=colors,
            title=title,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **plotly_kwargs,
        )
        return self._finalize(ds, fig, show)

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
        **plotly_kwargs,
    ) -> PlotResult:
        """Plot flows comparison. See StatisticsPlotAccessor.flows."""
        ds, title = self._combine_data('flows', start=start, end=end, component=component, select=select, unit=unit)
        if not ds.data_vars:
            return self._finalize(ds, None, show)
        fig = ds.fxplot.line(
            colors=colors,
            title=title,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **plotly_kwargs,
        )
        return self._finalize(ds, fig, show)

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
        **plotly_kwargs,
    ) -> PlotResult:
        """Plot storage operation comparison. See StatisticsPlotAccessor.storage."""
        ds, title = self._combine_data(
            'storage', storage, select=select, unit=unit, charge_state_color=charge_state_color
        )
        if not ds.data_vars:
            return self._finalize(ds, None, show)
        fig = ds.fxplot.stacked_bar(
            colors=colors,
            title=title,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **plotly_kwargs,
        )
        return self._finalize(ds, fig, show)

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
        **plotly_kwargs,
    ) -> PlotResult:
        """Plot charge states comparison. See StatisticsPlotAccessor.charge_states."""
        ds, title = self._combine_data('charge_states', storages, select=select)
        if not ds.data_vars:
            return self._finalize(ds, None, show)
        fig = ds.fxplot.line(
            colors=colors,
            title=title,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **plotly_kwargs,
        )
        fig.update_yaxes(title_text='Charge State')
        return self._finalize(ds, fig, show)

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
        **plotly_kwargs,
    ) -> PlotResult:
        """Plot duration curves comparison. See StatisticsPlotAccessor.duration_curve."""
        ds, title = self._combine_data('duration_curve', variables, select=select, normalize=normalize)
        if not ds.data_vars:
            return self._finalize(ds, None, show)
        fig = ds.fxplot.line(
            colors=colors,
            title=title,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **plotly_kwargs,
        )
        fig.update_xaxes(title_text='Duration [%]' if normalize else 'Timesteps')
        return self._finalize(ds, fig, show)

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
        **plotly_kwargs,
    ) -> PlotResult:
        """Plot investment sizes comparison. See StatisticsPlotAccessor.sizes."""
        ds, title = self._combine_data('sizes', max_size=max_size, select=select)
        if not ds.data_vars:
            return self._finalize(ds, None, show)
        fig = ds.fxplot.bar(
            x='variable',
            color='variable',
            colors=colors,
            title=title,
            ylabel='Size',
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **plotly_kwargs,
        )
        return self._finalize(ds, fig, show)

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
        **plotly_kwargs,
    ) -> PlotResult:
        """Plot effects comparison. See StatisticsPlotAccessor.effects."""
        ds, title = self._combine_data('effects', aspect, effect=effect, by=by, select=select)
        if not ds.data_vars:
            return self._finalize(ds, None, show)

        # After to_dataset(dim='effect'), effects become variables -> 'variable' column
        x_col = by if by else 'variable'

        fig = ds.fxplot.bar(
            x=x_col,
            colors=colors,
            title=title,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **plotly_kwargs,
        )
        fig.update_layout(bargap=0, bargroupgap=0)
        fig.update_traces(marker_line_width=0)
        return self._finalize(ds, fig, show)

    def heatmap(
        self,
        variables: str | list[str],
        *,
        select: SelectType | None = None,
        reshape: dict[str, int] | None = None,
        colors: str | list[str] | None = None,
        facet_col: str | Literal['auto'] | None = 'auto',
        animation_frame: str | Literal['auto'] | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs,
    ) -> PlotResult:
        """Plot heatmap comparison. See StatisticsPlotAccessor.heatmap."""
        ds, _ = self._combine_data('heatmap', variables, select=select, reshape=reshape)
        if not ds.data_vars:
            return self._finalize(ds, None, show)
        da = ds[next(iter(ds.data_vars))]
        fig = da.fxplot.heatmap(
            colors=colors,
            facet_col=facet_col,
            animation_frame=animation_frame,
            **plotly_kwargs,
        )
        return self._finalize(ds, fig, show)
