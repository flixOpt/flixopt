"""Compare multiple FlowSystems side-by-side."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, overload

import xarray as xr
from xarray_plotly import SLOT_ORDERS
from xarray_plotly.figures import add_secondary_y

from .config import CONFIG
from .plot_result import PlotResult
from .statistics_accessor import (
    _SLOT_DEFAULTS,
    ColorType,
    SelectType,
    _build_color_kwargs,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .flow_system import FlowSystem

__all__ = ['Comparison']

# Extract all unique slot names from xarray_plotly
_CASE_SLOTS = frozenset(slot for slots in SLOT_ORDERS.values() for slot in slots)


def _extract_nonindex_coords(datasets: list[xr.Dataset]) -> tuple[list[xr.Dataset], dict[str, tuple[str, dict]]]:
    """Extract and merge non-index coords, returning cleaned datasets and merged mappings.

    Non-index coords (like `component` on `contributor` dim) cause concat conflicts.
    This extracts them, merges the mappings, and returns datasets without them.
    """
    if not datasets:
        return datasets, {}

    # Find non-index coords and collect mappings
    merged: dict[str, tuple[str, dict]] = {}
    coords_to_drop: set[str] = set()

    for ds in datasets:
        for name, coord in ds.coords.items():
            if len(coord.dims) != 1:
                continue
            dim = coord.dims[0]
            if dim == name or dim not in ds.coords:
                continue

            coords_to_drop.add(name)
            if name not in merged:
                merged[name] = (dim, {})

            for dv, cv in zip(ds.coords[dim].values, coord.values, strict=False):
                if dv not in merged[name][1]:
                    merged[name][1][dv] = cv
                elif merged[name][1][dv] != cv:
                    warnings.warn(
                        f"Coordinate '{name}' has conflicting values for dim value '{dv}': "
                        f"'{merged[name][1][dv]}' vs '{cv}'. Keeping first value.",
                        stacklevel=4,
                    )

    # Drop these coords from datasets
    if coords_to_drop:
        datasets = [ds.drop_vars(coords_to_drop, errors='ignore') for ds in datasets]

    return datasets, merged


def _apply_merged_coords(ds: xr.Dataset, merged: dict[str, tuple[str, dict]]) -> xr.Dataset:
    """Apply merged coord mappings to concatenated dataset."""
    if not merged:
        return ds

    new_coords = {}
    for name, (dim, mapping) in merged.items():
        if dim not in ds.dims:
            continue
        new_coords[name] = (dim, [mapping.get(dv, dv) for dv in ds.coords[dim].values])

    return ds.assign_coords(new_coords)


def _apply_slot_defaults(plotly_kwargs: dict, defaults: dict[str, str | None]) -> None:
    """Apply default slot assignments to plotly kwargs.

    Args:
        plotly_kwargs: The kwargs dict to update (modified in place).
        defaults: Default slot assignments. None values block slots.
    """
    # Check if 'case' is already assigned by user to any slot
    case_already_assigned = any(plotly_kwargs.get(s) == 'case' for s in _CASE_SLOTS)

    for slot, value in defaults.items():
        if value == 'case' and case_already_assigned:
            # Skip case assignment if user already assigned 'case' to another slot
            continue
        plotly_kwargs.setdefault(slot, value)


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
        comp.stats.plot.balance('Heat')
        comp.stats.flow_rates.plotly.line()

        # Access combined data
        comp.solution  # xr.Dataset with 'case' dimension
        comp.stats.flow_rates  # xr.Dataset with 'case' dimension

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

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        lines = ['Comparison', '=' * 10]

        # Case info with optimization status
        lines.append(f'Cases ({len(self._names)}):')
        for name, fs in zip(self._names, self._systems, strict=True):
            status = '✓' if fs.solution is not None else '○'
            lines.append(f'  {status} {name}')

        # Shared dimensions
        shared_dims = self.dims
        if shared_dims:
            dims_str = ', '.join(f'{k}: {v}' for k, v in shared_dims.items())
            lines.append(f'Shared dims: {dims_str}')

        return '\n'.join(lines)

    def __len__(self) -> int:
        """Return number of cases."""
        return len(self._systems)

    @overload
    def __getitem__(self, key: int) -> FlowSystem: ...
    @overload
    def __getitem__(self, key: str) -> FlowSystem: ...

    def __getitem__(self, key: int | str) -> FlowSystem:
        """Access FlowSystem by name or index.

        Args:
            key: Case name (str) or index (int).

        Returns:
            The FlowSystem for that case.

        Raises:
            KeyError: If name not found.
            IndexError: If index out of range.
        """
        if isinstance(key, int):
            return self._systems[key]
        if key in self._names:
            idx = self._names.index(key)
            return self._systems[idx]
        raise KeyError(f"Case '{key}' not found. Available: {self._names}")

    def __iter__(self) -> Iterator[tuple[str, FlowSystem]]:
        """Iterate over (name, FlowSystem) pairs."""
        yield from zip(self._names, self._systems, strict=True)

    def __contains__(self, key: str) -> bool:
        """Check if a case name exists."""
        return key in self._names

    @property
    def flow_systems(self) -> dict[str, FlowSystem]:
        """Access underlying FlowSystems as a dict mapping name → FlowSystem."""
        return dict(zip(self._names, self._systems, strict=True))

    @property
    def is_optimized(self) -> bool:
        """Check if all FlowSystems have been optimized."""
        return all(fs.solution is not None for fs in self._systems)

    @property
    def dims(self) -> dict[str, int]:
        """Shared dimensions across all FlowSystems.

        Returns dimensions that exist in all systems with matching sizes.
        """
        if not self._systems:
            return {}

        # Start with first system's dims
        ref_dims = dict(self._systems[0].solution.sizes) if self._systems[0].solution else {}
        if not ref_dims:
            return {}

        # Keep only dims that match across all systems
        shared = {}
        for dim, size in ref_dims.items():
            if all(fs.solution is not None and fs.solution.sizes.get(dim) == size for fs in self._systems[1:]):
                shared[dim] = size

        return shared

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
            expanded = [ds.expand_dims(case=[name]) for ds, name in zip(datasets, self._names, strict=True)]
            expanded, merged_coords = _extract_nonindex_coords(expanded)
            result = xr.concat(expanded, dim='case', join='outer', coords='minimal', fill_value=float('nan'))
            self._solution = _apply_merged_coords(result, merged_coords)
        return self._solution

    @property
    def stats(self) -> ComparisonStatistics:
        """Combined statistics accessor with 'case' dimension."""
        if self._statistics is None:
            self._statistics = ComparisonStatistics(self)
        return self._statistics

    @property
    def statistics(self) -> ComparisonStatistics:
        """Deprecated: Use :attr:`stats` instead."""
        warnings.warn(
            "The 'statistics' accessor is deprecated. Use 'stats' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.stats

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
            expanded = [ds.expand_dims(case=[name]) for ds, name in zip(datasets, self._names, strict=True)]
            expanded, merged_coords = _extract_nonindex_coords(expanded)
            result = xr.concat(expanded, dim='case', join='outer', coords='minimal', fill_value=float('nan'))
            self._inputs = _apply_merged_coords(result, merged_coords)
        return self._inputs


class ComparisonStatistics:
    """Combined statistics accessor for comparing FlowSystems.

    Mirrors StatisticsAccessor properties, concatenating data with a 'case' dimension.
    Access via ``Comparison.stats``.
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
        self._flow_colors: dict[str, str] | None = None
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
                ds = getattr(fs.stats, prop_name)
                datasets.append(ds.expand_dims(case=[name]))
            except RuntimeError as e:
                warnings.warn(f"Skipping case '{name}': {e}", stacklevel=3)
                continue
        if not datasets:
            return xr.Dataset()
        datasets, merged_coords = _extract_nonindex_coords(datasets)
        result = xr.concat(datasets, dim='case', join='outer', coords='minimal', fill_value=float('nan'))
        return _apply_merged_coords(result, merged_coords)

    def _merge_dict_property(self, prop_name: str) -> dict[str, str]:
        """Merge a dict property from all cases (later cases override)."""
        result: dict[str, str] = {}
        for fs in self._comp._systems:
            result.update(getattr(fs.stats, prop_name))
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
    def flow_colors(self) -> dict[str, str]:
        """Merged flow colors from all cases (derived from parent components)."""
        if self._flow_colors is None:
            self._flow_colors = self._merge_dict_property('flow_colors')
        return self._flow_colors

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
                result = getattr(fs.stats.plot, method_name)(*args, **kwargs)
                datasets.append(result.data.expand_dims(case=[case_name]))
            except (KeyError, ValueError) as e:
                warnings.warn(
                    f"Skipping case '{case_name}' in {method_name}: {e}",
                    stacklevel=3,
                )
                continue

        if not datasets:
            return xr.Dataset(), ''

        datasets, merged_coords = _extract_nonindex_coords(datasets)
        combined = xr.concat(datasets, dim='case', join='outer', coords='minimal', fill_value=float('nan'))
        return _apply_merged_coords(combined, merged_coords), title

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
        include: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
        unit: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        colors: ColorType | None = None,
        threshold: float | None = 1e-5,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot node balance comparison across cases.

        Args:
            node: Bus or component label to plot balance for.
            select: xarray-style selection.
            include: Filter to include only matching flow labels.
            exclude: Filter to exclude matching flow labels.
            unit: 'flow_rate' or 'flow_hours'.
            colors: Color specification (dict, list, or colorscale name).
            threshold: Filter out variables where max absolute value is below this.
            show: Whether to display the figure.
            data_only: If True, skip figure creation and return only data.
            **plotly_kwargs: Additional arguments passed to plotly.

        Returns:
            PlotResult with combined balance data and figure.
        """
        ds, _ = self._combine_data(
            'balance', node, select=select, include=include, exclude=exclude, unit=unit, threshold=threshold
        )
        if not ds.data_vars or data_only:
            return self._finalize(ds, None, show if not data_only else False)

        defaults = {'x': 'time', 'color': 'variable', 'pattern_shape': None, 'facet_col': 'case'}
        _apply_slot_defaults(plotly_kwargs, defaults)
        color_kwargs = _build_color_kwargs(colors, list(ds.data_vars))
        fig = ds.plotly.bar(
            title=f'{node} Balance Comparison',
            **color_kwargs,
            **plotly_kwargs,
        )
        fig.update_layout(barmode='relative', bargap=0, bargroupgap=0)
        fig.update_traces(marker_line_width=0)
        return self._finalize(ds, fig, show)

    def carrier_balance(
        self,
        carrier: str,
        *,
        select: SelectType | None = None,
        include: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
        unit: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        colors: ColorType | None = None,
        threshold: float | None = 1e-5,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot carrier balance comparison across cases.

        Args:
            carrier: Carrier name to plot balance for.
            select: xarray-style selection.
            include: Filter to include only matching flow labels.
            exclude: Filter to exclude matching flow labels.
            unit: 'flow_rate' or 'flow_hours'.
            colors: Color specification (dict, list, or colorscale name).
            threshold: Filter out variables where max absolute value is below this.
            show: Whether to display the figure.
            data_only: If True, skip figure creation and return only data.
            **plotly_kwargs: Additional arguments passed to plotly.

        Returns:
            PlotResult with combined carrier balance data and figure.
        """
        ds, _ = self._combine_data(
            'carrier_balance', carrier, select=select, include=include, exclude=exclude, unit=unit, threshold=threshold
        )
        if not ds.data_vars or data_only:
            return self._finalize(ds, None, show if not data_only else False)

        defaults = {'x': 'time', 'color': 'variable', 'pattern_shape': None, 'facet_col': 'case'}
        _apply_slot_defaults(plotly_kwargs, defaults)
        color_kwargs = _build_color_kwargs(colors, list(ds.data_vars))
        fig = ds.plotly.bar(
            title=f'{carrier.capitalize()} Balance Comparison',
            **color_kwargs,
            **plotly_kwargs,
        )
        fig.update_layout(barmode='relative', bargap=0, bargroupgap=0)
        fig.update_traces(marker_line_width=0)
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
        threshold: float | None = 1e-5,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot flows comparison across cases.

        Args:
            start: Filter by source node(s).
            end: Filter by destination node(s).
            component: Filter by parent component(s).
            select: xarray-style selection.
            unit: 'flow_rate' or 'flow_hours'.
            colors: Color specification (dict, list, or colorscale name).
            threshold: Filter out variables where max absolute value is below this.
            show: Whether to display the figure.
            data_only: If True, skip figure creation and return only data.
            **plotly_kwargs: Additional arguments passed to plotly.

        Returns:
            PlotResult with combined flows data and figure.
        """
        ds, _ = self._combine_data(
            'flows', start=start, end=end, component=component, select=select, unit=unit, threshold=threshold
        )
        if not ds.data_vars or data_only:
            return self._finalize(ds, None, show if not data_only else False)

        defaults = {'x': 'time', 'color': 'variable', 'symbol': None, 'line_dash': 'case'}
        _apply_slot_defaults(plotly_kwargs, defaults)
        color_kwargs = _build_color_kwargs(colors, list(ds.data_vars))
        fig = ds.plotly.line(
            title='Flows Comparison',
            **color_kwargs,
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
        threshold: float | None = 1e-5,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot storage operation comparison across cases.

        Args:
            storage: Storage component label.
            select: xarray-style selection.
            unit: 'flow_rate' or 'flow_hours'.
            colors: Color specification for flow bars.
            threshold: Filter out variables where max absolute value is below this.
            show: Whether to display the figure.
            data_only: If True, skip figure creation and return only data.
            **plotly_kwargs: Additional arguments passed to plotly.

        Returns:
            PlotResult with combined storage operation data and figure.
        """
        ds, _ = self._combine_data('storage', storage, select=select, unit=unit, threshold=threshold)
        if not ds.data_vars or data_only:
            return self._finalize(ds, None, show if not data_only else False)

        # Separate flows from charge_state
        flow_vars = [v for v in ds.data_vars if v != 'charge_state']
        flow_ds = ds[flow_vars] if flow_vars else xr.Dataset()

        defaults = {'x': 'time', 'color': 'variable', 'pattern_shape': None, 'facet_col': 'case'}
        _apply_slot_defaults(plotly_kwargs, defaults)
        color_kwargs = _build_color_kwargs(colors, flow_vars)
        fig = flow_ds.plotly.bar(
            title=f'{storage} Operation Comparison',
            **color_kwargs,
            **plotly_kwargs,
        )
        fig.update_layout(barmode='relative', bargap=0, bargroupgap=0)
        fig.update_traces(marker_line_width=0)

        # Add charge state as line overlay on secondary y-axis
        if 'charge_state' in ds:
            # Filter out bar-only kwargs, apply line defaults, override color for comparison
            line_kwargs = {k: v for k, v in plotly_kwargs.items() if k not in ('pattern_shape', 'color')}
            _apply_slot_defaults(line_kwargs, {**_SLOT_DEFAULTS['storage_line'], 'color': 'case'})
            line_fig = ds['charge_state'].plotly.line(**line_kwargs)
            fig = add_secondary_y(fig, line_fig, secondary_y_title='Charge State')

        return self._finalize(ds, fig, show)

    def charge_states(
        self,
        storages: str | list[str] | None = None,
        *,
        select: SelectType | None = None,
        colors: ColorType | None = None,
        threshold: float | None = 1e-5,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot charge states comparison across cases.

        Args:
            storages: Storage label(s) to plot. If None, plots all.
            select: xarray-style selection.
            colors: Color specification (dict, list, or colorscale name).
            threshold: Filter out variables where max absolute value is below this.
            show: Whether to display the figure.
            data_only: If True, skip figure creation and return only data.
            **plotly_kwargs: Additional arguments passed to plotly.

        Returns:
            PlotResult with combined charge state data and figure.
        """
        ds, _ = self._combine_data('charge_states', storages, select=select, threshold=threshold)
        if not ds.data_vars or data_only:
            return self._finalize(ds, None, show if not data_only else False)

        defaults = {'x': 'time', 'color': 'variable', 'symbol': None, 'line_dash': 'case'}
        _apply_slot_defaults(plotly_kwargs, defaults)
        color_kwargs = _build_color_kwargs(colors, list(ds.data_vars))
        fig = ds.plotly.line(
            title='Charge States Comparison',
            **color_kwargs,
            **plotly_kwargs,
        )
        return self._finalize(ds, fig, show)

    def duration_curve(
        self,
        variables: str | list[str],
        *,
        select: SelectType | None = None,
        normalize: bool = False,
        colors: ColorType | None = None,
        threshold: float | None = 1e-5,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot duration curves comparison across cases.

        Args:
            variables: Flow label(s) or variable name(s) to plot.
            select: xarray-style selection.
            normalize: If True, normalize x-axis to 0-100%.
            colors: Color specification (dict, list, or colorscale name).
            threshold: Filter out variables where max absolute value is below this.
            show: Whether to display the figure.
            data_only: If True, skip figure creation and return only data.
            **plotly_kwargs: Additional arguments passed to plotly.

        Returns:
            PlotResult with combined duration curve data and figure.
        """
        ds, _ = self._combine_data('duration_curve', variables, select=select, normalize=normalize, threshold=threshold)
        if not ds.data_vars or data_only:
            return self._finalize(ds, None, show if not data_only else False)

        defaults = {
            'x': 'duration_pct' if normalize else 'duration',
            'color': 'variable',
            'symbol': None,
            'line_dash': 'case',
        }
        _apply_slot_defaults(plotly_kwargs, defaults)
        color_kwargs = _build_color_kwargs(colors, list(ds.data_vars))
        fig = ds.plotly.line(
            title='Duration Curve Comparison',
            **color_kwargs,
            **plotly_kwargs,
        )
        return self._finalize(ds, fig, show)

    def sizes(
        self,
        *,
        max_size: float | None = 1e6,
        select: SelectType | None = None,
        colors: ColorType | None = None,
        threshold: float | None = 1e-5,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot investment sizes comparison across cases.

        Args:
            max_size: Maximum size to include (filters defaults).
            select: xarray-style selection.
            colors: Color specification (dict, list, or colorscale name).
            threshold: Filter out variables where max absolute value is below this.
            show: Whether to display the figure.
            data_only: If True, skip figure creation and return only data.
            **plotly_kwargs: Additional arguments passed to plotly.

        Returns:
            PlotResult with combined sizes data and figure.
        """
        ds, _ = self._combine_data('sizes', max_size=max_size, select=select, threshold=threshold)
        if not ds.data_vars or data_only:
            return self._finalize(ds, None, show if not data_only else False)

        defaults = {'x': 'variable', 'color': 'case'}
        _apply_slot_defaults(plotly_kwargs, defaults)
        color_kwargs = _build_color_kwargs(colors, list(ds.data_vars))
        fig = ds.plotly.bar(
            title='Investment Sizes Comparison',
            labels={'value': 'Size'},
            barmode='group',
            **color_kwargs,
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
        threshold: float | None = 1e-5,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot effects comparison across cases.

        Args:
            aspect: Which aspect to plot - 'total', 'temporal', or 'periodic'.
            effect: Specific effect name to plot. If None, plots all.
            by: Group by 'component', 'contributor', or 'time'.
            select: xarray-style selection.
            colors: Color specification (dict, list, or colorscale name).
            threshold: Filter out variables where max absolute value is below this.
            show: Whether to display the figure.
            data_only: If True, skip figure creation and return only data.
            **plotly_kwargs: Additional arguments passed to plotly.

        Returns:
            PlotResult with combined effects data and figure.
        """
        ds, _ = self._combine_data('effects', aspect, effect=effect, by=by, select=select, threshold=threshold)
        if not ds.data_vars or data_only:
            return self._finalize(ds, None, show if not data_only else False)

        defaults = {'x': by if by else 'variable', 'color': 'case'}
        _apply_slot_defaults(plotly_kwargs, defaults)
        color_kwargs = _build_color_kwargs(colors, list(ds.data_vars))
        fig = ds.plotly.bar(
            title=f'Effects Comparison ({aspect})',
            barmode='group',
            **color_kwargs,
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
        reshape: tuple[str, str] | Literal['auto'] | None = 'auto',
        colors: str | list[str] | None = None,
        threshold: float | None = 1e-5,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot heatmap comparison across cases.

        Args:
            variables: Flow label(s) or variable name(s) to plot.
            select: xarray-style selection.
            reshape: Time reshape frequencies, 'auto', or None.
            colors: Colorscale name or list of colors.
            threshold: Filter out variables where max absolute value is below this.
            show: Whether to display the figure.
            data_only: If True, skip figure creation and return only data.
            **plotly_kwargs: Additional arguments passed to plotly.

        Returns:
            PlotResult with combined heatmap data and figure.
        """
        ds, _ = self._combine_data('heatmap', variables, select=select, reshape=reshape, threshold=threshold)
        if not ds.data_vars or data_only:
            return self._finalize(ds, None, show if not data_only else False)

        da = ds[next(iter(ds.data_vars))]

        defaults = {'facet_col': 'case'}
        _apply_slot_defaults(plotly_kwargs, defaults)
        # Handle colorscale
        if colors is not None and 'color_continuous_scale' not in plotly_kwargs:
            plotly_kwargs['color_continuous_scale'] = colors

        fig = da.plotly.imshow(
            title='Heatmap Comparison',
            **plotly_kwargs,
        )
        return self._finalize(ds, fig, show)
