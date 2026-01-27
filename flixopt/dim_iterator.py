"""
Dimension iteration utilities for period/scenario slicing patterns.

This module provides `DimIterator`, a helper class that eliminates repetitive
boilerplate when working with datasets that may have period and/or scenario
dimensions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .flow_system import FlowSystem


class DimIterator:
    """Handles period/scenario iteration and combination patterns.

    Eliminates the need for:
    - ``[None]`` sentinel values for missing dimensions
    - 4-way if/elif branching for period/scenario combinations
    - Manual key conversion between internal and external formats

    Keys returned by this class are tuples that match ``dims`` order:
    - No dims: ``()``
    - Period only: ``(period_value,)``
    - Scenario only: ``(scenario_value,)``
    - Both: ``(period_value, scenario_value)``

    Example:
        >>> iterator = DimIterator.from_dataset(ds)
        >>>
        >>> # Iteration
        >>> results = {}
        >>> for key, ds_slice in iterator.iter_slices(ds):
        ...     results[key] = transform(ds_slice)
        >>>
        >>> # Combination
        >>> combined = iterator.combine(results, base_dims=['cluster', 'time'])

    Attributes:
        dims: Tuple of dimension names present (e.g., ``('period', 'scenario')``).
        coords: Dict mapping dimension names to coordinate values.
    """

    def __init__(
        self,
        periods: list | None = None,
        scenarios: list | None = None,
    ):
        """Initialize with explicit dimension values.

        Args:
            periods: Period coordinate values, or None if no period dimension.
            scenarios: Scenario coordinate values, or None if no scenario dimension.
        """
        self._periods = periods
        self._scenarios = scenarios

        # Build dim_names for external keys (no None sentinels)
        self._dim_names: list[str] = []
        if periods is not None:
            self._dim_names.append('period')
        if scenarios is not None:
            self._dim_names.append('scenario')

    # ==========================================================================
    # Factory methods
    # ==========================================================================

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> DimIterator:
        """Create iterator from dataset's period/scenario coordinates.

        Args:
            ds: Dataset with optional 'period' and/or 'scenario' coordinates.

        Returns:
            DimIterator configured for the dataset's dimensions.
        """
        periods = list(ds.period.values) if 'period' in ds.coords else None
        scenarios = list(ds.scenario.values) if 'scenario' in ds.coords else None
        return cls(periods=periods, scenarios=scenarios)

    @classmethod
    def from_flow_system(cls, fs: FlowSystem) -> DimIterator:
        """Create iterator from FlowSystem's period/scenario structure.

        Args:
            fs: FlowSystem with optional periods and/or scenarios.

        Returns:
            DimIterator configured for the FlowSystem's dimensions.
        """
        return cls(
            periods=list(fs.periods) if fs.periods is not None else None,
            scenarios=list(fs.scenarios) if fs.scenarios is not None else None,
        )

    @classmethod
    def from_sentinel_lists(cls, periods: list, scenarios: list) -> DimIterator:
        """Create iterator from lists that may contain [None] sentinel.

        This handles the common pattern where code uses [None] to indicate
        no dimension exists:

            periods = list(fs.periods) if has_periods else [None]
            scenarios = list(fs.scenarios) if has_scenarios else [None]

        Args:
            periods: Period values, or [None] if no period dimension.
            scenarios: Scenario values, or [None] if no scenario dimension.

        Returns:
            DimIterator configured for the actual dimensions present.
        """
        return cls(
            periods=periods if periods != [None] else None,
            scenarios=scenarios if scenarios != [None] else None,
        )

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimension names as tuple (xarray-like)."""
        return tuple(self._dim_names)

    @property
    def coords(self) -> dict[str, list]:
        """Coordinate values for each dimension.

        Returns:
            Dict mapping dimension names to lists of coordinate values.
            Empty dict if no dimensions.
        """
        result = {}
        if self._periods is not None:
            result['period'] = self._periods
        if self._scenarios is not None:
            result['scenario'] = self._scenarios
        return result

    @property
    def has_periods(self) -> bool:
        """Whether period dimension exists."""
        return self._periods is not None

    @property
    def has_scenarios(self) -> bool:
        """Whether scenario dimension exists."""
        return self._scenarios is not None

    @property
    def n_slices(self) -> int:
        """Total number of slices (product of dimension sizes)."""
        n = 1
        if self._periods is not None:
            n *= len(self._periods)
        if self._scenarios is not None:
            n *= len(self._scenarios)
        return n

    # ==========================================================================
    # Iteration
    # ==========================================================================

    def iter_slices(
        self,
        ds: xr.Dataset,
        drop: bool = True,
    ) -> Iterator[tuple[tuple, xr.Dataset]]:
        """Iterate over (period, scenario) slices of a dataset.

        Args:
            ds: Dataset with optional period/scenario dimensions.
            drop: If True (default), drop period/scenario dims from sliced datasets.
                If False, keep them as scalar coordinates.

        Yields:
            ``(key, ds_slice)`` tuples where key matches ``dims`` order.
            For no dims, key is ``()``.

        Example:
            >>> for key, slice_ds in iterator.iter_slices(ds):
            ...     print(f'Processing {key}')
            ...     result = process(slice_ds)
        """
        periods = self._periods if self._periods is not None else [None]
        scenarios = self._scenarios if self._scenarios is not None else [None]

        for p in periods:
            for s in scenarios:
                selector = self._build_selector(p, s)
                key = self._to_key(p, s)
                if selector:
                    yield key, ds.sel(**selector, drop=drop)
                else:
                    yield key, ds

    def iter_keys(self) -> Iterator[tuple]:
        """Iterate over keys only (no data access).

        Yields:
            Key tuples matching ``dims`` order.

        Example:
            >>> for key in iterator.iter_keys():
            ...     results[key] = compute_something()
        """
        periods = self._periods if self._periods is not None else [None]
        scenarios = self._scenarios if self._scenarios is not None else [None]

        for p in periods:
            for s in scenarios:
                yield self._to_key(p, s)

    # ==========================================================================
    # Key conversion
    # ==========================================================================

    def _build_selector(self, period: Any, scenario: Any) -> dict:
        """Build xarray selector dict (only for non-None values)."""
        selector = {}
        if period is not None:
            selector['period'] = period
        if scenario is not None:
            selector['scenario'] = scenario
        return selector

    def _to_key(self, period: Any, scenario: Any) -> tuple:
        """Convert (period, scenario) to external key matching dims."""
        parts = []
        if self._periods is not None:
            parts.append(period)
        if self._scenarios is not None:
            parts.append(scenario)
        return tuple(parts)

    def _from_key(self, key: tuple) -> tuple[Any, Any]:
        """Convert external key back to (period, scenario) with Nones.

        Returns:
            Tuple of (period, scenario) where missing dims are None.
        """
        period = None
        scenario = None
        idx = 0
        if self._periods is not None:
            period = key[idx]
            idx += 1
        if self._scenarios is not None:
            scenario = key[idx]
        return period, scenario

    def selector_for_key(self, key: tuple) -> dict:
        """Get xarray selector dict for an external key.

        Args:
            key: Key tuple as returned by ``iter_slices`` or ``iter_keys``.

        Returns:
            Dict suitable for ``ds.sel(**selector)``.

        Example:
            >>> selector = iterator.selector_for_key(key)
            >>> ds_slice = ds.sel(**selector)
        """
        period, scenario = self._from_key(key)
        return self._build_selector(period, scenario)

    def key_from_values(self, period: Any, scenario: Any) -> tuple:
        """Convert (period, scenario) values to a key tuple.

        This is useful when you have separate period/scenario values
        (possibly None) and need to create a key for use with combine().

        Args:
            period: Period value, or None if no period dimension.
            scenario: Scenario value, or None if no scenario dimension.

        Returns:
            Key tuple matching ``dims`` order.

        Example:
            >>> iterator = DimIterator(periods=[2024, 2025])
            >>> iterator.key_from_values(2024, None)
            (2024,)
        """
        return self._to_key(period, scenario)

    # ==========================================================================
    # Combination
    # ==========================================================================

    def combine(
        self,
        slices: dict[tuple, xr.DataArray],
        base_dims: list[str],
        name: str | None = None,
        attrs: dict | None = None,
        join: str = 'exact',
        fill_value: Any = None,
    ) -> xr.DataArray:
        """Combine per-slice DataArrays into multi-dimensional DataArray.

        Args:
            slices: Dict mapping keys (from ``iter_slices``) to DataArrays.
                All DataArrays must have the same ``base_dims``.
            base_dims: Base dimensions of each slice (e.g., ``['cluster', 'time']``).
                Used for final transpose to ensure consistent dimension order.
            name: Optional name for resulting DataArray.
            attrs: Optional attributes for resulting DataArray.
            join: How to handle coordinate differences between slices.
                - ``'exact'``: Coordinates must match exactly (default).
                - ``'outer'``: Union of coordinates, fill missing with fill_value.
            fill_value: Value to use for missing entries when ``join='outer'``.
                Only used when coordinates differ between slices.

        Returns:
            DataArray with dims ``[*base_dims, period?, scenario?]``.

        Example:
            >>> results = {key: process(s) for key, s in iterator.iter_slices(ds)}
            >>> combined = iterator.combine(results, base_dims=['cluster', 'time'])
        """
        # Build concat kwargs
        concat_kwargs = {'join': join}
        if fill_value is not None:
            concat_kwargs['fill_value'] = fill_value

        if not self._dim_names:
            # No extra dims - return single slice
            result = slices[()]
        elif self.has_periods and self.has_scenarios:
            # Both dimensions: concat scenarios first, then periods
            period_arrays = []
            for p in self._periods:
                scenario_arrays = [slices[self._to_key(p, s)] for s in self._scenarios]
                period_arrays.append(
                    xr.concat(
                        scenario_arrays,
                        dim=pd.Index(self._scenarios, name='scenario'),
                        **concat_kwargs,
                    )
                )
            result = xr.concat(
                period_arrays,
                dim=pd.Index(self._periods, name='period'),
                **concat_kwargs,
            )
        elif self.has_periods:
            result = xr.concat(
                [slices[(p,)] for p in self._periods],
                dim=pd.Index(self._periods, name='period'),
                **concat_kwargs,
            )
        else:  # has_scenarios only
            result = xr.concat(
                [slices[(s,)] for s in self._scenarios],
                dim=pd.Index(self._scenarios, name='scenario'),
                **concat_kwargs,
            )

        # Transpose to standard order: base_dims first, then period/scenario
        result = result.transpose(*base_dims, ...)

        if name:
            result = result.rename(name)
        if attrs:
            result = result.assign_attrs(attrs)
        return result

    def combine_arrays(
        self,
        slices: dict[tuple, np.ndarray],
        base_dims: list[str],
        base_coords: dict[str, Any] | None = None,
        name: str | None = None,
        attrs: dict | None = None,
    ) -> xr.DataArray:
        """Combine per-slice numpy arrays into multi-dimensional DataArray.

        More efficient than ``combine()`` when working with raw numpy arrays,
        as it avoids intermediate DataArray creation.

        Args:
            slices: Dict mapping keys to numpy arrays. All arrays must have
                the same shape and dtype.
            base_dims: Dimension names for the base array axes.
            base_coords: Optional coordinates for base dimensions.
            name: Optional name for resulting DataArray.
            attrs: Optional attributes for resulting DataArray.

        Returns:
            DataArray with dims ``[*base_dims, period?, scenario?]``.

        Example:
            >>> arrays = {key: np.array([1, 2, 3]) for key in iterator.iter_keys()}
            >>> da = iterator.combine_arrays(arrays, base_dims=['cluster'])
        """
        base_coords = base_coords or {}

        if not self._dim_names:
            # No extra dims - wrap single array
            return xr.DataArray(
                slices[()],
                dims=base_dims,
                coords=base_coords,
                name=name,
                attrs=attrs or {},
            )

        # Build full shape: base_shape + extra_dims_shape
        first = next(iter(slices.values()))
        extra_shape = [len(coords) for coords in self.coords.values()]
        shape = list(first.shape) + extra_shape
        data = np.empty(shape, dtype=first.dtype)

        # Fill using np.ndindex for flexibility with any number of extra dims
        for combo in np.ndindex(*extra_shape):
            key = tuple(list(self.coords.values())[i][idx] for i, idx in enumerate(combo))
            data[(...,) + combo] = slices[key]

        return xr.DataArray(
            data,
            dims=base_dims + self._dim_names,
            coords={**base_coords, **self.coords},
            name=name,
            attrs=attrs or {},
        )

    # ==========================================================================
    # Dunder methods
    # ==========================================================================

    def __repr__(self) -> str:
        if not self.dims:
            return 'DimIterator(dims=())'
        coords_str = ', '.join(f'{k}: {len(v)}' for k, v in self.coords.items())
        return f'DimIterator(dims={self.dims}, coords=({coords_str}))'

    def __len__(self) -> int:
        """Number of slices."""
        return self.n_slices
