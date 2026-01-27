"""
Helper for period/scenario dimension handling.

Provides a simple utility for adding period/scenario dimensions back to
DataArrays after transformation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import xarray as xr


def add_slice_dims(
    da: xr.DataArray,
    period: Any = None,
    scenario: Any = None,
) -> xr.DataArray:
    """Add period/scenario dimensions back to a transformed DataArray.

    After selecting a slice with `ds.sel(period=p, scenario=s, drop=True)`
    and transforming it, use this to re-add the period/scenario coordinates
    so results can be combined with `xr.combine_by_coords`.

    Args:
        da: DataArray without period/scenario dimensions.
        period: Period value to add, or None to skip.
        scenario: Scenario value to add, or None to skip.

    Returns:
        DataArray with period/scenario dims added (as length-1 dims).

    Example:
        >>> results = []
        >>> for p in periods:
        ...     for s in scenarios:
        ...         selector = {k: v for k, v in [('period', p), ('scenario', s)] if v is not None}
        ...         ds_slice = ds.sel(**selector, drop=True) if selector else ds
        ...         result = transform(ds_slice)
        ...         results.append(add_slice_dims(result, period=p, scenario=s))
        >>> combined = xr.combine_by_coords(results)
    """
    if period is not None:
        da = da.expand_dims(period=[period])
    if scenario is not None:
        da = da.expand_dims(scenario=[scenario])
    return da
