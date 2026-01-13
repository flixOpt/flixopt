"""Xarray accessor for statistics and transformations (``.fxstats``)."""

from __future__ import annotations

import numpy as np
import xarray as xr


@xr.register_dataset_accessor('fxstats')
class DatasetStatsAccessor:
    """Statistics/transformation accessor for any xr.Dataset. Access via ``dataset.fxstats``.

    Provides data transformation methods that return new datasets.
    Chain with ``.plotly`` for visualization.

    Examples:
        Duration curve::

            ds.fxstats.to_duration_curve().plotly.line()
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
            >>> ds.fxstats.to_duration_curve().plotly.line(title='Duration Curve')
        """
        if 'time' not in self._ds.dims:
            raise ValueError("Duration curve requires a 'time' dimension.")

        # Sort each variable along time dimension (descending), preserving attributes
        sorted_vars = {}
        for var in self._ds.data_vars:
            da = self._ds[var]
            if 'time' not in da.dims:
                # Keep variables without time dimension unchanged
                sorted_vars[var] = da
                continue
            time_axis = da.dims.index('time')
            sorted_values = np.flip(np.sort(da.values, axis=time_axis), axis=time_axis)
            sorted_vars[var] = xr.DataArray(
                sorted_values,
                dims=da.dims,
                coords={k: v for k, v in da.coords.items() if k != 'time'},
                attrs=da.attrs,
            )

        # Preserve non-time coordinates from the original dataset
        non_time_coords = {k: v for k, v in self._ds.coords.items() if k != 'time'}
        sorted_ds = xr.Dataset(sorted_vars, coords=non_time_coords, attrs=self._ds.attrs)

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
