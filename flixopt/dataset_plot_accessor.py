"""Xarray accessor for statistics (``.fxstats``).

Note:
    Generic plotting accessors (``.plotly``) are provided by the ``xarray_plotly`` package.
    This module only contains the statistics/transformation accessor.
"""

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

        # Sort each variable along time dimension (descending)
        sorted_ds = self._ds.copy()
        for var in sorted_ds.data_vars:
            da = sorted_ds[var]
            if 'time' not in da.dims:
                # Skip variables without time dimension (e.g., scalar metadata)
                continue
            time_axis = da.dims.index('time')
            # Sort along time axis (descending) - use flip for correct axis
            sorted_values = np.flip(np.sort(da.values, axis=time_axis), axis=time_axis)
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
