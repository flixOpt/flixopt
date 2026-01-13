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

        def _sort_descending(data: np.ndarray, axis: int) -> np.ndarray:
            """Sort array in descending order along axis."""
            return np.flip(np.sort(data, axis=axis), axis=axis)

        # Sort each variable along time dimension (descending) using apply_ufunc
        # This preserves dask laziness and DataArray attributes
        sorted_vars = {}
        for var in self._ds.data_vars:
            da = self._ds[var]
            if 'time' not in da.dims:
                # Keep variables without time dimension unchanged
                sorted_vars[var] = da
                continue
            sorted_da = xr.apply_ufunc(
                _sort_descending,
                da,
                input_core_dims=[['time']],
                output_core_dims=[['time']],
                kwargs={'axis': -1},  # Core dim is moved to last position
                dask='parallelized',
                output_dtypes=[da.dtype],
            )
            sorted_vars[var] = sorted_da

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
