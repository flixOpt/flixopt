"""
ModelCoordinates encapsulates all time/period/scenario/cluster coordinate metadata for a FlowSystem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from .core import ConversionError, DataConverter

if TYPE_CHECKING:
    from .types import Numeric_S, Numeric_TPS


class ModelCoordinates:
    """Holds all coordinate/weight/duration state and the pure computation methods.

    This class is the single source of truth for time, period, scenario, and cluster
    metadata used by FlowSystem.

    Args:
        timesteps: The timesteps of the model.
        periods: The periods of the model.
        scenarios: The scenarios of the model.
        clusters: Cluster dimension index.
        hours_of_last_timestep: Duration of the last timestep.
        hours_of_previous_timesteps: Duration of previous timesteps.
        weight_of_last_period: Weight/duration of the last period.
        scenario_weights: The weights of each scenario.
        cluster_weight: Weight for each cluster.
        timestep_duration: Explicit timestep duration (for segmented systems).
        fit_to_model_coords: Callable to broadcast data to model dimensions.
    """

    def __init__(
        self,
        timesteps: pd.DatetimeIndex | pd.RangeIndex,
        periods: pd.Index | None = None,
        scenarios: pd.Index | None = None,
        clusters: pd.Index | None = None,
        hours_of_last_timestep: int | float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
        weight_of_last_period: int | float | None = None,
        scenario_weights: Numeric_S | None = None,
        cluster_weight: Numeric_TPS | None = None,
        timestep_duration: xr.DataArray | None = None,
        fit_to_model_coords=None,
    ):
        self.timesteps = self._validate_timesteps(timesteps)
        self.periods = None if periods is None else self._validate_periods(periods)
        self.scenarios = None if scenarios is None else self._validate_scenarios(scenarios)
        self.clusters = clusters
        self._fit_to_model_coords = fit_to_model_coords

        # Compute all time-related metadata
        (
            self.timesteps_extra,
            self.hours_of_last_timestep,
            self.hours_of_previous_timesteps,
            computed_timestep_duration,
        ) = self._compute_time_metadata(self.timesteps, hours_of_last_timestep, hours_of_previous_timesteps)

        # Use provided timestep_duration if given (for segmented systems), otherwise use computed value
        if timestep_duration is not None:
            self.timestep_duration = timestep_duration
        elif computed_timestep_duration is not None:
            self.timestep_duration = self._fit_data('timestep_duration', computed_timestep_duration)
        else:
            if isinstance(self.timesteps, pd.RangeIndex):
                raise ValueError(
                    'timestep_duration is required when using RangeIndex timesteps (segmented systems). '
                    'Provide timestep_duration explicitly or use DatetimeIndex timesteps.'
                )
            self.timestep_duration = None

        # Cluster weight
        self.cluster_weight: xr.DataArray | None = (
            self._fit_data('cluster_weight', cluster_weight) if cluster_weight is not None else None
        )

        # Scenario weights (set via property for normalization)
        self._scenario_weights: xr.DataArray | None = None
        if scenario_weights is not None:
            self.scenario_weights = scenario_weights
        else:
            self._scenario_weights = None

        # Compute all period-related metadata
        (self.periods_extra, self.weight_of_last_period, weight_per_period) = self._compute_period_metadata(
            self.periods, weight_of_last_period
        )
        self.period_weights: xr.DataArray | None = weight_per_period

    def _fit_data(self, name: str, data, dims=None) -> xr.DataArray:
        """Broadcast data to model coordinate dimensions."""
        coords = self.indexes
        if dims is not None:
            coords = {k: coords[k] for k in dims if k in coords}
        return DataConverter.to_dataarray(data, coords=coords).rename(name)

    # --- Validation ---

    @staticmethod
    def _validate_timesteps(
        timesteps: pd.DatetimeIndex | pd.RangeIndex,
    ) -> pd.DatetimeIndex | pd.RangeIndex:
        """Validate timesteps format and rename if needed."""
        if not isinstance(timesteps, (pd.DatetimeIndex, pd.RangeIndex)):
            raise TypeError('timesteps must be a pandas DatetimeIndex or RangeIndex')
        if len(timesteps) < 2:
            raise ValueError('timesteps must contain at least 2 timestamps')
        if timesteps.name != 'time':
            timesteps = timesteps.rename('time')
        if not timesteps.is_monotonic_increasing:
            raise ValueError('timesteps must be sorted')
        return timesteps

    @staticmethod
    def _validate_scenarios(scenarios: pd.Index) -> pd.Index:
        """Validate and prepare scenario index."""
        if not isinstance(scenarios, pd.Index) or len(scenarios) == 0:
            raise ConversionError('Scenarios must be a non-empty Index')
        if scenarios.name != 'scenario':
            scenarios = scenarios.rename('scenario')
        return scenarios

    @staticmethod
    def _validate_periods(periods: pd.Index) -> pd.Index:
        """Validate and prepare period index."""
        if not isinstance(periods, pd.Index) or len(periods) == 0:
            raise ConversionError(f'Periods must be a non-empty Index. Got {periods}')
        if not (periods.dtype.kind == 'i' and periods.is_monotonic_increasing and periods.is_unique):
            raise ConversionError(f'Periods must be a monotonically increasing and unique Index. Got {periods}')
        if periods.name != 'period':
            periods = periods.rename('period')
        return periods

    # --- Timestep computation ---

    @staticmethod
    def _create_timesteps_with_extra(
        timesteps: pd.DatetimeIndex | pd.RangeIndex, hours_of_last_timestep: float | None
    ) -> pd.DatetimeIndex | pd.RangeIndex:
        """Create timesteps with an extra step at the end."""
        if isinstance(timesteps, pd.RangeIndex):
            return pd.RangeIndex(len(timesteps) + 1, name='time')

        if hours_of_last_timestep is None:
            hours_of_last_timestep = (timesteps[-1] - timesteps[-2]) / pd.Timedelta(hours=1)

        last_date = pd.DatetimeIndex([timesteps[-1] + pd.Timedelta(hours=hours_of_last_timestep)], name='time')
        return pd.DatetimeIndex(timesteps.append(last_date), name='time')

    @staticmethod
    def calculate_timestep_duration(
        timesteps_extra: pd.DatetimeIndex | pd.RangeIndex,
    ) -> xr.DataArray | None:
        """Calculate duration of each timestep in hours as a 1D DataArray."""
        if isinstance(timesteps_extra, pd.RangeIndex):
            return None

        hours_per_step = np.diff(timesteps_extra) / pd.Timedelta(hours=1)
        return xr.DataArray(
            hours_per_step, coords={'time': timesteps_extra[:-1]}, dims='time', name='timestep_duration'
        )

    @staticmethod
    def _calculate_hours_of_previous_timesteps(
        timesteps: pd.DatetimeIndex | pd.RangeIndex, hours_of_previous_timesteps: float | np.ndarray | None
    ) -> float | np.ndarray | None:
        """Calculate duration of regular timesteps."""
        if hours_of_previous_timesteps is not None:
            return hours_of_previous_timesteps
        if isinstance(timesteps, pd.RangeIndex):
            return None
        first_interval = timesteps[1] - timesteps[0]
        return first_interval.total_seconds() / 3600

    @classmethod
    def _compute_time_metadata(
        cls,
        timesteps: pd.DatetimeIndex | pd.RangeIndex,
        hours_of_last_timestep: int | float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
    ) -> tuple[
        pd.DatetimeIndex | pd.RangeIndex,
        float | None,
        float | np.ndarray | None,
        xr.DataArray | None,
    ]:
        """Compute all time-related metadata from timesteps."""
        timesteps_extra = cls._create_timesteps_with_extra(timesteps, hours_of_last_timestep)
        timestep_duration = cls.calculate_timestep_duration(timesteps_extra)

        if hours_of_last_timestep is None and timestep_duration is not None:
            hours_of_last_timestep = timestep_duration.isel(time=-1).item()

        hours_of_previous_timesteps = cls._calculate_hours_of_previous_timesteps(timesteps, hours_of_previous_timesteps)

        return timesteps_extra, hours_of_last_timestep, hours_of_previous_timesteps, timestep_duration

    # --- Period computation ---

    @staticmethod
    def _create_periods_with_extra(periods: pd.Index, weight_of_last_period: int | float | None) -> pd.Index:
        """Create periods with an extra period at the end."""
        if weight_of_last_period is None:
            if len(periods) < 2:
                raise ValueError(
                    'FlowSystem: weight_of_last_period must be provided explicitly when only one period is defined.'
                )
            weight_of_last_period = int(periods[-1]) - int(periods[-2])

        last_period_value = int(periods[-1]) + weight_of_last_period
        periods_extra = periods.append(pd.Index([last_period_value], name='period'))
        return periods_extra

    @staticmethod
    def calculate_weight_per_period(periods_extra: pd.Index) -> xr.DataArray:
        """Calculate weight of each period from period index differences."""
        weights = np.diff(periods_extra.to_numpy().astype(int))
        return xr.DataArray(weights, coords={'period': periods_extra[:-1]}, dims='period', name='weight_per_period')

    @classmethod
    def _compute_period_metadata(
        cls, periods: pd.Index | None, weight_of_last_period: int | float | None = None
    ) -> tuple[pd.Index | None, int | float | None, xr.DataArray | None]:
        """Compute all period-related metadata from periods."""
        if periods is None:
            return None, None, None

        periods_extra = cls._create_periods_with_extra(periods, weight_of_last_period)
        weight_per_period = cls.calculate_weight_per_period(periods_extra)

        if weight_of_last_period is None:
            weight_of_last_period = weight_per_period.isel(period=-1).item()

        return periods_extra, weight_of_last_period, weight_per_period

    # --- Dataset update methods (used by TransformAccessor) ---

    @classmethod
    def _update_time_metadata(
        cls,
        dataset: xr.Dataset,
        hours_of_last_timestep: int | float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
    ) -> xr.Dataset:
        """Update time-related attributes and data variables in dataset based on its time index."""
        new_time_index = dataset.indexes.get('time')
        if new_time_index is not None and len(new_time_index) >= 2:
            _, hours_of_last_timestep, hours_of_previous_timesteps, timestep_duration = cls._compute_time_metadata(
                new_time_index, hours_of_last_timestep, hours_of_previous_timesteps
            )

            if 'timestep_duration' in dataset.data_vars:
                dataset['timestep_duration'] = timestep_duration

        if hours_of_last_timestep is not None:
            dataset.attrs['hours_of_last_timestep'] = hours_of_last_timestep
        if hours_of_previous_timesteps is not None:
            dataset.attrs['hours_of_previous_timesteps'] = hours_of_previous_timesteps

        return dataset

    @classmethod
    def _update_period_metadata(
        cls,
        dataset: xr.Dataset,
        weight_of_last_period: int | float | None = None,
    ) -> xr.Dataset:
        """Update period-related attributes and data variables in dataset based on its period index."""
        new_period_index = dataset.indexes.get('period')

        if new_period_index is None:
            if 'period' in dataset.coords:
                dataset = dataset.drop_vars('period')
            dataset = dataset.drop_vars(['period_weights'], errors='ignore')
            dataset.attrs.pop('weight_of_last_period', None)
            return dataset

        if len(new_period_index) >= 1:
            if weight_of_last_period is None:
                weight_of_last_period = dataset.attrs.get('weight_of_last_period')

            _, weight_of_last_period, period_weights = cls._compute_period_metadata(
                new_period_index, weight_of_last_period
            )

            if 'period_weights' in dataset.data_vars:
                dataset['period_weights'] = period_weights

        if weight_of_last_period is not None:
            dataset.attrs['weight_of_last_period'] = weight_of_last_period

        return dataset

    @classmethod
    def _update_scenario_metadata(cls, dataset: xr.Dataset) -> xr.Dataset:
        """Update scenario-related attributes and data variables in dataset based on its scenario index."""
        new_scenario_index = dataset.indexes.get('scenario')

        if new_scenario_index is None:
            if 'scenario' in dataset.coords:
                dataset = dataset.drop_vars('scenario')
            dataset = dataset.drop_vars(['scenario_weights'], errors='ignore')
            dataset.attrs.pop('scenario_weights', None)
            return dataset

        if len(new_scenario_index) <= 1:
            dataset.attrs.pop('scenario_weights', None)

        return dataset

    # --- Properties ---

    @property
    def scenario_weights(self) -> xr.DataArray | None:
        """Weights for each scenario."""
        return self._scenario_weights

    @scenario_weights.setter
    def scenario_weights(self, value: Numeric_S | None) -> None:
        """Set scenario weights (always normalized to sum to 1)."""
        if value is None:
            self._scenario_weights = None
            return

        if self.scenarios is None:
            raise ValueError(
                'scenario_weights cannot be set when no scenarios are defined. '
                'Either define scenarios or set scenario_weights to None.'
            )

        weights = self._fit_data('scenario_weights', value, dims=['scenario'])

        # Normalize to sum to 1
        norm = weights.sum('scenario')
        if np.isclose(norm, 0.0).any().item():
            if norm.ndim > 0:
                zero_locations = np.argwhere(np.isclose(norm.values, 0.0))
                coords_info = ', '.join(
                    f'{dim}={norm.coords[dim].values[idx]}'
                    for idx, dim in zip(zero_locations[0], norm.dims, strict=False)
                )
                raise ValueError(
                    f'scenario_weights sum to 0 at {coords_info}; cannot normalize. '
                    f'Ensure all scenario weight combinations sum to a positive value.'
                )
            raise ValueError('scenario_weights sum to 0; cannot normalize.')
        self._scenario_weights = weights / norm

    @property
    def dims(self) -> list[str]:
        """Active dimension names."""
        result = []
        if self.clusters is not None:
            result.append('cluster')
        result.append('time')
        if self.periods is not None:
            result.append('period')
        if self.scenarios is not None:
            result.append('scenario')
        return result

    @property
    def indexes(self) -> dict[str, pd.Index]:
        """Indexes for active dimensions."""
        result: dict[str, pd.Index] = {}
        if self.clusters is not None:
            result['cluster'] = self.clusters
        result['time'] = self.timesteps
        if self.periods is not None:
            result['period'] = self.periods
        if self.scenarios is not None:
            result['scenario'] = self.scenarios
        return result

    @property
    def temporal_dims(self) -> list[str]:
        """Temporal dimensions for summing over time."""
        if self.clusters is not None:
            return ['time', 'cluster']
        return ['time']

    @property
    def temporal_weight(self) -> xr.DataArray:
        """Combined temporal weight (timestep_duration x cluster_weight)."""
        cluster_weight = self.weights.get('cluster', self.cluster_weight if self.cluster_weight is not None else 1.0)
        return self.weights['time'] * cluster_weight

    @property
    def is_segmented(self) -> bool:
        """Check if this uses segmented time (RangeIndex)."""
        return isinstance(self.timesteps, pd.RangeIndex)

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps."""
        return len(self.timesteps)

    def _unit_weight(self, dim: str) -> xr.DataArray:
        """Create a unit weight DataArray (all 1.0) for a dimension."""
        index = self.indexes[dim]
        return xr.DataArray(
            np.ones(len(index), dtype=float),
            coords={dim: index},
            dims=[dim],
            name=f'{dim}_weight',
        )

    @property
    def weights(self) -> dict[str, xr.DataArray]:
        """Weights for active dimensions (unit weights if not explicitly set)."""
        result: dict[str, xr.DataArray] = {'time': self.timestep_duration}
        if self.clusters is not None:
            result['cluster'] = self.cluster_weight if self.cluster_weight is not None else self._unit_weight('cluster')
        if self.periods is not None:
            result['period'] = self.period_weights if self.period_weights is not None else self._unit_weight('period')
        if self.scenarios is not None:
            result['scenario'] = (
                self.scenario_weights if self.scenario_weights is not None else self._unit_weight('scenario')
            )
        return result

    def sum_temporal(self, data: xr.DataArray) -> xr.DataArray:
        """Sum data over temporal dimensions with full temporal weighting."""
        return (data * self.temporal_weight).sum(self.temporal_dims)
