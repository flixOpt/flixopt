"""
This module provides backwards-compatible aliases for the renamed Optimization classes.

DEPRECATED: This module is deprecated. Use the optimization module instead.
The following classes have been renamed:
    - Calculation -> Optimization
    - FullCalculation -> FullOptimization
    - AggregatedCalculation -> AggregatedOptimization
    - SegmentedCalculation -> SegmentedOptimization

Import from flixopt.optimization or use the new names from flixopt directly.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from .optimization import (
    AggregatedOptimization as _AggregatedOptimization,
)
from .optimization import (
    FullOptimization as _FullOptimization,
)
from .optimization import (
    Optimization as _Optimization,
)
from .optimization import (
    SegmentedOptimization as _SegmentedOptimization,
)

if TYPE_CHECKING:
    import pathlib
    from typing import Annotated, Any

    import pandas as pd

    from .aggregation import AggregationParameters
    from .elements import Component
    from .flow_system import FlowSystem
    from .solvers import _Solver


def _deprecation_warning(old_name: str, new_name: str):
    """Issue a deprecation warning for renamed classes."""
    warnings.warn(
        f'{old_name} is deprecated and will be removed in a future version. Use {new_name} instead.',
        DeprecationWarning,
        stacklevel=3,
    )


class Calculation(_Optimization):
    """
    DEPRECATED: Use Optimization instead.

    class for defined way of solving a flow_system optimization

    Args:
        name: name of calculation
        flow_system: flow_system which should be calculated
        folder: folder where results should be saved. If None, then the current working directory is used.
        normalize_weights: Whether to automatically normalize the weights of scenarios to sum up to 1 when solving.
        active_timesteps: Deprecated. Use FlowSystem.sel(time=...) or FlowSystem.isel(time=...) instead.
    """

    def __init__(
        self,
        name: str,
        flow_system: FlowSystem,
        active_timesteps: Annotated[
            pd.DatetimeIndex | None,
            'DEPRECATED: Use flow_system.sel(time=...) or flow_system.isel(time=...) instead',
        ] = None,
        folder: pathlib.Path | None = None,
        normalize_weights: bool = True,
    ):
        _deprecation_warning('Calculation', 'Optimization')
        super().__init__(name, flow_system, active_timesteps, folder, normalize_weights)


class FullCalculation(_FullOptimization):
    """
    DEPRECATED: Use FullOptimization instead.

    FullCalculation solves the complete optimization problem using all time steps.

    This is the most comprehensive calculation type that considers every time step
    in the optimization, providing the most accurate but computationally intensive solution.

    Args:
        name: name of calculation
        flow_system: flow_system which should be calculated
        folder: folder where results should be saved. If None, then the current working directory is used.
        normalize_weights: Whether to automatically normalize the weights of scenarios to sum up to 1 when solving.
        active_timesteps: Deprecated. Use FlowSystem.sel(time=...) or FlowSystem.isel(time=...) instead.
    """

    def __init__(
        self,
        name: str,
        flow_system: FlowSystem,
        active_timesteps: Annotated[
            pd.DatetimeIndex | None,
            'DEPRECATED: Use flow_system.sel(time=...) or flow_system.isel(time=...) instead',
        ] = None,
        folder: pathlib.Path | None = None,
        normalize_weights: bool = True,
    ):
        _deprecation_warning('FullCalculation', 'FullOptimization')
        super().__init__(name, flow_system, active_timesteps, folder, normalize_weights)


class AggregatedCalculation(_AggregatedOptimization):
    """
    DEPRECATED: Use AggregatedOptimization instead.

    AggregatedCalculation reduces computational complexity by clustering time series into typical periods.

    This calculation approach aggregates time series data using clustering techniques (tsam) to identify
    representative time periods, significantly reducing computation time while maintaining solution accuracy.

    Args:
        name: Name of the calculation
        flow_system: FlowSystem to be optimized
        aggregation_parameters: Parameters for aggregation. See AggregationParameters class documentation
        components_to_clusterize: list of Components to perform aggregation on. If None, all components are aggregated.
            This equalizes variables in the components according to the typical periods computed in the aggregation
        active_timesteps: DatetimeIndex of timesteps to use for calculation. If None, all timesteps are used
        folder: Folder where results should be saved. If None, current working directory is used
    """

    def __init__(
        self,
        name: str,
        flow_system: FlowSystem,
        aggregation_parameters: AggregationParameters,
        components_to_clusterize: list[Component] | None = None,
        active_timesteps: Annotated[
            pd.DatetimeIndex | None,
            'DEPRECATED: Use flow_system.sel(time=...) or flow_system.isel(time=...) instead',
        ] = None,
        folder: pathlib.Path | None = None,
    ):
        _deprecation_warning('AggregatedCalculation', 'AggregatedOptimization')
        super().__init__(name, flow_system, aggregation_parameters, components_to_clusterize, active_timesteps, folder)


class SegmentedCalculation(_SegmentedOptimization):
    """
    DEPRECATED: Use SegmentedOptimization instead.

    Solve large optimization problems by dividing time horizon into (overlapping) segments.

    Args:
        name: Unique identifier for the calculation, used in result files and logging.
        flow_system: The FlowSystem to optimize, containing all components, flows, and buses.
        timesteps_per_segment: Number of timesteps in each segment (excluding overlap).
        overlap_timesteps: Number of additional timesteps added to each segment.
        nr_of_previous_values: Number of previous timestep values to transfer between segments for initialization.
        folder: Directory for saving results. Defaults to current working directory + 'results'.
    """

    def __init__(
        self,
        name: str,
        flow_system: FlowSystem,
        timesteps_per_segment: int,
        overlap_timesteps: int,
        nr_of_previous_values: int = 1,
        folder: pathlib.Path | None = None,
    ):
        _deprecation_warning('SegmentedCalculation', 'SegmentedOptimization')
        super().__init__(name, flow_system, timesteps_per_segment, overlap_timesteps, nr_of_previous_values, folder)


__all__ = ['Calculation', 'FullCalculation', 'AggregatedCalculation', 'SegmentedCalculation']
