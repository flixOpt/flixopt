"""
Iteration utilities for clustering operations.

This module provides standardized iteration patterns for (period, scenario)
combinations used throughout the clustering module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..flow_system import FlowSystem


@dataclass
class DimSliceContext:
    """Context for a single (period, scenario) slice during clustering/expansion.

    Provides all the information needed to select and process a single slice
    of multi-dimensional data during clustering or expansion operations.

    Attributes:
        key: Tuple of (period, scenario) values. None values indicate
            that dimension doesn't exist. E.g., (2024, 'high') or (2024, None).
        selector: Dict for use with xarray .sel() calls. Only contains
            non-None dimensions. E.g., {'period': 2024, 'scenario': 'high'}.
        period: The period value, or None if no periods dimension.
        scenario: The scenario value, or None if no scenarios dimension.

    Example:
        >>> ctx = DimSliceContext(
        ...     key=(2024, 'high'),
        ...     selector={'period': 2024, 'scenario': 'high'},
        ...     period=2024,
        ...     scenario='high',
        ... )
        >>> ds.sel(**ctx.selector)  # Select data for this slice
    """

    key: tuple[Any, Any]
    selector: dict[str, Any]
    period: Any | None
    scenario: Any | None


@dataclass
class DimInfo:
    """Dimension metadata for combining slice results.

    Holds information about the (period, scenario) dimensions present in
    the data, used for combining per-slice results back into multi-dimensional
    DataArrays.

    Attributes:
        periods: List of period values, or [None] if no periods dimension.
        scenarios: List of scenario values, or [None] if no scenarios dimension.
        has_periods: True if the data has a periods dimension.
        has_scenarios: True if the data has a scenarios dimension.
        extra_dims: List of extra dimension names (e.g., ['period', 'scenario']).
        dim_coords: Dict mapping dimension names to their coordinate values.
        dim_names: List of dimension names for ClusteringResults format
            (excludes None dimensions).

    Example:
        >>> dim_info = DimInfo.from_flow_system(fs)
        >>> dim_info.periods
        [2024, 2025]
        >>> dim_info.has_periods
        True
        >>> dim_info.extra_dims
        ['period', 'scenario']
    """

    periods: list[Any]
    scenarios: list[Any]
    has_periods: bool = field(init=False)
    has_scenarios: bool = field(init=False)
    extra_dims: list[str] = field(init=False)
    dim_coords: dict[str, list[Any]] = field(init=False)
    dim_names: list[str] = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived fields after initialization."""
        self.has_periods = self.periods != [None]
        self.has_scenarios = self.scenarios != [None]

        self.extra_dims = []
        self.dim_coords = {}
        self.dim_names = []

        if self.has_periods:
            self.extra_dims.append('period')
            self.dim_coords['period'] = self.periods
            self.dim_names.append('period')
        if self.has_scenarios:
            self.extra_dims.append('scenario')
            self.dim_coords['scenario'] = self.scenarios
            self.dim_names.append('scenario')

    @classmethod
    def from_flow_system(cls, flow_system: FlowSystem) -> DimInfo:
        """Create DimInfo from a FlowSystem.

        Args:
            flow_system: The FlowSystem to extract dimension info from.

        Returns:
            DimInfo populated with the FlowSystem's period/scenario dimensions.
        """
        periods = list(flow_system.periods) if flow_system.periods is not None else [None]
        scenarios = list(flow_system.scenarios) if flow_system.scenarios is not None else [None]
        return cls(periods=periods, scenarios=scenarios)

    def to_clustering_key(self, period: Any | None, scenario: Any | None) -> tuple:
        """Convert (period, scenario) to ClusteringResults key format.

        The key format used by ClusteringResults excludes None values,
        so (2024, None) becomes (2024,) and (None, 'high') becomes ('high',).

        Args:
            period: Period value or None.
            scenario: Scenario value or None.

        Returns:
            Tuple key for ClusteringResults.
        """
        key_parts = []
        if self.has_periods:
            key_parts.append(period)
        if self.has_scenarios:
            key_parts.append(scenario)
        return tuple(key_parts)

    def from_clustering_key(self, cr_key: tuple) -> tuple[Any | None, Any | None]:
        """Convert ClusteringResults key back to (period, scenario) format.

        Args:
            cr_key: Key from ClusteringResults (excludes None dimensions).

        Returns:
            Tuple of (period, scenario) with None for missing dimensions.
        """
        if self.has_periods and self.has_scenarios:
            return (cr_key[0], cr_key[1])
        elif self.has_periods:
            return (cr_key[0], None)
        elif self.has_scenarios:
            return (None, cr_key[0])
        else:
            return (None, None)


def iter_dim_slices(dim_info: DimInfo) -> Iterator[DimSliceContext]:
    """Iterate over all (period, scenario) combinations.

    This generator provides a standardized way to iterate over period/scenario
    combinations, yielding context objects with all necessary information for
    selecting and processing each slice.

    Args:
        dim_info: DimInfo containing the periods and scenarios to iterate.

    Yields:
        DimSliceContext for each (period, scenario) combination.

    Example:
        >>> dim_info = DimInfo.from_flow_system(fs)
        >>> for ctx in iter_dim_slices(dim_info):
        ...     data_slice = ds.sel(**ctx.selector, drop=True) if ctx.selector else ds
        ...     results[ctx.key] = process_slice(data_slice)
    """
    for period in dim_info.periods:
        for scenario in dim_info.scenarios:
            key = (period, scenario)
            selector = {}
            if period is not None:
                selector['period'] = period
            if scenario is not None:
                selector['scenario'] = scenario
            yield DimSliceContext(
                key=key,
                selector=selector,
                period=period,
                scenario=scenario,
            )


def iter_dim_slices_simple(
    periods: list[Any] | None,
    scenarios: list[Any] | None,
) -> Iterator[DimSliceContext]:
    """Iterate over (period, scenario) combinations from raw lists.

    Convenience function that doesn't require a DimInfo object.

    Args:
        periods: List of period values, or None for no periods.
        scenarios: List of scenario values, or None for no scenarios.

    Yields:
        DimSliceContext for each (period, scenario) combination.
    """
    dim_info = DimInfo(
        periods=list(periods) if periods is not None else [None],
        scenarios=list(scenarios) if scenarios is not None else [None],
    )
    yield from iter_dim_slices(dim_info)
