"""
Expansion helpers for expanding clustered data back to full resolution.

This module provides utilities for expanding clustered FlowSystem data and
solutions back to their original temporal resolution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from ..structure import EXPAND_DIVIDE, EXPAND_FIRST_TIMESTEP, EXPAND_INTERPOLATE, VariableCategory

if TYPE_CHECKING:
    import pandas as pd

    from .base import Clustering


class VariableExpansionHandler:
    """Handles variable-specific expansion logic for clustered solutions.

    This class encapsulates the logic for determining how each variable should
    be expanded from clustered to full resolution, based on variable categories
    and fallback pattern matching for backwards compatibility.

    Attributes:
        variable_categories: Dict mapping variable names to their VariableCategory.
        clustering: The Clustering object with cluster structure info.
        is_segmented: Whether the clustering used segmentation.
        segment_total_vars: Set of variable names that are segment totals.
        expansion_divisor: DataArray for dividing segment totals (or None).
    """

    def __init__(
        self,
        variable_categories: dict[str, VariableCategory],
        clustering: Clustering,
        flow_system: Any,
    ) -> None:
        """Initialize the expansion handler.

        Args:
            variable_categories: Dict mapping variable names to VariableCategory.
            clustering: The Clustering object with cluster structure info.
            flow_system: The FlowSystem being expanded (for fallback pattern building).
        """
        self.variable_categories = variable_categories
        self.clustering = clustering
        self.is_segmented = clustering.is_segmented

        # Build segment total vars using registry first, fall back to pattern matching
        self.segment_total_vars: set[str] = set()
        self.expansion_divisor: xr.DataArray | None = None

        if self.is_segmented:
            original_timesteps = clustering.original_timesteps
            self.expansion_divisor = clustering.build_expansion_divisor(original_time=original_timesteps)

            # Build from registry
            self.segment_total_vars = {name for name, cat in variable_categories.items() if cat in EXPAND_DIVIDE}
            # Fall back to pattern matching for backwards compatibility
            if not self.segment_total_vars:
                self.segment_total_vars = build_segment_total_varnames(flow_system)

    def is_state_variable(self, var_name: str) -> bool:
        """Check if a variable is a state variable (should be interpolated).

        Args:
            var_name: Name of the variable.

        Returns:
            True if the variable should be interpolated during expansion.
        """
        if var_name in self.variable_categories:
            return self.variable_categories[var_name] in EXPAND_INTERPOLATE
        # Fall back to pattern matching for backwards compatibility
        return var_name.endswith('|charge_state')

    def is_first_timestep_variable(self, var_name: str) -> bool:
        """Check if a variable is a binary event (first timestep only).

        Args:
            var_name: Name of the variable.

        Returns:
            True if the variable should only appear at first timestep of segments.
        """
        if var_name in self.variable_categories:
            return self.variable_categories[var_name] in EXPAND_FIRST_TIMESTEP
        # Fall back to pattern matching for backwards compatibility
        return var_name.endswith('|startup') or var_name.endswith('|shutdown')

    def is_segment_total_variable(self, var_name: str) -> bool:
        """Check if a variable is a segment total (should be divided).

        Args:
            var_name: Name of the variable.

        Returns:
            True if the variable should be divided by expansion divisor.
        """
        return var_name in self.segment_total_vars


def build_segment_total_varnames(flow_system: Any) -> set[str]:
    """Build segment total variable names - BACKWARDS COMPATIBILITY FALLBACK.

    This function is only used when variable_categories is empty (old FlowSystems
    saved before category registration was implemented). New FlowSystems use
    the VariableCategory registry with EXPAND_DIVIDE categories.

    For segmented systems, these variables contain values that are summed over
    segments. When expanded to hourly resolution, they need to be divided by
    segment duration to get correct hourly rates.

    Args:
        flow_system: The FlowSystem to extract variable patterns from.

    Returns:
        Set of variable names that should be divided by expansion divisor.
    """
    segment_total_vars: set[str] = set()

    # Get all effect names
    effect_names = list(flow_system.effects.keys())

    # 1. Per-timestep totals for each effect: {effect}(temporal)|per_timestep
    for effect in effect_names:
        segment_total_vars.add(f'{effect}(temporal)|per_timestep')

    # 2. Flow contributions to effects: {flow}->{effect}(temporal)
    for flow_label in flow_system.flows:
        for effect in effect_names:
            segment_total_vars.add(f'{flow_label}->{effect}(temporal)')

    # 3. Component contributions to effects: {component}->{effect}(temporal)
    for component_label in flow_system.components:
        for effect in effect_names:
            segment_total_vars.add(f'{component_label}->{effect}(temporal)')

    # 4. Effect-to-effect contributions (from share_from_temporal)
    for target_effect_name, target_effect in flow_system.effects.items():
        if target_effect.share_from_temporal:
            for source_effect_name in target_effect.share_from_temporal:
                segment_total_vars.add(f'{source_effect_name}(temporal)->{target_effect_name}(temporal)')

    return segment_total_vars


def interpolate_charge_state_segmented(
    da: xr.DataArray,
    clustering: Clustering,
    original_timesteps: pd.DatetimeIndex,
) -> xr.DataArray:
    """Interpolate charge_state values within segments for segmented systems.

    For segmented systems, charge_state has values at segment boundaries (n_segments+1).
    Instead of repeating the start boundary value for all timesteps in a segment,
    this function interpolates between start and end boundary values to show the
    actual charge trajectory as the storage charges/discharges.

    Uses vectorized xarray operations via Clustering class properties.

    Args:
        da: charge_state DataArray with dims (cluster, time) where time has n_segments+1 entries.
        clustering: Clustering object with segment info.
        original_timesteps: Original timesteps to expand to.

    Returns:
        Interpolated charge_state with dims (time, ...) for original timesteps.
    """
    # Get multi-dimensional properties from Clustering
    segment_assignments = clustering.results.segment_assignments
    segment_durations = clustering.results.segment_durations
    position_within_segment = clustering.results.position_within_segment
    cluster_assignments = clustering.cluster_assignments

    # Compute original period index and position within period directly
    n_original_timesteps = len(original_timesteps)
    timesteps_per_cluster = clustering.timesteps_per_cluster
    n_original_clusters = clustering.n_original_clusters

    # For each original timestep, compute which original period it belongs to
    original_period_indices = np.minimum(
        np.arange(n_original_timesteps) // timesteps_per_cluster,
        n_original_clusters - 1,
    )
    # Position within the period (0 to timesteps_per_cluster-1)
    positions_in_period = np.arange(n_original_timesteps) % timesteps_per_cluster

    # Create DataArrays for indexing (with original_time dimension, coords added later)
    original_period_da = xr.DataArray(original_period_indices, dims=['original_time'])
    position_in_period_da = xr.DataArray(positions_in_period, dims=['original_time'])

    # Map original period to cluster
    cluster_indices = cluster_assignments.isel(original_cluster=original_period_da)

    # Get segment index and position for each original timestep
    seg_indices = segment_assignments.isel(cluster=cluster_indices, time=position_in_period_da)
    positions = position_within_segment.isel(cluster=cluster_indices, time=position_in_period_da)
    durations = segment_durations.isel(cluster=cluster_indices, segment=seg_indices)

    # Calculate interpolation factor: position within segment (0 to 1)
    factor = xr.where(durations > 1, (positions + 0.5) / durations, 0.5)

    # Get start and end boundary values from charge_state
    start_vals = da.isel(cluster=cluster_indices, time=seg_indices)
    end_vals = da.isel(cluster=cluster_indices, time=seg_indices + 1)

    # Linear interpolation
    interpolated = start_vals + (end_vals - start_vals) * factor

    # Clean up coordinate artifacts and rename
    interpolated = interpolated.drop_vars(['cluster', 'time', 'segment'], errors='ignore')
    interpolated = interpolated.rename({'original_time': 'time'}).assign_coords(time=original_timesteps)

    return interpolated.transpose('time', ...).assign_attrs(da.attrs)


def expand_first_timestep_only(
    da: xr.DataArray,
    clustering: Clustering,
    original_timesteps: pd.DatetimeIndex,
) -> xr.DataArray:
    """Expand binary event variables (startup/shutdown) to first timestep of each segment.

    For segmented systems, binary event variables like startup and shutdown indicate
    that an event occurred somewhere in the segment. When expanding, we place the
    event at the first timestep of each segment and set all other timesteps to 0.

    This function is only used for segmented systems. For non-segmented systems,
    the timing within the cluster is preserved by normal expansion.

    Args:
        da: Binary event DataArray with dims including (cluster, time).
        clustering: Clustering object with segment info (must be segmented).
        original_timesteps: Original timesteps to expand to.

    Returns:
        Expanded DataArray with event values only at first timestep of each segment.
    """
    # First expand normally (repeats values)
    expanded = clustering.expand_data(da, original_time=original_timesteps)

    # Build mask: True only at first timestep of each segment
    n_original_timesteps = len(original_timesteps)
    timesteps_per_cluster = clustering.timesteps_per_cluster
    n_original_clusters = clustering.n_original_clusters

    position_within_segment = clustering.results.position_within_segment
    cluster_assignments = clustering.cluster_assignments

    # Compute original period index and position within period
    original_period_indices = np.minimum(
        np.arange(n_original_timesteps) // timesteps_per_cluster,
        n_original_clusters - 1,
    )
    positions_in_period = np.arange(n_original_timesteps) % timesteps_per_cluster

    # Create DataArrays for indexing (coords added later after rename)
    original_period_da = xr.DataArray(original_period_indices, dims=['original_time'])
    position_in_period_da = xr.DataArray(positions_in_period, dims=['original_time'])

    # Map to cluster and get position within segment
    cluster_indices = cluster_assignments.isel(original_cluster=original_period_da)
    pos_in_segment = position_within_segment.isel(cluster=cluster_indices, time=position_in_period_da)

    # Clean up and create mask
    pos_in_segment = pos_in_segment.drop_vars(['cluster', 'time'], errors='ignore')
    pos_in_segment = pos_in_segment.rename({'original_time': 'time'}).assign_coords(time=original_timesteps)

    # First timestep of segment has position 0
    is_first = pos_in_segment == 0

    # Apply mask: keep value at first timestep, zero elsewhere
    result = xr.where(is_first, expanded, 0)
    return result.assign_attrs(da.attrs)


def append_final_state(
    expanded: xr.DataArray,
    da: xr.DataArray,
    clustering: Clustering,
    original_timesteps_extra: pd.DatetimeIndex,
) -> xr.DataArray:
    """Append final state value from original data to expanded data.

    For state variables like charge_state, the expanded data has n timesteps but
    we need n+1 to include the final state. This function appends the final value.

    Args:
        expanded: Already expanded DataArray with n timesteps.
        da: Original clustered DataArray with (cluster, time) dims.
        clustering: Clustering object with cluster assignments.
        original_timesteps_extra: Original timesteps including the extra final timestep.

    Returns:
        Expanded DataArray with the final state appended.
    """
    n_original_timesteps = len(original_timesteps_extra) - 1
    timesteps_per_cluster = clustering.timesteps_per_cluster
    n_original_clusters = clustering.n_original_clusters

    # Index of last valid original cluster
    last_original_cluster_idx = min(
        (n_original_timesteps - 1) // timesteps_per_cluster,
        n_original_clusters - 1,
    )

    cluster_assignments = clustering.cluster_assignments
    if cluster_assignments.ndim == 1:
        last_cluster = int(cluster_assignments.values[last_original_cluster_idx])
        extra_val = da.isel(cluster=last_cluster, time=-1)
    else:
        last_clusters = cluster_assignments.isel(original_cluster=last_original_cluster_idx)
        extra_val = da.isel(cluster=last_clusters, time=-1)

    extra_val = extra_val.drop_vars(['cluster', 'time'], errors='ignore')
    extra_val = extra_val.expand_dims(time=[original_timesteps_extra[-1]])
    return xr.concat([expanded, extra_val], dim='time')
