"""
Optimization accessor for FlowSystem.

This module provides the OptimizeAccessor class that enables the
`flow_system.optimize(...)` pattern with extensible optimization methods.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import xarray as xr
from tqdm import tqdm

from .config import CONFIG

if TYPE_CHECKING:
    from .flow_system import FlowSystem
    from .solvers import _Solver

logger = logging.getLogger('flixopt')


class OptimizeAccessor:
    """
    Accessor for optimization methods on FlowSystem.

    This class provides the optimization API for FlowSystem, accessible via
    `flow_system.optimize`. It supports both direct calling (standard optimization)
    and method access for specialized optimization modes.

    Examples:
        Standard optimization (via __call__):

        >>> flow_system.optimize(solver)
        >>> print(flow_system.solution)

        Future specialized modes:

        >>> flow_system.optimize.clustered(solver, aggregation=params)
        >>> flow_system.optimize.mga(solver, alternatives=5)
    """

    def __init__(self, flow_system: FlowSystem) -> None:
        """
        Initialize the accessor with a reference to the FlowSystem.

        Args:
            flow_system: The FlowSystem to optimize.
        """
        self._fs = flow_system

    def __call__(self, solver: _Solver, normalize_weights: bool = True) -> FlowSystem:
        """
        Build and solve the optimization model in one step.

        This is a convenience method that combines `build_model()` and `solve()`.
        Use this for simple optimization workflows. For more control (e.g., inspecting
        the model before solving, or adding custom constraints), use `build_model()`
        and `solve()` separately.

        Args:
            solver: The solver to use (e.g., HighsSolver, GurobiSolver).
            normalize_weights: Whether to normalize scenario/period weights to sum to 1.

        Returns:
            The FlowSystem, for method chaining.

        Examples:
            Simple optimization:

            >>> flow_system.optimize(HighsSolver())
            >>> print(flow_system.solution['Boiler(Q_th)|flow_rate'])

            Access element solutions directly:

            >>> flow_system.optimize(solver)
            >>> boiler = flow_system.components['Boiler']
            >>> print(boiler.solution)

            Method chaining:

            >>> solution = flow_system.optimize(solver).solution
        """
        self._fs.build_model(normalize_weights)
        self._fs.solve(solver)
        return self._fs

    def rolling_horizon(
        self,
        solver: _Solver,
        horizon: int = 100,
        overlap: int = 0,
        nr_of_previous_values: int = 1,
    ) -> list[FlowSystem]:
        """
        Solve the optimization using a rolling horizon approach.

        Divides the time horizon into overlapping segments that are solved sequentially.
        Each segment uses final values from the previous segment as initial conditions,
        ensuring dynamic continuity across the solution. The combined solution is stored
        on the original FlowSystem.

        This approach is useful for:
        - Large-scale problems that exceed memory limits
        - Annual planning with seasonal variations
        - Operational planning with limited foresight

        Args:
            solver: The solver to use (e.g., HighsSolver, GurobiSolver).
            horizon: Number of timesteps in each segment (excluding overlap).
                Must be > 2. Larger values provide better optimization at the cost
                of memory and computation time. Default: 100.
            overlap: Number of additional timesteps added to each segment for lookahead.
                Improves storage optimization by providing foresight. Higher values
                improve solution quality but increase computational cost. Default: 0.
            nr_of_previous_values: Number of previous timestep values to transfer between
                segments for initialization (e.g., for uptime/downtime tracking). Default: 1.

        Returns:
            List of segment FlowSystems, each with their individual solution.
            The combined solution (with overlaps trimmed) is stored on the original FlowSystem.

        Raises:
            ValueError: If horizon <= 2 or overlap < 0.
            ValueError: If horizon + overlap > total timesteps.
            ValueError: If InvestParameters are used (not supported in rolling horizon).

        Examples:
            Basic rolling horizon optimization:

            >>> segments = flow_system.optimize.rolling_horizon(
            ...     solver,
            ...     horizon=168,  # Weekly segments
            ...     overlap=24,  # 1-day lookahead
            ... )
            >>> print(flow_system.solution)  # Combined result

            Inspect individual segments:

            >>> for i, seg in enumerate(segments):
            ...     print(f'Segment {i}: {seg.solution["costs(total)"].item():.2f}')

        Note:
            - InvestParameters are not supported as investment decisions require
              full-horizon optimization.
            - Global constraints (flow_hours_max, etc.) may produce suboptimal results
              as they cannot be enforced globally across segments.
            - Storage optimization may be suboptimal compared to full-horizon solutions
              due to limited foresight in each segment.
        """

        # Validation
        if horizon <= 2:
            raise ValueError('horizon must be greater than 2 to avoid internal side effects.')
        if overlap < 0:
            raise ValueError('overlap must be non-negative.')
        if nr_of_previous_values < 0:
            raise ValueError('nr_of_previous_values must be non-negative.')
        if nr_of_previous_values > horizon:
            raise ValueError('nr_of_previous_values cannot exceed horizon.')

        total_timesteps = len(self._fs.timesteps)
        horizon_with_overlap = horizon + overlap

        if horizon_with_overlap > total_timesteps:
            raise ValueError(
                f'horizon + overlap ({horizon_with_overlap}) cannot exceed total timesteps ({total_timesteps}).'
            )

        # Ensure flow system is connected
        if not self._fs.connected_and_transformed:
            self._fs.connect_and_transform()

        # Calculate segment indices
        segment_indices = self._calculate_segment_indices(total_timesteps, horizon, overlap)
        n_segments = len(segment_indices)

        logger.info(f'{"":#^80}')
        logger.info(f'{" Rolling Horizon Optimization ":#^80}')
        logger.info(f'Segments: {n_segments}, Horizon: {horizon}, Overlap: {overlap}')

        # Store original initial values for restoration and state transfer
        original_initial_values = self._store_initial_values()

        # Create and solve segments
        segment_flow_systems: list[FlowSystem] = []

        progress_bar = tqdm(
            enumerate(segment_indices),
            total=n_segments,
            desc='Solving segments',
            unit='segment',
            file=sys.stdout,
            disable=not CONFIG.Solving.log_to_console,
        )

        try:
            for i, (start_idx, end_idx) in progress_bar:
                progress_bar.set_description(f'Segment {i + 1}/{n_segments} (timesteps {start_idx}-{end_idx})')

                # Create segment FlowSystem
                segment_fs = self._fs.transform.isel(time=slice(start_idx, end_idx))

                # Transfer state from previous segment
                if i > 0 and nr_of_previous_values > 0:
                    self._transfer_state(
                        source_fs=segment_flow_systems[i - 1],
                        target_fs=segment_fs,
                        horizon=horizon,
                        nr_of_previous_values=nr_of_previous_values,
                    )

                # Build and solve
                segment_fs.build_model()

                # Check for investments (only on first segment)
                if i == 0:
                    self._check_no_investments(segment_fs)

                segment_fs.solve(solver)
                segment_flow_systems.append(segment_fs)

        finally:
            progress_bar.close()

        # Combine solutions and store on original FlowSystem
        combined_solution = self._combine_solutions(segment_flow_systems, horizon)
        self._fs._solution = combined_solution

        # Restore original initial values
        self._restore_initial_values(original_initial_values)

        logger.info(f'Rolling horizon optimization completed: {n_segments} segments solved.')

        return segment_flow_systems

    def _calculate_segment_indices(self, total_timesteps: int, horizon: int, overlap: int) -> list[tuple[int, int]]:
        """Calculate start and end indices for each segment."""
        segments = []
        start = 0
        while start < total_timesteps:
            end = min(start + horizon + overlap, total_timesteps)
            segments.append((start, end))
            start += horizon  # Move by horizon (not horizon + overlap)
            if end == total_timesteps:
                break
        return segments

    def _store_initial_values(self) -> dict:
        """Store original initial values for later restoration."""
        from .components import Storage

        values = {}
        for flow in self._fs.flows.values():
            values[f'flow|{flow.label_full}'] = flow.previous_flow_rate

        for comp in self._fs.components.values():
            if isinstance(comp, Storage):
                values[f'storage|{comp.label_full}'] = comp.initial_charge_state

        return values

    def _restore_initial_values(self, values: dict) -> None:
        """Restore original initial values after rolling horizon."""
        from .components import Storage

        for flow in self._fs.flows.values():
            key = f'flow|{flow.label_full}'
            if key in values:
                flow.previous_flow_rate = values[key]

        for comp in self._fs.components.values():
            if isinstance(comp, Storage):
                key = f'storage|{comp.label_full}'
                if key in values:
                    comp.initial_charge_state = values[key]

    def _transfer_state(
        self,
        source_fs: FlowSystem,
        target_fs: FlowSystem,
        horizon: int,
        nr_of_previous_values: int,
    ) -> None:
        """Transfer final state from source segment to target segment."""

        from .components import Storage

        # Transfer flow rates (for uptime/downtime tracking)
        for source_flow in source_fs.flows.values():
            target_flow = target_fs.flows.get(source_flow.label_full)
            if target_flow is None:
                continue

            # Get last nr_of_previous_values from source solution
            flow_rate_var = f'{source_flow.label_full}|flow_rate'
            if flow_rate_var in source_fs.solution:
                # Select from the non-overlap portion (first 'horizon' timesteps)
                values = (
                    source_fs.solution[flow_rate_var].isel(time=slice(horizon - nr_of_previous_values, horizon)).values
                )
                target_flow.previous_flow_rate = values if len(values) > 1 else values.item()

        # Transfer storage charge states
        for source_comp in source_fs.components.values():
            if not isinstance(source_comp, Storage):
                continue

            target_comp = target_fs.components.get(source_comp.label_full)
            if target_comp is None or not isinstance(target_comp, Storage):
                continue

            charge_var = f'{source_comp.label_full}|charge_state'
            if charge_var in source_fs.solution:
                # Get charge state at the end of the non-overlap portion
                # Use horizon index (0-indexed, so horizon-1 is last non-overlap)
                charge_state = source_fs.solution[charge_var].isel(time=horizon - 1).values.item()
                target_comp.initial_charge_state = charge_state

    def _check_no_investments(self, segment_fs: FlowSystem) -> None:
        """Check that no InvestParameters are used (not supported in rolling horizon)."""
        from .features import InvestmentModel

        invest_elements = []
        for component in segment_fs.components.values():
            for model in component.submodel.all_submodels:
                if isinstance(model, InvestmentModel):
                    invest_elements.append(model.label_full)

        if invest_elements:
            raise ValueError(
                f'InvestParameters are not supported in rolling horizon optimization. '
                f'Found InvestmentModels: {invest_elements}. '
                f'Use standard optimize() for problems with investments.'
            )

    def _combine_solutions(self, segment_flow_systems: list[FlowSystem], horizon: int) -> xr.Dataset:
        """Combine segment solutions, trimming overlaps."""
        if not segment_flow_systems:
            raise ValueError('No segments to combine.')

        # Get all variable names from first segment
        var_names = list(segment_flow_systems[0].solution.data_vars)

        # Identify effect names for special handling
        effect_labels = {e.label for e in self._fs.effects.values()}

        combined_vars = {}
        for var_name in var_names:
            arrays = []
            for i, seg_fs in enumerate(segment_flow_systems):
                da = seg_fs.solution[var_name]

                # Check if this variable has a time dimension
                if 'time' in da.dims:
                    if i < len(segment_flow_systems) - 1:
                        # Not the last segment: trim to horizon (exclude overlap)
                        da = da.isel(time=slice(None, horizon))
                    # Last segment: keep all timesteps
                arrays.append(da)

            # Concatenate along time if time dimension exists
            if 'time' in segment_flow_systems[0].solution[var_name].dims:
                combined_vars[var_name] = xr.concat(arrays, dim='time')
            else:
                # For non-time scalars, just take the last value for now
                # Effect totals will be recalculated below
                combined_vars[var_name] = arrays[-1]

        # Recalculate effect totals from combined per-timestep data
        for effect_label in effect_labels:
            per_timestep_key = f'{effect_label}(temporal)|per_timestep'
            temporal_key = f'{effect_label}(temporal)'
            total_key = effect_label

            if per_timestep_key in combined_vars:
                # Recalculate temporal sum from per-timestep values
                combined_vars[temporal_key] = combined_vars[per_timestep_key].sum()

                # Recalculate total (temporal + periodic)
                periodic_key = f'{effect_label}(periodic)'
                if periodic_key in combined_vars:
                    combined_vars[total_key] = combined_vars[temporal_key] + combined_vars[periodic_key]
                else:
                    combined_vars[total_key] = combined_vars[temporal_key]

        return xr.Dataset(combined_vars)
