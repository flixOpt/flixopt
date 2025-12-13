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
from .io import suppress_output

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

        Rolling horizon optimization:

        >>> segments = flow_system.optimize.rolling_horizon(solver, horizon=168)
        >>> print(flow_system.solution)  # Combined result
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
            ...     print(f'Segment {i}: {seg.solution["costs"].item():.2f}')

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
        logger.info(
            f'Starting Rolling Horizon Optimization - Segments: {n_segments}, Horizon: {horizon}, Overlap: {overlap}'
        )

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

                # Suppress output when progress bar is shown (including logger and solver)
                if CONFIG.Solving.log_to_console:
                    # Temporarily raise logger level to suppress INFO messages
                    original_level = logger.level
                    logger.setLevel(logging.WARNING)
                    try:
                        with suppress_output():
                            segment_fs = self._fs.transform.isel(time=slice(start_idx, end_idx))
                            if i > 0 and nr_of_previous_values > 0:
                                self._transfer_state(
                                    source_fs=segment_flow_systems[i - 1],
                                    target_fs=segment_fs,
                                    horizon=horizon,
                                    nr_of_previous_values=nr_of_previous_values,
                                )
                            segment_fs.build_model()
                            if i == 0:
                                self._check_no_investments(segment_fs)
                            segment_fs.solve(solver)
                    finally:
                        logger.setLevel(original_level)
                else:
                    segment_fs = self._fs.transform.isel(time=slice(start_idx, end_idx))
                    if i > 0 and nr_of_previous_values > 0:
                        self._transfer_state(
                            source_fs=segment_flow_systems[i - 1],
                            target_fs=segment_fs,
                            horizon=horizon,
                            nr_of_previous_values=nr_of_previous_values,
                        )
                    segment_fs.build_model()
                    if i == 0:
                        self._check_no_investments(segment_fs)
                    segment_fs.solve(solver)

                segment_flow_systems.append(segment_fs)

        finally:
            progress_bar.close()

        # Combine segment solutions
        logger.info('Combining segment solutions...')
        self._finalize_solution(segment_flow_systems, horizon)

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

    def _transfer_state(
        self,
        source_fs: FlowSystem,
        target_fs: FlowSystem,
        horizon: int,
        nr_of_previous_values: int,
    ) -> None:
        """Transfer final state from source segment to target segment.

        Transfers:
        - Flow previous_flow_rate: Last nr_of_previous_values from non-overlap portion
        - Storage initial_charge_state: Charge state at end of non-overlap portion
        """
        from .components import Storage

        solution = source_fs.solution
        time_slice = slice(horizon - nr_of_previous_values, horizon)

        # Transfer flow rates (for uptime/downtime tracking)
        for label, target_flow in target_fs.flows.items():
            var_name = f'{label}|flow_rate'
            if var_name in solution:
                values = solution[var_name].isel(time=time_slice).values
                target_flow.previous_flow_rate = values.item() if values.size == 1 else values

        # Transfer storage charge states
        for label, target_comp in target_fs.components.items():
            if isinstance(target_comp, Storage):
                var_name = f'{label}|charge_state'
                if var_name in solution:
                    target_comp.initial_charge_state = solution[var_name].isel(time=horizon - 1).item()

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

    def _finalize_solution(
        self,
        segment_flow_systems: list[FlowSystem],
        horizon: int,
    ) -> None:
        """Combine segment solutions and compute derived values directly (no re-solve)."""
        # Combine all solution variables from segments
        combined_solution = self._combine_solutions(segment_flow_systems, horizon)

        # Assign combined solution to the original FlowSystem
        self._fs._solution = combined_solution

    def _combine_solutions(
        self,
        segment_flow_systems: list[FlowSystem],
        horizon: int,
    ) -> xr.Dataset:
        """Combine segment solutions into a single Dataset, recomputing effect totals.

        Effect variables are identified from the FlowSystem's effect labels:
        - {effect}: Total = temporal + periodic (recomputed)
        - {effect}(temporal): Sum of per_timestep values (recomputed)
        - {effect}(periodic): Sum across segments (no overlap)
        - {effect}(temporal)|per_timestep: Time-dependent (concatenated with overlap trimming)
        """
        if not segment_flow_systems:
            raise ValueError('No segments to combine.')

        # Get effect labels from the original FlowSystem (includes Penalty)
        effect_labels = set(self._fs.effects.keys())

        combined_vars: dict[str, xr.DataArray] = {}
        first_solution = segment_flow_systems[0].solution

        for var_name in first_solution.data_vars:
            first_var = first_solution[var_name]

            if 'time' in first_var.dims:
                # Time-dependent: concatenate with overlap trimming
                combined_vars[var_name] = self._concat_time_variable(segment_flow_systems, var_name, horizon)

            elif var_name in effect_labels:
                # Effect total: skip for now, will recompute from temporal + periodic
                pass

            elif var_name.endswith('(temporal)') and var_name[:-10] in effect_labels:
                # Temporal effect total: skip for now, will recompute from per_timestep
                pass

            elif var_name.endswith('(periodic)') and var_name[:-10] in effect_labels:
                # Periodic effect: sum across all segments (no overlap issue)
                combined_vars[var_name] = sum(seg.solution[var_name] for seg in segment_flow_systems)

            else:
                # Other scalar: use last segment's value
                combined_vars[var_name] = segment_flow_systems[-1].solution[var_name]

        # Recompute effect totals from combined per-timestep values
        for effect_label in effect_labels:
            per_timestep_var = f'{effect_label}(temporal)|per_timestep'
            temporal_var = f'{effect_label}(temporal)'
            periodic_var = f'{effect_label}(periodic)'

            # Temporal total = sum of per_timestep
            if per_timestep_var in combined_vars:
                combined_vars[temporal_var] = combined_vars[per_timestep_var].sum(dim='time', skipna=True)

            # Total = temporal + periodic
            total = xr.DataArray(0.0)
            if temporal_var in combined_vars:
                total = total + combined_vars[temporal_var]
            if periodic_var in combined_vars:
                total = total + combined_vars[periodic_var]
            combined_vars[effect_label] = total

        return xr.Dataset(combined_vars)

    def _concat_time_variable(
        self,
        segment_flow_systems: list[FlowSystem],
        var_name: str,
        horizon: int,
    ) -> xr.DataArray:
        """Concatenate a time-dependent variable from segments, trimming overlaps."""
        arrays = []
        for i, seg_fs in enumerate(segment_flow_systems):
            da = seg_fs.solution[var_name]
            # Trim overlap for all segments except the last
            if i < len(segment_flow_systems) - 1:
                da = da.isel(time=slice(None, horizon))
            arrays.append(da)
        return xr.concat(arrays, dim='time')
