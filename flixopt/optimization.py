"""
This module contains the Optimization functionality for the flixopt framework.
It is used to optimize a FlowSystemModel for a given FlowSystem through a solver.

There are two Optimization types:
    1. Optimization: Optimizes the FlowSystemModel for the full FlowSystem
    2. SegmentedOptimization: Solves a FlowSystemModel for each individual Segment of the FlowSystem.

For time series aggregation (clustering), use FlowSystem.transform.cluster() instead.
"""

from __future__ import annotations

import logging
import math
import pathlib
import sys
import timeit
import warnings
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from tqdm import tqdm

from . import io as fx_io
from .components import Storage
from .config import CONFIG, DEPRECATION_REMOVAL_VERSION, SUCCESS_LEVEL
from .effects import PENALTY_EFFECT_LABEL
from .features import InvestmentModel
from .results import Results, SegmentedResults

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

    from .flow_system import FlowSystem
    from .solvers import _Solver
    from .structure import FlowSystemModel

logger = logging.getLogger('flixopt')


@runtime_checkable
class OptimizationProtocol(Protocol):
    """
    Protocol defining the interface that all optimization types should implement.

    This protocol ensures type consistency across different optimization approaches
    without forcing them into an artificial inheritance hierarchy.

    Attributes:
        name: Name of the optimization
        flow_system: FlowSystem being optimized
        folder: Directory where results are saved
        results: Results object after solving
        durations: Dictionary tracking time spent in different phases
    """

    name: str
    flow_system: FlowSystem
    folder: pathlib.Path
    results: Results | SegmentedResults | None
    durations: dict[str, float]

    @property
    def modeled(self) -> bool:
        """Returns True if the optimization has been modeled."""
        ...

    @property
    def main_results(self) -> dict[str, int | float | dict]:
        """Returns main results including objective, effects, and investment decisions."""
        ...

    @property
    def summary(self) -> dict:
        """Returns summary information about the optimization."""
        ...


def _initialize_optimization_common(
    obj: Any,
    name: str,
    flow_system: FlowSystem,
    folder: pathlib.Path | None = None,
    normalize_weights: bool = True,
) -> None:
    """
    Shared initialization logic for all optimization types.

    This helper function encapsulates common initialization code to avoid duplication
    across Optimization, ClusteredOptimization, and SegmentedOptimization.

    Args:
        obj: The optimization object being initialized
        name: Name of the optimization
        flow_system: FlowSystem to optimize
        folder: Directory for saving results
        normalize_weights: Whether to normalize scenario weights
    """
    obj.name = name

    if flow_system.used_in_calculation:
        logger.warning(
            f'This FlowSystem is already used in an optimization:\n{flow_system}\n'
            f'Creating a copy of the FlowSystem for Optimization "{obj.name}".'
        )
        flow_system = flow_system.copy()

    obj.normalize_weights = normalize_weights

    flow_system._used_in_optimization = True

    obj.flow_system = flow_system
    obj.model = None

    obj.durations = {'modeling': 0.0, 'solving': 0.0, 'saving': 0.0}
    obj.folder = pathlib.Path.cwd() / 'results' if folder is None else pathlib.Path(folder)
    obj.results = None

    if obj.folder.exists() and not obj.folder.is_dir():
        raise NotADirectoryError(f'Path {obj.folder} exists and is not a directory.')
    # Create folder and any necessary parent directories
    obj.folder.mkdir(parents=True, exist_ok=True)


class Optimization:
    """
    Standard optimization that solves the complete problem using all time steps.

    This is the default optimization approach that considers every time step,
    providing the most accurate but computationally intensive solution.

    For large problems, consider using ClusteredOptimization (time aggregation)
    or SegmentedOptimization (temporal decomposition) instead.

    Args:
        name: name of optimization
        flow_system: flow_system which should be optimized
        folder: folder where results should be saved. If None, then the current working directory is used.
        normalize_weights: Whether to automatically normalize the weights of scenarios to sum up to 1 when solving.

    Examples:
        Basic usage:
        ```python
        from flixopt import Optimization

        opt = Optimization(name='my_optimization', flow_system=energy_system, folder=Path('results'))
        opt.do_modeling()
        opt.solve(solver=gurobi)
        results = opt.results
        ```
    """

    # Attributes set by __init__ / _initialize_optimization_common
    name: str
    flow_system: FlowSystem
    folder: pathlib.Path
    results: Results | None
    durations: dict[str, float]
    model: FlowSystemModel | None
    normalize_weights: bool

    def __init__(
        self,
        name: str,
        flow_system: FlowSystem,
        folder: pathlib.Path | None = None,
        normalize_weights: bool = True,
    ):
        warnings.warn(
            f'Optimization is deprecated and will be removed in v{DEPRECATION_REMOVAL_VERSION}. '
            'Use FlowSystem.optimize(solver) or FlowSystem.build_model() + FlowSystem.solve(solver) instead. '
            'Access results via FlowSystem.solution.',
            DeprecationWarning,
            stacklevel=2,
        )
        _initialize_optimization_common(
            self,
            name=name,
            flow_system=flow_system,
            folder=folder,
            normalize_weights=normalize_weights,
        )

    def do_modeling(self) -> Optimization:
        t_start = timeit.default_timer()
        self.flow_system.connect_and_transform()

        self.model = self.flow_system.create_model(self.normalize_weights)
        self.model.do_modeling()

        self.durations['modeling'] = round(timeit.default_timer() - t_start, 2)
        return self

    def fix_sizes(self, ds: xr.Dataset | None = None, decimal_rounding: int | None = 5) -> Optimization:
        """Fix the sizes of the optimizations to specified values.

        Args:
            ds: The dataset that contains the variable names mapped to their sizes. If None, the dataset is loaded from the results.
            decimal_rounding: The number of decimal places to round the sizes to. If no rounding is applied, numerical errors might lead to infeasibility.
        """
        if not self.modeled:
            raise RuntimeError('Model was not created. Call do_modeling() first.')

        if ds is None:
            if self.results is None:
                raise RuntimeError('No dataset provided and no results available to load sizes from.')
            ds = self.results.solution

        if decimal_rounding is not None:
            ds = ds.round(decimal_rounding)

        for name, da in ds.data_vars.items():
            if '|size' not in name:
                continue
            if name not in self.model.variables:
                logger.debug(f'Variable {name} not found in calculation model. Skipping.')
                continue

            con = self.model.add_constraints(
                self.model[name] == da,
                name=f'{name}-fixed',
            )
            logger.debug(f'Fixed "{name}":\n{con}')

        return self

    def solve(
        self, solver: _Solver, log_file: pathlib.Path | None = None, log_main_results: bool | None = None
    ) -> Optimization:
        # Auto-call do_modeling() if not already done
        if not self.modeled:
            logger.info('Model not yet created. Calling do_modeling() automatically.')
            self.do_modeling()

        t_start = timeit.default_timer()

        self.model.solve(
            log_fn=pathlib.Path(log_file) if log_file is not None else self.folder / f'{self.name}.log',
            solver_name=solver.name,
            progress=CONFIG.Solving.log_to_console,
            **solver.options,
        )
        self.durations['solving'] = round(timeit.default_timer() - t_start, 2)
        logger.log(SUCCESS_LEVEL, f'Model solved with {solver.name} in {self.durations["solving"]:.2f} seconds.')
        logger.info(f'Model status after solve: {self.model.status}')

        if self.model.status == 'warning':
            # Save the model and the flow_system to file in case of infeasibility
            self.folder.mkdir(parents=True, exist_ok=True)
            paths = fx_io.ResultsPaths(self.folder, self.name)
            from .io import document_linopy_model

            document_linopy_model(self.model, paths.model_documentation)
            self.flow_system.to_netcdf(paths.flow_system, overwrite=True)
            raise RuntimeError(
                f'Model was infeasible. Please check {paths.model_documentation=} and {paths.flow_system=} for more information.'
            )

        # Log the formatted output
        should_log = log_main_results if log_main_results is not None else CONFIG.Solving.log_main_results
        if should_log and logger.isEnabledFor(logging.INFO):
            logger.log(
                SUCCESS_LEVEL,
                f'{" Main Results ":#^80}\n' + fx_io.format_yaml_string(self.main_results, compact_numeric_lists=True),
            )

        # Store solution on FlowSystem for direct Element access
        self.flow_system.solution = self.model.solution

        self.results = Results.from_optimization(self)

        return self

    @property
    def main_results(self) -> dict[str, int | float | dict]:
        if self.model is None:
            raise RuntimeError('Optimization has not been solved yet. Call solve() before accessing main_results.')

        try:
            penalty_effect = self.flow_system.effects.penalty_effect
            penalty_section = {
                'temporal': penalty_effect.submodel.temporal.total.solution.values,
                'periodic': penalty_effect.submodel.periodic.total.solution.values,
                'total': penalty_effect.submodel.total.solution.values,
            }
        except KeyError:
            penalty_section = {'temporal': 0.0, 'periodic': 0.0, 'total': 0.0}

        main_results = {
            'Objective': self.model.objective.value,
            'Penalty': penalty_section,
            'Effects': {
                f'{effect.label} [{effect.unit}]': {
                    'temporal': effect.submodel.temporal.total.solution.values,
                    'periodic': effect.submodel.periodic.total.solution.values,
                    'total': effect.submodel.total.solution.values,
                }
                for effect in sorted(self.flow_system.effects.values(), key=lambda e: e.label_full.upper())
                if effect.label_full != PENALTY_EFFECT_LABEL
            },
            'Invest-Decisions': {
                'Invested': {
                    model.label_of_element: model.size.solution
                    for component in self.flow_system.components.values()
                    for model in component.submodel.all_submodels
                    if isinstance(model, InvestmentModel)
                    and model.size.solution.max().item() >= CONFIG.Modeling.epsilon
                },
                'Not invested': {
                    model.label_of_element: model.size.solution
                    for component in self.flow_system.components.values()
                    for model in component.submodel.all_submodels
                    if isinstance(model, InvestmentModel) and model.size.solution.max().item() < CONFIG.Modeling.epsilon
                },
            },
            'Buses with excess': [
                {
                    bus.label_full: {
                        'virtual_supply': bus.submodel.virtual_supply.solution.sum('time'),
                        'virtual_demand': bus.submodel.virtual_demand.solution.sum('time'),
                    }
                }
                for bus in self.flow_system.buses.values()
                if bus.allows_imbalance
                and (
                    bus.submodel.virtual_supply.solution.sum().item() > 1e-3
                    or bus.submodel.virtual_demand.solution.sum().item() > 1e-3
                )
            ],
        }

        return fx_io.round_nested_floats(main_results)

    @property
    def summary(self):
        if self.model is None:
            raise RuntimeError('Optimization has not been solved yet. Call solve() before accessing summary.')

        return {
            'Name': self.name,
            'Number of timesteps': len(self.flow_system.timesteps),
            'Optimization Type': self.__class__.__name__,
            'Constraints': self.model.constraints.ncons,
            'Variables': self.model.variables.nvars,
            'Main Results': self.main_results,
            'Durations': self.durations,
            'Config': CONFIG.to_dict(),
        }

    @property
    def modeled(self) -> bool:
        return True if self.model is not None else False


class SegmentedOptimization:
    """Solve large optimization problems by dividing time horizon into (overlapping) segments.

    This class addresses memory and computational limitations of large-scale optimization
    problems by decomposing the time horizon into smaller overlapping segments that are
    solved sequentially. Each segment uses final values from the previous segment as
    initial conditions, ensuring dynamic continuity across the solution.

    Key Concepts:
        **Temporal Decomposition**: Divides long time horizons into manageable segments
        **Overlapping Windows**: Segments share timesteps to improve storage dynamics
        **Value Transfer**: Final states of one segment become initial states of the next
        **Sequential Solving**: Each segment solved independently but with coupling

    Limitations and Constraints:
        **Investment Parameters**: InvestParameters are not supported in segmented optimizations
        as investment decisions must be made for the entire time horizon, not per segment.

        **Global Constraints**: Time-horizon-wide constraints (flow_hours_total_min/max,
        load_factor_min/max) may produce suboptimal results as they cannot be enforced
        globally across segments.

        **Storage Dynamics**: While overlap helps, storage optimization may be suboptimal
        compared to full-horizon solutions due to limited foresight in each segment.

    Args:
        name: Unique identifier for the calculation, used in result files and logging.
        flow_system: The FlowSystem to optimize, containing all components, flows, and buses.
        timesteps_per_segment: Number of timesteps in each segment (excluding overlap).
            Must be > 2 to avoid internal side effects. Larger values provide better
            optimization at the cost of memory and computation time.
        overlap_timesteps: Number of additional timesteps added to each segment.
            Improves storage optimization by providing lookahead. Higher values
            improve solution quality but increase computational cost.
        nr_of_previous_values: Number of previous timestep values to transfer between
            segments for initialization. Typically 1 is sufficient.
        folder: Directory for saving results. Defaults to current working directory + 'results'.

    Examples:
        Annual optimization with monthly segments:

        ```python
        # 8760 hours annual data with monthly segments (730 hours) and 48-hour overlap
        segmented_calc = SegmentedOptimization(
            name='annual_energy_system',
            flow_system=energy_system,
            timesteps_per_segment=730,  # ~1 month
            overlap_timesteps=48,  # 2 days overlap
            folder=Path('results/segmented'),
        )
        segmented_calc.do_modeling_and_solve(solver='gurobi')
        ```

        Weekly optimization with daily overlap:

        ```python
        # Weekly segments for detailed operational planning
        weekly_calc = SegmentedOptimization(
            name='weekly_operations',
            flow_system=industrial_system,
            timesteps_per_segment=168,  # 1 week (hourly data)
            overlap_timesteps=24,  # 1 day overlap
            nr_of_previous_values=1,
        )
        ```

        Large-scale system with minimal overlap:

        ```python
        # Large system with minimal overlap for computational efficiency
        large_calc = SegmentedOptimization(
            name='large_scale_grid',
            flow_system=grid_system,
            timesteps_per_segment=100,  # Shorter segments
            overlap_timesteps=5,  # Minimal overlap
        )
        ```

    Design Considerations:
        **Segment Size**: Balance between solution quality and computational efficiency.
        Larger segments provide better optimization but require more memory and time.

        **Overlap Duration**: More overlap improves storage dynamics and reduces
        end-effects but increases computational cost. Typically 5-10% of segment length.

        **Storage Systems**: Systems with large storage components benefit from longer
        overlaps to capture charge/discharge cycles effectively.

        **Investment Decisions**: Use Optimization for problems requiring investment
        optimization, as SegmentedOptimization cannot handle investment parameters.

    Common Use Cases:
        - **Annual Planning**: Long-term planning with seasonal variations
        - **Large Networks**: Spatially or temporally large energy systems
        - **Memory-Limited Systems**: When full optimization exceeds available memory
        - **Operational Planning**: Detailed short-term optimization with limited foresight
        - **Sensitivity Analysis**: Quick approximate solutions for parameter studies

    Performance Tips:
        - Start with Optimization and use this class if memory issues occur
        - Use longer overlaps for systems with significant storage
        - Monitor solution quality at segment boundaries for discontinuities

    Warning:
        The evaluation of the solution is a bit more complex than Optimization or ClusteredOptimization
        due to the overlapping individual solutions.

    """

    # Attributes set by __init__ / _initialize_optimization_common
    name: str
    flow_system: FlowSystem
    folder: pathlib.Path
    results: SegmentedResults | None
    durations: dict[str, float]
    model: None  # SegmentedOptimization doesn't use a single model
    normalize_weights: bool

    def __init__(
        self,
        name: str,
        flow_system: FlowSystem,
        timesteps_per_segment: int,
        overlap_timesteps: int,
        nr_of_previous_values: int = 1,
        folder: pathlib.Path | None = None,
    ):
        warnings.warn(
            f'SegmentedOptimization is deprecated and will be removed in v{DEPRECATION_REMOVAL_VERSION}. '
            'A replacement API for segmented optimization will be provided in a future release.',
            DeprecationWarning,
            stacklevel=2,
        )
        _initialize_optimization_common(
            self,
            name=name,
            flow_system=flow_system,
            folder=folder,
        )
        self.timesteps_per_segment = timesteps_per_segment
        self.overlap_timesteps = overlap_timesteps
        self.nr_of_previous_values = nr_of_previous_values

        # Validate overlap_timesteps early
        if self.overlap_timesteps < 0:
            raise ValueError('overlap_timesteps must be non-negative.')

        # Validate timesteps_per_segment early (before using in arithmetic)
        if self.timesteps_per_segment <= 2:
            raise ValueError('timesteps_per_segment must be greater than 2 due to internal side effects.')

        # Validate nr_of_previous_values
        if self.nr_of_previous_values < 0:
            raise ValueError('nr_of_previous_values must be non-negative.')
        if self.nr_of_previous_values > self.timesteps_per_segment:
            raise ValueError('nr_of_previous_values cannot exceed timesteps_per_segment.')

        self.sub_optimizations: list[Optimization] = []

        self.segment_names = [
            f'Segment_{i + 1}' for i in range(math.ceil(len(self.all_timesteps) / self.timesteps_per_segment))
        ]
        self._timesteps_per_segment = self._calculate_timesteps_per_segment()

        if self.timesteps_per_segment_with_overlap > len(self.all_timesteps):
            raise ValueError(
                f'timesteps_per_segment_with_overlap ({self.timesteps_per_segment_with_overlap}) '
                f'cannot exceed total timesteps ({len(self.all_timesteps)}).'
            )

        self.flow_system._connect_network()  # Connect network to ensure that all Flows know their Component
        # Storing all original start values
        self._original_start_values = {
            **{flow.label_full: flow.previous_flow_rate for flow in self.flow_system.flows.values()},
            **{
                comp.label_full: comp.initial_charge_state
                for comp in self.flow_system.components.values()
                if isinstance(comp, Storage)
            },
        }
        self._transfered_start_values: list[dict[str, Any]] = []

    def _create_sub_optimizations(self):
        for i, (segment_name, timesteps_of_segment) in enumerate(
            zip(self.segment_names, self._timesteps_per_segment, strict=True)
        ):
            calc = Optimization(f'{self.name}-{segment_name}', self.flow_system.sel(time=timesteps_of_segment))
            calc.flow_system._connect_network()  # Connect to have Correct names of Flows!

            self.sub_optimizations.append(calc)
            logger.info(
                f'{segment_name} [{i + 1:>2}/{len(self.segment_names):<2}] '
                f'({timesteps_of_segment[0]} -> {timesteps_of_segment[-1]}):'
            )

    def _solve_single_segment(
        self,
        i: int,
        optimization: Optimization,
        solver: _Solver,
        log_file: pathlib.Path | None,
        log_main_results: bool,
        suppress_output: bool,
    ) -> None:
        """Solve a single segment optimization."""
        if i > 0 and self.nr_of_previous_values > 0:
            self._transfer_start_values(i)

        optimization.do_modeling()

        # Check for unsupported Investments, but only in first run
        if i == 0:
            invest_elements = [
                model.label_full
                for component in optimization.flow_system.components.values()
                for model in component.submodel.all_submodels
                if isinstance(model, InvestmentModel)
            ]
            if invest_elements:
                raise ValueError(
                    f'Investments are not supported in SegmentedOptimization. '
                    f'Found InvestmentModels: {invest_elements}. '
                    f'Please use Optimization instead for problems with investments.'
                )

        log_path = pathlib.Path(log_file) if log_file is not None else self.folder / f'{self.name}.log'

        if suppress_output:
            with fx_io.suppress_output():
                optimization.solve(solver, log_file=log_path, log_main_results=log_main_results)
        else:
            optimization.solve(solver, log_file=log_path, log_main_results=log_main_results)

    def do_modeling_and_solve(
        self,
        solver: _Solver,
        log_file: pathlib.Path | None = None,
        log_main_results: bool = False,
        show_individual_solves: bool = False,
    ) -> SegmentedOptimization:
        """Model and solve all segments of the segmented optimization.

        This method creates sub-optimizations for each time segment, then iteratively
        models and solves each segment. It supports two output modes: a progress bar
        for compact output, or detailed individual solve information.

        Args:
            solver: The solver instance to use for optimization (e.g., Gurobi, HiGHS).
            log_file: Optional path to the solver log file. If None, defaults to
                folder/name.log.
            log_main_results: Whether to log main results (objective, effects, etc.)
                after each segment solve. Defaults to False.
            show_individual_solves: If True, shows detailed output for each segment
                solve with logger messages. If False (default), shows a compact progress
                bar with suppressed solver output for cleaner display.

        Returns:
            Self, for method chaining.

        Note:
            The method automatically transfers all start values between segments to ensure
            continuity of storage states and flow rates across segment boundaries.
        """
        logger.info(f'{"":#^80}')
        logger.info(f'{" Segmented Solving ":#^80}')
        self._create_sub_optimizations()

        if show_individual_solves:
            # Path 1: Show individual solves with detailed output
            for i, optimization in enumerate(self.sub_optimizations):
                logger.info(
                    f'Solving segment {i + 1}/{len(self.sub_optimizations)}: '
                    f'{optimization.flow_system.timesteps[0]} -> {optimization.flow_system.timesteps[-1]}'
                )
                self._solve_single_segment(i, optimization, solver, log_file, log_main_results, suppress_output=False)
        else:
            # Path 2: Show only progress bar with suppressed output
            progress_bar = tqdm(
                enumerate(self.sub_optimizations),
                total=len(self.sub_optimizations),
                desc='Solving segments',
                unit='segment',
                file=sys.stdout,
                disable=not CONFIG.Solving.log_to_console,
            )

            try:
                for i, optimization in progress_bar:
                    progress_bar.set_description(
                        f'Solving ({optimization.flow_system.timesteps[0]} -> {optimization.flow_system.timesteps[-1]})'
                    )
                    self._solve_single_segment(
                        i, optimization, solver, log_file, log_main_results, suppress_output=True
                    )
            finally:
                progress_bar.close()

        for calc in self.sub_optimizations:
            for key, value in calc.durations.items():
                self.durations[key] += value

        logger.log(SUCCESS_LEVEL, f'Model solved with {solver.name} in {self.durations["solving"]:.2f} seconds.')

        self.results = SegmentedResults.from_optimization(self)

        return self

    def _transfer_start_values(self, i: int):
        """
        This function gets the last values of the previous solved segment and
        inserts them as start values for the next segment
        """
        timesteps_of_prior_segment = self.sub_optimizations[i - 1].flow_system.timesteps_extra

        start = self.sub_optimizations[i].flow_system.timesteps[0]
        start_previous_values = timesteps_of_prior_segment[self.timesteps_per_segment - self.nr_of_previous_values]
        end_previous_values = timesteps_of_prior_segment[self.timesteps_per_segment - 1]

        logger.debug(
            f'Start of next segment: {start}. Indices of previous values: {start_previous_values} -> {end_previous_values}'
        )
        current_flow_system = self.sub_optimizations[i - 1].flow_system
        next_flow_system = self.sub_optimizations[i].flow_system

        start_values_of_this_segment = {}

        for current_flow in current_flow_system.flows.values():
            next_flow = next_flow_system.flows[current_flow.label_full]
            next_flow.previous_flow_rate = current_flow.submodel.flow_rate.solution.sel(
                time=slice(start_previous_values, end_previous_values)
            ).values
            start_values_of_this_segment[current_flow.label_full] = next_flow.previous_flow_rate

        for current_comp in current_flow_system.components.values():
            next_comp = next_flow_system.components[current_comp.label_full]
            if isinstance(next_comp, Storage):
                next_comp.initial_charge_state = current_comp.submodel.charge_state.solution.sel(time=start).item()
                start_values_of_this_segment[current_comp.label_full] = next_comp.initial_charge_state

        self._transfered_start_values.append(start_values_of_this_segment)

    def _calculate_timesteps_per_segment(self) -> list[pd.DatetimeIndex]:
        timesteps_per_segment = []
        for i, _ in enumerate(self.segment_names):
            start = self.timesteps_per_segment * i
            end = min(start + self.timesteps_per_segment_with_overlap, len(self.all_timesteps))
            timesteps_per_segment.append(self.all_timesteps[start:end])
        return timesteps_per_segment

    @property
    def timesteps_per_segment_with_overlap(self):
        return self.timesteps_per_segment + self.overlap_timesteps

    @property
    def start_values_of_segments(self) -> list[dict[str, Any]]:
        """Gives an overview of the start values of all Segments"""
        return [{name: value for name, value in self._original_start_values.items()}] + [
            start_values for start_values in self._transfered_start_values
        ]

    @property
    def all_timesteps(self) -> pd.DatetimeIndex:
        return self.flow_system.timesteps

    @property
    def modeled(self) -> bool:
        """Returns True if all segments have been modeled."""
        if len(self.sub_optimizations) == 0:
            return False
        return all(calc.modeled for calc in self.sub_optimizations)

    @property
    def main_results(self) -> dict[str, int | float | dict]:
        """Aggregated main results from all segments.

        Note:
            For SegmentedOptimization, results are aggregated from SegmentedResults
            which handles the overlapping segments properly. Individual segment results
            should not be summed directly as they contain overlapping timesteps.

            The objective value shown is the sum of all segment objectives and includes
            double-counting from overlapping regions. It does not represent a true
            full-horizon objective value.
        """
        if self.results is None:
            raise RuntimeError(
                'SegmentedOptimization has not been solved yet. '
                'Call do_modeling_and_solve() first to access main_results.'
            )

        # Use SegmentedResults to get the proper aggregated solution
        return {
            'Note': 'SegmentedOptimization results are aggregated via SegmentedResults',
            'Number of segments': len(self.sub_optimizations),
            'Total timesteps': len(self.all_timesteps),
            'Objective (sum of segments, includes overlaps)': sum(
                calc.model.objective.value for calc in self.sub_optimizations if calc.modeled
            ),
        }

    @property
    def summary(self):
        """Summary of the segmented optimization with aggregated information from all segments."""
        if len(self.sub_optimizations) == 0:
            raise RuntimeError(
                'SegmentedOptimization has no segments yet. Call do_modeling_and_solve() first to access summary.'
            )

        # Aggregate constraints and variables from all segments
        total_constraints = sum(calc.model.constraints.ncons for calc in self.sub_optimizations if calc.modeled)
        total_variables = sum(calc.model.variables.nvars for calc in self.sub_optimizations if calc.modeled)

        return {
            'Name': self.name,
            'Number of timesteps': len(self.flow_system.timesteps),
            'Optimization Type': self.__class__.__name__,
            'Number of segments': len(self.sub_optimizations),
            'Timesteps per segment': self.timesteps_per_segment,
            'Overlap timesteps': self.overlap_timesteps,
            'Constraints (total across segments)': total_constraints,
            'Variables (total across segments)': total_variables,
            'Main Results': self.main_results if self.results else 'Not yet solved',
            'Durations': self.durations,
            'Config': CONFIG.to_dict(),
        }
