import logging
from typing import Dict, List, Optional, Tuple, Union

import linopy
import numpy as np

from .config import CONFIG
from .core import NonTemporalData, Scalar, TemporalData, FlowSystemDimensions
from .structure import Model, FlowSystemModel, BaseFeatureModel

logger = logging.getLogger('flixopt')


class ModelingUtilities:
    """Utility functions for modeling calculations - used across different classes"""

    @staticmethod
    def compute_consecutive_hours_in_state(
        binary_values: TemporalData, hours_per_timestep: Union[int, float, np.ndarray]
    ) -> Scalar:
        """
        Computes the final consecutive duration in state 'on' (=1) in hours, from a binary array.

        Args:
            binary_values: An int or 1D binary array containing only `0`s and `1`s.
            hours_per_timestep: The duration of each timestep in hours.
                If a scalar is provided, it is used for all timesteps.
                If an array is provided, it must be as long as the last consecutive duration in binary_values.

        Returns:
            The duration of the binary variable in hours.

        Raises
        ------
        TypeError
            If the length of binary_values and dt_in_hours is not equal, but None is a scalar.
        """
        if np.isscalar(binary_values) and np.isscalar(hours_per_timestep):
            return binary_values * hours_per_timestep
        elif np.isscalar(binary_values) and not np.isscalar(hours_per_timestep):
            return binary_values * hours_per_timestep[-1]

        if np.isclose(binary_values[-1], 0, atol=CONFIG.modeling.EPSILON):
            return 0

        if np.isscalar(hours_per_timestep):
            hours_per_timestep = np.ones(len(binary_values)) * hours_per_timestep
        hours_per_timestep: np.ndarray

        indexes_with_zero_values = np.where(np.isclose(binary_values, 0, atol=CONFIG.modeling.EPSILON))[0]
        if len(indexes_with_zero_values) == 0:
            nr_of_indexes_with_consecutive_ones = len(binary_values)
        else:
            nr_of_indexes_with_consecutive_ones = len(binary_values) - indexes_with_zero_values[-1] - 1

        if len(hours_per_timestep) < nr_of_indexes_with_consecutive_ones:
            raise ValueError(
                f'When trying to calculate the consecutive duration, the length of the last duration '
                f'({nr_of_indexes_with_consecutive_ones}) is longer than the provided hours_per_timestep ({len(hours_per_timestep)}), '
                f'as {binary_values=}'
            )

        return np.sum(
            binary_values[-nr_of_indexes_with_consecutive_ones:]
            * hours_per_timestep[-nr_of_indexes_with_consecutive_ones:]
        )

    @staticmethod
    def compute_previous_states(previous_values: List[TemporalData], epsilon: float = None) -> np.ndarray:
        """
        Computes the previous states {0, 1} of defining variables as a binary array from their previous values.

        Args:
            previous_values: List of previous values for variables
            epsilon: Tolerance for zero detection (uses CONFIG.modeling.EPSILON if None)

        Returns:
            Binary array of previous states
        """
        if epsilon is None:
            epsilon = CONFIG.modeling.EPSILON

        if not previous_values or all(val is None for val in previous_values):
            return np.array([0])

        # Convert to 2D-array and compute binary on/off states
        previous_values = np.array([values for values in previous_values if values is not None])  # Filter out None
        if previous_values.ndim > 1:
            return np.any(~np.isclose(previous_values, 0, atol=epsilon), axis=0).astype(int)

        return (~np.isclose(previous_values, 0, atol=epsilon)).astype(int)

    @staticmethod
    def compute_previous_on_duration(previous_values: List[TemporalData], hours_per_step: Union[int, float]) -> Scalar:
        """
        Convenience method to compute previous consecutive 'on' duration.

        Args:
            previous_values: List of previous values for variables
            hours_per_step: Duration of each timestep in hours

        Returns:
            Previous consecutive on duration in hours
        """
        if not previous_values:
            return 0

        previous_states = ModelingUtilities.compute_previous_states(previous_values)
        return ModelingUtilities.compute_consecutive_hours_in_state(previous_states, hours_per_step)

    @staticmethod
    def compute_previous_off_duration(previous_values: List[TemporalData], hours_per_step: Union[int, float]) -> Scalar:
        """
        Convenience method to compute previous consecutive 'off' duration.

        Args:
            previous_values: List of previous values for variables
            hours_per_step: Duration of each timestep in hours

        Returns:
            Previous consecutive off duration in hours
        """
        if not previous_values:
            return 0

        previous_states = ModelingUtilities.compute_previous_states(previous_values)
        previous_off_states = 1 - previous_states
        return ModelingUtilities.compute_consecutive_hours_in_state(previous_off_states, hours_per_step)

    @staticmethod
    def get_most_recent_state(previous_values: List[TemporalData]) -> int:
        """
        Get the most recent binary state from previous values.

        Args:
            previous_values: List of previous values for variables

        Returns:
            Most recent binary state (0 or 1)
        """
        if not previous_values:
            return 0

        previous_states = ModelingUtilities.compute_previous_states(previous_values)
        return int(previous_states[-1])


class ModelingPrimitives:
    """Mathematical modeling primitives returning (variables, constraints) tuples"""

    @staticmethod
    def binary_state_pair(
        model: FlowSystemModel, name: str, coords: List[str] = None
    ) -> Tuple[Dict[str, linopy.Variable], Dict[str, linopy.Constraint]]:
        """
        Creates complementary binary variables with completeness constraint.

        Mathematical formulation:
            on[t] + off[t] = 1  ∀t
            on[t], off[t] ∈ {0, 1}

        Returns:
            variables: {'on': binary_var, 'off': binary_var}
            constraints: {'complementary': constraint}
        """
        coords = coords or ['time']

        on = model.add_variables(binary=True, name=f'{name}|on', coords=model.get_coords(coords))
        off = model.add_variables(binary=True, name=f'{name}|off', coords=model.get_coords(coords))

        # Constraint: on + off = 1
        complementary = model.add_constraints(on + off == 1, name=f'{name}|complementary')

        variables = {'on': on, 'off': off}
        constraints = {'complementary': complementary}

        return variables, constraints

    @staticmethod
    def proportionally_bounded_variable(
        model: FlowSystemModel,
        name: str,
        controlling_variable,
        bounds: Tuple[TemporalData, TemporalData],
        coords: List[str] = None,
    ) -> Tuple[Dict[str, linopy.Variable], Dict[str, linopy.Constraint]]:
        """
        Creates variable with bounds proportional to another variable.

        Mathematical formulation:
            lower_factor[t] * controller[t] ≤ variable[t] ≤ upper_factor[t] * controller[t]  ∀t

        Returns:
            variables: {'variable': bounded_var}
            constraints: {'lower_bound': constraint, 'upper_bound': constraint}
        """
        coords = coords or ['time']
        variable = model.add_variables(name=f'{name}|bounded', coords=model.get_coords(coords))

        lower_factor, upper_factor = bounds

        # Constraints: lower_factor * controller ≤ var ≤ upper_factor * controller
        lower_bound = model.add_constraints(
            variable >= controlling_variable * lower_factor, name=f'{name}|proportional_lb'
        )
        upper_bound = model.add_constraints(
            variable <= controlling_variable * upper_factor, name=f'{name}|proportional_ub'
        )

        variables = {'variable': variable}
        constraints = {'lower_bound': lower_bound, 'upper_bound': upper_bound}

        return variables, constraints

    @staticmethod
    def expression_tracking_variable(
        model: FlowSystemModel,
        name: str,
        tracked_expression,
        bounds: Tuple[TemporalData, TemporalData] = None,
        coords: List[str] = None,
    ) -> Tuple[Dict[str, linopy.Variable], Dict[str, linopy.Constraint]]:
        """
        Creates variable that equals a given expression.

        Mathematical formulation:
            tracker = expression
            lower ≤ tracker ≤ upper (if bounds provided)

        Returns:
            variables: {'tracker': tracker_var}
            constraints: {'tracking': constraint}
        """
        coords = coords or ['year', 'scenario']

        if not bounds:
            tracker = model.add_variables(name=f'{name}', coords=model.get_coords(coords))
        else:
            tracker = model.add_variables(
                lower=bounds[0] if bounds[0] is not None else -np.inf,
                upper=bounds[1] if bounds[1] is not None else np.inf,
                name=f'{name}',
                coords=model.get_coords(coords),
            )

        # Constraint: tracker = expression
        tracking = model.add_constraints(tracker == tracked_expression, name=f'{name}')

        variables = {'tracker': tracker}
        constraints = {'tracking': tracking}

        return variables, constraints

    @staticmethod
    def state_transition_variables(
        model: FlowSystemModel, name: str, state_variable, previous_state=0
    ) -> Tuple[Dict[str, linopy.Variable], Dict[str, linopy.Constraint]]:
        """
        Creates switch-on/off variables with state transition logic.

        Mathematical formulation:
            switch_on[t] - switch_off[t] = state[t] - state[t-1]  ∀t > 0
            switch_on[0] - switch_off[0] = state[0] - previous_state
            switch_on[t] + switch_off[t] ≤ 1  ∀t
            switch_on[t], switch_off[t] ∈ {0, 1}

        Returns:
            variables: {'switch_on': binary_var, 'switch_off': binary_var}
            constraints: {'transition': constraint, 'initial': constraint, 'mutex': constraint}
        """
        switch_on = model.add_variables(binary=True, name=f'{name}|switch_on', coords=model.get_coords(['time']))
        switch_off = model.add_variables(binary=True, name=f'{name}|switch_off', coords=model.get_coords(['time']))

        # State transition constraints for t > 0
        transition = model.add_constraints(
            switch_on.isel(time=slice(1, None)) - switch_off.isel(time=slice(1, None))
            == state_variable.isel(time=slice(1, None)) - state_variable.isel(time=slice(None, -1)),
            name=f'{name}|state_transition',
        )

        # Initial state transition for t = 0
        initial = model.add_constraints(
            switch_on.isel(time=0) - switch_off.isel(time=0) == state_variable.isel(time=0) - previous_state,
            name=f'{name}|initial_transition',
        )

        # At most one switch per timestep
        mutex = model.add_constraints(switch_on + switch_off <= 1, name=f'{name}|switch_mutex')

        variables = {'switch_on': switch_on, 'switch_off': switch_off}
        constraints = {'transition': transition, 'initial': initial, 'mutex': mutex}

        return variables, constraints

    @staticmethod
    def big_m_binary_bounds(
        model: FlowSystemModel,
        variable,
        binary_control,
        size_variable,
        relative_bounds: Tuple[TemporalData, TemporalData],
        upper_bound_name: str,
        lower_bound_name: str,
    ) -> Tuple[Dict, Dict[str, linopy.Constraint]]:
        """
        Creates bounds controlled by both binary and continuous variables.

        Mathematical formulation:
            variable[t] ≤ size[t] * upper_factor[t]  ∀t

            If binary_control provided:
                variable[t] ≥ M * (binary[t] - 1) + size[t] * lower_factor[t]  ∀t
                where M = max(size) * max(upper_factor)
            Else:
                variable[t] ≥ size[t] * lower_factor[t]  ∀t

        Returns:
            variables: {} (no new variables created)
            constraints: {'upper_bound': constraint, 'lower_bound': constraint}
        """
        rel_lower, rel_upper = relative_bounds

        # Upper bound: variable ≤ size * upper_factor
        upper_bound = model.add_constraints(variable <= size_variable * rel_upper, name=upper_bound_name)

        if binary_control is None:
            lower_bound = model.add_constraints(variable >= size_variable * rel_lower, name=lower_bound_name)
        else:
            # Big-M lower bound: variable ≥ M*(binary-1) + size*lower_factor
            big_m = CONFIG.modeling.BIG  #size_variable.max() * rel_upper.max()  # Conservative big-M
            lower_bound = model.add_constraints(
                variable >= big_m * (binary_control - 1) + size_variable * rel_lower, name=lower_bound_name
            )


        variables = {}  # No new variables created
        constraints = {'upper_bound': upper_bound, 'lower_bound': lower_bound}

        return variables, constraints

    @staticmethod
    def consecutive_duration_tracking(
        model: FlowSystemModel,
        name: str,
        state_variable: linopy.Variable,
        minimum_duration: Optional[TemporalData] = None,
        maximum_duration: Optional[TemporalData] = None,
        previous_duration: TemporalData = 0,
    ) -> Tuple[Dict[str, linopy.Variable], Dict[str, linopy.Constraint]]:
        """
        Creates consecutive duration tracking for a binary state variable.

        Mathematical formulation:
            duration[t] ≤ state[t] * M  ∀t
            duration[t+1] ≤ duration[t] + hours_per_step[t]  ∀t
            duration[t+1] ≥ duration[t] + hours_per_step[t] + (state[t+1] - 1) * M  ∀t
            duration[0] = (hours_per_step[0] + previous_duration) * state[0]

            If minimum_duration provided:
                duration[t] ≥ (state[t-1] - state[t]) * minimum_duration[t-1]  ∀t > 0

        Args:
            state_variable: Binary state variable to track duration for
            minimum_duration: Optional minimum consecutive duration
            maximum_duration: Optional maximum consecutive duration
            previous_duration: Duration from before first timestep

        Returns:
            variables: {'duration': duration_var}
            constraints: {'upper_bound': constraint, 'forward': constraint, 'backward': constraint, ...}
        """
        hours_per_step = model.hours_per_step
        mega = hours_per_step.sum('time') + previous_duration  # Big-M value

        # Duration variable
        duration = model.add_variables(
            lower=0,
            upper=maximum_duration if maximum_duration is not None else mega,
            coords=model.get_coords(['time']),
            name=f'{name}|duration',
        )

        constraints = {}

        # Upper bound: duration[t] ≤ state[t] * M
        constraints['upper_bound'] = model.add_constraints(
            duration <= state_variable * mega, name=f'{name}|duration_upper_bound'
        )

        # Forward constraint: duration[t+1] ≤ duration[t] + hours_per_step[t]
        constraints['forward'] = model.add_constraints(
            duration.isel(time=slice(1, None))
            <= duration.isel(time=slice(None, -1)) + hours_per_step.isel(time=slice(None, -1)),
            name=f'{name}|duration_forward',
        )

        # Backward constraint: duration[t+1] ≥ duration[t] + hours_per_step[t] + (state[t+1] - 1) * M
        constraints['backward'] = model.add_constraints(
            duration.isel(time=slice(1, None))
            >= duration.isel(time=slice(None, -1))
            + hours_per_step.isel(time=slice(None, -1))
            + (state_variable.isel(time=slice(1, None)) - 1) * mega,
            name=f'{name}|duration_backward',
        )

        # Initial condition: duration[0] = (hours_per_step[0] + previous_duration) * state[0]
        constraints['initial'] = model.add_constraints(
            duration.isel(time=0)
            == (hours_per_step.isel(time=0) + previous_duration) * state_variable.isel(time=0),
            name=f'{name}|duration_initial',
        )

        # Minimum duration constraint if provided
        if minimum_duration is not None:
            constraints['minimum'] = model.add_constraints(
                duration.isel(time=slice(1, None))
                >= (state_variable.isel(time=slice(None, -1)) - state_variable.isel(time=slice(1, None)))
                * minimum_duration.isel(time=slice(None, -1)),
                name=f'{name}|duration_minimum',
            )

            # Handle initial condition for minimum duration
            if previous_duration > 0 and previous_duration < minimum_duration.isel(time=0).max():
                constraints['initial_minimum'] = model.add_constraints(
                    state_variable.isel(time=0) == 1, name=f'{name}|duration_initial_minimum'
                )

        variables = {'duration': duration}

        return variables, constraints

    @staticmethod
    def mutual_exclusivity_constraint(
        model: FlowSystemModel, name: str, binary_variables: List[linopy.Variable], tolerance: float = 1.1
    ) -> Tuple[Dict, Dict[str, linopy.Constraint]]:
        """
        Creates mutual exclusivity constraint for binary variables.

        Mathematical formulation:
            Σ(binary_vars[i]) ≤ tolerance  ∀t

        Ensures at most one binary variable can be 1 at any time.
        Tolerance > 1.0 accounts for binary variable numerical precision.

        Args:
            binary_variables: List of binary variables that should be mutually exclusive
            tolerance: Upper bound (typically 1.1 for numerical stability)

        Returns:
            variables: {} (no new variables created)
            constraints: {'mutual_exclusivity': constraint}

        Raises:
            AssertionError: If fewer than 2 variables provided or variables aren't binary
        """
        assert len(binary_variables) >= 2, (
            f'Mutual exclusivity requires at least 2 variables, got {len(binary_variables)}'
        )

        for var in binary_variables:
            assert var.attrs.get('binary', False), (
                f'Variable {var.name} must be binary for mutual exclusivity constraint'
            )

        # Create mutual exclusivity constraint
        mutual_exclusivity = model.add_constraints(
            sum(binary_variables) <= tolerance, name=f'{name}|mutual_exclusivity'
        )

        variables = {}  # No new variables created
        constraints = {'mutual_exclusivity': mutual_exclusivity}

        return variables, constraints


class ModelingPatterns:
    """High-level patterns that compose primitives and return (variables, constraints) tuples"""

    @staticmethod
    def investment_sizing_pattern(
        model: FlowSystemModel,
        name: str,
        size_bounds: Tuple[TemporalData, TemporalData],
        controlled_variables: List[linopy.Variable] = None,
        control_factors: List[Tuple[TemporalData, TemporalData]] = None,
        state_variables: List[linopy.Variable] = None,
        optional: bool = False,
    ) -> Tuple[Dict[str, linopy.Variable], Dict[str, linopy.Constraint]]:
        """
        Complete investment sizing pattern with optional binary decision.

        Args:
            model: The model to add the variables to.
            name: The name of the investment variable.
            size_bounds: The minimum and maximum investment size.
            controlled_variables: The variables that are controlled by the investment decision.
            control_factors: The control factors for the controlled variables.
            state_variables: State variable defining the state of the controlled variables.
            optional: Whether the investment decision is optional.

        Returns:
            variables: {'size': size_var, 'is_invested': binary_var (if optional)}
            constraints: {'upper_bound': constraint, 'lower_bound': constraint, ...}
        """
        variables = {}
        constraints = {}

        # Investment size variable
        size_min, size_max = size_bounds
        variables['size'] = model.add_variables(
            lower=0 if optional else size_min,
            upper=size_max,
            name=f'{name}|size',
            coords=model.get_coords(['year', 'scenario']),
        )

        # Optional binary investment decision
        if optional:
            variables['is_invested'] = model.add_variables(
                binary=True, name=f'{name}|is_invested', coords=model.get_coords(['year', 'scenario'])
            )

            # Link size to investment decision
            if abs(size_min - size_max) < 1e-10:  # Fixed size case
                constraints['fixed_size'] = model.add_constraints(
                    variables['size'] == variables['is_invested'] * size_max, name=f'{name}|fixed_size'
                )
            else:  # Variable size case
                constraints['upper_bound'] = model.add_constraints(
                    variables['size'] <= variables['is_invested'] * size_max, name=f'{name}|size|upper_bound'
                )
                constraints['lower_bound'] = model.add_constraints(
                    variables['size'] >= variables['is_invested'] * np.maximum(CONFIG.modeling.EPSILON, size_min),
                    name=f'{name}|size|lower_bound',
                )

        # Control dependent variables
        if controlled_variables and control_factors:
            for i, (var, factors, state_variable) in enumerate(zip(controlled_variables, control_factors, state_variables)):
                upper_bound_name = f'{var.name}|upper_bound'
                lower_bound_name = f'{var.name}|lower_bound'
                _, control_constraints = ModelingPrimitives.big_m_binary_bounds(
                    model=model,
                    variable=var,
                    binary_control=state_variable,
                    size_variable=variables['size'],
                    relative_bounds=factors,
                    upper_bound_name=upper_bound_name,
                    lower_bound_name=lower_bound_name,
                )
                # Flatten control constraints with indexed names
                constraints[upper_bound_name] = control_constraints['upper_bound']
                constraints[lower_bound_name] = control_constraints['lower_bound']

        return variables, constraints

    @staticmethod
    def operational_binary_control_pattern(
        model: FlowSystemModel,
        name: str,
        controlled_variables: List[linopy.Variable],
        variable_bounds: List[Tuple[TemporalData, TemporalData]],
        use_complement: bool = False,
        track_total_duration: bool = False,
        track_switches: bool = False,
        previous_state=0,
        duration_bounds: Tuple[TemporalData, TemporalData] = None,
        track_consecutive_on: bool = False,
        consecutive_on_bounds: Tuple[Optional[TemporalData], Optional[TemporalData]] = (None, None),
        previous_on_duration: TemporalData = 0,
        track_consecutive_off: bool = False,
        consecutive_off_bounds: Tuple[Optional[TemporalData], Optional[TemporalData]] = (None, None),
        previous_off_duration: TemporalData = 0,
    ) -> Tuple[Dict[str, linopy.Variable], Dict[str, linopy.Constraint]]:
        """
        Enhanced operational binary control with consecutive duration tracking.

        New Args:
            track_consecutive_on: Whether to track consecutive on duration
            consecutive_on_bounds: (min_duration, max_duration) for consecutive on
            previous_on_duration: Previous consecutive on duration
            track_consecutive_off: Whether to track consecutive off duration
            consecutive_off_bounds: (min_duration, max_duration) for consecutive off
            previous_off_duration: Previous consecutive off duration
        """
        variables = {}
        constraints = {}

        # Main binary state (existing logic)
        if use_complement:
            state_vars, state_constraints = ModelingPrimitives.binary_state_pair(model, name)
            variables.update(state_vars)
            constraints.update(state_constraints)
        else:
            variables['on'] = model.add_variables(binary=True, name=f'{name}|on', coords=model.get_coords(['time']))

        # Control variables (existing logic)
        for i, (var, (lower_bound, upper_bound)) in enumerate(zip(controlled_variables, variable_bounds)):
            constraints[f'control_{i}_lower'] = model.add_constraints(
                variables['on'] * np.maximum(lower_bound, CONFIG.modeling.EPSILON) <= var, name=f'{name}|control_{i}_lower'
            )
            constraints[f'control_{i}_upper'] = model.add_constraints(
                var <= variables['on'] * upper_bound, name=f'{name}|control_{i}_upper'
            )

        # Total duration tracking (existing logic)
        if track_total_duration:
            duration_expr = (variables['on'] * model.hours_per_step).sum('time')
            duration_vars, duration_constraints = ModelingPrimitives.expression_tracking_variable(
                model, f'{name}|duration', duration_expr, duration_bounds
            )
            variables['total_duration'] = duration_vars['tracker']
            constraints['duration_tracking'] = duration_constraints['tracking']

        # Switch tracking (existing logic)
        if track_switches:
            switch_vars, switch_constraints = ModelingPrimitives.state_transition_variables(
                model, f'{name}|switches', variables['on'], previous_state
            )
            variables.update(switch_vars)
            for switch_name, switch_constraint in switch_constraints.items():
                constraints[f'switch_{switch_name}'] = switch_constraint

        # NEW: Consecutive on duration tracking
        if track_consecutive_on:
            min_on, max_on = consecutive_on_bounds
            consecutive_on_vars, consecutive_on_constraints = ModelingPrimitives.consecutive_duration_tracking(
                model,
                f'{name}|consecutive_on',
                variables['on'],
                minimum_duration=min_on,
                maximum_duration=max_on,
                previous_duration=previous_on_duration,
            )
            variables['consecutive_on_duration'] = consecutive_on_vars['duration']
            for cons_name, cons_constraint in consecutive_on_constraints.items():
                constraints[f'consecutive_on_{cons_name}'] = cons_constraint

        # NEW: Consecutive off duration tracking
        if track_consecutive_off and 'off' in variables:
            min_off, max_off = consecutive_off_bounds
            consecutive_off_vars, consecutive_off_constraints = ModelingPrimitives.consecutive_duration_tracking(
                model,
                f'{name}|consecutive_off',
                variables['off'],
                minimum_duration=min_off,
                maximum_duration=max_off,
                previous_duration=previous_off_duration,
            )
            variables['consecutive_off_duration'] = consecutive_off_vars['duration']
            for cons_name, cons_constraint in consecutive_off_constraints.items():
                constraints[f'consecutive_off_{cons_name}'] = cons_constraint

        return variables, constraints
