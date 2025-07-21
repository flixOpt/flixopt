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
    def expression_tracking_variable(
        model: Model,
        tracked_expression,
        name: str = None,
        short_name: str = None,
        bounds: Tuple[TemporalData, TemporalData] = None,
        coords: List[str] = None,
    ) -> Tuple[linopy.Variable, linopy.Constraint]:
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
            tracker = model.add_variables(name=name, coords=model.get_coords(coords), short_name=short_name)
        else:
            tracker = model.add_variables(
                lower=bounds[0] if bounds[0] is not None else -np.inf,
                upper=bounds[1] if bounds[1] is not None else np.inf,
                name=name,
                coords=model.get_coords(coords),
                short_name=short_name,
            )

        # Constraint: tracker = expression
        tracking = model.add_constraints(tracker == tracked_expression, name=name, short_name=short_name)

        return tracker, tracking

    @staticmethod
    def state_transition_variables(
        model: Union[FlowSystemModel, Model],
        state_variable: linopy.Variable,
        switch_on: linopy.Variable,
        switch_off: linopy.Variable,
        name: str,
        previous_state=0,
    ) -> Tuple[linopy.Constraint, linopy.Constraint, linopy.Constraint]:
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
        # State transition constraints for t > 0
        transition = model.add_constraints(
            switch_on.isel(time=slice(1, None)) - switch_off.isel(time=slice(1, None))
            == state_variable.isel(time=slice(1, None)) - state_variable.isel(time=slice(None, -1)),
            name=f'{name}|transition',
        )

        # Initial state transition for t = 0
        initial = model.add_constraints(
            switch_on.isel(time=0) - switch_off.isel(time=0) == state_variable.isel(time=0) - previous_state,
            name=f'{name}|initial',
        )

        # At most one switch per timestep
        mutex = model.add_constraints(switch_on + switch_off <= 1, name=f'{name}|mutex')

        return transition, initial, mutex

    @staticmethod
    def sum_up_variable(
        model: FlowSystemModel,
        variable_to_count: linopy.Variable,
        name: str = None,
        short_name: str = None,
        bounds: Tuple[NonTemporalData, NonTemporalData] = None,
        factor: TemporalData = 1,
    ) -> Tuple[linopy.Variable, linopy.Constraint]:
        """
        SUms up a variable over time, applying a factor to the variable.

        Args:
            model: The optimization model instance
            variable_to_count: The variable to be summed up
            name: The name of the constraint
            bounds: The bounds of the constraint
            factor: The factor to be applied to the variable
        """
        if bounds is None:
            bounds = (0, np.inf)
        else:
            bounds = (bounds[0] if bounds[0] is not None else 0, bounds[1] if bounds[1] is not None else np.inf)

        count = model.add_variables(
            lower=bounds[0],
            upper=bounds[1],
            coords=model.get_coords(['year', 'scenario']),
            name=name,
        )

        count_constraint = model.add_constraints(count == (variable_to_count * factor).sum('time'), name=name)

        return count, count_constraint

    @staticmethod
    def consecutive_duration_tracking(
        model: FlowSystemModel,
        state_variable: linopy.Variable,
        name: str = None,
        short_name: str = None,
        minimum_duration: Optional[TemporalData] = None,
        maximum_duration: Optional[TemporalData] = None,
        previous_duration: TemporalData = 0,
    ) -> Tuple[linopy.Variable, Tuple[linopy.Constraint, linopy.Constraint, linopy.Constraint]]:
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
            name: Name of the duration variable
            state_variable: Binary state variable to track duration for
            minimum_duration: Optional minimum consecutive duration
            maximum_duration: Optional maximum consecutive duration
            previous_duration: Duration from before first timestep

        Returns:
            variables: {'duration': duration_var}
            constraints: {'ub': constraint, 'forward': constraint, 'backward': constraint, ...}
        """
        hours_per_step = model.hours_per_step
        mega = hours_per_step.sum('time') + previous_duration  # Big-M value

        # Duration variable
        duration = model.add_variables(
            lower=0,
            upper=maximum_duration if maximum_duration is not None else mega,
            coords=model.get_coords(['time']),
            name=name,
            short_name=short_name,
        )

        constraints = {}

        # Upper bound: duration[t] ≤ state[t] * M
        constraints['ub'] = model.add_constraints(
            duration <= state_variable * mega, name=f'{duration.name}|ub'
        )

        # Forward constraint: duration[t+1] ≤ duration[t] + hours_per_step[t]
        constraints['forward'] = model.add_constraints(
            duration.isel(time=slice(1, None))
            <= duration.isel(time=slice(None, -1)) + hours_per_step.isel(time=slice(None, -1)),
            name=f'{duration.name}|forward',
        )

        # Backward constraint: duration[t+1] ≥ duration[t] + hours_per_step[t] + (state[t+1] - 1) * M
        constraints['backward'] = model.add_constraints(
            duration.isel(time=slice(1, None))
            >= duration.isel(time=slice(None, -1))
            + hours_per_step.isel(time=slice(None, -1))
            + (state_variable.isel(time=slice(1, None)) - 1) * mega,
            name=f'{duration.name}|backward',
        )

        # Initial condition: duration[0] = (hours_per_step[0] + previous_duration) * state[0]
        constraints['initial'] = model.add_constraints(
            duration.isel(time=0)
            == (hours_per_step.isel(time=0) + previous_duration) * state_variable.isel(time=0),
            name=f'{duration.name}|initial',
        )

        # Minimum duration constraint if provided
        if minimum_duration is not None:
            constraints['lb'] = model.add_constraints(
                duration
                >= (state_variable.isel(time=slice(None, -1)) - state_variable.isel(time=slice(1, None)))
                * minimum_duration.isel(time=slice(None, -1)),
                name=f'{duration.name}|lb',
            )

            # Handle initial condition for minimum duration
            if previous_duration > 0 and previous_duration < minimum_duration.isel(time=0).max():
                constraints['initial_lb'] = model.add_constraints(
                    state_variable.isel(time=0) == 1, name=f'{duration.name}|initial_lb'
                )

        variables = {'duration': duration}

        return variables, constraints

    @staticmethod
    def mutual_exclusivity_constraint(
        model: FlowSystemModel, name: str, binary_variables: List[linopy.Variable], tolerance: float = 1
    ) -> linopy.Constraint:
        """
        Creates mutual exclusivity constraint for binary variables.

        Mathematical formulation:
            Σ(binary_vars[i]) ≤ tolerance  ∀t

        Ensures at most one binary variable can be 1 at any time.
        Tolerance > 1.0 accounts for binary variable numerical precision.

        Args:
            binary_variables: List of binary variables that should be mutually exclusive
            tolerance: Upper bound

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

        return mutual_exclusivity


class BoundingPatterns:
    """High-level patterns that compose primitives and return (variables, constraints) tuples"""

    @staticmethod
    def basic_bounds(
        model: FlowSystemModel,
        variable: linopy.Variable,
        bounds: Tuple[TemporalData, TemporalData],
        name: str = None,
    ):
        """Create simple bounds.
        variable ∈ [lower_bound, upper_bound]

        Mathematical Formulation:
            lower_bound ≤ variable ≤ upper_bound

        Args:
            model: The optimization model instance
            variable: Variable to be bounded
            bounds: Tuple of (lower_bound, upper_bound) absolute bounds

        Returns:
            Tuple containing:
                - variables (Dict): Empty dict
                - constraints (Dict[str, linopy.Constraint]): 'ub', 'lb'
        """
        lower_bound, upper_bound = bounds
        name = name or f'{variable.name}'

        upper_constraint = model.add_constraints(variable <= upper_bound, name=f'{name}|ub')
        lower_constraint = model.add_constraints(variable >= lower_bound, name=f'{name}|lb')

        return [lower_constraint, upper_constraint]

    @staticmethod
    def bounds_with_state(
        model: FlowSystemModel,
        variable: linopy.Variable,
        bounds: Tuple[TemporalData, TemporalData],
        variable_state: linopy.Variable,
        name: str = None,
    ) -> List[linopy.Constraint]:
        """Constraint a variable to bounds, that can be escaped from to 0 by a binary variable.
        variable ∈ {0, [max(ε, lower_bound), upper_bound]}

        Mathematical Formulation:
            - variable_state * max(ε, lower_bound) ≤ variable ≤ variable_state * upper_bound

        Use Cases:
            - Investment decisions
            - Unit commitment (on/off states)

        Args:
            model: The optimization model instance
            variable: Variable to be bounded
            bounds: Tuple of (lower_bound, upper_bound) absolute bounds
            variable_state: Binary variable controlling the bounds

        Returns:
            Tuple containing:
                - variables (Dict): Empty dict
                - constraints (Dict[str, linopy.Constraint]): 'ub', 'lb'
        """
        lower_bound, upper_bound = bounds
        name = name or f'{variable.name}'

        if np.all(lower_bound - upper_bound) < 1e-10:
            fix_constraint = model.add_constraints(
                variable == variable_state * upper_bound, name=f'{name}|fix'
            )
            return [fix_constraint]

        epsilon = np.maximum(CONFIG.modeling.EPSILON, lower_bound)

        upper_constraint = model.add_constraints(variable <= variable_state * upper_bound, name=f'{name}|ub')
        lower_constraint = model.add_constraints(variable >= variable_state * epsilon, name=f'{name}|lb')

        return [lower_constraint, upper_constraint]

    @staticmethod
    def scaled_bounds(
        model: FlowSystemModel,
        variable: linopy.Variable,
        scaling_variable: linopy.Variable,
        relative_bounds: Tuple[TemporalData, TemporalData],
        name: str = None,
    ) -> List[linopy.Constraint]:
        """Constraint a variable by scaling bounds, dependent on another variable.
        variable ∈ [lower_bound * scaling_variable, upper_bound * scaling_variable]

        Mathematical Formulation:
            scaling_variable * lower_factor ≤ variable ≤ scaling_variable * upper_factor

        Use Cases:
            - Flow rates bounded by equipment capacity
            - Production levels scaled by plant size

        Args:
            model: The optimization model instance
            variable: Variable to be bounded
            scaling_variable: Variable that scales the bound factors
            relative_bounds: Tuple of (lower_factor, upper_factor) relative to scaling variable

        Returns:
            Tuple containing:
                - variables (Dict): Empty dict
                - constraints (Dict[str, linopy.Constraint]): 'ub', 'lb'
        """
        rel_lower, rel_upper = relative_bounds
        name = name or f'{variable.name}'

        if np.abs(rel_lower - rel_upper).all() < 10e-10:
            return [model.add_constraints(variable == scaling_variable * rel_lower, name=f'{name}|fixed')]

        upper_constraint = model.add_constraints(variable <= scaling_variable * rel_upper, name=f'{name}|ub')
        lower_constraint = model.add_constraints(variable >= scaling_variable * rel_lower, name=f'{name}|lb')

        return [lower_constraint, upper_constraint]

    @staticmethod
    def scaled_bounds_with_state(
        model: FlowSystemModel,
        variable: linopy.Variable,
        scaling_variable: linopy.Variable,
        relative_bounds: Tuple[TemporalData, TemporalData],
        scaling_bounds: Tuple[TemporalData, TemporalData],
        variable_state: linopy.Variable,
        name: str = None,
    ) -> List[linopy.Constraint]:
        """Constraint a variable by scaling bounds with binary state control.

        variable ∈ {0, [max(ε, lower_relative_bound) * scaling_variable, upper_relative_bound * scaling_variable]}

        Mathematical Formulation (Big-M):
            (variable_state - 1) * M_misc + scaling_variable * rel_lower ≤ variable ≤ scaling_variable * rel_upper
            variable_state * big_m_lower ≤ variable ≤ variable_state * big_m_upper

        Where:
            M_misc = scaling_max * rel_lower
            big_m_upper = scaling_max * rel_upper
            big_m_lower = max(ε, scaling_min * rel_lower)

        Args:
            model: The optimization model instance
            variable: Variable to be bounded
            scaling_variable: Variable that scales the bound factors
            relative_bounds: Tuple of (lower_factor, upper_factor) relative to scaling variable
            scaling_bounds: Tuple of (scaling_min, scaling_max) bounds of the scaling variable
            variable_state: Binary variable for on/off control
            name: Optional name prefix for constraints

        Returns:
            List[linopy.Constraint]: List of constraint objects
        """
        rel_lower, rel_upper = relative_bounds
        scaling_min, scaling_max = scaling_bounds
        name = name or f'{variable.name}'

        big_m_misc = scaling_max * rel_lower

        scaling_lower = model.add_constraints(
            variable >= (variable_state - 1) * big_m_misc + scaling_variable * rel_lower, name=f'{name}|lb2'
        )
        scaling_upper = model.add_constraints(
            variable <= scaling_variable * rel_upper, name=f'{name}|ub2'
        )

        big_m_upper = scaling_max * rel_upper
        big_m_lower = np.maximum(CONFIG.modeling.EPSILON, scaling_min * rel_lower)

        binary_upper = model.add_constraints(variable_state * big_m_upper >= variable, name=f'{name}|ub1')
        binary_lower = model.add_constraints(variable_state * big_m_lower <= variable, name=f'{name}|lb1')

        return [scaling_lower, scaling_upper, binary_lower, binary_upper]

    @staticmethod
    def auto_bounds(
        model: FlowSystemModel,
        variable: linopy.Variable,
        bounds: Tuple[TemporalData, TemporalData],
        scaling_variable: linopy.Variable = None,
        scaling_state: linopy.Variable = None,
        scaling_bounds: Tuple[TemporalData, TemporalData] = None,
        variable_state: linopy.Variable = None,
    ) -> List[linopy.Constraint]:
        """Automatically select the appropriate bounds method.

        Parameter Combinations:
        1. Only bounds → basic_bounds()
        2. bounds + scaling_variable → scaled_bounds()
        3. bounds + variable_state → bounds_with_state()
        4. bounds + scaling_variable + variable_state → binary_scaled_bounds()
        5. bounds + scaling_variable + scaling_state + variable_state → scaled_bounds_with_state_on_both_scaling_and_variable()

        Args:
            model: The optimization model instance
            variable: Variable to be bounded
            bounds: Tuple of (lower, upper) bounds or relative factors
            scaling_variable: Optional variable to scale bounds by
            scaling_state: Optional binary variable for scaling_variable state
            scaling_bounds: Required for cases 4,5 - bounds of scaling variable
            variable_state: Optional binary variable for variable state

        Returns:
            Tuple from the selected method

        Raises:
            ValueError: If required parameters are missing
        """
        # Case 5: Dual binary control
        if scaling_variable is not None and scaling_state is not None and variable_state is not None:
            if scaling_bounds is None:
                raise ValueError('scaling_bounds is required for dual binary control')
            return BoundingPatterns.scaled_bounds_with_state_on_both_scaling_and_variable(
                model=model,
                variable=variable,
                scaling_variable=scaling_variable,
                relative_bounds=bounds,
                scaling_state=scaling_state,
                variable_state=variable_state,
                scaling_bounds=scaling_bounds,
            )

        # Case 4: Binary scaled bounds
        if scaling_variable is not None and variable_state is not None:
            if scaling_bounds is None:
                raise ValueError('scaling_bounds is required for binary scaled bounds')
            return BoundingPatterns.binary_scaled_bounds(
                model=model,
                variable=variable,
                scaling_variable=scaling_variable,
                relative_bounds=bounds,
                variable_state=variable_state,
                scaling_bounds=scaling_bounds,
            )

        # Case 3: Binary controlled bounds
        if variable_state is not None and scaling_variable is None:
            return BoundingPatterns.bounds_with_state(
                model=model,
                variable=variable,
                bounds=bounds,
                variable_state=variable_state,
            )

        # Case 2: Scaled bounds
        if scaling_variable is not None and variable_state is None:
            return BoundingPatterns.scaled_bounds(
                model=model,
                variable=variable,
                scaling_variable=scaling_variable,
                relative_bounds=bounds,
            )

        # Case 1: Basic bounds
        if scaling_variable is None and variable_state is None:
            return BoundingPatterns.basic_bounds(model, variable, bounds)

        raise ValueError('Invalid combination of arguments')
