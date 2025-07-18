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
        model: FlowSystemModel, name: str, coords: List[str] = None, use_complement: bool = True
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
        if use_complement:
            off = model.add_variables(binary=True, name=f'{name}|off', coords=model.get_coords(coords))

            # Constraint: on + off = 1
            complementary = model.add_constraints(on + off == 1, name=f'{name}|complementary')

            variables = {'on': on, 'off': off}
            constraints = {'complementary': complementary}
            return variables, constraints
        return {'on': on}, {}

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
            constraints: {'lb': constraint, 'ub': constraint}
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
        constraints = {'lb': lower_bound, 'ub': upper_bound}

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
            constraints: {'ub': constraint, 'forward': constraint, 'backward': constraint, ...}
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
        constraints['ub'] = model.add_constraints(
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


class BoundingPatterns:
    """High-level patterns that compose primitives and return (variables, constraints) tuples"""

    @staticmethod
    def basic_bounds(
        model: FlowSystemModel,
        variable: linopy.Variable,
        bounds: Tuple[TemporalData, TemporalData],
    ):
        """Create simple bounds.

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

        upper_constraint = model.add_constraints(variable <= upper_bound, name=f'{variable.name}|ub')
        lower_constraint = model.add_constraints(variable >= lower_bound, name=f'{variable.name}|lb')

        return {}, {'ub': upper_constraint, 'lb': lower_constraint}

    @staticmethod
    def binary_controlled_bounds(
        model: FlowSystemModel,
        variable: linopy.Variable,
        bounds: Tuple[TemporalData, TemporalData],
        variable_state: linopy.Variable,
    ) -> Tuple[Dict, Dict[str, linopy.Constraint]]:
        """Create bounds controlled by a binary variable.

        Mathematical Formulation:
            variable_state * max(ε, lower_bound) ≤ variable ≤ variable_state * upper_bound

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

        if np.all(lower_bound - upper_bound) < 1e-10:
            fix_constraint = model.add_constraints(
                variable == variable_state * upper_bound, name=f'{variable.name}|fixed_size'
            )
            return {}, {'ub': fix_constraint, 'lb': fix_constraint}

        epsilon = np.maximum(CONFIG.modeling.EPSILON, lower_bound)

        upper_constraint = model.add_constraints(variable <= variable_state * upper_bound, name=f'{variable.name}|ub')
        lower_constraint = model.add_constraints(variable >= variable_state * epsilon, name=f'{variable.name}|lb')

        return {}, {'ub': upper_constraint, 'lb': lower_constraint}

    @staticmethod
    def scaled_bounds(
        model: FlowSystemModel,
        variable: linopy.Variable,
        scaling_variable: linopy.Variable,
        relative_bounds: Tuple[TemporalData, TemporalData],
    ) -> Tuple[Dict, Dict[str, linopy.Constraint]]:
        """Create bounds scaled by another variable.

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

        upper_constraint = model.add_constraints(variable <= scaling_variable * rel_upper, name=f'{variable.name}|ub')
        lower_constraint = model.add_constraints(variable >= scaling_variable * rel_lower, name=f'{variable.name}|lb')

        variables = {}
        constraints = {'ub': upper_constraint, 'lb': lower_constraint}
        return variables, constraints

    @staticmethod
    def binary_scaled_bounds(
        model: FlowSystemModel,
        variable: linopy.Variable,
        scaling_variable: linopy.Variable,
        relative_bounds: Tuple[TemporalData, TemporalData],
        variable_state: linopy.Variable,
        scaling_bounds: Tuple[TemporalData, TemporalData],
    ) -> Tuple[Dict, Dict[str, linopy.Constraint]]:
        """Create scaled bounds controlled by a binary variable.

        Mathematical Formulation (Big-M):
            scaling_variable * lower_factor ≤ variable ≤ scaling_variable * upper_factor
            variable ≤ variable_state * M_upper
            variable ≥ variable_state * M_lower

        Where: M_upper = scaling_max * upper_factor, M_lower = max(ε, scaling_min * lower_factor)

        Use Cases:
            - Equipment with capacity and on/off control
            - Variable-size units with operational states

        Args:
            model: The optimization model instance
            variable: Variable to be bounded
            scaling_variable: Variable that scales the bound factors
            relative_bounds: Tuple of (lower_factor, upper_factor) relative to scaling variable
            variable_state: Binary variable for on/off control
            scaling_bounds: Tuple of (scaling_min, scaling_max) bounds of the scaling variable

        Returns:
            Tuple containing:
                - variables (Dict): Empty dict
                - constraints (Dict[str, linopy.Constraint]): 'ub', 'lb', 'binary_upper', 'binary_lower'
        """
        rel_lower, rel_upper = relative_bounds
        scaling_min, scaling_max = scaling_bounds

        big_m_upper = scaling_max * rel_upper
        big_m_lower = np.maximum(CONFIG.modeling.EPSILON, scaling_min * rel_lower)

        scaling_upper = model.add_constraints(
            variable <= scaling_variable * rel_upper, name=f'{variable.name}|scaling_ub'
        )
        binary_upper = model.add_constraints(variable <= variable_state * big_m_upper, name=f'{variable.name}|ub')

        scaling_lower = model.add_constraints(
            variable >= scaling_variable * rel_lower, name=f'{variable.name}|scaling_lb'
        )
        binary_lower = model.add_constraints(variable >= variable_state * big_m_lower, name=f'{variable.name}|lb')

        variables = {}
        constraints = {
            'ub': scaling_upper,
            'lb': scaling_lower,
            'binary_upper': binary_upper,
            'binary_lower': binary_lower,
        }
        return variables, constraints

    @staticmethod
    def dual_binary_scaled_bounds(
        model: FlowSystemModel,
        variable: linopy.Variable,
        scaling_variable: linopy.Variable,
        relative_bounds: Tuple[TemporalData, TemporalData],
        scaling_state: linopy.Variable,
        variable_state: linopy.Variable,
        scaling_bounds: Tuple[TemporalData, TemporalData],
    ) -> Tuple[Dict, Dict[str, linopy.Constraint]]:
        """Create bounds with dual binary control over a scaled variable.

        Mathematical Formulation:
            scaling_state * max(ε, scaling_min) ≤ scaling_variable ≤ scaling_state * scaling_max
            variable_state * max(ε, rel_lower * scaling_min) ≤ variable ≤ variable_state * M
            M * (variable_state - 1) + scaling_variable * rel_lower ≤ variable ≤ scaling_variable * rel_upper

        Where: M = rel_upper * scaling_max

        Use Cases:
            - Investment + operational control (capacity sizing + on/off dispatch)
            - Equipment sizing + utilization

        Args:
            model: The optimization model instance
            variable: Variable to be bounded
            scaling_variable: Variable that scales the bounds
            relative_bounds: Tuple of (rel_lower, rel_upper) relative bound multipliers
            scaling_state: Binary controlling scaling_variable existence
            variable_state: Binary controlling variable operation
            scaling_bounds: Tuple of (scaling_min, scaling_max) bounds for scaling_variable

        Returns:
            Tuple containing:
                - variables (Dict): Empty dict
                - constraints (Dict[str, linopy.Constraint]): Multiple constraint keys
        """
        rel_lower, rel_upper = relative_bounds
        scaling_min, scaling_max = scaling_bounds

        big_m = rel_upper * scaling_max

        # 1. SCALING VARIABLE CONSTRAINTS
        epsilon_scaling = np.maximum(CONFIG.modeling.EPSILON, scaling_min)

        scaling_ub = model.add_constraints(
            scaling_variable <= scaling_state * scaling_max, name=f'{scaling_variable.name}|ub'
        )

        scaling_lb = model.add_constraints(
            scaling_variable >= scaling_state * epsilon_scaling, name=f'{scaling_variable.name}|lb'
        )

        # 2. VARIABLE STATE CONSTRAINTS
        epsilon_variable = np.maximum(CONFIG.modeling.EPSILON, rel_lower * scaling_min)

        variable_ub = model.add_constraints(variable <= variable_state * big_m, name=f'{variable.name}|ub')

        variable_lb = model.add_constraints(variable >= variable_state * epsilon_variable, name=f'{variable.name}|lb')

        # 3. SCALING-DEPENDENT CONSTRAINTS
        scaling_variable_ub = model.add_constraints(
            variable <= scaling_variable * rel_upper, name=f'{variable.name}|scaling_ub'
        )

        scaling_variable_lb = model.add_constraints(
            big_m * (variable_state - 1) + scaling_variable * rel_lower <= variable,
            name=f'{variable.name}|scaling_lb',
        )

        variables = {}
        constraints = {
            'scaling_ub': scaling_ub,
            'scaling_lb': scaling_lb,
            'variable_ub': variable_ub,
            'variable_lb': variable_lb,
            'scaling_variable_ub': scaling_variable_ub,
            'scaling_variable_lb': scaling_variable_lb,
        }

        return variables, constraints

    @staticmethod
    def auto_bounds(
        model: FlowSystemModel,
        variable: linopy.Variable,
        bounds: Tuple[TemporalData, TemporalData],
        scaling_variable: linopy.Variable = None,
        scaling_state: linopy.Variable = None,
        scaling_bounds: Tuple[TemporalData, TemporalData] = None,
        variable_state: linopy.Variable = None,
    ) -> Tuple[Dict, Dict[str, linopy.Constraint]]:
        """Automatically select the appropriate bounds method.

        Parameter Combinations:
        1. Only bounds → basic_bounds()
        2. bounds + scaling_variable → scaled_bounds()
        3. bounds + variable_state → binary_controlled_bounds()
        4. bounds + scaling_variable + variable_state → binary_scaled_bounds()
        5. bounds + scaling_variable + scaling_state + variable_state → dual_binary_scaled_bounds()

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
            return BoundingPatterns.dual_binary_scaled_bounds(
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
            return BoundingPatterns.binary_controlled_bounds(
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


class ModelingPatterns:
    """High-level patterns that compose primitives and return (variables, constraints) tuples"""

    @staticmethod
    def investment_sizing_pattern(
        model: FlowSystemModel,
        name: str,
        size_bounds: Tuple[TemporalData, TemporalData],
        controlled_variable: linopy.Variable,
        control_factors: Tuple[TemporalData, TemporalData],
        state_variable: List[linopy.Variable] = None,
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
            constraints: {'ub': constraint, 'lb': constraint, ...}
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

        _, new_cons = BoundingPatterns.auto_bounds(
            model=model,
            variable=controlled_variable,
            bounds=control_factors,
            upper_bound_name=f'{controlled_variable.name}|ub',
            lower_bound_name=f'{controlled_variable.name}|lb',
            scaling_variable=variables['size'],
            binary_control=variables['is_invested'] if optional else None,
            scaling_bounds=(size_min, size_max),
            constraint_name_prefix=name,
        )

        constraints.update(new_cons)

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
        Enhanced operational binary control using composable patterns.
        """
        variables = {}
        constraints = {}

        # 1. Main binary state using existing pattern
        if use_complement:
            state_vars, state_constraints = ModelingPrimitives.binary_state_pair(model, name)
            variables.update(state_vars)
            constraints.update(state_constraints)
        else:
            variables['on'] = model.add_variables(binary=True, name=f'{name}|on', coords=model.get_coords(['time']))

        # 2. Control variables - use big_m_binary_bounds pattern for consistency
        for i, (var, (lower_bound, upper_bound)) in enumerate(zip(controlled_variables, variable_bounds)):
            # Use the big_m pattern but without binary control (None)
            _, control_constraints = BoundingPatterns.big_m_binary_bounds(
                model=model,
                variable=var,
                binary_control=variables['on'],  # The on state controls the variables
                size_variable=1,  # No size scaling, just on/off
                relative_bounds=(lower_bound, upper_bound),
                upper_bound_name=f'{name}|control_{i}_upper',
                lower_bound_name=f'{name}|control_{i}_lower',
            )
            constraints[f'control_{i}_upper'] = control_constraints['ub']
            constraints[f'control_{i}_lower'] = control_constraints['lb']

        # 3. Total duration tracking using existing pattern
        if track_total_duration:
            duration_expr = (variables['on'] * model.hours_per_step).sum('time')
            duration_vars, duration_constraints = ModelingPrimitives.expression_tracking_variable(
                model, f'{name}|on_hours_total', duration_expr, duration_bounds
            )
            variables['total_duration'] = duration_vars['tracker']
            constraints['duration_tracking'] = duration_constraints['tracking']

        # 4. Switch tracking using existing pattern
        if track_switches:
            switch_vars, switch_constraints = ModelingPrimitives.state_transition_variables(
                model, f'{name}|switches', variables['on'], previous_state
            )
            variables.update(switch_vars)
            for switch_name, switch_constraint in switch_constraints.items():
                constraints[f'switch_{switch_name}'] = switch_constraint

        # 5. Consecutive on duration using existing pattern
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

        # 6. Consecutive off duration using existing pattern
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
