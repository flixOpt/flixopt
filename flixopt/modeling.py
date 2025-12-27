import logging

import linopy
import numpy as np
import xarray as xr

from .config import CONFIG
from .structure import Submodel

logger = logging.getLogger('flixopt')


class ModelingUtilitiesAbstract:
    """Utility functions for modeling - leveraging xarray for temporal data"""

    @staticmethod
    def to_binary(
        values: xr.DataArray,
        epsilon: float | None = None,
        dims: str | list[str] | None = None,
    ) -> xr.DataArray:
        """
        Converts a DataArray to binary {0, 1} values.

        Args:
            values: Input DataArray to convert to binary
            epsilon: Tolerance for zero detection (uses CONFIG.Modeling.epsilon if None)
            dims: Dims to keep. Other dimensions are collapsed using .any() -> If any value is 1, all are 1.

        Returns:
            Binary DataArray with same shape (or collapsed if collapse_non_time=True)
        """
        if not isinstance(values, xr.DataArray):
            values = xr.DataArray(values, dims=['time'], coords={'time': range(len(values))})

        if epsilon is None:
            epsilon = CONFIG.Modeling.epsilon

        if values.size == 0:
            return xr.DataArray(0) if values.item() < epsilon else xr.DataArray(1)

        # Convert to binary states
        binary_states = np.abs(values) >= epsilon

        # Optionally collapse dimensions using .any()
        if dims is not None:
            dims = [dims] if isinstance(dims, str) else dims

            binary_states = binary_states.any(dim=[d for d in binary_states.dims if d not in dims])

        return binary_states.astype(int)

    @staticmethod
    def count_consecutive_states(
        binary_values: xr.DataArray | np.ndarray | list[int, float],
        dim: str = 'time',
        epsilon: float | None = None,
    ) -> float:
        """Count consecutive steps in the final active state of a binary time series.

        This function counts how many consecutive time steps the series remains "on"
        (non-zero) at the end of the time series. If the final state is "inactive", returns 0.

        Args:
            binary_values: Binary DataArray with values close to 0 (inactive) or 1 (active).
            dim: Dimension along which to count consecutive states.
            epsilon: Tolerance for zero detection. Uses CONFIG.Modeling.epsilon if None.

        Returns:
            Sum of values in the final consecutive "active" period. Returns 0.0 if the
            final state is "inactive".

        Examples:
            >>> arr = xr.DataArray([0, 0, 1, 1, 1, 0, 1, 1], dims=['time'])
            >>> ModelingUtilitiesAbstract.count_consecutive_states(arr)
            2.0

            >>> arr = [0, 0, 1, 0, 1, 1, 1, 1]
            >>> ModelingUtilitiesAbstract.count_consecutive_states(arr)
            4.0
        """
        epsilon = epsilon or CONFIG.Modeling.epsilon

        if isinstance(binary_values, xr.DataArray):
            # xarray path
            other_dims = [d for d in binary_values.dims if d != dim]
            if other_dims:
                binary_values = binary_values.any(dim=other_dims)
            arr = binary_values.values
        else:
            # numpy/array-like path
            arr = np.asarray(binary_values)

        # Flatten to 1D if needed
        arr = arr.ravel() if arr.ndim > 1 else arr

        # Handle edge cases
        if arr.size == 0:
            return 0.0
        if arr.size == 1:
            return float(arr[0]) if not np.isclose(arr[0], 0, atol=epsilon) else 0.0

        # Return 0 if final state is inactive
        if np.isclose(arr[-1], 0, atol=epsilon):
            return 0.0

        # Find the last zero position (treat NaNs as inactive)
        arr = np.nan_to_num(arr, nan=0.0)
        is_zero = np.isclose(arr, 0, atol=epsilon)
        zero_indices = np.where(is_zero)[0]

        # Calculate sum from last zero to end
        start_idx = zero_indices[-1] + 1 if zero_indices.size > 0 else 0

        return float(np.sum(arr[start_idx:]))


class ModelingUtilities:
    @staticmethod
    def compute_consecutive_hours_in_state(
        binary_values: xr.DataArray,
        hours_per_timestep: int | float,
        epsilon: float = None,
    ) -> float:
        """
        Computes the final consecutive duration in state 'active' (=1) in hours.

        Args:
            binary_values: Binary DataArray with 'time' dim, or scalar/array
            hours_per_timestep: Duration of each timestep in hours
            epsilon: Tolerance for zero detection (uses CONFIG.Modeling.epsilon if None)

        Returns:
            The duration of the final consecutive 'active' period in hours
        """
        if not isinstance(hours_per_timestep, (int, float)):
            raise TypeError(f'hours_per_timestep must be a scalar, got {type(hours_per_timestep)}')

        return (
            ModelingUtilitiesAbstract.count_consecutive_states(binary_values=binary_values, epsilon=epsilon)
            * hours_per_timestep
        )

    @staticmethod
    def compute_previous_states(previous_values: xr.DataArray | None, epsilon: float | None = None) -> xr.DataArray:
        return ModelingUtilitiesAbstract.to_binary(values=previous_values, epsilon=epsilon, dims='time')

    @staticmethod
    def compute_previous_on_duration(
        previous_values: xr.DataArray, hours_per_step: xr.DataArray | float | int
    ) -> float:
        return (
            ModelingUtilitiesAbstract.count_consecutive_states(ModelingUtilitiesAbstract.to_binary(previous_values))
            * hours_per_step
        )

    @staticmethod
    def compute_previous_off_duration(
        previous_values: xr.DataArray, hours_per_step: xr.DataArray | float | int
    ) -> float:
        """
        Compute previous consecutive 'inactive' duration.

        Args:
            previous_values: DataArray with 'time' dimension
            hours_per_step: Duration of each timestep in hours

        Returns:
            Previous consecutive inactive duration in hours
        """
        if previous_values is None or previous_values.size == 0:
            return 0.0

        previous_states = ModelingUtilities.compute_previous_states(previous_values)
        previous_off_states = 1 - previous_states
        return ModelingUtilities.compute_consecutive_hours_in_state(previous_off_states, hours_per_step)

    @staticmethod
    def get_most_recent_state(previous_values: xr.DataArray | None) -> int:
        """
        Get the most recent binary state from previous values.

        Args:
            previous_values: DataArray with 'time' dimension

        Returns:
            Most recent binary state (0 or 1)
        """
        if previous_values is None or previous_values.size == 0:
            return 0

        previous_states = ModelingUtilities.compute_previous_states(previous_values)
        return int(previous_states.isel(time=-1).item())


class ModelingPrimitives:
    """Mathematical modeling primitives returning (variables, constraints) tuples"""

    @staticmethod
    def expression_tracking_variable(
        model: Submodel,
        tracked_expression: linopy.expressions.LinearExpression | linopy.Variable,
        name: str = None,
        short_name: str = None,
        bounds: tuple[xr.DataArray, xr.DataArray] = None,
        coords: str | list[str] | None = None,
    ) -> tuple[linopy.Variable, linopy.Constraint]:
        """Creates a variable constrained to equal a given expression.

        Mathematical formulation:
            tracker = expression
            lower ≤ tracker ≤ upper  (if bounds provided)

        Args:
            model: The submodel to add variables and constraints to
            tracked_expression: Expression that the tracker variable must equal
            name: Full name for the variable and constraint
            short_name: Short name for display purposes
            bounds: Optional (lower_bound, upper_bound) tuple for the tracker variable
            coords: Coordinate dimensions for the variable (None uses all model coords)

        Returns:
            Tuple of (tracker_variable, tracking_constraint)
        """
        if not isinstance(model, Submodel):
            raise ValueError('ModelingPrimitives.expression_tracking_variable() can only be used with a Submodel')

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
    def consecutive_duration_tracking(
        model: Submodel,
        state: linopy.Variable,
        name: str = None,
        short_name: str = None,
        minimum_duration: xr.DataArray | None = None,
        maximum_duration: xr.DataArray | None = None,
        duration_dim: str = 'time',
        duration_per_step: int | float | xr.DataArray = None,
        previous_duration: xr.DataArray | None = None,
    ) -> tuple[dict[str, linopy.Variable], dict[str, linopy.Constraint]]:
        """Creates consecutive duration tracking for a binary state variable.

        Tracks how long a binary state has been continuously active (=1).
        Duration resets to 0 when state becomes inactive (=0).

        Mathematical formulation:
            duration[t] ≤ state[t] · M  ∀t
            duration[t+1] ≤ duration[t] + duration_per_step[t]  ∀t
            duration[t+1] ≥ duration[t] + duration_per_step[t] + (state[t+1] - 1) · M  ∀t
            duration[0] = (duration_per_step[0] + previous_duration) · state[0]  (if previous_duration is not None)

            If minimum_duration provided:
                duration[t] ≥ (state[t-1] - state[t]) · minimum_duration[t-1]  ∀t > 0

        Where M is a big-M value (sum of all duration_per_step + previous_duration).

        Args:
            model: The submodel to add variables and constraints to
            state: Binary state variable (1=active, 0=inactive) to track duration for
            name: Full name for the duration variable
            short_name: Short name for display purposes
            minimum_duration: Optional minimum consecutive duration (enforced at state transitions)
            maximum_duration: Optional maximum consecutive duration (upper bound on duration variable)
            duration_dim: Dimension name to track duration along (default 'time')
            duration_per_step: Time increment per step in duration_dim
            previous_duration: Initial duration value before first timestep. If None (default),
                the initial constraint is skipped ("relaxed" mode) - duration at t=0 is unconstrained.

        Returns:
            Tuple of (duration_variable, constraints_dict)
            where constraints_dict contains: 'ub', 'forward', 'backward', and optionally 'initial', 'lb', 'initial_lb'
        """
        if not isinstance(model, Submodel):
            raise ValueError('ModelingPrimitives.consecutive_duration_tracking() can only be used with a Submodel')

        # Big-M value (use 0 for previous_duration if None/relaxed mode)
        prev_for_mega = previous_duration if previous_duration is not None else 0
        mega = duration_per_step.sum(duration_dim) + prev_for_mega

        # Duration variable
        duration = model.add_variables(
            lower=0,
            upper=maximum_duration if maximum_duration is not None else mega,
            coords=state.coords,
            name=name,
            short_name=short_name,
        )

        constraints = {}

        # Upper bound: duration[t] ≤ state[t] * M
        constraints['ub'] = model.add_constraints(duration <= state * mega, name=f'{duration.name}|ub')

        # Forward constraint: duration[t+1] ≤ duration[t] + duration_per_step[t]
        constraints['forward'] = model.add_constraints(
            duration.isel({duration_dim: slice(1, None)})
            <= duration.isel({duration_dim: slice(None, -1)}) + duration_per_step.isel({duration_dim: slice(None, -1)}),
            name=f'{duration.name}|forward',
        )

        # Backward constraint: duration[t+1] ≥ duration[t] + duration_per_step[t] + (state[t+1] - 1) * M
        constraints['backward'] = model.add_constraints(
            duration.isel({duration_dim: slice(1, None)})
            >= duration.isel({duration_dim: slice(None, -1)})
            + duration_per_step.isel({duration_dim: slice(None, -1)})
            + (state.isel({duration_dim: slice(1, None)}) - 1) * mega,
            name=f'{duration.name}|backward',
        )

        # Initial condition (skip if previous_duration is None = relaxed mode)
        if previous_duration is not None:
            # duration[0] = (duration_per_step[0] + previous_duration) * state[0]
            constraints['initial'] = model.add_constraints(
                duration.isel({duration_dim: 0})
                == (duration_per_step.isel({duration_dim: 0}) + previous_duration) * state.isel({duration_dim: 0}),
                name=f'{duration.name}|initial',
            )

        # Minimum duration constraint if provided
        if minimum_duration is not None:
            constraints['lb'] = model.add_constraints(
                duration
                >= (state.isel({duration_dim: slice(None, -1)}) - state.isel({duration_dim: slice(1, None)}))
                * minimum_duration.isel({duration_dim: slice(None, -1)}),
                name=f'{duration.name}|lb',
            )

            # Handle initial condition for minimum duration (only if not relaxed mode)
            if previous_duration is not None:
                prev = (
                    float(previous_duration)
                    if not isinstance(previous_duration, xr.DataArray)
                    else float(previous_duration.max().item())
                )
                min0 = float(minimum_duration.isel({duration_dim: 0}).max().item())
                if prev > 0 and prev < min0:
                    constraints['initial_lb'] = model.add_constraints(
                        state.isel({duration_dim: 0}) == 1, name=f'{duration.name}|initial_lb'
                    )

        variables = {'duration': duration}

        return variables, constraints

    @staticmethod
    def mutual_exclusivity_constraint(
        model: Submodel,
        binary_variables: list[linopy.Variable],
        tolerance: float = 1,
        short_name: str = 'mutual_exclusivity',
    ) -> linopy.Constraint:
        """Creates mutual exclusivity constraint for binary variables.

        Ensures at most one binary variable can be active (=1) at any time.

        Mathematical formulation:
            Σᵢ binary_vars[i] ≤ tolerance  ∀t

        Args:
            model: The submodel to add the constraint to
            binary_variables: List of binary variables that should be mutually exclusive
            tolerance: Upper bound on the sum (default 1, allows slight numerical tolerance)
            short_name: Short name for the constraint

        Returns:
            Mutual exclusivity constraint

        Raises:
            AssertionError: If fewer than 2 variables provided or variables aren't binary
        """
        if not isinstance(model, Submodel):
            raise ValueError('ModelingPrimitives.mutual_exclusivity_constraint() can only be used with a Submodel')

        assert len(binary_variables) >= 2, (
            f'Mutual exclusivity requires at least 2 variables, got {len(binary_variables)}'
        )

        for var in binary_variables:
            assert var.attrs.get('binary', False), (
                f'Variable {var.name} must be binary for mutual exclusivity constraint'
            )

        # Create mutual exclusivity constraint
        mutual_exclusivity = model.add_constraints(sum(binary_variables) <= tolerance, short_name=short_name)

        return mutual_exclusivity


class BoundingPatterns:
    """High-level patterns that compose primitives and return (variables, constraints) tuples"""

    @staticmethod
    def basic_bounds(
        model: Submodel,
        variable: linopy.Variable,
        bounds: tuple[xr.DataArray, xr.DataArray],
        name: str = None,
    ) -> list[linopy.constraints.Constraint]:
        """Creates simple lower and upper bounds for a variable.

        Mathematical formulation:
            lower_bound ≤ variable ≤ upper_bound

        Args:
            model: The submodel to add constraints to
            variable: Variable to be bounded
            bounds: Tuple of (lower_bound, upper_bound) absolute bounds
            name: Optional name prefix for constraints

        Returns:
            List of [lower_constraint, upper_constraint]
        """
        if not isinstance(model, Submodel):
            raise ValueError('BoundingPatterns.basic_bounds() can only be used with a Submodel')

        lower_bound, upper_bound = bounds
        name = name or f'{variable.name}'

        upper_constraint = model.add_constraints(variable <= upper_bound, name=f'{name}|ub')
        lower_constraint = model.add_constraints(variable >= lower_bound, name=f'{name}|lb')

        return [lower_constraint, upper_constraint]

    @staticmethod
    def bounds_with_state(
        model: Submodel,
        variable: linopy.Variable,
        bounds: tuple[xr.DataArray, xr.DataArray],
        state: linopy.Variable,
        name: str = None,
    ) -> list[linopy.Constraint]:
        """Creates bounds controlled by a binary state variable.

        Variable is forced to 0 when state=0, bounded when state=1.

        Mathematical formulation:
            state · max(ε, lower_bound) ≤ variable ≤ state · upper_bound

        Where ε is a small positive number (CONFIG.Modeling.epsilon) ensuring
        numerical stability when lower_bound is 0.

        Args:
            model: The submodel to add constraints to
            variable: Variable to be bounded
            bounds: Tuple of (lower_bound, upper_bound) absolute bounds when state=1
            state: Binary variable (0=force variable to 0, 1=allow bounds)
            name: Optional name prefix for constraints

        Returns:
            List of [lower_constraint, upper_constraint] (or [fix_constraint] if lower=upper)
        """
        if not isinstance(model, Submodel):
            raise ValueError('BoundingPatterns.bounds_with_state() can only be used with a Submodel')

        lower_bound, upper_bound = bounds
        name = name or f'{variable.name}'

        if np.allclose(lower_bound, upper_bound, atol=1e-10, equal_nan=True):
            fix_constraint = model.add_constraints(variable == state * upper_bound, name=f'{name}|fix')
            return [fix_constraint]

        epsilon = np.maximum(CONFIG.Modeling.epsilon, lower_bound)

        upper_constraint = model.add_constraints(variable <= state * upper_bound, name=f'{name}|ub')
        lower_constraint = model.add_constraints(variable >= state * epsilon, name=f'{name}|lb')

        return [lower_constraint, upper_constraint]

    @staticmethod
    def scaled_bounds(
        model: Submodel,
        variable: linopy.Variable,
        scaling_variable: linopy.Variable,
        relative_bounds: tuple[xr.DataArray, xr.DataArray],
        name: str = None,
    ) -> list[linopy.Constraint]:
        """Creates bounds scaled by another variable.

        Variable is bounded relative to a scaling variable (e.g., flow rate relative to size).

        Mathematical formulation:
            scaling_variable · lower_factor ≤ variable ≤ scaling_variable · upper_factor

        Args:
            model: The submodel to add constraints to
            variable: Variable to be bounded
            scaling_variable: Variable that scales the bound factors (e.g., equipment size)
            relative_bounds: Tuple of (lower_factor, upper_factor) relative to scaling_variable
            name: Optional name prefix for constraints

        Returns:
            List of [lower_constraint, upper_constraint] (or [fix_constraint] if lower=upper)
        """
        if not isinstance(model, Submodel):
            raise ValueError('BoundingPatterns.scaled_bounds() can only be used with a Submodel')

        rel_lower, rel_upper = relative_bounds
        name = name or f'{variable.name}'

        if np.allclose(rel_lower, rel_upper, atol=1e-10, equal_nan=True):
            return [model.add_constraints(variable == scaling_variable * rel_lower, name=f'{name}|fixed')]

        upper_constraint = model.add_constraints(variable <= scaling_variable * rel_upper, name=f'{name}|ub')
        lower_constraint = model.add_constraints(variable >= scaling_variable * rel_lower, name=f'{name}|lb')

        return [lower_constraint, upper_constraint]

    @staticmethod
    def scaled_bounds_with_state(
        model: Submodel,
        variable: linopy.Variable,
        scaling_variable: linopy.Variable,
        relative_bounds: tuple[xr.DataArray, xr.DataArray],
        scaling_bounds: tuple[xr.DataArray, xr.DataArray],
        state: linopy.Variable,
        name: str = None,
    ) -> list[linopy.Constraint]:
        """Creates bounds scaled by a variable and controlled by a binary state.

        Variable is forced to 0 when state=0, bounded relative to scaling_variable when state=1.

        Mathematical formulation (Big-M):
            (state - 1) · M_misc + scaling_variable · rel_lower ≤ variable ≤ scaling_variable · rel_upper
            state · big_m_lower ≤ variable ≤ state · big_m_upper

        Where:
            M_misc = scaling_max · rel_lower
            big_m_upper = scaling_max · rel_upper
            big_m_lower = max(ε, scaling_min · rel_lower)

        Args:
            model: The submodel to add constraints to
            variable: Variable to be bounded
            scaling_variable: Variable that scales the bound factors (e.g., equipment size)
            relative_bounds: Tuple of (lower_factor, upper_factor) relative to scaling_variable
            scaling_bounds: Tuple of (scaling_min, scaling_max) bounds of the scaling_variable
            state: Binary variable (0=force variable to 0, 1=allow scaled bounds)
            name: Optional name prefix for constraints

        Returns:
            List of [scaling_lower, scaling_upper, binary_lower, binary_upper] constraints
        """
        if not isinstance(model, Submodel):
            raise ValueError('BoundingPatterns.scaled_bounds_with_state() can only be used with a Submodel')

        rel_lower, rel_upper = relative_bounds
        scaling_min, scaling_max = scaling_bounds
        name = name or f'{variable.name}'

        big_m_misc = scaling_max * rel_lower

        scaling_lower = model.add_constraints(
            variable >= (state - 1) * big_m_misc + scaling_variable * rel_lower, name=f'{name}|lb2'
        )
        scaling_upper = model.add_constraints(variable <= scaling_variable * rel_upper, name=f'{name}|ub2')

        big_m_upper = rel_upper * scaling_max
        big_m_lower = np.maximum(CONFIG.Modeling.epsilon, rel_lower * scaling_min)

        binary_upper = model.add_constraints(state * big_m_upper >= variable, name=f'{name}|ub1')
        binary_lower = model.add_constraints(state * big_m_lower <= variable, name=f'{name}|lb1')

        return [scaling_lower, scaling_upper, binary_lower, binary_upper]

    @staticmethod
    def state_transition_bounds(
        model: Submodel,
        state: linopy.Variable,
        activate: linopy.Variable,
        deactivate: linopy.Variable,
        name: str,
        previous_state: float | xr.DataArray | None = None,
        coord: str = 'time',
    ) -> tuple[linopy.Constraint, linopy.Constraint | None, linopy.Constraint]:
        """Creates state transition constraints for binary state variables.

        Tracks transitions between active (1) and inactive (0) states using
        separate binary variables for activation and deactivation events.

        Mathematical formulation:
            activate[t] - deactivate[t] = state[t] - state[t-1]  ∀t > 0
            activate[0] - deactivate[0] = state[0] - previous_state  (if previous_state is not None)
            activate[t] + deactivate[t] ≤ 1  ∀t
            activate[t], deactivate[t] ∈ {0, 1}

        Args:
            model: The submodel to add constraints to
            state: Binary state variable (0=inactive, 1=active)
            activate: Binary variable for transitions from inactive to active (0→1)
            deactivate: Binary variable for transitions from active to inactive (1→0)
            name: Base name for constraints
            previous_state: State value before first timestep. If None (default), the initial
                constraint is skipped ("relaxed" mode) - startup/shutdown at t=0 is not forced.
            coord: Time dimension name (default 'time')

        Returns:
            Tuple of (transition_constraint, initial_constraint, mutex_constraint).
            initial_constraint is None if previous_state is None (relaxed mode).
        """
        if not isinstance(model, Submodel):
            raise ValueError('BoundingPatterns.state_transition_bounds() can only be used with a Submodel')

        # State transition constraints for t > 0
        transition = model.add_constraints(
            activate.isel({coord: slice(1, None)}) - deactivate.isel({coord: slice(1, None)})
            == state.isel({coord: slice(1, None)}) - state.isel({coord: slice(None, -1)}),
            name=f'{name}|transition',
        )

        # Initial state transition for t = 0 (skip if previous_state is None = relaxed mode)
        if previous_state is not None:
            initial = model.add_constraints(
                activate.isel({coord: 0}) - deactivate.isel({coord: 0}) == state.isel({coord: 0}) - previous_state,
                name=f'{name}|initial',
            )
        else:
            initial = None  # Relaxed mode: no constraint at t=0, startup/shutdown is "free"

        # At most one transition per timestep (mutual exclusivity)
        mutex = model.add_constraints(activate + deactivate <= 1, name=f'{name}|mutex')

        return transition, initial, mutex

    @staticmethod
    def continuous_transition_bounds(
        model: Submodel,
        continuous_variable: linopy.Variable,
        activate: linopy.Variable,
        deactivate: linopy.Variable,
        name: str,
        max_change: float | xr.DataArray,
        previous_value: float | xr.DataArray = 0.0,
        coord: str = 'time',
    ) -> tuple[linopy.Constraint, linopy.Constraint, linopy.Constraint, linopy.Constraint]:
        """Constrains a continuous variable to only change during state transitions.

        Ensures a continuous variable remains constant unless a transition event occurs.
        Uses Big-M formulation to enforce change bounds.

        Mathematical formulation:
            -max_change · (activate[t] + deactivate[t]) ≤ continuous[t] - continuous[t-1] ≤ max_change · (activate[t] + deactivate[t])  ∀t > 0
            -max_change · (activate[0] + deactivate[0]) ≤ continuous[0] - previous_value ≤ max_change · (activate[0] + deactivate[0])
            activate[t], deactivate[t] ∈ {0, 1}

        Behavior:
            - When activate=0 and deactivate=0: variable must stay constant
            - When activate=1 or deactivate=1: variable can change within ±max_change

        Args:
            model: The submodel to add constraints to
            continuous_variable: Continuous variable to constrain
            activate: Binary variable for transitions from inactive to active (0→1)
            deactivate: Binary variable for transitions from active to inactive (1→0)
            name: Base name for constraints
            max_change: Maximum allowed change (Big-M value, should be ≥ actual max change)
            previous_value: Initial value before first timestep (default 0.0)
            coord: Time dimension name (default 'time')

        Returns:
            Tuple of (transition_upper, transition_lower, initial_upper, initial_lower) constraints
        """
        if not isinstance(model, Submodel):
            raise ValueError('ModelingPrimitives.continuous_transition_bounds() can only be used with a Submodel')

        # Transition constraints for t > 0: continuous variable can only change when transitions occur
        transition_upper = model.add_constraints(
            continuous_variable.isel({coord: slice(1, None)}) - continuous_variable.isel({coord: slice(None, -1)})
            <= max_change * (activate.isel({coord: slice(1, None)}) + deactivate.isel({coord: slice(1, None)})),
            name=f'{name}|transition_ub',
        )

        transition_lower = model.add_constraints(
            -(continuous_variable.isel({coord: slice(1, None)}) - continuous_variable.isel({coord: slice(None, -1)}))
            <= max_change * (activate.isel({coord: slice(1, None)}) + deactivate.isel({coord: slice(1, None)})),
            name=f'{name}|transition_lb',
        )

        # Initial constraints for t = 0
        initial_upper = model.add_constraints(
            continuous_variable.isel({coord: 0}) - previous_value
            <= max_change * (activate.isel({coord: 0}) + deactivate.isel({coord: 0})),
            name=f'{name}|initial_ub',
        )

        initial_lower = model.add_constraints(
            -continuous_variable.isel({coord: 0}) + previous_value
            <= max_change * (activate.isel({coord: 0}) + deactivate.isel({coord: 0})),
            name=f'{name}|initial_lb',
        )

        return transition_upper, transition_lower, initial_upper, initial_lower

    @staticmethod
    def link_changes_to_level_with_binaries(
        model: Submodel,
        level_variable: linopy.Variable,
        increase_variable: linopy.Variable,
        decrease_variable: linopy.Variable,
        increase_binary: linopy.Variable,
        decrease_binary: linopy.Variable,
        name: str,
        max_change: float | xr.DataArray,
        initial_level: float | xr.DataArray = 0.0,
        coord: str = 'period',
    ) -> tuple[linopy.Constraint, linopy.Constraint, linopy.Constraint, linopy.Constraint, linopy.Constraint]:
        """
        Link changes to level evolution with binary control and mutual exclusivity.

        Creates the complete constraint system for ALL time periods:
        1. level[0] = initial_level + increase[0] - decrease[0]
        2. level[t] = level[t-1] + increase[t] - decrease[t]  ∀t > 0
        3. increase[t] <= max_change * increase_binary[t]  ∀t
        4. decrease[t] <= max_change * decrease_binary[t]  ∀t
        5. increase_binary[t] + decrease_binary[t] <= 1  ∀t

        Args:
            model: The submodel to add constraints to
            increase_variable: Incremental additions for ALL periods (>= 0)
            decrease_variable: Incremental reductions for ALL periods (>= 0)
            increase_binary: Binary indicators for increases for ALL periods
            decrease_binary: Binary indicators for decreases for ALL periods
            level_variable: Level variable for ALL periods
            name: Base name for constraints
            max_change: Maximum change per period
            initial_level: Starting level before first period
            coord: Time coordinate name

        Returns:
            Tuple of (initial_constraint, transition_constraints, increase_bounds, decrease_bounds, mutual_exclusion)
        """
        if not isinstance(model, Submodel):
            raise ValueError('BoundingPatterns.link_changes_to_level_with_binaries() can only be used with a Submodel')

        # 1. Initial period: level[0] - initial_level =  increase[0] - decrease[0]
        initial_constraint = model.add_constraints(
            level_variable.isel({coord: 0}) - initial_level
            == increase_variable.isel({coord: 0}) - decrease_variable.isel({coord: 0}),
            name=f'{name}|initial_level',
        )

        # 2. Transition periods: level[t] = level[t-1] + increase[t] - decrease[t] for t > 0
        transition_constraints = model.add_constraints(
            level_variable.isel({coord: slice(1, None)})
            == level_variable.isel({coord: slice(None, -1)})
            + increase_variable.isel({coord: slice(1, None)})
            - decrease_variable.isel({coord: slice(1, None)}),
            name=f'{name}|transitions',
        )

        # 3. Increase bounds: increase[t] <= max_change * increase_binary[t] for all t
        increase_bounds = model.add_constraints(
            increase_variable <= increase_binary * max_change,
            name=f'{name}|increase_bounds',
        )

        # 4. Decrease bounds: decrease[t] <= max_change * decrease_binary[t] for all t
        decrease_bounds = model.add_constraints(
            decrease_variable <= decrease_binary * max_change,
            name=f'{name}|decrease_bounds',
        )

        # 5. Mutual exclusivity: increase_binary[t] + decrease_binary[t] <= 1 for all t
        mutual_exclusion = model.add_constraints(
            increase_binary + decrease_binary <= 1,
            name=f'{name}|mutual_exclusion',
        )

        return initial_constraint, transition_constraints, increase_bounds, decrease_bounds, mutual_exclusion
