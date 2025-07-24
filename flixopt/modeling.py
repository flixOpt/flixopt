import logging
from typing import Dict, List, Optional, Tuple, Union

import linopy
import numpy as np
import xarray as xr

from .config import CONFIG
from .core import FlowSystemDimensions, NonTemporalData, Scalar, TemporalData
from .structure import FlowSystemModel, Submodel

logger = logging.getLogger('flixopt')


class ModelingUtilitiesAbstract:
    """Utility functions for modeling calculations - leveraging xarray for temporal data"""

    @staticmethod
    def to_binary(
        values: xr.DataArray,
        epsilon: Optional[float] = None,
        dims: Optional[Union[str, List[str]]] = None,
    ) -> xr.DataArray:
        """
        Converts a DataArray to binary {0, 1} values.

        Args:
            values: Input DataArray to convert to binary
            epsilon: Tolerance for zero detection (uses CONFIG.modeling.EPSILON if None)
            dims: Dims to keep. Other dimensions are collapsed using .any() -> If any value is 1, all are 1.

        Returns:
            Binary DataArray with same shape (or collapsed if collapse_non_time=True)
        """
        if not isinstance(values, xr.DataArray):
            values = xr.DataArray(values, dims=['time'], coords={'time': range(len(values))})

        if epsilon is None:
            epsilon = CONFIG.modeling.EPSILON

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
        binary_values: xr.DataArray,
        dim: str = 'time',
        epsilon: float = None,
    ) -> float:
        """
        Counts the number of consecutive states in a binary time series.

        Args:
            binary_values: Binary DataArray
            dim: Dimension to count consecutive states over
            epsilon: Tolerance for zero detection (uses CONFIG.modeling.EPSILON if None)

        Returns:
            The consecutive number of steps spent in the final state of the timeseries
        """
        if epsilon is None:
            epsilon = CONFIG.modeling.EPSILON

        binary_values = binary_values.any(dim=[d for d in binary_values.dims if d != dim])

        # Handle scalar case
        if binary_values.ndim == 0:
            return float(binary_values.item())

        # Check if final state is off
        if np.isclose(binary_values.isel({dim: -1}).item(), 0, atol=epsilon).all():
            return 0.0

        # Find consecutive 'on' period from the end
        is_zero = np.isclose(binary_values, 0, atol=epsilon)

        # Find the last zero, then sum everything after it
        zero_indices = np.where(is_zero)[0]
        if len(zero_indices) == 0:
            # All 'on' - sum everything
            start_idx = 0
        else:
            # Start after last zero
            start_idx = zero_indices[-1] + 1

        consecutive_values = binary_values.isel({dim: slice(start_idx, None)})

        return float(consecutive_values.sum().item())  # TODO: Som only over one dim?


class ModelingUtilities:
    @staticmethod
    def compute_consecutive_hours_in_state(
        binary_values: TemporalData,
        hours_per_timestep: Union[int, float],
        epsilon: float = None,
    ) -> float:
        """
        Computes the final consecutive duration in state 'on' (=1) in hours.

        Args:
            binary_values: Binary DataArray with 'time' dim, or scalar/array
            hours_per_timestep: Duration of each timestep in hours
            epsilon: Tolerance for zero detection (uses CONFIG.modeling.EPSILON if None)

        Returns:
            The duration of the final consecutive 'on' period in hours
        """
        if not isinstance(hours_per_timestep, (int, float)):
            raise TypeError(f'hours_per_timestep must be a scalar, got {type(hours_per_timestep)}')

        return (
            ModelingUtilitiesAbstract.count_consecutive_states(binary_values=binary_values, epsilon=epsilon)
            * hours_per_timestep
        )

    @staticmethod
    def compute_previous_states(
        previous_values: Optional[xr.DataArray], epsilon: Optional[float] = None
    ) -> xr.DataArray:
        return ModelingUtilitiesAbstract.to_binary(values=previous_values, epsilon=epsilon, dims='time')

    @staticmethod
    def compute_previous_on_duration(
        previous_values: xr.DataArray, hours_per_step: Union[xr.DataArray, float, int]
    ) -> float:
        return (
            ModelingUtilitiesAbstract.count_consecutive_states(ModelingUtilitiesAbstract.to_binary(previous_values))
            * hours_per_step
        )

    @staticmethod
    def compute_previous_off_duration(
        previous_values: xr.DataArray, hours_per_step: Union[xr.DataArray, float, int]
    ) -> float:
        """
        Compute previous consecutive 'off' duration.

        Args:
            previous_values: DataArray with 'time' dimension
            hours_per_step: Duration of each timestep in hours

        Returns:
            Previous consecutive off duration in hours
        """
        if previous_values is None or previous_values.size == 0:
            return 0.0

        previous_states = ModelingUtilities.compute_previous_states(previous_values)
        previous_off_states = 1 - previous_states
        return ModelingUtilities.compute_consecutive_hours_in_state(previous_off_states, hours_per_step)

    @staticmethod
    def get_most_recent_state(previous_values: Optional[xr.DataArray]) -> int:
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
        tracked_expression,
        name: str = None,
        short_name: str = None,
        bounds: Tuple[TemporalData, TemporalData] = None,
        coords: Optional[Union[str, List[str]]] = None,
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
        if not isinstance(model, Submodel):
            raise ValueError('ModelingPrimitives.sum_up_variable() can only be used with a Submodel')

        hours_per_step = model.hours_per_step
        mega = hours_per_step.sum('time') + previous_duration  # Big-M value

        # Duration variable
        duration = model.add_variables(
            lower=0,
            upper=maximum_duration if maximum_duration is not None else mega,
            coords=model.get_coords(),
            name=name,
            short_name=short_name,
        )

        constraints = {}

        # Upper bound: duration[t] ≤ state[t] * M
        constraints['ub'] = model.add_constraints(duration <= state_variable * mega, name=f'{duration.name}|ub')

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
            duration.isel(time=0) == (hours_per_step.isel(time=0) + previous_duration) * state_variable.isel(time=0),
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
        model: Submodel,
        binary_variables: List[linopy.Variable],
        tolerance: float = 1,
        short_name: str = 'mutual_exclusivity',
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
            short_name: Short name of the constraint

        Returns:
            variables: {} (no new variables created)
            constraints: {'mutual_exclusivity': constraint}

        Raises:
            AssertionError: If fewer than 2 variables provided or variables aren't binary
        """
        if not isinstance(model, Submodel):
            raise ValueError('ModelingPrimitives.sum_up_variable() can only be used with a Submodel')

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
        if not isinstance(model, Submodel):
            raise ValueError('BoundingPatterns.bounds_with_state() can only be used with a Submodel')

        lower_bound, upper_bound = bounds
        name = name or f'{variable.name}'

        if np.all(lower_bound - upper_bound) < 1e-10:
            fix_constraint = model.add_constraints(variable == variable_state * upper_bound, name=f'{name}|fix')
            return [fix_constraint]

        epsilon = np.maximum(CONFIG.modeling.EPSILON, lower_bound)

        upper_constraint = model.add_constraints(variable <= variable_state * upper_bound, name=f'{name}|ub')
        lower_constraint = model.add_constraints(variable >= variable_state * epsilon, name=f'{name}|lb')

        return [lower_constraint, upper_constraint]

    @staticmethod
    def scaled_bounds(
        model: Submodel,
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
        if not isinstance(model, Submodel):
            raise ValueError('BoundingPatterns.scaled_bounds() can only be used with a Submodel')

        rel_lower, rel_upper = relative_bounds
        name = name or f'{variable.name}'

        if np.abs(rel_lower - rel_upper).all() < 10e-10:
            return [model.add_constraints(variable == scaling_variable * rel_lower, name=f'{name}|fixed')]

        upper_constraint = model.add_constraints(variable <= scaling_variable * rel_upper, name=f'{name}|ub')
        lower_constraint = model.add_constraints(variable >= scaling_variable * rel_lower, name=f'{name}|lb')

        return [lower_constraint, upper_constraint]

    @staticmethod
    def scaled_bounds_with_state(
        model: Submodel,
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
        if not isinstance(model, Submodel):
            raise ValueError('BoundingPatterns.active_bounds_with_state() can only be used with a Submodel')

        rel_lower, rel_upper = relative_bounds
        scaling_min, scaling_max = scaling_bounds
        name = name or f'{variable.name}'

        big_m_misc = scaling_max * rel_lower

        scaling_lower = model.add_constraints(
            variable >= (variable_state - 1) * big_m_misc + scaling_variable * rel_lower, name=f'{name}|lb2'
        )
        scaling_upper = model.add_constraints(variable <= scaling_variable * rel_upper, name=f'{name}|ub2')

        big_m_upper = rel_upper * scaling_max
        big_m_lower = np.maximum(CONFIG.modeling.EPSILON, rel_lower * scaling_min)

        binary_upper = model.add_constraints(variable_state * big_m_upper >= variable, name=f'{name}|ub1')
        binary_lower = model.add_constraints(variable_state * big_m_lower <= variable, name=f'{name}|lb1')

        return [scaling_lower, scaling_upper, binary_lower, binary_upper]

    @staticmethod
    def state_transition_bounds(
        model: Submodel,
        state_variable: linopy.Variable,
        switch_on: linopy.Variable,
        switch_off: linopy.Variable,
        name: str,
        previous_state=0,
        coord: str = 'time',
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
        if not isinstance(model, Submodel):
            raise ValueError('ModelingPrimitives.state_transition_variables() can only be used with a Submodel')

        # State transition constraints for t > 0
        transition = model.add_constraints(
            switch_on.isel({coord: slice(1, None)}) - switch_off.isel({coord: slice(1, None)})
            == state_variable.isel({coord: slice(1, None)}) - state_variable.isel({coord: slice(None, -1)}),
            name=f'{name}|transition',
        )

        # Initial state transition for t = 0
        initial = model.add_constraints(
            switch_on.isel({coord: 0}) - switch_off.isel({coord: 0})
            == state_variable.isel({coord: 0}) - previous_state,
            name=f'{name}|initial',
        )

        # At most one switch per timestep
        mutex = model.add_constraints(switch_on + switch_off <= 1, name=f'{name}|mutex')

        return transition, initial, mutex
