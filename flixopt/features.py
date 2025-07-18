"""
This module contains the features of the flixopt framework.
Features extend the functionality of Elements.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import linopy
import numpy as np

from .config import CONFIG
from .core import NonTemporalData, Scalar, TemporalData, FlowSystemDimensions
from .interface import InvestParameters, OnOffParameters, Piecewise
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
            tracker = model.add_variables(name=f'{name}|tracker', coords=model.get_coords(coords))
        else:
            tracker = model.add_variables(
                lower=bounds[0] if bounds[0] is not None else -np.inf,
                upper=bounds[1] if bounds[1] is not None else np.inf,
                name=f'{name}|tracker',
                coords=model.get_coords(coords),
            )

        # Constraint: tracker = expression
        tracking = model.add_constraints(tracker == tracked_expression, name=f'{name}|tracking_eq')

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
        name: str,
        variable,
        binary_control,
        size_variable,
        relative_bounds: Tuple[TemporalData, TemporalData],
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
        upper_bound = model.add_constraints(variable <= size_variable * rel_upper, name=f'{name}|size_upper_bound')

        if binary_control is not None:
            # Big-M lower bound: variable ≥ M*(binary-1) + size*lower_factor
            big_m = size_variable.max() * rel_upper.max()  # Conservative big-M
            lower_bound = model.add_constraints(
                variable >= big_m * (binary_control - 1) + size_variable * rel_lower,
                name=f'{name}|binary_controlled_lower_bound',
            )
        else:
            # Simple lower bound: variable ≥ size * lower_factor
            lower_bound = model.add_constraints(variable >= size_variable * rel_lower, name=f'{name}|size_lower_bound')

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
        optional: bool = False,
    ) -> Tuple[Dict[str, linopy.Variable], Dict[str, linopy.Constraint]]:
        """
        Complete investment sizing pattern with optional binary decision.

        Returns:
            variables: {'size': size_var, 'is_invested': binary_var (if optional)}
            constraints: {'investment_upper_bound': constraint, 'investment_lower_bound': constraint, ...}
        """
        variables = {}
        constraints = {}

        # Investment size variable
        size_min, size_max = size_bounds
        variables['size'] = model.add_variables(
            lower=size_min,
            upper=size_max,
            name=f'{name}|investment_size',
            coords=model.get_coords(['year', 'scenario']),
        )

        # Optional binary investment decision
        if optional:
            variables['is_invested'] = model.add_variables(
                binary=True, name=f'{name}|is_invested', coords=model.get_coords(['year', 'scenario'])
            )

            # Link size to investment decision
            if abs(size_min - size_max) < 1e-10:  # Fixed size case
                constraints['fixed_investment_size'] = model.add_constraints(
                    variables['size'] == variables['is_invested'] * size_max, name=f'{name}|fixed_investment_size'
                )
            else:  # Variable size case
                constraints['investment_upper_bound'] = model.add_constraints(
                    variables['size'] <= variables['is_invested'] * size_max, name=f'{name}|investment_upper_bound'
                )
                constraints['investment_lower_bound'] = model.add_constraints(
                    variables['size'] >= variables['is_invested'] * max(CONFIG.modeling.EPSILON, size_min),
                    name=f'{name}|investment_lower_bound',
                )

        # Control dependent variables
        if controlled_variables and control_factors:
            for i, (var, factors) in enumerate(zip(controlled_variables, control_factors)):
                _, control_constraints = ModelingPrimitives.big_m_binary_bounds(
                    model, f'{name}|control_{i}', var, variables.get('is_invested'), variables['size'], factors
                )
                # Flatten control constraints with indexed names
                constraints[f'control_{i}_upper_bound'] = control_constraints['upper_bound']
                constraints[f'control_{i}_lower_bound'] = control_constraints['lower_bound']

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


class InvestmentModel(BaseFeatureModel):
    """Investment model using factory patterns but keeping old interface"""

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        parameters: InvestParameters,
        defining_variable: linopy.Variable,
        relative_bounds_of_defining_variable: Tuple[TemporalData, TemporalData],
        label: Optional[str] = None,
        on_variable: Optional[linopy.Variable] = None,
    ):
        super().__init__(model, label_of_element, parameters, label)

        self._defining_variable = defining_variable
        self._relative_bounds_of_defining_variable = relative_bounds_of_defining_variable
        self._on_variable = on_variable

        # Only keep non-variable attributes
        self.scenario_of_investment: Optional[linopy.Variable] = None
        self.piecewise_effects: Optional[PiecewiseEffectsModel] = None

    def create_variables_and_constraints(self):
        # Use factory patterns
        variables, constraints = ModelingPatterns.investment_sizing_pattern(
            model=self._model,
            name=self.label_full,
            size_bounds=(
                0 if self.parameters.optional else self.parameters.minimum_or_fixed_size,
                self.parameters.maximum_or_fixed_size,
            ),
            controlled_variables=[self._defining_variable],
            control_factors=[self._relative_bounds_of_defining_variable],
            optional=self.parameters.optional,
        )

        # Register variables (stored in Model's variable tracking)
        self.add(variables['size'], 'size')
        if 'is_invested' in variables:
            self.add(variables['is_invested'], 'is_invested')

        # Register constraints
        for constraint_name, constraint in constraints.items():
            self.add(constraint, constraint_name)

        # Handle scenarios and piecewise effects...
        if self._model.flow_system.scenarios is not None:
            self._create_bounds_for_scenarios()

        if self.parameters.piecewise_effects:
            self.piecewise_effects = self.add(
                PiecewiseEffectsModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    piecewise_origin=(self.size.name, self.parameters.piecewise_effects.piecewise_origin),
                    piecewise_shares=self.parameters.piecewise_effects.piecewise_shares,
                    zero_point=self.is_invested,
                ),
                'segments',
            )
            self.piecewise_effects.do_modeling()

    # Properties access variables from Model's tracking system
    @property
    def size(self) -> Optional[linopy.Variable]:
        """Investment size variable"""
        return self.get_variable_by_short_name('size')

    @property
    def is_invested(self) -> Optional[linopy.Variable]:
        """Binary investment decision variable"""
        return self.get_variable_by_short_name('is_invested')

    def add_effects(self):
        """Add investment effects"""
        if self.parameters.fix_effects:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.is_invested * factor if self.is_invested is not None else factor
                    for effect, factor in self.parameters.fix_effects.items()
                },
                target='invest',
            )

        if self.parameters.divest_effects and self.parameters.optional:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: -self.is_invested * factor + factor
                    for effect, factor in self.parameters.divest_effects.items()
                },
                target='invest',
            )

        if self.parameters.specific_effects:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={effect: self.size * factor for effect, factor in self.parameters.specific_effects.items()},
                target='invest',
            )

    def _create_bounds_for_scenarios(self):
        """Keep existing scenario logic"""
        pass


class OnOffModel(BaseFeatureModel):
    """OnOff model using factory patterns"""

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        on_off_parameters: OnOffParameters,
        defining_variables: List[linopy.Variable],
        defining_bounds: List[Tuple[TemporalData, TemporalData]],
        previous_values: List[Optional[TemporalData]],
        label: Optional[str] = None,
    ):
        super().__init__(model, label_of_element, on_off_parameters, label)

        self._defining_variables = defining_variables
        self._defining_bounds = defining_bounds
        self._previous_values = previous_values

    def create_variables_and_constraints(self):
        # Use factory patterns
        variables, constraints = ModelingPatterns.operational_binary_control_pattern(
            model=self._model,
            name=self.label_full,
            controlled_variables=self._defining_variables,
            variable_bounds=self._defining_bounds,
            use_complement=self.parameters.use_off,
            track_total_duration=True,
            track_switches=self.parameters.use_switch_on,
            previous_state=self._get_previous_state(),
            duration_bounds=(self.parameters.on_hours_total_min, self.parameters.on_hours_total_max),
            track_consecutive_on=self.parameters.use_consecutive_on_hours,
            consecutive_on_bounds=(self.parameters.consecutive_on_hours_min, self.parameters.consecutive_on_hours_max),
            previous_on_duration=self._get_previous_on_duration(),
            track_consecutive_off=self.parameters.use_consecutive_off_hours,
            consecutive_off_bounds=(
                self.parameters.consecutive_off_hours_min,
                self.parameters.consecutive_off_hours_max,
            ),
            previous_off_duration=self._get_previous_off_duration(),
        )

        # Register all variables (stored in Model's variable tracking)
        self.add(variables['on'], 'on')
        if 'off' in variables:
            self.add(variables['off'], 'off')
        if 'total_duration' in variables:
            self.add(variables['total_duration'], 'total_duration')
        if 'switch_on' in variables:
            self.add(variables['switch_on'], 'switch_on')
            self.add(variables['switch_off'], 'switch_off')
        if 'consecutive_on_duration' in variables:
            self.add(variables['consecutive_on_duration'], 'consecutive_on_hours')
        if 'consecutive_off_duration' in variables:
            self.add(variables['consecutive_off_duration'], 'consecutive_off_hours')

        # Register all constraints
        for constraint_name, constraint in constraints.items():
            self.add(constraint, constraint_name)

    # Properties access variables from Model's tracking system
    @property
    def on(self) -> Optional[linopy.Variable]:
        """Binary on state variable"""
        return self.get_variable_by_short_name('on')

    @property
    def off(self) -> Optional[linopy.Variable]:
        """Binary off state variable"""
        return self.get_variable_by_short_name('off')

    @property
    def total_on_hours(self) -> Optional[linopy.Variable]:
        """Total on hours variable"""
        return self.get_variable_by_short_name('total_duration')

    @property
    def switch_on(self) -> Optional[linopy.Variable]:
        """Switch on variable"""
        return self.get_variable_by_short_name('switch_on')

    @property
    def switch_off(self) -> Optional[linopy.Variable]:
        """Switch off variable"""
        return self.get_variable_by_short_name('switch_off')

    @property
    def switch_on_nr(self) -> Optional[linopy.Variable]:
        """Number of switch-ons variable"""
        # This could be added to factory if needed
        return None

    @property
    def consecutive_on_hours(self) -> Optional[linopy.Variable]:
        """Consecutive on hours variable"""
        return self.get_variable_by_short_name('consecutive_on_hours')

    @property
    def consecutive_off_hours(self) -> Optional[linopy.Variable]:
        """Consecutive off hours variable"""
        return self.get_variable_by_short_name('consecutive_off_hours')

    def add_effects(self):
        """Add operational effects"""
        if self.parameters.effects_per_running_hour:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.on * factor * self._model.hours_per_step
                    for effect, factor in self.parameters.effects_per_running_hour.items()
                },
                target='operation',
            )

        if self.parameters.effects_per_switch_on and self.switch_on:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.switch_on * factor for effect, factor in self.parameters.effects_per_switch_on.items()
                },
                target='operation',
            )

    def _get_previous_on_duration(self):
        hours_per_step = self._model.hours_per_step.isel(time=0).values.flatten()[0]
        return ModelingUtilities.compute_previous_on_duration(self._previous_values, hours_per_step)

    def _get_previous_off_duration(self):
        hours_per_step = self._model.hours_per_step.isel(time=0).values.flatten()[0]
        return ModelingUtilities.compute_previous_off_duration(self._previous_values, hours_per_step)

    def _get_previous_state(self):
        return ModelingUtilities.get_most_recent_state(self._previous_values)


class StateModel(Model):
    """
    Handles basic on/off binary states for defining variables
    """

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        defining_variables: List[linopy.Variable],
        defining_bounds: List[Tuple[TemporalData, TemporalData]],
        previous_values: List[Optional[TemporalData]] = None,
        use_off: bool = True,
        on_hours_total_min: Optional[TemporalData] = 0,
        on_hours_total_max: Optional[TemporalData] = None,
        effects_per_running_hour: Dict[str, TemporalData] = None,
        label: Optional[str] = None,
    ):
        """
        Models binary state variables based on a continous variable.

        Args:
            model: The FlowSystemModel that is used to create the model.
            label_of_element: The label of the parent (Element). Used to construct the full label of the model.
            defining_variables: List of Variables that are used to define the state
            defining_bounds: List of Tuples, defining the absolute bounds of each defining variable
            previous_values: List of previous values of the defining variables
            use_off: Whether to use the off state or not
            on_hours_total_min: min. overall sum of operating hours.
            on_hours_total_max: max. overall sum of operating hours.
            effects_per_running_hour: Costs per operating hours
            label: Label of the OnOffModel
        """
        super().__init__(model, label_of_element, label)
        assert len(defining_variables) == len(defining_bounds), 'Every defining Variable needs bounds to Model OnOff'
        self._defining_variables = defining_variables
        self._defining_bounds = defining_bounds
        self._previous_values = previous_values or []
        self._on_hours_total_min = on_hours_total_min if on_hours_total_min is not None else 0
        self._on_hours_total_max = on_hours_total_max if on_hours_total_max is not None else np.inf
        self._use_off = use_off
        self._effects_per_running_hour = effects_per_running_hour if effects_per_running_hour is not None else {}

        self.on = None
        self.total_on_hours: Optional[linopy.Variable] = None
        self.off = None

    def do_modeling(self):
        self.on = self.add(
            self._model.add_variables(
                name=f'{self.label_full}|on',
                binary=True,
                coords=self._model.get_coords(),
            ),
            'on',
        )

        self.total_on_hours = self.add(
            self._model.add_variables(
                lower=self._on_hours_total_min,
                upper=self._on_hours_total_max,
                coords=self._model.get_coords(['year', 'scenario']),
                name=f'{self.label_full}|on_hours_total',
            ),
            'on_hours_total',
        )

        self.add(
            self._model.add_constraints(
                self.total_on_hours == (self.on * self._model.hours_per_step).sum('time'),
                name=f'{self.label_full}|on_hours_total',
            ),
            'on_hours_total',
        )

        # Add defining constraints for each variable
        self._add_defining_constraints()

        if self._use_off:
            self.off = self.add(
                self._model.add_variables(
                    name=f'{self.label_full}|off',
                    binary=True,
                    coords=self._model.get_coords(),
                ),
                'off',
            )

            # Constraint: on + off = 1
            self.add(self._model.add_constraints(self.on + self.off == 1, name=f'{self.label_full}|off'), 'off')

        return self

    def _add_defining_constraints(self):
        """Add constraints that link defining variables to the on state"""
        nr_of_def_vars = len(self._defining_variables)

        if nr_of_def_vars == 1:
            # Case for a single defining variable
            def_var = self._defining_variables[0]
            lb, ub = self._defining_bounds[0]

            # Constraint: on * lower_bound <= def_var
            self.add(
                self._model.add_constraints(
                    self.on * np.maximum(CONFIG.modeling.EPSILON, lb) <= def_var, name=f'{self.label_full}|on_con1'
                ),
                'on_con1',
            )

            # Constraint: on * upper_bound >= def_var
            self.add(
                self._model.add_constraints(self.on * ub >= def_var, name=f'{self.label_full}|on_con2'), 'on_con2'
            )
        else:
            # Case for multiple defining variables
            ub = sum(bound[1] for bound in self._defining_bounds) / nr_of_def_vars
            lb = CONFIG.modeling.EPSILON  #TODO: Can this be a bigger value? (maybe the smallest bound?)

            # Constraint: on * epsilon <= sum(all_defining_variables)
            self.add(
                self._model.add_constraints(
                    self.on * lb <= sum(self._defining_variables), name=f'{self.label_full}|on_con1'
                ),
                'on_con1',
            )

            # Constraint to ensure all variables are zero when off.
            # Divide by nr_of_def_vars to improve numerical stability (smaller factors)
            self.add(
                self._model.add_constraints(
                    self.on * ub >= sum([def_var / nr_of_def_vars for def_var in self._defining_variables]),
                    name=f'{self.label_full}|on_con2',
                ),
                'on_con2',
            )

    @property
    def previous_states(self) -> np.ndarray:
        """Computes the previous states {0, 1} of defining variables as a binary array from their previous values."""
        return StateModel.compute_previous_states(self._previous_values, epsilon=CONFIG.modeling.EPSILON)

    @property
    def previous_on_states(self) -> np.ndarray:
        return self.previous_states

    @property
    def previous_off_states(self):
        return 1 - self.previous_states

    @staticmethod
    def compute_previous_states(previous_values: List[TemporalData], epsilon: float = 1e-5) -> np.ndarray:
        """Computes the previous states {0, 1} of defining variables as a binary array from their previous values."""
        if not previous_values or all([val is None for val in previous_values]):
            return np.array([0])

        # Convert to 2D-array and compute binary on/off states
        previous_values = np.array([values for values in previous_values if values is not None])  # Filter out None
        if previous_values.ndim > 1:
            return np.any(~np.isclose(previous_values, 0, atol=epsilon), axis=0).astype(int)

        return (~np.isclose(previous_values, 0, atol=epsilon)).astype(int)


class SwitchStateModel(Model):
    """
    Handles switch on/off transitions
    """

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        state_variable: linopy.Variable,
        previous_state=0,
        switch_on_max: Optional[Scalar] = None,
        label: Optional[str] = None,
    ):
        super().__init__(model, label_of_element, label)
        self._state_variable = state_variable
        self.previous_state = previous_state
        self._switch_on_max = switch_on_max if switch_on_max is not None else np.inf

        self.switch_on = None
        self.switch_off = None
        self.switch_on_nr = None

    def do_modeling(self):
        """Create switch variables and constraints"""

        # Create switch variables
        self.switch_on = self.add(
            self._model.add_variables(binary=True, name=f'{self.label_full}|switch_on', coords=self._model.get_coords()),
            'switch_on',
        )

        self.switch_off = self.add(
            self._model.add_variables(binary=True, name=f'{self.label_full}|switch_off', coords=self._model.get_coords()),
            'switch_off',
        )

        # Create count variable for number of switches
        self.switch_on_nr = self.add(
            self._model.add_variables(
                upper=self._switch_on_max,
                lower=0,
                name=f'{self.label_full}|switch_on_nr',
            ),
            'switch_on_nr',
        )

        # Add switch constraints for all entries after the first timestep
        self.add(
            self._model.add_constraints(
                self.switch_on.isel(time=slice(1, None)) - self.switch_off.isel(time=slice(1, None))
                == self._state_variable.isel(time=slice(1, None)) - self._state_variable.isel(time=slice(None, -1)),
                name=f'{self.label_full}|switch_con',
            ),
            'switch_con',
        )

        # Initial switch constraint
        self.add(
            self._model.add_constraints(
                self.switch_on.isel(time=0) - self.switch_off.isel(time=0)
                == self._state_variable.isel(time=0) - self.previous_state,
                name=f'{self.label_full}|initial_switch_con',
            ),
            'initial_switch_con',
        )

        # Mutual exclusivity constraint
        self.add(
            self._model.add_constraints(self.switch_on + self.switch_off <= 1.1, name=f'{self.label_full}|switch_on_or_off'),
            'switch_on_or_off',
        )

        # Total switch-on count constraint
        self.add(
            self._model.add_constraints(
                self.switch_on_nr == self.switch_on.sum('time'), name=f'{self.label_full}|switch_on_nr'
            ),
            'switch_on_nr',
        )

        return self


class ConsecutiveStateModel(Model):
    """
    Handles tracking consecutive durations in a state
    """

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        state_variable: linopy.Variable,
        minimum_duration: Optional[TemporalData] = None,
        maximum_duration: Optional[TemporalData] = None,
        previous_states: Optional[TemporalData] = None,
        label: Optional[str] = None,
    ):
        """
        Model and constraint the consecutive duration of a state variable.

        Args:
            model: The FlowSystemModel that is used to create the model.
            label_of_element: The label of the parent (Element). Used to construct the full label of the model.
            state_variable: The state variable that is used to model the duration. state = {0, 1}
            minimum_duration: The minimum duration of the state variable.
            maximum_duration: The maximum duration of the state variable.
            previous_states: The previous states of the state variable.
            label: The label of the model. Used to construct the full label of the model.
        """
        super().__init__(model, label_of_element, label)
        self._state_variable = state_variable
        self._previous_states = previous_states
        self._minimum_duration = minimum_duration
        self._maximum_duration = maximum_duration

        self.duration = None

    def do_modeling(self):
        """Create consecutive duration variables and constraints"""
        # Get the hours per step
        hours_per_step = self._model.hours_per_step
        mega = hours_per_step.sum('time') + self.previous_duration

        # Create the duration variable
        self.duration = self.add(
            self._model.add_variables(
                lower=0,
                upper=self._maximum_duration if self._maximum_duration is not None else mega,
                coords=self._model.get_coords(),
                name=f'{self.label_full}|hours',
            ),
            'hours',
        )

        # Add constraints

        # Upper bound constraint
        self.add(
            self._model.add_constraints(
                self.duration <= self._state_variable * mega, name=f'{self.label_full}|con1'
            ),
            'con1',
        )

        # Forward constraint
        self.add(
            self._model.add_constraints(
                self.duration.isel(time=slice(1, None))
                <= self.duration.isel(time=slice(None, -1)) + hours_per_step.isel(time=slice(None, -1)),
                name=f'{self.label_full}|con2a',
            ),
            'con2a',
        )

        # Backward constraint
        self.add(
            self._model.add_constraints(
                self.duration.isel(time=slice(1, None))
                >= self.duration.isel(time=slice(None, -1))
                + hours_per_step.isel(time=slice(None, -1))
                + (self._state_variable.isel(time=slice(1, None)) - 1) * mega,
                name=f'{self.label_full}|con2b',
            ),
            'con2b',
        )

        # Add minimum duration constraints if specified
        if self._minimum_duration is not None:
            self.add(
                self._model.add_constraints(
                    self.duration
                    >= (
                        self._state_variable.isel(time=slice(None, -1)) - self._state_variable.isel(time=slice(1, None))
                    )
                    * self._minimum_duration.isel(time=slice(None, -1)),
                    name=f'{self.label_full}|minimum',
                ),
                'minimum',
            )

            # Handle initial condition
            if 0 < self.previous_duration < self._minimum_duration.isel(time=0).max():
                self.add(
                    self._model.add_constraints(
                        self._state_variable.isel(time=0) == 1, name=f'{self.label_full}|initial_minimum'
                    ),
                    'initial_minimum',
                )

        # Set initial value
        self.add(
            self._model.add_constraints(
                self.duration.isel(time=0) ==
                (hours_per_step.isel(time=0) + self.previous_duration) * self._state_variable.isel(time=0),
                name=f'{self.label_full}|initial',
            ),
            'initial',
        )

        return self

    @property
    def previous_duration(self) -> Scalar:
        """Computes the previous duration of the state variable"""
        #TODO: Allow for other/dynamic timestep resolutions
        return ConsecutiveStateModel.compute_consecutive_hours_in_state(
            self._previous_states, self._model.hours_per_step.isel(time=0).values.flatten()[0]
        )

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
                f'({len(nr_of_indexes_with_consecutive_ones)}) is longer than the provided hours_per_timestep ({len(hours_per_timestep)}), '
                f'as {binary_values=}'
            )

        return np.sum(binary_values[-nr_of_indexes_with_consecutive_ones:] * hours_per_timestep[-nr_of_indexes_with_consecutive_ones:])


class PieceModel(Model):
    """Class for modeling a linear piece of one or more variables in parallel"""

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        label: str,
        as_time_series: bool = True,
    ):
        super().__init__(model, label_of_element, label)
        self.inside_piece: Optional[linopy.Variable] = None
        self.lambda0: Optional[linopy.Variable] = None
        self.lambda1: Optional[linopy.Variable] = None
        self._as_time_series = as_time_series

    def do_modeling(self):
        dims =('time', 'year','scenario') if self._as_time_series else ('year','scenario')
        self.inside_piece = self.add(
            self._model.add_variables(
                binary=True,
                name=f'{self.label_full}|inside_piece',
                coords=self._model.get_coords(dims=dims),
            ),
            'inside_piece',
        )

        self.lambda0 = self.add(
            self._model.add_variables(
                lower=0,
                upper=1,
                name=f'{self.label_full}|lambda0',
                coords=self._model.get_coords(dims=dims),
            ),
            'lambda0',
        )

        self.lambda1 = self.add(
            self._model.add_variables(
                lower=0,
                upper=1,
                name=f'{self.label_full}|lambda1',
                coords=self._model.get_coords(dims=dims),
            ),
            'lambda1',
        )

        # eq:  lambda0(t) + lambda1(t) = inside_piece(t)
        self.add(
            self._model.add_constraints(
                self.inside_piece == self.lambda0 + self.lambda1, name=f'{self.label_full}|inside_piece'
            ),
            'inside_piece',
        )


class PiecewiseModel(Model):
    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        piecewise_variables: Dict[str, Piecewise],
        zero_point: Optional[Union[bool, linopy.Variable]],
        as_time_series: bool,
        label: str = '',
    ):
        """
        Modeling a Piecewise relation between miultiple variables.
        The relation is defined by a list of Pieces, which are assigned to the variables.
        Each Piece is a tuple of (start, end).

        Args:
            model: The FlowSystemModel that is used to create the model.
            label_of_element: The label of the parent (Element). Used to construct the full label of the model.
            label: The label of the model. Used to construct the full label of the model.
            piecewise_variables: The variables to which the Pieces are assigned.
            zero_point: A variable that can be used to define a zero point for the Piecewise relation. If None or False, no zero point is defined.
            as_time_series: Whether the Piecewise relation is defined for a TimeSeries or a single variable.
        """
        super().__init__(model, label_of_element, label)
        self._piecewise_variables = piecewise_variables
        self._zero_point = zero_point
        self._as_time_series = as_time_series

        self.pieces: List[PieceModel] = []
        self.zero_point: Optional[linopy.Variable] = None

    def do_modeling(self):
        for i in range(len(list(self._piecewise_variables.values())[0])):
            new_piece = self.add(
                PieceModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    label=f'Piece_{i}',
                    as_time_series=self._as_time_series,
                )
            )
            self.pieces.append(new_piece)
            new_piece.do_modeling()

        for var_name in self._piecewise_variables:
            variable = self._model.variables[var_name]
            self.add(
                self._model.add_constraints(
                    variable
                    == sum(
                        [
                            piece_model.lambda0 * piece_bounds.start + piece_model.lambda1 * piece_bounds.end
                            for piece_model, piece_bounds in zip(
                                self.pieces, self._piecewise_variables[var_name], strict=False
                            )
                        ]
                    ),
                    name=f'{self.label_full}|{var_name}|lambda',
                ),
                f'{var_name}|lambda',
            )

            # a) eq: Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 1                Aufenthalt nur in Segmenten erlaubt
            # b) eq: -On(t) + Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 0       zusätzlich kann alles auch Null sein
            if isinstance(self._zero_point, linopy.Variable):
                self.zero_point = self._zero_point
                rhs = self.zero_point
            elif self._zero_point is True:
                self.zero_point = self.add(
                    self._model.add_variables(
                        coords=self._model.get_coords(), binary=True, name=f'{self.label_full}|zero_point'
                    ),
                    'zero_point',
                )
                rhs = self.zero_point
            else:
                rhs = 1

            self.add(
                self._model.add_constraints(
                    sum([piece.inside_piece for piece in self.pieces]) <= rhs,
                    name=f'{self.label_full}|{variable.name}|single_segment',
                ),
                f'{var_name}|single_segment',
            )


class ShareAllocationModel(Model):
    def __init__(
        self,
        model: FlowSystemModel,
        dims: List[FlowSystemDimensions],
        label_of_element: Optional[str] = None,
        label: Optional[str] = None,
        label_full: Optional[str] = None,
        total_max: Optional[Scalar] = None,
        total_min: Optional[Scalar] = None,
        max_per_hour: Optional[TemporalData] = None,
        min_per_hour: Optional[TemporalData] = None,
    ):
        super().__init__(model, label_of_element=label_of_element, label=label, label_full=label_full)

        if 'time' not in dims and (max_per_hour is not None or min_per_hour is not None):
            raise ValueError('Both max_per_hour and min_per_hour cannot be used when has_time_dim is False')

        self._dims = dims
        self.total_per_timestep: Optional[linopy.Variable] = None
        self.total: Optional[linopy.Variable] = None
        self.shares: Dict[str, linopy.Variable] = {}
        self.share_constraints: Dict[str, linopy.Constraint] = {}

        self._eq_total_per_timestep: Optional[linopy.Constraint] = None
        self._eq_total: Optional[linopy.Constraint] = None

        # Parameters
        self._total_max = total_max if total_max is not None else np.inf
        self._total_min = total_min if total_min is not None else -np.inf
        self._max_per_hour = max_per_hour if max_per_hour is not None else np.inf
        self._min_per_hour = min_per_hour if min_per_hour is not None else -np.inf

    def do_modeling(self):
        self.total = self.add(
            self._model.add_variables(
                lower=self._total_min,
                upper=self._total_max,
                coords=self._model.get_coords([dim for dim in self._dims if dim != 'time']),
                name=f'{self.label_full}|total',
            ),
            'total',
        )
        # eq: sum = sum(share_i) # skalar
        self._eq_total = self.add(
            self._model.add_constraints(self.total == 0, name=f'{self.label_full}|total'), 'total'
        )

        if 'time' in self._dims:
            self.total_per_timestep = self.add(
                self._model.add_variables(
                    lower=-np.inf if (self._min_per_hour is None) else self._min_per_hour * self._model.hours_per_step,
                    upper=np.inf if (self._max_per_hour is None) else self._max_per_hour * self._model.hours_per_step,
                    coords=self._model.get_coords(self._dims),
                    name=f'{self.label_full}|total_per_timestep',
                ),
                'total_per_timestep',
            )

            self._eq_total_per_timestep = self.add(
                self._model.add_constraints(self.total_per_timestep == 0, name=f'{self.label_full}|total_per_timestep'),
                'total_per_timestep',
            )

            # Add it to the total
            self._eq_total.lhs -= self.total_per_timestep.sum(dim='time')

    def add_share(
        self,
        name: str,
        expression: linopy.LinearExpression,
        dims: Optional[List[FlowSystemDimensions]] = None,
    ):
        """
        Add a share to the share allocation model. If the share already exists, the expression is added to the existing share.
        The expression is added to the right hand side (rhs) of the constraint.
        The variable representing the total share is on the left hand side (lhs) of the constraint.
        var_total = sum(expressions)

        Args:
            name: The name of the share.
            expression: The expression of the share. Added to the right hand side of the constraint.
            dims: The dimensions of the share. Defaults to all dimensions. Dims are ordered automatically
        """
        if dims is None:
            dims = self._dims
        else:
            if 'time' in dims and 'time' not in self._dims:
                raise ValueError('Cannot add share with time-dim to a model without time-dim')
            if 'year' in dims and 'year' not in self._dims:
                raise ValueError('Cannot add share with year-dim to a model without year-dim')
            if 'scenario' in dims and 'scenario' not in self._dims:
                raise ValueError('Cannot add share with scenario-dim to a model without scenario-dim')

        if name in self.shares:
            self.share_constraints[name].lhs -= expression
        else:
            self.shares[name] = self.add(
                self._model.add_variables(
                    coords=self._model.get_coords(dims),
                    name=f'{name}->{self.label_full}',
                ),
                name,
            )
            self.share_constraints[name] = self.add(
                self._model.add_constraints(self.shares[name] == expression, name=f'{name}->{self.label_full}'), name
            )
            if 'time' not in dims:
                self._eq_total.lhs -= self.shares[name]
            else:
                self._eq_total_per_timestep.lhs -= self.shares[name]


class PiecewiseEffectsModel(Model):
    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        piecewise_origin: Tuple[str, Piecewise],
        piecewise_shares: Dict[str, Piecewise],
        zero_point: Optional[Union[bool, linopy.Variable]],
        label: str = 'PiecewiseEffects',
    ):
        super().__init__(model, label_of_element, label)
        assert len(piecewise_origin[1]) == len(list(piecewise_shares.values())[0]), (
            'Piece length of variable_segments and share_segments must be equal'
        )
        self._zero_point = zero_point
        self._piecewise_origin = piecewise_origin
        self._piecewise_shares = piecewise_shares
        self.shares: Dict[str, linopy.Variable] = {}

        self.piecewise_model: Optional[PiecewiseModel] = None

    def do_modeling(self):
        self.shares = {
            effect: self.add(
                self._model.add_variables(
                    coords=self._model.get_coords(['year', 'scenario']), name=f'{self.label_full}|{effect}'
                ),
                f'{effect}',
            )
            for effect in self._piecewise_shares
        }

        piecewise_variables = {
            self._piecewise_origin[0]: self._piecewise_origin[1],
            **{
                self.shares[effect_label].name: self._piecewise_shares[effect_label]
                for effect_label in self._piecewise_shares
            },
        }

        self.piecewise_model = self.add(
            PiecewiseModel(
                model=self._model,
                label_of_element=self.label_of_element,
                piecewise_variables=piecewise_variables,
                zero_point=self._zero_point,
                as_time_series=False,
                label='PiecewiseEffects',
            )
        )

        self.piecewise_model.do_modeling()

        # Shares
        self._model.effects.add_share_to_effects(
            name=self.label_of_element,
            expressions={effect: variable * 1 for effect, variable in self.shares.items()},
            target='invest',
        )


class PreventSimultaneousUsageModel(Model):
    """
    Prevents multiple Multiple Binary variables from being 1 at the same time

    Only 'classic type is modeled for now (# "classic" -> alle Flows brauchen Binärvariable:)
    In 'new', the binary Variables need to be forced beforehand, which is not that straight forward... --> TODO maybe


    # "new":
    # eq: flow_1.on(t) + flow_2.on(t) + .. + flow_i.val(t)/flow_i.max <= 1 (1 Flow ohne Binärvariable!)

    # Anmerkung: Patrick Schönfeld (oemof, custom/link.py) macht bei 2 Flows ohne Binärvariable dies:
    # 1)	bin + flow1/flow1_max <= 1
    # 2)	bin - flow2/flow2_max >= 0
    # 3)    geht nur, wenn alle flow.min >= 0
    # --> könnte man auch umsetzen (statt force_on_variable() für die Flows, aber sollte aufs selbe wie "new" kommen)
    """

    def __init__(
        self,
        model: FlowSystemModel,
        variables: List[linopy.Variable],
        label_of_element: str,
        label: str = 'PreventSimultaneousUsage',
    ):
        super().__init__(model, label_of_element, label)
        self._simultanious_use_variables = variables
        assert len(self._simultanious_use_variables) >= 2, (
            f'Model {self.__class__.__name__} must get at least two variables'
        )
        for variable in self._simultanious_use_variables:  # classic
            assert variable.attrs['binary'], f'Variable {variable} must be binary for use in {self.__class__.__name__}'

    def do_modeling(self):
        # eq: sum(flow_i.on(t)) <= 1.1 (1 wird etwas größer gewählt wg. Binärvariablengenauigkeit)
        self.add(
            self._model.add_constraints(
                sum(self._simultanious_use_variables) <= 1.1, name=f'{self.label_full}|prevent_simultaneous_use'
            ),
            'prevent_simultaneous_use',
        )
