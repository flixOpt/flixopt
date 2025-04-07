"""
This module contains the features of the flixopt framework.
Features extend the functionality of Elements.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import linopy
import numpy as np

from . import utils
from .config import CONFIG
from .core import Scalar, TimestepData, TimeSeries
from .interface import InvestParameters, Piecewise, OnOffParameters
from .structure import Model, SystemModel

logger = logging.getLogger('flixopt')


class InvestmentModel(Model):
    """Class for modeling an investment"""

    def __init__(
        self,
        model: SystemModel,
        label_of_element: str,
        parameters: InvestParameters,
        defining_variable: [linopy.Variable],
        relative_bounds_of_defining_variable: Tuple[NumericData, NumericData],
        label: Optional[str] = None,
        on_variable: Optional[linopy.Variable] = None,
    ):
        super().__init__(model, label_of_element, label)
        self.size: Optional[Union[Scalar, linopy.Variable]] = None
        self.is_invested: Optional[linopy.Variable] = None

        self.piecewise_effects: Optional[PiecewiseEffectsModel] = None

        self._on_variable = on_variable
        self._defining_variable = defining_variable
        self._relative_bounds_of_defining_variable = relative_bounds_of_defining_variable
        self.parameters = parameters

    def do_modeling(self):
        if self.parameters.fixed_size and not self.parameters.optional:
            self.size = self.add(
                self._model.add_variables(
                    lower=self.parameters.fixed_size, upper=self.parameters.fixed_size, name=f'{self.label_full}|size'
                ),
                'size',
            )
        else:
            self.size = self.add(
                self._model.add_variables(
                    lower=0 if self.parameters.optional else self.parameters.minimum_size,
                    upper=self.parameters.maximum_size,
                    name=f'{self.label_full}|size',
                ),
                'size',
            )

        # Optional
        if self.parameters.optional:
            self.is_invested = self.add(
                self._model.add_variables(binary=True, name=f'{self.label_full}|is_invested'), 'is_invested'
            )

            self._create_bounds_for_optional_investment()

        # Bounds for defining variable
        self._create_bounds_for_defining_variable()

        self._create_shares()

    def _create_shares(self):
        # fix_effects:
        fix_effects = self.parameters.fix_effects
        if fix_effects != {}:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.is_invested * factor if self.is_invested is not None else factor
                    for effect, factor in fix_effects.items()
                },
                target='invest',
            )

        if self.parameters.divest_effects != {} and self.parameters.optional:
            # share: divest_effects - isInvested * divest_effects
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={effect: -self.is_invested * factor + factor for effect, factor in self.parameters.divest_effects.items()},
                target='invest',
            )

        if self.parameters.specific_effects != {}:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={effect: self.size * factor for effect, factor in self.parameters.specific_effects.items()},
                target='invest',
            )

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

    def _create_bounds_for_optional_investment(self):
        if self.parameters.fixed_size:
            # eq: investment_size = isInvested * fixed_size
            self.add(
                self._model.add_constraints(
                    self.size == self.is_invested * self.parameters.fixed_size, name=f'{self.label_full}|is_invested'
                ),
                'is_invested',
            )

        else:
            # eq1: P_invest <= isInvested * investSize_max
            self.add(
                self._model.add_constraints(
                    self.size <= self.is_invested * self.parameters.maximum_size,
                    name=f'{self.label_full}|is_invested_ub',
                ),
                'is_invested_ub',
            )

            # eq2: P_invest >= isInvested * max(epsilon, investSize_min)
            self.add(
                self._model.add_constraints(
                    self.size >= self.is_invested * np.maximum(CONFIG.modeling.EPSILON, self.parameters.minimum_size),
                    name=f'{self.label_full}|is_invested_lb',
                ),
                'is_invested_lb',
            )

    def _create_bounds_for_defining_variable(self):
        variable = self._defining_variable
        lb_relative, ub_relative = self._relative_bounds_of_defining_variable
        if np.all(lb_relative == ub_relative):
            self.add(
                self._model.add_constraints(
                    variable == self.size * ub_relative, name=f'{self.label_full}|fix_{variable.name}'
                ),
                f'fix_{variable.name}',
            )
            if self._on_variable is not None:
                raise ValueError(
                    f'Flow {self.label_full} has a fixed relative flow rate and an on_variable.'
                    f'This combination is currently not supported.'
                )
            return

        # eq: defining_variable(t)  <= size * upper_bound(t)
        self.add(
            self._model.add_constraints(
                variable <= self.size * ub_relative, name=f'{self.label_full}|ub_{variable.name}'
            ),
            f'ub_{variable.name}',
        )

        if self._on_variable is None:
            # eq: defining_variable(t) >= investment_size * relative_minimum(t)
            self.add(
                self._model.add_constraints(
                    variable >= self.size * lb_relative, name=f'{self.label_full}|lb_{variable.name}'
                ),
                f'lb_{variable.name}',
            )
        else:
            ## 2. Gleichung: Minimum durch Investmentgröße und On
            # eq: defining_variable(t) >= mega * (On(t)-1) + size * relative_minimum(t)
            #     ... mit mega = relative_maximum * maximum_size
            # äquivalent zu:.
            # eq: - defining_variable(t) + mega * On(t) + size * relative_minimum(t) <= + mega
            mega = lb_relative * self.parameters.maximum_size
            on = self._on_variable
            self.add(
                self._model.add_constraints(
                    variable >= mega * (on - 1) + self.size * lb_relative, name=f'{self.label_full}|lb_{variable.name}'
                ),
                f'lb_{variable.name}',
            )
            # anmerkung: Glg bei Spezialfall relative_minimum = 0 redundant zu OnOff ??


class StateModel(Model):
    """
    Handles basic on/off binary states for defining variables
    """

    def __init__(
        self,
        model: SystemModel,
        label_of_element: str,
        defining_variables: List[linopy.Variable],
        defining_bounds: List[Tuple[TimestepData, TimestepData]],
        previous_values: List[Optional[TimestepData]] = None,
        use_off: bool = True,
        on_hours_total_min: Optional[TimestepData] = 0,
        on_hours_total_max: Optional[TimestepData] = None,
        effects_per_running_hour: Dict[str, TimestepData] = None,
        label: Optional[str] = None,
    ):
        """
        Models binary state variables based on a continous variable.

        Args:
            model: The SystemModel that is used to create the model.
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
        self._effects_per_running_hour = effects_per_running_hour or {}

        self.on = None
        self.total_on_hours: Optional[linopy.Variable] = None
        self.off = None

    def do_modeling(self):
        self.on = self.add(
            self._model.add_variables(
                name=f'{self.label_full}|on',
                binary=True,
                coords=self._model.coords,
            ),
            'on',
        )

        self.total_on_hours = self.add(
            self._model.add_variables(
                lower=self._on_hours_total_min,
                upper=self._on_hours_total_max,
                coords=self._model.get_coords(time_dim=False),
                name=f'{self.label_full}|on_hours_total',
            ),
            'on_hours_total',
        )

        self.add(
            self._model.add_constraints(
                self.total_on_hours == (self.on * self._model.hours_per_step).sum(),
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
                    coords=self._model.coords,
                ),
                'off',
            )

            # Constraint: on + off = 1
            self.add(self._model.add_constraints(self.on + self.off == 1, name=f'{self.label_full}|off'), 'off')

        self._create_shares()

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

            # Constraint: def_var <= on * upper_bound
            self.add(
                self._model.add_constraints(def_var <= self.on * ub, name=f'{self.label_full}|on_con2'), 'on_con2'
            )
        else:
            # Case for multiple defining variables
            ub = sum(bound[1] for bound in self._defining_bounds)
            lb = CONFIG.modeling.EPSILON

            # Constraint: on * epsilon <= sum(all_defining_variables)
            self.add(
                self._model.add_constraints(
                    self.on * lb <= sum(self._defining_variables), name=f'{self.label_full}|on_con1'
                ),
                'on_con1',
            )

            # Constraint to ensure all variables are zero when off
            self.add(
                self._model.add_constraints(
                    sum([def_var / nr_of_def_vars for def_var in self._defining_variables])
                    <= self.on * ub / nr_of_def_vars,
                    name=f'{self.label_full}|on_con2',
                ),
                'on_con2',
            )

    def _create_shares(self):
        if self._effects_per_running_hour:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.on * factor * self._model.hours_per_step
                    for effect, factor in self._effects_per_running_hour.items()
                },
                target='operation',
            )

    @property
    def previous_states(self) -> np.ndarray:
        """Computes the previous states {0, 1} of defining variables as a binary array from their previous values."""
        if not self._previous_values or all([val is None for val in self._previous_values]):
            return np.array([0])

        # Convert to 2D-array and compute binary on/off states
        previous_values = np.array([values for values in self._previous_values if values is not None])  # Filter out None
        if previous_values.ndim > 1:
            return np.any(~np.isclose(previous_values, 0, atol=CONFIG.modeling.EPSILON), axis=0).astype(int)

        return (~np.isclose(previous_values, 0, atol=CONFIG.modeling.EPSILON)).astype(int)

    @property
    def previous_on_states(self) -> np.ndarray:
        return self.previous_states

    @property
    def previous_off_states(self):
        return 1 - self.previous_states


class SwitchStateModel(Model):
    """
    Handles switch on/off transitions
    """

    def __init__(
        self,
        model: SystemModel,
        label_of_element: str,
        state_variable: linopy.Variable,
        previous_state=0,
        switch_on_max: Optional[Scalar] = None,
        effects_per_switch_on: Dict[str, TimestepData] = None,
        label: Optional[str] = None,
    ):
        super().__init__(model, label_of_element, label)
        self._state_variable = state_variable
        self.previous_state = previous_state
        self._switch_on_max = switch_on_max if switch_on_max is not None else np.inf
        self._effects_per_switch_on = effects_per_switch_on or {}

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

        self._create_shares()

        return self

    def _create_shares(self):
        if self._effects_per_switch_on:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.switch_on * factor for effect, factor in self._effects_per_switch_on.items()
                },
                target='operation',
            )


class ConsecutiveStateModel(Model):
    """
    Handles tracking consecutive durations in a state
    """

    def __init__(
        self,
        model: SystemModel,
        label_of_element: str,
        state_variable: linopy.Variable,
        minimum_duration: Optional[TimestepData] = None,
        maximum_duration: Optional[TimestepData] = None,
        previous_states: Optional[TimestepData] = None,
        label: Optional[str] = None,
    ):
        super().__init__(model, label_of_element, label)
        self._state_variable = state_variable
        self._previous_states = previous_states
        self._minimum_duration = minimum_duration
        self._maximum_duration = maximum_duration

        if isinstance(self._minimum_duration, TimeSeries):
            self._minimum_duration = self._minimum_duration.selected_data
        if isinstance(self._maximum_duration, TimeSeries):
            self._maximum_duration = self._maximum_duration.selected_data

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
                name=f'{self.label_full}|consecutive',
            ),
            f'consecutive',
        )

        # Add constraints

        # Upper bound constraint
        self.add(
            self._model.add_constraints(
                self.duration <= self._state_variable * mega, name=f'{self.label_full}|consecutive_con1'
            ),
            f'consecutive_con1',
        )

        # Forward constraint
        self.add(
            self._model.add_constraints(
                self.duration.isel(time=slice(1, None))
                <= self.duration.isel(time=slice(None, -1)) + hours_per_step.isel(time=slice(None, -1)),
                name=f'{self.label_full}|consecutive_con2a',
            ),
            f'consecutive_con2a',
        )

        # Backward constraint
        self.add(
            self._model.add_constraints(
                self.duration.isel(time=slice(1, None))
                >= self.duration.isel(time=slice(None, -1))
                + hours_per_step.isel(time=slice(None, -1))
                + (self._state_variable.isel(time=slice(1, None)) - 1) * mega,
                name=f'{self.label_full}|consecutive_con2b',
            ),
            f'consecutive_con2b',
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
                    name=f'{self.label_full}|consecutive_minimum',
                ),
                f'consecutive_minimum',
            )

            # Handle initial condition
            if 0 < self.previous_duration < self._minimum_duration.isel(time=0):
                self.add(
                    self._model.add_constraints(
                        self._state_variable.isel(time=0) == 1, name=f'{self.label_full}|consecutive_minimum_initial'
                    ),
                    f'consecutive_minimum_initial',
                )

        # Set initial value
        self.add(
            self._model.add_constraints(
                self.duration.isel(time=0) == hours_per_step.isel(time=0) * self._state_variable.isel(time=0),
                name=f'{self.label_full}|consecutive_initial',
            ),
            f'consecutive_initial',
        )

        return self

    @property
    def previous_duration(self) -> Scalar:
        """Computes the previous duration of the state variable"""
        if not self._previous_states:
            return 0

        if np.isscalar(self._previous_states) and np.isscalar(self._model.hours_per_step):
            return self._previous_states * self._model.hours_per_step

        elif np.isscalar(self._previous_states) and not np.isscalar(self._model.hours_per_step):
            return self._previous_states * self._model.hours_per_step[-1]

        # Find the indexes where value=`0` in a 1D-array
        zero_indices = np.where(np.isclose(self._previous_states, 0, atol=CONFIG.modeling.EPSILON))[0]
        length_of_last_duration = zero_indices[-1] + 1 if zero_indices.size > 0 else len(self._previous_states)

        if not np.isscalar(self._previous_states) and np.isscalar(self._model.hours_per_step):
            return np.sum(self._previous_states[-length_of_last_duration:] * self._model.hours_per_step)
        elif not np.isscalar(self._previous_states) and not np.isscalar(self._model.hours_per_step):
            if length_of_last_duration > len(self._model.hours_per_step):  # check that lengths are compatible
                raise TypeError(
                    f'When trying to calculate the consecutive duration, the length of the last duration '
                    f'({length_of_last_duration}) is longer than the hours_per_timestep ({len(self._model.hours_per_step)})'
                )
            return np.sum(self._previous_states[-length_of_last_duration:] * self._model.hours_per_step[-length_of_last_duration:])
        else:
            raise Exception(
                f'Unexpected state reached in function compute_consecutive_duration(). binary_values={self._previous_states}; '
                f'hours_per_timestep={self._model.hours_per_step}'
            )


class OnOffModel(Model):
    """
    Class for modeling the on and off state of a variable
    Uses component models to create a modular implementation
    """

    def __init__(
        self,
        model: SystemModel,
        on_off_parameters: OnOffParameters,
        label_of_element: str,
        defining_variables: List[linopy.Variable],
        defining_bounds: List[Tuple[TimestepData, TimestepData]],
        previous_values: List[Optional[TimestepData]],
        label: Optional[str] = None,
    ):
        """
        Constructor for OnOffModel

        Args:
            model: Reference to the SystemModel
            on_off_parameters: Parameters for the OnOffModel
            label_of_element: Label of the Parent
            defining_variables: List of Variables that are used to define the OnOffModel
            defining_bounds: List of Tuples, defining the absolute bounds of each defining variable
            previous_values: List of previous values of the defining variables
            label: Label of the OnOffModel
        """
        super().__init__(model, label_of_element, label)
        self.parameters = on_off_parameters
        self._defining_variables = defining_variables
        self._defining_bounds = defining_bounds
        self._previous_values = previous_values

        self.state_model = None
        self.switch_state_model = None
        self.consecutive_on_model = None
        self.consecutive_off_model = None

    def do_modeling(self):
        """Create all variables and constraints for the OnOffModel"""

        # Create binary state component
        self.state_model = StateModel(
            model=self._model,
            label_of_element=self.label_of_element,
            defining_variables=self._defining_variables,
            defining_bounds=self._defining_bounds,
            previous_values=self._previous_values,
            use_off=self.parameters.use_off,
            on_hours_total_min=self.parameters.on_hours_total_min,
            on_hours_total_max=self.parameters.on_hours_total_max,
            effects_per_running_hour=self.parameters.effects_per_running_hour,
            label='State',
        )
        self.add(self.state_model)
        self.state_model.do_modeling()

        # Create switch component if needed
        if self.parameters.use_switch_on:
            self.switch_state_model = SwitchStateModel(
                model=self._model,
                label_of_element=self.label_of_element,
                state_variable=self.state_model.on,
                previous_state=self.state_model.previous_on_states[-1],
                switch_on_max=self.parameters.switch_on_total_max,
                effects_per_switch_on=self.parameters.effects_per_switch_on,
                label=f'SwitchState',
            )
            self.add(self.switch_state_model)
            self.switch_state_model.do_modeling()

        # Create consecutive on hours component if needed
        if self.parameters.use_consecutive_on_hours:
            self.consecutive_on_model = ConsecutiveStateModel(
                model=self._model,
                label_of_element=self.label_of_element,
                state_variable=self.state_model.on,
                minimum_duration=self.parameters.consecutive_on_hours_min,
                maximum_duration=self.parameters.consecutive_on_hours_max,
                previous_states=self.state_model.previous_on_states,
                label=f'ConsecutiveOn',
            )
            self.add(self.consecutive_on_model)
            self.consecutive_on_model.do_modeling()

        # Create consecutive off hours component if needed
        if self.parameters.use_consecutive_off_hours:
            if not self.parameters.use_off:
                logger.warning(
                    f'In {self.label_full}, consecutive_off_hours are requested, but use_off=False. '
                    f'Setting use_off=True automatically.'
                )

            self.consecutive_off_model = ConsecutiveStateModel(
                model=self._model,
                label_of_element=self.label_of_element,
                state_variable=self.state_model.off,
                minimum_duration=self.parameters.consecutive_off_hours_min,
                maximum_duration=self.parameters.consecutive_off_hours_max,
                previous_states=self.state_model.previous_off_states,
                label=f'ConsecutiveOff',
            )
            self.add(self.consecutive_off_model)
            self.consecutive_off_model.do_modeling()

    @property
    def on(self):
        return self.state_model.on


class PieceModel(Model):
    """Class for modeling a linear piece of one or more variables in parallel"""

    def __init__(
        self,
        model: SystemModel,
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
        self.inside_piece = self.add(
            self._model.add_variables(
                binary=True,
                name=f'{self.label_full}|inside_piece',
                coords=self._model.coords if self._as_time_series else None,
            ),
            'inside_piece',
        )

        self.lambda0 = self.add(
            self._model.add_variables(
                lower=0,
                upper=1,
                name=f'{self.label_full}|lambda0',
                coords=self._model.coords if self._as_time_series else None,
            ),
            'lambda0',
        )

        self.lambda1 = self.add(
            self._model.add_variables(
                lower=0,
                upper=1,
                name=f'{self.label_full}|lambda1',
                coords=self._model.coords if self._as_time_series else None,
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
        model: SystemModel,
        label_of_element: str,
        label: str,
        piecewise_variables: Dict[str, Piecewise],
        zero_point: Optional[Union[bool, linopy.Variable]],
        as_time_series: bool,
    ):
        """
        Modeling a Piecewise relation between miultiple variables.
        The relation is defined by a list of Pieces, which are assigned to the variables.
        Each Piece is a tuple of (start, end).

        Args:
            model: The SystemModel that is used to create the model.
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
                    name=f'{self.label_full}|{var_name}_lambda',
                ),
                f'{var_name}_lambda',
            )

            # a) eq: Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 1                Aufenthalt nur in Segmenten erlaubt
            # b) eq: -On(t) + Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 0       zusätzlich kann alles auch Null sein
            if isinstance(self._zero_point, linopy.Variable):
                self.zero_point = self._zero_point
                rhs = self.zero_point
            elif self._zero_point is True:
                self.zero_point = self.add(
                    self._model.add_variables(
                        coords=self._model.coords, binary=True, name=f'{self.label_full}|zero_point'
                    ),
                    'zero_point',
                )
                rhs = self.zero_point
            else:
                rhs = 1

            self.add(
                self._model.add_constraints(
                    sum([piece.inside_piece for piece in self.pieces]) <= rhs,
                    name=f'{self.label_full}|{variable.name}_single_segment',
                ),
                'single_segment',
            )


class ShareAllocationModel(Model):
    def __init__(
        self,
        model: SystemModel,
        shares_are_time_series: bool,
        label_of_element: Optional[str] = None,
        label: Optional[str] = None,
        label_full: Optional[str] = None,
        total_max: Optional[Scalar] = None,
        total_min: Optional[Scalar] = None,
        max_per_hour: Optional[NumericData] = None,
        min_per_hour: Optional[NumericData] = None,
    ):
        super().__init__(model, label_of_element=label_of_element, label=label, label_full=label_full)
        if not shares_are_time_series:  # If the condition is True
            assert max_per_hour is None and min_per_hour is None, (
                'Both max_per_hour and min_per_hour cannot be used when shares_are_time_series is False'
            )
        self.total_per_timestep: Optional[linopy.Variable] = None
        self.total: Optional[linopy.Variable] = None
        self.shares: Dict[str, linopy.Variable] = {}
        self.share_constraints: Dict[str, linopy.Constraint] = {}

        self._eq_total_per_timestep: Optional[linopy.Constraint] = None
        self._eq_total: Optional[linopy.Constraint] = None

        # Parameters
        self._shares_are_time_series = shares_are_time_series
        self._total_max = total_max if total_min is not None else np.inf
        self._total_min = total_min if total_min is not None else -np.inf
        self._max_per_hour = max_per_hour if max_per_hour is not None else np.inf
        self._min_per_hour = min_per_hour if min_per_hour is not None else -np.inf

    def do_modeling(self):
        self.total = self.add(
            self._model.add_variables(
                lower=self._total_min, upper=self._total_max, coords=None, name=f'{self.label_full}|total'
            ),
            'total',
        )
        # eq: sum = sum(share_i) # skalar
        self._eq_total = self.add(
            self._model.add_constraints(self.total == 0, name=f'{self.label_full}|total'), 'total'
        )

        if self._shares_are_time_series:
            self.total_per_timestep = self.add(
                self._model.add_variables(
                    lower=-np.inf
                    if (self._min_per_hour is None)
                    else np.multiply(self._min_per_hour, self._model.hours_per_step),
                    upper=np.inf
                    if (self._max_per_hour is None)
                    else np.multiply(self._max_per_hour, self._model.hours_per_step),
                    coords=self._model.coords,
                    name=f'{self.label_full}|total_per_timestep',
                ),
                'total_per_timestep',
            )

            self._eq_total_per_timestep = self.add(
                self._model.add_constraints(self.total_per_timestep == 0, name=f'{self.label_full}|total_per_timestep'),
                'total_per_timestep',
            )

            # Add it to the total
            self._eq_total.lhs -= self.total_per_timestep.sum()

    def add_share(
        self,
        name: str,
        expression: linopy.LinearExpression,
    ):
        """
        Add a share to the share allocation model. If the share already exists, the expression is added to the existing share.
        The expression is added to the right hand side (rhs) of the constraint.
        The variable representing the total share is on the left hand side (lhs) of the constraint.
        var_total = sum(expressions)

        Args:
            name: The name of the share.
            expression: The expression of the share. Added to the right hand side of the constraint.
        """
        if name in self.shares:
            self.share_constraints[name].lhs -= expression
        else:
            self.shares[name] = self.add(
                self._model.add_variables(
                    coords=None
                    if isinstance(expression, linopy.LinearExpression)
                    and expression.ndim == 0
                    or not isinstance(expression, linopy.LinearExpression)
                    else self._model.coords,
                    name=f'{name}->{self.label_full}',
                ),
                name,
            )
            self.share_constraints[name] = self.add(
                self._model.add_constraints(self.shares[name] == expression, name=f'{name}->{self.label_full}'), name
            )
            if self.shares[name].ndim == 0:
                self._eq_total.lhs -= self.shares[name]
            else:
                self._eq_total_per_timestep.lhs -= self.shares[name]


class PiecewiseEffectsModel(Model):
    def __init__(
        self,
        model: SystemModel,
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
            effect: self.add(self._model.add_variables(coords=None, name=f'{self.label_full}|{effect}'), f'{effect}')
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
                label=f'{self.label_full}|PiecewiseModel',
                piecewise_variables=piecewise_variables,
                zero_point=self._zero_point,
                as_time_series=False,
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
        model: SystemModel,
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
