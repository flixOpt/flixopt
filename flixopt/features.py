"""
This module contains the features of the flixopt framework.
Features extend the functionality of Elements.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

from .config import CONFIG
from .core import Numeric, Skalar, TimeSeries
from .interface import InvestParameters, OnOffParameters
from .math_modeling import Equation, Variable, VariableTS
from .structure import (
    Element,
    ElementModel,
    SystemModel,
    create_equation,
    create_variable,
)

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from .components import Storage
    from .effects import Effect
    from .elements import Flow


logger = logging.getLogger('flixopt')


class InvestmentModel(ElementModel):
    """Class for modeling an investment"""

    def __init__(
        self,
        element: Union['Flow', 'Storage'],
        invest_parameters: InvestParameters,
        defining_variable: [VariableTS],
        relative_bounds_of_defining_variable: Tuple[Numeric, Numeric],
        fixed_relative_profile: Optional[Numeric] = None,
        label: str = 'Investment',
        on_variable: Optional[VariableTS] = None,
    ):
        """
        If fixed relative profile is used, the relative bounds are ignored
        """
        super().__init__(element, label)
        self.element: Union['Flow', 'Storage'] = element
        self.size: Optional[Union[Skalar, Variable]] = None
        self.is_invested: Optional[Variable] = None

        self._segments: Optional[SegmentedSharesModel] = None

        self._on_variable = on_variable
        self._defining_variable = defining_variable
        self._relative_bounds_of_defining_variable = relative_bounds_of_defining_variable
        self._fixed_relative_profile = fixed_relative_profile
        self._invest_parameters = invest_parameters

    def do_modeling(self, system_model: SystemModel):
        invest_parameters = self._invest_parameters
        if invest_parameters.fixed_size and not invest_parameters.optional:
            self.size = create_variable('size', self, 1, fixed_value=invest_parameters.fixed_size)
        else:
            lower_bound = 0 if invest_parameters.optional else invest_parameters.minimum_size
            self.size = create_variable(
                'size', self, 1, lower_bound=lower_bound, upper_bound=invest_parameters.maximum_size
            )
        # Optional
        if invest_parameters.optional:
            self.is_invested = create_variable('isInvested', self, 1, is_binary=True)
            self._create_bounds_for_optional_investment(system_model)

        # Bounds for defining variable
        self._create_bounds_for_defining_variable(system_model)

        self._create_shares(system_model)

    def _create_shares(self, system_model: SystemModel):
        effect_collection = system_model.effect_collection_model
        invest_parameters = self._invest_parameters

        # fix_effects:
        fix_effects = invest_parameters.fix_effects
        if fix_effects != {}:
            if invest_parameters.optional:  # share: + isInvested * fix_effects
                variable_is_invested = self.is_invested
            else:
                variable_is_invested = None
            effect_collection.add_share_to_invest('fix_effects', self.element, fix_effects, 1, variable_is_invested)

        # divest_effects:
        divest_effects = invest_parameters.divest_effects
        if divest_effects != {}:
            if invest_parameters.optional:  # share: [divest_effects - isInvested * divest_effects]
                # 1. part of share [+ divest_effects]:
                effect_collection.add_share_to_invest('divest_effects', self.element, divest_effects, 1, None)
                # 2. part of share [- isInvested * divest_effects]:
                effect_collection.add_share_to_invest(
                    'divest_cancellation_effects', self.element, divest_effects, -1, self.is_invested
                )
                # TODO : these 2 parts should be one share! -> SingleShareModel...?

        # # specific_effects:
        specific_effects = invest_parameters.specific_effects
        if specific_effects != {}:
            # share: + investment_size (=var)   * specific_effects
            effect_collection.add_share_to_invest('specific_effects', self.element, specific_effects, 1, self.size)
        # segmented Effects
        invest_segments = invest_parameters.effects_in_segments
        if invest_segments:
            self._segments = SegmentedSharesModel(
                self.element, (self.size, invest_segments[0]), invest_segments[1], self.is_invested
            )
            self.sub_models.append(self._segments)
            self._segments.do_modeling(system_model)

    def _create_bounds_for_optional_investment(self, system_model: SystemModel):
        if self._invest_parameters.fixed_size:
            # eq: investment_size = isInvested * fixed_size
            eq_is_invested = create_equation('is_invested', self, 'eq')
            eq_is_invested.add_summand(self.size, -1)
            eq_is_invested.add_summand(self.is_invested, self._invest_parameters.fixed_size)
        else:
            # eq1: P_invest <= isInvested * investSize_max
            eq_is_invested_ub = create_equation('is_invested_ub', self, 'ineq')
            eq_is_invested_ub.add_summand(self.size, 1)
            eq_is_invested_ub.add_summand(self.is_invested, np.multiply(-1, self._invest_parameters.maximum_size))

            # eq2: P_invest >= isInvested * max(epsilon, investSize_min)
            eq_is_invested_lb = create_equation('is_invested_lb', self, 'ineq')
            eq_is_invested_lb.add_summand(self.size, -1)
            eq_is_invested_lb.add_summand(
                self.is_invested, np.maximum(CONFIG.modeling.EPSILON, self._invest_parameters.minimum_size)
            )

    def _create_bounds_for_defining_variable(self, system_model: SystemModel):
        label = self._defining_variable.label
        # fixed relative value
        if self._fixed_relative_profile is not None:
            # TODO: Allow Off? Currently not...
            eq_fixed = create_equation(f'fixed_{label}', self)
            eq_fixed.add_summand(self._defining_variable, 1)
            eq_fixed.add_summand(self.size, np.multiply(-1, self._fixed_relative_profile))
        else:
            relative_minimum, relative_maximum = self._relative_bounds_of_defining_variable
            eq_upper = create_equation(f'ub_{label}', self, 'ineq')
            # eq: defining_variable(t)  <= size * upper_bound(t)
            eq_upper.add_summand(self._defining_variable, 1)
            eq_upper.add_summand(self.size, np.multiply(-1, relative_maximum))

            ## 2. Gleichung: Minimum durch Investmentgröße ##
            eq_lower = create_equation(f'lb_{label}', self, 'ineq')
            if self._on_variable is None:
                # eq: defining_variable(t) >= investment_size * relative_minimum(t)
                eq_lower.add_summand(self._defining_variable, -1)
                eq_lower.add_summand(self.size, relative_minimum)
            else:
                ## 2. Gleichung: Minimum durch Investmentgröße und On
                # eq: defining_variable(t) >= mega * (On(t)-1) + size * relative_minimum(t)
                #     ... mit mega = relative_maximum * maximum_size
                # äquivalent zu:.
                # eq: - defining_variable(t) + mega * On(t) + size * relative_minimum(t) <= + mega
                mega = relative_maximum * self._invest_parameters.maximum_size
                eq_lower.add_summand(self._defining_variable, -1)
                eq_lower.add_summand(self._on_variable, mega)
                eq_lower.add_summand(self.size, relative_minimum)
                eq_lower.add_constant(mega)
                # Anmerkung: Glg bei Spezialfall relative_minimum = 0 redundant zu OnOff ??


class OnOffModel(ElementModel):
    """
    Class for modeling the on and off state of a variable
    If defining_bounds are given, creates sufficient lower bounds
    """

    def __init__(
        self,
        element: Element,
        on_off_parameters: OnOffParameters,
        defining_variables: List[VariableTS],
        defining_bounds: List[Tuple[Numeric, Numeric]],
        label: str = 'OnOff',
    ):
        """
        defining_bounds: a list of Numeric, that can be  used to create the bound for On/Off more efficiently
        """
        super().__init__(element, label)
        self.element = element
        self.on: Optional[VariableTS] = None
        self.total_on_hours: Optional[Variable] = None

        self.consecutive_on_hours: Optional[VariableTS] = None
        self.consecutive_off_hours: Optional[VariableTS] = None

        self.off: Optional[VariableTS] = None

        self.switch_on: Optional[VariableTS] = None
        self.switch_off: Optional[VariableTS] = None
        self.nr_switch_on: Optional[VariableTS] = None

        self._on_off_parameters = on_off_parameters
        self._defining_variables = defining_variables
        # Ensure that no lower bound is below a certain threshold
        self._defining_bounds = [(np.maximum(lb, CONFIG.modeling.EPSILON), ub) for lb, ub in defining_bounds]
        assert len(defining_variables) == len(defining_bounds), 'Every defining Variable needs bounds to Model OnOff'

    def do_modeling(self, system_model: SystemModel):
        self.on = create_variable(
            'on',
            self,
            system_model.nr_of_time_steps,
            is_binary=True,
            previous_values=self._previous_on_values(CONFIG.modeling.EPSILON),
        )

        self.total_on_hours = create_variable(
            'totalOnHours',
            self,
            1,
            lower_bound=self._on_off_parameters.on_hours_total_min,
            upper_bound=self._on_off_parameters.on_hours_total_max,
        )
        eq_total_on = create_equation('totalOnHours', self)
        eq_total_on.add_summand(self.on, system_model.dt_in_hours, as_sum=True)
        eq_total_on.add_summand(self.total_on_hours, -1)

        self._add_on_constraints(system_model, system_model.indices)

        if self._on_off_parameters.use_off:
            self.off = create_variable(
                'off',
                self,
                system_model.nr_of_time_steps,
                is_binary=True,
                previous_values=1 - self._previous_on_values(CONFIG.modeling.EPSILON),
            )

            self._add_off_constraints(system_model, system_model.indices)

        if self._on_off_parameters.use_consecutive_on_hours:
            self.consecutive_on_hours = self._get_duration_in_hours(
                'consecutiveOnHours',
                self.on,
                self._on_off_parameters.consecutive_on_hours_min,
                self._on_off_parameters.consecutive_on_hours_max,
                system_model,
                system_model.indices,
            )

        if self._on_off_parameters.use_consecutive_off_hours:
            self.consecutive_off_hours = self._get_duration_in_hours(
                'consecutiveOffHours',
                self.off,
                self._on_off_parameters.consecutive_off_hours_min,
                self._on_off_parameters.consecutive_off_hours_max,
                system_model,
                system_model.indices,
            )

        if self._on_off_parameters.use_switch_on:
            self.switch_on = create_variable('switchOn', self, system_model.nr_of_time_steps, is_binary=True)
            self.switch_off = create_variable('switchOff', self, system_model.nr_of_time_steps, is_binary=True)
            self.nr_switch_on = create_variable(
                'nrSwitchOn', self, 1, upper_bound=self._on_off_parameters.switch_on_total_max
            )
            self._add_switch_constraints(system_model)

        self._create_shares(system_model)

    def _add_on_constraints(self, system_model: SystemModel, time_indices: Union[list[int], range]):
        assert self.on is not None, f'On variable of {self.element} must be defined to add constraints'
        # % Bedingungen 1) und 2) müssen erfüllt sein:

        # % Anmerkung: Falls "abschnittsweise linear" gewählt, dann ist eigentlich nur Bedingung 1) noch notwendig
        # %            (und dann auch nur wenn erstes Segment bei Q_th=0 beginnt. Dann soll bei Q_th=0 (d.h. die Maschine ist Aus) On = 0 und segment1.onSeg = 0):)
        # %            Fazit: Wenn kein Performance-Verlust durch mehr Gleichungen, dann egal!

        nr_of_defining_variables = len(self._defining_variables)
        assert nr_of_defining_variables > 0, 'Achtung: mindestens 1 Flow notwendig'

        eq_on_1 = create_equation('On_Constraint_1', self, eq_type='ineq')
        eq_on_2 = create_equation('On_Constraint_2', self, eq_type='ineq')
        if nr_of_defining_variables == 1:
            variable = self._defining_variables[0]
            lower_bound, upper_bound = self._defining_bounds[0]
            #### Bedingung 1) ####
            # eq: On(t) * max(epsilon, lower_bound) <= Q_th(t)
            eq_on_1.add_summand(variable, -1, time_indices)
            eq_on_1.add_summand(self.on, np.maximum(CONFIG.modeling.EPSILON, lower_bound), time_indices)

            #### Bedingung 2) ####
            # eq: Q_th(t) <= Q_th_max * On(t)
            eq_on_2.add_summand(variable, 1, time_indices)
            eq_on_2.add_summand(self.on, -1 * upper_bound, time_indices)

        else:  # Bei mehreren Leistungsvariablen:
            #### Bedingung 1) ####
            # When all defining variables are 0, On is 0
            # eq: - sum(alle Leistungen(t)) + Epsilon * On(t) <= 0
            for variable in self._defining_variables:
                eq_on_1.add_summand(variable, -1, time_indices)
            eq_on_1.add_summand(self.on, CONFIG.modeling.EPSILON, time_indices)

            #### Bedingung 2) ####
            ## sum(alle Leistung) >0 -> On = 1 | On=0 -> sum(Leistung)=0
            #  eq: sum( Leistung(t,i))              - sum(Leistung_max(i))             * On(t) <= 0
            #  --> damit Gleichungswerte nicht zu groß werden, noch durch nr_of_flows geteilt:
            #  eq: sum( Leistung(t,i) / nr_of_flows ) - sum(Leistung_max(i)) / nr_of_flows * On(t) <= 0
            absolute_maximum: Numeric = 0.0
            for variable, bounds in zip(self._defining_variables, self._defining_bounds, strict=False):
                eq_on_2.add_summand(variable, 1 / nr_of_defining_variables, time_indices)
                absolute_maximum += bounds[
                    1
                ]  # der maximale Nennwert reicht als Obergrenze hier aus. (immer noch math. günster als BigM)

            upper_bound = absolute_maximum / nr_of_defining_variables
            eq_on_2.add_summand(self.on, -1 * upper_bound, time_indices)

        if np.max(upper_bound) > CONFIG.modeling.BIG_BINARY_BOUND:
            logger.warning(
                f'In "{self.element.label_full}", a binary definition was created with a big upper bound '
                f'({np.max(upper_bound)}). This can lead to wrong results regarding the on and off variables. '
                f'Avoid this warning by reducing the size of {self.element.label_full} '
                f'(or the maximum_size of the corresponding InvestParameters). '
                f'If its a Component, you might need to adjust the sizes of all of its flows.'
            )

    def _add_off_constraints(self, system_model: SystemModel, time_indices: Union[list[int], range]):
        assert self.off is not None, f'Off variable of {self.element} must be defined to add constraints'
        # Definition var_off:
        # eq: var_on(t) + var_off(t) = 1
        eq_off = create_equation('var_off', self, eq_type='eq')
        eq_off.add_summand(self.off, 1, time_indices)
        eq_off.add_summand(self.on, 1, time_indices)
        eq_off.add_constant(1)

    def _get_duration_in_hours(
        self,
        variable_label: str,
        binary_variable: VariableTS,
        minimum_duration: Optional[TimeSeries],
        maximum_duration: Optional[TimeSeries],
        system_model: SystemModel,
        time_indices: Union[list[int], range],
    ) -> VariableTS:
        """
        creates duration variable and adds constraints to a time-series variable to enforce duration limits based on
        binary activity.
        The minimum duration in the last time step is not restricted.
        Previous values before t=0 are not recognised!

        Parameters:
            variable_label (str):
                Label for the duration variable to be created.
            binary_variable (VariableTS):
                Time-series binary variable (e.g., [0, 0, 1, 1, 1, 0, ...]) representing activity states.
            minimum_duration (Optional[TimeSeries]):
                Minimum duration the activity must remain active once started.
                If None, no minimum duration constraint is applied.
            maximum_duration (Optional[TimeSeries]):
                Maximum duration the activity can remain active.
                If None, the maximum duration is set to the total available time.
            system_model (SystemModel):
                The system model containing time step information.
            time_indices (Union[list[int], range]):
                List or range of indices to which to apply the constraints.

        Returns:
            VariableTS: The created duration variable representing consecutive active durations.

        Example:
            binary_variable: [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, ...]
            duration_in_hours: [0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0, ...] (only if dt_in_hours=1)

            Here, duration_in_hours increments while binary_variable is 1. Minimum and maximum durations
            can be enforced to constrain how long the activity remains active.

        Notes:
            - To count consecutive zeros instead of ones, use a transformed binary variable
              (e.g., `1 - binary_variable`).
            - Constraints ensure the duration variable properly resets or increments based on activity.

        Raises:
            AssertionError: If the binary_variable is None, indicating the duration constraints cannot be applied.

        """
        try:
            previous_duration: Skalar = self.get_consecutive_duration(
                binary_variable.previous_values, system_model.previous_dt_in_hours
            )
        except TypeError as e:
            raise TypeError(f'The consecutive_duration of "{variable_label}" could not be calculated. {e}') from e
        mega = system_model.dt_in_hours_total + previous_duration

        if maximum_duration is not None:
            first_step_max: Skalar = (
                maximum_duration.active_data[0] if maximum_duration.is_array else maximum_duration.active_data
            )
            if previous_duration + system_model.dt_in_hours[0] > first_step_max:
                logger.warning(
                    f'The maximum duration of "{variable_label}" is set to {maximum_duration.active_data}h, '
                    f'but the consecutive_duration previous to this model is {previous_duration}h. '
                    f'This forces "{binary_variable.label} = 0" in the first time step '
                    f'(dt={system_model.dt_in_hours[0]}h)!'
                )

        duration_in_hours = create_variable(
            variable_label,
            self,
            system_model.nr_of_time_steps,
            lower_bound=0,
            upper_bound=maximum_duration.active_data if maximum_duration is not None else mega,
            previous_values=previous_duration,
        )
        label_prefix = duration_in_hours.label

        assert binary_variable is not None, f'Duration Variable of {self.element} must be defined to add constraints'
        # TODO: Einfachere Variante von Peter umsetzen!

        # 1) eq: duration(t) - On(t) * BIG <= 0
        constraint_1 = create_equation(f'{label_prefix}_constraint_1', self, eq_type='ineq')
        constraint_1.add_summand(duration_in_hours, 1)
        constraint_1.add_summand(binary_variable, -1 * mega)

        # 2a) eq: duration(t) - duration(t-1) <= dt(t)
        #    on(t)=1 -> duration(t) - duration(t-1) <= dt(t)
        #    on(t)=0 -> duration(t-1) >= negat. value
        constraint_2a = create_equation(f'{label_prefix}_constraint_2a', self, eq_type='ineq')
        constraint_2a.add_summand(duration_in_hours, 1, time_indices[1:])  # duration(t)
        constraint_2a.add_summand(duration_in_hours, -1, time_indices[0:-1])  # duration(t-1)
        constraint_2a.add_constant(system_model.dt_in_hours[1:])  # dt(t)

        # 2b) eq: dt(t) - BIG * ( 1-On(t) ) <= duration(t) - duration(t-1)
        # eq: -duration(t) + duration(t-1) + On(t) * BIG <= -dt(t) + BIG
        # with BIG = dt_in_hours_total.
        #   on(t)=1 -> duration(t)- duration(t-1) >= dt(t)
        #   on(t)=0 -> duration(t)- duration(t-1) >= negat. value

        constraint_2b = create_equation(f'{label_prefix}_constraint_2b', self, eq_type='ineq')
        constraint_2b.add_summand(duration_in_hours, -1, time_indices[1:])  # duration(t)
        constraint_2b.add_summand(duration_in_hours, 1, time_indices[0:-1])  # duration(t-1)
        constraint_2b.add_summand(binary_variable, mega, time_indices[1:])  # on(t)
        constraint_2b.add_constant(-1 * system_model.dt_in_hours[1:] + mega)  # dt(t)

        # 3) check minimum_duration before switchOff-step

        if minimum_duration is not None:
            # Note: switchOff-step is when: On(t) - On(t+1) == 1
            # Note: (last on-time period (with last timestep of period t=n) is not checked and can be shorter)
            # Note: (previous values before t=1 are not recognised!)
            # eq: duration(t) >= minimum_duration(t) * [On(t) - On(t+1)] for t=1..(n-1)
            # eq: -duration(t) + minimum_duration(t) * On(t) - minimum_duration(t) * On(t+1) <= 0
            if minimum_duration.is_scalar:
                minimum_duration_used = minimum_duration.active_data
            else:
                minimum_duration_used = minimum_duration.active_data[0:-1]  # only checked for t=1...(n-1)
            eq_min_duration = create_equation(f'{label_prefix}_minimum_duration', self, eq_type='ineq')
            eq_min_duration.add_summand(duration_in_hours, -1, time_indices[0:-1])  # -duration(t)
            eq_min_duration.add_summand(
                binary_variable, -1 * minimum_duration_used, time_indices[1:]
            )  # - minimum_duration (t) * On(t+1)
            eq_min_duration.add_summand(
                binary_variable, minimum_duration_used, time_indices[0:-1]
            )  # minimum_duration * On(t)

            first_step_min: Skalar = (
                minimum_duration.active_data[0] if minimum_duration.is_array else minimum_duration.active_data
            )
            if 0 < duration_in_hours.previous_values < first_step_min:
                # Force the first step to be = 1, if the minimum_duration is not reached in previous_values
                # Note: Only if the previous consecutive_duration is smaller than the minimum duration,
                # and the previous_values is greater 0!
                # eq: duration(t=0) = duration(t=-1) + dt(0)
                eq_min_duration_inital = create_equation(f'{label_prefix}_minimum_duration_inital', self, eq_type='eq')
                eq_min_duration_inital.add_summand(binary_variable, 1, time_indices[0])
                eq_min_duration_inital.add_constant(1)

        # 4) first index:
        # eq: duration(t=0)= dt(0) * On(0)
        first_index = time_indices[0]  # only first element
        eq_first = create_equation(f'{label_prefix}_initial', self)
        eq_first.add_summand(duration_in_hours, 1, first_index)
        eq_first.add_summand(
            binary_variable,
            -1 * (system_model.dt_in_hours[first_index] + duration_in_hours.previous_values),
            first_index,
        )

        return duration_in_hours

    def _add_switch_constraints(self, system_model: SystemModel):
        assert self.switch_on is not None, f'Switch On Variable of {self.element} must be defined to add constraints'
        assert self.switch_off is not None, f'Switch Off Variable of {self.element} must be defined to add constraints'
        assert self.nr_switch_on is not None, (
            f'Nr of Switch On Variable of {self.element} must be defined to add constraints'
        )
        assert self.on is not None, f'On Variable of {self.element} must be defined to add constraints'
        # % Schaltänderung aus On-Variable
        # % SwitchOn(t)-SwitchOff(t) = On(t)-On(t-1)
        eq_switch = create_equation('Switch', self)
        eq_switch.add_summand(self.switch_on, 1, system_model.indices[1:])  # SwitchOn(t)
        eq_switch.add_summand(self.switch_off, -1, system_model.indices[1:])  # SwitchOff(t)
        eq_switch.add_summand(self.on, -1, system_model.indices[1:])  # On(t)
        eq_switch.add_summand(self.on, +1, system_model.indices[0:-1])  # On(t-1)

        # Initital switch on
        # eq: SwitchOn(t=0)-SwitchOff(t=0) = On(t=0) - On(t=-1)
        eq_initial_switch = create_equation('Initial_Switch', self)
        eq_initial_switch.add_summand(self.switch_on, 1, indices_of_variable=0)  # SwitchOn(t=0)
        eq_initial_switch.add_summand(self.switch_off, -1, indices_of_variable=0)  # SwitchOff(t=0)
        eq_initial_switch.add_summand(self.on, -1, indices_of_variable=0)  # On(t=0)
        eq_initial_switch.add_constant(-1 * self.on.previous_values[-1])  # On(t-1)

        ## Entweder SwitchOff oder SwitchOn
        # eq: SwitchOn(t) + SwitchOff(t) <= 1.1
        eq_switch_on_or_off = create_equation('Switch_On_or_Off', self, eq_type='ineq')
        eq_switch_on_or_off.add_summand(self.switch_on, 1)
        eq_switch_on_or_off.add_summand(self.switch_off, 1)
        eq_switch_on_or_off.add_constant(1.1)

        ## Anzahl Starts:
        # eq: nrSwitchOn = sum(SwitchOn(t))
        eq_nr_switch_on = create_equation('NrSwitchOn', self)
        eq_nr_switch_on.add_summand(self.nr_switch_on, 1)
        eq_nr_switch_on.add_summand(self.switch_on, -1, as_sum=True)

    def _create_shares(self, system_model: SystemModel):
        # Anfahrkosten:
        effect_collection = system_model.effect_collection_model
        effects_per_switch_on = self._on_off_parameters.effects_per_switch_on
        if effects_per_switch_on != {}:
            effect_collection.add_share_to_operation(
                'switch_on_effects', self.element, effects_per_switch_on, 1, self.switch_on
            )

        # Betriebskosten:
        effects_per_running_hour = self._on_off_parameters.effects_per_running_hour
        if effects_per_running_hour != {}:
            effect_collection.add_share_to_operation(
                'running_hour_effects', self.element, effects_per_running_hour, system_model.dt_in_hours, self.on
            )

    def _previous_on_values(self, epsilon: float = 1e-5) -> np.ndarray:
        """
        Returns the previous 'on' states of defining variables as a binary array.

        Parameters:
        ----------
        epsilon : float, optional
            Tolerance for equality to determine "off" state, default is 1e-5.

        Returns:
        -------
        np.ndarray
            A binary array (0 and 1) indicating the previous on/off states of the variables.
            Returns `array([0])` if no previous values are available.
        """
        previous_values = [var.previous_values for var in self._defining_variables if var.previous_values is not None]

        if not previous_values:
            return np.array([0])
        else:  # Convert to 2D-array and compute binary on/off states
            previous_values = np.array(previous_values)
            if previous_values.ndim > 1:
                return np.any(~np.isclose(previous_values, 0, atol=epsilon), axis=0).astype(int)
            else:
                return (~np.isclose(previous_values, 0, atol=epsilon)).astype(int)

    @classmethod
    def get_consecutive_duration(
        cls, binary_values: Union[int, np.ndarray], dt_in_hours: Union[int, float, np.ndarray]
    ) -> Skalar:
        """
        Returns the current consecutive duration in hours, computed from binary values.
        If only one binary value is availlable, the last dt_in_hours is used.
        Of both binary_values and dt_in_hours are arrays, checks that the length of dt_in_hours has at least as
        many elements as the last  consecutive duration in binary_values.

        Parameters
        ----------
        binary_values : int, np.ndarray
            An int or 1D binary array containing only `0`s and `1`s.
        dt_in_hours : int, float, np.ndarray
            The duration of each time step in hours.

        Returns
        -------
        np.ndarray
            The duration of the binary variable in hours.

        Raises
        ------
        TypeError
            If the length of binary_values and dt_in_hours is not equal, but None is a scalar.
        """
        if np.isscalar(binary_values) and np.isscalar(dt_in_hours):
            return binary_values * dt_in_hours
        elif np.isscalar(binary_values) and not np.isscalar(dt_in_hours):
            return binary_values * dt_in_hours[-1]

        # Find the indexes where value=`0` in a 1D-array
        zero_indices = np.where(np.isclose(binary_values, 0, atol=CONFIG.modeling.EPSILON))[0]
        length_of_last_duration = zero_indices[-1] + 1 if zero_indices.size > 0 else len(binary_values)

        if not np.isscalar(binary_values) and np.isscalar(dt_in_hours):
            return np.sum(binary_values[-length_of_last_duration:] * dt_in_hours)

        elif not np.isscalar(binary_values) and not np.isscalar(dt_in_hours):
            if length_of_last_duration > len(dt_in_hours):  # check that lengths are compatible
                raise TypeError(
                    f'When trying to calculate the consecutive duration, the length of the last duration '
                    f'({len(length_of_last_duration)}) is longer than the dt_in_hours ({len(dt_in_hours)}), '
                    f'as {binary_values=}'
                )
            return np.sum(binary_values[-length_of_last_duration:] * dt_in_hours[-length_of_last_duration:])

        else:
            raise Exception(
                f'Unexpected state reached in function get_consecutive_duration(). binary_values={binary_values}; '
                f'dt_in_hours={dt_in_hours}'
            )


class SegmentModel(ElementModel):
    """Class for modeling a linear segment of one or more variables in parallel"""

    def __init__(
        self,
        element: Element,
        segment_index: Union[int, str],
        sample_points: Dict[Variable, Tuple[Union[Numeric, TimeSeries], Union[Numeric, TimeSeries]]],
        as_time_series: bool = True,
    ):
        super().__init__(element, f'Segment_{segment_index}')
        self.element = element
        self.in_segment: Optional[VariableTS] = None
        self.lambda0: Optional[VariableTS] = None
        self.lambda1: Optional[VariableTS] = None

        self._segment_index = segment_index
        self._as_time_series = as_time_series
        self.sample_points = sample_points

    def do_modeling(self, system_model: SystemModel):
        length = system_model.nr_of_time_steps if self._as_time_series else 1
        self.in_segment = create_variable('inSegment', self, length, is_binary=True)
        self.lambda0 = create_variable('lambda0', self, length, lower_bound=0, upper_bound=1)  # Wertebereich 0..1
        self.lambda1 = create_variable('lambda1', self, length, lower_bound=0, upper_bound=1)  # Wertebereich 0..1

        # eq: -aSegment.onSeg(t) + aSegment.lambda1(t) + aSegment.lambda2(t)  = 0
        equation = create_equation('inSegment', self)

        equation.add_summand(self.in_segment, -1)
        equation.add_summand(self.lambda0, 1)
        equation.add_summand(self.lambda1, 1)


class MultipleSegmentsModel(ElementModel):
    # TODO: Length...
    def __init__(
        self,
        element: Element,
        sample_points: Dict[Variable, List[Tuple[Numeric, Numeric]]],
        can_be_outside_segments: Optional[Union[bool, Variable]],
        as_time_series: bool = True,
        label: str = 'MultipleSegments',
    ):
        """
        can_be_outside_segments:    True -> Variable gets created;
                                    False or None -> No Variable gets_created;
                                    Variable -> the Variable gets used
        """
        super().__init__(element, label)
        self.element = element

        self.outside_segments: Optional[VariableTS] = None

        self._as_time_series = as_time_series
        self._can_be_outside_segments = can_be_outside_segments
        self._sample_points = sample_points
        self._segment_models: List[SegmentModel] = []

    def do_modeling(self, system_model: SystemModel):
        restructured_variables_with_segments: List[Dict[Variable, Tuple[Numeric, Numeric]]] = [
            {key: values[i] for key, values in self._sample_points.items()} for i in range(self._nr_of_segments)
        ]

        self._segment_models = [
            SegmentModel(self.element, i, sample_points, self._as_time_series)
            for i, sample_points in enumerate(restructured_variables_with_segments)
        ]

        self.sub_models.extend(self._segment_models)

        for segment_model in self._segment_models:
            segment_model.do_modeling(system_model)

        #  eq: - v(t) + (v_0_0 * lambda_0_0 + v_0_1 * lambda_0_1) + (v_1_0 * lambda_1_0 + v_1_1 * lambda_1_1) ... = 0
        #  -> v_0_0, v_0_1 = Stützstellen des Segments 0
        for variable in self._sample_points.keys():
            lambda_eq = create_equation(f'lambda_{variable.label}', self)
            lambda_eq.add_summand(variable, -1)
            for segment_model in self._segment_models:
                lambda_eq.add_summand(segment_model.lambda0, segment_model.sample_points[variable][0])
                lambda_eq.add_summand(segment_model.lambda1, segment_model.sample_points[variable][1])

        # a) eq: Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 1                Aufenthalt nur in Segmenten erlaubt
        # b) eq: -On(t) + Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 0       zusätzlich kann alles auch Null sein
        in_single_segment = create_equation('in_single_Segment', self)
        for segment_model in self._segment_models:
            in_single_segment.add_summand(segment_model.in_segment, 1)

        # a) or b) ?
        if isinstance(self._can_be_outside_segments, Variable):  # Use existing Variable
            self.outside_segments = self._can_be_outside_segments
            in_single_segment.add_summand(self.outside_segments, -1)
        elif self._can_be_outside_segments is True:  # Create Variable
            length = system_model.nr_of_time_steps if self._as_time_series else 1
            self.outside_segments = create_variable('outside_segments', self, length, is_binary=True)
            in_single_segment.add_summand(self.outside_segments, -1)
        else:  # Dont allow outside Segments
            in_single_segment.add_constant(1)

    @property
    def _nr_of_segments(self):
        return len(next(iter(self._sample_points.values())))


class ShareAllocationModel(ElementModel):
    def __init__(
        self,
        element: Element,
        label: str,
        shares_are_time_series: bool,
        total_max: Optional[Skalar] = None,
        total_min: Optional[Skalar] = None,
        max_per_hour: Optional[Numeric] = None,
        min_per_hour: Optional[Numeric] = None,
    ):
        super().__init__(element, label)
        if not shares_are_time_series:  # If the condition is True
            assert max_per_hour is None and min_per_hour is None, (
                'Both max_per_hour and min_per_hour cannot be used when shares_are_time_series is False'
            )
        self.element = element
        self.sum_TS: Optional[VariableTS] = None
        self.sum: Optional[Variable] = None
        self.shares: Dict[str, Variable] = {}

        self._eq_time_series: Optional[Equation] = None
        self._eq_sum: Optional[Equation] = None

        # Parameters
        self._shares_are_time_series = shares_are_time_series
        self._total_max = total_max
        self._total_min = total_min
        self._max_per_hour = max_per_hour
        self._min_per_hour = min_per_hour

    def do_modeling(self, system_model: SystemModel):
        self.sum = create_variable(
            f'{self.label}_sum', self, 1, lower_bound=self._total_min, upper_bound=self._total_max
        )
        # eq: sum = sum(share_i) # skalar
        self._eq_sum = create_equation(f'{self.label}_sum', self)
        self._eq_sum.add_summand(self.sum, -1)

        if self._shares_are_time_series:
            lb_ts = None if (self._min_per_hour is None) else np.multiply(self._min_per_hour, system_model.dt_in_hours)
            ub_ts = None if (self._max_per_hour is None) else np.multiply(self._max_per_hour, system_model.dt_in_hours)
            self.sum_TS = create_variable(
                f'{self.label}_sum_TS', self, system_model.nr_of_time_steps, lower_bound=lb_ts, upper_bound=ub_ts
            )

            # eq: sum_TS = sum(share_TS_i) # TS
            self._eq_time_series = create_equation(f'{self.label}_time_series', self)
            self._eq_time_series.add_summand(self.sum_TS, -1)

            # eq: sum = sum(sum_TS(t)) # additionaly to self.sum
            self._eq_sum.add_summand(self.sum_TS, 1, as_sum=True)

    def add_share(
        self,
        system_model: SystemModel,
        name_of_share: str,
        variable: Optional[Variable],
        factor: Numeric,
        share_as_sum: bool = False,
    ):
        """
        Adding a Share to a Share Allocation Model.
        """
        # TODO: accept only one factor or accept unlimited factors -> *factors

        # Check to which equation the share should be added
        if share_as_sum or not self._shares_are_time_series:
            target_eq = self._eq_sum
        else:
            target_eq = self._eq_time_series

        new_share = SingleShareModel(self.element, name_of_share, variable, factor, share_as_sum)
        target_eq.add_summand(new_share.single_share, 1)

        self.sub_models.append(new_share)
        assert new_share.label not in self.shares, (
            f'A Share with the label {new_share.label} was already present in {self.label}'
        )
        self.shares[new_share.label] = new_share.single_share

    def results(self):
        return {
            **{variable.label_short: variable.result for variable in self.variables.values()},
            **{'Shares': {variable.label_short: variable.result for variable in self.shares.values()}},
        }


class SingleShareModel(ElementModel):
    """Holds a Variable and an Equation. Summands can be added to the Equation. Used to publish Shares"""

    def __init__(self, element: Element, name: str, variable: Optional[Variable], factor: Numeric, share_as_sum: bool):
        super().__init__(element, name)
        if variable is not None:
            assert not (variable.length == 1 and share_as_sum), 'A Variable with the length 1 cannot be summed up!'

        if (
            share_as_sum
            or (variable is not None and variable.length == 1)
            or (variable is None and np.isscalar(factor))
        ):
            self.single_share = Variable(self.label_full, 1, self.label)
        elif variable is not None:
            self.single_share = VariableTS(self.label_full, variable.length, self.label)
        else:
            raise Exception('This case is not yet covered for a SingleShareModel')

        self.add_variables(self.single_share)
        self.single_equation = create_equation(self.label_full, self)
        self.single_equation.add_summand(self.single_share, -1)

        if variable is None:
            self.single_equation.add_constant(-1 * np.sum(factor) if share_as_sum else -1 * factor)
        else:
            self.single_equation.add_summand(variable, factor, as_sum=share_as_sum)


class SegmentedSharesModel(ElementModel):
    # TODO: Length...
    def __init__(
        self,
        element: Element,
        variable_segments: Tuple[Variable, List[Tuple[Skalar, Skalar]]],
        share_segments: Dict['Effect', List[Tuple[Skalar, Skalar]]],
        can_be_outside_segments: Optional[Union[bool, Variable]],
        label: str = 'SegmentedShares',
    ):
        super().__init__(element, label)
        assert len(variable_segments[1]) == len(list(share_segments.values())[0]), (
            'Segment length of variable_segments and share_segments must be equal'
        )
        self.element: Element
        self._can_be_outside_segments = can_be_outside_segments
        self._variable_segments = variable_segments
        self._share_segments = share_segments
        self._shares: Optional[Dict['Effect', SingleShareModel]] = None
        self._segments_model: Optional[MultipleSegmentsModel] = None
        self._as_tme_series: bool = isinstance(self._variable_segments[0], VariableTS)

    def do_modeling(self, system_model: SystemModel):
        length = system_model.nr_of_time_steps if self._as_tme_series else 1
        self._shares = {
            effect: create_variable(f'{effect.label}_segmented', self, length) for effect in self._share_segments
        }

        segments: Dict[Variable, List[Tuple[Skalar, Skalar]]] = {
            **{self._shares[effect]: segment for effect, segment in self._share_segments.items()},
            **{self._variable_segments[0]: self._variable_segments[1]},
        }

        self._segments_model = MultipleSegmentsModel(
            self.element,
            segments,
            can_be_outside_segments=self._can_be_outside_segments,
            as_time_series=self._as_tme_series,
        )
        self._segments_model.do_modeling(system_model)
        self.sub_models.append(self._segments_model)

        # Shares
        effect_collection = system_model.effect_collection_model
        for effect, variable in self._shares.items():
            if self._as_tme_series:
                effect_collection.add_share_to_operation(
                    name='segmented_effects',
                    element=self.element,
                    effect_values={effect: 1},
                    factor=1,
                    variable=variable,
                )
            else:
                effect_collection.add_share_to_invest(
                    name='segmented_effects',
                    element=self.element,
                    effect_values={effect: 1},
                    factor=1,
                    variable=variable,
                )


class PreventSimultaneousUsageModel(ElementModel):
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

    def __init__(self, element: Element, variables: List[VariableTS], label: str = 'PreventSimultaneousUsage'):
        super().__init__(element, label)
        self._variables = variables
        assert len(self._variables) >= 2, f'Model {self.__class__.__name__} must get at least two variables'
        for variable in self._variables:  # classic
            assert variable.is_binary, f'Variable {variable} must be binary for use in {self.__class__.__name__}'

    def do_modeling(self, system_model: SystemModel):
        # eq: sum(flow_i.on(t)) <= 1.1 (1 wird etwas größer gewählt wg. Binärvariablengenauigkeit)
        eq = create_equation('prevent_simultaneous_use', self, eq_type='ineq')
        for variable in self._variables:
            eq.add_summand(variable, 1)
        eq.add_constant(1.1)
