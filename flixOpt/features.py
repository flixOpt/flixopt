"""
This module contains the features of the flixOpt framework.
Features extend the functionality of Elements.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import linopy
import numpy as np

from . import utils
from .config import CONFIG
from .core import NumericData, Scalar, TimeSeries
from .interface import InvestParameters, OnOffParameters
from .structure import Model, SystemModel

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from .effects import Effect


logger = logging.getLogger('flixOpt')


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

        self._segments: Optional[SegmentedSharesModel] = None

        self._on_variable = on_variable
        self._defining_variable = defining_variable
        self._relative_bounds_of_defining_variable = relative_bounds_of_defining_variable
        self.parameters = parameters

    def do_modeling(self):
        if self.parameters.fixed_size and not self.parameters.optional:
            self.size = self.add(self._model.add_variables(
                lower=self.parameters.fixed_size,
                upper=self.parameters.fixed_size,
                name=f'{self.label_full}|size'),
                'size')
        else:
            self.size = self.add(self._model.add_variables(
                lower=0 if self.parameters.optional else self.parameters.minimum_size,
                upper=self.parameters.maximum_size,
                name=f'{self.label_full}|size'),
                'size')

        # Optional
        if self.parameters.optional:
            self.is_invested = self.add(self._model.add_variables(
                binary=True,
                name=f'{self.label_full}|is_invested'),
                'is_invested')

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
                expressions={effect: self.is_invested * factor if self.is_invested is not None else factor
                             for effect, factor in fix_effects.items()},
                target='invest',
            )

        if self.parameters.divest_effects != {} and self.parameters.optional:
            # share: divest_effects - isInvested * divest_effects
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={effect: -self.is_invested * factor + factor for effect, factor in fix_effects.items()},
                target='invest',
            )

        if self.parameters.specific_effects != {}:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={effect: self.size * factor for effect, factor in self.parameters.specific_effects.items()},
                target='invest',
            )

        if self.parameters.effects_in_segments:
            self._segments = self.add(
                SegmentedSharesModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    variable_segments=(self.size, self.parameters.effects_in_segments[0]),
                    share_segments=self.parameters.effects_in_segments[1],
                    can_be_outside_segments=self.is_invested),
                'segments'
            )
            self._segments.do_modeling()

    def _create_bounds_for_optional_investment(self):
        if self.parameters.fixed_size:
            # eq: investment_size = isInvested * fixed_size
            self.add(self._model.add_constraints(
                self.size == self.is_invested * self.parameters.fixed_size,
                name=f'{self.label_full}|is_invested'),
                'is_invested')

        else:
            # eq1: P_invest <= isInvested * investSize_max
            self.add(self._model.add_constraints(
                self.size <= self.is_invested * self.parameters.maximum_size,
                name=f'{self.label_full}|is_invested_ub'),
                'is_invested_ub')

            # eq2: P_invest >= isInvested * max(epsilon, investSize_min)
            self.add(self._model.add_constraints(
                self.size >= self.is_invested * np.maximum(CONFIG.modeling.EPSILON,self.parameters.minimum_size),
                name=f'{self.label_full}|is_invested_lb'),
                'is_invested_lb')

    def _create_bounds_for_defining_variable(self):
        variable = self._defining_variable
        lb_relative, ub_relative = self._relative_bounds_of_defining_variable
        if np.all(lb_relative == ub_relative):
            self.add(self._model.add_constraints(
                variable == self.size * ub_relative,
                name=f'{self.label_full}|fix_{variable.name}'),
                f'fix_{variable.name}')
            if self._on_variable is not None:
                raise ValueError(f'Flow {self.label} has a fixed relative flow rate and an on_variable.'
                                 f'This combination is currently not supported.')
            return

        # eq: defining_variable(t)  <= size * upper_bound(t)
        self.add(self._model.add_constraints(
            variable <= self.size * ub_relative,
            name=f'{self.label_full}|ub_{variable.name}'),
            f'ub_{variable.name}')

        if self._on_variable is None:
            # eq: defining_variable(t) >= investment_size * relative_minimum(t)
            self.add(self._model.add_constraints(
                variable >= self.size * lb_relative,
                name=f'{self.label_full}|lb_{variable.name}'),
                f'lb_{variable.name}')
        else:
            ## 2. Gleichung: Minimum durch Investmentgröße und On
            # eq: defining_variable(t) >= mega * (On(t)-1) + size * relative_minimum(t)
            #     ... mit mega = relative_maximum * maximum_size
            # äquivalent zu:.
            # eq: - defining_variable(t) + mega * On(t) + size * relative_minimum(t) <= + mega
            mega = lb_relative * self.parameters.maximum_size
            on = self._on_variable
            self.add(self._model.add_constraints(
                variable >= mega * (on - 1) + self.size * lb_relative,
                name=f'{self.label_full}|lb_{variable.name}'),
                f'lb_{variable.name}')
            # anmerkung: Glg bei Spezialfall relative_minimum = 0 redundant zu OnOff ??


class OnOffModel(Model):
    """
    Class for modeling the on and off state of a variable
    If defining_bounds are given, creates sufficient lower bounds
    """

    def __init__(
        self,
        model: SystemModel,
        on_off_parameters: OnOffParameters,
        label_of_element: str,
        defining_variables: List[linopy.Variable],
        defining_bounds: List[Tuple[NumericData, NumericData]],
        previous_values: List[Optional[NumericData]],
        label: Optional[str] = None,
    ):
        """
        Constructor for OnOffModel

        Parameters
        ----------
        model: SystemModel
            Reference to the SystemModel
        on_off_parameters: OnOffParameters
            Parameters for the OnOffModel
        label_of_element:
            Label of the Parent
        defining_variables:
            List of Variables that are used to define the OnOffModel
        defining_bounds:
            List of Tuples, defining the absolute bounds of each defining variable
        previous_values:
            List of previous values of the defining variables
        label:
            Label of the OnOffModel
        """
        super().__init__(model, label_of_element, label)
        assert len(defining_variables) == len(defining_bounds), 'Every defining Variable needs bounds to Model OnOff'
        self.parameters = on_off_parameters
        self._defining_variables = defining_variables
        # Ensure that no lower bound is below a certain threshold
        self._defining_bounds = [(np.maximum(lb, CONFIG.modeling.EPSILON), ub) for lb, ub in defining_bounds]
        self._previous_values = previous_values

        self.on: Optional[linopy.Variable] = None
        self.total_on_hours: Optional[linopy.Variable] = None

        self.consecutive_on_hours: Optional[linopy.Variable] = None
        self.consecutive_off_hours: Optional[linopy.Variable] = None

        self.off: Optional[linopy.Variable] = None

        self.switch_on: Optional[linopy.Variable] = None
        self.switch_off: Optional[linopy.Variable] = None
        self.switch_on_nr: Optional[linopy.Variable] = None

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
                lower=self.parameters.on_hours_total_min if self.parameters.on_hours_total_min is not None else 0,
                upper=self.parameters.on_hours_total_max if self.parameters.on_hours_total_max is not None else np.inf,
                name=f'{self.label_full}|on_hours_total'
            ),
            'on_hours_total'
        )

        self.add(
            self._model.add_constraints(
                self.total_on_hours == (self.on * self._model.hours_per_step).sum(),
                name=f'{self.label_full}|on_hours_total'
            ),
            'on_hours_total'
        )

        self._add_on_constraints()

        if self.parameters.use_off:
            self.off = self.add(
                self._model.add_variables(
                    name=f'{self.label_full}|off',
                    binary=True,
                    coords=self._model.coords,
                ),
                'off'
            )

            # eq: var_on(t) + var_off(t) = 1
            self.add(self._model.add_constraints(self.on + self.off == 1, name=f'{self.label_full}|off'), 'off')

        if self.parameters.use_consecutive_on_hours:
            self.consecutive_on_hours = self._get_duration_in_hours(
                'consecutive_on_hours',
                self.on,
                self.previous_consecutive_on_hours,
                self.parameters.consecutive_on_hours_min,
                self.parameters.consecutive_on_hours_max,
            )

        if self.parameters.use_consecutive_off_hours:
            self.consecutive_off_hours = self._get_duration_in_hours(
                'consecutive_off_hours',
                self.off,
                self.previous_consecutive_off_hours,
                self.parameters.consecutive_off_hours_min,
                self.parameters.consecutive_off_hours_max,
            )

        if self.parameters.use_switch_on:
            self.switch_on = self.add(self._model.add_variables(
                binary=True, name=f'{self.label_full}|switch_on', coords=self._model.coords),'switch_on')

            self.switch_off = self.add(self._model.add_variables(
                binary=True, name=f'{self.label_full}|switch_off', coords=self._model.coords), 'switch_off')

            self.switch_on_nr = self.add(self._model.add_variables(
                upper=self.parameters.switch_on_total_max if self.parameters.switch_on_total_max is not None else np.inf,
                name=f'{self.label_full}|switch_on_nr'),
                'switch_on_nr')

            self._add_switch_constraints()

        self._create_shares()

    def _add_on_constraints(self):
        assert self.on is not None, f'On variable of {self.label_full} must be defined to add constraints'
        # % Bedingungen 1) und 2) müssen erfüllt sein:

        # % Anmerkung: Falls "abschnittsweise linear" gewählt, dann ist eigentlich nur Bedingung 1) noch notwendig
        # %            (und dann auch nur wenn erstes Segment bei Q_th=0 beginnt. Dann soll bei Q_th=0 (d.h. die Maschine ist Aus) On = 0 und segment1.onSeg = 0):)
        # %            Fazit: Wenn kein Performance-Verlust durch mehr Gleichungen, dann egal!

        nr_of_def_vars = len(self._defining_variables)
        assert nr_of_def_vars > 0, 'Achtung: mindestens 1 Flow notwendig'

        if nr_of_def_vars == 1:
            def_var = self._defining_variables[0]
            lb, ub = self._defining_bounds[0]

            # eq: On(t) * max(epsilon, lower_bound) <= Q_th(t)
            self.add(
                self._model.add_constraints(
                    self.on * np.maximum(CONFIG.modeling.EPSILON, lb) <= def_var,
                    name=f'{self.label_full}|on_con1'
                ),
                'on_con1'
            )

            # eq: Q_th(t) <= Q_th_max * On(t)
            self.add(
                self._model.add_constraints(
                    self.on * np.maximum(CONFIG.modeling.EPSILON, ub) >= def_var,
                    name=f'{self.label_full}|on_con2'
                ),
                'on_con2'
            )

        else:  # Bei mehreren Leistungsvariablen:
            ub = sum(bound[1] for bound in self._defining_bounds)
            lb = CONFIG.modeling.EPSILON

            # When all defining variables are 0, On is 0
            # eq: On(t) * Epsilon <= sum(alle Leistungen(t))
            self.add(
                self._model.add_constraints(
                    self.on * lb <= sum(self._defining_variables),
                    name=f'{self.label_full}|on_con1'
                ),
                'on_con1'
            )

            ## sum(alle Leistung) >0 -> On = 1|On=0 -> sum(Leistung)=0
            #  eq: sum( Leistung(t,i))              - sum(Leistung_max(i))             * On(t) <= 0
            #  --> damit Gleichungswerte nicht zu groß werden, noch durch nr_of_flows geteilt:
            #  eq: sum( Leistung(t,i) / nr_of_flows ) - sum(Leistung_max(i)) / nr_of_flows * On(t) <= 0
            self.add(
                self._model.add_constraints(
                    self.on * ub >= sum([def_var / nr_of_def_vars for def_var in self._defining_variables]),
                    name=f'{self.label_full}|on_con2'
                ),
                'on_con2'
            )

        if np.max(ub) > CONFIG.modeling.BIG_BINARY_BOUND:
            logger.warning(
                f'In "{self.label_full}", a binary definition was created with a big upper bound '
                f'({np.max(ub)}). This can lead to wrong results regarding the on and off variables. '
                f'Avoid this warning by reducing the size of {self.label_full} '
                f'(or the maximum_size of the corresponding InvestParameters). '
                f'If its a Component, you might need to adjust the sizes of all of its flows.'
            )

    def _get_duration_in_hours(
        self,
        variable_name: str,
        binary_variable: linopy.Variable,
        previous_duration: Scalar,
        minimum_duration: Optional[TimeSeries],
        maximum_duration: Optional[TimeSeries],
    ) -> linopy.Variable:
        """
        creates duration variable and adds constraints to a time-series variable to enforce duration limits based on
        binary activity.
        The minimum duration in the last time step is not restricted.
        Previous values before t=0 are not recognised!

        Parameters:
            variable_label (str):
                Label for the duration variable to be created.
            binary_variable (linopy.Variable):
                Time-series binary variable (e.g., [0, 0, 1, 1, 1, 0, ...]) representing activity states.
            minimum_duration (Optional[TimeSeries]):
                Minimum duration the activity must remain active once started.
                If None, no minimum duration constraint is applied.
            maximum_duration (Optional[TimeSeries]):
                Maximum duration the activity can remain active.
                If None, the maximum duration is set to the total available time.

        Returns:
            linopy.Variable: The created duration variable representing consecutive active durations.

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
        assert binary_variable is not None, f'Duration Variable of {self.label_full} must be defined to add constraints'

        mega = self._model.hours_per_step.sum() + previous_duration

        if maximum_duration is not None:
            first_step_max: Scalar = maximum_duration.isel(time=0)

            if previous_duration + self._model.hours_per_step[0] > first_step_max:
                logger.warning(
                    f'The maximum duration of "{variable_name}" is set to {maximum_duration.active_data}h, '
                    f'but the consecutive_duration previous to this model is {previous_duration}h. '
                    f'This forces "{binary_variable.name} = 0" in the first time step '
                    f'(dt={self._model.hours_per_step[0]}h)!'
                )

        duration_in_hours = self.add(self._model.add_variables(
            lower=0,
            upper=maximum_duration.active_data if maximum_duration is not None else mega,
            coords=self._model.coords,
            name=f'{self.label_full}|{variable_name}'),
            variable_name
        )

        # 1) eq: duration(t) - On(t) * BIG <= 0
        self.add(self._model.add_constraints(
            duration_in_hours <= binary_variable * mega,
            name=f'{self.label_full}|{variable_name}_con1'),
            f'{variable_name}_con1'
        )

        # 2a) eq: duration(t) - duration(t-1) <= dt(t)
        #    on(t)=1 -> duration(t) - duration(t-1) <= dt(t)
        #    on(t)=0 -> duration(t-1) >= negat. value
        self.add(self._model.add_constraints(
            duration_in_hours.isel(time=slice(1, None))
            <=
            duration_in_hours.isel(time=slice(None, -1)) + self._model.hours_per_step.isel(time=slice(None, -1)),
            name=f'{self.label_full}|{variable_name}_con2a'),
            f'{variable_name}_con2a'
        )

        # 2b) eq: dt(t) - BIG * ( 1-On(t) ) <= duration(t) - duration(t-1)
        # eq: -duration(t) + duration(t-1) + On(t) * BIG <= -dt(t) + BIG
        # with BIG = dt_in_hours_total.
        #   on(t)=1 -> duration(t)- duration(t-1) >= dt(t)
        #   on(t)=0 -> duration(t)- duration(t-1) >= negat. value

        self.add(self._model.add_constraints(
            duration_in_hours.isel(time=slice(1, None))
            >=
            duration_in_hours.isel(time=slice(None, -1)) + self._model.hours_per_step.isel(time=slice(None, -1))
            + (binary_variable.isel(time=slice(1, None)) - 1) * mega,
            name=f'{self.label_full}|{variable_name}_con2b'),
            f'{variable_name}_con2b'
        )

        # 3) check minimum_duration before switchOff-step

        if minimum_duration is not None:
            # Note: switchOff-step is when: On(t) - On(t+1) == 1
            # Note: (last on-time period (with last timestep of period t=n) is not checked and can be shorter)
            # Note: (previous values before t=1 are not recognised!)
            # eq: duration(t) >= minimum_duration(t) * [On(t) - On(t+1)] for t=1..(n-1)
            # eq: -duration(t) + minimum_duration(t) * On(t) - minimum_duration(t) * On(t+1) <= 0
            self.add(self._model.add_constraints(
                duration_in_hours
                >=
                (binary_variable.isel(time=slice(None, -1)) - binary_variable.isel(time=slice(1, None)))
                * minimum_duration.isel(time=slice(None, -1)),
                name=f'{self.label_full}|{variable_name}_minimum_duration'),
                f'{variable_name}_minimum_duration'
            )

            if 0 < previous_duration < minimum_duration.isel(time=0):
                # Force the first step to be = 1, if the minimum_duration is not reached in previous_values
                # Note: Only if the previous consecutive_duration is smaller than the minimum duration
                # and the previous_duration is greater 0!
                # eq: On(t=0) = 1
                self.add(self._model.add_constraints(
                    binary_variable.isel(time=0) == 1,
                    name=f'{self.label_full}|{variable_name}_minimum_inital'),
                    f'{variable_name}_minimum_inital'
                )

            # 4) first index:
            # eq: duration(t=0)= dt(0) * On(0)
            self.add(self._model.add_constraints(
                duration_in_hours.isel(time=0) == self._model.hours_per_step.isel(time=0) * binary_variable.isel(time=0),
                name=f'{self.label_full}|{variable_name}_initial'),
                f'{variable_name}_initial'
            )

        return duration_in_hours

    def _add_switch_constraints(self):
        assert self.switch_on is not None, f'Switch On Variable of {self.label_full} must be defined to add constraints'
        assert self.switch_off is not None, f'Switch Off Variable of {self.label_full} must be defined to add constraints'
        assert self.switch_on_nr is not None, (
            f'Nr of Switch On Variable of {self.label_full} must be defined to add constraints'
        )
        assert self.on is not None, f'On Variable of {self.label_full} must be defined to add constraints'
        # % Schaltänderung aus On-Variable
        # % SwitchOn(t)-SwitchOff(t) = On(t)-On(t-1)
        self.add(
            self._model.add_constraints(
                self.switch_on.isel(time=slice(1, None)) - self.switch_off.isel(time=slice(1, None))
                ==
                self.on.isel(time=slice(1,None)) - self.on.isel(time=slice(None,-1)),
                name=f'{self.label_full}|switch_con'
            ),
            'switch_con'
        )
        # Initital switch on
        # eq: SwitchOn(t=0)-SwitchOff(t=0) = On(t=0) - On(t=-1)
        self.add(
            self._model.add_constraints(
                self.switch_on.isel(time=0) - self.switch_off.isel(time=0)
                ==
                self.on.isel(time=0) - self.previous_on_values[-1],
                name=f'{self.label_full}|initial_switch_con'
            ),
            'initial_switch_con'
        )
        ## Entweder SwitchOff oder SwitchOn
        # eq: SwitchOn(t) + SwitchOff(t) <= 1.1
        self.add(
            self._model.add_constraints(
                self.switch_on + self.switch_off <= 1.1,
                name=f'{self.label_full}|switch_on_or_off'
            ),
            'switch_on_or_off'
        )

        ## Anzahl Starts:
        # eq: nrSwitchOn = sum(SwitchOn(t))
        self.add(
            self._model.add_constraints(
                self.switch_on_nr == self.switch_on.sum(),
                name=f'{self.label_full}|switch_on_nr'
            ),
            'switch_on_nr'
        )

    def _create_shares(self):
        # Anfahrkosten:
        effects_per_switch_on = self.parameters.effects_per_switch_on
        if effects_per_switch_on != {}:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={effect: self.switch_on * factor for effect, factor in effects_per_switch_on.items()},
                target='operation',
            )

        # Betriebskosten:
        effects_per_running_hour = self.parameters.effects_per_running_hour
        if effects_per_running_hour != {}:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={effect: self.on * factor * self._model.hours_per_step
                             for effect, factor in effects_per_running_hour.items()},
                target='operation',
            )

    @property
    def previous_on_values(self) -> np.ndarray:
        return self.compute_previous_on_states(self._previous_values)

    @property
    def previous_off_values(self) -> np.ndarray:
        return 1 - self.previous_on_values

    @property
    def previous_consecutive_on_hours(self) -> Scalar:
        return self.compute_consecutive_duration(self.previous_on_values, self._model.hours_per_step)

    @property
    def previous_consecutive_off_hours(self) -> Scalar:
        return self.compute_consecutive_duration(self.previous_off_values, self._model.hours_per_step)

    @staticmethod
    def compute_previous_on_states(previous_values: List[Optional[NumericData]], epsilon: float = 1e-5) -> np.ndarray:
        """
        Computes the previous 'on' states {0, 1} of defining variables as a binary array from their previous values.

        Parameters:
        ----------
        previous_values: List[NumericData]
            List of previous values of the defining variables. In Range [0, inf] or None (ignored)
        epsilon : float, optional
            Tolerance for equality to determine "off" state, default is 1e-5.

        Returns:
        -------
        np.ndarray
            A binary array (0 and 1) indicating the previous on/off states of the variables.
            Returns `array([0])` if no previous values are available.
        """

        if not previous_values or all([val is None for val in previous_values]):
            return np.array([0])
        else:  # Convert to 2D-array and compute binary on/off states
            previous_values = np.array([values for values in previous_values if values is not None])  # Filter out None
            if previous_values.ndim > 1:
                return np.any(~np.isclose(previous_values, 0, atol=epsilon), axis=0).astype(int)
            else:
                return (~np.isclose(previous_values, 0, atol=epsilon)).astype(int)

    @staticmethod
    def compute_consecutive_duration(
        binary_values: NumericData,
        hours_per_timestep: Union[int, float, np.ndarray]
    ) -> Scalar:
        """
        Computes the final consecutive duration in State 'on' (=1) in hours, from a binary.

        hours_per_timestep is handled in a way, that maximizes compatability.
        Its length must only be as long as the last consecutive duration in binary_values.

        Parameters
        ----------
        binary_values : int, np.ndarray
            An int or 1D binary array containing only `0`s and `1`s.
        hours_per_timestep : int, float, np.ndarray
            The duration of each timestep in hours.

        Returns
        -------
        np.ndarray
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

        # Find the indexes where value=`0` in a 1D-array
        zero_indices = np.where(np.isclose(binary_values, 0, atol=CONFIG.modeling.EPSILON))[0]
        length_of_last_duration = zero_indices[-1] + 1 if zero_indices.size > 0 else len(binary_values)

        if not np.isscalar(binary_values) and np.isscalar(hours_per_timestep):
            return np.sum(binary_values[-length_of_last_duration:] * hours_per_timestep)

        elif not np.isscalar(binary_values) and not np.isscalar(hours_per_timestep):
            if length_of_last_duration > len(hours_per_timestep):  # check that lengths are compatible
                raise TypeError(
                    f'When trying to calculate the consecutive duration, the length of the last duration '
                    f'({len(length_of_last_duration)}) is longer than the hours_per_timestep ({len(hours_per_timestep)}), '
                    f'as {binary_values=}'
                )
            return np.sum(binary_values[-length_of_last_duration:] * hours_per_timestep[-length_of_last_duration:])

        else:
            raise Exception(
                f'Unexpected state reached in function get_consecutive_duration(). binary_values={binary_values}; '
                f'hours_per_timestep={hours_per_timestep}'
            )


class SegmentModel(Model):
    """Class for modeling a linear segment of one or more variables in parallel"""

    def __init__(
        self,
        model: SystemModel,
        label_of_element: str,
        segment_index: Union[int, str],
        sample_points: Dict[str, Tuple[Union[NumericData, TimeSeries], Union[NumericData, TimeSeries]]],
        as_time_series: bool = True,
    ):
        super().__init__(model, label_of_element, f'Segment{segment_index}')
        self.in_segment: Optional[linopy.Variable] = None
        self.lambda0: Optional[linopy.Variable] = None
        self.lambda1: Optional[linopy.Variable] = None

        self._segment_index = segment_index
        self._as_time_series = as_time_series
        self.sample_points = sample_points

    def do_modeling(self):
        self.in_segment = self.add(self._model.add_variables(
            binary=True,
            name=f'{self.label_full}|in_segment',
            coords=self._model.coords if self._as_time_series else None),
            'in_segment'
        )

        self.lambda0 = self.add(self._model.add_variables(
            lower=0, upper=1,
            name=f'{self.label_full}|lambda0',
            coords=self._model.coords if self._as_time_series else None),
            'lambda0'
        )

        self.lambda1 = self.add(self._model.add_variables(
            lower=0, upper=1,
            name=f'{self.label_full}|lambda1',
            coords=self._model.coords if self._as_time_series else None),
            'lambda1'
        )

        # eq:  lambda0(t) + lambda1(t) = in_segment(t)
        self.add(self._model.add_constraints(
            self.in_segment == self.lambda0 + self.lambda1,
            name=f'{self.label_full}|in_segment'),
            'in_segment'
        )


class MultipleSegmentsModel(Model):
    # TODO: Length...
    def __init__(
        self,
        model: SystemModel,
        label_of_element: str,
        sample_points: Dict[str, List[Tuple[NumericData, NumericData]]],
        can_be_outside_segments: Optional[Union[bool, linopy.Variable]],
        as_time_series: bool = True,
        label: str = 'MultipleSegments',
    ):
        """
        Parameters
        ----------
        model : linopy.Model
            Model to which the segmented variable belongs.
        label_of_element : str
            Name of the parent variable.
        sample_points : dict[str, list[tuple[float, float]]]
            Dictionary mapping variables (names) to their sample points for each segment.
            The sample points are tuples of the form (start, end).
        can_be_outside_segments : bool or linopy.Variable, optional
            Whether the variable can be outside the segments. If True, a variable is created.
            If False or None, no variable is created. If a Variable is passed, it is used.
        as_time_series : bool, optional
        """
        super().__init__(model, label_of_element, label)
        self.outside_segments: Optional[linopy.Variable] = None

        self._as_time_series = as_time_series
        self._can_be_outside_segments = can_be_outside_segments
        self._sample_points = sample_points
        self._segment_models: List[SegmentModel] = []

    def do_modeling(self):
        restructured_variables_with_segments: List[Dict[str, Tuple[NumericData, NumericData]]] = [
            {key: values[i] for key, values in self._sample_points.items()} for i in range(self._nr_of_segments)
        ]

        self._segment_models = [
            self.add(
                SegmentModel(
                    self._model,
                    label_of_element=self.label_of_element,
                    segment_index=i,
                    sample_points=sample_points,
                    as_time_series=self._as_time_series),
                f'Segment_{i}')
            for i, sample_points in enumerate(restructured_variables_with_segments)
        ]

        for segment_model in self._segment_models:
            segment_model.do_modeling()

        #  eq: - v(t) + (v_0_0 * lambda_0_0 + v_0_1 * lambda_0_1) + (v_1_0 * lambda_1_0 + v_1_1 * lambda_1_1) ... = 0
        #  -> v_0_0, v_0_1 = Stützstellen des Segments 0
        for var_name in self._sample_points.keys():
            variable = self._model.variables[var_name]
            self.add(self._model.add_constraints(
                variable == sum([segment.lambda0 * segment.sample_points[var_name][0]
                                 + segment.lambda1 * segment.sample_points[var_name][1]
                                 for segment in self._segment_models]),
                name=f'{self.label_full}|{var_name}_lambda'),
                f'{var_name}_lambda'
            )

            # a) eq: Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 1                Aufenthalt nur in Segmenten erlaubt
            # b) eq: -On(t) + Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 0       zusätzlich kann alles auch Null sein
            if isinstance(self._can_be_outside_segments, linopy.Variable):
                self.outside_segments = self._can_be_outside_segments
                rhs = self.outside_segments
            elif self._can_be_outside_segments is True:
                self.outside_segments = self.add(self._model.add_variables(
                    coords=self._model.coords,
                    binary=True,
                    name=f'{self.label_full}|outside_segments'),
                    'outside_segments'
                )
                rhs = self.outside_segments
            else:
                rhs = 1

            self.add(self._model.add_constraints(
                sum([segment.in_segment for segment in self._segment_models]) <= rhs,
                name=f'{self.label_full}|{variable.name}_single_segment'),
                'single_segment'
            )

    @property
    def _nr_of_segments(self):
        return len(next(iter(self._sample_points.values())))


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
            'total'
        )
        # eq: sum = sum(share_i) # skalar
        self._eq_total = self.add(self._model.add_constraints(self.total == 0, name=f'{self.label_full}|total'), 'total')

        if self._shares_are_time_series:
            self.total_per_timestep = self.add(
                    self._model.add_variables(
                    lower=-np.inf if (self._min_per_hour is None) else np.multiply(self._min_per_hour, self._model.hours_per_step),
                    upper=np.inf if (self._max_per_hour is None) else np.multiply(self._max_per_hour, self._model.hours_per_step),
                    coords=self._model.coords,
                    name=f'{self.label_full}|total_per_timestep'
                ),
                'total_per_timestep'
            )

            self._eq_total_per_timestep = self.add(
                self._model.add_constraints(self.total_per_timestep == 0, name=f'{self.label_full}|total_per_timestep'),
                'total_per_timestep'
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

        Parameters
        ----------
        system_model : SystemModel
            The system model.
        name : str
            The name of the share.
        expression : linopy.LinearExpression
            The expression of the share. Added to the right hand side of the constraint.
        """
        if name in self.shares:
            self.share_constraints[name].lhs -= expression
        else:
            self.shares[name] = self.add(
                self._model.add_variables(
                    coords=None if isinstance(expression, linopy.LinearExpression) and expression.ndim == 0 or not isinstance(expression, linopy.LinearExpression) else self._model.coords,
                    name=f'{name}->{self.label_full}'
                ),
                name
            )
            self.share_constraints[name] = self.add(
                self._model.add_constraints(
                    self.shares[name] == expression, name=f'{name}->{self.label_full}'
                ),
                name
            )
            if self.shares[name].ndim == 0:
                self._eq_total.lhs -= self.shares[name]
            else:
                self._eq_total_per_timestep.lhs -= self.shares[name]


class SegmentedSharesModel(Model):
    # TODO: Length...
    def __init__(
        self,
        model: SystemModel,
        label_of_element: str,
        variable_segments: Tuple[linopy.Variable, List[Tuple[Scalar, Scalar]]],
        share_segments: Dict[str, List[Tuple[Scalar, Scalar]]],
        can_be_outside_segments: Optional[Union[bool, linopy.Variable]],
        label: str = 'SegmentedShares',
    ):
        super().__init__(model, label_of_element, label)
        assert len(variable_segments[1]) == len(list(share_segments.values())[0]), (
            'Segment length of variable_segments and share_segments must be equal'
        )
        self._can_be_outside_segments = can_be_outside_segments
        self._variable_segments = variable_segments
        self._share_segments = share_segments
        self._shares: Dict['Effect', linopy.Variable] = {}
        self._segments_model: Optional[MultipleSegmentsModel] = None
        self._as_tme_series: bool = 'time' in self._variable_segments[0].indexes

    def do_modeling(self):
        self._shares = {
            effect: self.add(self._model.add_variables(
                coords=self._model.coords if self._as_tme_series else None,
                name=f'{self.label_full}|{effect}'),
                f'{effect}'
            ) for effect in self._share_segments
        }

        # Mapping variable names to segments
        segments: Dict[str, List[Tuple[Scalar, Scalar]]] = {
            **{self._shares[effect].name: segment for effect, segment in self._share_segments.items()},
            **{self._variable_segments[0].name: self._variable_segments[1]},
        }

        self._segments_model = self.add(
            MultipleSegmentsModel(
                model=self._model,
                label_of_element=self.label_of_element,
                sample_points=segments,
                can_be_outside_segments=self._can_be_outside_segments,
                as_time_series=self._as_tme_series),
            'segments'
        )
        self._segments_model.do_modeling()

        # Shares
        self._model.effects.add_share_to_effects(
            name=self.label_of_element,
            expressions={effect: variable*1 for effect, variable in self._shares.items()},
            target='operation' if self._as_tme_series else 'invest',
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

    def __init__(self, model: SystemModel, variables: List[linopy.Variable], label_of_element: str, label: str = 'PreventSimultaneousUsage'):
        super().__init__(model, label_of_element, label)
        self._simultanious_use_variables = variables
        assert len(self._simultanious_use_variables) >= 2, f'Model {self.__class__.__name__} must get at least two variables'
        for variable in self._simultanious_use_variables:  # classic
            assert variable.attrs['binary'], f'Variable {variable} must be binary for use in {self.__class__.__name__}'

    def do_modeling(self):
        # eq: sum(flow_i.on(t)) <= 1.1 (1 wird etwas größer gewählt wg. Binärvariablengenauigkeit)
        self.add(self._model.add_constraints(sum(self._simultanious_use_variables) <= 1.1,
                                             name=f'{self.label_full}|prevent_simultaneous_use'),
                 'prevent_simultaneous_use')
