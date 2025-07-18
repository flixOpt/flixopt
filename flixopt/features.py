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
from .interface import InvestParameters, OnOffParameters, Piecewise, PiecewiseEffects
from .structure import Model, FlowSystemModel, BaseFeatureModel
from .modeling import ModelingPatterns, ModelingUtilities, ModelingPrimitives

logger = logging.getLogger('flixopt')


class InvestmentModel(BaseFeatureModel):
    """Investment model using factory patterns but keeping old interface"""

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        parameters: InvestParameters,
        defining_variable: linopy.Variable,
        relative_bounds_of_defining_variable: Tuple[TemporalData, TemporalData],
        label_of_model: Optional[str] = None,
        on_variable: Optional[linopy.Variable] = None,
    ):
        super().__init__(model, label_of_element=label_of_element, parameters=parameters, label_of_model=label_of_model)

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
            size_bounds=(self.parameters.minimum_or_fixed_size, self.parameters.maximum_or_fixed_size,),
            controlled_variables=[self._defining_variable],
            control_factors=[self._relative_bounds_of_defining_variable],
            state_variables=[self._on_variable],
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
        parameters: OnOffParameters,
        flow_rates: List[linopy.Variable],
        flow_rate_bounds: List[Tuple[TemporalData, TemporalData]],
        previous_flow_rates: List[Optional[TemporalData]],
        label_of_model: Optional[str] = None,
    ):
        super().__init__(model, label_of_element, parameters=parameters, label_of_model=label_of_model)
        self._flow_rates = flow_rates
        self._flow_rate_bounds = flow_rate_bounds
        self._previous_flow_rates = previous_flow_rates

    def create_variables_and_constraints(self):
        variables = {}
        constraints = {}

        # 1. Main binary state using existing pattern
        state_vars, state_constraints = ModelingPrimitives.binary_state_pair(self._model, self.label_of_model, use_complement=self.parameters.use_off)
        variables.update(state_vars)
        constraints.update(state_constraints)

        # 2. Control variables - use big_m_binary_bounds pattern for consistency
        for i, (flow_rate, (lower_bound, upper_bound)) in enumerate(zip(self._flow_rates, self._flow_rate_bounds)):
            suffix = f'_{i}' if len(self._flow_rates) > 1 else ''
            # Use the big_m pattern but without binary control (None)
            _, control_constraints = ModelingPrimitives.big_m_binary_bounds(
                model=self._model,
                variable=flow_rate,
                binary_control=None,
                size_variable=variables['on'],
                relative_bounds=(lower_bound, upper_bound),
                upper_bound_name=f'{variables['on'].name}|ub{suffix}',
                lower_bound_name=f'{variables['on'].name}|lb{suffix}',
            )
            constraints[f'ub_{i}'] = control_constraints['upper_bound']
            constraints[f'lb_{i}'] = control_constraints['lower_bound']

        # 3. Total duration tracking using existing pattern
        duration_expr = (variables['on'] * self._model.hours_per_step).sum('time')
        duration_vars, duration_constraints = ModelingPrimitives.expression_tracking_variable(
            self._model, f'{self.label_of_model}|on_hours_total', duration_expr,
            (self.parameters.on_hours_total_min if self.parameters.on_hours_total_min is not None else 0,
             self.parameters.on_hours_total_max if self.parameters.on_hours_total_max is not None else np.inf),#TODO: self._model.hours_per_step.sum('time').item() + self._get_previous_on_duration())
        )
        variables['on_hours_total'] = duration_vars['tracker']
        constraints['on_hours_total'] = duration_constraints['tracking']

        # 4. Switch tracking using existing pattern
        if self.parameters.use_switch_on:
            switch_vars, switch_constraints = ModelingPrimitives.state_transition_variables(
                self._model, f'{self.label_of_model}|switches', variables['on'],
                previous_state=ModelingUtilities.get_most_recent_state(self._previous_flow_rates)
            )
            variables.update(switch_vars)
            for switch_name, switch_constraint in switch_constraints.items():
                constraints[f'switch_{switch_name}'] = switch_constraint

        # 5. Consecutive on duration using existing pattern
        if self.parameters.use_consecutive_on_hours:
            consecutive_on_vars, consecutive_on_constraints = ModelingPrimitives.consecutive_duration_tracking(
                self._model,
                f'{self.label_of_model}|consecutive_on',
                variables['on'],
                minimum_duration=self.parameters.consecutive_on_hours_min,
                maximum_duration=self.parameters.consecutive_on_hours_max,
                previous_duration=ModelingUtilities.compute_previous_on_duration(self._previous_flow_rates, self._model.hours_per_step),
            )
            variables['consecutive_on_duration'] = consecutive_on_vars['duration']
            for cons_name, cons_constraint in consecutive_on_constraints.items():
                constraints[f'consecutive_on_{cons_name}'] = cons_constraint

        # 6. Consecutive off duration using existing pattern
        if self.parameters.use_consecutive_off_hours:
            consecutive_off_vars, consecutive_off_constraints = ModelingPrimitives.consecutive_duration_tracking(
                self._model,
                f'{self.label_of_model}|consecutive_off',
                variables['off'],
                minimum_duration=self.parameters.consecutive_off_hours_min,
                maximum_duration=self.parameters.consecutive_off_hours_max,
                previous_duration=ModelingUtilities.compute_previous_off_duration(self._previous_flow_rates, self._model.hours_per_step),
            )
            variables['consecutive_off_duration'] = consecutive_off_vars['duration']
            for cons_name, cons_constraint in consecutive_off_constraints.items():
                constraints[f'consecutive_off_{cons_name}'] = cons_constraint

        # Register all constraints and variables
        for constraint_name, constraint in constraints.items():
            self.add(constraint, constraint_name)
        for variable_name, variable in variables.items():
            self.add(variable, variable_name)

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
        return ModelingUtilities.compute_previous_on_duration(self._previous_flow_rates, hours_per_step)

    def _get_previous_off_duration(self):
        hours_per_step = self._model.hours_per_step.isel(time=0).values.flatten()[0]
        return ModelingUtilities.compute_previous_off_duration(self._previous_flow_rates, hours_per_step)

    def _get_previous_state(self):
        return ModelingUtilities.get_most_recent_state(self._previous_flow_rates)


class PieceModel(Model):
    """Class for modeling a linear piece of one or more variables in parallel"""

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        label_of_model: str,
        as_time_series: bool = True,
    ):
        super().__init__(model, label_of_element, label_of_model)
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
        label_of_model: str = '',
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
        super().__init__(model, label_of_element=label_of_element, label_of_model=label_of_model)
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
                    label_of_model=f'{self.label_of_element}|Piece_{i}',
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
                label_of_model=f'{self.label_of_element}|PiecewiseEffects',
            )
        )

        self.piecewise_model.do_modeling()

        # Shares
        self._model.effects.add_share_to_effects(
            name=self.label_of_element,
            expressions={effect: variable * 1 for effect, variable in self.shares.items()},
            target='invest',
        )


class ShareAllocationModel(Model):
    def __init__(
        self,
        model: FlowSystemModel,
        dims: List[FlowSystemDimensions],
        label_of_element: Optional[str] = None,
        label_of_model: Optional[str] = None,
        total_max: Optional[Scalar] = None,
        total_min: Optional[Scalar] = None,
        max_per_hour: Optional[TemporalData] = None,
        min_per_hour: Optional[TemporalData] = None,
    ):
        super().__init__(model, label_of_element=label_of_element, label_of_model=label_of_model)

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
