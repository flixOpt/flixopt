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
from .structure import Submodel, FlowSystemModel, BaseFeatureModel
from .modeling import ModelingUtilities, ModelingPrimitives, BoundingPatterns

logger = logging.getLogger('flixopt')


class InvestmentModel(BaseFeatureModel):
    """Investment model using factory patterns but keeping old interface"""

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        parameters: InvestParameters,
        label_of_model: Optional[str] = None,
    ):
        """
        This feature model is used to model the investment of a variable.
        It applies the corresponding bounds to the variable and the on/off state of the variable.

        Args:
            model: The optimization model instance
            label_of_element: The label of the parent (Element). Used to construct the full label of the model.
            parameters: The parameters of the feature model.
            defining_variable: The variable to be invested
            relative_bounds_of_defining_variable: The bounds of the variable, with respect to the minimum/maximum investment sizes
            label_of_model: The label of the model. This is needed to construct the full label of the model.

        """
        super().__init__(model, label_of_element=label_of_element, parameters=parameters, label_of_model=label_of_model)

        self.piecewise_effects: Optional[PiecewiseEffectsModel] = None


    def create_variables_and_constraints(self):
        size_min, size_max = (self.parameters.minimum_or_fixed_size, self.parameters.maximum_or_fixed_size)
        self.add_variables(
            short_name='size',
            lower=0 if self.parameters.optional else size_min,
            upper=size_max,
            coords=self._model.get_coords(['year', 'scenario']),
        )

        if self.parameters.optional:
            self.add_variables(
                binary=True, coords=self._model.get_coords(['year', 'scenario']), short_name='is_invested',
            )

            BoundingPatterns.bounds_with_state(
                self,
                variable=self.size,
                variable_state=self.is_invested,
                bounds=(self.parameters.minimum_or_fixed_size, self.parameters.maximum_or_fixed_size),
            )

        if self.parameters.piecewise_effects:
            self.piecewise_effects = self.register_sub_model(
                PiecewiseEffectsModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    label_of_model=f'{self.label_of_element}|PiecewiseEffects',
                    piecewise_origin=(self.size.name, self.parameters.piecewise_effects.piecewise_origin),
                    piecewise_shares=self.parameters.piecewise_effects.piecewise_shares,
                    zero_point=self.is_invested,
                ),
                short_name='segments',
            )
            self.piecewise_effects.do_modeling()

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

    @property
    def size(self) -> linopy.Variable:
        """Investment size variable"""
        return self._variables['size']

    @property
    def is_invested(self) -> Optional[linopy.Variable]:
        """Binary investment decision variable"""
        if 'is_invested' not in self._variables:
            return None
        return self._variables['is_invested']


class OnOffModel(BaseFeatureModel):
    """OnOff model using factory patterns"""

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        parameters: OnOffParameters,
        on_variable: linopy.Variable,
        previous_states: Optional[TemporalData],
        label_of_model: Optional[str] = None,
    ):
        """
        This feature model is used to model the on/off state of flow_rate(s). It does not matter of the flow_rates are
        bounded by a size variable or by a hard bound. THe used bound here is the absolute highest/lowest bound!

        Args:
            model: The optimization model instance
            label_of_element: The label of the parent (Element). Used to construct the full label of the model.
            parameters: The parameters of the feature model.
            on_variable: The variable that determines the on state
            previous_states: The previous flow_rates
            label_of_model: The label of the model. This is needed to construct the full label of the model.
        """
        super().__init__(model, label_of_element, parameters=parameters, label_of_model=label_of_model)
        self.on = on_variable
        self._previous_states = previous_states

    def create_variables_and_constraints(self):
        if self.parameters.use_off:
            off = self.add_variables(binary=True, short_name='off', coords=self._model.get_coords())
            self.add_constraints(self.on + off == 1, short_name='complementary')

        # 3. Total duration tracking using existing pattern
        duration_expr = (self.on * self._model.hours_per_step).sum('time')
        ModelingPrimitives.expression_tracking_variable(
            self, duration_expr, short_name='on_hours_total',
            bounds=(
                self.parameters.on_hours_total_min if self.parameters.on_hours_total_min is not None else 0,
                self.parameters.on_hours_total_max if self.parameters.on_hours_total_max is not None else np.inf,
            ),   #TODO: self._model.hours_per_step.sum('time').item() + self._get_previous_on_duration())
        )

        # 4. Switch tracking using existing pattern
        if self.parameters.use_switch_on:
            self.add_variables(binary=True, short_name='switch|on', coords=self.get_coords())
            self.add_variables(binary=True, short_name='switch|off', coords=self.get_coords())

            ModelingPrimitives.state_transition_variables(
                self,
                state_variable=self.on,
                switch_on=self.switch_on,
                switch_off=self.switch_off,
                name=f'{self.label_of_model}|switch',
                previous_state=self._previous_states.isel(time=-1) if self._previous_states is not None else 0,
        )

            if self.parameters.switch_on_total_max is not None:
                count = self.add_variables(lower=0, upper=self.parameters.switch_on_total_max, coords=self._model.get_coords(('year', 'scenario')), short_name='switch|count')
                self.add_constraints(count == self.switch_on.sum('time'), short_name='switch|count')

        # 5. Consecutive on duration using existing pattern
        if self.parameters.use_consecutive_on_hours:
            ModelingPrimitives.consecutive_duration_tracking(
                self,
                state_variable=self.on,
                short_name='consecutive_on_hours',
                minimum_duration=self.parameters.consecutive_on_hours_min,
                maximum_duration=self.parameters.consecutive_on_hours_max,
                previous_duration=self._get_previous_on_duration(),
            )

        # 6. Consecutive off duration using existing pattern
        if self.parameters.use_consecutive_off_hours:
            ModelingPrimitives.consecutive_duration_tracking(
                self,
                state_variable=self.off,
                short_name='consecutive_off_hours',
                minimum_duration=self.parameters.consecutive_off_hours_min,
                maximum_duration=self.parameters.consecutive_off_hours_max,
                previous_duration=self._get_previous_off_duration(),
            )
            #TODO:

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

        if self.parameters.effects_per_switch_on:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.switch_on * factor for effect, factor in self.parameters.effects_per_switch_on.items()
                },
                target='operation',
            )

    # Properties access variables from Submodel's tracking system

    @property
    def total_on_hours(self) -> Optional[linopy.Variable]:
        """Total on hours variable"""
        return self['total_on_hours']

    @property
    def off(self) -> Optional[linopy.Variable]:
        """Binary off state variable"""
        return self.get('off')

    @property
    def switch_on(self) -> Optional[linopy.Variable]:
        """Switch on variable"""
        return self.get('switch|on')

    @property
    def switch_off(self) -> Optional[linopy.Variable]:
        """Switch off variable"""
        return self.get('switch|off')

    @property
    def switch_on_nr(self) -> Optional[linopy.Variable]:
        """Number of switch-ons variable"""
        return self.get('switch|count')

    @property
    def consecutive_on_hours(self) -> Optional[linopy.Variable]:
        """Consecutive on hours variable"""
        return self.get('consecutive_on_hours')

    @property
    def consecutive_off_hours(self) -> Optional[linopy.Variable]:
        """Consecutive off hours variable"""
        return self.get('consecutive_off_hours')

    def _get_previous_on_duration(self):
        """Get previous on duration. Previously OFF by default, for one timestep"""
        hours_per_step = self._model.hours_per_step.isel(time=0).min().item()
        if self._previous_states is None:
            return 0
        else:
            return ModelingUtilities.compute_consecutive_hours_in_state(self._previous_states, hours_per_step)

    def _get_previous_off_duration(self):
        """Get previous off duration. Previously OFF by default, for one timestep"""
        hours_per_step = self._model.hours_per_step.isel(time=0).min().item()
        if self._previous_states is None:
            return hours_per_step
        else:
            return ModelingUtilities.compute_consecutive_hours_in_state(self._previous_states  * -1 + 1, hours_per_step)


class PieceModel(Submodel):
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
        self.inside_piece = self.add_variables(
            binary=True,
            short_name='inside_piece',
            coords=self._model.get_coords(dims=dims),
        )
        self.lambda0 = self.add_variables(
            lower=0,
            upper=1,
            short_name='lambda0',
            coords=self._model.get_coords(dims=dims),
        )

        self.lambda1 = self.add_variables(
            lower=0,
            upper=1,
            short_name='lambda1',
            coords=self._model.get_coords(dims=dims),
        )

        # eq:  lambda0(t) + lambda1(t) = inside_piece(t)
        self.add_constraints(self.inside_piece == self.lambda0 + self.lambda1, short_name='inside_piece')


class PiecewiseModel(Submodel):
    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        label_of_model: str,
        piecewise_variables: Dict[str, Piecewise],
        zero_point: Optional[Union[bool, linopy.Variable]],
        as_time_series: bool,
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
            new_piece = self.register_sub_model(
                PieceModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    label_of_model=f'{self.label_of_element}|Piece_{i}',
                    as_time_series=self._as_time_series,
                ),
                short_name=f'Piece_{i}',
            )
            self.pieces.append(new_piece)
            new_piece.do_modeling()

        for var_name in self._piecewise_variables:
            variable = self._model.variables[var_name]
            self.add_constraints(
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
                short_name=f'{var_name}|lambda',
                )

            # a) eq: Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 1                Aufenthalt nur in Segmenten erlaubt
            # b) eq: -On(t) + Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 0       zusätzlich kann alles auch Null sein
            if isinstance(self._zero_point, linopy.Variable):
                self.zero_point = self._zero_point
                rhs = self.zero_point
            elif self._zero_point is True:
                self.zero_point = self.add_variables(
                    coords=self._model.get_coords(), binary=True, short_name='zero_point'
                )
                rhs = self.zero_point
            else:
                rhs = 1

            self.add_constraints(
                sum([piece.inside_piece for piece in self.pieces]) <= rhs,
                name=f'{self.label_full}|{variable.name}|single_segment',
                short_name=f'{var_name}|single_segment',
            )


class PiecewiseEffectsModel(Submodel):
    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        label_of_model: str,
        piecewise_origin: Tuple[str, Piecewise],
        piecewise_shares: Dict[str, Piecewise],
        zero_point: Optional[Union[bool, linopy.Variable]],
    ):
        super().__init__(model, label_of_element=label_of_element, label_of_model=label_of_model)
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
            effect: self.add_variables(coords=self._model.get_coords(['year', 'scenario']), short_name=effect)
            for effect in self._piecewise_shares
        }

        piecewise_variables = {
            self._piecewise_origin[0]: self._piecewise_origin[1],
            **{
                self.shares[effect_label].name: self._piecewise_shares[effect_label]
                for effect_label in self._piecewise_shares
            },
        }

        self.piecewise_model = self.register_sub_model(
            PiecewiseModel(
                model=self._model,
                label_of_element=self.label_of_element,
                piecewise_variables=piecewise_variables,
                zero_point=self._zero_point,
                as_time_series=False,
                label_of_model=f'{self.label_of_element}|PiecewiseEffects',
            ),
            short_name='PiecewiseEffects',
        )

        self.piecewise_model.do_modeling()

        # Shares
        self._model.effects.add_share_to_effects(
            name=self.label_of_element,
            expressions={effect: variable * 1 for effect, variable in self.shares.items()},
            target='invest',
        )


class ShareAllocationModel(Submodel):
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
        self.total = self.add_variables(
            lower=self._total_min,
            upper=self._total_max,
            coords=self._model.get_coords([dim for dim in self._dims if dim != 'time']),
            short_name='total'
        )
        # eq: sum = sum(share_i) # skalar
        self._eq_total =  self.add_constraints(self.total == 0, short_name='total')

        if 'time' in self._dims:
            self.total_per_timestep = self.add_variables(
                lower=-np.inf if (self._min_per_hour is None) else self._min_per_hour * self._model.hours_per_step,
                upper=np.inf if (self._max_per_hour is None) else self._max_per_hour * self._model.hours_per_step,
                coords=self._model.get_coords(self._dims),
                short_name='total_per_timestep',
            )

            self._eq_total_per_timestep = self.add_constraints(self.total_per_timestep == 0, short_name='total_per_timestep')

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
            self.shares[name] = self.add_variables(
                coords=self._model.get_coords(dims),
                name=f'{name}->{self.label_full}',
                short_name=name,
            )

            self.share_constraints[name] = self.add_constraints(
                self.shares[name] == expression, name=f'{name}->{self.label_full}'
            )

            if 'time' not in dims:
                self._eq_total.lhs -= self.shares[name]
            else:
                self._eq_total_per_timestep.lhs -= self.shares[name]
