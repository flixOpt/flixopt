"""
This module contains the features of the flixopt framework.
Features extend the functionality of Elements.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Protocol

import linopy
import numpy as np

from .modeling import BoundingPatterns, ModelingPrimitives, ModelingUtilities
from .structure import FlowSystemModel, Submodel

if TYPE_CHECKING:
    from .core import FlowSystemDimensions, PeriodicData, Scalar, TemporalData
    from .effects import PeriodicEffects
    from .interface import (
        InvestmentParameters,
        OnOffParameters,
        Piecewise,
        SizingParameters,
    )

logger = logging.getLogger('flixopt')


class _SizeModel(Submodel):
    """A model that creates the size variable together with a Binary"""

    def _create_sizing_variables_and_constraints(
        self,
        size_min: PeriodicData,
        size_max: PeriodicData,
        mandatory: PeriodicData,
        dims: list[FlowSystemDimensions],
    ):
        """Create timing variables and constraints."""
        if not np.issubdtype(mandatory.dtype, np.bool_):
            raise TypeError(f'Expected all bool values, got {mandatory.dtype=}: {mandatory}')

        size = self.add_variables(
            short_name='size',
            lower=size_min.where(mandatory, 0),
            upper=size_max,
            coords=self._model.get_coords(dims),
        )

        if mandatory.any():
            self.add_variables(
                binary=True,
                coords=self._model.get_coords(dims),
                short_name='available',
            )
            self.add_constraints(
                self.available.where(mandatory) == 1,
                short_name='mandatory',
            )
            BoundingPatterns.bounds_with_state(
                self,
                variable=size,
                variable_state=self._variables['available'],
                bounds=(size_min, size_max),
            )

    def _add_sizing_effects(self, effects_per_size: PeriodicEffects, effects_of_size: PeriodicEffects):
        if effects_per_size:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={effect: self.size * factor for effect, factor in effects_per_size.items()},
                target='periodic',
            )

        if effects_of_size:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.available * factor if self.available is not None else factor
                    for effect, factor in effects_of_size.items()
                },
                target='periodic',
            )

    @property
    def size(self) -> linopy.Variable:
        """Capacity size variable"""
        return self._variables['size']

    @property
    def available(self) -> linopy.Variable | None:
        """Capacity size variable"""
        return self._variables.get('available')


class SizingModel(_SizeModel):
    """
    This feature model is used to model capacity sizing decisions.
    It applies bounds to the size variable and optionally creates a binary investment decision.

    Args:
        model: The optimization model instance
        label_of_element: The label of the parent (Element). Used to construct the full label of the model.
        parameters: The sizing parameters.
        label_of_model: The label of the model. This is needed to construct the full label of the model.
    """

    parameters: SizingParameters

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        parameters: SizingParameters,
        label_of_model: str | None = None,
    ):
        self.parameters = parameters
        super().__init__(model, label_of_element=label_of_element, label_of_model=label_of_model)

    def _do_modeling(self):
        super()._do_modeling()
        self._create_sizing_variables_and_constraints(
            size_min=self.parameters.minimum_or_fixed_size,
            size_max=self.parameters.maximum_or_fixed_size,
            mandatory=self.parameters.mandatory,
            dims=['period', 'scenario'],
        )
        self._add_sizing_effects(
            effects_per_size=self.parameters.effects_per_size,
            effects_of_size=self.parameters.effects_of_size,
        )

    @property
    def invested(self) -> linopy.Variable | None:
        warnings.warn('Deprecated, use availlable instead', DeprecationWarning, stacklevel=2)
        return self.available


class InvestmentModel(_SizeModel):
    """
    Model investment timing with fixed lifetime.

    This feature works in conjunction with SizingModel to provide full investment modeling:
    - SizingModel: Determines HOW MUCH capacity to install
    - InvestmentModel: Determines WHEN to invest

    The model creates binary variables to track:
    - When the investment occurs (one period)
    - Which periods the investment is active (based on fixed lifetime)

    The investment capacity (from SizingModel) is only active during the investment's lifetime.

    Args:
        model: The optimization model instance
        label_of_element: The label of the parent element
        parameters: InvestmentParameters defining timing constraints
        label_of_model: Optional custom label for the model
    """

    parameters: InvestmentParameters

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        parameters: InvestmentParameters,
        label_of_model: str | None = None,
    ):
        self.parameters = parameters
        super().__init__(model, label_of_element=label_of_element, label_of_model=label_of_model)

    def _do_modeling(self):
        super()._do_modeling()
        self._create_variables_and_constraints()
        self._add_effects()

    def _create_variables_and_constraints(self):
        """Create timing variables and constraints."""
        # Regular sizing ===============================================================================================
        self._create_sizing_variables_and_constraints(
            size_min=self.parameters.minimum_or_fixed_size,
            size_max=self.parameters.maximum_or_fixed_size,
            mandatory=self.parameters.mandatory,
            dims=['period', 'scenario'],
        )

        self._track_investment_and_decomissioning_period()
        self._track_investment_and_decomissioning_size()
        self._apply_investment_period_constraints()

        self._add_effects()

    def _track_investment_and_decomissioning_period(self):
        """Track investment and decomissioning period absed on binary state variable."""
        self.add_variables(
            binary=True,
            coords=self._model.get_coords(['period', 'scenario']),
            short_name='size|investment_occurs',
        )
        self.add_constraints(
            self.investment_occurs.sum('period') <= 1,
            short_name='invest_once',
        )

        self.add_variables(
            binary=True,
            coords=self._model.get_coords(['period', 'scenario']),
            short_name='size|decommissioning_occurs',
        )
        self.add_constraints(
            self.decommissioning_occurs.sum('period') <= 1,
            short_name='decommission_once',
        )

        BoundingPatterns.state_transition_bounds(
            self,
            state_variable=self.is_invested,
            switch_on=self.investment_occurs,
            switch_off=self.decommissioning_occurs,
            name=self.is_invested.name,
            previous_state=0,
            coord='period',
        )

    def _track_investment_and_decomissioning_size(self):
        self.add_variables(
            coords=self._model.get_coords(['period', 'scenario']),
            short_name='size|increase',
            lower=0,
            upper=self.parameters.maximum_or_fixed_size,
        )
        self.add_variables(
            coords=self._model.get_coords(['period', 'scenario']),
            short_name='size|decrease',
            lower=0,
            upper=self.parameters.maximum_or_fixed_size,
        )
        BoundingPatterns.link_changes_to_level_with_binaries(
            self,
            level_variable=self.size,
            increase_variable=self.size_increase,
            decrease_variable=self.size_decrease,
            increase_binary=self.investment_occurs,
            decrease_binary=self.decommissioning_occurs,
            name=f'{self.label_of_element}|size|changes',
            max_change=self.parameters.maximum_or_fixed_size,
            previous_level=self.parameters.previous_size,
            coord='period',
        )

    def _apply_investment_period_constraints(self):
        # Constraint: Apply allow_investment restrictions
        if (self.parameters.allow_investment == 0).any():
            if (self.parameters.allow_investment == 0).all('period'):
                logger.error(f'In "{self.label_full}": Need to allow Investment in at least one period.')
            self.add_constraints(
                self.investment_occurs <= self.parameters.allow_investment,
                short_name='allow_investment',
            )

        # If a specific period is forced, investment must occur there
        if (self.parameters.force_investment == 1).any():
            if (self.parameters.force_investment.sum('period') > 1).any():
                raise ValueError('Can not force Investment in more than one period')
            self.add_constraints(
                self.investment_occurs == self.parameters.force_investment,
                short_name='force_investment',
            )

    def _add_effects(self):
        """Add investment effects to the model."""
        self._add_sizing_effects(
            self.parameters.effects_per_size,
            self.parameters.effects_of_size,
        )

        # New kind of effects ==========================================================================================

        if self.parameters.effects_of_investment:
            # Effects depending on when the investment is made
            remapped_variable = self.investment_occurs.rename({'period': 'period_of_investment'})

            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: (remapped_variable * factor).sum('period_of_investment')
                    for effect, factor in self.parameters.effects_of_investment.items()
                },
                target='periodic',
            )

        if self.parameters.effects_of_investment_per_size:
            # Effects depending on when the investment is made proportional to investment size
            remapped_variable = self.size_increase.rename({'period': 'period_of_investment'})

            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: (remapped_variable * factor).sum('period_of_investment')
                    for effect, factor in self.parameters.effects_of_investment_per_size.items()
                },
                target='periodic',
            )

    @property
    def investment_occurs(self) -> linopy.Variable:
        """Binary variable indicating when investment occurs (at most one period)"""
        return self._variables['size|investment_occurs']

    @property
    def decommissioning_occurs(self) -> linopy.Variable:
        """Binary decrease decision variable"""
        return self._variables['size|decommissioning_occurs']

    @property
    def is_invested(self) -> linopy.Variable:
        """Binary variable indicating which periods have active investment"""
        return self._variables['is_invested']

    @property
    def size_decrease(self) -> linopy.Variable:
        """Binary decrease decision variable"""
        return self._variables['size|decrease']

    @property
    def size_increase(self) -> linopy.Variable:
        """Binary increase decision variable"""
        return self._variables['size|increase']


class OnOffModel(Submodel):
    """OnOff model using factory patterns"""

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        parameters: OnOffParameters,
        on_variable: linopy.Variable,
        previous_states: TemporalData | None,
        label_of_model: str | None = None,
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
        self.on = on_variable
        self._previous_states = previous_states
        self.parameters = parameters
        super().__init__(model, label_of_element, label_of_model=label_of_model)

    def _do_modeling(self):
        super()._do_modeling()

        if self.parameters.use_off:
            off = self.add_variables(binary=True, short_name='off', coords=self._model.get_coords())
            self.add_constraints(self.on + off == 1, short_name='complementary')

        # 3. Total duration tracking using existing pattern
        ModelingPrimitives.expression_tracking_variable(
            self,
            tracked_expression=(self.on * self._model.hours_per_step).sum('time'),
            bounds=(
                self.parameters.on_hours_total_min if self.parameters.on_hours_total_min is not None else 0,
                self.parameters.on_hours_total_max if self.parameters.on_hours_total_max is not None else np.inf,
            ),  # TODO: self._model.hours_per_step.sum('time').item() + self._get_previous_on_duration())
            short_name='on_hours_total',
            coords=['period', 'scenario'],
        )

        # 4. Switch tracking using existing pattern
        if self.parameters.use_switch_on:
            self.add_variables(binary=True, short_name='switch|on', coords=self.get_coords())
            self.add_variables(binary=True, short_name='switch|off', coords=self.get_coords())

            BoundingPatterns.state_transition_bounds(
                self,
                state_variable=self.on,
                switch_on=self.switch_on,
                switch_off=self.switch_off,
                name=f'{self.label_of_model}|switch',
                previous_state=self._previous_states.isel(time=-1) if self._previous_states is not None else 0,
                coord='time',
            )

            if self.parameters.switch_on_total_max is not None:
                count = self.add_variables(
                    lower=0,
                    upper=self.parameters.switch_on_total_max,
                    coords=self._model.get_coords(('period', 'scenario')),
                    short_name='switch|count',
                )
                self.add_constraints(count == self.switch_on.sum('time'), short_name='switch|count')

        # 5. Consecutive on duration using existing pattern
        if self.parameters.use_consecutive_on_hours:
            ModelingPrimitives.consecutive_duration_tracking(
                self,
                state_variable=self.on,
                short_name='consecutive_on_hours',
                minimum_duration=self.parameters.consecutive_on_hours_min,
                maximum_duration=self.parameters.consecutive_on_hours_max,
                duration_per_step=self.hours_per_step,
                duration_dim='time',
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
                duration_per_step=self.hours_per_step,
                duration_dim='time',
                previous_duration=self._get_previous_off_duration(),
            )
            # TODO:

        self._add_effects()

    def _add_effects(self):
        """Add operational effects"""
        if self.parameters.effects_per_running_hour:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.on * factor * self._model.hours_per_step
                    for effect, factor in self.parameters.effects_per_running_hour.items()
                },
                target='temporal',
            )

        if self.parameters.effects_per_switch_on:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.switch_on * factor for effect, factor in self.parameters.effects_per_switch_on.items()
                },
                target='temporal',
            )

    # Properties access variables from Submodel's tracking system

    @property
    def on_hours_total(self) -> linopy.Variable:
        """Total on hours variable"""
        return self['on_hours_total']

    @property
    def off(self) -> linopy.Variable | None:
        """Binary off state variable"""
        return self.get('off')

    @property
    def switch_on(self) -> linopy.Variable | None:
        """Switch on variable"""
        return self.get('switch|on')

    @property
    def switch_off(self) -> linopy.Variable | None:
        """Switch off variable"""
        return self.get('switch|off')

    @property
    def switch_on_nr(self) -> linopy.Variable | None:
        """Number of switch-ons variable"""
        return self.get('switch|count')

    @property
    def consecutive_on_hours(self) -> linopy.Variable | None:
        """Consecutive on hours variable"""
        return self.get('consecutive_on_hours')

    @property
    def consecutive_off_hours(self) -> linopy.Variable | None:
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
            return ModelingUtilities.compute_consecutive_hours_in_state(self._previous_states * -1 + 1, hours_per_step)


class PieceModel(Submodel):
    """Class for modeling a linear piece of one or more variables in parallel"""

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        label_of_model: str,
        dims: FlowSystemDimensions | None,
    ):
        self.inside_piece: linopy.Variable | None = None
        self.lambda0: linopy.Variable | None = None
        self.lambda1: linopy.Variable | None = None
        self.dims = dims

        super().__init__(model, label_of_element, label_of_model)

    def _do_modeling(self):
        super()._do_modeling()
        self.inside_piece = self.add_variables(
            binary=True,
            short_name='inside_piece',
            coords=self._model.get_coords(dims=self.dims),
        )
        self.lambda0 = self.add_variables(
            lower=0,
            upper=1,
            short_name='lambda0',
            coords=self._model.get_coords(dims=self.dims),
        )

        self.lambda1 = self.add_variables(
            lower=0,
            upper=1,
            short_name='lambda1',
            coords=self._model.get_coords(dims=self.dims),
        )

        # eq:  lambda0(t) + lambda1(t) = inside_piece(t)
        self.add_constraints(self.inside_piece == self.lambda0 + self.lambda1, short_name='inside_piece')


class PiecewiseModel(Submodel):
    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        label_of_model: str,
        piecewise_variables: dict[str, Piecewise],
        zero_point: bool | linopy.Variable | None,
        dims: FlowSystemDimensions | None,
    ):
        """
        Modeling a Piecewise relation between miultiple variables.
        The relation is defined by a list of Pieces, which are assigned to the variables.
        Each Piece is a tuple of (start, end).

        Args:
            model: The FlowSystemModel that is used to create the model.
            label_of_element: The label of the parent (Element). Used to construct the full label of the model.
            label_of_model: The label of the model. Used to construct the full label of the model.
            piecewise_variables: The variables to which the Pieces are assigned.
            zero_point: A variable that can be used to define a zero point for the Piecewise relation. If None or False, no zero point is defined.
            dims: The dimensions used for variable creation. If None, all dimensions are used.
        """
        self._piecewise_variables = piecewise_variables
        self._zero_point = zero_point
        self.dims = dims

        self.pieces: list[PieceModel] = []
        self.zero_point: linopy.Variable | None = None
        super().__init__(model, label_of_element=label_of_element, label_of_model=label_of_model)

    def _do_modeling(self):
        super()._do_modeling()
        # Validate all piecewise variables have the same number of segments
        segment_counts = [len(pw) for pw in self._piecewise_variables.values()]
        if not all(count == segment_counts[0] for count in segment_counts):
            raise ValueError(f'All piecewises must have the same number of pieces, got {segment_counts}')

        for i in range(len(list(self._piecewise_variables.values())[0])):
            new_piece = self.add_submodels(
                PieceModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    label_of_model=f'{self.label_of_element}|Piece_{i}',
                    dims=self.dims,
                ),
                short_name=f'Piece_{i}',
            )
            self.pieces.append(new_piece)

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
                    coords=self._model.get_coords(self.dims),
                    binary=True,
                    short_name='zero_point',
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
        piecewise_origin: tuple[str, Piecewise],
        piecewise_shares: dict[str, Piecewise],
        zero_point: bool | linopy.Variable | None,
    ):
        origin_count = len(piecewise_origin[1])
        share_counts = [len(pw) for pw in piecewise_shares.values()]
        if not all(count == origin_count for count in share_counts):
            raise ValueError(
                f'Piece count mismatch: piecewise_origin has {origin_count} segments, '
                f'but piecewise_shares have {share_counts}'
            )
        self._zero_point = zero_point
        self._piecewise_origin = piecewise_origin
        self._piecewise_shares = piecewise_shares
        self.shares: dict[str, linopy.Variable] = {}

        self.piecewise_model: PiecewiseModel | None = None

        super().__init__(model, label_of_element=label_of_element, label_of_model=label_of_model)

    def _do_modeling(self):
        self.shares = {
            effect: self.add_variables(coords=self._model.get_coords(['period', 'scenario']), short_name=effect)
            for effect in self._piecewise_shares
        }

        piecewise_variables = {
            self._piecewise_origin[0]: self._piecewise_origin[1],
            **{
                self.shares[effect_label].name: self._piecewise_shares[effect_label]
                for effect_label in self._piecewise_shares
            },
        }

        self.piecewise_model = self.add_submodels(
            PiecewiseModel(
                model=self._model,
                label_of_element=self.label_of_element,
                piecewise_variables=piecewise_variables,
                zero_point=self._zero_point,
                dims=('period', 'scenario'),
                label_of_model=f'{self.label_of_element}|PiecewiseEffects',
            ),
            short_name='PiecewiseEffects',
        )

        # Shares
        self._model.effects.add_share_to_effects(
            name=self.label_of_element,
            expressions={effect: variable * 1 for effect, variable in self.shares.items()},
            target='periodic',
        )


class ShareAllocationModel(Submodel):
    def __init__(
        self,
        model: FlowSystemModel,
        dims: list[FlowSystemDimensions],
        label_of_element: str | None = None,
        label_of_model: str | None = None,
        total_max: Scalar | None = None,
        total_min: Scalar | None = None,
        max_per_hour: TemporalData | None = None,
        min_per_hour: TemporalData | None = None,
    ):
        if 'time' not in dims and (max_per_hour is not None or min_per_hour is not None):
            raise ValueError('Both max_per_hour and min_per_hour cannot be used when has_time_dim is False')

        self._dims = dims
        self.total_per_timestep: linopy.Variable | None = None
        self.total: linopy.Variable | None = None
        self.shares: dict[str, linopy.Variable] = {}
        self.share_constraints: dict[str, linopy.Constraint] = {}

        self._eq_total_per_timestep: linopy.Constraint | None = None
        self._eq_total: linopy.Constraint | None = None

        # Parameters
        self._total_max = total_max if total_max is not None else np.inf
        self._total_min = total_min if total_min is not None else -np.inf
        self._max_per_hour = max_per_hour if max_per_hour is not None else np.inf
        self._min_per_hour = min_per_hour if min_per_hour is not None else -np.inf

        super().__init__(model, label_of_element=label_of_element, label_of_model=label_of_model)

    def _do_modeling(self):
        super()._do_modeling()
        self.total = self.add_variables(
            lower=self._total_min,
            upper=self._total_max,
            coords=self._model.get_coords([dim for dim in self._dims if dim != 'time']),
            name=self.label_full,
            short_name='total',
        )
        # eq: sum = sum(share_i) # skalar
        self._eq_total = self.add_constraints(self.total == 0, name=self.label_full)

        if 'time' in self._dims:
            self.total_per_timestep = self.add_variables(
                lower=-np.inf if (self._min_per_hour is None) else self._min_per_hour * self._model.hours_per_step,
                upper=np.inf if (self._max_per_hour is None) else self._max_per_hour * self._model.hours_per_step,
                coords=self._model.get_coords(self._dims),
                short_name='per_timestep',
            )

            self._eq_total_per_timestep = self.add_constraints(self.total_per_timestep == 0, short_name='per_timestep')

            # Add it to the total
            self._eq_total.lhs -= self.total_per_timestep.sum(dim='time')

    def add_share(
        self,
        name: str,
        expression: linopy.LinearExpression,
        dims: list[FlowSystemDimensions] | None = None,
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
            if 'period' in dims and 'period' not in self._dims:
                raise ValueError('Cannot add share with period-dim to a model without period-dim')
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
