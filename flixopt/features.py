"""
This module contains the features of the flixopt framework.
Features extend the functionality of Elements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import linopy
import numpy as np

from .modeling import BoundingPatterns, ModelingPrimitives, ModelingUtilities
from .structure import FlowSystemModel, Submodel

if TYPE_CHECKING:
    from collections.abc import Collection

    import xarray as xr

    from .core import FlowSystemDimensions
    from .interface import InvestParameters, Piecewise, StatusParameters
    from .types import Numeric_PS, Numeric_TPS


class InvestmentModel(Submodel):
    """Mathematical model implementation for investment decisions.

    Creates optimization variables and constraints for investment sizing decisions,
    supporting both binary and continuous sizing with comprehensive effect modeling.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/features/InvestParameters/>

    Args:
        model: The optimization model instance
        label_of_element: The label of the parent (Element). Used to construct the full label of the model.
        parameters: The parameters of the feature model.
        label_of_model: The label of the model. This is needed to construct the full label of the model.
    """

    parameters: InvestParameters

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        parameters: InvestParameters,
        label_of_model: str | None = None,
    ):
        self.piecewise_effects: PiecewiseEffectsModel | None = None
        self.parameters = parameters
        super().__init__(model, label_of_element=label_of_element, label_of_model=label_of_model)

    def _do_modeling(self):
        super()._do_modeling()
        self._create_variables_and_constraints()
        self._add_effects()

    def _create_variables_and_constraints(self):
        size_min, size_max = (self.parameters.minimum_or_fixed_size, self.parameters.maximum_or_fixed_size)
        if self.parameters.linked_periods is not None:
            # Mask size bounds: linked_periods is a binary DataArray that zeros out non-linked periods
            size_min = size_min * self.parameters.linked_periods
            size_max = size_max * self.parameters.linked_periods

        self.add_variables(
            short_name='size',
            lower=size_min if self.parameters.mandatory else 0,
            upper=size_max,
            coords=self._model.get_coords(['period', 'scenario']),
        )

        if not self.parameters.mandatory:
            self.add_variables(
                binary=True,
                coords=self._model.get_coords(['period', 'scenario']),
                short_name='invested',
            )
            BoundingPatterns.bounds_with_state(
                self,
                variable=self.size,
                state=self._variables['invested'],
                bounds=(self.parameters.minimum_or_fixed_size, self.parameters.maximum_or_fixed_size),
            )

        if self.parameters.linked_periods is not None:
            masked_size = self.size.where(self.parameters.linked_periods, drop=True)
            self.add_constraints(
                masked_size.isel(period=slice(None, -1)) == masked_size.isel(period=slice(1, None)),
                short_name='linked_periods',
            )

    def _add_effects(self):
        """Add investment effects"""
        if self.parameters.effects_of_investment:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.invested * factor if self.invested is not None else factor
                    for effect, factor in self.parameters.effects_of_investment.items()
                },
                target='periodic',
            )

        if self.parameters.effects_of_retirement and not self.parameters.mandatory:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: -self.invested * factor + factor
                    for effect, factor in self.parameters.effects_of_retirement.items()
                },
                target='periodic',
            )

        if self.parameters.effects_of_investment_per_size:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.size * factor
                    for effect, factor in self.parameters.effects_of_investment_per_size.items()
                },
                target='periodic',
            )

        if self.parameters.piecewise_effects_of_investment:
            self.piecewise_effects = self.add_submodels(
                PiecewiseEffectsModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    label_of_model=f'{self.label_of_element}|PiecewiseEffects',
                    piecewise_origin=(self.size.name, self.parameters.piecewise_effects_of_investment.piecewise_origin),
                    piecewise_shares=self.parameters.piecewise_effects_of_investment.piecewise_shares,
                    zero_point=self.invested,
                ),
                short_name='segments',
            )

    @property
    def size(self) -> linopy.Variable:
        """Investment size variable"""
        return self._variables['size']

    @property
    def invested(self) -> linopy.Variable | None:
        """Binary investment decision variable"""
        if 'invested' not in self._variables:
            return None
        return self._variables['invested']


class StatusModel(Submodel):
    """Mathematical model implementation for binary status.

    Creates optimization variables and constraints for binary status modeling,
    state transitions, duration tracking, and operational effects.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/features/StatusParameters/>
    """

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        parameters: StatusParameters,
        status: linopy.Variable,
        previous_status: xr.DataArray | None,
        label_of_model: str | None = None,
    ):
        """
        This feature model is used to model the status (active/inactive) state of flow_rate(s).
        It does not matter if the flow_rates are bounded by a size variable or by a hard bound.
        The used bound here is the absolute highest/lowest bound!

        Args:
            model: The optimization model instance
            label_of_element: The label of the parent (Element). Used to construct the full label of the model.
            parameters: The parameters of the feature model.
            status: The variable that determines the active state
            previous_status: The previous flow_rates
            label_of_model: The label of the model. This is needed to construct the full label of the model.
        """
        self.status = status
        self._previous_status = previous_status
        self.parameters = parameters
        super().__init__(model, label_of_element, label_of_model=label_of_model)

    def _do_modeling(self):
        """Create variables, constraints, and nested submodels"""
        super()._do_modeling()

        # Create a separate binary 'inactive' variable when needed for downtime tracking or explicit use
        # When not needed, the expression (1 - self.status) can be used instead
        if self.parameters.use_downtime_tracking:
            inactive = self.add_variables(binary=True, short_name='inactive', coords=self._model.get_coords())
            self.add_constraints(self.status + inactive == 1, short_name='complementary')

        # 3. Total duration tracking using existing pattern
        # Sum over all temporal dimensions (time, and cluster if present)
        weighted_status = self.status * self._model.aggregation_weight
        temporal_dims = [d for d in weighted_status.dims if d not in ('period', 'scenario')]
        agg_weight_sum = self._model.aggregation_weight.sum(temporal_dims)
        ModelingPrimitives.expression_tracking_variable(
            self,
            tracked_expression=weighted_status.sum(temporal_dims),
            bounds=(
                self.parameters.active_hours_min if self.parameters.active_hours_min is not None else 0,
                self.parameters.active_hours_max
                if self.parameters.active_hours_max is not None
                else agg_weight_sum.max().item()
                if hasattr(agg_weight_sum, 'max')
                else agg_weight_sum,
            ),
            short_name='active_hours',
            coords=['period', 'scenario'],
        )

        # 4. Switch tracking using existing pattern
        if self.parameters.use_startup_tracking:
            self.add_variables(binary=True, short_name='startup', coords=self.get_coords())
            self.add_variables(binary=True, short_name='shutdown', coords=self.get_coords())

            BoundingPatterns.state_transition_bounds(
                self,
                state=self.status,
                activate=self.startup,
                deactivate=self.shutdown,
                name=f'{self.label_of_model}|switch',
                previous_state=self._previous_status.isel(time=-1) if self._previous_status is not None else 0,
                coord='time',
            )

            if self.parameters.startup_limit is not None:
                count = self.add_variables(
                    lower=0,
                    upper=self.parameters.startup_limit,
                    coords=self._model.get_coords(('period', 'scenario')),
                    short_name='startup_count',
                )
                # Sum over all temporal dimensions (time, and cluster if present)
                startup_temporal_dims = [d for d in self.startup.dims if d not in ('period', 'scenario')]
                self.add_constraints(count == self.startup.sum(startup_temporal_dims), short_name='startup_count')

        # 5. Consecutive active duration (uptime) using existing pattern
        if self.parameters.use_uptime_tracking:
            ModelingPrimitives.consecutive_duration_tracking(
                self,
                state=self.status,
                short_name='uptime',
                minimum_duration=self.parameters.min_uptime,
                maximum_duration=self.parameters.max_uptime,
                duration_per_step=self.timestep_duration,
                duration_dim='time',
                previous_duration=self._get_previous_uptime(),
            )

        # 6. Consecutive inactive duration (downtime) using existing pattern
        if self.parameters.use_downtime_tracking:
            ModelingPrimitives.consecutive_duration_tracking(
                self,
                state=self.inactive,
                short_name='downtime',
                minimum_duration=self.parameters.min_downtime,
                maximum_duration=self.parameters.max_downtime,
                duration_per_step=self.timestep_duration,
                duration_dim='time',
                previous_duration=self._get_previous_downtime(),
            )

        self._add_effects()

    def _add_effects(self):
        """Add operational effects (use timestep_duration only, cluster_weight is applied when summing to total)"""
        if self.parameters.effects_per_active_hour:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.status * factor * self._model.timestep_duration
                    for effect, factor in self.parameters.effects_per_active_hour.items()
                },
                target='temporal',
            )

        if self.parameters.effects_per_startup:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.startup * factor for effect, factor in self.parameters.effects_per_startup.items()
                },
                target='temporal',
            )

    # Properties access variables from Submodel's tracking system

    @property
    def active_hours(self) -> linopy.Variable:
        """Total active hours variable"""
        return self['active_hours']

    @property
    def inactive(self) -> linopy.Variable | None:
        """Binary inactive state variable.

        Note:
            Only created when downtime tracking is enabled (min_downtime or max_downtime set).
            For general use, prefer the expression `1 - status` instead of this variable.
        """
        return self.get('inactive')

    @property
    def startup(self) -> linopy.Variable | None:
        """Startup variable"""
        return self.get('startup')

    @property
    def shutdown(self) -> linopy.Variable | None:
        """Shutdown variable"""
        return self.get('shutdown')

    @property
    def startup_count(self) -> linopy.Variable | None:
        """Number of startups variable"""
        return self.get('startup_count')

    @property
    def uptime(self) -> linopy.Variable | None:
        """Consecutive active hours (uptime) variable"""
        return self.get('uptime')

    @property
    def downtime(self) -> linopy.Variable | None:
        """Consecutive inactive hours (downtime) variable"""
        return self.get('downtime')

    def _get_previous_uptime(self):
        """Get previous uptime (consecutive active hours).

        Returns 0 if no previous status is provided (assumes previously inactive).
        """
        hours_per_step = self._model.timestep_duration.isel(time=0).min().item()
        if self._previous_status is None:
            return 0
        else:
            return ModelingUtilities.compute_consecutive_hours_in_state(self._previous_status, hours_per_step)

    def _get_previous_downtime(self):
        """Get previous downtime (consecutive inactive hours).

        Returns one timestep duration if no previous status is provided (assumes previously inactive).
        """
        hours_per_step = self._model.timestep_duration.isel(time=0).min().item()
        if self._previous_status is None:
            return hours_per_step
        else:
            return ModelingUtilities.compute_consecutive_hours_in_state(1 - self._previous_status, hours_per_step)


class PieceModel(Submodel):
    """Class for modeling a linear piece of one or more variables in parallel"""

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        label_of_model: str,
        dims: Collection[FlowSystemDimensions] | None,
    ):
        self.inside_piece: linopy.Variable | None = None
        self.lambda0: linopy.Variable | None = None
        self.lambda1: linopy.Variable | None = None
        self.dims = dims

        super().__init__(model, label_of_element, label_of_model)

    def _do_modeling(self):
        """Create variables, constraints, and nested submodels"""
        super()._do_modeling()

        # Create variables
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

        # Create constraints
        # eq:  lambda0(t) + lambda1(t) = inside_piece(t)
        self.add_constraints(self.inside_piece == self.lambda0 + self.lambda1, short_name='inside_piece')


class PiecewiseModel(Submodel):
    """Mathematical model implementation for piecewise linear approximations.

    Creates optimization variables and constraints for piecewise linear relationships,
    including lambda variables, piece activation binaries, and coupling constraints.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/features/Piecewise/>
    """

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        label_of_model: str,
        piecewise_variables: dict[str, Piecewise],
        zero_point: bool | linopy.Variable | None,
        dims: Collection[FlowSystemDimensions] | None,
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
        """Create variables, constraints, and nested submodels"""
        super()._do_modeling()

        # Validate all piecewise variables have the same number of segments
        segment_counts = [len(pw) for pw in self._piecewise_variables.values()]
        if not all(count == segment_counts[0] for count in segment_counts):
            raise ValueError(f'All piecewises must have the same number of pieces, got {segment_counts}')

        # Create PieceModel submodels (which creates their variables and constraints)
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

            # This constraint ensures at most one segment is active at a time.
            # When zero_point is a binary variable, it acts as a gate:
            # - zero_point=1: at most one segment can be active (normal piecewise operation)
            # - zero_point=0: all segments must be inactive (effectively disables the piecewise)
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
        """Create variables, constraints, and nested submodels"""
        super()._do_modeling()

        # Create variables
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

        # Create piecewise model (which creates its variables and constraints)
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

        # Add shares to effects
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
        total_max: Numeric_PS | None = None,
        total_min: Numeric_PS | None = None,
        max_per_hour: Numeric_TPS | None = None,
        min_per_hour: Numeric_TPS | None = None,
    ):
        if 'time' not in dims and (max_per_hour is not None or min_per_hour is not None):
            raise ValueError("max_per_hour and min_per_hour require 'time' dimension in dims")

        self._dims = dims
        self.total_per_timestep: linopy.Variable | None = None
        self.total: linopy.Variable | None = None
        self.shares: dict[str, linopy.Variable] = {}
        self.share_constraints: dict[str, linopy.Constraint] = {}

        self._eq_total_per_timestep: linopy.Constraint | None = None
        self._eq_total: linopy.Constraint | None = None

        # Parameters
        self._total_max = total_max
        self._total_min = total_min
        self._max_per_hour = max_per_hour
        self._min_per_hour = min_per_hour

        super().__init__(model, label_of_element=label_of_element, label_of_model=label_of_model)

    def _do_modeling(self):
        """Create variables, constraints, and nested submodels"""
        super()._do_modeling()

        # Create variables
        self.total = self.add_variables(
            lower=self._total_min if self._total_min is not None else -np.inf,
            upper=self._total_max if self._total_max is not None else np.inf,
            coords=self._model.get_coords([dim for dim in self._dims if dim != 'time']),
            name=self.label_full,
            short_name='total',
        )
        # eq: sum = sum(share_i) # skalar
        self._eq_total = self.add_constraints(self.total == 0, name=self.label_full)

        if 'time' in self._dims:
            self.total_per_timestep = self.add_variables(
                lower=-np.inf if (self._min_per_hour is None) else self._min_per_hour * self._model.timestep_duration,
                upper=np.inf if (self._max_per_hour is None) else self._max_per_hour * self._model.timestep_duration,
                coords=self._model.get_coords(self._dims),
                short_name='per_timestep',
            )

            self._eq_total_per_timestep = self.add_constraints(self.total_per_timestep == 0, short_name='per_timestep')

            # Add it to the total (cluster_weight handles cluster representation, defaults to 1.0)
            # Sum over all temporal dimensions (time, and cluster if present)
            weighted_per_timestep = self.total_per_timestep * self._model.cluster_weight
            temporal_dims = [d for d in weighted_per_timestep.dims if d not in ('period', 'scenario')]
            self._eq_total.lhs -= weighted_per_timestep.sum(dim=temporal_dims)

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
