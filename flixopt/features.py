"""
This module contains the features of the flixopt framework.
Features extend the functionality of Elements.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import linopy
import numpy as np
import xarray as xr

from .modeling import BoundingPatterns, ModelingPrimitives, ModelingUtilities
from .structure import FlowSystemModel, Submodel

logger = logging.getLogger('flixopt')

if TYPE_CHECKING:
    from collections.abc import Collection

    from .core import FlowSystemDimensions
    from .interface import InvestmentParameters, Piecewise, SizingParameters, StatusParameters
    from .types import Numeric_PS, Numeric_TPS, PeriodicData, PeriodicEffects


class _SizeModel(Submodel):
    """A model that creates the size variable together with a Binary"""

    def _create_sizing_variables_and_constraints(
        self,
        size_min: PeriodicData,
        size_max: PeriodicData,
        mandatory: PeriodicData,
        dims: list[FlowSystemDimensions],
        force_available: bool = False,
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

        if force_available or mandatory.any():
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
            force_available=True,
        )

        self._track_investment_and_decomissioning_period()
        self._track_investment_and_decomissioning_size()
        self._track_lifetime()
        self._apply_investment_period_constraints()

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
            state_variable=self.available,
            switch_on=self.investment_occurs,
            switch_off=self.decommissioning_occurs,
            name=self.available.name,
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
            previous_level=0
            if self.parameters.previous_lifetime is None
            else self.size.isel(period=0),  # TODO: What value?
            coord='period',
        )

    def _track_lifetime(self):
        periods = self._model.flow_system.fit_to_model_coords(
            'periods', self._model.flow_system.periods.values, dims=['period', 'scenario']
        )

        # Calculate decommissioning periods (vectorized)
        is_first = periods == periods.isel(period=0)
        decom_period = periods + self.parameters.lifetime - xr.where(is_first, self.parameters.previous_lifetime, 0)

        # Map to available periods (drop invalid ones for sel to work)
        valid = decom_period.where(decom_period <= self._model.flow_system.periods.values[-1], drop=True)
        avail_decom = periods.sel(period=valid, method='bfill').assign_coords(period=valid.period)

        # One constraint per unique decommissioning period
        for decom_val in np.unique(avail_decom.values):
            mask = (avail_decom == decom_val).reindex_like(periods).fillna(0)
            self.add_constraints(
                self.investment_occurs.where(mask).sum('period') == self.decommissioning_occurs.sel(period=decom_val),
                short_name=f'size|lifetime{int(decom_val)}',
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
        return self._variables['ava']

    @property
    def size_decrease(self) -> linopy.Variable:
        """Binary decrease decision variable"""
        return self._variables['size|decrease']

    @property
    def size_increase(self) -> linopy.Variable:
        """Binary increase decision variable"""
        return self._variables['size|increase']


class StatusModel(Submodel):
    """Status model for equipment with binary active/inactive states"""

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
        ModelingPrimitives.expression_tracking_variable(
            self,
            tracked_expression=(self.status * self._model.hours_per_step).sum('time'),
            bounds=(
                self.parameters.active_hours_min if self.parameters.active_hours_min is not None else 0,
                self.parameters.active_hours_max
                if self.parameters.active_hours_max is not None
                else self._model.hours_per_step.sum('time').max().item(),
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
                self.add_constraints(count == self.startup.sum('time'), short_name='startup_count')

        # 5. Consecutive active duration (uptime) using existing pattern
        if self.parameters.use_uptime_tracking:
            ModelingPrimitives.consecutive_duration_tracking(
                self,
                state=self.status,
                short_name='uptime',
                minimum_duration=self.parameters.min_uptime,
                maximum_duration=self.parameters.max_uptime,
                duration_per_step=self.hours_per_step,
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
                duration_per_step=self.hours_per_step,
                duration_dim='time',
                previous_duration=self._get_previous_downtime(),
            )

        self._add_effects()

    def _add_effects(self):
        """Add operational effects"""
        if self.parameters.effects_per_active_hour:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.status * factor * self._model.hours_per_step
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
        hours_per_step = self._model.hours_per_step.isel(time=0).min().item()
        if self._previous_status is None:
            return 0
        else:
            return ModelingUtilities.compute_consecutive_hours_in_state(self._previous_status, hours_per_step)

    def _get_previous_downtime(self):
        """Get previous downtime (consecutive inactive hours).

        Returns one timestep duration if no previous status is provided (assumes previously inactive).
        """
        hours_per_step = self._model.hours_per_step.isel(time=0).min().item()
        if self._previous_status is None:
            return hours_per_step
        else:
            return ModelingUtilities.compute_consecutive_hours_in_state(self._previous_status * -1 + 1, hours_per_step)


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
