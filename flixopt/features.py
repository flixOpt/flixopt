"""
This module contains the features of the flixopt framework.
Features extend the functionality of Elements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import linopy
import numpy as np

from .modeling import BoundingPatterns, ModelingPrimitives, ModelingUtilities
from .structure import FlowSystemModel, Submodel, VariableCategory

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
        size_category: Category for the size variable (FLOW_SIZE, STORAGE_SIZE, or SIZE for generic).
    """

    parameters: InvestParameters

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        parameters: InvestParameters,
        label_of_model: str | None = None,
        size_category: VariableCategory = VariableCategory.SIZE,
    ):
        self.piecewise_effects: PiecewiseEffectsModel | None = None
        self.parameters = parameters
        self._size_category = size_category
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
            category=self._size_category,
        )

        if not self.parameters.mandatory:
            self.add_variables(
                binary=True,
                coords=self._model.get_coords(['period', 'scenario']),
                short_name='invested',
                category=VariableCategory.INVESTED,
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


class InvestmentProxy:
    """Proxy providing access to batched InvestmentsModel for a specific element.

    This class provides the same interface as InvestmentModel.size/invested
    but returns slices from the batched InvestmentsModel variables.
    """

    def __init__(self, investments_model: InvestmentsModel, element_id: str):
        self._investments_model = investments_model
        self._element_id = element_id

    @property
    def size(self):
        """Investment size variable for this element."""
        return self._investments_model.get_variable('size', self._element_id)

    @property
    def invested(self):
        """Binary investment decision variable for this element (if non-mandatory)."""
        return self._investments_model.get_variable('invested', self._element_id)


class InvestmentsModel:
    """Type-level model for batched investment decisions across multiple elements.

    Unlike InvestmentModel (one per element), InvestmentsModel handles ALL elements
    with investment in a single instance with batched variables.

    This enables:
    - Batched `size` and `invested` variables with element dimension
    - Vectorized constraint creation
    - Batched effect shares

    The model categorizes elements by investment type:
    - mandatory: Required investment (only size variable, with bounds)
    - non_mandatory: Optional investment (size + invested variables, state-controlled bounds)

    Example:
        >>> investments_model = InvestmentsModel(
        ...     model=flow_system_model,
        ...     elements=storages_with_investment,
        ...     parameters_getter=lambda s: s.capacity_in_flow_hours,
        ...     size_category=VariableCategory.STORAGE_SIZE,
        ... )
        >>> investments_model.create_variables()
        >>> investments_model.create_constraints()
        >>> investments_model.create_effect_shares()
    """

    def __init__(
        self,
        model: FlowSystemModel,
        elements: list,
        parameters_getter: callable,
        size_category: VariableCategory = VariableCategory.SIZE,
    ):
        """Initialize the type-level investment model.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            elements: List of elements with InvestParameters.
            parameters_getter: Function to get InvestParameters from element.
                e.g., lambda storage: storage.capacity_in_flow_hours
            size_category: Category for size variable expansion.
        """
        import logging

        import pandas as pd
        import xarray as xr

        self._logger = logging.getLogger('flixopt')
        self.model = model
        self.elements = elements
        self.element_ids: list[str] = [e.label_full for e in elements]
        self._parameters_getter = parameters_getter
        self._size_category = size_category

        # Storage for created variables
        self._variables: dict[str, linopy.Variable] = {}

        # Categorize by mandatory/non-mandatory
        self._mandatory_elements: list = []
        self._mandatory_ids: list[str] = []
        self._non_mandatory_elements: list = []
        self._non_mandatory_ids: list[str] = []

        for element in elements:
            params = parameters_getter(element)
            if params.mandatory:
                self._mandatory_elements.append(element)
                self._mandatory_ids.append(element.label_full)
            else:
                self._non_mandatory_elements.append(element)
                self._non_mandatory_ids.append(element.label_full)

        # Store xr and pd for use in methods
        self._xr = xr
        self._pd = pd

    def create_variables(self) -> None:
        """Create batched investment variables with element dimension.

        Creates:
        - size: For ALL elements (with element dimension)
        - invested: For non-mandatory elements only (binary, with element dimension)
        """
        from .structure import VARIABLE_TYPE_TO_EXPANSION, VariableType

        if not self.elements:
            return

        xr = self._xr
        pd = self._pd

        # Get base coords (period, scenario) - may be None if neither exist
        base_coords = self.model.get_coords(['period', 'scenario'])
        base_coords_dict = dict(base_coords) if base_coords is not None else {}

        # === size: ALL elements ===
        # Collect bounds per element
        lower_bounds_list = []
        upper_bounds_list = []

        for element in self.elements:
            params = self._parameters_getter(element)
            size_min = params.minimum_or_fixed_size
            size_max = params.maximum_or_fixed_size

            # Handle linked_periods masking
            if params.linked_periods is not None:
                size_min = size_min * params.linked_periods
                size_max = size_max * params.linked_periods

            # For non-mandatory, lower bound is 0 (invested variable controls actual minimum)
            if not params.mandatory:
                size_min = xr.zeros_like(size_min) if isinstance(size_min, xr.DataArray) else 0

            lower_bounds_list.append(size_min if isinstance(size_min, xr.DataArray) else xr.DataArray(size_min))
            upper_bounds_list.append(size_max if isinstance(size_max, xr.DataArray) else xr.DataArray(size_max))

        # Stack bounds into DataArrays with element dimension
        lower_bounds = xr.concat(lower_bounds_list, dim='element').assign_coords(element=self.element_ids)
        upper_bounds = xr.concat(upper_bounds_list, dim='element').assign_coords(element=self.element_ids)

        # Build coords with element dimension
        size_coords = xr.Coordinates(
            {
                'element': pd.Index(self.element_ids, name='element'),
                **base_coords_dict,
            }
        )

        size_var = self.model.add_variables(
            lower=lower_bounds,
            upper=upper_bounds,
            coords=size_coords,
            name='investment|size',
        )
        self._variables['size'] = size_var

        # Register category for segment expansion
        expansion_category = VARIABLE_TYPE_TO_EXPANSION.get(VariableType.SIZE)
        if expansion_category is not None:
            self.model.variable_categories[size_var.name] = expansion_category

        # === invested: non-mandatory elements only ===
        if self._non_mandatory_elements:
            invested_coords = xr.Coordinates(
                {
                    'element': pd.Index(self._non_mandatory_ids, name='element'),
                    **base_coords_dict,
                }
            )

            invested_var = self.model.add_variables(
                binary=True,
                coords=invested_coords,
                name='investment|invested',
            )
            self._variables['invested'] = invested_var

            # Register category
            expansion_category = VARIABLE_TYPE_TO_EXPANSION.get(VariableType.INVESTED)
            if expansion_category is not None:
                self.model.variable_categories[invested_var.name] = expansion_category

        self._logger.debug(
            f'InvestmentsModel created variables: {len(self.elements)} elements '
            f'({len(self._mandatory_elements)} mandatory, {len(self._non_mandatory_elements)} non-mandatory)'
        )

    def create_constraints(self) -> None:
        """Create batched investment constraints.

        For non-mandatory investments, creates state-controlled bounds:
            invested * min_size <= size <= invested * max_size
        """
        if not self._non_mandatory_elements:
            return

        xr = self._xr

        size_var = self._variables['size']
        invested_var = self._variables['invested']

        # Collect bounds for non-mandatory elements
        min_bounds_list = []
        max_bounds_list = []

        for element in self._non_mandatory_elements:
            params = self._parameters_getter(element)
            min_bounds_list.append(
                params.minimum_or_fixed_size
                if isinstance(params.minimum_or_fixed_size, xr.DataArray)
                else xr.DataArray(params.minimum_or_fixed_size)
            )
            max_bounds_list.append(
                params.maximum_or_fixed_size
                if isinstance(params.maximum_or_fixed_size, xr.DataArray)
                else xr.DataArray(params.maximum_or_fixed_size)
            )

        min_bounds = xr.concat(min_bounds_list, dim='element').assign_coords(element=self._non_mandatory_ids)
        max_bounds = xr.concat(max_bounds_list, dim='element').assign_coords(element=self._non_mandatory_ids)

        # Select size for non-mandatory elements
        size_non_mandatory = size_var.sel(element=self._non_mandatory_ids)

        # State-controlled bounds: invested * min <= size <= invested * max
        # Lower bound with epsilon to force non-zero when invested
        from .config import CONFIG

        epsilon = CONFIG.Modeling.epsilon
        effective_min = xr.where(min_bounds > epsilon, min_bounds, epsilon)

        self.model.add_constraints(
            size_non_mandatory >= invested_var * effective_min,
            name='investment|size|lb',
        )
        self.model.add_constraints(
            size_non_mandatory <= invested_var * max_bounds,
            name='investment|size|ub',
        )

        # Handle linked_periods constraints
        self._add_linked_periods_constraints()

        self._logger.debug(
            f'InvestmentsModel created constraints for {len(self._non_mandatory_elements)} non-mandatory elements'
        )

    def _add_linked_periods_constraints(self) -> None:
        """Add linked periods constraints for elements that have them."""
        size_var = self._variables['size']

        for element in self.elements:
            params = self._parameters_getter(element)
            if params.linked_periods is not None:
                element_size = size_var.sel(element=element.label_full)
                masked_size = element_size.where(params.linked_periods, drop=True)
                if 'period' in masked_size.dims and masked_size.sizes.get('period', 0) > 1:
                    self.model.add_constraints(
                        masked_size.isel(period=slice(None, -1)) == masked_size.isel(period=slice(1, None)),
                        name=f'{element.label_full}|linked_periods',
                    )

    def create_effect_shares(self) -> None:
        """Create batched effect shares for investment effects.

        Handles:
        - effects_of_investment (fixed costs)
        - effects_of_investment_per_size (variable costs)
        - effects_of_retirement (divestment costs)

        Note: piecewise_effects_of_investment is handled per-element due to complexity.
        """
        size_var = self._variables['size']
        invested_var = self._variables.get('invested')

        # Collect effect shares by effect name
        fix_effects: dict[str, list[tuple[str, any]]] = {}  # effect_name -> [(element_id, factor), ...]
        per_size_effects: dict[str, list[tuple[str, any]]] = {}
        retirement_effects: dict[str, list[tuple[str, any]]] = {}

        for element in self.elements:
            params = self._parameters_getter(element)
            element_id = element.label_full

            if params.effects_of_investment:
                for effect_name, factor in params.effects_of_investment.items():
                    if effect_name not in fix_effects:
                        fix_effects[effect_name] = []
                    fix_effects[effect_name].append((element_id, factor))

            if params.effects_of_investment_per_size:
                for effect_name, factor in params.effects_of_investment_per_size.items():
                    if effect_name not in per_size_effects:
                        per_size_effects[effect_name] = []
                    per_size_effects[effect_name].append((element_id, factor))

            if params.effects_of_retirement and not params.mandatory:
                for effect_name, factor in params.effects_of_retirement.items():
                    if effect_name not in retirement_effects:
                        retirement_effects[effect_name] = []
                    retirement_effects[effect_name].append((element_id, factor))

        # Apply fixed effects (factor * invested or factor if mandatory)
        for effect_name, element_factors in fix_effects.items():
            expressions = {}
            for element_id, factor in element_factors:
                element = next(e for e in self.elements if e.label_full == element_id)
                params = self._parameters_getter(element)
                if params.mandatory:
                    # Always incurred
                    expressions[element_id] = factor
                else:
                    # Only if invested
                    invested_elem = invested_var.sel(element=element_id)
                    expressions[element_id] = invested_elem * factor

            # Add to effects (per-element for now, could be batched further)
            for element_id, expr in expressions.items():
                self.model.effects.add_share_to_effects(
                    name=f'{element_id}|invest_fix',
                    expressions={effect_name: expr},
                    target='periodic',
                )

        # Apply per-size effects (size * factor)
        for effect_name, element_factors in per_size_effects.items():
            for element_id, factor in element_factors:
                size_elem = size_var.sel(element=element_id)
                self.model.effects.add_share_to_effects(
                    name=f'{element_id}|invest_per_size',
                    expressions={effect_name: size_elem * factor},
                    target='periodic',
                )

        # Apply retirement effects (-invested * factor + factor)
        for effect_name, element_factors in retirement_effects.items():
            for element_id, factor in element_factors:
                invested_elem = invested_var.sel(element=element_id)
                self.model.effects.add_share_to_effects(
                    name=f'{element_id}|invest_retire',
                    expressions={effect_name: -invested_elem * factor + factor},
                    target='periodic',
                )

        self._logger.debug('InvestmentsModel created effect shares')

    def get_variable(self, name: str, element_id: str | None = None):
        """Get a variable, optionally selecting a specific element."""
        var = self._variables.get(name)
        if var is None:
            return None
        if element_id is not None:
            if element_id in var.coords.get('element', []):
                return var.sel(element=element_id)
            return None
        return var

    @property
    def size(self) -> linopy.Variable:
        """Batched size variable with element dimension."""
        return self._variables['size']

    @property
    def invested(self) -> linopy.Variable | None:
        """Batched invested variable with element dimension (non-mandatory only)."""
        return self._variables.get('invested')


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
            inactive = self.add_variables(
                binary=True,
                short_name='inactive',
                coords=self._model.get_coords(),
                category=VariableCategory.INACTIVE,
            )
            self.add_constraints(self.status + inactive == 1, short_name='complementary')

        # 3. Total duration tracking
        total_hours = self._model.temporal_weight.sum(self._model.temporal_dims)
        ModelingPrimitives.expression_tracking_variable(
            self,
            tracked_expression=self._model.sum_temporal(self.status),
            bounds=(
                self.parameters.active_hours_min if self.parameters.active_hours_min is not None else 0,
                self.parameters.active_hours_max if self.parameters.active_hours_max is not None else total_hours,
            ),
            short_name='active_hours',
            coords=['period', 'scenario'],
            category=VariableCategory.TOTAL,
        )

        # 4. Switch tracking using existing pattern
        if self.parameters.use_startup_tracking:
            self.add_variables(
                binary=True,
                short_name='startup',
                coords=self.get_coords(),
                category=VariableCategory.STARTUP,
            )
            self.add_variables(
                binary=True,
                short_name='shutdown',
                coords=self.get_coords(),
                category=VariableCategory.SHUTDOWN,
            )

            # Determine previous_state: None means relaxed (no constraint at t=0)
            previous_state = self._previous_status.isel(time=-1) if self._previous_status is not None else None

            BoundingPatterns.state_transition_bounds(
                self,
                state=self.status,
                activate=self.startup,
                deactivate=self.shutdown,
                name=f'{self.label_of_model}|switch',
                previous_state=previous_state,
                coord='time',
            )

            if self.parameters.startup_limit is not None:
                count = self.add_variables(
                    lower=0,
                    upper=self.parameters.startup_limit,
                    coords=self._model.get_coords(('period', 'scenario')),
                    short_name='startup_count',
                    category=VariableCategory.STARTUP_COUNT,
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

        # 7. Cyclic constraint for clustered systems
        self._add_cluster_cyclic_constraint()

        self._add_effects()

    def _add_cluster_cyclic_constraint(self):
        """For 'cyclic' cluster mode: each cluster's start status equals its end status."""
        if self._model.flow_system.clusters is not None and self.parameters.cluster_mode == 'cyclic':
            self.add_constraints(
                self.status.isel(time=0) == self.status.isel(time=-1),
                short_name='cluster_cyclic',
            )

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

        Returns None if no previous status is provided (relaxed mode - no constraint at t=0).
        """
        if self._previous_status is None:
            return None  # Relaxed mode
        hours_per_step = self._model.timestep_duration.isel(time=0).min().item()
        return ModelingUtilities.compute_consecutive_hours_in_state(self._previous_status, hours_per_step)

    def _get_previous_downtime(self):
        """Get previous downtime (consecutive inactive hours).

        Returns None if no previous status is provided (relaxed mode - no constraint at t=0).
        """
        if self._previous_status is None:
            return None  # Relaxed mode
        hours_per_step = self._model.timestep_duration.isel(time=0).min().item()
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
            category=VariableCategory.INSIDE_PIECE,
        )
        self.lambda0 = self.add_variables(
            lower=0,
            upper=1,
            short_name='lambda0',
            coords=self._model.get_coords(dims=self.dims),
            category=VariableCategory.LAMBDA0,
        )

        self.lambda1 = self.add_variables(
            lower=0,
            upper=1,
            short_name='lambda1',
            coords=self._model.get_coords(dims=self.dims),
            category=VariableCategory.LAMBDA1,
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
                    category=VariableCategory.ZERO_POINT,
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
            category=VariableCategory.TOTAL,
        )
        # eq: sum = sum(share_i) # skalar
        self._eq_total = self.add_constraints(self.total == 0, name=self.label_full)

        if 'time' in self._dims:
            self.total_per_timestep = self.add_variables(
                lower=-np.inf if (self._min_per_hour is None) else self._min_per_hour * self._model.timestep_duration,
                upper=np.inf if (self._max_per_hour is None) else self._max_per_hour * self._model.timestep_duration,
                coords=self._model.get_coords(self._dims),
                short_name='per_timestep',
                category=VariableCategory.PER_TIMESTEP,
            )

            self._eq_total_per_timestep = self.add_constraints(self.total_per_timestep == 0, short_name='per_timestep')

            # Add it to the total (cluster_weight handles cluster representation, defaults to 1.0)
            # Sum over all temporal dimensions (time, and cluster if present)
            weighted_per_timestep = self.total_per_timestep * self._model.weights.get('cluster', 1.0)
            self._eq_total.lhs -= weighted_per_timestep.sum(dim=self._model.temporal_dims)

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
            # Temporal shares (with 'time' dim) are segment totals that need division
            category = VariableCategory.SHARE if 'time' in dims else None
            self.shares[name] = self.add_variables(
                coords=self._model.get_coords(dims),
                name=f'{name}->{self.label_full}',
                short_name=name,
                category=category,
            )

            self.share_constraints[name] = self.add_constraints(
                self.shares[name] == expression, name=f'{name}->{self.label_full}'
            )

            if 'time' not in dims:
                self._eq_total.lhs -= self.shares[name]
            else:
                self._eq_total_per_timestep.lhs -= self.shares[name]
