"""
This module contains the effects of the flixopt framework.
Furthermore, it contains the EffectCollection, which is used to collect all effects of a system.
Different Datatypes are used to represent the effects with assigned values by the user,
which are then transformed into the internal data structure.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Literal

import linopy
import numpy as np
import xarray as xr

from .core import PlausibilityError
from .features import ShareAllocationModel
from .structure import Element, ElementContainer, ElementModel, FlowSystemModel, Submodel, register_class_for_io

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .types import Effect_PS, Effect_TPS, Effect_VS, Numeric_PS, Numeric_S, Numeric_TPS, Numeric_VS, Scalar

logger = logging.getLogger('flixopt')

# Penalty effect label constant
PENALTY_EFFECT_LABEL = 'Penalty'


@register_class_for_io
class Effect(Element):
    """Represents system-wide impacts like costs, emissions, or resource consumption.

    Effects quantify impacts aggregating contributions from Elements across the FlowSystem.
    One Effect serves as the optimization objective, while others can be constrained or tracked.
    Supports operational and investment contributions, cross-effect relationships (e.g., carbon
    pricing), and flexible constraint formulation.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/effects-and-dimensions/>

    Args:
        label: The label of the Element. Used to identify it in the FlowSystem.
        unit: The unit of the effect (e.g., '€', 'kg_CO2', 'kWh_primary', 'm²').
            This is informative only and does not affect optimization.
        description: Descriptive name explaining what this effect represents.
        is_standard: If True, this is a standard effect allowing direct value input
            without effect dictionaries. Used for simplified effect specification (and less boilerplate code).
        is_objective: If True, this effect serves as the optimization objective function.
            Only one effect can be marked as objective per optimization.
        period_weights: Optional custom weights for periods and scenarios (Numeric_PS).
            If provided, overrides the FlowSystem's default period weights for this effect.
            Useful for effect-specific weighting (e.g., discounting for costs vs equal weights for CO2).
            If None, uses FlowSystem's default weights.
        vintage_weights: Optional custom weights for vintages and scenarios (Numeric_VS).
            If provided, overrides the FlowSystem's default vintage weights for this effect.
            Useful for discounting one-time investment costs.
            If None, uses FlowSystem's default weights (typically 1).
        share_from_temporal: Temporal cross-effect contributions.
            Maps temporal contributions from other effects to this effect.
        share_from_periodic: Periodic cross-effect contributions.
            Maps periodic contributions from other effects to this effect.
        share_from_vintage: Vintage cross-effect contributions.
            Maps vintage contributions from other effects to this effect.
        minimum_temporal: Minimum allowed total contribution across all timesteps (per period).
        maximum_temporal: Maximum allowed total contribution across all timesteps (per period).
        minimum_per_hour: Minimum allowed contribution per hour.
        maximum_per_hour: Maximum allowed contribution per hour.
        minimum_periodic: Minimum allowed total periodic contribution (per period).
        maximum_periodic: Maximum allowed total periodic contribution (per period).
        minimum_vintage: Minimum allowed total vintage contribution (per vintage).
        maximum_vintage: Maximum allowed total vintage contribution (per vintage).
        minimum_over_periods: Minimum allowed weighted sum of (temporal + periodic) across ALL periods.
            Weighted by effect-specific weights if defined, otherwise by FlowSystem period weights.
            Requires FlowSystem to have a 'period' dimension (i.e., periods must be defined).
        maximum_over_periods: Maximum allowed weighted sum of (temporal + periodic) across ALL periods.
            Weighted by effect-specific weights if defined, otherwise by FlowSystem period weights.
            Requires FlowSystem to have a 'period' dimension (i.e., periods must be defined).
        minimum_over_vintages: Minimum allowed weighted sum of vintage across ALL vintages.
            Weighted by vintage_weights if defined, otherwise by FlowSystem vintage weights.
            Requires FlowSystem to have a 'vintage' dimension.
        maximum_over_vintages: Maximum allowed weighted sum of vintage across ALL vintages.
            Weighted by vintage_weights if defined, otherwise by FlowSystem vintage weights.
            Requires FlowSystem to have a 'vintage' dimension.
        minimum_overall: Minimum allowed sum of over_periods + over_vintages.
            This is the grand total across all effect categories.
        maximum_overall: Maximum allowed sum of over_periods + over_vintages.
            This is the grand total across all effect categories.
        meta_data: Used to store additional information. Not used internally but saved
            in results. Only use Python native types.

    **Deprecated Parameters** (for backwards compatibility):
        minimum_operation: Use `minimum_temporal` instead.
        maximum_operation: Use `maximum_temporal` instead.
        minimum_invest: Use `minimum_periodic` instead.
        maximum_invest: Use `maximum_periodic` instead.
        minimum_operation_per_hour: Use `minimum_per_hour` instead.
        maximum_operation_per_hour: Use `maximum_per_hour` instead.

    Examples:
        Basic cost objective:

        ```python
        cost_effect = Effect(
            label='system_costs',
            unit='€',
            description='Total system costs',
            is_objective=True,
        )
        ```

        CO2 emissions with per-period limit:

        ```python
        co2_effect = Effect(
            label='CO2',
            unit='kg_CO2',
            description='Carbon dioxide emissions',
            maximum_total=100_000,  # 100 t CO2 per period
        )
        ```

        CO2 emissions with total limit across all periods:

        ```python
        co2_effect = Effect(
            label='CO2',
            unit='kg_CO2',
            description='Carbon dioxide emissions',
            maximum_over_periods=1_000_000,  # 1000 t CO2 total across all periods
        )
        ```

        Land use constraint:

        ```python
        land_use = Effect(
            label='land_usage',
            unit='m²',
            description='Land area requirement',
            maximum_total=50_000,  # Maximum 5 hectares per period
        )
        ```

        Primary energy tracking:

        ```python
        primary_energy = Effect(
            label='primary_energy',
            unit='kWh_primary',
            description='Primary energy consumption',
        )
        ```

       Cost objective with carbon and primary energy pricing:

        ```python
        cost_effect = Effect(
            label='system_costs',
            unit='€',
            description='Total system costs',
            is_objective=True,
            share_from_temporal={
                'primary_energy': 0.08,  # 0.08 €/kWh_primary
                'CO2': 0.2,  # Carbon pricing: 0.2 €/kg_CO2 into costs if used on a cost effect
            },
        )
        ```

        Water consumption with tiered constraints:

        ```python
        water_usage = Effect(
            label='water_consumption',
            unit='m³',
            description='Industrial water usage',
            minimum_per_hour=10,  # Minimum 10 m³/h for process stability
            maximum_per_hour=500,  # Maximum 500 m³/h capacity limit
            maximum_over_periods=100_000,  # Annual permit limit: 100,000 m³
        )
        ```

    Note:
        Effect bounds can be None to indicate no constraint in that direction.

        Cross-effect relationships enable sophisticated modeling like carbon pricing,
        resource valuation, or multi-criteria optimization with weighted objectives.

        The unit field is purely informational - ensure dimensional consistency
        across all contributions to each effect manually.

        Effects are accumulated as:
        - Total = Σ(temporal contributions) + Σ(periodic contributions)

    """

    submodel: EffectModel | None

    def __init__(
        self,
        label: str,
        unit: str,
        description: str,
        meta_data: dict | None = None,
        is_standard: bool = False,
        is_objective: bool = False,
        period_weights: Numeric_PS | None = None,
        vintage_weights: Numeric_VS | None = None,
        share_from_temporal: Effect_TPS | Numeric_TPS | None = None,
        share_from_periodic: Effect_PS | Numeric_PS | None = None,
        share_from_vintage: Effect_VS | Numeric_VS | None = None,
        minimum_temporal: Numeric_PS | None = None,
        maximum_temporal: Numeric_PS | None = None,
        minimum_periodic: Numeric_PS | None = None,
        maximum_periodic: Numeric_PS | None = None,
        minimum_vintage: Numeric_VS | None = None,
        maximum_vintage: Numeric_VS | None = None,
        minimum_per_hour: Numeric_TPS | None = None,
        maximum_per_hour: Numeric_TPS | None = None,
        minimum_total: Numeric_PS | None = None,
        maximum_total: Numeric_PS | None = None,
        minimum_over_periods: Numeric_S | None = None,
        maximum_over_periods: Numeric_S | None = None,
        minimum_over_vintages: Numeric_S | None = None,
        maximum_over_vintages: Numeric_S | None = None,
        minimum_overall: Numeric_S | None = None,
        maximum_overall: Numeric_S | None = None,
    ):
        super().__init__(label, meta_data=meta_data)
        self.unit = unit
        self.description = description
        self.is_standard = is_standard

        # Validate that Penalty cannot be set as objective
        if is_objective and label == PENALTY_EFFECT_LABEL:
            raise ValueError(
                f'The Penalty effect ("{PENALTY_EFFECT_LABEL}") cannot be set as the objective effect. '
                f'Please use a different effect as the optimization objective.'
            )

        self.is_objective = is_objective
        self.period_weights = period_weights
        self.vintage_weights = vintage_weights
        # Share parameters accept Effect_* | Numeric_* unions (dict or single value).
        # Store as-is here; transform_data() will normalize via fit_effects_to_model_coords().
        # Default to {} when None (no shares defined).
        self.share_from_temporal = share_from_temporal if share_from_temporal is not None else {}
        self.share_from_periodic = share_from_periodic if share_from_periodic is not None else {}
        self.share_from_vintage = share_from_vintage if share_from_vintage is not None else {}

        # Set attributes directly
        self.minimum_temporal = minimum_temporal
        self.maximum_temporal = maximum_temporal
        self.minimum_periodic = minimum_periodic
        self.maximum_periodic = maximum_periodic
        self.minimum_vintage = minimum_vintage
        self.maximum_vintage = maximum_vintage
        self.minimum_per_hour = minimum_per_hour
        self.maximum_per_hour = maximum_per_hour
        self.minimum_over_periods = minimum_over_periods
        self.maximum_over_periods = maximum_over_periods
        self.minimum_over_vintages = minimum_over_vintages
        self.maximum_over_vintages = maximum_over_vintages
        self.minimum_overall = minimum_overall
        self.maximum_overall = maximum_overall
        self.minimum_total = minimum_total
        self.maximum_total = maximum_total

    def transform_data(self, name_prefix: str = '') -> None:
        prefix = '|'.join(filter(None, [name_prefix, self.label_full]))
        self.minimum_per_hour = self._fit_coords(f'{prefix}|minimum_per_hour', self.minimum_per_hour)
        self.maximum_per_hour = self._fit_coords(f'{prefix}|maximum_per_hour', self.maximum_per_hour)

        # Cross-effect shares
        self.share_from_temporal = self._fit_effect_coords(
            prefix=None,
            effect_values=self.share_from_temporal,
            suffix=f'(temporal)->{prefix}(temporal)',
            dims=['time', 'period', 'scenario'],
        )
        self.share_from_periodic = self._fit_effect_coords(
            prefix=None,
            effect_values=self.share_from_periodic,
            suffix=f'(periodic)->{prefix}(periodic)',
            dims=['period', 'scenario'],
        )
        self.share_from_vintage = self._fit_effect_coords(
            prefix=None,
            effect_values=self.share_from_vintage,
            suffix=f'(vintage)->{prefix}(vintage)',
            dims=['vintage', 'scenario'],
        )

        # Temporal bounds
        self.minimum_temporal = self._fit_coords(
            f'{prefix}|minimum_temporal', self.minimum_temporal, dims=['period', 'scenario']
        )
        self.maximum_temporal = self._fit_coords(
            f'{prefix}|maximum_temporal', self.maximum_temporal, dims=['period', 'scenario']
        )

        # Periodic bounds
        self.minimum_periodic = self._fit_coords(
            f'{prefix}|minimum_periodic', self.minimum_periodic, dims=['period', 'scenario']
        )
        self.maximum_periodic = self._fit_coords(
            f'{prefix}|maximum_periodic', self.maximum_periodic, dims=['period', 'scenario']
        )

        # Total bounds (temporal + periodic per period)
        self.minimum_total = self._fit_coords(
            f'{prefix}|minimum_total', self.minimum_total, dims=['period', 'scenario']
        )
        self.maximum_total = self._fit_coords(
            f'{prefix}|maximum_total', self.maximum_total, dims=['period', 'scenario']
        )

        # Vintage bounds
        self.minimum_vintage = self._fit_coords(
            f'{prefix}|minimum_vintage', self.minimum_vintage, dims=['vintage', 'scenario']
        )
        self.maximum_vintage = self._fit_coords(
            f'{prefix}|maximum_vintage', self.maximum_vintage, dims=['vintage', 'scenario']
        )

        # Aggregated bounds
        self.minimum_over_periods = self._fit_coords(
            f'{prefix}|minimum_over_periods', self.minimum_over_periods, dims=['scenario']
        )
        self.maximum_over_periods = self._fit_coords(
            f'{prefix}|maximum_over_periods', self.maximum_over_periods, dims=['scenario']
        )
        self.minimum_over_vintages = self._fit_coords(
            f'{prefix}|minimum_over_vintages', self.minimum_over_vintages, dims=['scenario']
        )
        self.maximum_over_vintages = self._fit_coords(
            f'{prefix}|maximum_over_vintages', self.maximum_over_vintages, dims=['scenario']
        )
        self.minimum_overall = self._fit_coords(f'{prefix}|minimum_overall', self.minimum_overall, dims=['scenario'])
        self.maximum_overall = self._fit_coords(f'{prefix}|maximum_overall', self.maximum_overall, dims=['scenario'])

        # Weights
        self.period_weights = self._fit_coords(
            f'{prefix}|period_weights', self.period_weights, dims=['period', 'scenario']
        )
        self.vintage_weights = self._fit_coords(
            f'{prefix}|vintage_weights', self.vintage_weights, dims=['vintage', 'scenario']
        )

    def create_model(self, model: FlowSystemModel) -> EffectModel:
        self._plausibility_checks()
        self.submodel = EffectModel(model, self)
        return self.submodel

    def _plausibility_checks(self) -> None:
        # Check that minimum_over_periods and maximum_over_periods require a period dimension
        if (
            self.minimum_over_periods is not None or self.maximum_over_periods is not None
        ) and self.flow_system.periods is None:
            raise PlausibilityError(
                f"Effect '{self.label}': minimum_over_periods and maximum_over_periods require "
                f"the FlowSystem to have a 'period' dimension. Please define periods when creating "
                f'the FlowSystem, or remove these constraints.'
            )

        # Check that minimum_over_vintages and maximum_over_vintages require a vintage dimension
        if (
            self.minimum_over_vintages is not None or self.maximum_over_vintages is not None
        ) and self.flow_system.vintages is None:
            raise PlausibilityError(
                f"Effect '{self.label}': minimum_over_vintages and maximum_over_vintages require "
                f"the FlowSystem to have a 'vintage' dimension. Please define periods when creating "
                f'the FlowSystem (vintages are derived from periods), or remove these constraints.'
            )

        # Check that minimum_overall and maximum_overall require both dimensions
        if (self.minimum_overall is not None or self.maximum_overall is not None) and self.flow_system.periods is None:
            raise PlausibilityError(
                f"Effect '{self.label}': minimum_overall and maximum_overall require "
                f"the FlowSystem to have a 'period' dimension. Please define periods when creating "
                f'the FlowSystem, or remove these constraints.'
            )


class EffectModel(ElementModel):
    """Mathematical model implementation for Effects.

    Creates optimization variables and constraints for effect aggregation,
    including temporal, periodic, and vintage tracking, cross-effect contributions,
    and effect bounds.

    The three effect categories have different dimensions and are only combined
    in the objective function after proper weighting:
    - temporal: (time, period, scenario) - operational costs over time
    - periodic: (period, scenario) - recurring costs per period
    - vintage: (vintage, scenario) - one-time investment costs

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/effects-and-dimensions/>
    """

    element: Effect  # Type hint

    def __init__(self, model: FlowSystemModel, element: Effect):
        super().__init__(model, element)

    @property
    def period_weights(self) -> xr.DataArray:
        """
        Get period weights for this effect.

        Returns effect-specific weights if defined, otherwise falls back to FlowSystem period weights.
        This allows different effects to have different weighting schemes over periods (e.g., discounting for costs,
        equal weights for CO2 emissions).

        Returns:
            Weights with period dimensions (if applicable)
        """
        effect_weights = self.element.period_weights
        default_weights = self.element._flow_system.period_weights
        if effect_weights is not None:  # Use effect-specific weights
            return effect_weights
        elif default_weights is not None:  # Fall back to FlowSystem weights
            return default_weights
        return self.element._fit_coords(name='period_weights', data=1, dims=['period'])

    @property
    def vintage_weights(self) -> xr.DataArray:
        """
        Get vintage weights for this effect.

        Returns effect-specific weights if defined, otherwise falls back to FlowSystem vintage weights.
        Default vintage weights are 1 (one-time costs not scaled by period duration).

        Returns:
            Weights with vintage dimensions (if applicable)
        """
        effect_weights = self.element.vintage_weights
        default_weights = self.element._flow_system.vintage_weights
        if effect_weights is not None:  # Use effect-specific weights
            return effect_weights
        elif default_weights is not None:  # Fall back to FlowSystem weights
            return default_weights
        return self.element._fit_coords(name='vintage_weights', data=1, dims=['vintage'])

    def _do_modeling(self):
        """Create variables, constraints, and nested submodels"""
        super()._do_modeling()

        # Temporal effects: operational costs over time
        self.temporal: ShareAllocationModel = self.add_submodels(
            ShareAllocationModel(
                model=self._model,
                dims=('time', 'period', 'scenario'),
                label_of_element=self.label_of_element,
                label_of_model=f'{self.label_of_model}(temporal)',
                total_max=self.element.maximum_temporal,
                total_min=self.element.minimum_temporal,
                min_per_hour=self.element.minimum_per_hour if self.element.minimum_per_hour is not None else None,
                max_per_hour=self.element.maximum_per_hour if self.element.maximum_per_hour is not None else None,
            ),
            short_name='temporal',
        )

        # Periodic effects: recurring costs per period
        self.periodic: ShareAllocationModel = self.add_submodels(
            ShareAllocationModel(
                model=self._model,
                dims=('period', 'scenario'),
                label_of_element=self.label_of_element,
                label_of_model=f'{self.label_of_model}(periodic)',
                total_max=self.element.maximum_periodic,
                total_min=self.element.minimum_periodic,
            ),
            short_name='periodic',
        )

        # Vintage effects: one-time investment costs
        # Only create if FlowSystem has vintages defined
        self.vintage: ShareAllocationModel | None = None
        if self._model.flow_system.vintages is not None:
            self.vintage = self.add_submodels(
                ShareAllocationModel(
                    model=self._model,
                    dims=('vintage', 'scenario'),
                    label_of_element=self.label_of_element,
                    label_of_model=f'{self.label_of_model}(vintage)',
                    total_max=self.element.maximum_vintage,
                    total_min=self.element.minimum_vintage,
                ),
                short_name='vintage',
            )

        # NOTE: Vintage has a different dimension than temporal/periodic, so we cannot have a
        # unified 'total' that includes all three. However, we keep total = temporal + periodic
        # for backwards compatibility and per-period constraints.

        # Add total variable and constraint for temporal + periodic (same period dimension)
        self._add_total_constraint()

        # Add weighted sum over all periods constraint for temporal + periodic
        self._add_over_periods_constraints()

        # Add weighted sum over all vintages constraint
        self._add_over_vintages_constraints()

        # Add overall constraint (over_periods + over_vintages)
        self._add_overall_constraints()

    def _add_total_constraint(self):
        """Add total = temporal + periodic constraint (per period, does not include vintage)."""
        # Only create if bounds are specified or for backwards compatibility tracking
        # For now, always create to maintain backwards compatibility
        lower = self.element.minimum_total if self.element.minimum_total is not None else -np.inf
        upper = self.element.maximum_total if self.element.maximum_total is not None else np.inf

        self.total = self.add_variables(
            lower=lower,
            upper=upper,
            coords=self._model.get_coords(['period', 'scenario']),
            name=self.label_full,
        )

        self.add_constraints(
            self.total == self.temporal.total + self.periodic.total,
            name=self.label_full,
            short_name='total',
        )

    def _add_over_periods_constraints(self):
        """Add constraints for weighted sum of temporal + periodic over all periods."""
        if self.element.minimum_over_periods is not None or self.element.maximum_over_periods is not None:
            # Sum temporal and periodic (they share the same period dimension)
            period_total = self.temporal.total + self.periodic.total

            # Calculate weighted sum over all periods
            weighted_total = (period_total * self.period_weights).sum('period')

            # Create tracking variable for the weighted sum
            self.total_over_periods = self.add_variables(
                lower=self.element.minimum_over_periods if self.element.minimum_over_periods is not None else -np.inf,
                upper=self.element.maximum_over_periods if self.element.maximum_over_periods is not None else np.inf,
                coords=self._model.get_coords(['scenario']),
                short_name='total_over_periods',
            )

            self.add_constraints(self.total_over_periods == weighted_total, short_name='total_over_periods')

    def _add_over_vintages_constraints(self):
        """Add constraints for weighted sum of vintage over all vintages."""
        if self.vintage is None:
            return

        if self.element.minimum_over_vintages is not None or self.element.maximum_over_vintages is not None:
            # Calculate weighted sum over all vintages
            weighted_total = (self.vintage.total * self.vintage_weights).sum('vintage')

            # Create tracking variable for the weighted sum
            self.total_over_vintages = self.add_variables(
                lower=self.element.minimum_over_vintages if self.element.minimum_over_vintages is not None else -np.inf,
                upper=self.element.maximum_over_vintages if self.element.maximum_over_vintages is not None else np.inf,
                coords=self._model.get_coords(['scenario']),
                short_name='total_over_vintages',
            )

            self.add_constraints(self.total_over_vintages == weighted_total, short_name='total_over_vintages')

    def _add_overall_constraints(self):
        """Add constraints for overall sum (over_periods + over_vintages)."""
        if self.element.minimum_overall is None and self.element.maximum_overall is None:
            return

        # Calculate period contribution (temporal + periodic weighted sum)
        period_total = self.temporal.total + self.periodic.total
        weighted_periods = (period_total * self.period_weights).sum('period')

        # Calculate vintage contribution (if exists)
        if self.vintage is not None:
            weighted_vintages = (self.vintage.total * self.vintage_weights).sum('vintage')
            overall_total = weighted_periods + weighted_vintages
        else:
            overall_total = weighted_periods

        # Create tracking variable for overall
        self.total_overall = self.add_variables(
            lower=self.element.minimum_overall if self.element.minimum_overall is not None else -np.inf,
            upper=self.element.maximum_overall if self.element.maximum_overall is not None else np.inf,
            coords=self._model.get_coords(['scenario']),
            short_name='total_overall',
        )

        self.add_constraints(self.total_overall == overall_total, short_name='total_overall')


EffectExpr = dict[str, linopy.LinearExpression]  # Used to create Shares


class EffectCollection(ElementContainer[Effect]):
    """
    Handling all Effects
    """

    submodel: EffectCollectionModel | None

    def __init__(self, *effects: Effect, truncate_repr: int | None = None):
        """
        Initialize the EffectCollection.

        Args:
            *effects: Effects to register in the collection.
            truncate_repr: Maximum number of items to show in repr. If None, show all items. Default: None
        """
        super().__init__(element_type_name='effects', truncate_repr=truncate_repr)
        self._standard_effect: Effect | None = None
        self._objective_effect: Effect | None = None
        self._penalty_effect: Effect | None = None

        self.submodel = None
        self.add_effects(*effects)

    def create_model(self, model: FlowSystemModel) -> EffectCollectionModel:
        self._plausibility_checks()
        self.submodel = EffectCollectionModel(model, self)
        return self.submodel

    def _create_penalty_effect(self) -> Effect:
        """
        Create and register the penalty effect (called internally by FlowSystem).
        Only creates if user hasn't already defined a Penalty effect.
        """
        # Check if user has already defined a Penalty effect
        if PENALTY_EFFECT_LABEL in self:
            self._penalty_effect = self[PENALTY_EFFECT_LABEL]
            logger.info(f'Using user-defined Penalty Effect: {PENALTY_EFFECT_LABEL}')
            return self._penalty_effect

        # Auto-create penalty effect
        self._penalty_effect = Effect(
            label=PENALTY_EFFECT_LABEL,
            unit='penalty_units',
            description='Penalty for constraint violations and modeling artifacts',
            is_standard=False,
            is_objective=False,
        )
        self.add(self._penalty_effect)  # Add to container
        logger.info(f'Auto-created Penalty Effect: {PENALTY_EFFECT_LABEL}')
        return self._penalty_effect

    def add_effects(self, *effects: Effect) -> None:
        for effect in list(effects):
            if effect in self:
                raise ValueError(f'Effect with label "{effect.label=}" already added!')
            if effect.is_standard:
                self.standard_effect = effect
            if effect.is_objective:
                self.objective_effect = effect
            self.add(effect)  # Use the inherited add() method from ElementContainer
            logger.info(f'Registered new Effect: {effect.label}')

    def create_effect_values_dict(self, effect_values_user: Numeric_TPS | Effect_TPS | None) -> Effect_TPS | None:
        """Converts effect values into a dictionary. If a scalar is provided, it is associated with a default effect type.

        Examples:
            ```python
            effect_values_user = 20                               -> {'<standard_effect_label>': 20}
            effect_values_user = {None: 20}                       -> {'<standard_effect_label>': 20}
            effect_values_user = None                             -> None
            effect_values_user = {'effect1': 20, 'effect2': 0.3}  -> {'effect1': 20, 'effect2': 0.3}
            ```

        Returns:
            A dictionary keyed by effect label, or None if input is None.
            Note: a standard effect must be defined when passing scalars or None labels.
        """

        def get_effect_label(eff: str | None) -> str:
            """Get the label of an effect"""
            if eff is None:
                return self.standard_effect.label
            if isinstance(eff, Effect):
                raise TypeError(
                    f'Effect objects are no longer accepted when specifying EffectValues. '
                    f'Use the label string instead. Got: {eff.label_full}'
                )
            return eff

        if effect_values_user is None:
            return None
        if isinstance(effect_values_user, dict):
            return {get_effect_label(effect): value for effect, value in effect_values_user.items()}
        return {self.standard_effect.label: effect_values_user}

    def _plausibility_checks(self) -> None:
        # Check circular loops in effects:
        temporal, periodic = self.calculate_effect_share_factors()

        # Validate all referenced effects (both sources and targets) exist
        edges = list(temporal.keys()) + list(periodic.keys())
        unknown_sources = {src for src, _ in edges if src not in self}
        unknown_targets = {tgt for _, tgt in edges if tgt not in self}
        unknown = unknown_sources | unknown_targets
        if unknown:
            raise KeyError(f'Unknown effects used in effect share mappings: {sorted(unknown)}')

        temporal_cycles = detect_cycles(tuples_to_adjacency_list([key for key in temporal]))
        periodic_cycles = detect_cycles(tuples_to_adjacency_list([key for key in periodic]))

        if temporal_cycles:
            cycle_str = '\n'.join([' -> '.join(cycle) for cycle in temporal_cycles])
            raise ValueError(f'Error: circular temporal-shares detected:\n{cycle_str}')

        if periodic_cycles:
            cycle_str = '\n'.join([' -> '.join(cycle) for cycle in periodic_cycles])
            raise ValueError(f'Error: circular periodic-shares detected:\n{cycle_str}')

    def __getitem__(self, effect: str | Effect | None) -> Effect:
        """
        Get an effect by label, or return the standard effect if None is passed

        Raises:
            KeyError: If no effect with the given label is found.
            KeyError: If no standard effect is specified.
        """
        if effect is None:
            return self.standard_effect
        if isinstance(effect, Effect):
            if effect in self:
                return effect
            else:
                raise KeyError(f'Effect {effect} not found!')
        try:
            return super().__getitem__(effect)  # Leverage ContainerMixin suggestions
        except KeyError as e:
            # Extract the original message and append context for cleaner output
            original_msg = str(e).strip('\'"')
            raise KeyError(f'{original_msg} Add the effect to the FlowSystem first.') from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())  # Iterate over keys like a normal dict

    def __contains__(self, item: str | Effect) -> bool:
        """Check if the effect exists. Checks for label or object"""
        if isinstance(item, str):
            return super().__contains__(item)  # Check if the label exists
        elif isinstance(item, Effect):
            return item.label_full in self and self[item.label_full] is item
        return False

    @property
    def standard_effect(self) -> Effect:
        if self._standard_effect is None:
            raise KeyError(
                'No standard-effect specified! Either set an effect through is_standard=True '
                'or pass a mapping when specifying effect values: {effect_label: value}.'
            )
        return self._standard_effect

    @standard_effect.setter
    def standard_effect(self, value: Effect) -> None:
        if self._standard_effect is not None:
            raise ValueError(f'A standard-effect already exists! ({self._standard_effect.label=})')
        self._standard_effect = value

    @property
    def objective_effect(self) -> Effect:
        if self._objective_effect is None:
            raise KeyError('No objective-effect specified!')
        return self._objective_effect

    @objective_effect.setter
    def objective_effect(self, value: Effect) -> None:
        # Check Penalty first to give users a more specific error message
        if value.label == PENALTY_EFFECT_LABEL:
            raise ValueError(
                f'The Penalty effect ("{PENALTY_EFFECT_LABEL}") cannot be set as the objective effect. '
                f'Please use a different effect as the optimization objective.'
            )
        if self._objective_effect is not None:
            raise ValueError(f'An objective-effect already exists! ({self._objective_effect.label=})')
        self._objective_effect = value

    @property
    def penalty_effect(self) -> Effect:
        """
        The penalty effect (auto-created during modeling if not user-defined).

        Returns the Penalty effect whether user-defined or auto-created.
        """
        # If already set, return it
        if self._penalty_effect is not None:
            return self._penalty_effect

        # Check if user has defined a Penalty effect
        if PENALTY_EFFECT_LABEL in self:
            self._penalty_effect = self[PENALTY_EFFECT_LABEL]
            return self._penalty_effect

        # Not yet created - will be created during modeling
        raise KeyError(
            f'Penalty effect not yet created. It will be auto-created during modeling, '
            f'or you can define your own using: Effect("{PENALTY_EFFECT_LABEL}", ...)'
        )

    def calculate_effect_share_factors(
        self,
    ) -> tuple[
        dict[tuple[str, str], xr.DataArray],
        dict[tuple[str, str], xr.DataArray],
    ]:
        shares_periodic = {}
        for name, effect in self.items():
            if effect.share_from_periodic:
                for source, data in effect.share_from_periodic.items():
                    if source not in shares_periodic:
                        shares_periodic[source] = {}
                    shares_periodic[source][name] = data
        shares_periodic = calculate_all_conversion_paths(shares_periodic)

        shares_temporal = {}
        for name, effect in self.items():
            if effect.share_from_temporal:
                for source, data in effect.share_from_temporal.items():
                    if source not in shares_temporal:
                        shares_temporal[source] = {}
                    shares_temporal[source][name] = data
        shares_temporal = calculate_all_conversion_paths(shares_temporal)

        return shares_temporal, shares_periodic


class EffectCollectionModel(Submodel):
    """
    Handling all Effects
    """

    def __init__(self, model: FlowSystemModel, effects: EffectCollection):
        self.effects = effects
        super().__init__(model, label_of_element='Effects')

    def add_share_to_effects(
        self,
        name: str,
        expressions: EffectExpr,
        target: Literal['temporal', 'periodic', 'vintage'],
    ) -> None:
        for effect, expression in expressions.items():
            if target == 'temporal':
                self.effects[effect].submodel.temporal.add_share(
                    name,
                    expression,
                    dims=('time', 'period', 'scenario'),
                )
            elif target == 'periodic':
                self.effects[effect].submodel.periodic.add_share(
                    name,
                    expression,
                    dims=('period', 'scenario'),
                )
            elif target == 'vintage':
                if self.effects[effect].submodel.vintage is None:
                    raise ValueError(
                        f"Cannot add vintage share to effect '{effect}': "
                        f'FlowSystem has no vintage dimension (periods not defined).'
                    )
                self.effects[effect].submodel.vintage.add_share(
                    name,
                    expression,
                    dims=('vintage', 'scenario'),
                )
            else:
                raise ValueError(f'Target {target} not supported! Use "temporal", "periodic", or "vintage".')

    def _do_modeling(self):
        """Create variables, constraints, and nested submodels"""
        super()._do_modeling()

        # Ensure penalty effect exists (auto-create if user hasn't defined one)
        if self.effects._penalty_effect is None:
            penalty_effect = self.effects._create_penalty_effect()
            # Link to FlowSystem (should already be linked, but ensure it)
            if penalty_effect._flow_system is None:
                penalty_effect._set_flow_system(self._model.flow_system)

        # Create EffectModel for each effect
        for effect in self.effects.values():
            effect.create_model(self._model)

        # Add cross-effect shares
        self._add_share_between_effects()

        # Build objective function with separate weighting for each effect category
        self._add_objective()

    def _add_objective(self):
        """Build objective function combining temporal, periodic, and vintage effects."""
        obj_effect = self.effects.objective_effect.submodel
        pen_effect = self.effects.penalty_effect.submodel

        # Period-weighted components: (temporal + periodic) × period_weights × scenario_weights
        period_weights = obj_effect.period_weights
        scenario_weights = self._model.scenario_weights

        obj_temporal_weighted = (obj_effect.temporal.total * period_weights * scenario_weights).sum()
        obj_periodic_weighted = (obj_effect.periodic.total * period_weights * scenario_weights).sum()

        pen_temporal_weighted = (pen_effect.temporal.total * period_weights * scenario_weights).sum()
        pen_periodic_weighted = (pen_effect.periodic.total * period_weights * scenario_weights).sum()

        objective = obj_temporal_weighted + obj_periodic_weighted + pen_temporal_weighted + pen_periodic_weighted

        # Vintage-weighted component: vintage × vintage_weights × scenario_weights
        # Only if vintage dimension exists
        if obj_effect.vintage is not None:
            vintage_weights = obj_effect.vintage_weights
            obj_vintage_weighted = (obj_effect.vintage.total * vintage_weights * scenario_weights).sum()
            pen_vintage_weighted = (pen_effect.vintage.total * vintage_weights * scenario_weights).sum()
            objective = objective + obj_vintage_weighted + pen_vintage_weighted

        self._model.add_objective(objective)

    def _add_share_between_effects(self):
        for target_effect in self.effects.values():
            # 1. temporal: <- receiving temporal shares from other effects
            for source_effect, time_series in target_effect.share_from_temporal.items():
                target_effect.submodel.temporal.add_share(
                    self.effects[source_effect].submodel.temporal.label_full,
                    self.effects[source_effect].submodel.temporal.total_per_timestep * time_series,
                    dims=('time', 'period', 'scenario'),
                )
            # 2. periodic: <- receiving periodic shares from other effects
            for source_effect, factor in target_effect.share_from_periodic.items():
                target_effect.submodel.periodic.add_share(
                    self.effects[source_effect].submodel.periodic.label_full,
                    self.effects[source_effect].submodel.periodic.total * factor,
                    dims=('period', 'scenario'),
                )
            # 3. vintage: <- receiving vintage shares from other effects
            for source_effect, factor in target_effect.share_from_vintage.items():
                source_vintage = self.effects[source_effect].submodel.vintage
                target_vintage = target_effect.submodel.vintage
                if source_vintage is None or target_vintage is None:
                    raise ValueError(
                        f"Cannot add vintage cross-effect share from '{source_effect}' to '{target_effect.label}': "
                        f'FlowSystem has no vintage dimension (periods not defined).'
                    )
                target_vintage.add_share(
                    source_vintage.label_full,
                    source_vintage.total * factor,
                    dims=('vintage', 'scenario'),
                )


def calculate_all_conversion_paths(
    conversion_dict: dict[str, dict[str, Scalar | xr.DataArray]],
) -> dict[tuple[str, str], xr.DataArray]:
    """
    Calculates all possible direct and indirect conversion factors between units/domains.
    This function uses Breadth-First Search (BFS) to find all possible conversion paths
    between different units or domains in a conversion graph. It computes both direct
    conversions (explicitly provided in the input) and indirect conversions (derived
    through intermediate units).
    Args:
        conversion_dict: A nested dictionary where:
            - Outer keys represent origin units/domains
            - Inner dictionaries map target units/domains to their conversion factors
            - Conversion factors can be integers, floats, or numpy arrays
    Returns:
        A dictionary mapping (origin, target) tuples to their respective conversion factors.
        Each key is a tuple of strings representing the origin and target units/domains.
        Each value is the conversion factor (int, float, or numpy array) from origin to target.
    """
    # Initialize the result dictionary to accumulate all paths
    result = {}

    # Add direct connections to the result first
    for origin, targets in conversion_dict.items():
        for target, factor in targets.items():
            result[(origin, target)] = factor

    # Track all paths by keeping path history to avoid cycles
    # Iterate over each domain in the dictionary
    for origin in conversion_dict:
        # Keep track of visited paths to avoid repeating calculations
        processed_paths = set()
        # Use a queue with (current_domain, factor, path_history)
        queue = deque([(origin, 1, [origin])])

        while queue:
            current_domain, factor, path = queue.popleft()

            # Skip if we've processed this exact path before
            path_key = tuple(path)
            if path_key in processed_paths:
                continue
            processed_paths.add(path_key)

            # Iterate over the neighbors of the current domain
            for target, conversion_factor in conversion_dict.get(current_domain, {}).items():
                # Skip if target would create a cycle
                if target in path:
                    continue

                # Calculate the indirect conversion factor
                indirect_factor = factor * conversion_factor
                new_path = path + [target]

                # Only consider paths starting at origin and ending at some target
                if len(new_path) > 2 and new_path[0] == origin:
                    # Update the result dictionary - accumulate factors from different paths
                    if (origin, target) in result:
                        result[(origin, target)] = result[(origin, target)] + indirect_factor
                    else:
                        result[(origin, target)] = indirect_factor

                # Add new path to queue for further exploration
                queue.append((target, indirect_factor, new_path))

    # Convert all values to DataArrays
    result = {key: value if isinstance(value, xr.DataArray) else xr.DataArray(value) for key, value in result.items()}

    return result


def detect_cycles(graph: dict[str, list[str]]) -> list[list[str]]:
    """
    Detects cycles in a directed graph using DFS.

    Args:
        graph: Adjacency list representation of the graph

    Returns:
        List of cycles found, where each cycle is a list of nodes
    """
    # Track nodes in current recursion stack
    visiting = set()
    # Track nodes that have been fully explored
    visited = set()
    # Store all found cycles
    cycles = []

    def dfs_find_cycles(node, path=None):
        if path is None:
            path = []

        # Current path to this node
        current_path = path + [node]
        # Add node to current recursion stack
        visiting.add(node)

        # Check all neighbors
        for neighbor in graph.get(node, []):
            # If neighbor is in current path, we found a cycle
            if neighbor in visiting:
                # Get the cycle by extracting the relevant portion of the path
                cycle_start = current_path.index(neighbor)
                cycle = current_path[cycle_start:] + [neighbor]
                cycles.append(cycle)
            # If neighbor hasn't been fully explored, check it
            elif neighbor not in visited:
                dfs_find_cycles(neighbor, current_path)

        # Remove node from current path and mark as fully explored
        visiting.remove(node)
        visited.add(node)

    # Check each unvisited node
    for node in graph:
        if node not in visited:
            dfs_find_cycles(node)

    return cycles


def tuples_to_adjacency_list(edges: list[tuple[str, str]]) -> dict[str, list[str]]:
    """
    Converts a list of edge tuples (source, target) to an adjacency list representation.

    Args:
        edges: List of (source, target) tuples representing directed edges

    Returns:
        Dictionary mapping each source node to a list of its target nodes
    """
    graph = {}

    for source, target in edges:
        if source not in graph:
            graph[source] = []
        graph[source].append(target)

        # Ensure target nodes with no outgoing edges are in the graph
        if target not in graph:
            graph[target] = []

    return graph
