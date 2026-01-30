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
from .structure import (
    Element,
    ElementContainer,
    FlowSystemModel,
    register_class_for_io,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .types import Effect_PS, Effect_TPS, Numeric_PS, Numeric_S, Numeric_TPS, Scalar

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
        share_from_temporal: Temporal cross-effect contributions.
            Maps temporal contributions from other effects to this effect.
        share_from_periodic: Periodic cross-effect contributions.
            Maps periodic contributions from other effects to this effect.
        minimum_temporal: Minimum allowed total contribution across all timesteps (per period).
        maximum_temporal: Maximum allowed total contribution across all timesteps (per period).
        minimum_per_hour: Minimum allowed contribution per hour.
        maximum_per_hour: Maximum allowed contribution per hour.
        minimum_periodic: Minimum allowed total periodic contribution (per period).
        maximum_periodic: Maximum allowed total periodic contribution (per period).
        minimum_total: Minimum allowed total effect (temporal + periodic combined) per period.
        maximum_total: Maximum allowed total effect (temporal + periodic combined) per period.
        minimum_over_periods: Minimum allowed weighted sum of total effect across ALL periods.
            Weighted by effect-specific weights if defined, otherwise by FlowSystem period weights.
            Requires FlowSystem to have a 'period' dimension (i.e., periods must be defined).
        maximum_over_periods: Maximum allowed weighted sum of total effect across ALL periods.
            Weighted by effect-specific weights if defined, otherwise by FlowSystem period weights.
            Requires FlowSystem to have a 'period' dimension (i.e., periods must be defined).
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

    def __init__(
        self,
        label: str,
        unit: str,
        description: str = '',
        meta_data: dict | None = None,
        is_standard: bool = False,
        is_objective: bool = False,
        period_weights: Numeric_PS | None = None,
        share_from_temporal: Effect_TPS | Numeric_TPS | None = None,
        share_from_periodic: Effect_PS | Numeric_PS | None = None,
        minimum_temporal: Numeric_PS | None = None,
        maximum_temporal: Numeric_PS | None = None,
        minimum_periodic: Numeric_PS | None = None,
        maximum_periodic: Numeric_PS | None = None,
        minimum_per_hour: Numeric_TPS | None = None,
        maximum_per_hour: Numeric_TPS | None = None,
        minimum_total: Numeric_PS | None = None,
        maximum_total: Numeric_PS | None = None,
        minimum_over_periods: Numeric_S | None = None,
        maximum_over_periods: Numeric_S | None = None,
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
        # Share parameters accept Effect_* | Numeric_* unions (dict or single value).
        # Store as-is here; transform_data() will normalize via fit_effects_to_model_coords().
        # Default to {} when None (no shares defined).
        self.share_from_temporal = share_from_temporal if share_from_temporal is not None else {}
        self.share_from_periodic = share_from_periodic if share_from_periodic is not None else {}

        # Set attributes directly
        self.minimum_temporal = minimum_temporal
        self.maximum_temporal = maximum_temporal
        self.minimum_periodic = minimum_periodic
        self.maximum_periodic = maximum_periodic
        self.minimum_per_hour = minimum_per_hour
        self.maximum_per_hour = maximum_per_hour
        self.minimum_total = minimum_total
        self.maximum_total = maximum_total
        self.minimum_over_periods = minimum_over_periods
        self.maximum_over_periods = maximum_over_periods

    def link_to_flow_system(self, flow_system, prefix: str = '') -> None:
        """Link this effect to a FlowSystem.

        Elements use their label_full as prefix by default, ignoring the passed prefix.
        """
        super().link_to_flow_system(flow_system, self.label_full)

    def transform_data(self) -> None:
        self.minimum_per_hour = self._fit_coords(f'{self.prefix}|minimum_per_hour', self.minimum_per_hour)
        self.maximum_per_hour = self._fit_coords(f'{self.prefix}|maximum_per_hour', self.maximum_per_hour)

        self.share_from_temporal = self._fit_effect_coords(
            prefix=None,
            effect_values=self.share_from_temporal,
            suffix=f'(temporal)->{self.prefix}(temporal)',
        )
        self.share_from_periodic = self._fit_effect_coords(
            prefix=None,
            effect_values=self.share_from_periodic,
            suffix=f'(periodic)->{self.prefix}(periodic)',
            dims=['period', 'scenario'],
        )

        self.minimum_temporal = self._fit_coords(
            f'{self.prefix}|minimum_temporal', self.minimum_temporal, dims=['period', 'scenario']
        )
        self.maximum_temporal = self._fit_coords(
            f'{self.prefix}|maximum_temporal', self.maximum_temporal, dims=['period', 'scenario']
        )
        self.minimum_periodic = self._fit_coords(
            f'{self.prefix}|minimum_periodic', self.minimum_periodic, dims=['period', 'scenario']
        )
        self.maximum_periodic = self._fit_coords(
            f'{self.prefix}|maximum_periodic', self.maximum_periodic, dims=['period', 'scenario']
        )
        self.minimum_total = self._fit_coords(
            f'{self.prefix}|minimum_total', self.minimum_total, dims=['period', 'scenario']
        )
        self.maximum_total = self._fit_coords(
            f'{self.prefix}|maximum_total', self.maximum_total, dims=['period', 'scenario']
        )
        self.minimum_over_periods = self._fit_coords(
            f'{self.prefix}|minimum_over_periods', self.minimum_over_periods, dims=['scenario']
        )
        self.maximum_over_periods = self._fit_coords(
            f'{self.prefix}|maximum_over_periods', self.maximum_over_periods, dims=['scenario']
        )
        self.period_weights = self._fit_coords(
            f'{self.prefix}|period_weights', self.period_weights, dims=['period', 'scenario']
        )

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


class EffectsModel:
    """Type-level model for ALL effects with batched variables using 'effect' dimension.

    Unlike EffectModel (one per Effect), EffectsModel handles ALL effects in a single
    instance with batched variables. This provides:
    - Compact model structure with 'effect' dimension
    - Vectorized constraint creation
    - Direct expression building for effect shares

    Variables created (all with 'effect' dimension):
        - effect|periodic: Periodic (investment) contributions per effect
        - effect|temporal: Temporal (operation) total per effect
        - effect|per_timestep: Per-timestep contributions per effect
        - effect|total: Total effect (periodic + temporal)

    Usage:
        1. Call create_variables() to create effect variables
        2. Call finalize_shares() to add share expressions to effect constraints
    """

    def __init__(self, model: FlowSystemModel, effects: list[Effect]):
        import pandas as pd

        self.model = model
        self.effects = effects
        self.effect_ids = [e.label for e in effects]
        self._effect_index = pd.Index(self.effect_ids, name='effect')

        # Variables (set during create_variables)
        self.periodic: linopy.Variable | None = None
        self.temporal: linopy.Variable | None = None
        self.per_timestep: linopy.Variable | None = None
        self.total: linopy.Variable | None = None
        self.total_over_periods: linopy.Variable | None = None

        # Constraints for effect tracking (created in create_variables and finalize_shares)
        self._eq_periodic: linopy.Constraint | None = None
        self._eq_temporal: linopy.Constraint | None = None
        self._eq_total: linopy.Constraint | None = None

        self._eq_per_timestep: linopy.Constraint | None = None

        # Share variables (created in create_share_variables)
        self.share_temporal: linopy.Variable | None = None
        self.share_periodic: linopy.Variable | None = None

        # Registered contributions from type models (FlowsModel, StoragesModel, etc.)
        # Each entry: (contributor_ids, defining_expr with 'contributor' dim)
        self._temporal_share_defs: list[tuple[list[str], linopy.LinearExpression]] = []
        self._periodic_share_defs: list[tuple[list[str], linopy.LinearExpression]] = []
        # Extra contributions that don't go through share variables (status effects, invested, constants)
        self._temporal_contributions: list = []
        self._periodic_contributions: list = []

    @property
    def effect_index(self):
        """Public access to the effect index for type models."""
        return self._effect_index

    def register_temporal_share(self, contributor_ids: list[str], defining_expr) -> None:
        """Register contributors for the share|temporal variable.

        The defining_expr must have a 'contributor' dimension matching contributor_ids.
        """
        self._temporal_share_defs.append((contributor_ids, defining_expr))

    def register_periodic_share(self, contributor_ids: list[str], defining_expr) -> None:
        """Register contributors for the share|periodic variable.

        The defining_expr must have a 'contributor' dimension matching contributor_ids.
        """
        self._periodic_share_defs.append((contributor_ids, defining_expr))

    def add_temporal_contribution(self, expr) -> None:
        """Register a temporal effect contribution expression (not via share variable).

        For contributions like status effects that bypass the share variable.
        """
        self._temporal_contributions.append(expr)

    def add_periodic_contribution(self, expr) -> None:
        """Register a periodic effect contribution expression (not via share variable).

        For contributions like invested-based effects that bypass the share variable.
        """
        self._periodic_contributions.append(expr)

    def _stack_bounds(self, attr_name: str, default: float = np.inf) -> xr.DataArray:
        """Stack per-effect bounds into a single DataArray with effect dimension."""

        def as_dataarray(effect: Effect) -> xr.DataArray:
            val = getattr(effect, attr_name, None)
            if val is None:
                return xr.DataArray(default)
            return val if isinstance(val, xr.DataArray) else xr.DataArray(val)

        return xr.concat(
            [as_dataarray(e).expand_dims(effect=[e.label]) for e in self.effects],
            dim='effect',
            fill_value=default,
        )

    def _get_period_weights(self, effect: Effect) -> xr.DataArray:
        """Get period weights for an effect."""
        effect_weights = effect.period_weights
        default_weights = effect._flow_system.period_weights
        if effect_weights is not None:
            return effect_weights
        elif default_weights is not None:
            return default_weights
        return effect._fit_coords(name='period_weights', data=1, dims=['period'])

    def create_variables(self) -> None:
        """Create batched effect variables with 'effect' dimension."""

        # Helper to safely merge coordinates
        def _merge_coords(base_dict: dict, model_coords) -> dict:
            if model_coords is not None:
                base_dict.update({k: v for k, v in model_coords.items()})
            return base_dict

        # === Periodic (investment) ===
        periodic_coords = xr.Coordinates(
            _merge_coords(
                {'effect': self._effect_index},
                self.model.get_coords(['period', 'scenario']),
            )
        )
        self.periodic = self.model.add_variables(
            lower=self._stack_bounds('minimum_periodic', -np.inf),
            upper=self._stack_bounds('maximum_periodic', np.inf),
            coords=periodic_coords,
            name='effect|periodic',
        )
        # Constraint: periodic == sum(shares) - start with 0, shares subtract from LHS
        self._eq_periodic = self.model.add_constraints(
            self.periodic == 0,
            name='effect|periodic',
        )

        # === Temporal (operation total over time) ===
        self.temporal = self.model.add_variables(
            lower=self._stack_bounds('minimum_temporal', -np.inf),
            upper=self._stack_bounds('maximum_temporal', np.inf),
            coords=periodic_coords,
            name='effect|temporal',
        )
        self._eq_temporal = self.model.add_constraints(
            self.temporal == 0,
            name='effect|temporal',
        )

        # === Per-timestep (temporal contributions per timestep) ===
        temporal_coords = xr.Coordinates(
            _merge_coords(
                {'effect': self._effect_index},
                self.model.get_coords(None),  # All dims
            )
        )

        # Build per-hour bounds
        min_per_hour = self._stack_bounds('minimum_per_hour', -np.inf)
        max_per_hour = self._stack_bounds('maximum_per_hour', np.inf)

        self.per_timestep = self.model.add_variables(
            lower=min_per_hour * self.model.timestep_duration if min_per_hour is not None else -np.inf,
            upper=max_per_hour * self.model.timestep_duration if max_per_hour is not None else np.inf,
            coords=temporal_coords,
            name='effect|per_timestep',
        )
        self._eq_per_timestep = self.model.add_constraints(
            self.per_timestep == 0,
            name='effect|per_timestep',
        )

        # Link per_timestep to temporal (sum over time)
        weighted_per_timestep = self.per_timestep * self.model.weights.get('cluster', 1.0)
        self._eq_temporal.lhs -= weighted_per_timestep.sum(dim=self.model.temporal_dims)

        # === Total (periodic + temporal) ===
        self.total = self.model.add_variables(
            lower=self._stack_bounds('minimum_total', -np.inf),
            upper=self._stack_bounds('maximum_total', np.inf),
            coords=periodic_coords,
            name='effect|total',
        )
        self._eq_total = self.model.add_constraints(
            self.total == self.periodic + self.temporal,
            name='effect|total',
        )

        # === Total over periods (for effects with min/max_over_periods) ===
        # Only applicable when periods exist in the flow system
        if self.model.flow_system.periods is None:
            return
        effects_with_over_periods = [
            e for e in self.effects if e.minimum_over_periods is not None or e.maximum_over_periods is not None
        ]
        if effects_with_over_periods:
            over_periods_ids = [e.label for e in effects_with_over_periods]
            over_periods_coords = xr.Coordinates(
                _merge_coords(
                    {'effect': over_periods_ids},
                    self.model.get_coords(['scenario']),
                )
            )

            # Stack bounds for over_periods
            lower_over = []
            upper_over = []
            for e in effects_with_over_periods:
                lower_over.append(e.minimum_over_periods if e.minimum_over_periods is not None else -np.inf)
                upper_over.append(e.maximum_over_periods if e.maximum_over_periods is not None else np.inf)

            self.total_over_periods = self.model.add_variables(
                lower=xr.DataArray(lower_over, coords={'effect': over_periods_ids}, dims=['effect']),
                upper=xr.DataArray(upper_over, coords={'effect': over_periods_ids}, dims=['effect']),
                coords=over_periods_coords,
                name='effect|total_over_periods',
            )

            # Create constraint: total_over_periods == weighted sum for each effect
            # Can't use xr.concat with LinearExpression objects, so create individual constraints
            for e in effects_with_over_periods:
                total_e = self.total.sel(effect=e.label)
                weights_e = self._get_period_weights(e)
                weighted_total = (total_e * weights_e).sum('period')
                self.model.add_constraints(
                    self.total_over_periods.sel(effect=e.label) == weighted_total,
                    name=f'effect|total_over_periods|{e.label}',
                )

    def add_share_periodic(
        self,
        name: str,
        effect_id: str,
        expression: linopy.LinearExpression,
    ) -> None:
        """Add a periodic share to a specific effect.

        Args:
            name: Element identifier (for debugging)
            effect_id: Target effect identifier
            expression: The share expression to add
        """
        # Expand expression to have effect dimension (with zeros for other effects)
        effect_mask = xr.DataArray(
            [1 if eid == effect_id else 0 for eid in self.effect_ids],
            coords={'effect': self.effect_ids},
            dims=['effect'],
        )
        self._eq_periodic.lhs -= expression * effect_mask

    def add_share_temporal(
        self,
        name: str,
        effect_id: str,
        expression: linopy.LinearExpression,
    ) -> None:
        """Add a temporal (per-timestep) share to a specific effect.

        Args:
            name: Element identifier (for debugging)
            effect_id: Target effect identifier
            expression: The share expression to add
        """
        # Expand expression to have effect dimension (with zeros for other effects)
        effect_mask = xr.DataArray(
            [1 if eid == effect_id else 0 for eid in self.effect_ids],
            coords={'effect': self.effect_ids},
            dims=['effect'],
        )
        self._eq_per_timestep.lhs -= expression * effect_mask

    def finalize_shares(self) -> None:
        """Collect effect contributions from type models (push-based).

        Each type model (FlowsModel, StoragesModel) registers its share definitions
        via register_temporal_share() / register_periodic_share(). This method
        creates the two share variables (share|temporal, share|periodic) with a
        unified 'contributor' dimension, then applies all contributions.
        """
        if (fm := self.model._flows_model) is not None:
            fm.add_effect_contributions(self)
        if (sm := self.model._storages_model) is not None:
            sm.add_effect_contributions(self)

        # === Create share|temporal variable ===
        if self._temporal_share_defs:
            self.share_temporal = self._create_share_var(self._temporal_share_defs, 'share|temporal', temporal=True)
            self._eq_per_timestep.lhs -= self.share_temporal.sum('contributor')

        # === Create share|periodic variable ===
        if self._periodic_share_defs:
            self.share_periodic = self._create_share_var(self._periodic_share_defs, 'share|periodic', temporal=False)
            self._eq_periodic.lhs -= self.share_periodic.sum('contributor').reindex({'effect': self._effect_index})

        # Apply extra temporal contributions (status effects, etc.)
        if self._temporal_contributions:
            self._eq_per_timestep.lhs -= sum(self._temporal_contributions)

        # Apply extra periodic contributions (invested-based effects, etc.)
        for expr in self._periodic_contributions:
            self._eq_periodic.lhs -= expr.reindex({'effect': self._effect_index})

    def _share_coords(self, element_dim: str, element_index, temporal: bool = True) -> xr.Coordinates:
        """Build coordinates for share variables: (element, effect) + time/period/scenario."""
        base_dims = None if temporal else ['period', 'scenario']
        return xr.Coordinates(
            {
                element_dim: element_index,
                'effect': self._effect_index,
                **{k: v for k, v in (self.model.get_coords(base_dims) or {}).items()},
            }
        )

    def _create_share_var(
        self,
        share_defs: list[tuple[list[str], linopy.LinearExpression]],
        name: str,
        temporal: bool,
    ) -> linopy.Variable:
        """Create a share variable from registered contributor definitions.

        Concatenates all contributor expressions along a unified 'contributor' dimension,
        creates one variable and one defining constraint.
        """
        import pandas as pd

        all_ids = [str(cid) for ids, _ in share_defs for cid in ids]
        contributor_index = pd.Index(all_ids, name='contributor')
        coords = self._share_coords('contributor', contributor_index, temporal=temporal)
        var = self.model.add_variables(lower=-np.inf, upper=np.inf, coords=coords, name=name)

        # Concatenate defining expressions along contributor dim
        expr_datasets = [expr.data for _, expr in share_defs]
        combined_data = xr.concat(expr_datasets, dim='contributor')
        combined_expr = linopy.LinearExpression(combined_data, self.model)

        self.model.add_constraints(var == combined_expr, name=name)
        return var

    def get_periodic(self, effect_id: str) -> linopy.Variable:
        """Get periodic variable for a specific effect."""
        return self.periodic.sel(effect=effect_id)

    def get_temporal(self, effect_id: str) -> linopy.Variable:
        """Get temporal variable for a specific effect."""
        return self.temporal.sel(effect=effect_id)

    def get_per_timestep(self, effect_id: str) -> linopy.Variable:
        """Get per_timestep variable for a specific effect."""
        return self.per_timestep.sel(effect=effect_id)

    def get_total(self, effect_id: str) -> linopy.Variable:
        """Get total variable for a specific effect."""
        return self.total.sel(effect=effect_id)


EffectExpr = dict[str, linopy.LinearExpression]  # Used to create Shares


class EffectCollection(ElementContainer[Effect]):
    """
    Handling all Effects
    """

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

        self.add_effects(*effects)

    def create_model(self, model: FlowSystemModel) -> EffectCollectionModel:
        self._plausibility_checks()
        effects_model = EffectCollectionModel(model, self)
        effects_model.do_modeling()
        return effects_model

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


class EffectCollectionModel:
    """
    Handling all Effects using type-level (batched) modeling.
    """

    def __init__(self, model: FlowSystemModel, effects: EffectCollection):
        self.effects = effects
        self._model = model
        self._batched_model: EffectsModel | None = None

    def add_share_to_effects(
        self,
        name: str,
        expressions: EffectExpr,
        target: Literal['temporal', 'periodic'],
    ) -> None:
        """Add effect shares using batched EffectsModel."""
        for effect, expression in expressions.items():
            if self._batched_model is None:
                raise RuntimeError('EffectsModel not initialized. Call do_modeling() first.')
            effect_id = self.effects[effect].label
            if target == 'temporal':
                self._batched_model.add_share_temporal(name, effect_id, expression)
            elif target == 'periodic':
                self._batched_model.add_share_periodic(name, effect_id, expression)
            else:
                raise ValueError(f'Target {target} not supported!')

    def do_modeling(self):
        """Create variables and constraints using batched EffectsModel."""
        # Ensure penalty effect exists (auto-create if user hasn't defined one)
        if self.effects._penalty_effect is None:
            penalty_effect = self.effects._create_penalty_effect()
            # Link to FlowSystem (should already be linked, but ensure it)
            if penalty_effect._flow_system is None:
                penalty_effect.link_to_flow_system(self._model.flow_system)

        # Create batched EffectsModel
        self._batched_model = EffectsModel(
            model=self._model,
            effects=list(self.effects.values()),
        )
        self._batched_model.create_variables()

        # Add cross-effect shares
        self._add_share_between_effects()

        # Objective: sum over effect dim for objective and penalty effects
        obj_id = self.effects.objective_effect.label
        pen_id = self.effects.penalty_effect.label
        self._model.add_objective(
            (self._batched_model.total.sel(effect=obj_id) * self._model.objective_weights).sum()
            + (self._batched_model.total.sel(effect=pen_id) * self._model.objective_weights).sum()
        )

    def _add_share_between_effects(self):
        """Type-level mode: Add cross-effect shares using batched EffectsModel."""
        for target_effect in self.effects.values():
            target_id = target_effect.label
            # 1. temporal: <- receiving temporal shares from other effects
            for source_effect, time_series in target_effect.share_from_temporal.items():
                source_id = self.effects[source_effect].label
                source_per_timestep = self._batched_model.get_per_timestep(source_id)
                self._batched_model.add_share_temporal(
                    f'{source_id}(temporal)',
                    target_id,
                    source_per_timestep * time_series,
                )
            # 2. periodic: <- receiving periodic shares from other effects
            for source_effect, factor in target_effect.share_from_periodic.items():
                source_id = self.effects[source_effect].label
                source_periodic = self._batched_model.get_periodic(source_id)
                self._batched_model.add_share_periodic(
                    f'{source_id}(periodic)',
                    target_id,
                    source_periodic * factor,
                )

    def apply_batched_flow_effect_shares(
        self,
        flow_rate: linopy.Variable,
        effect_specs: dict[str, list[tuple[str, float | xr.DataArray]]],
    ) -> None:
        """Apply batched effect shares for flows to all relevant effects.

        This method receives pre-grouped effect specifications and applies them
        efficiently using vectorized operations.

        In type_level mode:
            - Tracks contributions for unified share variable creation
            - Adds sum of shares to effect's per_timestep constraint

        In traditional mode:
            - Creates per-effect share variables
            - Adds sum to effect's total_per_timestep

        Args:
            flow_rate: The batched flow_rate variable with flow dimension.
            effect_specs: Dict mapping effect_name to list of (element_id, factor) tuples.
                Example: {'costs': [('Boiler(gas_in)', 0.05), ('HP(elec_in)', 0.1)]}
        """

        # Detect the element dimension name from flow_rate (e.g., 'flow')
        flow_rate_dims = [d for d in flow_rate.dims if d not in ('time', 'period', 'scenario', '_term')]
        dim = flow_rate_dims[0] if flow_rate_dims else 'flow'

        for effect_name, element_factors in effect_specs.items():
            if effect_name not in self.effects:
                logger.warning(f'Effect {effect_name} not found, skipping shares')
                continue

            element_ids = [eid for eid, _ in element_factors]
            factors = [factor for _, factor in element_factors]

            # Build factors array with element dimension
            # Use coords='minimal' since factors may have different dimensions (e.g., some have period, others don't)
            factors_da = xr.concat(
                [xr.DataArray(f) if not isinstance(f, xr.DataArray) else f for f in factors],
                dim=dim,
                coords='minimal',
            ).assign_coords({dim: element_ids})

            # Select relevant flow rates and compute expression per element
            flow_rate_subset = flow_rate.sel({dim: element_ids})

            # Add sum of shares to effect's per_timestep constraint
            expression_all = flow_rate_subset * self._model.timestep_duration * factors_da
            share_sum = expression_all.sum(dim)
            effect_mask = xr.DataArray(
                [1 if eid == effect_name else 0 for eid in self._batched_model.effect_ids],
                coords={'effect': self._batched_model.effect_ids},
                dims=['effect'],
            )
            self._batched_model._eq_per_timestep.lhs -= share_sum * effect_mask

    def apply_batched_penalty_shares(
        self,
        penalty_expressions: list[tuple[str, linopy.LinearExpression]],
    ) -> None:
        """Apply batched penalty effect shares.

        Creates individual share variables to preserve per-element contribution info.

        Args:
            penalty_expressions: List of (element_label, penalty_expression) tuples.
        """
        for element_label, expression in penalty_expressions:
            # Create share variable for this element (preserves per-element info in results)
            share_var = self._model.add_variables(
                coords=self._model.get_coords(self._model.temporal_dims),
                name=f'{element_label}->Penalty(temporal)',
            )

            # Constraint: share_var == penalty_expression
            self._model.add_constraints(
                share_var == expression,
                name=f'{element_label}->Penalty(temporal)',
            )

            # Add to Penalty effect's per_timestep constraint
            # Expand share_var to have effect dimension (with zeros for other effects)
            effect_mask = xr.DataArray(
                [1 if eid == PENALTY_EFFECT_LABEL else 0 for eid in self._batched_model.effect_ids],
                coords={'effect': self._batched_model.effect_ids},
                dims=['effect'],
            )
            expanded_share = share_var * effect_mask
            self._batched_model._eq_per_timestep.lhs -= expanded_share


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
