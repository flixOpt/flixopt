"""
This module contains the effects of the flixopt framework.
Furthermore, it contains the EffectCollection, which is used to collect all effects of a system.
Different Datatypes are used to represent the effects with assigned values by the user,
which are then transformed into the internal data structure.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

import linopy
import numpy as np
import xarray as xr

from .core import PlausibilityError
from .id_list import IdList
from .structure import (
    Element,
    FlowSystemModel,
    register_class_for_io,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .types import Effect_PS, Effect_TPS, Numeric_PS, Numeric_S, Numeric_TPS, Scalar

logger = logging.getLogger('flixopt')

# Penalty effect ID constant
PENALTY_EFFECT_ID = 'Penalty'

# Deprecated alias
PENALTY_EFFECT_LABEL = PENALTY_EFFECT_ID


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
        id: The id of the Element. Used to identify it in the FlowSystem.
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
            id='system_costs',
            unit='€',
            description='Total system costs',
            is_objective=True,
        )
        ```

        CO2 emissions with per-period limit:

        ```python
        co2_effect = Effect(
            id='CO2',
            unit='kg_CO2',
            description='Carbon dioxide emissions',
            maximum_total=100_000,  # 100 t CO2 per period
        )
        ```

        CO2 emissions with total limit across all periods:

        ```python
        co2_effect = Effect(
            id='CO2',
            unit='kg_CO2',
            description='Carbon dioxide emissions',
            maximum_over_periods=1_000_000,  # 1000 t CO2 total across all periods
        )
        ```

        Land use constraint:

        ```python
        land_use = Effect(
            id='land_usage',
            unit='m²',
            description='Land area requirement',
            maximum_total=50_000,  # Maximum 5 hectares per period
        )
        ```

        Primary energy tracking:

        ```python
        primary_energy = Effect(
            id='primary_energy',
            unit='kWh_primary',
            description='Primary energy consumption',
        )
        ```

       Cost objective with carbon and primary energy pricing:

        ```python
        cost_effect = Effect(
            id='system_costs',
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
            id='water_consumption',
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
        id: str | None = None,
        unit: str = '',
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
        **kwargs,
    ):
        super().__init__(id, meta_data=meta_data, **kwargs)
        self.unit = unit
        self.description = description
        self.is_standard = is_standard

        # Validate that Penalty cannot be set as objective (compare resolved self.id, not the id argument,
        # so the check is not bypassed when a deprecated label is used with id=None)
        if is_objective and self.id == PENALTY_EFFECT_ID:
            raise ValueError(
                f'The Penalty effect ("{PENALTY_EFFECT_ID}") cannot be set as the objective effect. '
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

        Elements use their id as prefix by default, ignoring the passed prefix.
        """
        super().link_to_flow_system(flow_system, self.id)

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

    def validate_config(self) -> None:
        """Validate configuration consistency.

        Called BEFORE transformation via FlowSystem._run_config_validation().
        These are simple checks that don't require DataArray operations.
        """
        # Check that minimum_over_periods and maximum_over_periods require a period dimension
        if (
            self.minimum_over_periods is not None or self.maximum_over_periods is not None
        ) and self.flow_system.periods is None:
            raise PlausibilityError(
                f"Effect '{self.id}': minimum_over_periods and maximum_over_periods require "
                f"the FlowSystem to have a 'period' dimension. Please define periods when creating "
                f'the FlowSystem, or remove these constraints.'
            )

    def _plausibility_checks(self) -> None:
        """Legacy validation method - delegates to validate_config()."""
        self.validate_config()


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

    def __init__(self, model: FlowSystemModel, data):
        self.model = model
        self.data = data

        # Variables (set during do_modeling / create_variables)
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
        # Per-effect, per-contributor accumulation: effect_id -> {contributor_id -> expr (no effect dim)}
        self._temporal_shares: dict[str, dict[str, linopy.LinearExpression]] = {}
        self._periodic_shares: dict[str, dict[str, linopy.LinearExpression]] = {}
        # Constant (xr.DataArray) contributions with 'contributor' + 'effect' dims
        self._temporal_constant_defs: list[xr.DataArray] = []
        self._periodic_constant_defs: list[xr.DataArray] = []

        self.create_variables()
        self._add_share_between_effects()
        self._set_objective()

    @property
    def effect_index(self):
        """Public access to the effect index for type models."""
        return self.data.effect_index

    def add_temporal_contribution(
        self,
        defining_expr,
        contributor_dim: str = 'contributor',
        effect: str | None = None,
    ) -> None:
        """Register contributors for the share|temporal variable.

        Args:
            defining_expr: Expression with a contributor dimension (no effect dim if effect is given).
            contributor_dim: Name of the element dimension to rename to 'contributor'.
            effect: If provided, the expression is for this specific effect (no effect dim needed).
        """
        if contributor_dim != 'contributor':
            defining_expr = defining_expr.rename({contributor_dim: 'contributor'})
        if isinstance(defining_expr, xr.DataArray):
            if effect is not None:
                defining_expr = defining_expr.expand_dims(effect=[effect])
            elif 'effect' not in defining_expr.dims:
                raise ValueError(
                    "DataArray contribution must have an 'effect' dimension or an explicit effect= argument."
                )
            self._temporal_constant_defs.append(defining_expr)
        else:
            self._accumulate_shares(self._temporal_shares, self._as_expression(defining_expr), effect)

    def add_periodic_contribution(
        self,
        defining_expr,
        contributor_dim: str = 'contributor',
        effect: str | None = None,
    ) -> None:
        """Register contributors for the share|periodic variable.

        Args:
            defining_expr: Expression with a contributor dimension (no effect dim if effect is given).
            contributor_dim: Name of the element dimension to rename to 'contributor'.
            effect: If provided, the expression is for this specific effect (no effect dim needed).
        """
        if contributor_dim != 'contributor':
            defining_expr = defining_expr.rename({contributor_dim: 'contributor'})
        if isinstance(defining_expr, xr.DataArray):
            if effect is not None:
                defining_expr = defining_expr.expand_dims(effect=[effect])
            elif 'effect' not in defining_expr.dims:
                raise ValueError(
                    "DataArray contribution must have an 'effect' dimension or an explicit effect= argument."
                )
            self._periodic_constant_defs.append(defining_expr)
        else:
            self._accumulate_shares(self._periodic_shares, self._as_expression(defining_expr), effect)

    @staticmethod
    def _accumulate_shares(
        accum: dict[str, list],
        expr: linopy.LinearExpression,
        effect: str | None = None,
    ) -> None:
        """Append expression to per-effect list, dropping zero-coefficient contributors."""
        # accum structure: {effect_id: [expr1, expr2, ...]}
        if effect is not None:
            # Expression has no effect dim — tagged with specific effect
            accum.setdefault(effect, []).append(expr)
        elif 'effect' in expr.dims:
            # Expression has effect dim — split per effect, drop all-zero contributors
            # to avoid inflating the model with unused (contributor, effect) variable slots.
            for eid in expr.coords['effect'].values:
                sliced = expr.sel(effect=eid, drop=True)
                # Keep only contributors with at least one non-zero coefficient
                reduce_dims = [d for d in sliced.coeffs.dims if d != 'contributor']
                nonzero = (sliced.coeffs != 0).any(dim=reduce_dims)
                if nonzero.any():
                    active_contributors = nonzero.coords['contributor'].values[nonzero.values]
                    accum.setdefault(str(eid), []).append(sliced.sel(contributor=active_contributors))
        else:
            raise ValueError('Expression must have effect dim or effect parameter must be given')

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
                {'effect': self.data.effect_index},
                self.model.get_coords(['period', 'scenario']),
            )
        )
        self.periodic = self.model.add_variables(
            lower=self.data.minimum_periodic,
            upper=self.data.maximum_periodic,
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
            lower=self.data.minimum_temporal,
            upper=self.data.maximum_temporal,
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
                {'effect': self.data.effect_index},
                self.model.get_coords(None),  # All dims
            )
        )

        # Build per-hour bounds
        min_per_hour = self.data.minimum_per_hour
        max_per_hour = self.data.maximum_per_hour

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
            lower=self.data.minimum_total,
            upper=self.data.maximum_total,
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
        effects_with_over_periods = self.data.effects_with_over_periods
        if effects_with_over_periods:
            over_periods_ids = [e.id for e in effects_with_over_periods]
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
                total_e = self.total.sel(effect=e.id)
                weights_e = self.data.period_weights[e.id]
                weighted_total = (total_e * weights_e).sum('period')
                self.model.add_constraints(
                    self.total_over_periods.sel(effect=e.id) == weighted_total,
                    name=f'effect|total_over_periods|{e.id}',
                )

    def _as_expression(self, expr) -> linopy.LinearExpression:
        """Convert Variable to LinearExpression if needed."""
        if isinstance(expr, linopy.Variable):
            return expr * 1
        return expr

    def add_share_periodic(self, expression) -> None:
        """Add a periodic share expression with effect dimension to effect|periodic.

        The expression must have an 'effect' dimension aligned with the effect index.
        """
        self._eq_periodic.lhs -= self._as_expression(expression).reindex({'effect': self.data.effect_index})

    def add_share_temporal(self, expression) -> None:
        """Add a temporal share expression with effect dimension to effect|per_timestep.

        The expression must have an 'effect' dimension aligned with the effect index.
        """
        self._eq_per_timestep.lhs -= self._as_expression(expression).reindex({'effect': self.data.effect_index})

    def finalize_shares(self) -> None:
        """Collect effect contributions from type models (push-based).

        Each type model (FlowsModel, StoragesModel, ComponentsModel) registers its
        share definitions via add_temporal_contribution() / add_periodic_contribution().
        This method creates the two share variables (share|temporal, share|periodic)
        with a unified 'contributor' dimension, then applies all contributions.
        """
        if (fm := self.model._flows_model) is not None:
            fm.add_effect_contributions(self)
        if (sm := self.model._storages_model) is not None:
            sm.add_effect_contributions(self)
        if (cm := self.model._components_model) is not None:
            cm.add_effect_contributions(self)

        # === Create share|temporal variable (one combined with contributor × effect dims) ===
        if self._temporal_shares:
            self.share_temporal = self._create_share_var(self._temporal_shares, 'share|temporal', temporal=True)
            self._eq_per_timestep.lhs -= self.share_temporal.sum('contributor')

        # === Apply temporal constants directly ===
        for const in self._temporal_constant_defs:
            self._eq_per_timestep.lhs -= const.sum('contributor').reindex({'effect': self.data.effect_index})

        # === Create share|periodic variable (one combined with contributor × effect dims) ===
        if self._periodic_shares:
            self.share_periodic = self._create_share_var(self._periodic_shares, 'share|periodic', temporal=False)
            self._eq_periodic.lhs -= self.share_periodic.sum('contributor')

        # === Apply periodic constants directly ===
        for const in self._periodic_constant_defs:
            self._eq_periodic.lhs -= const.sum('contributor').reindex({'effect': self.data.effect_index})

    def _share_coords(self, element_dim: str, element_index, temporal: bool = True) -> xr.Coordinates:
        """Build coordinates for share variables: (element, effect) + time/period/scenario."""
        base_dims = None if temporal else ['period', 'scenario']
        return xr.Coordinates(
            {
                element_dim: element_index,
                'effect': self.data.effect_index,
                **{k: v for k, v in (self.model.get_coords(base_dims) or {}).items()},
            }
        )

    def _create_share_var(
        self,
        accum: dict[str, list[linopy.LinearExpression]],
        name: str,
        temporal: bool,
    ) -> linopy.Variable:
        """Create one share variable with (contributor, effect, ...) dims.

        accum structure: {effect_id: [expr1, expr2, ...]} where each expr has
        (contributor, ...other_dims) dims — no effect dim.

        Constraints are added per-effect: var.sel(effect=eid) == merged_for_eid,
        which avoids cross-effect alignment.

        Returns:
            linopy.Variable with dims (contributor, effect, time/period).
        """
        import pandas as pd

        if not accum:
            return None

        # Collect all contributor IDs across all effects
        all_contributor_ids: set[str] = set()
        for expr_list in accum.values():
            for expr in expr_list:
                all_contributor_ids.update(str(c) for c in expr.data.coords['contributor'].values)

        contributor_index = pd.Index(sorted(all_contributor_ids), name='contributor')
        effect_index = self.data.effect_index
        coords = self._share_coords('contributor', contributor_index, temporal=temporal)

        # Build mask: only create variables for (effect, contributor) combos that have expressions
        mask = xr.DataArray(
            np.zeros((len(contributor_index), len(effect_index)), dtype=bool),
            dims=['contributor', 'effect'],
            coords={'contributor': contributor_index, 'effect': effect_index},
        )
        covered_map: dict[str, list[str]] = {}
        for eid, expr_list in accum.items():
            cids = set()
            for expr in expr_list:
                cids.update(str(c) for c in expr.data.coords['contributor'].values)
            covered_map[eid] = sorted(cids)
            mask.loc[dict(effect=eid, contributor=covered_map[eid])] = True

        var = self.model.add_variables(lower=-np.inf, upper=np.inf, coords=coords, name=name, mask=mask)

        # Add per-effect constraints (only for covered combos)
        for eid, expr_list in accum.items():
            contributors = covered_map[eid]
            if len(expr_list) == 1:
                merged = expr_list[0].reindex(contributor=contributors)
            else:
                # Reindex all to common contributor set, then sum via linopy.merge (_term addition)
                aligned = [e.reindex(contributor=contributors) for e in expr_list]
                merged = aligned[0]
                for a in aligned[1:]:
                    merged = merged + a
            var_slice = var.sel(effect=eid, contributor=contributors)
            self.model.add_constraints(var_slice == merged, name=f'{name}({eid})')

        accum.clear()
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

    def _add_share_between_effects(self):
        """Register cross-effect shares as contributions (tracked in share variables).

        Effect-to-effect shares are registered via add_temporal/periodic_contribution()
        so they appear in the share variables and can be reconstructed by statistics.
        """
        for target_effect in self.data.values():
            target_id = target_effect.id
            # 1. temporal: <- receiving temporal shares from other effects
            for source_effect, time_series in target_effect.share_from_temporal.items():
                source_id = self.data[source_effect].id
                source_per_timestep = self.get_per_timestep(source_id)
                expr = (source_per_timestep * time_series).expand_dims(effect=[target_id], contributor=[source_id])
                self.add_temporal_contribution(expr)
            # 2. periodic: <- receiving periodic shares from other effects
            for source_effect, factor in target_effect.share_from_periodic.items():
                source_id = self.data[source_effect].id
                source_periodic = self.get_periodic(source_id)
                expr = (source_periodic * factor).expand_dims(effect=[target_id], contributor=[source_id])
                self.add_periodic_contribution(expr)

    def _set_objective(self):
        """Set the optimization objective function."""
        obj_id = self.data.objective_effect_id
        pen_id = self.data.penalty_effect_id
        self.model.add_objective(
            (self.total.sel(effect=obj_id) * self.model.objective_weights).sum()
            + (self.total.sel(effect=pen_id) * self.model.objective_weights).sum()
        )


class EffectCollection(IdList[Effect]):
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
        super().__init__(key_fn=lambda e: e.id, display_name='effects', truncate_repr=truncate_repr)
        self._standard_effect: Effect | None = None
        self._objective_effect: Effect | None = None
        self._penalty_effect: Effect | None = None

        self.add_effects(*effects)

    def _create_penalty_effect(self) -> Effect:
        """
        Create and register the penalty effect (called internally by FlowSystem).
        Only creates if user hasn't already defined a Penalty effect.
        """
        # Check if user has already defined a Penalty effect
        if PENALTY_EFFECT_ID in self:
            self._penalty_effect = self[PENALTY_EFFECT_ID]
            logger.info(f'Using user-defined Penalty Effect: {PENALTY_EFFECT_ID}')
            return self._penalty_effect

        # Auto-create penalty effect
        self._penalty_effect = Effect(
            id=PENALTY_EFFECT_ID,
            unit='penalty_units',
            description='Penalty for constraint violations and modeling artifacts',
            is_standard=False,
            is_objective=False,
        )
        self.add(self._penalty_effect)  # Add to container
        logger.info(f'Auto-created Penalty Effect: {PENALTY_EFFECT_ID}')
        return self._penalty_effect

    def add_effects(self, *effects: Effect) -> None:
        for effect in list(effects):
            if effect in self:
                raise ValueError(f'Effect with id "{effect.id=}" already added!')
            if effect.is_standard:
                self.standard_effect = effect
            if effect.is_objective:
                self.objective_effect = effect
            self.add(effect)
            logger.info(f'Registered new Effect: {effect.id}')

    def create_effect_values_dict(self, effect_values_user: Numeric_TPS | Effect_TPS | None) -> Effect_TPS | None:
        """Converts effect values into a dictionary. If a scalar is provided, it is associated with a default effect type.

        Examples:
            ```python
            effect_values_user = 20                               -> {'<standard_effect_id>': 20}
            effect_values_user = {None: 20}                       -> {'<standard_effect_id>': 20}
            effect_values_user = None                             -> None
            effect_values_user = {'effect1': 20, 'effect2': 0.3}  -> {'effect1': 20, 'effect2': 0.3}
            ```

        Returns:
            A dictionary keyed by effect id, or None if input is None.
            Note: a standard effect must be defined when passing scalars or None ids.
        """

        def get_effect_id(eff: str | None) -> str:
            """Get the id of an effect"""
            if eff is None:
                return self.standard_effect.id
            if isinstance(eff, Effect):
                raise TypeError(
                    f'Effect objects are no longer accepted when specifying EffectValues. '
                    f'Use the id string instead. Got: {eff.id}'
                )
            return eff

        if effect_values_user is None:
            return None
        if isinstance(effect_values_user, dict):
            return {get_effect_id(effect): value for effect, value in effect_values_user.items()}
        return {self.standard_effect.id: effect_values_user}

    def validate_config(self) -> None:
        """Deprecated: Validation is now handled by EffectsData.validate().

        This method is kept for backwards compatibility but does nothing.
        Collection-level validation (cycles, unknown refs) is now in EffectsData._validate_share_structure().
        """
        pass

    def _plausibility_checks(self) -> None:
        """Deprecated: Legacy validation method.

        Kept for backwards compatibility but does nothing.
        Validation is now handled by EffectsData.validate().
        """
        pass

    def __getitem__(self, effect: str | Effect | None) -> Effect:
        """
        Get an effect by id, or return the standard effect if None is passed

        Raises:
            KeyError: If no effect with the given id is found.
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
            return super().__getitem__(effect)
        except KeyError as e:
            # Extract the original message and append context for cleaner output
            original_msg = str(e).strip('\'"')
            raise KeyError(f'{original_msg} Add the effect to the FlowSystem first.') from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())  # Iterate over keys like a normal dict

    def __contains__(self, item: str | Effect) -> bool:
        """Check if the effect exists. Checks for id or object"""
        if isinstance(item, str):
            return super().__contains__(item)  # Check if the id exists
        elif isinstance(item, Effect):
            return item.id in self and self[item.id] is item
        return False

    @property
    def standard_effect(self) -> Effect:
        if self._standard_effect is None:
            raise KeyError(
                'No standard-effect specified! Either set an effect through is_standard=True '
                'or pass a mapping when specifying effect values: {effect_id: value}.'
            )
        return self._standard_effect

    @standard_effect.setter
    def standard_effect(self, value: Effect) -> None:
        if self._standard_effect is not None:
            raise ValueError(f'A standard-effect already exists! ({self._standard_effect.id=})')
        self._standard_effect = value

    @property
    def objective_effect(self) -> Effect:
        if self._objective_effect is None:
            raise KeyError('No objective-effect specified!')
        return self._objective_effect

    @objective_effect.setter
    def objective_effect(self, value: Effect) -> None:
        # Check Penalty first to give users a more specific error message
        if value.id == PENALTY_EFFECT_ID:
            raise ValueError(
                f'The Penalty effect ("{PENALTY_EFFECT_ID}") cannot be set as the objective effect. '
                f'Please use a different effect as the optimization objective.'
            )
        if self._objective_effect is not None:
            raise ValueError(f'An objective-effect already exists! ({self._objective_effect.id=})')
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
        if PENALTY_EFFECT_ID in self:
            self._penalty_effect = self[PENALTY_EFFECT_ID]
            return self._penalty_effect

        # Not yet created - will be created during modeling
        raise KeyError(
            f'Penalty effect not yet created. It will be auto-created during modeling, '
            f'or you can define your own using: Effect("{PENALTY_EFFECT_ID}", ...)'
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
