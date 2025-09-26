"""
This module contains the effects of the flixopt framework.
Furthermore, it contains the EffectCollection, which is used to collect all effects of a system.
Different Datatypes are used to represent the effects with assigned values by the user,
which are then transformed into the internal data structure.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Literal

import linopy
import numpy as np

from .core import NumericDataTS, Scalar, TimeSeries
from .features import ShareAllocationModel
from .structure import Element, ElementModel, Model, SystemModel, register_class_for_io

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


@register_class_for_io
class Effect(Element):
    """
    Represents system-wide impacts like costs, emissions, resource consumption, or other effects.

    Effects capture the broader impacts of system operation and investment decisions beyond
    the primary energy/material flows. Each Effect accumulates contributions from Components,
    Flows, and other system elements. One Effect is typically chosen as the optimization
    objective, while others can serve as constraints or tracking metrics.

    Effects support comprehensive modeling including operational and investment contributions,
    cross-effect relationships (e.g., carbon pricing), and flexible constraint formulation.

    Args:
        label: The label of the Element. Used to identify it in the FlowSystem.
        unit: The unit of the effect (e.g., '€', 'kg_CO2', 'kWh_primary', 'm²').
            This is informative only and does not affect optimization calculations.
        description: Descriptive name explaining what this effect represents.
        is_standard: If True, this is a standard effect allowing direct value input
            without effect dictionaries. Used for simplified effect specification (and less boilerplate code).
        is_objective: If True, this effect serves as the optimization objective function.
            Only one effect can be marked as objective per optimization.
        specific_share_to_other_effects_operation: Operational cross-effect contributions.
            Maps this effect's operational values to contributions to other effects
        specific_share_to_other_effects_invest: Investment cross-effect contributions.
            Maps this effect's investment values to contributions to other effects.
        minimum_temporal: Minimum allowed total contribution across all timesteps.
        maximum_temporal: Maximum allowed total contribution across all timesteps.
        minimum_temporal_per_hour: Minimum allowed contribution per hour.
        maximum_temporal_per_hour: Maximum allowed contribution per hour.
        minimum_nontemporal: Minimum allowed total nontemporal contribution.
        maximum_nontemporal: Maximum allowed total nontemporal contribution.
        minimum_total: Minimum allowed total effect (temporal + nontemporal combined).
        maximum_total: Maximum allowed total effect (temporal + nontemporal combined).
        meta_data: Used to store additional information. Not used internally but saved
            in results. Only use Python native types.

    Examples:
        Basic cost objective:

        ```python
        cost_effect = Effect(label='system_costs', unit='€', description='Total system costs', is_objective=True)
        ```

        CO2 emissions with carbon pricing:

        ```python
        co2_effect = Effect(
            label='co2_emissions',
            unit='kg_CO2',
            description='Carbon dioxide emissions',
            specific_share_to_other_effects_operation={'costs': 50},  # €50/t_CO2
            maximum_total=1_000_000,  # 1000 t CO2 annual limit
        )
        ```

        Land use constraint:

        ```python
        land_use = Effect(
            label='land_usage',
            unit='m²',
            description='Land area requirement',
            maximum_total=50_000,  # Maximum 5 hectares available
        )
        ```

        Primary energy tracking:

        ```python
        primary_energy = Effect(
            label='primary_energy',
            unit='kWh_primary',
            description='Primary energy consumption',
            specific_share_to_other_effects_operation={'costs': 0.08},  # €0.08/kWh
        )
        ```

        Water consumption with tiered constraints:

        ```python
        water_usage = Effect(
            label='water_consumption',
            unit='m³',
            description='Industrial water usage',
            minimum_temporal_per_hour=10,  # Minimum 10 m³/h for process stability
            maximum_temporal_per_hour=500,  # Maximum 500 m³/h capacity limit
            maximum_total=100_000,  # Annual permit limit: 100,000 m³
        )
        ```

    Note:
        Effect bounds can be None to indicate no constraint in that direction.

        Cross-effect relationships enable sophisticated modeling like carbon pricing,
        resource valuation, or multi-criteria optimization with weighted objectives.

        The unit field is purely informational - ensure dimensional consistency
        across all contributions to each effect manually.

        Effects are accumulated as:
        - Total = Σ(operational contributions) + Σ(investment contributions)
        - Cross-effects add to target effects based on specific_share ratios

    """

    def __init__(
        self,
        label: str,
        unit: str,
        description: str,
        meta_data: dict | None = None,
        is_standard: bool = False,
        is_objective: bool = False,
        specific_share_to_other_effects_operation: EffectValuesUser | None = None,
        specific_share_to_other_effects_invest: EffectValuesUser | None = None,
        minimum_temporal: Scalar | None = None,
        maximum_temporal: Scalar | None = None,
        minimum_nontemporal: Scalar | None = None,
        maximum_nontemporal: Scalar | None = None,
        minimum_temporal_per_hour: NumericDataTS | None = None,
        maximum_temporal_per_hour: NumericDataTS | None = None,
        minimum_total: Scalar | None = None,
        maximum_total: Scalar | None = None,
    ):
        super().__init__(label, meta_data=meta_data)
        self.label = label
        self.unit = unit
        self.description = description
        self.is_standard = is_standard
        self.is_objective = is_objective
        self.specific_share_to_other_effects_operation: EffectValuesUser = (
            specific_share_to_other_effects_operation or {}
        )
        self.specific_share_to_other_effects_invest: EffectValuesUser = specific_share_to_other_effects_invest or {}
        self.minimum_temporal = minimum_temporal
        self.maximum_temporal = maximum_temporal
        self.minimum_temporal_per_hour = minimum_temporal_per_hour
        self.maximum_temporal_per_hour = maximum_temporal_per_hour
        self.minimum_nontemporal = minimum_nontemporal
        self.maximum_nontemporal = maximum_nontemporal
        self.minimum_total = minimum_total
        self.maximum_total = maximum_total

    def transform_data(self, flow_system: FlowSystem):
        self.minimum_temporal_per_hour = flow_system.create_time_series(
            f'{self.label_full}|minimum_temporal_per_hour', self.minimum_temporal_per_hour
        )
        self.maximum_temporal_per_hour = flow_system.create_time_series(
            f'{self.label_full}|maximum_temporal_per_hour',
            self.maximum_temporal_per_hour,
        )

        self.specific_share_to_other_effects_operation = flow_system.create_effect_time_series(
            f'{self.label_full}|operation->', self.specific_share_to_other_effects_operation, 'operation'
        )

    def create_model(self, model: SystemModel) -> EffectModel:
        self._plausibility_checks()
        self.model = EffectModel(model, self)
        return self.model

    def _plausibility_checks(self) -> None:
        # TODO: Check for plausibility
        pass


class EffectModel(ElementModel):
    def __init__(self, model: SystemModel, element: Effect):
        super().__init__(model, element)
        self.element: Effect = element
        self.total: linopy.Variable | None = None
        self.nontemporal: ShareAllocationModel = self.add(
            ShareAllocationModel(
                self._model,
                False,
                self.label_of_element,
                'nontemporal',
                label_full=f'{self.label_full}(nontemporal)',
                total_max=self.element.maximum_nontemporal,
                total_min=self.element.minimum_nontemporal,
            )
        )

        self.temporal: ShareAllocationModel = self.add(
            ShareAllocationModel(
                self._model,
                True,
                self.label_of_element,
                'temporal',
                label_full=f'{self.label_full}(temporal)',
                total_max=self.element.maximum_temporal,
                total_min=self.element.minimum_temporal,
                min_per_hour=self.element.minimum_temporal_per_hour.active_data
                if self.element.minimum_temporal_per_hour is not None
                else None,
                max_per_hour=self.element.maximum_temporal_per_hour.active_data
                if self.element.maximum_temporal_per_hour is not None
                else None,
            )
        )

    def do_modeling(self):
        for model in self.sub_models:
            model.do_modeling()

        self.total = self.add(
            self._model.add_variables(
                lower=self.element.minimum_total if self.element.minimum_total is not None else -np.inf,
                upper=self.element.maximum_total if self.element.maximum_total is not None else np.inf,
                coords=None,
                name=f'{self.label_full}',
            ),
            'total',
        )

        self.add(
            self._model.add_constraints(
                self.total == self.temporal.total.sum() + self.nontemporal.total.sum(), name=f'{self.label_full}'
            ),
            'total',
        )


EffectValuesExpr = dict[str, linopy.LinearExpression]  # Used to create Shares
EffectTimeSeries = dict[str, TimeSeries]  # Used internally to index values
EffectValuesDict = dict[str, NumericDataTS]  # How effect values are stored
EffectValuesUser = NumericDataTS | dict[str, NumericDataTS]  # User-specified Shares to Effects
""" This datatype is used to define the share to an effect by a certain attribute. """

EffectValuesUserScalar = Scalar | dict[str, Scalar]  # User-specified Shares to Effects
""" This datatype is used to define the share to an effect by a certain attribute. Only scalars are allowed. """


class EffectCollection:
    """
    Handling all Effects
    """

    def __init__(self, *effects: Effect):
        self._effects = {}
        self._standard_effect: Effect | None = None
        self._objective_effect: Effect | None = None

        self.model: EffectCollectionModel | None = None
        self.add_effects(*effects)

    def create_model(self, model: SystemModel) -> EffectCollectionModel:
        self._plausibility_checks()
        self.model = EffectCollectionModel(model, self)
        return self.model

    def add_effects(self, *effects: Effect) -> None:
        for effect in list(effects):
            if effect in self:
                raise ValueError(f'Effect with label "{effect.label=}" already added!')
            if effect.is_standard:
                self.standard_effect = effect
            if effect.is_objective:
                self.objective_effect = effect
            self._effects[effect.label] = effect
            logger.info(f'Registered new Effect: {effect.label}')

    def create_effect_values_dict(self, effect_values_user: EffectValuesUser) -> EffectValuesDict | None:
        """
        Converts effect values into a dictionary. If a scalar is provided, it is associated with a default effect type.

        Examples
        --------
        effect_values_user = 20                             -> {None: 20}
        effect_values_user = None                           -> None
        effect_values_user = {effect1: 20, effect2: 0.3}    -> {effect1: 20, effect2: 0.3}

        Returns
        -------
        dict or None
            A dictionary with None or Effect as the key, or None if input is None.
        """

        def get_effect_label(eff: Effect | str) -> str:
            """Temporary function to get the label of an effect and warn for deprecation"""
            if isinstance(eff, Effect):
                warnings.warn(
                    f'The use of effect objects when specifying EffectValues is deprecated. '
                    f'Use the label of the effect instead. Used effect: {eff.label_full}',
                    UserWarning,
                    stacklevel=2,
                )
                return eff.label
            else:
                return eff

        if effect_values_user is None:
            return None
        if isinstance(effect_values_user, dict):
            return {get_effect_label(effect): value for effect, value in effect_values_user.items()}
        return {self.standard_effect.label: effect_values_user}

    def _plausibility_checks(self) -> None:
        # Check circular loops in effects:
        # TODO: Improve checks!! Only most basic case covered...

        def error_str(effect_label: str, share_ffect_label: str):
            return (
                f'  {effect_label} -> has share in: {share_ffect_label}\n'
                f'  {share_ffect_label} -> has share in: {effect_label}'
            )

        for effect in self.effects.values():
            # Effekt darf nicht selber als Share in seinen ShareEffekten auftauchen:
            # operation:
            for target_effect in effect.specific_share_to_other_effects_operation.keys():
                assert effect not in self[target_effect].specific_share_to_other_effects_operation.keys(), (
                    f'Error: circular operation-shares \n{error_str(effect.label, self[target_effect].label)}'
                )
            # invest:
            for target_effect in effect.specific_share_to_other_effects_invest.keys():
                assert effect not in self[target_effect].specific_share_to_other_effects_invest.keys(), (
                    f'Error: circular invest-shares \n{error_str(effect.label, self[target_effect].label)}'
                )

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
            return self.effects[effect]
        except KeyError as e:
            raise KeyError(f'Effect "{effect}" not found! Add it to the FlowSystem first!') from e

    def __iter__(self) -> Iterator[Effect]:
        return iter(self._effects.values())

    def __len__(self) -> int:
        return len(self._effects)

    def __contains__(self, item: str | Effect) -> bool:
        """Check if the effect exists. Checks for label or object"""
        if isinstance(item, str):
            return item in self.effects  # Check if the label exists
        elif isinstance(item, Effect):
            return item in self.effects.values()  # Check if the object exists
        return False

    @property
    def effects(self) -> dict[str, Effect]:
        return self._effects

    @property
    def standard_effect(self) -> Effect:
        if self._standard_effect is None:
            raise KeyError('No standard-effect specified!')
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
        if self._objective_effect is not None:
            raise ValueError(f'An objective-effect already exists! ({self._objective_effect.label=})')
        self._objective_effect = value


class EffectCollectionModel(Model):
    """
    Handling all Effects
    """

    def __init__(self, model: SystemModel, effects: EffectCollection):
        super().__init__(model, label_of_element='Effects')
        self.effects = effects
        self.penalty: ShareAllocationModel | None = None

    def add_share_to_effects(
        self,
        name: str,
        expressions: EffectValuesExpr,
        target: Literal['temporal', 'nontemporal'],
    ) -> None:
        for effect, expression in expressions.items():
            if target == 'temporal':
                self.effects[effect].model.temporal.add_share(name, expression)
            elif target == 'nontemporal':
                self.effects[effect].model.nontemporal.add_share(name, expression)
            else:
                raise ValueError(f'Target {target} not supported!')

    def add_share_to_penalty(self, name: str, expression: linopy.LinearExpression) -> None:
        if expression.ndim != 0:
            raise TypeError(f'Penalty shares must be scalar expressions! ({expression.ndim=})')
        self.penalty.add_share(name, expression)

    def do_modeling(self):
        for effect in self.effects:
            effect.create_model(self._model)
        self.penalty = self.add(
            ShareAllocationModel(self._model, shares_are_time_series=False, label_of_element='Penalty')
        )
        for model in [effect.model for effect in self.effects] + [self.penalty]:
            model.do_modeling()

        self._add_share_between_effects()

        self._model.add_objective(self.effects.objective_effect.model.total + self.penalty.total)

    def _add_share_between_effects(self):
        for origin_effect in self.effects:
            # 1. temporal: -> hier sind es Zeitreihen (share_TS)
            for target_effect, time_series in origin_effect.specific_share_to_other_effects_operation.items():
                self.effects[target_effect].model.temporal.add_share(
                    origin_effect.model.temporal.label_full,
                    origin_effect.model.temporal.total_per_timestep * time_series.active_data,
                )
            # 2. nontemporal:    -> hier ist es Scalar (share)
            for target_effect, factor in origin_effect.specific_share_to_other_effects_invest.items():
                self.effects[target_effect].model.nontemporal.add_share(
                    origin_effect.model.nontemporal.label_full,
                    origin_effect.model.nontemporal.total * factor,
                )
