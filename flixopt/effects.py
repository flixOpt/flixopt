"""
This module contains the effects of the flixopt framework.
Furthermore, it contains the EffectCollection, which is used to collect all effects of a system.
Different Datatypes are used to represent the effects with assigned values by the user,
which are then transformed into the internal data structure.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

import linopy
import numpy as np
import xarray as xr

from .core import NonTemporalDataUser, Scalar, TemporalData, TemporalDataUser
from .features import ShareAllocationModel
from .structure import Element, ElementModel, FlowSystemModel, Submodel, register_class_for_io

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
        minimum_operation: Minimum allowed total operational contribution across all timesteps.
        maximum_operation: Maximum allowed total operational contribution across all timesteps.
        minimum_operation_per_hour: Minimum allowed operational contribution per timestep.
        maximum_operation_per_hour: Maximum allowed operational contribution per timestep.
        minimum_invest: Minimum allowed total investment contribution.
        maximum_invest: Maximum allowed total investment contribution.
        minimum_total: Minimum allowed total effect (operation + investment combined).
        maximum_total: Maximum allowed total effect (operation + investment combined).
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
            minimum_operation_per_hour=10,  # Minimum 10 m³/h for process stability
            maximum_operation_per_hour=500,  # Maximum 500 m³/h capacity limit
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
        specific_share_to_other_effects_operation: TemporalEffectsUser | None = None,
        specific_share_to_other_effects_invest: NonTemporalEffectsUser | None = None,
        minimum_operation: Scalar | None = None,
        maximum_operation: Scalar | None = None,
        minimum_invest: Scalar | None = None,
        maximum_invest: Scalar | None = None,
        minimum_operation_per_hour: TemporalDataUser | None = None,
        maximum_operation_per_hour: TemporalDataUser | None = None,
        minimum_total: Scalar | None = None,
        maximum_total: Scalar | None = None,
    ):
        super().__init__(label, meta_data=meta_data)
        self.label = label
        self.unit = unit
        self.description = description
        self.is_standard = is_standard
        self.is_objective = is_objective
        self.specific_share_to_other_effects_operation: TemporalEffectsUser = (
            specific_share_to_other_effects_operation if specific_share_to_other_effects_operation is not None else {}
        )
        self.specific_share_to_other_effects_invest: NonTemporalEffectsUser = (
            specific_share_to_other_effects_invest if specific_share_to_other_effects_invest is not None else {}
        )
        self.minimum_operation = minimum_operation
        self.maximum_operation = maximum_operation
        self.minimum_operation_per_hour = minimum_operation_per_hour
        self.maximum_operation_per_hour = maximum_operation_per_hour
        self.minimum_invest = minimum_invest
        self.maximum_invest = maximum_invest
        self.minimum_total = minimum_total
        self.maximum_total = maximum_total

    def transform_data(self, flow_system: FlowSystem, name_prefix: str = '') -> None:
        self.minimum_operation_per_hour = flow_system.fit_to_model_coords(
            f'{self.label_full}|minimum_operation_per_hour', self.minimum_operation_per_hour
        )

        self.maximum_operation_per_hour = flow_system.fit_to_model_coords(
            f'{self.label_full}|maximum_operation_per_hour', self.maximum_operation_per_hour
        )

        self.specific_share_to_other_effects_operation = flow_system.fit_effects_to_model_coords(
            f'{self.label_full}|operation->', self.specific_share_to_other_effects_operation, 'operation'
        )

        self.minimum_operation = flow_system.fit_to_model_coords(
            f'{self.label_full}|minimum_operation', self.minimum_operation, dims=['year', 'scenario']
        )
        self.maximum_operation = flow_system.fit_to_model_coords(
            f'{self.label_full}|maximum_operation', self.maximum_operation, dims=['year', 'scenario']
        )
        self.minimum_invest = flow_system.fit_to_model_coords(
            f'{self.label_full}|minimum_invest', self.minimum_invest, dims=['year', 'scenario']
        )
        self.maximum_invest = flow_system.fit_to_model_coords(
            f'{self.label_full}|maximum_invest', self.maximum_invest, dims=['year', 'scenario']
        )
        self.minimum_total = flow_system.fit_to_model_coords(
            f'{self.label_full}|minimum_total',
            self.minimum_total,
            dims=['year', 'scenario'],
        )
        self.maximum_total = flow_system.fit_to_model_coords(
            f'{self.label_full}|maximum_total', self.maximum_total, dims=['year', 'scenario']
        )
        self.specific_share_to_other_effects_invest = flow_system.fit_effects_to_model_coords(
            f'{self.label_full}|invest->',
            self.specific_share_to_other_effects_invest,
            'invest',
            dims=['year', 'scenario'],
        )

    def create_model(self, model: FlowSystemModel) -> EffectModel:
        self._plausibility_checks()
        self.submodel = EffectModel(model, self)
        return self.submodel

    def _plausibility_checks(self) -> None:
        # TODO: Check for plausibility
        pass


class EffectModel(ElementModel):
    element: Effect  # Type hint

    def __init__(self, model: FlowSystemModel, element: Effect):
        super().__init__(model, element)

    def _do_modeling(self):
        self.total: linopy.Variable | None = None
        self.invest: ShareAllocationModel = self.add_submodels(
            ShareAllocationModel(
                model=self._model,
                dims=('year', 'scenario'),
                label_of_element=self.label_of_element,
                label_of_model=f'{self.label_of_model}(invest)',
                total_max=self.element.maximum_invest,
                total_min=self.element.minimum_invest,
            ),
            short_name='invest',
        )

        self.operation: ShareAllocationModel = self.add_submodels(
            ShareAllocationModel(
                model=self._model,
                dims=('time', 'year', 'scenario'),
                label_of_element=self.label_of_element,
                label_of_model=f'{self.label_of_model}(operation)',
                total_max=self.element.maximum_operation,
                total_min=self.element.minimum_operation,
                min_per_hour=self.element.minimum_operation_per_hour
                if self.element.minimum_operation_per_hour is not None
                else None,
                max_per_hour=self.element.maximum_operation_per_hour
                if self.element.maximum_operation_per_hour is not None
                else None,
            ),
            short_name='operation',
        )

        self.total = self.add_variables(
            lower=self.element.minimum_total if self.element.minimum_total is not None else -np.inf,
            upper=self.element.maximum_total if self.element.maximum_total is not None else np.inf,
            coords=self._model.get_coords(['year', 'scenario']),
            short_name='total',
        )

        self.add_constraints(self.total == self.operation.total + self.invest.total, short_name='total')


TemporalEffectsUser = TemporalDataUser | dict[str, TemporalDataUser]  # User-specified Shares to Effects
""" This datatype is used to define a temporal share to an effect by a certain attribute. """

NonTemporalEffectsUser = NonTemporalDataUser | dict[str, NonTemporalDataUser]  # User-specified Shares to Effects
""" This datatype is used to define a scalar share to an effect by a certain attribute. """

TemporalEffects = dict[str, TemporalData]  # User-specified Shares to Effects
""" This datatype is used internally to handle temporal shares to an effect. """

NonTemporalEffects = dict[str, Scalar]
""" This datatype is used internally to handle scalar shares to an effect. """

EffectExpr = dict[str, linopy.LinearExpression]  # Used to create Shares


class EffectCollection:
    """
    Handling all Effects
    """

    def __init__(self, *effects: Effect):
        self._effects = {}
        self._standard_effect: Effect | None = None
        self._objective_effect: Effect | None = None

        self.submodel: EffectCollectionModel | None = None
        self.add_effects(*effects)

    def create_model(self, model: FlowSystemModel) -> EffectCollectionModel:
        self._plausibility_checks()
        self.submodel = EffectCollectionModel(model, self)
        return self.submodel

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

    def create_effect_values_dict(
        self, effect_values_user: NonTemporalEffectsUser | TemporalEffectsUser
    ) -> dict[str, Scalar | TemporalDataUser] | None:
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
            elif eff is None:
                return self.standard_effect.label
            else:
                return eff

        if effect_values_user is None:
            return None
        if isinstance(effect_values_user, dict):
            return {get_effect_label(effect): value for effect, value in effect_values_user.items()}
        return {self.standard_effect.label: effect_values_user}

    def _plausibility_checks(self) -> None:
        # Check circular loops in effects:
        operation, invest = self.calculate_effect_share_factors()

        operation_cycles = detect_cycles(tuples_to_adjacency_list([key for key in operation]))
        invest_cycles = detect_cycles(tuples_to_adjacency_list([key for key in invest]))

        if operation_cycles:
            cycle_str = '\n'.join([' -> '.join(cycle) for cycle in operation_cycles])
            raise ValueError(f'Error: circular operation-shares detected:\n{cycle_str}')

        if invest_cycles:
            cycle_str = '\n'.join([' -> '.join(cycle) for cycle in invest_cycles])
            raise ValueError(f'Error: circular invest-shares detected:\n{cycle_str}')

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
            if item.label_full in self.effects:
                return True
            if item in self.effects.values():  # Check if the object exists
                return True
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

    def calculate_effect_share_factors(
        self,
    ) -> tuple[
        dict[tuple[str, str], xr.DataArray],
        dict[tuple[str, str], xr.DataArray],
    ]:
        shares_invest = {}
        for name, effect in self.effects.items():
            if effect.specific_share_to_other_effects_invest:
                shares_invest[name] = {
                    target: data for target, data in effect.specific_share_to_other_effects_invest.items()
                }
        shares_invest = calculate_all_conversion_paths(shares_invest)

        shares_operation = {}
        for name, effect in self.effects.items():
            if effect.specific_share_to_other_effects_operation:
                shares_operation[name] = {
                    target: data for target, data in effect.specific_share_to_other_effects_operation.items()
                }
        shares_operation = calculate_all_conversion_paths(shares_operation)

        return shares_operation, shares_invest


class EffectCollectionModel(Submodel):
    """
    Handling all Effects
    """

    def __init__(self, model: FlowSystemModel, effects: EffectCollection):
        self.effects = effects
        self.penalty: ShareAllocationModel | None = None
        super().__init__(model, label_of_element='Effects')

    def add_share_to_effects(
        self,
        name: str,
        expressions: EffectExpr,
        target: Literal['operation', 'invest'],
    ) -> None:
        for effect, expression in expressions.items():
            if target == 'operation':
                self.effects[effect].submodel.operation.add_share(
                    name,
                    expression,
                    dims=('time', 'year', 'scenario'),
                )
            elif target == 'invest':
                self.effects[effect].submodel.invest.add_share(
                    name,
                    expression,
                    dims=('year', 'scenario'),
                )
            else:
                raise ValueError(f'Target {target} not supported!')

    def add_share_to_penalty(self, name: str, expression: linopy.LinearExpression) -> None:
        if expression.ndim != 0:
            raise TypeError(f'Penalty shares must be scalar expressions! ({expression.ndim=})')
        self.penalty.add_share(name, expression, dims=())

    def _do_modeling(self):
        super()._do_modeling()
        for effect in self.effects:
            effect.create_model(self._model)
        self.penalty = self.add_submodels(
            ShareAllocationModel(self._model, dims=(), label_of_element='Penalty'),
            short_name='penalty',
        )

        self._add_share_between_effects()

        self._model.add_objective(
            (self.effects.objective_effect.submodel.total * self._model.weights).sum() + self.penalty.total.sum()
        )

    def _add_share_between_effects(self):
        for origin_effect in self.effects:
            # 1. operation: -> hier sind es Zeitreihen (share_TS)
            for target_effect, time_series in origin_effect.specific_share_to_other_effects_operation.items():
                self.effects[target_effect].submodel.operation.add_share(
                    origin_effect.submodel.operation.label_full,
                    origin_effect.submodel.operation.total_per_timestep * time_series,
                    dims=('time', 'year', 'scenario'),
                )
            # 2. invest:    -> hier ist es Scalar (share)
            for target_effect, factor in origin_effect.specific_share_to_other_effects_invest.items():
                self.effects[target_effect].submodel.invest.add_share(
                    origin_effect.submodel.invest.label_full,
                    origin_effect.submodel.invest.total * factor,
                    dims=('year', 'scenario'),
                )


def calculate_all_conversion_paths(
    conversion_dict: dict[str, dict[str, xr.DataArray]],
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
        queue = [(origin, 1, [origin])]

        while queue:
            current_domain, factor, path = queue.pop(0)

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
