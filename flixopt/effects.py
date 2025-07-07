"""
This module contains the effects of the flixopt framework.
Furthermore, it contains the EffectCollection, which is used to collect all effects of a system.
Different Datatypes are used to represent the effects with assigned values by the user,
which are then transformed into the internal data structure.
"""

import logging
import warnings
from typing import TYPE_CHECKING, Dict, Iterator, List, Literal, Optional, Set, Tuple, Union

import linopy
import numpy as np
import xarray as xr

from .core import Scalar, TemporalData, TemporalDataUser
from .features import ShareAllocationModel
from .structure import Element, ElementModel, Interface, Model, SystemModel, register_class_for_io

if TYPE_CHECKING:
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


@register_class_for_io
class Effect(Element):
    """
    Effect, i.g. costs, CO2 emissions, area, ...
    Components, FLows, and so on can contribute to an Effect. One Effect is chosen as the Objective of the Optimization
    """

    def __init__(
        self,
        label: str,
        unit: str,
        description: str,
        meta_data: Optional[Dict] = None,
        is_standard: bool = False,
        is_objective: bool = False,
        specific_share_to_other_effects_operation: Optional['TemporalEffectsUser'] = None,
        specific_share_to_other_effects_invest: Optional['NonTemporalEffectsUser'] = None,
        minimum_operation: Optional[Scalar] = None,
        maximum_operation: Optional[Scalar] = None,
        minimum_invest: Optional[Scalar] = None,
        maximum_invest: Optional[Scalar] = None,
        minimum_operation_per_hour: Optional[TemporalDataUser] = None,
        maximum_operation_per_hour: Optional[TemporalDataUser] = None,
        minimum_total: Optional[Scalar] = None,
        maximum_total: Optional[Scalar] = None,
    ):
        """
        Args:
            label: The label of the Element. Used to identify it in the FlowSystem
            unit: The unit of effect, i.g. €, kg_CO2, kWh_primaryEnergy
            description: The long name
            is_standard: true, if Standard-Effect (for direct input of value without effect (alternatively to dict)) , else false
            is_objective: true, if optimization target
            specific_share_to_other_effects_operation: {effectType: TS, ...}, i.g. 180 €/t_CO2, input as {costs: 180}, optional
                share to other effects (only operation)
            specific_share_to_other_effects_invest: {effectType: TS, ...}, i.g. 180 €/t_CO2, input as {costs: 180}, optional
                share to other effects (only invest).
            minimum_operation: minimal sum (only operation) of the effect.
            maximum_operation: maximal sum (nur operation) of the effect.
            minimum_operation_per_hour: max. value per hour (only operation) of effect (=sum of all effect-shares) for each timestep!
            maximum_operation_per_hour:  min. value per hour (only operation) of effect (=sum of all effect-shares) for each timestep!
            minimum_invest: minimal sum (only invest) of the effect
            maximum_invest: maximal sum (only invest) of the effect
            minimum_total: min sum of effect (invest+operation).
            maximum_total: max sum of effect (invest+operation).
            meta_data: used to store more information about the Element. Is not used internally, but saved in the results. Only use python native types.
        """
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
        self.minimum_operation_per_hour: TemporalDataUser = minimum_operation_per_hour
        self.maximum_operation_per_hour: TemporalDataUser = maximum_operation_per_hour
        self.minimum_invest = minimum_invest
        self.maximum_invest = maximum_invest
        self.minimum_total = minimum_total
        self.maximum_total = maximum_total

    def transform_data(self, flow_system: 'FlowSystem'):
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
            f'{self.label_full}|minimum_operation', self.minimum_operation, has_time_dim=False
        )
        self.maximum_operation = flow_system.fit_to_model_coords(
            f'{self.label_full}|maximum_operation', self.maximum_operation, has_time_dim=False
        )
        self.minimum_invest = flow_system.fit_to_model_coords(
            f'{self.label_full}|minimum_invest', self.minimum_invest, has_time_dim=False
        )
        self.maximum_invest = flow_system.fit_to_model_coords(
            f'{self.label_full}|maximum_invest', self.maximum_invest, has_time_dim=False
        )
        self.minimum_total = flow_system.fit_to_model_coords(
            f'{self.label_full}|minimum_total', self.minimum_total, has_time_dim=False,
        )
        self.maximum_total = flow_system.fit_to_model_coords(
            f'{self.label_full}|maximum_total', self.maximum_total, has_time_dim=False
        )
        self.specific_share_to_other_effects_invest = flow_system.fit_effects_to_model_coords(
            f'{self.label_full}|invest->', self.specific_share_to_other_effects_invest, 'invest',
            has_time_dim=False
        )

    def create_model(self, model: SystemModel) -> 'EffectModel':
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
        self.total: Optional[linopy.Variable] = None
        self.invest: ShareAllocationModel = self.add(
            ShareAllocationModel(
                model=self._model,
                has_time_dim=False,
                has_scenario_dim=True,
                label_of_element=self.label_of_element,
                label='invest',
                label_full=f'{self.label_full}(invest)',
                total_max=self.element.maximum_invest,
                total_min=self.element.minimum_invest,
            )
        )

        self.operation: ShareAllocationModel = self.add(
            ShareAllocationModel(
                model=self._model,
                has_time_dim=True,
                has_scenario_dim=True,
                label_of_element=self.label_of_element,
                label='operation',
                label_full=f'{self.label_full}(operation)',
                total_max=self.element.maximum_operation,
                total_min=self.element.minimum_operation,
                min_per_hour=self.element.minimum_operation_per_hour
                if self.element.minimum_operation_per_hour is not None
                else None,
                max_per_hour=self.element.maximum_operation_per_hour
                if self.element.maximum_operation_per_hour is not None
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
                coords=self._model.get_coords(time_dim=False),
                name=f'{self.label_full}|total',
            ),
            'total',
        )

        self.add(
            self._model.add_constraints(
                self.total == self.operation.total + self.invest.total, name=f'{self.label_full}|total'
            ),
            'total',
        )


TemporalEffectsUser = Union[TemporalDataUser, Dict[str, TemporalDataUser]]  # User-specified Shares to Effects
""" This datatype is used to define a temporal share to an effect by a certain attribute. """

NonTemporalEffectsUser = Union[Scalar, Dict[str, Scalar]]  # User-specified Shares to Effects
""" This datatype is used to define a scalar share to an effect by a certain attribute. """

TemporalEffects = Dict[str, TemporalData]  # User-specified Shares to Effects
""" This datatype is used internally to handle temporal shares to an effect. """

NonTemporalEffects = Dict[str, Scalar]
""" This datatype is used internally to handle scalar shares to an effect. """

EffectExpr = Dict[str, linopy.LinearExpression]  # Used to create Shares


class EffectCollection:
    """
    Handling all Effects
    """

    def __init__(self, *effects: List[Effect]):
        self._effects = {}
        self._standard_effect: Optional[Effect] = None
        self._objective_effect: Optional[Effect] = None

        self.model: Optional[EffectCollectionModel] = None
        self.add_effects(*effects)

    def create_model(self, model: SystemModel) -> 'EffectCollectionModel':
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

    def create_effect_values_dict(
        self,
        effect_values_user: Union[NonTemporalEffectsUser, TemporalEffectsUser]
    ) -> Optional[Dict[str, Union[Scalar, TemporalDataUser]]]:
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

        def get_effect_label(eff: Union[Effect, str]) -> str:
            """Temporary function to get the label of an effect and warn for deprecation"""
            if isinstance(eff, Effect):
                warnings.warn(
                    f'The use of effect objects when specifying EffectValues is deprecated. '
                    f'Use the label of the effect instead. Used effect: {eff.label_full}',
                    UserWarning,
                    stacklevel=2,
                )
                return eff.label_full
            elif eff is None:
                return self.standard_effect.label_full
            else:
                return eff

        if effect_values_user is None:
            return None
        if isinstance(effect_values_user, dict):
            return {get_effect_label(effect): value for effect, value in effect_values_user.items()}
        return {self.standard_effect.label_full: effect_values_user}

    def _plausibility_checks(self) -> None:
        # Check circular loops in effects:
        operation, invest = self.calculate_effect_share_factors()

        operation_cycles = detect_cycles(tuples_to_adjacency_list([key for key in operation]))
        invest_cycles = detect_cycles(tuples_to_adjacency_list([key for key in invest]))

        if operation_cycles:
            cycle_str = "\n".join([" -> ".join(cycle) for cycle in operation_cycles])
            raise ValueError(f'Error: circular operation-shares detected:\n{cycle_str}')

        if invest_cycles:
            cycle_str = "\n".join([" -> ".join(cycle) for cycle in invest_cycles])
            raise ValueError(f'Error: circular invest-shares detected:\n{cycle_str}')

    def __getitem__(self, effect: Union[str, Effect]) -> 'Effect':
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

    def __contains__(self, item: Union[str, 'Effect']) -> bool:
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
    def effects(self) -> Dict[str, Effect]:
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

    def calculate_effect_share_factors(self) -> Tuple[
        Dict[Tuple[str, str], xr.DataArray],
        Dict[Tuple[str, str], xr.DataArray],
    ]:
        shares_invest = {}
        for name, effect in self.effects.items():
            if effect.specific_share_to_other_effects_invest:
                shares_invest[name] = {
                    target: data
                    for target, data in effect.specific_share_to_other_effects_invest.items()
                }
        shares_invest = calculate_all_conversion_paths(shares_invest)

        shares_operation = {}
        for name, effect in self.effects.items():
            if effect.specific_share_to_other_effects_operation:
                shares_operation[name] = {
                    target: data
                    for target, data in effect.specific_share_to_other_effects_operation.items()
                }
        shares_operation = calculate_all_conversion_paths(shares_operation)

        return shares_operation, shares_invest


class EffectCollectionModel(Model):
    """
    Handling all Effects
    """

    def __init__(self, model: SystemModel, effects: EffectCollection):
        super().__init__(model, label_of_element='Effects')
        self.effects = effects
        self.penalty: Optional[ShareAllocationModel] = None

    def add_share_to_effects(
        self,
        name: str,
        expressions: EffectExpr,
        target: Literal['operation', 'invest'],
    ) -> None:
        for effect, expression in expressions.items():
            if target == 'operation':
                self.effects[effect].model.operation.add_share(
                    name,
                    expression,
                    has_time_dim=True,
                    has_scenario_dim=True,
                )
            elif target == 'invest':
                self.effects[effect].model.invest.add_share(
                    name,
                    expression,
                    has_time_dim=False,
                    has_scenario_dim=True,
                )
            else:
                raise ValueError(f'Target {target} not supported!')

    def add_share_to_penalty(self, name: str, expression: linopy.LinearExpression) -> None:
        if expression.ndim != 0:
            raise TypeError(f'Penalty shares must be scalar expressions! ({expression.ndim=})')
        self.penalty.add_share(name, expression, has_time_dim=False, has_scenario_dim=False)

    def do_modeling(self):
        for effect in self.effects:
            effect.create_model(self._model)
        self.penalty = self.add(
            ShareAllocationModel(self._model, has_time_dim=False, has_scenario_dim=False, label_of_element='Penalty')
        )
        for model in [effect.model for effect in self.effects] + [self.penalty]:
            model.do_modeling()

        self._add_share_between_effects()

        self._model.add_objective(
            (self.effects.objective_effect.model.total * self._model.scenario_weights).sum()
            + self.penalty.total.sum()
        )

    def _add_share_between_effects(self):
        for origin_effect in self.effects:
            # 1. operation: -> hier sind es Zeitreihen (share_TS)
            for target_effect, time_series in origin_effect.specific_share_to_other_effects_operation.items():
                self.effects[target_effect].model.operation.add_share(
                    origin_effect.model.operation.label_full,
                    origin_effect.model.operation.total_per_timestep * time_series,
                    has_time_dim=True,
                    has_scenario_dim=True,
                )
            # 2. invest:    -> hier ist es Scalar (share)
            for target_effect, factor in origin_effect.specific_share_to_other_effects_invest.items():
                self.effects[target_effect].model.invest.add_share(
                    origin_effect.model.invest.label_full,
                    origin_effect.model.invest.total * factor,
                    has_time_dim=False,
                    has_scenario_dim=True,
                )


def calculate_all_conversion_paths(
        conversion_dict: Dict[str, Dict[str, xr.DataArray]],
) -> Dict[Tuple[str, str], xr.DataArray]:
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
    result = {key: value if isinstance(value, xr.DataArray) else xr.DataArray(value)
              for key, value in result.items()}

    return result


def detect_cycles(graph: Dict[str, List[str]]) -> List[List[str]]:
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


def tuples_to_adjacency_list(edges: List[Tuple[str, str]]) -> Dict[str, List[str]]:
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
