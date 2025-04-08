"""
This module contains classes to collect Parameters for the Investment and OnOff decisions.
These are tightly connected to features.py
"""

import logging
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Union, Literal

from .config import CONFIG
from .core import NumericDataTS, Scalar, ScenarioData, TimestepData
from .structure import Interface, register_class_for_io

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from .effects import EffectValuesUserScenario, EffectValuesUserTimestep
    from .flow_system import FlowSystem


logger = logging.getLogger('flixopt')


@register_class_for_io
class Piece(Interface):
    def __init__(self, start: TimestepData, end: TimestepData):
        """
        Define a Piece, which is part of a Piecewise object.

        Args:
            start: The x-values of the piece.
            end: The end of the piece.
        """
        self.start = start
        self.end = end
        self.has_time_dim = False

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str):
        self.start = flow_system.create_time_series(
            name=f'{name_prefix}|start', data=self.start, has_time_dim=self.has_time_dim, has_scenario_dim=True
        )
        self.end = flow_system.create_time_series(
            name=f'{name_prefix}|end', data=self.end, has_time_dim=self.has_time_dim, has_scenario_dim=True
        )


@register_class_for_io
class Piecewise(Interface):
    def __init__(self, pieces: List[Piece]):
        """
        Define a Piecewise, consisting of a list of Pieces.

        Args:
            pieces: The pieces of the piecewise.
        """
        self.pieces = pieces
        self._has_time_dim = False

    @property
    def has_time_dim(self):
        return self._has_time_dim

    @has_time_dim.setter
    def has_time_dim(self, value):
        self._has_time_dim = value
        for piece in self.pieces:
            piece.has_time_dim = value

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, index) -> Piece:
        return self.pieces[index]  # Enables indexing like piecewise[i]

    def __iter__(self) -> Iterator[Piece]:
        return iter(self.pieces)  # Enables iteration like for piece in piecewise: ...

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str):
        for i, piece in enumerate(self.pieces):
            piece.transform_data(flow_system, f'{name_prefix}|Piece{i}')


@register_class_for_io
class PiecewiseConversion(Interface):
    def __init__(self, piecewises: Dict[str, Piecewise]):
        """
        Define a piecewise conversion between multiple Flows.
        --> "gaps" can be expressed by a piece not starting at the end of the prior piece: [(1,3), (4,5)]
        --> "points" can expressed as piece with same begin and end: [(3,3), (4,4)]

        Args:
            piecewises: Dict of Piecewises defining the conversion factors. flow labels as keys, piecewise as values
        """
        self.piecewises = piecewises
        self._has_time_dim = True
        self.has_time_dim = True  # Inital propagation

    @property
    def has_time_dim(self):
        return self._has_time_dim

    @has_time_dim.setter
    def has_time_dim(self, value):
        self._has_time_dim = value
        for piecewise in self.piecewises.values():
            piecewise.has_time_dim = value

    def items(self):
        return self.piecewises.items()

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str):
        for name, piecewise in self.piecewises.items():
            piecewise.transform_data(flow_system, f'{name_prefix}|{name}')


@register_class_for_io
class PiecewiseEffects(Interface):
    def __init__(self, piecewise_origin: Piecewise, piecewise_shares: Dict[str, Piecewise]):
        """
        Define piecewise effects related to a variable.

        Args:
            piecewise_origin: Piecewise of the related variable
            piecewise_shares: Piecewise defining the shares to different Effects
        """
        self.piecewise_origin = piecewise_origin
        self.piecewise_shares = piecewise_shares
        self._has_time_dim = False
        self.has_time_dim = False  # Inital propagation

    @property
    def has_time_dim(self):
        return self._has_time_dim

    @has_time_dim.setter
    def has_time_dim(self, value):
        self._has_time_dim = value
        self.piecewise_origin.has_time_dim = value
        for piecewise in self.piecewise_shares.values():
            piecewise.has_time_dim = value

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str):
        self.piecewise_origin.transform_data(flow_system, f'{name_prefix}|PiecewiseEffects|origin')
        for effect, piecewise in self.piecewise_shares.items():
            piecewise.transform_data(flow_system, f'{name_prefix}|PiecewiseEffects|{effect}')


@register_class_for_io
class InvestParameters(Interface):
    """
    collects arguments for invest-stuff
    """

    def __init__(
        self,
        fixed_size: Optional[ScenarioData] = None,
        minimum_size: ScenarioData = 0,  # TODO: Use EPSILON?
        maximum_size: Optional[ScenarioData] = None,
        optional: bool = True,  # Investition ist weglassbar
        fix_effects: Optional['EffectValuesUserScenario'] = None,
        specific_effects: Optional['EffectValuesUserScenario'] = None,  # costs per Flow-Unit/Storage-Size/...
        piecewise_effects: Optional[PiecewiseEffects] = None,
        divest_effects: Optional['EffectValuesUserScenario'] = None,
        size_per_scenario: Literal['equal', 'individual', 'increment_once'] = 'equal',
    ):
        """
        Args:
            fix_effects: Fixed investment costs if invested. (Attention: Annualize costs to chosen period!)
            divest_effects: Fixed divestment costs (if not invested, e.g., demolition costs or contractual penalty).
            fixed_size: Determines if the investment size is fixed.
            optional: If True, investment is not forced.
            specific_effects: Specific costs, e.g., in €/kW_nominal or €/m²_nominal.
                Example: {costs: 3, CO2: 0.3} with costs and CO2 representing an Object of class Effect
                (Attention: Annualize costs to chosen period!)
            piecewise_effects: Define the effects of the investment as a piecewise function of the size of the investment.
            minimum_size: Minimum possible size of the investment.
            maximum_size: Maximum possible size of the investment.
            size_per_scenario: How to treat the size in each scenario
                - 'equal': Equalize the size of all scenarios
                - 'individual': Optimize the size of each scenario individually
                - 'increment_once': Allow the size to increase only once. This is useful if the scenarios are related to
                    different periods (years, months). Tune the timing by setting the maximum size to 0 in the first scenarios.
        """
        self.fix_effects: EffectValuesUserScenario = fix_effects if fix_effects is not None else {}
        self.divest_effects: EffectValuesUserScenario = divest_effects if divest_effects is not None else {}
        self.fixed_size = fixed_size
        self.optional = optional
        self.specific_effects: EffectValuesUserScenario = specific_effects if specific_effects is not None else {}
        self.piecewise_effects = piecewise_effects
        self._minimum_size = minimum_size
        self._maximum_size = CONFIG.modeling.BIG if maximum_size is None else maximum_size  # default maximum
        self.size_per_scenario = size_per_scenario

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str):
        self.fix_effects = flow_system.create_effect_time_series(
            label_prefix=name_prefix,
            effect_values=self.fix_effects,
            label_suffix='fix_effects',
            has_time_dim=False,
            has_scenario_dim=True,
        )
        self.divest_effects = flow_system.create_effect_time_series(
            label_prefix=name_prefix,
            effect_values=self.divest_effects,
            label_suffix='divest_effects',
            has_time_dim=False,
            has_scenario_dim=True,
        )
        self.specific_effects = flow_system.create_effect_time_series(
            label_prefix=name_prefix,
            effect_values=self.specific_effects,
            label_suffix='specific_effects',
            has_time_dim=False,
            has_scenario_dim=True,
        )
        if self.piecewise_effects is not None:
            self.piecewise_effects.has_time_dim = False
            self.piecewise_effects.transform_data(flow_system, f'{name_prefix}|PiecewiseEffects')

        self._minimum_size = flow_system.create_time_series(
            f'{name_prefix}|minimum_size', self.minimum_size, has_time_dim=False, has_scenario_dim=True
        )
        self._maximum_size = flow_system.create_time_series(
            f'{name_prefix}|maximum_size', self.maximum_size, has_time_dim=False, has_scenario_dim=True
        )
        if self.fixed_size is not None:
            self.fixed_size = flow_system.create_time_series(
                f'{name_prefix}|fixed_size', self.fixed_size, has_time_dim=False, has_scenario_dim=True
            )

    @property
    def minimum_size(self):
        return self.fixed_size if self.fixed_size is not None else self._minimum_size

    @property
    def maximum_size(self):
        return self.fixed_size if self.fixed_size is not None else self._maximum_size


@register_class_for_io
class OnOffParameters(Interface):
    def __init__(
        self,
        effects_per_switch_on: Optional['EffectValuesUserTimestep'] = None,
        effects_per_running_hour: Optional['EffectValuesUserTimestep'] = None,
        on_hours_total_min: Optional[ScenarioData] = None,
        on_hours_total_max: Optional[ScenarioData] = None,
        consecutive_on_hours_min: Optional[TimestepData] = None,
        consecutive_on_hours_max: Optional[TimestepData] = None,
        consecutive_off_hours_min: Optional[TimestepData] = None,
        consecutive_off_hours_max: Optional[TimestepData] = None,
        switch_on_total_max: Optional[ScenarioData] = None,
        force_switch_on: bool = False,
    ):
        """
        Bundles information about the on and off state of an Element.
        If no parameters are given, the default is to create a binary variable for the on state
        without further constraints or effects and a variable for the total on hours.

        Args:
            effects_per_switch_on: cost of one switch from off (var_on=0) to on (var_on=1),
                unit i.g. in Euro
            effects_per_running_hour: costs for operating, i.g. in € per hour
            on_hours_total_min: min. overall sum of operating hours.
            on_hours_total_max: max. overall sum of operating hours.
            consecutive_on_hours_min: min sum of operating hours in one piece
                (last on-time period of timeseries is not checked and can be shorter)
            consecutive_on_hours_max: max sum of operating hours in one piece
            consecutive_off_hours_min: min sum of non-operating hours in one piece
                (last off-time period of timeseries is not checked and can be shorter)
            consecutive_off_hours_max: max sum of non-operating hours in one piece
            switch_on_total_max: max nr of switchOn operations
            force_switch_on: force creation of switch on variable, even if there is no switch_on_total_max
        """
        self.effects_per_switch_on: EffectValuesUserTimestep = effects_per_switch_on or {}
        self.effects_per_running_hour: EffectValuesUserTimestep = effects_per_running_hour or {}
        self.on_hours_total_min: Scalar = on_hours_total_min
        self.on_hours_total_max: Scalar = on_hours_total_max
        self.consecutive_on_hours_min: NumericDataTS = consecutive_on_hours_min
        self.consecutive_on_hours_max: NumericDataTS = consecutive_on_hours_max
        self.consecutive_off_hours_min: NumericDataTS = consecutive_off_hours_min
        self.consecutive_off_hours_max: NumericDataTS = consecutive_off_hours_max
        self.switch_on_total_max: Scalar = switch_on_total_max
        self.force_switch_on: bool = force_switch_on

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str):
        self.effects_per_switch_on = flow_system.create_effect_time_series(
            name_prefix, self.effects_per_switch_on, 'per_switch_on'
        )
        self.effects_per_running_hour = flow_system.create_effect_time_series(
            name_prefix, self.effects_per_running_hour, 'per_running_hour'
        )
        self.consecutive_on_hours_min = flow_system.create_time_series(
            f'{name_prefix}|consecutive_on_hours_min', self.consecutive_on_hours_min
        )
        self.consecutive_on_hours_max = flow_system.create_time_series(
            f'{name_prefix}|consecutive_on_hours_max', self.consecutive_on_hours_max
        )
        self.consecutive_off_hours_min = flow_system.create_time_series(
            f'{name_prefix}|consecutive_off_hours_min', self.consecutive_off_hours_min
        )
        self.consecutive_off_hours_max = flow_system.create_time_series(
            f'{name_prefix}|consecutive_off_hours_max', self.consecutive_off_hours_max
        )
        self.on_hours_total_max = flow_system.create_time_series(
            f'{name_prefix}|on_hours_total_max', self.on_hours_total_max, has_time_dim=False
        )
        self.on_hours_total_min = flow_system.create_time_series(
            f'{name_prefix}|on_hours_total_min', self.on_hours_total_min, has_time_dim=False
        )
        self.switch_on_total_max = flow_system.create_time_series(
            f'{name_prefix}|switch_on_total_max', self.switch_on_total_max, has_time_dim=False
        )

    @property
    def use_off(self) -> bool:
        """Determines wether the OFF Variable is needed or not"""
        return self.use_consecutive_off_hours

    @property
    def use_consecutive_on_hours(self) -> bool:
        """Determines wether a Variable for consecutive off hours is needed or not"""
        return any(param is not None for param in [self.consecutive_on_hours_min, self.consecutive_on_hours_max])

    @property
    def use_consecutive_off_hours(self) -> bool:
        """Determines wether a Variable for consecutive off hours is needed or not"""
        return any(param is not None for param in [self.consecutive_off_hours_min, self.consecutive_off_hours_max])

    @property
    def use_switch_on(self) -> bool:
        """Determines wether a Variable for SWITCH-ON is needed or not"""
        return (
            any(
                param not in (None, {})
                for param in [
                    self.effects_per_switch_on,
                    self.switch_on_total_max,
                    self.on_hours_total_min,
                    self.on_hours_total_max,
                ]
            )
            or self.force_switch_on
        )
