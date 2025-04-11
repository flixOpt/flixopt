"""
This module contains classes to collect Parameters for the Investment and OnOff decisions.
These are tightly connected to features.py
"""

import logging
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Union

from .config import CONFIG
from .core import NumericData, NumericDataTS, Scalar
from .structure import Interface, register_class_for_io

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from .effects import EffectValuesUser, EffectValuesUserScalar
    from .flow_system import FlowSystem


logger = logging.getLogger('flixopt')


@register_class_for_io
class Piece(Interface):
    def __init__(self, start: NumericData, end: NumericData):
        """
        Define a Piece, which is part of a Piecewise object.

        Args:
            start: The x-values of the piece.
            end: The end of the piece.
        """
        self.start = start
        self.end = end

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str):
        self.start = flow_system.create_time_series(f'{name_prefix}|start', self.start)
        self.end = flow_system.create_time_series(f'{name_prefix}|end', self.end)


@register_class_for_io
class Piecewise(Interface):
    def __init__(self, pieces: List[Piece]):
        """
        Define a Piecewise, consisting of a list of Pieces.

        Args:
            pieces: The pieces of the piecewise.
        """
        self.pieces = pieces

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

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str):
        raise NotImplementedError('PiecewiseEffects is not yet implemented for non scalar shares')
        # self.piecewise_origin.transform_data(flow_system, f'{name_prefix}|PiecewiseEffects|origin')
        # for name, piecewise in self.piecewise_shares.items():
        #    piecewise.transform_data(flow_system, f'{name_prefix}|PiecewiseEffects|{name}')


@register_class_for_io
class InvestParameters(Interface):
    """
    collects arguments for invest-stuff
    """

    def __init__(
        self,
        fixed_size: Optional[Union[int, float]] = None,
        minimum_size: Optional[Union[int, float]] = None,
        maximum_size: Optional[Union[int, float]] = None,
        optional: bool = True,  # Investition ist weglassbar
        fix_effects: Optional['EffectValuesUserScalar'] = None,
        specific_effects: Optional['EffectValuesUserScalar'] = None,  # costs per Flow-Unit/Storage-Size/...
        piecewise_effects: Optional[PiecewiseEffects] = None,
        divest_effects: Optional['EffectValuesUserScalar'] = None,
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
            piecewise_effects: Linear piecewise relation [invest_pieces, cost_pieces].
                Example 1:
                    [           [5, 25, 25, 100],       # size in kW
                     {costs:    [50,250,250,800],       # €
                      PE:       [5, 25, 25, 100]        # kWh_PrimaryEnergy
                      }
                    ]
                Example 2 (if only standard-effect):
                    [   [5, 25, 25, 100],  # kW # size in kW
                        [50,250,250,800]        # value for standart effect, typically €
                     ]  # €
                (Attention: Annualize costs to chosen period!)
                (Args 'specific_effects' and 'fix_effects' can be used in parallel to Investsizepieces)
            minimum_size: Min nominal value (only if: size_is_fixed = False). Defaults to CONFIG.modeling.EPSILON.
            maximum_size: Max nominal value (only if: size_is_fixed = False). Defaults to CONFIG.modeling.BIG.
        """
        self.fix_effects: EffectValuesUser = fix_effects or {}
        self.divest_effects: EffectValuesUser = divest_effects or {}
        self.fixed_size = fixed_size
        self.optional = optional
        self.specific_effects: EffectValuesUser = specific_effects or {}
        self.piecewise_effects = piecewise_effects
        self._minimum_size = minimum_size if minimum_size is not None else CONFIG.modeling.EPSILON
        self._maximum_size = maximum_size if maximum_size is not None else CONFIG.modeling.BIG  # default maximum

    def transform_data(self, flow_system: 'FlowSystem'):
        self.fix_effects = flow_system.effects.create_effect_values_dict(self.fix_effects)
        self.divest_effects = flow_system.effects.create_effect_values_dict(self.divest_effects)
        self.specific_effects = flow_system.effects.create_effect_values_dict(self.specific_effects)

    @property
    def minimum_size(self):
        return self.fixed_size or self._minimum_size

    @property
    def maximum_size(self):
        return self.fixed_size or self._maximum_size


@register_class_for_io
class OnOffParameters(Interface):
    def __init__(
        self,
        effects_per_switch_on: Optional['EffectValuesUser'] = None,
        effects_per_running_hour: Optional['EffectValuesUser'] = None,
        on_hours_total_min: Optional[int] = None,
        on_hours_total_max: Optional[int] = None,
        consecutive_on_hours_min: Optional[NumericData] = None,
        consecutive_on_hours_max: Optional[NumericData] = None,
        consecutive_off_hours_min: Optional[NumericData] = None,
        consecutive_off_hours_max: Optional[NumericData] = None,
        switch_on_total_max: Optional[int] = None,
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
        self.effects_per_switch_on: EffectValuesUser = effects_per_switch_on or {}
        self.effects_per_running_hour: EffectValuesUser = effects_per_running_hour or {}
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
