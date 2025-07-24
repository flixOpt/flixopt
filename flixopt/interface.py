"""
This module contains classes to collect Parameters for the Investment and OnOff decisions.
These are tightly connected to features.py
"""

import logging
from typing import TYPE_CHECKING, Dict, Iterator, List, Literal, Optional, Union

from .config import CONFIG
from .core import NonTemporalData, NonTemporalDataUser, Scalar, TemporalDataUser
from .structure import Interface, register_class_for_io

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from .effects import NonTemporalEffectsUser, TemporalEffectsUser
    from .flow_system import FlowSystem


logger = logging.getLogger('flixopt')


@register_class_for_io
class Piece(Interface):
    def __init__(self, start: TemporalDataUser, end: TemporalDataUser):
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
        self.start = flow_system.fit_to_model_coords(f'{name_prefix}|start', self.start, has_time_dim=self.has_time_dim)
        self.end = flow_system.fit_to_model_coords(f'{name_prefix}|end', self.end, has_time_dim=self.has_time_dim)


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
        fixed_size: Optional[NonTemporalDataUser] = None,
        minimum_size: Optional[NonTemporalDataUser] = None,
        maximum_size: Optional[NonTemporalDataUser] = None,
        optional: bool = True,  # Investition ist weglassbar
        fix_effects: Optional['NonTemporalEffectsUser'] = None,
        specific_effects: Optional['NonTemporalEffectsUser'] = None,  # costs per Flow-Unit/Storage-Size/...
        piecewise_effects: Optional[PiecewiseEffects] = None,
        divest_effects: Optional['NonTemporalEffectsUser'] = None,
        investment_scenarios: Optional[Union[Literal['individual'], List[Union[int, str]]]] = None,
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
            investment_scenarios: For which scenarios to optimize the size for.
                - 'individual': Optimize the size of each scenario individually
                - List of scenario names: Optimize the size for the passed scenario names (equal size in all). All other scenarios will have the size 0.
                - None: Equals to a list of all scenarios (default)
        """
        self.fix_effects: 'NonTemporalEffectsUser' = fix_effects if fix_effects is not None else {}
        self.divest_effects: 'NonTemporalEffectsUser' = divest_effects if divest_effects is not None else {}
        self.fixed_size = fixed_size
        self.optional = optional
        self.specific_effects: 'NonTemporalEffectsUser' = specific_effects if specific_effects is not None else {}
        self.piecewise_effects = piecewise_effects
        self.minimum_size = minimum_size if minimum_size is not None else CONFIG.modeling.EPSILON
        self.maximum_size = maximum_size if maximum_size is not None else CONFIG.modeling.BIG  # default maximum
        self.investment_scenarios = investment_scenarios

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str):
        self._plausibility_checks(flow_system)
        self.fix_effects = flow_system.fit_effects_to_model_coords(
            label_prefix=name_prefix,
            effect_values=self.fix_effects,
            label_suffix='fix_effects',
            has_time_dim=False,
        )
        self.divest_effects = flow_system.fit_effects_to_model_coords(
            label_prefix=name_prefix,
            effect_values=self.divest_effects,
            label_suffix='divest_effects',
            has_time_dim=False,
        )
        self.specific_effects = flow_system.fit_effects_to_model_coords(
            label_prefix=name_prefix,
            effect_values=self.specific_effects,
            label_suffix='specific_effects',
            has_time_dim=False,
        )
        if self.piecewise_effects is not None:
            self.piecewise_effects.has_time_dim = False
            self.piecewise_effects.transform_data(flow_system, f'{name_prefix}|PiecewiseEffects')

        self.minimum_size = flow_system.fit_to_model_coords(
            f'{name_prefix}|minimum_size', self.minimum_size, has_time_dim=False
        )
        self.maximum_size = flow_system.fit_to_model_coords(
            f'{name_prefix}|maximum_size', self.maximum_size, has_time_dim=False
        )
        if self.fixed_size is not None:
            self.fixed_size = flow_system.fit_to_model_coords(
                f'{name_prefix}|fixed_size', self.fixed_size, has_time_dim=False
            )

    def _plausibility_checks(self, flow_system):
        if isinstance(self.investment_scenarios, list):
            if not set(self.investment_scenarios).issubset(flow_system.scenarios):
                raise ValueError(
                    f'Some scenarios in investment_scenarios are not present in the time_series_collection: '
                    f'{set(self.investment_scenarios) - set(flow_system.scenarios)}'
                )
        if self.investment_scenarios is not None:
            if not self.optional:
                if self.minimum_size is not None or self.fixed_size is not None:
                    logger.warning(
                        'When using investment_scenarios, minimum_size and fixed_size should only ne used if optional is True.'
                        'Otherwise the investment cannot be 0 incertain scenarios while being non-zero in others.'
                    )

    @property
    def minimum_or_fixed_size(self) -> NonTemporalData:
        return self.fixed_size if self.fixed_size is not None else self.minimum_size

    @property
    def maximum_or_fixed_size(self) -> NonTemporalData:
        return self.fixed_size if self.fixed_size is not None else self.maximum_size


@register_class_for_io
class OnOffParameters(Interface):
    def __init__(
        self,
        effects_per_switch_on: Optional['TemporalEffectsUser'] = None,
        effects_per_running_hour: Optional['TemporalEffectsUser'] = None,
        on_hours_total_min: Optional[int] = None,
        on_hours_total_max: Optional[int] = None,
        consecutive_on_hours_min: Optional[TemporalDataUser] = None,
        consecutive_on_hours_max: Optional[TemporalDataUser] = None,
        consecutive_off_hours_min: Optional[TemporalDataUser] = None,
        consecutive_off_hours_max: Optional[TemporalDataUser] = None,
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
        self.effects_per_switch_on: 'TemporalEffectsUser' = (
            effects_per_switch_on if effects_per_switch_on is not None else {}
        )
        self.effects_per_running_hour: 'TemporalEffectsUser' = (
            effects_per_running_hour if effects_per_running_hour is not None else {}
        )
        self.on_hours_total_min: Scalar = on_hours_total_min
        self.on_hours_total_max: Scalar = on_hours_total_max
        self.consecutive_on_hours_min: TemporalDataUser = consecutive_on_hours_min
        self.consecutive_on_hours_max: TemporalDataUser = consecutive_on_hours_max
        self.consecutive_off_hours_min: TemporalDataUser = consecutive_off_hours_min
        self.consecutive_off_hours_max: TemporalDataUser = consecutive_off_hours_max
        self.switch_on_total_max: Scalar = switch_on_total_max
        self.force_switch_on: bool = force_switch_on

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str):
        self.effects_per_switch_on = flow_system.fit_effects_to_model_coords(
            name_prefix, self.effects_per_switch_on, 'per_switch_on'
        )
        self.effects_per_running_hour = flow_system.fit_effects_to_model_coords(
            name_prefix, self.effects_per_running_hour, 'per_running_hour'
        )
        self.consecutive_on_hours_min = flow_system.fit_to_model_coords(
            f'{name_prefix}|consecutive_on_hours_min', self.consecutive_on_hours_min
        )
        self.consecutive_on_hours_max = flow_system.fit_to_model_coords(
            f'{name_prefix}|consecutive_on_hours_max', self.consecutive_on_hours_max
        )
        self.consecutive_off_hours_min = flow_system.fit_to_model_coords(
            f'{name_prefix}|consecutive_off_hours_min', self.consecutive_off_hours_min
        )
        self.consecutive_off_hours_max = flow_system.fit_to_model_coords(
            f'{name_prefix}|consecutive_off_hours_max', self.consecutive_off_hours_max
        )
        self.on_hours_total_max = flow_system.fit_to_model_coords(
            f'{name_prefix}|on_hours_total_max', self.on_hours_total_max, has_time_dim=False
        )
        self.on_hours_total_min = flow_system.fit_to_model_coords(
            f'{name_prefix}|on_hours_total_min', self.on_hours_total_min, has_time_dim=False
        )
        self.switch_on_total_max = flow_system.fit_to_model_coords(
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
        if self.force_switch_on:
            return True

        return any(
            param is not None and param != {}
            for param in [
                self.effects_per_switch_on,
                self.switch_on_total_max,
            ]
        )
