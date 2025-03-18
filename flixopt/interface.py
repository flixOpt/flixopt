"""
This module contains classes to collect Parameters for the Investment and OnOff decisions.
These are tightly connected to features.py
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from .config import CONFIG
from .core import Numeric, Numeric_TS, Skalar
from .structure import Element, Interface

if TYPE_CHECKING:
    from .effects import Effect, EffectTimeSeries, EffectValues, EffectValuesInvest

logger = logging.getLogger('flixOpt')


class InvestParameters(Interface):
    """
    collects arguments for invest-stuff
    """

    def __init__(
        self,
        fixed_size: Optional[Union[int, float]] = None,
        minimum_size: Union[int, float] = 0,  # TODO: Use EPSILON?
        maximum_size: Optional[Union[int, float]] = None,
        optional: bool = True,  # Investition ist weglassbar
        fix_effects: Union[Dict, int, float] = None,
        specific_effects: Union[Dict, int, float] = None,  # costs per Flow-Unit/Storage-Size/...
        effects_in_segments: Optional[
            Tuple[List[Tuple[Skalar, Skalar]], Dict['Effect', List[Tuple[Skalar, Skalar]]]]
        ] = None,
        divest_effects: Union[Dict, int, float] = None,
    ):
        """
        Parameters
        ----------
        fix_effects : None or scalar, optional
            Fixed investment costs if invested.
            (Attention: Annualize costs to chosen period!)
        divest_effects : None or scalar, optional
            Fixed divestment costs (if not invested, e.g., demolition costs or contractual penalty).
        fixed_size : int, float, optional
            Determines if the investment size is fixed.
        optional : bool, optional
            If True, investment is not forced.
        specific_effects : scalar or Dict[Effect: Union[int, float, np.ndarray], optional
            Specific costs, e.g., in €/kW_nominal or €/m²_nominal.
            Example: {costs: 3, CO2: 0.3} with costs and CO2 representing an Object of class Effect
            (Attention: Annualize costs to chosen period!)
        effects_in_segments : list or List[ List[Union[int,float]], Dict[cEffecType: Union[List[Union[int,float]], optional
            Linear relation in segments [invest_segments, cost_segments].
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
            (Args 'specific_effects' and 'fix_effects' can be used in parallel to InvestsizeSegments)
        minimum_size : scalar
            Min nominal value (only if: size_is_fixed = False).
        maximum_size : scalar, Optional
            Max nominal value (only if: size_is_fixed = False).
        """
        self.fix_effects: EffectValuesInvest = fix_effects or {}
        self.divest_effects: EffectValuesInvest = divest_effects or {}
        self.fixed_size = fixed_size
        self.optional = optional
        self.specific_effects: EffectValuesInvest = specific_effects or {}
        self.effects_in_segments = effects_in_segments
        self._minimum_size = minimum_size
        self._maximum_size = maximum_size or CONFIG.modeling.BIG  # default maximum

    def transform_data(self):
        from .effects import as_effect_dict

        self.fix_effects = as_effect_dict(self.fix_effects)
        self.divest_effects = as_effect_dict(self.divest_effects)
        self.specific_effects = as_effect_dict(self.specific_effects)

    @property
    def minimum_size(self):
        return self.fixed_size or self._minimum_size

    @property
    def maximum_size(self):
        return self.fixed_size or self._maximum_size


class OnOffParameters(Interface):
    def __init__(
        self,
        effects_per_switch_on: Union[Dict, Numeric] = None,
        effects_per_running_hour: Union[Dict, Numeric] = None,
        on_hours_total_min: Optional[int] = None,
        on_hours_total_max: Optional[int] = None,
        consecutive_on_hours_min: Optional[Numeric] = None,
        consecutive_on_hours_max: Optional[Numeric] = None,
        consecutive_off_hours_min: Optional[Numeric] = None,
        consecutive_off_hours_max: Optional[Numeric] = None,
        switch_on_total_max: Optional[int] = None,
        force_switch_on: bool = False,
    ):
        """
        on_off_parameters class for modeling on and off state of an Element.
        If no parameters are given, the default is to create a binary variable for the on state
        without further constraints or effects and a variable for the total on hours.

        Parameters
        ----------
        effects_per_switch_on : scalar, array, TimeSeriesData, optional
            cost of one switch from off (var_on=0) to on (var_on=1),
            unit i.g. in Euro
        effects_per_running_hour : scalar or TS, optional
            costs for operating, i.g. in € per hour
        on_hours_total_min : scalar, optional
            min. overall sum of operating hours.
        on_hours_total_max : scalar, optional
            max. overall sum of operating hours.
        consecutive_on_hours_min : scalar, optional
            min sum of operating hours in one piece
            (last on-time period of timeseries is not checked and can be shorter)
        consecutive_on_hours_max : scalar, optional
            max sum of operating hours in one piece
        consecutive_off_hours_min : scalar, optional
            min sum of non-operating hours in one piece
            (last off-time period of timeseries is not checked and can be shorter)
        consecutive_off_hours_max : scalar, optional
            max sum of non-operating hours in one piece
        switch_on_total_max : integer, optional
            max nr of switchOn operations
        force_switch_on : bool
            force creation of switch on variable, even if there is no switch_on_total_max
        """
        self.effects_per_switch_on: Union[EffectValues, EffectTimeSeries] = effects_per_switch_on or {}
        self.effects_per_running_hour: Union[EffectValues, EffectTimeSeries] = effects_per_running_hour or {}
        self.on_hours_total_min: Skalar = on_hours_total_min
        self.on_hours_total_max: Skalar = on_hours_total_max
        self.consecutive_on_hours_min: Numeric_TS = consecutive_on_hours_min
        self.consecutive_on_hours_max: Numeric_TS = consecutive_on_hours_max
        self.consecutive_off_hours_min: Numeric_TS = consecutive_off_hours_min
        self.consecutive_off_hours_max: Numeric_TS = consecutive_off_hours_max
        self.switch_on_total_max: Skalar = switch_on_total_max
        self.force_switch_on: bool = force_switch_on

    def transform_data(self, owner: 'Element'):
        from .effects import effect_values_to_time_series
        from .structure import _create_time_series

        self.effects_per_switch_on = effect_values_to_time_series('per_switch_on', self.effects_per_switch_on, owner)
        self.effects_per_running_hour = effect_values_to_time_series(
            'per_running_hour', self.effects_per_running_hour, owner
        )
        self.consecutive_on_hours_min = _create_time_series(
            'consecutive_on_hours_min', self.consecutive_on_hours_min, owner
        )
        self.consecutive_on_hours_max = _create_time_series(
            'consecutive_on_hours_max', self.consecutive_on_hours_max, owner
        )
        self.consecutive_off_hours_min = _create_time_series(
            'consecutive_off_hours_min', self.consecutive_off_hours_min, owner
        )
        self.consecutive_off_hours_max = _create_time_series(
            'consecutive_off_hours_max', self.consecutive_off_hours_max, owner
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
