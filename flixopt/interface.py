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
    """Define piecewise linear conversion relationships between multiple flows.

    This class models complex conversion processes where the relationship between
    input and output flows changes at different operating points, such as:

    - Variable efficiency equipment (heat pumps, engines, turbines)
    - Multi-stage chemical processes with different conversion rates
    - Equipment with discrete operating modes
    - Systems with capacity constraints and thresholds

    Args:
        piecewises: Dictionary mapping flow labels to their Piecewise conversion functions.
            Keys are flow names (e.g., 'electricity_in', 'heat_out', 'fuel_consumed').
            Values are Piecewise objects defining conversion factors at different operating points.
            All Piecewise objects must have the same number of pieces and compatible domains
            to ensure consistent conversion relationships across operating ranges.

    Note:
        Special modeling features:

        - **Gaps**: Express forbidden operating ranges by creating non-contiguous pieces.
          Example: `[(0,50), (100,200)]` - cannot operate between 50-100 units
        - **Points**: Express discrete operating points using pieces with identical start/end.
          Example: `[(50,50), (100,100)]` - can only operate at exactly 50 or 100 units

    Examples:
        Heat pump with variable COP (Coefficient of Performance):

        ```python
        PiecewiseConversion(
            {
                'electricity_in': Piecewise(
                    [
                        Piece(0, 10),  # Low load: 0-10 kW electricity
                        Piece(10, 25),  # High load: 10-25 kW electricity
                    ]
                ),
                'heat_out': Piecewise(
                    [
                        Piece(0, 35),  # Low load COP=3.5: 0-35 kW heat output
                        Piece(35, 75),  # High load COP=3.0: 35-75 kW heat output
                    ]
                ),
            }
        )
        # At 15 kW electricity input → 52.5 kW heat output (interpolated)
        ```

        Engine with fuel consumption and emissions:

        ```python
        PiecewiseConversion(
            {
                'fuel_input': Piecewise(
                    [
                        Piece(5, 15),  # Part load: 5-15 L/h fuel
                        Piece(15, 30),  # Full load: 15-30 L/h fuel
                    ]
                ),
                'power_output': Piecewise(
                    [
                        Piece(10, 25),  # Part load: 10-25 kW output
                        Piece(25, 45),  # Full load: 25-45 kW output
                    ]
                ),
                'co2_emissions': Piecewise(
                    [
                        Piece(12, 35),  # Part load: 12-35 kg/h CO2
                        Piece(35, 78),  # Full load: 35-78 kg/h CO2
                    ]
                ),
            }
        )
        ```

        Discrete operating modes (on/off equipment):

        ```python
        PiecewiseConversion(
            {
                'electricity_in': Piecewise(
                    [
                        Piece(0, 0),  # Off mode: no consumption
                        Piece(20, 20),  # On mode: fixed 20 kW consumption
                    ]
                ),
                'cooling_out': Piecewise(
                    [
                        Piece(0, 0),  # Off mode: no cooling
                        Piece(60, 60),  # On mode: fixed 60 kW cooling
                    ]
                ),
            }
        )
        ```

        Equipment with forbidden operating range:

        ```python
        PiecewiseConversion(
            {
                'steam_input': Piecewise(
                    [
                        Piece(0, 100),  # Low pressure operation
                        Piece(200, 500),  # High pressure (gap: 100-200)
                    ]
                ),
                'power_output': Piecewise(
                    [
                        Piece(0, 80),  # Low efficiency at low pressure
                        Piece(180, 400),  # High efficiency at high pressure
                    ]
                ),
            }
        )
        ```

        Multi-product chemical reactor:

        ```python
        fx.PiecewiseConversion(
            {
                'feedstock': fx.Piecewise(
                    [
                        fx.Piece(10, 50),  # Small batch: 10-50 kg/h
                        fx.Piece(50, 200),  # Large batch: 50-200 kg/h
                    ]
                ),
                'product_A': fx.Piecewise(
                    [
                        fx.Piece(7, 32),  # Small batch yield: 70%
                        fx.Piece(32, 140),  # Large batch yield: 70%
                    ]
                ),
                'product_B': fx.Piecewise(
                    [
                        fx.Piece(2, 12),  # Small batch: 20% to product B
                        fx.Piece(12, 45),  # Large batch: better selectivity
                    ]
                ),
                'waste': fx.Piecewise(
                    [
                        fx.Piece(1, 6),  # Small batch waste: 10%
                        fx.Piece(6, 15),  # Large batch waste: 7.5%
                    ]
                ),
            }
        )
        ```

    Common Use Cases:
        - Heat pumps/chillers: COP varies with load and ambient conditions
        - Power plants: Heat rate curves showing fuel efficiency vs output
        - Chemical reactors: Conversion rates and selectivity vs throughput
        - Compressors/pumps: Power consumption vs flow rate
        - Multi-stage processes: Different conversion rates per stage
        - Equipment with minimum loads: Cannot operate below threshold
        - Batch processes: Discrete production campaigns

    """

    def __init__(self, piecewises: Dict[str, Piecewise]):
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
class PiecewiseEffectsPerFlowHour(Interface):
    """
    Define piecewise linear relationships between flow rate and various effects (costs, emissions, etc.).

    This class models situations where the relationship between flow rate and effects changes at
    different flow rate levels, such as:
    - Pump efficiency curves across operating ranges
    - Emission factors that vary with operating levels
    - Capacity-dependent transportation costs
    - Decision between different operating modes or suppliers
    - Optional equipment activation with minimum flow requirements

    Args:
        piecewise_flow_rate: `Piecewise` defining the valid flow rate segments.
            Each Piece represents a linear segment with (min_flow, max_flow) bounds.

        piecewise_shares: Dictionary mapping effect names to their `Piecewise`.
            Keys are effect names (e.g., 'Costs', 'CO2', 'Maintenance').
            Values are `Piecewise` objects defining the absolute effect values (not rates/prices).

            ⚠️  IMPORTANT: Values represent total effect amounts, not unit rates.
            For a flow rate of X, the effect value is interpolated from the `Piecewise`.
            This is NOT flow_rate × unit_price (which would be non-linear).

    Behavior:
        - If the first piece doesn't start at zero, flow rate is automatically bounded
          by piecewise_flow_rate (when OnOffParameters are not used)
        - Each segment represents a linear relationship within that flow rate range
        - Effects are interpolated linearly within each piece
        - All `Piece`s of the different `Piecewise`s at index i are active at the same time
        - A decision whether to utilize the effect can be modeled by defining multiple Pieces for the same flow rate range

    Examples:
        # Tiered cost structure with increasing rates
        PiecewiseEffectsPerFlowHour(
            piecewise_flow_rate=Piecewise([
                Piece(0, 50),    # Low flow segment: 0-50 units
                Piece(50, 200)   # High flow segment: 50-200 units
            ]),
            piecewise_shares={
                'Costs': Piecewise([
                    Piece(0, 500),     # At flow=0: cost=0, at flow=50: cost=500
                    Piece(500, 2000)   # At flow=50: cost=500, at flow=200: cost=2000
                ]),
                'CO2': Piecewise([
                    Piece(0, 100),     # At flow=0: CO2=0, at flow=50: CO2=100
                    Piece(100, 800)    # At flow=50: CO2=100, at flow=200: CO2=800
                ])
            }
        )

        # Decision between two suppliers with overlapping flow ranges
        PiecewiseEffectsPerFlowHour(
            piecewise_flow_rate=Piecewise([
                Piece(0, 100),     # Supplier A: 0-100 units
                Piece(50, 150)     # Supplier B: 50-150 units (overlaps with A)
            ]),
            piecewise_shares={
                'Costs': Piecewise([
                    Piece(0, 800),     # Supplier A: cheaper for low volumes
                    Piece(400, 1200)   # Supplier B: better rates for high volumes
                ])
            }
        )
        # Flow range 50-100: Optimizer chooses between suppliers based on cost

        # Optional equipment with minimum activation threshold
        PiecewiseEffectsPerFlowHour(
            piecewise_flow_rate=Piecewise([
                Piece(0, 0),       # Equipment off: no flow
                Piece(20, 100)     # Equipment on: minimum 20 units required
            ]),
            piecewise_shares={
                'Costs': Piecewise([
                    Piece(0, 0),       # No cost when off
                    Piece(200, 800)    # Fixed startup cost + variable cost
                ]),
                'CO2': Piecewise([
                    Piece(0, 0),       # No CO2 when off
                    Piece(50, 300)     # Decreasing CO2 per fuel burn with higher power
                ])
            }
        )
        # Decision: Either flow=0 (off) or flow≥20 (on with minimum threshold)

        # Equipment efficiency curve (although this might be better modeled as a Flow rather than an effect)
        PiecewiseEffectsPerFlowHour(
            piecewise_flow_rate=Piecewise([Piece(10, 100)]),  # Min 10, max 100 units
            piecewise_shares={
                'PowerConsumption': Piecewise([Piece(50, 800)])  # Non-linear efficiency
            }
        )

    """

    def __init__(self, piecewise_flow_rate: Piecewise, piecewise_shares: Dict[str, Piecewise]):
        self.piecewise_flow_rate = piecewise_flow_rate
        self.piecewise_shares = piecewise_shares

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str):
        self.piecewise_flow_rate.transform_data(flow_system, f'{name_prefix}|PiecewiseEffectsPerFlowHour|origin')
        for name, piecewise in self.piecewise_shares.items():
           piecewise.transform_data(flow_system, f'{name_prefix}|PiecewiseEffectsPerFlowHour|{name}')


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
