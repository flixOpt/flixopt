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
    """Define a linear segment within a piecewise linear function.

    This class represents a single linear segment that forms part of a larger
    piecewise linear relationship. Each Piece defines the domain boundaries
    (start and end points) for one segment of the function, enabling the
    modeling of complex non-linear relationships through linear approximations.

    Args:
        start: Values marking the beginning of this linear segment.
            These define the lower bound of the domain where this piece is active.
        end: Values marking the end of this linear segment.
            These define the upper bound of the domain where this piece is active.

    Examples:
        Creating a piece for an efficiency curve segment:

        ```python
        # Represents efficiency from 40% to 80% load
        efficiency_piece = Piece(start=40, end=80)
        ```

        Multiple pieces forming a complete piecewise function:

        ```python
        # Low efficiency at part load (0-50% load)
        low_load_piece = Piece(start=0, end=50)

        # High efficiency at full load (50-100% load)
        full_load_piece = Piece(start=50, end=100)
        ```

        Time-varying piece boundaries:

        ```python
        # Operating range that changes with time)
        time_dependent_piece = Piece(
            start=np.array([10, 20, 30, 25]),  # Minimum capacity by timestep
            end=np.array([80, 100, 90, 70]),  # Maximum capacity by timestep
        )
        ```

    Note:
        Pieces typically "touch" at their boundaries (end of one piece = start of next)
        to ensure continuity in the piecewise function. Gaps between pieces can be
        used to model forbidden operating regions.
        Overlapping Pieces effectively leave the decision on which piece is active open.

    """

    def __init__(self, start: NumericData, end: NumericData):
        self.start = start
        self.end = end

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str):
        self.start = flow_system.create_time_series(f'{name_prefix}|start', self.start)
        self.end = flow_system.create_time_series(f'{name_prefix}|end', self.end)


@register_class_for_io
class Piecewise(Interface):
    """Define a piecewise linear function composed of `Pieces`.

    This class creates complex non-linear relationships by combining multiple
    Piece objects into a single piecewise linear function. Each piece represents
    a different linear segment that applies over a specific domain range,
    allowing accurate approximation of curved relationships through linear
    interpolation between breakpoints.

    Args:
        pieces: List of Piece objects defining the linear segments of the function.
            Pieces should typically be ordered by their domain (start values) and
            may overlap at boundaries to ensure continuity. Gaps between pieces
            can represent forbidden operating regions.

    Examples:
        Heat pump COP (Coefficient of Performance) curve:

        ```python
        cop_curve = Piecewise(
            [
                Piece(start=0, end=25),  # Low ambient temp: poor COP
                Piece(start=25, end=50),  # Moderate temp: better COP
                Piece(start=50, end=75),  # High temp: best COP
            ]
        )
        ```

        Tiered electricity pricing:

        ```python
        electricity_cost = Piecewise(
            [
                Piece(start=0, end=100),  # First 100 kWh: low rate
                Piece(start=100, end=500),  # Next 400 kWh: medium rate
                Piece(start=500, end=1000),  # Above 500 kWh: high rate
            ]
        )
        ```

        Equipment with minimum load and forbidden range:

        ```python
        turbine_operation = Piecewise(
            [
                Piece(start=0, end=0),  # Off state (point)
                Piece(start=40, end=100),  # Operating range (gap 0-40)
            ]
        )
        ```

        Seasonal capacity variation:

        ```python
        seasonal_capacity = Piecewise(
            [
                Piece(start=[10, 15, 20, 12], end=[80, 90, 85, 75]),  # By season
            ]
        )
        ```

    Note:
        The Piecewise class supports standard Python container operations:

        - Length: `len(piecewise)` returns number of pieces
        - Indexing: `piecewise[i]` accesses the i-th piece
        - Iteration: `for piece in piecewise:` loops over all pieces

    Common Use Cases:
        - Power plant heat rate curves: fuel consumption vs electrical output
        - HVAC equipment: capacity and efficiency vs outdoor temperature
        - Industrial processes: conversion efficiency vs throughput
        - Financial modeling: progressive tax rates, bulk pricing discounts
        - Transportation: fuel consumption vs speed, load capacity vs distance
        - Storage systems: charge/discharge efficiency vs state of charge

    """

    def __init__(self, pieces: List[Piece]):
        self.pieces = pieces

    def __len__(self):
        """
        Return the number of Piece segments in this Piecewise container.

        Returns:
            int: Count of contained Piece objects.
        """
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
        """
        Return an iterator over (flow_label, Piecewise) pairs stored in this PiecewiseConversion.

        This is a thin convenience wrapper around the internal mapping and yields the same view
        as dict.items(), where each key is a flow label (str) and each value is a Piecewise.
        """
        return self.piecewises.items()

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str):
        for name, piecewise in self.piecewises.items():
            piecewise.transform_data(flow_system, f'{name_prefix}|{name}')


@register_class_for_io
class PiecewiseEffects(Interface):
    """Define variable-dependent effects using piecewise linear functions.

    This class models complex relationships where the effects (costs, emissions,
    resources) associated with a decision variable change non-linearly based on
    the variable's value. The origin piecewise function defines the primary
    variable's behavior, while the shares define how this variable contributes
    to different system effects at different operating levels.

    This is particularly useful for modeling:
    - Scale-dependent costs (bulk discounts, economies of scale)
    - Load-dependent emissions (efficiency changes with throughput)
    - Capacity-dependent resource consumption
    - Multi-tier pricing structures
    - Performance-dependent environmental impacts

    Args:
        piecewise_origin: Piecewise function defining the behavior of the primary variable
            that drives the effects. This establishes the domain and operating ranges.
        piecewise_shares: Dictionary mapping effect names to their corresponding
            Piecewise functions. Keys are effect identifiers (e.g., 'cost', 'CO2', 'water').
            Values are Piecewise objects defining how much of each effect is generated
            per unit of the origin variable at different operating levels.

    Note:
        **Implementation Status**: This functionality is not yet fully implemented
        for non-scalar shares. Currently limited to scalar effect relationships.

    Examples:
        Manufacturing process with scale-dependent costs and emissions:

        ```python
        # Production volume affects unit costs and emissions
        manufacturing_effects = PiecewiseEffects(
            piecewise_origin=Piecewise(
                [
                    Piece(0, 1000),  # Small scale production
                    Piece(1000, 5000),  # Medium scale production
                    Piece(5000, 10000),  # Large scale production
                ]
            ),
            piecewise_shares={
                'unit_cost': Piecewise(
                    [
                        Piece(50, 45),  # High unit cost at low volume
                        Piece(45, 35),  # Decreasing cost with scale
                        Piece(35, 30),  # Lowest cost at high volume
                    ]
                ),
                'CO2_intensity': Piecewise(
                    [
                        Piece(2.5, 2.0),  # Higher emissions per unit at low efficiency
                        Piece(2.0, 1.5),  # Better efficiency at medium scale
                        Piece(1.5, 1.2),  # Best efficiency at large scale
                    ]
                ),
            },
        )
        ```

        Power plant with load-dependent heat rate and emissions:

        ```python
        power_plant_effects = PiecewiseEffects(
            piecewise_origin=Piecewise(
                [
                    Piece(100, 300),  # Minimum load to rated capacity (MW)
                    Piece(300, 500),  # Overload operation
                ]
            ),
            piecewise_shares={
                'fuel_rate': Piecewise(
                    [
                        Piece(11.5, 10.2),  # Heat rate: BTU/kWh (less efficient at part load)
                        Piece(10.2, 10.8),  # Heat rate increases at overload
                    ]
                ),
                'NOx_rate': Piecewise(
                    [
                        Piece(0.15, 0.12),  # NOx emissions: lb/MWh
                        Piece(0.12, 0.18),  # Higher emissions at overload
                    ]
                ),
            },
        )
        ```

        Tiered water pricing with consumption-dependent rates:

        ```python
        water_pricing = PiecewiseEffects(
            piecewise_origin=Piecewise(
                [
                    Piece(0, 10),  # Basic tier: 0-10 m³/month
                    Piece(10, 50),  # Standard tier: 10-50 m³/month
                    Piece(50, 200),  # High consumption: >50 m³/month
                ]
            ),
            piecewise_shares={
                'cost_per_m3': Piecewise(
                    [
                        Piece(1.20, 1.20),  # Flat rate for basic consumption
                        Piece(2.50, 2.50),  # Higher rate for standard tier
                        Piece(4.00, 4.00),  # Premium rate for high consumption
                    ]
                ),
                'infrastructure_fee': Piecewise(
                    [
                        Piece(0.10, 0.10),  # Low infrastructure impact
                        Piece(0.25, 0.25),  # Medium infrastructure impact
                        Piece(0.50, 0.50),  # High infrastructure impact
                    ]
                ),
            },
        )
        ```

    Common Use Cases:
        - Manufacturing: Scale economies, learning curves, capacity utilization effects
        - Energy systems: Load-dependent efficiency, emissions, and maintenance costs
        - Transportation: Distance-dependent rates, fuel efficiency curves
        - Utilities: Tiered pricing, demand charges, infrastructure costs
        - Environmental modeling: Threshold effects, cumulative impacts
        - Financial instruments: Progressive rates, risk premiums

    """

    def __init__(self, piecewise_origin: Piecewise, piecewise_shares: Dict[str, Piecewise]):
        self.piecewise_origin = piecewise_origin
        self.piecewise_shares = piecewise_shares

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str):
        raise NotImplementedError('PiecewiseEffects is not yet implemented for non scalar shares')
        # self.piecewise_origin.transform_data(flow_system, f'{name_prefix}|PiecewiseEffects|origin')
        # for name, piecewise in self.piecewise_shares.items():
        #    piecewise.transform_data(flow_system, f'{name_prefix}|PiecewiseEffects|{name}')


@register_class_for_io
class InvestParameters(Interface):
    """Define comprehensive investment decision parameters for optimization models.

    This class encapsulates all parameters needed to model investment decisions
    in optimization problems, including sizing constraints, cost structures,
    and operational flexibility. It supports multiple cost modeling approaches
    from simple linear relationships to complex piecewise functions, enabling
    accurate representation of real-world investment economics.

    Investment modeling capabilities include:
    - Fixed vs. continuous sizing decisions
    - Multiple cost components (fixed, variable, piecewise)
    - Optional investments with divestment penalties
    - Technology learning curves and economies of scale
    - Multi-period investment planning with proper annualization

    Args:
        fixed_size: When specified, constrains the investment to exactly this size,
            creating a binary invest/don't-invest decision. When None, allows
            continuous sizing between minimum_size and maximum_size bounds.
        minimum_size: Minimum investment size for continuous sizing decisions.
            Defaults to CONFIG.modeling.EPSILON to avoid numerical issues.
            Ignored when fixed_size is specified.
        maximum_size: Maximum investment size for continuous sizing decisions.
            Defaults to CONFIG.modeling.BIG to represent unlimited capacity.
            Ignored when fixed_size is specified.
        optional: Controls investment optionality. When True (default), the
            optimization can choose not to invest. When False, forces investment
            to occur, useful for mandatory infrastructure or replacement decisions.
        fix_effects: Fixed costs incurred once if the investment is made, regardless
            of the investment size. Typical examples include permitting costs,
            connection fees, or base equipment costs. Dictionary mapping effect
            names to scalar values (e.g., {'cost': 10000, 'CO2': 500}).
            **Important**: Costs must be annualized to the optimization time period.
        specific_effects: Variable costs proportional to investment size, representing
            per-unit costs like €/kW_nominal or €/m²_nominal. Dictionary mapping
            effect names to unit cost values (e.g., {'cost': 1200, 'CO2': 0.5}).
            **Important**: Costs must be annualized to the optimization time period.
        piecewise_effects: Complex non-linear cost relationships using PiecewiseEffects,
            enabling modeling of bulk discounts, technology learning curves, or
            economies of scale. Can be combined with fix_effects and specific_effects.
            **Important**: Costs must be annualized to the optimization time period.
        divest_effects: Costs incurred if the investment is NOT made, such as
            demolition costs for existing equipment, contractual penalties, or
            opportunity costs. Dictionary mapping effect names to scalar values.

    Note:
        **Cost Annualization**: All cost values must be properly annualized to match
        the optimization model's time period. For example, if modeling annual decisions
        but equipment has a 20-year lifetime, divide capital costs by 20 or use
        appropriate discount rates to convert to equivalent annual costs.

    Examples:
        Simple binary investment (solar panels):

        ```python
        solar_investment = InvestParameters(
            fixed_size=100,  # 100 kW system (binary decision)
            optional=True,
            fix_effects={
                'cost': 25000,  # Installation and permitting costs
                'CO2': -50000,  # Avoided emissions over lifetime
            },
            specific_effects={
                'cost': 1200,  # €1200/kW for panels (annualized)
                'CO2': -800,  # kg CO2 avoided per kW annually
            },
        )
        ```

        Flexible sizing with economies of scale:

        ```python
        battery_investment = InvestParameters(
            minimum_size=10,  # Minimum viable system size (kWh)
            maximum_size=1000,  # Maximum installable capacity
            optional=True,
            fix_effects={
                'cost': 5000,  # Grid connection and control system
                'installation_time': 2,  # Days for fixed components
            },
            piecewise_effects=PiecewiseEffects(
                piecewise_origin=Piecewise(
                    [
                        Piece(0, 100),  # Small systems
                        Piece(100, 500),  # Medium systems
                        Piece(500, 1000),  # Large systems
                    ]
                ),
                piecewise_shares={
                    'cost': Piecewise(
                        [
                            Piece(800, 750),  # High cost/kWh for small systems
                            Piece(750, 600),  # Medium cost/kWh
                            Piece(600, 500),  # Bulk discount for large systems
                        ]
                    )
                },
            ),
        )
        ```

        Mandatory replacement with divestment costs:

        ```python
        boiler_replacement = InvestParameters(
            minimum_size=50,
            maximum_size=200,
            optional=True,  # Can choose not to replace
            fix_effects={
                'cost': 15000,  # Installation costs
                'disruption': 3,  # Days of downtime
            },
            specific_effects={
                'cost': 400,  # €400/kW capacity
                'maintenance': 25,  # Annual maintenance per kW
            },
            divest_effects={
                'cost': 8000,  # Demolition if not replaced
                'environmental': 100,  # Disposal fees
            },
        )
        ```

        Multi-technology comparison:

        ```python
        # Gas turbine option
        gas_turbine = InvestParameters(
            fixed_size=50,  # MW
            fix_effects={'cost': 2500000, 'CO2': 1250000},
            specific_effects={'fuel_cost': 45, 'maintenance': 12},
        )

        # Wind farm option
        wind_farm = InvestParameters(
            minimum_size=20,
            maximum_size=100,
            fix_effects={'cost': 1000000, 'CO2': -5000000},
            specific_effects={'cost': 1800000, 'land_use': 0.5},
        )
        ```

        Technology learning curve:

        ```python
        hydrogen_electrolyzer = InvestParameters(
            minimum_size=1,
            maximum_size=50,  # MW
            piecewise_effects=PiecewiseEffects(
                piecewise_origin=Piecewise(
                    [
                        Piece(0, 5),  # Small scale: early adoption
                        Piece(5, 20),  # Medium scale: cost reduction
                        Piece(20, 50),  # Large scale: mature technology
                    ]
                ),
                piecewise_shares={
                    'capex': Piecewise(
                        [
                            Piece(2000, 1800),  # Learning reduces costs
                            Piece(1800, 1400),  # Continued cost reduction
                            Piece(1400, 1200),  # Technology maturity
                        ]
                    ),
                    'efficiency': Piecewise(
                        [
                            Piece(65, 68),  # Improving efficiency
                            Piece(68, 72),  # with scale and experience
                            Piece(72, 75),  # Best efficiency at scale
                        ]
                    ),
                },
            ),
        )
        ```

    Common Use Cases:
        - Power generation: Plant sizing, technology selection, retrofit decisions
        - Industrial equipment: Capacity expansion, efficiency upgrades, replacements
        - Infrastructure: Network expansion, facility construction, system upgrades
        - Energy storage: Battery sizing, pumped hydro, compressed air systems
        - Transportation: Fleet expansion, charging infrastructure, modal shifts
        - Buildings: HVAC systems, insulation upgrades, renewable integration

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
    """Define binary operation constraints and costs for equipment with discrete states.

    This class models equipment that operates in discrete on/off states rather than
    continuous operation, capturing the operational constraints and economic impacts
    of binary decisions. It addresses real-world equipment behavior including
    minimum run times, startup costs, cycling limitations, and maintenance schedules.

    Common applications include:
    - Power plants with minimum load requirements and startup costs
    - Industrial equipment with batch processes or discrete operating modes
    - HVAC systems with thermostat control and equipment cycling
    - Process equipment with startup/shutdown sequences
    - Backup generators and emergency systems
    - Maintenance scheduling and equipment availability

    Args:
        effects_per_switch_on: Costs or impacts incurred for each transition from
            off state (var_on=0) to on state (var_on=1). Represents startup costs,
            wear and tear, or other switching impacts. Dictionary mapping effect
            names to values (e.g., {'cost': 500, 'maintenance_hours': 2}).
        effects_per_running_hour: Ongoing costs or impacts while equipment operates
            in the on state. Includes fuel costs, labor, consumables, or emissions.
            Dictionary mapping effect names to hourly values (e.g., {'fuel_cost': 45}).
        on_hours_total_min: Minimum total operating hours across the entire time horizon.
            Ensures equipment meets minimum utilization requirements or contractual
            obligations (e.g., power purchase agreements, maintenance schedules).
        on_hours_total_max: Maximum total operating hours across the entire time horizon.
            Limits equipment usage due to maintenance schedules, fuel availability,
            environmental permits, or equipment lifetime constraints.
        consecutive_on_hours_min: Minimum continuous operating duration once started.
            Models minimum run times due to thermal constraints, process stability,
            or efficiency considerations. Can be time-varying to reflect different
            constraints across the planning horizon.
        consecutive_on_hours_max: Maximum continuous operating duration in one campaign.
            Models mandatory maintenance intervals, process batch sizes, or
            equipment thermal limits requiring periodic shutdowns.
        consecutive_off_hours_min: Minimum continuous shutdown duration between operations.
            Models cooling periods, maintenance requirements, or process constraints
            that prevent immediate restart after shutdown.
        consecutive_off_hours_max: Maximum continuous shutdown duration before mandatory
            restart. Models equipment preservation, process stability, or contractual
            requirements for minimum activity levels.
        switch_on_total_max: Maximum number of startup operations across the time horizon.
            Limits equipment cycling to reduce wear, maintenance costs, or comply
            with operational constraints (e.g., grid stability requirements).
        force_switch_on: When True, creates switch-on variables even without explicit
            switch_on_total_max constraint. Useful for tracking or reporting startup
            events without enforcing limits.

    Note:
        **Time Series Boundary Handling**: The final time period constraints for
        consecutive_on_hours_min/max and consecutive_off_hours_min/max are not
        enforced, allowing the optimization to end with ongoing campaigns that
        may be shorter than the specified minimums or longer than maximums.

    Examples:
        Combined cycle power plant with startup costs and minimum run time:

        ```python
        power_plant_operation = OnOffParameters(
            effects_per_switch_on={
                'startup_cost': 25000,  # €25,000 per startup
                'startup_fuel': 150,  # GJ natural gas for startup
                'startup_time': 4,  # Hours to reach full output
                'maintenance_impact': 0.1,  # Fractional life consumption
            },
            effects_per_running_hour={
                'fixed_om': 125,  # Fixed O&M costs while running
                'auxiliary_power': 2.5,  # MW parasitic loads
            },
            consecutive_on_hours_min=8,  # Minimum 8-hour run once started
            consecutive_off_hours_min=4,  # Minimum 4-hour cooling period
            on_hours_total_max=6000,  # Annual operating limit
        )
        ```

        Industrial batch process with cycling limits:

        ```python
        batch_reactor = OnOffParameters(
            effects_per_switch_on={
                'setup_cost': 1500,  # Labor and materials for startup
                'catalyst_consumption': 5,  # kg catalyst per batch
                'cleaning_chemicals': 200,  # L cleaning solution
            },
            effects_per_running_hour={
                'steam': 2.5,  # t/h process steam
                'electricity': 150,  # kWh electrical load
                'cooling_water': 50,  # m³/h cooling water
            },
            consecutive_on_hours_min=12,  # Minimum batch size (12 hours)
            consecutive_on_hours_max=24,  # Maximum batch size (24 hours)
            consecutive_off_hours_min=6,  # Cleaning and setup time
            switch_on_total_max=200,  # Maximum 200 batches per year
            on_hours_total_max=4000,  # Maximum production time
        )
        ```

        HVAC system with thermostat control and maintenance:

        ```python
        hvac_operation = OnOffParameters(
            effects_per_switch_on={
                'compressor_wear': 0.5,  # Hours of compressor life per start
                'inrush_current': 15,  # kW peak demand on startup
            },
            effects_per_running_hour={
                'electricity': 25,  # kW electrical consumption
                'maintenance': 0.12,  # €/hour maintenance reserve
            },
            consecutive_on_hours_min=1,  # Minimum 1-hour run to avoid cycling
            consecutive_off_hours_min=0.5,  # 30-minute minimum off time
            switch_on_total_max=2000,  # Limit cycling for compressor life
            on_hours_total_min=2000,  # Minimum operation for humidity control
            on_hours_total_max=5000,  # Maximum operation for energy budget
        )
        ```

        Backup generator with testing and maintenance requirements:

        ```python
        backup_generator = OnOffParameters(
            effects_per_switch_on={
                'fuel_priming': 50,  # L diesel for system priming
                'wear_factor': 1.0,  # Start cycles impact on maintenance
                'testing_labor': 2,  # Hours technician time per test
            },
            effects_per_running_hour={
                'fuel_consumption': 180,  # L/h diesel consumption
                'emissions_permit': 15,  # € emissions allowance cost
                'noise_penalty': 25,  # € noise compliance cost
            },
            consecutive_on_hours_min=0.5,  # Minimum test duration (30 min)
            consecutive_off_hours_max=720,  # Maximum 30 days between tests
            switch_on_total_max=52,  # Weekly testing limit
            on_hours_total_min=26,  # Minimum annual testing (0.5h × 52)
            on_hours_total_max=200,  # Maximum runtime (emergencies + tests)
        )
        ```

        Peak shaving battery with cycling degradation:

        ```python
        battery_cycling = OnOffParameters(
            effects_per_switch_on={
                'cycle_degradation': 0.01,  # % capacity loss per cycle
                'inverter_startup': 0.5,  # kWh losses during startup
            },
            effects_per_running_hour={
                'standby_losses': 2,  # kW standby consumption
                'cooling': 5,  # kW thermal management
                'inverter_losses': 8,  # kW conversion losses
            },
            consecutive_on_hours_min=1,  # Minimum discharge duration
            consecutive_on_hours_max=4,  # Maximum continuous discharge
            consecutive_off_hours_min=1,  # Minimum rest between cycles
            switch_on_total_max=365,  # Daily cycling limit
            force_switch_on=True,  # Track all cycling events
        )
        ```

    Common Use Cases:
        - Power generation: Thermal plant cycling, renewable curtailment, grid services
        - Industrial processes: Batch production, maintenance scheduling, equipment rotation
        - Buildings: HVAC control, lighting systems, elevator operations
        - Transportation: Fleet management, charging infrastructure, maintenance windows
        - Storage systems: Battery cycling, pumped hydro, compressed air systems
        - Emergency equipment: Backup generators, safety systems, emergency lighting

    """

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
