"""
This module contains classes to collect Parameters for the Investment and Status decisions.
These are tightly connected to features.py
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import xarray as xr

from .config import CONFIG
from .structure import Interface, register_class_for_io
from .types import Bool_PS, Numeric_PS, PeriodicData, PeriodicEffectsUser

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from collections.abc import Iterator

    from .flow_system import FlowSystem
    from .types import Effect_PS, Effect_TPS, Numeric_TPS

# Backwards compatibility alias
PeriodicDataUser = Numeric_PS

logger = logging.getLogger('flixopt')


@register_class_for_io
class Piece(Interface):
    """Define a single linear segment with specified domain boundaries.

    This class represents one linear segment that will be combined with other
    pieces to form complete piecewise linear functions. Each piece defines
    a domain interval [start, end] where a linear relationship applies.

    Args:
        start: Lower bound of the domain interval for this linear segment.
            Can be scalar values or time series arrays for time-varying boundaries.
        end: Upper bound of the domain interval for this linear segment.
            Can be scalar values or time series arrays for time-varying boundaries.

    Examples:
        Basic piece for equipment efficiency curve:

        ```python
        # Single segment from 40% to 80% load
        efficiency_segment = Piece(start=40, end=80)
        ```

        Piece with time-varying boundaries:

        ```python
        # Capacity limits that change seasonally
        seasonal_piece = Piece(
            start=np.array([10, 20, 30, 25]),  # Minimum capacity by season
            end=np.array([80, 100, 90, 70]),  # Maximum capacity by season
        )
        ```

        Fixed operating point (start equals end):

        ```python
        # Equipment that operates at exactly 50 MW
        fixed_output = Piece(start=50, end=50)
        ```

    Note:
        Individual pieces are building blocks that gain meaning when combined
        into Piecewise functions. See the Piecewise class for information about
        how pieces interact and relate to each other.

    """

    def __init__(self, start: Numeric_TPS, end: Numeric_TPS):
        self.start = start
        self.end = end
        self.has_time_dim = False

    def transform_data(self, name_prefix: str = '') -> None:
        dims = None if self.has_time_dim else ['period', 'scenario']
        self.start = self._fit_coords(f'{name_prefix}|start', self.start, dims=dims)
        self.end = self._fit_coords(f'{name_prefix}|end', self.end, dims=dims)


@register_class_for_io
class Piecewise(Interface):
    """
    Define a Piecewise, consisting of a list of Pieces.

    Args:
        pieces: list of Piece objects defining the linear segments. The arrangement
            and relationships between pieces determine the function behavior:
            - Touching pieces (end of one = start of next) ensure continuity
            - Gaps between pieces create forbidden regions
            - Overlapping pieces provide an extra choice for the optimizer

    Piece Relationship Patterns:
        **Touching Pieces (Continuous Function)**:
        Pieces that share boundary points create smooth, continuous functions
        without gaps or overlaps.

        **Gaps Between Pieces (Forbidden Regions)**:
        Non-contiguous pieces with gaps represent forbidden regions.
        For example minimum load requirements or safety zones.

        **Overlapping Pieces (Flexible Operation)**:
        Pieces with overlapping domains provide optimization flexibility,
        allowing the solver to choose which segment to operate in.

    Examples:
        Continuous efficiency curve (touching pieces):

        ```python
        efficiency_curve = Piecewise(
            [
                Piece(start=0, end=25),  # Low load: 0-25 MW
                Piece(start=25, end=75),  # Medium load: 25-75 MW (touches at 25)
                Piece(start=75, end=100),  # High load: 75-100 MW (touches at 75)
            ]
        )
        ```

        Equipment with forbidden operating range (gap):

        ```python
        turbine_operation = Piecewise(
            [
                Piece(start=0, end=0),  # Off state (point operation)
                Piece(start=40, end=100),  # Operating range (gap: 0-40 forbidden)
            ]
        )
        ```

        Flexible operation with overlapping options:

        ```python
        flexible_operation = Piecewise(
            [
                Piece(start=20, end=60),  # Standard efficiency mode
                Piece(start=50, end=90),  # High efficiency mode (overlap: 50-60)
            ]
        )
        ```

        Tiered pricing structure:

        ```python
        electricity_pricing = Piecewise(
            [
                Piece(start=0, end=100),  # Tier 1: 0-100 kWh
                Piece(start=100, end=500),  # Tier 2: 100-500 kWh
                Piece(start=500, end=1000),  # Tier 3: 500-1000 kWh
            ]
        )
        ```

        Seasonal capacity variation:

        ```python
        seasonal_capacity = Piecewise(
            [
                Piece(start=[10, 15, 20, 12], end=[80, 90, 85, 75]),  # Varies by time
            ]
        )
        ```

    Container Operations:
        The Piecewise class supports standard Python container operations:

        ```python
        piecewise = Piecewise([piece1, piece2, piece3])

        len(piecewise)  # Returns number of pieces (3)
        piecewise[0]  # Access first piece
        for piece in piecewise:  # Iterate over all pieces
            print(piece.start, piece.end)
        ```

    Validation Considerations:
        - Pieces are typically ordered by their start values
        - Check for unintended gaps that might create infeasible regions
        - Consider whether overlaps provide desired flexibility or create ambiguity
        - Ensure time-varying pieces have consistent dimensions

    Common Use Cases:
        - Power plants: Heat rate curves, efficiency vs load, emissions profiles
        - HVAC systems: COP vs temperature, capacity vs conditions
        - Industrial processes: Conversion rates vs throughput, quality vs speed
        - Financial modeling: Tiered rates, progressive taxes, bulk discounts
        - Transportation: Fuel efficiency curves, capacity vs speed
        - Storage systems: Efficiency vs state of charge, power vs energy
        - Renewable energy: Output vs weather conditions, curtailment strategies

    """

    def __init__(self, pieces: list[Piece]):
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

    def _set_flow_system(self, flow_system) -> None:
        """Propagate flow_system reference to nested Piece objects."""
        super()._set_flow_system(flow_system)
        for piece in self.pieces:
            piece._set_flow_system(flow_system)

    def transform_data(self, name_prefix: str = '') -> None:
        for i, piece in enumerate(self.pieces):
            piece.transform_data(f'{name_prefix}|Piece{i}')


@register_class_for_io
class PiecewiseConversion(Interface):
    """Define coordinated piecewise linear relationships between multiple flows.

    This class models conversion processes where multiple flows (inputs, outputs,
    auxiliaries) have synchronized piecewise relationships. All flows change
    together based on the same operating point, enabling accurate modeling of
    complex equipment with variable performance characteristics.

    Multi-Flow Coordination:
        All piecewise functions must have matching piece structures (same number
        of pieces with compatible domains) to ensure synchronized operation.
        When the equipment operates at a given point, ALL flows scale proportionally
        within their respective pieces.

    Mathematical Formulation:
        See the complete mathematical model in the documentation:
        [Piecewise](../user-guide/mathematical-notation/features/Piecewise.md)

    Args:
        piecewises: Dictionary mapping flow labels to their Piecewise functions.
            Keys are flow identifiers (e.g., 'electricity_in', 'heat_out', 'fuel_consumed').
            Values are Piecewise objects that define each flow's behavior.
            **Critical Requirement**: All Piecewise objects must have the same
            number of pieces with compatible domains to ensure consistent operation.

    Operating Point Coordination:
        When equipment operates at any point within a piece, all flows scale
        proportionally within their corresponding pieces. This ensures realistic
        equipment behavior where efficiency, consumption, and production rates
        all change together.

    Examples:
        Heat pump with coordinated efficiency changes:

        ```python
        heat_pump_pc = PiecewiseConversion(
            {
                'electricity_in': Piecewise(
                    [
                        Piece(0, 10),  # Low load: 0-10 kW electricity
                        Piece(10, 25),  # High load: 10-25 kW electricity
                    ]
                ),
                'heat_out': Piecewise(
                    [
                        Piece(0, 35),  # Low load COP=3.5: 0-35 kW heat
                        Piece(35, 75),  # High load COP=3.0: 35-75 kW heat
                    ]
                ),
                'cooling_water': Piecewise(
                    [
                        Piece(0, 2.5),  # Low load: 0-2.5 m³/h cooling
                        Piece(2.5, 6),  # High load: 2.5-6 m³/h cooling
                    ]
                ),
            }
        )
        # At 15 kW electricity → 52.5 kW heat + 3.75 m³/h cooling water
        ```

        Combined cycle power plant with synchronized flows:

        ```python
        power_plant_pc = PiecewiseConversion(
            {
                'natural_gas': Piecewise(
                    [
                        Piece(150, 300),  # Part load: 150-300 MW_th fuel
                        Piece(300, 500),  # Full load: 300-500 MW_th fuel
                    ]
                ),
                'electricity': Piecewise(
                    [
                        Piece(60, 135),  # Part load: 60-135 MW_e (45% efficiency)
                        Piece(135, 250),  # Full load: 135-250 MW_e (50% efficiency)
                    ]
                ),
                'steam_export': Piecewise(
                    [
                        Piece(20, 35),  # Part load: 20-35 MW_th steam
                        Piece(35, 50),  # Full load: 35-50 MW_th steam
                    ]
                ),
                'co2_emissions': Piecewise(
                    [
                        Piece(30, 60),  # Part load: 30-60 t/h CO2
                        Piece(60, 100),  # Full load: 60-100 t/h CO2
                    ]
                ),
            }
        )
        ```

        Chemical reactor with multiple products and waste:

        ```python
        reactor_pc = PiecewiseConversion(
            {
                'feedstock': Piecewise(
                    [
                        Piece(10, 50),  # Small batch: 10-50 kg/h
                        Piece(50, 200),  # Large batch: 50-200 kg/h
                    ]
                ),
                'product_A': Piecewise(
                    [
                        Piece(7, 35),  # Small batch: 70% yield
                        Piece(35, 140),  # Large batch: 70% yield
                    ]
                ),
                'product_B': Piecewise(
                    [
                        Piece(2, 10),  # Small batch: 20% yield
                        Piece(10, 45),  # Large batch: 22.5% yield (improved)
                    ]
                ),
                'waste_stream': Piecewise(
                    [
                        Piece(1, 5),  # Small batch: 10% waste
                        Piece(5, 15),  # Large batch: 7.5% waste (efficiency)
                    ]
                ),
            }
        )
        ```

        Equipment with discrete operating modes:

        ```python
        compressor_pc = PiecewiseConversion(
            {
                'electricity': Piecewise(
                    [
                        Piece(0, 0),  # Off mode: no consumption
                        Piece(45, 45),  # Low mode: fixed 45 kW
                        Piece(85, 85),  # High mode: fixed 85 kW
                    ]
                ),
                'compressed_air': Piecewise(
                    [
                        Piece(0, 0),  # Off mode: no production
                        Piece(250, 250),  # Low mode: 250 Nm³/h
                        Piece(500, 500),  # High mode: 500 Nm³/h
                    ]
                ),
            }
        )
        ```

        Equipment with forbidden operating range:

        ```python
        steam_turbine_pc = PiecewiseConversion(
            {
                'steam_in': Piecewise(
                    [
                        Piece(0, 100),  # Low pressure operation
                        Piece(200, 500),  # High pressure (gap: 100-200 forbidden)
                    ]
                ),
                'electricity_out': Piecewise(
                    [
                        Piece(0, 30),  # Low pressure: poor efficiency
                        Piece(80, 220),  # High pressure: good efficiency
                    ]
                ),
                'condensate_out': Piecewise(
                    [
                        Piece(0, 100),  # Low pressure condensate
                        Piece(200, 500),  # High pressure condensate
                    ]
                ),
            }
        )
        ```

    Design Patterns:
        **Forbidden Ranges**: Use gaps between pieces to model equipment that cannot
        operate in certain ranges (e.g., minimum loads, unstable regions).

        **Discrete Modes**: Use pieces with identical start/end values to model
        equipment with fixed operating points (e.g., on/inactive, discrete speeds).

        **Efficiency Changes**: Coordinate input and output pieces to reflect
        changing conversion efficiency across operating ranges.

    Common Use Cases:
        - Power generation: Multi-fuel plants, cogeneration systems, renewable hybrids
        - HVAC systems: Heat pumps, chillers with variable COP and auxiliary loads
        - Industrial processes: Multi-product reactors, separation units, heat exchangers
        - Transportation: Multi-modal systems, hybrid vehicles, charging infrastructure
        - Water treatment: Multi-stage processes with varying energy and chemical needs
        - Energy storage: Systems with efficiency changes and auxiliary power requirements

    """

    def __init__(self, piecewises: dict[str, Piecewise]):
        self.piecewises = piecewises
        self._has_time_dim = True
        self.has_time_dim = True  # Initial propagation

    @property
    def has_time_dim(self):
        return self._has_time_dim

    @has_time_dim.setter
    def has_time_dim(self, value):
        self._has_time_dim = value
        for piecewise in self.piecewises.values():
            piecewise.has_time_dim = value

    def items(self):
        """
        Return an iterator over (flow_label, Piecewise) pairs stored in this PiecewiseConversion.

        This is a thin convenience wrapper around the internal mapping and yields the same view
        as dict.items(), where each key is a flow label (str) and each value is a Piecewise.
        """
        return self.piecewises.items()

    def _set_flow_system(self, flow_system) -> None:
        """Propagate flow_system reference to nested Piecewise objects."""
        super()._set_flow_system(flow_system)
        for piecewise in self.piecewises.values():
            piecewise._set_flow_system(flow_system)

    def transform_data(self, name_prefix: str = '') -> None:
        for name, piecewise in self.piecewises.items():
            piecewise.transform_data(f'{name_prefix}|{name}')


@register_class_for_io
class PiecewiseEffects(Interface):
    """Define how a single decision variable contributes to system effects with piecewise rates.

    This class models situations where a decision variable (the origin) generates
    different types of system effects (costs, emissions, resource consumption) at
    rates that change non-linearly with the variable's operating level. Unlike
    PiecewiseConversion which coordinates multiple flows, PiecewiseEffects focuses
    on how one variable impacts multiple system-wide effects.

    Key Concept - Origin vs. Effects:
        - **Origin**: The primary decision variable (e.g., production level, capacity, size)
        - **Shares**: The amounts which this variable contributes to different system effects

    Relationship to PiecewiseConversion:
        **PiecewiseConversion**: Models synchronized relationships between multiple
        flow variables (e.g., fuel_in, electricity_out, emissions_out all coordinated).

        **PiecewiseEffects**: Models how one variable contributes to system-wide
        effects at variable rates (e.g., production_level → costs, emissions, resources).

    Args:
        piecewise_origin: Piecewise function defining the behavior of the primary
            decision variable. This establishes the operating domain and ranges.
        piecewise_shares: Dictionary mapping effect names to their rate functions.
            Keys are effect identifiers (e.g., 'cost_per_unit', 'CO2_intensity').
            Values are Piecewise objects defining the contribution rate per unit
            of the origin variable at different operating levels.

    Mathematical Relationship:
        For each effect: Total_Effect = Origin_Variable × Share_Rate(Origin_Level)

        This enables modeling of:
        - Economies of scale (decreasing unit costs with volume)
        - Learning curves (improving efficiency with experience)
        - Threshold effects (changing rates at different scales)
        - Progressive pricing (increasing rates with consumption)

    Examples:
        Manufacturing with economies of scale:

        ```python
        production_effects = PiecewiseEffects(
            piecewise_origin=Piecewise(
                [
                    Piece(0, 1000),  # Small scale: 0-1000 units/month
                    Piece(1000, 5000),  # Medium scale: 1000-5000 units/month
                    Piece(5000, 10000),  # Large scale: 5000-10000 units/month
                ]
            ),
            piecewise_shares={
                'unit_cost': Piecewise(
                    [
                        Piece(50, 45),  # €50-45/unit (scale benefits)
                        Piece(45, 35),  # €45-35/unit (bulk materials)
                        Piece(35, 30),  # €35-30/unit (automation benefits)
                    ]
                ),
                'labor_hours': Piecewise(
                    [
                        Piece(2.5, 2.0),  # 2.5-2.0 hours/unit (learning curve)
                        Piece(2.0, 1.5),  # 2.0-1.5 hours/unit (efficiency gains)
                        Piece(1.5, 1.2),  # 1.5-1.2 hours/unit (specialization)
                    ]
                ),
                'CO2_intensity': Piecewise(
                    [
                        Piece(15, 12),  # 15-12 kg CO2/unit (process optimization)
                        Piece(12, 9),  # 12-9 kg CO2/unit (equipment efficiency)
                        Piece(9, 7),  # 9-7 kg CO2/unit (renewable energy)
                    ]
                ),
            },
        )
        ```

        Power generation with load-dependent characteristics:

        ```python
        generator_effects = PiecewiseEffects(
            piecewise_origin=Piecewise(
                [
                    Piece(50, 200),  # Part load operation: 50-200 MW
                    Piece(200, 350),  # Rated operation: 200-350 MW
                    Piece(350, 400),  # Overload operation: 350-400 MW
                ]
            ),
            piecewise_shares={
                'fuel_rate': Piecewise(
                    [
                        Piece(12.0, 10.5),  # Heat rate: 12.0-10.5 GJ/MWh (part load penalty)
                        Piece(10.5, 9.8),  # Heat rate: 10.5-9.8 GJ/MWh (optimal efficiency)
                        Piece(9.8, 11.2),  # Heat rate: 9.8-11.2 GJ/MWh (overload penalty)
                    ]
                ),
                'maintenance_factor': Piecewise(
                    [
                        Piece(0.8, 1.0),  # Low stress operation
                        Piece(1.0, 1.0),  # Design operation
                        Piece(1.0, 1.5),  # High stress operation
                    ]
                ),
                'NOx_rate': Piecewise(
                    [
                        Piece(0.20, 0.15),  # NOx: 0.20-0.15 kg/MWh
                        Piece(0.15, 0.12),  # NOx: 0.15-0.12 kg/MWh (optimal combustion)
                        Piece(0.12, 0.25),  # NOx: 0.12-0.25 kg/MWh (overload penalties)
                    ]
                ),
            },
        )
        ```

        Progressive utility pricing structure:

        ```python
        electricity_billing = PiecewiseEffects(
            piecewise_origin=Piecewise(
                [
                    Piece(0, 200),  # Basic usage: 0-200 kWh/month
                    Piece(200, 800),  # Standard usage: 200-800 kWh/month
                    Piece(800, 2000),  # High usage: 800-2000 kWh/month
                ]
            ),
            piecewise_shares={
                'energy_rate': Piecewise(
                    [
                        Piece(0.12, 0.12),  # Basic rate: €0.12/kWh
                        Piece(0.18, 0.18),  # Standard rate: €0.18/kWh
                        Piece(0.28, 0.28),  # Premium rate: €0.28/kWh
                    ]
                ),
                'carbon_tax': Piecewise(
                    [
                        Piece(0.02, 0.02),  # Low carbon tax: €0.02/kWh
                        Piece(0.03, 0.03),  # Medium carbon tax: €0.03/kWh
                        Piece(0.05, 0.05),  # High carbon tax: €0.05/kWh
                    ]
                ),
            },
        )
        ```

        Data center with capacity-dependent efficiency:

        ```python
        datacenter_effects = PiecewiseEffects(
            piecewise_origin=Piecewise(
                [
                    Piece(100, 500),  # Low utilization: 100-500 servers
                    Piece(500, 2000),  # Medium utilization: 500-2000 servers
                    Piece(2000, 5000),  # High utilization: 2000-5000 servers
                ]
            ),
            piecewise_shares={
                'power_per_server': Piecewise(
                    [
                        Piece(0.8, 0.6),  # 0.8-0.6 kW/server (inefficient cooling)
                        Piece(0.6, 0.4),  # 0.6-0.4 kW/server (optimal efficiency)
                        Piece(0.4, 0.5),  # 0.4-0.5 kW/server (thermal limits)
                    ]
                ),
                'cooling_overhead': Piecewise(
                    [
                        Piece(0.4, 0.3),  # 40%-30% cooling overhead
                        Piece(0.3, 0.2),  # 30%-20% cooling overhead
                        Piece(0.2, 0.25),  # 20%-25% cooling overhead
                    ]
                ),
            },
        )
        ```

    Design Patterns:
        **Economies of Scale**: Decreasing unit costs/impacts with increased scale
        **Learning Curves**: Improving efficiency rates with experience/volume
        **Threshold Effects**: Step changes in rates at specific operating levels
        **Progressive Pricing**: Increasing rates for higher consumption levels
        **Capacity Utilization**: Optimal efficiency at design points, penalties at extremes

    Common Use Cases:
        - Manufacturing: Production scaling, learning effects, quality improvements
        - Energy systems: Generator efficiency curves, renewable capacity factors
        - Logistics: Transportation rates, warehouse utilization, delivery optimization
        - Utilities: Progressive pricing, infrastructure cost allocation
        - Financial services: Risk premiums, transaction fees, volume discounts
        - Environmental modeling: Pollution intensity, resource consumption rates

    """

    def __init__(self, piecewise_origin: Piecewise, piecewise_shares: dict[str, Piecewise]):
        self.piecewise_origin = piecewise_origin
        self.piecewise_shares = piecewise_shares
        self._has_time_dim = False
        self.has_time_dim = False  # Initial propagation

    @property
    def has_time_dim(self):
        return self._has_time_dim

    @has_time_dim.setter
    def has_time_dim(self, value):
        self._has_time_dim = value
        self.piecewise_origin.has_time_dim = value
        for piecewise in self.piecewise_shares.values():
            piecewise.has_time_dim = value

    def _set_flow_system(self, flow_system) -> None:
        """Propagate flow_system reference to nested Piecewise objects."""
        super()._set_flow_system(flow_system)
        self.piecewise_origin._set_flow_system(flow_system)
        for piecewise in self.piecewise_shares.values():
            piecewise._set_flow_system(flow_system)

    def transform_data(self, name_prefix: str = '') -> None:
        self.piecewise_origin.transform_data(f'{name_prefix}|PiecewiseEffects|origin')
        for effect, piecewise in self.piecewise_shares.items():
            piecewise.transform_data(f'{name_prefix}|PiecewiseEffects|{effect}')


class _SizeParameters(Interface):
    """Base class for sizing and investment parameters."""

    def __init__(
        self,
        fixed_size: Numeric_PS | None = None,
        minimum_size: Numeric_PS | None = None,
        maximum_size: Numeric_PS | None = None,
        mandatory: bool | Bool_PS = False,
        effects_of_size: Effect_PS | Numeric_PS | None = None,
        effects_per_size: Effect_PS | Numeric_PS | None = None,
        piecewise_effects_per_size: PiecewiseEffects | None = None,
    ):
        self.effects_of_size: PeriodicEffectsUser = effects_of_size if effects_of_size is not None else {}
        self.fixed_size = fixed_size
        self.mandatory = mandatory
        self.effects_per_size: PeriodicEffectsUser = effects_per_size if effects_per_size is not None else {}
        self.piecewise_effects_per_size = piecewise_effects_per_size
        self.minimum_size = minimum_size if minimum_size is not None else CONFIG.Modeling.epsilon
        self.maximum_size = maximum_size if maximum_size is not None else CONFIG.Modeling.big  # default maximum

    def transform_data(self, flow_system: FlowSystem, name_prefix: str = '') -> None:
        self.effects_of_size = flow_system.fit_effects_to_model_coords(
            label_prefix=name_prefix,
            effect_values=self.effects_of_size,
            label_suffix='effects_of_size',
            dims=['period', 'scenario'],
        )
        self.effects_per_size = flow_system.fit_effects_to_model_coords(
            label_prefix=name_prefix,
            effect_values=self.effects_per_size,
            label_suffix='effects_per_size',
            dims=['period', 'scenario'],
        )

        if self.piecewise_effects_per_size is not None:
            self.piecewise_effects_per_size.has_time_dim = False
            self.piecewise_effects_per_size.transform_data(flow_system, f'{name_prefix}|PiecewiseEffects')

        self.minimum_size = flow_system.fit_to_model_coords(
            f'{name_prefix}|minimum_size', self.minimum_size, dims=['period', 'scenario']
        )
        self.maximum_size = flow_system.fit_to_model_coords(
            f'{name_prefix}|maximum_size', self.maximum_size, dims=['period', 'scenario']
        )
        self.fixed_size = flow_system.fit_to_model_coords(
            f'{name_prefix}|fixed_size', self.fixed_size, dims=['period', 'scenario']
        )
        self.mandatory = flow_system.fit_to_model_coords(
            f'{name_prefix}|mandatory', self.mandatory, dims=['period', 'scenario']
        )

    @property
    def minimum_or_fixed_size(self) -> PeriodicData:
        return self.fixed_size if self.fixed_size is not None else self.minimum_size

    @property
    def maximum_or_fixed_size(self) -> PeriodicData:
        return self.fixed_size if self.fixed_size is not None else self.maximum_size


@register_class_for_io
class SizingParameters(_SizeParameters):
    """Define investment decision parameters with flexible sizing and effect modeling.

    This class models investment decisions in optimization problems, supporting
    both binary (invest/don't invest) and continuous sizing choices with
    comprehensive cost structures. It enables realistic representation of
    investment economics including fixed costs, scale effects, and divestment penalties.

    Investment Decision Types:
        **Binary Investments**: Fixed size investments creating yes/no decisions
        (e.g., install a specific generator, build a particular facility)

        **Continuous Sizing**: Variable size investments with minimum/maximum bounds
        (e.g., battery capacity from 10-1000 kWh, pipeline diameter optimization)

    Cost Modeling Approaches:
        - **Fixed Effects**: One-time costs independent of size (permits, connections)
        - **Specific Effects**: Linear costs proportional to size (€/kW, €/m²)
        - **Piecewise Effects**: Non-linear relationships (bulk discounts, learning curves)
        - **Divestment Effects**: Penalties for not investing (demolition, opportunity costs)

    Mathematical Formulation:
        See the complete mathematical model in the documentation:
        [SizingParameters](../user-guide/mathematical-notation/features/SizingParameters.md)

    Args:
        fixed_size: Creates binary decision at this exact size. None allows continuous sizing.
        minimum_size: Lower bound for continuous sizing. Default: CONFIG.Modeling.epsilon.
            Ignored if fixed_size is specified.
        maximum_size: Upper bound for continuous sizing. Default: CONFIG.Modeling.big.
            Ignored if fixed_size is specified.
        mandatory: Controls whether investment is required. When True, forces investment
            to occur (useful for mandatory upgrades or replacement decisions).
            When False (default), optimization can choose not to invest.
            With multiple periods, at least one period has to have an investment.
        effects_of_size: Fixed costs if investment is made, regardless of size.
            Dict: {'effect_name': value} (e.g., {'cost': 10000}).
        effects_per_size: Variable costs proportional to size (per-unit costs).
            Dict: {'effect_name': value/unit} (e.g., {'cost': 1200}).
        piecewise_effects_per_size: Non-linear costs using PiecewiseEffects.
            Combinable with effects_of_size and effects_per_size.

    Cost Annualization Requirements:
        All cost values must be properly weighted to match the optimization model's time horizon.
        For long-term investments, the cost values should be annualized to the corresponding operation time (annuity).

        - Use equivalent annual cost (capital cost / equipment lifetime)
        - Apply appropriate discount rates for present value optimizations
        - Account for inflation, escalation, and financing costs

        Example: €1M equipment with 20-year life → €50k/year fixed cost

    Examples:
        Simple binary investment (solar panels):

        ```python
        solar_investment = SizingParameters(
            fixed_size=100,  # 100 kW system (binary decision)
            mandatory=False,  # Investment is optional
            effects_of_size={
                'cost': 25000,  # Installation and permitting costs
                'CO2': -50000,  # Avoided emissions over lifetime
            },
            effects_per_size={
                'cost': 1200,  # €1200/kW for panels (annualized)
                'CO2': -800,  # kg CO2 avoided per kW annually
            },
        )
        ```

        Flexible sizing with economies of scale:

        ```python
        battery_investment = SizingParameters(
            minimum_size=10,  # Minimum viable system size (kWh)
            maximum_size=1000,  # Maximum installable capacity
            mandatory=False,  # Investment is optional
            effects_of_size={
                'cost': 5000,  # Grid connection and control system
                'installation_time': 2,  # Days for fixed components
            },
            piecewise_effects_per_size=PiecewiseEffects(
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

        Mandatory replacement with retirement costs:

        ```python
        boiler_replacement = SizingParameters(
            minimum_size=50,
            maximum_size=200,
            mandatory=False,  # Can choose not to replace
            effects_of_size={
                'cost': 15000,  # Installation costs
                'disruption': 3,  # Days of downtime
            },
            effects_per_size={
                'cost': 400,  # €400/kW capacity
                'maintenance': 25,  # Annual maintenance per kW
            },
        )
        ```

        Multi-technology comparison:

        ```python
        # Gas turbine option
        gas_turbine = SizingParameters(
            fixed_size=50,  # MW
            effects_of_size={'cost': 2500000, 'CO2': 1250000},
            effects_per_size={'fuel_cost': 45, 'maintenance': 12},
        )

        # Wind farm option
        wind_farm = SizingParameters(
            minimum_size=20,
            maximum_size=100,
            effects_of_size={'cost': 1000000, 'CO2': -5000000},
            effects_per_size={'cost': 1800000, 'land_use': 0.5},
        )
        ```

        Technology learning curve:

        ```python
        hydrogen_electrolyzer = SizingParameters(
            minimum_size=1,
            maximum_size=50,  # MW
            piecewise_effects_per_size=PiecewiseEffects(
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

    # SizingParameters now inherits all functionality from _SizeParameters
    # No additional implementation needed


@register_class_for_io
class InvestParameters(SizingParameters):
    def __init__(self, **kwargs):
        warnings.warn(
            'InvestParameters is deprecated, use SizingParameters instead',
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)


InvestmentPeriodData = PeriodicDataUser
"""This datatype is used to define things related to the period of investment."""
InvestmentPeriodDataBool = bool | InvestmentPeriodData
"""This datatype is used to define things with boolean data related to the period of investment."""


@register_class_for_io
class InvestmentParameters(_SizeParameters):
    """Define investment timing parameters with fixed lifetime.

    This class models WHEN to invest with a fixed lifetime duration.
    It includes all sizing parameters (capacity bounds, effects) plus timing controls.

    InvestmentParameters combines both TIMING (when to invest) and CAPACITY (how much)
    aspects in a single class, optimizing when to make an investment that will last
    for a fixed duration.

    Investment Timing Features:
        **Single Investment Decision**: Decide which period to invest in (at most once)
        **Fixed Lifetime**: Investment lasts for a specified number of periods
        **Timing-Dependent Effects**: Effects that vary based on when investment occurs
            (e.g., technology learning curves, time-varying costs)

    Mathematical Formulation:
        See the complete mathematical model in the documentation:
        [InvestmentParameters](../user-guide/mathematical-notation/features/InvestmentParameters.md)

    Args:
        lifetime: REQUIRED. The investment lifetime in number of periods.
            Once invested, the asset operates for this many periods.
        allow_investment: Allow investment in specific periods. Default: True (all periods).
        force_investment: Force investment to occur in a specific period. Default: False.
        effects_of_investment: Effects that depend on when investment occurs.
            Dict mapping effect names to xr.DataArray with dimensions [period, scenario, investment_period].
            These effects can vary by the investment period, enabling modeling of:
            - Technology learning curves (costs decrease over time)
            - Time-varying financing costs
            - Period-specific subsidies or regulations
        effects_of_investment_per_size: Size-dependent effects that also depend on investment period.
            Dict mapping effect names to xr.DataArray with dimensions [period, scenario, investment_period].
        previous_size: Size of existing capacity from previous periods. Default: 0.
        fixed_size: Creates binary decision at this exact size. None allows continuous sizing.
        minimum_size: Lower bound for continuous sizing. Default: CONFIG.Modeling.epsilon.
        maximum_size: Upper bound for continuous sizing. Default: CONFIG.Modeling.big.
        mandatory: Controls whether investment is required. When True, forces investment.
        effects_of_size: Fixed costs if investment is made, regardless of size.
        effects_per_size: Variable costs proportional to size (per-unit costs).
        piecewise_effects_per_size: Non-linear costs using PiecewiseEffects.

    Examples:
        Basic investment timing:

        ```python
        timing = InvestmentParameters(
            lifetime=10,  # Investment lasts 10 periods
            allow_investment=True,  # Can invest in any period
        )
        ```

        Force investment in specific period:

        ```python
        timing = InvestmentParameters(
            lifetime=10,  # Must operate for 10 periods
            force_investment=xr.DataArray(
                [0, 0, 1, 0, 0],  # Force in period 3 (2030)
                coords=[('period', [2020, 2025, 2030, 2035, 2040])],
            ),
        )
        ```

        Technology learning curve (costs decrease over time):

        ```python
        # Create investment-period-dependent effects
        periods = [2020, 2025, 2030, 2035, 2040]

        # Cost per kW depends on WHEN you invest (learning curve)
        # Each row is a current period, columns are investment periods
        learning_costs = xr.DataArray(
            [
                [1200, 0, 0, 0, 0],  # 2020: only if invested in 2020
                [1200, 1100, 0, 0, 0],  # 2025: depends on when invested
                [1200, 1100, 1000, 0, 0],  # 2030: costs lower if invested later
                [1200, 1100, 1000, 950, 0],  # 2035
                [1200, 1100, 1000, 950, 900],  # 2040: benefit from all past investments
            ],
            coords=[('period', periods), ('investment_period', periods)],
        )

        timing = InvestmentParameters(
            lifetime=10,
            effects_of_investment_per_size={
                'cost': learning_costs  # €/kW depends on investment year
            },
        )
        ```

        Restrict investment to early periods:

        ```python
        timing = InvestmentParameters(
            lifetime=15,
            allow_investment=xr.DataArray(
                [1, 1, 1, 0, 0],  # Only allow investment in first 3 periods
                coords=[('period', [2020, 2025, 2030, 2035, 2040])],
            ),
        )
        ```

    Common Use Cases:
        - Technology learning: Model cost reductions over time
        - Multi-period optimization: Optimize investment timing across periods
        - Regulatory changes: Model period-specific incentives or constraints
        - Strategic timing: Find optimal investment timing considering future conditions
    """

    def __init__(
        self,
        lifetime: InvestmentPeriodData,
        allow_investment: InvestmentPeriodDataBool = True,
        force_investment: InvestmentPeriodDataBool = False,
        effects_of_investment: PeriodicEffectsUser | None = None,
        effects_of_investment_per_size: PeriodicEffectsUser | None = None,
        previous_lifetime: int = 0,
        # Sizing parameters (inherited from _SizeParameters)
        fixed_size: PeriodicDataUser | None = None,
        minimum_size: PeriodicDataUser | None = None,
        maximum_size: PeriodicDataUser | None = None,
        mandatory: bool | xr.DataArray = False,
        effects_of_size: PeriodicEffectsUser | None = None,
        effects_per_size: PeriodicEffectsUser | None = None,
        piecewise_effects_per_size: PiecewiseEffects | None = None,
    ):
        if lifetime is None:
            raise ValueError('InvestmentParameters requires lifetime to be specified.')

        # Initialize investment-specific attributes
        self.lifetime = lifetime
        self.allow_investment = allow_investment
        self.force_investment = force_investment
        self.previous_lifetime = previous_lifetime

        self.effects_of_investment: dict[str, xr.DataArray] = (
            effects_of_investment if effects_of_investment is not None else {}
        )
        self.effects_of_investment_per_size: dict[str, xr.DataArray] = (
            effects_of_investment_per_size if effects_of_investment_per_size is not None else {}
        )

        # Initialize base sizing parameters
        super().__init__(
            fixed_size=fixed_size,
            minimum_size=minimum_size,
            maximum_size=maximum_size,
            mandatory=mandatory,
            effects_of_size=effects_of_size,
            effects_per_size=effects_per_size,
            piecewise_effects_per_size=piecewise_effects_per_size,
        )

    def transform_data(self, flow_system: FlowSystem, name_prefix: str = '') -> None:
        """Transform user data into internal model coordinates."""
        super().transform_data(flow_system, name_prefix)
        # Transform boolean/data flags to DataArrays
        self.allow_investment = flow_system.fit_to_model_coords(
            f'{name_prefix}|allow_investment', self.allow_investment, dims=['period', 'scenario']
        )
        self.force_investment = flow_system.fit_to_model_coords(
            f'{name_prefix}|force_investment', self.force_investment, dims=['period', 'scenario']
        )
        self.previous_lifetime = flow_system.fit_to_model_coords(
            f'{name_prefix}|previous_lifetime', self.previous_lifetime, dims=['scenario']
        )
        self.lifetime = flow_system.fit_to_model_coords(f'{name_prefix}|lifetime', self.lifetime, dims=['scenario'])
        self.effects_of_investment = flow_system.fit_effects_to_model_coords(
            f'{name_prefix}|effects_of_investment', self.effects_of_investment, dims=['period', 'scenario']
        )  # TODO: investment period dim

        self.effects_of_investment_per_size = flow_system.fit_effects_to_model_coords(
            f'{name_prefix}|effects_of_investment_per_size',
            self.effects_of_investment_per_size,
            dims=['period', 'scenario'],
        )  # TODO: investment period dim

    def _plausibility_checks(self, flow_system: FlowSystem) -> None:
        """Validate parameter consistency."""
        # super()._plausibility_checks(flow_system)
        if flow_system.periods is None:
            raise ValueError("InvestmentParameters requires the flow_system to have a 'periods' dimension.")

        # Check force_investment uniqueness (can only force in one period per scenario)
        if (self.force_investment.sum('investment_period') > 1).any():
            raise ValueError('force_investment can only be True for a single investment_period per scenario.')

        # Check lifetime feasibility
        _periods = flow_system.periods.values
        if len(_periods) > 1:
            # Warn if investment in late periods would extend beyond model horizon
            max_horizon = _periods[-1] - _periods[0]
            if self.lifetime > max_horizon:
                logger.warning(
                    f'Fixed lifetime ({self.lifetime}) if Investment exceeds model horizon ({max_horizon}). '
                )


YearOfInvestmentData = PeriodicDataUser
"""This datatype is used to define things related to the year of investment."""
YearOfInvestmentDataBool = bool | YearOfInvestmentData
"""This datatype is used to define things with boolean data related to the year of investment."""


@register_class_for_io
class _InvestTimingParameters(Interface):
    """
    Investment with fixed start and end years.

    This is the simplest variant - investment is completely scheduled.
    No optimization variables needed for timing, just size optimization.
    """

    def __init__(
        self,
        allow_investment: YearOfInvestmentDataBool = True,
        allow_decommissioning: YearOfInvestmentDataBool = True,
        force_investment: YearOfInvestmentDataBool = False,  # TODO: Allow to simply pass the year
        force_decommissioning: YearOfInvestmentDataBool = False,  # TODO: Allow to simply pass the year
        lifetime: int | None = None,
        minimum_lifetime: int | None = None,
        maximum_lifetime: int | None = None,
        minimum_size: YearOfInvestmentData | None = None,
        maximum_size: YearOfInvestmentData | None = None,
        fixed_size: YearOfInvestmentData | None = None,
        fix_effects: Numeric_PS | Effect_PS | None = None,
        specific_effects: Numeric_PS | Effect_PS | None | None = None,  # costs per Flow-Unit/Storage-Size/...
        fixed_effects_by_investment_year: xr.DataArray | None = None,
        specific_effects_by_investment_year: xr.DataArray | None = None,
        previous_lifetime: int | None = None,
    ):
        """
        These parameters are used to include the timing of investments in the model.
        Two out of three parameters (year_of_investment, year_of_decommissioning, duration_in_years) can be fixed.
        This has a 'year_of_investment' dimension in some parameters:
            allow_investment: Whether investment is allowed in a certain year
            allow_decommissioning: Whether divestment is allowed in a certain year
            duration_between_investment_and_decommissioning: Duration between investment and decommissioning

        Args:
            allow_investment: Allow investment in a certain year. By default, allow it in all years.
            allow_decommissioning: Allow decommissioning in a certain year. By default, allow it in all years.
            force_investment: Force the investment to occur in a certain year.
            force_decommissioning: Force the decommissioning to occur in a certain year.
            lifetime: Fix the lifetime of an investment (duration between investment and decommissioning).
            minimum_size: Minimum possible size of the investment. Can depend on the year of investment.
            maximum_size: Maximum possible size of the investment. Can depend on the year of investment.
            fixed_size: Fix the size of the investment. Can depend on the year of investment. Can still be 0 if not forced.
            specific_effects: Effects dependent on the size.
                These will occur in each year, depending on the size in that year.
            fix_effects: Effects of the Investment, independent of the size.
                These will occur in each year, depending on wether the size is greater zero in that year.

            fixed_effects_by_investment_year: Effects dependent on the year of investment.
                These effects will depend on the year of the investment. The actual effects can occur in other years,
                letting you model things like annuities, which depend on when an investment was taken.
                The passed xr.DataArray needs to match the FlowSystem dimensions (except time, but including "year_of_investment"). No internal Broadcasting!
                "year_of_investment" has the same values as the year dimension. Access it through `flow_system.year_of_investment`.
            specific_effects_by_investment_year: Effects dependent on the year of investment and the chosen size.
                These effects will depend on the year of the investment. The actual effects can occur in other years,
                letting you model things like annuities, which depend on when an investment was taken.
                The passed xr.DataArray needs to match the FlowSystem dimensions (except time, but including "year_of_investment"). No internal Broadcasting!
                "year_of_investment" has the same values as the year dimension. Access it through `flow_system.year_of_investment`.

        """
        self.minimum_size = minimum_size if minimum_size is not None else CONFIG.modeling.EPSILON
        self.maximum_size = maximum_size if maximum_size is not None else CONFIG.modeling.BIG
        self.fixed_size = fixed_size

        self.allow_investment = allow_investment
        self.allow_decommissioning = allow_decommissioning
        self.force_investment = force_investment
        self.force_decommissioning = force_decommissioning

        self.maximum_lifetime = maximum_lifetime
        self.minimum_lifetime = minimum_lifetime
        self.lifetime = lifetime
        self.previous_lifetime = previous_lifetime

        self.fix_effects: Numeric_PS | Effect_PS | None = fix_effects if fix_effects is not None else {}
        self.specific_effects: Numeric_PS | Effect_PS | None = specific_effects if specific_effects is not None else {}
        self.fixed_effects_by_investment_year = (
            fixed_effects_by_investment_year if fixed_effects_by_investment_year is not None else {}
        )
        self.specific_effects_by_investment_year = (
            specific_effects_by_investment_year if specific_effects_by_investment_year is not None else {}
        )

    def _plausibility_checks(self, flow_system):
        """Validate parameter consistency."""
        if flow_system.years is None:
            raise ValueError("InvestmentParameters requires the flow_system to have a 'period' dimension.")

        if (self.force_investment.sum('year') > 1).any():
            raise ValueError('force_investment can only be True for a single year.')
        if (self.force_decommissioning.sum('year') > 1).any():
            raise ValueError('force_decommissioning can only be True for a single year.')

        if (self.force_investment.sum('year') == 1).any() and (self.force_decommissioning.sum('year') == 1).any():
            year_of_forced_investment = (
                self.force_investment.where(self.force_investment) * self.force_investment.year
            ).sum('year')
            year_of_forced_decommissioning = (
                self.force_decommissioning.where(self.force_decommissioning) * self.force_decommissioning.year
            ).sum('year')
            if not (year_of_forced_investment < year_of_forced_decommissioning).all():
                raise ValueError(
                    f'force_investment needs to be before force_decommissioning. Got:\n'
                    f'{self.force_investment}\nand\n{self.force_decommissioning}'
                )

        if self.previous_lifetime is not None:
            if self.fixed_size is None:
                # TODO: Might be only a warning
                raise ValueError('previous_lifetime can only be used if fixed_size is defined.')
            if self.force_investment is False:
                # TODO: Might be only a warning
                raise ValueError('previous_lifetime can only be used if force_investment is True.')

        if self.minimum_or_lifetime is not None and self.maximum_or_lifetime is not None:
            years = flow_system.years.values

            infeasible_years = []
            for i, inv_year in enumerate(years[:-1]):  # Exclude last year
                future_years = years[i + 1 :]  # All years after investment
                min_decomm = self.minimum_or_lifetime + inv_year
                max_decomm = self.maximum_or_lifetime + inv_year
                if max_decomm >= years[-1]:
                    continue

                # Check if any future year falls in decommissioning window
                future_years_da = xr.DataArray(future_years, dims=['year'])
                valid_decomm = ((min_decomm <= future_years_da) & (future_years_da <= max_decomm)).any('year')
                if not valid_decomm.all():
                    infeasible_years.append(inv_year)

            if infeasible_years:
                logger.warning(
                    f'Plausibility Check in {self.__class__.__name__}:\n'
                    f'  Investment years with no feasible decommissioning: {[int(year) for year in infeasible_years]}\n'
                    f'  Consider relaxing the lifetime constraints or including more years into your model.\n'
                    f'  Lifetime:\n'
                    f'      min={self.minimum_or_lifetime}\n'
                    f'      max={self.maximum_or_lifetime}\n'
                    f'  Model years: {list(flow_system.years)}\n'
                )

        specify_timing = (
            (self.lifetime is not None)
            + bool((self.force_investment.sum('year') > 1).any())
            + bool((self.force_decommissioning.sum('year') > 1).any())
        )

        if specify_timing in (0, 3):
            # TODO: Is there a valid use case for this? Should this be checked at all?
            logger.warning(
                'Either the the lifetime of an investment should be fixed, or the investment or decommissioning '
                'needs to be forced in a certain year.'
            )

    def transform_data(self, flow_system: FlowSystem, name_prefix: str = '') -> None:
        """Transform all parameter data to match the flow system's coordinate structure."""
        self.fix_effects = flow_system.fit_effects_to_model_coords(
            label_prefix=name_prefix,
            effect_values=self.fix_effects,
            label_suffix='fix_effects',
            dims=['year', 'scenario'],
        )
        self.specific_effects = flow_system.fit_effects_to_model_coords(
            label_prefix=name_prefix,
            effect_values=self.specific_effects,
            label_suffix='specific_effects',
            dims=['year', 'scenario'],
        )
        self.maximum_lifetime = flow_system.fit_to_model_coords(
            f'{name_prefix}|maximum_lifetime', self.maximum_lifetime, dims=['scenario']
        )
        self.minimum_lifetime = flow_system.fit_to_model_coords(
            f'{name_prefix}|minimum_lifetime', self.minimum_lifetime, dims=['scenario']
        )
        self.lifetime = flow_system.fit_to_model_coords(f'{name_prefix}|lifetime', self.lifetime, dims=['scenario'])

        self.force_investment = flow_system.fit_to_model_coords(
            f'{name_prefix}|force_investment', self.force_investment, dims=['year', 'scenario']
        )
        self.force_decommissioning = flow_system.fit_to_model_coords(
            f'{name_prefix}|force_decommissioning', self.force_decommissioning, dims=['year', 'scenario']
        )

        self.minimum_size = flow_system.fit_to_model_coords(
            f'{name_prefix}|minimum_size', self.minimum_size, dims=['year', 'scenario']
        )
        self.maximum_size = flow_system.fit_to_model_coords(
            f'{name_prefix}|maximum_size', self.maximum_size, dims=['year', 'scenario']
        )
        if self.fixed_size is not None:
            self.fixed_size = flow_system.fit_to_model_coords(
                f'{name_prefix}|fixed_size', self.fixed_size, dims=['year', 'scenario']
            )

        # TODO: self.previous_size to only scenarios

        # No Broadcasting! Until a safe way is established, we need to do check for this!
        self.fixed_effects_by_investment_year = flow_system.effects.create_effect_values_dict(
            self.fixed_effects_by_investment_year
        )
        for effect, da in self.fixed_effects_by_investment_year.items():
            dims = set(da.coords)
            if not {'year_of_investment', 'year'}.issubset(dims):
                raise ValueError(
                    f'fixed_effects_by_investment_year need to have a "year_of_investment" dimension and a '
                    f'"year" dimension. Got {dims} for effect {effect}'
                )
        self.specific_effects_by_investment_year = flow_system.effects.create_effect_values_dict(
            self.specific_effects_by_investment_year
        )
        for effect, da in self.specific_effects_by_investment_year.items():
            dims = set(da.coords)
            if not {'year_of_investment', 'year'}.issubset(dims):
                raise ValueError(
                    f'specific_effects_by_investment_year need to have a "year_of_investment" dimension and a '
                    f'"year" dimension. Got {dims} for effect {effect}'
                )
        self.fixed_effects_by_investment_year = flow_system.fit_effects_to_model_coords(
            label_prefix=name_prefix,
            effect_values=self.fixed_effects_by_investment_year,
            label_suffix='fixed_effects_by_investment_year',
            dims=['year', 'scenario'],
            with_year_of_investment=True,
        )
        self.specific_effects_by_investment_year = flow_system.fit_effects_to_model_coords(
            label_prefix=name_prefix,
            effect_values=self.specific_effects_by_investment_year,
            label_suffix='specific_effects_by_investment_year',
            dims=['year', 'scenario'],
            with_year_of_investment=True,
        )

        self._plausibility_checks(flow_system)

    @property
    def minimum_or_fixed_size(self) -> PeriodicDataUser:
        """Get the effective minimum size (fixed size takes precedence)."""
        return self.fixed_size if self.fixed_size is not None else self.minimum_size

    @property
    def maximum_or_fixed_size(self) -> PeriodicDataUser:
        """Get the effective maximum size (fixed size takes precedence)."""
        return self.fixed_size if self.fixed_size is not None else self.maximum_size

    @property
    def is_fixed_size(self) -> bool:
        """Check if investment size is fixed."""
        return self.fixed_size is not None

    @property
    def minimum_or_lifetime(self) -> PeriodicDataUser:
        """Get the effective minimum lifetime (fixed lifetime takes precedence)."""
        return self.lifetime if self.lifetime is not None else self.minimum_lifetime

    @property
    def maximum_or_lifetime(self) -> PeriodicDataUser:
        """Get the effective maximum lifetime (fixed lifetime takes precedence)."""
        return self.lifetime if self.lifetime is not None else self.maximum_lifetime


@register_class_for_io
class StatusParameters(Interface):
    """Define operational constraints and effects for binary status equipment behavior.

    This class models equipment that operates in discrete states (active/inactive) rather than
    continuous operation, capturing realistic operational constraints and associated
    costs. It handles complex equipment behavior including startup costs, minimum
    run times, cycling limitations, and maintenance scheduling requirements.

    Key Modeling Capabilities:
        **Startup Costs**: One-time costs for starting equipment (fuel, wear, labor)
        **Runtime Constraints**: Minimum and maximum continuous operation periods (uptime/downtime)
        **Cycling Limits**: Maximum number of startups to prevent excessive wear
        **Operating Hours**: Total active hours limits and requirements over time horizon

    Typical Equipment Applications:
        - **Power Plants**: Combined cycle units, steam turbines with startup costs
        - **Industrial Processes**: Batch reactors, furnaces with thermal cycling
        - **HVAC Systems**: Chillers, boilers with minimum run times
        - **Backup Equipment**: Emergency generators, standby systems
        - **Process Equipment**: Compressors, pumps with operational constraints

    Mathematical Formulation:
        See the complete mathematical model in the documentation:
        [StatusParameters](../user-guide/mathematical-notation/features/StatusParameters.md)

    Args:
        effects_per_startup: Costs or impacts incurred for each transition from
            inactive state (status=0) to active state (status=1). Represents startup costs,
            wear and tear, or other switching impacts. Dictionary mapping effect
            names to values (e.g., {'cost': 500, 'maintenance_hours': 2}).
        effects_per_active_hour: Ongoing costs or impacts while equipment operates
            in the active state. Includes fuel costs, labor, consumables, or emissions.
            Dictionary mapping effect names to hourly values (e.g., {'fuel_cost': 45}).
        active_hours_min: Minimum total active hours across the entire time horizon per period.
            Ensures equipment meets minimum utilization requirements or contractual
            obligations (e.g., power purchase agreements, maintenance schedules).
        active_hours_max: Maximum total active hours across the entire time horizon per period.
            Limits equipment usage due to maintenance schedules, fuel availability,
            environmental permits, or equipment lifetime constraints.
        min_uptime: Minimum continuous operating duration once started (unit commitment term).
            Models minimum run times due to thermal constraints, process stability,
            or efficiency considerations. Can be time-varying to reflect different
            constraints across the planning horizon.
        max_uptime: Maximum continuous operating duration in one campaign (unit commitment term).
            Models mandatory maintenance intervals, process batch sizes, or
            equipment thermal limits requiring periodic shutdowns.
        min_downtime: Minimum continuous shutdown duration between operations (unit commitment term).
            Models cooling periods, maintenance requirements, or process constraints
            that prevent immediate restart after shutdown.
        max_downtime: Maximum continuous shutdown duration before mandatory
            restart. Models equipment preservation, process stability, or contractual
            requirements for minimum activity levels.
        startup_limit: Maximum number of startup operations across the time horizon per period..
            Limits equipment cycling to reduce wear, maintenance costs, or comply
            with operational constraints (e.g., grid stability requirements).
        force_startup_tracking: When True, creates startup variables even without explicit
            startup_limit constraint. Useful for tracking or reporting startup
            events without enforcing limits.

    Note:
        **Time Series Boundary Handling**: The final time period constraints for
        min_uptime/max_uptime and min_downtime/max_downtime are not
        enforced, allowing the optimization to end with ongoing campaigns that
        may be shorter than the specified minimums or longer than maximums.

    Examples:
        Combined cycle power plant with startup costs and minimum run time:

        ```python
        power_plant_operation = StatusParameters(
            effects_per_startup={
                'startup_cost': 25000,  # €25,000 per startup
                'startup_fuel': 150,  # GJ natural gas for startup
                'startup_time': 4,  # Hours to reach full output
                'maintenance_impact': 0.1,  # Fractional life consumption
            },
            effects_per_active_hour={
                'fixed_om': 125,  # Fixed O&M costs while active
                'auxiliary_power': 2.5,  # MW parasitic loads
            },
            min_uptime=8,  # Minimum 8-hour run once started
            min_downtime=4,  # Minimum 4-hour cooling period
            active_hours_max=6000,  # Annual operating limit
        )
        ```

        Industrial batch process with cycling limits:

        ```python
        batch_reactor = StatusParameters(
            effects_per_startup={
                'setup_cost': 1500,  # Labor and materials for startup
                'catalyst_consumption': 5,  # kg catalyst per batch
                'cleaning_chemicals': 200,  # L cleaning solution
            },
            effects_per_active_hour={
                'steam': 2.5,  # t/h process steam
                'electricity': 150,  # kWh electrical load
                'cooling_water': 50,  # m³/h cooling water
            },
            min_uptime=12,  # Minimum batch size (12 hours)
            max_uptime=24,  # Maximum batch size (24 hours)
            min_downtime=6,  # Cleaning and setup time
            startup_limit=200,  # Maximum 200 batches per period
            active_hours_max=4000,  # Maximum production time
        )
        ```

        HVAC system with thermostat control and maintenance:

        ```python
        hvac_operation = StatusParameters(
            effects_per_startup={
                'compressor_wear': 0.5,  # Hours of compressor life per start
                'inrush_current': 15,  # kW peak demand on startup
            },
            effects_per_active_hour={
                'electricity': 25,  # kW electrical consumption
                'maintenance': 0.12,  # €/hour maintenance reserve
            },
            min_uptime=1,  # Minimum 1-hour run to avoid cycling
            min_downtime=0.5,  # 30-minute minimum inactive time
            startup_limit=2000,  # Limit cycling for compressor life
            active_hours_min=2000,  # Minimum operation for humidity control
            active_hours_max=5000,  # Maximum operation for energy budget
        )
        ```

        Backup generator with testing and maintenance requirements:

        ```python
        backup_generator = StatusParameters(
            effects_per_startup={
                'fuel_priming': 50,  # L diesel for system priming
                'wear_factor': 1.0,  # Start cycles impact on maintenance
                'testing_labor': 2,  # Hours technician time per test
            },
            effects_per_active_hour={
                'fuel_consumption': 180,  # L/h diesel consumption
                'emissions_permit': 15,  # € emissions allowance cost
                'noise_penalty': 25,  # € noise compliance cost
            },
            min_uptime=0.5,  # Minimum test duration (30 min)
            max_downtime=720,  # Maximum 30 days between tests
            startup_limit=52,  # Weekly testing limit
            active_hours_min=26,  # Minimum annual testing (0.5h × 52)
            active_hours_max=200,  # Maximum runtime (emergencies + tests)
        )
        ```

        Peak shaving battery with cycling degradation:

        ```python
        battery_cycling = StatusParameters(
            effects_per_startup={
                'cycle_degradation': 0.01,  # % capacity loss per cycle
                'inverter_startup': 0.5,  # kWh losses during startup
            },
            effects_per_active_hour={
                'standby_losses': 2,  # kW standby consumption
                'cooling': 5,  # kW thermal management
                'inverter_losses': 8,  # kW conversion losses
            },
            min_uptime=1,  # Minimum discharge duration
            max_uptime=4,  # Maximum continuous discharge
            min_downtime=1,  # Minimum rest between cycles
            startup_limit=365,  # Daily cycling limit
            force_startup_tracking=True,  # Track all cycling events
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
        effects_per_startup: Effect_TPS | Numeric_TPS | None = None,
        effects_per_active_hour: Effect_TPS | Numeric_TPS | None = None,
        active_hours_min: Numeric_PS | None = None,
        active_hours_max: Numeric_PS | None = None,
        min_uptime: Numeric_TPS | None = None,
        max_uptime: Numeric_TPS | None = None,
        min_downtime: Numeric_TPS | None = None,
        max_downtime: Numeric_TPS | None = None,
        startup_limit: Numeric_PS | None = None,
        force_startup_tracking: bool = False,
    ):
        self.effects_per_startup = effects_per_startup if effects_per_startup is not None else {}
        self.effects_per_active_hour = effects_per_active_hour if effects_per_active_hour is not None else {}
        self.active_hours_min = active_hours_min
        self.active_hours_max = active_hours_max
        self.min_uptime = min_uptime
        self.max_uptime = max_uptime
        self.min_downtime = min_downtime
        self.max_downtime = max_downtime
        self.startup_limit = startup_limit
        self.force_startup_tracking: bool = force_startup_tracking

    def transform_data(self, name_prefix: str = '') -> None:
        self.effects_per_startup = self._fit_effect_coords(
            prefix=name_prefix,
            effect_values=self.effects_per_startup,
            suffix='per_startup',
        )
        self.effects_per_active_hour = self._fit_effect_coords(
            prefix=name_prefix,
            effect_values=self.effects_per_active_hour,
            suffix='per_active_hour',
        )
        self.min_uptime = self._fit_coords(f'{name_prefix}|min_uptime', self.min_uptime)
        self.max_uptime = self._fit_coords(f'{name_prefix}|max_uptime', self.max_uptime)
        self.min_downtime = self._fit_coords(f'{name_prefix}|min_downtime', self.min_downtime)
        self.max_downtime = self._fit_coords(f'{name_prefix}|max_downtime', self.max_downtime)
        self.active_hours_max = self._fit_coords(
            f'{name_prefix}|active_hours_max', self.active_hours_max, dims=['period', 'scenario']
        )
        self.active_hours_min = self._fit_coords(
            f'{name_prefix}|active_hours_min', self.active_hours_min, dims=['period', 'scenario']
        )
        self.startup_limit = self._fit_coords(
            f'{name_prefix}|startup_limit', self.startup_limit, dims=['period', 'scenario']
        )

    @property
    def use_uptime_tracking(self) -> bool:
        """Determines whether a Variable for uptime (consecutive active hours) is needed or not"""
        return any(param is not None for param in [self.min_uptime, self.max_uptime])

    @property
    def use_downtime_tracking(self) -> bool:
        """Determines whether a Variable for downtime (consecutive inactive hours) is needed or not"""
        return any(param is not None for param in [self.min_downtime, self.max_downtime])

    @property
    def use_startup_tracking(self) -> bool:
        """Determines whether a variable for startup is needed or not"""
        if self.force_startup_tracking:
            return True

        return any(
            self._has_value(param)
            for param in [
                self.effects_per_startup,
                self.startup_limit,
            ]
        )
