"""
This module contains classes to collect Parameters for the Investment and Status decisions.
These are tightly connected to features.py
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr

from .config import CONFIG
from .plot_result import PlotResult
from .structure import Interface, register_class_for_io

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from collections.abc import Iterator

    from .types import Effect_PS, Effect_TPS, Numeric_PS, Numeric_TPS

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

    def transform_data(self) -> None:
        dims = None if self.has_time_dim else ['period', 'scenario']
        self.start = self._fit_coords(f'{self.prefix}|start', self.start, dims=dims)
        self.end = self._fit_coords(f'{self.prefix}|end', self.end, dims=dims)


@register_class_for_io
class Piecewise(Interface):
    """Define piecewise linear approximations for modeling non-linear relationships.

    Enables modeling of non-linear relationships through piecewise linear segments
    while maintaining problem linearity. Consists of a collection of Pieces that
    define valid ranges for variables.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/features/Piecewise/>

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

    def link_to_flow_system(self, flow_system, prefix: str = '') -> None:
        """Propagate flow_system reference to nested Piece objects."""
        super().link_to_flow_system(flow_system, prefix)
        for i, piece in enumerate(self.pieces):
            piece.link_to_flow_system(flow_system, self._sub_prefix(f'Piece{i}'))

    def transform_data(self) -> None:
        for piece in self.pieces:
            piece.transform_data()


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

    def link_to_flow_system(self, flow_system, prefix: str = '') -> None:
        """Propagate flow_system reference to nested Piecewise objects."""
        super().link_to_flow_system(flow_system, prefix)
        for name, piecewise in self.piecewises.items():
            piecewise.link_to_flow_system(flow_system, self._sub_prefix(name))

    def transform_data(self) -> None:
        for piecewise in self.piecewises.values():
            piecewise.transform_data()

    def plot(
        self,
        x_flow: str | None = None,
        title: str = '',
        select: dict[str, Any] | None = None,
        colorscale: str | None = None,
        show: bool | None = None,
    ) -> PlotResult:
        """Plot multi-flow piecewise conversion with time variation visualization.

        Visualizes the piecewise linear relationships between flows. Each flow
        is shown in a separate subplot (faceted by flow). Pieces are distinguished
        by line dash style. If boundaries vary over time, color shows time progression.

        Note:
            Requires FlowSystem to be connected and transformed (call
            flow_system.connect_and_transform() first).

        Args:
            x_flow: Flow label to use for X-axis. Defaults to first flow in dict.
            title: Plot title.
            select: xarray-style selection dict to filter data,
                e.g. {'time': slice('2024-01-01', '2024-01-02')}.
            colorscale: Colorscale name for time coloring (e.g., 'RdYlBu_r', 'viridis').
                Defaults to CONFIG.Plotting.default_sequential_colorscale.
            show: Whether to display the figure.
                Defaults to CONFIG.Plotting.default_show.

        Returns:
            PlotResult containing the figure and underlying piecewise data.

        Examples:
            >>> flow_system.connect_and_transform()
            >>> chp.piecewise_conversion.plot(x_flow='Gas', title='CHP Curves')
            >>> # Select specific time range
            >>> chp.piecewise_conversion.plot(select={'time': slice(0, 12)})
        """
        if not self.flow_system.connected_and_transformed:
            logger.debug('Connecting flow_system for plotting PiecewiseConversion')
            self.flow_system.connect_and_transform()

        colorscale = colorscale or CONFIG.Plotting.default_sequential_colorscale

        flow_labels = list(self.piecewises.keys())
        x_label = x_flow if x_flow is not None else flow_labels[0]
        if x_label not in flow_labels:
            raise ValueError(f"x_flow '{x_label}' not found. Available: {flow_labels}")

        y_flows = [label for label in flow_labels if label != x_label]
        if not y_flows:
            raise ValueError('Need at least two flows to plot')

        x_piecewise = self.piecewises[x_label]

        # Build Dataset with all piece data
        datasets = []
        for y_label in y_flows:
            y_piecewise = self.piecewises[y_label]
            for i, (x_piece, y_piece) in enumerate(zip(x_piecewise, y_piecewise, strict=False)):
                ds = xr.Dataset(
                    {
                        x_label: xr.concat([x_piece.start, x_piece.end], dim='point'),
                        'output': xr.concat([y_piece.start, y_piece.end], dim='point'),
                    }
                )
                ds = ds.assign_coords(point=['start', 'end'])
                ds['flow'] = y_label
                ds['piece'] = f'Piece {i}'
                datasets.append(ds)

        combined = xr.concat(datasets, dim='trace')

        # Apply selection if provided
        if select:
            valid_select = {k: v for k, v in select.items() if k in combined.dims or k in combined.coords}
            if valid_select:
                combined = combined.sel(valid_select)

        df = combined.to_dataframe().reset_index()

        # Check if values vary over time
        has_time = 'time' in df.columns
        varies_over_time = False
        if has_time:
            varies_over_time = df.groupby(['trace', 'point'])[[x_label, 'output']].nunique().max().max() > 1

        if varies_over_time:
            # Time-varying: color by time, dash by piece
            df['time_idx'] = df.groupby('time').ngroup()
            df['line_id'] = df['trace'].astype(str) + '_' + df['time_idx'].astype(str)
            n_times = df['time_idx'].nunique()
            colors = px.colors.sample_colorscale(colorscale, n_times)

            fig = px.line(
                df,
                x=x_label,
                y='output',
                color='time_idx',
                line_dash='piece',
                line_group='line_id',
                facet_col='flow' if len(y_flows) > 1 else None,
                title=title or 'Piecewise Conversion',
                markers=True,
                color_discrete_sequence=colors,
            )
        else:
            # Static: dash by piece
            if has_time:
                df = df.groupby(['trace', 'point', 'flow', 'piece']).first().reset_index()
            df['line_id'] = df['trace'].astype(str)

            fig = px.line(
                df,
                x=x_label,
                y='output',
                line_dash='piece',
                line_group='line_id',
                facet_col='flow' if len(y_flows) > 1 else None,
                title=title or 'Piecewise Conversion',
                markers=True,
            )

        # Clean up facet titles and axis labels
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace('flow=', '')))
        fig.update_yaxes(title_text='')
        fig.update_xaxes(title_text=x_label)

        result = PlotResult(data=combined, figure=fig)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            result.show()

        return result


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

    def link_to_flow_system(self, flow_system, prefix: str = '') -> None:
        """Propagate flow_system reference to nested Piecewise objects."""
        super().link_to_flow_system(flow_system, prefix)
        self.piecewise_origin.link_to_flow_system(flow_system, self._sub_prefix('origin'))
        for effect, piecewise in self.piecewise_shares.items():
            piecewise.link_to_flow_system(flow_system, self._sub_prefix(effect))

    def transform_data(self) -> None:
        self.piecewise_origin.transform_data()
        for piecewise in self.piecewise_shares.values():
            piecewise.transform_data()

    def plot(
        self,
        title: str = '',
        select: dict[str, Any] | None = None,
        colorscale: str | None = None,
        show: bool | None = None,
    ) -> PlotResult:
        """Plot origin vs effect shares with time variation visualization.

        Visualizes the piecewise linear relationships between the origin variable
        and its effect shares. Each effect is shown in a separate subplot (faceted
        by effect). Pieces are distinguished by line dash style.

        Note:
            Requires FlowSystem to be connected and transformed (call
            flow_system.connect_and_transform() first).

        Args:
            title: Plot title.
            select: xarray-style selection dict to filter data,
                e.g. {'time': slice('2024-01-01', '2024-01-02')}.
            colorscale: Colorscale name for time coloring (e.g., 'RdYlBu_r', 'viridis').
                Defaults to CONFIG.Plotting.default_sequential_colorscale.
            show: Whether to display the figure.
                Defaults to CONFIG.Plotting.default_show.

        Returns:
            PlotResult containing the figure and underlying piecewise data.

        Examples:
            >>> flow_system.connect_and_transform()
            >>> invest_params.piecewise_effects_of_investment.plot(title='Investment Effects')
        """
        if not self.flow_system.connected_and_transformed:
            logger.debug('Connecting flow_system for plotting PiecewiseEffects')
            self.flow_system.connect_and_transform()

        colorscale = colorscale or CONFIG.Plotting.default_sequential_colorscale

        effect_labels = list(self.piecewise_shares.keys())
        if not effect_labels:
            raise ValueError('Need at least one effect share to plot')

        # Build Dataset with all piece data
        datasets = []
        for effect_label in effect_labels:
            y_piecewise = self.piecewise_shares[effect_label]
            for i, (x_piece, y_piece) in enumerate(zip(self.piecewise_origin, y_piecewise, strict=False)):
                ds = xr.Dataset(
                    {
                        'origin': xr.concat([x_piece.start, x_piece.end], dim='point'),
                        'share': xr.concat([y_piece.start, y_piece.end], dim='point'),
                    }
                )
                ds = ds.assign_coords(point=['start', 'end'])
                ds['effect'] = effect_label
                ds['piece'] = f'Piece {i}'
                datasets.append(ds)

        combined = xr.concat(datasets, dim='trace')

        # Apply selection if provided
        if select:
            valid_select = {k: v for k, v in select.items() if k in combined.dims or k in combined.coords}
            if valid_select:
                combined = combined.sel(valid_select)

        df = combined.to_dataframe().reset_index()

        # Check if values vary over time
        has_time = 'time' in df.columns
        varies_over_time = False
        if has_time:
            varies_over_time = df.groupby(['trace', 'point'])[['origin', 'share']].nunique().max().max() > 1

        if varies_over_time:
            # Time-varying: color by time, dash by piece
            df['time_idx'] = df.groupby('time').ngroup()
            df['line_id'] = df['trace'].astype(str) + '_' + df['time_idx'].astype(str)
            n_times = df['time_idx'].nunique()
            colors = px.colors.sample_colorscale(colorscale, n_times)

            fig = px.line(
                df,
                x='origin',
                y='share',
                color='time_idx',
                line_dash='piece',
                line_group='line_id',
                facet_col='effect' if len(effect_labels) > 1 else None,
                title=title or 'Piecewise Effects',
                markers=True,
                color_discrete_sequence=colors,
            )
        else:
            # Static: dash by piece
            if has_time:
                df = df.groupby(['trace', 'point', 'effect', 'piece']).first().reset_index()
            df['line_id'] = df['trace'].astype(str)

            fig = px.line(
                df,
                x='origin',
                y='share',
                line_dash='piece',
                line_group='line_id',
                facet_col='effect' if len(effect_labels) > 1 else None,
                title=title or 'Piecewise Effects',
                markers=True,
            )

        # Clean up facet titles and axis labels
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace('effect=', '')))
        fig.update_yaxes(title_text='')
        fig.update_xaxes(title_text='Origin')

        result = PlotResult(data=combined, figure=fig)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            result.show()

        return result


@register_class_for_io
class InvestParameters(Interface):
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
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/features/InvestParameters/>

    Args:
        fixed_size: Creates binary decision at this exact size. None allows continuous sizing.
        minimum_size: Lower bound for continuous sizing. Default: CONFIG.Modeling.epsilon.
            Ignored if fixed_size is specified.
        maximum_size: Upper bound for continuous sizing. Required if fixed_size is not set.
            Ignored if fixed_size is specified.
        mandatory: Controls whether investment is required. When True, forces investment
            to occur (useful for mandatory upgrades or replacement decisions).
            When False (default), optimization can choose not to invest.
            With multiple periods, at least one period has to have an investment.
        effects_of_investment: Fixed costs if investment is made, regardless of size.
            Dict: {'effect_name': value} (e.g., {'cost': 10000}).
        effects_of_investment_per_size: Variable costs proportional to size (per-unit costs).
            Dict: {'effect_name': value/unit} (e.g., {'cost': 1200}).
        piecewise_effects_of_investment: Non-linear costs using PiecewiseEffects.
            Combinable with effects_of_investment and effects_of_investment_per_size.
        effects_of_retirement: Costs incurred if NOT investing (demolition, penalties).
            Dict: {'effect_name': value}.
        linked_periods: Describes which periods are linked. 1 means linked, 0 means size=0. None means no linked periods.
            For convenience, pass a tuple containing the first and last period (2025, 2039), linking them and those in between

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
        solar_investment = InvestParameters(
            fixed_size=100,  # 100 kW system (binary decision)
            mandatory=False,  # Investment is optional
            effects_of_investment={
                'cost': 25000,  # Installation and permitting costs
                'CO2': -50000,  # Avoided emissions over lifetime
            },
            effects_of_investment_per_size={
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
            mandatory=False,  # Investment is optional
            effects_of_investment={
                'cost': 5000,  # Grid connection and control system
                'installation_time': 2,  # Days for fixed components
            },
            piecewise_effects_of_investment=PiecewiseEffects(
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
        boiler_replacement = InvestParameters(
            minimum_size=50,
            maximum_size=200,
            mandatory=False,  # Can choose not to replace
            effects_of_investment={
                'cost': 15000,  # Installation costs
                'disruption': 3,  # Days of downtime
            },
            effects_of_investment_per_size={
                'cost': 400,  # €400/kW capacity
                'maintenance': 25,  # Annual maintenance per kW
            },
            effects_of_retirement={
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
            effects_of_investment={'cost': 2500000, 'CO2': 1250000},
            effects_of_investment_per_size={'fuel_cost': 45, 'maintenance': 12},
        )

        # Wind farm option
        wind_farm = InvestParameters(
            minimum_size=20,
            maximum_size=100,
            effects_of_investment={'cost': 1000000, 'CO2': -5000000},
            effects_of_investment_per_size={'cost': 1800000, 'land_use': 0.5},
        )
        ```

        Technology learning curve:

        ```python
        hydrogen_electrolyzer = InvestParameters(
            minimum_size=1,
            maximum_size=50,  # MW
            piecewise_effects_of_investment=PiecewiseEffects(
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
        fixed_size: Numeric_PS | None = None,
        minimum_size: Numeric_PS | None = None,
        maximum_size: Numeric_PS | None = None,
        mandatory: bool = False,
        effects_of_investment: Effect_PS | Numeric_PS | None = None,
        effects_of_investment_per_size: Effect_PS | Numeric_PS | None = None,
        effects_of_retirement: Effect_PS | Numeric_PS | None = None,
        piecewise_effects_of_investment: PiecewiseEffects | None = None,
        linked_periods: Numeric_PS | tuple[int, int] | None = None,
    ):
        self._effects_of_investment = effects_of_investment if effects_of_investment is not None else {}
        self._effects_of_retirement = effects_of_retirement if effects_of_retirement is not None else {}
        self._fixed_size = fixed_size
        self._mandatory = mandatory
        self._effects_of_investment_per_size = (
            effects_of_investment_per_size if effects_of_investment_per_size is not None else {}
        )
        self._piecewise_effects_of_investment = piecewise_effects_of_investment
        self._minimum_size = minimum_size if minimum_size is not None else CONFIG.Modeling.epsilon
        self._maximum_size = maximum_size
        self._linked_periods = linked_periods

    # region Properties
    @property
    def effects_of_investment(self) -> Effect_PS:
        """Fixed costs if investment is made."""
        return (
            dict(self._effects_of_investment)
            if isinstance(self._effects_of_investment, dict)
            else self._effects_of_investment
        )

    @effects_of_investment.setter
    def effects_of_investment(self, value: Effect_PS | Numeric_PS | None) -> None:
        self._invalidate()
        self._effects_of_investment = value if value is not None else {}

    @property
    def effects_of_retirement(self) -> Effect_PS:
        """Costs incurred if NOT investing."""
        return (
            dict(self._effects_of_retirement)
            if isinstance(self._effects_of_retirement, dict)
            else self._effects_of_retirement
        )

    @effects_of_retirement.setter
    def effects_of_retirement(self, value: Effect_PS | Numeric_PS | None) -> None:
        self._invalidate()
        self._effects_of_retirement = value if value is not None else {}

    @property
    def effects_of_investment_per_size(self) -> Effect_PS:
        """Variable costs proportional to size."""
        return (
            dict(self._effects_of_investment_per_size)
            if isinstance(self._effects_of_investment_per_size, dict)
            else self._effects_of_investment_per_size
        )

    @effects_of_investment_per_size.setter
    def effects_of_investment_per_size(self, value: Effect_PS | Numeric_PS | None) -> None:
        self._invalidate()
        self._effects_of_investment_per_size = value if value is not None else {}

    @property
    def fixed_size(self) -> Numeric_PS | None:
        """Creates binary decision at this exact size."""
        return self._fixed_size

    @fixed_size.setter
    def fixed_size(self, value: Numeric_PS | None) -> None:
        self._invalidate()
        self._fixed_size = value

    @property
    def mandatory(self) -> bool:
        """Whether investment is required."""
        return self._mandatory

    @mandatory.setter
    def mandatory(self, value: bool) -> None:
        self._invalidate()
        self._mandatory = value

    @property
    def piecewise_effects_of_investment(self) -> PiecewiseEffects | None:
        """Non-linear costs using PiecewiseEffects."""
        return self._piecewise_effects_of_investment

    @piecewise_effects_of_investment.setter
    def piecewise_effects_of_investment(self, value: PiecewiseEffects | None) -> None:
        self._invalidate()
        self._piecewise_effects_of_investment = value

    @property
    def minimum_size(self) -> Numeric_PS:
        """Lower bound for continuous sizing."""
        return self._minimum_size

    @minimum_size.setter
    def minimum_size(self, value: Numeric_PS | None) -> None:
        self._invalidate()
        self._minimum_size = value if value is not None else CONFIG.Modeling.epsilon

    @property
    def maximum_size(self) -> Numeric_PS | None:
        """Upper bound for continuous sizing."""
        return self._maximum_size

    @maximum_size.setter
    def maximum_size(self, value: Numeric_PS | None) -> None:
        self._invalidate()
        self._maximum_size = value

    @property
    def linked_periods(self) -> Numeric_PS | tuple[int, int] | None:
        """Describes which periods are linked."""
        return self._linked_periods

    @linked_periods.setter
    def linked_periods(self, value: Numeric_PS | tuple[int, int] | None) -> None:
        self._invalidate()
        self._linked_periods = value

    # endregion

    def link_to_flow_system(self, flow_system, prefix: str = '') -> None:
        """Propagate flow_system reference to nested PiecewiseEffects object if present."""
        super().link_to_flow_system(flow_system, prefix)
        if self.piecewise_effects_of_investment is not None:
            self.piecewise_effects_of_investment.link_to_flow_system(flow_system, self._sub_prefix('PiecewiseEffects'))

    def transform_data(self) -> None:
        # Validate that either fixed_size or maximum_size is set
        if self._fixed_size is None and self._maximum_size is None:
            raise ValueError(
                f'InvestParameters in "{self.prefix}" requires either fixed_size or maximum_size to be set. '
                f'An upper bound is needed to properly scale the optimization model.'
            )
        # Use backing fields directly to avoid triggering invalidation
        self._effects_of_investment = self._fit_effect_coords(
            prefix=self.prefix,
            effect_values=self._effects_of_investment,
            suffix='effects_of_investment',
            dims=['period', 'scenario'],
        )
        self._effects_of_retirement = self._fit_effect_coords(
            prefix=self.prefix,
            effect_values=self._effects_of_retirement,
            suffix='effects_of_retirement',
            dims=['period', 'scenario'],
        )
        self._effects_of_investment_per_size = self._fit_effect_coords(
            prefix=self.prefix,
            effect_values=self._effects_of_investment_per_size,
            suffix='effects_of_investment_per_size',
            dims=['period', 'scenario'],
        )

        if self._piecewise_effects_of_investment is not None:
            self._piecewise_effects_of_investment.has_time_dim = False
            self._piecewise_effects_of_investment.transform_data()

        self._minimum_size = self._fit_coords(
            f'{self.prefix}|minimum_size', self._minimum_size, dims=['period', 'scenario']
        )
        self._maximum_size = self._fit_coords(
            f'{self.prefix}|maximum_size', self._maximum_size, dims=['period', 'scenario']
        )
        # Convert tuple (first_period, last_period) to DataArray if needed
        if isinstance(self._linked_periods, (tuple, list)):
            if len(self._linked_periods) != 2:
                raise TypeError(
                    f'If you provide a tuple to "linked_periods", it needs to be len=2. Got {len(self._linked_periods)=}'
                )
            if self.flow_system.periods is None:
                raise ValueError(
                    f'Cannot use linked_periods={self._linked_periods} when FlowSystem has no periods defined. '
                    f'Please define periods in FlowSystem or use linked_periods=None.'
                )
            logger.debug(f'Computing linked_periods from {self._linked_periods}')
            start, end = self._linked_periods
            if start not in self.flow_system.periods.values:
                logger.warning(
                    f'Start of linked periods ({start} not found in periods directly: {self.flow_system.periods.values}'
                )
            if end not in self.flow_system.periods.values:
                logger.warning(
                    f'End of linked periods ({end} not found in periods directly: {self.flow_system.periods.values}'
                )
            self._linked_periods = self.compute_linked_periods(start, end, self.flow_system.periods)
            logger.debug(f'Computed {self._linked_periods=}')

        self._linked_periods = self._fit_coords(
            f'{self.prefix}|linked_periods', self._linked_periods, dims=['period', 'scenario']
        )
        self._fixed_size = self._fit_coords(f'{self.prefix}|fixed_size', self._fixed_size, dims=['period', 'scenario'])

    @property
    def minimum_or_fixed_size(self) -> Numeric_PS:
        return self.fixed_size if self.fixed_size is not None else self.minimum_size

    @property
    def maximum_or_fixed_size(self) -> Numeric_PS:
        return self.fixed_size if self.fixed_size is not None else self.maximum_size

    def format_for_repr(self) -> str:
        """Format InvestParameters for display in repr methods.

        Returns:
            Formatted string showing size information
        """
        from .io import numeric_to_str_for_repr

        if self.fixed_size is not None:
            val = numeric_to_str_for_repr(self.fixed_size)
            status = 'mandatory' if self.mandatory else 'optional'
            return f'{val} ({status})'

        # Show range if available
        parts = []
        if self.minimum_size is not None:
            parts.append(f'min: {numeric_to_str_for_repr(self.minimum_size)}')
        if self.maximum_size is not None:
            parts.append(f'max: {numeric_to_str_for_repr(self.maximum_size)}')
        return ', '.join(parts) if parts else 'invest'

    @staticmethod
    def compute_linked_periods(first_period: int, last_period: int, periods: pd.Index | list[int]) -> xr.DataArray:
        return xr.DataArray(
            xr.where(
                (first_period <= np.array(periods)) & (np.array(periods) <= last_period),
                1,
                0,
            ),
            coords=(pd.Index(periods, name='period'),),
        ).rename('linked_periods')


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
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/features/StatusParameters/>

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
        cluster_mode: How inter-timestep constraints are handled at cluster boundaries.
            Only relevant when using ``transform.cluster()``. Options:

            - ``'relaxed'``: No constraint at cluster boundaries. Startups at the first
              timestep of each cluster are not forced - the optimizer is free to choose.
              This prevents clustering from inducing "phantom" startups. (default)
            - ``'cyclic'``: Each cluster's final status equals its initial status.
              Ensures consistent behavior within each representative period.

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
        cluster_mode: Literal['relaxed', 'cyclic'] = 'relaxed',
    ):
        self._effects_per_startup = effects_per_startup if effects_per_startup is not None else {}
        self._effects_per_active_hour = effects_per_active_hour if effects_per_active_hour is not None else {}
        self._active_hours_min = active_hours_min
        self._active_hours_max = active_hours_max
        self._min_uptime = min_uptime
        self._max_uptime = max_uptime
        self._min_downtime = min_downtime
        self._max_downtime = max_downtime
        self._startup_limit = startup_limit
        self._force_startup_tracking: bool = force_startup_tracking
        self._cluster_mode = cluster_mode

    # region Properties
    @property
    def effects_per_startup(self) -> Effect_TPS:
        """Costs per startup transition."""
        return (
            dict(self._effects_per_startup)
            if isinstance(self._effects_per_startup, dict)
            else self._effects_per_startup
        )

    @effects_per_startup.setter
    def effects_per_startup(self, value: Effect_TPS | Numeric_TPS | None) -> None:
        self._invalidate()
        self._effects_per_startup = value if value is not None else {}

    @property
    def effects_per_active_hour(self) -> Effect_TPS:
        """Ongoing costs while active."""
        return (
            dict(self._effects_per_active_hour)
            if isinstance(self._effects_per_active_hour, dict)
            else self._effects_per_active_hour
        )

    @effects_per_active_hour.setter
    def effects_per_active_hour(self, value: Effect_TPS | Numeric_TPS | None) -> None:
        self._invalidate()
        self._effects_per_active_hour = value if value is not None else {}

    @property
    def active_hours_min(self) -> Numeric_PS | None:
        """Minimum total active hours per period."""
        return self._active_hours_min

    @active_hours_min.setter
    def active_hours_min(self, value: Numeric_PS | None) -> None:
        self._invalidate()
        self._active_hours_min = value

    @property
    def active_hours_max(self) -> Numeric_PS | None:
        """Maximum total active hours per period."""
        return self._active_hours_max

    @active_hours_max.setter
    def active_hours_max(self, value: Numeric_PS | None) -> None:
        self._invalidate()
        self._active_hours_max = value

    @property
    def min_uptime(self) -> Numeric_TPS | None:
        """Minimum continuous operating duration once started."""
        return self._min_uptime

    @min_uptime.setter
    def min_uptime(self, value: Numeric_TPS | None) -> None:
        self._invalidate()
        self._min_uptime = value

    @property
    def max_uptime(self) -> Numeric_TPS | None:
        """Maximum continuous operating duration."""
        return self._max_uptime

    @max_uptime.setter
    def max_uptime(self, value: Numeric_TPS | None) -> None:
        self._invalidate()
        self._max_uptime = value

    @property
    def min_downtime(self) -> Numeric_TPS | None:
        """Minimum continuous shutdown duration."""
        return self._min_downtime

    @min_downtime.setter
    def min_downtime(self, value: Numeric_TPS | None) -> None:
        self._invalidate()
        self._min_downtime = value

    @property
    def max_downtime(self) -> Numeric_TPS | None:
        """Maximum continuous shutdown duration."""
        return self._max_downtime

    @max_downtime.setter
    def max_downtime(self, value: Numeric_TPS | None) -> None:
        self._invalidate()
        self._max_downtime = value

    @property
    def startup_limit(self) -> Numeric_PS | None:
        """Maximum number of startups per period."""
        return self._startup_limit

    @startup_limit.setter
    def startup_limit(self, value: Numeric_PS | None) -> None:
        self._invalidate()
        self._startup_limit = value

    @property
    def force_startup_tracking(self) -> bool:
        """Whether to create startup variables even without limits."""
        return self._force_startup_tracking

    @force_startup_tracking.setter
    def force_startup_tracking(self, value: bool) -> None:
        self._invalidate()
        self._force_startup_tracking = value

    @property
    def cluster_mode(self) -> Literal['relaxed', 'cyclic']:
        """How this on/off is treated during clustering optimization."""
        return self._cluster_mode

    @cluster_mode.setter
    def cluster_mode(self, value: Literal['relaxed', 'cyclic']) -> None:
        self._invalidate()
        self._cluster_mode = value

    # endregion

    def transform_data(self) -> None:
        # Use backing fields directly to avoid triggering invalidation
        self._effects_per_startup = self._fit_effect_coords(
            prefix=self.prefix,
            effect_values=self._effects_per_startup,
            suffix='per_startup',
        )
        self._effects_per_active_hour = self._fit_effect_coords(
            prefix=self.prefix,
            effect_values=self._effects_per_active_hour,
            suffix='per_active_hour',
        )
        self._min_uptime = self._fit_coords(f'{self.prefix}|min_uptime', self._min_uptime)
        self._max_uptime = self._fit_coords(f'{self.prefix}|max_uptime', self._max_uptime)
        self._min_downtime = self._fit_coords(f'{self.prefix}|min_downtime', self._min_downtime)
        self._max_downtime = self._fit_coords(f'{self.prefix}|max_downtime', self._max_downtime)
        self._active_hours_max = self._fit_coords(
            f'{self.prefix}|active_hours_max', self._active_hours_max, dims=['period', 'scenario']
        )
        self._active_hours_min = self._fit_coords(
            f'{self.prefix}|active_hours_min', self._active_hours_min, dims=['period', 'scenario']
        )
        self._startup_limit = self._fit_coords(
            f'{self.prefix}|startup_limit', self._startup_limit, dims=['period', 'scenario']
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
