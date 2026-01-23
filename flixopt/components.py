"""
This module contains the basic components of the flixopt framework.
"""

from __future__ import annotations

import functools
import logging
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from . import io as fx_io
from .core import PlausibilityError
from .elements import Component, ComponentModel, Flow
from .features import InvestmentModel, PiecewiseModel
from .interface import InvestParameters, PiecewiseConversion, StatusParameters
from .modeling import BoundingPatterns, _scalar_safe_isel, _scalar_safe_isel_drop, _scalar_safe_reduce
from .structure import FlowSystemModel, VariableCategory, register_class_for_io

if TYPE_CHECKING:
    import linopy

    from .types import Numeric_PS, Numeric_TPS

logger = logging.getLogger('flixopt')


@register_class_for_io
class LinearConverter(Component):
    """
    Converts input-Flows into output-Flows via linear conversion factors.

    LinearConverter models equipment that transforms one or more input flows into one or
    more output flows through linear relationships. This includes heat exchangers,
    electrical converters, chemical reactors, and other equipment where the
    relationship between inputs and outputs can be expressed as linear equations.

    The component supports two modeling approaches: simple conversion factors for
    straightforward linear relationships, or piecewise conversion for complex non-linear
    behavior approximated through piecewise linear segments.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/elements/LinearConverter/>

    Args:
        label: The label of the Element. Used to identify it in the FlowSystem.
        inputs: list of input Flows that feed into the converter.
        outputs: list of output Flows that are produced by the converter.
        status_parameters: Information about active and inactive state of LinearConverter.
            Component is active/inactive if all connected Flows are active/inactive. This induces a
            status variable (binary) in all Flows! If possible, use StatusParameters in a
            single Flow instead to keep the number of binary variables low.
        conversion_factors: Linear relationships between flows expressed as a list of
            dictionaries. Each dictionary maps flow labels to their coefficients in one
            linear equation. The number of conversion factors must be less than the total
            number of flows to ensure degrees of freedom > 0. Either 'conversion_factors'
            OR 'piecewise_conversion' can be used, but not both.
            For examples also look into the linear_converters.py file.
        piecewise_conversion: Define piecewise linear relationships between flow rates
            of different flows. Enables modeling of non-linear conversion behavior through
            linear approximation. Either 'conversion_factors' or 'piecewise_conversion'
            can be used, but not both.
        meta_data: Used to store additional information about the Element. Not used
            internally, but saved in results. Only use Python native types.

    Examples:
        Simple 1:1 heat exchanger with 95% efficiency:

        ```python
        heat_exchanger = LinearConverter(
            label='primary_hx',
            inputs=[hot_water_in],
            outputs=[hot_water_out],
            conversion_factors=[{'hot_water_in': 0.95, 'hot_water_out': 1}],
        )
        ```

        Multi-input heat pump with COP=3:

        ```python
        heat_pump = LinearConverter(
            label='air_source_hp',
            inputs=[electricity_in],
            outputs=[heat_output],
            conversion_factors=[{'electricity_in': 3, 'heat_output': 1}],
        )
        ```

        Combined heat and power (CHP) unit with multiple outputs:

        ```python
        chp_unit = LinearConverter(
            label='gas_chp',
            inputs=[natural_gas],
            outputs=[electricity_out, heat_out],
            conversion_factors=[
                {'natural_gas': 0.35, 'electricity_out': 1},
                {'natural_gas': 0.45, 'heat_out': 1},
            ],
        )
        ```

        Electrolyzer with multiple conversion relationships:

        ```python
        electrolyzer = LinearConverter(
            label='pem_electrolyzer',
            inputs=[electricity_in, water_in],
            outputs=[hydrogen_out, oxygen_out],
            conversion_factors=[
                {'electricity_in': 1, 'hydrogen_out': 50},  # 50 kWh/kg H2
                {'water_in': 1, 'hydrogen_out': 9},  # 9 kg H2O/kg H2
                {'hydrogen_out': 8, 'oxygen_out': 1},  # Mass balance
            ],
        )
        ```

        Complex converter with piecewise efficiency:

        ```python
        variable_efficiency_converter = LinearConverter(
            label='variable_converter',
            inputs=[fuel_in],
            outputs=[power_out],
            piecewise_conversion=PiecewiseConversion(
                {
                    'fuel_in': Piecewise(
                        [
                            Piece(0, 10),  # Low load operation
                            Piece(10, 25),  # High load operation
                        ]
                    ),
                    'power_out': Piecewise(
                        [
                            Piece(0, 3.5),  # Lower efficiency at part load
                            Piece(3.5, 10),  # Higher efficiency at full load
                        ]
                    ),
                }
            ),
        )
        ```

    Note:
        Conversion factors define linear relationships where the sum of (coefficient × flow_rate)
        equals zero for each equation: factor1×flow1 + factor2×flow2 + ... = 0
        Conversion factors define linear relationships:
        `{flow1: a1, flow2: a2, ...}` yields `a1×flow_rate1 + a2×flow_rate2 + ... = 0`.
        Note: The input format may be unintuitive. For example,
        `{"electricity": 1, "H2": 50}` implies `1×electricity = 50×H2`,
        i.e., 50 units of electricity produce 1 unit of H2.

        The system must have fewer conversion factors than total flows (degrees of freedom > 0)
        to avoid over-constraining the problem. For n total flows, use at most n-1 conversion factors.

        When using piecewise_conversion, the converter operates on one piece at a time,
        with binary variables determining which piece is active.

    """

    submodel: LinearConverterModel | None

    def __init__(
        self,
        label: str,
        inputs: list[Flow],
        outputs: list[Flow],
        status_parameters: StatusParameters | None = None,
        conversion_factors: list[dict[str, Numeric_TPS]] | None = None,
        piecewise_conversion: PiecewiseConversion | None = None,
        meta_data: dict | None = None,
        color: str | None = None,
    ):
        super().__init__(label, inputs, outputs, status_parameters, meta_data=meta_data, color=color)
        self.conversion_factors = conversion_factors or []
        self.piecewise_conversion = piecewise_conversion

    def create_model(self, model: FlowSystemModel) -> LinearConverterModel:
        self._plausibility_checks()
        self.submodel = LinearConverterModel(model, self)
        return self.submodel

    def link_to_flow_system(self, flow_system, prefix: str = '') -> None:
        """Propagate flow_system reference to parent Component and piecewise_conversion."""
        super().link_to_flow_system(flow_system, prefix)
        if self.piecewise_conversion is not None:
            self.piecewise_conversion.link_to_flow_system(flow_system, self._sub_prefix('PiecewiseConversion'))

    def _plausibility_checks(self) -> None:
        super()._plausibility_checks()
        if not self.conversion_factors and not self.piecewise_conversion:
            raise PlausibilityError('Either conversion_factors or piecewise_conversion must be defined!')
        if self.conversion_factors and self.piecewise_conversion:
            raise PlausibilityError('Only one of conversion_factors or piecewise_conversion can be defined, not both!')

        if self.conversion_factors:
            if self.degrees_of_freedom <= 0:
                raise PlausibilityError(
                    f'Too Many conversion_factors_specified. Care that you use less conversion_factors '
                    f'then inputs + outputs!! With {len(self.inputs + self.outputs)} inputs and outputs, '
                    f'use not more than {len(self.inputs + self.outputs) - 1} conversion_factors!'
                )

            for conversion_factor in self.conversion_factors:
                for flow in conversion_factor:
                    if flow not in self.flows:
                        raise PlausibilityError(
                            f'{self.label}: Flow {flow} in conversion_factors is not in inputs/outputs'
                        )
        if self.piecewise_conversion:
            for flow in self.flows.values():
                if isinstance(flow.size, InvestParameters) and flow.size.fixed_size is None:
                    logger.warning(
                        f'Using a Flow with variable size (InvestParameters without fixed_size) '
                        f'and a piecewise_conversion in {self.label_full} is uncommon. Please verify intent '
                        f'({flow.label_full}).'
                    )

    def transform_data(self) -> None:
        super().transform_data()
        if self.conversion_factors:
            self.conversion_factors = self._transform_conversion_factors()
        if self.piecewise_conversion:
            self.piecewise_conversion.has_time_dim = True
            self.piecewise_conversion.transform_data()

    def _transform_conversion_factors(self) -> list[dict[str, xr.DataArray]]:
        """Converts all conversion factors to internal datatypes"""
        list_of_conversion_factors = []
        for idx, conversion_factor in enumerate(self.conversion_factors):
            transformed_dict = {}
            for flow, values in conversion_factor.items():
                # TODO: Might be better to use the label of the component instead of the flow
                ts = self._fit_coords(f'{self.flows[flow].label_full}|conversion_factor{idx}', values)
                if ts is None:
                    raise PlausibilityError(f'{self.label_full}: conversion factor for flow "{flow}" must not be None')
                transformed_dict[flow] = ts
            list_of_conversion_factors.append(transformed_dict)
        return list_of_conversion_factors

    @property
    def degrees_of_freedom(self):
        return len(self.inputs + self.outputs) - len(self.conversion_factors)


@register_class_for_io
class Storage(Component):
    """
    A Storage models the temporary storage and release of energy or material.

    Storages have one incoming and one outgoing Flow, each with configurable efficiency
    factors. They maintain a charge state variable that represents the stored amount,
    bounded by capacity limits and evolving over time based on charging, discharging,
    and self-discharge losses.

    The storage model handles complex temporal dynamics including initial conditions,
    final state constraints, and time-varying parameters. It supports both fixed-size
    and investment-optimized storage systems with comprehensive techno-economic modeling.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/elements/Storage/>

    Args:
        label: Element identifier used in the FlowSystem.
        charging: Incoming flow for loading the storage.
        discharging: Outgoing flow for unloading the storage.
        capacity_in_flow_hours: Storage capacity in flow-hours (kWh, m³, kg).
            Scalar for fixed size, InvestParameters for optimization, or None (unbounded).
            Default: None (unbounded capacity). When using InvestParameters,
            maximum_size (or fixed_size) must be explicitly set for proper model scaling.
        relative_minimum_charge_state: Minimum charge state (0-1). Default: 0.
        relative_maximum_charge_state: Maximum charge state (0-1). Default: 1.
        initial_charge_state: Charge at start. Numeric, 'equals_final', or None (free). Default: 0.
        minimal_final_charge_state: Minimum absolute charge required at end (optional).
        maximal_final_charge_state: Maximum absolute charge allowed at end (optional).
        relative_minimum_final_charge_state: Minimum relative charge at end.
            Defaults to last value of relative_minimum_charge_state.
        relative_maximum_final_charge_state: Maximum relative charge at end.
            Defaults to last value of relative_maximum_charge_state.
        eta_charge: Charging efficiency (0-1). Default: 1.
        eta_discharge: Discharging efficiency (0-1). Default: 1.
        relative_loss_per_hour: Self-discharge per hour (0-0.1). Default: 0.
        prevent_simultaneous_charge_and_discharge: Prevent charging and discharging
            simultaneously. Adds binary variables. Default: True.
        cluster_mode: How this storage is treated during clustering optimization.
            Only relevant when using ``transform.cluster()``. Options:

            - ``'independent'``: Clusters are fully decoupled. No constraints between
              clusters, each cluster has free start/end SOC. Fast but ignores
              seasonal storage value.
            - ``'cyclic'``: Each cluster is self-contained. The SOC at the start of
              each cluster equals its end (cluster returns to initial state).
              Good for "average day" modeling.
            - ``'intercluster'``: Link storage state across the original timeline using
              SOC boundary variables (Kotzur et al. approach). Properly values
              seasonal storage patterns. Overall SOC can drift.
            - ``'intercluster_cyclic'`` (default): Like 'intercluster' but also enforces
              that overall SOC returns to initial state (yearly cyclic).

        meta_data: Additional information stored in results. Python native types only.

    Examples:
        Battery energy storage system:

        ```python
        battery = Storage(
            label='lithium_battery',
            charging=battery_charge_flow,
            discharging=battery_discharge_flow,
            capacity_in_flow_hours=100,  # 100 kWh capacity
            eta_charge=0.95,  # 95% charging efficiency
            eta_discharge=0.95,  # 95% discharging efficiency
            relative_loss_per_hour=0.001,  # 0.1% loss per hour
            relative_minimum_charge_state=0.1,  # Never below 10% SOC
            relative_maximum_charge_state=0.9,  # Never above 90% SOC
        )
        ```

        Thermal storage with cycling constraints:

        ```python
        thermal_storage = Storage(
            label='hot_water_tank',
            charging=heat_input,
            discharging=heat_output,
            capacity_in_flow_hours=500,  # 500 kWh thermal capacity
            initial_charge_state=250,  # Start half full
            # Impact of temperature on energy capacity
            relative_maximum_charge_state=water_temperature_spread / rated_temeprature_spread,
            eta_charge=0.90,  # Heat exchanger losses
            eta_discharge=0.85,  # Distribution losses
            relative_loss_per_hour=0.02,  # 2% thermal loss per hour
            prevent_simultaneous_charge_and_discharge=True,
        )
        ```

        Pumped hydro storage with investment optimization:

        ```python
        pumped_hydro = Storage(
            label='pumped_hydro',
            charging=pump_flow,
            discharging=turbine_flow,
            capacity_in_flow_hours=InvestParameters(
                minimum_size=1000,  # Minimum economic scale
                maximum_size=10000,  # Site constraints
                specific_effects={'cost': 150},  # €150/MWh capacity
                fix_effects={'cost': 50_000_000},  # €50M fixed costs
            ),
            eta_charge=0.85,  # Pumping efficiency
            eta_discharge=0.90,  # Turbine efficiency
            initial_charge_state='equals_final',  # Ensuring no deficit compared to start
            relative_loss_per_hour=0.0001,  # Minimal evaporation
        )
        ```

        Material storage with inventory management:

        ```python
        fuel_storage = Storage(
            label='natural_gas_storage',
            charging=gas_injection,
            discharging=gas_withdrawal,
            capacity_in_flow_hours=10000,  # 10,000 m³ storage volume
            initial_charge_state=3000,  # Start with 3,000 m³
            minimal_final_charge_state=1000,  # Strategic reserve
            maximal_final_charge_state=9000,  # Prevent overflow
            eta_charge=0.98,  # Compression losses
            eta_discharge=0.95,  # Pressure reduction losses
            relative_loss_per_hour=0.0005,  # 0.05% leakage per hour
            prevent_simultaneous_charge_and_discharge=False,  # Allow flow-through
        )
        ```

    Note:
        **Mathematical formulation**: See [Storage](../user-guide/mathematical-notation/elements/Storage.md)
        for charge state evolution equations and balance constraints.

        **Efficiency parameters** (eta_charge, eta_discharge) are dimensionless (0-1 range).
        The relative_loss_per_hour represents exponential decay per hour.

        **Binary variables**: When prevent_simultaneous_charge_and_discharge is True, binary
        variables enforce mutual exclusivity, increasing solution time but preventing unrealistic
        simultaneous charging and discharging.

        **Unbounded capacity**: When capacity_in_flow_hours is None (default), the storage has
        unlimited capacity. Note that prevent_simultaneous_charge_and_discharge requires the
        charging and discharging flows to have explicit sizes. Use prevent_simultaneous_charge_and_discharge=False
        with unbounded storages, or set flow sizes explicitly.

        **Units**: Flow rates and charge states are related by the concept of 'flow hours' (=flow_rate * time).
        With flow rates in kW, the charge state is therefore (usually) kWh.
        With flow rates in m3/h, the charge state is therefore in m3.
    """

    submodel: StorageModel | None

    def __init__(
        self,
        label: str,
        charging: Flow,
        discharging: Flow,
        capacity_in_flow_hours: Numeric_PS | InvestParameters | None = None,
        relative_minimum_charge_state: Numeric_TPS = 0,
        relative_maximum_charge_state: Numeric_TPS = 1,
        initial_charge_state: Numeric_PS | Literal['equals_final'] | None = 0,
        minimal_final_charge_state: Numeric_PS | None = None,
        maximal_final_charge_state: Numeric_PS | None = None,
        relative_minimum_final_charge_state: Numeric_PS | None = None,
        relative_maximum_final_charge_state: Numeric_PS | None = None,
        eta_charge: Numeric_TPS = 1,
        eta_discharge: Numeric_TPS = 1,
        relative_loss_per_hour: Numeric_TPS = 0,
        prevent_simultaneous_charge_and_discharge: bool = True,
        balanced: bool = False,
        cluster_mode: Literal['independent', 'cyclic', 'intercluster', 'intercluster_cyclic'] = 'intercluster_cyclic',
        meta_data: dict | None = None,
        color: str | None = None,
    ):
        # TODO: fixed_relative_chargeState implementieren
        super().__init__(
            label,
            inputs=[charging],
            outputs=[discharging],
            prevent_simultaneous_flows=[charging, discharging] if prevent_simultaneous_charge_and_discharge else None,
            meta_data=meta_data,
            color=color,
        )

        self.charging = charging
        self.discharging = discharging
        self.capacity_in_flow_hours = capacity_in_flow_hours
        self.relative_minimum_charge_state: Numeric_TPS = relative_minimum_charge_state
        self.relative_maximum_charge_state: Numeric_TPS = relative_maximum_charge_state

        self.relative_minimum_final_charge_state = relative_minimum_final_charge_state
        self.relative_maximum_final_charge_state = relative_maximum_final_charge_state

        self.initial_charge_state = initial_charge_state
        self.minimal_final_charge_state = minimal_final_charge_state
        self.maximal_final_charge_state = maximal_final_charge_state

        self.eta_charge: Numeric_TPS = eta_charge
        self.eta_discharge: Numeric_TPS = eta_discharge
        self.relative_loss_per_hour: Numeric_TPS = relative_loss_per_hour
        self.prevent_simultaneous_charge_and_discharge = prevent_simultaneous_charge_and_discharge
        self.balanced = balanced
        self.cluster_mode = cluster_mode

    def create_model(self, model: FlowSystemModel) -> StorageModel:
        """Create the appropriate storage model based on cluster_mode and flow system state.

        For intercluster modes ('intercluster', 'intercluster_cyclic'), uses
        :class:`InterclusterStorageModel` which implements S-N linking.
        For other modes, uses the base :class:`StorageModel`.

        Args:
            model: The FlowSystemModel to add constraints to.

        Returns:
            StorageModel or InterclusterStorageModel instance.
        """
        self._plausibility_checks()

        # Use InterclusterStorageModel for intercluster modes when clustering is active
        clustering = model.flow_system.clustering
        is_intercluster = clustering is not None and self.cluster_mode in (
            'intercluster',
            'intercluster_cyclic',
        )

        if is_intercluster:
            self.submodel = InterclusterStorageModel(model, self)
        else:
            self.submodel = StorageModel(model, self)

        return self.submodel

    def link_to_flow_system(self, flow_system, prefix: str = '') -> None:
        """Propagate flow_system reference to parent Component and capacity_in_flow_hours if it's InvestParameters."""
        super().link_to_flow_system(flow_system, prefix)
        if isinstance(self.capacity_in_flow_hours, InvestParameters):
            self.capacity_in_flow_hours.link_to_flow_system(flow_system, self._sub_prefix('InvestParameters'))

    def transform_data(self) -> None:
        super().transform_data()
        self.relative_minimum_charge_state = self._fit_coords(
            f'{self.prefix}|relative_minimum_charge_state', self.relative_minimum_charge_state
        )
        self.relative_maximum_charge_state = self._fit_coords(
            f'{self.prefix}|relative_maximum_charge_state', self.relative_maximum_charge_state
        )
        self.eta_charge = self._fit_coords(f'{self.prefix}|eta_charge', self.eta_charge)
        self.eta_discharge = self._fit_coords(f'{self.prefix}|eta_discharge', self.eta_discharge)
        self.relative_loss_per_hour = self._fit_coords(
            f'{self.prefix}|relative_loss_per_hour', self.relative_loss_per_hour
        )
        if self.initial_charge_state is not None and not isinstance(self.initial_charge_state, str):
            self.initial_charge_state = self._fit_coords(
                f'{self.prefix}|initial_charge_state', self.initial_charge_state, dims=['period', 'scenario']
            )
        self.minimal_final_charge_state = self._fit_coords(
            f'{self.prefix}|minimal_final_charge_state', self.minimal_final_charge_state, dims=['period', 'scenario']
        )
        self.maximal_final_charge_state = self._fit_coords(
            f'{self.prefix}|maximal_final_charge_state', self.maximal_final_charge_state, dims=['period', 'scenario']
        )
        self.relative_minimum_final_charge_state = self._fit_coords(
            f'{self.prefix}|relative_minimum_final_charge_state',
            self.relative_minimum_final_charge_state,
            dims=['period', 'scenario'],
        )
        self.relative_maximum_final_charge_state = self._fit_coords(
            f'{self.prefix}|relative_maximum_final_charge_state',
            self.relative_maximum_final_charge_state,
            dims=['period', 'scenario'],
        )
        if isinstance(self.capacity_in_flow_hours, InvestParameters):
            self.capacity_in_flow_hours.transform_data()
        else:
            self.capacity_in_flow_hours = self._fit_coords(
                f'{self.prefix}|capacity_in_flow_hours', self.capacity_in_flow_hours, dims=['period', 'scenario']
            )

    def _plausibility_checks(self) -> None:
        """
        Check for infeasible or uncommon combinations of parameters
        """
        super()._plausibility_checks()

        # Validate string values and set flag
        initial_equals_final = False
        if isinstance(self.initial_charge_state, str):
            if not self.initial_charge_state == 'equals_final':
                raise PlausibilityError(f'initial_charge_state has undefined value: {self.initial_charge_state}')
            initial_equals_final = True

        # Capacity is required when using non-default relative bounds
        if self.capacity_in_flow_hours is None:
            if np.any(self.relative_minimum_charge_state > 0):
                raise PlausibilityError(
                    f'Storage "{self.label_full}" has relative_minimum_charge_state > 0 but no capacity_in_flow_hours. '
                    f'A capacity is required because the lower bound is capacity * relative_minimum_charge_state.'
                )
            if np.any(self.relative_maximum_charge_state < 1):
                raise PlausibilityError(
                    f'Storage "{self.label_full}" has relative_maximum_charge_state < 1 but no capacity_in_flow_hours. '
                    f'A capacity is required because the upper bound is capacity * relative_maximum_charge_state.'
                )
            if self.relative_minimum_final_charge_state is not None:
                raise PlausibilityError(
                    f'Storage "{self.label_full}" has relative_minimum_final_charge_state but no capacity_in_flow_hours. '
                    f'A capacity is required for relative final charge state constraints.'
                )
            if self.relative_maximum_final_charge_state is not None:
                raise PlausibilityError(
                    f'Storage "{self.label_full}" has relative_maximum_final_charge_state but no capacity_in_flow_hours. '
                    f'A capacity is required for relative final charge state constraints.'
                )

        # Skip capacity-related checks if capacity is None (unbounded)
        if self.capacity_in_flow_hours is not None:
            # Use new InvestParameters methods to get capacity bounds
            if isinstance(self.capacity_in_flow_hours, InvestParameters):
                minimum_capacity = self.capacity_in_flow_hours.minimum_or_fixed_size
                maximum_capacity = self.capacity_in_flow_hours.maximum_or_fixed_size
            else:
                maximum_capacity = self.capacity_in_flow_hours
                minimum_capacity = self.capacity_in_flow_hours

            # Initial charge state should not constrain investment decision
            # If initial > (min_cap * rel_max), investment is forced to increase capacity
            # If initial < (max_cap * rel_min), investment is forced to decrease capacity
            min_initial_at_max_capacity = maximum_capacity * _scalar_safe_isel(
                self.relative_minimum_charge_state, {'time': 0}
            )
            max_initial_at_min_capacity = minimum_capacity * _scalar_safe_isel(
                self.relative_maximum_charge_state, {'time': 0}
            )

            # Only perform numeric comparisons if using a numeric initial_charge_state
            if not initial_equals_final and self.initial_charge_state is not None:
                if (self.initial_charge_state > max_initial_at_min_capacity).any():
                    raise PlausibilityError(
                        f'{self.label_full}: {self.initial_charge_state=} '
                        f'is constraining the investment decision. Choose a value <= {max_initial_at_min_capacity}.'
                    )
                if (self.initial_charge_state < min_initial_at_max_capacity).any():
                    raise PlausibilityError(
                        f'{self.label_full}: {self.initial_charge_state=} '
                        f'is constraining the investment decision. Choose a value >= {min_initial_at_max_capacity}.'
                    )

        if self.balanced:
            if not isinstance(self.charging.size, InvestParameters) or not isinstance(
                self.discharging.size, InvestParameters
            ):
                raise PlausibilityError(
                    f'Balancing charging and discharging Flows in {self.label_full} is only possible with Investments.'
                )

            if (self.charging.size.minimum_or_fixed_size > self.discharging.size.maximum_or_fixed_size).any() or (
                self.charging.size.maximum_or_fixed_size < self.discharging.size.minimum_or_fixed_size
            ).any():
                raise PlausibilityError(
                    f'Balancing charging and discharging Flows in {self.label_full} need compatible minimum and maximum sizes.'
                    f'Got: {self.charging.size.minimum_or_fixed_size=}, {self.charging.size.maximum_or_fixed_size=} and '
                    f'{self.discharging.size.minimum_or_fixed_size=}, {self.discharging.size.maximum_or_fixed_size=}.'
                )

    def __repr__(self) -> str:
        """Return string representation."""
        # Use build_repr_from_init directly to exclude charging and discharging
        return fx_io.build_repr_from_init(
            self,
            excluded_params={'self', 'label', 'charging', 'discharging', 'kwargs'},
            skip_default_size=True,
        ) + fx_io.format_flow_details(self)


@register_class_for_io
class Transmission(Component):
    """
    Models transmission infrastructure that transports flows between two locations with losses.

    Transmission components represent physical infrastructure like pipes, cables,
    transmission lines, or conveyor systems that transport energy or materials between
    two points. They can model both unidirectional and bidirectional flow with
    configurable loss mechanisms and operational constraints.

    The component supports complex transmission scenarios including relative losses
    (proportional to flow), absolute losses (fixed when active), and bidirectional
    operation with flow direction constraints.

    Args:
        label: The label of the Element. Used to identify it in the FlowSystem.
        in1: The primary inflow (side A). Pass InvestParameters here for capacity optimization.
        out1: The primary outflow (side B).
        in2: Optional secondary inflow (side B) for bidirectional operation.
            If in1 has InvestParameters, in2 will automatically have matching capacity.
        out2: Optional secondary outflow (side A) for bidirectional operation.
        relative_losses: Proportional losses as fraction of throughput (e.g., 0.02 for 2% loss).
            Applied as: output = input × (1 - relative_losses)
        absolute_losses: Fixed losses that occur when transmission is active.
            Automatically creates binary variables for active/inactive states.
        status_parameters: Parameters defining binary operation constraints and costs.
        prevent_simultaneous_flows_in_both_directions: If True, prevents simultaneous
            flow in both directions. Increases binary variables but reflects physical
            reality for most transmission systems. Default is True.
        balanced: Whether to equate the size of the in1 and in2 Flow. Needs InvestParameters in both Flows.
        meta_data: Used to store additional information. Not used internally but saved
            in results. Only use Python native types.

    Examples:
        Simple electrical transmission line:

        ```python
        power_line = Transmission(
            label='110kv_line',
            in1=substation_a_out,
            out1=substation_b_in,
            relative_losses=0.03,  # 3% line losses
        )
        ```

        Bidirectional natural gas pipeline:

        ```python
        gas_pipeline = Transmission(
            label='interstate_pipeline',
            in1=compressor_station_a,
            out1=distribution_hub_b,
            in2=compressor_station_b,
            out2=distribution_hub_a,
            relative_losses=0.005,  # 0.5% friction losses
            absolute_losses=50,  # 50 kW compressor power when active
            prevent_simultaneous_flows_in_both_directions=True,
        )
        ```

        District heating network with investment optimization:

        ```python
        heating_network = Transmission(
            label='dh_main_line',
            in1=Flow(
                label='heat_supply',
                bus=central_plant_bus,
                size=InvestParameters(
                    minimum_size=1000,  # Minimum 1 MW capacity
                    maximum_size=10000,  # Maximum 10 MW capacity
                    specific_effects={'cost': 200},  # €200/kW capacity
                    fix_effects={'cost': 500000},  # €500k fixed installation
                ),
            ),
            out1=district_heat_demand,
            relative_losses=0.15,  # 15% thermal losses in distribution
        )
        ```

        Material conveyor with active/inactive status:

        ```python
        conveyor_belt = Transmission(
            label='material_transport',
            in1=loading_station,
            out1=unloading_station,
            absolute_losses=25,  # 25 kW motor power when running
            status_parameters=StatusParameters(
                effects_per_startup={'maintenance': 0.1},
                min_uptime=2,  # Minimum 2-hour operation
                startup_limit=10,  # Maximum 10 starts per period
            ),
        )
        ```

    Note:
        The transmission equation balances flows with losses:
        output_flow = input_flow × (1 - relative_losses) - absolute_losses

        For bidirectional transmission, each direction has independent loss calculations.

        When using InvestParameters on in1, the capacity automatically applies to in2
        to maintain consistent bidirectional capacity without additional investment variables.

        Absolute losses force the creation of binary on/inactive variables, which increases
        computational complexity but enables realistic modeling of equipment with
        standby power consumption.

    """

    submodel: TransmissionModel | None

    def __init__(
        self,
        label: str,
        in1: Flow,
        out1: Flow,
        in2: Flow | None = None,
        out2: Flow | None = None,
        relative_losses: Numeric_TPS | None = None,
        absolute_losses: Numeric_TPS | None = None,
        status_parameters: StatusParameters | None = None,
        prevent_simultaneous_flows_in_both_directions: bool = True,
        balanced: bool = False,
        meta_data: dict | None = None,
        color: str | None = None,
    ):
        super().__init__(
            label,
            inputs=[flow for flow in (in1, in2) if flow is not None],
            outputs=[flow for flow in (out1, out2) if flow is not None],
            status_parameters=status_parameters,
            prevent_simultaneous_flows=None
            if in2 is None or prevent_simultaneous_flows_in_both_directions is False
            else [in1, in2],
            meta_data=meta_data,
            color=color,
        )
        self.in1 = in1
        self.out1 = out1
        self.in2 = in2
        self.out2 = out2

        self.relative_losses = relative_losses
        self.absolute_losses = absolute_losses
        self.balanced = balanced

    def _plausibility_checks(self):
        super()._plausibility_checks()
        # check buses:
        if self.in2 is not None:
            assert self.in2.bus == self.out1.bus, (
                f'Output 1 and Input 2 do not start/end at the same Bus: {self.out1.bus=}, {self.in2.bus=}'
            )
        if self.out2 is not None:
            assert self.out2.bus == self.in1.bus, (
                f'Input 1 and Output 2 do not start/end at the same Bus: {self.in1.bus=}, {self.out2.bus=}'
            )

        if self.balanced:
            if self.in2 is None:
                raise ValueError('Balanced Transmission needs InvestParameters in both in-Flows')
            if not isinstance(self.in1.size, InvestParameters) or not isinstance(self.in2.size, InvestParameters):
                raise ValueError('Balanced Transmission needs InvestParameters in both in-Flows')
            if (self.in1.size.minimum_or_fixed_size > self.in2.size.maximum_or_fixed_size).any() or (
                self.in1.size.maximum_or_fixed_size < self.in2.size.minimum_or_fixed_size
            ).any():
                raise ValueError(
                    f'Balanced Transmission needs compatible minimum and maximum sizes.'
                    f'Got: {self.in1.size.minimum_or_fixed_size=}, {self.in1.size.maximum_or_fixed_size=} and '
                    f'{self.in2.size.minimum_or_fixed_size=}, {self.in2.size.maximum_or_fixed_size=}.'
                )

    def create_model(self, model) -> TransmissionModel:
        self._plausibility_checks()
        self.submodel = TransmissionModel(model, self)
        return self.submodel

    def transform_data(self) -> None:
        super().transform_data()
        self.relative_losses = self._fit_coords(f'{self.prefix}|relative_losses', self.relative_losses)
        self.absolute_losses = self._fit_coords(f'{self.prefix}|absolute_losses', self.absolute_losses)


class TransmissionModel(ComponentModel):
    element: Transmission

    def __init__(self, model: FlowSystemModel, element: Transmission):
        if (element.absolute_losses is not None) and np.any(element.absolute_losses != 0):
            for flow in (element.inputs + element.outputs).values():
                if flow.status_parameters is None:
                    flow.status_parameters = StatusParameters()
                    flow.status_parameters.link_to_flow_system(
                        model.flow_system, f'{flow.label_full}|status_parameters'
                    )

        super().__init__(model, element)

    def _do_modeling(self):
        """Create transmission efficiency equations and optional absolute loss constraints for both flow directions"""
        super()._do_modeling()

        # first direction
        self.create_transmission_equation('dir1', self.element.in1, self.element.out1)

        # second direction:
        if self.element.in2 is not None:
            self.create_transmission_equation('dir2', self.element.in2, self.element.out2)

        # equate size of both directions
        if self.element.balanced:
            # eq: in1.size = in2.size
            self.add_constraints(
                self.element.in1.submodel._investment.size == self.element.in2.submodel._investment.size,
                short_name='same_size',
            )

    def create_transmission_equation(self, name: str, in_flow: Flow, out_flow: Flow) -> linopy.Constraint:
        """Creates an Equation for the Transmission efficiency and adds it to the model"""
        # eq: out(t) + on(t)*loss_abs(t) = in(t)*(1 - loss_rel(t))
        rel_losses = 0 if self.element.relative_losses is None else self.element.relative_losses
        con_transmission = self.add_constraints(
            out_flow.submodel.flow_rate == in_flow.submodel.flow_rate * (1 - rel_losses),
            short_name=name,
        )

        if (self.element.absolute_losses is not None) and np.any(self.element.absolute_losses != 0):
            con_transmission.lhs += in_flow.submodel.status.status * self.element.absolute_losses

        return con_transmission


class LinearConverterModel(ComponentModel):
    """Mathematical model implementation for LinearConverter components.

    Creates optimization constraints for linear conversion relationships between
    input and output flows, supporting both simple conversion factors and piecewise
    non-linear approximations.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/elements/LinearConverter/>
    """

    element: LinearConverter

    def __init__(self, model: FlowSystemModel, element: LinearConverter):
        self.piecewise_conversion: PiecewiseConversion | None = None
        super().__init__(model, element)

    def _do_modeling(self):
        """Create linear conversion equations or piecewise conversion constraints between input and output flows"""
        super()._do_modeling()

        # Create conversion factor constraints if specified
        if self.element.conversion_factors:
            all_input_flows = set(self.element.inputs.values())
            all_output_flows = set(self.element.outputs.values())

            # für alle linearen Gleichungen:
            for i, conv_factors in enumerate(self.element.conversion_factors):
                used_flows = set([self.element.flows[flow_label] for flow_label in conv_factors])
                used_inputs: set[Flow] = all_input_flows & used_flows
                used_outputs: set[Flow] = all_output_flows & used_flows

                self.add_constraints(
                    sum([flow.submodel.flow_rate * conv_factors[flow.label] for flow in used_inputs])
                    == sum([flow.submodel.flow_rate * conv_factors[flow.label] for flow in used_outputs]),
                    short_name=f'conversion_{i}',
                )

        else:
            # TODO: Improve Inclusion of StatusParameters. Instead of creating a Binary in every flow, the binary could only be part of the Piece itself
            piecewise_conversion = {
                self.element.flows[flow].submodel.flow_rate.name: piecewise
                for flow, piecewise in self.element.piecewise_conversion.items()
            }

            self.piecewise_conversion = self.add_submodels(
                PiecewiseModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    label_of_model=f'{self.label_of_element}',
                    piecewise_variables=piecewise_conversion,
                    zero_point=self.status.status if self.status is not None else False,
                    dims=('time', 'period', 'scenario'),
                ),
                short_name='PiecewiseConversion',
            )


class StorageModel(ComponentModel):
    """Mathematical model implementation for Storage components.

    Creates optimization variables and constraints for charge state tracking,
    storage balance equations, and optional investment sizing.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/elements/Storage/>

    Note:
        This class uses a template method pattern. Subclasses (e.g., InterclusterStorageModel)
        can override individual methods to customize behavior without duplicating code.
    """

    element: Storage

    def __init__(self, model: FlowSystemModel, element: Storage):
        super().__init__(model, element)

    def _do_modeling(self):
        """Create charge state variables, energy balance equations, and optional investment submodels."""
        super()._do_modeling()
        self._create_storage_variables()
        self._add_netto_discharge_constraint()
        self._add_energy_balance_constraint()
        self._add_cluster_cyclic_constraint()
        self._add_investment_model()
        self._add_initial_final_constraints()
        self._add_balanced_sizes_constraint()

    def _create_storage_variables(self):
        """Create charge_state and netto_discharge variables."""
        lb, ub = self._absolute_charge_state_bounds
        self.add_variables(
            lower=lb,
            upper=ub,
            coords=self._model.get_coords(extra_timestep=True),
            short_name='charge_state',
            category=VariableCategory.CHARGE_STATE,
        )
        self.add_variables(
            coords=self._model.get_coords(),
            short_name='netto_discharge',
            category=VariableCategory.NETTO_DISCHARGE,
        )

    def _add_netto_discharge_constraint(self):
        """Add constraint: netto_discharge = discharging - charging."""
        self.add_constraints(
            self.netto_discharge
            == self.element.discharging.submodel.flow_rate - self.element.charging.submodel.flow_rate,
            short_name='netto_discharge',
        )

    def _add_energy_balance_constraint(self):
        """Add energy balance constraint linking charge states across timesteps."""
        self.add_constraints(self._build_energy_balance_lhs() == 0, short_name='charge_state')

    def _add_cluster_cyclic_constraint(self):
        """For 'cyclic' cluster mode: each cluster's start equals its end."""
        if self._model.flow_system.clusters is not None and self.element.cluster_mode == 'cyclic':
            self.add_constraints(
                self.charge_state.isel(time=0) == self.charge_state.isel(time=-2),
                short_name='cluster_cyclic',
            )

    def _add_investment_model(self):
        """Create InvestmentModel and add capacity-scaled bounds if using investment sizing."""
        if isinstance(self.element.capacity_in_flow_hours, InvestParameters):
            self.add_submodels(
                InvestmentModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    label_of_model=self.label_of_element,
                    parameters=self.element.capacity_in_flow_hours,
                    size_category=VariableCategory.STORAGE_SIZE,
                ),
                short_name='investment',
            )
            BoundingPatterns.scaled_bounds(
                self,
                variable=self.charge_state,
                scaling_variable=self.investment.size,
                relative_bounds=self._relative_charge_state_bounds,
            )

    def _add_initial_final_constraints(self):
        """Add initial and final charge state constraints.

        For clustered systems with 'independent' or 'cyclic' mode, these constraints
        are skipped because:
        - 'independent': Each cluster has free start/end SOC
        - 'cyclic': Start == end is handled by _add_cluster_cyclic_constraint,
          but no specific initial value is enforced
        """
        # Skip initial/final constraints for clustered systems with independent/cyclic mode
        # These modes should have free or cyclic SOC, not a fixed initial value per cluster
        if self._model.flow_system.clusters is not None and self.element.cluster_mode in (
            'independent',
            'cyclic',
        ):
            return

        if self.element.initial_charge_state is not None:
            if isinstance(self.element.initial_charge_state, str):
                self.add_constraints(
                    self.charge_state.isel(time=0) == self.charge_state.isel(time=-1),
                    short_name='initial_charge_state',
                )
            else:
                self.add_constraints(
                    self.charge_state.isel(time=0) == self.element.initial_charge_state,
                    short_name='initial_charge_state',
                )

        if self.element.maximal_final_charge_state is not None:
            self.add_constraints(
                self.charge_state.isel(time=-1) <= self.element.maximal_final_charge_state,
                short_name='final_charge_max',
            )

        if self.element.minimal_final_charge_state is not None:
            self.add_constraints(
                self.charge_state.isel(time=-1) >= self.element.minimal_final_charge_state,
                short_name='final_charge_min',
            )

    def _add_balanced_sizes_constraint(self):
        """Add constraint ensuring charging and discharging capacities are equal."""
        if self.element.balanced:
            self.add_constraints(
                self.element.charging.submodel._investment.size - self.element.discharging.submodel._investment.size
                == 0,
                short_name='balanced_sizes',
            )

    def _build_energy_balance_lhs(self):
        """Build the left-hand side of the energy balance constraint.

        The energy balance equation is:
            charge_state[t+1] = charge_state[t] * (1 - loss)^dt
                              + charge_rate * eta_charge * dt
                              - discharge_rate / eta_discharge * dt

        Rearranged as LHS = 0:
            charge_state[t+1] - charge_state[t] * (1 - loss)^dt
            - charge_rate * eta_charge * dt
            + discharge_rate / eta_discharge * dt = 0

        Returns:
            The LHS expression (should equal 0).
        """
        charge_state = self.charge_state
        rel_loss = self.element.relative_loss_per_hour
        timestep_duration = self._model.timestep_duration
        charge_rate = self.element.charging.submodel.flow_rate
        discharge_rate = self.element.discharging.submodel.flow_rate
        eff_charge = self.element.eta_charge
        eff_discharge = self.element.eta_discharge

        return (
            charge_state.isel(time=slice(1, None))
            - charge_state.isel(time=slice(None, -1)) * ((1 - rel_loss) ** timestep_duration)
            - charge_rate * eff_charge * timestep_duration
            + discharge_rate * timestep_duration / eff_discharge
        )

    @property
    def _absolute_charge_state_bounds(self) -> tuple[xr.DataArray, xr.DataArray]:
        """Get absolute bounds for charge_state variable.

        For base StorageModel, charge_state represents absolute SOC with bounds
        derived from relative bounds scaled by capacity.

        Note:
            InterclusterStorageModel overrides this to provide symmetric bounds
            since charge_state represents ΔE (relative change from cluster start).
        """
        relative_lower_bound, relative_upper_bound = self._relative_charge_state_bounds

        if self.element.capacity_in_flow_hours is None:
            return 0, np.inf
        elif isinstance(self.element.capacity_in_flow_hours, InvestParameters):
            cap_min = self.element.capacity_in_flow_hours.minimum_or_fixed_size
            cap_max = self.element.capacity_in_flow_hours.maximum_or_fixed_size
            return (
                relative_lower_bound * cap_min,
                relative_upper_bound * cap_max,
            )
        else:
            cap = self.element.capacity_in_flow_hours
            return (
                relative_lower_bound * cap,
                relative_upper_bound * cap,
            )

    @functools.cached_property
    def _relative_charge_state_bounds(self) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Get relative charge state bounds with final timestep values.

        Returns:
            Tuple of (minimum_bounds, maximum_bounds) DataArrays extending to final timestep
        """
        timesteps_extra = self._model.flow_system.timesteps_extra

        # Get the original bounds (may be scalar or have time dim)
        rel_min = self.element.relative_minimum_charge_state
        rel_max = self.element.relative_maximum_charge_state

        # Get final minimum charge state
        if self.element.relative_minimum_final_charge_state is None:
            min_final_value = _scalar_safe_isel_drop(rel_min, 'time', -1)
        else:
            min_final_value = self.element.relative_minimum_final_charge_state

        # Get final maximum charge state
        if self.element.relative_maximum_final_charge_state is None:
            max_final_value = _scalar_safe_isel_drop(rel_max, 'time', -1)
        else:
            max_final_value = self.element.relative_maximum_final_charge_state

        # Build bounds arrays for timesteps_extra (includes final timestep)
        # Handle case where original data may be scalar (no time dim)
        if 'time' in rel_min.dims:
            # Original has time dim - concat with final value
            min_final_da = (
                min_final_value.expand_dims('time') if 'time' not in min_final_value.dims else min_final_value
            )
            min_final_da = min_final_da.assign_coords(time=[timesteps_extra[-1]])
            min_bounds = xr.concat([rel_min, min_final_da], dim='time')
        else:
            # Original is scalar - broadcast to full time range (constant value)
            min_bounds = rel_min.expand_dims(time=timesteps_extra)

        if 'time' in rel_max.dims:
            # Original has time dim - concat with final value
            max_final_da = (
                max_final_value.expand_dims('time') if 'time' not in max_final_value.dims else max_final_value
            )
            max_final_da = max_final_da.assign_coords(time=[timesteps_extra[-1]])
            max_bounds = xr.concat([rel_max, max_final_da], dim='time')
        else:
            # Original is scalar - broadcast to full time range (constant value)
            max_bounds = rel_max.expand_dims(time=timesteps_extra)

        # Ensure both bounds have matching dimensions (broadcast once here,
        # so downstream code doesn't need to handle dimension mismatches)
        return xr.broadcast(min_bounds, max_bounds)

    @property
    def _investment(self) -> InvestmentModel | None:
        """Deprecated alias for investment"""
        return self.investment

    @property
    def investment(self) -> InvestmentModel | None:
        """Investment feature"""
        if 'investment' not in self.submodels:
            return None
        return self.submodels['investment']

    @property
    def charge_state(self) -> linopy.Variable:
        """Charge state variable"""
        return self['charge_state']

    @property
    def netto_discharge(self) -> linopy.Variable:
        """Netto discharge variable"""
        return self['netto_discharge']


class InterclusterStorageModel(StorageModel):
    """Storage model with inter-cluster linking for clustered optimization.

    This class extends :class:`StorageModel` to support inter-cluster storage linking
    when using time series aggregation (clustering). It implements the S-N linking model
    from Blanke et al. (2022) to properly value seasonal storage in clustered optimizations.

    The Problem with Naive Clustering
    ---------------------------------
    When time series are clustered (e.g., 365 days → 8 typical days), storage behavior
    is fundamentally misrepresented if each cluster operates independently:

    - **Seasonal patterns are lost**: A battery might charge in summer and discharge in
      winter, but with independent clusters, each "typical summer day" cannot transfer
      energy to the "typical winter day".
    - **Storage value is underestimated**: Without inter-cluster linking, storage can only
      provide intra-day flexibility, not seasonal arbitrage.

    The S-N Linking Model
    ---------------------
    This model introduces two key concepts:

    1. **SOC_boundary**: Absolute state-of-charge at the boundary between original periods.
       With N original periods, there are N+1 boundary points (including start and end).

    2. **charge_state (ΔE)**: Relative change in SOC within each representative cluster,
       measured from the cluster start (where ΔE = 0).

    The actual SOC at any timestep t within original period d is::

        SOC(t) = SOC_boundary[d] + ΔE(t)

    Key Constraints
    ---------------
    1. **Cluster start constraint**: ``ΔE(cluster_start) = 0``
       Each representative cluster starts with zero relative charge.

    2. **Linking constraint**: ``SOC_boundary[d+1] = SOC_boundary[d] + delta_SOC[cluster_assignments[d]]``
       The boundary SOC after period d equals the boundary before plus the net
       charge/discharge of the representative cluster for that period.

    3. **Combined bounds**: ``0 ≤ SOC_boundary[d] + ΔE(t) ≤ capacity``
       The actual SOC must stay within physical bounds.

    4. **Cyclic constraint** (for ``intercluster_cyclic`` mode):
       ``SOC_boundary[0] = SOC_boundary[N]``
       The storage returns to its initial state over the full time horizon.

    Variables Created
    -----------------
    - ``SOC_boundary``: Absolute SOC at each original period boundary.
      Shape: (n_original_clusters + 1,) plus any period/scenario dimensions.

    Constraints Created
    -------------------
    - ``cluster_start``: Forces ΔE = 0 at start of each representative cluster.
    - ``link``: Links consecutive SOC_boundary values via delta_SOC.
    - ``cyclic`` or ``initial_SOC_boundary``: Initial/final boundary condition.
    - ``soc_lb_start/mid/end``: Lower bound on combined SOC at sample points.
    - ``soc_ub_start/mid/end``: Upper bound on combined SOC (if investment).
    - ``SOC_boundary_ub``: Links SOC_boundary to investment size (if investment).
    - ``charge_state|lb/ub``: Symmetric bounds on ΔE for intercluster modes.

    References
    ----------
    - Blanke, T., et al. (2022). "Inter-Cluster Storage Linking for Time Series
      Aggregation in Energy System Optimization Models."
    - Kotzur, L., et al. (2018). "Time series aggregation for energy system design:
      Modeling seasonal storage."

    See Also
    --------
    :class:`StorageModel` : Base storage model without inter-cluster linking.
    :class:`Storage` : The element class that creates this model.

    Example
    -------
    The model is automatically used when a Storage has ``cluster_mode='intercluster'``
    or ``cluster_mode='intercluster_cyclic'`` and the FlowSystem has been clustered::

        storage = Storage(
            label='seasonal_storage',
            charging=charge_flow,
            discharging=discharge_flow,
            capacity_in_flow_hours=InvestParameters(maximum_size=10000),
            cluster_mode='intercluster_cyclic',  # Enable inter-cluster linking
        )

        # Cluster the flow system
        fs_clustered = flow_system.transform.cluster(n_clusters=8)
        fs_clustered.optimize(solver)

        # Access the SOC_boundary in results
        soc_boundary = fs_clustered.solution['seasonal_storage|SOC_boundary']
    """

    @property
    def _absolute_charge_state_bounds(self) -> tuple[xr.DataArray, xr.DataArray]:
        """Get symmetric bounds for charge_state (ΔE) variable.

        For InterclusterStorageModel, charge_state represents ΔE (relative change
        from cluster start), which can be negative. Therefore, we need symmetric
        bounds: -capacity <= ΔE <= capacity.

        Note that for investment-based sizing, additional constraints are added
        in _add_investment_model to link bounds to the actual investment size.
        """
        _, relative_upper_bound = self._relative_charge_state_bounds

        if self.element.capacity_in_flow_hours is None:
            return -np.inf, np.inf
        elif isinstance(self.element.capacity_in_flow_hours, InvestParameters):
            cap_max = self.element.capacity_in_flow_hours.maximum_or_fixed_size * relative_upper_bound
            # Adding 0.0 converts -0.0 to 0.0 (linopy LP writer bug workaround)
            return -cap_max + 0.0, cap_max + 0.0
        else:
            cap = self.element.capacity_in_flow_hours * relative_upper_bound
            # Adding 0.0 converts -0.0 to 0.0 (linopy LP writer bug workaround)
            return -cap + 0.0, cap + 0.0

    def _do_modeling(self):
        """Create storage model with inter-cluster linking constraints.

        Uses template method pattern: calls parent's _do_modeling, then adds
        inter-cluster linking. Overrides specific methods to customize behavior.
        """
        super()._do_modeling()
        self._add_intercluster_linking()

    def _add_cluster_cyclic_constraint(self):
        """Skip cluster cyclic constraint - handled by inter-cluster linking."""
        pass

    def _add_investment_model(self):
        """Create InvestmentModel with symmetric bounds for ΔE."""
        if isinstance(self.element.capacity_in_flow_hours, InvestParameters):
            self.add_submodels(
                InvestmentModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    label_of_model=self.label_of_element,
                    parameters=self.element.capacity_in_flow_hours,
                    size_category=VariableCategory.STORAGE_SIZE,
                ),
                short_name='investment',
            )
            # Symmetric bounds: -size <= charge_state <= size
            self.add_constraints(
                self.charge_state >= -self.investment.size,
                short_name='charge_state|lb',
            )
            self.add_constraints(
                self.charge_state <= self.investment.size,
                short_name='charge_state|ub',
            )

    def _add_initial_final_constraints(self):
        """Skip initial/final constraints - handled by SOC_boundary in inter-cluster linking."""
        pass

    def _add_intercluster_linking(self) -> None:
        """Add inter-cluster storage linking following the S-K model from Blanke et al. (2022).

        This method implements the core inter-cluster linking logic:

        1. Constrains charge_state (ΔE) at each cluster start to 0
        2. Creates SOC_boundary variables to track absolute SOC at period boundaries
        3. Links boundaries via Eq. 5: SOC_boundary[d+1] = SOC_boundary[d] * (1-loss)^N + delta_SOC
        4. Adds combined bounds per Eq. 9: 0 ≤ SOC_boundary * (1-loss)^t + ΔE ≤ capacity
        5. Enforces initial/cyclic constraint on SOC_boundary
        """
        from .clustering.intercluster_helpers import (
            build_boundary_coords,
            extract_capacity_bounds,
        )

        clustering = self._model.flow_system.clustering
        if clustering is None:
            return

        n_clusters = clustering.n_clusters
        timesteps_per_cluster = clustering.timesteps_per_cluster
        n_original_clusters = clustering.n_original_clusters
        cluster_assignments = clustering.cluster_assignments

        # 1. Constrain ΔE = 0 at cluster starts
        self._add_cluster_start_constraints(n_clusters, timesteps_per_cluster)

        # 2. Create SOC_boundary variable
        flow_system = self._model.flow_system
        boundary_coords, boundary_dims = build_boundary_coords(n_original_clusters, flow_system)
        capacity_bounds = extract_capacity_bounds(self.element.capacity_in_flow_hours, boundary_coords, boundary_dims)

        soc_boundary = self.add_variables(
            lower=capacity_bounds.lower,
            upper=capacity_bounds.upper,
            coords=boundary_coords,
            dims=boundary_dims,
            short_name='SOC_boundary',
            category=VariableCategory.SOC_BOUNDARY,
        )

        # 3. Link SOC_boundary to investment size
        if capacity_bounds.has_investment and self.investment is not None:
            self.add_constraints(
                soc_boundary <= self.investment.size,
                short_name='SOC_boundary_ub',
            )

        # 4. Compute delta_SOC for each cluster
        delta_soc = self._compute_delta_soc(n_clusters, timesteps_per_cluster)

        # 5. Add linking constraints
        self._add_linking_constraints(
            soc_boundary, delta_soc, cluster_assignments, n_original_clusters, timesteps_per_cluster
        )

        # 6. Add cyclic or initial constraint
        if self.element.cluster_mode == 'intercluster_cyclic':
            self.add_constraints(
                soc_boundary.isel(cluster_boundary=0) == soc_boundary.isel(cluster_boundary=n_original_clusters),
                short_name='cyclic',
            )
        else:
            # Apply initial_charge_state to SOC_boundary[0]
            initial = self.element.initial_charge_state
            if initial is not None:
                if isinstance(initial, str):
                    # 'equals_final' means cyclic
                    self.add_constraints(
                        soc_boundary.isel(cluster_boundary=0)
                        == soc_boundary.isel(cluster_boundary=n_original_clusters),
                        short_name='initial_SOC_boundary',
                    )
                else:
                    self.add_constraints(
                        soc_boundary.isel(cluster_boundary=0) == initial,
                        short_name='initial_SOC_boundary',
                    )

        # 7. Add combined bound constraints
        self._add_combined_bound_constraints(
            soc_boundary,
            cluster_assignments,
            capacity_bounds.has_investment,
            n_original_clusters,
            timesteps_per_cluster,
        )

    def _add_cluster_start_constraints(self, n_clusters: int, timesteps_per_cluster: int) -> None:
        """Constrain ΔE = 0 at the start of each representative cluster.

        This ensures that the relative charge state is measured from a known
        reference point (the cluster start).

        With 2D (cluster, time) structure, time=0 is the start of every cluster,
        so we simply select isel(time=0) which broadcasts across the cluster dimension.

        Args:
            n_clusters: Number of representative clusters (unused with 2D structure).
            timesteps_per_cluster: Timesteps in each cluster (unused with 2D structure).
        """
        # With 2D structure: time=0 is start of every cluster
        self.add_constraints(
            self.charge_state.isel(time=0) == 0,
            short_name='cluster_start',
        )

    def _compute_delta_soc(self, n_clusters: int, timesteps_per_cluster: int) -> xr.DataArray:
        """Compute net SOC change (delta_SOC) for each representative cluster.

        The delta_SOC is the difference between the charge_state at the end
        and start of each cluster: delta_SOC[c] = ΔE(end_c) - ΔE(start_c).

        Since ΔE(start) = 0 by constraint, this simplifies to delta_SOC[c] = ΔE(end_c).

        With 2D (cluster, time) structure, we can simply select isel(time=-1) and isel(time=0),
        which already have the 'cluster' dimension.

        Args:
            n_clusters: Number of representative clusters (unused with 2D structure).
            timesteps_per_cluster: Timesteps in each cluster (unused with 2D structure).

        Returns:
            DataArray with 'cluster' dimension containing delta_SOC for each cluster.
        """
        # With 2D structure: result already has cluster dimension
        return self.charge_state.isel(time=-1) - self.charge_state.isel(time=0)

    def _add_linking_constraints(
        self,
        soc_boundary: xr.DataArray,
        delta_soc: xr.DataArray,
        cluster_assignments: xr.DataArray,
        n_original_clusters: int,
        timesteps_per_cluster: int,
    ) -> None:
        """Add constraints linking consecutive SOC_boundary values.

        Per Blanke et al. (2022) Eq. 5, implements:
            SOC_boundary[d+1] = SOC_boundary[d] * (1-loss)^N + delta_SOC[cluster_assignments[d]]

        where N is timesteps_per_cluster and loss is self-discharge rate per timestep.

        This connects the SOC at the end of original period d to the SOC at the
        start of period d+1, accounting for self-discharge decay over the period.

        Args:
            soc_boundary: SOC_boundary variable.
            delta_soc: Net SOC change per cluster.
            cluster_assignments: Mapping from original periods to representative clusters.
            n_original_clusters: Number of original (non-clustered) periods.
            timesteps_per_cluster: Number of timesteps in each cluster period.
        """
        soc_after = soc_boundary.isel(cluster_boundary=slice(1, None))
        soc_before = soc_boundary.isel(cluster_boundary=slice(None, -1))

        # Rename for alignment
        soc_after = soc_after.rename({'cluster_boundary': 'original_cluster'})
        soc_after = soc_after.assign_coords(original_cluster=np.arange(n_original_clusters))
        soc_before = soc_before.rename({'cluster_boundary': 'original_cluster'})
        soc_before = soc_before.assign_coords(original_cluster=np.arange(n_original_clusters))

        # Get delta_soc for each original period using cluster_assignments
        delta_soc_ordered = delta_soc.isel(cluster=cluster_assignments)

        # Apply self-discharge decay factor (1-loss)^hours to soc_before per Eq. 5
        # relative_loss_per_hour is per-hour, so we need total hours per cluster
        # Use sum over time to get total duration (handles both regular and segmented systems)
        # Keep as DataArray to respect per-period/scenario values
        rel_loss = _scalar_safe_reduce(self.element.relative_loss_per_hour, 'time', 'mean')
        total_hours_per_cluster = _scalar_safe_reduce(self._model.timestep_duration, 'time', 'sum')
        decay_n = (1 - rel_loss) ** total_hours_per_cluster

        lhs = soc_after - soc_before * decay_n - delta_soc_ordered
        self.add_constraints(lhs == 0, short_name='link')

    def _add_combined_bound_constraints(
        self,
        soc_boundary: xr.DataArray,
        cluster_assignments: xr.DataArray,
        has_investment: bool,
        n_original_clusters: int,
        timesteps_per_cluster: int,
    ) -> None:
        """Add constraints ensuring actual SOC stays within bounds.

        Per Blanke et al. (2022) Eq. 9, the actual SOC at time t in period d is:
            SOC(t) = SOC_boundary[d] * (1-loss)^t + ΔE(t)

        This must satisfy: 0 ≤ SOC(t) ≤ capacity

        Since checking every timestep is expensive, we sample at the start,
        middle, and end of each cluster.

        With 2D (cluster, time) structure, we simply select charge_state at a
        given time offset, then reorder by cluster_assignments to get original_cluster order.

        Args:
            soc_boundary: SOC_boundary variable.
            cluster_assignments: Mapping from original periods to clusters.
            has_investment: Whether the storage has investment sizing.
            n_original_clusters: Number of original periods.
            timesteps_per_cluster: Timesteps in each cluster.
        """
        charge_state = self.charge_state

        # soc_d: SOC at start of each original period
        soc_d = soc_boundary.isel(cluster_boundary=slice(None, -1))
        soc_d = soc_d.rename({'cluster_boundary': 'original_cluster'})
        soc_d = soc_d.assign_coords(original_cluster=np.arange(n_original_clusters))

        # Get self-discharge rate for decay calculation
        # relative_loss_per_hour is per-hour, so we need to convert offsets to hours
        # Keep as DataArray to respect per-period/scenario values
        rel_loss = _scalar_safe_reduce(self.element.relative_loss_per_hour, 'time', 'mean')

        # Compute cumulative hours for accurate offset calculation with non-uniform timesteps
        timestep_duration = self._model.timestep_duration
        if isinstance(timestep_duration, xr.DataArray) and 'time' in timestep_duration.dims:
            # Use cumsum for accurate hours offset with non-uniform timesteps
            # Build cumulative_hours with N+1 elements to match charge_state's extra timestep:
            # index 0 = 0 hours, index i = sum of durations[0:i], index N = total duration
            cumsum = timestep_duration.cumsum('time')
            # Prepend 0 at the start, giving [0, cumsum[0], cumsum[1], ..., cumsum[N-1]]
            cumulative_hours = xr.concat(
                [xr.zeros_like(timestep_duration.isel(time=0)), cumsum],
                dim='time',
            )
        else:
            # Scalar or no time dim: fall back to mean-based calculation
            mean_timestep_duration = _scalar_safe_reduce(timestep_duration, 'time', 'mean')
            cumulative_hours = None

        # Use actual time dimension size (may be smaller than timesteps_per_cluster for segmented systems)
        actual_time_size = charge_state.sizes['time']
        sample_offsets = [0, actual_time_size // 2, actual_time_size - 1]

        for sample_name, offset in zip(['start', 'mid', 'end'], sample_offsets, strict=False):
            # With 2D structure: select time offset, then reorder by cluster_assignments
            cs_at_offset = charge_state.isel(time=offset)  # Shape: (cluster, ...)
            # Reorder to original_cluster order using cluster_assignments indexer
            cs_t = cs_at_offset.isel(cluster=cluster_assignments)
            # Suppress xarray warning about index loss - we immediately assign new coords anyway
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*does not create an index anymore.*')
                cs_t = cs_t.rename({'cluster': 'original_cluster'})
            cs_t = cs_t.assign_coords(original_cluster=np.arange(n_original_clusters))

            # Apply decay factor (1-loss)^hours to SOC_boundary per Eq. 9
            # Convert timestep offset to hours using cumulative duration for non-uniform timesteps
            if cumulative_hours is not None:
                hours_offset = cumulative_hours.isel(time=offset)
            else:
                hours_offset = offset * mean_timestep_duration
            decay_t = (1 - rel_loss) ** hours_offset
            combined = soc_d * decay_t + cs_t

            self.add_constraints(combined >= 0, short_name=f'soc_lb_{sample_name}')

            if has_investment and self.investment is not None:
                self.add_constraints(combined <= self.investment.size, short_name=f'soc_ub_{sample_name}')
            elif not has_investment and isinstance(self.element.capacity_in_flow_hours, (int, float)):
                # Fixed-capacity storage: upper bound is the fixed capacity
                self.add_constraints(
                    combined <= self.element.capacity_in_flow_hours, short_name=f'soc_ub_{sample_name}'
                )


@register_class_for_io
class SourceAndSink(Component):
    """
    A SourceAndSink combines both supply and demand capabilities in a single component.

    SourceAndSink components can both consume AND provide energy or material flows
    from and to the system, making them ideal for modeling markets, (simple) storage facilities,
    or bidirectional grid connections where buying and selling occur at the same location.

    Args:
        label: The label of the Element. Used to identify it in the FlowSystem.
        inputs: Input-flows into the SourceAndSink representing consumption/demand side.
        outputs: Output-flows from the SourceAndSink representing supply/generation side.
        prevent_simultaneous_flow_rates: If True, prevents simultaneous input and output
            flows. This enforces that the component operates either as a source OR sink
            at any given time, but not both simultaneously. Default is True.
        meta_data: Used to store additional information about the Element. Not used
            internally but saved in results. Only use Python native types.

    Examples:
        Electricity market connection (buy/sell to grid):

        ```python
        electricity_market = SourceAndSink(
            label='grid_connection',
            inputs=[electricity_purchase],  # Buy from grid
            outputs=[electricity_sale],  # Sell to grid
            prevent_simultaneous_flow_rates=True,  # Can't buy and sell simultaneously
        )
        ```

        Natural gas storage facility:

        ```python
        gas_storage_facility = SourceAndSink(
            label='underground_gas_storage',
            inputs=[gas_injection_flow],  # Inject gas into storage
            outputs=[gas_withdrawal_flow],  # Withdraw gas from storage
            prevent_simultaneous_flow_rates=True,  # Injection or withdrawal, not both
        )
        ```

        District heating network connection:

        ```python
        dh_connection = SourceAndSink(
            label='district_heating_tie',
            inputs=[heat_purchase_flow],  # Purchase heat from network
            outputs=[heat_sale_flow],  # Sell excess heat to network
            prevent_simultaneous_flow_rates=False,  # May allow simultaneous flows
        )
        ```

        Industrial waste heat exchange:

        ```python
        waste_heat_exchange = SourceAndSink(
            label='industrial_heat_hub',
            inputs=[
                waste_heat_input_a,  # Receive waste heat from process A
                waste_heat_input_b,  # Receive waste heat from process B
            ],
            outputs=[
                useful_heat_supply_c,  # Supply heat to process C
                useful_heat_supply_d,  # Supply heat to process D
            ],
            prevent_simultaneous_flow_rates=False,  # Multiple simultaneous flows allowed
        )
        ```

    Note:
        When prevent_simultaneous_flow_rates is True, binary variables are created to
        ensure mutually exclusive operation between input and output flows, which
        increases computational complexity but reflects realistic market or storage
        operation constraints.

        SourceAndSink is particularly useful for modeling:
        - Energy markets with bidirectional trading
        - Storage facilities with injection/withdrawal operations
        - Grid tie points with import/export capabilities
        - Waste exchange networks with multiple participants

    Deprecated:
        The deprecated `sink` and `source` kwargs are accepted for compatibility but will be removed in future releases.
    """

    def __init__(
        self,
        label: str,
        inputs: list[Flow] | None = None,
        outputs: list[Flow] | None = None,
        prevent_simultaneous_flow_rates: bool = True,
        meta_data: dict | None = None,
        color: str | None = None,
    ):
        super().__init__(
            label,
            inputs=inputs,
            outputs=outputs,
            prevent_simultaneous_flows=(inputs or []) + (outputs or []) if prevent_simultaneous_flow_rates else None,
            meta_data=meta_data,
            color=color,
        )
        self.prevent_simultaneous_flow_rates = prevent_simultaneous_flow_rates


@register_class_for_io
class Source(Component):
    """
    A Source generates or provides energy or material flows into the system.

    Sources represent supply points like power plants, fuel suppliers, renewable
    energy sources, or any system boundary where flows originate. They provide
    unlimited supply capability subject to flow constraints, demand patterns and effects.

    Args:
        label: The label of the Element. Used to identify it in the FlowSystem.
        outputs: Output-flows from the source. Can be single flow or list of flows
            for sources providing multiple commodities or services.
        meta_data: Used to store additional information about the Element. Not used
            internally but saved in results. Only use Python native types.
        prevent_simultaneous_flow_rates: If True, only one output flow can be active
            at a time. Useful for modeling mutually exclusive supply options. Default is False.

    Examples:
        Simple electricity grid connection:

        ```python
        grid_source = Source(label='electrical_grid', outputs=[grid_electricity_flow])
        ```

        Natural gas supply with cost and capacity constraints:

        ```python
        gas_supply = Source(
            label='gas_network',
            outputs=[
                Flow(
                    label='natural_gas_flow',
                    bus=gas_bus,
                    size=1000,  # Maximum 1000 kW supply capacity
                    effects_per_flow_hour={'cost': 0.04},  # €0.04/kWh gas cost
                )
            ],
        )
        ```

        Multi-fuel power plant with switching constraints:

        ```python
        multi_fuel_plant = Source(
            label='flexible_generator',
            outputs=[coal_electricity, gas_electricity, biomass_electricity],
            prevent_simultaneous_flow_rates=True,  # Can only use one fuel at a time
        )
        ```

        Renewable energy source with investment optimization:

        ```python
        solar_farm = Source(
            label='solar_pv',
            outputs=[
                Flow(
                    label='solar_power',
                    bus=electricity_bus,
                    size=InvestParameters(
                        minimum_size=0,
                        maximum_size=50000,  # Up to 50 MW
                        specific_effects={'cost': 800},  # €800/kW installed
                        fix_effects={'cost': 100000},  # €100k development costs
                    ),
                    fixed_relative_profile=solar_profile,  # Hourly generation profile
                )
            ],
        )
        ```

    Deprecated:
        The deprecated `source` kwarg is accepted for compatibility but will be removed in future releases.
    """

    def __init__(
        self,
        label: str,
        outputs: list[Flow] | None = None,
        meta_data: dict | None = None,
        prevent_simultaneous_flow_rates: bool = False,
        color: str | None = None,
    ):
        self.prevent_simultaneous_flow_rates = prevent_simultaneous_flow_rates
        super().__init__(
            label,
            outputs=outputs,
            meta_data=meta_data,
            prevent_simultaneous_flows=outputs if prevent_simultaneous_flow_rates else None,
            color=color,
        )


@register_class_for_io
class Sink(Component):
    """
    A Sink consumes energy or material flows from the system.

    Sinks represent demand points like electrical loads, heat demands, material
    consumption, or any system boundary where flows terminate. They provide
    unlimited consumption capability subject to flow constraints, demand patterns and effects.

    Args:
        label: The label of the Element. Used to identify it in the FlowSystem.
        inputs: Input-flows into the sink. Can be single flow or list of flows
            for sinks consuming multiple commodities or services.
        meta_data: Used to store additional information about the Element. Not used
            internally but saved in results. Only use Python native types.
        prevent_simultaneous_flow_rates: If True, only one input flow can be active
            at a time. Useful for modeling mutually exclusive consumption options. Default is False.

    Examples:
        Simple electrical demand:

        ```python
        electrical_load = Sink(label='building_load', inputs=[electricity_demand_flow])
        ```

        Heat demand with time-varying profile:

        ```python
        heat_demand = Sink(
            label='district_heating_load',
            inputs=[
                Flow(
                    label='heat_consumption',
                    bus=heat_bus,
                    fixed_relative_profile=hourly_heat_profile,  # Demand profile
                    size=2000,  # Peak demand of 2000 kW
                )
            ],
        )
        ```

        Multi-energy building with switching capabilities:

        ```python
        flexible_building = Sink(
            label='smart_building',
            inputs=[electricity_heating, gas_heating, heat_pump_heating],
            prevent_simultaneous_flow_rates=True,  # Can only use one heating mode
        )
        ```

        Industrial process with variable demand:

        ```python
        factory_load = Sink(
            label='manufacturing_plant',
            inputs=[
                Flow(
                    label='electricity_process',
                    bus=electricity_bus,
                    size=5000,  # Base electrical load
                    effects_per_flow_hour={'cost': -0.1},  # Value of service (negative cost)
                ),
                Flow(
                    label='steam_process',
                    bus=steam_bus,
                    size=3000,  # Process steam demand
                    fixed_relative_profile=production_schedule,
                ),
            ],
        )
        ```

    Deprecated:
        The deprecated `sink` kwarg is accepted for compatibility but will be removed in future releases.
    """

    def __init__(
        self,
        label: str,
        inputs: list[Flow] | None = None,
        meta_data: dict | None = None,
        prevent_simultaneous_flow_rates: bool = False,
        color: str | None = None,
    ):
        """Initialize a Sink (consumes flow from the system).

        Args:
            label: Unique element label.
            inputs: Input flows for the sink.
            meta_data: Arbitrary metadata attached to the element.
            prevent_simultaneous_flow_rates: If True, prevents simultaneous nonzero flow rates
                across the element's inputs by wiring that restriction into the base Component setup.
            color: Optional color for visualizations.
        """

        self.prevent_simultaneous_flow_rates = prevent_simultaneous_flow_rates
        super().__init__(
            label,
            inputs=inputs,
            meta_data=meta_data,
            prevent_simultaneous_flows=inputs if prevent_simultaneous_flow_rates else None,
            color=color,
        )
