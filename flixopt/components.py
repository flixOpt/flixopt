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
from .features import InvestmentModel, InvestmentProxy, MaskHelpers
from .interface import InvestParameters, PiecewiseConversion, StatusParameters
from .modeling import _scalar_safe_isel, _scalar_safe_isel_drop, _scalar_safe_reduce
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
    ):
        super().__init__(label, inputs, outputs, status_parameters, meta_data=meta_data)
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

    submodel: StorageModelProxy | InterclusterStorageModel | None

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
    ):
        # TODO: fixed_relative_chargeState implementieren
        super().__init__(
            label,
            inputs=[charging],
            outputs=[discharging],
            prevent_simultaneous_flows=[charging, discharging] if prevent_simultaneous_charge_and_discharge else None,
            meta_data=meta_data,
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

    def create_model(self, model: FlowSystemModel) -> InterclusterStorageModel | StorageModelProxy:
        """Create the appropriate storage model based on cluster_mode.

        For intercluster modes ('intercluster', 'intercluster_cyclic'), uses
        :class:`InterclusterStorageModel` which implements S-N linking.
        For basic storages, uses :class:`StorageModelProxy` which provides
        element-level access to the batched StoragesModel.

        Args:
            model: The FlowSystemModel to add constraints to.

        Returns:
            InterclusterStorageModel or StorageModelProxy instance.
        """
        self._plausibility_checks()

        # Use InterclusterStorageModel for intercluster modes when clustering is active
        clustering = model.flow_system.clustering
        is_intercluster = clustering is not None and self.cluster_mode in (
            'intercluster',
            'intercluster_cyclic',
        )

        if is_intercluster:
            # Intercluster storages use standalone model (too complex to batch)
            self.submodel = InterclusterStorageModel(model, self)
        else:
            # Basic storages use proxy to batched StoragesModel
            self.submodel = StorageModelProxy(model, self)

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
    """Lightweight proxy for Transmission elements when using type-level modeling.

    Transmission constraints are created by ComponentsModel.create_transmission_constraints().
    This proxy exists for:
    - Results structure compatibility
    - Submodel registration in FlowSystemModel
    """

    element: Transmission

    def _do_modeling(self):
        """No-op: transmission constraints handled by ComponentsModel."""
        super()._do_modeling()
        # Transmission efficiency constraints are now created by
        # ComponentsModel.create_transmission_constraints()
        pass


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

        # Both conversion factor and piecewise conversion constraints are now handled
        # by type-level model ConvertersModel in elements.py
        # This model is kept for component-specific logic and results structure
        pass


class InterclusterStorageModel(ComponentModel):
    """Storage model with inter-cluster linking for clustered optimization.

    This is a standalone model for storages with ``cluster_mode='intercluster'``
    or ``cluster_mode='intercluster_cyclic'``. It implements the S-N linking model
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
    - ``charge_state``: Relative change in SOC (ΔE) within each cluster.
    - ``netto_discharge``: Net discharge rate (discharge - charge).
    - ``SOC_boundary``: Absolute SOC at each original period boundary.
      Shape: (n_original_clusters + 1,) plus any period/scenario dimensions.

    Constraints Created
    -------------------
    - ``netto_discharge``: Links netto_discharge to charge/discharge flows.
    - ``charge_state``: Energy balance within clusters.
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

    element: Storage

    def __init__(self, model: FlowSystemModel, element: Storage):
        super().__init__(model, element)

    # =========================================================================
    # Variable and Constraint Creation
    # =========================================================================

    def _do_modeling(self):
        """Create charge state variables, energy balance equations, and inter-cluster linking."""
        super()._do_modeling()
        self._create_storage_variables()
        self._add_netto_discharge_constraint()
        self._add_energy_balance_constraint()
        self._add_investment_model()
        self._add_balanced_sizes_constraint()
        self._add_intercluster_linking()

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
        # Access flow rates from type-level FlowsModel
        flows_model = self._model._flows_model
        charge_rate = flows_model.get_variable('rate', self.element.charging.label_full)
        discharge_rate = flows_model.get_variable('rate', self.element.discharging.label_full)
        self.add_constraints(
            self.netto_discharge == discharge_rate - charge_rate,
            short_name='netto_discharge',
        )

    def _add_energy_balance_constraint(self):
        """Add energy balance constraint linking charge states across timesteps."""
        self.add_constraints(self._build_energy_balance_lhs() == 0, short_name='charge_state')

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
        # Access flow rates from type-level FlowsModel
        flows_model = self._model._flows_model
        charge_rate = flows_model.get_variable('rate', self.element.charging.label_full)
        discharge_rate = flows_model.get_variable('rate', self.element.discharging.label_full)
        eff_charge = self.element.eta_charge
        eff_discharge = self.element.eta_discharge

        return (
            charge_state.isel(time=slice(1, None))
            - charge_state.isel(time=slice(None, -1)) * ((1 - rel_loss) ** timestep_duration)
            - charge_rate * eff_charge * timestep_duration
            + discharge_rate * timestep_duration / eff_discharge
        )

    def _add_balanced_sizes_constraint(self):
        """Add constraint ensuring charging and discharging capacities are equal."""
        if self.element.balanced:
            # Access investment sizes from type-level FlowsModel
            flows_model = self._model._flows_model
            charge_size = flows_model.get_variable('size', self.element.charging.label_full)
            discharge_size = flows_model.get_variable('size', self.element.discharging.label_full)
            self.add_constraints(
                charge_size - discharge_size == 0,
                short_name='balanced_sizes',
            )

    # =========================================================================
    # Bounds Properties
    # =========================================================================

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

    @functools.cached_property
    def _relative_charge_state_bounds(self) -> tuple[xr.DataArray, xr.DataArray]:
        """Get relative charge state bounds with final timestep values."""
        timesteps_extra = self._model.flow_system.timesteps_extra

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
        if 'time' in rel_min.dims:
            min_final_da = (
                min_final_value.expand_dims('time') if 'time' not in min_final_value.dims else min_final_value
            )
            min_final_da = min_final_da.assign_coords(time=[timesteps_extra[-1]])
            min_bounds = xr.concat([rel_min, min_final_da], dim='time')
        else:
            min_bounds = rel_min.expand_dims(time=timesteps_extra)

        if 'time' in rel_max.dims:
            max_final_da = (
                max_final_value.expand_dims('time') if 'time' not in max_final_value.dims else max_final_value
            )
            max_final_da = max_final_da.assign_coords(time=[timesteps_extra[-1]])
            max_bounds = xr.concat([rel_max, max_final_da], dim='time')
        else:
            max_bounds = rel_max.expand_dims(time=timesteps_extra)

        return xr.broadcast(min_bounds, max_bounds)

    # =========================================================================
    # Variable Access Properties
    # =========================================================================

    @property
    def _investment(self) -> InvestmentModel | None:
        """Deprecated alias for investment."""
        return self.investment

    @property
    def investment(self) -> InvestmentModel | None:
        """Investment feature."""
        if 'investment' not in self.submodels:
            return None
        return self.submodels['investment']

    @property
    def charge_state(self) -> linopy.Variable:
        """Charge state variable."""
        return self['charge_state']

    @property
    def netto_discharge(self) -> linopy.Variable:
        """Netto discharge variable."""
        return self['netto_discharge']

    # =========================================================================
    # Investment Model
    # =========================================================================

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

    # =========================================================================
    # Inter-Cluster Linking
    # =========================================================================

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
        mean_timestep_duration = _scalar_safe_reduce(self._model.timestep_duration, 'time', 'mean')

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
            # Convert timestep offset to hours
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


class StoragesModel:
    """Type-level model for ALL basic (non-intercluster) storages in a FlowSystem.

    Unlike StorageModel (one per Storage instance), StoragesModel handles ALL
    basic storages in a single instance with batched variables.

    Note:
        InterclusterStorageModel storages are excluded and handled traditionally
        due to their complexity (SOC_boundary linking, etc.).

    This enables:
    - Batched charge_state and netto_discharge variables with element dimension
    - Batched investment variables via InvestmentsModel
    - Consistent architecture with FlowsModel and BusesModel

    Example:
        >>> storages_model = StoragesModel(model, basic_storages, flows_model)
        >>> storages_model.create_variables()
        >>> storages_model.create_constraints()
        >>> storages_model.create_investment_model()  # After storage variables exist
        >>> storages_model.create_investment_constraints()
    """

    def __init__(
        self,
        model: FlowSystemModel,
        elements: list[Storage],
        flows_model,  # FlowsModel - avoid circular import
    ):
        """Initialize the type-level model for basic storages.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            elements: List of basic (non-intercluster) Storage elements.
            flows_model: The FlowsModel containing flow_rate variables.
        """
        from .structure import ElementType

        self.model = model
        self.elements = elements
        self.element_ids: list[str] = [s.label_full for s in elements]
        self._flows_model = flows_model
        self.element_type = ElementType.STORAGE

        # Storage for created variables
        self._variables: dict[str, linopy.Variable] = {}

        # Categorize by features
        self.storages_with_investment: list[Storage] = [
            s for s in elements if isinstance(s.capacity_in_flow_hours, InvestParameters)
        ]
        self.storages_with_optional_investment: list[Storage] = [
            s for s in self.storages_with_investment if not s.capacity_in_flow_hours.mandatory
        ]
        self.investment_ids: list[str] = [s.label_full for s in self.storages_with_investment]
        self.optional_investment_ids: list[str] = [s.label_full for s in self.storages_with_optional_investment]

        # Investment params dict (populated in create_investment_model)
        self._invest_params: dict[str, InvestParameters] = {}

        # Set reference on each storage element
        for storage in elements:
            storage._storages_model = self

    @property
    def dim_name(self) -> str:
        """Dimension name for storage elements."""
        return self.element_type.value  # 'storage'

    # --- Investment Cached Properties ---

    @functools.cached_property
    def mandatory_investment_ids(self) -> list[str]:
        """List of storage IDs with mandatory investment."""
        return [s.label_full for s in self.storages_with_investment if s.capacity_in_flow_hours.mandatory]

    @functools.cached_property
    def _size_lower(self) -> xr.DataArray:
        """(storage,) - minimum size for investment storages."""
        from .features import InvestmentHelpers

        element_ids = self.investment_ids
        values = [s.capacity_in_flow_hours.minimum_or_fixed_size for s in self.storages_with_investment]
        return InvestmentHelpers.stack_bounds(values, element_ids, self.dim_name)

    @functools.cached_property
    def _size_upper(self) -> xr.DataArray:
        """(storage,) - maximum size for investment storages."""
        from .features import InvestmentHelpers

        element_ids = self.investment_ids
        values = [s.capacity_in_flow_hours.maximum_or_fixed_size for s in self.storages_with_investment]
        return InvestmentHelpers.stack_bounds(values, element_ids, self.dim_name)

    @functools.cached_property
    def _linked_periods_mask(self) -> xr.DataArray | None:
        """(storage, period) - linked periods for investment storages. None if no linking."""
        from .features import InvestmentHelpers

        linked_list = [s.capacity_in_flow_hours.linked_periods for s in self.storages_with_investment]
        if not any(lp is not None for lp in linked_list):
            return None

        element_ids = self.investment_ids
        values = [lp if lp is not None else np.nan for lp in linked_list]
        return InvestmentHelpers.stack_bounds(values, element_ids, self.dim_name)

    @functools.cached_property
    def _mandatory_mask(self) -> xr.DataArray:
        """(storage,) bool - True if mandatory, False if optional."""
        element_ids = self.investment_ids
        values = [s.capacity_in_flow_hours.mandatory for s in self.storages_with_investment]
        return xr.DataArray(values, dims=[self.dim_name], coords={self.dim_name: element_ids})

    @functools.cached_property
    def _optional_lower(self) -> xr.DataArray | None:
        """(storage,) - minimum size for optional investment storages."""
        if not self.optional_investment_ids:
            return None
        from .features import InvestmentHelpers

        element_ids = self.optional_investment_ids
        values = [s.capacity_in_flow_hours.minimum_or_fixed_size for s in self.storages_with_optional_investment]
        return InvestmentHelpers.stack_bounds(values, element_ids, self.dim_name)

    @functools.cached_property
    def _optional_upper(self) -> xr.DataArray | None:
        """(storage,) - maximum size for optional investment storages."""
        if not self.optional_investment_ids:
            return None
        from .features import InvestmentHelpers

        element_ids = self.optional_investment_ids
        values = [s.capacity_in_flow_hours.maximum_or_fixed_size for s in self.storages_with_optional_investment]
        return InvestmentHelpers.stack_bounds(values, element_ids, self.dim_name)

    @functools.cached_property
    def _flow_mask(self) -> xr.DataArray:
        """(storage, flow) mask: 1 if flow belongs to storage."""
        membership = MaskHelpers.build_flow_membership(
            self.elements,
            lambda s: s.inputs + s.outputs,
        )
        return MaskHelpers.build_mask(
            row_dim='storage',
            row_ids=self.element_ids,
            col_dim='flow',
            col_ids=self._flows_model.element_ids,
            membership=membership,
        )

    def create_variables(self) -> None:
        """Create batched variables for all storages.

        Creates:
        - storage|charge: For ALL storages (with storage dimension, extra timestep)
        - storage|netto: For ALL storages (with storage dimension)
        """
        import pandas as pd

        from .structure import VARIABLE_TYPE_TO_EXPANSION, VariableType

        if not self.elements:
            return

        dim = self.dim_name  # 'storage'

        # === storage|charge: ALL storages (with extra timestep) ===
        lower_bounds = self._collect_charge_state_bounds('lower')
        upper_bounds = self._collect_charge_state_bounds('upper')

        # Get coords with extra timestep
        coords_extra = self.model.get_coords(extra_timestep=True)
        charge_state_coords = xr.Coordinates(
            {
                dim: pd.Index(self.element_ids, name=dim),
                **{d: coords_extra[d] for d in coords_extra},
            }
        )

        charge_state = self.model.add_variables(
            lower=lower_bounds,
            upper=upper_bounds,
            coords=charge_state_coords,
            name='storage|charge',
        )
        self._variables['charge'] = charge_state

        # Register category for segment expansion
        expansion_category = VARIABLE_TYPE_TO_EXPANSION.get(VariableType.CHARGE_STATE)
        if expansion_category is not None:
            self.model.variable_categories[charge_state.name] = expansion_category

        # === storage|netto: ALL storages ===
        # Use full coords (including scenarios) not just temporal_dims
        full_coords = self.model.get_coords()
        netto_discharge_coords = xr.Coordinates(
            {
                dim: pd.Index(self.element_ids, name=dim),
                **{d: full_coords[d] for d in full_coords},
            }
        )

        netto_discharge = self.model.add_variables(
            coords=netto_discharge_coords,
            name='storage|netto',
        )
        self._variables['netto'] = netto_discharge

        # Register category for segment expansion
        expansion_category = VARIABLE_TYPE_TO_EXPANSION.get(VariableType.NETTO_DISCHARGE)
        if expansion_category is not None:
            self.model.variable_categories[netto_discharge.name] = expansion_category

        logger.debug(
            f'StoragesModel created variables: {len(self.elements)} storages, '
            f'{len(self.storages_with_investment)} with investment'
        )

    def _collect_charge_state_bounds(self, bound_type: str) -> xr.DataArray:
        """Collect charge_state bounds from all storages.

        Args:
            bound_type: 'lower' or 'upper'
        """
        dim = self.dim_name  # 'storage'
        bounds_list = []
        for storage in self.elements:
            rel_min, rel_max = self._get_relative_charge_state_bounds(storage)

            if storage.capacity_in_flow_hours is None:
                lb, ub = 0, np.inf
            elif isinstance(storage.capacity_in_flow_hours, InvestParameters):
                cap_min = storage.capacity_in_flow_hours.minimum_or_fixed_size
                cap_max = storage.capacity_in_flow_hours.maximum_or_fixed_size
                lb = rel_min * cap_min
                ub = rel_max * cap_max
            else:
                cap = storage.capacity_in_flow_hours
                lb = rel_min * cap
                ub = rel_max * cap

            if bound_type == 'lower':
                bounds_list.append(lb if isinstance(lb, xr.DataArray) else xr.DataArray(lb))
            else:
                bounds_list.append(ub if isinstance(ub, xr.DataArray) else xr.DataArray(ub))

        return xr.concat(bounds_list, dim=dim, coords='minimal').assign_coords({dim: self.element_ids})

    def _get_relative_charge_state_bounds(self, storage: Storage) -> tuple[xr.DataArray, xr.DataArray]:
        """Get relative charge state bounds with final timestep values."""
        timesteps_extra = self.model.flow_system.timesteps_extra

        rel_min = storage.relative_minimum_charge_state
        rel_max = storage.relative_maximum_charge_state

        # Get final values
        if storage.relative_minimum_final_charge_state is None:
            min_final_value = _scalar_safe_isel_drop(rel_min, 'time', -1)
        else:
            min_final_value = storage.relative_minimum_final_charge_state

        if storage.relative_maximum_final_charge_state is None:
            max_final_value = _scalar_safe_isel_drop(rel_max, 'time', -1)
        else:
            max_final_value = storage.relative_maximum_final_charge_state

        # Build bounds arrays for timesteps_extra
        if 'time' in rel_min.dims:
            min_final_da = (
                min_final_value.expand_dims('time') if 'time' not in min_final_value.dims else min_final_value
            )
            min_final_da = min_final_da.assign_coords(time=[timesteps_extra[-1]])
            min_bounds = xr.concat([rel_min, min_final_da], dim='time')
        else:
            min_bounds = rel_min.expand_dims(time=timesteps_extra)

        if 'time' in rel_max.dims:
            max_final_da = (
                max_final_value.expand_dims('time') if 'time' not in max_final_value.dims else max_final_value
            )
            max_final_da = max_final_da.assign_coords(time=[timesteps_extra[-1]])
            max_bounds = xr.concat([rel_max, max_final_da], dim='time')
        else:
            max_bounds = rel_max.expand_dims(time=timesteps_extra)

        return xr.broadcast(min_bounds, max_bounds)

    def create_constraints(self) -> None:
        """Create batched constraints for all storages.

        Uses vectorized operations for efficiency:
        - netto_discharge constraint (batched)
        - energy balance constraint (batched)
        - initial/final constraints (batched by type)
        """
        if not self.elements:
            return

        flow_rate = self._flows_model._variables['rate']
        charge_state = self._variables['charge']
        netto_discharge = self._variables['netto']
        timestep_duration = self.model.timestep_duration

        # === Batched netto_discharge constraint ===
        # Build charge and discharge flow_rate selections aligned with storage dimension
        charge_flow_ids = [s.charging.label_full for s in self.elements]
        discharge_flow_ids = [s.discharging.label_full for s in self.elements]

        # Detect flow dimension name from flow_rate variable
        flow_dim = 'flow' if 'flow' in flow_rate.dims else 'element'
        dim = self.dim_name

        # Select from flow dimension and rename to storage dimension
        charge_rates = flow_rate.sel({flow_dim: charge_flow_ids})
        charge_rates = charge_rates.rename({flow_dim: dim}).assign_coords({dim: self.element_ids})
        discharge_rates = flow_rate.sel({flow_dim: discharge_flow_ids})
        discharge_rates = discharge_rates.rename({flow_dim: dim}).assign_coords({dim: self.element_ids})

        self.model.add_constraints(
            netto_discharge == discharge_rates - charge_rates,
            name='storage|netto_eq',
        )

        # === Batched energy balance constraint ===
        # Stack parameters into DataArrays with element dimension
        eta_charge = self._stack_parameter([s.eta_charge for s in self.elements])
        eta_discharge = self._stack_parameter([s.eta_discharge for s in self.elements])
        rel_loss = self._stack_parameter([s.relative_loss_per_hour for s in self.elements])

        # Energy balance: cs[t+1] = cs[t] * (1-loss)^dt + charge * eta_c * dt - discharge * dt / eta_d
        # Rearranged: cs[t+1] - cs[t] * (1-loss)^dt - charge * eta_c * dt + discharge * dt / eta_d = 0
        energy_balance_lhs = (
            charge_state.isel(time=slice(1, None))
            - charge_state.isel(time=slice(None, -1)) * ((1 - rel_loss) ** timestep_duration)
            - charge_rates * eta_charge * timestep_duration
            + discharge_rates * timestep_duration / eta_discharge
        )
        self.model.add_constraints(
            energy_balance_lhs == 0,
            name='storage|balance',
        )

        # === Initial/final constraints (grouped by type) ===
        self._add_batched_initial_final_constraints(charge_state)

        # === Cluster cyclic constraints ===
        self._add_batched_cluster_cyclic_constraints(charge_state)

        # === Balanced flow sizes constraint ===
        self._add_balanced_flow_sizes_constraint()

        logger.debug(f'StoragesModel created batched constraints for {len(self.elements)} storages')

    def _add_balanced_flow_sizes_constraint(self) -> None:
        """Add constraint ensuring charging and discharging flow capacities are equal for balanced storages."""
        balanced_storages = [s for s in self.elements if s.balanced]
        if not balanced_storages:
            return

        # Access flow size variables from FlowsModel
        flows_model = self._flows_model
        size_var = flows_model.get_variable('size')
        if size_var is None:
            return

        flow_dim = flows_model.dim_name  # 'flow'

        for storage in balanced_storages:
            charge_id = storage.charging.label_full
            discharge_id = storage.discharging.label_full
            # Check if both flows have investment
            if charge_id not in flows_model.investment_ids or discharge_id not in flows_model.investment_ids:
                continue
            charge_size = size_var.sel({flow_dim: charge_id})
            discharge_size = size_var.sel({flow_dim: discharge_id})
            self.model.add_constraints(
                charge_size - discharge_size == 0,
                name=f'storage|{storage.label}|balanced_sizes',
            )

    def _stack_parameter(self, values: list, element_ids: list | None = None) -> xr.DataArray:
        """Stack parameter values into DataArray with storage dimension."""
        dim = self.dim_name
        ids = element_ids if element_ids is not None else self.element_ids
        das = [v if isinstance(v, xr.DataArray) else xr.DataArray(v) for v in values]
        return xr.concat(das, dim=dim, coords='minimal').assign_coords({dim: ids})

    def _add_batched_initial_final_constraints(self, charge_state) -> None:
        """Add batched initial and final charge state constraints."""
        # Group storages by constraint type
        storages_numeric_initial: list[tuple[Storage, float]] = []
        storages_equals_final: list[Storage] = []
        storages_max_final: list[tuple[Storage, float]] = []
        storages_min_final: list[tuple[Storage, float]] = []

        for storage in self.elements:
            # Skip for clustered independent/cyclic modes
            if self.model.flow_system.clusters is not None and storage.cluster_mode in ('independent', 'cyclic'):
                continue

            if storage.initial_charge_state is not None:
                if isinstance(storage.initial_charge_state, str):  # 'equals_final'
                    storages_equals_final.append(storage)
                else:
                    storages_numeric_initial.append((storage, storage.initial_charge_state))

            if storage.maximal_final_charge_state is not None:
                storages_max_final.append((storage, storage.maximal_final_charge_state))

            if storage.minimal_final_charge_state is not None:
                storages_min_final.append((storage, storage.minimal_final_charge_state))

        dim = self.dim_name

        # Batched numeric initial constraint
        if storages_numeric_initial:
            ids = [s.label_full for s, _ in storages_numeric_initial]
            values = self._stack_parameter([v for _, v in storages_numeric_initial], ids)
            cs_initial = charge_state.sel({dim: ids}).isel(time=0)
            self.model.add_constraints(
                cs_initial == values,
                name='storage|initial_charge_state',
            )

        # Batched equals_final constraint
        if storages_equals_final:
            ids = [s.label_full for s in storages_equals_final]
            cs_subset = charge_state.sel({dim: ids})
            self.model.add_constraints(
                cs_subset.isel(time=0) == cs_subset.isel(time=-1),
                name='storage|initial_equals_final',
            )

        # Batched max final constraint
        if storages_max_final:
            ids = [s.label_full for s, _ in storages_max_final]
            values = self._stack_parameter([v for _, v in storages_max_final], ids)
            cs_final = charge_state.sel({dim: ids}).isel(time=-1)
            self.model.add_constraints(
                cs_final <= values,
                name='storage|final_charge_max',
            )

        # Batched min final constraint
        if storages_min_final:
            ids = [s.label_full for s, _ in storages_min_final]
            values = self._stack_parameter([v for _, v in storages_min_final], ids)
            cs_final = charge_state.sel({dim: ids}).isel(time=-1)
            self.model.add_constraints(
                cs_final >= values,
                name='storage|final_charge_min',
            )

    def _add_batched_cluster_cyclic_constraints(self, charge_state) -> None:
        """Add batched cluster cyclic constraints for storages with cyclic mode."""
        if self.model.flow_system.clusters is None:
            return

        cyclic_storages = [s for s in self.elements if s.cluster_mode == 'cyclic']
        if not cyclic_storages:
            return

        ids = [s.label_full for s in cyclic_storages]
        cs_subset = charge_state.sel({self.dim_name: ids})
        self.model.add_constraints(
            cs_subset.isel(time=0) == cs_subset.isel(time=-2),
            name='storage|cluster_cyclic',
        )

    def create_investment_model(self) -> None:
        """Create investment variables and constraints for storages with investment.

        Creates:
        - storage|size: For all storages with investment
        - storage|invested: For storages with optional (non-mandatory) investment

        Must be called BEFORE create_investment_constraints().
        """
        if not self.storages_with_investment:
            return

        import pandas as pd

        from .features import InvestmentHelpers
        from .structure import VARIABLE_TYPE_TO_EXPANSION, VariableType

        # Build params dict for easy access
        self._invest_params = {s.label_full: s.capacity_in_flow_hours for s in self.storages_with_investment}

        dim = self.dim_name
        element_ids = self.investment_ids
        non_mandatory_ids = self.optional_investment_ids
        mandatory_ids = self.mandatory_investment_ids

        # Get base coords
        base_coords = self.model.get_coords(['period', 'scenario'])
        base_coords_dict = dict(base_coords) if base_coords is not None else {}

        # Use cached properties for bounds
        size_min = self._size_lower
        size_max = self._size_upper

        # Handle linked_periods masking
        linked_periods = self._linked_periods_mask
        if linked_periods is not None:
            linked = linked_periods.fillna(1.0)
            size_min = size_min * linked
            size_max = size_max * linked

        # Use cached mandatory mask
        mandatory_mask = self._mandatory_mask

        # For non-mandatory, lower bound is 0 (invested variable controls actual minimum)
        lower_bounds = xr.where(mandatory_mask, size_min, 0)
        upper_bounds = size_max

        # === storage|size variable ===
        size_coords = xr.Coordinates({dim: pd.Index(element_ids, name=dim), **base_coords_dict})
        size_var = self.model.add_variables(
            lower=lower_bounds,
            upper=upper_bounds,
            coords=size_coords,
            name='storage|size',
        )
        self._variables['size'] = size_var

        # Register category for segment expansion
        expansion_category = VARIABLE_TYPE_TO_EXPANSION.get(VariableType.SIZE)
        if expansion_category is not None:
            self.model.variable_categories[size_var.name] = expansion_category

        # === storage|invested variable (non-mandatory only) ===
        if non_mandatory_ids:
            invested_coords = xr.Coordinates({dim: pd.Index(non_mandatory_ids, name=dim), **base_coords_dict})
            invested_var = self.model.add_variables(
                binary=True,
                coords=invested_coords,
                name='storage|invested',
            )
            self._variables['invested'] = invested_var

            # State-controlled bounds constraints using cached properties
            InvestmentHelpers.add_optional_size_bounds(
                model=self.model,
                size_var=size_var,
                invested_var=invested_var,
                min_bounds=self._optional_lower,
                max_bounds=self._optional_upper,
                element_ids=non_mandatory_ids,
                dim_name=dim,
                name_prefix='storage',
            )

        # Linked periods constraints
        InvestmentHelpers.add_linked_periods_constraints(
            model=self.model,
            size_var=size_var,
            params=self._invest_params,
            element_ids=element_ids,
            dim_name=dim,
        )

        # Piecewise effects (requires per-element submodels, not batchable)
        self._create_piecewise_effects()

        logger.debug(
            f'StoragesModel created investment variables: {len(element_ids)} storages '
            f'({len(mandatory_ids)} mandatory, {len(non_mandatory_ids)} optional)'
        )

    def create_investment_constraints(self) -> None:
        """Create batched scaled bounds linking charge_state to investment size.

        Must be called AFTER create_investment_model().

        Mathematical formulation:
            charge_state >= size * relative_minimum_charge_state
            charge_state <= size * relative_maximum_charge_state

        Uses the batched size variable for true vectorized constraint creation.
        """
        if not self.storages_with_investment or 'size' not in self._variables:
            return

        charge_state = self._variables['charge']
        size_var = self._variables['size']  # Batched size with storage dimension

        # Collect relative bounds for all investment storages
        rel_lowers = []
        rel_uppers = []
        for storage in self.storages_with_investment:
            rel_lower, rel_upper = self._get_relative_charge_state_bounds(storage)
            rel_lowers.append(rel_lower)
            rel_uppers.append(rel_upper)

        # Stack relative bounds with storage dimension
        # Use coords='minimal' to handle dimension mismatches (some have 'period', some don't)
        dim = self.dim_name
        rel_lower_stacked = xr.concat(rel_lowers, dim=dim, coords='minimal').assign_coords({dim: self.investment_ids})
        rel_upper_stacked = xr.concat(rel_uppers, dim=dim, coords='minimal').assign_coords({dim: self.investment_ids})

        # Select charge_state for investment storages only
        cs_investment = charge_state.sel({dim: self.investment_ids})

        # Select size for these storages (it already has storage dimension)
        size_investment = size_var.sel({dim: self.investment_ids})

        # Check if all bounds are equal (fixed relative bounds)
        from .modeling import _xr_allclose

        if _xr_allclose(rel_lower_stacked, rel_upper_stacked):
            # Fixed bounds: charge_state == size * relative_bound
            self.model.add_constraints(
                cs_investment == size_investment * rel_lower_stacked,
                name='storage|charge|investment|fixed',
            )
        else:
            # Variable bounds: lower <= charge_state <= upper
            self.model.add_constraints(
                cs_investment >= size_investment * rel_lower_stacked,
                name='storage|charge|investment|lb',
            )
            self.model.add_constraints(
                cs_investment <= size_investment * rel_upper_stacked,
                name='storage|charge|investment|ub',
            )

        logger.debug(
            f'StoragesModel created batched investment constraints for {len(self.storages_with_investment)} storages'
        )

    def _add_initial_final_constraints_legacy(self, storage, cs) -> None:
        """Legacy per-element initial/final constraints (kept for reference)."""
        skip_initial_final = self.model.flow_system.clusters is not None and storage.cluster_mode in (
            'independent',
            'cyclic',
        )

        if not skip_initial_final:
            if storage.initial_charge_state is not None:
                if isinstance(storage.initial_charge_state, str):  # 'equals_final'
                    self.model.add_constraints(
                        cs.isel(time=0) == cs.isel(time=-1),
                        name=f'storage|{storage.label}|initial_charge_state',
                    )
                else:
                    self.model.add_constraints(
                        cs.isel(time=0) == storage.initial_charge_state,
                        name=f'storage|{storage.label}|initial_charge_state',
                    )

                if storage.maximal_final_charge_state is not None:
                    self.model.add_constraints(
                        cs.isel(time=-1) >= storage.minimal_final_charge_state,
                        name=f'storage|{storage.label}|final_charge_min',
                    )

        logger.debug(f'StoragesModel created constraints for {len(self.elements)} storages')

    # === Variable accessor properties ===

    @property
    def charge(self) -> linopy.Variable | None:
        """Batched charge state variable with (storage, time+1) dims."""
        return self.model.variables['storage|charge'] if 'storage|charge' in self.model.variables else None

    @property
    def netto(self) -> linopy.Variable | None:
        """Batched netto discharge variable with (storage, time) dims."""
        return self.model.variables['storage|netto'] if 'storage|netto' in self.model.variables else None

    @property
    def size(self) -> linopy.Variable | None:
        """Batched size variable with (storage,) dims, or None if no storages have investment."""
        return self.model.variables['storage|size'] if 'storage|size' in self.model.variables else None

    @property
    def invested(self) -> linopy.Variable | None:
        """Batched invested binary variable with (storage,) dims, or None if no optional investments."""
        return self.model.variables['storage|invested'] if 'storage|invested' in self.model.variables else None

    def get_variable(self, name: str, element_id: str | None = None):
        """Get a variable, optionally selecting a specific element."""
        var = self._variables.get(name)
        if var is None:
            return None
        if element_id is not None:
            return var.sel({self.dim_name: element_id})
        return var

    # === Investment effect properties (used by EffectsModel) ===

    @functools.cached_property
    def effects_per_size(self) -> xr.DataArray | None:
        """Combined effects_of_investment_per_size with (storage, effect) dims."""
        if not hasattr(self, '_invest_params') or not self._invest_params:
            return None
        from .features import InvestmentHelpers

        element_ids = [eid for eid in self.investment_ids if self._invest_params[eid].effects_of_investment_per_size]
        if not element_ids:
            return None
        effects_dict = InvestmentHelpers.collect_effects(
            self._invest_params, element_ids, 'effects_of_investment_per_size', self.dim_name
        )
        return InvestmentHelpers.build_effect_factors(effects_dict, element_ids, self.dim_name)

    @functools.cached_property
    def effects_of_investment(self) -> xr.DataArray | None:
        """Combined effects_of_investment with (storage, effect) dims for non-mandatory."""
        if not hasattr(self, '_invest_params') or not self._invest_params:
            return None
        from .features import InvestmentHelpers

        element_ids = [eid for eid in self.optional_investment_ids if self._invest_params[eid].effects_of_investment]
        if not element_ids:
            return None
        effects_dict = InvestmentHelpers.collect_effects(
            self._invest_params, element_ids, 'effects_of_investment', self.dim_name
        )
        return InvestmentHelpers.build_effect_factors(effects_dict, element_ids, self.dim_name)

    @functools.cached_property
    def effects_of_retirement(self) -> xr.DataArray | None:
        """Combined effects_of_retirement with (storage, effect) dims for non-mandatory."""
        if not hasattr(self, '_invest_params') or not self._invest_params:
            return None
        from .features import InvestmentHelpers

        element_ids = [eid for eid in self.optional_investment_ids if self._invest_params[eid].effects_of_retirement]
        if not element_ids:
            return None
        effects_dict = InvestmentHelpers.collect_effects(
            self._invest_params, element_ids, 'effects_of_retirement', self.dim_name
        )
        return InvestmentHelpers.build_effect_factors(effects_dict, element_ids, self.dim_name)

    @functools.cached_property
    def effects_of_investment_mandatory(self) -> list[tuple[str, dict[str, float | xr.DataArray]]]:
        """List of (element_id, effects_dict) for mandatory investments with fixed effects."""
        if not hasattr(self, '_invest_params') or not self._invest_params:
            return []

        result = []
        for eid in self.investment_ids:
            params = self._invest_params[eid]
            if params.mandatory and params.effects_of_investment:
                effects_dict = {
                    k: v
                    for k, v in params.effects_of_investment.items()
                    if v is not None and not (np.isscalar(v) and np.isnan(v))
                }
                if effects_dict:
                    result.append((eid, effects_dict))
        return result

    @functools.cached_property
    def effects_of_retirement_constant(self) -> list[tuple[str, dict[str, float | xr.DataArray]]]:
        """List of (element_id, effects_dict) for retirement constant parts."""
        if not hasattr(self, '_invest_params') or not self._invest_params:
            return []

        result = []
        for eid in self.optional_investment_ids:
            params = self._invest_params[eid]
            if params.effects_of_retirement:
                effects_dict = {
                    k: v
                    for k, v in params.effects_of_retirement.items()
                    if v is not None and not (np.isscalar(v) and np.isnan(v))
                }
                if effects_dict:
                    result.append((eid, effects_dict))
        return result

    def _create_piecewise_effects(self) -> None:
        """Create batched piecewise effects for storages with piecewise_effects_of_investment.

        Uses PiecewiseHelpers for pad-to-max batching across all storages with
        piecewise effects. Creates batched segment variables, share variables,
        and coupling constraints.
        """
        from .features import PiecewiseHelpers

        dim = self.dim_name
        size_var = self._variables.get('size')
        invested_var = self._variables.get('invested')

        if size_var is None:
            return

        # Find storages with piecewise effects
        storages_with_piecewise = [
            s
            for s in self.storages_with_investment
            if s.capacity_in_flow_hours.piecewise_effects_of_investment is not None
        ]

        if not storages_with_piecewise:
            return

        element_ids = [s.label_full for s in storages_with_piecewise]

        # Collect segment counts
        segment_counts = {
            s.label_full: len(self._invest_params[s.label_full].piecewise_effects_of_investment.piecewise_origin)
            for s in storages_with_piecewise
        }

        # Build segment mask
        max_segments, segment_mask = PiecewiseHelpers.collect_segment_info(element_ids, segment_counts, dim)

        # Collect origin breakpoints (for size)
        origin_breakpoints = {}
        for s in storages_with_piecewise:
            sid = s.label_full
            piecewise_origin = self._invest_params[sid].piecewise_effects_of_investment.piecewise_origin
            starts = [p.start for p in piecewise_origin]
            ends = [p.end for p in piecewise_origin]
            origin_breakpoints[sid] = (starts, ends)

        origin_starts, origin_ends = PiecewiseHelpers.pad_breakpoints(
            element_ids, origin_breakpoints, max_segments, dim
        )

        # Collect all effect names across all storages
        all_effect_names: set[str] = set()
        for s in storages_with_piecewise:
            sid = s.label_full
            shares = self._invest_params[sid].piecewise_effects_of_investment.piecewise_shares
            all_effect_names.update(shares.keys())

        # Collect breakpoints for each effect
        effect_breakpoints: dict[str, tuple[xr.DataArray, xr.DataArray]] = {}
        for effect_name in all_effect_names:
            breakpoints = {}
            for s in storages_with_piecewise:
                sid = s.label_full
                shares = self._invest_params[sid].piecewise_effects_of_investment.piecewise_shares
                if effect_name in shares:
                    piecewise = shares[effect_name]
                    starts = [p.start for p in piecewise]
                    ends = [p.end for p in piecewise]
                else:
                    # This storage doesn't have this effect - use zeros
                    starts = [0.0] * segment_counts[sid]
                    ends = [0.0] * segment_counts[sid]
                breakpoints[sid] = (starts, ends)

            starts, ends = PiecewiseHelpers.pad_breakpoints(element_ids, breakpoints, max_segments, dim)
            effect_breakpoints[effect_name] = (starts, ends)

        # Create batched piecewise variables
        base_coords = self.model.get_coords(['period', 'scenario'])
        name_prefix = f'{dim}|piecewise_effects'  # Tied to element type (storage)
        piecewise_vars = PiecewiseHelpers.create_piecewise_variables(
            self.model,
            element_ids,
            max_segments,
            dim,
            segment_mask,
            base_coords,
            name_prefix,
        )

        # Build zero_point array if any storages are non-mandatory
        zero_point = None
        if invested_var is not None:
            non_mandatory_ids = [sid for sid in element_ids if not self._invest_params[sid].mandatory]
            if non_mandatory_ids:
                available_ids = [sid for sid in non_mandatory_ids if sid in invested_var.coords.get(dim, [])]
                if available_ids:
                    zero_point = invested_var.sel({dim: element_ids})

        # Create piecewise constraints
        PiecewiseHelpers.create_piecewise_constraints(
            self.model,
            piecewise_vars,
            segment_mask,
            zero_point,
            dim,
            name_prefix,
        )

        # Create coupling constraint for size (origin)
        size_subset = size_var.sel({dim: element_ids})
        PiecewiseHelpers.create_coupling_constraint(
            self.model,
            size_subset,
            piecewise_vars['lambda0'],
            piecewise_vars['lambda1'],
            origin_starts,
            origin_ends,
            f'{name_prefix}|size|coupling',
        )

        # Create share variables and coupling constraints for each effect
        import pandas as pd

        coords_dict = {dim: pd.Index(element_ids, name=dim)}
        if base_coords is not None:
            coords_dict.update(dict(base_coords))
        share_coords = xr.Coordinates(coords_dict)

        for effect_name in all_effect_names:
            # Create batched share variable
            share_var = self.model.add_variables(
                lower=-np.inf,
                upper=np.inf,
                coords=share_coords,
                name=f'{name_prefix}|{effect_name}',
            )

            # Create coupling constraint for this share
            starts, ends = effect_breakpoints[effect_name]
            PiecewiseHelpers.create_coupling_constraint(
                self.model,
                share_var,
                piecewise_vars['lambda0'],
                piecewise_vars['lambda1'],
                starts,
                ends,
                f'{name_prefix}|{effect_name}|coupling',
            )

            # Add to effects (sum over element dimension for periodic share)
            self.model.effects.add_share_to_effects(
                name=f'{name_prefix}|{effect_name}',
                expressions={effect_name: share_var.sum(dim)},
                target='periodic',
            )

        logger.debug(f'Created batched piecewise effects for {len(element_ids)} storages')


class StorageModelProxy(ComponentModel):
    """Lightweight proxy for Storage elements when using type-level modeling.

    Instead of creating its own variables and constraints, this proxy
    provides access to the variables created by StoragesModel.
    """

    element: Storage

    def __init__(self, model: FlowSystemModel, element: Storage):
        # Set _storages_model BEFORE super().__init__() because _do_modeling() may use it
        self._storages_model = model._storages_model
        super().__init__(model, element)

        # Register variables from StoragesModel
        if self._storages_model is not None:
            charge_state = self._storages_model.get_variable('charge', self.label_full)
            if charge_state is not None:
                self.register_variable(charge_state, 'charge_state')

            netto_discharge = self._storages_model.get_variable('netto', self.label_full)
            if netto_discharge is not None:
                self.register_variable(netto_discharge, 'netto_discharge')

    def _do_modeling(self):
        """Skip most modeling - StoragesModel handles variables and constraints.

        Still creates FlowModels for charging/discharging flows and investment model.
        """
        # Create flow models for charging/discharging
        all_flows = self.element.inputs + self.element.outputs

        # Set status_parameters on flows if needed (from ComponentModel)
        if self.element.status_parameters:
            for flow in all_flows:
                if flow.status_parameters is None:
                    flow.status_parameters = StatusParameters()
                    flow.status_parameters.link_to_flow_system(
                        self._model.flow_system, f'{flow.label_full}|status_parameters'
                    )

        if self.element.prevent_simultaneous_flows:
            for flow in self.element.prevent_simultaneous_flows:
                if flow.status_parameters is None:
                    flow.status_parameters = StatusParameters()
                    flow.status_parameters.link_to_flow_system(
                        self._model.flow_system, f'{flow.label_full}|status_parameters'
                    )

        # Note: All storage modeling is now handled by StoragesModel type-level model:
        # - Variables: charge_state, netto_discharge, size, invested
        # - Constraints: netto_discharge, energy_balance, initial/final, balanced_sizes
        #
        # Flow modeling is handled by FlowsModel type-level model.
        # Investment modeling for storages is handled by StoragesModel.create_investment_model().
        # The balanced_sizes constraint is handled by StoragesModel._add_balanced_flow_sizes_constraint().
        #
        # This proxy class only exists for backwards compatibility.

    @property
    def investment(self):
        """Investment feature - provides access to batched investment variables for this storage.

        Returns a proxy object with size/invested properties that select this storage's
        portion of the batched investment variables.
        """
        if not isinstance(self.element.capacity_in_flow_hours, InvestParameters):
            return None

        if self._storages_model.size is None:
            return None

        # Return a proxy that provides size/invested for this specific element
        return InvestmentProxy(self._storages_model, self.label_full, dim_name='storage')

    @property
    def charge_state(self) -> linopy.Variable:
        """Charge state variable."""
        return self['charge_state']

    @property
    def netto_discharge(self) -> linopy.Variable:
        """Netto discharge variable."""
        return self['netto_discharge']


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
    ):
        super().__init__(
            label,
            inputs=inputs,
            outputs=outputs,
            prevent_simultaneous_flows=(inputs or []) + (outputs or []) if prevent_simultaneous_flow_rates else None,
            meta_data=meta_data,
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
    ):
        self.prevent_simultaneous_flow_rates = prevent_simultaneous_flow_rates
        super().__init__(
            label,
            outputs=outputs,
            meta_data=meta_data,
            prevent_simultaneous_flows=outputs if prevent_simultaneous_flow_rates else None,
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
    ):
        """Initialize a Sink (consumes flow from the system).

        Args:
            label: Unique element label.
            inputs: Input flows for the sink.
            meta_data: Arbitrary metadata attached to the element.
            prevent_simultaneous_flow_rates: If True, prevents simultaneous nonzero flow rates
                across the element's inputs by wiring that restriction into the base Component setup.
        """

        self.prevent_simultaneous_flow_rates = prevent_simultaneous_flow_rates
        super().__init__(
            label,
            inputs=inputs,
            meta_data=meta_data,
            prevent_simultaneous_flows=inputs if prevent_simultaneous_flow_rates else None,
        )
