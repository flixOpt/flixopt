"""
This module contains the basic components of the flixopt framework.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Literal

import linopy
import numpy as np

from . import utils
from .core import NumericData, NumericDataTS, PlausibilityError, Scalar, TimeSeries
from .elements import Component, ComponentModel, Flow
from .features import InvestmentModel, OnOffModel, PiecewiseModel
from .interface import InvestParameters, OnOffParameters, PiecewiseConversion
from .structure import SystemModel, register_class_for_io

if TYPE_CHECKING:
    from .flow_system import FlowSystem

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

    Args:
        label: The label of the Element. Used to identify it in the FlowSystem.
        inputs: list of input Flows that feed into the converter.
        outputs: list of output Flows that are produced by the converter.
        on_off_parameters: Information about on and off state of LinearConverter.
            Component is On/Off if all connected Flows are On/Off. This induces an
            On-Variable (binary) in all Flows! If possible, use OnOffParameters in a
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
        Conversion factors define linear relationships.
        `{flow1: a1, flow2: a2, ...}` leads to `a1×flow_rate1 + a2×flow_rate2 + ... = 0`
        Unfortunately the current input format doest read intuitively:
        {"electricity": 1, "H2": 50} means that the electricity_in flow rate is multiplied by 1
        and the hydrogen_out flow rate is multiplied by 50. THis leads to 50 electricity --> 1 H2.

        The system must have fewer conversion factors than total flows (degrees of freedom > 0)
        to avoid over-constraining the problem. For n total flows, use at most n-1 conversion factors.

        When using piecewise_conversion, the converter operates on one piece at a time,
        with binary variables determining which piece is active.

    """

    def __init__(
        self,
        label: str,
        inputs: list[Flow],
        outputs: list[Flow],
        on_off_parameters: OnOffParameters | None = None,
        conversion_factors: list[dict[str, NumericDataTS]] = None,
        piecewise_conversion: PiecewiseConversion | None = None,
        meta_data: dict | None = None,
    ):
        super().__init__(label, inputs, outputs, on_off_parameters, meta_data=meta_data)
        self.conversion_factors = conversion_factors or []
        self.piecewise_conversion = piecewise_conversion

    def create_model(self, model: SystemModel) -> LinearConverterModel:
        self._plausibility_checks()
        self.model = LinearConverterModel(model, self)
        return self.model

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
                    raise PlausibilityError(
                        f'piecewise_conversion (in {self.label_full}) and variable size '
                        f'(in flow {flow.label_full}) do not make sense together!'
                    )

    def transform_data(self, flow_system: FlowSystem):
        super().transform_data(flow_system)
        if self.conversion_factors:
            self.conversion_factors = self._transform_conversion_factors(flow_system)
        if self.piecewise_conversion:
            self.piecewise_conversion.transform_data(flow_system, f'{self.label_full}|PiecewiseConversion')

    def _transform_conversion_factors(self, flow_system: FlowSystem) -> list[dict[str, TimeSeries]]:
        """macht alle Faktoren, die nicht TimeSeries sind, zu TimeSeries"""
        list_of_conversion_factors = []
        for idx, conversion_factor in enumerate(self.conversion_factors):
            transformed_dict = {}
            for flow, values in conversion_factor.items():
                # TODO: Might be better to use the label of the component instead of the flow
                transformed_dict[flow] = flow_system.create_time_series(
                    f'{self.flows[flow].label_full}|conversion_factor{idx}', values
                )
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

    Args:
        label: The label of the Element. Used to identify it in the FlowSystem.
        charging: Incoming flow for loading the storage. Represents energy or material
            flowing into the storage system.
        discharging: Outgoing flow for unloading the storage. Represents energy or
            material flowing out of the storage system.
        capacity_in_flow_hours: Nominal capacity/size of the storage in flow-hours
            (e.g., kWh for electrical storage, m³ or kg for material storage). Can be a scalar
            for fixed capacity or InvestParameters for optimization.
        relative_minimum_charge_state: Minimum relative charge state (0-1 range).
            Prevents deep discharge that could damage equipment. Default is 0.
        relative_maximum_charge_state: Maximum relative charge state (0-1 range).
            Accounts for practical capacity limits, safety margins or temperature impacts. Default is 1.
        initial_charge_state: Storage charge state at the beginning of the time horizon.
            Can be numeric value or 'lastValueOfSim', which is recommended for if the initial start state is not known.
            Default is 0.
        minimal_final_charge_state: Minimum absolute charge state required at the end
            of the time horizon. Useful for ensuring energy security or meeting contracts.
        maximal_final_charge_state: Maximum absolute charge state allowed at the end
            of the time horizon. Useful for preventing overcharge or managing inventory.
        eta_charge: Charging efficiency factor (0-1 range). Accounts for conversion
            losses during charging. Default is 1 (perfect efficiency).
        eta_discharge: Discharging efficiency factor (0-1 range). Accounts for
            conversion losses during discharging. Default is 1 (perfect efficiency).
        relative_loss_per_hour: Self-discharge rate per hour (typically 0-0.1 range).
            Represents standby losses, leakage, or degradation. Default is 0.
        prevent_simultaneous_charge_and_discharge: If True, prevents charging and
            discharging simultaneously. Increases binary variables but improves model
            realism and solution interpretation. Default is True.
        meta_data: Used to store additional information about the Element. Not used
            internally, but saved in results. Only use Python native types.

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
            initial_charge_state='lastValueOfSim',  # Ensuring no deficit compared to start
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
        Charge state evolution follows the equation:
        charge[t+1] = charge[t] × (1-loss_rate)^hours_per_step +
                      charge_flow[t] × eta_charge × hours_per_step -
                      discharge_flow[t] × hours_per_step / eta_discharge

        All efficiency parameters (eta_charge, eta_discharge) are dimensionless (0-1 range).
        The relative_loss_per_hour parameter represents exponential decay per hour.

        When prevent_simultaneous_charge_and_discharge is True, binary variables are
        created to enforce mutual exclusivity, which increases solution time but
        prevents unrealistic simultaneous charging and discharging.

        Initial and final charge state constraints use absolute values (not relative),
        matching the capacity_in_flow_hours units.
    """

    def __init__(
        self,
        label: str,
        charging: Flow,
        discharging: Flow,
        capacity_in_flow_hours: Scalar | InvestParameters,
        relative_minimum_charge_state: NumericData = 0,
        relative_maximum_charge_state: NumericData = 1,
        initial_charge_state: Scalar | Literal['lastValueOfSim'] = 0,
        minimal_final_charge_state: Scalar | None = None,
        maximal_final_charge_state: Scalar | None = None,
        eta_charge: NumericData = 1,
        eta_discharge: NumericData = 1,
        relative_loss_per_hour: NumericData = 0,
        prevent_simultaneous_charge_and_discharge: bool = True,
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
        self.relative_minimum_charge_state: NumericDataTS = relative_minimum_charge_state
        self.relative_maximum_charge_state: NumericDataTS = relative_maximum_charge_state

        self.initial_charge_state = initial_charge_state
        self.minimal_final_charge_state = minimal_final_charge_state
        self.maximal_final_charge_state = maximal_final_charge_state

        self.eta_charge: NumericDataTS = eta_charge
        self.eta_discharge: NumericDataTS = eta_discharge
        self.relative_loss_per_hour: NumericDataTS = relative_loss_per_hour
        self.prevent_simultaneous_charge_and_discharge = prevent_simultaneous_charge_and_discharge

    def create_model(self, model: SystemModel) -> StorageModel:
        self._plausibility_checks()
        self.model = StorageModel(model, self)
        return self.model

    def transform_data(self, flow_system: FlowSystem) -> None:
        super().transform_data(flow_system)
        self.relative_minimum_charge_state = flow_system.create_time_series(
            f'{self.label_full}|relative_minimum_charge_state',
            self.relative_minimum_charge_state,
            needs_extra_timestep=True,
        )
        self.relative_maximum_charge_state = flow_system.create_time_series(
            f'{self.label_full}|relative_maximum_charge_state',
            self.relative_maximum_charge_state,
            needs_extra_timestep=True,
        )
        self.eta_charge = flow_system.create_time_series(f'{self.label_full}|eta_charge', self.eta_charge)
        self.eta_discharge = flow_system.create_time_series(f'{self.label_full}|eta_discharge', self.eta_discharge)
        self.relative_loss_per_hour = flow_system.create_time_series(
            f'{self.label_full}|relative_loss_per_hour', self.relative_loss_per_hour
        )
        if isinstance(self.capacity_in_flow_hours, InvestParameters):
            self.capacity_in_flow_hours.transform_data(flow_system)

    def _plausibility_checks(self) -> None:
        """
        Check for infeasible or uncommon combinations of parameters
        """
        super()._plausibility_checks()
        if utils.is_number(self.initial_charge_state):
            if isinstance(self.capacity_in_flow_hours, InvestParameters):
                if self.capacity_in_flow_hours.fixed_size is None:
                    maximum_capacity = self.capacity_in_flow_hours.maximum_size
                    minimum_capacity = self.capacity_in_flow_hours.minimum_size
                else:
                    maximum_capacity = self.capacity_in_flow_hours.fixed_size
                    minimum_capacity = self.capacity_in_flow_hours.fixed_size
            else:
                maximum_capacity = self.capacity_in_flow_hours
                minimum_capacity = self.capacity_in_flow_hours

            # initial capacity >= allowed min for maximum_size:
            minimum_inital_capacity = maximum_capacity * self.relative_minimum_charge_state.isel(time=1)
            # initial capacity <= allowed max for minimum_size:
            maximum_inital_capacity = minimum_capacity * self.relative_maximum_charge_state.isel(time=1)

            if self.initial_charge_state > maximum_inital_capacity:
                raise ValueError(
                    f'{self.label_full}: {self.initial_charge_state=} '
                    f'is above allowed maximum charge_state {maximum_inital_capacity}'
                )
            if self.initial_charge_state < minimum_inital_capacity:
                raise ValueError(
                    f'{self.label_full}: {self.initial_charge_state=} '
                    f'is below allowed minimum charge_state {minimum_inital_capacity}'
                )
        elif self.initial_charge_state != 'lastValueOfSim':
            raise ValueError(f'{self.label_full}: {self.initial_charge_state=} has an invalid value')


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
            Automatically creates binary variables for on/off states.
        on_off_parameters: Parameters defining binary operation constraints and costs.
        prevent_simultaneous_flows_in_both_directions: If True, prevents simultaneous
            flow in both directions. Increases binary variables but reflects physical
            reality for most transmission systems. Default is True.
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

        Material conveyor with on/off operation:

        ```python
        conveyor_belt = Transmission(
            label='material_transport',
            in1=loading_station,
            out1=unloading_station,
            absolute_losses=25,  # 25 kW motor power when running
            on_off_parameters=OnOffParameters(
                effects_per_switch_on={'maintenance': 0.1},
                consecutive_on_hours_min=2,  # Minimum 2-hour operation
                switch_on_total_max=10,  # Maximum 10 starts per day
            ),
        )
        ```

    Note:
        The transmission equation balances flows with losses:
        output_flow = input_flow × (1 - relative_losses) - absolute_losses

        For bidirectional transmission, each direction has independent loss calculations.

        When using InvestParameters on in1, the capacity automatically applies to in2
        to maintain consistent bidirectional capacity without additional investment variables.

        Absolute losses force the creation of binary on/off variables, which increases
        computational complexity but enables realistic modeling of equipment with
        standby power consumption.

    """

    def __init__(
        self,
        label: str,
        in1: Flow,
        out1: Flow,
        in2: Flow | None = None,
        out2: Flow | None = None,
        relative_losses: NumericDataTS | None = None,
        absolute_losses: NumericDataTS | None = None,
        on_off_parameters: OnOffParameters | None = None,
        prevent_simultaneous_flows_in_both_directions: bool = True,
        meta_data: dict | None = None,
    ):
        super().__init__(
            label,
            inputs=[flow for flow in (in1, in2) if flow is not None],
            outputs=[flow for flow in (out1, out2) if flow is not None],
            on_off_parameters=on_off_parameters,
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
        # Check Investments
        for flow in [self.out1, self.in2, self.out2]:
            if flow is not None and isinstance(flow.size, InvestParameters):
                raise ValueError(
                    'Transmission currently does not support separate InvestParameters for Flows. '
                    'Please use Flow in1. The size of in2 is equal to in1. THis is handled internally'
                )

    def create_model(self, model) -> TransmissionModel:
        self._plausibility_checks()
        self.model = TransmissionModel(model, self)
        return self.model

    def transform_data(self, flow_system: FlowSystem) -> None:
        super().transform_data(flow_system)
        self.relative_losses = flow_system.create_time_series(
            f'{self.label_full}|relative_losses', self.relative_losses
        )
        self.absolute_losses = flow_system.create_time_series(
            f'{self.label_full}|absolute_losses', self.absolute_losses
        )


class TransmissionModel(ComponentModel):
    def __init__(self, model: SystemModel, element: Transmission):
        super().__init__(model, element)
        self.element: Transmission = element
        self.on_off: OnOffModel | None = None

    def do_modeling(self):
        """Initiates all FlowModels"""
        # Force On Variable if absolute losses are present
        if (self.element.absolute_losses is not None) and np.any(self.element.absolute_losses.active_data != 0):
            for flow in self.element.inputs + self.element.outputs:
                if flow.on_off_parameters is None:
                    flow.on_off_parameters = OnOffParameters()

        # Make sure either None or both in Flows have InvestParameters
        if self.element.in2 is not None:
            if isinstance(self.element.in1.size, InvestParameters) and not isinstance(
                self.element.in2.size, InvestParameters
            ):
                self.element.in2.size = InvestParameters(maximum_size=self.element.in1.size.maximum_size)

        super().do_modeling()

        # first direction
        self.create_transmission_equation('dir1', self.element.in1, self.element.out1)

        # second direction:
        if self.element.in2 is not None:
            self.create_transmission_equation('dir2', self.element.in2, self.element.out2)

        # equate size of both directions
        if isinstance(self.element.in1.size, InvestParameters) and self.element.in2 is not None:
            # eq: in1.size = in2.size
            self.add(
                self._model.add_constraints(
                    self.element.in1.model._investment.size == self.element.in2.model._investment.size,
                    name=f'{self.label_full}|same_size',
                ),
                'same_size',
            )

    def create_transmission_equation(self, name: str, in_flow: Flow, out_flow: Flow) -> linopy.Constraint:
        """Creates an Equation for the Transmission efficiency and adds it to the model"""
        # eq: out(t) + on(t)*loss_abs(t) = in(t)*(1 - loss_rel(t))
        con_transmission = self.add(
            self._model.add_constraints(
                out_flow.model.flow_rate == -in_flow.model.flow_rate * (self.element.relative_losses.active_data - 1),
                name=f'{self.label_full}|{name}',
            ),
            name,
        )

        if self.element.absolute_losses is not None:
            con_transmission.lhs += in_flow.model.on_off.on * self.element.absolute_losses.active_data

        return con_transmission


class LinearConverterModel(ComponentModel):
    def __init__(self, model: SystemModel, element: LinearConverter):
        super().__init__(model, element)
        self.element: LinearConverter = element
        self.on_off: OnOffModel | None = None
        self.piecewise_conversion: PiecewiseConversion | None = None

    def do_modeling(self):
        super().do_modeling()

        # conversion_factors:
        if self.element.conversion_factors:
            all_input_flows = set(self.element.inputs)
            all_output_flows = set(self.element.outputs)

            # für alle linearen Gleichungen:
            for i, conv_factors in enumerate(self.element.conversion_factors):
                used_flows = set([self.element.flows[flow_label] for flow_label in conv_factors])
                used_inputs: set = all_input_flows & used_flows
                used_outputs: set = all_output_flows & used_flows

                self.add(
                    self._model.add_constraints(
                        sum([flow.model.flow_rate * conv_factors[flow.label].active_data for flow in used_inputs])
                        == sum([flow.model.flow_rate * conv_factors[flow.label].active_data for flow in used_outputs]),
                        name=f'{self.label_full}|conversion_{i}',
                    )
                )

        else:
            # TODO: Improve Inclusion of OnOffParameters. Instead of creating a Binary in every flow, the binary could only be part of the Piece itself
            piecewise_conversion = {
                self.element.flows[flow].model.flow_rate.name: piecewise
                for flow, piecewise in self.element.piecewise_conversion.items()
            }

            self.piecewise_conversion = self.add(
                PiecewiseModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    piecewise_variables=piecewise_conversion,
                    zero_point=self.on_off.on if self.on_off is not None else False,
                    as_time_series=True,
                )
            )
            self.piecewise_conversion.do_modeling()


class StorageModel(ComponentModel):
    """Model of Storage"""

    def __init__(self, model: SystemModel, element: Storage):
        super().__init__(model, element)
        self.element: Storage = element
        self.charge_state: linopy.Variable | None = None
        self.netto_discharge: linopy.Variable | None = None
        self._investment: InvestmentModel | None = None

    def do_modeling(self):
        super().do_modeling()

        lb, ub = self.absolute_charge_state_bounds
        self.charge_state = self.add(
            self._model.add_variables(
                lower=lb, upper=ub, coords=self._model.coords_extra, name=f'{self.label_full}|charge_state'
            ),
            'charge_state',
        )
        self.netto_discharge = self.add(
            self._model.add_variables(coords=self._model.coords, name=f'{self.label_full}|netto_discharge'),
            'netto_discharge',
        )
        # netto_discharge:
        # eq: nettoFlow(t) - discharging(t) + charging(t) = 0
        self.add(
            self._model.add_constraints(
                self.netto_discharge
                == self.element.discharging.model.flow_rate - self.element.charging.model.flow_rate,
                name=f'{self.label_full}|netto_discharge',
            ),
            'netto_discharge',
        )

        charge_state = self.charge_state
        rel_loss = self.element.relative_loss_per_hour.active_data
        hours_per_step = self._model.hours_per_step
        charge_rate = self.element.charging.model.flow_rate
        discharge_rate = self.element.discharging.model.flow_rate
        eff_charge = self.element.eta_charge.active_data
        eff_discharge = self.element.eta_discharge.active_data

        self.add(
            self._model.add_constraints(
                charge_state.isel(time=slice(1, None))
                == charge_state.isel(time=slice(None, -1)) * ((1 - rel_loss) ** hours_per_step)
                + charge_rate * eff_charge * hours_per_step
                - discharge_rate * eff_discharge * hours_per_step,
                name=f'{self.label_full}|charge_state',
            ),
            'charge_state',
        )

        if isinstance(self.element.capacity_in_flow_hours, InvestParameters):
            self._investment = InvestmentModel(
                model=self._model,
                label_of_element=self.label_of_element,
                parameters=self.element.capacity_in_flow_hours,
                defining_variable=self.charge_state,
                relative_bounds_of_defining_variable=self.relative_charge_state_bounds,
            )
            self.sub_models.append(self._investment)
            self._investment.do_modeling()

        # Initial charge state
        self._initial_and_final_charge_state()

    def _initial_and_final_charge_state(self):
        if self.element.initial_charge_state is not None:
            name_short = 'initial_charge_state'
            name = f'{self.label_full}|{name_short}'

            if utils.is_number(self.element.initial_charge_state):
                self.add(
                    self._model.add_constraints(
                        self.charge_state.isel(time=0) == self.element.initial_charge_state, name=name
                    ),
                    name_short,
                )
            elif self.element.initial_charge_state == 'lastValueOfSim':
                self.add(
                    self._model.add_constraints(
                        self.charge_state.isel(time=0) == self.charge_state.isel(time=-1), name=name
                    ),
                    name_short,
                )
            else:  # TODO: Validation in Storage Class, not in Model
                raise PlausibilityError(
                    f'initial_charge_state has undefined value: {self.element.initial_charge_state}'
                )

        if self.element.maximal_final_charge_state is not None:
            self.add(
                self._model.add_constraints(
                    self.charge_state.isel(time=-1) <= self.element.maximal_final_charge_state,
                    name=f'{self.label_full}|final_charge_max',
                ),
                'final_charge_max',
            )

        if self.element.minimal_final_charge_state is not None:
            self.add(
                self._model.add_constraints(
                    self.charge_state.isel(time=-1) >= self.element.minimal_final_charge_state,
                    name=f'{self.label_full}|final_charge_min',
                ),
                'final_charge_min',
            )

    @property
    def absolute_charge_state_bounds(self) -> tuple[NumericData, NumericData]:
        relative_lower_bound, relative_upper_bound = self.relative_charge_state_bounds
        if not isinstance(self.element.capacity_in_flow_hours, InvestParameters):
            return (
                relative_lower_bound * self.element.capacity_in_flow_hours,
                relative_upper_bound * self.element.capacity_in_flow_hours,
            )
        else:
            return (
                relative_lower_bound * self.element.capacity_in_flow_hours.minimum_size,
                relative_upper_bound * self.element.capacity_in_flow_hours.maximum_size,
            )

    @property
    def relative_charge_state_bounds(self) -> tuple[NumericData, NumericData]:
        return (
            self.element.relative_minimum_charge_state.active_data,
            self.element.relative_maximum_charge_state.active_data,
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
        inputs: list[Flow] = None,
        outputs: list[Flow] = None,
        prevent_simultaneous_flow_rates: bool = True,
        meta_data: dict | None = None,
        **kwargs,
    ):
        source = kwargs.pop('source', None)
        sink = kwargs.pop('sink', None)
        prevent_simultaneous_sink_and_source = kwargs.pop('prevent_simultaneous_sink_and_source', None)
        if source is not None:
            warnings.warn(
                'The use of the source argument is deprecated. Use the outputs argument instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            if outputs is not None:
                raise ValueError('Either source or outputs can be specified, but not both.')
            outputs = [source]

        if sink is not None:
            warnings.warn(
                'The use of the sink argument is deprecated. Use the inputs argument instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            if inputs is not None:
                raise ValueError('Either sink or outputs can be specified, but not both.')
            inputs = [sink]

        if prevent_simultaneous_sink_and_source is not None:
            warnings.warn(
                'The use of the prevent_simultaneous_sink_and_source argument is deprecated. Use the prevent_simultaneous_flow_rates argument instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            prevent_simultaneous_flow_rates = prevent_simultaneous_sink_and_source

        super().__init__(
            label,
            inputs=inputs,
            outputs=outputs,
            prevent_simultaneous_flows=(inputs or []) + (outputs or []) if prevent_simultaneous_flow_rates else None,
            meta_data=meta_data,
        )
        self.prevent_simultaneous_flow_rates = prevent_simultaneous_flow_rates

    @property
    def source(self) -> Flow:
        warnings.warn(
            'The source property is deprecated. Use the outputs property instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.outputs[0]

    @property
    def sink(self) -> Flow:
        warnings.warn(
            'The sink property is deprecated. Use the outputs property instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.inputs[0]

    @property
    def prevent_simultaneous_sink_and_source(self) -> bool:
        warnings.warn(
            'The prevent_simultaneous_sink_and_source property is deprecated. Use the prevent_simultaneous_flow_rates property instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.prevent_simultaneous_flow_rates


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
        outputs: list[Flow] = None,
        meta_data: dict | None = None,
        prevent_simultaneous_flow_rates: bool = False,
        **kwargs,
    ):
        source = kwargs.pop('source', None)
        if source is not None:
            warnings.warn(
                'The use of the source argument is deprecated. Use the outputs argument instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            if outputs is not None:
                raise ValueError('Either source or outputs can be specified, but not both.')
            outputs = [source]

        self.prevent_simultaneous_flow_rates = prevent_simultaneous_flow_rates
        super().__init__(
            label,
            outputs=outputs,
            meta_data=meta_data,
            prevent_simultaneous_flows=outputs if prevent_simultaneous_flow_rates else None,
        )

    @property
    def source(self) -> Flow:
        warnings.warn(
            'The source property is deprecated. Use the outputs property instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.outputs[0]


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
        inputs: list[Flow] = None,
        meta_data: dict | None = None,
        prevent_simultaneous_flow_rates: bool = False,
        **kwargs,
    ):
        """
        Initialize a Sink (consumes flow from the system).

        Supports legacy `sink=` keyword for backward compatibility (deprecated): if `sink` is provided it is used as the single input flow and a DeprecationWarning is issued; specifying both `inputs` and `sink` raises ValueError.

        Parameters:
            label (str): Unique element label.
            inputs (list[Flow], optional): Input flows for the sink.
            meta_data (dict, optional): Arbitrary metadata attached to the element.
            prevent_simultaneous_flow_rates (bool, optional): If True, prevents simultaneous nonzero flow rates across the element's inputs by wiring that restriction into the base Component setup.

        Note:
            The deprecated `sink` kwarg is accepted for compatibility but will be removed in future releases.
        """
        sink = kwargs.pop('sink', None)
        if sink is not None:
            warnings.warn(
                'The use of the sink argument is deprecated. Use the inputs argument instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            if inputs is not None:
                raise ValueError('Either sink or outputs can be specified, but not both.')
            inputs = [sink]

        self.prevent_simultaneous_flow_rates = prevent_simultaneous_flow_rates
        super().__init__(
            label,
            inputs=inputs,
            meta_data=meta_data,
            prevent_simultaneous_flows=inputs if prevent_simultaneous_flow_rates else None,
        )

    @property
    def sink(self) -> Flow:
        warnings.warn(
            'The sink property is deprecated. Use the outputs property instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.inputs[0]
