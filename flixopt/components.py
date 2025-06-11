"""
This module contains the basic components of the flixopt framework.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Set, Tuple, Union

import linopy
import numpy as np

from . import utils
from .config import CONFIG
from .core import NumericData, NumericDataTS, PlausibilityError, Scalar, TimeSeries
from .elements import Component, ComponentModel, Flow
from .features import InvestmentModel, OnOffModel, PiecewiseModel, StateModel, PreventSimultaneousUsageModel
from .interface import InvestParameters, OnOffParameters, PiecewiseConversion
from .structure import SystemModel, register_class_for_io

if TYPE_CHECKING:
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


@register_class_for_io
class LinearConverter(Component):
    """
    Converts input-Flows into output-Flows via linear conversion factors

    """

    def __init__(
        self,
        label: str,
        inputs: List[Flow],
        outputs: List[Flow],
        on_off_parameters: OnOffParameters = None,
        conversion_factors: List[Dict[str, NumericDataTS]] = None,
        piecewise_conversion: Optional[PiecewiseConversion] = None,
        meta_data: Optional[Dict] = None,
    ):
        """
        Args:
            label: The label of the Element. Used to identify it in the FlowSystem
            inputs: The input Flows
            outputs: The output Flows
            on_off_parameters: Information about on and off state of LinearConverter.
                Component is On/Off, if all connected Flows are On/Off. This induces an On-Variable (binary) in all Flows!
                If possible, use OnOffParameters in a single Flow instead to keep the number of binary variables low.
                See class OnOffParameters.
            conversion_factors: linear relation between flows.
                Either 'conversion_factors' or 'piecewise_conversion' can be used!
            piecewise_conversion: Define a piecewise linear relation between flow rates of different flows.
                Either 'conversion_factors' or 'piecewise_conversion' can be used!
            meta_data: used to store more information about the Element. Is not used internally, but saved in the results. Only use python native types.
        """
        super().__init__(label, inputs, outputs, on_off_parameters, meta_data=meta_data)
        self.conversion_factors = conversion_factors or []
        self.piecewise_conversion = piecewise_conversion

    def create_model(self, model: SystemModel) -> 'LinearConverterModel':
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

    def transform_data(self, flow_system: 'FlowSystem'):
        super().transform_data(flow_system)
        if self.conversion_factors:
            self.conversion_factors = self._transform_conversion_factors(flow_system)
        if self.piecewise_conversion:
            self.piecewise_conversion.transform_data(flow_system, f'{self.label_full}|PiecewiseConversion')

    def _transform_conversion_factors(self, flow_system: 'FlowSystem') -> List[Dict[str, TimeSeries]]:
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
    Used to model the storage of energy or material.
    """

    def __init__(
        self,
        label: str,
        charging: Flow,
        discharging: Flow,
        capacity_in_flow_hours: Union[Scalar, InvestParameters],
        relative_minimum_charge_state: NumericData = 0,
        relative_maximum_charge_state: NumericData = 1,
        initial_charge_state: Union[Scalar, Literal['lastValueOfSim']] = 0,
        minimal_final_charge_state: Optional[Scalar] = None,
        maximal_final_charge_state: Optional[Scalar] = None,
        eta_charge: NumericData = 1,
        eta_discharge: NumericData = 1,
        relative_loss_per_hour: NumericData = 0,
        prevent_simultaneous_charge_and_discharge: bool = True,
        meta_data: Optional[Dict] = None,
    ):
        """
        Storages have one incoming and one outgoing Flow each with an efficiency.
        Further, storages have a `size` and a `charge_state`.
        Similarly to the flow-rate of a Flow, the `size` combined with a relative upper and lower bound
        limits the `charge_state` of the storage.

        For mathematical details take a look at our online documentation

        Args:
            label: The label of the Element. Used to identify it in the FlowSystem
            charging: ingoing flow.
            discharging: outgoing flow.
            capacity_in_flow_hours: nominal capacity/size of the storage
            relative_minimum_charge_state: minimum relative charge state. The default is 0.
            relative_maximum_charge_state: maximum relative charge state. The default is 1.
            initial_charge_state: storage charge_state at the beginning. The default is 0.
            minimal_final_charge_state: minimal value of chargeState at the end of timeseries.
            maximal_final_charge_state: maximal value of chargeState at the end of timeseries.
            eta_charge: efficiency factor of charging/loading. The default is 1.
            eta_discharge: efficiency factor of uncharging/unloading. The default is 1.
            relative_loss_per_hour: loss per chargeState-Unit per hour. The default is 0.
            prevent_simultaneous_charge_and_discharge: If True, loading and unloading at the same time is not possible.
                Increases the number of binary variables, but is recommended for easier evaluation. The default is True.
            meta_data: used to store more information about the Element. Is not used internally, but saved in the results. Only use python native types.
        """
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

    def create_model(self, model: SystemModel) -> 'StorageModel':
        self._plausibility_checks()
        self.model = StorageModel(model, self)
        return self.model

    def transform_data(self, flow_system: 'FlowSystem') -> None:
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
    # TODO: automatic on-Value in Flows if loss_abs
    # TODO: loss_abs must be: investment_size * loss_abs_rel!!!
    # TODO: investmentsize only on 1 flow
    # TODO: automatic investArgs for both in-flows (or alternatively both out-flows!)
    # TODO: optional: capacities should be recognised for losses

    def __init__(
        self,
        label: str,
        in1: Flow,
        out1: Flow,
        in2: Optional[Flow] = None,
        out2: Optional[Flow] = None,
        relative_losses: Optional[NumericDataTS] = None,
        absolute_losses: Optional[NumericDataTS] = None,
        on_off_parameters: OnOffParameters = None,
        prevent_simultaneous_flows_in_both_directions: bool = True,
        meta_data: Optional[Dict] = None,
    ):
        """
        Initializes a Transmission component (Pipe, cable, ...) that models the flows between two sides
        with potential losses.

        Args:
            label: The label of the Element. Used to identify it in the FlowSystem
            in1: The inflow at side A. Pass InvestmentParameters here.
            out1: The outflow at side B.
            in2: The optional inflow at side B.
                If in1 got InvestParameters, the size of this Flow will be equal to in1 (with no extra effects!)
            out2: The optional outflow at side A.
            relative_losses: The relative loss between inflow and outflow, e.g., 0.02 for 2% loss.
            absolute_losses: The absolute loss, occur only when the Flow is on. Induces the creation of the ON-Variable
            on_off_parameters: Parameters defining the on/off behavior of the component.
            prevent_simultaneous_flows_in_both_directions: If True, inflow and outflow are not allowed to be both non-zero at same timestep.
            meta_data: used to store more information about the Element. Is not used internally, but saved in the results. Only use python native types.
        """
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

    def create_model(self, model) -> 'TransmissionModel':
        self._plausibility_checks()
        self.model = TransmissionModel(model, self)
        return self.model

    def transform_data(self, flow_system: 'FlowSystem') -> None:
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
        self.on_off: Optional[OnOffModel] = None

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
        self.on_off: Optional[OnOffModel] = None
        self.piecewise_conversion: Optional[PiecewiseConversion] = None

    def do_modeling(self):
        super().do_modeling()

        # conversion_factors:
        if self.element.conversion_factors:
            all_input_flows = set(self.element.inputs)
            all_output_flows = set(self.element.outputs)

            # für alle linearen Gleichungen:
            for i, conv_factors in enumerate(self.element.conversion_factors):
                used_flows = set([self.element.flows[flow_label] for flow_label in conv_factors])
                used_inputs: Set = all_input_flows & used_flows
                used_outputs: Set = all_output_flows & used_flows

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
        self.charge_state: Optional[linopy.Variable] = None
        self.netto_discharge: Optional[linopy.Variable] = None
        self._investment: Optional[InvestmentModel] = None

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
                == charge_state.isel(time=slice(None, -1)) * (1 - rel_loss * hours_per_step)
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
    def absolute_charge_state_bounds(self) -> Tuple[NumericData, NumericData]:
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
    def relative_charge_state_bounds(self) -> Tuple[NumericData, NumericData]:
        return (
            self.element.relative_minimum_charge_state.active_data,
            self.element.relative_maximum_charge_state.active_data,
        )


@register_class_for_io
class SourceAndSink(Component):
    """
    class for source (output-flow) and sink (input-flow) in one commponent
    """

    def __init__(
        self,
        label: str,
        source: Flow,
        sink: Flow,
        prevent_simultaneous_sink_and_source: bool = True,
        meta_data: Optional[Dict] = None,
    ):
        """
        Args:
            label: The label of the Element. Used to identify it in the FlowSystem
            source: output-flow of this component
            sink: input-flow of this component
            prevent_simultaneous_sink_and_source: If True, inflow and outflow can not be active simultaniously.
            meta_data: used to store more information about the Element. Is not used internally, but saved in the results. Only use python native types.
        """
        super().__init__(
            label,
            inputs=[sink],
            outputs=[source],
            prevent_simultaneous_flows=[sink, source] if prevent_simultaneous_sink_and_source is True else None,
            meta_data=meta_data,
        )
        self.source = source
        self.sink = sink
        self.prevent_simultaneous_sink_and_source = prevent_simultaneous_sink_and_source


@register_class_for_io
class Source(Component):
    def __init__(self, label: str, source: Flow, meta_data: Optional[Dict] = None):
        """
        Args:
            label: The label of the Element. Used to identify it in the FlowSystem
            source: output-flow of source
            meta_data: used to store more information about the Element. Is not used internally, but saved in the results. Only use python native types.
        """
        super().__init__(label, outputs=[source], meta_data=meta_data)
        self.source = source


@register_class_for_io
class Sink(Component):
    def __init__(self, label: str, sink: Flow, meta_data: Optional[Dict] = None):
        """
        Args:
            label: The label of the Element. Used to identify it in the FlowSystem
            meta_data: used to store more information about the element. Is not used internally, but saved in the results
            sink: input-flow of sink
        """
        super().__init__(label, inputs=[sink], meta_data=meta_data)
        self.sink = sink

@register_class_for_io
class DSMSink(Sink):
    """
    Used to model sinks with the ability to perform demand side management.
    """

    def __init__(
        self,
        label: str,
        sink: Flow,
        initial_demand: NumericData,
        maximum_flow_deficit: NumericData,
        maximum_flow_surplus: NumericData,
        maximum_cumulated_deficit: NumericData,
        maximum_cumulated_surplus: NumericData,
        forward_timeshift: Scalar = None,
        backward_timeshift: Scalar = None,
        relative_loss_per_hour_positive_charge_state: NumericData = 0,
        relative_loss_per_hour_negative_charge_state: NumericData = 0,
        allow_mixed_charge_states: bool = False,
        allow_parallel_charge_and_discharge: bool = False,
        penalty_costs_positive_charge_states: NumericData = None,
        penalty_costs_negative_charge_states: NumericData = None,
        meta_data: Optional[Dict] = None
    ):
        """
        Args:
            label: The label of the Element. Used to identify it in the FlowSystem
            sink: input-flow of DSM sink after DSM
            initial_demand: initial demand of DSM sink before DSM
            maximum_flow_deficit: maximum that the supply flow can fall short of the demand
            maximum_flow_surplus: maximum that the supply flow can exceed the demand
            maximum_cumulated_deficit: maximum cumulated supply deficit in flow hours
            maximum_cumulated_surplus: maximum cumulated supply surplus in flow hours
            forward_timeshift: Maximum number of hours by which the demand can be shifted forward in time. Default is infinite.
            backward_timeshift: Maximum number of hours by which the demand can be shifted backward in time. Default is infinite.
            relative_loss_per_hour_positive_charge_state: loss per chargeState-Unit per hour for positive charge states of the virtual storage. The default is 0.
            relative_loss_per_hour_negative_charge_state: loss per chargeState-Unit per hour for negative charge states of the virtual storage. The default is 0.
            allow_mixed_charge_states: If True, positive and negative charge states can occur simultaneously.
                If False, only one type of charge state is allowed at a time. The default is False.
            allow_parallel_charge_and_discharge: If True, allows simultaneous charging and discharging in one timestep.
                If False, charging and discharging cannot occur simultaneously. The default is False.
            penalty_costs_positive_charge_states: penalty costs per flow hour for loss of comfort due to positive charge states of the virtual storage (e.g. increased room temperature). The default is a small epsilon.
            penalty_costs_negative_charge_states: penalty costs per flow hour for loss of comfort due to negative charge states of the virtual storage (e.g. decreased room temperature). The default is a small epsilon.
            meta_data: used to store more information about the Element. Is not used internally, but saved in the results. Only use python native types.
        """
        super().__init__(
            label,
            sink,
            meta_data
        )

        self.initial_demand: NumericDataTS = initial_demand
        self.forward_timeshift = forward_timeshift
        self.backward_timeshift = backward_timeshift
        self.maximum_flow_deficit: NumericDataTS = maximum_flow_deficit
        self.maximum_flow_surplus: NumericDataTS = maximum_flow_surplus
        self.maximum_cumulated_deficit: NumericDataTS = maximum_cumulated_deficit
        self.maximum_cumulated_surplus: NumericDataTS = maximum_cumulated_surplus

        self.relative_loss_per_hour_positive_charge_state: NumericDataTS = relative_loss_per_hour_positive_charge_state
        self.relative_loss_per_hour_negative_charge_state: NumericDataTS = relative_loss_per_hour_negative_charge_state
        self.allow_mixed_charge_states = allow_mixed_charge_states
        self.allow_parallel_charge_and_discharge = allow_parallel_charge_and_discharge

        self.penalty_costs_positive_charge_states: NumericDataTS = penalty_costs_positive_charge_states if penalty_costs_positive_charge_states is not None else CONFIG.modeling.EPSILON
        self.penalty_costs_negative_charge_states: NumericDataTS = penalty_costs_negative_charge_states if penalty_costs_negative_charge_states is not None else CONFIG.modeling.EPSILON

    def create_model(self, model: SystemModel) -> 'DSMSinkModel':
        self._plausibility_checks(model)

        # calculate amount of timesteps, by which demand can be shifted forward/backward in time
        hours_per_step = model.hours_per_step
        self.timesteps_forward = int(self.forward_timeshift/hours_per_step.values[0]) if self.forward_timeshift is not None else None
        self.timesteps_backward = int(self.backward_timeshift/hours_per_step.values[0]) if self.backward_timeshift is not None else None

        self.model = DSMSinkModel(model, self)
        return self.model
    
    def transform_data(self, flow_system: 'FlowSystem') -> None:
        super().transform_data(flow_system)
        self.initial_demand = flow_system.create_time_series(
            f'{self.label_full}|initial_demand',
            self.initial_demand,
        )
        self.maximum_flow_surplus = flow_system.create_time_series(
            f'{self.label_full}|maximum_flow_surplus',
            self.maximum_flow_surplus,
        )
        self.maximum_flow_deficit = flow_system.create_time_series(
            f'{self.label_full}|maximum_flow_deficit',
            self.maximum_flow_deficit,
        )
        self.maximum_cumulated_deficit = flow_system.create_time_series(
            f'{self.label_full}|maximum_cumulated_deficit',
            self.maximum_cumulated_deficit,
            needs_extra_timestep=True,
        )
        self.maximum_cumulated_surplus = flow_system.create_time_series(
            f'{self.label_full}|maximum_cumulated_surplus',
            self.maximum_cumulated_surplus,
            needs_extra_timestep=True,
        )
        self.relative_loss_per_hour_negative_charge_state = flow_system.create_time_series(
            f'{self.label_full}|relative_loss_per_hour_negative_charge_state',
            self.relative_loss_per_hour_negative_charge_state,
        )
        self.relative_loss_per_hour_positive_charge_state = flow_system.create_time_series(
            f'{self.label_full}|relative_loss_per_hour_positive_charge_state',
            self.relative_loss_per_hour_positive_charge_state,
        )
        self.penalty_costs_negative_charge_states = flow_system.create_time_series(
            f'{self.label_full}|penalty_costs_negative_charge_states',
            self.penalty_costs_negative_charge_states,
        )
        self.penalty_costs_positive_charge_states = flow_system.create_time_series(
            f'{self.label_full}|penalty_costs_positive_charge_states',
            self.penalty_costs_positive_charge_states,
        )

    def _plausibility_checks(self, model: SystemModel):
        """
        Check for infeasible or uncommon combinations of parameters
        """
        super()._plausibility_checks()
        
        hours_per_step = model.hours_per_step

        if self.forward_timeshift != None or self.backward_timeshift != None:
            if any(hours_per_step.values[0]!=hours_per_step.values):
                raise ValueError(
                    f'{self.label_full}:'
                    f'limits to forward and backward timeshifts can only be used for timesteps of equal length'
                )

        if self.forward_timeshift is not None:
            if self.forward_timeshift%hours_per_step.values[0]!=0:
                raise ValueError(
                    f'{self.label_full}: {self.forward_timeshift=} '
                    f'must be a multiple of the timestep length.'
                )
        
        if self.backward_timeshift is not None:
            if self.backward_timeshift%hours_per_step.values[0]!=0:
                raise ValueError(
                    f'{self.label_full}: {self.backward_timeshift=} '
                    f'must be a multiple of the timestep length.'
                )
            
    #TODO: think about other implausibilities
    #INFO: investments not implemented
    
class DSMSinkModel(ComponentModel):
    """Model of DSM Sink"""
    
    def __init__(self, model: SystemModel, element: DSMSink):
        super().__init__(model, element)
        self.element: DSMSink = element
        self.positive_charge_state: Optional[linopy.Variable] = None
        self.negative_charge_state: Optional[linopy.Variable] = None
        self.positive_charge_rate: Optional[linopy.Variable] = None
        self.negative_charge_rate: Optional[linopy.Variable] = None

        self.is_positive_charge_state: Optional[linopy.Variable] = None
        self.is_negative_charge_state: Optional[linopy.Variable] = None
        
        # Add state models for charge rates
        self.is_positive_charge_rate: Optional[StateModel] = None
        self.is_negative_charge_rate: Optional[StateModel] = None
        self.prevent_simultaneous_charge_rates: Optional[PreventSimultaneousUsageModel] = None
    
    def do_modeling(self):
        super().do_modeling()

        # Add variables for positive and negative charge rates
        lb, ub = self.charge_rate_bounds
        self.positive_charge_rate = self.add(
            self._model.add_variables(
                lower=lb, upper=ub, coords=self._model.coords, name=f'{self.label_full}|positive_charge_rate'
            ),
            'positive_charge_rate',
        )

        lb, ub = self.discharge_rate_bounds
        self.negative_charge_rate = self.add(
            self._model.add_variables(
                lower=lb, upper=ub, coords=self._model.coords, name=f'{self.label_full}|negative_charge_rate'
            ),
            'negative_charge_rate',
        )
        
        # Add variables for negative charge states
        lb, ub = self.negative_charge_state_bounds
        self.negative_charge_state = self.add(
            self._model.add_variables(
                lower=lb, upper=ub, coords=self._model.coords_extra, name=f'{self.label_full}|negative_charge_state'
            ),
            'negative_charge_state',
        )

        # Add variables for positive charge states
        lb, ub = self.positive_charge_state_bounds
        self.positive_charge_state = self.add(
            self._model.add_variables(
                lower=lb, upper=ub, coords=self._model.coords_extra, name=f'{self.label_full}|positive_charge_state'
            ),
            'positive_charge_state'
        )

        positive_charge_state = self.positive_charge_state
        negative_charge_state = self.negative_charge_state
        etapos = 1 - self.element.relative_loss_per_hour_positive_charge_state.active_data
        etaneg = 1 - self.element.relative_loss_per_hour_negative_charge_state.active_data
        hours_per_step = self._model.hours_per_step
        positive_charge_rate = self.positive_charge_rate
        negative_charge_rate = self.negative_charge_rate
        initial_demand = self.element.initial_demand.active_data
        sink = self.element.sink.model.flow_rate

        # eq: positive_charge_state(t) + negative_charge_state(t) = positive_charge_state(t-1) * etapos + negative_charge_state(t-1) * etaneg + positive_charge_rate(t) + negative_charge_rate(t)
        self.add(
            self._model.add_constraints(
                positive_charge_state.isel(time=slice(1, None))
                + negative_charge_state.isel(time=slice(1,None))
                == positive_charge_state.isel(time=slice(None, -1)) * (etapos ** hours_per_step)
                + negative_charge_state.isel(time=slice(None,-1)) * (etaneg ** hours_per_step)
                + positive_charge_rate * hours_per_step
                + negative_charge_rate * hours_per_step,
                name=f'{self.label_full}|charge_state',
            ),
            'charge_state',
        )

        # eq: sink(t) = initial_demand(t) + positive_charge_rate(t) + negative_charge_rate(t)
        self.add(
            self._model.add_constraints(
                sink
                == positive_charge_rate
                + negative_charge_rate
                + initial_demand,
                name=f'{self.label_full}|resulting_load_profile',
            ),
            'resulting_load_profile',
        )

        # Add constraints for preventing mixed charge states if not allowed
        if not self.element.allow_mixed_charge_states:
            self._add_charge_state_exclusivity_constraints()

        # Add charge rate exclusivity constraints
        if not self.element.allow_parallel_charge_and_discharge:
            self._add_charge_rate_exclusivity_constraints()

        # Forward and backward timeshift constraints
        self._add_timeshift_limits()

        # Initial and final charge state constraints
        self._initial_and_final_charge_state()

        # Add penalty costs as effects for positive and negative charge states
        penalty_costs_pos = self.element.penalty_costs_positive_charge_states.active_data
        penalty_costs_neg = self.element.penalty_costs_negative_charge_states.active_data

        # Add effects for positive charge states
        if np.any(penalty_costs_pos != 0):
            # Multiply penalty costs with hours_per_step first to get a single coefficient per timestep
            penalty_coeff_pos = penalty_costs_pos * hours_per_step
            self._model.effects.add_share_to_penalty(
                name = self.label_full,
                # charge state is shifted backwards in time to apply penalty costs to the charge state at the end of a timestep
                expression = (positive_charge_state.shift(time=-1).isel(time=slice(None,-1)) * penalty_coeff_pos).sum()
            )

        # Add effects for negative charge states
        if np.any(penalty_costs_neg != 0):
            # Multiply penalty costs with hours_per_step first to get a single coefficient per timestep
            penalty_coeff_neg = - penalty_costs_neg * hours_per_step
            self._model.effects.add_share_to_penalty(
                name = self.label_full,
                # charge state is shifted backwards in time to apply penalty costs to the charge state at the end of a timestep
                expression = (negative_charge_state.shift(time=-1).isel(time=slice(None,-1)) * penalty_coeff_neg).sum()
            )

    def _add_charge_state_exclusivity_constraints(self):
        """Add constraints to prevent simultaneous positive and negative charge states"""
        #TODO: this method currently does not use the implemented statemodel and preventsimultaneous model
        #because they do not work with variables using coords_extra
        
        # Add binary variables to track charge state type
        self.is_positive_charge_state = self.add(
            self._model.add_variables(
                binary=True,
                coords=self._model.coords_extra,
                name=f'{self.label_full}|is_positive_charge_state'
            ),
            'is_positive_charge_state'
        )

        self.is_negative_charge_state = self.add(
            self._model.add_variables(
                binary=True,
                coords=self._model.coords_extra,
                name=f'{self.label_full}|is_negative_charge_state'
            ),
            'is_negative_charge_state'
        )

        positive_charge_state = self.positive_charge_state
        negative_charge_state = self.negative_charge_state

        # If positive_charge_state > 0, then is_positive_charge_state must be 1
        self.add(
            self._model.add_constraints(
                positive_charge_state <= self.positive_charge_state_bounds[1] * self.is_positive_charge_state,
                name=f'{self.label_full}|positive_charge_state_binary_upper'
            ),
            'positive_charge_state_binary_upper'
        )

        # If is_positive_charge_state is 1, then positive_charge_state must be > 0
        self.add(
            self._model.add_constraints(
                positive_charge_state >= CONFIG.modeling.EPSILON * self.positive_charge_state_bounds[1] * self.is_positive_charge_state,  # Small epsilon to avoid numerical issues
                name=f'{self.label_full}|positive_charge_state_binary_lower'
            ),
            'positive_charge_state_binary_lower'
        )

        # If negative_charge_state < 0, then is_negative_charge_state must be 1
        self.add(
            self._model.add_constraints(
                negative_charge_state >= self.negative_charge_state_bounds[0] * self.is_negative_charge_state,
                name=f'{self.label_full}|negative_charge_state_binary_upper'
            ),
            'negative_charge_state_binary_upper'
        )

        # If is_negative_charge_state is 1, then negative_charge_state must be < 0
        self.add(
            self._model.add_constraints(
                negative_charge_state <= CONFIG.modeling.EPSILON * self.negative_charge_state_bounds[0] * self.is_negative_charge_state,  # Small epsilon to avoid numerical issues
                name=f'{self.label_full}|negative_charge_state_binary_lower'
            ),
            'negative_charge_state_binary_lower'
        )

        # Ensure only one type of charge state can be active at a time
        self.add(
            self._model.add_constraints(
                self.is_positive_charge_state + self.is_negative_charge_state <= 1,
                name=f'{self.label_full}|mutually_exclusive_charge_states'
            ),
            'mutually_exclusive_charge_states'
        )

    def _add_charge_rate_exclusivity_constraints(self):
        """Add constraints to prevent simultaneous positive and negative charge rates using StateModel and PreventSimultaneousUsageModel"""
        # Create a time series of zeros to be used for both bounds
        timeseries_zeros = np.zeros_like(self._model.coords[0], dtype=float)
        
        # Create StateModel for positive charge rate
        self.is_positive_charge_rate = self.add(
            StateModel(
                model=self._model,
                label_of_element=f'{self.label_full}|positive_charge_rate',
                defining_variables=[self.positive_charge_rate],
                defining_bounds=[(timeseries_zeros, self.charge_rate_bounds[1])],
                use_off=False
            )
        )
        self.is_positive_charge_rate.do_modeling()

        # Create StateModel for negative charge rate
        self.is_negative_charge_rate = self.add(
            StateModel(
                model=self._model,
                label_of_element=f'{self.label_full}|negative_charge_rate',
                defining_variables=[-self.negative_charge_rate],  # StateModel can only handle positive variables
                defining_bounds=[(timeseries_zeros, -self.discharge_rate_bounds[0])],
                use_off=False
            )
        )
        self.is_negative_charge_rate.do_modeling()
        
        # Create PreventSimultaneousUsageModel for charge rates
        self.prevent_simultaneous_charge_rates = self.add(
            PreventSimultaneousUsageModel(
                model=self._model,
                variables=[self.is_positive_charge_rate.on, self.is_negative_charge_rate.on],
                label_of_element=self.label_full,
                label='PreventSimultaneousChargeRates'
            )
        )
        self.prevent_simultaneous_charge_rates.do_modeling()

    def _add_timeshift_limits(self):
        hours_per_step = self._model.hours_per_step
        timesteps_forward = self.element.timesteps_forward
        timesteps_backward = self.element.timesteps_backward

        etapos = 1 - self.element.relative_loss_per_hour_positive_charge_state.active_data
        etaneg = 1 - self.element.relative_loss_per_hour_negative_charge_state.active_data

        positive_charge_state = self.positive_charge_state
        negative_charge_state = self.negative_charge_state

        # Add constraints limiting the forward timeshift
        if timesteps_forward is not None:
            surplus_sum = 0
            for i in range(0, timesteps_forward):
                surplus_sum += self.positive_charge_rate.shift(time=i).isel(time=slice(i,None)) * hours_per_step * (etapos ** (hours_per_step * i))
            # eq: positive_charge_state(t) <= sum over n (positive_charge_rate(t-n) * hours_per_step * (etapos ^ (hours_per_step * i))
            # where n ranges from 0 to timesteps_forward
            # The positive charge state can't be any higher than the charge that was added during the last x timesteps minus the losses
            # x is defined by the timesteps_forward
            # This forces the virtual storage to discharge after the maximum timesteps that the demand is allowed to be shifted forward
            self.add(
                self._model.add_constraints(
                    positive_charge_state.isel(time = slice(1,None))
                    <= surplus_sum,
                    name=f'{self.label_full}|limit_forward_timeshift'
                ),
                f'limit_forward_timeshift'
            )

        # Add constraints limiting the backwards timeshift
        if timesteps_backward is not None:
            deficit_sum = 0
            for i in range(0, timesteps_backward):
                deficit_sum += self.negative_charge_rate.shift(time=i).isel(time=slice(i,None)) * hours_per_step * (etaneg ** (hours_per_step * i))
            # eq: -negative_charge_state(t) <= -sum over n (negative_charge_rate(t-n) * hours_per_step * (etaneg ^ (hours_per_step * i))
            # where n ranges from 0 to timesteps_backward
            # The negative charge state can't be any higher than the deficit that was accumulated during the last x timesteps minus the losses.
            # x is defined by the timesteps_backward.
            # This forces the virtual storage to "recharge" (i. e. "discharge" the negative charge state) after the maximum timesteps that the demand is allowed to be shifted backward.
            self.add(
                self._model.add_constraints(
                    - negative_charge_state.isel(time=slice(1,None))
                    <= - deficit_sum,
                    name=f'{self.label_full}|limit_backward_timeshift'
                ),
                f'limit_backward_timeshift'
            )

    def _initial_and_final_charge_state(self):
        """Add constraints for initial and final charge states to be zero"""
        # Set initial charge state to zero
        self.add(
            self._model.add_constraints(
                self.positive_charge_state.isel(time=0) + self.negative_charge_state.isel(time=0) == 0,
                name=f'{self.label_full}|initial_charge_state'
            ),
            'initial_charge_state'
        )

        # Set final charge state to zero
        self.add(
            self._model.add_constraints(
                self.positive_charge_state.isel(time=-1) + self.negative_charge_state.isel(time=-1) == 0,
                name=f'{self.label_full}|final_charge_state'
            ),
            'final_charge_state'
        )

    @property
    def positive_charge_state_bounds(self) -> Tuple[NumericData, NumericData]:
        return (
            0,
            self.element.maximum_cumulated_surplus.active_data,
        )
    
    @property
    def negative_charge_state_bounds(self) -> Tuple[NumericData, NumericData]:
        return (
            self.element.maximum_cumulated_deficit.active_data,
            0,
        )
        
    @property
    def charge_rate_bounds(self) -> Tuple[NumericData, NumericData]:
        return(
            0,
            self.element.maximum_flow_surplus.active_data,
        )
    
    @property
    def discharge_rate_bounds(self) -> Tuple[NumericData, NumericData]:
        return(
            self.element.maximum_flow_deficit.active_data,
            0,
        )