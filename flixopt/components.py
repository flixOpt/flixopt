"""
This module contains the basic components of the flixOpt framework.
"""

import logging
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np

from . import utils
from .core import Numeric, Numeric_TS, Skalar, TimeSeries
from .elements import Component, ComponentModel, Flow, _create_time_series
from .features import InvestmentModel, MultipleSegmentsModel, OnOffModel
from .interface import InvestParameters, OnOffParameters
from .math_modeling import Equation, VariableTS
from .structure import SystemModel, create_equation, create_variable

logger = logging.getLogger('flixOpt')


class LinearConverter(Component):
    """
    Converts one FLow into another via linear conversion factors
    """

    def __init__(
        self,
        label: str,
        inputs: List[Flow],
        outputs: List[Flow],
        on_off_parameters: OnOffParameters = None,
        conversion_factors: List[Dict[Flow, Numeric_TS]] = None,
        segmented_conversion_factors: Dict[Flow, List[Tuple[Numeric_TS, Numeric_TS]]] = None,
        meta_data: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        label : str
            name.
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        inputs : input flows.
        outputs : output flows.
        on_off_parameters: Information about on and off states. See class OnOffParameters.
        conversion_factors : linear relation between flows.
            Either 'conversion_factors' or 'segmented_conversion_factors' can be used!
            example heat pump:
        segmented_conversion_factors :  Segmented linear relation between flows.
            Each Flow gets a List of Segments assigned.
            If FLows need to be 0 (or Off), include a "Zero-Segment" "(0, 0)", or use on_off_parameters
            Either 'segmented_conversion_factors' or 'conversion_factors' can be used!
            --> "gaps" can be expressed by a segment not starting at the end of the prior segment : [(1,3), (4,5)]
            --> "points" can expressed as segment with same begin and end : [(3,3), (4,4)]

        """
        super().__init__(label, inputs, outputs, on_off_parameters, meta_data=meta_data)
        self.conversion_factors = conversion_factors or []
        self.segmented_conversion_factors = segmented_conversion_factors or {}
        self._plausibility_checks()

    def create_model(self) -> 'LinearConverterModel':
        self.model = LinearConverterModel(self)
        return self.model

    def _plausibility_checks(self) -> None:
        if not self.conversion_factors and not self.segmented_conversion_factors:
            raise Exception('Either conversion_factors or segmented_conversion_factors must be defined!')
        if self.conversion_factors and self.segmented_conversion_factors:
            raise Exception('Only one of conversion_factors or segmented_conversion_factors can be defined, not both!')

        if self.conversion_factors:
            if self.degrees_of_freedom <= 0:
                raise Exception(
                    f'Too Many conversion_factors_specified. Care that you use less conversion_factors '
                    f'then inputs + outputs!! With {len(self.inputs + self.outputs)} inputs and outputs, '
                    f'use not more than {len(self.inputs + self.outputs) - 1} conversion_factors!'
                )

            for conversion_factor in self.conversion_factors:
                for flow in conversion_factor:
                    if flow not in (self.inputs + self.outputs):
                        raise Exception(
                            f'{self.label}: Flow {flow.label} in conversion_factors is not in inputs/outputs'
                        )
        if self.segmented_conversion_factors:
            for flow in self.inputs + self.outputs:
                if isinstance(flow.size, InvestParameters) and flow.size.fixed_size is None:
                    raise Exception(
                        f'segmented_conversion_factors (in {self.label_full}) and variable size '
                        f'(in flow {flow.label_full}) do not make sense together!'
                    )

    def transform_data(self):
        super().transform_data()
        if self.conversion_factors:
            self.conversion_factors = self._transform_conversion_factors()
        else:
            segmented_conversion_factors = {}
            for flow, segments in self.segmented_conversion_factors.items():
                segmented_conversion_factors[flow] = [
                    (
                        _create_time_series('Stuetzstelle', segment[0], self),
                        _create_time_series('Stuetzstelle', segment[1], self),
                    )
                    for segment in segments
                ]
            self.segmented_conversion_factors = segmented_conversion_factors

    def _transform_conversion_factors(self) -> List[Dict[Flow, TimeSeries]]:
        """macht alle Faktoren, die nicht TimeSeries sind, zu TimeSeries"""
        list_of_conversion_factors = []
        for conversion_factor in self.conversion_factors:
            transformed_dict = {}
            for flow, values in conversion_factor.items():
                transformed_dict[flow] = _create_time_series(f'{flow.label}_factor', values, self)
            list_of_conversion_factors.append(transformed_dict)
        return list_of_conversion_factors

    @property
    def degrees_of_freedom(self):
        return len(self.inputs + self.outputs) - len(self.conversion_factors)


class Storage(Component):
    """
    Klasse Storage
    """

    # TODO: Dabei fällt mir auf. Vielleicht sollte man mal überlegen, ob man für Ladeleistungen bereits in dem
    #  jeweiligen Zeitschritt mit einem Verlust berücksichtigt. Zumindest für große Zeitschritte bzw. große Verluste
    #  eventuell relevant.
    #  -> Sprich: speicherverlust = charge_state(t) * relative_loss_per_hour * dt + 0.5 * Q_lade(t) * dt * relative_loss_per_hour * dt
    #  -> müsste man aber auch für den sich ändernden Ladezustand berücksichtigten

    def __init__(
        self,
        label: str,
        charging: Flow,
        discharging: Flow,
        capacity_in_flow_hours: Union[Skalar, InvestParameters],
        relative_minimum_charge_state: Numeric = 0,
        relative_maximum_charge_state: Numeric = 1,
        initial_charge_state: Optional[Union[Skalar, Literal['lastValueOfSim']]] = 0,
        minimal_final_charge_state: Optional[Skalar] = None,
        maximal_final_charge_state: Optional[Skalar] = None,
        eta_charge: Numeric = 1,
        eta_discharge: Numeric = 1,
        relative_loss_per_hour: Numeric = 0,
        prevent_simultaneous_charge_and_discharge: bool = True,
        meta_data: Optional[Dict] = None,
    ):
        """
        constructor of storage

        Parameters
        ----------
        label : str
            description.
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        charging : Flow
            ingoing flow.
        discharging : Flow
            outgoing flow.
        capacity_in_flow_hours : Skalar or InvestParameter
            nominal capacity of the storage
        relative_minimum_charge_state : float or TS, optional
            minimum relative charge state. The default is 0.
        relative_maximum_charge_state : float or TS, optional
            maximum relative charge state. The default is 1.
        initial_charge_state : None, float (0...1), 'lastValueOfSim',  optional
            storage charge_state at the beginning. The default is 0.
            float: defined charge_state at start of first timestep
            None: free to choose by optimizer
            'lastValueOfSim': chargeState0 is equal to chargestate of last timestep ("closed simulation")
        minimal_final_charge_state : float or None, optional
            minimal value of chargeState at the end of timeseries.
        maximal_final_charge_state : float or None, optional
            maximal value of chargeState at the end of timeseries.
        eta_charge : float, optional
            efficiency factor of charging/loading. The default is 1.
        eta_discharge : TYPE, optional
            efficiency factor of uncharging/unloading. The default is 1.
        relative_loss_per_hour : float or TS. optional
            loss per chargeState-Unit per hour. The default is 0.
        prevent_simultaneous_charge_and_discharge : boolean, optional
            should simultaneously Loading and Unloading be avoided? (Attention, Performance maybe becomes worse with avoidInAndOutAtOnce=True). The default is True.
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
        self.relative_minimum_charge_state: Numeric_TS = relative_minimum_charge_state
        self.relative_maximum_charge_state: Numeric_TS = relative_maximum_charge_state

        self.initial_charge_state = initial_charge_state
        self.minimal_final_charge_state = minimal_final_charge_state
        self.maximal_final_charge_state = maximal_final_charge_state

        self.eta_charge: Numeric_TS = eta_charge
        self.eta_discharge: Numeric_TS = eta_discharge
        self.relative_loss_per_hour: Numeric_TS = relative_loss_per_hour

    def create_model(self) -> 'StorageModel':
        self.model = StorageModel(self)
        return self.model

    def transform_data(self) -> None:
        super().transform_data()
        self.relative_minimum_charge_state = _create_time_series(
            'relative_minimum_charge_state', self.relative_minimum_charge_state, self
        )
        self.relative_maximum_charge_state = _create_time_series(
            'relative_maximum_charge_state', self.relative_maximum_charge_state, self
        )
        self.eta_charge = _create_time_series('eta_charge', self.eta_charge, self)
        self.eta_discharge = _create_time_series('eta_discharge', self.eta_discharge, self)
        self.relative_loss_per_hour = _create_time_series('relative_loss_per_hour', self.relative_loss_per_hour, self)
        if isinstance(self.capacity_in_flow_hours, InvestParameters):
            self.capacity_in_flow_hours.transform_data()


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
        relative_losses: Optional[Numeric_TS] = None,
        absolute_losses: Optional[Numeric_TS] = None,
        on_off_parameters: OnOffParameters = None,
        prevent_simultaneous_flows_in_both_directions: bool = True,
    ):
        """
        Initializes a Transmission component (Pipe, cable, ...) that models the flows between two sides
        with potential losses.

        Parameters
        ----------
        label : str
            The name of the transmission component.
        in1 : Flow
            The inflow at side A. Pass InvestmentParameters here.
        out1 : Flow
            The outflow at side B.
        in2 : Optional[Flow], optional
            The optional inflow at side B.
            If in1 got Investmentparameters, the size of this Flow will be equal to in1 (with no extra effects!)
        out2 : Optional[Flow], optional
            The optional outflow at side A.
        relative_losses : Optional[Numeric_TS], optional
            The relative loss between inflow and outflow, e.g., 0.02 for 2% loss.
        absolute_losses : Optional[Numeric_TS], optional
            The absolute loss, occur only when the Flow is on. Induces the creation of the ON-Variable
        on_off_parameters : OnOffParameters, optional
            Parameters defining the on/off behavior of the component.
        prevent_simultaneous_flows_in_both_directions : bool, default=True
            If True, prevents simultaneous flows in both directions.
        """
        super().__init__(
            label,
            inputs=[flow for flow in (in1, in2) if flow is not None],
            outputs=[flow for flow in (out1, out2) if flow is not None],
            on_off_parameters=on_off_parameters,
            prevent_simultaneous_flows=None
            if in2 is None or prevent_simultaneous_flows_in_both_directions is False
            else [in1, in2],
        )
        self.in1 = in1
        self.out1 = out1
        self.in2 = in2
        self.out2 = out2

        self.relative_losses = relative_losses
        self.absolute_losses = absolute_losses

    def _plausibility_checks(self):
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

    def create_model(self) -> 'TransmissionModel':
        self.model = TransmissionModel(self)
        return self.model

    def transform_data(self) -> None:
        super().transform_data()
        self.relative_losses = _create_time_series('relative_losses', self.relative_losses, self)
        self.absolute_losses = _create_time_series('absolute_losses', self.absolute_losses, self)


class TransmissionModel(ComponentModel):
    def __init__(self, element: Transmission):
        super().__init__(element)
        self.element: Transmission = element
        self._on: Optional[OnOffModel] = None

    def do_modeling(self, system_model: SystemModel):
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

        super().do_modeling(system_model)

        # first direction
        self.create_transmission_equation('direction_1', self.element.in1, self.element.out1)

        # second direction:
        if self.element.in2 is not None:
            self.create_transmission_equation('direction_2', self.element.in2, self.element.out2)

        # equate size of both directions
        if isinstance(self.element.in1.size, InvestParameters) and self.element.in2 is not None:
            # eq: in1.size = in2.size
            eq_equal_size = create_equation('equal_size_in_both_directions', self, 'eq')
            eq_equal_size.add_summand(self.element.in1.model._investment.size, 1)
            eq_equal_size.add_summand(self.element.in2.model._investment.size, -1)

    def create_transmission_equation(self, name: str, in_flow: Flow, out_flow: Flow) -> Equation:
        """Creates an Equation for the Transmission efficiency and adds it to the model"""
        # first direction
        # eq: in(t)*(1-loss_rel(t)) = out(t) + on(t)*loss_abs(t)
        eq_transmission = create_equation(name, self, 'eq')
        efficiency = 1 if self.element.relative_losses is None else (1 - self.element.relative_losses.active_data)
        eq_transmission.add_summand(in_flow.model.flow_rate, efficiency)
        eq_transmission.add_summand(out_flow.model.flow_rate, -1)
        if self.element.absolute_losses is not None:
            eq_transmission.add_summand(in_flow.model._on.on, -1 * self.element.absolute_losses.active_data)
        return eq_transmission


class LinearConverterModel(ComponentModel):
    def __init__(self, element: LinearConverter):
        super().__init__(element)
        self.element: LinearConverter = element
        self._on: Optional[OnOffModel] = None

    def do_modeling(self, system_model: SystemModel):
        super().do_modeling(system_model)

        # conversion_factors:
        if self.element.conversion_factors:
            all_input_flows = set(self.element.inputs)
            all_output_flows = set(self.element.outputs)

            # für alle linearen Gleichungen:
            for i, conversion_factor in enumerate(self.element.conversion_factors):
                # erstelle Gleichung für jedes t:
                # sum(inputs * factor) = sum(outputs * factor)
                # left = in1.flow_rate[t] * factor_in1[t] + in2.flow_rate[t] * factor_in2[t] + ...
                # right = out1.flow_rate[t] * factor_out1[t] + out2.flow_rate[t] * factor_out2[t] + ...
                # eq: left = right
                used_flows = set(conversion_factor.keys())
                used_inputs: Set = all_input_flows & used_flows
                used_outputs: Set = all_output_flows & used_flows

                eq_conversion = create_equation(f'conversion_{i}', self)
                for flow in used_inputs:
                    factor = conversion_factor[flow].active_data
                    eq_conversion.add_summand(flow.model.flow_rate, factor)  # flow1.flow_rate[t]      * factor[t]
                for flow in used_outputs:
                    factor = conversion_factor[flow].active_data
                    eq_conversion.add_summand(flow.model.flow_rate, -1 * factor)  # output.val[t] * -1 * factor[t]

                eq_conversion.add_constant(0)  # TODO: Is this necessary?

        # (linear) segments:
        else:
            # TODO: Improve Inclusion of OnOffParameters. Instead of creating a Binary in every flow, the binary could only be part of the Segment itself
            segments = {
                flow.model.flow_rate: [
                    (ts1.active_data, ts2.active_data) for ts1, ts2 in self.element.segmented_conversion_factors[flow]
                ]
                for flow in self.element.inputs + self.element.outputs
            }
            linear_segments = MultipleSegmentsModel(
                self.element, segments, self._on.on if self._on is not None else None
            )  # TODO: Add Outside_segments Variable (On)
            linear_segments.do_modeling(system_model)
            self.sub_models.append(linear_segments)


class StorageModel(ComponentModel):
    """Model of Storage"""

    # TODO: Add additional Timestep!!!
    def __init__(self, element: Storage):
        super().__init__(element)
        self.element: Storage = element
        self.charge_state: Optional[VariableTS] = None
        self.netto_discharge: Optional[VariableTS] = None
        self._investment: Optional[InvestmentModel] = None

    def do_modeling(self, system_model):
        super().do_modeling(system_model)

        lb, ub = self.absolute_charge_state_bounds
        self.charge_state = create_variable(
            'charge_state', self, system_model.nr_of_time_steps + 1, lower_bound=lb, upper_bound=ub
        )

        self.netto_discharge = create_variable(
            'netto_discharge', self, system_model.nr_of_time_steps, lower_bound=-np.inf
        )  # negative Werte zulässig!

        # netto_discharge:
        # eq: nettoFlow(t) - discharging(t) + charging(t) = 0
        eq_netto = create_equation('netto_discharge', self, eq_type='eq')
        eq_netto.add_summand(self.netto_discharge, 1)
        eq_netto.add_summand(self.element.charging.model.flow_rate, 1)
        eq_netto.add_summand(self.element.discharging.model.flow_rate, -1)

        indices_charge_state = range(system_model.indices.start, system_model.indices.stop + 1)  # additional

        ############# Charge State Equation
        # charge_state(n+1)
        # + charge_state(n) * [relative_loss_per_hour * dt(n) - 1]
        # - charging(n)     * eta_charge * dt(n)
        # + discharging(n)  * 1 / eta_discharge * dt(n)
        # = 0
        eq_charge_state = create_equation('charge_state', self, eq_type='eq')
        eq_charge_state.add_summand(self.charge_state, 1, indices_charge_state[1:])  # 1:end
        eq_charge_state.add_summand(
            self.charge_state,
            (self.element.relative_loss_per_hour.active_data * system_model.dt_in_hours) - 1,
            indices_charge_state[:-1],
        )  # sprich 0 .. end-1 % nach letztem Zeitschritt gibt es noch einen weiteren Ladezustand!
        eq_charge_state.add_summand(
            self.element.charging.model.flow_rate, -1 * self.element.eta_charge.active_data * system_model.dt_in_hours
        )
        eq_charge_state.add_summand(
            self.element.discharging.model.flow_rate,
            1 / self.element.eta_discharge.active_data * system_model.dt_in_hours,
        )

        if isinstance(self.element.capacity_in_flow_hours, InvestParameters):
            self._investment = InvestmentModel(
                self.element, self.element.capacity_in_flow_hours, self.charge_state, self.relative_charge_state_bounds
            )
            self.sub_models.append(self._investment)
            self._investment.do_modeling(system_model)

        # Initial charge state
        if self.element.initial_charge_state is not None:
            self._model_initial_and_final_charge_state(system_model)

    def _model_initial_and_final_charge_state(self, system_model):
        indices_charge_state = range(system_model.indices.start, system_model.indices.stop + 1)  # additional

        if self.element.initial_charge_state is not None:
            eq_initial = create_equation('initial_charge_state', self, eq_type='eq')
            if utils.is_number(self.element.initial_charge_state):
                # eq: Q_Ladezustand(1) = Q_Ladezustand_Start;
                eq_initial.add_constant(self.element.initial_charge_state)  # chargeState_0 !
                eq_initial.add_summand(self.charge_state, 1, system_model.indices[0])
            elif self.element.initial_charge_state == 'lastValueOfSim':
                # eq: Q_Ladezustand(1) - Q_Ladezustand(end) = 0;
                eq_initial.add_summand(self.charge_state, 1, system_model.indices[0])
                eq_initial.add_summand(self.charge_state, -1, system_model.indices[-1])
            else:
                raise Exception(f'initial_charge_state has undefined value: {self.element.initial_charge_state}')
                # TODO: Validation in Storage Class, not in Model

        ####################################
        # Final Charge State
        # 1: eq:  Q_charge_state(end) <= Q_max
        if self.element.maximal_final_charge_state is not None:
            eq_max = create_equation('eq_final_charge_state_max', self, eq_type='ineq')
            eq_max.add_summand(self.charge_state, 1, indices_charge_state[-1])
            eq_max.add_constant(self.element.maximal_final_charge_state)

        # 2: eq: - Q_charge_state(end) <= - Q_min
        if self.element.minimal_final_charge_state is not None:
            eq_min = create_equation('eq_charge_state_end_min', self, eq_type='ineq')
            eq_min.add_summand(self.charge_state, -1, indices_charge_state[-1])
            eq_min.add_constant(-self.element.minimal_final_charge_state)

    @property
    def absolute_charge_state_bounds(self) -> Tuple[Numeric, Numeric]:
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
    def relative_charge_state_bounds(self) -> Tuple[Numeric, Numeric]:
        return (
            self.element.relative_minimum_charge_state.active_data,
            self.element.relative_maximum_charge_state.active_data,
        )


class SourceAndSink(Component):
    """
    class for source (output-flow) and sink (input-flow) in one commponent
    """

    # source : Flow
    # sink   : Flow

    def __init__(
        self,
        label: str,
        source: Flow,
        sink: Flow,
        prevent_simultaneous_flows: bool = True,
        meta_data: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        label : str
            name of sourceAndSink
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        source : Flow
            output-flow of this component
        sink : Flow
            input-flow of this component
        prevent_simultaneous_flows: boolean. Default ist True.
            True: inflow and outflow are not allowed to be both non-zero at same timestep.
            False: inflow and outflow are working independently.

        """
        super().__init__(
            label,
            inputs=[sink],
            outputs=[source],
            prevent_simultaneous_flows=[sink, source] if prevent_simultaneous_flows is True else None,
            meta_data=meta_data,
        )
        self.source = source
        self.sink = sink


class Source(Component):
    def __init__(self, label: str, source: Flow, meta_data: Optional[Dict] = None):
        """
        Parameters
        ----------
        label : str
            name of source
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        source : Flow
            output-flow of source
        """
        super().__init__(label, outputs=[source], meta_data=meta_data)
        self.source = source


class Sink(Component):
    def __init__(self, label: str, sink: Flow, meta_data: Optional[Dict] = None):
        """
        constructor of sink

        Parameters
        ----------
        label : str
            name of sink.
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        sink : Flow
            input-flow of sink
        """
        super().__init__(label, inputs=[sink], meta_data=meta_data)
        self.sink = sink
