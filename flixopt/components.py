"""
This module contains the basic components of the flixopt framework.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Set, Tuple, Union

import linopy
import numpy as np
import xarray as xr

from . import utils
from .core import NonTemporalDataUser, PlausibilityError, Scalar, TemporalData, TemporalDataUser
from .elements import Component, ComponentModel, Flow
from .features import InvestmentModel, OnOffModel, PiecewiseModel
from .interface import InvestParameters, OnOffParameters, PiecewiseConversion
from .modeling import BoundingPatterns
from .structure import FlowSystemModel, register_class_for_io

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
        conversion_factors: List[Dict[str, TemporalDataUser]] = None,
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

    def create_model(self, model: FlowSystemModel) -> 'LinearConverterModel':
        self._plausibility_checks()
        self.submodel = LinearConverterModel(model, self)
        return self.submodel

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
                        f'Using a FLow with a fixed size ({flow.label_full}) AND a piecewise_conversion '
                        f'(in {self.label_full}) and variable size is uncommon. Please check if this is intended!'
                    )

    def transform_data(self, flow_system: 'FlowSystem'):
        super().transform_data(flow_system)
        if self.conversion_factors:
            self.conversion_factors = self._transform_conversion_factors(flow_system)
        if self.piecewise_conversion:
            self.piecewise_conversion.has_time_dim = True
            self.piecewise_conversion.transform_data(flow_system, f'{self.label_full}|PiecewiseConversion')

    def _transform_conversion_factors(self, flow_system: 'FlowSystem') -> List[Dict[str, xr.DataArray]]:
        """Converts all conversion factors to internal datatypes"""
        list_of_conversion_factors = []
        for idx, conversion_factor in enumerate(self.conversion_factors):
            transformed_dict = {}
            for flow, values in conversion_factor.items():
                # TODO: Might be better to use the label of the component instead of the flow
                transformed_dict[flow] = flow_system.fit_to_model_coords(
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
        capacity_in_flow_hours: Union[NonTemporalDataUser, InvestParameters],
        relative_minimum_charge_state: TemporalDataUser = 0,
        relative_maximum_charge_state: TemporalDataUser = 1,
        initial_charge_state: Union[NonTemporalDataUser, Literal['lastValueOfSim']] = 0,
        minimal_final_charge_state: Optional[NonTemporalDataUser] = None,
        maximal_final_charge_state: Optional[NonTemporalDataUser] = None,
        relative_minimum_final_charge_state: Optional[NonTemporalDataUser] = None,
        relative_maximum_final_charge_state: Optional[NonTemporalDataUser] = None,
        eta_charge: TemporalDataUser = 1,
        eta_discharge: TemporalDataUser = 1,
        relative_loss_per_hour: TemporalDataUser = 0,
        prevent_simultaneous_charge_and_discharge: bool = True,
        balanced: bool = False,
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
            minimal_final_charge_state: relative minimal value of chargeState at the end of timeseries.
            maximal_final_charge_state: relative maximal value of chargeState at the end of timeseries.
            eta_charge: efficiency factor of charging/loading. The default is 1.
            eta_discharge: efficiency factor of uncharging/unloading. The default is 1.
            relative_loss_per_hour: loss per chargeState-Unit per hour. The default is 0.
            prevent_simultaneous_charge_and_discharge: If True, loading and unloading at the same time is not possible.
                Increases the number of binary variables, but is recommended for easier evaluation. The default is True.
            balanced: Wether to equate the size of the charging and discharging flow. Only if not fixed.
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
        self.relative_minimum_charge_state: TemporalDataUser = relative_minimum_charge_state
        self.relative_maximum_charge_state: TemporalDataUser = relative_maximum_charge_state

        self.relative_minimum_final_charge_state = relative_minimum_final_charge_state
        self.relative_maximum_final_charge_state = relative_maximum_final_charge_state

        self.initial_charge_state = initial_charge_state
        self.minimal_final_charge_state = minimal_final_charge_state
        self.maximal_final_charge_state = maximal_final_charge_state

        self.eta_charge: TemporalDataUser = eta_charge
        self.eta_discharge: TemporalDataUser = eta_discharge
        self.relative_loss_per_hour: TemporalDataUser = relative_loss_per_hour
        self.prevent_simultaneous_charge_and_discharge = prevent_simultaneous_charge_and_discharge
        self.balanced = balanced

    def create_model(self, model: FlowSystemModel) -> 'StorageModel':
        self._plausibility_checks()
        self.submodel = StorageModel(model, self)
        return self.submodel

    def transform_data(self, flow_system: 'FlowSystem') -> None:
        super().transform_data(flow_system)
        self.relative_minimum_charge_state = flow_system.fit_to_model_coords(
            f'{self.label_full}|relative_minimum_charge_state',
            self.relative_minimum_charge_state,
        )
        self.relative_maximum_charge_state = flow_system.fit_to_model_coords(
            f'{self.label_full}|relative_maximum_charge_state',
            self.relative_maximum_charge_state,
        )
        self.eta_charge = flow_system.fit_to_model_coords(f'{self.label_full}|eta_charge', self.eta_charge)
        self.eta_discharge = flow_system.fit_to_model_coords(f'{self.label_full}|eta_discharge', self.eta_discharge)
        self.relative_loss_per_hour = flow_system.fit_to_model_coords(
            f'{self.label_full}|relative_loss_per_hour', self.relative_loss_per_hour
        )
        if not isinstance(self.initial_charge_state, str):
            self.initial_charge_state = flow_system.fit_to_model_coords(
                f'{self.label_full}|initial_charge_state', self.initial_charge_state, has_time_dim=False
            )
        self.minimal_final_charge_state = flow_system.fit_to_model_coords(
            f'{self.label_full}|minimal_final_charge_state', self.minimal_final_charge_state, has_time_dim=False
        )
        self.maximal_final_charge_state = flow_system.fit_to_model_coords(
            f'{self.label_full}|maximal_final_charge_state', self.maximal_final_charge_state, has_time_dim=False
        )
        self.relative_minimum_final_charge_state = flow_system.fit_to_model_coords(
            f'{self.label_full}|relative_minimum_final_charge_state', self.relative_minimum_final_charge_state, has_time_dim=False
        )
        self.relative_maximum_final_charge_state = flow_system.fit_to_model_coords(
            f'{self.label_full}|relative_maximum_final_charge_state', self.relative_maximum_final_charge_state, has_time_dim=False
        )
        if isinstance(self.capacity_in_flow_hours, InvestParameters):
            self.capacity_in_flow_hours.transform_data(flow_system, f'{self.label_full}|InvestParameters')
        else:
            self.capacity_in_flow_hours = flow_system.fit_to_model_coords(
                f'{self.label_full}|capacity_in_flow_hours', self.capacity_in_flow_hours, has_time_dim=False
            )

    def _plausibility_checks(self) -> None:
        """
        Check for infeasible or uncommon combinations of parameters
        """
        super()._plausibility_checks()
        if isinstance(self.initial_charge_state, str):
            if self.initial_charge_state != 'lastValueOfSim':
                raise PlausibilityError(f'initial_charge_state has undefined value: {self.initial_charge_state}')
            return
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
        minimum_initial_capacity = maximum_capacity * self.relative_minimum_charge_state.isel(time=0)
        # initial capacity <= allowed max for minimum_size:
        maximum_initial_capacity = minimum_capacity * self.relative_maximum_charge_state.isel(time=0)

        if (self.initial_charge_state > maximum_initial_capacity).any():
            raise ValueError(
                f'{self.label_full}: {self.initial_charge_state=} '
                f'is above allowed maximum charge_state {maximum_initial_capacity}'
            )
        if (self.initial_charge_state < minimum_initial_capacity).any():
            raise ValueError(
                f'{self.label_full}: {self.initial_charge_state=} '
                f'is below allowed minimum charge_state {minimum_initial_capacity}'
            )

        if self.balanced:
            if not isinstance(self.charging.size, InvestParameters) or not isinstance(self.discharging.size, InvestParameters):
                raise PlausibilityError(
                    f'Balancing charging and discharging Flows in {self.label_full} '
                    f'is only possible with Investments.')
            if (self.charging.size.minimum_size > self.discharging.size.maximum_size or
                self.charging.size.maximum_size < self.discharging.size.minimum_size):
                raise PlausibilityError(
                    f'Balancing charging and discharging Flows in {self.label_full} need compatible minimum and maximum sizes.'
                    f'Got: {self.charging.size.minimum_size=}, {self.charging.size.maximum_size=} and '
                    f'{self.charging.size.minimum_size=}, {self.charging.size.maximum_size=}.')


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
        relative_losses: Optional[TemporalDataUser] = None,
        absolute_losses: Optional[TemporalDataUser] = None,
        on_off_parameters: OnOffParameters = None,
        prevent_simultaneous_flows_in_both_directions: bool = True,
        balanced: bool = False,
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
            balanced: Wether to equate the size of the in1 and in2 Flow. Needs InvestParameters in both Flows.
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
            if (
                    (self.in1.size.minimum_or_fixed_size > self.in2.size.maximum_or_fixed_size).any() or
                    (self.in1.size.maximum_or_fixed_size < self.in2.size.minimum_or_fixed_size).any()
            ):
                raise ValueError(
                    f'Balanced Transmission needs compatible minimum and maximum sizes.'
                    f'Got: {self.in1.size.minimum_size=}, {self.in1.size.maximum_size=}, {self.in1.size.fixed_size=} and '
                    f'{self.in2.size.minimum_size=}, {self.in2.size.maximum_size=}, {self.in2.size.fixed_size=}.')

    def create_model(self, model) -> 'TransmissionModel':
        self._plausibility_checks()
        self.submodel = TransmissionModel(model, self)
        return self.submodel

    def transform_data(self, flow_system: 'FlowSystem') -> None:
        super().transform_data(flow_system)
        self.relative_losses = flow_system.fit_to_model_coords(
            f'{self.label_full}|relative_losses', self.relative_losses
        )
        self.absolute_losses = flow_system.fit_to_model_coords(
            f'{self.label_full}|absolute_losses', self.absolute_losses
        )


class TransmissionModel(ComponentModel):
    def __init__(self, model: FlowSystemModel, element: Transmission):
        if (element.absolute_losses is not None) and np.any(element.absolute_losses != 0):
            for flow in element.inputs + element.outputs:
                if flow.on_off_parameters is None:
                    flow.on_off_parameters = OnOffParameters()
        self.element: Transmission = element
        self.on_off: Optional[OnOffModel] = None

        super().__init__(model, element)

    def _do_modeling(self):
        """Initiates all FlowModels"""
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
        con_transmission = self.add_constraints(
            out_flow.submodel.flow_rate == -in_flow.submodel.flow_rate * (self.element.relative_losses - 1),
            short_name=name,
        )

        if self.element.absolute_losses is not None:
            con_transmission.lhs += in_flow.submodel.on_off.on * self.element.absolute_losses

        return con_transmission


class LinearConverterModel(ComponentModel):
    def __init__(self, model: FlowSystemModel, element: LinearConverter):
        self.element: LinearConverter = element
        self.on_off: Optional[OnOffModel] = None
        self.piecewise_conversion: Optional[PiecewiseConversion] = None
        super().__init__(model, element)

    def _do_modeling(self):
        super()._do_modeling()
        # conversion_factors:
        if self.element.conversion_factors:
            all_input_flows = set(self.element.inputs)
            all_output_flows = set(self.element.outputs)

            # für alle linearen Gleichungen:
            for i, conv_factors in enumerate(self.element.conversion_factors):
                used_flows = set([self.element.flows[flow_label] for flow_label in conv_factors])
                used_inputs: Set = all_input_flows & used_flows
                used_outputs: Set = all_output_flows & used_flows

                self.add_constraints(
                    sum([flow.submodel.flow_rate * conv_factors[flow.label] for flow in used_inputs])
                    == sum([flow.submodel.flow_rate * conv_factors[flow.label] for flow in used_outputs]),
                    short_name=f'conversion_{i}',
                )

        else:
            # TODO: Improve Inclusion of OnOffParameters. Instead of creating a Binary in every flow, the binary could only be part of the Piece itself
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
                    zero_point=self.on_off.on if self.on_off is not None else False,
                    as_time_series=True,
                ),
                short_name='PiecewiseConversion',
            )


class StorageModel(ComponentModel):
    """Submodel of Storage"""

    def __init__(self, model: FlowSystemModel, element: Storage):
        super().__init__(model, element)

    def _do_modeling(self):
        super()._do_modeling()

        lb, ub = self._absolute_charge_state_bounds
        self.add_variables(
            lower=lb,
            upper=ub,
            coords=self._model.get_coords(extra_timestep=True),
            short_name='charge_state',
        )

        self.add_variables(coords=self._model.get_coords(), short_name='netto_discharge')

        # netto_discharge:
        # eq: nettoFlow(t) - discharging(t) + charging(t) = 0
        self.add_constraints(
            self.netto_discharge
            == self.element.discharging.submodel.flow_rate - self.element.charging.submodel.flow_rate,
            short_name='netto_discharge',
        )

        charge_state = self.charge_state
        rel_loss = self.element.relative_loss_per_hour
        hours_per_step = self._model.hours_per_step
        charge_rate = self.element.charging.submodel.flow_rate
        discharge_rate = self.element.discharging.submodel.flow_rate
        eff_charge = self.element.eta_charge
        eff_discharge = self.element.eta_discharge

        self.add_constraints(
            charge_state.isel(time=slice(1, None))
            == charge_state.isel(time=slice(None, -1)) * ((1 - rel_loss) ** hours_per_step)
            + charge_rate * eff_charge * hours_per_step
            - discharge_rate * eff_discharge * hours_per_step,
            short_name='charge_state',
        )

        if isinstance(self.element.capacity_in_flow_hours, InvestParameters):
            self.add_submodels(
                InvestmentModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    label_of_model=self.label_of_element,
                    parameters=self.element.capacity_in_flow_hours,
                ),
                short_name='investment',
            )

            BoundingPatterns.scaled_bounds(
                self,
                variable=self.charge_state,
                scaling_variable=self.investment.size,
                relative_bounds=self._relative_charge_state_bounds,
            )

        # Initial charge state
        self._initial_and_final_charge_state()

        if self.element.balanced:
            self.add_constraints(
                self.element.charging.model._investment.size * 1 == self.element.discharging.model._investment.size * 1,
                short_name='balanced_sizes',
            )

    def _initial_and_final_charge_state(self):
        if self.element.initial_charge_state is not None:
            if isinstance(self.element.initial_charge_state, str):
                self.add_constraints(
                    self.charge_state.isel(time=0) == self.charge_state.isel(time=-1), short_name='initial_charge_state'
                )
            else:
                self.add_constraints(
                    self.charge_state.isel(time=0) == self.element.initial_charge_state, short_name='initial_charge_state'
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

    @property
    def _absolute_charge_state_bounds(self) -> Tuple[TemporalData, TemporalData]:
        relative_lower_bound, relative_upper_bound = self._relative_charge_state_bounds
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
    def _relative_charge_state_bounds(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Get relative charge state bounds with final timestep values.

        Returns:
            Tuple of (minimum_bounds, maximum_bounds) DataArrays extending to final timestep
        """
        final_coords = {'time': [self._model.flow_system.timesteps_extra[-1]]}

        # Get final minimum charge state
        if self.element.relative_minimum_final_charge_state is None:
            min_final = self.element.relative_minimum_charge_state.isel(time=-1, drop=True)
        else:
            min_final = self.element.relative_minimum_final_charge_state
        min_final = min_final.expand_dims('time').assign_coords(time=final_coords['time'])

        # Get final maximum charge state
        if self.element.relative_maximum_final_charge_state is None:
            max_final = self.element.relative_maximum_charge_state.isel(time=-1, drop=True)
        else:
            max_final = self.element.relative_maximum_final_charge_state
        max_final = max_final.expand_dims('time').assign_coords(time=final_coords['time'])
        # Concatenate with original bounds
        min_bounds = xr.concat([self.element.relative_minimum_charge_state, min_final], dim='time')
        max_bounds = xr.concat([self.element.relative_maximum_charge_state, max_final], dim='time')

        return min_bounds, max_bounds

    @property
    def _investment(self) -> Optional[InvestmentModel]:
        """Deprecated alias for investment"""
        return self.investment

    @property
    def investment(self) -> Optional[InvestmentModel]:
        """OnOff feature"""
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
