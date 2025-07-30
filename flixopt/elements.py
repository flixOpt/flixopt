"""
This module contains the basic elements of the flixopt framework.
"""

import logging
import warnings
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import linopy
import numpy as np
import xarray as xr

from .config import CONFIG
from .core import PlausibilityError, Scalar, TemporalData, TemporalDataUser
from .effects import TemporalEffectsUser
from .features import InvestmentModel, InvestmentTimingModel, ModelingPrimitives, OnOffModel
from .interface import InvestParameters, InvestTimingParameters, OnOffParameters
from .modeling import BoundingPatterns, ModelingUtilitiesAbstract
from .structure import Element, ElementModel, FlowSystemModel, register_class_for_io

if TYPE_CHECKING:
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


@register_class_for_io
class Component(Element):
    """
    A Component contains incoming and outgoing [`Flows`][flixopt.elements.Flow]. It defines how these Flows interact with each other.
    The On or Off state of the Component is defined by all its Flows. Its on, if any of its FLows is On.
    It's mathematically advisable to define the On/Off state in a FLow rather than a Component if possible,
    as this introduces less binary variables to the Submodel
    Constraints to the On/Off state are defined by the [`on_off_parameters`][flixopt.interface.OnOffParameters].
    """

    def __init__(
        self,
        label: str,
        inputs: Optional[List['Flow']] = None,
        outputs: Optional[List['Flow']] = None,
        on_off_parameters: Optional[OnOffParameters] = None,
        prevent_simultaneous_flows: Optional[List['Flow']] = None,
        meta_data: Optional[Dict] = None,
    ):
        """
        Args:
            label: The label of the Element. Used to identify it in the FlowSystem
            inputs: input flows.
            outputs: output flows.
            on_off_parameters: Information about on and off state of Component.
                Component is On/Off, if all connected Flows are On/Off. This induces an On-Variable (binary) in all Flows!
                If possible, use OnOffParameters in a single Flow instead to keep the number of binary variables low.
                See class OnOffParameters.
            prevent_simultaneous_flows: Define a Group of Flows. Only one them can be on at a time.
                Induces On-Variable in all Flows! If possible, use OnOffParameters in a single Flow instead.
            meta_data: used to store more information about the Element. Is not used internally, but saved in the results. Only use python native types.
        """
        super().__init__(label, meta_data=meta_data)
        self.inputs: List['Flow'] = inputs or []
        self.outputs: List['Flow'] = outputs or []
        self._check_unique_flow_labels()
        self.on_off_parameters = on_off_parameters
        self.prevent_simultaneous_flows: List['Flow'] = prevent_simultaneous_flows or []

        self.flows: Dict[str, Flow] = {flow.label: flow for flow in self.inputs + self.outputs}

    def create_model(self, model: FlowSystemModel) -> 'ComponentModel':
        self._plausibility_checks()
        self.submodel = ComponentModel(model, self)
        return self.submodel

    def transform_data(self, flow_system: 'FlowSystem') -> None:
        if self.on_off_parameters is not None:
            self.on_off_parameters.transform_data(flow_system, self.label_full)

        for flow in self.inputs + self.outputs:
            flow.transform_data(flow_system)

    def _check_unique_flow_labels(self):
        all_flow_labels = [flow.label for flow in self.inputs + self.outputs]

        if len(set(all_flow_labels)) != len(all_flow_labels):
            duplicates = {label for label in all_flow_labels if all_flow_labels.count(label) > 1}
            raise ValueError(f'Flow names must be unique! "{self.label_full}" got 2 or more of: {duplicates}')

    def _plausibility_checks(self) -> None:
        self._check_unique_flow_labels()


@register_class_for_io
class Bus(Element):
    """
    A Bus represents a nodal balance between the flow rates of its incoming and outgoing Flows.
    """

    def __init__(
        self,
        label: str,
        excess_penalty_per_flow_hour: Optional[TemporalDataUser] = 1e5,
        meta_data: Optional[Dict] = None,
    ):
        """
        Args:
            label: The label of the Element. Used to identify it in the FlowSystem
            excess_penalty_per_flow_hour: excess costs / penalty costs (bus balance compensation)
                (none/ 0 -> no penalty). The default is 1e5.
                (Take care: if you use a timeseries (no scalar), timeseries is aggregated if calculation_type = aggregated!)
            meta_data: used to store more information about the Element. Is not used internally, but saved in the results. Only use python native types.
        """
        super().__init__(label, meta_data=meta_data)
        self.excess_penalty_per_flow_hour = excess_penalty_per_flow_hour
        self.inputs: List[Flow] = []
        self.outputs: List[Flow] = []

    def create_model(self, model: FlowSystemModel) -> 'BusModel':
        self._plausibility_checks()
        self.submodel = BusModel(model, self)
        return self.submodel

    def transform_data(self, flow_system: 'FlowSystem'):
        self.excess_penalty_per_flow_hour = flow_system.fit_to_model_coords(
            f'{self.label_full}|excess_penalty_per_flow_hour', self.excess_penalty_per_flow_hour
        )

    def _plausibility_checks(self) -> None:
        if self.excess_penalty_per_flow_hour is not None and (self.excess_penalty_per_flow_hour == 0).all():
            logger.warning(
                f'In Bus {self.label_full}, the excess_penalty_per_flow_hour is 0. Use "None" or a value > 0.'
            )

    @property
    def with_excess(self) -> bool:
        return False if self.excess_penalty_per_flow_hour is None else True


@register_class_for_io
class Connection:
    # input/output-dock (TODO:
    # -> wäre cool, damit Komponenten auch auch ohne Knoten verbindbar
    # input wären wie Flow,aber statt bus: connectsTo -> hier andere Connection oder aber Bus (dort keine Connection, weil nicht notwendig)

    def __init__(self):
        """
        This class is not yet implemented!
        """
        raise NotImplementedError()


@register_class_for_io
class Flow(Element):
    r"""
    A **Flow** moves energy (or material) between a [Bus][flixopt.elements.Bus] and a [Component][flixopt.elements.Component] in a predefined direction.
    The flow-rate is the main optimization variable of the **Flow**.
    """

    def __init__(
        self,
        label: str,
        bus: str,
        size: Union[Scalar, InvestParameters, InvestTimingParameters] = None,
        fixed_relative_profile: Optional[TemporalDataUser] = None,
        relative_minimum: TemporalDataUser = 0,
        relative_maximum: TemporalDataUser = 1,
        effects_per_flow_hour: Optional[TemporalEffectsUser] = None,
        on_off_parameters: Optional[OnOffParameters] = None,
        flow_hours_total_max: Optional[Scalar] = None,
        flow_hours_total_min: Optional[Scalar] = None,
        load_factor_min: Optional[Scalar] = None,
        load_factor_max: Optional[Scalar] = None,
        previous_flow_rate: Optional[Union[Scalar, List[Scalar]]] = None,
        meta_data: Optional[Dict] = None,
    ):
        r"""
        Args:
            label: The label of the FLow. Used to identify it in the FlowSystem. Its `full_label` consists of the label of the Component and the label of the Flow.
            bus: blabel of the bus the flow is connected to.
            size: size of the flow. If InvestmentParameters is used, size is optimized.
                If size is None, a default value is used.
            relative_minimum: min value is relative_minimum multiplied by size
            relative_maximum: max value is relative_maximum multiplied by size. If size = max then relative_maximum=1
            load_factor_min: minimal load factor  general: avg Flow per nominalVal/investSize
                (e.g. boiler, kW/kWh=h; solarthermal: kW/m²;
                 def: :math:`load\_factor:= sumFlowHours/ (nominal\_val \cdot \Delta t_{tot})`
            load_factor_max: maximal load factor (see minimal load factor)
            effects_per_flow_hour: operational costs, costs per flow-"work"
            on_off_parameters: If present, flow can be "off", i.e. be zero (only relevant if relative_minimum > 0)
                Therefore a binary var "on" is used. Further, several other restrictions and effects can be modeled
                through this On/Off State (See OnOffParameters)
            flow_hours_total_max: maximum flow-hours ("flow-work")
                (if size is not const, maybe load_factor_max is the better choice!)
            flow_hours_total_min: minimum flow-hours ("flow-work")
                (if size is not predefined, maybe load_factor_min is the better choice!)
            fixed_relative_profile: fixed relative values for flow (if given).
                flow_rate(t) := fixed_relative_profile(t) * size(t)
                With this value, the flow_rate is no optimization-variable anymore.
                (relative_minimum and relative_maximum are ignored)
                used for fixed load or supply profiles, i.g. heat demand, wind-power, solarthermal
                If the load-profile is just an upper limit, use relative_maximum instead.
            previous_flow_rate: previous flow rate of the flow. Used to determine if and how long the
                flow is already on / off. If None, the flow is considered to be off for one timestep.
                Currently does not support different values in different years or scenarios!
            meta_data: used to store more information about the Element. Is not used internally, but saved in the results. Only use python native types.
        """
        super().__init__(label, meta_data=meta_data)
        self.size = size if size is not None else CONFIG.modeling.BIG  # Default size
        self.relative_minimum = relative_minimum
        self.relative_maximum = relative_maximum
        self.fixed_relative_profile = fixed_relative_profile

        self.load_factor_min = load_factor_min
        self.load_factor_max = load_factor_max
        # self.positive_gradient = TimeSeries('positive_gradient', positive_gradient, self)
        self.effects_per_flow_hour = effects_per_flow_hour if effects_per_flow_hour is not None else {}
        self.flow_hours_total_max = flow_hours_total_max
        self.flow_hours_total_min = flow_hours_total_min
        self.on_off_parameters = on_off_parameters

        self.previous_flow_rate = previous_flow_rate

        self.component: str = 'UnknownComponent'
        self.is_input_in_component: Optional[bool] = None
        if isinstance(bus, Bus):
            self.bus = bus.label_full
            warnings.warn(
                f'Bus {bus.label} is passed as a Bus object to {self.label}. This is deprecated and will be removed '
                f'in the future. Add the Bus to the FlowSystem instead and pass its label to the Flow.',
                UserWarning,
                stacklevel=1,
            )
            self._bus_object = bus
        else:
            self.bus = bus
            self._bus_object = None

    def create_model(self, model: FlowSystemModel) -> 'FlowModel':
        self._plausibility_checks()
        self.submodel = FlowModel(model, self)
        return self.submodel

    def transform_data(self, flow_system: 'FlowSystem'):
        self.relative_minimum = flow_system.fit_to_model_coords(
            f'{self.label_full}|relative_minimum', self.relative_minimum
        )
        self.relative_maximum = flow_system.fit_to_model_coords(
            f'{self.label_full}|relative_maximum', self.relative_maximum
        )
        self.fixed_relative_profile = flow_system.fit_to_model_coords(
            f'{self.label_full}|fixed_relative_profile', self.fixed_relative_profile
        )
        self.effects_per_flow_hour = flow_system.fit_effects_to_model_coords(
            self.label_full, self.effects_per_flow_hour, 'per_flow_hour'
        )
        self.flow_hours_total_max = flow_system.fit_to_model_coords(
            f'{self.label_full}|flow_hours_total_max', self.flow_hours_total_max, has_time_dim=False
        )
        self.flow_hours_total_min = flow_system.fit_to_model_coords(
            f'{self.label_full}|flow_hours_total_min', self.flow_hours_total_min, has_time_dim=False
        )
        self.load_factor_max = flow_system.fit_to_model_coords(
            f'{self.label_full}|load_factor_max', self.load_factor_max, has_time_dim=False
        )
        self.load_factor_min = flow_system.fit_to_model_coords(
            f'{self.label_full}|load_factor_min', self.load_factor_min, has_time_dim=False
        )

        if self.on_off_parameters is not None:
            self.on_off_parameters.transform_data(flow_system, self.label_full)
        if isinstance(self.size, (InvestParameters, InvestTimingParameters)):
            self.size.transform_data(flow_system, self.label_full)
        else:
            self.size = flow_system.fit_to_model_coords(f'{self.label_full}|size', self.size, has_time_dim=False)

    def _plausibility_checks(self) -> None:
        # TODO: Incorporate into Variable? (Lower_bound can not be greater than upper bound
        if np.any(self.relative_minimum > self.relative_maximum):
            raise PlausibilityError(self.label_full + ': Take care, that relative_minimum <= relative_maximum!')

        if not isinstance(self.size, (InvestParameters, InvestTimingParameters)) and (
            np.any(self.size == CONFIG.modeling.BIG) and self.fixed_relative_profile is not None
        ):  # Default Size --> Most likely by accident
            logger.warning(
                f'Flow "{self.label_full}" has no size assigned, but a "fixed_relative_profile". '
                f'The default size is {CONFIG.modeling.BIG}. As "flow_rate = size * fixed_relative_profile", '
                f'the resulting flow_rate will be very high. To fix this, assign a size to the Flow {self}.'
            )

        if self.fixed_relative_profile is not None and self.on_off_parameters is not None:
            logger.warning(
                f'Flow {self.label_full} has both a fixed_relative_profile and an on_off_parameters.'
                f'This will allow the flow to be switched on and off, effectively differing from the fixed_flow_rate.'
            )

        if (self.relative_minimum > 0).any() and self.on_off_parameters is None:
            logger.warning(
                f'Flow {self.label_full} has a relative_minimum of {self.relative_minimum} and no on_off_parameters. '
                f'This prevents the flow_rate from switching off (flow_rate = 0). '
                f'Consider using on_off_parameters to allow the flow to be switched on and off.'
            )

        if self.previous_flow_rate is not None:
            if not any(
                [
                    isinstance(self.previous_flow_rate, np.ndarray) and self.previous_flow_rate.ndim == 1,
                    isinstance(self.previous_flow_rate, (int, float, list)),
                ]
            ):
                raise TypeError(
                    f'previous_flow_rate must be None, a scalar, a list of scalars or a 1D-numpy-array. Got {type(self.previous_flow_rate)}.'
                    f'Different values in different years or scenarios are not yetsupported.'
                )

    @property
    def label_full(self) -> str:
        return f'{self.component}({self.label})'

    @property
    def size_is_fixed(self) -> bool:
        # Wenn kein InvestParameters existiert --> True; Wenn Investparameter, den Wert davon nehmen
        return False if (isinstance(self.size, InvestParameters) and self.size.fixed_size is None) else True

    @property
    def invest_is_optional(self) -> bool:
        # Wenn kein InvestParameters existiert: # Investment ist nicht optional -> Keine Variable --> False
        return False if (isinstance(self.size, InvestParameters) and not self.size.optional) else True


class FlowModel(ElementModel):
    element: Flow  # Type hint

    def __init__(self, model: FlowSystemModel, element: Flow):
        super().__init__(model, element)

    def _do_modeling(self):
        super()._do_modeling()
        # Main flow rate variable
        self.add_variables(
            lower=self.absolute_flow_rate_bounds[0],
            upper=self.absolute_flow_rate_bounds[1],
            coords=self._model.get_coords(),
            short_name='flow_rate',
        )

        self._constraint_flow_rate()

        # Total flow hours tracking
        ModelingPrimitives.expression_tracking_variable(
            model=self,
            name=f'{self.label_full}|total_flow_hours',
            tracked_expression=(self.flow_rate * self._model.hours_per_step).sum('time'),
            bounds=(
                self.element.flow_hours_total_min if self.element.flow_hours_total_min is not None else 0,
                self.element.flow_hours_total_max if self.element.flow_hours_total_max is not None else None,
            ),
            coords=['year', 'scenario'],
            short_name='total_flow_hours',
        )

        # Load factor constraints
        self._create_bounds_for_load_factor()

        # Effects
        self._create_shares()

    def _create_on_off_model(self):
        on = self.add_variables(binary=True, short_name='on', coords=self._model.get_coords())
        self.add_submodels(
            OnOffModel(
                model=self._model,
                label_of_element=self.label_of_element,
                parameters=self.element.on_off_parameters,
                on_variable=on,
                previous_states=self.previous_states,
                label_of_model=self.label_of_element,
            ),
            short_name='on_off',
        )

    def _create_investment_model(self):
        if isinstance(self.element.size, InvestParameters):
            self.add_submodels(
                InvestmentModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    parameters=self.element.size,
                    label_of_model=self.label_of_element,
                ),
                'investment',
            )
        elif isinstance(self.element.size, InvestTimingParameters):
            self.add_submodels(
                InvestmentTimingModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    parameters=self.element.size,
                    label_of_model=self.label_of_element,
                ),
                'investment',
            )
        else:
            raise ValueError(f'Invalid InvestParameters type: {type(self.element.size)}')

    def _constraint_flow_rate(self):
        if not self.with_investment and not self.with_on_off:
            # Most basic case. Already covered by direct variable bounds
            pass

        elif self.with_on_off and not self.with_investment:
            # OnOff, but no Investment
            self._create_on_off_model()
            bounds = self.relative_flow_rate_bounds
            BoundingPatterns.bounds_with_state(
                self,
                variable=self.flow_rate,
                bounds=(bounds[0] * self.element.size, bounds[1] * self.element.size),
                variable_state=self.on_off.on,
            )

        elif self.with_investment and not self.with_on_off:
            # Investment, but no OnOff
            self._create_investment_model()
            BoundingPatterns.scaled_bounds(
                self,
                variable=self.flow_rate,
                scaling_variable=self.investment.size,
                relative_bounds=self.relative_flow_rate_bounds,
            )

        elif self.with_investment and self.with_on_off:
            # Investment and OnOff
            self._create_investment_model()
            self._create_on_off_model()

            BoundingPatterns.scaled_bounds_with_state(
                model=self,
                variable=self.flow_rate,
                scaling_variable=self._investment.size,
                relative_bounds=self.relative_flow_rate_bounds,
                scaling_bounds=(self.element.size.minimum_or_fixed_size, self.element.size.maximum_or_fixed_size),
                variable_state=self.on_off.on,
            )
        else:
            raise Exception('Not valid')

    @property
    def with_on_off(self) -> bool:
        return self.element.on_off_parameters is not None

    @property
    def with_investment(self) -> bool:
        return isinstance(self.element.size, (InvestParameters, InvestTimingParameters))

    # Properties for clean access to variables
    @property
    def flow_rate(self) -> linopy.Variable:
        """Main flow rate variable"""
        return self['flow_rate']

    @property
    def total_flow_hours(self) -> linopy.Variable:
        """Total flow hours variable"""
        return self['total_flow_hours']

    def results_structure(self):
        return {
            **super().results_structure(),
            'start': self.element.bus if self.element.is_input_in_component else self.element.component,
            'end': self.element.component if self.element.is_input_in_component else self.element.bus,
            'component': self.element.component,
        }

    def _create_shares(self):
        # Effects per flow hour
        if self.element.effects_per_flow_hour:
            self._model.effects.add_share_to_effects(
                name=self.label_full,
                expressions={
                    effect: self.flow_rate * self._model.hours_per_step * factor
                    for effect, factor in self.element.effects_per_flow_hour.items()
                },
                target='operation',
            )

    def _create_bounds_for_load_factor(self):
        """Create load factor constraints using current approach"""
        # Get the size (either from element or investment)
        size = self.investment.size if self.with_investment else self.element.size

        # Maximum load factor constraint
        if self.element.load_factor_max is not None:
            flow_hours_per_size_max = self._model.hours_per_step.sum('time') * self.element.load_factor_max
            self.add_constraints(
                self.total_flow_hours <= size * flow_hours_per_size_max,
                short_name='load_factor_max',
            )

        # Minimum load factor constraint
        if self.element.load_factor_min is not None:
            flow_hours_per_size_min = self._model.hours_per_step.sum('time') * self.element.load_factor_min
            self.add_constraints(
                self.total_flow_hours >= size * flow_hours_per_size_min,
                short_name='load_factor_min',
            )

    @property
    def relative_flow_rate_bounds(self) -> Tuple[TemporalData, TemporalData]:
        if self.element.fixed_relative_profile is not None:
            return self.element.fixed_relative_profile, self.element.fixed_relative_profile
        return self.element.relative_minimum, self.element.relative_maximum

    @property
    def absolute_flow_rate_bounds(self) -> Tuple[TemporalData, TemporalData]:
        """
        Returns the absolute bounds the flow_rate can reach.
        Further constraining might be needed
        """
        lb_relative, ub_relative = self.relative_flow_rate_bounds

        lb = 0
        if not self.with_on_off:
            if not self.with_investment:
                # Basic case without investment and without OnOff
                lb = lb_relative * self.element.size
            elif isinstance(self.element.size, InvestParameters) and not self.element.size.optional:
                # With non-optional Investment
                lb = lb_relative * self.element.size.minimum_or_fixed_size

        if self.with_investment:
            ub = ub_relative * self.element.size.maximum_or_fixed_size
        else:
            ub = ub_relative * self.element.size

        return lb, ub

    @property
    def on_off(self) -> Optional[OnOffModel]:
        """OnOff feature"""
        if 'on_off' not in self.submodels:
            return None
        return self.submodels['on_off']

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
    def previous_states(self) -> Optional[TemporalData]:
        """Previous states of the flow rate"""
        # TODO: This would be nicer to handle in the Flow itself, and allow DataArrays as well.
        previous_flow_rate = self.element.previous_flow_rate
        if previous_flow_rate is None:
            return None

        return ModelingUtilitiesAbstract.to_binary(
            values=xr.DataArray(
                [previous_flow_rate] if np.isscalar(previous_flow_rate) else previous_flow_rate, dims='time'
            ),
            epsilon=CONFIG.modeling.EPSILON,
            dims='time',
        )


class BusModel(ElementModel):
    element: Bus  # Type hint

    def __init__(self, model: FlowSystemModel, element: Bus):
        self.excess_input: Optional[linopy.Variable] = None
        self.excess_output: Optional[linopy.Variable] = None
        super().__init__(model, element)

    def _do_modeling(self) -> None:
        super()._do_modeling()
        # inputs == outputs
        for flow in self.element.inputs + self.element.outputs:
            self.register_variable(flow.submodel.flow_rate, flow.label_full)
        inputs = sum([flow.submodel.flow_rate for flow in self.element.inputs])
        outputs = sum([flow.submodel.flow_rate for flow in self.element.outputs])
        eq_bus_balance = self.add_constraints(inputs == outputs, short_name='balance')

        # Fehlerplus/-minus:
        if self.element.with_excess:
            excess_penalty = np.multiply(self._model.hours_per_step, self.element.excess_penalty_per_flow_hour)

            self.excess_input = self.add_variables(lower=0, coords=self._model.get_coords(), short_name='excess_input')

            self.excess_output = self.add_variables(
                lower=0, coords=self._model.get_coords(), short_name='excess_output'
            )

            eq_bus_balance.lhs -= -self.excess_input + self.excess_output

            self._model.effects.add_share_to_penalty(self.label_of_element, (self.excess_input * excess_penalty).sum())
            self._model.effects.add_share_to_penalty(self.label_of_element, (self.excess_output * excess_penalty).sum())

    def results_structure(self):
        inputs = [flow.submodel.flow_rate.name for flow in self.element.inputs]
        outputs = [flow.submodel.flow_rate.name for flow in self.element.outputs]
        if self.excess_input is not None:
            inputs.append(self.excess_input.name)
        if self.excess_output is not None:
            outputs.append(self.excess_output.name)
        return {
            **super().results_structure(),
            'inputs': inputs,
            'outputs': outputs,
            'flows': [flow.label_full for flow in self.element.inputs + self.element.outputs],
        }


class ComponentModel(ElementModel):
    element: Component  # Type hint

    def __init__(self, model: FlowSystemModel, element: Component):
        self.on_off: Optional[OnOffModel] = None
        super().__init__(model, element)

    def _do_modeling(self):
        """Initiates all FlowModels"""
        super()._do_modeling()
        all_flows = self.element.inputs + self.element.outputs
        if self.element.on_off_parameters:
            for flow in all_flows:
                if flow.on_off_parameters is None:
                    flow.on_off_parameters = OnOffParameters()

        if self.element.prevent_simultaneous_flows:
            for flow in self.element.prevent_simultaneous_flows:
                if flow.on_off_parameters is None:
                    flow.on_off_parameters = OnOffParameters()

        for flow in all_flows:
            self.add_submodels(flow.create_model(self._model), short_name=flow.label)

        if self.element.on_off_parameters:
            on = self.add_variables(binary=True, short_name='on', coords=self._model.get_coords())
            if len(all_flows) == 1:
                self.add_constraints(on == all_flows[0].submodel.on_off.on, short_name='on')
            else:
                flow_ons = [flow.submodel.on_off.on for flow in all_flows]
                # TODO: Is the EPSILON even necessary?
                self.add_constraints(on <= sum(flow_ons) + CONFIG.modeling.EPSILON, short_name='on|ub')
                self.add_constraints(
                    on >= sum(flow_ons) / (len(flow_ons) + CONFIG.modeling.EPSILON), short_name='on|lb'
                )

            self.on_off = self.add_submodels(
                OnOffModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    parameters=self.element.on_off_parameters,
                    on_variable=on,
                    label_of_model=self.label_of_element,
                    previous_states=self.previous_states,
                ),
                short_name='on_off',
            )

        if self.element.prevent_simultaneous_flows:
            # Simultanious Useage --> Only One FLow is On at a time, but needs a Binary for every flow
            ModelingPrimitives.mutual_exclusivity_constraint(
                self,
                binary_variables=[flow.submodel.on_off.on for flow in self.element.prevent_simultaneous_flows],
                short_name='prevent_simultaneous_use',
            )

    def results_structure(self):
        return {
            **super().results_structure(),
            'inputs': [flow.submodel.flow_rate.name for flow in self.element.inputs],
            'outputs': [flow.submodel.flow_rate.name for flow in self.element.outputs],
            'flows': [flow.label_full for flow in self.element.inputs + self.element.outputs],
        }

    @property
    def previous_states(self) -> Optional[xr.DataArray]:
        """Previous state of the component, derived from its flows"""
        if self.element.on_off_parameters is None:
            raise ValueError(f'OnOffModel not present in \n{self}\nCant access previous_states')

        previous_states = [flow.submodel.on_off._previous_states for flow in self.element.inputs + self.element.outputs]
        previous_states = [da for da in previous_states if da is not None]

        if not previous_states:  # Empty list
            return None

        max_len = max(da.sizes['time'] for da in previous_states)

        padded_previous_states = [
            da.assign_coords(time=range(-da.sizes['time'], 0)).reindex(time=range(-max_len, 0), fill_value=0)
            for da in previous_states
        ]
        return xr.concat(padded_previous_states, dim='flow').any(dim='flow').astype(int)
