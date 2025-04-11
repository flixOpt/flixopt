"""
This module contains the basic elements of the flixopt framework.
"""

import logging
import warnings
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import linopy
import numpy as np

from .config import CONFIG
from .core import NumericData, NumericDataTS, PlausibilityError, Scalar, TimeSeriesCollection
from .effects import EffectValuesUser
from .features import InvestmentModel, OnOffModel, PreventSimultaneousUsageModel
from .interface import InvestParameters, OnOffParameters
from .structure import Element, ElementModel, SystemModel, register_class_for_io

if TYPE_CHECKING:
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


@register_class_for_io
class Component(Element):
    """
    A Component contains incoming and outgoing [`Flows`][flixopt.elements.Flow]. It defines how these Flows interact with each other.
    The On or Off state of the Component is defined by all its Flows. Its on, if any of its FLows is On.
    It's mathematically advisable to define the On/Off state in a FLow rather than a Component if possible,
    as this introduces less binary variables to the Model
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

    def create_model(self, model: SystemModel) -> 'ComponentModel':
        self._plausibility_checks()
        self.model = ComponentModel(model, self)
        return self.model

    def transform_data(self, flow_system: 'FlowSystem') -> None:
        if self.on_off_parameters is not None:
            self.on_off_parameters.transform_data(flow_system, self.label_full)

    def infos(self, use_numpy=True, use_element_label: bool = False) -> Dict:
        infos = super().infos(use_numpy, use_element_label)
        infos['inputs'] = [flow.infos(use_numpy, use_element_label) for flow in self.inputs]
        infos['outputs'] = [flow.infos(use_numpy, use_element_label) for flow in self.outputs]
        return infos

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
        self, label: str, excess_penalty_per_flow_hour: Optional[NumericDataTS] = 1e5, meta_data: Optional[Dict] = None
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

    def create_model(self, model: SystemModel) -> 'BusModel':
        self._plausibility_checks()
        self.model = BusModel(model, self)
        return self.model

    def transform_data(self, flow_system: 'FlowSystem'):
        self.excess_penalty_per_flow_hour = flow_system.create_time_series(
            f'{self.label_full}|excess_penalty_per_flow_hour', self.excess_penalty_per_flow_hour
        )

    def _plausibility_checks(self) -> None:
        if self.excess_penalty_per_flow_hour is not None and (self.excess_penalty_per_flow_hour == 0).all():
            logger.warning(f'In Bus {self.label}, the excess_penalty_per_flow_hour is 0. Use "None" or a value > 0.')

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
        size: Union[Scalar, InvestParameters] = None,
        fixed_relative_profile: Optional[NumericDataTS] = None,
        relative_minimum: NumericDataTS = 0,
        relative_maximum: NumericDataTS = 1,
        effects_per_flow_hour: Optional[EffectValuesUser] = None,
        on_off_parameters: Optional[OnOffParameters] = None,
        flow_hours_total_max: Optional[Scalar] = None,
        flow_hours_total_min: Optional[Scalar] = None,
        load_factor_min: Optional[Scalar] = None,
        load_factor_max: Optional[Scalar] = None,
        previous_flow_rate: Optional[NumericData] = None,
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
            meta_data: used to store more information about the Element. Is not used internally, but saved in the results. Only use python native types.
        """
        super().__init__(label, meta_data=meta_data)
        self.size = size or CONFIG.modeling.BIG  # Default size
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

        self.previous_flow_rate = (
            previous_flow_rate if not isinstance(previous_flow_rate, list) else np.array(previous_flow_rate)
        )

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

    def create_model(self, model: SystemModel) -> 'FlowModel':
        self._plausibility_checks()
        self.model = FlowModel(model, self)
        return self.model

    def transform_data(self, flow_system: 'FlowSystem'):
        self.relative_minimum = flow_system.create_time_series(
            f'{self.label_full}|relative_minimum', self.relative_minimum
        )
        self.relative_maximum = flow_system.create_time_series(
            f'{self.label_full}|relative_maximum', self.relative_maximum
        )
        self.fixed_relative_profile = flow_system.create_time_series(
            f'{self.label_full}|fixed_relative_profile', self.fixed_relative_profile
        )
        self.effects_per_flow_hour = flow_system.create_effect_time_series(
            self.label_full, self.effects_per_flow_hour, 'per_flow_hour'
        )
        if self.on_off_parameters is not None:
            self.on_off_parameters.transform_data(flow_system, self.label_full)
        if isinstance(self.size, InvestParameters):
            self.size.transform_data(flow_system)

    def infos(self, use_numpy: bool = True, use_element_label: bool = False) -> Dict:
        infos = super().infos(use_numpy, use_element_label)
        infos['is_input_in_component'] = self.is_input_in_component
        return infos

    def to_dict(self) -> Dict:
        data = super().to_dict()
        if isinstance(data.get('previous_flow_rate'), np.ndarray):
            data['previous_flow_rate'] = data['previous_flow_rate'].tolist()
        return data

    def _plausibility_checks(self) -> None:
        # TODO: Incorporate into Variable? (Lower_bound can not be greater than upper bound
        if np.any(self.relative_minimum > self.relative_maximum):
            raise PlausibilityError(self.label_full + ': Take care, that relative_minimum <= relative_maximum!')

        if (
            self.size == CONFIG.modeling.BIG and self.fixed_relative_profile is not None
        ):  # Default Size --> Most likely by accident
            logger.warning(
                f'Flow "{self.label}" has no size assigned, but a "fixed_relative_profile". '
                f'The default size is {CONFIG.modeling.BIG}. As "flow_rate = size * fixed_relative_profile", '
                f'the resulting flow_rate will be very high. To fix this, assign a size to the Flow {self}.'
            )

        if self.fixed_relative_profile is not None and self.on_off_parameters is not None:
            raise ValueError(
                f'Flow {self.label} has both a fixed_relative_profile and an on_off_parameters. This is not supported. '
                f'Use relative_minimum and relative_maximum instead, '
                f'if you want to allow flows to be switched on and off.'
            )

        if (self.relative_minimum > 0).any() and self.on_off_parameters is None:
            logger.warning(
                f'Flow {self.label} has a relative_minimum of {self.relative_minimum.active_data} and no on_off_parameters. '
                f'This prevents the flow_rate from switching off (flow_rate = 0). '
                f'Consider using on_off_parameters to allow the flow to be switched on and off.'
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
    def __init__(self, model: SystemModel, element: Flow):
        super().__init__(model, element)
        self.element: Flow = element
        self.flow_rate: Optional[linopy.Variable] = None
        self.total_flow_hours: Optional[linopy.Variable] = None

        self.on_off: Optional[OnOffModel] = None
        self._investment: Optional[InvestmentModel] = None

    def do_modeling(self):
        # eq relative_minimum(t) * size <= flow_rate(t) <= relative_maximum(t) * size
        self.flow_rate: linopy.Variable = self.add(
            self._model.add_variables(
                lower=self.flow_rate_lower_bound,
                upper=self.flow_rate_upper_bound,
                coords=self._model.coords,
                name=f'{self.label_full}|flow_rate',
            ),
            'flow_rate',
        )

        # OnOff
        if self.element.on_off_parameters is not None:
            self.on_off: OnOffModel = self.add(
                OnOffModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    on_off_parameters=self.element.on_off_parameters,
                    defining_variables=[self.flow_rate],
                    defining_bounds=[self.flow_rate_bounds_on],
                    previous_values=[self.element.previous_flow_rate],
                ),
                'on_off',
            )
            self.on_off.do_modeling()

        # Investment
        if isinstance(self.element.size, InvestParameters):
            self._investment: InvestmentModel = self.add(
                InvestmentModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    parameters=self.element.size,
                    defining_variable=self.flow_rate,
                    relative_bounds_of_defining_variable=(self.flow_rate_lower_bound_relative,
                                                          self.flow_rate_upper_bound_relative),
                    on_variable=self.on_off.on if self.on_off is not None else None,
                ),
                'investment',
            )
            self._investment.do_modeling()

        self.total_flow_hours = self.add(
            self._model.add_variables(
                lower=self.element.flow_hours_total_min if self.element.flow_hours_total_min is not None else 0,
                upper=self.element.flow_hours_total_max if self.element.flow_hours_total_max is not None else np.inf,
                coords=None,
                name=f'{self.label_full}|total_flow_hours',
            ),
            'total_flow_hours',
        )

        self.add(
            self._model.add_constraints(
                self.total_flow_hours == (self.flow_rate * self._model.hours_per_step).sum(),
                name=f'{self.label_full}|total_flow_hours',
            ),
            'total_flow_hours',
        )

        # Load factor
        self._create_bounds_for_load_factor()

        # Shares
        self._create_shares()

    def _create_shares(self):
        # Arbeitskosten:
        if self.element.effects_per_flow_hour != {}:
            self._model.effects.add_share_to_effects(
                name=self.label_full,  # Use the full label of the element
                expressions={
                    effect: self.flow_rate * self._model.hours_per_step * factor.active_data
                    for effect, factor in self.element.effects_per_flow_hour.items()
                },
                target='operation',
            )

    def _create_bounds_for_load_factor(self):
        # TODO: Add Variable load_factor for better evaluation?

        # eq: var_sumFlowHours <= size * dt_tot * load_factor_max
        if self.element.load_factor_max is not None:
            name_short = 'load_factor_max'
            flow_hours_per_size_max = self._model.hours_per_step.sum() * self.element.load_factor_max
            size = self.element.size if self._investment is None else self._investment.size

            self.add(
                self._model.add_constraints(
                    self.total_flow_hours <= size * flow_hours_per_size_max,
                    name=f'{self.label_full}|{name_short}',
                ),
                name_short,
            )

        #  eq: size * sum(dt)* load_factor_min <= var_sumFlowHours
        if self.element.load_factor_min is not None:
            name_short = 'load_factor_min'
            flow_hours_per_size_min = self._model.hours_per_step.sum() * self.element.load_factor_min
            size = self.element.size if self._investment is None else self._investment.size

            self.add(
                self._model.add_constraints(
                    self.total_flow_hours >= size * flow_hours_per_size_min,
                    name=f'{self.label_full}|{name_short}',
                ),
                name_short,
            )

    @property
    def flow_rate_bounds_on(self) -> Tuple[NumericData, NumericData]:
        """Returns absolute flow rate bounds. Important for OnOffModel"""
        relative_minimum, relative_maximum = self.flow_rate_lower_bound_relative, self.flow_rate_upper_bound_relative
        size = self.element.size
        if not isinstance(size, InvestParameters):
            return relative_minimum * size, relative_maximum * size
        if size.fixed_size is not None:
            return relative_minimum * size.fixed_size, relative_maximum * size.fixed_size
        return relative_minimum * size.minimum_size, relative_maximum * size.maximum_size

    @property
    def flow_rate_lower_bound_relative(self) -> NumericData:
        """Returns the lower bound of the flow_rate relative to its size"""
        fixed_profile = self.element.fixed_relative_profile
        if fixed_profile is None:
            return self.element.relative_minimum.active_data
        return fixed_profile.active_data

    @property
    def flow_rate_upper_bound_relative(self) -> NumericData:
        """ Returns the upper bound of the flow_rate relative to its size"""
        fixed_profile = self.element.fixed_relative_profile
        if fixed_profile is None:
            return self.element.relative_maximum.active_data
        return fixed_profile.active_data

    @property
    def flow_rate_lower_bound(self) -> NumericData:
        """
        Returns the minimum bound the flow_rate can reach.
        Further constraining might be done in OnOffModel and InvestmentModel
        """
        if self.element.on_off_parameters is not None:
            return 0
        if isinstance(self.element.size, InvestParameters):
            if self.element.size.optional:
                return 0
            return self.flow_rate_lower_bound_relative * self.element.size.minimum_size
        return self.flow_rate_lower_bound_relative * self.element.size

    @property
    def flow_rate_upper_bound(self) -> NumericData:
        """
        Returns the maximum bound the flow_rate can reach.
        Further constraining might be done in OnOffModel and InvestmentModel
        """
        if isinstance(self.element.size, InvestParameters):
            return self.flow_rate_upper_bound_relative * self.element.size.maximum_size
        return self.flow_rate_upper_bound_relative * self.element.size


class BusModel(ElementModel):
    def __init__(self, model: SystemModel, element: Bus):
        super().__init__(model, element)
        self.element: Bus = element
        self.excess_input: Optional[linopy.Variable] = None
        self.excess_output: Optional[linopy.Variable] = None

    def do_modeling(self) -> None:
        # inputs == outputs
        for flow in self.element.inputs + self.element.outputs:
            self.add(flow.model.flow_rate, flow.label_full)
        inputs = sum([flow.model.flow_rate for flow in self.element.inputs])
        outputs = sum([flow.model.flow_rate for flow in self.element.outputs])
        eq_bus_balance = self.add(self._model.add_constraints(inputs == outputs, name=f'{self.label_full}|balance'))

        # Fehlerplus/-minus:
        if self.element.with_excess:
            excess_penalty = np.multiply(
                self._model.hours_per_step, self.element.excess_penalty_per_flow_hour.active_data
            )
            self.excess_input = self.add(
                self._model.add_variables(lower=0, coords=self._model.coords, name=f'{self.label_full}|excess_input'),
                'excess_input',
            )
            self.excess_output = self.add(
                self._model.add_variables(lower=0, coords=self._model.coords, name=f'{self.label_full}|excess_output'),
                'excess_output',
            )
            eq_bus_balance.lhs -= -self.excess_input + self.excess_output

            self._model.effects.add_share_to_penalty(self.label_of_element, (self.excess_input * excess_penalty).sum())
            self._model.effects.add_share_to_penalty(self.label_of_element, (self.excess_output * excess_penalty).sum())

    def results_structure(self):
        inputs = [flow.model.flow_rate.name for flow in self.element.inputs]
        outputs = [flow.model.flow_rate.name for flow in self.element.outputs]
        if self.excess_input is not None:
            inputs.append(self.excess_input.name)
        if self.excess_output is not None:
            outputs.append(self.excess_output.name)
        return {**super().results_structure(), 'inputs': inputs, 'outputs': outputs}


class ComponentModel(ElementModel):
    def __init__(self, model: SystemModel, element: Component):
        super().__init__(model, element)
        self.element: Component = element
        self.on_off: Optional[OnOffModel] = None

    def do_modeling(self):
        """Initiates all FlowModels"""
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
            self.add(flow.create_model(self._model), flow.label)

        for sub_model in self.sub_models:
            sub_model.do_modeling()

        if self.element.on_off_parameters:
            self.on_off = self.add(
                OnOffModel(
                    self._model,
                    self.element.on_off_parameters,
                    self.label_of_element,
                    defining_variables=[flow.model.flow_rate for flow in all_flows],
                    defining_bounds=[flow.model.flow_rate_bounds_on for flow in all_flows],
                    previous_values=[flow.previous_flow_rate for flow in all_flows],
                )
            )

            self.on_off.do_modeling()

        if self.element.prevent_simultaneous_flows:
            # Simultanious Useage --> Only One FLow is On at a time, but needs a Binary for every flow
            on_variables = [flow.model.on_off.on for flow in self.element.prevent_simultaneous_flows]
            simultaneous_use = self.add(PreventSimultaneousUsageModel(self._model, on_variables, self.label_full))
            simultaneous_use.do_modeling()

    def results_structure(self):
        return {
            **super().results_structure(),
            'inputs': [flow.model.flow_rate.name for flow in self.element.inputs],
            'outputs': [flow.model.flow_rate.name for flow in self.element.outputs],
        }
