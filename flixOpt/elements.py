"""
This module contains the basic elements of the flixOpt framework.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .config import CONFIG
from .core import Numeric, Numeric_TS, Skalar
from .effects import EffectValues, effect_values_to_time_series
from .features import InvestmentModel, OnOffModel, PreventSimultaneousUsageModel
from .interface import InvestParameters, OnOffParameters
from .math_modeling import Variable, VariableTS
from .structure import (
    Element,
    ElementModel,
    Interface,
    SystemModel,
    _create_time_series,
    copy_and_convert_datatypes,
    create_equation,
    create_variable,
)

logger = logging.getLogger('flixOpt')


class MediumCategories:
    """
    A centralized class to define standard medium categories.
    """

    heat = 'heat'
    electricity = 'electricity'
    fuel = 'fuel'


class MediumColors:
    """
    A centralized class to define default colors for medium categories.
    Change values before creating new mediums!
    """

    colors = {
        MediumCategories.heat: 'orange',
        MediumCategories.electricity: 'steelblue',
        MediumCategories.fuel: 'brown',
    }

    default = 'gray'

    @classmethod
    def get_color(cls, category: str = '') -> str:
        """
        Returns the color associated with a category. Returns default color if category is not found.

        Parameters
        ----------
        category : str
            The category for which to retrieve the color.

        Returns
        -------
        str
            The color code (hex format).
        """
        return cls.colors.get(category, cls.default)


class Medium(Interface):
    """
    Represents a medium with attributes such as label, unit, color, and categories.
    """

    def __init__(
        self,
        label: str,
        unit: str,
        color: Optional[str] = None,
        categories: Optional[List[str]] = None,
        description: Optional[str] = None,
    ):
        """
        Creates a new Medium object.

        Parameters
        ----------
        label : str
            The label of the medium.
        unit : str
            The unit of the medium.
        color : str, optional
            The color of the medium. Defaults to the color of the first category, or 'unknown' if no category is specified.
        categories : List[str], optional
            The categories of the medium. Used for validation purposes.
            It's recommended to use the default categories from "MediumCategories".
        description : str, optional
            A description of the medium.
        """
        self.label = label
        self.unit = unit
        self.categories: List[str] = categories or []
        self.color = color or self._get_default_color()
        self.description = description

    def _get_default_color(self) -> str:
        """
        Determines the default color for the medium based on its categories.

        Returns
        -------
        str
            The default color of the medium.
        """
        if self.categories:
            # Use the first category's color as default
            return MediumColors.get_color(self.categories[0])
        return MediumColors.get_color('unknown')


class Component(Element):
    """
    basic component class for all components
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
        Parameters
        ----------
        label : str
            name.
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        inputs : input flows.
        outputs : output flows.
        on_off_parameters: Information about on and off state of Component.
            Component is On/Off, if all connected Flows are On/Off.
            Induces On-Variable in all FLows!
            See class OnOffParameters.
        prevent_simultaneous_flows: Define a Group of Flows. Only one them can be on at a time.
            Induces On-Variable in all FLows!
        """
        super().__init__(label, meta_data=meta_data)
        self.inputs: List['Flow'] = inputs or []
        self.outputs: List['Flow'] = outputs or []
        self.on_off_parameters = on_off_parameters
        self.prevent_simultaneous_flows: List['Flow'] = prevent_simultaneous_flows or []

        self.flows: Dict[str, Flow] = {flow.label: flow for flow in self.inputs + self.outputs}

    def create_model(self) -> 'ComponentModel':
        self.model = ComponentModel(self)
        return self.model

    def transform_data(self) -> None:
        if self.on_off_parameters is not None:
            self.on_off_parameters.transform_data(self)

    def connect_elements(self) -> None:
        """
        Connects components, flows and buses to each other.
        """
        for flow in self.inputs + self.outputs:
            flow.comp = self
        # Register flows in buses
        for flow in self.inputs:
            flow.bus.add_output(flow)
        for flow in self.outputs:
            flow.bus.add_input(flow)

    def infos(self, use_numpy=True, use_element_label=False) -> Dict:
        infos = super().infos(use_numpy, use_element_label)
        infos['inputs'] = [flow.infos(use_numpy, use_element_label) for flow in self.inputs]
        infos['outputs'] = [flow.infos(use_numpy, use_element_label) for flow in self.outputs]
        return infos


class Bus(Element):
    """
    realizing balance of all linked flows
    (penalty flow is excess can be activated)
    """

    def __init__(
        self,
        label: str,
        medium: Optional[Medium] = None,
        excess_penalty_per_flow_hour: Optional[Numeric_TS] = 1e5,
        meta_data: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        label : str
            name.
        medium : Medium
            The medium of the bus. This is used to validate flows connected to the bus.
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        excess_penalty_per_flow_hour : none or scalar, array or TimeSeriesData
            excess costs / penalty costs (bus balance compensation)
            (none/ 0 -> no penalty). The default is 1e5.
            (Take care: if you use a timeseries (no scalar), timeseries is aggregated if calculation_type = aggregated!)
        """
        super().__init__(label, meta_data=meta_data)
        self.medium = medium
        self.excess_penalty_per_flow_hour = excess_penalty_per_flow_hour
        self.inputs: List[Flow] = []
        self.outputs: List[Flow] = []

    def create_model(self) -> 'BusModel':
        self.model = BusModel(self)
        return self.model

    def transform_data(self):
        self.excess_penalty_per_flow_hour = _create_time_series(
            'excess_penalty_per_flow_hour', self.excess_penalty_per_flow_hour, self
        )

    def add_input(self, flow: 'Flow') -> None:
        self.inputs.append(flow)
        self._assign_medium(flow)

    def add_output(self, flow: 'Flow') -> None:
        self.outputs.append(flow)
        self._assign_medium(flow)

    def _assign_medium(self, flow: 'Flow') -> None:
        """
        Assigns the bus.medium to the flow.medium.
        Tries to ensure that the medium of the flow is compatible with the medium of the bus.
        If not, a logger warning is raised.
        """
        if flow.medium is None or self.medium is None:
            pass
        elif flow.medium in self.medium.categories:
            pass
        else:
            logger.warning(
                f'Flow "{flow.label_full}" is not compatible with its connected Bus "{self.label_full}" and its medium '
                f'"{self.medium.label}", according to medium categories: {flow.medium=}, '
                f'bus.medium.categories={self.medium.categories}. This might be the cause of connecting a Flow to the '
                f'wrong bus. Further, this might lead to inconsistent plotting regarding units and colors '
                f'(which doesnt interfere with the calculation itself!).'
            )
        flow.medium = self.medium

    def _plausibility_checks(self) -> None:
        if self.excess_penalty_per_flow_hour == 0:
            logger.warning(f'In Bus {self.label}, the excess_penalty_per_flow_hour is 0. Use "None" or a value > 0.')

    @property
    def with_excess(self) -> bool:
        return False if self.excess_penalty_per_flow_hour is None else True


class Connection:
    # input/output-dock (TODO:
    # -> wäre cool, damit Komponenten auch auch ohne Knoten verbindbar
    # input wären wie Flow,aber statt bus : connectsTo -> hier andere Connection oder aber Bus (dort keine Connection, weil nicht notwendig)

    def __init__(self):
        raise NotImplementedError()


class Flow(Element):
    """
    flows are inputs and outputs of components
    """

    def __init__(
        self,
        label: str,
        bus: Bus,
        size: Union[Skalar, InvestParameters] = None,
        fixed_relative_profile: Optional[Numeric_TS] = None,
        relative_minimum: Numeric_TS = 0,
        relative_maximum: Numeric_TS = 1,
        effects_per_flow_hour: EffectValues = None,
        on_off_parameters: Optional[OnOffParameters] = None,
        flow_hours_total_max: Optional[Skalar] = None,
        flow_hours_total_min: Optional[Skalar] = None,
        load_factor_min: Optional[Skalar] = None,
        load_factor_max: Optional[Skalar] = None,
        previous_flow_rate: Optional[Numeric] = None,
        medium: Optional[str] = None,
        meta_data: Optional[Dict] = None,
    ):
        r"""
        Parameters
        ----------
        label : str
            name of flow
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        bus : Bus, optional
            bus to which flow is linked
        size : scalar, InvestmentParameters, optional
            size of the flow. If InvestmentParameters is used, size is optimized.
            If size is None, a default value is used.
        relative_minimum : scalar, array, TimeSeriesData, optional
            min value is relative_minimum multiplied by size
        relative_maximum : scalar, array, TimeSeriesData, optional
            max value is relative_maximum multiplied by size. If size = max then relative_maximum=1
        load_factor_min : scalar, optional
            minimal load factor  general: avg Flow per nominalVal/investSize
            (e.g. boiler, kW/kWh=h; solarthermal: kW/m²;
             def: :math:`load\_factor:= sumFlowHours/ (nominal\_val \cdot \Delta t_{tot})`
        load_factor_max : scalar, optional
            maximal load factor (see minimal load factor)
        effects_per_flow_hour : scalar, array, TimeSeriesData, optional
            operational costs, costs per flow-"work"
        on_off_parameters : OnOffParameters, optional
            If present, flow can be "off", i.e. be zero (only relevant if relative_minimum > 0)
            Therefore a binary var "on" is used. Further, several other restrictions and effects can be modeled
            through this On/Off State (See OnOffParameters)
        flow_hours_total_max : TYPE, optional
            maximum flow-hours ("flow-work")
            (if size is not const, maybe load_factor_max fits better for you!)
        flow_hours_total_min : TYPE, optional
            minimum flow-hours ("flow-work")
            (if size is not const, maybe load_factor_min fits better for you!)
        fixed_relative_profile : scalar, array, TimeSeriesData, optional
            fixed relative values for flow (if given).
            val(t) := fixed_relative_profile(t) * size(t)
            With this value, the flow_rate is no opt-variable anymore;
            (relative_minimum u. relative_maximum are making sense anymore)
            used for fixed load profiles, i.g. heat demand, wind-power, solarthermal
            If the load-profile is just an upper limit, use relative_maximum instead.
        previous_flow_rate : scalar, array, optional
            previous flow rate of the component.
        medium : str, optional
            The medium category of the flow. Used for validation purposes.
        meta_data : Dict, optional
            Additional metadata for the flow.
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

        self.previous_flow_rate = previous_flow_rate

        self.bus = bus
        self.medium: Optional[Medium, str] = medium
        self.comp: Optional[Component] = None

        self._plausibility_checks()

    def create_model(self) -> 'FlowModel':
        self.model = FlowModel(self)
        return self.model

    def transform_data(self):
        self.relative_minimum = _create_time_series('relative_minimum', self.relative_minimum, self)
        self.relative_maximum = _create_time_series('relative_maximum', self.relative_maximum, self)
        self.fixed_relative_profile = _create_time_series('fixed_relative_profile', self.fixed_relative_profile, self)
        self.effects_per_flow_hour = effect_values_to_time_series('per_flow_hour', self.effects_per_flow_hour, self)
        if self.on_off_parameters is not None:
            self.on_off_parameters.transform_data(self)
        if isinstance(self.size, InvestParameters):
            self.size.transform_data()

    def infos(self, use_numpy=True, use_element_label=False) -> Dict:
        infos = super().infos(use_numpy, use_element_label)
        infos['is_input_in_component'] = self.is_input_in_comp
        return infos

    def _plausibility_checks(self) -> None:
        # TODO: Incorporate into Variable? (Lower_bound can not be greater than upper bound
        if np.any(self.relative_minimum > self.relative_maximum):
            raise Exception(self.label_full + ': Take care, that relative_minimum <= relative_maximum!')

        if (
            self.size == CONFIG.modeling.BIG and self.fixed_relative_profile is not None
        ):  # Default Size --> Most likely by accident
            logger.warning(
                f'Flow "{self.label}" has no size assigned, but a "fixed_relative_profile". '
                f'The default size is {CONFIG.modeling.BIG}. As "flow_rate = size * fixed_relative_profile", '
                f'the resulting flow_rate will be very high. To fix this, assign a size to the Flow {self}.'
            )

    @property
    def label_full(self) -> str:
        # Wenn im Erstellungsprozess comp noch nicht bekannt:
        comp_label = 'unknownComp' if self.comp is None else self.comp.label
        return f'{comp_label}__{self.label}'  # z.B. für results_struct (deswegen auch _  statt . dazwischen)

    @property  # Richtung
    def is_input_in_comp(self) -> bool:
        return True if self in self.comp.inputs else False

    @property
    def size_is_fixed(self) -> bool:
        # Wenn kein InvestParameters existiert --> True; Wenn Investparameter, den Wert davon nehmen
        return False if (isinstance(self.size, InvestParameters) and self.size.fixed_size is None) else True

    @property
    def invest_is_optional(self) -> bool:
        # Wenn kein InvestParameters existiert: # Investment ist nicht optional -> Keine Variable --> False
        return False if (isinstance(self.size, InvestParameters) and not self.size.optional) else True


class FlowModel(ElementModel):
    def __init__(self, element: Flow):
        super().__init__(element)
        self.element: Flow = element
        self.flow_rate: Optional[VariableTS] = None
        self.sum_flow_hours: Optional[Variable] = None

        self._on: Optional[OnOffModel] = None
        self._investment: Optional[InvestmentModel] = None

    def do_modeling(self, system_model: SystemModel):
        # eq relative_minimum(t) * size <= flow_rate(t) <= relative_maximum(t) * size
        self.flow_rate = create_variable(
            'flow_rate',
            self,
            system_model.nr_of_time_steps,
            lower_bound=self.absolute_flow_rate_bounds[0] if self.element.on_off_parameters is None else 0,
            upper_bound=self.absolute_flow_rate_bounds[1] if self.element.on_off_parameters is None else None,
            previous_values=self.element.previous_flow_rate,
        )

        # OnOff
        if self.element.on_off_parameters is not None:
            self._on = OnOffModel(
                self.element, self.element.on_off_parameters, [self.flow_rate], [self.absolute_flow_rate_bounds]
            )
            self._on.do_modeling(system_model)
            self.sub_models.append(self._on)

        # Investment
        if isinstance(self.element.size, InvestParameters):
            self._investment = InvestmentModel(
                self.element,
                self.element.size,
                self.flow_rate,
                self.relative_flow_rate_bounds,
                fixed_relative_profile=(None
                                        if self.element.fixed_relative_profile is None
                                        else self.element.fixed_relative_profile.active_data),
                on_variable=self._on.on if self._on is not None else None,
            )
            self._investment.do_modeling(system_model)
            self.sub_models.append(self._investment)

        # sumFLowHours
        self.sum_flow_hours = create_variable(
            'sumFlowHours',
            self,
            1,
            lower_bound=self.element.flow_hours_total_min,
            upper_bound=self.element.flow_hours_total_max,
        )
        eq_sum_flow_hours = create_equation('sumFlowHours', self, 'eq')
        eq_sum_flow_hours.add_summand(self.flow_rate, system_model.dt_in_hours, as_sum=True)
        eq_sum_flow_hours.add_summand(self.sum_flow_hours, -1)

        # Load factor
        self._create_bounds_for_load_factor(system_model)

        # Shares
        self._create_shares(system_model)

    def _create_shares(self, system_model: SystemModel):
        # Arbeitskosten:
        if self.element.effects_per_flow_hour != {}:
            system_model.effect_collection_model.add_share_to_operation(
                name='effects_per_flow_hour',
                element=self.element,
                variable=self.flow_rate,
                effect_values=self.element.effects_per_flow_hour,
                factor=system_model.dt_in_hours,
            )

    def _create_bounds_for_load_factor(self, system_model: SystemModel):
        # TODO: Add Variable load_factor for better evaluation?

        # eq: var_sumFlowHours <= size * dt_tot * load_factor_max
        if self.element.load_factor_max is not None:
            flow_hours_per_size_max = system_model.dt_in_hours_total * self.element.load_factor_max
            eq_load_factor_max = create_equation('load_factor_max', self, 'ineq')
            eq_load_factor_max.add_summand(self.sum_flow_hours, 1)
            # if investment:
            if self._investment is not None:
                eq_load_factor_max.add_summand(self._investment.size, -1 * flow_hours_per_size_max)
            else:
                eq_load_factor_max.add_constant(self.element.size * flow_hours_per_size_max)

        #  eq: size * sum(dt)* load_factor_min <= var_sumFlowHours
        if self.element.load_factor_min is not None:
            flow_hours_per_size_min = system_model.dt_in_hours_total * self.element.load_factor_min
            eq_load_factor_min = create_equation('load_factor_min', self, 'ineq')
            eq_load_factor_min.add_summand(self.sum_flow_hours, -1)
            if self._investment is not None:
                eq_load_factor_min.add_summand(self._investment.size, flow_hours_per_size_min)
            else:
                eq_load_factor_min.add_constant(-1 * self.element.size * flow_hours_per_size_min)

    @property
    def with_investment(self) -> bool:
        """Checks if the element's size is investment-driven."""
        return isinstance(self.element.size, InvestParameters)

    @property
    def absolute_flow_rate_bounds(self) -> Tuple[Numeric, Numeric]:
        """Returns absolute flow rate bounds. Important for OnOffModel"""
        rel_min, rel_max = self.relative_flow_rate_bounds
        size = self.element.size
        if not self.with_investment:
            return rel_min * size, rel_max * size
        if size.fixed_size is not None:
            return rel_min * size.fixed_size, rel_max * size.fixed_size
        return rel_min * size.minimum_size, rel_max * size.maximum_size


    @property
    def relative_flow_rate_bounds(self) -> Tuple[Numeric, Numeric]:
        """Returns relative flow rate bounds."""
        fixed_profile = self.element.fixed_relative_profile
        if fixed_profile is None:
            return self.element.relative_minimum.active_data, self.element.relative_maximum.active_data
        return fixed_profile.active_data, fixed_profile.active_data


class BusModel(ElementModel):
    def __init__(self, element: Bus):
        super().__init__(element)
        self.element: Bus
        self.excess_input: Optional[VariableTS] = None
        self.excess_output: Optional[VariableTS] = None

    def do_modeling(self, system_model: SystemModel) -> None:
        self.element: Bus
        # inputs = outputs
        eq_bus_balance = create_equation('busBalance', self)
        for flow in self.element.inputs:
            eq_bus_balance.add_summand(flow.model.flow_rate, 1)
        for flow in self.element.outputs:
            eq_bus_balance.add_summand(flow.model.flow_rate, -1)

        # Fehlerplus/-minus:
        if self.element.with_excess:
            excess_penalty = np.multiply(
                system_model.dt_in_hours, self.element.excess_penalty_per_flow_hour.active_data
            )
            self.excess_input = create_variable('excess_input', self, system_model.nr_of_time_steps, lower_bound=0)
            self.excess_output = create_variable('excess_output', self, system_model.nr_of_time_steps, lower_bound=0)

            eq_bus_balance.add_summand(self.excess_output, -1)
            eq_bus_balance.add_summand(self.excess_input, 1)

            fx_collection = system_model.effect_collection_model

            fx_collection.add_share_to_penalty(
                f'{self.element.label_full}__excess_input', self.excess_input, excess_penalty
            )
            fx_collection.add_share_to_penalty(
                f'{self.element.label_full}__excess_output', self.excess_output, excess_penalty
            )


class ComponentModel(ElementModel):
    def __init__(self, element: Component):
        super().__init__(element)
        self.element: Component = element
        self._on: Optional[OnOffModel] = None

    def do_modeling(self, system_model: SystemModel):
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

        self.sub_models.extend([flow.create_model() for flow in all_flows])
        for sub_model in self.sub_models:
            sub_model.do_modeling(system_model)

        if self.element.on_off_parameters:
            flow_rates: List[VariableTS] = [flow.model.flow_rate for flow in all_flows]
            bounds: List[Tuple[Numeric, Numeric]] = [flow.model.absolute_flow_rate_bounds for flow in all_flows]
            self._on = OnOffModel(self.element, self.element.on_off_parameters, flow_rates, bounds)
            self.sub_models.append(self._on)
            self._on.do_modeling(system_model)

        if self.element.prevent_simultaneous_flows:
            # Simultanious Useage --> Only One FLow is On at a time, but needs a Binary for every flow
            on_variables = [flow.model._on.on for flow in self.element.prevent_simultaneous_flows]
            simultaneous_use = PreventSimultaneousUsageModel(self.element, on_variables)
            self.sub_models.append(simultaneous_use)
            simultaneous_use.do_modeling(system_model)
