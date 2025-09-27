"""
This module contains the basic elements of the flixopt framework.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np

from .config import CONFIG
from .core import NumericData, NumericDataTS, PlausibilityError, Scalar
from .features import InvestmentModel, OnOffModel, PreventSimultaneousUsageModel
from .interface import InvestParameters, OnOffParameters
from .structure import Element, ElementModel, SystemModel, register_class_for_io

if TYPE_CHECKING:
    import linopy

    from .effects import EffectValuesUser
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


@register_class_for_io
class Component(Element):
    """
    Base class for all system components that transform, convert, or process flows.

    Components are the active elements in energy systems that define how input and output
    Flows interact with each other. They represent equipment, processes, or logical
    operations that transform energy or materials between different states, carriers,
    or locations.

    Components serve as connection points between Buses through their associated Flows,
    enabling the modeling of complex energy system topologies and operational constraints.

    Args:
        label: The label of the Element. Used to identify it in the FlowSystem.
        inputs: list of input Flows feeding into the component. These represent
            energy/material consumption by the component.
        outputs: list of output Flows leaving the component. These represent
            energy/material production by the component.
        on_off_parameters: Defines binary operation constraints and costs when the
            component has discrete on/off states. Creates binary variables for all
            connected Flows. For better performance, prefer defining OnOffParameters
            on individual Flows when possible.
        prevent_simultaneous_flows: list of Flows that cannot be active simultaneously.
            Creates binary variables to enforce mutual exclusivity. Use sparingly as
            it increases computational complexity.
        meta_data: Used to store additional information. Not used internally but saved
            in results. Only use Python native types.

    Note:
        Component operational state is determined by its connected Flows:
        - Component is "on" if ANY of its Flows is active (flow_rate > 0)
        - Component is "off" only when ALL Flows are inactive (flow_rate = 0)

        Binary variables and constraints:
        - on_off_parameters creates binary variables for ALL connected Flows
        - prevent_simultaneous_flows creates binary variables for specified Flows
        - For better computational performance, prefer Flow-level OnOffParameters

        Component is an abstract base class. In practice, use specialized subclasses:
        - LinearConverter: Linear input/output relationships
        - Storage: Temporal energy/material storage
        - Transmission: Transport between locations
        - Source/Sink: System boundaries

    """

    def __init__(
        self,
        label: str,
        inputs: list[Flow] | None = None,
        outputs: list[Flow] | None = None,
        on_off_parameters: OnOffParameters | None = None,
        prevent_simultaneous_flows: list[Flow] | None = None,
        meta_data: dict | None = None,
    ):
        super().__init__(label, meta_data=meta_data)
        self.inputs: list[Flow] = inputs or []
        self.outputs: list[Flow] = outputs or []
        self._check_unique_flow_labels()
        self.on_off_parameters = on_off_parameters
        self.prevent_simultaneous_flows: list[Flow] = prevent_simultaneous_flows or []

        self.flows: dict[str, Flow] = {flow.label: flow for flow in self.inputs + self.outputs}

    def create_model(self, model: SystemModel) -> ComponentModel:
        self._plausibility_checks()
        self.model = ComponentModel(model, self)
        return self.model

    def transform_data(self, flow_system: FlowSystem) -> None:
        if self.on_off_parameters is not None:
            self.on_off_parameters.transform_data(flow_system, self.label_full)

    def infos(self, use_numpy=True, use_element_label: bool = False) -> dict:
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
    Buses represent nodal balances between flow rates, serving as connection points.

    A Bus enforces energy or material balance constraints where the sum of all incoming
    flows must equal the sum of all outgoing flows at each time step. Buses represent
    physical or logical connection points for energy carriers (electricity, heat, gas)
    or material flows between different Components.

    Args:
        label: The label of the Element. Used to identify it in the FlowSystem.
        penalty_of_input_deficit: Penalty costs for bus balance violations. Per flow hour of not extra input.
            When None, no input deficit is allowed (hard constraint).
        penalty_of_output_deficit: Penalty costs for bus balance violations. Per flow hour of extra output.
            When None, no output deficit is allowed (hard constraint).
        meta_data: Used to store additional information. Not used internally but saved
            in results. Only use Python native types.

    Examples:
        Electrical bus with strict balance:

        ```python
        electricity_bus = Bus(
            label='main_electrical_bus',
            excess_penalty_per_flow_hour=None,  # No imbalance allowed
        )
        ```

        Heat network with penalty for imbalances:

        ```python
        heat_network = Bus(
            label='district_heating_network',
            excess_penalty_per_flow_hour=1000,  # €1000/MWh penalty for imbalance
        )
        ```

        Material flow with time-varying penalties:

        ```python
        material_hub = Bus(
            label='material_processing_hub',
            excess_penalty_per_flow_hour=waste_disposal_costs,  # Time series
        )
        ```

    Note:
        The bus balance equation enforced is: Σ(inflows) = Σ(outflows) + excess - deficit

        When excess_penalty_per_flow_hour is None, excess and deficit are forced to zero.
        When a penalty cost is specified, the optimization can choose to violate the
        balance if economically beneficial, paying the penalty.
        The penalty is added to the objective directly.

        Empty `inputs` and `outputs` lists are initialized and populated automatically
        by the FlowSystem during system setup.
    """

    def __init__(
        self,
        label: str,
        excess_penalty_per_flow_hour: NumericData | NumericDataTS | None = None,
        penalty_of_input_deficit: NumericData | NumericDataTS | None = 1e5,
        penalty_of_output_deficit: NumericData | NumericDataTS | None = 1e5,
        meta_data: dict | None = None,
    ):
        super().__init__(label, meta_data=meta_data)
        self.inputs: list[Flow] = []
        self.outputs: list[Flow] = []

        if excess_penalty_per_flow_hour is not None:
            if penalty_of_input_deficit is not None or penalty_of_output_deficit is not None:
                raise ValueError(
                    'Deprectated "excess_penalty_per_flow_hour" cannnot be used together with replacement '
                    '"penalty_of_input_deficit" and "penalty_of_output_deficit" parameters'
                )
            self.excess_penalty_per_flow_hour = excess_penalty_per_flow_hour
        else:
            self.penalty_of_input_deficit = penalty_of_input_deficit
            self.penalty_of_output_deficit = penalty_of_output_deficit

    def create_model(self, model: SystemModel) -> BusModel:
        self._plausibility_checks()
        self.model = BusModel(model, self)
        return self.model

    def transform_data(self, flow_system: FlowSystem):
        self.penalty_of_input_deficit = flow_system.create_time_series(
            f'{self.label_full}|penalty_of_input_deficit', self.penalty_of_input_deficit
        )
        self.penalty_of_output_deficit = flow_system.create_time_series(
            f'{self.label_full}|penalty_of_output_deficit', self.penalty_of_output_deficit
        )

    def _plausibility_checks(self) -> None:
        if self.penalty_of_output_deficit is not None:
            zero_penalty = np.all(np.equal(self.penalty_of_output_deficit, 0))
            if zero_penalty:
                logger.warning(f'In Bus {self.label}, "the penalty_of_output_deficit" is 0. Use "None" or a value > 0.')
        if self.penalty_of_input_deficit is not None:
            zero_penalty = np.all(np.equal(self.penalty_of_input_deficit, 0))
            if zero_penalty:
                logger.warning(f'In Bus {self.label}, the "penalty_of_input_deficit" is 0. Use "None" or a value > 0.')

    @property
    def with_excess(self) -> bool:
        return False if self.excess_penalty_per_flow_hour is None else True

    @property
    def excess_penalty_per_flow_hour(self):
        warnings.warn(
            'The property "excess_penalty_per_flow_hour" is deprecated. Use "penalty_per_flow_hour_excess" instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.penalty_of_input_deficit

    @excess_penalty_per_flow_hour.setter
    def excess_penalty_per_flow_hour(self, value):
        warnings.warn(
            'The property "excess_penalty_per_flow_hour" is deprecated. Use "penalty_per_flow_hour_excess" instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        self.penalty_of_input_deficit = value
        self.penalty_of_output_deficit = value


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
    """Define a directed flow of energy or material between bus and component.

    A Flow represents the transfer of energy (electricity, heat, fuel) or material
    between a Bus and a Component in a specific direction. The flow rate is the
    primary optimization variable, with constraints and costs defined through
    various parameters. Flows can have fixed or variable sizes, operational
    constraints, and complex on/off behavior.

    Key Concepts:
        **Flow Rate**: The instantaneous rate of energy/material transfer (optimization variable) [kW, m³/h, kg/h]
        **Flow Hours**: Amount of energy/material transferred per timestep. [kWh, m³, kg]
        **Flow Size**: The maximum capacity or nominal rating of the flow [kW, m³/h, kg/h]
        **Relative Bounds**: Flow rate limits expressed as fractions of flow size

    Integration with Parameter Classes:
        - **InvestParameters**: Used for `size` when flow Size is an investment decision
        - **OnOffParameters**: Used for `on_off_parameters` when flow has discrete states

    Args:
        label: Unique identifier for the flow within its component.
            The full label combines component and flow labels.
        bus: Label of the bus this flow connects to. Must match a bus in the FlowSystem.
        size: Flow capacity or nominal rating. Can be:
            - Scalar value for fixed capacity
            - InvestParameters for investment-based sizing decisions
            - None to use large default value (CONFIG.modeling.BIG)
        relative_minimum: Minimum flow rate as fraction of size.
            Example: 0.2 means flow cannot go below 20% of rated capacity.
        relative_maximum: Maximum flow rate as fraction of size (typically 1.0).
            Values >1.0 allow temporary overload operation.
        load_factor_min: Minimum average utilization over the time horizon (0-1).
            Calculated as total flow hours divided by (size × total time).
        load_factor_max: Maximum average utilization over the time horizon (0-1).
            Useful for equipment duty cycle limits or maintenance scheduling.
        effects_per_flow_hour: Operational costs and impacts per unit of flow-time.
            Dictionary mapping effect names to unit costs (e.g., fuel costs, emissions).
        on_off_parameters: Binary operation constraints using OnOffParameters.
            Enables modeling of startup costs, minimum run times, cycling limits.
            Only relevant when relative_minimum > 0 or discrete operation is required.
        flow_hours_total_max: Maximum cumulative flow-hours over time horizon.
            Alternative to load_factor_max for absolute energy/material limits.
        flow_hours_total_min: Minimum cumulative flow-hours over time horizon.
            Alternative to load_factor_min for contractual or operational requirements.
        fixed_relative_profile: Predetermined flow pattern as fraction of size.
            When specified, flow rate becomes: size × fixed_relative_profile(t).
            Used for: demand profiles, renewable generation, fixed schedules.
        previous_flow_rate: Initial flow state for startup/shutdown dynamics.
            Used with on_off_parameters to determine initial on/off status.
            If None, assumes flow was off in previous time period.
        meta_data: Additional information stored with results but not used in optimization.
            Must contain only Python native types (dict, list, str, int, float, bool).

    Examples:
        Basic power flow with fixed capacity:

        ```python
        generator_output = Flow(
            label='electricity_out',
            bus='electricity_grid',
            size=100,  # 100 MW capacity
            relative_minimum=0.4,  # Cannot operate below 40 MW
            effects_per_flow_hour={'fuel_cost': 45, 'co2_emissions': 0.8},
        )
        ```

        Investment decision for battery capacity:

        ```python
        battery_flow = Flow(
            label='electricity_storage',
            bus='electricity_grid',
            size=InvestParameters(
                minimum_size=10,  # Minimum 10 MWh
                maximum_size=100,  # Maximum 100 MWh
                specific_effects={'cost': 150_000},  # €150k/MWh annualized
            ),
        )
        ```

        Heat pump with startup costs and minimum run times:

        ```python
        heat_pump = Flow(
            label='heat_output',
            bus='heating_network',
            size=50,  # 50 kW thermal
            relative_minimum=0.3,  # Minimum 15 kW output when on
            effects_per_flow_hour={'electricity_cost': 25, 'maintenance': 2},
            on_off_parameters=OnOffParameters(
                effects_per_switch_on={'startup_cost': 100, 'wear': 0.1},
                consecutive_on_hours_min=2,  # Must run at least 2 hours
                consecutive_off_hours_min=1,  # Must stay off at least 1 hour
                switch_on_total_max=200,  # Maximum 200 starts per year
            ),
        )
        ```

        Fixed renewable generation profile:

        ```python
        solar_generation = Flow(
            label='solar_power',
            bus='electricity_grid',
            size=25,  # 25 MW installed capacity
            fixed_relative_profile=np.array([0, 0.1, 0.4, 0.8, 0.9, 0.7, 0.3, 0.1, 0]),
            effects_per_flow_hour={'maintenance_costs': 5},  # €5/MWh maintenance
        )
        ```

        Industrial process with annual utilization limits:

        ```python
        production_line = Flow(
            label='product_output',
            bus='product_market',
            size=1000,  # 1000 units/hour capacity
            load_factor_min=0.6,  # Must achieve 60% annual utilization
            load_factor_max=0.85,  # Cannot exceed 85% for maintenance
            effects_per_flow_hour={'variable_cost': 12, 'quality_control': 0.5},
        )
        ```

    Design Considerations:
        **Size vs Load Factors**: Use `flow_hours_total_min/max` for absolute limits,
        `load_factor_min/max` for utilization-based constraints.

        **Relative Bounds**: Set `relative_minimum > 0` only when equipment cannot
        operate below that level. Use `on_off_parameters` for discrete on/off behavior.

        **Fixed Profiles**: Use `fixed_relative_profile` for known exact patterns,
        `relative_maximum` for upper bounds on optimization variables.

    Notes:
        - Default size (CONFIG.modeling.BIG) is used when size=None
        - list inputs for previous_flow_rate are converted to NumPy arrays
        - Flow direction is determined by component input/output designation

    Deprecated:
        Passing Bus objects to `bus` parameter. Use bus label strings instead.

    """

    def __init__(
        self,
        label: str,
        bus: str,
        size: Scalar | InvestParameters | None = None,
        fixed_relative_profile: NumericDataTS | None = None,
        relative_minimum: NumericDataTS = 0,
        relative_maximum: NumericDataTS = 1,
        effects_per_flow_hour: EffectValuesUser | None = None,
        on_off_parameters: OnOffParameters | None = None,
        flow_hours_total_max: Scalar | None = None,
        flow_hours_total_min: Scalar | None = None,
        load_factor_min: Scalar | None = None,
        load_factor_max: Scalar | None = None,
        previous_flow_rate: NumericData | None = None,
        meta_data: dict | None = None,
    ):
        super().__init__(label, meta_data=meta_data)
        self.size = CONFIG.modeling.BIG if size is None else size
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
        self.is_input_in_component: bool | None = None
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

    def create_model(self, model: SystemModel) -> FlowModel:
        self._plausibility_checks()
        self.model = FlowModel(model, self)
        return self.model

    def transform_data(self, flow_system: FlowSystem):
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

    def infos(self, use_numpy: bool = True, use_element_label: bool = False) -> dict:
        infos = super().infos(use_numpy, use_element_label)
        infos['is_input_in_component'] = self.is_input_in_component
        return infos

    def to_dict(self) -> dict:
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
        self.flow_rate: linopy.Variable | None = None
        self.total_flow_hours: linopy.Variable | None = None

        self.on_off: OnOffModel | None = None
        self._investment: InvestmentModel | None = None

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
                    relative_bounds_of_defining_variable=(
                        self.flow_rate_lower_bound_relative,
                        self.flow_rate_upper_bound_relative,
                    ),
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
    def flow_rate_bounds_on(self) -> tuple[NumericData, NumericData]:
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
        """Returns the upper bound of the flow_rate relative to its size"""
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
        self.input_deficit: linopy.Variable | None = None
        self.output_deficit: linopy.Variable | None = None

    def do_modeling(self) -> None:
        # inputs == outputs
        for flow in self.element.inputs + self.element.outputs:
            self.add(flow.model.flow_rate, flow.label_full)
        inputs = sum([flow.model.flow_rate for flow in self.element.inputs])
        outputs = sum([flow.model.flow_rate for flow in self.element.outputs])
        eq_bus_balance = self.add(self._model.add_constraints(inputs == outputs, name=f'{self.label_full}|balance'))

        # Fehlerplus/-minus:
        if self.element.penalty_of_input_deficit is not None:
            input_deficit_penalty = np.multiply(
                self._model.hours_per_step, self.element.penalty_of_input_deficit.active_data
            )
            self.input_deficit = self.add(
                self._model.add_variables(lower=0, coords=self._model.coords, name=f'{self.label_full}|input_deficit'),
                'input_deficit',
            )

            eq_bus_balance.lhs += self.input_deficit
            self._model.effects.add_share_to_penalty(
                self.label_of_element, (self.input_deficit * input_deficit_penalty).sum()
            )

        if self.element.penalty_of_output_deficit is not None:
            output_deficit_penalty = np.multiply(
                self._model.hours_per_step, self.element.penalty_of_output_deficit.active_data
            )
            self.output_deficit = self.add(
                self._model.add_variables(lower=0, coords=self._model.coords, name=f'{self.label_full}|output_deficit'),
                'output_deficit',
            )

            eq_bus_balance.lhs -= self.output_deficit
            self._model.effects.add_share_to_penalty(
                self.label_of_element, (self.output_deficit * output_deficit_penalty).sum()
            )

    def results_structure(self):
        inputs = [flow.model.flow_rate.name for flow in self.element.inputs]
        outputs = [flow.model.flow_rate.name for flow in self.element.outputs]
        if self.input_deficit is not None:
            inputs.append(self.input_deficit.name)
        if self.output_deficit is not None:
            outputs.append(self.output_deficit.name)
        return {**super().results_structure(), 'inputs': inputs, 'outputs': outputs}


class ComponentModel(ElementModel):
    def __init__(self, model: SystemModel, element: Component):
        super().__init__(model, element)
        self.element: Component = element
        self.on_off: OnOffModel | None = None

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
