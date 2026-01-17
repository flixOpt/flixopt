"""
This module contains the basic elements of the flixopt framework.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from . import io as fx_io
from .config import CONFIG
from .core import PlausibilityError
from .features import InvestmentModel, StatusModel
from .interface import InvestParameters, StatusParameters
from .modeling import BoundingPatterns, ModelingPrimitives, ModelingUtilitiesAbstract
from .structure import (
    Element,
    ElementModel,
    FlowSystemModel,
    VariableCategory,
    register_class_for_io,
)
from .vectorized import ConstraintResult, ConstraintSpec, VariableHandle, VariableSpec

if TYPE_CHECKING:
    import linopy

    from .types import (
        Effect_TPS,
        Numeric_PS,
        Numeric_S,
        Numeric_TPS,
        Scalar,
    )

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
        status_parameters: Defines binary operation constraints and costs when the
            component has discrete active/inactive states. Creates binary variables for all
            connected Flows. For better performance, prefer defining StatusParameters
            on individual Flows when possible.
        prevent_simultaneous_flows: list of Flows that cannot be active simultaneously.
            Creates binary variables to enforce mutual exclusivity. Use sparingly as
            it increases computational complexity.
        meta_data: Used to store additional information. Not used internally but saved
            in results. Only use Python native types.

    Note:
        Component operational state is determined by its connected Flows:
        - Component is "active" if ANY of its Flows is active (flow_rate > 0)
        - Component is "inactive" only when ALL Flows are inactive (flow_rate = 0)

        Binary variables and constraints:
        - status_parameters creates binary variables for ALL connected Flows
        - prevent_simultaneous_flows creates binary variables for specified Flows
        - For better computational performance, prefer Flow-level StatusParameters

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
        status_parameters: StatusParameters | None = None,
        prevent_simultaneous_flows: list[Flow] | None = None,
        meta_data: dict | None = None,
        color: str | None = None,
    ):
        super().__init__(label, meta_data=meta_data, color=color)
        self.inputs: list[Flow] = inputs or []
        self.outputs: list[Flow] = outputs or []
        self.status_parameters = status_parameters
        self.prevent_simultaneous_flows: list[Flow] = prevent_simultaneous_flows or []

        self._check_unique_flow_labels()
        self._connect_flows()

        self.flows: dict[str, Flow] = {flow.label: flow for flow in self.inputs + self.outputs}

    def create_model(self, model: FlowSystemModel) -> ComponentModel:
        self._plausibility_checks()
        self.submodel = ComponentModel(model, self)
        return self.submodel

    def link_to_flow_system(self, flow_system, prefix: str = '') -> None:
        """Propagate flow_system reference to nested Interface objects and flows.

        Elements use their label_full as prefix by default, ignoring the passed prefix.
        """
        super().link_to_flow_system(flow_system, self.label_full)
        if self.status_parameters is not None:
            self.status_parameters.link_to_flow_system(flow_system, self._sub_prefix('status_parameters'))
        for flow in self.inputs + self.outputs:
            flow.link_to_flow_system(flow_system)

    def transform_data(self) -> None:
        if self.status_parameters is not None:
            self.status_parameters.transform_data()

        for flow in self.inputs + self.outputs:
            flow.transform_data()

    def _check_unique_flow_labels(self):
        all_flow_labels = [flow.label for flow in self.inputs + self.outputs]

        if len(set(all_flow_labels)) != len(all_flow_labels):
            duplicates = {label for label in all_flow_labels if all_flow_labels.count(label) > 1}
            raise ValueError(f'Flow names must be unique! "{self.label_full}" got 2 or more of: {duplicates}')

    def _plausibility_checks(self) -> None:
        self._check_unique_flow_labels()

        # Component with status_parameters requires all flows to have sizes set
        # (status_parameters are propagated to flows in _do_modeling, which need sizes for big-M constraints)
        if self.status_parameters is not None:
            flows_without_size = [flow.label for flow in self.inputs + self.outputs if flow.size is None]
            if flows_without_size:
                raise PlausibilityError(
                    f'Component "{self.label_full}" has status_parameters, but the following flows have no size: '
                    f'{flows_without_size}. All flows need explicit sizes when the component uses status_parameters '
                    f'(required for big-M constraints).'
                )

    def _connect_flows(self):
        # Inputs
        for flow in self.inputs:
            if flow.component not in ('UnknownComponent', self.label_full):
                raise ValueError(
                    f'Flow "{flow.label}" already assigned to component "{flow.component}". '
                    f'Cannot attach to "{self.label_full}".'
                )
            flow.component = self.label_full
            flow.is_input_in_component = True
        # Outputs
        for flow in self.outputs:
            if flow.component not in ('UnknownComponent', self.label_full):
                raise ValueError(
                    f'Flow "{flow.label}" already assigned to component "{flow.component}". '
                    f'Cannot attach to "{self.label_full}".'
                )
            flow.component = self.label_full
            flow.is_input_in_component = False

        # Validate prevent_simultaneous_flows: only allow local flows
        if self.prevent_simultaneous_flows:
            # Deduplicate while preserving order
            seen = set()
            self.prevent_simultaneous_flows = [
                f for f in self.prevent_simultaneous_flows if id(f) not in seen and not seen.add(id(f))
            ]
            local = set(self.inputs + self.outputs)
            foreign = [f for f in self.prevent_simultaneous_flows if f not in local]
            if foreign:
                names = ', '.join(f.label_full for f in foreign)
                raise ValueError(
                    f'prevent_simultaneous_flows for "{self.label_full}" must reference its own flows. '
                    f'Foreign flows detected: {names}'
                )

    def __repr__(self) -> str:
        """Return string representation with flow information."""
        return fx_io.build_repr_from_init(
            self, excluded_params={'self', 'label', 'inputs', 'outputs', 'kwargs'}, skip_default_size=True
        ) + fx_io.format_flow_details(self)


@register_class_for_io
class Bus(Element):
    """
    Buses represent nodal balances between flow rates, serving as connection points.

    A Bus enforces energy or material balance constraints where the sum of all incoming
    flows must equal the sum of all outgoing flows at each time step. Buses represent
    physical or logical connection points for energy carriers (electricity, heat, gas)
    or material flows between different Components.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/elements/Bus/>

    Args:
        label: The label of the Element. Used to identify it in the FlowSystem.
        carrier: Name of the energy/material carrier type (e.g., 'electricity', 'heat', 'gas').
            Carriers are registered via ``flow_system.add_carrier()`` or available as
            predefined defaults in CONFIG.Carriers. Used for automatic color assignment in plots.
        imbalance_penalty_per_flow_hour: Penalty costs for bus balance violations.
            When None (default), no imbalance is allowed (hard constraint). When set to a
            value > 0, allows bus imbalances at penalty cost.
        meta_data: Used to store additional information. Not used internally but saved
            in results. Only use Python native types.

    Examples:
        Using predefined carrier names:

        ```python
        electricity_bus = Bus(label='main_grid', carrier='electricity')
        heat_bus = Bus(label='district_heating', carrier='heat')
        ```

        Registering custom carriers on FlowSystem:

        ```python
        import flixopt as fx

        fs = fx.FlowSystem(timesteps)
        fs.add_carrier(fx.Carrier('biogas', '#228B22', 'kW'))
        biogas_bus = fx.Bus(label='biogas_network', carrier='biogas')
        ```

        Heat network with penalty for imbalances:

        ```python
        heat_bus = Bus(
            label='district_heating',
            carrier='heat',
            imbalance_penalty_per_flow_hour=1000,
        )
        ```

    Note:
        The bus balance equation enforced is: Σ(inflows) + virtual_supply = Σ(outflows) + virtual_demand

        When imbalance_penalty_per_flow_hour is None, virtual_supply and virtual_demand are forced to zero.
        When a penalty cost is specified, the optimization can choose to violate the
        balance if economically beneficial, paying the penalty.
        The penalty is added to the objective directly.

        Empty `inputs` and `outputs` lists are initialized and populated automatically
        by the FlowSystem during system setup.
    """

    submodel: BusModel | None

    def __init__(
        self,
        label: str,
        carrier: str | None = None,
        imbalance_penalty_per_flow_hour: Numeric_TPS | None = None,
        meta_data: dict | None = None,
        **kwargs,
    ):
        super().__init__(label, meta_data=meta_data)
        imbalance_penalty_per_flow_hour = self._handle_deprecated_kwarg(
            kwargs, 'excess_penalty_per_flow_hour', 'imbalance_penalty_per_flow_hour', imbalance_penalty_per_flow_hour
        )
        self._validate_kwargs(kwargs)
        self.carrier = carrier.lower() if carrier else None  # Store as lowercase string
        self.imbalance_penalty_per_flow_hour = imbalance_penalty_per_flow_hour
        self.inputs: list[Flow] = []
        self.outputs: list[Flow] = []

    def create_model(self, model: FlowSystemModel) -> BusModel:
        self._plausibility_checks()
        self.submodel = BusModel(model, self)
        return self.submodel

    def link_to_flow_system(self, flow_system, prefix: str = '') -> None:
        """Propagate flow_system reference to nested flows.

        Elements use their label_full as prefix by default, ignoring the passed prefix.
        """
        super().link_to_flow_system(flow_system, self.label_full)
        for flow in self.inputs + self.outputs:
            flow.link_to_flow_system(flow_system)

    def transform_data(self) -> None:
        self.imbalance_penalty_per_flow_hour = self._fit_coords(
            f'{self.prefix}|imbalance_penalty_per_flow_hour', self.imbalance_penalty_per_flow_hour
        )

    def _plausibility_checks(self) -> None:
        if self.imbalance_penalty_per_flow_hour is not None:
            zero_penalty = np.all(np.equal(self.imbalance_penalty_per_flow_hour, 0))
            if zero_penalty:
                logger.warning(
                    f'In Bus {self.label_full}, the imbalance_penalty_per_flow_hour is 0. Use "None" or a value > 0.'
                )
        if len(self.inputs) == 0 and len(self.outputs) == 0:
            raise ValueError(
                f'Bus "{self.label_full}" has no Flows connected to it. Please remove it from the FlowSystem'
            )

    @property
    def allows_imbalance(self) -> bool:
        return self.imbalance_penalty_per_flow_hour is not None

    def __repr__(self) -> str:
        """Return string representation."""
        return super().__repr__() + fx_io.format_flow_details(self)


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
    constraints, and complex on/inactive behavior.

    Key Concepts:
        **Flow Rate**: The instantaneous rate of energy/material transfer (optimization variable) [kW, m³/h, kg/h]
        **Flow Hours**: Amount of energy/material transferred per timestep. [kWh, m³, kg]
        **Flow Size**: The maximum capacity or nominal rating of the flow [kW, m³/h, kg/h]
        **Relative Bounds**: Flow rate limits expressed as fractions of flow size

    Integration with Parameter Classes:
        - **InvestParameters**: Used for `size` when flow Size is an investment decision
        - **StatusParameters**: Used for `status_parameters` when flow has discrete states

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/elements/Flow/>

    Args:
        label: Unique flow identifier within its component.
        bus: Bus label this flow connects to.
        size: Flow capacity. Scalar, InvestParameters, or None (unbounded).
        relative_minimum: Minimum flow rate as fraction of size (0-1). Default: 0.
        relative_maximum: Maximum flow rate as fraction of size. Default: 1.
        load_factor_min: Minimum average utilization (0-1). Default: 0.
        load_factor_max: Maximum average utilization (0-1). Default: 1.
        effects_per_flow_hour: Operational costs/impacts per flow-hour.
            Dict mapping effect names to values (e.g., {'cost': 45, 'CO2': 0.8}).
        status_parameters: Binary operation constraints (StatusParameters). Default: None.
        flow_hours_max: Maximum cumulative flow-hours per period. Alternative to load_factor_max.
        flow_hours_min: Minimum cumulative flow-hours per period. Alternative to load_factor_min.
        flow_hours_max_over_periods: Maximum weighted sum of flow-hours across ALL periods.
            Weighted by FlowSystem period weights.
        flow_hours_min_over_periods: Minimum weighted sum of flow-hours across ALL periods.
            Weighted by FlowSystem period weights.
        fixed_relative_profile: Predetermined pattern as fraction of size.
            Flow rate = size × fixed_relative_profile(t).
        previous_flow_rate: Initial flow state for active/inactive status at model start. Default: None (inactive).
        meta_data: Additional info stored in results. Python native types only.

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
            relative_minimum=0.3,  # Minimum 15 kW output when active
            effects_per_flow_hour={'electricity_cost': 25, 'maintenance': 2},
            status_parameters=StatusParameters(
                effects_per_startup={'startup_cost': 100, 'wear': 0.1},
                min_uptime=2,  # Must run at least 2 hours
                min_downtime=1,  # Must stay inactive at least 1 hour
                startup_limit=200,  # Maximum 200 starts per period
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
        **Size vs Load Factors**: Use `flow_hours_min/max` for absolute limits per period,
        `load_factor_min/max` for utilization-based constraints, or `flow_hours_min/max_over_periods` for
        limits across all periods.

        **Relative Bounds**: Set `relative_minimum > 0` only when equipment cannot
        operate below that level. Use `status_parameters` for discrete active/inactive behavior.

        **Fixed Profiles**: Use `fixed_relative_profile` for known exact patterns,
        `relative_maximum` for upper bounds on optimization variables.

    Notes:
        - size=None means unbounded (no capacity constraint)
        - size must be set when using status_parameters or fixed_relative_profile
        - list inputs for previous_flow_rate are converted to NumPy arrays
        - Flow direction is determined by component input/output designation

    Deprecated:
        Passing Bus objects to `bus` parameter. Use bus label strings instead.

    """

    submodel: FlowModel | None

    def __init__(
        self,
        label: str,
        bus: str,
        size: Numeric_PS | InvestParameters | None = None,
        fixed_relative_profile: Numeric_TPS | None = None,
        relative_minimum: Numeric_TPS = 0,
        relative_maximum: Numeric_TPS = 1,
        effects_per_flow_hour: Effect_TPS | Numeric_TPS | None = None,
        status_parameters: StatusParameters | None = None,
        flow_hours_max: Numeric_PS | None = None,
        flow_hours_min: Numeric_PS | None = None,
        flow_hours_max_over_periods: Numeric_S | None = None,
        flow_hours_min_over_periods: Numeric_S | None = None,
        load_factor_min: Numeric_PS | None = None,
        load_factor_max: Numeric_PS | None = None,
        previous_flow_rate: Scalar | list[Scalar] | None = None,
        meta_data: dict | None = None,
    ):
        super().__init__(label, meta_data=meta_data)
        self.size = size
        self.relative_minimum = relative_minimum
        self.relative_maximum = relative_maximum
        self.fixed_relative_profile = fixed_relative_profile

        self.load_factor_min = load_factor_min
        self.load_factor_max = load_factor_max

        # self.positive_gradient = TimeSeries('positive_gradient', positive_gradient, self)
        self.effects_per_flow_hour = effects_per_flow_hour if effects_per_flow_hour is not None else {}
        self.flow_hours_max = flow_hours_max
        self.flow_hours_min = flow_hours_min
        self.flow_hours_max_over_periods = flow_hours_max_over_periods
        self.flow_hours_min_over_periods = flow_hours_min_over_periods
        self.status_parameters = status_parameters

        self.previous_flow_rate = previous_flow_rate

        self.component: str = 'UnknownComponent'
        self.is_input_in_component: bool | None = None
        if isinstance(bus, Bus):
            raise TypeError(
                f'Bus {bus.label} is passed as a Bus object to Flow {self.label}. '
                f'This is no longer supported. Add the Bus to the FlowSystem and pass its label (string) to the Flow.'
            )
        self.bus = bus

    def create_model(self, model: FlowSystemModel) -> FlowModel:
        self._plausibility_checks()
        self.submodel = FlowModel(model, self)
        return self.submodel

    def link_to_flow_system(self, flow_system, prefix: str = '') -> None:
        """Propagate flow_system reference to nested Interface objects.

        Elements use their label_full as prefix by default, ignoring the passed prefix.
        """
        super().link_to_flow_system(flow_system, self.label_full)
        if self.status_parameters is not None:
            self.status_parameters.link_to_flow_system(flow_system, self._sub_prefix('status_parameters'))
        if isinstance(self.size, InvestParameters):
            self.size.link_to_flow_system(flow_system, self._sub_prefix('InvestParameters'))

    def transform_data(self) -> None:
        self.relative_minimum = self._fit_coords(f'{self.prefix}|relative_minimum', self.relative_minimum)
        self.relative_maximum = self._fit_coords(f'{self.prefix}|relative_maximum', self.relative_maximum)
        self.fixed_relative_profile = self._fit_coords(
            f'{self.prefix}|fixed_relative_profile', self.fixed_relative_profile
        )
        self.effects_per_flow_hour = self._fit_effect_coords(self.prefix, self.effects_per_flow_hour, 'per_flow_hour')
        self.flow_hours_max = self._fit_coords(
            f'{self.prefix}|flow_hours_max', self.flow_hours_max, dims=['period', 'scenario']
        )
        self.flow_hours_min = self._fit_coords(
            f'{self.prefix}|flow_hours_min', self.flow_hours_min, dims=['period', 'scenario']
        )
        self.flow_hours_max_over_periods = self._fit_coords(
            f'{self.prefix}|flow_hours_max_over_periods', self.flow_hours_max_over_periods, dims=['scenario']
        )
        self.flow_hours_min_over_periods = self._fit_coords(
            f'{self.prefix}|flow_hours_min_over_periods', self.flow_hours_min_over_periods, dims=['scenario']
        )
        self.load_factor_max = self._fit_coords(
            f'{self.prefix}|load_factor_max', self.load_factor_max, dims=['period', 'scenario']
        )
        self.load_factor_min = self._fit_coords(
            f'{self.prefix}|load_factor_min', self.load_factor_min, dims=['period', 'scenario']
        )

        if self.status_parameters is not None:
            self.status_parameters.transform_data()
        if isinstance(self.size, InvestParameters):
            self.size.transform_data()
        elif self.size is not None:
            self.size = self._fit_coords(f'{self.prefix}|size', self.size, dims=['period', 'scenario'])

    def _plausibility_checks(self) -> None:
        # TODO: Incorporate into Variable? (Lower_bound can not be greater than upper bound
        if (self.relative_minimum > self.relative_maximum).any():
            raise PlausibilityError(self.label_full + ': Take care, that relative_minimum <= relative_maximum!')

        # Size is required when using StatusParameters (for big-M constraints)
        if self.status_parameters is not None and self.size is None:
            raise PlausibilityError(
                f'Flow "{self.label_full}" has status_parameters but no size defined. '
                f'A size is required when using status_parameters to bound the flow rate.'
            )

        if self.size is None and self.fixed_relative_profile is not None:
            raise PlausibilityError(
                f'Flow "{self.label_full}" has a fixed_relative_profile but no size defined. '
                f'A size is required because flow_rate = size * fixed_relative_profile.'
            )

        # Size is required when using non-default relative bounds (flow_rate = size * relative_bound)
        if self.size is None and np.any(self.relative_minimum > 0):
            raise PlausibilityError(
                f'Flow "{self.label_full}" has relative_minimum > 0 but no size defined. '
                f'A size is required because the lower bound is size * relative_minimum.'
            )

        if self.size is None and np.any(self.relative_maximum < 1):
            raise PlausibilityError(
                f'Flow "{self.label_full}" has relative_maximum != 1 but no size defined. '
                f'A size is required because the upper bound is size * relative_maximum.'
            )

        # Size is required for load factor constraints (total_flow_hours / size)
        if self.size is None and self.load_factor_min is not None:
            raise PlausibilityError(
                f'Flow "{self.label_full}" has load_factor_min but no size defined. '
                f'A size is required because the constraint is total_flow_hours >= size * load_factor_min * hours.'
            )

        if self.size is None and self.load_factor_max is not None:
            raise PlausibilityError(
                f'Flow "{self.label_full}" has load_factor_max but no size defined. '
                f'A size is required because the constraint is total_flow_hours <= size * load_factor_max * hours.'
            )

        if self.fixed_relative_profile is not None and self.status_parameters is not None:
            logger.warning(
                f'Flow {self.label_full} has both a fixed_relative_profile and status_parameters.'
                f'This will allow the flow to be switched active and inactive, effectively differing from the fixed_flow_rate.'
            )

        if np.any(self.relative_minimum > 0) and self.status_parameters is None:
            logger.warning(
                f'Flow {self.label_full} has a relative_minimum of {self.relative_minimum} and no status_parameters. '
                f'This prevents the Flow from switching inactive (flow_rate = 0). '
                f'Consider using status_parameters to allow the Flow to be switched active and inactive.'
            )

        if self.previous_flow_rate is not None:
            if not any(
                [
                    isinstance(self.previous_flow_rate, np.ndarray) and self.previous_flow_rate.ndim == 1,
                    isinstance(self.previous_flow_rate, (int, float, list)),
                ]
            ):
                raise TypeError(
                    f'previous_flow_rate must be None, a scalar, a list of scalars or a 1D-numpy-array. Got {type(self.previous_flow_rate)}. '
                    f'Different values in different periods or scenarios are not yet supported.'
                )

    @property
    def label_full(self) -> str:
        return f'{self.component}({self.label})'

    @property
    def size_is_fixed(self) -> bool:
        # Wenn kein InvestParameters existiert --> True; Wenn Investparameter, den Wert davon nehmen
        return False if (isinstance(self.size, InvestParameters) and self.size.fixed_size is None) else True

    def _format_invest_params(self, params: InvestParameters) -> str:
        """Format InvestParameters for display."""
        return f'size: {params.format_for_repr()}'


class FlowModel(ElementModel):
    """Mathematical model implementation for Flow elements.

    Creates optimization variables and constraints for flow rate bounds,
    flow-hours tracking, and load factors.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/elements/Flow/>
    """

    element: Flow  # Type hint

    def __init__(self, model: FlowSystemModel, element: Flow):
        super().__init__(model, element)
        self._dce_handles: dict[str, VariableHandle] = {}

    # =========================================================================
    # DCE Pattern: Declaration-Collection-Execution
    # =========================================================================

    def declare_variables(self) -> list[VariableSpec]:
        """Declare variables needed by this Flow for batch creation.

        Returns VariableSpecs that will be collected by VariableRegistry
        and batch-created with other Flows' variables.
        """
        specs = []

        # Main flow rate variable (always needed)
        specs.append(
            VariableSpec(
                category='flow_rate',
                element_id=self.label_full,
                lower=self.absolute_flow_rate_bounds[0],
                upper=self.absolute_flow_rate_bounds[1],
                dims=self._model.temporal_dims,
                var_category=VariableCategory.FLOW_RATE,
            )
        )

        # Status variable (if using status_parameters)
        if self.with_status:
            specs.append(
                VariableSpec(
                    category='status',
                    element_id=self.label_full,
                    binary=True,
                    dims=self._model.temporal_dims,
                    var_category=VariableCategory.STATUS,
                )
            )

        # Total flow hours variable (per period)
        # Bounds from flow_hours_min/max
        specs.append(
            VariableSpec(
                category='total_flow_hours',
                element_id=self.label_full,
                lower=self.element.flow_hours_min if self.element.flow_hours_min is not None else 0,
                upper=self.element.flow_hours_max if self.element.flow_hours_max is not None else np.inf,
                dims=('period', 'scenario'),
                var_category=VariableCategory.TOTAL,
            )
        )

        # Flow hours over periods (if constrained)
        if self.element.flow_hours_min_over_periods is not None or self.element.flow_hours_max_over_periods is not None:
            specs.append(
                VariableSpec(
                    category='flow_hours_over_periods',
                    element_id=self.label_full,
                    lower=self.element.flow_hours_min_over_periods
                    if self.element.flow_hours_min_over_periods is not None
                    else 0,
                    upper=self.element.flow_hours_max_over_periods
                    if self.element.flow_hours_max_over_periods is not None
                    else np.inf,
                    dims=('scenario',),
                    var_category=VariableCategory.TOTAL_OVER_PERIODS,
                )
            )

        return specs

    def declare_constraints(self) -> list[ConstraintSpec]:
        """Declare constraints needed by this Flow for batch creation.

        Returns ConstraintSpecs with build functions that will be called
        after variables are created.
        """
        specs = []

        # Flow rate bounds (depends on status/investment configuration)
        if self.with_status and not self.with_investment:
            # Status-controlled bounds
            specs.append(
                ConstraintSpec(
                    category='flow_rate_ub',
                    element_id=self.label_full,
                    build_fn=self._build_status_upper_bound,
                )
            )
            specs.append(
                ConstraintSpec(
                    category='flow_rate_lb',
                    element_id=self.label_full,
                    build_fn=self._build_status_lower_bound,
                )
            )

        # Total flow hours tracking constraint
        specs.append(
            ConstraintSpec(
                category='total_flow_hours_eq',
                element_id=self.label_full,
                build_fn=self._build_total_flow_hours_tracking,
            )
        )

        # Flow hours over periods tracking (if needed)
        if self.element.flow_hours_min_over_periods is not None or self.element.flow_hours_max_over_periods is not None:
            specs.append(
                ConstraintSpec(
                    category='flow_hours_over_periods_eq',
                    element_id=self.label_full,
                    build_fn=self._build_flow_hours_over_periods_tracking,
                )
            )

        # Load factor constraints
        if self.element.load_factor_max is not None:
            specs.append(
                ConstraintSpec(
                    category='load_factor_max',
                    element_id=self.label_full,
                    build_fn=self._build_load_factor_max,
                )
            )

        if self.element.load_factor_min is not None:
            specs.append(
                ConstraintSpec(
                    category='load_factor_min',
                    element_id=self.label_full,
                    build_fn=self._build_load_factor_min,
                )
            )

        return specs

    def on_variables_created(self, handles: dict[str, VariableHandle]) -> None:
        """Called after batch variable creation with handles to our variables.

        Also creates effects since they need the flow_rate variable.
        """
        self._dce_handles = handles

        # Register the DCE variables in our local registry so properties like self.flow_rate work
        for category, handle in handles.items():
            self.register_variable(handle.variable, category)

        # Now create effects (needs flow_rate to be accessible)
        self._create_shares()

    # =========================================================================
    # DCE Constraint Build Functions
    # =========================================================================

    def _build_status_upper_bound(self, model: FlowSystemModel, handles: dict[str, VariableHandle]) -> ConstraintResult:
        """Build: flow_rate <= status * size * relative_max"""
        flow_rate = handles['flow_rate'].variable
        status = handles['status'].variable
        _, ub_relative = self.relative_flow_rate_bounds
        upper = status * ub_relative * self.element.size
        return ConstraintResult(lhs=flow_rate, rhs=upper, sense='<=')

    def _build_status_lower_bound(self, model: FlowSystemModel, handles: dict[str, VariableHandle]) -> ConstraintResult:
        """Build: flow_rate >= status * max(epsilon, size * relative_min)"""
        flow_rate = handles['flow_rate'].variable
        status = handles['status'].variable
        lb_relative, _ = self.relative_flow_rate_bounds
        lower_bound = lb_relative * self.element.size
        epsilon = np.maximum(CONFIG.Modeling.epsilon, lower_bound)
        lower = status * epsilon
        return ConstraintResult(lhs=flow_rate, rhs=lower, sense='>=')

    def _build_total_flow_hours_tracking(
        self, model: FlowSystemModel, handles: dict[str, VariableHandle]
    ) -> ConstraintResult:
        """Build: total_flow_hours = sum(flow_rate * dt)"""
        flow_rate = handles['flow_rate'].variable
        total_flow_hours = handles['total_flow_hours'].variable
        rhs = self._model.sum_temporal(flow_rate)
        return ConstraintResult(lhs=total_flow_hours, rhs=rhs, sense='==')

    def _build_flow_hours_over_periods_tracking(
        self, model: FlowSystemModel, handles: dict[str, VariableHandle]
    ) -> ConstraintResult:
        """Build: flow_hours_over_periods = sum(total_flow_hours * period_weight)"""
        total_flow_hours = handles['total_flow_hours'].variable
        flow_hours_over_periods = handles['flow_hours_over_periods'].variable
        weighted = (total_flow_hours * self._model.flow_system.period_weights).sum('period')
        return ConstraintResult(lhs=flow_hours_over_periods, rhs=weighted, sense='==')

    def _build_load_factor_max(self, model: FlowSystemModel, handles: dict[str, VariableHandle]) -> ConstraintResult:
        """Build: total_flow_hours <= size * load_factor_max * total_hours"""
        total_flow_hours = handles['total_flow_hours'].variable
        # Get size (from investment handle if available, else from element)
        if self.with_investment and 'size' in handles:
            size = handles['size'].variable
        else:
            size = self.element.size
        total_hours = self._model.temporal_weight.sum(self._model.temporal_dims)
        rhs = size * self.element.load_factor_max * total_hours
        return ConstraintResult(lhs=total_flow_hours, rhs=rhs, sense='<=')

    def _build_load_factor_min(self, model: FlowSystemModel, handles: dict[str, VariableHandle]) -> ConstraintResult:
        """Build: total_flow_hours >= size * load_factor_min * total_hours"""
        total_flow_hours = handles['total_flow_hours'].variable
        if self.with_investment and 'size' in handles:
            size = handles['size'].variable
        else:
            size = self.element.size
        total_hours = self._model.temporal_weight.sum(self._model.temporal_dims)
        rhs = size * self.element.load_factor_min * total_hours
        return ConstraintResult(lhs=total_flow_hours, rhs=rhs, sense='>=')

    # =========================================================================
    # Original Implementation (kept for backward compatibility)
    # =========================================================================

    def _do_modeling(self):
        """Create variables, constraints, and nested submodels.

        When FlowSystemModel._dce_mode is True, this method skips variable/constraint
        creation since those will be handled by the DCE registries. Only effects
        are still created here since they don't use DCE yet.
        """
        super()._do_modeling()

        # In DCE mode, skip all variable/constraint creation - handled by registries
        # Effects are created after variables exist via on_variables_created()
        if self._model._dce_mode:
            return

        # === Traditional (non-DCE) variable/constraint creation ===

        # Main flow rate variable
        self.add_variables(
            lower=self.absolute_flow_rate_bounds[0],
            upper=self.absolute_flow_rate_bounds[1],
            coords=self._model.get_coords(),
            short_name='flow_rate',
            category=VariableCategory.FLOW_RATE,
        )

        self._constraint_flow_rate()

        # Total flow hours tracking (per period)
        ModelingPrimitives.expression_tracking_variable(
            model=self,
            name=f'{self.label_full}|total_flow_hours',
            tracked_expression=self._model.sum_temporal(self.flow_rate),
            bounds=(
                self.element.flow_hours_min if self.element.flow_hours_min is not None else 0,
                self.element.flow_hours_max if self.element.flow_hours_max is not None else None,
            ),
            coords=['period', 'scenario'],
            short_name='total_flow_hours',
            category=VariableCategory.TOTAL,
        )

        # Weighted sum over all periods constraint
        if self.element.flow_hours_min_over_periods is not None or self.element.flow_hours_max_over_periods is not None:
            # Validate that period dimension exists
            if self._model.flow_system.periods is None:
                raise ValueError(
                    f"{self.label_full}: flow_hours_*_over_periods requires FlowSystem to define 'periods', "
                    f'but FlowSystem has no period dimension. Please define periods in FlowSystem constructor.'
                )
            # Get period weights from FlowSystem
            weighted_flow_hours_over_periods = (self.total_flow_hours * self._model.flow_system.period_weights).sum(
                'period'
            )

            # Create tracking variable for the weighted sum
            ModelingPrimitives.expression_tracking_variable(
                model=self,
                name=f'{self.label_full}|flow_hours_over_periods',
                tracked_expression=weighted_flow_hours_over_periods,
                bounds=(
                    self.element.flow_hours_min_over_periods
                    if self.element.flow_hours_min_over_periods is not None
                    else 0,
                    self.element.flow_hours_max_over_periods
                    if self.element.flow_hours_max_over_periods is not None
                    else None,
                ),
                coords=['scenario'],
                short_name='flow_hours_over_periods',
                category=VariableCategory.TOTAL_OVER_PERIODS,
            )

        # Load factor constraints
        self._create_bounds_for_load_factor()

        # Effects
        self._create_shares()

    def _create_status_model(self):
        status = self.add_variables(
            binary=True,
            short_name='status',
            coords=self._model.get_coords(),
            category=VariableCategory.STATUS,
        )
        self.add_submodels(
            StatusModel(
                model=self._model,
                label_of_element=self.label_of_element,
                parameters=self.element.status_parameters,
                status=status,
                previous_status=self.previous_status,
                label_of_model=self.label_of_element,
            ),
            short_name='status',
        )

    def _create_investment_model(self):
        self.add_submodels(
            InvestmentModel(
                model=self._model,
                label_of_element=self.label_of_element,
                parameters=self.element.size,
                label_of_model=self.label_of_element,
                size_category=VariableCategory.FLOW_SIZE,
            ),
            'investment',
        )

    def _constraint_flow_rate(self):
        """Create bounding constraints for flow_rate (models already created in _create_variables)"""
        if not self.with_investment and not self.with_status:
            # Most basic case. Already covered by direct variable bounds
            pass

        elif self.with_status and not self.with_investment:
            # Status, but no Investment
            self._create_status_model()
            bounds = self.relative_flow_rate_bounds
            BoundingPatterns.bounds_with_state(
                self,
                variable=self.flow_rate,
                bounds=(bounds[0] * self.element.size, bounds[1] * self.element.size),
                state=self.status.status,
            )

        elif self.with_investment and not self.with_status:
            # Investment, but no Status
            self._create_investment_model()
            BoundingPatterns.scaled_bounds(
                self,
                variable=self.flow_rate,
                scaling_variable=self.investment.size,
                relative_bounds=self.relative_flow_rate_bounds,
            )

        elif self.with_investment and self.with_status:
            # Investment and Status
            self._create_investment_model()
            self._create_status_model()

            BoundingPatterns.scaled_bounds_with_state(
                model=self,
                variable=self.flow_rate,
                scaling_variable=self._investment.size,
                relative_bounds=self.relative_flow_rate_bounds,
                scaling_bounds=(self.element.size.minimum_or_fixed_size, self.element.size.maximum_or_fixed_size),
                state=self.status.status,
            )
        else:
            raise Exception('Not valid')

    @property
    def with_status(self) -> bool:
        return self.element.status_parameters is not None

    @property
    def with_investment(self) -> bool:
        return isinstance(self.element.size, InvestParameters)

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
        # Effects per flow hour (use timestep_duration only, cluster_weight is applied when summing to total)
        if self.element.effects_per_flow_hour:
            self._model.effects.add_share_to_effects(
                name=self.label_full,
                expressions={
                    effect: self.flow_rate * self._model.timestep_duration * factor
                    for effect, factor in self.element.effects_per_flow_hour.items()
                },
                target='temporal',
            )

    def _create_bounds_for_load_factor(self):
        """Create load factor constraints using current approach"""
        # Get the size (either from element or investment)
        size = self.investment.size if self.with_investment else self.element.size

        # Total hours in the period (sum of temporal weights)
        total_hours = self._model.temporal_weight.sum(self._model.temporal_dims)

        # Maximum load factor constraint
        if self.element.load_factor_max is not None:
            flow_hours_per_size_max = total_hours * self.element.load_factor_max
            self.add_constraints(
                self.total_flow_hours <= size * flow_hours_per_size_max,
                short_name='load_factor_max',
            )

        # Minimum load factor constraint
        if self.element.load_factor_min is not None:
            flow_hours_per_size_min = total_hours * self.element.load_factor_min
            self.add_constraints(
                self.total_flow_hours >= size * flow_hours_per_size_min,
                short_name='load_factor_min',
            )

    @property
    def relative_flow_rate_bounds(self) -> tuple[xr.DataArray, xr.DataArray]:
        if self.element.fixed_relative_profile is not None:
            return self.element.fixed_relative_profile, self.element.fixed_relative_profile
        return self.element.relative_minimum, self.element.relative_maximum

    @property
    def absolute_flow_rate_bounds(self) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Returns the absolute bounds the flow_rate can reach.
        Further constraining might be needed
        """
        lb_relative, ub_relative = self.relative_flow_rate_bounds

        lb = 0
        if not self.with_status:
            if not self.with_investment:
                # Basic case without investment and without Status
                if self.element.size is not None:
                    lb = lb_relative * self.element.size
            elif self.with_investment and self.element.size.mandatory:
                # With mandatory Investment
                lb = lb_relative * self.element.size.minimum_or_fixed_size

        if self.with_investment:
            ub = ub_relative * self.element.size.maximum_or_fixed_size
        elif self.element.size is not None:
            ub = ub_relative * self.element.size
        else:
            ub = np.inf  # Unbounded when size is None

        return lb, ub

    @property
    def status(self) -> StatusModel | None:
        """Status feature"""
        if 'status' not in self.submodels:
            return None
        return self.submodels['status']

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
    def previous_status(self) -> xr.DataArray | None:
        """Previous status of the flow rate"""
        # TODO: This would be nicer to handle in the Flow itself, and allow DataArrays as well.
        previous_flow_rate = self.element.previous_flow_rate
        if previous_flow_rate is None:
            return None

        return ModelingUtilitiesAbstract.to_binary(
            values=xr.DataArray(
                [previous_flow_rate] if np.isscalar(previous_flow_rate) else previous_flow_rate, dims='time'
            ),
            epsilon=CONFIG.Modeling.epsilon,
            dims='time',
        )


class BusModel(ElementModel):
    """Mathematical model implementation for Bus elements.

    Creates optimization variables and constraints for nodal balance equations,
    and optional excess/deficit variables with penalty costs.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/elements/Bus/>
    """

    element: Bus  # Type hint

    def __init__(self, model: FlowSystemModel, element: Bus):
        self.virtual_supply: linopy.Variable | None = None
        self.virtual_demand: linopy.Variable | None = None
        super().__init__(model, element)

    def _do_modeling(self):
        """Create variables, constraints, and nested submodels"""
        super()._do_modeling()

        # In DCE mode, skip constraint creation - constraints will be added later
        if self._model._dce_mode:
            return

        # inputs == outputs
        for flow in self.element.inputs + self.element.outputs:
            self.register_variable(flow.submodel.flow_rate, flow.label_full)
        inputs = sum([flow.submodel.flow_rate for flow in self.element.inputs])
        outputs = sum([flow.submodel.flow_rate for flow in self.element.outputs])
        eq_bus_balance = self.add_constraints(inputs == outputs, short_name='balance')

        # Add virtual supply/demand to balance and penalty if needed
        if self.element.allows_imbalance:
            imbalance_penalty = self.element.imbalance_penalty_per_flow_hour * self._model.timestep_duration

            self.virtual_supply = self.add_variables(
                lower=0,
                coords=self._model.get_coords(),
                short_name='virtual_supply',
                category=VariableCategory.VIRTUAL_FLOW,
            )

            self.virtual_demand = self.add_variables(
                lower=0,
                coords=self._model.get_coords(),
                short_name='virtual_demand',
                category=VariableCategory.VIRTUAL_FLOW,
            )

            # Σ(inflows) + virtual_supply = Σ(outflows) + virtual_demand
            eq_bus_balance.lhs += self.virtual_supply - self.virtual_demand

            # Add penalty shares as temporal effects (time-dependent)
            from .effects import PENALTY_EFFECT_LABEL

            total_imbalance_penalty = (self.virtual_supply + self.virtual_demand) * imbalance_penalty
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={PENALTY_EFFECT_LABEL: total_imbalance_penalty},
                target='temporal',
            )

    def results_structure(self):
        inputs = [flow.submodel.flow_rate.name for flow in self.element.inputs]
        outputs = [flow.submodel.flow_rate.name for flow in self.element.outputs]
        if self.virtual_supply is not None:
            inputs.append(self.virtual_supply.name)
        if self.virtual_demand is not None:
            outputs.append(self.virtual_demand.name)
        return {
            **super().results_structure(),
            'inputs': inputs,
            'outputs': outputs,
            'flows': [flow.label_full for flow in self.element.inputs + self.element.outputs],
        }


class ComponentModel(ElementModel):
    element: Component  # Type hint

    def __init__(self, model: FlowSystemModel, element: Component):
        self.status: StatusModel | None = None
        super().__init__(model, element)

    def _do_modeling(self):
        """Create variables, constraints, and nested submodels"""
        super()._do_modeling()

        all_flows = self.element.inputs + self.element.outputs

        # Set status_parameters on flows if needed
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

        # Create FlowModels (which creates their variables and constraints)
        for flow in all_flows:
            self.add_submodels(flow.create_model(self._model), short_name=flow.label)

        # In DCE mode, skip constraint creation - constraints will be added later
        if self._model._dce_mode:
            return

        # Create component status variable and StatusModel if needed
        if self.element.status_parameters:
            status = self.add_variables(
                binary=True,
                short_name='status',
                coords=self._model.get_coords(),
                category=VariableCategory.STATUS,
            )
            if len(all_flows) == 1:
                self.add_constraints(status == all_flows[0].submodel.status.status, short_name='status')
            else:
                flow_statuses = [flow.submodel.status.status for flow in all_flows]
                # TODO: Is the EPSILON even necessary?
                self.add_constraints(status <= sum(flow_statuses) + CONFIG.Modeling.epsilon, short_name='status|ub')
                self.add_constraints(
                    status >= sum(flow_statuses) / (len(flow_statuses) + CONFIG.Modeling.epsilon),
                    short_name='status|lb',
                )

            self.status = self.add_submodels(
                StatusModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    parameters=self.element.status_parameters,
                    status=status,
                    label_of_model=self.label_of_element,
                    previous_status=self.previous_status,
                ),
                short_name='status',
            )

        if self.element.prevent_simultaneous_flows:
            # Simultanious Useage --> Only One FLow is On at a time, but needs a Binary for every flow
            ModelingPrimitives.mutual_exclusivity_constraint(
                self,
                binary_variables=[flow.submodel.status.status for flow in self.element.prevent_simultaneous_flows],
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
    def previous_status(self) -> xr.DataArray | None:
        """Previous status of the component, derived from its flows"""
        if self.element.status_parameters is None:
            raise ValueError(f'StatusModel not present in \n{self}\nCant access previous_status')

        previous_status = [flow.submodel.status._previous_status for flow in self.element.inputs + self.element.outputs]
        previous_status = [da for da in previous_status if da is not None]

        if not previous_status:  # Empty list
            return None

        max_len = max(da.sizes['time'] for da in previous_status)

        padded_previous_status = [
            da.assign_coords(time=range(-da.sizes['time'], 0)).reindex(time=range(-max_len, 0), fill_value=0)
            for da in previous_status
        ]
        return xr.concat(padded_previous_status, dim='flow').any(dim='flow').astype(int)
