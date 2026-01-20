"""
This module contains the basic elements of the flixopt framework.
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from . import io as fx_io
from .config import CONFIG
from .core import PlausibilityError
from .features import InvestmentModel, InvestmentProxy, MaskHelpers, StatusProxy
from .interface import InvestParameters, StatusParameters
from .modeling import ModelingUtilitiesAbstract
from .structure import (
    ComponentVarName,
    Element,
    ElementModel,
    ElementType,
    FlowSystemModel,
    FlowVarName,
    TypeModel,
    VariableType,
    register_class_for_io,
)

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

    submodel: BusModelProxy | None

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

    def create_model(self, model: FlowSystemModel) -> BusModelProxy:
        """Create the bus model proxy for this bus element.

        BusesModel creates the actual variables/constraints. The proxy provides
        element-level access to those batched variables.
        """
        self._plausibility_checks()
        self.submodel = BusModelProxy(model, self)
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

    submodel: FlowModelProxy | None

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

    def create_model(self, model: FlowSystemModel) -> FlowModelProxy:
        """Create the flow model proxy for this flow element.

        FlowsModel creates the actual variables/constraints. The proxy provides
        element-level access to those batched variables.
        """
        self._plausibility_checks()
        self.submodel = FlowModelProxy(model, self)
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

    # =========================================================================
    # Type-Level Model Access (for FlowsModel integration)
    # =========================================================================

    _flows_model: FlowsModel | None = None  # Set by FlowsModel during creation

    def set_flows_model(self, flows_model: FlowsModel) -> None:
        """Set reference to the type-level FlowsModel.

        Called by FlowsModel during initialization to enable element access.
        """
        self._flows_model = flows_model

    @property
    def flow_rate_from_type_model(self) -> linopy.Variable | None:
        """Get flow_rate from FlowsModel (if using type-level modeling).

        Returns the slice of the batched variable for this specific flow.
        """
        if self._flows_model is None:
            return None
        return self._flows_model.get_variable('flow_rate', self.label_full)

    @property
    def total_flow_hours_from_type_model(self) -> linopy.Variable | None:
        """Get total_flow_hours from FlowsModel (if using type-level modeling)."""
        if self._flows_model is None:
            return None
        return self._flows_model.get_variable('total_flow_hours', self.label_full)

    @property
    def status_from_type_model(self) -> linopy.Variable | None:
        """Get status from FlowsModel (if using type-level modeling)."""
        if self._flows_model is None or 'status' not in self._flows_model.variables:
            return None
        if self.label_full not in self._flows_model.status_ids:
            return None
        return self._flows_model.get_variable('status', self.label_full)

    @property
    def size_is_fixed(self) -> bool:
        # Wenn kein InvestParameters existiert --> True; Wenn Investparameter, den Wert davon nehmen
        return False if (isinstance(self.size, InvestParameters) and self.size.fixed_size is None) else True

    def _format_invest_params(self, params: InvestParameters) -> str:
        """Format InvestParameters for display."""
        return f'size: {params.format_for_repr()}'


class FlowModelProxy(ElementModel):
    """Lightweight proxy for Flow elements when using type-level modeling.

    Instead of creating its own variables and constraints, this proxy
    provides access to the variables created by FlowsModel. This enables
    the same interface (flow_rate, total_flow_hours, etc.) while avoiding
    duplicate variable/constraint creation.
    """

    element: Flow  # Type hint

    def __init__(self, model: FlowSystemModel, element: Flow):
        # Set _flows_model BEFORE super().__init__() because _do_modeling() uses it
        self._flows_model = model._flows_model
        super().__init__(model, element)

        # Register variables from FlowsModel in our local registry
        # so properties like self.flow_rate work
        if self._flows_model is not None:
            # Note: FlowsModel uses new names 'rate' and 'hours', but we register with legacy names
            # for backward compatibility with property access (self.flow_rate, self.total_flow_hours)
            flow_rate = self._flows_model.get_variable('rate', self.label_full)
            self.register_variable(flow_rate, 'flow_rate')

            total_flow_hours = self._flows_model.get_variable('hours', self.label_full)
            self.register_variable(total_flow_hours, 'total_flow_hours')

            # Status if applicable
            if self.label_full in self._flows_model.status_ids:
                status = self._flows_model.get_variable('status', self.label_full)
                self.register_variable(status, 'status')

            # Investment variables if applicable (from FlowsModel)
            if self.label_full in self._flows_model.investment_ids:
                size = self._flows_model.get_variable('size', self.label_full)
                if size is not None:
                    self.register_variable(size, 'size')

                if self.label_full in self._flows_model.optional_investment_ids:
                    invested = self._flows_model.get_variable('invested', self.label_full)
                    if invested is not None:
                        self.register_variable(invested, 'invested')

    def _do_modeling(self):
        """Skip modeling - FlowsModel already created everything via StatusHelpers."""
        # Status features are handled by StatusHelpers in FlowsModel
        pass

    @property
    def with_status(self) -> bool:
        return self.element.status_parameters is not None

    @property
    def with_investment(self) -> bool:
        return isinstance(self.element.size, InvestParameters)

    @property
    def flow_rate(self) -> linopy.Variable:
        """Main flow rate variable from FlowsModel."""
        return self['flow_rate']

    @property
    def total_flow_hours(self) -> linopy.Variable:
        """Total flow hours variable from FlowsModel."""
        return self['total_flow_hours']

    @property
    def status(self) -> StatusProxy | None:
        """Status feature - returns proxy to FlowsModel's batched status variables."""
        if not self.with_status:
            return None

        # Return a proxy that provides active_hours/startup/etc. for this specific element
        # FlowsModel has get_variable and _previous_status that StatusProxy needs
        return StatusProxy(self._flows_model, self.label_full)

    @property
    def investment(self) -> InvestmentModel | InvestmentProxy | None:
        """Investment feature - returns proxy to access investment variables."""
        if not self.with_investment:
            return None

        # Return a proxy that provides size/invested for this specific element from FlowsModel
        return InvestmentProxy(self._flows_model, self.label_full, dim_name='flow')

    @property
    def previous_status(self) -> xr.DataArray | None:
        """Previous status of the flow rate."""
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

    def results_structure(self):
        return {
            **super().results_structure(),
            'start': self.element.bus if self.element.is_input_in_component else self.element.component,
            'end': self.element.component if self.element.is_input_in_component else self.element.bus,
            'component': self.element.component,
        }


# =============================================================================
# Type-Level Model: FlowsModel
# =============================================================================


class FlowsModel(TypeModel):
    """Type-level model for ALL flows in a FlowSystem.

    Unlike FlowModel (one per Flow instance), FlowsModel handles ALL flows
    in a single instance with batched variables and constraints.

    This enables:
    - One `flow_rate` variable with element dimension for all flows
    - One constraint call for all flow rate bounds
    - Efficient batch creation instead of N individual calls

    The model handles heterogeneous flows by creating subsets:
    - All flows: flow_rate, total_flow_hours
    - Flows with status: status variable
    - Flows with investment: size, invested variables

    Example:
        >>> flows_model = FlowsModel(model, all_flows)
        >>> flows_model.create_variables()
        >>> flows_model.create_constraints()
        >>> # Access individual flow's variable:
        >>> boiler_rate = flows_model.get_variable('flow_rate', 'Boiler(gas_in)')
    """

    element_type = ElementType.FLOW

    def __init__(self, model: FlowSystemModel, elements: list[Flow]):
        """Initialize the type-level model for all flows.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            elements: List of all Flow elements to model.
        """
        super().__init__(model, elements)

        # Categorize flows by their features
        self.flows_with_status: list[Flow] = [f for f in elements if f.status_parameters is not None]
        self.flows_with_investment: list[Flow] = [f for f in elements if isinstance(f.size, InvestParameters)]
        self.flows_with_optional_investment: list[Flow] = [
            f for f in self.flows_with_investment if not f.size.mandatory
        ]
        self.flows_with_flow_hours_over_periods: list[Flow] = [
            f
            for f in elements
            if f.flow_hours_min_over_periods is not None or f.flow_hours_max_over_periods is not None
        ]

        # Element ID lists for subsets
        self.status_ids: list[str] = [f.label_full for f in self.flows_with_status]
        self.investment_ids: list[str] = [f.label_full for f in self.flows_with_investment]
        self.optional_investment_ids: list[str] = [f.label_full for f in self.flows_with_optional_investment]
        self.flow_hours_over_periods_ids: list[str] = [f.label_full for f in self.flows_with_flow_hours_over_periods]

        # Investment params dict (populated in create_investment_model)
        self._invest_params: dict[str, InvestParameters] = {}

        # Status params and previous status (populated in create_status_model)
        self._status_params: dict[str, StatusParameters] = {}
        self._previous_status: dict[str, xr.DataArray] = {}

        # Set reference on each flow element for element access pattern
        for flow in elements:
            flow.set_flows_model(self)

        # Cache for bounds computation
        self._bounds_cache: dict[str, xr.DataArray] = {}

    def create_variables(self) -> None:
        """Create all batched variables for flows.

        Creates:
        - flow|rate: For ALL flows (with flow dimension)
        - flow|hours: For ALL flows
        - flow|status: For flows with status_parameters
        - flow|size: For flows with investment (via InvestmentsModel)
        - flow|invested: For flows with optional investment (via InvestmentsModel)
        - flow|hours_over_periods: For flows with that constraint
        """
        # === flow|rate: ALL flows ===
        # Use dims=None to include ALL dimensions (time, period, scenario)
        # This matches traditional mode behavior where flow_rate has all coords
        lower_bounds = self._collect_bounds('absolute_lower')
        upper_bounds = self._collect_bounds('absolute_upper')

        self.add_variables(
            name='rate',
            var_type=VariableType.FLOW_RATE,
            lower=lower_bounds,
            upper=upper_bounds,
            dims=None,  # Include all dimensions (time, period, scenario)
        )

        # === flow|hours: ALL flows ===
        # Use cached properties with fillna at use time
        total_lower = self.flow_hours_minimum.fillna(0)
        total_upper = self.flow_hours_maximum.fillna(np.inf)

        self.add_variables(
            name='hours',
            var_type=VariableType.TOTAL,
            lower=total_lower,
            upper=total_upper,
            dims=('period', 'scenario'),
        )

        # === flow|status: Only flows with status_parameters ===
        if self.flows_with_status:
            self._add_subset_variables(
                name='status',
                var_type=VariableType.STATUS,
                element_ids=self.status_ids,
                binary=True,
                dims=None,  # Include all dimensions (time, period, scenario)
            )

        # Note: Investment variables (size, invested) are created by InvestmentsModel
        # via create_investment_model(), not inline here

        # === flow|hours_over_periods: Only flows that need it ===
        if self.flows_with_flow_hours_over_periods:
            # Use cached properties, select subset, and apply fillna
            fhop_lower = self.flow_hours_minimum_over_periods.sel(
                {self.dim_name: self.flow_hours_over_periods_ids}
            ).fillna(0)
            fhop_upper = self.flow_hours_maximum_over_periods.sel(
                {self.dim_name: self.flow_hours_over_periods_ids}
            ).fillna(np.inf)

            self._add_subset_variables(
                name='hours_over_periods',
                var_type=VariableType.TOTAL_OVER_PERIODS,
                element_ids=self.flow_hours_over_periods_ids,
                lower=fhop_lower,
                upper=fhop_upper,
                dims=('scenario',),
            )

        logger.debug(
            f'FlowsModel created variables: {len(self.elements)} flows, {len(self.flows_with_status)} with status'
        )

    def create_constraints(self) -> None:
        """Create all batched constraints for flows.

        Creates:
        - flow|hours_eq: Tracking constraint for all flows
        - flow|hours_over_periods_eq: For flows that need it
        - flow|rate bounds: Depending on status/investment configuration
        """
        # === flow|hours = sum_temporal(flow|rate) for ALL flows ===
        flow_rate = self._variables['rate']
        total_hours = self._variables['hours']
        rhs = self.model.sum_temporal(flow_rate)
        self.add_constraints(total_hours == rhs, name='hours_eq')

        # === flow|hours_over_periods tracking ===
        if self.flows_with_flow_hours_over_periods:
            hours_over_periods = self._variables['hours_over_periods']
            # Select only the relevant elements from hours
            hours_subset = total_hours.sel({self.dim_name: self.flow_hours_over_periods_ids})
            period_weights = self.model.flow_system.period_weights
            if period_weights is None:
                period_weights = 1.0
            weighted = (hours_subset * period_weights).sum('period')
            self.add_constraints(hours_over_periods == weighted, name='hours_over_periods_eq')

        # === Load factor constraints ===
        self._create_load_factor_constraints()

        # === Flow rate bounds (depends on status/investment) ===
        self._create_flow_rate_bounds()

        # Note: Investment constraints (size bounds) are created by InvestmentsModel
        # via create_investment_model(), not here

        logger.debug(f'FlowsModel created {len(self._constraints)} constraint types')

    def _create_load_factor_constraints(self) -> None:
        """Create load_factor_min/max constraints for flows that have them.

        Constraints:
        - load_factor_min: total_hours >= total_time * load_factor_min * size
        - load_factor_max: total_hours <= total_time * load_factor_max * size

        Only created for flows with non-NaN load factor values.
        """
        total_hours = self._variables['hours']
        total_time = self.model.timestep_duration.sum(self.model.temporal_dims)
        dim = self.dim_name

        # Helper to get dims other than flow
        def other_dims(arr: xr.DataArray) -> list[str]:
            return [d for d in arr.dims if d != dim]

        # Load factor min constraint
        lf_min = self.load_factor_minimum
        has_lf_min = lf_min.notnull().any(other_dims(lf_min)) if other_dims(lf_min) else lf_min.notnull()
        if has_lf_min.any():
            flow_ids_min = [
                eid
                for eid, has in zip(self.element_ids, has_lf_min.sel({dim: self.element_ids}).values, strict=False)
                if has
            ]
            size_min = self.size_minimum.sel({dim: flow_ids_min}).fillna(0)
            hours_subset = total_hours.sel({dim: flow_ids_min})
            lf_min_subset = lf_min.sel({dim: flow_ids_min}).fillna(0)
            rhs_min = total_time * lf_min_subset * size_min
            self.add_constraints(hours_subset >= rhs_min, name='load_factor_min')

        # Load factor max constraint
        lf_max = self.load_factor_maximum
        has_lf_max = lf_max.notnull().any(other_dims(lf_max)) if other_dims(lf_max) else lf_max.notnull()
        if has_lf_max.any():
            flow_ids_max = [
                eid
                for eid, has in zip(self.element_ids, has_lf_max.sel({dim: self.element_ids}).values, strict=False)
                if has
            ]
            size_max = self.size_maximum.sel({dim: flow_ids_max}).fillna(np.inf)
            hours_subset = total_hours.sel({dim: flow_ids_max})
            lf_max_subset = lf_max.sel({dim: flow_ids_max}).fillna(1)
            rhs_max = total_time * lf_max_subset * size_max
            self.add_constraints(hours_subset <= rhs_max, name='load_factor_max')

    def _add_subset_variables(
        self,
        name: str,
        var_type: VariableType,
        element_ids: list[str],
        dims: tuple[str, ...] | None,
        lower: xr.DataArray | float = -np.inf,
        upper: xr.DataArray | float = np.inf,
        binary: bool = False,
        **kwargs,
    ) -> None:
        """Create a variable for a subset of elements.

        Unlike add_variables() which uses self.element_ids, this creates
        a variable with a custom subset of element IDs.

        Args:
            dims: Dimensions to include. None means ALL model dimensions.
        """
        # Build coordinates with subset element-type dimension (e.g., 'flow')
        dim = self.dim_name
        coord_dict = {dim: pd.Index(element_ids, name=dim)}
        model_coords = self.model.get_coords(dims=dims)
        if model_coords is not None:
            if dims is None:
                # Include all model coords
                for d, coord in model_coords.items():
                    coord_dict[d] = coord
            else:
                for d in dims:
                    if d in model_coords:
                        coord_dict[d] = model_coords[d]
        coords = xr.Coordinates(coord_dict)

        # Create variable
        full_name = f'{self.element_type.value}|{name}'
        variable = self.model.add_variables(
            lower=lower if not binary else None,
            upper=upper if not binary else None,
            coords=coords,
            name=full_name,
            binary=binary,
            **kwargs,
        )

        # Register expansion category
        from .structure import VARIABLE_TYPE_TO_EXPANSION

        expansion_category = VARIABLE_TYPE_TO_EXPANSION.get(var_type)
        if expansion_category is not None:
            self.model.variable_categories[variable.name] = expansion_category

        self._variables[name] = variable

    def _collect_bounds(self, bound_type: str) -> xr.DataArray | float:
        """Collect bounds from all flows and stack them.

        Args:
            bound_type: 'absolute_lower', 'absolute_upper', 'relative_lower', 'relative_upper'

        Returns:
            Stacked bounds with element dimension.
        """
        bounds_list = []
        for flow in self.elements:
            if bound_type == 'absolute_lower':
                bounds_list.append(self._get_absolute_lower_bound(flow))
            elif bound_type == 'absolute_upper':
                bounds_list.append(self._get_absolute_upper_bound(flow))
            elif bound_type == 'relative_lower':
                bounds_list.append(self._get_relative_bounds(flow)[0])
            elif bound_type == 'relative_upper':
                bounds_list.append(self._get_relative_bounds(flow)[1])
            else:
                raise ValueError(f'Unknown bound type: {bound_type}')

        return self._stack_bounds(bounds_list)

    def _get_relative_bounds(self, flow: Flow) -> tuple[xr.DataArray, xr.DataArray]:
        """Get relative flow rate bounds for a flow."""
        if flow.fixed_relative_profile is not None:
            return flow.fixed_relative_profile, flow.fixed_relative_profile
        return xr.broadcast(flow.relative_minimum, flow.relative_maximum)

    def _get_absolute_lower_bound(self, flow: Flow) -> xr.DataArray | float:
        """Get absolute lower bound for a flow."""
        lb_relative, _ = self._get_relative_bounds(flow)

        # Flows with status have lb=0 (status controls activation)
        if flow.status_parameters is not None:
            return 0

        if not isinstance(flow.size, InvestParameters):
            # Basic case without investment
            if flow.size is not None:
                return lb_relative * flow.size
            return 0
        elif flow.size.mandatory:
            # Mandatory investment
            return lb_relative * flow.size.minimum_or_fixed_size
        else:
            # Optional investment - lower bound is 0
            return 0

    def _get_absolute_upper_bound(self, flow: Flow) -> xr.DataArray | float:
        """Get absolute upper bound for a flow."""
        _, ub_relative = self._get_relative_bounds(flow)

        if isinstance(flow.size, InvestParameters):
            return ub_relative * flow.size.maximum_or_fixed_size
        elif flow.size is not None:
            return ub_relative * flow.size
        else:
            return np.inf  # Unbounded

    def _create_flow_rate_bounds(self) -> None:
        """Create flow rate bounding constraints based on status/investment configuration."""
        # Group flows by their constraint type
        # 1. Status only (no investment) - exclude flows with size=None (bounds come from converter)
        status_only_flows = [
            f for f in self.flows_with_status if f not in self.flows_with_investment and f.size is not None
        ]
        if status_only_flows:
            self._create_status_bounds(status_only_flows)

        # 2. Investment only (no status)
        invest_only_flows = [f for f in self.flows_with_investment if f not in self.flows_with_status]
        if invest_only_flows:
            self._create_investment_bounds(invest_only_flows)

        # 3. Both status and investment
        both_flows = [f for f in self.flows_with_status if f in self.flows_with_investment]
        if both_flows:
            self._create_status_investment_bounds(both_flows)

    def _create_status_bounds(self, flows: list[Flow]) -> None:
        """Create bounds: rate <= status * size * relative_max, rate >= status * epsilon."""
        dim = self.dim_name  # 'flow'
        flow_ids = [f.label_full for f in flows]
        flow_rate = self._variables['rate'].sel({dim: flow_ids})
        status = self._variables['status'].sel({dim: flow_ids})

        # Get effective relative bounds and fixed size for the subset
        rel_max = self.effective_relative_maximum.sel({dim: flow_ids})
        rel_min = self.effective_relative_minimum.sel({dim: flow_ids})
        size = self.fixed_size.sel({dim: flow_ids})

        # Upper bound: rate <= status * size * relative_max
        upper_bounds = rel_max * size
        self.add_constraints(flow_rate <= status * upper_bounds, name='rate_status_ub')

        # Lower bound: rate >= status * max(epsilon, size * relative_min)
        lower_bounds = np.maximum(CONFIG.Modeling.epsilon, rel_min * size)
        self.add_constraints(flow_rate >= status * lower_bounds, name='rate_status_lb')

    def _create_investment_bounds(self, flows: list[Flow]) -> None:
        """Create bounds: rate <= size * relative_max, rate >= size * relative_min."""
        dim = self.dim_name  # 'flow'
        flow_ids = [f.label_full for f in flows]
        flow_rate = self._variables['rate'].sel({dim: flow_ids})
        size = self._variables['size'].sel({dim: flow_ids})

        # Get effective relative bounds for the subset
        rel_max = self.effective_relative_maximum.sel({dim: flow_ids})
        rel_min = self.effective_relative_minimum.sel({dim: flow_ids})

        # Upper bound: rate <= size * relative_max
        self.add_constraints(flow_rate <= size * rel_max, name='rate_invest_ub')

        # Lower bound: rate >= size * relative_min
        self.add_constraints(flow_rate >= size * rel_min, name='rate_invest_lb')

    def _create_status_investment_bounds(self, flows: list[Flow]) -> None:
        """Create bounds for flows with both status and investment."""
        dim = self.dim_name  # 'flow'
        flow_ids = [f.label_full for f in flows]
        flow_rate = self._variables['rate'].sel({dim: flow_ids})
        size = self._variables['size'].sel({dim: flow_ids})
        status = self._variables['status'].sel({dim: flow_ids})

        # Get effective relative bounds and size_maximum for the subset
        rel_max = self.effective_relative_maximum.sel({dim: flow_ids})
        rel_min = self.effective_relative_minimum.sel({dim: flow_ids})
        max_size = self.size_maximum.sel({dim: flow_ids})

        # Upper bound: rate <= size * relative_max
        self.add_constraints(flow_rate <= size * rel_max, name='rate_status_invest_ub')

        # Lower bound: rate >= (status - 1) * M + size * relative_min
        # big_M = max_size * relative_min
        big_m = max_size * rel_min
        rhs = (status - 1) * big_m + size * rel_min
        self.add_constraints(flow_rate >= rhs, name='rate_status_invest_lb')

    def create_investment_model(self) -> None:
        """Create investment variables and constraints for flows with investment.

        Creates:
        - flow|size: For all flows with investment
        - flow|invested: For flows with optional (non-mandatory) investment

        Must be called AFTER create_variables() and create_constraints().
        """
        if not self.flows_with_investment:
            return

        from .features import InvestmentHelpers
        from .structure import VARIABLE_TYPE_TO_EXPANSION, VariableType

        # Build params dict for easy access
        self._invest_params = {f.label_full: f.size for f in self.flows_with_investment}

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

        # === flow|size variable ===
        size_coords = xr.Coordinates({dim: pd.Index(element_ids, name=dim), **base_coords_dict})
        size_var = self.model.add_variables(
            lower=lower_bounds,
            upper=upper_bounds,
            coords=size_coords,
            name=FlowVarName.SIZE,
        )
        self._variables['size'] = size_var

        # Register category for segment expansion
        expansion_category = VARIABLE_TYPE_TO_EXPANSION.get(VariableType.SIZE)
        if expansion_category is not None:
            self.model.variable_categories[size_var.name] = expansion_category

        # === flow|invested variable (non-mandatory only) ===
        if non_mandatory_ids:
            invested_coords = xr.Coordinates({dim: pd.Index(non_mandatory_ids, name=dim), **base_coords_dict})
            invested_var = self.model.add_variables(
                binary=True,
                coords=invested_coords,
                name=FlowVarName.INVESTED,
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
                name_prefix='flow',
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
            f'FlowsModel created investment variables: {len(element_ids)} flows '
            f'({len(mandatory_ids)} mandatory, {len(non_mandatory_ids)} optional)'
        )

    def _create_piecewise_effects(self) -> None:
        """Create piecewise effect submodels for flows with piecewise_effects_of_investment.

        Piecewise effects require individual submodels (not batchable) because they
        involve SOS2 constraints with per-element segment definitions.

        For each flow with piecewise_effects_of_investment:
        - Gets the size variable slice for that flow
        - Gets the invested variable slice if applicable (for zero_point)
        - Creates a PiecewiseEffectsModel submodel
        """
        from .features import PiecewiseEffectsModel

        dim = self.dim_name
        size_var = self._variables.get('size')
        invested_var = self._variables.get('invested')

        if size_var is None:
            return

        # Find flows with piecewise effects
        flows_with_piecewise = [
            f for f in self.flows_with_investment if f.size.piecewise_effects_of_investment is not None
        ]

        if not flows_with_piecewise:
            return

        for flow in flows_with_piecewise:
            flow_id = flow.label_full
            params = self._invest_params[flow_id]

            # Get size variable for this flow
            flow_size = size_var.sel({dim: flow_id})

            # Get invested variable for zero_point (if non-mandatory)
            flow_invested = None
            if not params.mandatory and invested_var is not None:
                if flow_id in invested_var.coords.get(dim, []):
                    flow_invested = invested_var.sel({dim: flow_id})

            # Create piecewise effects model
            piecewise_model = PiecewiseEffectsModel(
                model=self.model,
                label_of_element=flow_id,
                label_of_model=f'{flow_id}|PiecewiseEffects',
                piecewise_origin=(flow_size.name, params.piecewise_effects_of_investment.piecewise_origin),
                piecewise_shares=params.piecewise_effects_of_investment.piecewise_shares,
                zero_point=flow_invested,
            )
            piecewise_model.do_modeling()

            logger.debug(f'Created piecewise effects for {flow_id}')

    # === Investment effect properties (used by EffectsModel) ===

    @cached_property
    def invest_effects_per_size(self) -> xr.DataArray | None:
        """Combined effects_of_investment_per_size with (flow, effect) dims."""
        if not hasattr(self, '_invest_params'):
            return None
        from .features import InvestmentHelpers

        element_ids = [eid for eid in self.investment_ids if self._invest_params[eid].effects_of_investment_per_size]
        if not element_ids:
            return None
        effects_dict = InvestmentHelpers.collect_effects(
            self._invest_params, element_ids, 'effects_of_investment_per_size', self.dim_name
        )
        return InvestmentHelpers.build_effect_factors(effects_dict, element_ids, self.dim_name)

    @cached_property
    def invest_effects_of_investment(self) -> xr.DataArray | None:
        """Combined effects_of_investment with (flow, effect) dims for non-mandatory."""
        if not hasattr(self, '_invest_params'):
            return None
        from .features import InvestmentHelpers

        element_ids = [eid for eid in self.optional_investment_ids if self._invest_params[eid].effects_of_investment]
        if not element_ids:
            return None
        effects_dict = InvestmentHelpers.collect_effects(
            self._invest_params, element_ids, 'effects_of_investment', self.dim_name
        )
        return InvestmentHelpers.build_effect_factors(effects_dict, element_ids, self.dim_name)

    @cached_property
    def invest_effects_of_retirement(self) -> xr.DataArray | None:
        """Combined effects_of_retirement with (flow, effect) dims for non-mandatory."""
        if not hasattr(self, '_invest_params'):
            return None
        from .features import InvestmentHelpers

        element_ids = [eid for eid in self.optional_investment_ids if self._invest_params[eid].effects_of_retirement]
        if not element_ids:
            return None
        effects_dict = InvestmentHelpers.collect_effects(
            self._invest_params, element_ids, 'effects_of_retirement', self.dim_name
        )
        return InvestmentHelpers.build_effect_factors(effects_dict, element_ids, self.dim_name)

    @cached_property
    def status_effects_per_active_hour(self) -> xr.DataArray | None:
        """Combined effects_per_active_hour with (flow, effect) dims."""
        if not hasattr(self, '_status_params') or not self._status_params:
            return None
        from .features import InvestmentHelpers, StatusHelpers

        element_ids = [eid for eid in self.status_ids if self._status_params[eid].effects_per_active_hour]
        if not element_ids:
            return None
        effects_dict = StatusHelpers.collect_status_effects(
            self._status_params, element_ids, 'effects_per_active_hour', self.dim_name
        )
        return InvestmentHelpers.build_effect_factors(effects_dict, element_ids, self.dim_name)

    @cached_property
    def status_effects_per_startup(self) -> xr.DataArray | None:
        """Combined effects_per_startup with (flow, effect) dims."""
        if not hasattr(self, '_status_params') or not self._status_params:
            return None
        from .features import InvestmentHelpers, StatusHelpers

        element_ids = [eid for eid in self.status_ids if self._status_params[eid].effects_per_startup]
        if not element_ids:
            return None
        effects_dict = StatusHelpers.collect_status_effects(
            self._status_params, element_ids, 'effects_per_startup', self.dim_name
        )
        return InvestmentHelpers.build_effect_factors(effects_dict, element_ids, self.dim_name)

    @cached_property
    def mandatory_invest_effects(self) -> list[tuple[str, dict[str, float | xr.DataArray]]]:
        """List of (element_id, effects_dict) for mandatory investments with fixed effects.

        These are constant effects always incurred, not dependent on the invested variable.
        Returns empty list if no such effects exist.
        """
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

    @cached_property
    def retirement_constant_effects(self) -> list[tuple[str, dict[str, float | xr.DataArray]]]:
        """List of (element_id, effects_dict) for retirement constant parts.

        For optional investments with effects_of_retirement, this is the constant "+factor"
        part of the formula: -invested * factor + factor.
        Returns empty list if no such effects exist.
        """
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

    def create_status_model(self) -> None:
        """Create status variables and constraints for flows with status.

        Uses StatusHelpers.create_status_features() to create:
        - flow|active_hours: For all flows with status
        - flow|startup, flow|shutdown: For flows needing startup tracking
        - flow|inactive: For flows needing downtime tracking
        - flow|startup_count: For flows with startup limit
        - flow|uptime, flow|downtime: Duration tracking variables

        Must be called AFTER create_variables() and create_constraints().
        """
        if not self.flows_with_status:
            return

        from .features import StatusHelpers

        # Build params and previous_status dicts
        self._status_params = {f.label_full: f.status_parameters for f in self.flows_with_status}
        for flow in self.flows_with_status:
            prev = self.get_previous_status(flow)
            if prev is not None:
                self._previous_status[flow.label_full] = prev

        status = self._variables.get('status')

        # Use helper to create all status features
        status_vars = StatusHelpers.create_status_features(
            model=self.model,
            status=status,
            params=self._status_params,
            dim_name=self.dim_name,
            var_names=FlowVarName,
            previous_status=self._previous_status,
            has_clusters=self.model.flow_system.clusters is not None,
        )

        # Store created variables in our variables dict
        self._variables.update(status_vars)

    def collect_effect_share_specs(self) -> dict[str, list[tuple[str, float | xr.DataArray]]]:
        """Collect effect share specifications for all flows.

        Returns:
            Dict mapping effect_name to list of (element_id, factor) tuples.
            Example: {'costs': [('Boiler(gas_in)', 0.05), ('HP(elec_in)', 0.1)]}
        """
        effect_specs: dict[str, list[tuple[str, float | xr.DataArray]]] = {}
        for flow in self.elements:
            if flow.effects_per_flow_hour:
                for effect_name, factor in flow.effects_per_flow_hour.items():
                    if effect_name not in effect_specs:
                        effect_specs[effect_name] = []
                    effect_specs[effect_name].append((flow.label_full, factor))
        return effect_specs

    @property
    def rate(self) -> linopy.Variable:
        """Batched flow rate variable with (flow, time) dims."""
        return self.model.variables[FlowVarName.RATE]

    @property
    def status(self) -> linopy.Variable | None:
        """Batched status variable with (flow, time) dims, or None if no flows have status."""
        return self.model.variables[FlowVarName.STATUS] if FlowVarName.STATUS in self.model.variables else None

    @property
    def startup(self) -> linopy.Variable | None:
        """Batched startup variable with (flow, time) dims, or None if no flows need startup tracking."""
        return self.model.variables[FlowVarName.STARTUP] if FlowVarName.STARTUP in self.model.variables else None

    @property
    def shutdown(self) -> linopy.Variable | None:
        """Batched shutdown variable with (flow, time) dims, or None if no flows need startup tracking."""
        return self.model.variables[FlowVarName.SHUTDOWN] if FlowVarName.SHUTDOWN in self.model.variables else None

    @property
    def size(self) -> linopy.Variable | None:
        """Batched size variable with (flow,) dims, or None if no flows have investment."""
        return self.model.variables[FlowVarName.SIZE] if FlowVarName.SIZE in self.model.variables else None

    @property
    def invested(self) -> linopy.Variable | None:
        """Batched invested binary variable with (flow,) dims, or None if no optional investments."""
        return self.model.variables[FlowVarName.INVESTED] if FlowVarName.INVESTED in self.model.variables else None

    @cached_property
    def effects_per_flow_hour(self) -> xr.DataArray | None:
        """Combined effect factors with (flow, effect, ...) dims.

        Missing (flow, effect) combinations are NaN - the xarray convention for
        missing data. This distinguishes "no effect defined" from "effect is zero".

        Use `.fillna(0)` to fill for computation, `.notnull()` as mask.
        """
        flows_with_effects = [f for f in self.elements if f.effects_per_flow_hour]
        if not flows_with_effects:
            return None

        effects_model = getattr(self.model.effects, '_batched_model', None)
        if effects_model is None:
            return None

        effect_ids = effects_model.effect_ids
        flow_ids = [f.label_full for f in flows_with_effects]

        # Use np.nan for missing effects (not 0!) to distinguish "not defined" from "zero"
        # Use coords='minimal' to handle dimension mismatches (some effects may have 'time', some scalars)
        flow_factors = [
            xr.concat(
                [xr.DataArray(flow.effects_per_flow_hour.get(eff, np.nan)) for eff in effect_ids],
                dim='effect',
                coords='minimal',
            ).assign_coords(effect=effect_ids)
            for flow in flows_with_effects
        ]

        # Use coords='minimal' to handle dimension mismatches (some effects may have 'period', some don't)
        return xr.concat(flow_factors, dim=self.dim_name, coords='minimal').assign_coords({self.dim_name: flow_ids})

    def get_previous_status(self, flow: Flow) -> xr.DataArray | None:
        """Get previous status for a flow based on its previous_flow_rate.

        This is used by ComponentsModel to compute component previous status.

        Args:
            flow: The flow to get previous status for.

        Returns:
            Binary DataArray with 1 where previous flow was active, None if no previous data.
        """
        previous_flow_rate = flow.previous_flow_rate
        if previous_flow_rate is None:
            return None

        return ModelingUtilitiesAbstract.to_binary(
            values=xr.DataArray(
                [previous_flow_rate] if np.isscalar(previous_flow_rate) else previous_flow_rate, dims='time'
            ),
            epsilon=CONFIG.Modeling.epsilon,
            dims='time',
        )

    # === Batched Parameter Properties ===

    # --- Flow Hours Bounds ---

    @cached_property
    def flow_hours_minimum(self) -> xr.DataArray:
        """(flow, period, scenario) - minimum total flow hours. NaN = no constraint."""
        values = [f.flow_hours_min if f.flow_hours_min is not None else np.nan for f in self.elements]
        return self._broadcast_to_model_coords(self._stack_bounds(values), dims=['period', 'scenario'])

    @cached_property
    def flow_hours_maximum(self) -> xr.DataArray:
        """(flow, period, scenario) - maximum total flow hours. NaN = no constraint."""
        values = [f.flow_hours_max if f.flow_hours_max is not None else np.nan for f in self.elements]
        return self._broadcast_to_model_coords(self._stack_bounds(values), dims=['period', 'scenario'])

    @cached_property
    def flow_hours_minimum_over_periods(self) -> xr.DataArray:
        """(flow, scenario) - minimum flow hours summed over all periods. NaN = no constraint."""
        values = [
            f.flow_hours_min_over_periods if f.flow_hours_min_over_periods is not None else np.nan
            for f in self.elements
        ]
        return self._broadcast_to_model_coords(self._stack_bounds(values), dims=['scenario'])

    @cached_property
    def flow_hours_maximum_over_periods(self) -> xr.DataArray:
        """(flow, scenario) - maximum flow hours summed over all periods. NaN = no constraint."""
        values = [
            f.flow_hours_max_over_periods if f.flow_hours_max_over_periods is not None else np.nan
            for f in self.elements
        ]
        return self._broadcast_to_model_coords(self._stack_bounds(values), dims=['scenario'])

    # --- Load Factor Bounds ---

    @cached_property
    def load_factor_minimum(self) -> xr.DataArray:
        """(flow, period, scenario) - minimum load factor. NaN = no constraint."""
        values = [f.load_factor_min if f.load_factor_min is not None else np.nan for f in self.elements]
        return self._broadcast_to_model_coords(self._stack_bounds(values), dims=['period', 'scenario'])

    @cached_property
    def load_factor_maximum(self) -> xr.DataArray:
        """(flow, period, scenario) - maximum load factor. NaN = no constraint."""
        values = [f.load_factor_max if f.load_factor_max is not None else np.nan for f in self.elements]
        return self._broadcast_to_model_coords(self._stack_bounds(values), dims=['period', 'scenario'])

    # --- Relative Bounds ---

    @cached_property
    def relative_minimum(self) -> xr.DataArray:
        """(flow, time, period, scenario) - relative lower bound on flow rate."""
        values = [f.relative_minimum for f in self.elements]  # Default is 0, never None
        return self._broadcast_to_model_coords(self._stack_bounds(values), dims=None)

    @cached_property
    def relative_maximum(self) -> xr.DataArray:
        """(flow, time, period, scenario) - relative upper bound on flow rate."""
        values = [f.relative_maximum for f in self.elements]  # Default is 1, never None
        return self._broadcast_to_model_coords(self._stack_bounds(values), dims=None)

    @cached_property
    def fixed_relative_profile(self) -> xr.DataArray:
        """(flow, time, period, scenario) - fixed profile. NaN = not fixed."""
        values = [f.fixed_relative_profile if f.fixed_relative_profile is not None else np.nan for f in self.elements]
        return self._broadcast_to_model_coords(self._stack_bounds(values), dims=None)

    @cached_property
    def effective_relative_minimum(self) -> xr.DataArray:
        """(flow, time, period, scenario) - effective lower bound (uses fixed_profile if set)."""
        # Where fixed_relative_profile is set, use it; otherwise use relative_minimum
        fixed = self.fixed_relative_profile
        rel_min = self.relative_minimum
        return xr.where(fixed.notnull(), fixed, rel_min)

    @cached_property
    def effective_relative_maximum(self) -> xr.DataArray:
        """(flow, time, period, scenario) - effective upper bound (uses fixed_profile if set)."""
        # Where fixed_relative_profile is set, use it; otherwise use relative_maximum
        fixed = self.fixed_relative_profile
        rel_max = self.relative_maximum
        return xr.where(fixed.notnull(), fixed, rel_max)

    @cached_property
    def fixed_size(self) -> xr.DataArray:
        """(flow, period, scenario) - fixed size for non-investment flows. NaN for investment/no-size flows."""
        values = []
        for f in self.elements:
            if f.size is None or isinstance(f.size, InvestParameters):
                values.append(np.nan)
            else:
                values.append(f.size)  # Fixed size
        return self._broadcast_to_model_coords(self._stack_bounds(values), dims=['period', 'scenario'])

    # --- Size Bounds ---

    @cached_property
    def size_minimum(self) -> xr.DataArray:
        """(flow, period, scenario) - minimum size. NaN for flows without size."""
        values = []
        for f in self.elements:
            if f.size is None:
                values.append(np.nan)
            elif isinstance(f.size, InvestParameters):
                values.append(f.size.minimum_or_fixed_size)
            else:
                values.append(f.size)  # Fixed size
        return self._broadcast_to_model_coords(self._stack_bounds(values), dims=['period', 'scenario'])

    @cached_property
    def size_maximum(self) -> xr.DataArray:
        """(flow, period, scenario) - maximum size. NaN for flows without size."""
        values = []
        for f in self.elements:
            if f.size is None:
                values.append(np.nan)
            elif isinstance(f.size, InvestParameters):
                values.append(f.size.maximum_or_fixed_size)
            else:
                values.append(f.size)  # Fixed size
        return self._broadcast_to_model_coords(self._stack_bounds(values), dims=['period', 'scenario'])

    # --- Investment Masks ---

    @cached_property
    def investment_mandatory(self) -> xr.DataArray:
        """(flow,) bool - True if investment is mandatory, False if optional, NaN if no investment."""
        values = []
        for f in self.elements:
            if not isinstance(f.size, InvestParameters):
                values.append(np.nan)
            else:
                values.append(f.size.mandatory)
        return xr.DataArray(values, dims=[self.dim_name], coords={self.dim_name: self.element_ids})

    @cached_property
    def linked_periods(self) -> xr.DataArray | None:
        """(flow, period) - period linking mask. 1=linked, 0=not linked, NaN=no linking."""
        has_linking = any(
            isinstance(f.size, InvestParameters) and f.size.linked_periods is not None for f in self.elements
        )
        if not has_linking:
            return None

        values = []
        for f in self.elements:
            if not isinstance(f.size, InvestParameters) or f.size.linked_periods is None:
                values.append(np.nan)
            else:
                values.append(f.size.linked_periods)
        return self._broadcast_to_model_coords(self._stack_bounds(values), dims=['period'])

    # --- Investment Subset Properties (for create_investment_model) ---

    @cached_property
    def mandatory_investment_ids(self) -> list[str]:
        """List of flow IDs with mandatory investment."""
        return [f.label_full for f in self.flows_with_investment if f.size.mandatory]

    @cached_property
    def _size_lower(self) -> xr.DataArray:
        """(flow,) - minimum size for investment flows."""
        from .features import InvestmentHelpers

        element_ids = self.investment_ids
        values = [f.size.minimum_or_fixed_size for f in self.flows_with_investment]
        return InvestmentHelpers.stack_bounds(values, element_ids, self.dim_name)

    @cached_property
    def _size_upper(self) -> xr.DataArray:
        """(flow,) - maximum size for investment flows."""
        from .features import InvestmentHelpers

        element_ids = self.investment_ids
        values = [f.size.maximum_or_fixed_size for f in self.flows_with_investment]
        return InvestmentHelpers.stack_bounds(values, element_ids, self.dim_name)

    @cached_property
    def _linked_periods_mask(self) -> xr.DataArray | None:
        """(flow, period) - linked periods for investment flows. None if no linking."""
        from .features import InvestmentHelpers

        linked_list = [f.size.linked_periods for f in self.flows_with_investment]
        if not any(lp is not None for lp in linked_list):
            return None

        element_ids = self.investment_ids
        values = [lp if lp is not None else np.nan for lp in linked_list]
        return InvestmentHelpers.stack_bounds(values, element_ids, self.dim_name)

    @cached_property
    def _mandatory_mask(self) -> xr.DataArray:
        """(flow,) bool - True if mandatory, False if optional."""
        element_ids = self.investment_ids
        values = [f.size.mandatory for f in self.flows_with_investment]
        return xr.DataArray(values, dims=[self.dim_name], coords={self.dim_name: element_ids})

    @cached_property
    def _optional_lower(self) -> xr.DataArray | None:
        """(flow,) - minimum size for optional investment flows."""
        if not self.optional_investment_ids:
            return None
        from .features import InvestmentHelpers

        flows = [f for f in self.flows_with_investment if not f.size.mandatory]
        element_ids = self.optional_investment_ids
        values = [f.size.minimum_or_fixed_size for f in flows]
        return InvestmentHelpers.stack_bounds(values, element_ids, self.dim_name)

    @cached_property
    def _optional_upper(self) -> xr.DataArray | None:
        """(flow,) - maximum size for optional investment flows."""
        if not self.optional_investment_ids:
            return None
        from .features import InvestmentHelpers

        flows = [f for f in self.flows_with_investment if not f.size.mandatory]
        element_ids = self.optional_investment_ids
        values = [f.size.maximum_or_fixed_size for f in flows]
        return InvestmentHelpers.stack_bounds(values, element_ids, self.dim_name)

    # --- Previous Status ---

    @cached_property
    def previous_status_batched(self) -> xr.DataArray | None:
        """Concatenated previous status (flow, time) from previous_flow_rate.

        Returns None if no flows have previous_flow_rate set.
        For flows without previous_flow_rate, their slice contains NaN values.

        The DataArray has dimensions (flow, time) where:
        - flow: subset of flows_with_status that have previous_flow_rate
        - time: negative time indices representing past timesteps
        """
        flows_with_previous = [f for f in self.flows_with_status if f.previous_flow_rate is not None]
        if not flows_with_previous:
            return None

        previous_arrays = []
        for flow in flows_with_previous:
            previous_flow_rate = flow.previous_flow_rate

            # Convert to DataArray and compute binary status
            previous_status = ModelingUtilitiesAbstract.to_binary(
                values=xr.DataArray(
                    [previous_flow_rate] if np.isscalar(previous_flow_rate) else previous_flow_rate,
                    dims='time',
                ),
                epsilon=CONFIG.Modeling.epsilon,
                dims='time',
            )
            # Expand dims to add flow dimension
            previous_status = previous_status.expand_dims({self.dim_name: [flow.label_full]})
            previous_arrays.append(previous_status)

        return xr.concat(previous_arrays, dim=self.dim_name)


class BusesModel(TypeModel):
    """Type-level model for ALL buses in a FlowSystem.

    Unlike BusModel (one per Bus instance), BusesModel handles ALL buses
    in a single instance with batched variables and constraints.

    This enables:
    - One constraint call for all bus balance constraints
    - Batched virtual_supply/virtual_demand for buses with imbalance
    - Efficient batch creation instead of N individual calls

    The model handles heterogeneous buses by creating subsets:
    - All buses: balance constraints
    - Buses with imbalance: virtual_supply, virtual_demand variables

    Example:
        >>> buses_model = BusesModel(model, all_buses, flows_model)
        >>> buses_model.create_variables()
        >>> buses_model.create_constraints()
    """

    element_type = ElementType.BUS

    def __init__(self, model: FlowSystemModel, elements: list[Bus], flows_model: FlowsModel):
        """Initialize the type-level model for all buses.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            elements: List of all Bus elements to model.
            flows_model: The FlowsModel containing flow_rate variables.
        """
        super().__init__(model, elements)
        self._flows_model = flows_model

        # Categorize buses by their features
        self.buses_with_imbalance: list[Bus] = [b for b in elements if b.allows_imbalance]

        # Element ID lists for subsets
        self.imbalance_ids: list[str] = [b.label_full for b in self.buses_with_imbalance]

        # Set reference on each bus element
        for bus in elements:
            bus._buses_model = self

    def create_variables(self) -> None:
        """Create all batched variables for buses.

        Creates:
        - virtual_supply: For buses with imbalance penalty
        - virtual_demand: For buses with imbalance penalty
        """
        if self.buses_with_imbalance:
            # virtual_supply: allows adding flow to meet demand
            self._add_subset_variables(
                name='virtual_supply',
                var_type=VariableType.VIRTUAL_FLOW,
                element_ids=self.imbalance_ids,
                lower=0.0,
                upper=np.inf,
                dims=self.model.temporal_dims,
            )

            # virtual_demand: allows removing excess flow
            self._add_subset_variables(
                name='virtual_demand',
                var_type=VariableType.VIRTUAL_FLOW,
                element_ids=self.imbalance_ids,
                lower=0.0,
                upper=np.inf,
                dims=self.model.temporal_dims,
            )

        logger.debug(
            f'BusesModel created variables: {len(self.elements)} buses, {len(self.buses_with_imbalance)} with imbalance'
        )

    def _add_subset_variables(
        self,
        name: str,
        var_type: VariableType,
        element_ids: list[str],
        dims: tuple[str, ...],
        lower: xr.DataArray | float = -np.inf,
        upper: xr.DataArray | float = np.inf,
        **kwargs,
    ) -> None:
        """Create a variable for a subset of elements."""
        # Build coordinates with subset element-type dimension (e.g., 'bus')
        dim = self.dim_name
        coord_dict = {dim: pd.Index(element_ids, name=dim)}
        model_coords = self.model.get_coords(dims=dims)
        if model_coords is not None:
            for d in dims:
                if d in model_coords:
                    coord_dict[d] = model_coords[d]
        coords = xr.Coordinates(coord_dict)

        # Create variable
        full_name = f'{self.element_type.value}|{name}'
        variable = self.model.add_variables(
            lower=lower,
            upper=upper,
            coords=coords,
            name=full_name,
            **kwargs,
        )

        # Register category for segment expansion
        from .structure import VARIABLE_TYPE_TO_EXPANSION

        expansion_category = VARIABLE_TYPE_TO_EXPANSION.get(var_type)
        if expansion_category is not None:
            self.model.variable_categories[variable.name] = expansion_category

        # Store reference
        self._variables[name] = variable

    def create_constraints(self) -> None:
        """Create all batched constraints for buses.

        Creates:
        - bus_balance: Sum(inputs) == Sum(outputs) for all buses
        - With virtual_supply/demand adjustment for buses with imbalance
        """
        flow_rate = self._flows_model._variables['rate']
        flow_dim = self._flows_model.dim_name  # 'flow'
        bus_dim = self.dim_name  # 'bus'

        # Build the balance constraint for each bus
        # We need to do this per-bus because each bus has different inputs/outputs
        # However, we can batch create using xr.concat
        lhs_list = []
        rhs_list = []

        for bus in self.elements:
            bus_label = bus.label_full

            # Get input flow IDs and output flow IDs for this bus
            input_ids = [f.label_full for f in bus.inputs]
            output_ids = [f.label_full for f in bus.outputs]

            # Sum of input flow rates
            if input_ids:
                inputs_sum = flow_rate.sel({flow_dim: input_ids}).sum(flow_dim)
            else:
                inputs_sum = 0

            # Sum of output flow rates
            if output_ids:
                outputs_sum = flow_rate.sel({flow_dim: output_ids}).sum(flow_dim)
            else:
                outputs_sum = 0

            # Add virtual supply/demand if this bus allows imbalance
            if bus.allows_imbalance:
                virtual_supply = self._variables['virtual_supply'].sel({bus_dim: bus_label})
                virtual_demand = self._variables['virtual_demand'].sel({bus_dim: bus_label})
                # inputs + virtual_supply == outputs + virtual_demand
                lhs = inputs_sum + virtual_supply
                rhs = outputs_sum + virtual_demand
            else:
                # inputs == outputs (strict balance)
                lhs = inputs_sum
                rhs = outputs_sum

            lhs_list.append(lhs)
            rhs_list.append(rhs)

        # Stack into a single constraint with bus dimension
        # Note: For efficiency, we create one constraint per bus but they share a name prefix
        for i, bus in enumerate(self.elements):
            lhs, rhs = lhs_list[i], rhs_list[i]
            # Skip if both sides are scalar zeros (no flows connected)
            if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
                continue
            constraint_name = f'{self.element_type.value}|{bus.label}|balance'
            self.model.add_constraints(
                lhs == rhs,
                name=constraint_name,
            )

        logger.debug(f'BusesModel created {len(self.elements)} balance constraints')

    def collect_penalty_share_specs(self) -> list[tuple[str, xr.DataArray]]:
        """Collect penalty effect share specifications for buses with imbalance.

        Returns:
            List of (element_label, penalty_expression) tuples.
        """
        if not self.buses_with_imbalance:
            return []

        dim = self.dim_name
        penalty_specs = []
        for bus in self.buses_with_imbalance:
            bus_label = bus.label_full
            imbalance_penalty = bus.imbalance_penalty_per_flow_hour * self.model.timestep_duration

            virtual_supply = self._variables['virtual_supply'].sel({dim: bus_label})
            virtual_demand = self._variables['virtual_demand'].sel({dim: bus_label})

            total_imbalance_penalty = (virtual_supply + virtual_demand) * imbalance_penalty
            penalty_specs.append((bus_label, total_imbalance_penalty))

        return penalty_specs

    def create_effect_shares(self) -> None:
        """Create penalty effect shares for buses with imbalance.

        Collects specs and delegates to EffectCollectionModel for application.
        """
        penalty_specs = self.collect_penalty_share_specs()
        if penalty_specs:
            self.model.effects.apply_batched_penalty_shares(penalty_specs)

    def get_variable(self, name: str, element_id: str | None = None):
        """Get a variable, optionally selecting a specific element.

        Args:
            name: Variable name (e.g., 'virtual_supply').
            element_id: Optional element label_full. If provided, returns slice for that element.

        Returns:
            Full batched variable, or element slice if element_id provided.
        """
        var = self._variables.get(name)
        if var is None:
            return None
        if element_id is not None:
            return var.sel({self.dim_name: element_id})
        return var


class BusModelProxy(ElementModel):
    """Lightweight proxy for Bus elements when using type-level modeling.

    Instead of creating its own variables and constraints, this proxy
    provides access to the variables created by BusesModel. This enables
    the same interface (virtual_supply, virtual_demand, etc.) while avoiding
    duplicate variable/constraint creation.
    """

    element: Bus  # Type hint

    def __init__(self, model: FlowSystemModel, element: Bus):
        self.virtual_supply: linopy.Variable | None = None
        self.virtual_demand: linopy.Variable | None = None
        # Set _buses_model BEFORE super().__init__() for consistency
        self._buses_model = model._buses_model
        super().__init__(model, element)

        # Register variables from BusesModel in our local registry
        if self._buses_model is not None and self.label_full in self._buses_model.imbalance_ids:
            self.virtual_supply = self._buses_model.get_variable('virtual_supply', self.label_full)
            self.register_variable(self.virtual_supply, 'virtual_supply')

            self.virtual_demand = self._buses_model.get_variable('virtual_demand', self.label_full)
            self.register_variable(self.virtual_demand, 'virtual_demand')

    def _do_modeling(self):
        """Skip modeling - BusesModel already created everything."""
        # Register flow variables in our local registry for results_structure
        for flow in self.element.inputs + self.element.outputs:
            self.register_variable(flow.submodel.flow_rate, flow.label_full)

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
        self.status: StatusProxy | None = None
        super().__init__(model, element)

    def _do_modeling(self):
        """Create variables, constraints, and nested submodels.

        Note: status_parameters setup is done in FlowSystemModel.do_modeling() preprocessing,
        before FlowsModel is created. This ensures FlowsModel knows which flows need status variables.
        """
        super()._do_modeling()

        # Create FlowModelProxy for each flow (variables/constraints handled by FlowsModel)
        for flow in self.element.inputs + self.element.outputs:
            self.add_submodels(flow.create_model(self._model), short_name=flow.label)
        # Status and prevent_simultaneous constraints handled by type-level models

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
            raise ValueError(f'status_parameters not present in \n{self}\nCant access previous_status')

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


class ComponentsModel:
    """Type-level model for batched component status across multiple components.

    This handles component-level status variables and constraints for ALL components
    with status_parameters in a single instance with batched variables.

    Component status is derived from flow statuses:
    - Single-flow component: status == flow_status
    - Multi-flow component: status is 1 if ANY flow is active

    This enables:
    - Batched `component|status` variable with component dimension
    - Batched constraints linking component status to flow statuses
    - Status features (active_hours, startup, shutdown, etc.) via StatusHelpers

    Example:
        >>> component_statuses = ComponentsModel(
        ...     model=flow_system_model,
        ...     components=components_with_status,
        ...     flows_model=flows_model,
        ... )
        >>> component_statuses.create_variables()
        >>> component_statuses.create_constraints()
        >>> component_statuses.create_status_features()
        >>> component_statuses.create_effect_shares()
    """

    def __init__(
        self,
        model: FlowSystemModel,
        components: list[Component],
        flows_model: FlowsModel,
    ):
        """Initialize the type-level component status model.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            components: List of components with status_parameters.
            flows_model: The FlowsModel that owns flow status variables.
        """

        self._logger = logging.getLogger('flixopt')
        self.model = model
        self.components = components
        self._flows_model = flows_model
        self.element_ids: list[str] = [c.label for c in components]
        self.dim_name = 'component'

        # Variables dict
        self._variables: dict[str, linopy.Variable] = {}

        # Status feature variables (active_hours, startup, shutdown, etc.) created by StatusHelpers
        self._status_variables: dict[str, linopy.Variable] = {}

        self._logger.debug(f'ComponentsModel initialized: {len(components)} components with status')

    # --- Cached Properties ---

    @cached_property
    def _status_params(self) -> dict[str, StatusParameters]:
        """Dict of component_id -> StatusParameters."""
        return {c.label: c.status_parameters for c in self.components}

    @cached_property
    def _previous_status_dict(self) -> dict[str, xr.DataArray]:
        """Dict of component_id -> previous_status DataArray."""
        result = {}
        for c in self.components:
            prev = self._get_previous_status_for_component(c)
            if prev is not None:
                result[c.label] = prev
        return result

    @cached_property
    def _flow_mask(self) -> xr.DataArray:
        """(component, flow) mask: 1 if flow belongs to component."""
        membership = MaskHelpers.build_flow_membership(
            self.components,
            lambda c: c.inputs + c.outputs,
        )
        return MaskHelpers.build_mask(
            row_dim='component',
            row_ids=self.element_ids,
            col_dim='flow',
            col_ids=self._flows_model.element_ids,
            membership=membership,
        )

    @cached_property
    def _flow_count(self) -> xr.DataArray:
        """(component,) number of flows per component."""
        counts = [len(c.inputs) + len(c.outputs) for c in self.components]
        return xr.DataArray(
            counts,
            dims=['component'],
            coords={'component': self.element_ids},
        )

    def create_variables(self) -> None:
        """Create batched component status variable with component dimension."""
        if not self.components:
            return

        dim = self.dim_name

        # Create component status binary variable
        temporal_coords = self.model.get_coords()
        status_coords = xr.Coordinates(
            {
                dim: pd.Index(self.element_ids, name=dim),
                **dict(temporal_coords),
            }
        )

        self._variables['status'] = self.model.add_variables(
            binary=True,
            coords=status_coords,
            name='component|status',
        )

        self._logger.debug(f'ComponentsModel created status variable for {len(self.components)} components')

    def create_constraints(self) -> None:
        """Create batched constraints linking component status to flow statuses.

        Uses mask matrix for batched constraint creation:
        - Single-flow components: comp_status == flow_status (equality)
        - Multi-flow components: bounded by flow sum with epsilon tolerance
        """
        if not self.components:
            return

        comp_status = self._variables['status']
        flow_status = self._flows_model._variables['status']
        mask = self._flow_mask
        n_flows = self._flow_count

        # Sum of flow statuses for each component: (component, time, ...)
        flow_sum = (flow_status * mask).sum('flow')

        # Separate single-flow vs multi-flow components
        single_flow_ids = [c.label for c in self.components if len(c.inputs) + len(c.outputs) == 1]
        multi_flow_ids = [c.label for c in self.components if len(c.inputs) + len(c.outputs) > 1]

        # Single-flow: exact equality
        if single_flow_ids:
            self.model.add_constraints(
                comp_status.sel(component=single_flow_ids) == flow_sum.sel(component=single_flow_ids),
                name='component|status|eq',
            )

        # Multi-flow: bounded constraints
        if multi_flow_ids:
            comp_status_multi = comp_status.sel(component=multi_flow_ids)
            flow_sum_multi = flow_sum.sel(component=multi_flow_ids)
            n_flows_multi = n_flows.sel(component=multi_flow_ids)

            # Upper bound: status <= sum(flow_statuses) + epsilon
            self.model.add_constraints(
                comp_status_multi <= flow_sum_multi + CONFIG.Modeling.epsilon,
                name='component|status|ub',
            )

            # Lower bound: status >= sum(flow_statuses) / (n + epsilon)
            self.model.add_constraints(
                comp_status_multi >= flow_sum_multi / (n_flows_multi + CONFIG.Modeling.epsilon),
                name='component|status|lb',
            )

        self._logger.debug(f'ComponentsModel created batched constraints for {len(self.components)} components')

    @cached_property
    def previous_status_batched(self) -> xr.DataArray | None:
        """Concatenated previous status (component, time) derived from component flows.

        Returns None if no components have previous status.
        For each component, previous status is OR of its flows' previous statuses.
        """
        previous_arrays = []
        components_with_previous = []

        for component in self.components:
            all_flows = component.inputs + component.outputs
            previous_status = []
            for flow in all_flows:
                prev = self._flows_model.get_previous_status(flow)
                if prev is not None:
                    previous_status.append(prev)

            if previous_status:
                # Combine flow statuses using OR (any flow active = component active)
                max_len = max(da.sizes['time'] for da in previous_status)
                padded = [
                    da.assign_coords(time=range(-da.sizes['time'], 0)).reindex(time=range(-max_len, 0), fill_value=0)
                    for da in previous_status
                ]
                comp_prev_status = xr.concat(padded, dim='flow').any(dim='flow').astype(int)
                comp_prev_status = comp_prev_status.expand_dims({self.dim_name: [component.label]})
                previous_arrays.append(comp_prev_status)
                components_with_previous.append(component)

        if not previous_arrays:
            return None

        return xr.concat(previous_arrays, dim=self.dim_name)

    def _get_previous_status_for_component(self, component) -> xr.DataArray | None:
        """Get previous status for a single component (OR of flow statuses).

        Args:
            component: The component to get previous status for.

        Returns:
            DataArray of previous status, or None if no flows have previous status.
        """
        all_flows = component.inputs + component.outputs
        previous_status = []
        for flow in all_flows:
            prev = self._flows_model.get_previous_status(flow)
            if prev is not None:
                previous_status.append(prev)

        if not previous_status:
            return None

        # Combine flow statuses using OR (any flow active = component active)
        max_len = max(da.sizes['time'] for da in previous_status)
        padded = [
            da.assign_coords(time=range(-da.sizes['time'], 0)).reindex(time=range(-max_len, 0), fill_value=0)
            for da in previous_status
        ]
        return xr.concat(padded, dim='flow').any(dim='flow').astype(int)

    def create_status_features(self) -> None:
        """Create status features (startup, shutdown, active_hours, etc.) using StatusHelpers."""
        if not self.components:
            return

        from .features import StatusHelpers

        # Use helper to create all status features with cached properties
        status_vars = StatusHelpers.create_status_features(
            model=self.model,
            status=self._variables['status'],
            params=self._status_params,
            dim_name=self.dim_name,
            var_names=ComponentVarName,
            previous_status=self._previous_status_dict,
            has_clusters=self.model.flow_system.clusters is not None,
        )

        # Store created variables
        self._status_variables = status_vars

        self._logger.debug(f'ComponentsModel created status features for {len(self.components)} components')

    def create_effect_shares(self) -> None:
        """No-op: effect shares are now collected centrally in EffectsModel.finalize_shares()."""
        pass

    # === Variable accessor properties ===

    @property
    def status(self) -> linopy.Variable | None:
        """Batched component status variable with (component, time) dims."""
        return self.model.variables['component|status'] if 'component|status' in self.model.variables else None

    def get_variable(self, var_name: str, component_id: str):
        """Get variable slice for a specific component."""
        dim = self.dim_name
        if var_name in self._variables:
            return self._variables[var_name].sel({dim: component_id})
        elif hasattr(self, '_status_variables') and var_name in self._status_variables:
            var = self._status_variables[var_name]
            if component_id in var.coords.get(dim, []):
                return var.sel({dim: component_id})
            return None
        else:
            raise KeyError(f'Variable {var_name} not found in ComponentsModel')


class PreventSimultaneousFlowsModel:
    """Type-level model for batched prevent_simultaneous_flows constraints.

    Handles mutual exclusivity constraints for components where flows cannot
    be active simultaneously (e.g., Storage charge/discharge, SourceAndSink buy/sell).

    Each constraint enforces: sum(flow_statuses) <= 1
    """

    def __init__(
        self,
        model: FlowSystemModel,
        components: list[Component],
        flows_model: FlowsModel,
    ):
        """Initialize the prevent simultaneous flows model.

        Args:
            model: The FlowSystemModel to create constraints in.
            components: List of components with prevent_simultaneous_flows set.
            flows_model: The FlowsModel that owns flow status variables.
        """
        self._logger = logging.getLogger('flixopt')
        self.model = model
        self.components = components
        self._flows_model = flows_model

        self._logger.debug(f'PreventSimultaneousFlowsModel initialized: {len(components)} components')

    @cached_property
    def _flow_mask(self) -> xr.DataArray:
        """(component, flow) mask: 1 if flow belongs to component's prevent_simultaneous_flows."""
        membership = MaskHelpers.build_flow_membership(
            self.components,
            lambda c: c.prevent_simultaneous_flows,
        )
        return MaskHelpers.build_mask(
            row_dim='component',
            row_ids=[c.label for c in self.components],
            col_dim='flow',
            col_ids=self._flows_model.element_ids,
            membership=membership,
        )

    def create_constraints(self) -> None:
        """Create batched mutual exclusivity constraints.

        Uses a mask matrix to batch all components into a single constraint:
        - mask: (component, flow) = 1 if flow in component's prevent_simultaneous_flows
        - status: (flow, time, ...)
        - (status * mask).sum('flow') <= 1 gives (component, time, ...) constraint
        """
        if not self.components:
            return

        status = self._flows_model._variables['status']
        mask = self._flow_mask

        # Batched constraint: sum of statuses for each component's flows <= 1
        # status * mask broadcasts to (component, flow, time, ...)
        # .sum('flow') reduces to (component, time, ...)
        self.model.add_constraints(
            (status * mask).sum('flow') <= 1,
            name='prevent_simultaneous',
        )

        self._logger.debug(
            f'PreventSimultaneousFlowsModel created batched constraint for {len(self.components)} components'
        )
