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
from .features import MaskHelpers, fast_notnull
from .interface import InvestParameters, StatusParameters
from .modeling import ModelingUtilitiesAbstract
from .structure import (
    ComponentVarName,
    ConverterVarName,
    Element,
    ElementType,
    FlowSystemModel,
    FlowVarName,
    TransmissionVarName,
    TypeModel,
    VariableType,
    register_class_for_io,
)

if TYPE_CHECKING:
    import linopy

    from .batched import FlowsData
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

    @property
    def data(self) -> FlowsData:
        """Access FlowsData from the batched accessor."""
        return self.model.flow_system.batched.flows

    # === Variables (cached_property) ===

    @cached_property
    def rate(self) -> linopy.Variable:
        """(flow, time, ...) - flow rate variable for ALL flows."""
        # Reindex bounds to match coords flow order (FlowsData uses sorted order, TypeModel uses insertion order)
        flow_order = self._build_coords(dims=None)[self.dim_name]
        lower = self.data.absolute_lower_bounds.reindex({self.dim_name: flow_order})
        upper = self.data.absolute_upper_bounds.reindex({self.dim_name: flow_order})
        return self.add_variables('rate', VariableType.FLOW_RATE, lower=lower, upper=upper, dims=None)

    @cached_property
    def status(self) -> linopy.Variable | None:
        """(flow, time, ...) - binary status variable, masked to flows with status."""
        if not self.data.with_status:
            return None
        return self.add_variables(
            'status',
            VariableType.STATUS,
            dims=None,
            mask=self.data.has_status,
            binary=True,
        )

    @cached_property
    def size(self) -> linopy.Variable | None:
        """(flow, period, scenario) - size variable, masked to flows with investment."""
        if not self.data.with_investment:
            return None
        # Reindex bounds to match TypeModel's flow order (FlowsData uses sorted order)
        flow_order = self._build_coords(dims=('period', 'scenario'))[self.dim_name]
        lower = self.data.size_minimum_all.reindex({self.dim_name: flow_order})
        upper = self.data.size_maximum_all.reindex({self.dim_name: flow_order})
        return self.add_variables(
            'size',
            VariableType.FLOW_SIZE,
            lower=lower,
            upper=upper,
            dims=('period', 'scenario'),
            mask=self.data.has_investment,
        )

    @cached_property
    def invested(self) -> linopy.Variable | None:
        """(flow, period, scenario) - binary invested variable, masked to optional investment."""
        if not self.data.with_optional_investment:
            return None
        return self.add_variables(
            'invested',
            dims=('period', 'scenario'),
            mask=self.data.has_optional_investment,
            binary=True,
        )

    def create_variables(self) -> None:
        """Create all batched variables for flows.

        Triggers cached property creation for:
        - flow|rate: For ALL flows
        - flow|status: For flows with status_parameters
        - flow|size: For flows with investment
        - flow|invested: For flows with optional investment
        """
        # Trigger variable creation via cached properties
        _ = self.rate
        _ = self.status
        _ = self.size
        _ = self.invested

        logger.debug(
            f'FlowsModel created variables: {len(self.elements)} flows, '
            f'{len(self.data.with_status)} with status, {len(self.data.with_investment)} with investment'
        )

    def create_constraints(self) -> None:
        """Create all batched constraints for flows."""
        # Trigger investment variable creation first (cached properties)
        # These must exist before rate bounds constraints that reference them
        _ = self.size  # Creates size variable if with_investment
        _ = self.invested  # Creates invested variable if with_optional_investment

        self.constraint_flow_hours()
        self.constraint_flow_hours_over_periods()
        self.constraint_load_factor()
        self.constraint_rate_bounds()
        self.constraint_investment()

        logger.debug(f'FlowsModel created {len(self._constraints)} constraint types')

    def constraint_investment(self) -> None:
        """Investment constraints: optional size bounds, linked periods, piecewise effects."""
        if self.size is None:
            return

        from .features import InvestmentHelpers

        dim = self.dim_name

        # Optional investment: size controlled by invested binary
        if self.invested is not None:
            InvestmentHelpers.add_optional_size_bounds(
                model=self.model,
                size_var=self.size,
                invested_var=self.invested,
                min_bounds=self.data.optional_investment_size_minimum,
                max_bounds=self.data.optional_investment_size_maximum,
                element_ids=self.data.with_optional_investment,
                dim_name=dim,
                name_prefix='flow',
            )

        # Linked periods constraints
        InvestmentHelpers.add_linked_periods_constraints(
            model=self.model,
            size_var=self.size,
            params=self.data.invest_params,
            element_ids=self.data.with_investment,
            dim_name=dim,
        )

        # Piecewise effects
        self._create_piecewise_effects()

    # === Constraints (methods with constraint_* naming) ===

    def constraint_flow_hours(self) -> None:
        """Constrain sum_temporal(rate) for flows with flow_hours bounds."""
        dim = self.dim_name

        # Min constraint
        if self.data.flow_hours_minimum is not None:
            flow_ids = self.data.with_flow_hours_min
            hours = self.model.sum_temporal(self.rate.sel({dim: flow_ids}))
            self.add_constraints(hours >= self.data.flow_hours_minimum, name='hours_min')

        # Max constraint
        if self.data.flow_hours_maximum is not None:
            flow_ids = self.data.with_flow_hours_max
            hours = self.model.sum_temporal(self.rate.sel({dim: flow_ids}))
            self.add_constraints(hours <= self.data.flow_hours_maximum, name='hours_max')

    def constraint_flow_hours_over_periods(self) -> None:
        """Constrain weighted sum of hours across periods."""
        dim = self.dim_name

        def compute_hours_over_periods(flow_ids: list[str]):
            rate_subset = self.rate.sel({dim: flow_ids})
            hours_per_period = self.model.sum_temporal(rate_subset)
            if self.model.flow_system.periods is not None:
                period_weights = self.model.flow_system.weights.get('period', 1)
                return (hours_per_period * period_weights).sum('period')
            return hours_per_period

        # Min constraint
        if self.data.flow_hours_minimum_over_periods is not None:
            flow_ids = self.data.with_flow_hours_over_periods_min
            hours = compute_hours_over_periods(flow_ids)
            self.add_constraints(hours >= self.data.flow_hours_minimum_over_periods, name='hours_over_periods_min')

        # Max constraint
        if self.data.flow_hours_maximum_over_periods is not None:
            flow_ids = self.data.with_flow_hours_over_periods_max
            hours = compute_hours_over_periods(flow_ids)
            self.add_constraints(hours <= self.data.flow_hours_maximum_over_periods, name='hours_over_periods_max')

    def constraint_load_factor(self) -> None:
        """Load factor min/max constraints for flows that have them."""
        dim = self.dim_name
        total_time = self.model.timestep_duration.sum(self.model.temporal_dims)

        # Min constraint: hours >= total_time * load_factor_min * size
        if self.data.load_factor_minimum is not None:
            flow_ids = self.data.with_load_factor_min
            hours = self.model.sum_temporal(self.rate.sel({dim: flow_ids}))
            size = self.data.effective_size_lower.sel({dim: flow_ids}).fillna(0)
            rhs = total_time * self.data.load_factor_minimum * size
            self.add_constraints(hours >= rhs, name='load_factor_min')

        # Max constraint: hours <= total_time * load_factor_max * size
        if self.data.load_factor_maximum is not None:
            flow_ids = self.data.with_load_factor_max
            hours = self.model.sum_temporal(self.rate.sel({dim: flow_ids}))
            size = self.data.effective_size_upper.sel({dim: flow_ids}).fillna(np.inf)
            rhs = total_time * self.data.load_factor_maximum * size
            self.add_constraints(hours <= rhs, name='load_factor_max')

    def __init__(self, model: FlowSystemModel, elements: list[Flow]):
        """Initialize the type-level model for all flows.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            elements: List of all Flow elements to model.
        """
        super().__init__(model, elements)

        # Set reference on each flow element for element access pattern
        for flow in elements:
            flow.set_flows_model(self)

    @property
    def _previous_status(self) -> dict[str, xr.DataArray]:
        """Previous status for flows that have it, keyed by label_full.

        Delegates to FlowsData.previous_states.
        """
        return self.data.previous_states

    def _build_constraint_mask(self, selected_ids: set[str], reference_var: linopy.Variable) -> xr.DataArray:
        """Build a mask for constraint creation from selected flow IDs.

        Args:
            selected_ids: Set of flow IDs to include (mask=True).
            reference_var: Variable whose dimensions the mask should match.

        Returns:
            Boolean DataArray matching reference_var dimensions, True where flow ID is in selected_ids.
        """
        dim = self.dim_name
        flow_ids = self.element_ids

        # Build 1D mask
        mask = xr.DataArray(
            [fid in selected_ids for fid in flow_ids],
            dims=[dim],
            coords={dim: flow_ids},
        )

        # Broadcast to match reference variable dimensions
        for d in reference_var.dims:
            if d != dim and d not in mask.dims:
                mask = mask.expand_dims({d: reference_var.coords[d]})
        return mask.transpose(*reference_var.dims)

    def constraint_rate_bounds(self) -> None:
        """Create flow rate bounding constraints based on status/investment configuration."""
        # Group flow IDs by their constraint type
        status_set = set(self.data.with_status)
        investment_set = set(self.data.with_investment)
        without_size_set = set(self.data.without_size)

        # 1. Status only (no investment) - exclude flows with size=None (bounds come from converter)
        status_only_ids = list(status_set - investment_set - without_size_set)
        if status_only_ids:
            self._constraint_status_bounds()

        # 2. Investment only (no status)
        invest_only_ids = [fid for fid in self.data.with_investment if fid not in status_set]
        if invest_only_ids:
            self._constraint_investment_bounds()

        # 3. Both status and investment
        both_ids = [fid for fid in self.data.with_status if fid in investment_set]
        if both_ids:
            self._constraint_status_investment_bounds()

    def _constraint_investment_bounds(self) -> None:
        """
        Case: With investment, without status.
        rate <= size * relative_max, rate >= size * relative_min.

        Uses mask-based constraint creation - creates constraints for all flows but
        masks out non-investment flows.
        """
        dim = self.dim_name
        flow_ids = self.element_ids

        # Build mask: True for investment flows without status
        invest_only_ids = set(self.data.with_investment) - set(self.data.with_status)
        mask = self._build_constraint_mask(invest_only_ids, self.rate)

        if not mask.any():
            return

        # Reindex data to match flow_ids order (FlowsData uses sorted order)
        rel_max = self.data.effective_relative_maximum.reindex({dim: flow_ids})
        rel_min = self.data.effective_relative_minimum.reindex({dim: flow_ids})

        # Upper bound: rate <= size * relative_max
        self.model.add_constraints(
            self.rate <= self.size * rel_max,
            name=f'{dim}|invest_ub',
            mask=mask,
        )

        # Lower bound: rate >= size * relative_min
        self.model.add_constraints(
            self.rate >= self.size * rel_min,
            name=f'{dim}|invest_lb',
            mask=mask,
        )

    def _constraint_status_bounds(self) -> None:
        """
        Case: With status, without investment.
        rate <= status * size * relative_max, rate >= status * epsilon."""
        flow_ids = sorted(
            [fid for fid in set(self.data.with_status) - set(self.data.with_investment) - set(self.data.without_size)]
        )
        dim = self.dim_name
        flow_rate = self.rate.sel({dim: flow_ids})
        status = self.status.sel({dim: flow_ids})

        # Get effective relative bounds and fixed size for the subset
        rel_max = self.data.effective_relative_maximum.sel({dim: flow_ids})
        rel_min = self.data.effective_relative_minimum.sel({dim: flow_ids})
        size = self.data.fixed_size.sel({dim: flow_ids})

        # Upper bound: rate <= status * size * relative_max
        upper_bounds = rel_max * size
        self.add_constraints(flow_rate <= status * upper_bounds, name='status_ub')

        # Lower bound: rate >= status * max(epsilon, size * relative_min)
        lower_bounds = np.maximum(CONFIG.Modeling.epsilon, rel_min * size)
        self.add_constraints(flow_rate >= status * lower_bounds, name='status_lb')

    def _constraint_status_investment_bounds(self) -> None:
        """Bounds for flows with both status and investment.

        Three constraints:
        1. rate <= status * M (big-M): forces status=1 when rate>0
        2. rate <= size * rel_max: limits rate by actual invested size
        3. rate >= (status - 1) * M + size * rel_min: enforces minimum when status=1
        """
        flow_ids = sorted([fid for fid in set(self.data.with_investment) & set(self.data.with_status)])
        dim = self.dim_name
        flow_rate = self.rate.sel({dim: flow_ids})
        size = self.size.sel({dim: flow_ids})
        status = self.status.sel({dim: flow_ids})

        # Get effective relative bounds and effective_size_upper for the subset
        rel_max = self.data.effective_relative_maximum.sel({dim: flow_ids})
        rel_min = self.data.effective_relative_minimum.sel({dim: flow_ids})
        max_size = self.data.effective_size_upper.sel({dim: flow_ids})

        # Upper bound 1: rate <= status * M where M = max_size * relative_max
        big_m_upper = max_size * rel_max
        self.add_constraints(flow_rate <= status * big_m_upper, name='status+invest_ub1')

        # Upper bound 2: rate <= size * relative_max
        self.add_constraints(flow_rate <= size * rel_max, name='status+invest_ub2')

        # Lower bound: rate >= (status - 1) * M + size * relative_min
        big_m_lower = max_size * rel_min
        rhs = (status - 1) * big_m_lower + size * rel_min
        self.add_constraints(flow_rate >= rhs, name='status+invest_lb')

    def _create_piecewise_effects(self) -> None:
        """Create batched piecewise effects for flows with piecewise_effects_of_investment.

        Uses PiecewiseHelpers for pad-to-max batching across all flows with
        piecewise effects. Creates batched segment variables, share variables,
        and coupling constraints.
        """
        from .features import PiecewiseHelpers

        dim = self.dim_name
        size_var = self._variables.get('size')
        invested_var = self._variables.get('invested')

        if size_var is None:
            return

        # Find flows with piecewise effects
        invest_params = self.data.invest_params
        with_piecewise = [
            fid for fid in self.data.with_investment if invest_params[fid].piecewise_effects_of_investment is not None
        ]

        if not with_piecewise:
            return

        element_ids = with_piecewise

        # Collect segment counts
        segment_counts = {
            fid: len(invest_params[fid].piecewise_effects_of_investment.piecewise_origin) for fid in with_piecewise
        }

        # Build segment mask
        max_segments, segment_mask = PiecewiseHelpers.collect_segment_info(element_ids, segment_counts, dim)

        # Collect origin breakpoints (for size)
        origin_breakpoints = {}
        for fid in with_piecewise:
            piecewise_origin = invest_params[fid].piecewise_effects_of_investment.piecewise_origin
            starts = [p.start for p in piecewise_origin]
            ends = [p.end for p in piecewise_origin]
            origin_breakpoints[fid] = (starts, ends)

        origin_starts, origin_ends = PiecewiseHelpers.pad_breakpoints(
            element_ids, origin_breakpoints, max_segments, dim
        )

        # Collect all effect names across all flows
        all_effect_names: set[str] = set()
        for fid in with_piecewise:
            shares = invest_params[fid].piecewise_effects_of_investment.piecewise_shares
            all_effect_names.update(shares.keys())

        # Collect breakpoints for each effect
        effect_breakpoints: dict[str, tuple[xr.DataArray, xr.DataArray]] = {}
        for effect_name in all_effect_names:
            breakpoints = {}
            for fid in with_piecewise:
                shares = invest_params[fid].piecewise_effects_of_investment.piecewise_shares
                if effect_name in shares:
                    piecewise = shares[effect_name]
                    starts = [p.start for p in piecewise]
                    ends = [p.end for p in piecewise]
                else:
                    # This flow doesn't have this effect - use NaN (will be masked)
                    starts = [0.0] * segment_counts[fid]
                    ends = [0.0] * segment_counts[fid]
                breakpoints[fid] = (starts, ends)

            starts, ends = PiecewiseHelpers.pad_breakpoints(element_ids, breakpoints, max_segments, dim)
            effect_breakpoints[effect_name] = (starts, ends)

        # Create batched piecewise variables
        base_coords = self.model.get_coords(['period', 'scenario'])
        name_prefix = f'{dim}|piecewise_effects'  # Tied to element type (flow)
        piecewise_vars = PiecewiseHelpers.create_piecewise_variables(
            self.model,
            element_ids,
            max_segments,
            dim,
            segment_mask,
            base_coords,
            name_prefix,
        )

        # Build zero_point array if any flows are non-mandatory
        zero_point = None
        if invested_var is not None:
            non_mandatory_ids = [fid for fid in element_ids if not invest_params[fid].mandatory]
            if non_mandatory_ids:
                # Select invested for non-mandatory flows in this batch
                available_ids = [fid for fid in non_mandatory_ids if fid in invested_var.coords.get(dim, [])]
                if available_ids:
                    zero_point = invested_var.sel({dim: element_ids})

        # Create piecewise constraints
        PiecewiseHelpers.create_piecewise_constraints(
            self.model,
            piecewise_vars,
            segment_mask,
            zero_point,
            dim,
            name_prefix,
        )

        # Create coupling constraint for size (origin)
        size_subset = size_var.sel({dim: element_ids})
        PiecewiseHelpers.create_coupling_constraint(
            self.model,
            size_subset,
            piecewise_vars['lambda0'],
            piecewise_vars['lambda1'],
            origin_starts,
            origin_ends,
            f'{name_prefix}|size|coupling',
        )

        # Create share variables and coupling constraints for each effect

        coords_dict = {dim: pd.Index(element_ids, name=dim)}
        if base_coords is not None:
            coords_dict.update(dict(base_coords))
        share_coords = xr.Coordinates(coords_dict)

        for effect_name in all_effect_names:
            # Create batched share variable
            share_var = self.model.add_variables(
                lower=-np.inf,  # Shares can be negative (e.g., costs)
                upper=np.inf,
                coords=share_coords,
                name=f'{name_prefix}|{effect_name}',
            )

            # Create coupling constraint for this share
            starts, ends = effect_breakpoints[effect_name]
            PiecewiseHelpers.create_coupling_constraint(
                self.model,
                share_var,
                piecewise_vars['lambda0'],
                piecewise_vars['lambda1'],
                starts,
                ends,
                f'{name_prefix}|{effect_name}|coupling',
            )

            # Add to effects (sum over element dimension for periodic share)
            self.model.effects.add_share_to_effects(
                name=f'{name_prefix}|{effect_name}',
                expressions={effect_name: share_var.sum(dim)},
                target='periodic',
            )

        logger.debug(f'Created batched piecewise effects for {len(element_ids)} flows')

    # === Effect properties (used by EffectsModel) ===
    # Investment effect properties are defined below, delegating to data._investment_data

    @property
    def effects_per_active_hour(self) -> xr.DataArray | None:
        """Combined effects_per_active_hour with (flow, effect) dims."""
        return self.data.effects_per_active_hour

    @property
    def effects_per_startup(self) -> xr.DataArray | None:
        """Combined effects_per_startup with (flow, effect) dims."""
        return self.data.effects_per_startup

    def add_effect_contributions(self, effects_model) -> None:
        """Register effect contributions with EffectsModel.

        Called by EffectsModel.finalize_shares() to collect contributions from FlowsModel.
        Adds temporal contributions (status effects) to effect|per_timestep constraint.

        Args:
            effects_model: The EffectsModel to register contributions with.
        """
        if self.status is None:
            return

        dim = self.dim_name
        dt = self.model.timestep_duration

        # Effects per active hour: status * factor * dt
        factor = self.data.effects_per_active_hour
        if factor is not None:
            flow_ids = factor.coords[dim].values
            status_subset = self.status.sel({dim: flow_ids})
            effects_model.add_temporal_contribution((status_subset * factor * dt).sum(dim))

        # Effects per startup: startup * factor
        factor = self.data.effects_per_startup
        if self.startup is not None and factor is not None:
            flow_ids = factor.coords[dim].values
            startup_subset = self.startup.sel({dim: flow_ids})
            effects_model.add_temporal_contribution((startup_subset * factor).sum(dim))

    # === Status Variables (cached_property) ===

    @cached_property
    def active_hours(self) -> linopy.Variable | None:
        """(flow, period, scenario) - total active hours for flows with status."""
        sd = self.data
        if not sd.with_status:
            return None

        dim = self.dim_name
        params = sd.status_params
        total_hours = self.model.temporal_weight.sum(self.model.temporal_dims)

        min_vals = [params[eid].active_hours_min or 0 for eid in sd.with_status]
        max_list = [params[eid].active_hours_max for eid in sd.with_status]
        lower = xr.DataArray(min_vals, dims=[dim], coords={dim: sd.with_status})
        has_max = xr.DataArray([v is not None for v in max_list], dims=[dim], coords={dim: sd.with_status})
        raw_max = xr.DataArray([v if v is not None else 0 for v in max_list], dims=[dim], coords={dim: sd.with_status})
        upper = xr.where(has_max, raw_max, total_hours)

        return self.add_variables(
            'active_hours',
            lower=lower,
            upper=upper,
            dims=('period', 'scenario'),
            element_ids=sd.with_status,
        )

    @cached_property
    def startup(self) -> linopy.Variable | None:
        """(flow, time, ...) - binary startup variable."""
        ids = self.data.with_startup_tracking
        if not ids:
            return None
        return self.add_variables('startup', dims=None, element_ids=ids, binary=True)

    @cached_property
    def shutdown(self) -> linopy.Variable | None:
        """(flow, time, ...) - binary shutdown variable."""
        ids = self.data.with_startup_tracking
        if not ids:
            return None
        return self.add_variables('shutdown', dims=None, element_ids=ids, binary=True)

    @cached_property
    def inactive(self) -> linopy.Variable | None:
        """(flow, time, ...) - binary inactive variable."""
        ids = self.data.with_downtime_tracking
        if not ids:
            return None
        return self.add_variables('inactive', dims=None, element_ids=ids, binary=True)

    @cached_property
    def startup_count(self) -> linopy.Variable | None:
        """(flow, period, scenario) - startup count."""
        ids = self.data.with_startup_limit
        if not ids:
            return None
        return self.add_variables(
            'startup_count',
            lower=0,
            upper=self.data.startup_limit_values,
            dims=('period', 'scenario'),
            element_ids=ids,
        )

    @cached_property
    def uptime(self) -> linopy.Variable | None:
        """(flow, time, ...) - consecutive uptime duration."""
        sd = self.data
        if not sd.with_uptime_tracking:
            return None
        from .features import StatusHelpers

        prev = sd.previous_uptime
        var = StatusHelpers.add_batched_duration_tracking(
            model=self.model,
            state=self.status.sel({self.dim_name: sd.with_uptime_tracking}),
            name=FlowVarName.UPTIME,
            dim_name=self.dim_name,
            timestep_duration=self.model.timestep_duration,
            minimum_duration=sd.min_uptime,
            maximum_duration=sd.max_uptime,
            previous_duration=prev if prev is not None and fast_notnull(prev).any() else None,
        )
        self._variables['uptime'] = var
        return var

    @cached_property
    def downtime(self) -> linopy.Variable | None:
        """(flow, time, ...) - consecutive downtime duration."""
        sd = self.data
        if not sd.with_downtime_tracking:
            return None
        from .features import StatusHelpers

        prev = sd.previous_downtime
        var = StatusHelpers.add_batched_duration_tracking(
            model=self.model,
            state=self.inactive,
            name=FlowVarName.DOWNTIME,
            dim_name=self.dim_name,
            timestep_duration=self.model.timestep_duration,
            minimum_duration=sd.min_downtime,
            maximum_duration=sd.max_downtime,
            previous_duration=prev if prev is not None and fast_notnull(prev).any() else None,
        )
        self._variables['downtime'] = var
        return var

    # === Status Constraints ===

    def _status_sel(self, element_ids: list[str]) -> linopy.Variable:
        """Select status variable for a subset of element IDs."""
        return self.status.sel({self.dim_name: element_ids})

    def constraint_active_hours(self) -> None:
        """Constrain active_hours == sum_temporal(status)."""
        if self.active_hours is None:
            return
        self.model.add_constraints(
            self.active_hours == self.model.sum_temporal(self.status),
            name=FlowVarName.Constraint.ACTIVE_HOURS,
        )

    def constraint_complementary(self) -> None:
        """Constrain status + inactive == 1 for downtime tracking flows."""
        if self.inactive is None:
            return
        self.model.add_constraints(
            self._status_sel(self.data.with_downtime_tracking) + self.inactive == 1,
            name=FlowVarName.Constraint.COMPLEMENTARY,
        )

    def constraint_switch_transition(self) -> None:
        """Constrain startup[t] - shutdown[t] == status[t] - status[t-1] for t > 0."""
        if self.startup is None:
            return
        status = self._status_sel(self.data.with_startup_tracking)
        self.model.add_constraints(
            self.startup.isel(time=slice(1, None)) - self.shutdown.isel(time=slice(1, None))
            == status.isel(time=slice(1, None)) - status.isel(time=slice(None, -1)),
            name=FlowVarName.Constraint.SWITCH_TRANSITION,
        )

    def constraint_switch_mutex(self) -> None:
        """Constrain startup + shutdown <= 1."""
        if self.startup is None:
            return
        self.model.add_constraints(
            self.startup + self.shutdown <= 1,
            name=FlowVarName.Constraint.SWITCH_MUTEX,
        )

    def constraint_switch_initial(self) -> None:
        """Constrain startup[0] - shutdown[0] == status[0] - previous_status[-1]."""
        if self.startup is None:
            return
        dim = self.dim_name
        ids = [eid for eid in self.data.with_startup_tracking if eid in self._previous_status]
        if not ids:
            return

        prev_arrays = [self._previous_status[eid].expand_dims({dim: [eid]}) for eid in ids]
        prev_state = xr.concat(prev_arrays, dim=dim).isel(time=-1)

        self.model.add_constraints(
            self.startup.sel({dim: ids}).isel(time=0) - self.shutdown.sel({dim: ids}).isel(time=0)
            == self._status_sel(ids).isel(time=0) - prev_state,
            name=FlowVarName.Constraint.SWITCH_INITIAL,
        )

    def constraint_startup_count(self) -> None:
        """Constrain startup_count == sum(startup) over temporal dims."""
        if self.startup_count is None:
            return
        dim = self.dim_name
        startup_subset = self.startup.sel({dim: self.data.with_startup_limit})
        temporal_dims = [d for d in startup_subset.dims if d not in ('period', 'scenario', dim)]
        self.model.add_constraints(
            self.startup_count == startup_subset.sum(temporal_dims),
            name=FlowVarName.Constraint.STARTUP_COUNT,
        )

    def constraint_cluster_cyclic(self) -> None:
        """Constrain status[0] == status[-1] for cyclic cluster mode."""
        if self.model.flow_system.clusters is None:
            return
        params = self.data.status_params
        cyclic_ids = [eid for eid in self.data.with_status if params[eid].cluster_mode == 'cyclic']
        if not cyclic_ids:
            return
        status = self._status_sel(cyclic_ids)
        self.model.add_constraints(
            status.isel(time=0) == status.isel(time=-1),
            name=FlowVarName.Constraint.CLUSTER_CYCLIC,
        )

    def create_status_model(self) -> None:
        """Create status variables and constraints for flows with status.

        Triggers cached property creation for all status variables and calls
        individual constraint methods.

        Creates:
        - flow|active_hours: For all flows with status
        - flow|startup, flow|shutdown: For flows needing startup tracking
        - flow|inactive: For flows needing downtime tracking
        - flow|startup_count: For flows with startup limit
        - flow|uptime, flow|downtime: Duration tracking variables

        Must be called AFTER create_variables() and create_constraints().
        """
        if not self.data.with_status:
            return

        # Trigger variable creation via cached properties
        _ = self.active_hours
        _ = self.startup
        _ = self.shutdown
        _ = self.inactive
        _ = self.startup_count
        _ = self.uptime
        _ = self.downtime

        # Create constraints
        self.constraint_active_hours()
        self.constraint_complementary()
        self.constraint_switch_transition()
        self.constraint_switch_mutex()
        self.constraint_switch_initial()
        self.constraint_startup_count()
        self.constraint_cluster_cyclic()

    @property
    def effects_per_flow_hour(self) -> xr.DataArray | None:
        """Combined effect factors with (flow, effect, ...) dims."""
        return self.data.effects_per_flow_hour

    # --- Investment Effect Properties (delegating to _investment_data) ---

    @property
    def effects_per_size(self) -> xr.DataArray | None:
        """(flow, effect) - effects per unit size."""
        inv = self.data._investment_data
        return inv.effects_per_size if inv else None

    @property
    def effects_of_investment(self) -> xr.DataArray | None:
        """(flow, effect) - fixed effects of investment (optional only)."""
        inv = self.data._investment_data
        return inv.effects_of_investment if inv else None

    @property
    def effects_of_retirement(self) -> xr.DataArray | None:
        """(flow, effect) - effects of retirement (optional only)."""
        inv = self.data._investment_data
        return inv.effects_of_retirement if inv else None

    @property
    def effects_of_investment_mandatory(self) -> list[tuple[str, dict[str, float | xr.DataArray]]]:
        """List of (element_id, effects_dict) for mandatory investments with fixed effects."""
        inv = self.data._investment_data
        return inv.effects_of_investment_mandatory if inv else []

    @property
    def effects_of_retirement_constant(self) -> list[tuple[str, dict[str, float | xr.DataArray]]]:
        """List of (element_id, effects_dict) for retirement constant parts."""
        inv = self.data._investment_data
        return inv.effects_of_retirement_constant if inv else []

    @property
    def investment_ids(self) -> list[str]:
        """IDs of flows with investment parameters (alias for data.with_investment)."""
        return self.data.with_investment

    # --- Previous Status ---

    @cached_property
    def previous_status_batched(self) -> xr.DataArray | None:
        """Concatenated previous status (flow, time) from previous_flow_rate."""
        with_previous = self.data.with_previous_flow_rate
        if not with_previous:
            return None

        previous_arrays = []
        for fid in with_previous:
            previous_flow_rate = self.data[fid].previous_flow_rate

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
            previous_status = previous_status.expand_dims({self.dim_name: [fid]})
            previous_arrays.append(previous_status)

        return xr.concat(previous_arrays, dim=self.dim_name)

    def get_previous_status(self, flow: Flow) -> xr.DataArray | None:
        """Get previous status for a specific flow.

        Args:
            flow: The Flow element to get previous status for.

        Returns:
            DataArray of previous status (time dimension), or None if no previous status.
        """
        fid = flow.label_full
        return self._previous_status.get(fid)


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
            self.add_variables(
                'virtual_supply',
                VariableType.VIRTUAL_FLOW,
                lower=0.0,
                dims=self.model.temporal_dims,
                element_ids=self.imbalance_ids,
            )

            # virtual_demand: allows removing excess flow
            self.add_variables(
                'virtual_demand',
                VariableType.VIRTUAL_FLOW,
                lower=0.0,
                dims=self.model.temporal_dims,
                element_ids=self.imbalance_ids,
            )

        logger.debug(
            f'BusesModel created variables: {len(self.elements)} buses, {len(self.buses_with_imbalance)} with imbalance'
        )

    def create_constraints(self) -> None:
        """Create all batched constraints for buses.

        Creates:
        - bus|balance: Sum(inputs) - Sum(outputs) == 0 for all buses
        - With virtual_supply/demand adjustment for buses with imbalance

        Uses dense coefficient matrix approach for fast vectorized computation.
        The coefficient matrix has +1 for inputs, -1 for outputs, 0 for unconnected flows.
        """
        flow_rate = self._flows_model._variables['rate']
        flow_dim = self._flows_model.dim_name  # 'flow'
        bus_dim = self.dim_name  # 'bus'

        # Get ordered lists for coefficient matrix
        bus_ids = list(self.elements.keys())
        flow_ids = list(flow_rate.coords[flow_dim].values)

        if not bus_ids or not flow_ids:
            logger.debug('BusesModel: no buses or flows, skipping balance constraints')
            return

        # Build coefficient matrix: +1 for inputs, -1 for outputs, 0 otherwise
        coeffs = np.zeros((len(bus_ids), len(flow_ids)), dtype=np.float64)
        for i, bus in enumerate(self.elements.values()):
            for f in bus.inputs:
                coeffs[i, flow_ids.index(f.label_full)] = 1.0
            for f in bus.outputs:
                coeffs[i, flow_ids.index(f.label_full)] = -1.0

        coeffs_da = xr.DataArray(coeffs, dims=[bus_dim, flow_dim], coords={bus_dim: bus_ids, flow_dim: flow_ids})

        # Balance = sum(inputs) - sum(outputs)
        balance = (coeffs_da * flow_rate).sum(flow_dim)

        if self.buses_with_imbalance:
            imbalance_ids = [b.label_full for b in self.buses_with_imbalance]
            is_imbalance = xr.DataArray(
                [b in imbalance_ids for b in bus_ids], dims=[bus_dim], coords={bus_dim: bus_ids}
            )

            # Buses without imbalance: balance == 0
            self.model.add_constraints(balance == 0, name='bus|balance', mask=~is_imbalance)

            # Buses with imbalance: balance + virtual_supply - virtual_demand == 0
            balance_imbalance = balance.sel({bus_dim: imbalance_ids})
            virtual_balance = balance_imbalance + self._variables['virtual_supply'] - self._variables['virtual_demand']
            self.model.add_constraints(virtual_balance == 0, name='bus|balance_imbalance')
        else:
            self.model.add_constraints(balance == 0, name='bus|balance')

        logger.debug(f'BusesModel created batched balance constraint for {len(bus_ids)} buses')

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


class ComponentsModel(TypeModel):
    """Type-level model for component status variables and constraints.

    This handles component status for components with status_parameters:
    - Status variables and constraints linking component status to flow statuses
    - Status features (startup, shutdown, active_hours, etc.)

    Component status is derived from flow statuses:
    - Single-flow component: status == flow_status
    - Multi-flow component: status is 1 if ANY flow is active

    Note:
        Piecewise conversion is handled by ConvertersModel.
        Transmission constraints are handled by TransmissionsModel.
    """

    element_type = ElementType.COMPONENT

    def __init__(
        self,
        model: FlowSystemModel,
        components_with_status: list[Component],
        flows_model: FlowsModel,
    ):
        super().__init__(model, components_with_status)
        self._logger = logging.getLogger('flixopt')
        self._flows_model = flows_model
        self._logger.debug(f'ComponentsModel initialized: {len(components_with_status)} with status')

    @property
    def components(self) -> list[Component]:
        """List of components with status (alias for elements.values())."""
        return list(self.elements.values())

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
    def _status_data(self):
        """StatusData instance for component status."""
        from .batched import StatusData

        return StatusData(
            params=self._status_params,
            dim_name=self.dim_name,
            effect_ids=list(self.model.flow_system.effects.keys()),
            timestep_duration=self.model.timestep_duration,
            previous_states=self._previous_status_dict,
        )

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

        self.add_variables('status', dims=None, binary=True)
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

    # === Status Variables (cached_property) ===

    @cached_property
    def active_hours(self) -> linopy.Variable | None:
        """(component, period, scenario) - total active hours for components with status."""
        if not self.components:
            return None

        sd = self._status_data
        dim = self.dim_name
        total_hours = self.model.temporal_weight.sum(self.model.temporal_dims)

        min_vals = [sd._params[eid].active_hours_min or 0 for eid in sd.ids]
        max_list = [sd._params[eid].active_hours_max for eid in sd.ids]
        lower = xr.DataArray(min_vals, dims=[dim], coords={dim: sd.ids})
        has_max = xr.DataArray([v is not None for v in max_list], dims=[dim], coords={dim: sd.ids})
        raw_max = xr.DataArray([v if v is not None else 0 for v in max_list], dims=[dim], coords={dim: sd.ids})
        upper = xr.where(has_max, raw_max, total_hours)

        return self.add_variables(
            'active_hours',
            lower=lower,
            upper=upper,
            dims=('period', 'scenario'),
            element_ids=sd.ids,
        )

    @cached_property
    def startup(self) -> linopy.Variable | None:
        """(component, time, ...) - binary startup variable."""
        ids = self._status_data.with_startup_tracking
        if not ids:
            return None
        return self.add_variables('startup', dims=None, element_ids=ids, binary=True)

    @cached_property
    def shutdown(self) -> linopy.Variable | None:
        """(component, time, ...) - binary shutdown variable."""
        ids = self._status_data.with_startup_tracking
        if not ids:
            return None
        return self.add_variables('shutdown', dims=None, element_ids=ids, binary=True)

    @cached_property
    def inactive(self) -> linopy.Variable | None:
        """(component, time, ...) - binary inactive variable."""
        ids = self._status_data.with_downtime_tracking
        if not ids:
            return None
        return self.add_variables('inactive', dims=None, element_ids=ids, binary=True)

    @cached_property
    def startup_count(self) -> linopy.Variable | None:
        """(component, period, scenario) - startup count."""
        ids = self._status_data.with_startup_limit
        if not ids:
            return None
        return self.add_variables(
            'startup_count',
            lower=0,
            upper=self._status_data.startup_limit,
            dims=('period', 'scenario'),
            element_ids=ids,
        )

    @cached_property
    def uptime(self) -> linopy.Variable | None:
        """(component, time, ...) - consecutive uptime duration."""
        sd = self._status_data
        if not sd.with_uptime_tracking:
            return None
        from .features import StatusHelpers

        prev = sd.previous_uptime
        var = StatusHelpers.add_batched_duration_tracking(
            model=self.model,
            state=self._variables['status'].sel({self.dim_name: sd.with_uptime_tracking}),
            name=ComponentVarName.UPTIME,
            dim_name=self.dim_name,
            timestep_duration=self.model.timestep_duration,
            minimum_duration=sd.min_uptime,
            maximum_duration=sd.max_uptime,
            previous_duration=prev if prev is not None and fast_notnull(prev).any() else None,
        )
        self._variables['uptime'] = var
        return var

    @cached_property
    def downtime(self) -> linopy.Variable | None:
        """(component, time, ...) - consecutive downtime duration."""
        sd = self._status_data
        if not sd.with_downtime_tracking:
            return None
        from .features import StatusHelpers

        _ = self.inactive  # ensure inactive variable exists
        prev = sd.previous_downtime
        var = StatusHelpers.add_batched_duration_tracking(
            model=self.model,
            state=self.inactive,
            name=ComponentVarName.DOWNTIME,
            dim_name=self.dim_name,
            timestep_duration=self.model.timestep_duration,
            minimum_duration=sd.min_downtime,
            maximum_duration=sd.max_downtime,
            previous_duration=prev if prev is not None and fast_notnull(prev).any() else None,
        )
        self._variables['downtime'] = var
        return var

    # === Status Constraints ===

    def _status_sel(self, element_ids: list[str]) -> linopy.Variable:
        """Select status variable for a subset of component IDs."""
        return self._variables['status'].sel({self.dim_name: element_ids})

    def constraint_active_hours(self) -> None:
        """Constrain active_hours == sum_temporal(status)."""
        if self.active_hours is None:
            return
        self.model.add_constraints(
            self.active_hours == self.model.sum_temporal(self._variables['status']),
            name=ComponentVarName.Constraint.ACTIVE_HOURS,
        )

    def constraint_complementary(self) -> None:
        """Constrain status + inactive == 1 for downtime tracking components."""
        if self.inactive is None:
            return
        self.model.add_constraints(
            self._status_sel(self._status_data.with_downtime_tracking) + self.inactive == 1,
            name=ComponentVarName.Constraint.COMPLEMENTARY,
        )

    def constraint_switch_transition(self) -> None:
        """Constrain startup[t] - shutdown[t] == status[t] - status[t-1] for t > 0."""
        if self.startup is None:
            return
        status = self._status_sel(self._status_data.with_startup_tracking)
        self.model.add_constraints(
            self.startup.isel(time=slice(1, None)) - self.shutdown.isel(time=slice(1, None))
            == status.isel(time=slice(1, None)) - status.isel(time=slice(None, -1)),
            name=ComponentVarName.Constraint.SWITCH_TRANSITION,
        )

    def constraint_switch_mutex(self) -> None:
        """Constrain startup + shutdown <= 1."""
        if self.startup is None:
            return
        self.model.add_constraints(
            self.startup + self.shutdown <= 1,
            name=ComponentVarName.Constraint.SWITCH_MUTEX,
        )

    def constraint_switch_initial(self) -> None:
        """Constrain startup[0] - shutdown[0] == status[0] - previous_status[-1]."""
        if self.startup is None:
            return
        dim = self.dim_name
        previous_status = self._status_data._previous_states
        ids = [eid for eid in self._status_data.with_startup_tracking if eid in previous_status]
        if not ids:
            return

        prev_arrays = [previous_status[eid].expand_dims({dim: [eid]}) for eid in ids]
        prev_state = xr.concat(prev_arrays, dim=dim).isel(time=-1)

        self.model.add_constraints(
            self.startup.sel({dim: ids}).isel(time=0) - self.shutdown.sel({dim: ids}).isel(time=0)
            == self._status_sel(ids).isel(time=0) - prev_state,
            name=ComponentVarName.Constraint.SWITCH_INITIAL,
        )

    def constraint_startup_count(self) -> None:
        """Constrain startup_count == sum(startup) over temporal dims."""
        if self.startup_count is None:
            return
        dim = self.dim_name
        startup_subset = self.startup.sel({dim: self._status_data.with_startup_limit})
        temporal_dims = [d for d in startup_subset.dims if d not in ('period', 'scenario', dim)]
        self.model.add_constraints(
            self.startup_count == startup_subset.sum(temporal_dims),
            name=ComponentVarName.Constraint.STARTUP_COUNT,
        )

    def constraint_cluster_cyclic(self) -> None:
        """Constrain status[0] == status[-1] for cyclic cluster mode."""
        if self.model.flow_system.clusters is None:
            return
        params = self._status_data._params
        cyclic_ids = [eid for eid in self._status_data.ids if params[eid].cluster_mode == 'cyclic']
        if not cyclic_ids:
            return
        status = self._status_sel(cyclic_ids)
        self.model.add_constraints(
            status.isel(time=0) == status.isel(time=-1),
            name=ComponentVarName.Constraint.CLUSTER_CYCLIC,
        )

    def create_status_features(self) -> None:
        """Create status variables and constraints for components with status.

        Triggers cached property creation for all status variables and calls
        individual constraint methods.
        """
        if not self.components:
            return

        # Trigger variable creation via cached properties
        _ = self.active_hours
        _ = self.startup
        _ = self.shutdown
        _ = self.inactive
        _ = self.startup_count
        _ = self.uptime
        _ = self.downtime

        # Create constraints
        self.constraint_active_hours()
        self.constraint_complementary()
        self.constraint_switch_transition()
        self.constraint_switch_mutex()
        self.constraint_switch_initial()
        self.constraint_startup_count()
        self.constraint_cluster_cyclic()

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
            var = self._variables[var_name]
            if component_id in var.coords.get(dim, []):
                return var.sel({dim: component_id})
            return None
        else:
            raise KeyError(f'Variable {var_name} not found in ComponentsModel')


class ConvertersModel:
    """Type-level model for ALL converter constraints.

    Handles LinearConverters with:
    1. Linear conversion factors: sum(flow * coeff * sign) == 0
    2. Piecewise conversion: inside_piece, lambda0, lambda1 + coupling constraints

    This consolidates converter logic that was previously split between
    LinearConvertersModel (linear) and ComponentsModel (piecewise).

    Example:
        >>> converters_model = ConvertersModel(
        ...     model=flow_system_model,
        ...     converters_with_factors=converters_with_linear_factors,
        ...     converters_with_piecewise=converters_with_piecewise,
        ...     flows_model=flows_model,
        ... )
        >>> converters_model.create_linear_constraints()
        >>> converters_model.create_piecewise_variables()
        >>> converters_model.create_piecewise_constraints()
    """

    def __init__(
        self,
        model: FlowSystemModel,
        converters_with_factors: list,  # list[LinearConverter] - avoid circular import
        converters_with_piecewise: list,  # list[LinearConverter] - avoid circular import
        flows_model: FlowsModel,
    ):
        """Initialize the converter model.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            converters_with_factors: List of LinearConverters with conversion_factors.
            converters_with_piecewise: List of LinearConverters with piecewise_conversion.
            flows_model: The FlowsModel that owns flow variables.
        """
        from .features import PiecewiseHelpers

        self._logger = logging.getLogger('flixopt')
        self.model = model
        self.converters_with_factors = converters_with_factors
        self.converters_with_piecewise = converters_with_piecewise
        self._flows_model = flows_model
        self._PiecewiseHelpers = PiecewiseHelpers

        # Element IDs for linear conversion
        self.element_ids: list[str] = [c.label for c in converters_with_factors]
        self.dim_name = 'converter'

        # Piecewise conversion variables
        self._piecewise_variables: dict[str, linopy.Variable] = {}

        self._logger.debug(
            f'ConvertersModel initialized: {len(converters_with_factors)} with factors, '
            f'{len(converters_with_piecewise)} with piecewise'
        )

    # === Linear Conversion Properties (from LinearConvertersModel) ===

    @cached_property
    def _max_equations(self) -> int:
        """Maximum number of conversion equations across all converters."""
        if not self.converters_with_factors:
            return 0
        return max(len(c.conversion_factors) for c in self.converters_with_factors)

    @cached_property
    def _flow_sign(self) -> xr.DataArray:
        """(converter, flow) sign: +1 for inputs, -1 for outputs, 0 if not involved."""
        all_flow_ids = self._flows_model.element_ids

        # Build sign array
        sign_data = np.zeros((len(self.element_ids), len(all_flow_ids)))
        for i, conv in enumerate(self.converters_with_factors):
            for flow in conv.inputs:
                if flow.label_full in all_flow_ids:
                    j = all_flow_ids.index(flow.label_full)
                    sign_data[i, j] = 1.0  # inputs are positive
            for flow in conv.outputs:
                if flow.label_full in all_flow_ids:
                    j = all_flow_ids.index(flow.label_full)
                    sign_data[i, j] = -1.0  # outputs are negative

        return xr.DataArray(
            sign_data,
            dims=['converter', 'flow'],
            coords={'converter': self.element_ids, 'flow': all_flow_ids},
        )

    @cached_property
    def _equation_mask(self) -> xr.DataArray:
        """(converter, equation_idx) mask: 1 if equation exists, 0 otherwise."""
        max_eq = self._max_equations
        mask_data = np.zeros((len(self.element_ids), max_eq))

        for i, conv in enumerate(self.converters_with_factors):
            for eq_idx in range(len(conv.conversion_factors)):
                mask_data[i, eq_idx] = 1.0

        return xr.DataArray(
            mask_data,
            dims=['converter', 'equation_idx'],
            coords={'converter': self.element_ids, 'equation_idx': list(range(max_eq))},
        )

    @cached_property
    def _coefficients(self) -> xr.DataArray:
        """(converter, equation_idx, flow, [time, ...]) conversion coefficients.

        Returns DataArray with dims (converter, equation_idx, flow) for constant coefficients,
        or (converter, equation_idx, flow, time, ...) for time-varying coefficients.
        Values are 0 where flow is not involved in equation.
        """
        max_eq = self._max_equations
        all_flow_ids = self._flows_model.element_ids
        n_conv = len(self.element_ids)
        n_flows = len(all_flow_ids)

        # Build flow_label -> flow_id mapping for each converter
        conv_flow_maps = []
        for conv in self.converters_with_factors:
            flow_map = {fl.label: fl.label_full for fl in conv.flows.values()}
            conv_flow_maps.append(flow_map)

        # First pass: collect all coefficients and check for time-varying
        coeff_values = {}  # (i, eq_idx, j) -> value
        has_dataarray = False
        extra_coords = {}

        flow_id_to_idx = {fid: j for j, fid in enumerate(all_flow_ids)}

        for i, (conv, flow_map) in enumerate(zip(self.converters_with_factors, conv_flow_maps, strict=False)):
            for eq_idx, conv_factors in enumerate(conv.conversion_factors):
                for flow_label, coeff in conv_factors.items():
                    flow_id = flow_map.get(flow_label)
                    if flow_id and flow_id in flow_id_to_idx:
                        j = flow_id_to_idx[flow_id]
                        coeff_values[(i, eq_idx, j)] = coeff
                        if isinstance(coeff, xr.DataArray) and coeff.ndim > 0:
                            has_dataarray = True
                            for d in coeff.dims:
                                if d not in extra_coords:
                                    extra_coords[d] = coeff.coords[d].values

        # Build the coefficient array
        if not has_dataarray:
            # Fast path: all scalars - use simple numpy array
            data = np.zeros((n_conv, max_eq, n_flows), dtype=np.float64)
            for (i, eq_idx, j), val in coeff_values.items():
                if isinstance(val, xr.DataArray):
                    data[i, eq_idx, j] = float(val.values)
                else:
                    data[i, eq_idx, j] = float(val)

            return xr.DataArray(
                data,
                dims=['converter', 'equation_idx', 'flow'],
                coords={
                    'converter': self.element_ids,
                    'equation_idx': list(range(max_eq)),
                    'flow': all_flow_ids,
                },
            )
        else:
            # Slow path: some time-varying coefficients - broadcast all to common shape
            extra_dims = list(extra_coords.keys())
            extra_shape = [len(c) for c in extra_coords.values()]
            full_shape = [n_conv, max_eq, n_flows] + extra_shape
            full_dims = ['converter', 'equation_idx', 'flow'] + extra_dims

            data = np.zeros(full_shape, dtype=np.float64)

            # Create template for broadcasting
            template = xr.DataArray(coords=extra_coords, dims=extra_dims) if extra_coords else None

            for (i, eq_idx, j), val in coeff_values.items():
                if isinstance(val, xr.DataArray):
                    if val.ndim == 0:
                        data[i, eq_idx, j, ...] = float(val.values)
                    elif template is not None:
                        broadcasted = val.broadcast_like(template)
                        data[i, eq_idx, j, ...] = broadcasted.values
                    else:
                        data[i, eq_idx, j, ...] = val.values
                else:
                    data[i, eq_idx, j, ...] = float(val)

            full_coords = {
                'converter': self.element_ids,
                'equation_idx': list(range(max_eq)),
                'flow': all_flow_ids,
            }
            full_coords.update(extra_coords)

            return xr.DataArray(data, dims=full_dims, coords=full_coords)

    def create_linear_constraints(self) -> None:
        """Create batched linear conversion factor constraints.

        For each converter c with equation i:
            sum_f(flow_rate[f] * coefficient[c,i,f] * sign[c,f]) == 0

        where:
            - Inputs have positive sign, outputs have negative sign
            - coefficient contains the conversion factors (may be time-varying)
        """
        if not self.converters_with_factors:
            return

        coefficients = self._coefficients
        flow_rate = self._flows_model._variables['rate']
        sign = self._flow_sign

        # Pre-combine coefficients and sign (both are xr.DataArrays, not linopy)
        # This avoids creating intermediate linopy expressions
        # coefficients: (converter, equation_idx, flow, [time, ...])
        # sign: (converter, flow)
        # Result: (converter, equation_idx, flow, [time, ...])
        signed_coeffs = coefficients * sign

        # Now multiply flow_rate by the combined coefficients
        # flow_rate: (flow, time, ...)
        # signed_coeffs: (converter, equation_idx, flow, [time, ...])
        # Result: (converter, equation_idx, flow, time, ...)
        weighted = flow_rate * signed_coeffs

        # Sum over flows: (converter, equation_idx, time, ...)
        flow_sum = weighted.sum('flow')

        # Build valid mask: (converter, equation_idx)
        # True where converter HAS that equation (keep constraint)
        n_equations_per_converter = xr.DataArray(
            [len(c.conversion_factors) for c in self.converters_with_factors],
            dims=['converter'],
            coords={'converter': self.element_ids},
        )
        equation_indices = xr.DataArray(
            list(range(self._max_equations)),
            dims=['equation_idx'],
            coords={'equation_idx': list(range(self._max_equations))},
        )
        valid_mask = equation_indices < n_equations_per_converter

        # Add all constraints at once using linopy's mask parameter
        # mask=True means KEEP constraint for that (converter, equation_idx) pair
        self.model.add_constraints(
            flow_sum == 0,
            name=ConverterVarName.Constraint.CONVERSION,
            mask=valid_mask,
        )

        self._logger.debug(
            f'ConvertersModel created linear constraints for {len(self.converters_with_factors)} converters'
        )

    # === Piecewise Conversion Properties (from ComponentsModel) ===

    @cached_property
    def _piecewise_element_ids(self) -> list[str]:
        """Element IDs for converters with piecewise conversion."""
        return [c.label for c in self.converters_with_piecewise]

    @cached_property
    def _piecewise_segment_counts(self) -> dict[str, int]:
        """Dict mapping converter_id -> number of segments."""
        return {
            c.label: len(list(c.piecewise_conversion.piecewises.values())[0]) for c in self.converters_with_piecewise
        }

    @cached_property
    def _piecewise_max_segments(self) -> int:
        """Maximum segment count across all converters."""
        if not self.converters_with_piecewise:
            return 0
        return max(self._piecewise_segment_counts.values())

    @cached_property
    def _piecewise_segment_mask(self) -> xr.DataArray:
        """(converter, segment) mask: 1=valid, 0=padded."""
        _, mask = self._PiecewiseHelpers.collect_segment_info(
            self._piecewise_element_ids, self._piecewise_segment_counts, self._piecewise_dim_name
        )
        return mask

    @cached_property
    def _piecewise_dim_name(self) -> str:
        """Dimension name for piecewise converters."""
        return 'converter'

    @cached_property
    def _piecewise_flow_breakpoints(self) -> dict[str, tuple[xr.DataArray, xr.DataArray]]:
        """Dict mapping flow_id -> (starts, ends) padded DataArrays."""
        # Collect all flow ids that appear in piecewise conversions
        all_flow_ids: set[str] = set()
        for conv in self.converters_with_piecewise:
            for flow_label in conv.piecewise_conversion.piecewises:
                flow_id = conv.flows[flow_label].label_full
                all_flow_ids.add(flow_id)

        result = {}
        for flow_id in all_flow_ids:
            breakpoints: dict[str, tuple[list[float], list[float]]] = {}
            for conv in self.converters_with_piecewise:
                # Check if this converter has this flow
                found = False
                for flow_label, piecewise in conv.piecewise_conversion.piecewises.items():
                    if conv.flows[flow_label].label_full == flow_id:
                        starts = [p.start for p in piecewise]
                        ends = [p.end for p in piecewise]
                        breakpoints[conv.label] = (starts, ends)
                        found = True
                        break
                if not found:
                    # This converter doesn't have this flow - use NaN
                    breakpoints[conv.label] = (
                        [np.nan] * self._piecewise_max_segments,
                        [np.nan] * self._piecewise_max_segments,
                    )

            # Get time coordinates from model for time-varying breakpoints
            time_coords = self.model.flow_system.timesteps
            starts, ends = self._PiecewiseHelpers.pad_breakpoints(
                self._piecewise_element_ids,
                breakpoints,
                self._piecewise_max_segments,
                self._piecewise_dim_name,
                time_coords=time_coords,
            )
            result[flow_id] = (starts, ends)

        return result

    @cached_property
    def piecewise_segment_counts(self) -> xr.DataArray | None:
        """(converter,) - number of segments per converter with piecewise conversion."""
        if not self.converters_with_piecewise:
            return None
        counts = [len(list(c.piecewise_conversion.piecewises.values())[0]) for c in self.converters_with_piecewise]
        return xr.DataArray(
            counts,
            dims=[self._piecewise_dim_name],
            coords={self._piecewise_dim_name: self._piecewise_element_ids},
        )

    @cached_property
    def piecewise_segment_mask(self) -> xr.DataArray | None:
        """(converter, segment) - 1=valid segment, 0=padded."""
        if not self.converters_with_piecewise:
            return None
        return self._piecewise_segment_mask

    @cached_property
    def piecewise_breakpoints(self) -> xr.Dataset | None:
        """Dataset with (converter, segment, flow) or (converter, segment, flow, time) breakpoints.

        Variables:
            - starts: segment start values
            - ends: segment end values

        When breakpoints are time-varying, an additional 'time' dimension is included.
        """
        if not self.converters_with_piecewise:
            return None

        # Collect all flows
        all_flows = list(self._piecewise_flow_breakpoints.keys())

        # Build a list of DataArrays for each flow, then combine with xr.concat
        starts_list = []
        ends_list = []
        for flow_id in all_flows:
            starts_da, ends_da = self._piecewise_flow_breakpoints[flow_id]
            # Add 'flow' as a new coordinate
            starts_da = starts_da.expand_dims(flow=[flow_id])
            ends_da = ends_da.expand_dims(flow=[flow_id])
            starts_list.append(starts_da)
            ends_list.append(ends_da)

        # Concatenate along 'flow' dimension
        starts_combined = xr.concat(starts_list, dim='flow')
        ends_combined = xr.concat(ends_list, dim='flow')

        return xr.Dataset({'starts': starts_combined, 'ends': ends_combined})

    def create_piecewise_variables(self) -> dict[str, linopy.Variable]:
        """Create batched piecewise conversion variables.

        Returns:
            Dict with 'inside_piece', 'lambda0', 'lambda1' variables.
        """
        if not self.converters_with_piecewise:
            return {}

        base_coords = self.model.get_coords(['time', 'period', 'scenario'])

        self._piecewise_variables = self._PiecewiseHelpers.create_piecewise_variables(
            self.model,
            self._piecewise_element_ids,
            self._piecewise_max_segments,
            self._piecewise_dim_name,
            self._piecewise_segment_mask,
            base_coords,
            ConverterVarName.PIECEWISE_PREFIX,
        )

        self._logger.debug(
            f'ConvertersModel created piecewise variables for {len(self.converters_with_piecewise)} converters'
        )
        return self._piecewise_variables

    def create_piecewise_constraints(self) -> None:
        """Create batched piecewise constraints and coupling constraints."""
        if not self.converters_with_piecewise:
            return

        # Get zero_point for each converter (status variable if available)
        # TODO: Integrate status from ComponentsModel when converters overlap
        zero_point = None

        # Create lambda_sum and single_segment constraints
        self._PiecewiseHelpers.create_piecewise_constraints(
            self.model,
            self._piecewise_variables,
            self._piecewise_segment_mask,
            zero_point,
            self._piecewise_dim_name,
            ConverterVarName.PIECEWISE_PREFIX,
        )

        # Create batched coupling constraints for all piecewise flows
        bp = self.piecewise_breakpoints  # Dataset with (converter, segment, flow) dims
        if bp is None:
            return

        flow_rate = self._flows_model._variables['rate']
        lambda0 = self._piecewise_variables['lambda0']
        lambda1 = self._piecewise_variables['lambda1']

        # Compute all reconstructed values at once: (converter, flow, time, period, ...)
        all_reconstructed = (lambda0 * bp['starts'] + lambda1 * bp['ends']).sum('segment')

        # Mask: valid where breakpoints exist (not NaN)
        valid_mask = fast_notnull(bp['starts']).any('segment')

        # Apply mask and sum over converter (each flow has exactly one valid converter)
        reconstructed_per_flow = all_reconstructed.where(valid_mask).sum('converter')

        # Get flow rates for piecewise flows
        flow_ids = list(bp.coords['flow'].values)
        piecewise_flow_rate = flow_rate.sel(flow=flow_ids)

        # Add single batched constraint
        self.model.add_constraints(
            piecewise_flow_rate == reconstructed_per_flow,
            name=ConverterVarName.Constraint.PIECEWISE_COUPLING,
        )

        self._logger.debug(
            f'ConvertersModel created piecewise constraints for {len(self.converters_with_piecewise)} converters'
        )


class TransmissionsModel:
    """Type-level model for batched transmission efficiency constraints.

    Handles Transmission components with batched constraints:
    - Efficiency: out = in * (1 - rel_losses) - status * abs_losses
    - Balanced size: in1.size == in2.size

    All constraints have a 'transmission' dimension for proper batching.

    Example:
        >>> transmissions_model = TransmissionsModel(
        ...     model=flow_system_model,
        ...     transmissions=transmissions,
        ...     flows_model=flows_model,
        ... )
        >>> transmissions_model.create_constraints()
    """

    def __init__(
        self,
        model: FlowSystemModel,
        transmissions: list,  # list[Transmission] - avoid circular import
        flows_model: FlowsModel,
    ):
        """Initialize the transmission model.

        Args:
            model: The FlowSystemModel to create constraints in.
            transmissions: List of Transmission components.
            flows_model: The FlowsModel that owns flow variables.
        """
        self._logger = logging.getLogger('flixopt')
        self.model = model
        self.transmissions = transmissions
        self._flows_model = flows_model
        self.element_ids: list[str] = [t.label for t in transmissions]
        self.dim_name = 'transmission'

        self._logger.debug(f'TransmissionsModel initialized: {len(transmissions)} transmissions')

    # === Flow Mapping Properties ===

    @cached_property
    def _bidirectional(self) -> list:
        """List of transmissions that are bidirectional."""
        return [t for t in self.transmissions if t.in2 is not None]

    @cached_property
    def _bidirectional_ids(self) -> list[str]:
        """Element IDs for bidirectional transmissions."""
        return [t.label for t in self._bidirectional]

    @cached_property
    def _balanced(self) -> list:
        """List of transmissions with balanced=True."""
        return [t for t in self.transmissions if t.balanced]

    @cached_property
    def _balanced_ids(self) -> list[str]:
        """Element IDs for balanced transmissions."""
        return [t.label for t in self._balanced]

    # === Flow Masks for Batched Selection ===

    def _build_flow_mask(self, transmission_ids: list[str], flow_getter) -> xr.DataArray:
        """Build (transmission, flow) mask: 1 if flow belongs to transmission.

        Args:
            transmission_ids: List of transmission labels to include.
            flow_getter: Function that takes a transmission and returns its flow label_full.
        """
        all_flow_ids = self._flows_model.element_ids
        mask_data = np.zeros((len(transmission_ids), len(all_flow_ids)))

        for t_idx, t_id in enumerate(transmission_ids):
            t = next(t for t in self.transmissions if t.label == t_id)
            flow_id = flow_getter(t)
            if flow_id in all_flow_ids:
                f_idx = all_flow_ids.index(flow_id)
                mask_data[t_idx, f_idx] = 1.0

        return xr.DataArray(
            mask_data,
            dims=[self.dim_name, 'flow'],
            coords={self.dim_name: transmission_ids, 'flow': all_flow_ids},
        )

    @cached_property
    def _in1_mask(self) -> xr.DataArray:
        """(transmission, flow) mask: 1 if flow is in1 for transmission."""
        return self._build_flow_mask(self.element_ids, lambda t: t.in1.label_full)

    @cached_property
    def _out1_mask(self) -> xr.DataArray:
        """(transmission, flow) mask: 1 if flow is out1 for transmission."""
        return self._build_flow_mask(self.element_ids, lambda t: t.out1.label_full)

    @cached_property
    def _in2_mask(self) -> xr.DataArray:
        """(transmission, flow) mask for bidirectional: 1 if flow is in2."""
        return self._build_flow_mask(self._bidirectional_ids, lambda t: t.in2.label_full)

    @cached_property
    def _out2_mask(self) -> xr.DataArray:
        """(transmission, flow) mask for bidirectional: 1 if flow is out2."""
        return self._build_flow_mask(self._bidirectional_ids, lambda t: t.out2.label_full)

    # === Loss Properties ===

    @cached_property
    def _relative_losses(self) -> xr.DataArray:
        """(transmission, [time, ...]) relative losses. 0 if None."""
        if not self.transmissions:
            return xr.DataArray()
        values = []
        for t in self.transmissions:
            loss = t.relative_losses if t.relative_losses is not None else 0
            values.append(loss)
        return self._stack_data(values)

    @cached_property
    def _absolute_losses(self) -> xr.DataArray:
        """(transmission, [time, ...]) absolute losses. 0 if None."""
        if not self.transmissions:
            return xr.DataArray()
        values = []
        for t in self.transmissions:
            loss = t.absolute_losses if t.absolute_losses is not None else 0
            values.append(loss)
        return self._stack_data(values)

    @cached_property
    def _has_absolute_losses_mask(self) -> xr.DataArray:
        """(transmission,) bool mask for transmissions with absolute losses."""
        if not self.transmissions:
            return xr.DataArray()
        has_abs = [t.absolute_losses is not None and np.any(t.absolute_losses != 0) for t in self.transmissions]
        return xr.DataArray(
            has_abs,
            dims=[self.dim_name],
            coords={self.dim_name: self.element_ids},
        )

    @cached_property
    def _transmissions_with_abs_losses(self) -> list[str]:
        """Element IDs for transmissions with absolute losses."""
        return [t.label for t in self.transmissions if t.absolute_losses is not None and np.any(t.absolute_losses != 0)]

    def _stack_data(self, values: list) -> xr.DataArray:
        """Stack transmission data into (transmission, [time, ...]) array."""
        if not values:
            return xr.DataArray()

        # Convert scalars to arrays with proper coords
        arrays = []
        for i, val in enumerate(values):
            if isinstance(val, xr.DataArray):
                arr = val.expand_dims({self.dim_name: [self.element_ids[i]]})
            else:
                # Scalar - create simple array
                arr = xr.DataArray(
                    val,
                    dims=[self.dim_name],
                    coords={self.dim_name: [self.element_ids[i]]},
                )
            arrays.append(arr)

        return xr.concat(arrays, dim=self.dim_name)

    def create_constraints(self) -> None:
        """Create batched transmission efficiency constraints.

        Uses mask-based batching: mask[transmission, flow] = 1 if flow belongs to transmission.
        Broadcasting (flow_rate * mask).sum('flow') gives (transmission, time, ...) rates.

        Creates batched constraints with transmission dimension:
        - Direction 1: out1 == in1 * (1 - rel_losses) - in1_status * abs_losses
        - Direction 2: out2 == in2 * (1 - rel_losses) - in2_status * abs_losses (bidirectional only)
        - Balanced: in1.size == in2.size (balanced only)
        """
        if not self.transmissions:
            return

        con = TransmissionVarName.Constraint
        flow_rate = self._flows_model._variables['rate']

        # === Direction 1: All transmissions (batched) ===
        # Use masks to batch flow selection: (flow_rate * mask).sum('flow') -> (transmission, time, ...)
        in1_rate = (flow_rate * self._in1_mask).sum('flow')
        out1_rate = (flow_rate * self._out1_mask).sum('flow')
        rel_losses = self._relative_losses
        abs_losses = self._absolute_losses

        # Build the efficiency expression: in1 * (1 - rel_losses) - abs_losses_term
        efficiency_expr = in1_rate * (1 - rel_losses)

        # Add absolute losses term if any transmission has them
        if self._transmissions_with_abs_losses:
            flow_status = self._flows_model._variables['status']
            in1_status = (flow_status * self._in1_mask).sum('flow')
            efficiency_expr = efficiency_expr - in1_status * abs_losses

        # out1 == in1 * (1 - rel_losses) - in1_status * abs_losses
        self.model.add_constraints(
            out1_rate == efficiency_expr,
            name=con.DIR1,
        )

        # === Direction 2: Bidirectional transmissions only (batched) ===
        if self._bidirectional:
            in2_rate = (flow_rate * self._in2_mask).sum('flow')
            out2_rate = (flow_rate * self._out2_mask).sum('flow')
            rel_losses_bidir = self._relative_losses.sel({self.dim_name: self._bidirectional_ids})
            abs_losses_bidir = self._absolute_losses.sel({self.dim_name: self._bidirectional_ids})

            # Build the efficiency expression for direction 2
            efficiency_expr_2 = in2_rate * (1 - rel_losses_bidir)

            # Add absolute losses for bidirectional if any have them
            bidir_with_abs = [t.label for t in self._bidirectional if t.label in self._transmissions_with_abs_losses]
            if bidir_with_abs:
                flow_status = self._flows_model._variables['status']
                in2_status = (flow_status * self._in2_mask).sum('flow')
                efficiency_expr_2 = efficiency_expr_2 - in2_status * abs_losses_bidir

            # out2 == in2 * (1 - rel_losses) - in2_status * abs_losses
            self.model.add_constraints(
                out2_rate == efficiency_expr_2,
                name=con.DIR2,
            )

        # === Balanced constraints: in1.size == in2.size (batched) ===
        if self._balanced:
            flow_size = self._flows_model._variables['size']
            # Build masks for balanced transmissions only
            in1_size_mask = self._build_flow_mask(self._balanced_ids, lambda t: t.in1.label_full)
            in2_size_mask = self._build_flow_mask(self._balanced_ids, lambda t: t.in2.label_full)

            in1_size_batched = (flow_size * in1_size_mask).sum('flow')
            in2_size_batched = (flow_size * in2_size_mask).sum('flow')

            self.model.add_constraints(
                in1_size_batched == in2_size_batched,
                name=con.BALANCED,
            )

        self._logger.debug(
            f'TransmissionsModel created batched constraints for {len(self.transmissions)} transmissions'
        )


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
