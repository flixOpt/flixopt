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
from .features import (
    MaskHelpers,
    StatusBuilder,
    fast_notnull,
    sparse_multiply_sum,
    sparse_weighted_sum,
)
from .id_list import IdList, flow_id_list
from .interface import InvestParameters, StatusParameters
from .modeling import ModelingUtilitiesAbstract
from .structure import (
    BusVarName,
    ComponentVarName,
    ConverterVarName,
    Element,
    FlowSystemModel,
    FlowVarName,
    TransmissionVarName,
    TypeModel,
    register_class_for_io,
)

if TYPE_CHECKING:
    import linopy

    from .batched import BusesData, ComponentsData, ConvertersData, FlowsData, TransmissionsData
    from .types import (
        Effect_TPS,
        Numeric_PS,
        Numeric_S,
        Numeric_TPS,
        Scalar,
    )

logger = logging.getLogger('flixopt')


def _add_prevent_simultaneous_constraints(
    components: list,
    flows_model,
    model,
    constraint_name: str,
) -> None:
    """Add prevent_simultaneous_flows constraints for the given components.

    For each component with prevent_simultaneous_flows set, adds:
        sum(flow_statuses) <= 1

    Args:
        components: Components to check for prevent_simultaneous_flows.
        flows_model: FlowsModel that owns flow status variables.
        model: The FlowSystemModel to add constraints to.
        constraint_name: Name for the constraint.
    """
    with_prevent = [c for c in components if c.prevent_simultaneous_flows]
    if not with_prevent:
        return

    membership = MaskHelpers.build_flow_membership(
        with_prevent,
        lambda c: c.prevent_simultaneous_flows,
    )
    mask = MaskHelpers.build_mask(
        row_dim='component',
        row_ids=[c.id for c in with_prevent],
        col_dim='flow',
        col_ids=flows_model.element_ids,
        membership=membership,
    )

    status = flows_model[FlowVarName.STATUS]
    model.add_constraints(
        sparse_weighted_sum(status, mask, sum_dim='flow', group_dim='component') <= 1,
        name=constraint_name,
    )


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
        id: The id of the Element. Used to identify it in the FlowSystem.
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
        id: str | None = None,
        inputs: list[Flow] | dict[str, Flow] | None = None,
        outputs: list[Flow] | dict[str, Flow] | None = None,
        status_parameters: StatusParameters | None = None,
        prevent_simultaneous_flows: list[Flow] | None = None,
        meta_data: dict | None = None,
        color: str | None = None,
        **kwargs,
    ):
        super().__init__(id, meta_data=meta_data, color=color, **kwargs)
        self.status_parameters = status_parameters
        if isinstance(prevent_simultaneous_flows, dict):
            prevent_simultaneous_flows = list(prevent_simultaneous_flows.values())
        self.prevent_simultaneous_flows: list[Flow] = prevent_simultaneous_flows or []

        # IdLists serialize as dicts, but constructor expects lists
        if isinstance(inputs, dict):
            inputs = list(inputs.values())
        if isinstance(outputs, dict):
            outputs = list(outputs.values())

        _inputs = inputs or []
        _outputs = outputs or []

        # Check uniqueness on raw lists (before connecting)
        all_flow_ids = [flow.flow_id for flow in _inputs + _outputs]
        if len(set(all_flow_ids)) != len(all_flow_ids):
            duplicates = {fid for fid in all_flow_ids if all_flow_ids.count(fid) > 1}
            raise ValueError(f'Flow names must be unique! "{self.id}" got 2 or more of: {duplicates}')

        # Connect flows (sets component name) before creating IdLists
        self._connect_flows(_inputs, _outputs)

        # Now flow.id is qualified, so IdList can key by it
        self.inputs: IdList = flow_id_list(_inputs, display_name='inputs')
        self.outputs: IdList = flow_id_list(_outputs, display_name='outputs')

    @cached_property
    def flows(self) -> IdList:
        """All flows (inputs and outputs) as an IdList."""
        return self.inputs + self.outputs

    def link_to_flow_system(self, flow_system, prefix: str = '') -> None:
        """Propagate flow_system reference to nested Interface objects and flows.

        Elements use their id_full as prefix by default, ignoring the passed prefix.
        """
        super().link_to_flow_system(flow_system, self.id)
        for flow in self.flows.values():
            flow.link_to_flow_system(flow_system)

    def transform_data(self) -> None:
        self._propagate_status_parameters()

    def _propagate_status_parameters(self) -> None:
        """Propagate status parameters from this component to flows that need them.

        Components with status_parameters require all their flows to have
        StatusParameters (for big-M constraints). Components with
        prevent_simultaneous_flows require those flows to have them too.
        """
        from .interface import StatusParameters

        if self.status_parameters:
            for flow in self.flows.values():
                if flow.status_parameters is None:
                    flow.status_parameters = StatusParameters()
        if self.prevent_simultaneous_flows:
            for flow in self.prevent_simultaneous_flows:
                if flow.status_parameters is None:
                    flow.status_parameters = StatusParameters()

    def _check_unique_flow_ids(self, inputs: list = None, outputs: list = None):
        if inputs is None:
            inputs = list(self.inputs.values())
        if outputs is None:
            outputs = list(self.outputs.values())
        all_flow_ids = [flow.flow_id for flow in inputs + outputs]

        if len(set(all_flow_ids)) != len(all_flow_ids):
            duplicates = {fid for fid in all_flow_ids if all_flow_ids.count(fid) > 1}
            raise ValueError(f'Flow names must be unique! "{self.id}" got 2 or more of: {duplicates}')

    def validate_config(self) -> None:
        """Validate configuration consistency.

        Called BEFORE transformation via FlowSystem._run_config_validation().
        These are simple checks that don't require DataArray operations.
        """
        self._check_unique_flow_ids()

        # Component with status_parameters requires all flows to have sizes set
        # (status_parameters are propagated to flows in _do_modeling, which need sizes for big-M constraints)
        if self.status_parameters is not None:
            flows_without_size = [flow.flow_id for flow in self.flows.values() if flow.size is None]
            if flows_without_size:
                raise PlausibilityError(
                    f'Component "{self.id}" has status_parameters, but the following flows have no size: '
                    f'{flows_without_size}. All flows need explicit sizes when the component uses status_parameters '
                    f'(required for big-M constraints).'
                )

    def _plausibility_checks(self) -> None:
        """Legacy validation method - delegates to validate_config()."""
        self.validate_config()

    def _connect_flows(self, inputs=None, outputs=None):
        if inputs is None:
            inputs = list(self.inputs.values())
        if outputs is None:
            outputs = list(self.outputs.values())
        # Inputs
        for flow in inputs:
            if flow.component not in ('UnknownComponent', self.id):
                raise ValueError(
                    f'Flow "{flow.id}" already assigned to component "{flow.component}". Cannot attach to "{self.id}".'
                )
            flow.component = self.id
            flow.is_input_in_component = True
        # Outputs
        for flow in outputs:
            if flow.component not in ('UnknownComponent', self.id):
                raise ValueError(
                    f'Flow "{flow.id}" already assigned to component "{flow.component}". Cannot attach to "{self.id}".'
                )
            flow.component = self.id
            flow.is_input_in_component = False

        # Validate prevent_simultaneous_flows: only allow local flows
        if self.prevent_simultaneous_flows:
            # Deduplicate while preserving order
            seen = set()
            self.prevent_simultaneous_flows = [
                f for f in self.prevent_simultaneous_flows if id(f) not in seen and not seen.add(id(f))
            ]
            local = set(inputs + outputs)
            foreign = [f for f in self.prevent_simultaneous_flows if f not in local]
            if foreign:
                names = ', '.join(f.id for f in foreign)
                raise ValueError(
                    f'prevent_simultaneous_flows for "{self.id}" must reference its own flows. '
                    f'Foreign flows detected: {names}'
                )

    def __repr__(self) -> str:
        """Return string representation with flow information."""
        return fx_io.build_repr_from_init(
            self, excluded_params={'self', 'id', 'inputs', 'outputs', 'kwargs'}, skip_default_size=True
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
        id: str | None = None,
        carrier: str | None = None,
        imbalance_penalty_per_flow_hour: Numeric_TPS | None = None,
        meta_data: dict | None = None,
        **kwargs,
    ):
        # Handle Bus-specific deprecated kwarg before passing kwargs to super
        old_penalty = kwargs.pop('excess_penalty_per_flow_hour', None)
        super().__init__(id, meta_data=meta_data, **kwargs)
        if old_penalty is not None:
            imbalance_penalty_per_flow_hour = self._handle_deprecated_kwarg(
                {'excess_penalty_per_flow_hour': old_penalty},
                'excess_penalty_per_flow_hour',
                'imbalance_penalty_per_flow_hour',
                imbalance_penalty_per_flow_hour,
            )
        self.carrier = carrier.lower() if carrier else None  # Store as lowercase string
        self.imbalance_penalty_per_flow_hour = imbalance_penalty_per_flow_hour
        self.inputs: IdList = flow_id_list(display_name='inputs')
        self.outputs: IdList = flow_id_list(display_name='outputs')

    @property
    def flows(self) -> IdList:
        """All flows (inputs and outputs) as an IdList."""
        return self.inputs + self.outputs

    def link_to_flow_system(self, flow_system, prefix: str = '') -> None:
        """Propagate flow_system reference to nested flows.

        Elements use their id_full as prefix by default, ignoring the passed prefix.
        """
        super().link_to_flow_system(flow_system, self.id)
        for flow in self.flows.values():
            flow.link_to_flow_system(flow_system)

    def transform_data(self) -> None:
        # No-op: alignment now handled by BusesData
        pass

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
        bus: Bus this flow connects to (string id). First positional argument.
        flow_id: Unique flow identifier within its component. Defaults to the bus name.
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
            'electricity_grid',
            flow_id='electricity_out',
            size=100,  # 100 MW capacity
            relative_minimum=0.4,  # Cannot operate below 40 MW
            effects_per_flow_hour={'fuel_cost': 45, 'co2_emissions': 0.8},
        )
        ```

        Investment decision for battery capacity (flow_id defaults to bus name):

        ```python
        battery_flow = Flow(
            'electricity_grid',
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
            'heating_network',
            flow_id='heat_output',
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
            'electricity_grid',
            flow_id='solar_power',
            size=25,  # 25 MW installed capacity
            fixed_relative_profile=np.array([0, 0.1, 0.4, 0.8, 0.9, 0.7, 0.3, 0.1, 0]),
            effects_per_flow_hour={'maintenance_costs': 5},  # €5/MWh maintenance
        )
        ```

        Industrial process with annual utilization limits:

        ```python
        production_line = Flow(
            'product_market',
            flow_id='product_output',
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
        *args,
        bus: str | None = None,
        flow_id: str | None = None,
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
        label: str | None = None,
        id: str | None = None,
        **kwargs,
    ):
        # --- Resolve positional args + deprecation bridge ---
        import warnings

        from .config import DEPRECATION_REMOVAL_VERSION

        # Handle deprecated 'id' kwarg (use flow_id instead)
        if id is not None:
            warnings.warn(
                f'Flow(id=...) is deprecated. Use Flow(flow_id=...) instead. '
                f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}.',
                DeprecationWarning,
                stacklevel=2,
            )
            if flow_id is not None:
                raise ValueError('Either id or flow_id can be specified, but not both.')
            flow_id = id

        if len(args) == 2:
            # Old API: Flow(label, bus)
            warnings.warn(
                f'Flow(label, bus) positional form is deprecated. '
                f'Use Flow(bus, flow_id=...) instead. Will be removed in v{DEPRECATION_REMOVAL_VERSION}.',
                DeprecationWarning,
                stacklevel=2,
            )
            if flow_id is None and label is None:
                flow_id = args[0]
            if bus is None:
                bus = args[1]
        elif len(args) == 1:
            if bus is not None:
                # Old API: Flow(label, bus=...)
                warnings.warn(
                    f'Flow(label, bus=...) positional form is deprecated. '
                    f'Use Flow(bus, flow_id=...) instead. Will be removed in v{DEPRECATION_REMOVAL_VERSION}.',
                    DeprecationWarning,
                    stacklevel=2,
                )
                if flow_id is None and label is None:
                    flow_id = args[0]
            else:
                # New API: Flow(bus) — bus is the positional arg
                bus = args[0]
        elif len(args) > 2:
            raise TypeError(f'Flow() takes at most 2 positional arguments ({len(args)} given)')

        # Handle deprecated label kwarg
        if label is not None:
            warnings.warn(
                f'The "label" argument is deprecated. Use "flow_id" instead. '
                f'Will be removed in v{DEPRECATION_REMOVAL_VERSION}.',
                DeprecationWarning,
                stacklevel=2,
            )
            if flow_id is not None:
                raise ValueError('Either label or flow_id can be specified, but not both.')
            flow_id = label

        # Default flow_id to bus name
        if flow_id is None:
            if bus is None:
                raise TypeError('Flow() requires a bus argument.')
            flow_id = bus if isinstance(bus, str) else str(bus)

        if bus is None:
            raise TypeError('Flow() requires a bus argument.')

        super().__init__(flow_id, meta_data=meta_data, **kwargs)
        self.size = size
        self.relative_minimum = relative_minimum
        self.relative_maximum = relative_maximum
        self.fixed_relative_profile = fixed_relative_profile

        self.load_factor_min = load_factor_min
        self.load_factor_max = load_factor_max

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
                f'Bus {bus.id} is passed as a Bus object to Flow {self.id}. '
                f'This is no longer supported. Add the Bus to the FlowSystem and pass its id (string) to the Flow.'
            )
        self.bus = bus

    def link_to_flow_system(self, flow_system, prefix: str = '') -> None:
        """Propagate flow_system reference to nested Interface objects.

        Elements use their id_full as prefix by default, ignoring the passed prefix.
        """
        super().link_to_flow_system(flow_system, self.id)

    def validate_config(self) -> None:
        """Validate configuration consistency.

        Called BEFORE transformation via FlowSystem._run_config_validation().
        These are simple checks that don't require DataArray operations.
        """
        # Size is required when using StatusParameters (for big-M constraints)
        if self.status_parameters is not None and self.size is None:
            raise PlausibilityError(
                f'Flow "{self.id}" has status_parameters but no size defined. '
                f'A size is required when using status_parameters to bound the flow rate.'
            )

        if self.size is None and self.fixed_relative_profile is not None:
            raise PlausibilityError(
                f'Flow "{self.id}" has a fixed_relative_profile but no size defined. '
                f'A size is required because flow_rate = size * fixed_relative_profile.'
            )

        # Size is required for load factor constraints (total_flow_hours / size)
        if self.size is None and self.load_factor_min is not None:
            raise PlausibilityError(
                f'Flow "{self.id}" has load_factor_min but no size defined. '
                f'A size is required because the constraint is total_flow_hours >= size * load_factor_min * hours.'
            )

        if self.size is None and self.load_factor_max is not None:
            raise PlausibilityError(
                f'Flow "{self.id}" has load_factor_max but no size defined. '
                f'A size is required because the constraint is total_flow_hours <= size * load_factor_max * hours.'
            )

        # Validate previous_flow_rate type
        if self.previous_flow_rate is not None:
            if not any(
                [
                    isinstance(self.previous_flow_rate, np.ndarray) and self.previous_flow_rate.ndim == 1,
                    isinstance(self.previous_flow_rate, (int, float, list)),
                ]
            ):
                raise TypeError(
                    f'previous_flow_rate must be None, a scalar, a list of scalars or a 1D-numpy-array. '
                    f'Got {type(self.previous_flow_rate)}. '
                    f'Different values in different periods or scenarios are not yet supported.'
                )

        # Warning: fixed_relative_profile + status_parameters is unusual
        if self.fixed_relative_profile is not None and self.status_parameters is not None:
            logger.warning(
                f'Flow {self.id} has both a fixed_relative_profile and status_parameters. '
                f'This will allow the flow to be switched active and inactive, effectively differing from the fixed_flow_rate.'
            )

    def _plausibility_checks(self) -> None:
        """Legacy validation method - delegates to validate_config().

        DataArray-based validation is now done in FlowsData.validate().
        """
        self.validate_config()

    @property
    def flow_id(self) -> str:
        """The short flow identifier (e.g. ``'Heat'``).

        This is the user-facing name. Defaults to the bus name if not set explicitly.
        """
        return self._short_id

    @flow_id.setter
    def flow_id(self, value: str) -> None:
        self._short_id = value

    @property
    def id(self) -> str:
        """The qualified identifier: ``component(flow_id)``."""
        return f'{self.component}({self._short_id})'

    @id.setter
    def id(self, value: str) -> None:
        self._short_id = value

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
        return self._flows_model.get_variable(FlowVarName.RATE, self.id)

    @property
    def total_flow_hours_from_type_model(self) -> linopy.Variable | None:
        """Get total_flow_hours from FlowsModel (if using type-level modeling)."""
        if self._flows_model is None:
            return None
        return self._flows_model.get_variable(FlowVarName.TOTAL_FLOW_HOURS, self.id)

    @property
    def status_from_type_model(self) -> linopy.Variable | None:
        """Get status from FlowsModel (if using type-level modeling)."""
        if self._flows_model is None or FlowVarName.STATUS not in self._flows_model:
            return None
        if self.id not in self._flows_model.status_ids:
            return None
        return self._flows_model.get_variable(FlowVarName.STATUS, self.id)

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
        >>> boiler_rate = flows_model.get_variable(FlowVarName.RATE, 'Boiler(gas_in)')
    """

    # === Variables (cached_property) ===

    @cached_property
    def rate(self) -> linopy.Variable:
        """(flow, time, ...) - flow rate variable for ALL flows."""
        return self.add_variables(
            FlowVarName.RATE,
            lower=self.data.absolute_lower_bounds,
            upper=self.data.absolute_upper_bounds,
            dims=None,
        )

    @cached_property
    def status(self) -> linopy.Variable | None:
        """(flow, time, ...) - binary status variable, masked to flows with status."""
        if not self.data.with_status:
            return None
        return self.add_variables(
            FlowVarName.STATUS,
            dims=None,
            mask=self.data.has_status,
            binary=True,
        )

    @cached_property
    def size(self) -> linopy.Variable | None:
        """(flow, period, scenario) - size variable, masked to flows with investment."""
        if not self.data.with_investment:
            return None
        return self.add_variables(
            FlowVarName.SIZE,
            lower=self.data.size_minimum_all,
            upper=self.data.size_maximum_all,
            dims=('period', 'scenario'),
            mask=self.data.has_investment,
        )

    @cached_property
    def invested(self) -> linopy.Variable | None:
        """(flow, period, scenario) - binary invested variable, masked to optional investment."""
        if not self.data.with_optional_investment:
            return None
        return self.add_variables(
            FlowVarName.INVESTED,
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

        from .features import InvestmentBuilder

        dim = self.dim_name

        # Optional investment: size controlled by invested binary
        if self.invested is not None:
            InvestmentBuilder.add_optional_size_bounds(
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
        InvestmentBuilder.add_linked_periods_constraints(
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
        total_time = self.model.temporal_weight.sum(self.model.temporal_dims)

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

    def __init__(self, model: FlowSystemModel, data: FlowsData):
        """Initialize the type-level model for all flows.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            data: FlowsData container with batched flow data.
        """
        super().__init__(model, data)

        # Set reference on each flow element for element access pattern
        for flow in self.elements.values():
            flow.set_flows_model(self)

        self.create_variables()
        self.create_status_model()
        self.create_constraints()

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
        if self.data.with_status_only:
            self._constraint_status_bounds()
        if self.data.with_investment_only:
            self._constraint_investment_bounds()
        if self.data.with_status_and_investment:
            self._constraint_status_investment_bounds()

    def _constraint_investment_bounds(self) -> None:
        """
        Case: With investment, without status.
        rate <= size * relative_max, rate >= size * relative_min.

        Uses mask-based constraint creation - creates constraints for all flows but
        masks out non-investment flows.
        """
        mask = self._build_constraint_mask(self.data.with_investment_only, self.rate)

        if not mask.any():
            return

        # Upper bound: rate <= size * relative_max
        self.model.add_constraints(
            self.rate <= self.size * self.data.effective_relative_maximum,
            name=f'{self.dim_name}|invest_ub',  # TODO Rename to size_ub
            mask=mask,
        )

        # Lower bound: rate >= size * relative_min
        self.model.add_constraints(
            self.rate >= self.size * self.data.effective_relative_minimum,
            name=f'{self.dim_name}|invest_lb',  # TODO Rename to size_lb
            mask=mask,
        )

    def _constraint_status_bounds(self) -> None:
        """
        Case: With status, without investment.
        rate <= status * size * relative_max, rate >= status * epsilon."""
        flow_ids = self.data.with_status_only
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
        flow_ids = self.data.with_status_and_investment
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
        self.add_constraints(
            flow_rate <= status * big_m_upper, name='status+invest_ub1'
        )  # TODO Rename to status+size_ub1

        # Upper bound 2: rate <= size * relative_max
        self.add_constraints(flow_rate <= size * rel_max, name='status+invest_ub2')  # TODO Rename to status+size_ub2

        # Lower bound: rate >= (status - 1) * M + size * relative_min
        big_m_lower = max_size * rel_min
        rhs = (status - 1) * big_m_lower + size * rel_min
        self.add_constraints(flow_rate >= rhs, name='status+invest_lb')  # TODO Rename to status+size_lb2

    def _create_piecewise_effects(self) -> None:
        """Create batched piecewise effects for flows with piecewise_effects_of_investment.

        Uses PiecewiseBuilder for pad-to-max batching across all flows with
        piecewise effects. Creates batched segment variables, share variables,
        and coupling constraints.
        """
        from .features import PiecewiseBuilder

        dim = self.dim_name
        size_var = self.get(FlowVarName.SIZE)
        invested_var = self.get(FlowVarName.INVESTED)

        if size_var is None:
            return

        inv = self.data._investment_data
        if inv is None or not inv.piecewise_element_ids:
            return

        element_ids = inv.piecewise_element_ids
        segment_mask = inv.piecewise_segment_mask
        origin_starts = inv.piecewise_origin_starts
        origin_ends = inv.piecewise_origin_ends
        effect_starts = inv.piecewise_effect_starts
        effect_ends = inv.piecewise_effect_ends
        effect_names = inv.piecewise_effect_names
        max_segments = inv.piecewise_max_segments

        # Create batched piecewise variables
        base_coords = self.model.get_coords(['period', 'scenario'])
        name_prefix = f'{dim}|piecewise_effects'
        piecewise_vars = PiecewiseBuilder.create_piecewise_variables(
            self.model,
            element_ids,
            max_segments,
            dim,
            segment_mask,
            base_coords,
            name_prefix,
        )

        # Create piecewise constraints
        PiecewiseBuilder.create_piecewise_constraints(
            self.model,
            piecewise_vars,
            name_prefix,
        )

        # Tighten single_segment constraint for optional elements: sum(inside_piece) <= invested
        # This helps the LP relaxation by immediately forcing inside_piece=0 when invested=0.
        if invested_var is not None:
            invested_ids = set(invested_var.coords[dim].values)
            optional_ids = [fid for fid in element_ids if fid in invested_ids]
            if optional_ids:
                inside_piece = piecewise_vars['inside_piece'].sel({dim: optional_ids})
                self.model.add_constraints(
                    inside_piece.sum('segment') <= invested_var.sel({dim: optional_ids}),
                    name=f'{name_prefix}|single_segment_invested',
                )

        # Create coupling constraint for size (origin)
        size_subset = size_var.sel({dim: element_ids})
        PiecewiseBuilder.create_coupling_constraint(
            self.model,
            size_subset,
            piecewise_vars['lambda0'],
            piecewise_vars['lambda1'],
            origin_starts,
            origin_ends,
            f'{name_prefix}|size|coupling',
        )

        # Create share variable with (dim, effect) and vectorized coupling constraint
        coords_dict = {dim: pd.Index(element_ids, name=dim), 'effect': effect_names}
        if base_coords is not None:
            coords_dict.update(dict(base_coords))

        share_var = self.model.add_variables(
            lower=-np.inf,
            upper=np.inf,
            coords=xr.Coordinates(coords_dict),
            name=f'{name_prefix}|share',
        )
        PiecewiseBuilder.create_coupling_constraint(
            self.model,
            share_var,
            piecewise_vars['lambda0'],
            piecewise_vars['lambda1'],
            effect_starts,
            effect_ends,
            f'{name_prefix}|coupling',
        )

        # Sum over element dim, keep effect dim
        self.model.effects.add_share_periodic(share_var.sum(dim))

        logger.debug(f'Created batched piecewise effects for {len(element_ids)} flows')

    def add_effect_contributions(self, effects_model) -> None:
        """Push ALL effect contributions from flows to EffectsModel.

        Called by EffectsModel.finalize_shares(). Pushes:
        - Temporal share: rate × effects_per_flow_hour × dt
        - Status effects: status × effects_per_active_hour × dt, startup × effects_per_startup
        - Periodic share: size × effects_per_size
        - Investment/retirement: invested × factor
        - Constants: mandatory fixed + retirement constants

        Args:
            effects_model: The EffectsModel to register contributions with.
        """
        dim = self.dim_name
        dt = self.model.timestep_duration

        # === Temporal: rate * effects_per_flow_hour * dt ===
        # Batched over flows and effects - _accumulate_shares handles effect dim internally
        factors = self.data.effects_per_flow_hour
        if factors is not None:
            flow_ids = factors.coords[dim].values
            rate_subset = self.rate.sel({dim: flow_ids})
            effects_model.add_temporal_contribution(rate_subset * (factors * dt), contributor_dim=dim)

        # === Temporal: status effects ===
        if self.status is not None:
            # effects_per_active_hour
            factor = self.data.effects_per_active_hour
            if factor is not None:
                flow_ids = factor.coords[dim].values
                status_subset = self.status.sel({dim: flow_ids})
                effects_model.add_temporal_contribution(status_subset * (factor * dt), contributor_dim=dim)

            # effects_per_startup
            factor = self.data.effects_per_startup
            if self.startup is not None and factor is not None:
                flow_ids = factor.coords[dim].values
                startup_subset = self.startup.sel({dim: flow_ids})
                effects_model.add_temporal_contribution(startup_subset * factor, contributor_dim=dim)

        # === Periodic: size * effects_per_size ===
        inv = self.data._investment_data
        if inv is not None and inv.effects_per_size is not None:
            factors = inv.effects_per_size
            flow_ids = factors.coords[dim].values
            size_subset = self.size.sel({dim: flow_ids})
            effects_model.add_periodic_contribution(size_subset * factors, contributor_dim=dim)

        # === Investment/retirement effects (optional investments) ===
        if inv is not None and self.invested is not None:
            if (ff := inv.effects_of_investment) is not None:
                flow_ids = ff.coords[dim].values
                invested_subset = self.invested.sel({dim: flow_ids})
                effects_model.add_periodic_contribution(invested_subset * ff, contributor_dim=dim)

            if (ff := inv.effects_of_retirement) is not None:
                flow_ids = ff.coords[dim].values
                invested_subset = self.invested.sel({dim: flow_ids})
                effects_model.add_periodic_contribution(invested_subset * (-ff), contributor_dim=dim)

        # === Constants: mandatory fixed + retirement ===
        if inv is not None:
            if inv.effects_of_investment_mandatory is not None:
                effects_model.add_periodic_contribution(inv.effects_of_investment_mandatory, contributor_dim=dim)
            if inv.effects_of_retirement_constant is not None:
                effects_model.add_periodic_contribution(inv.effects_of_retirement_constant, contributor_dim=dim)

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
            FlowVarName.ACTIVE_HOURS,
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
        return self.add_variables(FlowVarName.STARTUP, dims=None, element_ids=ids, binary=True)

    @cached_property
    def shutdown(self) -> linopy.Variable | None:
        """(flow, time, ...) - binary shutdown variable."""
        ids = self.data.with_startup_tracking
        if not ids:
            return None
        return self.add_variables(FlowVarName.SHUTDOWN, dims=None, element_ids=ids, binary=True)

    @cached_property
    def inactive(self) -> linopy.Variable | None:
        """(flow, time, ...) - binary inactive variable."""
        ids = self.data.with_downtime_tracking
        if not ids:
            return None
        return self.add_variables(FlowVarName.INACTIVE, dims=None, element_ids=ids, binary=True)

    @cached_property
    def startup_count(self) -> linopy.Variable | None:
        """(flow, period, scenario) - startup count."""
        ids = self.data.with_startup_limit
        if not ids:
            return None
        return self.add_variables(
            FlowVarName.STARTUP_COUNT,
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
        from .features import StatusBuilder

        prev = sd.previous_uptime
        var = StatusBuilder.add_batched_duration_tracking(
            model=self.model,
            state=self.status.sel({self.dim_name: sd.with_uptime_tracking}),
            name=FlowVarName.UPTIME,
            dim_name=self.dim_name,
            timestep_duration=self.model.timestep_duration,
            minimum_duration=sd.min_uptime,
            maximum_duration=sd.max_uptime,
            previous_duration=prev if prev is not None and fast_notnull(prev).any() else None,
        )
        self._variables[FlowVarName.UPTIME] = var
        return var

    @cached_property
    def downtime(self) -> linopy.Variable | None:
        """(flow, time, ...) - consecutive downtime duration."""
        sd = self.data
        if not sd.with_downtime_tracking:
            return None
        from .features import StatusBuilder

        prev = sd.previous_downtime
        var = StatusBuilder.add_batched_duration_tracking(
            model=self.model,
            state=self.inactive,
            name=FlowVarName.DOWNTIME,
            dim_name=self.dim_name,
            timestep_duration=self.model.timestep_duration,
            minimum_duration=sd.min_downtime,
            maximum_duration=sd.max_downtime,
            previous_duration=prev if prev is not None and fast_notnull(prev).any() else None,
        )
        self._variables[FlowVarName.DOWNTIME] = var
        return var

    # === Status Constraints ===

    def _status_sel(self, element_ids: list[str]) -> linopy.Variable:
        """Select status variable for a subset of element IDs."""
        return self.status.sel({self.dim_name: element_ids})

    def constraint_active_hours(self) -> None:
        """Constrain active_hours == sum_temporal(status)."""
        if self.active_hours is None:
            return
        StatusBuilder.add_active_hours_constraint(
            self.model,
            self.active_hours,
            self.status,
            FlowVarName.Constraint.ACTIVE_HOURS,
        )

    def constraint_complementary(self) -> None:
        """Constrain status + inactive == 1 for downtime tracking flows."""
        if self.inactive is None:
            return
        StatusBuilder.add_complementary_constraint(
            self.model,
            self._status_sel(self.data.with_downtime_tracking),
            self.inactive,
            FlowVarName.Constraint.COMPLEMENTARY,
        )

    def constraint_switch_transition(self) -> None:
        """Constrain startup[t] - shutdown[t] == status[t] - status[t-1] for t > 0."""
        if self.startup is None:
            return
        StatusBuilder.add_switch_transition_constraint(
            self.model,
            self._status_sel(self.data.with_startup_tracking),
            self.startup,
            self.shutdown,
            FlowVarName.Constraint.SWITCH_TRANSITION,
        )

    def constraint_switch_mutex(self) -> None:
        """Constrain startup + shutdown <= 1."""
        if self.startup is None:
            return
        StatusBuilder.add_switch_mutex_constraint(
            self.model,
            self.startup,
            self.shutdown,
            FlowVarName.Constraint.SWITCH_MUTEX,
        )

    def constraint_switch_initial(self) -> None:
        """Constrain startup[0] - shutdown[0] == status[0] - previous_status[-1]."""
        if self.startup is None:
            return
        dim = self.dim_name
        ids = [eid for eid in self.data.with_startup_tracking if eid in self.data.previous_states]
        if not ids:
            return

        prev_arrays = [self.data.previous_states[eid].expand_dims({dim: [eid]}) for eid in ids]
        prev_state = xr.concat(prev_arrays, dim=dim).isel(time=-1)

        StatusBuilder.add_switch_initial_constraint(
            self.model,
            self._status_sel(ids).isel(time=0),
            self.startup.sel({dim: ids}).isel(time=0),
            self.shutdown.sel({dim: ids}).isel(time=0),
            prev_state,
            FlowVarName.Constraint.SWITCH_INITIAL,
        )

    def constraint_startup_count(self) -> None:
        """Constrain startup_count == sum(startup) over temporal dims."""
        if self.startup_count is None:
            return
        startup_subset = self.startup.sel({self.dim_name: self.data.with_startup_limit})
        StatusBuilder.add_startup_count_constraint(
            self.model,
            self.startup_count,
            startup_subset,
            self.dim_name,
            FlowVarName.Constraint.STARTUP_COUNT,
        )

    def constraint_cluster_cyclic(self) -> None:
        """Constrain status[0] == status[-1] for cyclic cluster mode."""
        if self.model.flow_system.clusters is None:
            return
        params = self.data.status_params
        cyclic_ids = [eid for eid in self.data.with_status if params[eid].cluster_mode == 'cyclic']
        if not cyclic_ids:
            return
        StatusBuilder.add_cluster_cyclic_constraint(
            self.model,
            self._status_sel(cyclic_ids),
            FlowVarName.Constraint.CLUSTER_CYCLIC,
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
        fid = flow.id
        return self.data.previous_states.get(fid)


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

    def __init__(self, model: FlowSystemModel, data: BusesData, flows_model: FlowsModel):
        """Initialize the type-level model for all buses.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            data: BusesData container.
            flows_model: The FlowsModel containing flow_rate variables.
        """
        super().__init__(model, data)
        self._flows_model = flows_model

        # Categorize buses by their features
        self.buses_with_imbalance: list[Bus] = data.imbalance_elements

        # Element ID lists for subsets
        self.imbalance_ids: list[str] = data.with_imbalance

        # Set reference on each bus element
        for bus in self.elements.values():
            bus._buses_model = self

        self.create_variables()
        self.create_constraints()
        self.create_effect_shares()

    def create_variables(self) -> None:
        """Create all batched variables for buses.

        Creates:
        - virtual_supply: For buses with imbalance penalty
        - virtual_demand: For buses with imbalance penalty
        """
        if self.buses_with_imbalance:
            # virtual_supply: allows adding flow to meet demand
            self.add_variables(
                BusVarName.VIRTUAL_SUPPLY,
                lower=0.0,
                dims=self.model.temporal_dims,
                element_ids=self.imbalance_ids,
            )

            # virtual_demand: allows removing excess flow
            self.add_variables(
                BusVarName.VIRTUAL_DEMAND,
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
        flow_rate = self._flows_model[FlowVarName.RATE]
        flow_dim = self._flows_model.dim_name  # 'flow'
        bus_dim = self.dim_name  # 'bus'

        bus_ids = list(self.elements.keys())
        if not bus_ids:
            logger.debug('BusesModel: no buses, skipping balance constraints')
            return

        balance = sparse_multiply_sum(flow_rate, self.data.balance_coefficients, sum_dim=flow_dim, group_dim=bus_dim)

        if self.buses_with_imbalance:
            imbalance_ids = [b.id for b in self.buses_with_imbalance]
            is_imbalance = xr.DataArray(
                [b in imbalance_ids for b in bus_ids], dims=[bus_dim], coords={bus_dim: bus_ids}
            )

            # Buses without imbalance: balance == 0
            self.model.add_constraints(balance == 0, name='bus|balance', mask=~is_imbalance)

            # Buses with imbalance: balance + virtual_supply - virtual_demand == 0
            balance_imbalance = balance.sel({bus_dim: imbalance_ids})
            virtual_balance = balance_imbalance + self[BusVarName.VIRTUAL_SUPPLY] - self[BusVarName.VIRTUAL_DEMAND]
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
            bus_label = bus.id
            imbalance_penalty = self.data.aligned_imbalance_penalty(bus) * self.model.timestep_duration

            virtual_supply = self[BusVarName.VIRTUAL_SUPPLY].sel({dim: bus_label})
            virtual_demand = self[BusVarName.VIRTUAL_DEMAND].sel({dim: bus_label})

            total_imbalance_penalty = (virtual_supply + virtual_demand) * imbalance_penalty
            penalty_specs.append((bus_label, total_imbalance_penalty))

        return penalty_specs

    def create_effect_shares(self) -> None:
        """Create penalty effect shares for buses with imbalance."""
        from .effects import PENALTY_EFFECT_LABEL

        for element_label, expression in self.collect_penalty_share_specs():
            share_var = self.model.add_variables(
                coords=self.model.get_coords(self.model.temporal_dims),
                name=f'{element_label}->Penalty(temporal)',
            )
            self.model.add_constraints(
                share_var == expression,
                name=f'{element_label}->Penalty(temporal)',
            )
            self.model.effects.add_share_temporal(share_var.expand_dims(effect=[PENALTY_EFFECT_LABEL]))

    def get_variable(self, name: str, element_id: str | None = None):
        """Get a variable, optionally selecting a specific element.

        Args:
            name: Variable name (e.g., BusVarName.VIRTUAL_SUPPLY).
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

    def __init__(
        self,
        model: FlowSystemModel,
        data: ComponentsData,
        flows_model: FlowsModel,
    ):
        super().__init__(model, data)
        self._logger = logging.getLogger('flixopt')
        self._flows_model = flows_model
        self._logger.debug(f'ComponentsModel initialized: {len(self.element_ids)} with status')
        self.create_variables()
        self.create_constraints()
        self.create_status_features()
        self.create_effect_shares()
        self.constraint_prevent_simultaneous()

    @property
    def components(self) -> list[Component]:
        """List of components with status (alias for elements.values())."""
        return list(self.elements.values())

    def create_variables(self) -> None:
        """Create batched component status variable with component dimension."""
        if not self.components:
            return

        self.add_variables(ComponentVarName.STATUS, dims=None, binary=True)
        self._logger.debug(f'ComponentsModel created status variable for {len(self.components)} components')

    def create_constraints(self) -> None:
        """Create batched constraints linking component status to flow statuses.

        Uses mask matrix for batched constraint creation:
        - Single-flow components: comp_status == flow_status (equality)
        - Multi-flow components: bounded by flow sum with epsilon tolerance
        """
        if not self.components:
            return

        comp_status = self[ComponentVarName.STATUS]
        flow_status = self._flows_model[FlowVarName.STATUS]
        mask = self.data.flow_mask
        n_flows = self.data.flow_count

        # Sum of flow statuses for each component: (component, time, ...)
        flow_sum = sparse_weighted_sum(flow_status, mask, sum_dim='flow', group_dim='component')

        # Separate single-flow vs multi-flow components
        single_flow_ids = [c.id for c in self.components if len(c.inputs) + len(c.outputs) == 1]
        multi_flow_ids = [c.id for c in self.components if len(c.inputs) + len(c.outputs) > 1]

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
            previous_status = []
            for flow in component.flows.values():
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
                comp_prev_status = comp_prev_status.expand_dims({self.dim_name: [component.id]})
                previous_arrays.append(comp_prev_status)
                components_with_previous.append(component)

        if not previous_arrays:
            return None

        return xr.concat(previous_arrays, dim=self.dim_name)

    # === Status Variables (cached_property) ===

    @cached_property
    def active_hours(self) -> linopy.Variable | None:
        """(component, period, scenario) - total active hours for components with status."""
        if not self.components:
            return None

        sd = self.data.status_data
        dim = self.dim_name
        total_hours = self.model.temporal_weight.sum(self.model.temporal_dims)

        min_vals = [sd._params[eid].active_hours_min or 0 for eid in sd.ids]
        max_list = [sd._params[eid].active_hours_max for eid in sd.ids]
        lower = xr.DataArray(min_vals, dims=[dim], coords={dim: sd.ids})
        has_max = xr.DataArray([v is not None for v in max_list], dims=[dim], coords={dim: sd.ids})
        raw_max = xr.DataArray([v if v is not None else 0 for v in max_list], dims=[dim], coords={dim: sd.ids})
        upper = xr.where(has_max, raw_max, total_hours)

        return self.add_variables(
            ComponentVarName.ACTIVE_HOURS,
            lower=lower,
            upper=upper,
            dims=('period', 'scenario'),
            element_ids=sd.ids,
        )

    @cached_property
    def startup(self) -> linopy.Variable | None:
        """(component, time, ...) - binary startup variable."""
        ids = self.data.status_data.with_startup_tracking
        if not ids:
            return None
        return self.add_variables(ComponentVarName.STARTUP, dims=None, element_ids=ids, binary=True)

    @cached_property
    def shutdown(self) -> linopy.Variable | None:
        """(component, time, ...) - binary shutdown variable."""
        ids = self.data.status_data.with_startup_tracking
        if not ids:
            return None
        return self.add_variables(ComponentVarName.SHUTDOWN, dims=None, element_ids=ids, binary=True)

    @cached_property
    def inactive(self) -> linopy.Variable | None:
        """(component, time, ...) - binary inactive variable."""
        ids = self.data.status_data.with_downtime_tracking
        if not ids:
            return None
        return self.add_variables(ComponentVarName.INACTIVE, dims=None, element_ids=ids, binary=True)

    @cached_property
    def startup_count(self) -> linopy.Variable | None:
        """(component, period, scenario) - startup count."""
        ids = self.data.status_data.with_startup_limit
        if not ids:
            return None
        return self.add_variables(
            ComponentVarName.STARTUP_COUNT,
            lower=0,
            upper=self.data.status_data.startup_limit,
            dims=('period', 'scenario'),
            element_ids=ids,
        )

    @cached_property
    def uptime(self) -> linopy.Variable | None:
        """(component, time, ...) - consecutive uptime duration."""
        sd = self.data.status_data
        if not sd.with_uptime_tracking:
            return None
        from .features import StatusBuilder

        prev = sd.previous_uptime
        var = StatusBuilder.add_batched_duration_tracking(
            model=self.model,
            state=self[ComponentVarName.STATUS].sel({self.dim_name: sd.with_uptime_tracking}),
            name=ComponentVarName.UPTIME,
            dim_name=self.dim_name,
            timestep_duration=self.model.timestep_duration,
            minimum_duration=sd.min_uptime,
            maximum_duration=sd.max_uptime,
            previous_duration=prev if prev is not None and fast_notnull(prev).any() else None,
        )
        self._variables[ComponentVarName.UPTIME] = var
        return var

    @cached_property
    def downtime(self) -> linopy.Variable | None:
        """(component, time, ...) - consecutive downtime duration."""
        sd = self.data.status_data
        if not sd.with_downtime_tracking:
            return None
        from .features import StatusBuilder

        _ = self.inactive  # ensure inactive variable exists
        prev = sd.previous_downtime
        var = StatusBuilder.add_batched_duration_tracking(
            model=self.model,
            state=self.inactive,
            name=ComponentVarName.DOWNTIME,
            dim_name=self.dim_name,
            timestep_duration=self.model.timestep_duration,
            minimum_duration=sd.min_downtime,
            maximum_duration=sd.max_downtime,
            previous_duration=prev if prev is not None and fast_notnull(prev).any() else None,
        )
        self._variables[ComponentVarName.DOWNTIME] = var
        return var

    # === Status Constraints ===

    def _status_sel(self, element_ids: list[str]) -> linopy.Variable:
        """Select status variable for a subset of component IDs."""
        return self[ComponentVarName.STATUS].sel({self.dim_name: element_ids})

    def constraint_active_hours(self) -> None:
        """Constrain active_hours == sum_temporal(status)."""
        if self.active_hours is None:
            return
        StatusBuilder.add_active_hours_constraint(
            self.model,
            self.active_hours,
            self[ComponentVarName.STATUS],
            ComponentVarName.Constraint.ACTIVE_HOURS,
        )

    def constraint_complementary(self) -> None:
        """Constrain status + inactive == 1 for downtime tracking components."""
        if self.inactive is None:
            return
        StatusBuilder.add_complementary_constraint(
            self.model,
            self._status_sel(self.data.status_data.with_downtime_tracking),
            self.inactive,
            ComponentVarName.Constraint.COMPLEMENTARY,
        )

    def constraint_switch_transition(self) -> None:
        """Constrain startup[t] - shutdown[t] == status[t] - status[t-1] for t > 0."""
        if self.startup is None:
            return
        StatusBuilder.add_switch_transition_constraint(
            self.model,
            self._status_sel(self.data.status_data.with_startup_tracking),
            self.startup,
            self.shutdown,
            ComponentVarName.Constraint.SWITCH_TRANSITION,
        )

    def constraint_switch_mutex(self) -> None:
        """Constrain startup + shutdown <= 1."""
        if self.startup is None:
            return
        StatusBuilder.add_switch_mutex_constraint(
            self.model,
            self.startup,
            self.shutdown,
            ComponentVarName.Constraint.SWITCH_MUTEX,
        )

    def constraint_switch_initial(self) -> None:
        """Constrain startup[0] - shutdown[0] == status[0] - previous_status[-1]."""
        if self.startup is None:
            return
        dim = self.dim_name
        previous_status = self.data.status_data._previous_states
        ids = [eid for eid in self.data.status_data.with_startup_tracking if eid in previous_status]
        if not ids:
            return

        prev_arrays = [previous_status[eid].expand_dims({dim: [eid]}) for eid in ids]
        prev_state = xr.concat(prev_arrays, dim=dim).isel(time=-1)

        StatusBuilder.add_switch_initial_constraint(
            self.model,
            self._status_sel(ids).isel(time=0),
            self.startup.sel({dim: ids}).isel(time=0),
            self.shutdown.sel({dim: ids}).isel(time=0),
            prev_state,
            ComponentVarName.Constraint.SWITCH_INITIAL,
        )

    def constraint_startup_count(self) -> None:
        """Constrain startup_count == sum(startup) over temporal dims."""
        if self.startup_count is None:
            return
        startup_subset = self.startup.sel({self.dim_name: self.data.status_data.with_startup_limit})
        StatusBuilder.add_startup_count_constraint(
            self.model,
            self.startup_count,
            startup_subset,
            self.dim_name,
            ComponentVarName.Constraint.STARTUP_COUNT,
        )

    def constraint_cluster_cyclic(self) -> None:
        """Constrain status[0] == status[-1] for cyclic cluster mode."""
        if self.model.flow_system.clusters is None:
            return
        params = self.data.status_data._params
        cyclic_ids = [eid for eid in self.data.status_data.ids if params[eid].cluster_mode == 'cyclic']
        if not cyclic_ids:
            return
        StatusBuilder.add_cluster_cyclic_constraint(
            self.model,
            self._status_sel(cyclic_ids),
            ComponentVarName.Constraint.CLUSTER_CYCLIC,
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

    def add_effect_contributions(self, effects_model) -> None:
        """Push component-level status effect contributions to EffectsModel.

        Called by EffectsModel.finalize_shares(). Pushes:
        - Temporal: status × effects_per_active_hour × dt
        - Temporal: startup × effects_per_startup

        Args:
            effects_model: The EffectsModel to register contributions with.
        """
        dim = self.dim_name
        dt = self.model.timestep_duration
        sd = self.data.status_data

        # === Temporal: status * effects_per_active_hour * dt ===
        if self.status is not None:
            factor = sd.effects_per_active_hour
            if factor is not None:
                component_ids = factor.coords[dim].values
                status_subset = self.status.sel({dim: component_ids})
                effects_model.add_temporal_contribution(status_subset * (factor * dt), contributor_dim=dim)

        # === Temporal: startup * effects_per_startup ===
        if self.startup is not None:
            factor = sd.effects_per_startup
            if factor is not None:
                component_ids = factor.coords[dim].values
                startup_subset = self.startup.sel({dim: component_ids})
                effects_model.add_temporal_contribution(startup_subset * factor, contributor_dim=dim)

    def constraint_prevent_simultaneous(self) -> None:
        """Create mutual exclusivity constraints for components with prevent_simultaneous_flows."""
        _add_prevent_simultaneous_constraints(
            self.data.with_prevent_simultaneous, self._flows_model, self.model, 'prevent_simultaneous'
        )

    # === Variable accessor properties ===

    @property
    def status(self) -> linopy.Variable | None:
        """Batched component status variable with (component, time) dims."""
        return (
            self.model.variables[ComponentVarName.STATUS] if ComponentVarName.STATUS in self.model.variables else None
        )

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


class ConvertersModel(TypeModel):
    """Type-level model for ALL converter constraints.

    Handles LinearConverters with:
    1. Linear conversion factors: sum(flow * coeff * sign) == 0
    2. Piecewise conversion: inside_piece, lambda0, lambda1 + coupling constraints
    """

    def __init__(
        self,
        model: FlowSystemModel,
        data: ConvertersData,
        flows_model: FlowsModel,
    ):
        """Initialize the converter model.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            data: ConvertersData container.
            flows_model: The FlowsModel that owns flow variables.
        """
        from .features import PiecewiseBuilder

        super().__init__(model, data)
        self.converters_with_factors = data.with_factors
        self.converters_with_piecewise = data.with_piecewise
        self._flows_model = flows_model
        self._PiecewiseBuilder = PiecewiseBuilder

        # Piecewise conversion variables
        self._piecewise_variables: dict[str, linopy.Variable] = {}

        logger.debug(
            f'ConvertersModel initialized: {len(self.converters_with_factors)} with factors, '
            f'{len(self.converters_with_piecewise)} with piecewise'
        )
        self.create_variables()
        self.create_constraints()

    def create_linear_constraints(self) -> None:
        """Create batched linear conversion factor constraints.

        For each converter c with equation i:
            sum_f(flow_rate[f] * coefficient[c,i,f] * sign[c,f]) == 0

        Uses sparse_multiply_sum: each converter only touches its own 2-3 flows
        instead of allocating a dense coefficient array across all flows.
        """
        if not self.converters_with_factors:
            return

        d = self.data  # ConvertersData
        flow_rate = self._flows_model[FlowVarName.RATE]

        # Sparse sum: only multiplies non-zero (converter, flow) pairs
        flow_sum = sparse_multiply_sum(flow_rate, d.signed_coefficients, sum_dim='flow', group_dim='converter')

        # Build valid mask: True where converter HAS that equation
        equation_indices = xr.DataArray(
            list(range(d.max_equations)),
            dims=['equation_idx'],
            coords={'equation_idx': list(range(d.max_equations))},
        )
        valid_mask = equation_indices < d.n_equations_per_converter

        self.add_constraints(
            flow_sum == 0,
            name=ConverterVarName.Constraint.CONVERSION,
            mask=valid_mask,
        )

        logger.debug(f'ConvertersModel created linear constraints for {len(self.converters_with_factors)} converters')

    def create_variables(self) -> None:
        """Create all batched variables for converters (piecewise variables)."""
        self._create_piecewise_variables()

    def create_constraints(self) -> None:
        """Create all batched constraints for converters."""
        self.create_linear_constraints()
        self._create_piecewise_constraints()

    def _create_piecewise_variables(self) -> dict[str, linopy.Variable]:
        """Create batched piecewise conversion variables.

        Returns:
            Dict with 'inside_piece', 'lambda0', 'lambda1' variables.
        """
        if not self.converters_with_piecewise:
            return {}

        d = self.data  # ConvertersData
        base_coords = self.model.get_coords(['time', 'period', 'scenario'])

        self._piecewise_variables = self._PiecewiseBuilder.create_piecewise_variables(
            self.model,
            d.piecewise_element_ids,
            d.piecewise_max_segments,
            d.dim_name,
            d.piecewise_segment_mask,
            base_coords,
            ConverterVarName.PIECEWISE_PREFIX,
        )

        logger.debug(
            f'ConvertersModel created piecewise variables for {len(self.converters_with_piecewise)} converters'
        )
        return self._piecewise_variables

    def _create_piecewise_constraints(self) -> None:
        """Create batched piecewise constraints and coupling constraints."""
        if not self.converters_with_piecewise:
            return

        # Create lambda_sum and single_segment constraints
        # TODO: Integrate status from ComponentsModel when converters overlap
        self._PiecewiseBuilder.create_piecewise_constraints(
            self.model,
            self._piecewise_variables,
            ConverterVarName.PIECEWISE_PREFIX,
        )

        # Create batched coupling constraints for all piecewise flows
        bp = self.data.piecewise_breakpoints  # Dataset with (converter, segment, flow) dims
        if bp is None:
            return

        flow_rate = self._flows_model[FlowVarName.RATE]
        lambda0 = self._piecewise_variables['lambda0']
        lambda1 = self._piecewise_variables['lambda1']

        # Each flow belongs to exactly one converter. Select the owning converter
        # per flow directly instead of broadcasting across all (converter × flow).
        starts = bp['starts']  # (converter, segment, flow, [time])
        ends = bp['ends']

        # Find which converter owns each flow (first non-NaN along converter)
        notnull = fast_notnull(starts)
        for d in notnull.dims:
            if d not in ('flow', 'converter'):
                notnull = notnull.any(d)
        owner_idx = notnull.argmax('converter')  # (flow,)
        owner_ids = starts.coords['converter'].values[owner_idx.values]

        # Select breakpoints and lambdas for the owning converter per flow
        owner_da = xr.DataArray(owner_ids, dims=['flow'], coords={'flow': starts.coords['flow']})
        flow_starts = starts.sel(converter=owner_da).drop_vars('converter')
        flow_ends = ends.sel(converter=owner_da).drop_vars('converter')
        flow_lambda0 = lambda0.sel(converter=owner_da)
        flow_lambda1 = lambda1.sel(converter=owner_da)

        # Reconstruct: sum over segments only (no converter dim)
        reconstructed_per_flow = (flow_lambda0 * flow_starts + flow_lambda1 * flow_ends).sum('segment')
        # Drop dangling converter coord left by vectorized sel()
        reconstructed_per_flow = reconstructed_per_flow.drop_vars('converter', errors='ignore')

        # Get flow rates for piecewise flows
        flow_ids = list(bp.coords['flow'].values)
        piecewise_flow_rate = flow_rate.sel(flow=flow_ids)

        # Add single batched constraint
        self.add_constraints(
            piecewise_flow_rate == reconstructed_per_flow,
            name=ConverterVarName.Constraint.PIECEWISE_COUPLING,
        )

        logger.debug(
            f'ConvertersModel created piecewise constraints for {len(self.converters_with_piecewise)} converters'
        )


class TransmissionsModel(TypeModel):
    """Type-level model for batched transmission efficiency constraints.

    Handles Transmission components with batched constraints:
    - Efficiency: out = in * (1 - rel_losses) - status * abs_losses
    - Balanced size: in1.size == in2.size

    All constraints have a 'transmission' dimension for proper batching.
    """

    def __init__(
        self,
        model: FlowSystemModel,
        data: TransmissionsData,
        flows_model: FlowsModel,
    ):
        """Initialize the transmission model.

        Args:
            model: The FlowSystemModel to create constraints in.
            data: TransmissionsData container.
            flows_model: The FlowsModel that owns flow variables.
        """
        super().__init__(model, data)
        self.transmissions = list(self.elements.values())
        self._flows_model = flows_model

        logger.debug(f'TransmissionsModel initialized: {len(self.transmissions)} transmissions')
        self.create_variables()
        self.create_constraints()
        _add_prevent_simultaneous_constraints(
            self.transmissions, self._flows_model, self.model, 'transmission|prevent_simultaneous'
        )

    def create_variables(self) -> None:
        """No variables needed for transmissions (constraint-only model)."""
        pass

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
        flow_rate = self._flows_model[FlowVarName.RATE]
        d = self.data  # TransmissionsData

        # === Direction 1: All transmissions (batched) ===
        # Use masks to batch flow selection: (flow_rate * mask).sum('flow') -> (transmission, time, ...)
        in1_rate = (flow_rate * d.in1_mask).sum('flow')
        out1_rate = (flow_rate * d.out1_mask).sum('flow')
        rel_losses = d.relative_losses
        abs_losses = d.absolute_losses

        # Build the efficiency expression: in1 * (1 - rel_losses) - abs_losses_term
        efficiency_expr = in1_rate * (1 - rel_losses)

        # Add absolute losses term if any transmission has them
        if d.transmissions_with_abs_losses:
            flow_status = self._flows_model[FlowVarName.STATUS]
            in1_status = (flow_status * d.in1_mask).sum('flow')
            efficiency_expr = efficiency_expr - in1_status * abs_losses

        # out1 == in1 * (1 - rel_losses) - in1_status * abs_losses
        self.add_constraints(
            out1_rate == efficiency_expr,
            name=con.DIR1,
        )

        # === Direction 2: Bidirectional transmissions only (batched) ===
        if d.bidirectional:
            in2_rate = (flow_rate * d.in2_mask).sum('flow')
            out2_rate = (flow_rate * d.out2_mask).sum('flow')
            rel_losses_bidir = d.relative_losses.sel({self.dim_name: d.bidirectional_ids})
            abs_losses_bidir = d.absolute_losses.sel({self.dim_name: d.bidirectional_ids})

            # Build the efficiency expression for direction 2
            efficiency_expr_2 = in2_rate * (1 - rel_losses_bidir)

            # Add absolute losses for bidirectional if any have them
            bidir_with_abs = [t.id for t in d.bidirectional if t.id in d.transmissions_with_abs_losses]
            if bidir_with_abs:
                flow_status = self._flows_model[FlowVarName.STATUS]
                in2_status = (flow_status * d.in2_mask).sum('flow')
                efficiency_expr_2 = efficiency_expr_2 - in2_status * abs_losses_bidir

            # out2 == in2 * (1 - rel_losses) - in2_status * abs_losses
            self.add_constraints(
                out2_rate == efficiency_expr_2,
                name=con.DIR2,
            )

        # === Balanced constraints: in1.size == in2.size (batched) ===
        if d.balanced:
            flow_size = self._flows_model[FlowVarName.SIZE]

            in1_size_batched = (flow_size * d.balanced_in1_mask).sum('flow')
            in2_size_batched = (flow_size * d.balanced_in2_mask).sum('flow')

            self.add_constraints(
                in1_size_batched == in2_size_batched,
                name=con.BALANCED,
            )

        logger.debug(f'TransmissionsModel created batched constraints for {len(self.transmissions)} transmissions')
