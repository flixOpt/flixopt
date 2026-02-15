"""
This module contains the basic components of the flixopt framework.
"""

from __future__ import annotations

import functools
import logging
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from . import io as fx_io
from .elements import Component, Flow
from .features import MaskHelpers, stack_along_dim
from .interface import InvestParameters, PiecewiseConversion, StatusParameters
from .modeling import _scalar_safe_reduce
from .structure import (
    FlowSystemModel,
    FlowVarName,
    InterclusterStorageVarName,
    StorageVarName,
    TypeModel,
    register_class_for_io,
)

if TYPE_CHECKING:
    import linopy

    from .batched import StoragesData
    from .types import Numeric_PS, Numeric_TPS

logger = logging.getLogger('flixopt')


@register_class_for_io
class LinearConverter(Component):
    """
    Converts input-Flows into output-Flows via linear conversion factors.

    LinearConverter models equipment that transforms one or more input flows into one or
    more output flows through linear relationships. This includes heat exchangers,
    electrical converters, chemical reactors, and other equipment where the
    relationship between inputs and outputs can be expressed as linear equations.

    The component supports two modeling approaches: simple conversion factors for
    straightforward linear relationships, or piecewise conversion for complex non-linear
    behavior approximated through piecewise linear segments.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/elements/LinearConverter/>

    Args:
        id: The id of the Element. Used to identify it in the FlowSystem.
        inputs: list of input Flows that feed into the converter.
        outputs: list of output Flows that are produced by the converter.
        status_parameters: Information about active and inactive state of LinearConverter.
            Component is active/inactive if all connected Flows are active/inactive. This induces a
            status variable (binary) in all Flows! If possible, use StatusParameters in a
            single Flow instead to keep the number of binary variables low.
        conversion_factors: Linear relationships between flows expressed as a list of
            dictionaries. Each dictionary maps flow ids to their coefficients in one
            linear equation. The number of conversion factors must be less than the total
            number of flows to ensure degrees of freedom > 0. Either 'conversion_factors'
            OR 'piecewise_conversion' can be used, but not both.
            For examples also look into the linear_converters.py file.
        piecewise_conversion: Define piecewise linear relationships between flow rates
            of different flows. Enables modeling of non-linear conversion behavior through
            linear approximation. Either 'conversion_factors' or 'piecewise_conversion'
            can be used, but not both.
        meta_data: Used to store additional information about the Element. Not used
            internally, but saved in results. Only use Python native types.

    Examples:
        Simple 1:1 heat exchanger with 95% efficiency:

        ```python
        heat_exchanger = LinearConverter(
            id='primary_hx',
            inputs=[hot_water_in],
            outputs=[hot_water_out],
            conversion_factors=[{'hot_water_in': 0.95, 'hot_water_out': 1}],
        )
        ```

        Multi-input heat pump with COP=3:

        ```python
        heat_pump = LinearConverter(
            id='air_source_hp',
            inputs=[electricity_in],
            outputs=[heat_output],
            conversion_factors=[{'electricity_in': 3, 'heat_output': 1}],
        )
        ```

        Combined heat and power (CHP) unit with multiple outputs:

        ```python
        chp_unit = LinearConverter(
            id='gas_chp',
            inputs=[natural_gas],
            outputs=[electricity_out, heat_out],
            conversion_factors=[
                {'natural_gas': 0.35, 'electricity_out': 1},
                {'natural_gas': 0.45, 'heat_out': 1},
            ],
        )
        ```

        Electrolyzer with multiple conversion relationships:

        ```python
        electrolyzer = LinearConverter(
            id='pem_electrolyzer',
            inputs=[electricity_in, water_in],
            outputs=[hydrogen_out, oxygen_out],
            conversion_factors=[
                {'electricity_in': 1, 'hydrogen_out': 50},  # 50 kWh/kg H2
                {'water_in': 1, 'hydrogen_out': 9},  # 9 kg H2O/kg H2
                {'hydrogen_out': 8, 'oxygen_out': 1},  # Mass balance
            ],
        )
        ```

        Complex converter with piecewise efficiency:

        ```python
        variable_efficiency_converter = LinearConverter(
            id='variable_converter',
            inputs=[fuel_in],
            outputs=[power_out],
            piecewise_conversion=PiecewiseConversion(
                {
                    'fuel_in': Piecewise(
                        [
                            Piece(0, 10),  # Low load operation
                            Piece(10, 25),  # High load operation
                        ]
                    ),
                    'power_out': Piecewise(
                        [
                            Piece(0, 3.5),  # Lower efficiency at part load
                            Piece(3.5, 10),  # Higher efficiency at full load
                        ]
                    ),
                }
            ),
        )
        ```

    Note:
        Conversion factors define linear relationships where the sum of (coefficient × flow_rate)
        equals zero for each equation: factor1×flow1 + factor2×flow2 + ... = 0
        Conversion factors define linear relationships:
        `{flow1: a1, flow2: a2, ...}` yields `a1×flow_rate1 + a2×flow_rate2 + ... = 0`.
        Note: The input format may be unintuitive. For example,
        `{"electricity": 1, "H2": 50}` implies `1×electricity = 50×H2`,
        i.e., 50 units of electricity produce 1 unit of H2.

        The system must have fewer conversion factors than total flows (degrees of freedom > 0)
        to avoid over-constraining the problem. For n total flows, use at most n-1 conversion factors.

        When using piecewise_conversion, the converter operates on one piece at a time,
        with binary variables determining which piece is active.

    """

    def __init__(
        self,
        id: str | None = None,
        inputs: list[Flow] | None = None,
        outputs: list[Flow] | None = None,
        status_parameters: StatusParameters | None = None,
        conversion_factors: list[dict[str, Numeric_TPS]] | None = None,
        piecewise_conversion: PiecewiseConversion | None = None,
        meta_data: dict | None = None,
        color: str | None = None,
        **kwargs,
    ):
        super().__init__(id, inputs, outputs, status_parameters, meta_data=meta_data, color=color, **kwargs)
        self.conversion_factors = conversion_factors or []
        self.piecewise_conversion = piecewise_conversion

    def link_to_flow_system(self, flow_system) -> None:
        """Propagate flow_system reference to parent Component."""
        super().link_to_flow_system(flow_system)

    @property
    def degrees_of_freedom(self):
        return len(self.inputs + self.outputs) - len(self.conversion_factors)


@register_class_for_io
class Storage(Component):
    """
    A Storage models the temporary storage and release of energy or material.

    Storages have one incoming and one outgoing Flow, each with configurable efficiency
    factors. They maintain a charge state variable that represents the stored amount,
    bounded by capacity limits and evolving over time based on charging, discharging,
    and self-discharge losses.

    The storage model handles complex temporal dynamics including initial conditions,
    final state constraints, and time-varying parameters. It supports both fixed-size
    and investment-optimized storage systems with comprehensive techno-economic modeling.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/elements/Storage/>

    Args:
        id: Element identifier used in the FlowSystem.
        charging: Incoming flow for loading the storage.
        discharging: Outgoing flow for unloading the storage.
        capacity_in_flow_hours: Storage capacity in flow-hours (kWh, m³, kg).
            Scalar for fixed size, InvestParameters for optimization, or None (unbounded).
            Default: None (unbounded capacity). When using InvestParameters,
            maximum_size (or fixed_size) must be explicitly set for proper model scaling.
        relative_minimum_charge_state: Minimum charge state (0-1). Default: 0.
        relative_maximum_charge_state: Maximum charge state (0-1). Default: 1.
        initial_charge_state: Charge at start. Numeric, 'equals_final', or None (free). Default: 0.
        minimal_final_charge_state: Minimum absolute charge required at end (optional).
        maximal_final_charge_state: Maximum absolute charge allowed at end (optional).
        relative_minimum_final_charge_state: Minimum relative charge at end.
            Defaults to last value of relative_minimum_charge_state.
        relative_maximum_final_charge_state: Maximum relative charge at end.
            Defaults to last value of relative_maximum_charge_state.
        eta_charge: Charging efficiency (0-1). Default: 1.
        eta_discharge: Discharging efficiency (0-1). Default: 1.
        relative_loss_per_hour: Self-discharge per hour (0-0.1). Default: 0.
        prevent_simultaneous_charge_and_discharge: Prevent charging and discharging
            simultaneously. Adds binary variables. Default: True.
        cluster_mode: How this storage is treated during clustering optimization.
            Only relevant when using ``transform.cluster()``. Options:

            - ``'independent'``: Clusters are fully decoupled. No constraints between
              clusters, each cluster has free start/end SOC. Fast but ignores
              seasonal storage value.
            - ``'cyclic'``: Each cluster is self-contained. The SOC at the start of
              each cluster equals its end (cluster returns to initial state).
              Good for "average day" modeling.
            - ``'intercluster'``: Link storage state across the original timeline using
              SOC boundary variables (Kotzur et al. approach). Properly values
              seasonal storage patterns. Overall SOC can drift.
            - ``'intercluster_cyclic'`` (default): Like 'intercluster' but also enforces
              that overall SOC returns to initial state (yearly cyclic).

        meta_data: Additional information stored in results. Python native types only.

    Examples:
        Battery energy storage system:

        ```python
        battery = Storage(
            id='lithium_battery',
            charging=battery_charge_flow,
            discharging=battery_discharge_flow,
            capacity_in_flow_hours=100,  # 100 kWh capacity
            eta_charge=0.95,  # 95% charging efficiency
            eta_discharge=0.95,  # 95% discharging efficiency
            relative_loss_per_hour=0.001,  # 0.1% loss per hour
            relative_minimum_charge_state=0.1,  # Never below 10% SOC
            relative_maximum_charge_state=0.9,  # Never above 90% SOC
        )
        ```

        Thermal storage with cycling constraints:

        ```python
        thermal_storage = Storage(
            id='hot_water_tank',
            charging=heat_input,
            discharging=heat_output,
            capacity_in_flow_hours=500,  # 500 kWh thermal capacity
            initial_charge_state=250,  # Start half full
            # Impact of temperature on energy capacity
            relative_maximum_charge_state=water_temperature_spread / rated_temeprature_spread,
            eta_charge=0.90,  # Heat exchanger losses
            eta_discharge=0.85,  # Distribution losses
            relative_loss_per_hour=0.02,  # 2% thermal loss per hour
            prevent_simultaneous_charge_and_discharge=True,
        )
        ```

        Pumped hydro storage with investment optimization:

        ```python
        pumped_hydro = Storage(
            id='pumped_hydro',
            charging=pump_flow,
            discharging=turbine_flow,
            capacity_in_flow_hours=InvestParameters(
                minimum_size=1000,  # Minimum economic scale
                maximum_size=10000,  # Site constraints
                specific_effects={'cost': 150},  # €150/MWh capacity
                fix_effects={'cost': 50_000_000},  # €50M fixed costs
            ),
            eta_charge=0.85,  # Pumping efficiency
            eta_discharge=0.90,  # Turbine efficiency
            initial_charge_state='equals_final',  # Ensuring no deficit compared to start
            relative_loss_per_hour=0.0001,  # Minimal evaporation
        )
        ```

        Material storage with inventory management:

        ```python
        fuel_storage = Storage(
            id='natural_gas_storage',
            charging=gas_injection,
            discharging=gas_withdrawal,
            capacity_in_flow_hours=10000,  # 10,000 m³ storage volume
            initial_charge_state=3000,  # Start with 3,000 m³
            minimal_final_charge_state=1000,  # Strategic reserve
            maximal_final_charge_state=9000,  # Prevent overflow
            eta_charge=0.98,  # Compression losses
            eta_discharge=0.95,  # Pressure reduction losses
            relative_loss_per_hour=0.0005,  # 0.05% leakage per hour
            prevent_simultaneous_charge_and_discharge=False,  # Allow flow-through
        )
        ```

    Note:
        **Mathematical formulation**: See [Storage](../user-guide/mathematical-notation/elements/Storage.md)
        for charge state evolution equations and balance constraints.

        **Efficiency parameters** (eta_charge, eta_discharge) are dimensionless (0-1 range).
        The relative_loss_per_hour represents exponential decay per hour.

        **Binary variables**: When prevent_simultaneous_charge_and_discharge is True, binary
        variables enforce mutual exclusivity, increasing solution time but preventing unrealistic
        simultaneous charging and discharging.

        **Unbounded capacity**: When capacity_in_flow_hours is None (default), the storage has
        unlimited capacity. Note that prevent_simultaneous_charge_and_discharge requires the
        charging and discharging flows to have explicit sizes. Use prevent_simultaneous_charge_and_discharge=False
        with unbounded storages, or set flow sizes explicitly.

        **Units**: Flow rates and charge states are related by the concept of 'flow hours' (=flow_rate * time).
        With flow rates in kW, the charge state is therefore (usually) kWh.
        With flow rates in m3/h, the charge state is therefore in m3.
    """

    def __init__(
        self,
        id: str | None = None,
        charging: Flow | None = None,
        discharging: Flow | None = None,
        capacity_in_flow_hours: Numeric_PS | InvestParameters | None = None,
        relative_minimum_charge_state: Numeric_TPS = 0,
        relative_maximum_charge_state: Numeric_TPS = 1,
        initial_charge_state: Numeric_PS | Literal['equals_final'] | None = 0,
        minimal_final_charge_state: Numeric_PS | None = None,
        maximal_final_charge_state: Numeric_PS | None = None,
        relative_minimum_final_charge_state: Numeric_PS | None = None,
        relative_maximum_final_charge_state: Numeric_PS | None = None,
        eta_charge: Numeric_TPS = 1,
        eta_discharge: Numeric_TPS = 1,
        relative_loss_per_hour: Numeric_TPS = 0,
        prevent_simultaneous_charge_and_discharge: bool = True,
        balanced: bool = False,
        cluster_mode: Literal['independent', 'cyclic', 'intercluster', 'intercluster_cyclic'] = 'intercluster_cyclic',
        meta_data: dict | None = None,
        color: str | None = None,
        **kwargs,
    ):
        # TODO: fixed_relative_chargeState implementieren
        super().__init__(
            id,
            inputs=[charging],
            outputs=[discharging],
            prevent_simultaneous_flows=[charging, discharging] if prevent_simultaneous_charge_and_discharge else None,
            meta_data=meta_data,
            color=color,
            **kwargs,
        )

        self.charging = charging
        self.discharging = discharging
        self.capacity_in_flow_hours = capacity_in_flow_hours
        self.relative_minimum_charge_state: Numeric_TPS = relative_minimum_charge_state
        self.relative_maximum_charge_state: Numeric_TPS = relative_maximum_charge_state

        self.relative_minimum_final_charge_state = relative_minimum_final_charge_state
        self.relative_maximum_final_charge_state = relative_maximum_final_charge_state

        self.initial_charge_state = initial_charge_state
        self.minimal_final_charge_state = minimal_final_charge_state
        self.maximal_final_charge_state = maximal_final_charge_state

        self.eta_charge: Numeric_TPS = eta_charge
        self.eta_discharge: Numeric_TPS = eta_discharge
        self.relative_loss_per_hour: Numeric_TPS = relative_loss_per_hour
        self.prevent_simultaneous_charge_and_discharge = prevent_simultaneous_charge_and_discharge
        self.balanced = balanced
        self.cluster_mode = cluster_mode

    def link_to_flow_system(self, flow_system) -> None:
        """Propagate flow_system reference to parent Component."""
        super().link_to_flow_system(flow_system)

    def __repr__(self) -> str:
        """Return string representation."""
        # Use build_repr_from_init directly to exclude charging and discharging
        return fx_io.build_repr_from_init(
            self,
            excluded_params={'self', 'id', 'charging', 'discharging', 'kwargs'},
            skip_default_size=True,
        ) + fx_io.format_flow_details(self)


@register_class_for_io
class Transmission(Component):
    """
    Models transmission infrastructure that transports flows between two locations with losses.

    Transmission components represent physical infrastructure like pipes, cables,
    transmission lines, or conveyor systems that transport energy or materials between
    two points. They can model both unidirectional and bidirectional flow with
    configurable loss mechanisms and operational constraints.

    The component supports complex transmission scenarios including relative losses
    (proportional to flow), absolute losses (fixed when active), and bidirectional
    operation with flow direction constraints.

    Args:
        id: The id of the Element. Used to identify it in the FlowSystem.
        in1: The primary inflow (side A). Pass InvestParameters here for capacity optimization.
        out1: The primary outflow (side B).
        in2: Optional secondary inflow (side B) for bidirectional operation.
            If in1 has InvestParameters, in2 will automatically have matching capacity.
        out2: Optional secondary outflow (side A) for bidirectional operation.
        relative_losses: Proportional losses as fraction of throughput (e.g., 0.02 for 2% loss).
            Applied as: output = input × (1 - relative_losses)
        absolute_losses: Fixed losses that occur when transmission is active.
            Automatically creates binary variables for active/inactive states.
        status_parameters: Parameters defining binary operation constraints and costs.
        prevent_simultaneous_flows_in_both_directions: If True, prevents simultaneous
            flow in both directions. Increases binary variables but reflects physical
            reality for most transmission systems. Default is True.
        balanced: Whether to equate the size of the in1 and in2 Flow. Needs InvestParameters in both Flows.
        meta_data: Used to store additional information. Not used internally but saved
            in results. Only use Python native types.

    Examples:
        Simple electrical transmission line:

        ```python
        power_line = Transmission(
            id='110kv_line',
            in1=substation_a_out,
            out1=substation_b_in,
            relative_losses=0.03,  # 3% line losses
        )
        ```

        Bidirectional natural gas pipeline:

        ```python
        gas_pipeline = Transmission(
            id='interstate_pipeline',
            in1=compressor_station_a,
            out1=distribution_hub_b,
            in2=compressor_station_b,
            out2=distribution_hub_a,
            relative_losses=0.005,  # 0.5% friction losses
            absolute_losses=50,  # 50 kW compressor power when active
            prevent_simultaneous_flows_in_both_directions=True,
        )
        ```

        District heating network with investment optimization:

        ```python
        heating_network = Transmission(
            id='dh_main_line',
            in1=Flow(
                label='heat_supply',
                bus=central_plant_bus,
                size=InvestParameters(
                    minimum_size=1000,  # Minimum 1 MW capacity
                    maximum_size=10000,  # Maximum 10 MW capacity
                    specific_effects={'cost': 200},  # €200/kW capacity
                    fix_effects={'cost': 500000},  # €500k fixed installation
                ),
            ),
            out1=district_heat_demand,
            relative_losses=0.15,  # 15% thermal losses in distribution
        )
        ```

        Material conveyor with active/inactive status:

        ```python
        conveyor_belt = Transmission(
            id='material_transport',
            in1=loading_station,
            out1=unloading_station,
            absolute_losses=25,  # 25 kW motor power when running
            status_parameters=StatusParameters(
                effects_per_startup={'maintenance': 0.1},
                min_uptime=2,  # Minimum 2-hour operation
                startup_limit=10,  # Maximum 10 starts per period
            ),
        )
        ```

    Note:
        The transmission equation balances flows with losses:
        output_flow = input_flow × (1 - relative_losses) - absolute_losses

        For bidirectional transmission, each direction has independent loss calculations.

        When using InvestParameters on in1, the capacity automatically applies to in2
        to maintain consistent bidirectional capacity without additional investment variables.

        Absolute losses force the creation of binary on/inactive variables, which increases
        computational complexity but enables realistic modeling of equipment with
        standby power consumption.

    """

    def __init__(
        self,
        id: str | None = None,
        in1: Flow | None = None,
        out1: Flow | None = None,
        in2: Flow | None = None,
        out2: Flow | None = None,
        relative_losses: Numeric_TPS | None = None,
        absolute_losses: Numeric_TPS | None = None,
        status_parameters: StatusParameters | None = None,
        prevent_simultaneous_flows_in_both_directions: bool = True,
        balanced: bool = False,
        meta_data: dict | None = None,
        color: str | None = None,
        **kwargs,
    ):
        super().__init__(
            id,
            inputs=[flow for flow in (in1, in2) if flow is not None],
            outputs=[flow for flow in (out1, out2) if flow is not None],
            status_parameters=status_parameters,
            prevent_simultaneous_flows=None
            if in2 is None or prevent_simultaneous_flows_in_both_directions is False
            else [in1, in2],
            meta_data=meta_data,
            color=color,
            **kwargs,
        )
        self.in1 = in1
        self.out1 = out1
        self.in2 = in2
        self.out2 = out2

        self.relative_losses = relative_losses
        self.absolute_losses = absolute_losses
        self.balanced = balanced

    def _propagate_status_parameters(self) -> None:
        super()._propagate_status_parameters()
        # Transmissions with absolute_losses need status variables on input flows
        # Also need relative_minimum > 0 to link status to flow rate properly
        if self.absolute_losses is not None and np.any(self.absolute_losses != 0):
            from .config import CONFIG
            from .interface import StatusParameters

            input_flows = [self.in1]
            if self.in2 is not None:
                input_flows.append(self.in2)
            for flow in input_flows:
                if flow.status_parameters is None:
                    flow.status_parameters = StatusParameters()
                rel_min = flow.relative_minimum
                needs_update = (
                    rel_min is None
                    or (np.isscalar(rel_min) and rel_min <= 0)
                    or (isinstance(rel_min, np.ndarray) and np.all(rel_min <= 0))
                )
                if needs_update:
                    flow.relative_minimum = CONFIG.Modeling.epsilon


class StoragesModel(TypeModel):
    """Type-level model for ALL basic (non-intercluster) storages in a FlowSystem.

    Unlike StorageModel (one per Storage instance), StoragesModel handles ALL
    basic storages in a single instance with batched variables.

    Note:
        Intercluster storages are handled separately by InterclusterStoragesModel.

    This enables:
    - Batched charge_state and netto_discharge variables with element dimension
    - Batched investment variables via InvestmentsModel
    - Consistent architecture with FlowsModel and BusesModel

    Example:
        >>> storages_model = StoragesModel(model, basic_storages, flows_model)
        >>> storages_model.create_variables()
        >>> storages_model.create_constraints()
        >>> storages_model.create_investment_model()  # After storage variables exist
        >>> storages_model.create_investment_constraints()
    """

    def __init__(
        self,
        model: FlowSystemModel,
        data: StoragesData,
        flows_model,  # FlowsModel - avoid circular import
    ):
        """Initialize the type-level model for basic storages.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            data: StoragesData container for basic storages.
            flows_model: The FlowsModel containing flow_rate variables.
        """
        super().__init__(model, data)
        self._flows_model = flows_model

        # Set reference on each storage element
        for storage in self.elements.values():
            storage._storages_model = self

        self.create_variables()
        self.create_constraints()
        self.create_investment_model()
        self.create_investment_constraints()
        self._create_prevent_simultaneous_constraints()

    def _create_prevent_simultaneous_constraints(self) -> None:
        from .elements import _add_prevent_simultaneous_constraints

        _add_prevent_simultaneous_constraints(
            list(self.elements.values()), self._flows_model, self.model, 'storage|prevent_simultaneous'
        )

    def storage(self, label: str) -> Storage:
        """Get a storage by its id."""
        return self.elements[label]

    @property
    def investment_ids(self) -> list[str]:
        """IDs of storages with investment parameters. Used by external code (optimization.py)."""
        return self.data.with_investment

    def add_effect_contributions(self, effects_model) -> None:
        """Push ALL effect contributions from storages to EffectsModel.

        Called by EffectsModel.finalize_shares(). Pushes:
        - Periodic share: size × effects_per_size
        - Investment/retirement: invested × factor
        - Constants: mandatory fixed + retirement constants

        Args:
            effects_model: The EffectsModel to register contributions with.
        """
        inv = self.data.investment_data
        if inv is None:
            return

        dim = self.dim_name

        # === Periodic: size * effects_per_size ===
        # Batched over storages and effects - _accumulate_shares handles effect dim internally
        if inv.effects_per_size is not None:
            factors = inv.effects_per_size
            storage_ids = factors.coords[dim].values
            size_subset = self.size.sel({dim: storage_ids})
            effects_model.add_periodic_contribution(size_subset * factors, contributor_dim=dim)

        # === Investment/retirement effects (optional investments) ===
        invested = self.invested
        if invested is not None:
            if (ff := inv.effects_of_investment) is not None:
                storage_ids = ff.coords[dim].values
                invested_subset = invested.sel({dim: storage_ids})
                effects_model.add_periodic_contribution(invested_subset * ff, contributor_dim=dim)

            if (ff := inv.effects_of_retirement) is not None:
                storage_ids = ff.coords[dim].values
                invested_subset = invested.sel({dim: storage_ids})
                effects_model.add_periodic_contribution(invested_subset * (-ff), contributor_dim=dim)

        # === Constants: mandatory fixed + retirement ===
        if inv.effects_of_investment_mandatory is not None:
            effects_model.add_periodic_contribution(inv.effects_of_investment_mandatory, contributor_dim=dim)
        if inv.effects_of_retirement_constant is not None:
            effects_model.add_periodic_contribution(inv.effects_of_retirement_constant, contributor_dim=dim)

    # --- Investment Cached Properties ---

    @functools.cached_property
    def _size_lower(self) -> xr.DataArray:
        """(storage,) - minimum size for investment storages."""
        element_ids = self.data.with_investment
        values = [self.storage(sid).capacity_in_flow_hours.minimum_or_fixed_size for sid in element_ids]
        return stack_along_dim(values, self.dim_name, element_ids)

    @functools.cached_property
    def _size_upper(self) -> xr.DataArray:
        """(storage,) - maximum size for investment storages."""
        element_ids = self.data.with_investment
        values = [self.storage(sid).capacity_in_flow_hours.maximum_or_fixed_size for sid in element_ids]
        return stack_along_dim(values, self.dim_name, element_ids)

    @functools.cached_property
    def _linked_periods_mask(self) -> xr.DataArray | None:
        """(storage, period) - linked periods for investment storages. None if no linking."""
        element_ids = self.data.with_investment
        linked_list = [self.storage(sid).capacity_in_flow_hours.linked_periods for sid in element_ids]
        if not any(lp is not None for lp in linked_list):
            return None

        values = [lp if lp is not None else np.nan for lp in linked_list]
        return stack_along_dim(values, self.dim_name, element_ids)

    @functools.cached_property
    def _mandatory_mask(self) -> xr.DataArray:
        """(storage,) bool - True if mandatory, False if optional."""
        element_ids = self.data.with_investment
        values = [self.storage(sid).capacity_in_flow_hours.mandatory for sid in element_ids]
        return xr.DataArray(values, dims=[self.dim_name], coords={self.dim_name: element_ids})

    @functools.cached_property
    def _optional_lower(self) -> xr.DataArray | None:
        """(storage,) - minimum size for optional investment storages."""
        if not self.data.with_optional_investment:
            return None

        element_ids = self.data.with_optional_investment
        values = [self.storage(sid).capacity_in_flow_hours.minimum_or_fixed_size for sid in element_ids]
        return stack_along_dim(values, self.dim_name, element_ids)

    @functools.cached_property
    def _optional_upper(self) -> xr.DataArray | None:
        """(storage,) - maximum size for optional investment storages."""
        if not self.data.with_optional_investment:
            return None

        element_ids = self.data.with_optional_investment
        values = [self.storage(sid).capacity_in_flow_hours.maximum_or_fixed_size for sid in element_ids]
        return stack_along_dim(values, self.dim_name, element_ids)

    @functools.cached_property
    def _flow_mask(self) -> xr.DataArray:
        """(storage, flow) mask: 1 if flow belongs to storage."""
        membership = MaskHelpers.build_flow_membership(
            self.elements,
            lambda s: list(s.flows.values()),
        )
        return MaskHelpers.build_mask(
            row_dim='storage',
            row_ids=self.element_ids,
            col_dim='flow',
            col_ids=self._flows_model.element_ids,
            membership=membership,
        )

    @functools.cached_property
    def charge(self) -> linopy.Variable:
        """(storage, time+1, ...) - charge state variable for ALL storages."""
        return self.add_variables(
            StorageVarName.CHARGE,
            lower=self.data.charge_state_lower_bounds,
            upper=self.data.charge_state_upper_bounds,
            dims=None,
            extra_timestep=True,
        )

    @functools.cached_property
    def netto(self) -> linopy.Variable:
        """(storage, time, ...) - netto discharge variable for ALL storages."""
        return self.add_variables(
            StorageVarName.NETTO,
            dims=None,
        )

    def create_variables(self) -> None:
        """Create all batched variables for storages.

        Triggers cached property creation for:
        - storage|charge: For ALL storages (with extra timestep)
        - storage|netto: For ALL storages
        """
        if not self.elements:
            return

        _ = self.charge
        _ = self.netto

        logger.debug(
            f'StoragesModel created variables: {len(self.elements)} storages, '
            f'{len(self.data.with_investment)} with investment'
        )

    def create_constraints(self) -> None:
        """Create batched constraints for all storages.

        Uses vectorized operations for efficiency:
        - netto_discharge constraint (batched)
        - energy balance constraint (batched)
        - initial/final constraints (batched by type)
        """
        if not self.elements:
            return

        flow_rate = self._flows_model[FlowVarName.RATE]
        charge_state = self.charge
        netto_discharge = self.netto
        timestep_duration = self.model.timestep_duration

        # === Batched netto_discharge constraint ===
        # Build charge and discharge flow_rate selections aligned with storage dimension
        charge_flow_ids = self.data.charging_flow_ids
        discharge_flow_ids = self.data.discharging_flow_ids

        # Detect flow dimension name from flow_rate variable
        flow_dim = 'flow' if 'flow' in flow_rate.dims else 'element'
        dim = self.dim_name

        # Select from flow dimension and rename to storage dimension
        charge_rates = flow_rate.sel({flow_dim: charge_flow_ids})
        charge_rates = charge_rates.rename({flow_dim: dim}).assign_coords({dim: self.element_ids})
        discharge_rates = flow_rate.sel({flow_dim: discharge_flow_ids})
        discharge_rates = discharge_rates.rename({flow_dim: dim}).assign_coords({dim: self.element_ids})

        self.model.add_constraints(
            netto_discharge == discharge_rates - charge_rates,
            name='storage|netto_eq',
        )

        # === Batched energy balance constraint ===
        eta_charge = self.data.eta_charge
        eta_discharge = self.data.eta_discharge
        rel_loss = self.data.relative_loss_per_hour

        # Energy balance: cs[t+1] = cs[t] * (1-loss)^dt + charge * eta_c * dt - discharge * dt / eta_d
        # Rearranged: cs[t+1] - cs[t] * (1-loss)^dt - charge * eta_c * dt + discharge * dt / eta_d = 0
        # Pre-combine pure xarray coefficients to minimize linopy operations
        loss_factor = (1 - rel_loss) ** timestep_duration
        charge_factor = eta_charge * timestep_duration
        discharge_factor = timestep_duration / eta_discharge
        energy_balance_lhs = (
            charge_state.isel(time=slice(1, None))
            - charge_state.isel(time=slice(None, -1)) * loss_factor
            - charge_rates * charge_factor
            + discharge_rates * discharge_factor
        )
        self.model.add_constraints(
            energy_balance_lhs == 0,
            name='storage|balance',
        )

        # === Initial/final constraints (grouped by type) ===
        self._add_batched_initial_final_constraints(charge_state)

        # === Cluster cyclic constraints ===
        self._add_batched_cluster_cyclic_constraints(charge_state)

        # === Balanced flow sizes constraint ===
        self._add_balanced_flow_sizes_constraint()

        logger.debug(f'StoragesModel created batched constraints for {len(self.elements)} storages')

    def _add_balanced_flow_sizes_constraint(self) -> None:
        """Add constraint ensuring charging and discharging flow capacities are equal for balanced storages."""
        balanced_ids = self.data.with_balanced
        if not balanced_ids:
            return

        flows_model = self._flows_model
        size_var = flows_model.get_variable(FlowVarName.SIZE)
        if size_var is None:
            return

        flow_dim = flows_model.dim_name
        investment_ids_set = set(flows_model.investment_ids)

        # Filter to balanced storages where both flows have investment
        charge_ids = []
        discharge_ids = []
        for sid in balanced_ids:
            s = self.data[sid]
            cid = s.charging.id
            did = s.discharging.id
            if cid in investment_ids_set and did in investment_ids_set:
                charge_ids.append(cid)
                discharge_ids.append(did)

        if not charge_ids:
            return

        charge_sizes = size_var.sel({flow_dim: charge_ids})
        discharge_sizes = size_var.sel({flow_dim: discharge_ids})
        # Rename to a shared dim so the constraint is element-wise
        balanced_dim = 'balanced_storage'
        charge_sizes = charge_sizes.rename({flow_dim: balanced_dim}).assign_coords({balanced_dim: charge_ids})
        discharge_sizes = discharge_sizes.rename({flow_dim: balanced_dim}).assign_coords({balanced_dim: charge_ids})
        self.model.add_constraints(
            charge_sizes - discharge_sizes == 0,
            name='storage|balanced_sizes',
        )

    def _add_batched_initial_final_constraints(self, charge_state) -> None:
        """Add batched initial and final charge state constraints."""
        # Group storages by constraint type
        storages_numeric_initial: list[tuple[Storage, float]] = []
        storages_equals_final: list[Storage] = []
        storages_max_final: list[tuple[Storage, float]] = []
        storages_min_final: list[tuple[Storage, float]] = []

        for storage in self.elements.values():
            # Skip for clustered independent/cyclic modes
            if self.model.flow_system.clusters is not None and storage.cluster_mode in ('independent', 'cyclic'):
                continue

            if storage.initial_charge_state is not None:
                if isinstance(storage.initial_charge_state, str):  # 'equals_final'
                    storages_equals_final.append(storage)
                else:
                    storages_numeric_initial.append((storage, self.data.aligned_initial_charge_state(storage)))

            aligned_max_final = self.data.aligned_maximal_final_charge_state(storage)
            if aligned_max_final is not None:
                storages_max_final.append((storage, aligned_max_final))

            aligned_min_final = self.data.aligned_minimal_final_charge_state(storage)
            if aligned_min_final is not None:
                storages_min_final.append((storage, aligned_min_final))

        dim = self.dim_name

        # Batched numeric initial constraint
        if storages_numeric_initial:
            ids = [s.id for s, _ in storages_numeric_initial]
            values = stack_along_dim([v for _, v in storages_numeric_initial], self.dim_name, ids)
            cs_initial = charge_state.sel({dim: ids}).isel(time=0)
            self.model.add_constraints(
                cs_initial == values,
                name='storage|initial_charge_state',
            )

        # Batched equals_final constraint
        if storages_equals_final:
            ids = [s.id for s in storages_equals_final]
            cs_subset = charge_state.sel({dim: ids})
            self.model.add_constraints(
                cs_subset.isel(time=0) == cs_subset.isel(time=-1),
                name='storage|initial_equals_final',
            )

        # Batched max final constraint
        if storages_max_final:
            ids = [s.id for s, _ in storages_max_final]
            values = stack_along_dim([v for _, v in storages_max_final], self.dim_name, ids)
            cs_final = charge_state.sel({dim: ids}).isel(time=-1)
            self.model.add_constraints(
                cs_final <= values,
                name='storage|final_charge_max',
            )

        # Batched min final constraint
        if storages_min_final:
            ids = [s.id for s, _ in storages_min_final]
            values = stack_along_dim([v for _, v in storages_min_final], self.dim_name, ids)
            cs_final = charge_state.sel({dim: ids}).isel(time=-1)
            self.model.add_constraints(
                cs_final >= values,
                name='storage|final_charge_min',
            )

    def _add_batched_cluster_cyclic_constraints(self, charge_state) -> None:
        """Add batched cluster cyclic constraints for storages with cyclic mode."""
        if self.model.flow_system.clusters is None:
            return

        cyclic_storages = [s for s in self.elements.values() if s.cluster_mode == 'cyclic']
        if not cyclic_storages:
            return

        ids = [s.id for s in cyclic_storages]
        cs_subset = charge_state.sel({self.dim_name: ids})
        self.model.add_constraints(
            cs_subset.isel(time=0) == cs_subset.isel(time=-2),
            name='storage|cluster_cyclic',
        )

    @functools.cached_property
    def size(self) -> linopy.Variable | None:
        """(storage, period, scenario) - size variable for storages with investment."""
        if not self.data.with_investment:
            return None

        size_min = self._size_lower
        size_max = self._size_upper

        # Handle linked_periods masking
        linked_periods = self._linked_periods_mask
        if linked_periods is not None:
            linked = linked_periods.fillna(1.0)
            size_min = size_min * linked
            size_max = size_max * linked

        # For non-mandatory, lower bound is 0 (invested variable controls actual minimum)
        lower_bounds = xr.where(self._mandatory_mask, size_min, 0)

        return self.add_variables(
            StorageVarName.SIZE,
            lower=lower_bounds,
            upper=size_max,
            dims=('period', 'scenario'),
            element_ids=self.investment_ids,
        )

    @functools.cached_property
    def invested(self) -> linopy.Variable | None:
        """(storage, period, scenario) - binary invested variable for optional investment."""
        if not self.data.with_optional_investment:
            return None
        return self.add_variables(
            StorageVarName.INVESTED,
            dims=('period', 'scenario'),
            element_ids=self.data.with_optional_investment,
            binary=True,
        )

    def create_investment_model(self) -> None:
        """Create investment variables and constraints for storages with investment.

        Must be called BEFORE create_investment_constraints().
        """
        if not self.data.with_investment:
            return

        from .features import InvestmentBuilder

        dim = self.dim_name
        element_ids = self.investment_ids
        non_mandatory_ids = self.data.with_optional_investment
        mandatory_ids = self.data.with_mandatory_investment

        # Trigger variable creation via cached properties
        size_var = self.size
        invested_var = self.invested

        if invested_var is not None:
            # State-controlled bounds constraints using cached properties
            InvestmentBuilder.add_optional_size_bounds(
                model=self.model,
                size_var=size_var,
                invested_var=invested_var,
                min_bounds=self._optional_lower,
                max_bounds=self._optional_upper,
                element_ids=non_mandatory_ids,
                dim_name=dim,
                name_prefix='storage',
            )

        # Linked periods constraints
        InvestmentBuilder.add_linked_periods_constraints(
            model=self.model,
            size_var=size_var,
            params=self.data.invest_params,
            element_ids=element_ids,
            dim_name=dim,
        )

        # Piecewise effects (handled per-element, not batchable)
        self._create_piecewise_effects()

        logger.debug(
            f'StoragesModel created investment variables: {len(element_ids)} storages '
            f'({len(mandatory_ids)} mandatory, {len(non_mandatory_ids)} optional)'
        )

    def create_investment_constraints(self) -> None:
        """Create batched scaled bounds linking charge_state to investment size.

        Must be called AFTER create_investment_model().

        Mathematical formulation:
            charge_state >= size * relative_minimum_charge_state
            charge_state <= size * relative_maximum_charge_state

        Uses the batched size variable for true vectorized constraint creation.
        """
        if not self.data.with_investment or StorageVarName.SIZE not in self:
            return

        charge_state = self.charge
        size_var = self.size  # Batched size with storage dimension

        dim = self.dim_name
        rel_lower_stacked = self.data.relative_minimum_charge_state_extra.sel({dim: self.investment_ids})
        rel_upper_stacked = self.data.relative_maximum_charge_state_extra.sel({dim: self.investment_ids})

        # Select charge_state for investment storages only
        cs_investment = charge_state.sel({dim: self.investment_ids})

        # Select size for these storages (it already has storage dimension)
        size_investment = size_var.sel({dim: self.investment_ids})

        # Check if all bounds are equal (fixed relative bounds)
        from .modeling import _xr_allclose

        if _xr_allclose(rel_lower_stacked, rel_upper_stacked):
            # Fixed bounds: charge_state == size * relative_bound
            self.model.add_constraints(
                cs_investment == size_investment * rel_lower_stacked,
                name='storage|charge|investment|fixed',
            )
        else:
            # Variable bounds: lower <= charge_state <= upper
            self.model.add_constraints(
                cs_investment >= size_investment * rel_lower_stacked,
                name='storage|charge|investment|lb',
            )
            self.model.add_constraints(
                cs_investment <= size_investment * rel_upper_stacked,
                name='storage|charge|investment|ub',
            )

        logger.debug(
            f'StoragesModel created batched investment constraints for {len(self.data.with_investment)} storages'
        )

    def _add_initial_final_constraints_legacy(self, storage, cs) -> None:
        """Legacy per-element initial/final constraints (kept for reference)."""
        skip_initial_final = self.model.flow_system.clusters is not None and storage.cluster_mode in (
            'independent',
            'cyclic',
        )

        if not skip_initial_final:
            if storage.initial_charge_state is not None:
                if isinstance(storage.initial_charge_state, str):  # 'equals_final'
                    self.model.add_constraints(
                        cs.isel(time=0) == cs.isel(time=-1),
                        name=f'storage|{storage.id}|initial_charge_state',
                    )
                else:
                    aligned_initial = self.data.aligned_initial_charge_state(storage)
                    self.model.add_constraints(
                        cs.isel(time=0) == aligned_initial,
                        name=f'storage|{storage.id}|initial_charge_state',
                    )

                aligned_min_final = self.data.aligned_minimal_final_charge_state(storage)
                if aligned_min_final is not None:
                    self.model.add_constraints(
                        cs.isel(time=-1) >= aligned_min_final,
                        name=f'storage|{storage.id}|final_charge_min',
                    )

        logger.debug(f'StoragesModel created constraints for {len(self.elements)} storages')

    # === Variable accessor properties ===

    def get_variable(self, name: str, element_id: str | None = None):
        """Get a variable, optionally selecting a specific element."""
        var = self._variables.get(name)
        if var is None:
            return None
        if element_id is not None:
            return var.sel({self.dim_name: element_id})
        return var

    # Investment effect properties are defined above, delegating to _investment_data

    def _create_piecewise_effects(self) -> None:
        """Create batched piecewise effects for storages with piecewise_effects_of_investment.

        Uses PiecewiseBuilder for pad-to-max batching across all storages with
        piecewise effects. Creates batched segment variables, share variables,
        and coupling constraints.
        """
        from .features import PiecewiseBuilder

        dim = self.dim_name
        size_var = self.size
        invested_var = self.invested

        if size_var is None:
            return

        inv = self.data.investment_data
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
            optional_ids = [sid for sid in element_ids if sid in invested_ids]
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
        import pandas as pd

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

        logger.debug(f'Created batched piecewise effects for {len(element_ids)} storages')


class InterclusterStoragesModel(TypeModel):
    """Type-level batched model for ALL intercluster storages.

    Replaces per-element InterclusterStorageModel with a single batched implementation.
    Handles SOC_boundary linking, energy balance, and investment for all intercluster
    storages together using vectorized operations.

    This is only created when:
    - The FlowSystem has been clustered
    - There are storages with cluster_mode='intercluster' or 'intercluster_cyclic'
    """

    def __init__(
        self,
        model: FlowSystemModel,
        data: StoragesData,
        flows_model,  # FlowsModel - avoid circular import
    ):
        """Initialize the batched model for intercluster storages.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            data: StoragesData container for intercluster storages.
            flows_model: The FlowsModel containing flow_rate variables.
        """
        from .features import InvestmentBuilder

        super().__init__(model, data)
        self._flows_model = flows_model
        self._InvestmentBuilder = InvestmentBuilder

        # Clustering info (required for intercluster)
        self._clustering = model.flow_system.clustering
        if not self.elements:
            return  # Nothing to model

        if self._clustering is None:
            raise ValueError('InterclusterStoragesModel requires a clustered FlowSystem')

        self.create_variables()
        self.create_constraints()
        self.create_investment_model()
        self.create_investment_constraints()
        self.create_effect_shares()

    def get_variable(self, name: str, element_id: str | None = None) -> linopy.Variable:
        """Get a variable, optionally selecting a specific element."""
        var = self._variables.get(name)
        if var is None:
            return None
        if element_id is not None and self.dim_name in var.dims:
            return var.sel({self.dim_name: element_id})
        return var

    # =========================================================================
    # Variable Creation
    # =========================================================================

    @functools.cached_property
    def charge_state(self) -> linopy.Variable:
        """(intercluster_storage, time+1, ...) - relative SOC change."""
        return self.add_variables(
            InterclusterStorageVarName.CHARGE_STATE,
            lower=-self.data.capacity_upper,
            upper=self.data.capacity_upper,
            dims=None,
            extra_timestep=True,
        )

    @functools.cached_property
    def netto_discharge(self) -> linopy.Variable:
        """(intercluster_storage, time, ...) - net discharge rate."""
        return self.add_variables(
            InterclusterStorageVarName.NETTO_DISCHARGE,
            dims=None,
        )

    def create_variables(self) -> None:
        """Create batched variables for all intercluster storages."""
        if not self.elements:
            return

        _ = self.charge_state
        _ = self.netto_discharge
        _ = self.soc_boundary

    @functools.cached_property
    def soc_boundary(self) -> linopy.Variable:
        """(cluster_boundary, intercluster_storage, ...) - absolute SOC at period boundaries."""
        import pandas as pd

        from .clustering.intercluster_helpers import build_boundary_coords, extract_capacity_bounds

        dim = self.dim_name
        n_original_clusters = self._clustering.n_original_clusters
        flow_system = self.model.flow_system

        # Build coords for boundary dimension (returns dict, not xr.Coordinates)
        boundary_coords_dict, boundary_dims = build_boundary_coords(n_original_clusters, flow_system)

        # Build per-storage bounds using original boundary dims (without storage dim)
        per_storage_coords = dict(boundary_coords_dict)
        per_storage_dims = list(boundary_dims)

        # Add storage dimension with pd.Index for proper indexing
        boundary_coords_dict[dim] = pd.Index(self.element_ids, name=dim)
        boundary_dims = list(boundary_dims) + [dim]

        # Convert to xr.Coordinates for variable creation
        boundary_coords = xr.Coordinates(boundary_coords_dict)

        # Compute bounds per storage
        lowers = []
        uppers = []
        for storage in self.elements.values():
            cap_bounds = extract_capacity_bounds(storage.capacity_in_flow_hours, per_storage_coords, per_storage_dims)
            lowers.append(cap_bounds.lower)
            uppers.append(cap_bounds.upper)

        # Stack bounds
        lower = stack_along_dim(lowers, dim, self.element_ids)
        upper = stack_along_dim(uppers, dim, self.element_ids)

        soc_boundary = self.model.add_variables(
            lower=lower,
            upper=upper,
            coords=boundary_coords,
            name=f'{self.dim_name}|SOC_boundary',
        )
        self._variables[InterclusterStorageVarName.SOC_BOUNDARY] = soc_boundary
        return soc_boundary

    # =========================================================================
    # Constraint Creation
    # =========================================================================

    def create_constraints(self) -> None:
        """Create batched constraints for all intercluster storages."""
        if not self.elements:
            return

        self._add_netto_discharge_constraints()
        self._add_energy_balance_constraints()
        self._add_cluster_start_constraints()
        self._add_linking_constraints()
        self._add_cyclic_or_initial_constraints()
        self._add_combined_bound_constraints()

    def _add_netto_discharge_constraints(self) -> None:
        """Add constraint: netto_discharge = discharging - charging for all storages."""
        netto = self.netto_discharge
        dim = self.dim_name

        # Get batched flow_rate variable and select charge/discharge flows
        flow_rate = self._flows_model[FlowVarName.RATE]
        flow_dim = 'flow' if 'flow' in flow_rate.dims else 'element'

        charge_flow_ids = self.data.charging_flow_ids
        discharge_flow_ids = self.data.discharging_flow_ids

        # Select and rename to match storage dimension
        charge_rates = flow_rate.sel({flow_dim: charge_flow_ids})
        charge_rates = charge_rates.rename({flow_dim: dim}).assign_coords({dim: self.element_ids})
        discharge_rates = flow_rate.sel({flow_dim: discharge_flow_ids})
        discharge_rates = discharge_rates.rename({flow_dim: dim}).assign_coords({dim: self.element_ids})

        self.model.add_constraints(
            netto == discharge_rates - charge_rates,
            name=f'{self.dim_name}|netto_discharge',
        )

    def _add_energy_balance_constraints(self) -> None:
        """Add energy balance constraints for all storages."""
        charge_state = self.charge_state
        timestep_duration = self.model.timestep_duration
        dim = self.dim_name

        # Select and rename flow rates to storage dimension
        flow_rate = self._flows_model[FlowVarName.RATE]
        flow_dim = 'flow' if 'flow' in flow_rate.dims else 'element'

        charge_rates = flow_rate.sel({flow_dim: self.data.charging_flow_ids})
        charge_rates = charge_rates.rename({flow_dim: dim}).assign_coords({dim: self.element_ids})
        discharge_rates = flow_rate.sel({flow_dim: self.data.discharging_flow_ids})
        discharge_rates = discharge_rates.rename({flow_dim: dim}).assign_coords({dim: self.element_ids})

        rel_loss = self.data.relative_loss_per_hour
        eta_charge = self.data.eta_charge
        eta_discharge = self.data.eta_discharge

        # Pre-combine pure xarray coefficients to minimize linopy operations
        loss_factor = (1 - rel_loss) ** timestep_duration
        charge_factor = eta_charge * timestep_duration
        discharge_factor = timestep_duration / eta_discharge
        lhs = (
            charge_state.isel(time=slice(1, None))
            - charge_state.isel(time=slice(None, -1)) * loss_factor
            - charge_rates * charge_factor
            + discharge_rates * discharge_factor
        )
        self.model.add_constraints(lhs == 0, name=f'{self.dim_name}|energy_balance')

    def _add_cluster_start_constraints(self) -> None:
        """Constrain ΔE = 0 at the start of each cluster for all storages."""
        charge_state = self.charge_state
        self.model.add_constraints(
            charge_state.isel(time=0) == 0,
            name=f'{self.dim_name}|cluster_start',
        )

    def _add_linking_constraints(self) -> None:
        """Add constraints linking consecutive SOC_boundary values."""
        soc_boundary = self.soc_boundary
        charge_state = self.charge_state
        n_original_clusters = self._clustering.n_original_clusters
        cluster_assignments = self._clustering.cluster_assignments

        # delta_SOC = charge_state at end of cluster (start is 0 by constraint)
        delta_soc = charge_state.isel(time=-1) - charge_state.isel(time=0)

        # Link each original period
        soc_after = soc_boundary.isel(cluster_boundary=slice(1, None))
        soc_before = soc_boundary.isel(cluster_boundary=slice(None, -1))

        # Rename for alignment
        soc_after = soc_after.rename({'cluster_boundary': 'original_cluster'})
        soc_after = soc_after.assign_coords(original_cluster=np.arange(n_original_clusters))
        soc_before = soc_before.rename({'cluster_boundary': 'original_cluster'})
        soc_before = soc_before.assign_coords(original_cluster=np.arange(n_original_clusters))

        # Get delta_soc for each original period using cluster_assignments
        delta_soc_ordered = delta_soc.isel(cluster=cluster_assignments)

        # Decay factor: (1 - mean_loss)^total_hours, stacked across storages
        rel_loss = _scalar_safe_reduce(self.data.relative_loss_per_hour, 'time', 'mean')
        total_hours = _scalar_safe_reduce(self.model.timestep_duration, 'time', 'sum')
        decay_stacked = (1 - rel_loss) ** total_hours

        lhs = soc_after - soc_before * decay_stacked - delta_soc_ordered
        self.model.add_constraints(lhs == 0, name=f'{self.dim_name}|link')

    def _add_cyclic_or_initial_constraints(self) -> None:
        """Add cyclic or initial SOC_boundary constraints per storage."""
        soc_boundary = self.soc_boundary
        n_original_clusters = self._clustering.n_original_clusters

        # Group by constraint type
        cyclic_ids = []
        initial_fixed_ids = []
        initial_values = []

        for storage in self.elements.values():
            if storage.cluster_mode == 'intercluster_cyclic':
                cyclic_ids.append(storage.id)
            else:
                initial = storage.initial_charge_state
                if initial is not None:
                    if isinstance(initial, str) and initial == 'equals_final':
                        cyclic_ids.append(storage.id)
                    else:
                        initial_fixed_ids.append(storage.id)
                        initial_values.append(self.data.aligned_initial_charge_state(storage))

        # Add cyclic constraints
        if cyclic_ids:
            soc_cyclic = soc_boundary.sel({self.dim_name: cyclic_ids})
            self.model.add_constraints(
                soc_cyclic.isel(cluster_boundary=0) == soc_cyclic.isel(cluster_boundary=n_original_clusters),
                name=f'{self.dim_name}|cyclic',
            )

        # Add fixed initial constraints
        if initial_fixed_ids:
            soc_initial = soc_boundary.sel({self.dim_name: initial_fixed_ids})
            initial_stacked = stack_along_dim(initial_values, self.dim_name, initial_fixed_ids)
            self.model.add_constraints(
                soc_initial.isel(cluster_boundary=0) == initial_stacked,
                name=f'{self.dim_name}|initial_SOC_boundary',
            )

    def _add_combined_bound_constraints(self) -> None:
        """Add constraints ensuring actual SOC stays within bounds at sample points."""
        charge_state = self.charge_state
        soc_boundary = self.soc_boundary
        n_original_clusters = self._clustering.n_original_clusters
        cluster_assignments = self._clustering.cluster_assignments

        # soc_d: SOC at start of each original period
        soc_d = soc_boundary.isel(cluster_boundary=slice(None, -1))
        soc_d = soc_d.rename({'cluster_boundary': 'original_cluster'})
        soc_d = soc_d.assign_coords(original_cluster=np.arange(n_original_clusters))

        actual_time_size = charge_state.sizes['time']
        sample_offsets = [0, actual_time_size // 2, actual_time_size - 1]

        for sample_name, offset in zip(['start', 'mid', 'end'], sample_offsets, strict=False):
            # Get charge_state at offset, reorder by cluster_assignments
            cs_at_offset = charge_state.isel(time=offset)
            cs_t = cs_at_offset.isel(cluster=cluster_assignments)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*does not create an index anymore.*')
                cs_t = cs_t.rename({'cluster': 'original_cluster'})
            cs_t = cs_t.assign_coords(original_cluster=np.arange(n_original_clusters))

            # Decay factor at offset: (1 - mean_loss)^(offset * mean_dt)
            rel_loss = _scalar_safe_reduce(self.data.relative_loss_per_hour, 'time', 'mean')
            mean_dt = _scalar_safe_reduce(self.model.timestep_duration, 'time', 'mean')
            decay_stacked = (1 - rel_loss) ** (offset * mean_dt)

            combined = soc_d * decay_stacked + cs_t

            # Lower bound: combined >= 0
            self.model.add_constraints(combined >= 0, name=f'{self.dim_name}|soc_lb_{sample_name}')

            # Upper bound depends on investment
            self._add_upper_bound_constraint(combined, sample_name)

    def _add_upper_bound_constraint(self, combined: xr.DataArray, sample_name: str) -> None:
        """Add upper bound constraint for combined SOC."""
        # Group storages by upper bound type
        invest_ids = []
        fixed_ids = []
        fixed_caps = []

        for storage in self.elements.values():
            if isinstance(storage.capacity_in_flow_hours, InvestParameters):
                invest_ids.append(storage.id)
            elif storage.capacity_in_flow_hours is not None:
                fixed_ids.append(storage.id)
                fixed_caps.append(storage.capacity_in_flow_hours)

        # Investment storages: combined <= size
        if invest_ids:
            combined_invest = combined.sel({self.dim_name: invest_ids})
            size_var = self.size
            if size_var is not None:
                size_invest = size_var.sel({self.dim_name: invest_ids})
                self.model.add_constraints(
                    combined_invest <= size_invest,
                    name=f'{self.dim_name}|soc_ub_{sample_name}_invest',
                )

        # Fixed capacity storages: combined <= capacity
        if fixed_ids:
            combined_fixed = combined.sel({self.dim_name: fixed_ids})
            caps_stacked = stack_along_dim(fixed_caps, self.dim_name, fixed_ids)
            self.model.add_constraints(
                combined_fixed <= caps_stacked,
                name=f'{self.dim_name}|soc_ub_{sample_name}_fixed',
            )

    # =========================================================================
    # Investment
    # =========================================================================

    @functools.cached_property
    def size(self) -> linopy.Variable | None:
        """(intercluster_storage, period, scenario) - size variable for storages with investment."""
        if not self.data.with_investment:
            return None
        inv = self.data.investment_data
        return self.add_variables(
            InterclusterStorageVarName.SIZE,
            lower=inv.size_minimum,
            upper=inv.size_maximum,
            dims=('period', 'scenario'),
            element_ids=self.data.with_investment,
        )

    @functools.cached_property
    def invested(self) -> linopy.Variable | None:
        """(intercluster_storage, period, scenario) - binary invested variable for optional investment."""
        if not self.data.with_optional_investment:
            return None
        return self.add_variables(
            InterclusterStorageVarName.INVESTED,
            dims=('period', 'scenario'),
            element_ids=self.data.with_optional_investment,
            binary=True,
        )

    def create_investment_model(self) -> None:
        """Create batched investment variables using InvestmentBuilder."""
        if not self.data.with_investment:
            return

        _ = self.size
        _ = self.invested

    def create_investment_constraints(self) -> None:
        """Create investment-related constraints."""
        if not self.data.with_investment:
            return

        investment_ids = self.data.with_investment
        optional_ids = self.data.with_optional_investment

        size_var = self.size
        invested_var = self.invested
        charge_state = self.charge_state
        soc_boundary = self.soc_boundary

        # Symmetric bounds on charge_state: -size <= charge_state <= size
        size_for_all = size_var.sel({self.dim_name: investment_ids})
        cs_for_invest = charge_state.sel({self.dim_name: investment_ids})

        self.model.add_constraints(
            cs_for_invest >= -size_for_all,
            name=f'{self.dim_name}|charge_state|lb',
        )
        self.model.add_constraints(
            cs_for_invest <= size_for_all,
            name=f'{self.dim_name}|charge_state|ub',
        )

        # SOC_boundary <= size
        soc_for_invest = soc_boundary.sel({self.dim_name: investment_ids})
        self.model.add_constraints(
            soc_for_invest <= size_for_all,
            name=f'{self.dim_name}|SOC_boundary_ub',
        )

        # Optional investment bounds using InvestmentBuilder
        inv = self.data.investment_data
        if optional_ids and invested_var is not None:
            optional_lower = inv.optional_size_minimum
            optional_upper = inv.optional_size_maximum
            size_optional = size_var.sel({self.dim_name: optional_ids})

            self._InvestmentBuilder.add_optional_size_bounds(
                self.model,
                size_optional,
                invested_var,
                optional_lower,
                optional_upper,
                optional_ids,
                self.dim_name,
                f'{self.dim_name}|size',
            )

    def create_effect_shares(self) -> None:
        """Add investment effects to the EffectsModel."""
        if not self.data.with_investment:
            return

        from .features import InvestmentBuilder

        investment_ids = self.data.with_investment
        optional_ids = self.data.with_optional_investment
        storages_with_investment = [self.data[sid] for sid in investment_ids]

        size_var = self.size
        invested_var = self.invested

        # Collect effects
        effects = InvestmentBuilder.collect_effects(
            storages_with_investment,
            lambda s: s.capacity_in_flow_hours,
        )

        # Add effect shares
        for effect_name, effect_type, factors in effects:
            factor_stacked = stack_along_dim(factors, self.dim_name, investment_ids)

            if effect_type == 'per_size':
                expr = (size_var * factor_stacked).sum(self.dim_name)
            elif effect_type == 'fixed':
                if invested_var is not None:
                    mandatory_ids = self.data.with_mandatory_investment

                    expr_parts = []
                    if mandatory_ids:
                        factor_mandatory = factor_stacked.sel({self.dim_name: mandatory_ids})
                        expr_parts.append(factor_mandatory.sum(self.dim_name))
                    if optional_ids:
                        factor_optional = factor_stacked.sel({self.dim_name: optional_ids})
                        invested_optional = invested_var.sel({self.dim_name: optional_ids})
                        expr_parts.append((invested_optional * factor_optional).sum(self.dim_name))
                    expr = sum(expr_parts) if expr_parts else 0
                else:
                    expr = factor_stacked.sum(self.dim_name)
            else:
                continue

            if isinstance(expr, (int, float)) and expr == 0:
                continue
            if isinstance(expr, (int, float)):
                expr = xr.DataArray(expr)
            self.model.effects.add_share_periodic(expr.expand_dims(effect=[effect_name]))


@register_class_for_io
class SourceAndSink(Component):
    """
    A SourceAndSink combines both supply and demand capabilities in a single component.

    SourceAndSink components can both consume AND provide energy or material flows
    from and to the system, making them ideal for modeling markets, (simple) storage facilities,
    or bidirectional grid connections where buying and selling occur at the same location.

    Args:
        id: The id of the Element. Used to identify it in the FlowSystem.
        inputs: Input-flows into the SourceAndSink representing consumption/demand side.
        outputs: Output-flows from the SourceAndSink representing supply/generation side.
        prevent_simultaneous_flow_rates: If True, prevents simultaneous input and output
            flows. This enforces that the component operates either as a source OR sink
            at any given time, but not both simultaneously. Default is True.
        meta_data: Used to store additional information about the Element. Not used
            internally but saved in results. Only use Python native types.

    Examples:
        Electricity market connection (buy/sell to grid):

        ```python
        electricity_market = SourceAndSink(
            id='grid_connection',
            inputs=[electricity_purchase],  # Buy from grid
            outputs=[electricity_sale],  # Sell to grid
            prevent_simultaneous_flow_rates=True,  # Can't buy and sell simultaneously
        )
        ```

        Natural gas storage facility:

        ```python
        gas_storage_facility = SourceAndSink(
            id='underground_gas_storage',
            inputs=[gas_injection_flow],  # Inject gas into storage
            outputs=[gas_withdrawal_flow],  # Withdraw gas from storage
            prevent_simultaneous_flow_rates=True,  # Injection or withdrawal, not both
        )
        ```

        District heating network connection:

        ```python
        dh_connection = SourceAndSink(
            id='district_heating_tie',
            inputs=[heat_purchase_flow],  # Purchase heat from network
            outputs=[heat_sale_flow],  # Sell excess heat to network
            prevent_simultaneous_flow_rates=False,  # May allow simultaneous flows
        )
        ```

        Industrial waste heat exchange:

        ```python
        waste_heat_exchange = SourceAndSink(
            id='industrial_heat_hub',
            inputs=[
                waste_heat_input_a,  # Receive waste heat from process A
                waste_heat_input_b,  # Receive waste heat from process B
            ],
            outputs=[
                useful_heat_supply_c,  # Supply heat to process C
                useful_heat_supply_d,  # Supply heat to process D
            ],
            prevent_simultaneous_flow_rates=False,  # Multiple simultaneous flows allowed
        )
        ```

    Note:
        When prevent_simultaneous_flow_rates is True, binary variables are created to
        ensure mutually exclusive operation between input and output flows, which
        increases computational complexity but reflects realistic market or storage
        operation constraints.

        SourceAndSink is particularly useful for modeling:
        - Energy markets with bidirectional trading
        - Storage facilities with injection/withdrawal operations
        - Grid tie points with import/export capabilities
        - Waste exchange networks with multiple participants

    Deprecated:
        The deprecated `sink` and `source` kwargs are accepted for compatibility but will be removed in future releases.
    """

    def __init__(
        self,
        id: str | None = None,
        inputs: list[Flow] | None = None,
        outputs: list[Flow] | None = None,
        prevent_simultaneous_flow_rates: bool = True,
        meta_data: dict | None = None,
        color: str | None = None,
        **kwargs,
    ):
        # Convert dict to list for deserialization compatibility (IdLists serialize as dicts)
        _inputs_list = list(inputs.values()) if isinstance(inputs, dict) else (inputs or [])
        _outputs_list = list(outputs.values()) if isinstance(outputs, dict) else (outputs or [])
        super().__init__(
            id,
            inputs=_inputs_list,
            outputs=_outputs_list,
            prevent_simultaneous_flows=_inputs_list + _outputs_list if prevent_simultaneous_flow_rates else None,
            meta_data=meta_data,
            color=color,
            **kwargs,
        )
        self.prevent_simultaneous_flow_rates = prevent_simultaneous_flow_rates


@register_class_for_io
class Source(Component):
    """
    A Source generates or provides energy or material flows into the system.

    Sources represent supply points like power plants, fuel suppliers, renewable
    energy sources, or any system boundary where flows originate. They provide
    unlimited supply capability subject to flow constraints, demand patterns and effects.

    Args:
        id: The id of the Element. Used to identify it in the FlowSystem.
        outputs: Output-flows from the source. Can be single flow or list of flows
            for sources providing multiple commodities or services.
        meta_data: Used to store additional information about the Element. Not used
            internally but saved in results. Only use Python native types.
        prevent_simultaneous_flow_rates: If True, only one output flow can be active
            at a time. Useful for modeling mutually exclusive supply options. Default is False.

    Examples:
        Simple electricity grid connection:

        ```python
        grid_source = Source(id='electrical_grid', outputs=[grid_electricity_flow])
        ```

        Natural gas supply with cost and capacity constraints:

        ```python
        gas_supply = Source(
            id='gas_network',
            outputs=[
                Flow(
                    label='natural_gas_flow',
                    bus=gas_bus,
                    size=1000,  # Maximum 1000 kW supply capacity
                    effects_per_flow_hour={'cost': 0.04},  # €0.04/kWh gas cost
                )
            ],
        )
        ```

        Multi-fuel power plant with switching constraints:

        ```python
        multi_fuel_plant = Source(
            id='flexible_generator',
            outputs=[coal_electricity, gas_electricity, biomass_electricity],
            prevent_simultaneous_flow_rates=True,  # Can only use one fuel at a time
        )
        ```

        Renewable energy source with investment optimization:

        ```python
        solar_farm = Source(
            id='solar_pv',
            outputs=[
                Flow(
                    label='solar_power',
                    bus=electricity_bus,
                    size=InvestParameters(
                        minimum_size=0,
                        maximum_size=50000,  # Up to 50 MW
                        specific_effects={'cost': 800},  # €800/kW installed
                        fix_effects={'cost': 100000},  # €100k development costs
                    ),
                    fixed_relative_profile=solar_profile,  # Hourly generation profile
                )
            ],
        )
        ```

    Deprecated:
        The deprecated `source` kwarg is accepted for compatibility but will be removed in future releases.
    """

    def __init__(
        self,
        id: str | None = None,
        outputs: list[Flow] | None = None,
        meta_data: dict | None = None,
        prevent_simultaneous_flow_rates: bool = False,
        color: str | None = None,
        **kwargs,
    ):
        self.prevent_simultaneous_flow_rates = prevent_simultaneous_flow_rates
        super().__init__(
            id,
            outputs=outputs,
            meta_data=meta_data,
            prevent_simultaneous_flows=outputs if prevent_simultaneous_flow_rates else None,
            color=color,
            **kwargs,
        )


@register_class_for_io
class Sink(Component):
    """
    A Sink consumes energy or material flows from the system.

    Sinks represent demand points like electrical loads, heat demands, material
    consumption, or any system boundary where flows terminate. They provide
    unlimited consumption capability subject to flow constraints, demand patterns and effects.

    Args:
        id: The id of the Element. Used to identify it in the FlowSystem.
        inputs: Input-flows into the sink. Can be single flow or list of flows
            for sinks consuming multiple commodities or services.
        meta_data: Used to store additional information about the Element. Not used
            internally but saved in results. Only use Python native types.
        prevent_simultaneous_flow_rates: If True, only one input flow can be active
            at a time. Useful for modeling mutually exclusive consumption options. Default is False.

    Examples:
        Simple electrical demand:

        ```python
        electrical_load = Sink(id='building_load', inputs=[electricity_demand_flow])
        ```

        Heat demand with time-varying profile:

        ```python
        heat_demand = Sink(
            id='district_heating_load',
            inputs=[
                Flow(
                    label='heat_consumption',
                    bus=heat_bus,
                    fixed_relative_profile=hourly_heat_profile,  # Demand profile
                    size=2000,  # Peak demand of 2000 kW
                )
            ],
        )
        ```

        Multi-energy building with switching capabilities:

        ```python
        flexible_building = Sink(
            id='smart_building',
            inputs=[electricity_heating, gas_heating, heat_pump_heating],
            prevent_simultaneous_flow_rates=True,  # Can only use one heating mode
        )
        ```

        Industrial process with variable demand:

        ```python
        factory_load = Sink(
            id='manufacturing_plant',
            inputs=[
                Flow(
                    label='electricity_process',
                    bus=electricity_bus,
                    size=5000,  # Base electrical load
                    effects_per_flow_hour={'cost': -0.1},  # Value of service (negative cost)
                ),
                Flow(
                    label='steam_process',
                    bus=steam_bus,
                    size=3000,  # Process steam demand
                    fixed_relative_profile=production_schedule,
                ),
            ],
        )
        ```

    Deprecated:
        The deprecated `sink` kwarg is accepted for compatibility but will be removed in future releases.
    """

    def __init__(
        self,
        id: str | None = None,
        inputs: list[Flow] | None = None,
        meta_data: dict | None = None,
        prevent_simultaneous_flow_rates: bool = False,
        color: str | None = None,
        **kwargs,
    ):
        """Initialize a Sink (consumes flow from the system).

        Args:
            id: Unique element id.
            inputs: Input flows for the sink.
            meta_data: Arbitrary metadata attached to the element.
            prevent_simultaneous_flow_rates: If True, prevents simultaneous nonzero flow rates
                across the element's inputs by wiring that restriction into the base Component setup.
            color: Optional color for visualizations.
        """

        self.prevent_simultaneous_flow_rates = prevent_simultaneous_flow_rates
        super().__init__(
            id,
            inputs=inputs,
            meta_data=meta_data,
            prevent_simultaneous_flows=inputs if prevent_simultaneous_flow_rates else None,
            color=color,
            **kwargs,
        )
