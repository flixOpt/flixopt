"""
This Module contains high-level classes to easily model a FlowSystem.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .components import LinearConverter
from .structure import register_class_for_io

if TYPE_CHECKING:
    from .elements import Flow
    from .interface import OnOffParameters
    from .types import Numeric_TPS

logger = logging.getLogger('flixopt')


@register_class_for_io
class Boiler(LinearConverter):
    """
    A specialized LinearConverter representing a fuel-fired boiler for thermal energy generation.

    Boilers convert fuel input into thermal energy with a specified efficiency factor.
    This is a simplified wrapper around LinearConverter with predefined conversion
    relationships for thermal generation applications.

    Args:
        label: The label of the Element. Used to identify it in the FlowSystem.
        thermal_efficiency: Thermal efficiency factor (0-1 range). Defines the ratio of thermal
            output to fuel input energy content.
        fuel_flow: Fuel input-flow representing fuel consumption.
        thermal_flow: Thermal output-flow representing heat generation.
        on_off_parameters: Parameters defining binary operation constraints and costs.
        meta_data: Used to store additional information. Not used internally but
            saved in results. Only use Python native types.

    Examples:
        Natural gas boiler:

        ```python
        gas_boiler = Boiler(
            label='natural_gas_boiler',
            thermal_efficiency=0.85,  # 85% thermal efficiency
            fuel_flow=natural_gas_flow,
            thermal_flow=hot_water_flow,
        )
        ```

        Biomass boiler with seasonal efficiency variation:

        ```python
        biomass_boiler = Boiler(
            label='wood_chip_boiler',
            thermal_efficiency=seasonal_efficiency_profile,  # Time-varying efficiency
            fuel_flow=biomass_flow,
            thermal_flow=district_heat_flow,
            on_off_parameters=OnOffParameters(
                consecutive_on_hours_min=4,  # Minimum 4-hour operation
                effects_per_switch_on={'startup_fuel': 50},  # Startup fuel penalty
            ),
        )
        ```

    Note:
        The conversion relationship is: thermal_flow = fuel_flow × thermal_efficiency

        Efficiency should be between 0 and 1, where 1 represents perfect conversion
        (100% of fuel energy converted to useful thermal output).
    """

    def __init__(
        self,
        label: str,
        thermal_efficiency: Numeric_TPS | None = None,
        fuel_flow: Flow | None = None,
        thermal_flow: Flow | None = None,
        on_off_parameters: OnOffParameters | None = None,
        meta_data: dict | None = None,
    ):
        # Validate required parameters
        if fuel_flow is None:
            raise ValueError(f"'{label}': fuel_flow is required and cannot be None")
        if thermal_flow is None:
            raise ValueError(f"'{label}': thermal_flow is required and cannot be None")
        if thermal_efficiency is None:
            raise ValueError(f"'{label}': thermal_efficiency is required and cannot be None")

        super().__init__(
            label,
            inputs=[fuel_flow],
            outputs=[thermal_flow],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )
        self.fuel_flow = fuel_flow
        self.thermal_flow = thermal_flow
        self.thermal_efficiency = thermal_efficiency  # Uses setter

    @property
    def thermal_efficiency(self):
        return self.conversion_factors[0][self.fuel_flow.label]

    @thermal_efficiency.setter
    def thermal_efficiency(self, value):
        check_bounds(value, 'thermal_efficiency', self.label_full, 0, 1)
        self.conversion_factors = [{self.fuel_flow.label: value, self.thermal_flow.label: 1}]


@register_class_for_io
class Power2Heat(LinearConverter):
    """
    A specialized LinearConverter representing electric resistance heating or power-to-heat conversion.

    Power2Heat components convert electrical energy directly into thermal energy through
    resistance heating elements, electrode boilers, or other direct electric heating
    technologies. This is a simplified wrapper around LinearConverter with predefined
    conversion relationships for electric heating applications.

    Args:
        label: The label of the Element. Used to identify it in the FlowSystem.
        thermal_efficiency: Thermal efficiency factor (0-1 range). For resistance heating this is
            typically close to 1.0 (nearly 100% efficiency), but may be lower for
            electrode boilers or systems with distribution losses.
        electrical_flow: Electrical input-flow representing electricity consumption.
        thermal_flow: Thermal output-flow representing heat generation.
        on_off_parameters: Parameters defining binary operation constraints and costs.
        meta_data: Used to store additional information. Not used internally but
            saved in results. Only use Python native types.

    Examples:
        Electric resistance heater:

        ```python
        electric_heater = Power2Heat(
            label='resistance_heater',
            thermal_efficiency=0.98,  # 98% efficiency (small losses)
            electrical_flow=electricity_flow,
            thermal_flow=space_heating_flow,
        )
        ```

        Electrode boiler for industrial steam:

        ```python
        electrode_boiler = Power2Heat(
            label='electrode_steam_boiler',
            thermal_efficiency=0.95,  # 95% efficiency including boiler losses
            electrical_flow=industrial_electricity,
            thermal_flow=process_steam_flow,
            on_off_parameters=OnOffParameters(
                consecutive_on_hours_min=1,  # Minimum 1-hour operation
                effects_per_switch_on={'startup_cost': 100},
            ),
        )
        ```

    Note:
        The conversion relationship is: thermal_flow = electrical_flow × thermal_efficiency

        Unlike heat pumps, Power2Heat systems cannot exceed 100% efficiency (thermal_efficiency ≤ 1.0)
        as they only convert electrical energy without extracting additional energy
        from the environment. However, they provide fast response times and precise
        temperature control.
    """

    def __init__(
        self,
        label: str,
        thermal_efficiency: Numeric_TPS | None = None,
        electrical_flow: Flow | None = None,
        thermal_flow: Flow | None = None,
        on_off_parameters: OnOffParameters | None = None,
        meta_data: dict | None = None,
    ):
        # Validate required parameters
        if electrical_flow is None:
            raise ValueError(f"'{label}': electrical_flow is required and cannot be None")
        if thermal_flow is None:
            raise ValueError(f"'{label}': thermal_flow is required and cannot be None")
        if thermal_efficiency is None:
            raise ValueError(f"'{label}': thermal_efficiency is required and cannot be None")

        super().__init__(
            label,
            inputs=[electrical_flow],
            outputs=[thermal_flow],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )

        self.electrical_flow = electrical_flow
        self.thermal_flow = thermal_flow
        self.thermal_efficiency = thermal_efficiency  # Uses setter

    @property
    def thermal_efficiency(self):
        return self.conversion_factors[0][self.electrical_flow.label]

    @thermal_efficiency.setter
    def thermal_efficiency(self, value):
        check_bounds(value, 'thermal_efficiency', self.label_full, 0, 1)
        self.conversion_factors = [{self.electrical_flow.label: value, self.thermal_flow.label: 1}]


@register_class_for_io
class HeatPump(LinearConverter):
    """
    A specialized LinearConverter representing an electric heat pump for thermal energy generation.

    Heat pumps convert electrical energy into thermal energy with a Coefficient of
    Performance (COP) greater than 1, making them more efficient than direct electric
    heating. This is a simplified wrapper around LinearConverter with predefined
    conversion relationships for heat pump applications.

    Args:
        label: The label of the Element. Used to identify it in the FlowSystem.
        cop: Coefficient of Performance (typically 1-20 range). Defines the ratio of
            thermal output to electrical input. COP > 1 indicates the heat pump extracts
            additional energy from the environment.
        electrical_flow: Electrical input-flow representing electricity consumption.
        thermal_flow: Thermal output-flow representing heat generation.
        on_off_parameters: Parameters defining binary operation constraints and costs.
        meta_data: Used to store additional information. Not used internally but
            saved in results. Only use Python native types.

    Examples:
        Air-source heat pump with constant COP:

        ```python
        air_hp = HeatPump(
            label='air_source_heat_pump',
            cop=3.5,  # COP of 3.5 (350% efficiency)
            electrical_flow=electricity_flow,
            thermal_flow=heating_flow,
        )
        ```

        Ground-source heat pump with temperature-dependent COP:

        ```python
        ground_hp = HeatPump(
            label='geothermal_heat_pump',
            cop=temperature_dependent_cop,  # Time-varying COP based on ground temp
            electrical_flow=electricity_flow,
            thermal_flow=radiant_heating_flow,
            on_off_parameters=OnOffParameters(
                consecutive_on_hours_min=2,  # Avoid frequent cycling
                effects_per_running_hour={'maintenance': 0.5},
            ),
        )
        ```

    Note:
        The conversion relationship is: thermal_flow = electrical_flow × COP

        COP should be greater than 1 for realistic heat pump operation, with typical
        values ranging from 2-6 depending on technology and operating conditions.
        Higher COP values indicate more efficient heat extraction from the environment.
    """

    def __init__(
        self,
        label: str,
        cop: Numeric_TPS | None = None,
        electrical_flow: Flow | None = None,
        thermal_flow: Flow | None = None,
        on_off_parameters: OnOffParameters | None = None,
        meta_data: dict | None = None,
    ):
        # Validate required parameters
        if electrical_flow is None:
            raise ValueError(f"'{label}': electrical_flow is required and cannot be None")
        if thermal_flow is None:
            raise ValueError(f"'{label}': thermal_flow is required and cannot be None")
        if cop is None:
            raise ValueError(f"'{label}': cop is required and cannot be None")

        super().__init__(
            label,
            inputs=[electrical_flow],
            outputs=[thermal_flow],
            conversion_factors=[],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )
        self.electrical_flow = electrical_flow
        self.thermal_flow = thermal_flow
        self.cop = cop  # Uses setter

    @property
    def cop(self):
        return self.conversion_factors[0][self.electrical_flow.label]

    @cop.setter
    def cop(self, value):
        check_bounds(value, 'cop', self.label_full, 1, 20)
        self.conversion_factors = [{self.electrical_flow.label: value, self.thermal_flow.label: 1}]


@register_class_for_io
class CoolingTower(LinearConverter):
    """
    A specialized LinearConverter representing a cooling tower for waste heat rejection.

    Cooling towers consume electrical energy (for fans, pumps) to reject thermal energy
    to the environment through evaporation and heat transfer. The electricity demand
    is typically a small fraction of the thermal load being rejected. This component
    has no thermal outputs as the heat is rejected to the environment.

    Args:
        label: The label of the Element. Used to identify it in the FlowSystem.
        specific_electricity_demand: Auxiliary electricity demand per unit of cooling
            power (dimensionless, typically 0.01-0.05 range). Represents the fraction
            of thermal power that must be supplied as electricity for fans and pumps.
        electrical_flow: Electrical input-flow representing electricity consumption for fans/pumps.
        thermal_flow: Thermal input-flow representing waste heat to be rejected to environment.
        on_off_parameters: Parameters defining binary operation constraints and costs.
        meta_data: Used to store additional information. Not used internally but
            saved in results. Only use Python native types.

    Examples:
        Industrial cooling tower:

        ```python
        cooling_tower = CoolingTower(
            label='process_cooling_tower',
            specific_electricity_demand=0.025,  # 2.5% auxiliary power
            electrical_flow=cooling_electricity,
            thermal_flow=waste_heat_flow,
        )
        ```

        Power plant condenser cooling:

        ```python
        condenser_cooling = CoolingTower(
            label='power_plant_cooling',
            specific_electricity_demand=0.015,  # 1.5% auxiliary power
            electrical_flow=auxiliary_electricity,
            thermal_flow=condenser_waste_heat,
            on_off_parameters=OnOffParameters(
                consecutive_on_hours_min=4,  # Minimum operation time
                effects_per_running_hour={'water_consumption': 2.5},  # m³/h
            ),
        )
        ```

    Note:
        The conversion relationship is: electrical_flow = thermal_flow × specific_electricity_demand

        The cooling tower consumes electrical power proportional to the thermal load.
        No thermal energy is produced - all thermal input is rejected to the environment.

        Typical specific electricity demands range from 1-5% of the thermal cooling load,
        depending on tower design, climate conditions, and operational requirements.
    """

    def __init__(
        self,
        label: str,
        specific_electricity_demand: Numeric_TPS,
        electrical_flow: Flow | None = None,
        thermal_flow: Flow | None = None,
        on_off_parameters: OnOffParameters | None = None,
        meta_data: dict | None = None,
    ):
        # Validate required parameters
        if electrical_flow is None:
            raise ValueError(f"'{label}': electrical_flow is required and cannot be None")
        if thermal_flow is None:
            raise ValueError(f"'{label}': thermal_flow is required and cannot be None")

        super().__init__(
            label,
            inputs=[electrical_flow, thermal_flow],
            outputs=[],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )

        self.electrical_flow = electrical_flow
        self.thermal_flow = thermal_flow
        self.specific_electricity_demand = specific_electricity_demand  # Uses setter

    @property
    def specific_electricity_demand(self):
        return self.conversion_factors[0][self.thermal_flow.label]

    @specific_electricity_demand.setter
    def specific_electricity_demand(self, value):
        check_bounds(value, 'specific_electricity_demand', self.label_full, 0, 1)
        self.conversion_factors = [{self.electrical_flow.label: -1, self.thermal_flow.label: value}]


@register_class_for_io
class CHP(LinearConverter):
    """
    A specialized LinearConverter representing a Combined Heat and Power (CHP) unit.

    CHP units simultaneously generate both electrical and thermal energy from a single
    fuel input, providing higher overall efficiency than separate generation. This is
    a wrapper around LinearConverter with predefined conversion relationships for
    cogeneration applications.

    Args:
        label: The label of the Element. Used to identify it in the FlowSystem.
        thermal_efficiency: Thermal efficiency factor (0-1 range). Defines the fraction of fuel
            energy converted to useful thermal output.
        electrical_efficiency: Electrical efficiency factor (0-1 range). Defines the fraction of fuel
            energy converted to electrical output.
        fuel_flow: Fuel input-flow representing fuel consumption.
        electrical_flow: Electrical output-flow representing electricity generation.
        thermal_flow: Thermal output-flow representing heat generation.
        on_off_parameters: Parameters defining binary operation constraints and costs.
        meta_data: Used to store additional information. Not used internally but
            saved in results. Only use Python native types.

    Examples:
        Natural gas CHP unit:

        ```python
        gas_chp = CHP(
            label='natural_gas_chp',
            thermal_efficiency=0.45,  # 45% thermal efficiency
            electrical_efficiency=0.35,  # 35% electrical efficiency (80% total)
            fuel_flow=natural_gas_flow,
            electrical_flow=electricity_flow,
            thermal_flow=district_heat_flow,
        )
        ```

        Industrial CHP with operational constraints:

        ```python
        industrial_chp = CHP(
            label='industrial_chp',
            thermal_efficiency=0.40,
            electrical_efficiency=0.38,
            fuel_flow=fuel_gas_flow,
            electrical_flow=plant_electricity,
            thermal_flow=process_steam,
            on_off_parameters=OnOffParameters(
                consecutive_on_hours_min=8,  # Minimum 8-hour operation
                effects_per_switch_on={'startup_cost': 5000},
                on_hours_max=6000,  # Annual operating limit
            ),
        )
        ```

    Note:
        The conversion relationships are:
        - thermal_flow = fuel_flow × thermal_efficiency (thermal output)
        - electrical_flow = fuel_flow × electrical_efficiency (electrical output)

        Total efficiency (thermal_efficiency + electrical_efficiency) should be ≤ 1.0, with typical combined
        efficiencies of 80-90% for modern CHP units. This provides significant
        efficiency gains compared to separate heat and power generation.
    """

    def __init__(
        self,
        label: str,
        thermal_efficiency: Numeric_TPS | None = None,
        electrical_efficiency: Numeric_TPS | None = None,
        fuel_flow: Flow | None = None,
        electrical_flow: Flow | None = None,
        thermal_flow: Flow | None = None,
        on_off_parameters: OnOffParameters | None = None,
        meta_data: dict | None = None,
    ):
        # Validate required parameters
        if fuel_flow is None:
            raise ValueError(f"'{label}': fuel_flow is required and cannot be None")
        if electrical_flow is None:
            raise ValueError(f"'{label}': electrical_flow is required and cannot be None")
        if thermal_flow is None:
            raise ValueError(f"'{label}': thermal_flow is required and cannot be None")
        if thermal_efficiency is None:
            raise ValueError(f"'{label}': thermal_efficiency is required and cannot be None")
        if electrical_efficiency is None:
            raise ValueError(f"'{label}': electrical_efficiency is required and cannot be None")

        super().__init__(
            label,
            inputs=[fuel_flow],
            outputs=[thermal_flow, electrical_flow],
            conversion_factors=[{}, {}],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )

        self.fuel_flow = fuel_flow
        self.electrical_flow = electrical_flow
        self.thermal_flow = thermal_flow
        self.thermal_efficiency = thermal_efficiency  # Uses setter
        self.electrical_efficiency = electrical_efficiency  # Uses setter

        check_bounds(
            electrical_efficiency + thermal_efficiency,
            'thermal_efficiency+electrical_efficiency',
            self.label_full,
            0,
            1,
        )

    @property
    def thermal_efficiency(self):
        return self.conversion_factors[0][self.fuel_flow.label]

    @thermal_efficiency.setter
    def thermal_efficiency(self, value):
        check_bounds(value, 'thermal_efficiency', self.label_full, 0, 1)
        self.conversion_factors[0] = {self.fuel_flow.label: value, self.thermal_flow.label: 1}

    @property
    def electrical_efficiency(self):
        return self.conversion_factors[1][self.fuel_flow.label]

    @electrical_efficiency.setter
    def electrical_efficiency(self, value):
        check_bounds(value, 'electrical_efficiency', self.label_full, 0, 1)
        self.conversion_factors[1] = {self.fuel_flow.label: value, self.electrical_flow.label: 1}


@register_class_for_io
class HeatPumpWithSource(LinearConverter):
    """
    A specialized LinearConverter representing a heat pump with explicit heat source modeling.

    This component models a heat pump that extracts thermal energy from a heat source
    (ground, air, water) and upgrades it using electrical energy to provide higher-grade
    thermal output. Unlike the simple HeatPump class, this explicitly models both the
    heat source extraction and electrical consumption with their interdependent relationships.

    Args:
        label: The label of the Element. Used to identify it in the FlowSystem.
        cop: Coefficient of Performance (typically 1-20 range). Defines the ratio of
            thermal output to electrical input. The heat source extraction is automatically
            calculated as heat_source_flow = thermal_flow × (COP-1)/COP.
        electrical_flow: Electrical input-flow representing electricity consumption for compressor.
        heat_source_flow: Heat source input-flow representing thermal energy extracted from environment
            (ground, air, water source).
        thermal_flow: Thermal output-flow representing useful heat delivered to the application.
        on_off_parameters: Parameters defining binary operation constraints and costs.
        meta_data: Used to store additional information. Not used internally but
            saved in results. Only use Python native types.

    Examples:
        Ground-source heat pump with explicit ground coupling:

        ```python
        ground_source_hp = HeatPumpWithSource(
            label='geothermal_heat_pump',
            cop=4.5,  # High COP due to stable ground temperature
            electrical_flow=electricity_flow,
            heat_source_flow=ground_heat_extraction,  # Heat extracted from ground loop
            thermal_flow=building_heating_flow,
        )
        ```

        Air-source heat pump with temperature-dependent performance:

        ```python
        waste_heat_pump = HeatPumpWithSource(
            label='waste_heat_pump',
            cop=temperature_dependent_cop,  # Varies with temperature of heat source
            electrical_flow=electricity_consumption,
            heat_source_flow=industrial_heat_extraction,  # Heat extracted from a industrial process or waste water
            thermal_flow=heat_supply,
            on_off_parameters=OnOffParameters(
                consecutive_on_hours_min=0.5,  # 30-minute minimum runtime
                effects_per_switch_on={'costs': 1000},
            ),
        )
        ```

    Note:
        The conversion relationships are:
        - thermal_flow = electrical_flow × COP (thermal output from electrical input)
        - heat_source_flow = thermal_flow × (COP-1)/COP (heat source extraction)
        - Energy balance: thermal_flow = electrical_flow + heat_source_flow

        This formulation explicitly tracks the heat source, which is
        important for systems where the source capacity or temperature is limited,
        or where the impact of heat extraction must be considered.

        COP should be > 1 for thermodynamically valid operation, with typical
        values of 2-6 depending on source and sink temperatures.
    """

    def __init__(
        self,
        label: str,
        cop: Numeric_TPS | None = None,
        electrical_flow: Flow | None = None,
        heat_source_flow: Flow | None = None,
        thermal_flow: Flow | None = None,
        on_off_parameters: OnOffParameters | None = None,
        meta_data: dict | None = None,
    ):
        # Validate required parameters
        if electrical_flow is None:
            raise ValueError(f"'{label}': electrical_flow is required and cannot be None")
        if heat_source_flow is None:
            raise ValueError(f"'{label}': heat_source_flow is required and cannot be None")
        if thermal_flow is None:
            raise ValueError(f"'{label}': thermal_flow is required and cannot be None")
        if cop is None:
            raise ValueError(f"'{label}': cop is required and cannot be None")

        super().__init__(
            label,
            inputs=[electrical_flow, heat_source_flow],
            outputs=[thermal_flow],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )
        self.electrical_flow = electrical_flow
        self.heat_source_flow = heat_source_flow
        self.thermal_flow = thermal_flow
        self.cop = cop  # Uses setter

    @property
    def cop(self):
        return self.conversion_factors[0][self.electrical_flow.label]

    @cop.setter
    def cop(self, value):
        check_bounds(value, 'cop', self.label_full, 1, 20)
        if np.any(np.asarray(value) == 1):
            raise ValueError(f'{self.label_full}.cop must be strictly !=1 for HeatPumpWithSource.')
        self.conversion_factors = [
            {self.electrical_flow.label: value, self.thermal_flow.label: 1},
            {self.heat_source_flow.label: value / (value - 1), self.thermal_flow.label: 1},
        ]


def check_bounds(
    value: Numeric_TPS,
    parameter_label: str,
    element_label: str,
    lower_bound: Numeric_TPS,
    upper_bound: Numeric_TPS,
) -> None:
    """
    Check if the value is within the bounds. The bounds are exclusive.
    If not, log a warning.
    Args:
        value: The value to check.
        parameter_label: The label of the value.
        element_label: The label of the element.
        lower_bound: The lower bound.
        upper_bound: The upper bound.
    """
    # Convert to array for shape and statistics
    value_arr = np.asarray(value)

    if not np.all(value_arr > lower_bound):
        logger.warning(
            f"'{element_label}.{parameter_label}' <= lower bound {lower_bound}. "
            f'{parameter_label}.min={float(np.min(value_arr))}, shape={np.shape(value_arr)}'
        )
    if not np.all(value_arr < upper_bound):
        logger.warning(
            f"'{element_label}.{parameter_label}' >= upper bound {upper_bound}. "
            f'{parameter_label}.max={float(np.max(value_arr))}, shape={np.shape(value_arr)}'
        )
