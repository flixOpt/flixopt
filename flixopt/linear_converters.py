"""
This Module contains high-level classes to easily model a FlowSystem.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .components import LinearConverter
from .core import TimeSeriesData
from .structure import register_class_for_io

if TYPE_CHECKING:
    from .elements import Flow
    from .interface import OnOffParameters
    from .types import NumericData, Period, Scenario, Time

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
        eta: Thermal efficiency factor (0-1 range). Defines the ratio of thermal
            output to fuel input energy content.
        Q_fu: Fuel input-flow representing fuel consumption.
        Q_th: Thermal output-flow representing heat generation.
        on_off_parameters: Parameters defining binary operation constraints and costs.
        meta_data: Used to store additional information. Not used internally but
            saved in results. Only use Python native types.

    Examples:
        Natural gas boiler:

        ```python
        gas_boiler = Boiler(
            label='natural_gas_boiler',
            eta=0.85,  # 85% thermal efficiency
            Q_fu=natural_gas_flow,
            Q_th=hot_water_flow,
        )
        ```

        Biomass boiler with seasonal efficiency variation:

        ```python
        biomass_boiler = Boiler(
            label='wood_chip_boiler',
            eta=seasonal_efficiency_profile,  # Time-varying efficiency
            Q_fu=biomass_flow,
            Q_th=district_heat_flow,
            on_off_parameters=OnOffParameters(
                consecutive_on_hours_min=4,  # Minimum 4-hour operation
                effects_per_switch_on={'startup_fuel': 50},  # Startup fuel penalty
            ),
        )
        ```

    Note:
        The conversion relationship is: Q_th = Q_fu × eta

        Efficiency should be between 0 and 1, where 1 represents perfect conversion
        (100% of fuel energy converted to useful thermal output).
    """

    def __init__(
        self,
        label: str,
        eta: Numeric_TPS,
        Q_fu: Flow,
        Q_th: Flow,
        on_off_parameters: OnOffParameters | None = None,
        meta_data: dict | None = None,
    ):
        super().__init__(
            label,
            inputs=[Q_fu],
            outputs=[Q_th],
            conversion_factors=[{Q_fu.label: eta, Q_th.label: 1}],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )
        self.Q_fu = Q_fu
        self.Q_th = Q_th

    @property
    def eta(self):
        return self.conversion_factors[0][self.Q_fu.label]

    @eta.setter
    def eta(self, value):
        check_bounds(value, 'eta', self.label_full, 0, 1)
        self.conversion_factors[0][self.Q_fu.label] = value


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
        eta: Thermal efficiency factor (0-1 range). For resistance heating this is
            typically close to 1.0 (nearly 100% efficiency), but may be lower for
            electrode boilers or systems with distribution losses.
        P_el: Electrical input-flow representing electricity consumption.
        Q_th: Thermal output-flow representing heat generation.
        on_off_parameters: Parameters defining binary operation constraints and costs.
        meta_data: Used to store additional information. Not used internally but
            saved in results. Only use Python native types.

    Examples:
        Electric resistance heater:

        ```python
        electric_heater = Power2Heat(
            label='resistance_heater',
            eta=0.98,  # 98% efficiency (small losses)
            P_el=electricity_flow,
            Q_th=space_heating_flow,
        )
        ```

        Electrode boiler for industrial steam:

        ```python
        electrode_boiler = Power2Heat(
            label='electrode_steam_boiler',
            eta=0.95,  # 95% efficiency including boiler losses
            P_el=industrial_electricity,
            Q_th=process_steam_flow,
            on_off_parameters=OnOffParameters(
                consecutive_on_hours_min=1,  # Minimum 1-hour operation
                effects_per_switch_on={'startup_cost': 100},
            ),
        )
        ```

    Note:
        The conversion relationship is: Q_th = P_el × eta

        Unlike heat pumps, Power2Heat systems cannot exceed 100% efficiency (eta ≤ 1.0)
        as they only convert electrical energy without extracting additional energy
        from the environment. However, they provide fast response times and precise
        temperature control.
    """

    def __init__(
        self,
        label: str,
        eta: Numeric_TPS,
        P_el: Flow,
        Q_th: Flow,
        on_off_parameters: OnOffParameters | None = None,
        meta_data: dict | None = None,
    ):
        super().__init__(
            label,
            inputs=[P_el],
            outputs=[Q_th],
            conversion_factors=[{P_el.label: eta, Q_th.label: 1}],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )

        self.P_el = P_el
        self.Q_th = Q_th

    @property
    def eta(self):
        return self.conversion_factors[0][self.P_el.label]

    @eta.setter
    def eta(self, value):
        check_bounds(value, 'eta', self.label_full, 0, 1)
        self.conversion_factors[0][self.P_el.label] = value


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
        COP: Coefficient of Performance (typically 1-20 range). Defines the ratio of
            thermal output to electrical input. COP > 1 indicates the heat pump extracts
            additional energy from the environment.
        P_el: Electrical input-flow representing electricity consumption.
        Q_th: Thermal output-flow representing heat generation.
        on_off_parameters: Parameters defining binary operation constraints and costs.
        meta_data: Used to store additional information. Not used internally but
            saved in results. Only use Python native types.

    Examples:
        Air-source heat pump with constant COP:

        ```python
        air_hp = HeatPump(
            label='air_source_heat_pump',
            COP=3.5,  # COP of 3.5 (350% efficiency)
            P_el=electricity_flow,
            Q_th=heating_flow,
        )
        ```

        Ground-source heat pump with temperature-dependent COP:

        ```python
        ground_hp = HeatPump(
            label='geothermal_heat_pump',
            COP=temperature_dependent_cop,  # Time-varying COP based on ground temp
            P_el=electricity_flow,
            Q_th=radiant_heating_flow,
            on_off_parameters=OnOffParameters(
                consecutive_on_hours_min=2,  # Avoid frequent cycling
                effects_per_running_hour={'maintenance': 0.5},
            ),
        )
        ```

    Note:
        The conversion relationship is: Q_th = P_el × COP

        COP should be greater than 1 for realistic heat pump operation, with typical
        values ranging from 2-6 depending on technology and operating conditions.
        Higher COP values indicate more efficient heat extraction from the environment.
    """

    def __init__(
        self,
        label: str,
        COP: Numeric_TPS,
        P_el: Flow,
        Q_th: Flow,
        on_off_parameters: OnOffParameters | None = None,
        meta_data: dict | None = None,
    ):
        super().__init__(
            label,
            inputs=[P_el],
            outputs=[Q_th],
            conversion_factors=[{P_el.label: COP, Q_th.label: 1}],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )
        self.P_el = P_el
        self.Q_th = Q_th
        self.COP = COP

    @property
    def COP(self):  # noqa: N802
        return self.conversion_factors[0][self.P_el.label]

    @COP.setter
    def COP(self, value):  # noqa: N802
        check_bounds(value, 'COP', self.label_full, 1, 20)
        self.conversion_factors[0][self.P_el.label] = value


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
        P_el: Electrical input-flow representing electricity consumption for fans/pumps.
        Q_th: Thermal input-flow representing waste heat to be rejected to environment.
        on_off_parameters: Parameters defining binary operation constraints and costs.
        meta_data: Used to store additional information. Not used internally but
            saved in results. Only use Python native types.

    Examples:
        Industrial cooling tower:

        ```python
        cooling_tower = CoolingTower(
            label='process_cooling_tower',
            specific_electricity_demand=0.025,  # 2.5% auxiliary power
            P_el=cooling_electricity,
            Q_th=waste_heat_flow,
        )
        ```

        Power plant condenser cooling:

        ```python
        condenser_cooling = CoolingTower(
            label='power_plant_cooling',
            specific_electricity_demand=0.015,  # 1.5% auxiliary power
            P_el=auxiliary_electricity,
            Q_th=condenser_waste_heat,
            on_off_parameters=OnOffParameters(
                consecutive_on_hours_min=4,  # Minimum operation time
                effects_per_running_hour={'water_consumption': 2.5},  # m³/h
            ),
        )
        ```

    Note:
        The conversion relationship is: P_el = Q_th × specific_electricity_demand

        The cooling tower consumes electrical power proportional to the thermal load.
        No thermal energy is produced - all thermal input is rejected to the environment.

        Typical specific electricity demands range from 1-5% of the thermal cooling load,
        depending on tower design, climate conditions, and operational requirements.
    """

    def __init__(
        self,
        label: str,
        specific_electricity_demand: Numeric_TPS,
        P_el: Flow,
        Q_th: Flow,
        on_off_parameters: OnOffParameters | None = None,
        meta_data: dict | None = None,
    ):
        super().__init__(
            label,
            inputs=[P_el, Q_th],
            outputs=[],
            conversion_factors=[{P_el.label: -1, Q_th.label: specific_electricity_demand}],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )

        self.P_el = P_el
        self.Q_th = Q_th

        check_bounds(specific_electricity_demand, 'specific_electricity_demand', self.label_full, 0, 1)

    @property
    def specific_electricity_demand(self):
        return self.conversion_factors[0][self.Q_th.label]

    @specific_electricity_demand.setter
    def specific_electricity_demand(self, value):
        check_bounds(value, 'specific_electricity_demand', self.label_full, 0, 1)
        self.conversion_factors[0][self.Q_th.label] = value


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
        eta_th: Thermal efficiency factor (0-1 range). Defines the fraction of fuel
            energy converted to useful thermal output.
        eta_el: Electrical efficiency factor (0-1 range). Defines the fraction of fuel
            energy converted to electrical output.
        Q_fu: Fuel input-flow representing fuel consumption.
        P_el: Electrical output-flow representing electricity generation.
        Q_th: Thermal output-flow representing heat generation.
        on_off_parameters: Parameters defining binary operation constraints and costs.
        meta_data: Used to store additional information. Not used internally but
            saved in results. Only use Python native types.

    Examples:
        Natural gas CHP unit:

        ```python
        gas_chp = CHP(
            label='natural_gas_chp',
            eta_th=0.45,  # 45% thermal efficiency
            eta_el=0.35,  # 35% electrical efficiency (80% total)
            Q_fu=natural_gas_flow,
            P_el=electricity_flow,
            Q_th=district_heat_flow,
        )
        ```

        Industrial CHP with operational constraints:

        ```python
        industrial_chp = CHP(
            label='industrial_chp',
            eta_th=0.40,
            eta_el=0.38,
            Q_fu=fuel_gas_flow,
            P_el=plant_electricity,
            Q_th=process_steam,
            on_off_parameters=OnOffParameters(
                consecutive_on_hours_min=8,  # Minimum 8-hour operation
                effects_per_switch_on={'startup_cost': 5000},
                on_hours_total_max=6000,  # Annual operating limit
            ),
        )
        ```

    Note:
        The conversion relationships are:
        - Q_th = Q_fu × eta_th (thermal output)
        - P_el = Q_fu × eta_el (electrical output)

        Total efficiency (eta_th + eta_el) should be ≤ 1.0, with typical combined
        efficiencies of 80-90% for modern CHP units. This provides significant
        efficiency gains compared to separate heat and power generation.
    """

    def __init__(
        self,
        label: str,
        eta_th: Numeric_TPS,
        eta_el: Numeric_TPS,
        Q_fu: Flow,
        P_el: Flow,
        Q_th: Flow,
        on_off_parameters: OnOffParameters | None = None,
        meta_data: dict | None = None,
    ):
        heat = {Q_fu.label: eta_th, Q_th.label: 1}
        electricity = {Q_fu.label: eta_el, P_el.label: 1}

        super().__init__(
            label,
            inputs=[Q_fu],
            outputs=[Q_th, P_el],
            conversion_factors=[heat, electricity],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )

        self.Q_fu = Q_fu
        self.P_el = P_el
        self.Q_th = Q_th

        check_bounds(eta_el + eta_th, 'eta_th+eta_el', self.label_full, 0, 1)

    @property
    def eta_th(self):
        return self.conversion_factors[0][self.Q_fu.label]

    @eta_th.setter
    def eta_th(self, value):
        check_bounds(value, 'eta_th', self.label_full, 0, 1)
        self.conversion_factors[0][self.Q_fu.label] = value

    @property
    def eta_el(self):
        return self.conversion_factors[1][self.Q_fu.label]

    @eta_el.setter
    def eta_el(self, value):
        check_bounds(value, 'eta_el', self.label_full, 0, 1)
        self.conversion_factors[1][self.Q_fu.label] = value


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
        COP: Coefficient of Performance (typically 1-20 range). Defines the ratio of
            thermal output to electrical input. The heat source extraction is automatically
            calculated as Q_ab = Q_th × (COP-1)/COP.
        P_el: Electrical input-flow representing electricity consumption for compressor.
        Q_ab: Heat source input-flow representing thermal energy extracted from environment
            (ground, air, water source).
        Q_th: Thermal output-flow representing useful heat delivered to the application.
        on_off_parameters: Parameters defining binary operation constraints and costs.
        meta_data: Used to store additional information. Not used internally but
            saved in results. Only use Python native types.

    Examples:
        Ground-source heat pump with explicit ground coupling:

        ```python
        ground_source_hp = HeatPumpWithSource(
            label='geothermal_heat_pump',
            COP=4.5,  # High COP due to stable ground temperature
            P_el=electricity_flow,
            Q_ab=ground_heat_extraction,  # Heat extracted from ground loop
            Q_th=building_heating_flow,
        )
        ```

        Air-source heat pump with temperature-dependent performance:

        ```python
        waste_heat_pump = HeatPumpWithSource(
            label='waste_heat_pump',
            COP=temperature_dependent_cop,  # Varies with temperature of heat source
            P_el=electricity_consumption,
            Q_ab=industrial_heat_extraction,  # Heat extracted from a industrial process or waste water
            Q_th=heat_supply,
            on_off_parameters=OnOffParameters(
                consecutive_on_hours_min=0.5,  # 30-minute minimum runtime
                effects_per_switch_on={'costs': 1000},
            ),
        )
        ```

    Note:
        The conversion relationships are:
        - Q_th = P_el × COP (thermal output from electrical input)
        - Q_ab = Q_th × (COP-1)/COP (heat source extraction)
        - Energy balance: Q_th = P_el + Q_ab

        This formulation explicitly tracks the heat source, which is
        important for systems where the source capacity or temperature is limited,
        or where the impact of heat extraction must be considered.

        COP should be > 1 for thermodynamically valid operation, with typical
        values of 2-6 depending on source and sink temperatures.
    """

    def __init__(
        self,
        label: str,
        COP: Numeric_TPS,
        P_el: Flow,
        Q_ab: Flow,
        Q_th: Flow,
        on_off_parameters: OnOffParameters | None = None,
        meta_data: dict | None = None,
    ):
        super().__init__(
            label,
            inputs=[P_el, Q_ab],
            outputs=[Q_th],
            conversion_factors=[{P_el.label: COP, Q_th.label: 1}, {Q_ab.label: COP / (COP - 1), Q_th.label: 1}],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )
        self.P_el = P_el
        self.Q_ab = Q_ab
        self.Q_th = Q_th

        if np.any(np.asarray(self.COP) <= 1):
            raise ValueError(f'{self.label_full}.COP must be strictly > 1 for HeatPumpWithSource.')

    @property
    def COP(self):  # noqa: N802
        return self.conversion_factors[0][self.P_el.label]

    @COP.setter
    def COP(self, value):  # noqa: N802
        check_bounds(value, 'COP', self.label_full, 1, 20)
        if np.any(np.asarray(value) <= 1):
            raise ValueError(f'{self.label_full}.COP must be strictly > 1 for HeatPumpWithSource.')
        self.conversion_factors = [
            {self.P_el.label: value, self.Q_th.label: 1},
            {self.Q_ab.label: value / (value - 1), self.Q_th.label: 1},
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
    if isinstance(value, TimeSeriesData):
        value = value.data
    if isinstance(lower_bound, TimeSeriesData):
        lower_bound = lower_bound.data
    if isinstance(upper_bound, TimeSeriesData):
        upper_bound = upper_bound.data
    if not np.all(value > lower_bound):
        logger.warning(
            f"'{element_label}.{parameter_label}' is equal or below the common lower bound {lower_bound}."
            f'    {parameter_label}.min={np.min(value)};    {parameter_label}={value}'
        )
    if not np.all(value < upper_bound):
        logger.warning(
            f"'{element_label}.{parameter_label}' exceeds or matches the common upper bound {upper_bound}."
            f'    {parameter_label}.max={np.max(value)};    {parameter_label}={value}'
        )
