"""
This Module contains high-level classes to easily model a FlowSystem.
"""

import logging
from typing import Dict, Optional

import numpy as np

from .components import LinearConverter
from .core import NumericDataTS, TimeSeriesData
from .elements import Flow
from .interface import OnOffParameters
from .structure import register_class_for_io

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
        eta: NumericDataTS,
        Q_fu: Flow,
        Q_th: Flow,
        on_off_parameters: OnOffParameters = None,
        meta_data: Optional[Dict] = None,
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
    def __init__(
        self,
        label: str,
        eta: NumericDataTS,
        P_el: Flow,
        Q_th: Flow,
        on_off_parameters: OnOffParameters = None,
        meta_data: Optional[Dict] = None,
    ):
        """
        Args:
            label: The label of the Element. Used to identify it in the FlowSystem
            eta: thermal efficiency.
            P_el: electric input-flow
            Q_th: thermal output-flow.
            on_off_parameters: Parameters defining the on/off behavior of the component.
            meta_data: used to store more information about the Element. Is not used internally, but saved in the results. Only use python native types.
        """
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
        COP: NumericDataTS,
        P_el: Flow,
        Q_th: Flow,
        on_off_parameters: OnOffParameters = None,
        meta_data: Optional[Dict] = None,
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
    def __init__(
        self,
        label: str,
        specific_electricity_demand: NumericDataTS,
        P_el: Flow,
        Q_th: Flow,
        on_off_parameters: OnOffParameters = None,
        meta_data: Optional[Dict] = None,
    ):
        """
        Args:
            label: The label of the Element. Used to identify it in the FlowSystem
            specific_electricity_demand: auxiliary electricty demand per cooling power, i.g. 0.02 (2 %).
            P_el: electricity input-flow.
            Q_th: thermal input-flow.
            on_off_parameters: Parameters defining the on/off behavior of the component.
            meta_data: used to store more information about the Element. Is not used internally, but saved in the results. Only use python native types.
        """
        super().__init__(
            label,
            inputs=[P_el, Q_th],
            outputs=[],
            conversion_factors=[{P_el.label: 1, Q_th.label: -specific_electricity_demand}],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )

        self.P_el = P_el
        self.Q_th = Q_th

        check_bounds(specific_electricity_demand, 'specific_electricity_demand', self.label_full, 0, 1)

    @property
    def specific_electricity_demand(self):
        return -self.conversion_factors[0][self.Q_th.label]

    @specific_electricity_demand.setter
    def specific_electricity_demand(self, value):
        check_bounds(value, 'specific_electricity_demand', self.label_full, 0, 1)
        self.conversion_factors[0][self.Q_th.label] = -value


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
        eta_th: NumericDataTS,
        eta_el: NumericDataTS,
        Q_fu: Flow,
        P_el: Flow,
        Q_th: Flow,
        on_off_parameters: OnOffParameters = None,
        meta_data: Optional[Dict] = None,
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
    def __init__(
        self,
        label: str,
        COP: NumericDataTS,
        P_el: Flow,
        Q_ab: Flow,
        Q_th: Flow,
        on_off_parameters: OnOffParameters = None,
        meta_data: Optional[Dict] = None,
    ):
        """
        Args:
            label: The label of the Element. Used to identify it in the FlowSystem
            COP: Coefficient of performance.
            Q_ab: Heatsource input-flow.
            P_el: electricity input-flow.
            Q_th: thermal output-flow.
            on_off_parameters: Parameters defining the on/off behavior of the component.
            meta_data: used to store more information about the Element. Is not used internally, but saved in the results. Only use python native types.
        """

        # super:
        electricity = {P_el.label: COP, Q_th.label: 1}
        heat_source = {Q_ab.label: COP / (COP - 1), Q_th.label: 1}

        super().__init__(
            label,
            inputs=[P_el, Q_ab],
            outputs=[Q_th],
            conversion_factors=[electricity, heat_source],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )
        self.P_el = P_el
        self.Q_ab = Q_ab
        self.Q_th = Q_th

    @property
    def COP(self):  # noqa: N802
        return self.conversion_factors[0][self.Q_th.label]

    @COP.setter
    def COP(self, value):  # noqa: N802
        check_bounds(value, 'COP', self.label_full, 1, 20)
        self.conversion_factors[0][self.Q_th.label] = value
        self.conversion_factors[1][self.Q_th.label] = value / (value - 1)


def check_bounds(
    value: NumericDataTS,
    parameter_label: str,
    element_label: str,
    lower_bound: NumericDataTS,
    upper_bound: NumericDataTS,
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
