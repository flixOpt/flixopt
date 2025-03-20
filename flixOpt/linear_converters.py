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

logger = logging.getLogger('flixOpt')


@register_class_for_io
class Boiler(LinearConverter):
    def __init__(
        self,
        label: str,
        eta: NumericDataTS,
        Q_fu: Flow,
        Q_th: Flow,
        on_off_parameters: OnOffParameters = None,
        meta_data: Optional[Dict] = None,
    ):
        """
        constructor for boiler

        Parameters
        ----------
        label : str
            name of bolier.
        eta : float or TS
            thermal efficiency.
        Q_fu : Flow
            fuel input-flow
        Q_th : Flow
            thermal output-flow.
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        """
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
        Parameters
        ----------
        label : str
            name of bolier.
        eta : float or TS
            thermal efficiency.
        P_el : Flow
            electric input-flow
        Q_th : Flow
            thermal output-flow.
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results

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
    def __init__(
        self,
        label: str,
        COP: NumericDataTS,
        P_el: Flow,
        Q_th: Flow,
        on_off_parameters: OnOffParameters = None,
        meta_data: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        label : str
            name of heatpump.
        COP : float or TS
            Coefficient of performance.
        P_el : Flow
            electricity input-flow.
        Q_th : Flow
            thermal output-flow.
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        """
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
        Parameters
        ----------
        label : str
            name of cooling tower.
        specific_electricity_demand : float or TS
            auxiliary electricty demand per cooling power, i.g. 0.02 (2 %).
        P_el : Flow
            electricity input-flow.
        Q_th : Flow
            thermal input-flow.
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results

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
        """
        constructor of cCHP

        Parameters
        ----------
        label : str
            name of CHP-unit.
        eta_th : float or TS
            thermal efficiency.
        eta_el : float or TS
            electrical efficiency.
        Q_fu : cFlow
            fuel input-flow.
        P_el : cFlow
            electricity output-flow.
        Q_th : cFlow
            heat output-flow.
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        """
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
        Parameters
        ----------
        label : str
            name of heatpump.
        COP : float, TS
            Coefficient of performance.
        Q_ab : Flow
            Heatsource input-flow.
        P_el : Flow
            electricity input-flow.
        Q_th : Flow
            thermal output-flow.
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
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
    value: NumericDataTS, parameter_label: str, element_label: str, lower_bound: NumericDataTS, upper_bound: NumericDataTS
):
    """
    Check if the value is within the bounds. The bounds are exclusive.
    If not, log a warning.
    Parameters
    ----------
    value: NumericDataTS
        The value to check.
    parameter_label: str
        The label of the value.
    element_label: str
        The label of the element.
    lower_bound: NumericDataTS
        The lower bound.
    upper_bound: NumericDataTS
        The upper bound.

    Returns
    -------

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
