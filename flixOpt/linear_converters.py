"""
This Module contains high-level classes to easily model a FlowSystem.
"""

import logging
from typing import Dict, Optional

import numpy as np

from .components import LinearConverter
from .core import Numeric_TS, TimeSeriesData
from .elements import Flow, MediumCategories
from .interface import OnOffParameters

logger = logging.getLogger('flixOpt')


class Boiler(LinearConverter):
    def __init__(
        self,
        label: str,
        eta: Numeric_TS,
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
            conversion_factors=[{Q_fu: eta, Q_th: 1}],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )

        self.eta = eta
        self.Q_fu = Q_fu
        self.Q_th = Q_th

        assign_medium_category(self.Q_fu, MediumCategories.fuel)
        assign_medium_category(self.Q_th, MediumCategories.heat)
        check_bounds(eta, 'eta', self.label_full, 0, 1)


class Power2Heat(LinearConverter):
    def __init__(
        self,
        label: str,
        eta: Numeric_TS,
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
            conversion_factors=[{P_el: eta, Q_th: 1}],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )

        self.eta = eta
        self.P_el = P_el
        self.Q_th = Q_th

        assign_medium_category(self.P_el, MediumCategories.electricity)
        assign_medium_category(self.Q_th, MediumCategories.heat)
        check_bounds(eta, 'eta', self.label_full, 0, 1)


class HeatPump(LinearConverter):
    def __init__(
        self,
        label: str,
        COP: Numeric_TS,
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
            conversion_factors=[{P_el: COP, Q_th: 1}],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )

        self.COP = COP
        self.P_el = P_el
        self.Q_th = Q_th

        assign_medium_category(self.P_el, MediumCategories.electricity)
        assign_medium_category(self.Q_th, MediumCategories.heat)
        check_bounds(COP, 'COP', self.label_full, 1, 20)


class CoolingTower(LinearConverter):
    def __init__(
        self,
        label: str,
        specific_electricity_demand: Numeric_TS,
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
            conversion_factors=[{P_el: 1, Q_th: -specific_electricity_demand}],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )

        self.specific_electricity_demand = specific_electricity_demand
        self.P_el = P_el
        self.Q_th = Q_th

        assign_medium_category(self.P_el, MediumCategories.electricity)
        assign_medium_category(self.Q_th, MediumCategories.heat)
        check_bounds(specific_electricity_demand, 'specific_electricity_demand', self.label_full, 0, 1)


class CHP(LinearConverter):
    def __init__(
        self,
        label: str,
        eta_th: Numeric_TS,
        eta_el: Numeric_TS,
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
        heat = {Q_fu: eta_th, Q_th: 1}
        electricity = {Q_fu: eta_el, P_el: 1}

        super().__init__(
            label,
            inputs=[Q_fu],
            outputs=[Q_th, P_el],
            conversion_factors=[heat, electricity],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )

        # args to attributes:
        self.eta_th = eta_th
        self.eta_el = eta_el
        self.Q_fu = Q_fu
        self.P_el = P_el
        self.Q_th = Q_th

        assign_medium_category(self.P_el, MediumCategories.electricity)
        assign_medium_category(self.Q_th, MediumCategories.heat)
        assign_medium_category(self.Q_fu, MediumCategories.fuel)
        check_bounds(eta_th, 'eta_th', self.label_full, 0, 1)
        check_bounds(eta_el, 'eta_el', self.label_full, 0, 1)
        check_bounds(eta_el + eta_th, 'eta_th+eta_el', self.label_full, 0, 1)


class HeatPumpWithSource(LinearConverter):
    def __init__(
        self,
        label: str,
        COP: Numeric_TS,
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
        electricity = {P_el: COP, Q_th: 1}
        heat_source = {Q_ab: COP / (COP - 1), Q_th: 1}

        super().__init__(
            label,
            inputs=[P_el, Q_ab],
            outputs=[Q_th],
            conversion_factors=[electricity, heat_source],
            on_off_parameters=on_off_parameters,
            meta_data=meta_data,
        )

        self.COP = COP
        self.P_el = P_el
        self.Q_ab = Q_ab
        self.Q_th = Q_th

        assign_medium_category(self.P_el, MediumCategories.electricity)
        assign_medium_category(self.Q_th, MediumCategories.heat)
        assign_medium_category(self.Q_ab, MediumCategories.heat)  # TODO: Check if this is necessary
        check_bounds(COP, 'eta_th', self.label_full, 1, 20)


def check_bounds(
    value: Numeric_TS, parameter_label: str, element_label: str, lower_bound: Numeric_TS, upper_bound: Numeric_TS
):
    """
    Check if the value is within the bounds. The bounds are exclusive.
    If not, log a warning.
    Parameters
    ----------
    value: Numeric_TS
        The value to check.
    parameter_label: str
        The label of the value.
    element_label: str
        The label of the element.
    lower_bound: Numeric_TS
        The lower bound.
    upper_bound: Numeric_TS
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


def assign_medium_category(flow: Flow, medium_category: str) -> None:
    """
    Assigns a medium category to a flow.
    If the flow already has a category assigned, a warning is raised.

    Parameters
    ----------
    flow: Flow
    medium_category: str
        The medium category to assign to the flow.

    Returns
    -------
    None
    """
    if flow.medium is not None:
        logger.warning(
            f'Flow {flow.label} already has a medium category assigned ({flow.medium}). '
            f'The new medium category {medium_category} will be ignored.'
        )
    else:
        flow.medium = medium_category
        logger.debug(f'Automatically assigned {medium_category=} to flow {flow.label_full}.')
