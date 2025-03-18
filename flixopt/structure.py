"""
This module contains the core structure of the flixopt framework.
These classes are not directly used by the end user, but are used by other modules.
"""

import inspect
import json
import logging
import pathlib
from datetime import datetime
from io import StringIO
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import numpy as np
from rich.console import Console
from rich.pretty import Pretty

from . import utils
from .config import CONFIG
from .core import Numeric, Numeric_TS, Skalar, TimeSeries, TimeSeriesData
from .math_modeling import Equation, Inequation, MathModel, Solver, Variable, VariableTS

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from .elements import BusModel, ComponentModel
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


class SystemModel(MathModel):
    """
    Hier kommen die ModellingLanguage-spezifischen Sachen rein
    """

    def __init__(
        self,
        label: str,
        modeling_language: Literal['pyomo', 'cvxpy'],
        flow_system: 'FlowSystem',
        time_indices: Optional[Union[List[int], range]],
    ):
        super().__init__(label, modeling_language)
        self.flow_system = flow_system
        # Zeitdaten generieren:
        self.time_series, self.time_series_with_end, self.dt_in_hours, self.dt_in_hours_total = (
            flow_system.get_time_data_from_indices(time_indices)
        )
        self.previous_dt_in_hours = flow_system.previous_dt_in_hours
        self.nr_of_time_steps = len(self.time_series)
        self.indices = range(self.nr_of_time_steps)

        self.effect_collection_model = flow_system.effect_collection.create_model(self)
        self.component_models: List['ComponentModel'] = []
        self.bus_models: List['BusModel'] = []
        self.other_models: List[ElementModel] = []

    def do_modeling(self):
        self.effect_collection_model.do_modeling(self)
        self.component_models = [component.create_model() for component in self.flow_system.components.values()]
        self.bus_models = [bus.create_model() for bus in self.flow_system.buses.values()]
        for component_model in self.component_models:
            component_model.do_modeling(self)
        for bus_model in self.bus_models:  # Buses after Components, because FlowModels are created in ComponentModels
            bus_model.do_modeling(self)

    def solve(self, solver: Solver, excess_threshold: Union[int, float] = 0.1):
        """
        Parameters
        ----------
        solver : Solver
            An Instance of the class Solver. Choose from flixopt.solvers
        excess_threshold : float, positive!
            threshold for excess: If sum(Excess)>excess_threshold a warning is raised, that an excess occurs
        """

        logger.info(f'{" starting solving ":#^80}')
        logger.info(f'{self.describe_size()}')

        super().solve(solver)

        logger.info(f'Termination message: "{self.solver.termination_message}"')

        logger.info(f'{" finished solving ":#^80}')
        logger.info(f'{" Main Results ":#^80}')
        for effect_name, effect_results in self.main_results['Effects'].items():
            logger.info(
                f'{effect_name}:\n'
                f'  {"operation":<15}: {effect_results["operation"]:>10.2f}\n'
                f'  {"invest":<15}: {effect_results["invest"]:>10.2f}\n'
                f'  {"sum":<15}: {effect_results["sum"]:>10.2f}'
            )

        logger.info(
            # f'{"SUM":<15}: ...todo...\n'
            f'{"Penalty":<17}: {self.main_results["penalty"]:>10.2f}\n'
            f'{"":-^80}\n'
            f'{"Objective":<17}: {self.main_results["Objective"]:>10.2f}\n'
            f'{"":-^80}'
        )

        logger.info('Investment Decisions:')
        logger.info(
            utils.apply_formating(
                data_dict={
                    **self.main_results['Invest-Decisions']['invested'],
                    **self.main_results['Invest-Decisions']['not invested'],
                },
                key_format='<30',
                indent=2,
                sort_by='value',
            )
        )

        for bus in self.main_results['buses with excess']:
            logger.warning(f'A penalty occurred in Bus "{bus}"!')

        if self.main_results['penalty'] > 10:
            logger.warning(f'A total penalty of {self.main_results["penalty"]} occurred.This might distort the results')
        logger.info(f'{" End of Main Results ":#^80}')

    def description_of_variables(self, structured: bool = True) -> Dict[str, Union[str, List[str]]]:
        return {
            'Components': {
                label: comp.model.description_of_variables(structured)
                for label, comp in self.flow_system.components.items()
            },
            'Buses': {
                label: bus.model.description_of_variables(structured) for label, bus in self.flow_system.buses.items()
            },
            'Effects': self.flow_system.effect_collection.model.description_of_variables(structured),
            'Others': {model.element.label: model.description_of_variables(structured) for model in self.other_models},
        }

    def description_of_constraints(self, structured: bool = True) -> Dict[str, Union[str, List[str]]]:
        return {
            'Components': {
                label: comp.model.description_of_constraints(structured)
                for label, comp in self.flow_system.components.items()
            },
            'Buses': {
                label: bus.model.description_of_constraints(structured) for label, bus in self.flow_system.buses.items()
            },
            'Objective': self.objective.description(),
            'Effects': self.flow_system.effect_collection.model.description_of_constraints(structured),
            'Others': {
                model.element.label: model.description_of_constraints(structured) for model in self.other_models
            },
        }

    def results(self):
        return {
            'Components': {model.element.label: model.results() for model in self.component_models},
            'Effects': self.effect_collection_model.results(),
            'Buses': {model.element.label: model.results() for model in self.bus_models},
            'Others': {model.element.label: model.results() for model in self.other_models},
            'Objective': self.result_of_objective,
            'Time': self.time_series_with_end,
            'Time intervals in hours': self.dt_in_hours,
        }

    @property
    def main_results(self) -> Dict[str, Union[Skalar, Dict]]:
        main_results = {}
        effect_results = {}
        main_results['Effects'] = effect_results
        for effect in self.flow_system.effect_collection.effects.values():
            effect_results[f'{effect.label} [{effect.unit}]'] = {
                'operation': float(effect.model.operation.sum.result),
                'invest': float(effect.model.invest.sum.result),
                'sum': float(effect.model.all.sum.result),
            }
        main_results['penalty'] = float(self.effect_collection_model.penalty.sum.result)
        main_results['Objective'] = self.result_of_objective
        main_results['lower bound'] = self.solver.best_bound
        buses_with_excess = []
        main_results['buses with excess'] = buses_with_excess
        for bus in self.flow_system.buses.values():
            if bus.with_excess:
                if np.sum(bus.model.excess_input.result) > 1e-3 or np.sum(bus.model.excess_output.result) > 1e-3:
                    buses_with_excess.append(bus.label)

        invest_decisions = {'invested': {}, 'not invested': {}}
        main_results['Invest-Decisions'] = invest_decisions
        from flixopt.features import InvestmentModel

        for sub_model in self.sub_models:
            if isinstance(sub_model, InvestmentModel):
                invested_size = float(sub_model.size.result)  # bei np.floats Probleme bei Speichern
                if invested_size > 1e-3:
                    invest_decisions['invested'][sub_model.element.label_full] = invested_size
                else:
                    invest_decisions['not invested'][sub_model.element.label_full] = invested_size

        return main_results

    @property
    def infos(self) -> Dict:
        infos = super().infos
        infos['Constraints'] = self.description_of_constraints()
        infos['Variables'] = self.description_of_variables()
        infos['Main Results'] = self.main_results
        infos['Config'] = CONFIG.to_dict()
        return infos

    @property
    def all_variables(self) -> Dict[str, Variable]:
        all_vars = {}
        for model in self.sub_models:
            for label, variable in model.variables.items():
                if label in all_vars:
                    raise KeyError(f'Duplicate Variable found in SystemModel:{model=} {label=}; {variable=}')
                all_vars[label] = variable
        return all_vars

    @property
    def all_constraints(self) -> Dict[str, Union[Equation, Inequation]]:
        all_constr = {}
        for model in self.sub_models:
            for label, constr in model.constraints.items():
                if label in all_constr:
                    raise KeyError(f'Duplicate Constraint found in SystemModel: {label=}; {constr=}')
                else:
                    all_constr[label] = constr
        return all_constr

    @property
    def all_equations(self) -> Dict[str, Equation]:
        return {key: value for key, value in self.all_constraints.items() if isinstance(value, Equation)}

    @property
    def all_inequations(self) -> Dict[str, Inequation]:
        return {key: value for key, value in self.all_constraints.items() if isinstance(value, Inequation)}

    @property
    def sub_models(self) -> List['ElementModel']:
        direct_models = [self.effect_collection_model] + self.component_models + self.bus_models + self.other_models
        sub_models = [sub_model for direct_model in direct_models for sub_model in direct_model.all_sub_models]
        return direct_models + sub_models

    @property
    def variables(self) -> List[Variable]:
        """Needed for Mother class"""
        return list(self.all_variables.values())

    @property
    def equations(self) -> List[Equation]:
        """Needed for Mother class"""
        return list(self.all_equations.values())

    @property
    def inequations(self) -> List[Inequation]:
        """Needed for Mother class"""
        return list(self.all_inequations.values())

    @property
    def objective(self) -> Equation:
        return self.effect_collection_model.objective


class Interface:
    """
    This class is used to collect arguments about a Model.
    """

    def transform_data(self):
        raise NotImplementedError('Every Interface needs a transform_data() method')

    def infos(self, use_numpy=True, use_element_label=False) -> Dict:
        """
        Generate a dictionary representation of the object's constructor arguments.
        Excludes default values and empty dictionaries and lists.
        Converts data to be compatible with JSON.

        Parameters:
        -----------
        use_numpy bool:
            Whether to convert NumPy arrays to lists. Defaults to True.
            If True, numeric numpy arrays (`np.ndarray`) are preserved as-is.
            If False, they are converted to lists.
        use_element_label bool:
            Whether to use the element label instead of the infos of the element. Defaults to False.
            Note that Elements used as keys in dictionaries are always converted to their labels.

        Returns:
            Dict: A dictionary representation of the object's constructor arguments.

        """
        # Get the constructor arguments and their default values
        init_params = sorted(
            inspect.signature(self.__init__).parameters.items(),
            key=lambda x: (x[0].lower() != 'label', x[0].lower()),  # Prioritize 'label'
        )
        # Build a dict of attribute=value pairs, excluding defaults
        details = {'class': ':'.join([cls.__name__ for cls in self.__class__.__mro__])}
        for name, param in init_params:
            if name == 'self':
                continue
            value, default = getattr(self, name, None), param.default
            # Ignore default values and empty dicts and list
            if np.all(value == default) or (isinstance(value, (dict, list)) and not value):
                continue
            details[name] = copy_and_convert_datatypes(value, use_numpy, use_element_label)
        return details

    def to_json(self, path: Union[str, pathlib.Path]):
        """
        Saves the element to a json file.
        This not meant to be reloaded and recreate the object, but rather used to document or compare the object.

        Parameters:
        -----------
        path : Union[str, pathlib.Path]
            The path to the json file.
        """
        data = get_compact_representation(self.infos(use_numpy=True, use_element_label=True))
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def __repr__(self):
        # Get the constructor arguments and their current values
        init_signature = inspect.signature(self.__init__)
        init_args = init_signature.parameters

        # Create a dictionary with argument names and their values
        args_str = ', '.join(f'{name}={repr(getattr(self, name, None))}' for name in init_args if name != 'self')
        return f'{self.__class__.__name__}({args_str})'

    def __str__(self):
        return get_str_representation(self.infos(use_numpy=True, use_element_label=True))


class Element(Interface):
    """Basic Element of flixopt"""

    def __init__(self, label: str, meta_data: Dict = None):
        """
        Parameters
        ----------
        label : str
            label of the element
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        """
        if not utils.label_is_valid(label):
            logger.critical(
                f"'{label}' cannot be used as a label. Leading or Trailing '_' and '__' are reserved. "
                f'Use any other symbol instead'
            )
        self.label = label
        self.meta_data = meta_data if meta_data is not None else {}
        self.used_time_series: List[TimeSeries] = []  # Used for better access
        self.model: Optional[ElementModel] = None

    def _plausibility_checks(self) -> None:
        """This function is used to do some basic plausibility checks for each Element during initialization"""
        raise NotImplementedError('Every Element needs a _plausibility_checks() method')

    def create_model(self) -> None:
        raise NotImplementedError('Every Element needs a create_model() method')

    @property
    def label_full(self) -> str:
        return self.label


class ElementModel:
    """Interface to create the mathematical Models for Elements"""

    def __init__(self, element: Element, label: Optional[str] = None):
        logger.debug(f'Created {self.__class__.__name__} for {element.label_full}')
        self.element = element
        self.variables = {}
        self.constraints = {}
        self.sub_models = []
        self._label = label

    def add_variables(self, *variables: Variable) -> None:
        for variable in variables:
            if variable.label not in self.variables.keys():
                self.variables[variable.label] = variable
            elif variable in self.variables.values():
                raise Exception(f'Variable "{variable.label}" already exists')
            else:
                raise Exception(f'A Variable with the label "{variable.label}" already exists')

    def add_constraints(self, *constraints: Union[Equation, Inequation]) -> None:
        for constraint in constraints:
            if constraint.label not in self.constraints.keys():
                self.constraints[constraint.label] = constraint
            else:
                raise Exception(f'Constraint "{constraint.label}" already exists')

    def description_of_variables(self, structured: bool = True) -> Union[Dict[str, Union[List[str], Dict]], List[str]]:
        if structured:
            # Gather descriptions of this model's variables
            descriptions = {'_self': [var.description() for var in self.variables.values()]}

            # Recursively gather descriptions from sub-models
            for sub_model in self.sub_models:
                descriptions[sub_model.label] = sub_model.description_of_variables(structured=structured)

            return descriptions
        else:
            return [var.description() for var in self.all_variables.values()]

    def description_of_constraints(self, structured: bool = True) -> Union[Dict[str, str], List[str]]:
        if structured:
            # Gather descriptions of this model's variables
            descriptions = {'_self': [constr.description() for constr in self.constraints.values()]}

            # Recursively gather descriptions from sub-models
            for sub_model in self.sub_models:
                descriptions[sub_model.label] = sub_model.description_of_constraints(structured=structured)

            return descriptions
        else:
            return [eq.description() for eq in self.all_equations.values()]

    @property
    def overview_of_model_size(self) -> Dict[str, int]:
        all_vars, all_eqs, all_ineqs = self.all_variables, self.all_equations, self.all_inequations
        return {
            'no of Euations': len(all_eqs),
            'no of Equations single': sum(eq.nr_of_single_equations for eq in all_eqs.values()),
            'no of Inequations': len(all_ineqs),
            'no of Inequations single': sum(ineq.nr_of_single_equations for ineq in all_ineqs.values()),
            'no of Variables': len(all_vars),
            'no of Variables single': sum(var.length for var in all_vars.values()),
        }

    @property
    def inequations(self) -> Dict[str, Inequation]:
        return {name: ineq for name, ineq in self.constraints.items() if isinstance(ineq, Inequation)}

    @property
    def equations(self) -> Dict[str, Equation]:
        return {name: eq for name, eq in self.constraints.items() if isinstance(eq, Equation)}

    @property
    def all_variables(self) -> Dict[str, Variable]:
        all_vars = self.variables.copy()
        for sub_model in self.sub_models:
            for key, value in sub_model.all_variables.items():
                if key in all_vars:
                    raise KeyError(f"Duplicate key found: '{key}' in both main model and submodel!")
                all_vars[key] = value
        return all_vars

    @property
    def all_constraints(self) -> Dict[str, Union[Equation, Inequation]]:
        all_constr = self.constraints.copy()
        for sub_model in self.sub_models:
            for key, value in sub_model.all_constraints.items():
                if key in all_constr:
                    raise KeyError(f"Duplicate key found: '{key}' in both main model and submodel!")
                all_constr[key] = value
        return all_constr

    @property
    def all_equations(self) -> Dict[str, Equation]:
        all_eqs = self.equations.copy()
        for sub_model in self.sub_models:
            for key, value in sub_model.all_equations.items():
                if key in all_eqs:
                    raise KeyError(f"Duplicate key found: '{key}' in both main model and submodel!")
                all_eqs[key] = value
        return all_eqs

    @property
    def all_inequations(self) -> Dict[str, Inequation]:
        all_ineqs = self.inequations.copy()
        for sub_model in self.sub_models:
            for key in sub_model.all_inequations:
                if key in all_ineqs:
                    raise KeyError(f"Duplicate key found: '{key}' in both main model and submodel!")
        return all_ineqs

    @property
    def all_sub_models(self) -> List['ElementModel']:
        all_subs = []
        to_process = self.sub_models.copy()
        for model in to_process:
            all_subs.append(model)
            to_process.extend(model.sub_models)
        return all_subs

    def results(self) -> Dict:
        return {
            **{variable.label_short: variable.result for variable in self.variables.values()},
            **{model.label: model.results() for model in self.sub_models},
        }

    @property
    def label_full(self) -> str:
        return f'{self.element.label_full}__{self._label}' if self._label else self.element.label_full

    @property
    def label(self):
        return self._label or self.element.label


def _create_time_series(
    label: str, data: Optional[Union[Numeric_TS, TimeSeries]], element: Element
) -> Optional[TimeSeries]:
    """Creates a TimeSeries from Numeric Data and adds it to the list of time_series of an Element.
    If the data already is a TimeSeries, nothing happens and the TimeSeries gets cleaned and returned"""
    if data is None:
        return None
    elif isinstance(data, TimeSeries):
        data.clear_indices_and_aggregated_data()
        return data
    else:
        time_series = TimeSeries(label=f'{element.label_full}__{label}', data=data)
        element.used_time_series.append(time_series)
        return time_series


def create_equation(
    label: str, element_model: ElementModel, eq_type: Literal['eq', 'ineq'] = 'eq'
) -> Union[Equation, Inequation]:
    """Creates an Equation and adds it to the model of the Element"""
    if eq_type == 'eq':
        constr = Equation(f'{element_model.label_full}_{label}', label)
    elif eq_type == 'ineq':
        constr = Inequation(f'{element_model.label_full}_{label}', label)
    element_model.add_constraints(constr)
    return constr


def create_variable(
    label: str,
    element_model: ElementModel,
    length: int,
    is_binary: bool = False,
    fixed_value: Optional[Numeric] = None,
    lower_bound: Optional[Numeric] = None,
    upper_bound: Optional[Numeric] = None,
    previous_values: Optional[Numeric] = None,
    avoid_use_of_variable_ts: bool = False,
) -> VariableTS:
    """Creates a VariableTS and adds it to the model of the Element"""
    variable_label = f'{element_model.label_full}_{label}'
    if length > 1 and not avoid_use_of_variable_ts:
        var = VariableTS(
            variable_label, length, label, is_binary, fixed_value, lower_bound, upper_bound, previous_values
        )
        logger.debug(f'Created VariableTS "{variable_label}": [{length}]')
    else:
        var = Variable(variable_label, length, label, is_binary, fixed_value, lower_bound, upper_bound)
        logger.debug(f'Created Variable "{variable_label}": [{length}]')
    element_model.add_variables(var)
    return var


def copy_and_convert_datatypes(data: Any, use_numpy: bool = True, use_element_label: bool = False) -> Any:
    """
    Converts values in a nested data structure into JSON-compatible types while preserving or transforming numpy arrays
    and custom `Element` objects based on the specified options.

    The function handles various data types and transforms them into a consistent, readable format:
    - Primitive types (`int`, `float`, `str`, `bool`, `None`) are returned as-is.
    - Numpy scalars are converted to their corresponding Python scalar types.
    - Collections (`list`, `tuple`, `set`, `dict`) are recursively processed to ensure all elements are compatible.
    - Numpy arrays are preserved or converted to lists, depending on `use_numpy`.
    - Custom `Element` objects can be represented either by their `label` or their initialization parameters as a dictionary.
    - Timestamps (`datetime`) are converted to ISO 8601 strings.

    Parameters
    ----------
    data : Any
        The input data to process, which may be deeply nested and contain a mix of types.
    use_numpy : bool, optional
        If `True`, numeric numpy arrays (`np.ndarray`) are preserved as-is. If `False`, they are converted to lists.
        Default is `True`.
    use_element_label : bool, optional
        If `True`, `Element` objects are represented by their `label`. If `False`, they are converted into a dictionary
        based on their initialization parameters. Default is `False`.

    Returns
    -------
    Any
        A transformed version of the input data, containing only JSON-compatible types:
        - `int`, `float`, `str`, `bool`, `None`
        - `list`, `dict`
        - `np.ndarray` (if `use_numpy=True`. This is NOT JSON-compatible)

    Raises
    ------
    TypeError
        If the data cannot be converted to the specified types.

    Examples
    --------
    >>> copy_and_convert_datatypes({'a': np.array([1, 2, 3]), 'b': Element(label='example')})
    {'a': array([1, 2, 3]), 'b': {'class': 'Element', 'label': 'example'}}

    >>> copy_and_convert_datatypes({'a': np.array([1, 2, 3]), 'b': Element(label='example')}, use_numpy=False)
    {'a': [1, 2, 3], 'b': {'class': 'Element', 'label': 'example'}}

    Notes
    -----
    - The function gracefully handles unexpected types by issuing a warning and returning a deep copy of the data.
    - Empty collections (lists, dictionaries) and default parameter values in `Element` objects are omitted from the output.
    - Numpy arrays with non-numeric data types are automatically converted to lists.
    """
    if isinstance(data, np.integer):  # This must be checked before checking for regular int and float!
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)

    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    elif isinstance(data, datetime):
        return data.isoformat()

    elif isinstance(data, (tuple, set)):
        return copy_and_convert_datatypes([item for item in data], use_numpy, use_element_label)
    elif isinstance(data, dict):
        return {
            copy_and_convert_datatypes(key, use_numpy, use_element_label=True): copy_and_convert_datatypes(
                value, use_numpy, use_element_label
            )
            for key, value in data.items()
        }
    elif isinstance(data, list):  # Shorten arrays/lists to be readable
        if use_numpy and all([isinstance(value, (int, float)) for value in data]):
            return np.array([item for item in data])
        else:
            return [copy_and_convert_datatypes(item, use_numpy, use_element_label) for item in data]

    elif isinstance(data, np.ndarray):
        if not use_numpy:
            return copy_and_convert_datatypes(data.tolist(), use_numpy, use_element_label)
        elif use_numpy and np.issubdtype(data.dtype, np.number):
            return data
        else:
            logger.critical(
                f'An np.array with non-numeric content was found: {data=}.It will be converted to a list instead'
            )
            return copy_and_convert_datatypes(data.tolist(), use_numpy, use_element_label)

    elif isinstance(data, TimeSeries):
        return copy_and_convert_datatypes(data.active_data, use_numpy, use_element_label)
    elif isinstance(data, TimeSeriesData):
        return copy_and_convert_datatypes(data.data, use_numpy, use_element_label)

    elif isinstance(data, Interface):
        if use_element_label and isinstance(data, Element):
            return data.label
        return data.infos(use_numpy, use_element_label)
    else:
        raise TypeError(f'copy_and_convert_datatypes() did get unexpected data of type "{type(data)}": {data=}')


def get_compact_representation(data: Any, array_threshold: int = 50, decimals: int = 2) -> Dict:
    """
    Generate a compact json serializable representation of deeply nested data.
    Numpy arrays are statistically described if they exceed a threshold and converted to lists.

    Args:
        data (Any): The data to format and represent.
        array_threshold (int): Maximum length of NumPy arrays to display. Longer arrays are statistically described.
        decimals (int): Number of decimal places in which to describe the arrays.

    Returns:
        Dict: A dictionary representation of the data
    """

    def format_np_array_if_found(value: Any) -> Any:
        """Recursively processes the data, formatting NumPy arrays."""
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        elif isinstance(value, np.ndarray):
            return describe_numpy_arrays(value)
        elif isinstance(value, dict):
            return {format_np_array_if_found(k): format_np_array_if_found(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple, set)):
            return [format_np_array_if_found(v) for v in value]
        else:
            logger.warning(
                f'Unexpected value found when trying to format numpy array numpy array: {type(value)=}; {value=}'
            )
            return value

    def describe_numpy_arrays(arr: np.ndarray) -> Union[str, List]:
        """Shortens NumPy arrays if they exceed the specified length."""

        def normalized_center_of_mass(array: Any) -> float:
            # position in array (0 bis 1 normiert)
            positions = np.linspace(0, 1, len(array))  # weights w_i
            # mass center
            if np.sum(array) == 0:
                return np.nan
            else:
                return np.sum(positions * array) / np.sum(array)

        if arr.size > array_threshold:  # Calculate basic statistics
            fmt = f'.{decimals}f'
            return (
                f'Array (min={np.min(arr):{fmt}}, max={np.max(arr):{fmt}}, mean={np.mean(arr):{fmt}}, '
                f'median={np.median(arr):{fmt}}, std={np.std(arr):{fmt}}, len={len(arr)}, '
                f'center={normalized_center_of_mass(arr):{fmt}})'
            )
        else:
            return np.around(arr, decimals=decimals).tolist()

    # Process the data to handle NumPy arrays
    formatted_data = format_np_array_if_found(copy_and_convert_datatypes(data, use_numpy=True))

    return formatted_data


def get_str_representation(data: Any, array_threshold: int = 50, decimals: int = 2) -> str:
    """
    Generate a string representation of deeply nested data using `rich.print`.
    NumPy arrays are shortened to the specified length and converted to strings.

    Args:
        data (Any): The data to format and represent.
        array_threshold (int): Maximum length of NumPy arrays to display. Longer arrays are statistically described.
        decimals (int): Number of decimal places in which to describe the arrays.

    Returns:
        str: The formatted string representation of the data.
    """

    formatted_data = get_compact_representation(data, array_threshold, decimals)

    # Use Rich to format and print the data
    with StringIO() as output_buffer:
        console = Console(file=output_buffer, width=1000)  # Adjust width as needed
        console.print(Pretty(formatted_data, expand_all=True, indent_guides=True))
        return output_buffer.getvalue()
