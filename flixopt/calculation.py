"""
This module contains the Calculation functionality for the flixopt framework.
It is used to calculate a SystemModel for a given FlowSystem through a solver.
There are three different Calculation types:
    1. FullCalculation: Calculates the SystemModel for the full FlowSystem
    2. AggregatedCalculation: Calculates the SystemModel for the full FlowSystem, but aggregates the TimeSeriesData.
        This simplifies the mathematical model and usually speeds up the solving process.
    3. SegmentedCalculation: Solves a SystemModel for each individual Segment of the FlowSystem.
"""

import datetime
import json
import logging
import math
import pathlib
import timeit
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import yaml

from . import utils as utils
from .aggregation import AggregationModel, AggregationParameters, TimeSeriesCollection
from .components import Storage
from .core import Numeric, Skalar
from .elements import Component
from .features import InvestmentModel
from .flow_system import FlowSystem
from .solvers import Solver
from .structure import SystemModel, copy_and_convert_datatypes, get_compact_representation

logger = logging.getLogger('flixopt')


class Calculation:
    """
    class for defined way of solving a flow_system optimization
    """

    def __init__(
        self,
        name,
        flow_system: FlowSystem,
        modeling_language: Literal['pyomo', 'cvxpy'] = 'pyomo',
        time_indices: Optional[Union[range, List[int]]] = None,
    ):
        """
        Parameters
        ----------
        name : str
            name of calculation
        flow_system : FlowSystem
            flow_system which should be calculated
        modeling_language : 'pyomo','cvxpy' (not implemeted yet)
            choose optimization modeling language
        time_indices : List[int] or None
            list with indices, which should be used for calculation. If None, then all timesteps are used.
        """
        self.name = name
        self.flow_system = flow_system
        self.modeling_language = modeling_language
        self.time_indices = time_indices

        self.system_model: Optional[SystemModel] = None
        self.durations = {'modeling': 0.0, 'solving': 0.0, 'saving': 0.0}  # Dauer der einzelnen Dinge

        self._paths: Dict[str, Optional[Union[pathlib.Path, List[pathlib.Path]]]] = {
            'log': None,
            'data': None,
            'info': None,
        }
        self._results = None

    def _define_path_names(self, save_results: Union[bool, str, pathlib.Path], include_timestamp: bool = False):
        """
        Creates the path for saving results and alters the name of the calculation to have a timestamp
        """
        if include_timestamp:
            timestamp = datetime.datetime.now()
            self.name = f'{timestamp.strftime("%Y-%m-%d")}_{self.name.replace(" ", "")}'

        if save_results:
            if not isinstance(save_results, (str, pathlib.Path)):
                save_results = 'results/'  # Standard path for results
            path = pathlib.Path.cwd() / save_results  # absoluter Pfad:

            path.mkdir(parents=True, exist_ok=True)  # Pfad anlegen, fall noch nicht vorhanden:

            self._paths['log'] = path / f'{self.name}_solver.log'
            self._paths['data'] = path / f'{self.name}_data.json'
            self._paths['results'] = path / f'{self.name}_results.json'
            self._paths['infos'] = path / f'{self.name}_infos.yaml'

    def _save_solve_infos(self):
        t_start = timeit.default_timer()
        indent = 4 if len(self.flow_system.time_series) < 50 else None
        with open(self._paths['results'], 'w', encoding='utf-8') as f:
            results = copy_and_convert_datatypes(self.results(), use_numpy=False, use_element_label=False)
            json.dump(results, f, indent=indent)

        with open(self._paths['data'], 'w', encoding='utf-8') as f:
            data = copy_and_convert_datatypes(self.flow_system.infos(), use_numpy=False, use_element_label=False)
            json.dump(data, f, indent=indent)

        self.durations['saving'] = round(timeit.default_timer() - t_start, 2)

        t_start = timeit.default_timer()
        nodes_info, edges_info = self.flow_system.network_infos()
        infos = {
            'Calculation': self.infos,
            'Model': self.system_model.infos,
            'FlowSystem': get_compact_representation(self.flow_system.infos(use_numpy=True, use_element_label=True)),
            'Network': {'Nodes': nodes_info, 'Edges': edges_info},
        }

        with open(self._paths['infos'], 'w', encoding='utf-8') as f:
            yaml.dump(
                infos,
                f,
                width=1000,  # Verhinderung Zeilenumbruch für lange equations
                allow_unicode=True,
                sort_keys=False,
            )

        message = f' Saved Calculation: {self.name} '
        logger.info(f'{"":#^80}\n{message:#^80}\n{"":#^80}')
        logger.info(f'Saving calculation to .json took {self.durations["saving"]:>8.2f} seconds')
        logger.info(f'Saving calculation to .yaml took {(timeit.default_timer() - t_start):>8.2f} seconds')

    def results(self):
        if self._results is None:
            self._results = self.system_model.results()
        return self._results

    @property
    def infos(self):
        return {
            'Name': self.name,
            'Number of indices': len(self.time_indices) if self.time_indices else 'all',
            'Calculation Type': self.__class__.__name__,
            'Durations': self.durations,
        }


class FullCalculation(Calculation):
    """
    class for defined way of solving a flow_system optimization
    """

    def do_modeling(self) -> SystemModel:
        t_start = timeit.default_timer()

        self.flow_system.transform_data()
        for time_series in self.flow_system.all_time_series:
            time_series.activate_indices(self.time_indices)

        self.system_model = SystemModel(self.name, self.modeling_language, self.flow_system, self.time_indices)
        self.system_model.do_modeling()
        self.system_model.translate_to_modeling_language()

        self.durations['modeling'] = round(timeit.default_timer() - t_start, 2)
        return self.system_model

    def solve(self, solver: Solver, save_results: Union[bool, str, pathlib.Path] = False):
        self._define_path_names(save_results)
        t_start = timeit.default_timer()
        solver.logfile_name = self._paths['log']
        self.system_model.solve(solver)
        self.durations['solving'] = round(timeit.default_timer() - t_start, 2)

        if save_results:
            self._save_solve_infos()


class AggregatedCalculation(Calculation):
    """
    class for defined way of solving a flow_system optimization
    """

    def __init__(
        self,
        name,
        flow_system: FlowSystem,
        aggregation_parameters: AggregationParameters,
        components_to_clusterize: Optional[List[Component]] = None,
        modeling_language: Literal['pyomo', 'cvxpy'] = 'pyomo',
        time_indices: Optional[Union[range, List[int]]] = None,
    ):
        """
        Class for Optimizing the FLowSystem including:
            1. Aggregating TimeSeriesData via typical periods using tsam.
            2. Equalizing variables of typical periods.
        Parameters
        ----------
        name : str
            name of calculation
        aggregation_parameters : AggregationParameters
            Parameters for aggregation. See documentation of AggregationParameters class.
        components_to_clusterize: List[Component] or None
            List of Components to perform aggregation on. If None, then all components are aggregated.
            This means, teh variables in the components are equalized to each other, according to the typical periods
            computed in the DataAggregation
        flow_system : FlowSystem
            flow_system which should be calculated
        modeling_language : 'pyomo','cvxpy' (not implemeted yet)
            choose optimization modeling language
        time_indices : List[int] or None
            list with indices, which should be used for calculation. If None, then all timesteps are used.
        """
        super().__init__(name, flow_system, modeling_language, time_indices)
        self.aggregation_parameters = aggregation_parameters
        self.components_to_clusterize = components_to_clusterize
        self.time_series_for_aggregation = None
        self.aggregation = None
        self.time_series_collection: Optional[TimeSeriesCollection] = None

    def do_modeling(self) -> SystemModel:
        self.flow_system.transform_data()
        for time_series in self.flow_system.all_time_series:
            time_series.activate_indices(self.time_indices)

        from .aggregation import Aggregation

        (chosen_time_series, chosen_time_series_with_end, dt_in_hours, dt_in_hours_total) = (
            self.flow_system.get_time_data_from_indices(self.time_indices)
        )

        t_start_agg = timeit.default_timer()

        # Validation
        dt_min, dt_max = np.min(dt_in_hours), np.max(dt_in_hours)
        if not dt_min == dt_max:
            raise ValueError(
                f'Aggregation failed due to inconsistent time step sizes:'
                f'delta_t varies from {dt_min} to {dt_max} hours.'
            )
        steps_per_period = self.aggregation_parameters.hours_per_period / dt_in_hours[0]
        if not steps_per_period.is_integer():
            raise Exception(
                f'The selected {self.aggregation_parameters.hours_per_period=} does not match the time '
                f'step size of {dt_in_hours[0]} hours). It must be a multiple of {dt_in_hours[0]} hours.'
            )

        logger.info(f'{"":#^80}')
        logger.info(f'{" Aggregating TimeSeries Data ":#^80}')

        self.time_series_collection = TimeSeriesCollection(
            [ts for ts in self.flow_system.all_time_series if ts.is_array]
        )

        import pandas as pd

        original_data = pd.DataFrame(self.time_series_collection.data, index=chosen_time_series)

        # Aggregation - creation of aggregated timeseries:
        self.aggregation = Aggregation(
            original_data=original_data,
            hours_per_time_step=dt_min,
            hours_per_period=self.aggregation_parameters.hours_per_period,
            nr_of_periods=self.aggregation_parameters.nr_of_periods,
            weights=self.time_series_collection.weights,
            time_series_for_high_peaks=self.aggregation_parameters.labels_for_high_peaks,
            time_series_for_low_peaks=self.aggregation_parameters.labels_for_low_peaks,
        )

        self.aggregation.cluster()
        self.aggregation.plot()
        if self.aggregation_parameters.aggregate_data_and_fix_non_binary_vars:
            self.time_series_collection.insert_data(  # Converting it into a dict with labels as keys
                {
                    col: np.array(values)
                    for col, values in self.aggregation.aggregated_data.to_dict(orient='list').items()
                }
            )
        self.durations['aggregation'] = round(timeit.default_timer() - t_start_agg, 2)

        # Model the System
        t_start = timeit.default_timer()

        self.system_model = SystemModel(self.name, self.modeling_language, self.flow_system, self.time_indices)
        self.system_model.do_modeling()
        # Add Aggregation Model after modeling the rest
        aggregation_model = AggregationModel(
            self.aggregation_parameters, self.flow_system, self.aggregation, self.components_to_clusterize
        )
        self.system_model.other_models.append(aggregation_model)
        aggregation_model.do_modeling(self.system_model)

        self.system_model.translate_to_modeling_language()

        self.durations['modeling'] = round(timeit.default_timer() - t_start, 2)
        return self.system_model

    def solve(self, solver: Solver, save_results: Union[bool, str, pathlib.Path] = False):
        self._define_path_names(save_results)
        t_start = timeit.default_timer()
        solver.logfile_name = self._paths['log']
        self.system_model.solve(solver)
        self.durations['solving'] = round(timeit.default_timer() - t_start, 2)

        if save_results:
            self._save_solve_infos()


class SegmentedCalculation(Calculation):
    def __init__(
        self,
        name,
        flow_system: FlowSystem,
        segment_length: int,
        overlap_length: int,
        modeling_language: Literal['pyomo', 'cvxpy'] = 'pyomo',
        time_indices: Optional[Union[range, list[int]]] = None,
    ):
        """
        Dividing and Modeling the problem in (overlapping) segments.
        The final values of each Segment are recognized by the following segment, effectively coupling
        charge_states and flow_rates between segments.
        Because of this intersection, both modeling and solving is done in one step

        Take care:
        Parameters like InvestParameters, sum_of_flow_hours and other restrictions over the total time_series
        don't really work in this Calculation. Lower bounds to such SUMS can lead to weird results.
        This is NOT yet explicitly checked for...

        Parameters
        ----------
        name : str
            name of calculation
        flow_system : FlowSystem
            flow_system which should be calculated
        segment_length : int
            The number of time_steps per individual segment (without the overlap)
        overlap_length : int
            The number of time_steps that are added to each individual model. Used for better
            results of storages)
        modeling_language : 'pyomo', 'cvxpy' (not implemeted yet)
            choose optimization modeling language
        time_indices : List[int] or None
            list with indices, which should be used for calculation. If None, then all timesteps are used.

        """
        super().__init__(name, flow_system, modeling_language, time_indices)
        self.segment_length = segment_length
        self.overlap_length = overlap_length
        self._total_length = len(self.time_indices) if self.time_indices is not None else len(flow_system.time_series)
        self.number_of_segments = math.ceil(self._total_length / self.segment_length)
        self.sub_calculations: List[FullCalculation] = []

        assert segment_length > 2, 'The Segment length must be greater 2, due to unwanted internal side effects'
        assert self.segment_length_with_overlap <= self._total_length, (
            f'{self.segment_length_with_overlap=} cant be greater than the total length {self._total_length}'
        )

        # Storing all original start values
        self._original_start_values = {
            **{flow: flow.previous_flow_rate for flow in self.flow_system.flows.values()},
            **{
                comp: comp.initial_charge_state
                for comp in self.flow_system.components.values()
                if isinstance(comp, Storage)
            },
        }
        self._transfered_start_values: Dict[str, Dict[str, Any]] = {}

    def do_modeling_and_solve(self, solver: Solver, save_results: Union[bool, str, pathlib.Path] = True):
        logger.info(f'{"":#^80}')
        logger.info(f'{" Segmented Solving ":#^80}')
        self._define_path_names(save_results)

        for i in range(self.number_of_segments):
            name_of_segment = f'Segment_{i + 1}'
            if self.sub_calculations:
                self._transfer_start_values(name_of_segment)
            time_indices = self._get_indices(i)
            logger.info(f'{name_of_segment}. (flow_system indices {time_indices.start}...{time_indices.stop - 1}):')
            calculation = FullCalculation(name_of_segment, self.flow_system, self.modeling_language, time_indices)
            # TODO: Add Before Values if available
            self.sub_calculations.append(calculation)
            calculation.do_modeling()
            invest_elements = [
                model.element.label_full
                for model in calculation.system_model.sub_models
                if isinstance(model, InvestmentModel)
            ]
            if invest_elements:
                logger.critical(
                    f'Investments are not supported in Segmented Calculation! '
                    f'Following elements Contain Investments: {invest_elements}'
                )
            calculation.solve(solver, save_results=False)

        self._reset_start_values()

        for calc in self.sub_calculations:
            for key, value in calc.durations.items():
                self.durations[key] += value

        if save_results:
            self._save_solve_infos()

    def results(
        self, combined_arrays: bool = False, combined_scalars: bool = False, individual_results: bool = False
    ) -> Dict[str, Union[Numeric, Dict[str, Numeric]]]:
        """
        Retrieving the results of a Segmented Calculation is not as straight forward as with other Calculation types.
        You have 3 options:
        1.  combined_arrays:
            Retrieve the combined array Results of all Segments as 'combined_arrays'. All result arrays ar concatenated,
            taking care of removing the overlap. These results can be directly compared to other Calculation results.
            Unfortunately, Scalar values like the total of effects can not be combined in a deterministic way.
            Rather convert the time series effect results to a sum yourself.
        2.  combined_scalars:
            Retrieve the combined scalar Results of all Segments. All Scalar Values like the total of effects are
            combined and stored in a List. Take care that the total of multiple Segment is not equivalent to the
            total of the total timeSeries, as it includes the Overlap!
        3.  individual_results:
            Retrieve the individual results of each Segment

        """
        options_chosen = combined_arrays + combined_scalars + individual_results
        assert options_chosen == 1, (
            'Exactly one of the three options to retrieve the results needs to be chosen! You chose {options_chosen}!'
        )
        all_results = {f'Segment_{i + 1}': calculation.results() for i, calculation in enumerate(self.sub_calculations)}
        if combined_arrays:
            return _combine_nested_arrays(*list(all_results.values()), length_per_array=self.segment_length)
        elif combined_scalars:
            return _combine_nested_scalars(*list(all_results.values()))
        else:
            return all_results

    def _save_solve_infos(self):
        t_start = timeit.default_timer()
        indent = 4 if len(self.flow_system.time_series) < 50 else None
        with open(self._paths['results'], 'w', encoding='utf-8') as f:
            results = copy_and_convert_datatypes(
                self.results(combined_arrays=True), use_numpy=False, use_element_label=False
            )
            json.dump(results, f, indent=indent)

        with open(self._paths['data'], 'w', encoding='utf-8') as f:
            data = copy_and_convert_datatypes(self.flow_system.infos(), use_numpy=False, use_element_label=False)
            json.dump(data, f, indent=indent)

        with open(self._paths['results'].parent / f'{self.name}_results_extra.json', 'w', encoding='utf-8') as f:
            results = {
                'Individual Results': copy_and_convert_datatypes(
                    self.results(individual_results=True), use_numpy=False, use_element_label=False
                ),
                'Skalar Results': copy_and_convert_datatypes(
                    self.results(combined_scalars=True), use_numpy=False, use_element_label=False
                ),
            }
            json.dump(results, f, indent=indent)
        self.durations['saving'] = round(timeit.default_timer() - t_start, 2)

        t_start = timeit.default_timer()
        nodes_info, edges_info = self.flow_system.network_infos()
        infos = {
            'Calculation': self.infos,
            'Model': self.sub_calculations[0].system_model.infos,
            'FlowSystem': get_compact_representation(self.flow_system.infos(use_numpy=True, use_element_label=True)),
            'Network': {'Nodes': nodes_info, 'Edges': edges_info},
        }

        with open(self._paths['infos'], 'w', encoding='utf-8') as f:
            yaml.dump(
                infos,
                f,
                width=1000,  # Verhinderung Zeilenumbruch für lange equations
                allow_unicode=True,
                sort_keys=False,
            )

        message = f' Saved Calculation: {self.name} '
        logger.info(f'{"":#^80}\n{message:#^80}\n{"":#^80}')
        logger.info(f'Saving calculation to .json took {self.durations["saving"]:>8.2f} seconds')
        logger.info(f'Saving calculation to .yaml took {(timeit.default_timer() - t_start):>8.2f} seconds')

    def _transfer_start_values(self, segment_name: str):
        """
        This function gets the last values of the previous solved segment and
        inserts them as start values for the nest segment
        """
        final_index_of_prior_segment = -(1 + self.overlap_length)
        start_values_of_this_segment = {}
        for flow in self.flow_system.flows.values():
            flow.previous_flow_rate = flow.model.flow_rate.result[
                final_index_of_prior_segment
            ]  # TODO: maybe more values?
            start_values_of_this_segment[flow.label_full] = flow.previous_flow_rate
        for comp in self.flow_system.components.values():
            if isinstance(comp, Storage):
                comp.initial_charge_state = comp.model.charge_state.result[final_index_of_prior_segment]
                start_values_of_this_segment[comp.label_full] = comp.initial_charge_state

        self._transfered_start_values[segment_name] = start_values_of_this_segment

    def _reset_start_values(self):
        """This resets the start values of all Elements to its original state"""
        for flow in self.flow_system.flows.values():
            flow.previous_flow_rate = self._original_start_values[flow]
        for comp in self.flow_system.components.values():
            if isinstance(comp, Storage):
                comp.initial_charge_state = self._original_start_values[comp]

    def _get_indices(self, segment_index: int) -> range:
        start = segment_index * self.segment_length
        return range(start, min(start + self.segment_length + self.overlap_length, self._total_length))

    @property
    def segment_length_with_overlap(self):
        return self.segment_length + self.overlap_length

    @property
    def start_values_of_segments(self) -> Dict[str, Dict[str, Any]]:
        """Gives an overview of the start values of all Segments"""
        return {
            self.sub_calculations[0].name: {
                element.label_full: value for element, value in self._original_start_values.items()
            },
            **self._transfered_start_values,
        }


def _remove_none_values(d: Dict[Any, Optional[Any]]) -> Dict[Any, Any]:
    # Remove None values from a dictionary
    return {k: _remove_none_values(v) if isinstance(v, dict) else v for k, v in d.items() if v is not None}


def _remove_empty_dicts(d: Dict[Any, Any]) -> Dict[Any, Any]:
    """Recursively removes empty dictionaries from a nested dictionary."""
    return {
        k: _remove_empty_dicts(v) if isinstance(v, dict) else v
        for k, v in d.items()
        if not isinstance(v, dict) or _remove_empty_dicts(v)
    }


def _combine_nested_arrays(
    *dicts: Dict[str, Union[Numeric, dict]],
    trim: Optional[int] = None,
    length_per_array: Optional[int] = None,
) -> Dict[str, Union[np.ndarray, dict]]:
    """
    Combines multiple dictionaries with identical structures by concatenating their arrays,
    with optional trimming. Filters out all other values.

    Parameters
    ----------
    *dicts : Dict[str, Union[np.ndarray, dict]]
        Dictionaries with matching structures and Numeric values.
    trim : int, optional
        Number of elements to trim from the end of each array except the last. Defaults to None.
    length_per_array : int, optional
        Trims the arrays to the desired length. Defaults to None.
        If None, then trim is used.

    Returns
    -------
    Dict[str, Union[np.ndarray, dict]]
        A single dictionary with concatenated arrays at each key, ignoring non-array values.

    Example
    -------
    >>> dict1 = {'a': np.array([1, 2, 3]), 'b': {'c': np.array([4, 5, 6])}}
    >>> dict2 = {'a': np.array([7, 8, 9]), 'b': {'c': np.array([10, 11, 12])}}
    >>> _combine_nested_arrays(dict1, dict2, trim=1)
    {'a': array([1, 2, 7, 8, 9]), 'b': {'c': array([4, 5, 10, 11, 12])}}
    """
    assert (trim is None) != (length_per_array is None), (
        'Either trim or length_per_array must be provided,But not both!'
    )

    def combine_arrays_recursively(
        *values: Union[Numeric, Dict[str, Numeric], Any],
    ) -> Optional[Union[np.ndarray, Dict[str, Union[np.ndarray, dict]]]]:
        if all(isinstance(val, dict) for val in values):  # If all values are dictionaries, recursively combine each key
            return {key: combine_arrays_recursively(*(val[key] for val in values)) for key in values[0]}

        if all(isinstance(val, np.ndarray) for val in values):

            def limit(idx: int, arr: np.ndarray) -> np.ndarray:
                # Performs the trimming of the arrays. Doesn't trim the last array!
                if trim and idx < len(values) - 1:
                    return arr[:-trim]
                elif length_per_array and idx < len(values) - 1:
                    return arr[:length_per_array]
                return arr

            values: List[np.ndarray]
            return np.concatenate([limit(idx, arr) for idx, arr in enumerate(values)])

        else:  # Ignore non-array values
            return None

    combined_arrays = combine_arrays_recursively(*dicts)
    combined_arrays = _remove_none_values(combined_arrays)
    return _remove_empty_dicts(combined_arrays)


def _combine_nested_scalars(*dicts: Dict[str, Union[Numeric, dict]]) -> Dict[str, Union[List[Skalar], dict]]:
    """
    Combines multiple dictionaries with identical structures by combining its skalar values to a list.
    Filters out all other values.

    Parameters
    ----------
    *dicts : Dict[str, Union[np.ndarray, dict]]
        Dictionaries with matching structures and Numeric values.
    """

    def combine_scalars_recursively(
        *values: Union[Numeric, Dict[str, Numeric], Any],
    ) -> Optional[Union[List[Skalar], Dict[str, Union[List[Skalar], dict]]]]:
        # If all values are dictionaries, recursively combine each key
        if all(isinstance(val, dict) for val in values):
            return {key: combine_scalars_recursively(*(val[key] for val in values)) for key in values[0]}

        # Concatenate arrays with optional trimming
        if all(np.isscalar(val) for val in values):
            return [val for val in values]
        else:  # Ignore non-skalar values
            return None

    combined_scalars = combine_scalars_recursively(*dicts)
    combined_scalars = _remove_none_values(combined_scalars)
    return _remove_empty_dicts(combined_scalars)
