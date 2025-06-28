"""
This module contains the Calculation functionality for the flixopt framework.
It is used to calculate a SystemModel for a given FlowSystem through a solver.
There are three different Calculation types:
    1. FullCalculation: Calculates the SystemModel for the full FlowSystem
    2. AggregatedCalculation: Calculates the SystemModel for the full FlowSystem, but aggregates the TimeSeriesData.
        This simplifies the mathematical model and usually speeds up the solving process.
    3. SegmentedCalculation: Solves a SystemModel for each individual Segment of the FlowSystem.
"""

import logging
import math
import pathlib
import timeit
import warnings
from collections import Counter
from typing import Annotated, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from . import io as fx_io
from . import utils as utils
from .aggregation import AggregationModel, AggregationParameters
from .components import Storage
from .config import CONFIG
from .core import DataConverter, Scalar, TimeSeriesData, drop_constant_arrays
from .elements import Component
from .features import InvestmentModel
from .flow_system import FlowSystem
from .results import CalculationResults, SegmentedCalculationResults
from .solvers import _Solver
from .structure import SystemModel

logger = logging.getLogger('flixopt')


class Calculation:
    """
    class for defined way of solving a flow_system optimization
    """

    def __init__(
        self,
        name: str,
        flow_system: FlowSystem,
        active_timesteps: Annotated[
            Optional[pd.DatetimeIndex],
            'DEPRECATED: Use flow_system.sel(time=...) or flow_system.isel(time=...) instead',
        ] = None,
        folder: Optional[pathlib.Path] = None,
    ):
        """
        Args:
            name: name of calculation
            flow_system: flow_system which should be calculated
            folder: folder where results should be saved. If None, then the current working directory is used.
            active_timesteps: Deprecated. Use FLowSystem.sel(time=...) or FlowSystem.isel(time=...) instead.
        """
        self.name = name
        if flow_system.used_in_calculation:
            logging.warning(f'FlowSystem {flow_system} is already used in a calculation. '
                            f'Creating a copy for Calculation "{self.name}".')
            flow_system = flow_system.copy()

        if active_timesteps is not None:
            warnings.warn(
                "The 'active_timesteps' parameter is deprecated and will be removed in a future version. "
                'Use flow_system.sel(time=timesteps) or flow_system.isel(time=indices) before passing '
                'the FlowSystem to the Calculation instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            flow_system = flow_system.sel(time=active_timesteps)
        self._active_timesteps = active_timesteps  # deprecated

        flow_system._used_in_calculation = True

        self.flow_system = flow_system
        self.model: Optional[SystemModel] = None

        self.durations = {'modeling': 0.0, 'solving': 0.0, 'saving': 0.0}
        self.folder = pathlib.Path.cwd() / 'results' if folder is None else pathlib.Path(folder)
        self.results: Optional[CalculationResults] = None

        if not self.folder.exists():
            try:
                self.folder.mkdir(parents=False)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f'Folder {self.folder} and its parent do not exist. Please create them first.'
                ) from e

    @property
    def main_results(self) -> Dict[str, Union[Scalar, Dict]]:
        from flixopt.features import InvestmentModel

        main_results = {
            'Objective': self.model.objective.value,
            'Penalty': self.model.effects.penalty.total.solution.values,
            'Effects': {
                f'{effect.label} [{effect.unit}]': {
                    'operation': effect.model.operation.total.solution.values,
                    'invest': effect.model.invest.total.solution.values,
                    'total': effect.model.total.solution.values,
                }
                for effect in self.flow_system.effects
            },
            'Invest-Decisions': {
                'Invested': {
                    model.label_of_element: model.size.solution
                    for component in self.flow_system.components.values()
                    for model in component.model.all_sub_models
                    if isinstance(model, InvestmentModel) and model.size.solution.max() >= CONFIG.modeling.EPSILON
                },
                'Not invested': {
                    model.label_of_element: model.size.solution
                    for component in self.flow_system.components.values()
                    for model in component.model.all_sub_models
                    if isinstance(model, InvestmentModel) and model.size.solution.max() < CONFIG.modeling.EPSILON
                },
            },
            'Buses with excess': [
                {
                    bus.label_full: {
                        'input': bus.model.excess_input.solution.sum('time'),
                        'output': bus.model.excess_output.solution.sum('time'),
                    }
                }
                for bus in self.flow_system.buses.values()
                if bus.with_excess
                and (
                    bus.model.excess_input.solution.sum() > 1e-3
                    or bus.model.excess_output.solution.sum() > 1e-3
                )
            ],
        }

        return utils.round_floats(main_results)

    @property
    def summary(self):
        return {
            'Name': self.name,
            'Number of timesteps': len(self.flow_system.timesteps),
            'Calculation Type': self.__class__.__name__,
            'Constraints': self.model.constraints.ncons,
            'Variables': self.model.variables.nvars,
            'Main Results': self.main_results,
            'Durations': self.durations,
            'Config': CONFIG.to_dict(),
        }

    @property
    def active_timesteps(self) -> pd.DatetimeIndex:
        warnings.warn(
            'active_timesteps is deprecated. Use active_timesteps instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self._active_timesteps


class FullCalculation(Calculation):
    """
    class for defined way of solving a flow_system optimization
    """

    def do_modeling(self) -> SystemModel:
        t_start = timeit.default_timer()
        self.flow_system.connect_and_transform()

        self.model = self.flow_system.create_model()
        self.model.do_modeling()

        self.durations['modeling'] = round(timeit.default_timer() - t_start, 2)
        return self.model

    def solve(self, solver: _Solver, log_file: Optional[pathlib.Path] = None, log_main_results: bool = True):
        t_start = timeit.default_timer()

        self.model.solve(
            log_fn=pathlib.Path(log_file) if log_file is not None else self.folder / f'{self.name}.log',
            solver_name=solver.name,
            **solver.options,
        )
        self.durations['solving'] = round(timeit.default_timer() - t_start, 2)

        if self.model.status == 'warning':
            # Save the model and the flow_system to file in case of infeasibility
            paths = fx_io.CalculationResultsPaths(self.folder, self.name)
            from .io import document_linopy_model

            document_linopy_model(self.model, paths.model_documentation)
            self.flow_system.to_netcdf(paths.flow_system)
            raise RuntimeError(
                f'Model was infeasible. Please check {paths.model_documentation=} and {paths.flow_system=} for more information.'
            )

        # Log the formatted output
        if log_main_results:
            logger.info(f'{" Main Results ":#^80}')
            logger.info(
                '\n'
                + yaml.dump(
                    utils.round_floats(self.main_results),
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    indent=4,
                )
            )

        self.results = CalculationResults.from_calculation(self)


class AggregatedCalculation(FullCalculation):
    """
    class for defined way of solving a flow_system optimization
    """

    def __init__(
        self,
        name: str,
        flow_system: FlowSystem,
        aggregation_parameters: AggregationParameters,
        components_to_clusterize: Optional[List[Component]] = None,
        active_timesteps: Annotated[
            Optional[pd.DatetimeIndex],
            'DEPRECATED: Use flow_system.sel(time=...) or flow_system.isel(time=...) instead',
        ] = None,
        folder: Optional[pathlib.Path] = None,
    ):
        """
        Class for Optimizing the `FlowSystem` including:
            1. Aggregating TimeSeriesData via typical periods using tsam.
            2. Equalizing variables of typical periods.
        Args:
            name: name of calculation
            flow_system: flow_system which should be calculated
            aggregation_parameters: Parameters for aggregation. See documentation of AggregationParameters class.
            components_to_clusterize: List of Components to perform aggregation on. If None, then all components are aggregated.
                This means, teh variables in the components are equalized to each other, according to the typical periods
                computed in the DataAggregation
            active_timesteps: pd.DatetimeIndex or None
                list with indices, which should be used for calculation. If None, then all timesteps are used.
            folder: folder where results should be saved. If None, then the current working directory is used.
        """
        if flow_system.scenarios is not None:
            raise ValueError('Aggregation is not supported for scenarios yet. Please use FullCalculation instead.')
        super().__init__(name, flow_system, folder=folder, active_timesteps=active_timesteps)
        self.aggregation_parameters = aggregation_parameters
        self.components_to_clusterize = components_to_clusterize
        self.aggregation = None

    def do_modeling(self) -> SystemModel:
        t_start = timeit.default_timer()
        self.flow_system.connect_and_transform()
        self._perform_aggregation()

        # Model the System
        self.model = self.flow_system.create_model()
        self.model.do_modeling()
        # Add Aggregation Model after modeling the rest
        self.aggregation = AggregationModel(
            self.model, self.aggregation_parameters, self.flow_system, self.aggregation, self.components_to_clusterize
        )
        self.aggregation.do_modeling()
        self.durations['modeling'] = round(timeit.default_timer() - t_start, 2)
        return self.model

    def _perform_aggregation(self):
        from .aggregation import Aggregation

        t_start_agg = timeit.default_timer()

        # Validation
        dt_min, dt_max = (
            np.min(self.flow_system.hours_per_timestep),
            np.max(self.flow_system.hours_per_timestep),
        )
        if not dt_min == dt_max:
            raise ValueError(
                f'Aggregation failed due to inconsistent time step sizes:'
                f'delta_t varies from {dt_min} to {dt_max} hours.'
            )
        steps_per_period = (
            self.aggregation_parameters.hours_per_period
            / self.flow_system.hours_per_timestep.max()
        )
        is_integer = (
            self.aggregation_parameters.hours_per_period
            % self.flow_system.hours_per_timestep.max()
        ).item() == 0
        if not (steps_per_period.size == 1 and is_integer):
            raise ValueError(
                f'The selected {self.aggregation_parameters.hours_per_period=} does not match the time '
                f'step size of {dt_min} hours). It must be a multiple of {dt_min} hours.'
            )

        logger.info(f'{"":#^80}')
        logger.info(f'{" Aggregating TimeSeries Data ":#^80}')

        ds = self.flow_system.to_dataset()

        temporaly_changing_ds = drop_constant_arrays(ds, dim='time')

        # Aggregation - creation of aggregated timeseries:
        self.aggregation = Aggregation(
            original_data=temporaly_changing_ds.to_dataframe(),
            hours_per_time_step=float(dt_min),
            hours_per_period=self.aggregation_parameters.hours_per_period,
            nr_of_periods=self.aggregation_parameters.nr_of_periods,
            weights=self.calculate_aggregation_weights(temporaly_changing_ds),
            time_series_for_high_peaks=self.aggregation_parameters.labels_for_high_peaks,
            time_series_for_low_peaks=self.aggregation_parameters.labels_for_low_peaks,
        )

        self.aggregation.cluster()
        self.aggregation.plot(show=True, save=self.folder / 'aggregation.html')
        if self.aggregation_parameters.aggregate_data_and_fix_non_binary_vars:
            ds = self.flow_system.to_dataset()
            for name, series in self.aggregation.aggregated_data.items():
                da = DataConverter.to_dataarray(series, timesteps=self.flow_system.timesteps).rename(name).assign_attrs(ds[name].attrs)
                if TimeSeriesData.is_timeseries_data(da):
                    da = TimeSeriesData.from_dataarray(da)

                ds[name] = da

            self.flow_system = FlowSystem.from_dataset(ds)
        self.flow_system.connect_and_transform()
        self.durations['aggregation'] = round(timeit.default_timer() - t_start_agg, 2)

    @classmethod
    def calculate_aggregation_weights(cls, ds: xr.Dataset) -> Dict[str, float]:
        """Calculate weights for all datavars in the dataset. Weights are pulled from the attrs of the datavars."""

        groups = [da.attrs['aggregation_group'] for da in ds.values() if 'aggregation_group' in da.attrs]
        group_counts = Counter(groups)

        # Calculate weight for each group (1/count)
        group_weights = {group: 1 / count for group, count in group_counts.items()}

        weights = {}
        for name, da in ds.data_vars.items():
            group_weight = group_weights.get(da.attrs.get('aggregation_group'))
            if group_weight is not None:
                weights[name] = group_weight
            else:
                weights[name] = da.attrs.get('aggregation_weight', 1)

        if np.all(np.isclose(list(weights.values()), 1, atol=1e-6)):
            logger.info('All Aggregation weights were set to 1')

        return weights


class SegmentedCalculation(Calculation):
    def __init__(
        self,
        name: str,
        flow_system: FlowSystem,
        timesteps_per_segment: int,
        overlap_timesteps: int,
        nr_of_previous_values: int = 1,
        folder: Optional[pathlib.Path] = None,
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

        Args:
            name: name of calculation
            flow_system: flow_system which should be calculated
            timesteps_per_segment: The number of time_steps per individual segment (without the overlap)
            overlap_timesteps: The number of time_steps that are added to each individual model. Used for better
                results of storages)
            folder: folder where results should be saved. If None, then the current working directory is used.
        """
        super().__init__(name, flow_system, folder=folder)
        self.timesteps_per_segment = timesteps_per_segment
        self.overlap_timesteps = overlap_timesteps
        self.nr_of_previous_values = nr_of_previous_values
        self.sub_calculations: List[FullCalculation] = []


        self.segment_names = [
            f'Segment_{i + 1}' for i in range(math.ceil(len(self.all_timesteps) / self.timesteps_per_segment))
        ]
        self._timesteps_per_segment = self._calculate_timesteps_per_segment()

        assert timesteps_per_segment > 2, 'The Segment length must be greater 2, due to unwanted internal side effects'
        assert self.timesteps_per_segment_with_overlap <= len(self.all_timesteps), (
            f'{self.timesteps_per_segment_with_overlap=} cant be greater than the total length {len(self.all_timesteps)}'
        )

        self.flow_system._connect_network()  # Connect network to ensure that all Flows know their Component
        # Storing all original start values
        self._original_start_values = {
            **{flow.label_full: flow.previous_flow_rate for flow in self.flow_system.flows.values()},
            **{
                comp.label_full: comp.initial_charge_state
                for comp in self.flow_system.components.values()
                if isinstance(comp, Storage)
            },
        }
        self._transfered_start_values: List[Dict[str, Any]] = []

    def _create_sub_calculations(self):
        for i, (segment_name, timesteps_of_segment) in enumerate(
            zip(self.segment_names, self._timesteps_per_segment, strict=True)
        ):
            calc = FullCalculation(f'{self.name}-{segment_name}', self.flow_system.sel(timesteps_of_segment))
            calc.flow_system._connect_network()  # Connect to have Correct names of Flows!

            self.sub_calculations.append(calc)
            logger.info(
                f'{segment_name} [{i + 1:>2}/{len(self.segment_names):<2}] '
                f'({timesteps_of_segment[0]} -> {timesteps_of_segment[-1]}):'
            )

    def do_modeling_and_solve(
        self, solver: _Solver, log_file: Optional[pathlib.Path] = None, log_main_results: bool = False
    ):
        logger.info(f'{"":#^80}')
        logger.info(f'{" Segmented Solving ":#^80}')
        self._create_sub_calculations()

        for i, calculation in enumerate(self.sub_calculations):
            logger.info(
                f'{self.segment_names[i]} [{i + 1:>2}/{len(self.segment_names):<2}] '
                f'({calculation.flow_system.timesteps[0]} -> {calculation.flow_system.timesteps[-1]}):'
            )

            if i > 0 and self.nr_of_previous_values > 0:
                self._transfer_start_values(i)

            calculation.do_modeling()

            # Warn about Investments, but only in fist run
            if i == 0:
                invest_elements = [
                    model.label_full
                    for component in calculation.flow_system.components.values()
                    for model in component.model.all_sub_models
                    if isinstance(model, InvestmentModel)
                ]
                if invest_elements:
                    logger.critical(
                        f'Investments are not supported in Segmented Calculation! '
                        f'Following InvestmentModels were found: {invest_elements}'
                    )

            calculation.solve(
                solver,
                log_file=pathlib.Path(log_file) if log_file is not None else self.folder / f'{self.name}.log',
                log_main_results=log_main_results,
            )

        for calc in self.sub_calculations:
            for key, value in calc.durations.items():
                self.durations[key] += value

        self.results = SegmentedCalculationResults.from_calculation(self)

    def _transfer_start_values(self, i: int):
        """
        This function gets the last values of the previous solved segment and
        inserts them as start values for the next segment
        """
        timesteps_of_prior_segment = self.sub_calculations[i - 1].flow_system.timesteps_extra

        start = self.sub_calculations[i].flow_system.timesteps[0]
        start_previous_values = timesteps_of_prior_segment[self.timesteps_per_segment - self.nr_of_previous_values]
        end_previous_values = timesteps_of_prior_segment[self.timesteps_per_segment - 1]

        logger.debug(
            f'Start of next segment: {start}. Indices of previous values: {start_previous_values} -> {end_previous_values}'
        )
        current_flow_system = self.sub_calculations[i -1].flow_system
        next_flow_system = self.sub_calculations[i].flow_system

        start_values_of_this_segment = {}

        for current_flow in current_flow_system.flows.values():
            next_flow = next_flow_system.flows[current_flow.label_full]
            next_flow.previous_flow_rate = current_flow.model.flow_rate.solution.sel(
                time=slice(start_previous_values, end_previous_values)
            ).values
            start_values_of_this_segment[current_flow.label_full] = next_flow.previous_flow_rate

        for current_comp in current_flow_system.components.values():
            next_comp = next_flow_system.components[current_comp.label_full]
            if isinstance(next_comp, Storage):
                next_comp.initial_charge_state = current_comp.model.charge_state.solution.sel(time=start).item()
                start_values_of_this_segment[current_comp.label_full] = next_comp.initial_charge_state

        self._transfered_start_values.append(start_values_of_this_segment)

    def _calculate_timesteps_per_segment(self) -> List[pd.DatetimeIndex]:
        timesteps_per_segment = []
        for i, _ in enumerate(self.segment_names):
            start = self.timesteps_per_segment * i
            end = min(start + self.timesteps_per_segment_with_overlap, len(self.all_timesteps))
            timesteps_per_segment.append(self.all_timesteps[start:end])
        return timesteps_per_segment

    @property
    def timesteps_per_segment_with_overlap(self):
        return self.timesteps_per_segment + self.overlap_timesteps

    @property
    def start_values_of_segments(self) -> List[Dict[str, Any]]:
        """Gives an overview of the start values of all Segments"""
        return [
            {name: value for name, value in self._original_start_values.items()}
        ] + [start_values for start_values in self._transfered_start_values]

    @property
    def all_timesteps(self) -> pd.DatetimeIndex:
        return self.flow_system.timesteps
