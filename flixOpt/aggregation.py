"""
This module contains the Aggregation functionality for the flixOpt framework.
Through this, aggregating TimeSeriesData is possible.
"""

import copy
import logging
import timeit
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import linopy
import numpy as np
import pandas as pd

try:
    import tsam.timeseriesaggregation as tsam
    TSAM_AVAILABLE = True
except ImportError:
    TSAM_AVAILABLE = False

from .components import Storage
from .core import Scalar, TimeSeriesData
from .elements import Component
from .flow_system import FlowSystem
from .structure import (
    Element,
    Model,
    SystemModel,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go

warnings.filterwarnings('ignore', category=DeprecationWarning)
logger = logging.getLogger('flixOpt')


class Aggregation:
    """
    aggregation organizing class
    """

    def __init__(
        self,
        original_data: pd.DataFrame,
        hours_per_time_step: Scalar,
        hours_per_period: Scalar,
        nr_of_periods: int = 8,
        weights: Dict[str, float] = None,
        time_series_for_high_peaks: List[str] = None,
        time_series_for_low_peaks: List[str] = None,
    ):

        """
        Args:
            original_data: The original data to aggregate
            hours_per_time_step: The duration of each timestep in hours.
            hours_per_period: The duration of each period in hours.
            nr_of_periods: The number of typical periods to use in the aggregation.
            weights: The weights for aggregation. If None, all time series are equally weighted.
            time_series_for_high_peaks: List of time series to use for explicitly selecting periods with high values.
            time_series_for_low_peaks: List of time series to use for explicitly selecting periods with low values.
        """
        if not TSAM_AVAILABLE:
            raise ImportError("The 'tsam' package is required for clustering functionality. "
                              "Install it with 'pip install tsam'.")
        self.original_data = copy.deepcopy(original_data)
        self.hours_per_time_step = hours_per_time_step
        self.hours_per_period = hours_per_period
        self.nr_of_periods = nr_of_periods
        self.nr_of_time_steps = len(self.original_data.index)
        self.weights = weights or {}
        self.time_series_for_high_peaks = time_series_for_high_peaks or []
        self.time_series_for_low_peaks = time_series_for_low_peaks or []

        self.aggregated_data: Optional[pd.DataFrame] = None
        self.clustering_duration_seconds = None
        self.tsam: Optional[tsam.TimeSeriesAggregation] = None

    def cluster(self) -> None:
        """
        Durchführung der Zeitreihenaggregation
        """
        start_time = timeit.default_timer()
        # Erstellen des aggregation objects
        self.tsam = tsam.TimeSeriesAggregation(
            self.original_data,
            noTypicalPeriods=self.nr_of_periods,
            hoursPerPeriod=self.hours_per_period,
            resolution=self.hours_per_time_step,
            clusterMethod='k_means',
            extremePeriodMethod='new_cluster_center'
            if self.use_extreme_periods
            else 'None',  # Wenn Extremperioden eingebunden werden sollen, nutze die Methode 'new_cluster_center' aus tsam
            weightDict={name: weight for name, weight in self.weights.items() if name in self.original_data.columns},
            addPeakMax=self.time_series_for_high_peaks,
            addPeakMin=self.time_series_for_low_peaks,
        )

        self.tsam.createTypicalPeriods()  # Ausführen der Aggregation/Clustering
        self.aggregated_data = self.tsam.predictOriginalData()

        self.clustering_duration_seconds = timeit.default_timer() - start_time  # Zeit messen:
        logger.info(self.describe_clusters())

    def describe_clusters(self) -> str:
        description = {}
        for cluster in self.get_cluster_indices().keys():
            description[cluster] = [
                str(indexVector[0]) + '...' + str(indexVector[-1])
                for indexVector in self.get_cluster_indices()[cluster]
            ]

        if self.use_extreme_periods:
            # Zeitreihe rauslöschen:
            extreme_periods = self.tsam.extremePeriods.copy()
            for key in extreme_periods:
                del extreme_periods[key]['profile']
        else:
            extreme_periods = {}

        return (
            f'{"":#^80}\n'
            f'{" Clustering ":#^80}\n'
            f'periods_order:\n'
            f'{self.tsam.clusterOrder}\n'
            f'clusterPeriodNoOccur:\n'
            f'{self.tsam.clusterPeriodNoOccur}\n'
            f'index_vectors_of_clusters:\n'
            f'{description}\n'
            f'{"":#^80}\n'
            f'extreme_periods:\n'
            f'{extreme_periods}\n'
            f'{"":#^80}'
        )

    @property
    def use_extreme_periods(self):
        return self.time_series_for_high_peaks or self.time_series_for_low_peaks

    def plot(self, colormap: str = 'viridis', show: bool = True) -> 'go.Figure':
        from . import plotting

        df_org = self.original_data.copy().rename(
            columns={col: f'Original - {col}' for col in self.original_data.columns}
        )
        df_agg = self.aggregated_data.copy().rename(
            columns={col: f'Aggregated - {col}' for col in self.aggregated_data.columns}
        )
        fig = plotting.with_plotly(df_org, 'line', colors=colormap)
        for trace in fig.data:
            trace.update(dict(line=dict(dash='dash')))
        fig = plotting.with_plotly(df_agg, 'line', colors=colormap, show=show, fig=fig)

        fig.update_layout(
            title='Original vs Aggregated Data (original = ---)', xaxis_title='Index', yaxis_title='Value'
        )
        return fig

    def get_cluster_indices(self) -> Dict[str, List[np.ndarray]]:
        """
        Generates a dictionary that maps each cluster to a list of index vectors representing the time steps
        assigned to that cluster for each period.

        Returns:
            dict: {cluster_0: [index_vector_3, index_vector_7, ...],
                   cluster_1: [index_vector_1],
                   ...}
        """
        clusters = self.tsam.clusterPeriodNoOccur.keys()
        index_vectors = {cluster: [] for cluster in clusters}

        period_length = len(self.tsam.stepIdx)
        total_steps = len(self.tsam.timeSeries)

        for period, cluster_id in enumerate(self.tsam.clusterOrder):
            start_idx = period * period_length
            end_idx = np.min([start_idx + period_length, total_steps])
            index_vectors[cluster_id].append(np.arange(start_idx, end_idx))

        return index_vectors

    def get_equation_indices(self, skip_first_index_of_period: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates pairs of indices for the equations by comparing index vectors of the same cluster.
        If `skip_first_index_of_period` is True, the first index of each period is skipped.

        Args:
            skip_first_index_of_period (bool): Whether to include or skip the first index of each period.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two arrays of indices.
        """
        idx_var1 = []
        idx_var2 = []

        # Iterate through cluster index vectors
        for index_vectors in self.get_cluster_indices().values():
            if len(index_vectors) <= 1:  # Only proceed if cluster has more than one period
                continue

            # Process the first vector, optionally skip first index
            first_vector = index_vectors[0][1:] if skip_first_index_of_period else index_vectors[0]

            # Compare first vector to others in the cluster
            for other_vector in index_vectors[1:]:
                if skip_first_index_of_period:
                    other_vector = other_vector[1:]

                # Compare elements up to the minimum length of both vectors
                min_len = min(len(first_vector), len(other_vector))
                idx_var1.extend(first_vector[:min_len])
                idx_var2.extend(other_vector[:min_len])

        # Convert lists to numpy arrays
        return np.array(idx_var1), np.array(idx_var2)


class AggregationParameters:
    def __init__(
        self,
        hours_per_period: float,
        nr_of_periods: int,
        fix_storage_flows: bool,
        aggregate_data_and_fix_non_binary_vars: bool,
        percentage_of_period_freedom: float = 0,
        penalty_of_period_freedom: float = 0,
        time_series_for_high_peaks: List[TimeSeriesData] = None,
        time_series_for_low_peaks: List[TimeSeriesData] = None,
    ):
        """
        Initializes aggregation parameters for time series data

        Args:
            hours_per_period: Duration of each period in hours.
            nr_of_periods: Number of typical periods to use in the aggregation.
            fix_storage_flows: Whether to aggregate storage flows (load/unload); if other flows
                are fixed, fixing storage flows is usually not required.
            aggregate_data_and_fix_non_binary_vars: Whether to aggregate all time series data, which allows to fix all time series variables (like flow_rate),
                or only fix binary variables. If False non time_series data is changed!! If True, the mathematical Problem
                is simplified even further.
            percentage_of_period_freedom: Specifies the maximum percentage (0–100) of binary values within each period
                that can deviate as "free variables", chosen by the solver (default is 0).
                This allows binary variables to be 'partly equated' between aggregated periods.
            penalty_of_period_freedom: The penalty associated with each "free variable"; defaults to 0. Added to Penalty
            time_series_for_high_peaks: List of TimeSeriesData to use for explicitly selecting periods with high values.
            time_series_for_low_peaks: List of TimeSeriesData to use for explicitly selecting periods with low values.
        """
        self.hours_per_period = hours_per_period
        self.nr_of_periods = nr_of_periods
        self.fix_storage_flows = fix_storage_flows
        self.aggregate_data_and_fix_non_binary_vars = aggregate_data_and_fix_non_binary_vars
        self.percentage_of_period_freedom = percentage_of_period_freedom
        self.penalty_of_period_freedom = penalty_of_period_freedom
        self.time_series_for_high_peaks: List[TimeSeriesData] = time_series_for_high_peaks or []
        self.time_series_for_low_peaks: List[TimeSeriesData] = time_series_for_low_peaks or []

    @property
    def use_extreme_periods(self):
        return self.time_series_for_high_peaks or self.time_series_for_low_peaks

    @property
    def labels_for_high_peaks(self) -> List[str]:
        return [ts.label for ts in self.time_series_for_high_peaks]

    @property
    def labels_for_low_peaks(self) -> List[str]:
        return [ts.label for ts in self.time_series_for_low_peaks]

    @property
    def use_low_peaks(self):
        return self.time_series_for_low_peaks is not None


class AggregationModel(Model):
    """The AggregationModel holds equations and variables related to the Aggregation of a FLowSystem.
    It creates Equations that equates indices of variables, and introduces penalties related to binary variables, that
    escape the equation to their related binaries in other periods"""

    def __init__(
        self,
        model: SystemModel,
        aggregation_parameters: AggregationParameters,
        flow_system: FlowSystem,
        aggregation_data: Aggregation,
        components_to_clusterize: Optional[List[Component]],
    ):
        """
        Modeling-Element for "index-equating"-equations
        """
        super().__init__(model, label_of_element='Aggregation', label_full='Aggregation')
        self.flow_system = flow_system
        self.aggregation_parameters = aggregation_parameters
        self.aggregation_data = aggregation_data
        self.components_to_clusterize = components_to_clusterize

    def do_modeling(self):
        if not self.components_to_clusterize:
            components = self.flow_system.components.values()
        else:
            components = [component for component in self.components_to_clusterize]

        indices = self.aggregation_data.get_equation_indices(skip_first_index_of_period=True)

        time_variables: Set[str] = {k for k, v in self._model.variables.data.items() if 'time' in v.indexes}
        binary_variables: Set[str] = {k for k, v in self._model.variables.data.items() if k in self._model.binaries}
        binary_time_variables: Set[str] = time_variables & binary_variables

        for component in components:
            if isinstance(component, Storage) and not self.aggregation_parameters.fix_storage_flows:
                continue  # Fix Nothing in The Storage

            all_variables_of_component = set(component.model.variables)

            if self.aggregation_parameters.aggregate_data_and_fix_non_binary_vars:
                relevant_variables = component.model.variables[all_variables_of_component & time_variables]
            else:
                relevant_variables = component.model.variables[all_variables_of_component & binary_time_variables]
            for variable in relevant_variables:
                self._equate_indices(component.model.variables[variable], indices)

        penalty = self.aggregation_parameters.penalty_of_period_freedom
        if (self.aggregation_parameters.percentage_of_period_freedom > 0) and penalty != 0:
            for variable in self.variables_direct.values():
                self._model.effects.add_share_to_penalty(
                    'Aggregation',
                    variable * penalty
                )

    def _equate_indices(self, variable: linopy.Variable, indices: Tuple[np.ndarray, np.ndarray]) -> None:
        assert len(indices[0]) == len(indices[1]), 'The length of the indices must match!!'
        length = len(indices[0])

        # Gleichung:
        # eq1: x(p1,t) - x(p3,t) = 0 # wobei p1 und p3 im gleichen Cluster sind und t = 0..N_p
        con = self.add(self._model.add_constraints(
            variable.isel(time=indices[0]) - variable.isel(time=indices[1]) == 0,
            name=f'{self.label_full}|equate_indices|{variable.name}'),
            f'equate_indices|{variable.name}')

        # Korrektur: (bisher nur für Binärvariablen:)
        if variable.name in self._model.variables.binaries and self.aggregation_parameters.percentage_of_period_freedom > 0:
            var_k1 = self.add(self._model.add_variables(
                binary=True,
                coords={'time': variable.isel(time=indices[0]).indexes['time']},
                name=f'{self.label_full}|correction1|{variable.name}'), f'correction1|{variable.name}')

            var_k0 = self.add(self._model.add_variables(
                binary=True,
                coords={'time': variable.isel(time=indices[0]).indexes['time']},
                name=f'{self.label_full}|correction0|{variable.name}'), f'correction0|{variable.name}')

            # equation extends ...
            # --> On(p3) can be 0/1 independent of On(p1,t)!
            # eq1: On(p1,t) - On(p3,t) + K1(p3,t) - K0(p3,t) = 0
            # --> correction On(p3) can be:
            #  On(p1,t) = 1 -> On(p3) can be 0 -> K0=1 (,K1=0)
            #  On(p1,t) = 0 -> On(p3) can be 1 -> K1=1 (,K0=1)
            con.lhs += 1 * var_k1 - 1 * var_k0

            # interlock var_k1 and var_K2:
            # eq: var_k0(t)+var_k1(t) <= 1.1
            self.add(self._model.add_constraints(
                var_k0 + var_k1 <= 1.1,
                name=f'{self.label_full}|lock_k0_and_k1|{variable.name}'),
                f'lock_k0_and_k1|{variable.name}'
            )

            # Begrenzung der Korrektur-Anzahl:
            # eq: sum(K) <= n_Corr_max
            self.add(self._model.add_constraints(
                sum(var_k0) + sum(var_k1) <= round(self.aggregation_parameters.percentage_of_period_freedom / 100 * length),
                name=f'{self.label_full}|limit_corrections|{variable.name}'),
                f'limit_corrections|{variable.name}'
            )
