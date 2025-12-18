"""
This module contains the Clustering functionality for the flixopt framework.
Through this, clustering TimeSeriesData is possible.
"""

from __future__ import annotations

import copy
import logging
import timeit
from typing import TYPE_CHECKING

import numpy as np

try:
    import tsam.timeseriesaggregation as tsam

    TSAM_AVAILABLE = True
except ImportError:
    TSAM_AVAILABLE = False

from .color_processing import process_colors
from .components import Storage
from .config import CONFIG
from .plot_result import PlotResult
from .structure import (
    FlowSystemModel,
    Submodel,
)

if TYPE_CHECKING:
    import linopy
    import pandas as pd

    from .core import Scalar, TimeSeriesData
    from .elements import Component
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


class Clustering:
    """
    Clustering organizing class
    """

    def __init__(
        self,
        original_data: pd.DataFrame,
        hours_per_time_step: Scalar,
        hours_per_period: Scalar,
        nr_of_periods: int = 8,
        weights: dict[str, float] | None = None,
        time_series_for_high_peaks: list[str] | None = None,
        time_series_for_low_peaks: list[str] | None = None,
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
            raise ImportError(
                "The 'tsam' package is required for clustering functionality. Install it with 'pip install tsam'."
            )
        self.original_data = copy.deepcopy(original_data)
        self.hours_per_time_step = hours_per_time_step
        self.hours_per_period = hours_per_period
        self.nr_of_periods = nr_of_periods
        self.nr_of_time_steps = len(self.original_data.index)
        self.weights = weights or {}
        self.time_series_for_high_peaks = time_series_for_high_peaks or []
        self.time_series_for_low_peaks = time_series_for_low_peaks or []

        self.aggregated_data: pd.DataFrame | None = None
        self.clustering_duration_seconds = None
        self.tsam: tsam.TimeSeriesAggregation | None = None

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
        if logger.isEnabledFor(logging.INFO):
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

    def plot(self, colormap: str | None = None, show: bool | None = None) -> PlotResult:
        """Plot original vs aggregated data comparison.

        Visualizes the original time series (dashed lines) overlaid with
        the aggregated/clustered time series (solid lines) for comparison.

        Args:
            colormap: Colorscale name for the time series colors.
                Defaults to CONFIG.Plotting.default_qualitative_colorscale.
            show: Whether to display the figure.
                Defaults to CONFIG.Plotting.default_show.

        Returns:
            PlotResult containing the comparison figure and underlying data.

        Examples:
            >>> clustering.cluster()
            >>> clustering.plot()
            >>> clustering.plot(colormap='Set2', show=False).to_html('clustering.html')
        """
        import plotly.express as px
        import xarray as xr

        df_org = self.original_data.copy().rename(
            columns={col: f'Original - {col}' for col in self.original_data.columns}
        )
        df_agg = self.aggregated_data.copy().rename(
            columns={col: f'Aggregated - {col}' for col in self.aggregated_data.columns}
        )
        colors = list(
            process_colors(colormap or CONFIG.Plotting.default_qualitative_colorscale, list(df_org.columns)).values()
        )

        # Create line plot for original data (dashed)
        index_name = df_org.index.name or 'index'
        df_org_long = df_org.reset_index().melt(id_vars=index_name, var_name='variable', value_name='value')
        fig = px.line(df_org_long, x=index_name, y='value', color='variable', color_discrete_sequence=colors)
        for trace in fig.data:
            trace.update(line=dict(dash='dash'))

        # Add aggregated data (solid lines)
        df_agg_long = df_agg.reset_index().melt(id_vars=index_name, var_name='variable', value_name='value')
        fig2 = px.line(df_agg_long, x=index_name, y='value', color='variable', color_discrete_sequence=colors)
        for trace in fig2.data:
            fig.add_trace(trace)

        fig.update_layout(
            title='Original vs Aggregated Data (original = ---)',
            xaxis_title='Time in h',
            yaxis_title='Value',
        )

        # Build xarray Dataset with both original and aggregated data
        data = xr.Dataset(
            {
                'original': self.original_data.to_xarray().to_array(dim='variable'),
                'aggregated': self.aggregated_data.to_xarray().to_array(dim='variable'),
            }
        )
        result = PlotResult(data=data, figure=fig)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            result.show()

        return result

    def get_cluster_indices(self) -> dict[str, list[np.ndarray]]:
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

    def get_equation_indices(self, skip_first_index_of_period: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates pairs of indices for the equations by comparing index vectors of the same cluster.
        If `skip_first_index_of_period` is True, the first index of each period is skipped.

        Args:
            skip_first_index_of_period (bool): Whether to include or skip the first index of each period.

        Returns:
            tuple[np.ndarray, np.ndarray]: Two arrays of indices.
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


class ClusteringParameters:
    def __init__(
        self,
        hours_per_period: float,
        nr_of_periods: int,
        fix_storage_flows: bool,
        aggregate_data_and_fix_non_binary_vars: bool,
        percentage_of_period_freedom: float = 0,
        penalty_of_period_freedom: float = 0,
        time_series_for_high_peaks: list[TimeSeriesData] | None = None,
        time_series_for_low_peaks: list[TimeSeriesData] | None = None,
    ):
        """
        Initializes clustering parameters for time series data

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
        self.time_series_for_high_peaks: list[TimeSeriesData] = time_series_for_high_peaks or []
        self.time_series_for_low_peaks: list[TimeSeriesData] = time_series_for_low_peaks or []

    @property
    def use_extreme_periods(self):
        return self.time_series_for_high_peaks or self.time_series_for_low_peaks

    @property
    def labels_for_high_peaks(self) -> list[str]:
        return [ts.name for ts in self.time_series_for_high_peaks]

    @property
    def labels_for_low_peaks(self) -> list[str]:
        return [ts.name for ts in self.time_series_for_low_peaks]

    @property
    def use_low_peaks(self) -> bool:
        return bool(self.time_series_for_low_peaks)


class ClusteringModel(Submodel):
    """The ClusteringModel holds equations and variables related to the Clustering of a FlowSystem.
    It creates Equations that equates indices of variables, and introduces penalties related to binary variables, that
    escape the equation to their related binaries in other periods"""

    def __init__(
        self,
        model: FlowSystemModel,
        clustering_parameters: ClusteringParameters,
        flow_system: FlowSystem,
        clustering_data: Clustering,
        components_to_clusterize: list[Component] | None,
    ):
        """
        Modeling-Element for "index-equating"-equations
        """
        super().__init__(model, label_of_element='Clustering', label_of_model='Clustering')
        self.flow_system = flow_system
        self.clustering_parameters = clustering_parameters
        self.clustering_data = clustering_data
        self.components_to_clusterize = components_to_clusterize

    def do_modeling(self):
        if not self.components_to_clusterize:
            components = self.flow_system.components.values()
        else:
            components = [component for component in self.components_to_clusterize]

        indices = self.clustering_data.get_equation_indices(skip_first_index_of_period=True)

        time_variables: set[str] = {
            name for name in self._model.variables if 'time' in self._model.variables[name].dims
        }
        binary_variables: set[str] = set(self._model.variables.binaries)
        binary_time_variables: set[str] = time_variables & binary_variables

        for component in components:
            if isinstance(component, Storage) and not self.clustering_parameters.fix_storage_flows:
                continue  # Fix Nothing in The Storage

            all_variables_of_component = set(component.submodel.variables)

            if self.clustering_parameters.aggregate_data_and_fix_non_binary_vars:
                relevant_variables = component.submodel.variables[all_variables_of_component & time_variables]
            else:
                relevant_variables = component.submodel.variables[all_variables_of_component & binary_time_variables]
            for variable in relevant_variables:
                self._equate_indices(component.submodel.variables[variable], indices)

        penalty = self.clustering_parameters.penalty_of_period_freedom
        if (self.clustering_parameters.percentage_of_period_freedom > 0) and penalty != 0:
            from .effects import PENALTY_EFFECT_LABEL

            for variable_name in self.variables_direct:
                variable = self.variables_direct[variable_name]
                # Sum correction variables over all dimensions to get periodic penalty contribution
                self._model.effects.add_share_to_effects(
                    name='Aggregation',
                    expressions={PENALTY_EFFECT_LABEL: (variable * penalty).sum('time')},
                    target='periodic',
                )

    def _equate_indices(self, variable: linopy.Variable, indices: tuple[np.ndarray, np.ndarray]) -> None:
        assert len(indices[0]) == len(indices[1]), 'The length of the indices must match!!'
        length = len(indices[0])

        # Gleichung:
        # eq1: x(p1,t) - x(p3,t) = 0 # wobei p1 und p3 im gleichen Cluster sind und t = 0..N_p
        con = self.add_constraints(
            variable.isel(time=indices[0]) - variable.isel(time=indices[1]) == 0,
            short_name=f'equate_indices|{variable.name}',
        )

        # Korrektur: (bisher nur für Binärvariablen:)
        if (
            variable.name in self._model.variables.binaries
            and self.clustering_parameters.percentage_of_period_freedom > 0
        ):
            sel = variable.isel(time=indices[0])
            coords = {d: sel.indexes[d] for d in sel.dims}
            var_k1 = self.add_variables(binary=True, coords=coords, short_name=f'correction1|{variable.name}')

            var_k0 = self.add_variables(binary=True, coords=coords, short_name=f'correction0|{variable.name}')

            # equation extends ...
            # --> On(p3) can be 0/1 independent of On(p1,t)!
            # eq1: On(p1,t) - On(p3,t) + K1(p3,t) - K0(p3,t) = 0
            # --> correction On(p3) can be:
            #  On(p1,t) = 1 -> On(p3) can be 0 -> K0=1 (,K1=0)
            #  On(p1,t) = 0 -> On(p3) can be 1 -> K1=1 (,K0=1)
            con.lhs += 1 * var_k1 - 1 * var_k0

            # interlock var_k1 and var_K2:
            # eq: var_k0(t)+var_k1(t) <= 1
            self.add_constraints(var_k0 + var_k1 <= 1, short_name=f'lock_k0_and_k1|{variable.name}')

            # Begrenzung der Korrektur-Anzahl:
            # eq: sum(K) <= n_Corr_max
            limit = int(np.floor(self.clustering_parameters.percentage_of_period_freedom / 100 * length))
            self.add_constraints(
                var_k0.sum(dim='time') + var_k1.sum(dim='time') <= limit,
                short_name=f'limit_corrections|{variable.name}',
            )


class TypicalPeriodsModel(Submodel):
    """
    Model for typical periods optimization with reduced timesteps.

    This model handles:
    1. Timestep weighting for operational costs (each typical period represents multiple original periods)
    2. Inter-period storage state linking via boundary variables
    3. Cyclic storage constraints

    The inter-period storage linking works as follows:
    - SOC_boundary[d] represents the storage state at the boundary of original period d
    - For each original period d with cluster assignment c = cluster_order[d]:
      SOC_boundary[d+1] = SOC_boundary[d] + ΔSOC[c]
    - Where ΔSOC[c] = charge_state[c, end] - charge_state[c, start] (change during typical period c)
    - Cyclic constraint: SOC_boundary[0] = SOC_boundary[n_original_periods]
    """

    def __init__(
        self,
        model: FlowSystemModel,
        flow_system: FlowSystem,
        timestep_weights: np.ndarray,
        cluster_order: np.ndarray,
        nr_of_typical_periods: int,
        timesteps_per_period: int,
        storage_inter_period_linking: bool = True,
        storage_cyclic: bool = True,
    ):
        """
        Initialize the TypicalPeriodsModel.

        Args:
            model: The FlowSystemModel to add variables and constraints to.
            flow_system: The FlowSystem being modeled.
            timestep_weights: Array of weights for each timestep in the reduced model.
            cluster_order: Array mapping original period index to cluster/typical period index.
            nr_of_typical_periods: Number of typical periods (clusters).
            timesteps_per_period: Number of timesteps per typical period.
            storage_inter_period_linking: Whether to link storage states between periods.
            storage_cyclic: Whether to enforce cyclic constraint on storage boundaries.
        """
        super().__init__(model, label_of_element='TypicalPeriods', label_of_model='TypicalPeriods')
        self.flow_system = flow_system
        self.timestep_weights = timestep_weights
        self.cluster_order = cluster_order
        self.nr_of_typical_periods = nr_of_typical_periods
        self.timesteps_per_period = timesteps_per_period
        self.storage_inter_period_linking = storage_inter_period_linking
        self.storage_cyclic = storage_cyclic
        self.nr_of_original_periods = len(cluster_order)

    def do_modeling(self):
        """Create variables and constraints for typical periods optimization."""

        # Apply timestep weighting to operational effects
        self._apply_timestep_weighting()

        # Add inter-period storage linking if requested
        if self.storage_inter_period_linking:
            self._add_storage_linking()

    def _apply_timestep_weighting(self):
        """
        Log timestep weighting status.

        The actual timestep weights are applied in build_model() before do_modeling(),
        so that ShareAllocationModel can use them when summing temporal effects.
        This method just logs the weighting for transparency.
        """
        # Weights are already set on the model in build_model()
        # This method just logs the weighting configuration
        logger.info(
            f'Timestep weighting active: weights range from {self.timestep_weights.min()} '
            f'to {self.timestep_weights.max()}'
        )

    def _add_storage_linking(self):
        """
        Add inter-period storage state linking constraints.

        Creates SOC_boundary variables that track storage state at period boundaries
        and links them via the ΔSOC of each typical period.
        """

        storages = [comp for comp in self.flow_system.components.values() if isinstance(comp, Storage)]

        if not storages:
            logger.info('No storages found - skipping inter-period storage linking')
            return

        logger.info(f'Adding inter-period storage linking for {len(storages)} storage(s)')

        for storage in storages:
            self._add_storage_linking_for_component(storage)

    def _add_storage_linking_for_component(self, storage: Storage):
        """
        Add inter-period linking constraints for a single storage component.

        Creates:
        - SOC_boundary[d] variables for d = 0..n_original_periods (n+1 boundary points)
        - delta_SOC[c] variables for c = 0..n_typical_periods-1 (change during each typical period)
        - Linking constraints: SOC_boundary[d+1] = SOC_boundary[d] + delta_SOC[cluster_order[d]]
        - Optional cyclic constraint: SOC_boundary[0] = SOC_boundary[n_original_periods]

        Args:
            storage: The Storage component to add linking for.
        """
        import xarray as xr

        # Get storage capacity bounds for SOC_boundary limits
        if storage.capacity_in_flow_hours is None:
            # Unbounded storage
            capacity_lb, capacity_ub = 0, np.inf
        elif hasattr(storage.capacity_in_flow_hours, 'minimum_or_fixed_size'):
            # InvestParameters
            capacity_lb = storage.capacity_in_flow_hours.minimum_or_fixed_size
            capacity_ub = storage.capacity_in_flow_hours.maximum_or_fixed_size
        else:
            # Fixed capacity
            capacity_lb = capacity_ub = storage.capacity_in_flow_hours

        # Create SOC_boundary variables (one for each original period boundary)
        # Boundary d is BEFORE original period d starts, boundary n_original_periods is at the end
        boundary_indices = np.arange(self.nr_of_original_periods + 1)

        # Get relative charge state bounds (use scalar values for simplicity)
        rel_min = (
            float(storage.relative_minimum_charge_state.min().item())
            if hasattr(storage.relative_minimum_charge_state, 'min')
            else float(storage.relative_minimum_charge_state)
        )
        rel_max = (
            float(storage.relative_maximum_charge_state.max().item())
            if hasattr(storage.relative_maximum_charge_state, 'max')
            else float(storage.relative_maximum_charge_state)
        )

        # Create bounds as DataArrays with proper dimensions
        soc_lower = xr.DataArray(
            np.full(len(boundary_indices), rel_min * capacity_lb),
            dims=['boundary'],
            coords={'boundary': boundary_indices},
        )
        soc_upper = xr.DataArray(
            np.full(len(boundary_indices), rel_max * capacity_ub),
            dims=['boundary'],
            coords={'boundary': boundary_indices},
        )

        soc_boundary = self.add_variables(
            lower=soc_lower,
            upper=soc_upper,
            short_name=f'SOC_boundary|{storage.label}',
        )

        # Create delta_SOC variables (change in SOC during each typical period)
        # These are free variables (can be positive or negative)
        # Note: Using large bounds instead of inf for linopy compatibility
        max_delta = rel_max * capacity_ub if capacity_ub != np.inf else 1e9
        typical_period_indices = np.arange(self.nr_of_typical_periods)

        delta_lower = xr.DataArray(
            np.full(self.nr_of_typical_periods, -max_delta),
            dims=['typical_period'],
            coords={'typical_period': typical_period_indices},
        )
        delta_upper = xr.DataArray(
            np.full(self.nr_of_typical_periods, max_delta),
            dims=['typical_period'],
            coords={'typical_period': typical_period_indices},
        )

        delta_soc = self.add_variables(
            lower=delta_lower,
            upper=delta_upper,
            short_name=f'delta_SOC|{storage.label}',
        )

        # Link delta_SOC to actual charge state change during each typical period
        # delta_SOC[c] = charge_state[c, end] - charge_state[c, start]
        charge_state = storage.submodel.charge_state

        for c in range(self.nr_of_typical_periods):
            # Indices for this typical period's timesteps
            start_idx = c * self.timesteps_per_period
            end_idx = start_idx + self.timesteps_per_period  # charge_state has extra timestep

            # charge_state has n+1 timesteps (includes final state)
            # start state is at index start_idx, end state is at index end_idx
            self.add_constraints(
                delta_soc.isel(typical_period=c) == charge_state.isel(time=end_idx) - charge_state.isel(time=start_idx),
                short_name=f'delta_SOC_def|{storage.label}|tp{c}',
            )

        # Link SOC_boundary variables via delta_SOC
        # SOC_boundary[d+1] = SOC_boundary[d] + delta_SOC[cluster_order[d]]
        for d in range(self.nr_of_original_periods):
            cluster_idx = int(self.cluster_order[d])
            self.add_constraints(
                soc_boundary.isel(boundary=d + 1)
                == soc_boundary.isel(boundary=d) + delta_soc.isel(typical_period=cluster_idx),
                short_name=f'SOC_boundary_link|{storage.label}|d{d}',
            )

        # Cyclic constraint: SOC_boundary[0] = SOC_boundary[n_original_periods]
        if self.storage_cyclic:
            self.add_constraints(
                soc_boundary.isel(boundary=0) == soc_boundary.isel(boundary=self.nr_of_original_periods),
                short_name=f'SOC_boundary_cyclic|{storage.label}',
            )
            logger.debug(f'Added cyclic storage constraint for {storage.label}')

        # Link the typical period's initial charge state to SOC_boundary
        # This ensures each typical period starts from a feasible state
        # We use the first boundary that maps to each typical period
        # Actually, we need to be careful here - the typical period's charge state
        # should be "free floating" within its bounds, with only delta_SOC mattering
        # for inter-period linking. So we don't constrain the absolute starting point.

        logger.debug(
            f'Added inter-period storage linking for {storage.label}: '
            f'{self.nr_of_original_periods + 1} boundary variables, '
            f'{self.nr_of_typical_periods} delta_SOC variables'
        )
