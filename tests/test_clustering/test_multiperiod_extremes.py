"""Tests for clustering multi-period flow systems with different time series and extreme configurations."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose
from tsam import ExtremeConfig, SegmentConfig

import flixopt as fx

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def timesteps_8_days():
    """192 hour timesteps (8 days) for clustering tests."""
    return pd.date_range('2020-01-01', periods=192, freq='h')


@pytest.fixture
def timesteps_14_days():
    """336 hour timesteps (14 days) for more comprehensive clustering tests."""
    return pd.date_range('2020-01-01', periods=336, freq='h')


@pytest.fixture
def periods_2():
    """Two periods for testing."""
    return pd.Index([2025, 2030], name='period')


@pytest.fixture
def periods_3():
    """Three periods for testing."""
    return pd.Index([2025, 2030, 2035], name='period')


@pytest.fixture
def scenarios_2():
    """Two scenarios for testing."""
    return pd.Index(['low', 'high'], name='scenario')


@pytest.fixture
def scenarios_3():
    """Three scenarios for testing."""
    return pd.Index(['low', 'medium', 'high'], name='scenario')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_multiperiod_system_with_different_profiles(
    timesteps: pd.DatetimeIndex,
    periods: pd.Index,
) -> fx.FlowSystem:
    """Create a multi-period FlowSystem with different demand profiles per period.

    Each period has a distinctly different demand pattern to test that clustering
    produces different cluster assignments per period.
    """
    hours = len(timesteps)
    hour_of_day = np.array([t.hour for t in timesteps])
    day_idx = np.arange(hours) // 24

    # Create different demand profiles for each period
    demand_data = {}
    for i, period in enumerate(periods):
        # Base pattern varies by hour
        base = np.where((hour_of_day >= 8) & (hour_of_day < 20), 25, 8)

        # Add period-specific variation:
        # - Period 0: Higher morning peaks
        # - Period 1: Higher evening peaks
        # - Period 2+: Flatter profile with higher base
        if i == 0:
            # Morning peak pattern
            morning_boost = np.where((hour_of_day >= 6) & (hour_of_day < 10), 15, 0)
            demand = base + morning_boost
        elif i == 1:
            # Evening peak pattern
            evening_boost = np.where((hour_of_day >= 17) & (hour_of_day < 21), 20, 0)
            demand = base + evening_boost
        else:
            # Flatter profile
            demand = base * 0.8 + 10

        # Add day-to-day variation for clustering diversity
        demand = demand * (1 + 0.2 * (day_idx % 3))
        demand_data[period] = demand

    # Create xarray DataArray with period dimension
    demand_array = np.column_stack([demand_data[p] for p in periods])
    demand_da = xr.DataArray(
        demand_array,
        dims=['time', 'period'],
        coords={'time': timesteps, 'period': periods},
    )

    flow_system = fx.FlowSystem(timesteps, periods=periods)
    flow_system.add_elements(
        fx.Bus('Heat'),
        fx.Bus('Gas'),
        fx.Effect('costs', '€', is_standard=True, is_objective=True),
        fx.Sink(
            'HeatDemand',
            inputs=[fx.Flow('Q', bus='Heat', fixed_relative_profile=demand_da, size=1)],
        ),
        fx.Source('GasSource', outputs=[fx.Flow('Gas', bus='Gas', effects_per_flow_hour=0.05)]),
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.9,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow('Q_th', bus='Heat'),
        ),
    )
    return flow_system


def create_system_with_extreme_peaks(
    timesteps: pd.DatetimeIndex,
    periods: pd.Index | None = None,
    scenarios: pd.Index | None = None,
    peak_day: int = 5,
    peak_magnitude: float = 100,
) -> fx.FlowSystem:
    """Create a FlowSystem with clearly identifiable extreme peak days.

    Args:
        timesteps: Time coordinates.
        periods: Optional period dimension.
        scenarios: Optional scenario dimension.
        peak_day: Which day (0-indexed) should have the extreme peak.
        peak_magnitude: Magnitude of the peak demand.
    """
    hours = len(timesteps)
    hour_of_day = np.arange(hours) % 24
    day_idx = np.arange(hours) // 24

    # Base demand pattern
    base_demand = np.where((hour_of_day >= 8) & (hour_of_day < 18), 20, 8)

    # Add extreme peak on specified day during hours 10-14
    peak_mask = (day_idx == peak_day) & (hour_of_day >= 10) & (hour_of_day < 14)
    demand = np.where(peak_mask, peak_magnitude, base_demand)

    # Add moderate variation to other days
    demand = demand * (1 + 0.15 * (day_idx % 3))

    # Handle multi-dimensional cases
    if periods is not None and scenarios is not None:
        # Create 3D array: (time, period, scenario)
        demand_3d = np.zeros((hours, len(periods), len(scenarios)))
        for i, _period in enumerate(periods):
            for j, _scenario in enumerate(scenarios):
                # Scale demand by period and scenario
                scale = (1 + 0.1 * i) * (1 + 0.15 * j)
                demand_3d[:, i, j] = demand * scale
        demand_input = xr.DataArray(
            demand_3d,
            dims=['time', 'period', 'scenario'],
            coords={'time': timesteps, 'period': periods, 'scenario': scenarios},
        )
        flow_system = fx.FlowSystem(timesteps, periods=periods, scenarios=scenarios)
    elif periods is not None:
        # Create 2D array: (time, period)
        demand_2d = np.column_stack([demand * (1 + 0.1 * i) for i in range(len(periods))])
        demand_input = xr.DataArray(
            demand_2d,
            dims=['time', 'period'],
            coords={'time': timesteps, 'period': periods},
        )
        flow_system = fx.FlowSystem(timesteps, periods=periods)
    elif scenarios is not None:
        # Create 2D array: (time, scenario)
        demand_2d = np.column_stack([demand * (1 + 0.15 * j) for j in range(len(scenarios))])
        demand_input = xr.DataArray(
            demand_2d,
            dims=['time', 'scenario'],
            coords={'time': timesteps, 'scenario': scenarios},
        )
        flow_system = fx.FlowSystem(timesteps, scenarios=scenarios)
    else:
        demand_input = demand
        flow_system = fx.FlowSystem(timesteps)

    flow_system.add_elements(
        fx.Bus('Heat'),
        fx.Bus('Gas'),
        fx.Effect('costs', '€', is_standard=True, is_objective=True),
        fx.Sink(
            'HeatDemand',
            inputs=[fx.Flow('Q', bus='Heat', fixed_relative_profile=demand_input, size=1)],
        ),
        fx.Source('GasSource', outputs=[fx.Flow('Gas', bus='Gas', effects_per_flow_hour=0.05)]),
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.9,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow('Q_th', bus='Heat'),
        ),
    )
    return flow_system


def create_multiperiod_multiscenario_system(
    timesteps: pd.DatetimeIndex,
    periods: pd.Index,
    scenarios: pd.Index,
) -> fx.FlowSystem:
    """Create a FlowSystem with both periods and scenarios dimensions."""
    hours = len(timesteps)
    hour_of_day = np.array([t.hour for t in timesteps])
    day_idx = np.arange(hours) // 24

    # Create 3D demand array: (time, period, scenario)
    demand_3d = np.zeros((hours, len(periods), len(scenarios)))

    for i, _period in enumerate(periods):
        for j, _scenario in enumerate(scenarios):
            # Base pattern
            base = np.where((hour_of_day >= 8) & (hour_of_day < 18), 20, 8)

            # Period variation: demand growth over time
            period_factor = 1 + 0.15 * i

            # Scenario variation: different load levels
            scenario_factor = 0.8 + 0.2 * j

            # Day variation for clustering
            day_factor = 1 + 0.2 * (day_idx % 4)

            demand_3d[:, i, j] = base * period_factor * scenario_factor * day_factor

    demand_da = xr.DataArray(
        demand_3d,
        dims=['time', 'period', 'scenario'],
        coords={'time': timesteps, 'period': periods, 'scenario': scenarios},
    )

    flow_system = fx.FlowSystem(timesteps, periods=periods, scenarios=scenarios)
    flow_system.add_elements(
        fx.Bus('Heat'),
        fx.Bus('Gas'),
        fx.Effect('costs', '€', is_standard=True, is_objective=True),
        fx.Sink(
            'HeatDemand',
            inputs=[fx.Flow('Q', bus='Heat', fixed_relative_profile=demand_da, size=1)],
        ),
        fx.Source('GasSource', outputs=[fx.Flow('Gas', bus='Gas', effects_per_flow_hour=0.05)]),
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.9,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow('Q_th', bus='Heat'),
        ),
    )
    return flow_system


# ============================================================================
# MULTI-PERIOD CLUSTERING WITH DIFFERENT TIME SERIES
# ============================================================================


class TestMultiPeriodDifferentTimeSeries:
    """Tests for clustering multi-period systems where each period has different time series."""

    def test_different_profiles_create_different_assignments(self, timesteps_8_days, periods_2):
        """Test that different demand profiles per period lead to different cluster assignments."""
        fs = create_multiperiod_system_with_different_profiles(timesteps_8_days, periods_2)

        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        # Verify clustering structure
        assert fs_clustered.periods is not None
        assert len(fs_clustered.periods) == 2
        assert fs_clustered.clustering is not None

        # Cluster assignments should have period dimension
        cluster_assignments = fs_clustered.clustering.cluster_assignments
        assert 'period' in cluster_assignments.dims

        # Each period should have n_original_clusters assignments
        n_original_clusters = 8  # 8 days
        for period in periods_2:
            period_assignments = cluster_assignments.sel(period=period)
            assert len(period_assignments) == n_original_clusters

    def test_different_profiles_can_be_optimized(self, solver_fixture, timesteps_8_days, periods_2):
        """Test that multi-period systems with different profiles optimize correctly."""
        fs = create_multiperiod_system_with_different_profiles(timesteps_8_days, periods_2)

        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        assert fs_clustered.solution is not None

        # Solution should have period dimension
        flow_var = 'flow|rate'
        assert flow_var in fs_clustered.solution
        assert 'period' in fs_clustered.solution[flow_var].dims

    def test_different_profiles_expand_correctly(self, solver_fixture, timesteps_8_days, periods_2):
        """Test that expansion handles period-specific cluster assignments."""
        fs = create_multiperiod_system_with_different_profiles(timesteps_8_days, periods_2)

        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        fs_expanded = fs_clustered.transform.expand()

        # Should have original timesteps
        assert len(fs_expanded.timesteps) == 192

        # Should have period dimension preserved
        assert fs_expanded.periods is not None
        assert len(fs_expanded.periods) == 2

        # Each period should map using its own cluster assignments
        flow_var = 'flow|rate'
        for period in periods_2:
            flow_period = fs_expanded.solution[flow_var].sel(period=period)
            assert len(flow_period.coords['time']) == 193  # 192 + 1 extra

    def test_three_periods_with_different_profiles(self, solver_fixture, timesteps_8_days, periods_3):
        """Test clustering with three periods, each having different demand characteristics."""
        fs = create_multiperiod_system_with_different_profiles(timesteps_8_days, periods_3)

        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        # Verify 3 periods
        assert len(fs_clustered.periods) == 3

        # Cluster assignments should span all periods
        cluster_assignments = fs_clustered.clustering.cluster_assignments
        assert cluster_assignments.sizes['period'] == 3

        # Optimize and expand
        fs_clustered.optimize(solver_fixture)
        fs_expanded = fs_clustered.transform.expand()

        assert len(fs_expanded.periods) == 3
        assert len(fs_expanded.timesteps) == 192

    def test_statistics_correct_per_period(self, solver_fixture, timesteps_8_days, periods_2):
        """Test that statistics are computed correctly for each period."""
        fs = create_multiperiod_system_with_different_profiles(timesteps_8_days, periods_2)

        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Get stats from clustered system
        total_effects_clustered = fs_clustered.stats.total_effects.sel(effect='costs')

        # Expand and get stats
        fs_expanded = fs_clustered.transform.expand()
        total_effects_expanded = fs_expanded.stats.total_effects.sel(effect='costs')

        # Total effects should match between clustered and expanded
        assert_allclose(
            total_effects_clustered.sum('contributor').values,
            total_effects_expanded.sum('contributor').values,
            rtol=1e-5,
        )


# ============================================================================
# EXTREME CLUSTER CONFIGURATION TESTS
# ============================================================================


class TestExtremeConfigNewCluster:
    """Tests for ExtremeConfig with method='new_cluster'."""

    def test_new_cluster_captures_peak_day(self, solver_fixture, timesteps_8_days):
        """Test that new_cluster method captures extreme peak day."""
        fs = create_system_with_extreme_peaks(timesteps_8_days, peak_day=5, peak_magnitude=100)

        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            extremes=ExtremeConfig(
                method='new_cluster',
                max_value=['HeatDemand(Q)|fixed_relative_profile'],
            ),
        )

        fs_clustered.optimize(solver_fixture)

        # The peak should be captured in the solution
        flow_rates = fs_clustered.solution['flow|rate'].sel(flow='Boiler(Q_th)')
        max_flow = float(flow_rates.max())
        # Peak demand is ~100, boiler efficiency 0.9, so max flow should be ~100
        assert max_flow >= 90, f'Peak not captured: max_flow={max_flow}'

    def test_new_cluster_can_increase_cluster_count(self, timesteps_8_days):
        """Test that new_cluster may increase the effective cluster count."""
        fs = create_system_with_extreme_peaks(timesteps_8_days, peak_day=5, peak_magnitude=150)

        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            extremes=ExtremeConfig(
                method='new_cluster',
                max_value=['HeatDemand(Q)|fixed_relative_profile'],
            ),
        )

        # n_clusters should be >= 2 (may be higher with extreme periods)
        assert fs_clustered.clustering.n_clusters >= 2

        # Sum of occurrences should equal original clusters (8 days)
        assert int(fs_clustered.clustering.cluster_occurrences.sum()) == 8

    def test_new_cluster_with_min_value(self, solver_fixture, timesteps_8_days):
        """Test new_cluster with min_value parameter."""
        # Create system with low demand day
        hours = len(timesteps_8_days)
        hour_of_day = np.arange(hours) % 24
        day_idx = np.arange(hours) // 24

        # Normal demand with one very low day
        demand = np.where((hour_of_day >= 8) & (hour_of_day < 18), 25, 10)
        low_day_mask = day_idx == 3
        demand = np.where(low_day_mask, 2, demand)  # Very low on day 3

        fs = fx.FlowSystem(timesteps_8_days)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink('HeatDemand', inputs=[fx.Flow('Q', bus='Heat', fixed_relative_profile=demand, size=1)]),
            fx.Source('GasSource', outputs=[fx.Flow('Gas', bus='Gas', effects_per_flow_hour=0.05)]),
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=0.9,
                fuel_flow=fx.Flow('Q_fu', bus='Gas'),
                thermal_flow=fx.Flow('Q_th', bus='Heat'),
            ),
        )

        fs_clustered = fs.transform.cluster(
            n_clusters=3,
            cluster_duration='1D',
            extremes=ExtremeConfig(
                method='new_cluster',
                min_value=['HeatDemand(Q)|fixed_relative_profile'],
            ),
        )

        assert fs_clustered.clustering.n_clusters >= 3
        fs_clustered.optimize(solver_fixture)
        assert fs_clustered.solution is not None


class TestExtremeConfigReplace:
    """Tests for ExtremeConfig with method='replace'."""

    def test_replace_maintains_cluster_count(self, solver_fixture, timesteps_8_days):
        """Test that replace method maintains the requested cluster count."""
        fs = create_system_with_extreme_peaks(timesteps_8_days, peak_day=5, peak_magnitude=100)

        fs_clustered = fs.transform.cluster(
            n_clusters=3,
            cluster_duration='1D',
            extremes=ExtremeConfig(
                method='replace',
                max_value=['HeatDemand(Q)|fixed_relative_profile'],
            ),
        )

        # Replace should maintain exactly n_clusters
        assert fs_clustered.clustering.n_clusters == 3

        fs_clustered.optimize(solver_fixture)
        assert fs_clustered.solution is not None

    def test_replace_with_multiperiod(self, solver_fixture, timesteps_8_days, periods_2):
        """Test replace method with multi-period system."""
        fs = create_system_with_extreme_peaks(timesteps_8_days, periods=periods_2, peak_day=5)

        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            extremes=ExtremeConfig(
                method='replace',
                max_value=['HeatDemand(Q)|fixed_relative_profile'],
            ),
        )

        assert fs_clustered.clustering.n_clusters == 2
        assert len(fs_clustered.periods) == 2

        fs_clustered.optimize(solver_fixture)
        assert fs_clustered.solution is not None


class TestExtremeConfigAppend:
    """Tests for ExtremeConfig with method='append'."""

    def test_append_with_segments(self, solver_fixture, timesteps_8_days):
        """Test append method combined with segmentation."""
        fs = create_system_with_extreme_peaks(timesteps_8_days, peak_day=5, peak_magnitude=80)

        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            extremes=ExtremeConfig(
                method='append',
                max_value=['HeatDemand(Q)|fixed_relative_profile'],
            ),
            segments=SegmentConfig(n_segments=4),
        )

        # Verify segmentation
        assert fs_clustered.clustering.is_segmented is True
        assert fs_clustered.clustering.n_segments == 4

        # n_representatives = n_clusters * n_segments
        n_clusters = fs_clustered.clustering.n_clusters
        assert fs_clustered.clustering.n_representatives == n_clusters * 4

        fs_clustered.optimize(solver_fixture)
        assert fs_clustered.solution is not None

    def test_append_expand_preserves_objective(self, solver_fixture, timesteps_8_days):
        """Test that expansion after append preserves objective value."""
        fs = create_system_with_extreme_peaks(timesteps_8_days, peak_day=5, peak_magnitude=80)

        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            extremes=ExtremeConfig(
                method='append',
                max_value=['HeatDemand(Q)|fixed_relative_profile'],
            ),
            segments=SegmentConfig(n_segments=4),
        )

        fs_clustered.optimize(solver_fixture)
        clustered_objective = fs_clustered.solution['objective'].item()

        fs_expanded = fs_clustered.transform.expand()
        expanded_objective = fs_expanded.solution['objective'].item()

        assert_allclose(clustered_objective, expanded_objective, rtol=1e-5)


class TestExtremeConfigMultiPeriod:
    """Tests for extreme configurations with multi-period systems."""

    def test_extremes_require_replace_method_multiperiod(self, timesteps_8_days, periods_2):
        """Test that only method='replace' is allowed for multi-period systems."""
        fs = create_system_with_extreme_peaks(timesteps_8_days, periods=periods_2)

        # method='new_cluster' should be rejected
        with pytest.raises(ValueError, match="method='new_cluster'.*not supported"):
            fs.transform.cluster(
                n_clusters=2,
                cluster_duration='1D',
                extremes=ExtremeConfig(
                    method='new_cluster',
                    max_value=['HeatDemand(Q)|fixed_relative_profile'],
                ),
            )

        # method='append' should also be rejected
        with pytest.raises(ValueError, match="method='append'.*not supported"):
            fs.transform.cluster(
                n_clusters=2,
                cluster_duration='1D',
                extremes=ExtremeConfig(
                    method='append',
                    max_value=['HeatDemand(Q)|fixed_relative_profile'],
                ),
            )

    def test_extremes_with_replace_multiperiod(self, solver_fixture, timesteps_8_days, periods_2):
        """Test that extremes work with method='replace' for multi-period."""
        fs = create_system_with_extreme_peaks(timesteps_8_days, periods=periods_2)

        # Only method='replace' is allowed for multi-period systems
        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            extremes=ExtremeConfig(
                method='replace',
                max_value=['HeatDemand(Q)|fixed_relative_profile'],
            ),
        )

        assert fs_clustered.clustering.n_clusters == 2
        assert len(fs_clustered.periods) == 2

        fs_clustered.optimize(solver_fixture)
        assert fs_clustered.solution is not None

    def test_extremes_with_periods_and_scenarios(self, solver_fixture, timesteps_8_days, periods_2, scenarios_2):
        """Test extremes with both periods and scenarios."""
        fs = create_system_with_extreme_peaks(
            timesteps_8_days,
            periods=periods_2,
            scenarios=scenarios_2,
            peak_day=5,
        )

        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            extremes=ExtremeConfig(
                method='replace',
                max_value=['HeatDemand(Q)|fixed_relative_profile'],
            ),
        )

        # Verify dimensions
        assert len(fs_clustered.periods) == 2
        assert len(fs_clustered.scenarios) == 2
        assert fs_clustered.clustering.n_clusters == 2

        fs_clustered.optimize(solver_fixture)

        # Solution should have both dimensions
        flow_var = 'flow|rate'
        assert 'period' in fs_clustered.solution[flow_var].dims
        assert 'scenario' in fs_clustered.solution[flow_var].dims


# ============================================================================
# COMBINED MULTI-PERIOD AND EXTREME TESTS
# ============================================================================


class TestMultiPeriodWithExtremes:
    """Tests combining multi-period systems with extreme configurations."""

    def test_different_profiles_with_extremes(self, solver_fixture, timesteps_8_days, periods_2):
        """Test multi-period with different profiles AND extreme capture."""
        fs = create_multiperiod_system_with_different_profiles(timesteps_8_days, periods_2)

        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            extremes=ExtremeConfig(
                method='replace',
                max_value=['HeatDemand(Q)|fixed_relative_profile'],
            ),
        )

        assert fs_clustered.clustering.n_clusters == 2
        assert len(fs_clustered.periods) == 2

        fs_clustered.optimize(solver_fixture)
        fs_expanded = fs_clustered.transform.expand()

        # Verify expansion
        assert len(fs_expanded.timesteps) == 192
        assert len(fs_expanded.periods) == 2

    def test_multiperiod_extremes_with_segmentation(self, solver_fixture, timesteps_8_days, periods_2):
        """Test multi-period with extremes and segmentation."""
        fs = create_multiperiod_system_with_different_profiles(timesteps_8_days, periods_2)

        # Note: method='replace' is required for multi-period systems (method='append' has tsam bug)
        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            extremes=ExtremeConfig(
                method='replace',
                max_value=['HeatDemand(Q)|fixed_relative_profile'],
            ),
            segments=SegmentConfig(n_segments=6),
        )

        # Verify structure
        assert fs_clustered.clustering.is_segmented is True
        assert fs_clustered.clustering.n_segments == 6
        assert len(fs_clustered.periods) == 2

        fs_clustered.optimize(solver_fixture)

        # Verify expansion
        fs_expanded = fs_clustered.transform.expand()
        assert len(fs_expanded.timesteps) == 192

    def test_cluster_assignments_independent_per_period(self, timesteps_8_days, periods_3):
        """Test that each period gets independent cluster assignments."""
        fs = create_multiperiod_system_with_different_profiles(timesteps_8_days, periods_3)

        fs_clustered = fs.transform.cluster(n_clusters=3, cluster_duration='1D')

        cluster_assignments = fs_clustered.clustering.cluster_assignments

        # Each period should have its own assignments
        assert 'period' in cluster_assignments.dims
        assert cluster_assignments.sizes['period'] == 3

        # Assignments are computed independently per period
        # (may or may not be different depending on the data)
        for period in periods_3:
            period_assignments = cluster_assignments.sel(period=period)
            # Should have 8 assignments (one per original day)
            assert len(period_assignments) == 8
            # Each assignment should be in range [0, n_clusters-1]
            assert period_assignments.min() >= 0
            assert period_assignments.max() < 3


# ============================================================================
# MULTI-SCENARIO WITH CLUSTERING TESTS
# ============================================================================


class TestMultiScenarioWithClustering:
    """Tests for clustering systems with scenario dimension."""

    def test_cluster_with_scenarios(self, solver_fixture, timesteps_8_days, scenarios_2):
        """Test clustering with scenarios dimension."""
        hours = len(timesteps_8_days)
        demand_data = np.column_stack(
            [np.sin(np.linspace(0, 4 * np.pi, hours)) * 10 + 15 * (1 + 0.2 * i) for i in range(len(scenarios_2))]
        )
        demand_da = xr.DataArray(
            demand_data,
            dims=['time', 'scenario'],
            coords={'time': timesteps_8_days, 'scenario': scenarios_2},
        )

        fs = fx.FlowSystem(timesteps_8_days, scenarios=scenarios_2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink('HeatDemand', inputs=[fx.Flow('Q', bus='Heat', fixed_relative_profile=demand_da, size=1)]),
            fx.Source('GasSource', outputs=[fx.Flow('Gas', bus='Gas', effects_per_flow_hour=0.05)]),
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=0.9,
                fuel_flow=fx.Flow('Q_fu', bus='Gas'),
                thermal_flow=fx.Flow('Q_th', bus='Heat'),
            ),
        )

        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        assert len(fs_clustered.scenarios) == 2
        assert fs_clustered.clustering.n_clusters == 2

        fs_clustered.optimize(solver_fixture)

        # Solution should have scenario dimension
        flow_var = 'flow|rate'
        assert 'scenario' in fs_clustered.solution[flow_var].dims

    def test_scenarios_with_extremes(self, solver_fixture, timesteps_8_days, scenarios_2):
        """Test scenarios combined with extreme configuration."""
        fs = create_system_with_extreme_peaks(timesteps_8_days, scenarios=scenarios_2, peak_day=5)

        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            extremes=ExtremeConfig(
                method='replace',
                max_value=['HeatDemand(Q)|fixed_relative_profile'],
            ),
        )

        assert len(fs_clustered.scenarios) == 2
        fs_clustered.optimize(solver_fixture)
        assert fs_clustered.solution is not None


class TestFullDimensionalClustering:
    """Tests for clustering with all dimensions (periods + scenarios)."""

    def test_periods_and_scenarios_clustering(self, solver_fixture, timesteps_8_days, periods_2, scenarios_2):
        """Test clustering with both periods and scenarios."""
        fs = create_multiperiod_multiscenario_system(timesteps_8_days, periods_2, scenarios_2)

        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')

        # Verify all dimensions
        assert len(fs_clustered.periods) == 2
        assert len(fs_clustered.scenarios) == 2
        assert fs_clustered.clustering.n_clusters == 2

        # Cluster assignments should have both dimensions
        cluster_assignments = fs_clustered.clustering.cluster_assignments
        assert 'period' in cluster_assignments.dims
        assert 'scenario' in cluster_assignments.dims

        fs_clustered.optimize(solver_fixture)

        # Solution should have all dimensions
        flow_var = 'flow|rate'
        assert 'period' in fs_clustered.solution[flow_var].dims
        assert 'scenario' in fs_clustered.solution[flow_var].dims
        assert 'cluster' in fs_clustered.solution[flow_var].dims

    def test_full_dimensional_expand(self, solver_fixture, timesteps_8_days, periods_2, scenarios_2):
        """Test expansion of system with all dimensions."""
        fs = create_multiperiod_multiscenario_system(timesteps_8_days, periods_2, scenarios_2)

        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        fs_expanded = fs_clustered.transform.expand()

        # Verify all dimensions preserved after expansion
        assert len(fs_expanded.timesteps) == 192
        assert len(fs_expanded.periods) == 2
        assert len(fs_expanded.scenarios) == 2

        # Solution should maintain dimensions
        flow_var = 'flow|rate'
        assert 'period' in fs_expanded.solution[flow_var].dims
        assert 'scenario' in fs_expanded.solution[flow_var].dims

    def test_full_dimensional_with_extremes(self, solver_fixture, timesteps_8_days, periods_2, scenarios_2):
        """Test full dimensional system with extreme configuration."""
        fs = create_multiperiod_multiscenario_system(timesteps_8_days, periods_2, scenarios_2)

        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            extremes=ExtremeConfig(
                method='replace',
                max_value=['HeatDemand(Q)|fixed_relative_profile'],
            ),
        )

        assert fs_clustered.clustering.n_clusters == 2

        fs_clustered.optimize(solver_fixture)
        fs_expanded = fs_clustered.transform.expand()

        # Objectives should match
        assert_allclose(
            fs_clustered.solution['objective'].item(),
            fs_expanded.solution['objective'].item(),
            rtol=1e-5,
        )

    def test_full_dimensional_with_segmentation(self, solver_fixture, timesteps_8_days, periods_2, scenarios_2):
        """Test full dimensional system with segmentation."""
        fs = create_multiperiod_multiscenario_system(timesteps_8_days, periods_2, scenarios_2)

        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            segments=SegmentConfig(n_segments=6),
        )

        assert fs_clustered.clustering.is_segmented is True
        assert fs_clustered.clustering.n_segments == 6

        fs_clustered.optimize(solver_fixture)
        fs_expanded = fs_clustered.transform.expand()

        # Should restore original timesteps
        assert len(fs_expanded.timesteps) == 192


# ============================================================================
# IO ROUND-TRIP TESTS WITH MULTI-PERIOD
# ============================================================================


class TestMultiPeriodClusteringIO:
    """Tests for IO round-trip of multi-period clustered systems."""

    def test_multiperiod_clustering_roundtrip(self, solver_fixture, timesteps_8_days, periods_2, tmp_path):
        """Test that multi-period clustered system survives IO round-trip."""
        fs = create_multiperiod_system_with_different_profiles(timesteps_8_days, periods_2)

        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Save and load
        path = tmp_path / 'multiperiod_clustered.nc4'
        fs_clustered.to_netcdf(path)
        fs_loaded = fx.FlowSystem.from_netcdf(path)

        # Verify clustering preserved
        assert fs_loaded.clustering is not None
        assert fs_loaded.clustering.n_clusters == 2

        # Verify periods preserved
        assert fs_loaded.periods is not None
        assert len(fs_loaded.periods) == 2

        # Verify solution preserved
        assert_allclose(
            fs_loaded.solution['objective'].item(),
            fs_clustered.solution['objective'].item(),
            rtol=1e-6,
        )

    def test_multiperiod_expand_after_load(self, solver_fixture, timesteps_8_days, periods_2, tmp_path):
        """Test that expand works after loading multi-period clustered system."""
        fs = create_multiperiod_system_with_different_profiles(timesteps_8_days, periods_2)

        fs_clustered = fs.transform.cluster(n_clusters=2, cluster_duration='1D')
        fs_clustered.optimize(solver_fixture)

        # Save, load, and expand
        path = tmp_path / 'multiperiod_clustered.nc4'
        fs_clustered.to_netcdf(path)
        fs_loaded = fx.FlowSystem.from_netcdf(path)
        fs_expanded = fs_loaded.transform.expand()

        # Should have original timesteps
        assert len(fs_expanded.timesteps) == 192

        # Should have periods preserved
        assert len(fs_expanded.periods) == 2

    def test_extremes_preserved_after_io(self, solver_fixture, timesteps_8_days, periods_2, tmp_path):
        """Test that extremes configuration results are preserved after IO."""
        fs = create_system_with_extreme_peaks(timesteps_8_days, periods=periods_2, peak_day=5)

        fs_clustered = fs.transform.cluster(
            n_clusters=2,
            cluster_duration='1D',
            extremes=ExtremeConfig(
                method='replace',
                max_value=['HeatDemand(Q)|fixed_relative_profile'],
            ),
        )
        fs_clustered.optimize(solver_fixture)

        # Save and load
        path = tmp_path / 'extremes_clustered.nc4'
        fs_clustered.to_netcdf(path)
        fs_loaded = fx.FlowSystem.from_netcdf(path)

        # Clustering structure should be preserved
        assert fs_loaded.clustering.n_clusters == 2

        # Expand should work
        fs_expanded = fs_loaded.transform.expand()
        assert len(fs_expanded.timesteps) == 192


# ============================================================================
# EDGE CASES AND VALIDATION TESTS
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases in multi-period clustering."""

    def test_single_cluster_multiperiod(self, solver_fixture, timesteps_8_days, periods_2):
        """Test clustering with n_clusters=1 for multi-period system."""
        fs = create_multiperiod_system_with_different_profiles(timesteps_8_days, periods_2)

        fs_clustered = fs.transform.cluster(n_clusters=1, cluster_duration='1D')

        assert fs_clustered.clustering.n_clusters == 1
        assert len(fs_clustered.clusters) == 1

        # All days should be assigned to cluster 0
        cluster_assignments = fs_clustered.clustering.cluster_assignments
        assert (cluster_assignments == 0).all()

        fs_clustered.optimize(solver_fixture)
        assert fs_clustered.solution is not None

    def test_cluster_occurrences_sum_to_original(self, timesteps_8_days, periods_2):
        """Test that cluster occurrences always sum to original cluster count."""
        fs = create_multiperiod_system_with_different_profiles(timesteps_8_days, periods_2)

        for n_clusters in [1, 2, 4, 6]:
            fs_clustered = fs.transform.cluster(n_clusters=n_clusters, cluster_duration='1D')

            # For each period, occurrences should sum to 8 (original days)
            occurrences = fs_clustered.clustering.cluster_occurrences
            for period in periods_2:
                period_occurrences = occurrences.sel(period=period)
                assert int(period_occurrences.sum()) == 8, (
                    f'Occurrences for period {period} with n_clusters={n_clusters}: '
                    f'{int(period_occurrences.sum())} != 8'
                )

    def test_timestep_mapping_valid_range(self, timesteps_8_days, periods_2):
        """Test that timestep_mapping values are within valid range."""
        fs = create_multiperiod_system_with_different_profiles(timesteps_8_days, periods_2)

        fs_clustered = fs.transform.cluster(n_clusters=3, cluster_duration='1D')

        mapping = fs_clustered.clustering.timestep_mapping

        # Mapping values should be in [0, n_clusters * timesteps_per_cluster - 1]
        max_valid = 3 * 24 - 1  # n_clusters * timesteps_per_cluster - 1
        assert mapping.min().item() >= 0
        assert mapping.max().item() <= max_valid

        # Each period should have valid mappings
        for period in periods_2:
            period_mapping = mapping.sel(period=period)
            assert period_mapping.min().item() >= 0
            assert period_mapping.max().item() <= max_valid
