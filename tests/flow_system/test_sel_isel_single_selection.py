"""Tests for sel/isel with single period/scenario selection."""

import numpy as np
import pandas as pd
import pytest

import flixopt as fx


@pytest.fixture
def fs_with_scenarios():
    """FlowSystem with scenarios for testing single selection."""
    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
    scenarios = pd.Index(['A', 'B', 'C'], name='scenario')
    scenario_weights = np.array([0.5, 0.3, 0.2])

    fs = fx.FlowSystem(timesteps, scenarios=scenarios, scenario_weights=scenario_weights)
    fs.add_elements(
        fx.Bus('heat'),
        fx.Effect('costs', unit='EUR', description='costs', is_objective=True, is_standard=True),
    )
    fs.add_elements(
        fx.Sink('demand', inputs=[fx.Flow('in', bus='heat', fixed_relative_profile=np.ones(24), size=10)]),
        fx.Source('source', outputs=[fx.Flow('out', bus='heat', size=50, effects_per_flow_hour={'costs': 0.05})]),
    )
    return fs


@pytest.fixture
def fs_with_periods():
    """FlowSystem with periods for testing single selection."""
    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
    periods = pd.Index([2020, 2030, 2040], name='period')

    fs = fx.FlowSystem(timesteps, periods=periods, weight_of_last_period=10)
    fs.add_elements(
        fx.Bus('heat'),
        fx.Effect('costs', unit='EUR', description='costs', is_objective=True, is_standard=True),
    )
    fs.add_elements(
        fx.Sink('demand', inputs=[fx.Flow('in', bus='heat', fixed_relative_profile=np.ones(24), size=10)]),
        fx.Source('source', outputs=[fx.Flow('out', bus='heat', size=50, effects_per_flow_hour={'costs': 0.05})]),
    )
    return fs


@pytest.fixture
def fs_with_periods_and_scenarios():
    """FlowSystem with both periods and scenarios."""
    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
    periods = pd.Index([2020, 2030], name='period')
    scenarios = pd.Index(['Low', 'High'], name='scenario')

    fs = fx.FlowSystem(timesteps, periods=periods, scenarios=scenarios, weight_of_last_period=10)
    fs.add_elements(
        fx.Bus('heat'),
        fx.Effect('costs', unit='EUR', description='costs', is_objective=True, is_standard=True),
    )
    fs.add_elements(
        fx.Sink('demand', inputs=[fx.Flow('in', bus='heat', fixed_relative_profile=np.ones(24), size=10)]),
        fx.Source('source', outputs=[fx.Flow('out', bus='heat', size=50, effects_per_flow_hour={'costs': 0.05})]),
    )
    return fs


class TestIselSingleScenario:
    """Test isel with single scenario selection."""

    def test_isel_single_scenario_drops_dimension(self, fs_with_scenarios):
        """Selecting a single scenario with isel should drop the scenario dimension."""
        fs_selected = fs_with_scenarios.transform.isel(scenario=0)

        assert fs_selected.scenarios is None
        assert 'scenario' not in fs_selected.to_dataset().dims

    def test_isel_single_scenario_removes_scenario_weights(self, fs_with_scenarios):
        """scenario_weights should be removed when scenario dimension is dropped."""
        fs_selected = fs_with_scenarios.transform.isel(scenario=0)

        ds = fs_selected.to_dataset()
        assert 'scenario_weights' not in ds.data_vars
        assert 'scenario_weights' not in ds.attrs

    def test_isel_single_scenario_preserves_time(self, fs_with_scenarios):
        """Time dimension should be preserved."""
        fs_selected = fs_with_scenarios.transform.isel(scenario=0)

        assert len(fs_selected.timesteps) == 24

    def test_isel_single_scenario_roundtrip(self, fs_with_scenarios):
        """FlowSystem should survive to_dataset/from_dataset roundtrip after single selection."""
        fs_selected = fs_with_scenarios.transform.isel(scenario=0)

        ds = fs_selected.to_dataset()
        fs_restored = fx.FlowSystem.from_dataset(ds)

        assert fs_restored.scenarios is None
        assert len(fs_restored.timesteps) == 24


class TestSelSingleScenario:
    """Test sel with single scenario selection."""

    def test_sel_single_scenario_drops_dimension(self, fs_with_scenarios):
        """Selecting a single scenario with sel should drop the scenario dimension."""
        fs_selected = fs_with_scenarios.transform.sel(scenario='B')

        assert fs_selected.scenarios is None


class TestIselSinglePeriod:
    """Test isel with single period selection."""

    def test_isel_single_period_drops_dimension(self, fs_with_periods):
        """Selecting a single period with isel should drop the period dimension."""
        fs_selected = fs_with_periods.transform.isel(period=0)

        assert fs_selected.periods is None
        assert 'period' not in fs_selected.to_dataset().dims

    def test_isel_single_period_removes_period_weights(self, fs_with_periods):
        """period_weights should be removed when period dimension is dropped."""
        fs_selected = fs_with_periods.transform.isel(period=0)

        ds = fs_selected.to_dataset()
        assert 'period_weights' not in ds.data_vars
        assert 'weight_of_last_period' not in ds.attrs

    def test_isel_single_period_roundtrip(self, fs_with_periods):
        """FlowSystem should survive roundtrip after single period selection."""
        fs_selected = fs_with_periods.transform.isel(period=0)

        ds = fs_selected.to_dataset()
        fs_restored = fx.FlowSystem.from_dataset(ds)

        assert fs_restored.periods is None


class TestSelSinglePeriod:
    """Test sel with single period selection."""

    def test_sel_single_period_drops_dimension(self, fs_with_periods):
        """Selecting a single period with sel should drop the period dimension."""
        fs_selected = fs_with_periods.transform.sel(period=2030)

        assert fs_selected.periods is None


class TestMixedSelection:
    """Test mixed selections (single + multiple)."""

    def test_single_period_multiple_scenarios(self, fs_with_periods_and_scenarios):
        """Single period but multiple scenarios should only drop period."""
        fs_selected = fs_with_periods_and_scenarios.transform.isel(period=0)

        assert fs_selected.periods is None
        assert fs_selected.scenarios is not None
        assert len(fs_selected.scenarios) == 2

    def test_multiple_periods_single_scenario(self, fs_with_periods_and_scenarios):
        """Multiple periods but single scenario should only drop scenario."""
        fs_selected = fs_with_periods_and_scenarios.transform.isel(scenario=0)

        assert fs_selected.periods is not None
        assert len(fs_selected.periods) == 2
        assert fs_selected.scenarios is None

    def test_single_period_single_scenario(self, fs_with_periods_and_scenarios):
        """Single period and single scenario should drop both."""
        fs_selected = fs_with_periods_and_scenarios.transform.isel(period=0, scenario=0)

        assert fs_selected.periods is None
        assert fs_selected.scenarios is None


class TestSliceSelection:
    """Test that slice selection preserves dimensions."""

    def test_slice_scenarios_preserves_dimension(self, fs_with_scenarios):
        """Slice selection should preserve dimension even with 1 element."""
        # Select a slice that results in 2 elements
        fs_selected = fs_with_scenarios.transform.isel(scenario=slice(0, 2))

        assert fs_selected.scenarios is not None
        assert len(fs_selected.scenarios) == 2

    def test_list_selection_preserves_dimension(self, fs_with_scenarios):
        """List selection should preserve dimension even with 1 element."""
        fs_selected = fs_with_scenarios.transform.isel(scenario=[0])

        # List selection should preserve dimension
        assert fs_selected.scenarios is not None
        assert len(fs_selected.scenarios) == 1
