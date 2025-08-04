import linopy
import numpy as np
import pandas as pd
import pytest

import flixopt as fx

from .conftest import (
    Buses,
    Effects,
    LoadProfiles,
    Sinks,
    Sources,
    assert_conequal,
    assert_sets_equal,
    assert_var_equal,
    create_linopy_model,
)


@pytest.fixture
def flow_system() -> fx.FlowSystem:
    """Create basic elements for component testing with coordinate parametrization."""
    years = pd.Index([2020, 2021, 2022, 2023, 2024], name='year')
    timesteps = pd.date_range('2020-01-01', periods=24, freq='h', name='time')
    flow_system = fx.FlowSystem(timesteps=timesteps, years=years)

    thermal_load = LoadProfiles.random_thermal(len(timesteps))
    p_el = LoadProfiles.random_electrical(len(timesteps))

    costs = Effects.costs()
    heat_load = Sinks.heat_load(thermal_load)
    gas_source = Sources.gas_with_costs()
    electricity_sink = Sinks.electricity_feed_in(p_el)

    flow_system.add_elements(*Buses.defaults())
    flow_system.add_elements(costs, heat_load, gas_source, electricity_sink)

    return flow_system


class TestYearAwareInvestParameters:
    """Test the YearAwareInvestParameters interface."""

    def test_basic_initialization(self):
        """Test basic parameter initialization."""
        params = fx.YearAwareInvestParameters(
            minimum_size=10,
            maximum_size=100,
        )

        assert params.minimum_size == 10
        assert params.maximum_size == 100
        assert params.fixed_size is None
        assert not params.allow_divestment
        assert params.fixed_year_of_investment is None
        assert params.fixed_year_of_decommissioning is None
        assert params.fixed_duration is None

    def test_fixed_size_initialization(self):
        """Test initialization with fixed size."""
        params = fx.YearAwareInvestParameters(fixed_size=50)

        assert params.minimum_or_fixed_size == 50
        assert params.maximum_or_fixed_size == 50
        assert params.is_fixed_size

    def test_timing_constraints_initialization(self):
        """Test initialization with various timing constraints."""
        params = fx.YearAwareInvestParameters(
            fixed_year_of_investment=2,
            minimum_duration=3,
            maximum_duration=5,
            earliest_year_of_decommissioning=4,
        )

        assert params.fixed_year_of_investment == 2
        assert params.minimum_duration == 3
        assert params.maximum_duration == 5
        assert params.earliest_year_of_decommissioning == 4

    def test_effects_initialization(self):
        """Test initialization with effects."""
        params = fx.YearAwareInvestParameters(
            effects_of_investment={'costs': 1000},
            effects_of_investment_per_size={'costs': 100},
            allow_divestment=True,
            effects_of_divestment={'costs': 500},
            effects_of_divestment_per_size={'costs': 50},
        )

        assert params.effects_of_investment == {'costs': 1000}
        assert params.effects_of_investment_per_size == {'costs': 100}
        assert params.allow_divestment
        assert params.effects_of_divestment == {'costs': 500}
        assert params.effects_of_divestment_per_size == {'costs': 50}

    def test_property_methods(self):
        """Test property methods."""
        # Test with fixed size
        params_fixed = fx.YearAwareInvestParameters(fixed_size=50)
        assert params_fixed.minimum_or_fixed_size == 50
        assert params_fixed.maximum_or_fixed_size == 50
        assert params_fixed.is_fixed_size

        # Test with min/max size
        params_range = fx.YearAwareInvestParameters(minimum_size=10, maximum_size=100)
        assert params_range.minimum_or_fixed_size == 10
        assert params_range.maximum_or_fixed_size == 100
        assert not params_range.is_fixed_size


class TestYearAwareInvestmentModelDirect:
    """Test the YearAwareInvestmentModel class directly with linopy."""

    def test_flow_invest_new(self, flow_system):
        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestTimingParameters(
                year_of_investment=2021,
                year_of_decommissioning=2023,
                minimum_size=900,
                maximum_size=1000,
                effects_of_investment_per_size=200,
            ),
            relative_maximum=np.linspace(0.5, 1, flow_system.timesteps.size),
        )

        flow_system.add_elements(fx.Source('Source', source=flow))
        calculation = fx.FullCalculation('GenericName', flow_system)
        calculation.do_modeling()
        # calculation.model.add_constraints(calculation.model['Source(Wärme)|decrease'].isel(year=2) == 1)
        calculation.solve(fx.solvers.HighsSolver(0, 60))

        ds = calculation.results['Source'].solution
        filtered_ds = ds[[v for v in ds.data_vars if ds[v].dims == ('year',)]]
        print(filtered_ds.round(0).to_pandas().T)

        print('##')
