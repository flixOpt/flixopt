from typing import Union

import linopy
import numpy as np
import pandas as pd
import pytest
import xarray as xr

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


def calculate_annual_payment(total_cost: float, remaining_years: int, discount_rate: float) -> float:
    """Calculate annualized payment for given remaining years.

    Args:
        total_cost: Total cost to be annualized.
        remaining_years: Number of remaining years.
        discount_rate: Discount rate for annualization.

    Returns:
        Annual payment amount.
    """
    if remaining_years == 1:
        return total_cost

    return (
        total_cost
        * (discount_rate * (1 + discount_rate) ** remaining_years)
        / ((1 + discount_rate) ** remaining_years - 1)
    )


def create_annualized_effects(
    year_of_investments: Union[range, list, pd.Index],
    all_years: Union[range, list, pd.Index],
    total_cost: float,
    discount_rate: float,
    horizon_end: int,
    extra_dim: str = 'year_of_investment',
) -> xr.DataArray:
    """Create a 2D effects array for annualized costs.

    Creates an array where investing in year Y results in annualized costs
    applied to years Y through horizon_end.

    Args:
        year_of_investments: Years when investment decisions can be made.
        all_years: All years in the model (for the 'year' dimension).
        total_cost: Total upfront cost to be annualized.
        discount_rate: Discount rate for annualization calculation.
        horizon_end: Last year when effects apply.
        extra_dim: Name for the investment year dimension.

    Returns:
        xr.DataArray with dimensions [extra_dim, 'year'] containing annualized costs.
    """

    # Convert to lists for easier iteration
    year_of_investments_list = list(year_of_investments)
    all_years_list = list(all_years)

    # Initialize cost matrix
    cost_matrix = np.zeros((len(year_of_investments_list), len(all_years_list)))

    # Fill matrix with annualized costs
    for i, year_of_investment in enumerate(year_of_investments_list):
        remaining_years = horizon_end - year_of_investment + 1
        if remaining_years > 0:
            annual_cost = calculate_annual_payment(total_cost, remaining_years, discount_rate)

            # Apply cost to years from year_of_investment through horizon_end
            for j, cost_year in enumerate(all_years_list):
                if year_of_investment <= cost_year <= horizon_end:
                    cost_matrix[i, j] = annual_cost

    return xr.DataArray(
        cost_matrix, coords={extra_dim: year_of_investments_list, 'year': all_years_list}, dims=[extra_dim, 'year']
    )


@pytest.fixture
def flow_system() -> fx.FlowSystem:
    """Create basic elements for component testing with coordinate parametrization."""
    years = pd.Index([2020, 2021, 2022, 2023, 2024, 2030], name='year')
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
        da = xr.DataArray(
            [25, 30, 35, 40, 45, 50],
            coords=(flow_system.years_of_investment,),
        ).expand_dims(year=flow_system.years)
        da = da.where(da.year >= da.year_of_investment).fillna(0)

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestTimingParameters(
                force_investment=xr.DataArray(
                    [False if year != 2021 else True for year in flow_system.years], coords=(flow_system.years,)
                ),
                # year_of_decommissioning=2030,
                duration_in_years=2,
                minimum_size=900,
                maximum_size=1000,
                specific_effects=xr.DataArray(
                    [25, 30, 35, 40, 45, 50],
                    coords=(flow_system.years,),
                )
                * 0,
                # fix_effects=-2e3,
                specific_effects_by_investment_year=da,
            ),
            relative_maximum=np.linspace(0.5, 1, flow_system.timesteps.size),
        )

        flow_system.add_elements(fx.Source('Source', source=flow))
        calculation = fx.FullCalculation('GenericName', flow_system)
        calculation.do_modeling()
        # calculation.model.add_constraints(calculation.model['Source(Wärme)|decrease'].isel(year=2) == 1)
        calculation.solve(fx.solvers.GurobiSolver(0, 60))

        calculation = fx.FullCalculation('GenericName', flow_system.sel(year=[2022, 2030]))
        calculation.do_modeling()
        # calculation.model.add_constraints(calculation.model['Source(Wärme)|decrease'].isel(year=2) == 1)
        calculation.solve(fx.solvers.GurobiSolver(0, 60))

        ds = calculation.results['Source'].solution
        filtered_ds_year = ds[[v for v in ds.data_vars if ds[v].dims == ('year',)]]
        print(filtered_ds_year.round(0).to_pandas().T)

        filtered_ds_scalar = ds[[v for v in ds.data_vars if ds[v].dims == tuple()]]
        print(filtered_ds_scalar.round(0).to_pandas().T)

        print(calculation.results.solution['costs(invest)|total'].to_pandas())

        print('##')
