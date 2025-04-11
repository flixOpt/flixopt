import numpy as np
import pandas as pd
import pytest
import xarray as xr

import flixopt as fx

from .conftest import assert_conequal, assert_var_equal, create_linopy_model


class TestFlowModel:
    """Test the FlowModel class."""

    def test_flow_minimal(self, basic_flow_system_linopy):
        """Test that flow model constraints are correctly generated."""
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps
        flow = fx.Flow('Wärme', bus='Fernwärme', size=100)

        flow_system.add_elements(fx.Sink('Sink', sink=flow))

        model = create_linopy_model(flow_system)

        assert_conequal(
            model.constraints['Sink(Wärme)|total_flow_hours'],
            flow.model.variables['Sink(Wärme)|total_flow_hours'] == (flow.model.variables['Sink(Wärme)|flow_rate'] * model.hours_per_step).sum()
        )
        assert_var_equal(flow.model.flow_rate,
                            model.add_variables(lower=0, upper=100, coords=(timesteps,)))
        assert_var_equal(flow.model.total_flow_hours, model.add_variables(lower=0))

        assert set(flow.model.variables) == set(['Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|flow_rate'])
        assert set(flow.model.constraints) == set(['Sink(Wärme)|total_flow_hours'])

    def test_flow(self, basic_flow_system_linopy):
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps
        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            relative_minimum=np.linspace(0, 0.5, timesteps.size),
            relative_maximum=np.linspace(0.5, 1, timesteps.size),
            flow_hours_total_max=1000,
            flow_hours_total_min=10,
            load_factor_min=0.1,
            load_factor_max=0.9,
        )

        flow_system.add_elements(fx.Sink('Sink', sink=flow))
        model = create_linopy_model(flow_system)

        # total_flow_hours
        assert_conequal(
            model.constraints['Sink(Wärme)|total_flow_hours'],
            flow.model.variables['Sink(Wärme)|total_flow_hours']
            == (flow.model.variables['Sink(Wärme)|flow_rate'] * model.hours_per_step).sum(),
        )

        assert_var_equal(
            flow.model.total_flow_hours,
            model.add_variables(lower=10, upper=1000)
        )

        assert_var_equal(
            flow.model.flow_rate,
            model.add_variables(lower=np.linspace(0, 0.5, timesteps.size) * 100,
                                upper=np.linspace(0.5, 1, timesteps.size) * 100,
                                coords=(timesteps,))
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|load_factor_min'],
            flow.model.variables['Sink(Wärme)|total_flow_hours']
            >= model.hours_per_step.sum('time') * 0.1 * 100,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|load_factor_max'],
            flow.model.variables['Sink(Wärme)|total_flow_hours']
            <= model.hours_per_step.sum('time') * 0.9 * 100,
        )

        assert set(flow.model.variables) == set(['Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|flow_rate'])
        assert set(flow.model.constraints) == set(['Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|load_factor_max', 'Sink(Wärme)|load_factor_min'])

    def test_effects_per_flow_hour(self, basic_flow_system_linopy):
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps

        costs_per_flow_hour = xr.DataArray(np.linspace(1,2,timesteps.size), coords=(timesteps,))
        co2_per_flow_hour = xr.DataArray(np.linspace(4, 5, timesteps.size), coords=(timesteps,))

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            effects_per_flow_hour={'Costs': costs_per_flow_hour, 'CO2': co2_per_flow_hour}
        )
        flow_system.add_elements(fx.Sink('Sink', sink=flow), fx.Effect('CO2', 't', ''))
        model = create_linopy_model(flow_system)
        costs, co2 = flow_system.effects['Costs'], flow_system.effects['CO2']

        assert set(flow.model.variables) == {'Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|flow_rate'}
        assert set(flow.model.constraints) == {'Sink(Wärme)|total_flow_hours'}

        assert 'Sink(Wärme)->Costs(operation)' in set(costs.model.constraints)
        assert 'Sink(Wärme)->CO2(operation)' in set(co2.model.constraints)

        assert_conequal(
            model.constraints['Sink(Wärme)->Costs(operation)'],
            model.variables['Sink(Wärme)->Costs(operation)'] == flow.model.variables['Sink(Wärme)|flow_rate'] * model.hours_per_step * costs_per_flow_hour)

        assert_conequal(
            model.constraints['Sink(Wärme)->CO2(operation)'],
            model.variables['Sink(Wärme)->CO2(operation)'] == flow.model.variables['Sink(Wärme)|flow_rate'] * model.hours_per_step * co2_per_flow_hour)


class TestFlowInvestModel:
    """Test the FlowModel class."""

    def test_flow_invest(self, basic_flow_system_linopy):
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(minimum_size=20, maximum_size=100, optional=False),
            relative_minimum=np.linspace(0.1, 0.5, timesteps.size),
            relative_maximum=np.linspace(0.5, 1, timesteps.size),
        )

        flow_system.add_elements(fx.Sink('Sink', sink=flow))
        model = create_linopy_model(flow_system)

        assert set(flow.model.variables) == set(
            [
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|flow_rate',
                'Sink(Wärme)|size',
            ]
        )
        assert set(flow.model.constraints) == set(
            [
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|lb_Sink(Wärme)|flow_rate',
                'Sink(Wärme)|ub_Sink(Wärme)|flow_rate',
            ]
        )

        # size
        assert_var_equal(model['Sink(Wärme)|size'], model.add_variables(lower=20, upper=100))

        # flow_rate
        assert_var_equal(
            flow.model.flow_rate,
            model.add_variables(
                lower=np.linspace(0.1, 0.5, timesteps.size) * 20,
                upper=np.linspace(0.5, 1, timesteps.size) * 100,
                coords=(timesteps,),
            ),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|lb_Sink(Wärme)|flow_rate'],
            flow.model.variables['Sink(Wärme)|flow_rate']
            >= flow.model.variables['Sink(Wärme)|size']
            * xr.DataArray(np.linspace(0.1, 0.5, timesteps.size), coords=(timesteps,)),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|ub_Sink(Wärme)|flow_rate'],
            flow.model.variables['Sink(Wärme)|flow_rate']
            <= flow.model.variables['Sink(Wärme)|size']
            * xr.DataArray(np.linspace(0.5, 1, timesteps.size), coords=(timesteps,)),
        )

    def test_flow_invest_optional(self, basic_flow_system_linopy):
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(minimum_size=20, maximum_size=100, optional=True),
            relative_minimum=np.linspace(0.1, 0.5, timesteps.size),
            relative_maximum=np.linspace(0.5, 1, timesteps.size),
        )

        flow_system.add_elements(fx.Sink('Sink', sink=flow))
        model = create_linopy_model(flow_system)

        assert set(flow.model.variables) == set(
            ['Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|flow_rate', 'Sink(Wärme)|size', 'Sink(Wärme)|is_invested']
        )
        assert set(flow.model.constraints) == set(
            [
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|is_invested_ub',
                'Sink(Wärme)|is_invested_lb',
                'Sink(Wärme)|lb_Sink(Wärme)|flow_rate',
                'Sink(Wärme)|ub_Sink(Wärme)|flow_rate',
            ]
        )

        assert_var_equal(model['Sink(Wärme)|size'], model.add_variables(lower=0, upper=100))

        assert_var_equal(model['Sink(Wärme)|is_invested'], model.add_variables(binary=True))

        # flow_rate
        assert_var_equal(
            flow.model.flow_rate,
            model.add_variables(
                lower=0,  # Optional investment
                upper=np.linspace(0.5, 1, timesteps.size) * 100,
                coords=(timesteps,),
            ),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|lb_Sink(Wärme)|flow_rate'],
            flow.model.variables['Sink(Wärme)|flow_rate']
            >= flow.model.variables['Sink(Wärme)|size']
            * xr.DataArray(np.linspace(0.1, 0.5, timesteps.size), coords=(timesteps,)),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|ub_Sink(Wärme)|flow_rate'],
            flow.model.variables['Sink(Wärme)|flow_rate']
            <= flow.model.variables['Sink(Wärme)|size']
            * xr.DataArray(np.linspace(0.5, 1, timesteps.size), coords=(timesteps,)),
        )

        # Is invested
        assert_conequal(
            model.constraints['Sink(Wärme)|is_invested_ub'],
            flow.model.variables['Sink(Wärme)|size'] <= flow.model.variables['Sink(Wärme)|is_invested'] * 100,
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|is_invested_lb'],
            flow.model.variables['Sink(Wärme)|size'] >= flow.model.variables['Sink(Wärme)|is_invested'] * 20,
        )

    def test_flow_invest_optional_wo_min_size(self, basic_flow_system_linopy):
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(maximum_size=100, optional=True),
            relative_minimum=np.linspace(0.1, 0.5, timesteps.size),
            relative_maximum=np.linspace(0.5, 1, timesteps.size),
        )

        flow_system.add_elements(fx.Sink('Sink', sink=flow))
        model = create_linopy_model(flow_system)

        assert set(flow.model.variables) == set(
            ['Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|flow_rate', 'Sink(Wärme)|size', 'Sink(Wärme)|is_invested']
        )
        assert set(flow.model.constraints) == set(
            [
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|is_invested_ub',
                'Sink(Wärme)|is_invested_lb',
                'Sink(Wärme)|lb_Sink(Wärme)|flow_rate',
                'Sink(Wärme)|ub_Sink(Wärme)|flow_rate',
            ]
        )

        assert_var_equal(model['Sink(Wärme)|size'], model.add_variables(lower=0, upper=100))

        assert_var_equal(model['Sink(Wärme)|is_invested'], model.add_variables(binary=True))

        # flow_rate
        assert_var_equal(
            flow.model.flow_rate,
            model.add_variables(
                lower=0,  # Optional investment
                upper=np.linspace(0.5, 1, timesteps.size) * 100,
                coords=(timesteps,),
            ),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|lb_Sink(Wärme)|flow_rate'],
            flow.model.variables['Sink(Wärme)|flow_rate']
            >= flow.model.variables['Sink(Wärme)|size']
            * xr.DataArray(np.linspace(0.1, 0.5, timesteps.size), coords=(timesteps,)),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|ub_Sink(Wärme)|flow_rate'],
            flow.model.variables['Sink(Wärme)|flow_rate']
            <= flow.model.variables['Sink(Wärme)|size']
            * xr.DataArray(np.linspace(0.5, 1, timesteps.size), coords=(timesteps,)),
        )

        # Is invested
        assert_conequal(
            model.constraints['Sink(Wärme)|is_invested_ub'],
            flow.model.variables['Sink(Wärme)|size'] <= flow.model.variables['Sink(Wärme)|is_invested'] * 100,
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|is_invested_lb'],
            flow.model.variables['Sink(Wärme)|size'] >= flow.model.variables['Sink(Wärme)|is_invested'] * 1e-5,
        )

    def test_flow_invest_wo_min_size_non_optional(self, basic_flow_system_linopy):
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(maximum_size=100, optional=False),
            relative_minimum=np.linspace(0.1, 0.5, timesteps.size),
            relative_maximum=np.linspace(0.5, 1, timesteps.size),
        )

        flow_system.add_elements(fx.Sink('Sink', sink=flow))
        model = create_linopy_model(flow_system)

        assert set(flow.model.variables) == set(
            ['Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|flow_rate', 'Sink(Wärme)|size']
        )
        assert set(flow.model.constraints) == set(
            [
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|lb_Sink(Wärme)|flow_rate',
                'Sink(Wärme)|ub_Sink(Wärme)|flow_rate',
            ]
        )

        assert_var_equal(model['Sink(Wärme)|size'], model.add_variables(lower=1e-5, upper=100))

        # flow_rate
        assert_var_equal(
            flow.model.flow_rate,
            model.add_variables(
                lower=np.linspace(0.1, 0.5, timesteps.size) * 1e-5,
                upper=np.linspace(0.5, 1, timesteps.size) * 100,
                coords=(timesteps,),
            ),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|lb_Sink(Wärme)|flow_rate'],
            flow.model.variables['Sink(Wärme)|flow_rate']
            >= flow.model.variables['Sink(Wärme)|size']
            * xr.DataArray(np.linspace(0.1, 0.5, timesteps.size), coords=(timesteps,)),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|ub_Sink(Wärme)|flow_rate'],
            flow.model.variables['Sink(Wärme)|flow_rate']
            <= flow.model.variables['Sink(Wärme)|size']
            * xr.DataArray(np.linspace(0.5, 1, timesteps.size), coords=(timesteps,)),
        )

    def test_flow_invest_fixed_size(self, basic_flow_system_linopy):
        """Test flow with fixed size investment."""
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(fixed_size=75, optional=False),
            relative_minimum=0.2,
            relative_maximum=0.9,
        )

        flow_system.add_elements(fx.Sink('Sink', sink=flow))
        model = create_linopy_model(flow_system)

        assert set(flow.model.variables) == {'Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|flow_rate', 'Sink(Wärme)|size'}

        # Check that size is fixed to 75
        assert_var_equal(flow.model.variables['Sink(Wärme)|size'], model.add_variables(lower=75, upper=75))

        # Check flow rate bounds
        assert_var_equal(flow.model.flow_rate, model.add_variables(lower=0.2 * 75, upper=0.9 * 75, coords=(timesteps,)))

    def test_flow_invest_with_effects(self, basic_flow_system_linopy):
        """Test flow with investment effects."""
        flow_system = basic_flow_system_linopy

        # Create effects
        co2 = fx.Effect(label='CO2', unit='ton', description='CO2 emissions')

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(
                minimum_size=20,
                maximum_size=100,
                optional=True,
                fix_effects={'Costs': 1000, 'CO2': 5},  # Fixed investment effects
                specific_effects={'Costs': 500, 'CO2': 0.1},  # Specific investment effects
            ),
        )

        flow_system.add_elements( fx.Sink('Sink', sink=flow), co2)
        model = create_linopy_model(flow_system)

        # Check investment effects
        assert 'Sink(Wärme)->Costs(invest)' in model.variables
        assert 'Sink(Wärme)->CO2(invest)' in model.variables

        # Check fix effects (applied only when is_invested=1)
        assert_conequal(
            model.constraints['Sink(Wärme)->Costs(invest)'],
            model.variables['Sink(Wärme)->Costs(invest)']
            == flow.model.variables['Sink(Wärme)|is_invested'] * 1000 + flow.model.variables['Sink(Wärme)|size'] * 500,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)->CO2(invest)'],
            model.variables['Sink(Wärme)->CO2(invest)']
            == flow.model.variables['Sink(Wärme)|is_invested'] * 5 + flow.model.variables['Sink(Wärme)|size'] * 0.1,
        )

    def test_flow_invest_divest_effects(self, basic_flow_system_linopy):
        """Test flow with divestment effects."""
        flow_system = basic_flow_system_linopy

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(
                minimum_size=20,
                maximum_size=100,
                optional=True,
                divest_effects={'Costs': 500},  # Cost incurred when NOT investing
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', sink=flow))
        model = create_linopy_model(flow_system)

        # Check divestment effects
        assert 'Sink(Wärme)->Costs(invest)' in model.constraints

        assert_conequal(
            model.constraints['Sink(Wärme)->Costs(invest)'],
            model.variables['Sink(Wärme)->Costs(invest)'] + (model.variables['Sink(Wärme)|is_invested'] -1) * 500 == 0
        )


class TestFlowOnModel:
    """Test the FlowModel class."""

    def test_flow_on(self, basic_flow_system_linopy):
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps
        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            relative_minimum=xr.DataArray(0.2, coords=(timesteps,)),
            relative_maximum=xr.DataArray(0.8, coords=(timesteps,)),
            on_off_parameters=fx.OnOffParameters(),
        )
        flow_system.add_elements(fx.Sink('Sink', sink=flow))
        model = create_linopy_model(flow_system)

        assert set(flow.model.variables) == set(
            ['Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|flow_rate', 'Sink(Wärme)|on', 'Sink(Wärme)|on_hours_total']
        )

        assert set(flow.model.constraints) == set(
            [
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|on_hours_total',
                'Sink(Wärme)|on_con1',
                'Sink(Wärme)|on_con2',
            ]
        )
        # flow_rate
        assert_var_equal(
            flow.model.flow_rate,
            model.add_variables(
                lower=0,
                upper=0.8 * 100,
                coords=(timesteps,),
            ),
        )

        # OnOff
        assert_var_equal(
            flow.model.on_off.on,
            model.add_variables(binary=True, coords=(timesteps,)),
        )
        assert_var_equal(
            model.variables['Sink(Wärme)|on_hours_total'],
            model.add_variables(lower=0),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|on_con1'],
            flow.model.variables['Sink(Wärme)|on'] * 0.2 * 100 <= flow.model.variables['Sink(Wärme)|flow_rate'],
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|on_con2'],
            flow.model.variables['Sink(Wärme)|on'] * 0.8 * 100 >= flow.model.variables['Sink(Wärme)|flow_rate'],
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|on_hours_total'],
            flow.model.variables['Sink(Wärme)|on_hours_total']
            == (flow.model.variables['Sink(Wärme)|on'] * model.hours_per_step).sum(),
        )

    def test_effects_per_running_hour(self, basic_flow_system_linopy):
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps

        costs_per_running_hour = xr.DataArray(np.linspace(1, 2, timesteps.size), coords=(timesteps,))
        co2_per_running_hour = xr.DataArray(np.linspace(4, 5, timesteps.size), coords=(timesteps,))

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            on_off_parameters=fx.OnOffParameters(
                effects_per_running_hour={'Costs': costs_per_running_hour, 'CO2': co2_per_running_hour}
            ),
        )
        flow_system.add_elements(fx.Sink('Sink', sink=flow), fx.Effect('CO2', 't', ''))
        model = create_linopy_model(flow_system)
        costs, co2 = flow_system.effects['Costs'], flow_system.effects['CO2']

        assert set(flow.model.variables) == {
            'Sink(Wärme)|total_flow_hours',
            'Sink(Wärme)|flow_rate',
            'Sink(Wärme)|on',
            'Sink(Wärme)|on_hours_total',
        }
        assert set(flow.model.constraints) == {
            'Sink(Wärme)|total_flow_hours',
            'Sink(Wärme)|on_con1',
            'Sink(Wärme)|on_con2',
            'Sink(Wärme)|on_hours_total',
        }

        assert 'Sink(Wärme)->Costs(operation)' in set(costs.model.constraints)
        assert 'Sink(Wärme)->CO2(operation)' in set(co2.model.constraints)

        assert_conequal(
            model.constraints['Sink(Wärme)->Costs(operation)'],
            model.variables['Sink(Wärme)->Costs(operation)']
            == flow.model.variables['Sink(Wärme)|on'] * model.hours_per_step * costs_per_running_hour,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)->CO2(operation)'],
            model.variables['Sink(Wärme)->CO2(operation)']
            == flow.model.variables['Sink(Wärme)|on'] * model.hours_per_step * co2_per_running_hour,
        )

    def test_consecutive_on_hours(self, basic_flow_system_linopy):
        """Test flow with minimum and maximum consecutive on hours."""
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            on_off_parameters=fx.OnOffParameters(
                consecutive_on_hours_min=2,  # Must run for at least 2 hours when turned on
                consecutive_on_hours_max=8,  # Can't run more than 8 consecutive hours
            ),
        )

        flow_system.add_elements( fx.Sink('Sink', sink=flow))
        model = create_linopy_model(flow_system)

        assert {'Sink(Wärme)|ConsecutiveOn|hours', 'Sink(Wärme)|on'}.issubset(set(flow.model.variables))

        assert {'Sink(Wärme)|ConsecutiveOn|con1',
         'Sink(Wärme)|ConsecutiveOn|con2a',
         'Sink(Wärme)|ConsecutiveOn|con2b',
         'Sink(Wärme)|ConsecutiveOn|initial',
         'Sink(Wärme)|ConsecutiveOn|minimum',
        }.issubset(set(flow.model.constraints))

        assert_var_equal(
            model.variables['Sink(Wärme)|ConsecutiveOn|hours'],
            model.add_variables(lower=0, upper=8, coords=(timesteps,))
        )

        mega = model.hours_per_step.sum('time')

        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOn|con1'],
            model.variables['Sink(Wärme)|ConsecutiveOn|hours'] <= model.variables['Sink(Wärme)|on'] * mega
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOn|con2a'],
            model.variables['Sink(Wärme)|ConsecutiveOn|hours'].isel(time=slice(1, None))
            <= model.variables['Sink(Wärme)|ConsecutiveOn|hours'].isel(time=slice(None, -1)) + model.hours_per_step.isel(time=slice(None, -1))
        )

        # eq: duration(t) >= duration(t - 1) + dt(t) + (On(t) - 1) * BIG
        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOn|con2b'],
            model.variables['Sink(Wärme)|ConsecutiveOn|hours'].isel(time=slice(1, None))
            >= model.variables['Sink(Wärme)|ConsecutiveOn|hours'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1))
            + (model.variables['Sink(Wärme)|on'].isel(time=slice(1, None)) - 1) * mega
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOn|initial'],
            model.variables['Sink(Wärme)|ConsecutiveOn|hours'].isel(time=0)
            == model.variables['Sink(Wärme)|on'].isel(time=0) * model.hours_per_step.isel(time=0),
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOn|minimum'],
            model.variables['Sink(Wärme)|ConsecutiveOn|hours']
            >= (model.variables['Sink(Wärme)|on'].isel(time=slice(None, -1)) - model.variables['Sink(Wärme)|on'].isel(time=slice(1, None))) * 2
        )

    def test_consecutive_on_hours_previous(self, basic_flow_system_linopy):
        """Test flow with minimum and maximum consecutive on hours."""
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            on_off_parameters=fx.OnOffParameters(
                consecutive_on_hours_min=2,  # Must run for at least 2 hours when turned on
                consecutive_on_hours_max=8,  # Can't run more than 8 consecutive hours
            ),
            previous_flow_rate=np.array([10, 20, 30, 0, 20, 20, 30]) # Previously on for 3 steps
        )

        flow_system.add_elements( fx.Sink('Sink', sink=flow))
        model = create_linopy_model(flow_system)

        assert {'Sink(Wärme)|ConsecutiveOn|hours', 'Sink(Wärme)|on'}.issubset(set(flow.model.variables))

        assert {'Sink(Wärme)|ConsecutiveOn|con1',
         'Sink(Wärme)|ConsecutiveOn|con2a',
         'Sink(Wärme)|ConsecutiveOn|con2b',
         'Sink(Wärme)|ConsecutiveOn|initial',
         'Sink(Wärme)|ConsecutiveOn|minimum',
        }.issubset(set(flow.model.constraints))

        assert_var_equal(
            model.variables['Sink(Wärme)|ConsecutiveOn|hours'],
            model.add_variables(lower=0, upper=8, coords=(timesteps,))
        )

        mega = model.hours_per_step.sum('time') + model.hours_per_step.isel(time=0) * 3

        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOn|con1'],
            model.variables['Sink(Wärme)|ConsecutiveOn|hours'] <= model.variables['Sink(Wärme)|on'] * mega
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOn|con2a'],
            model.variables['Sink(Wärme)|ConsecutiveOn|hours'].isel(time=slice(1, None))
            <= model.variables['Sink(Wärme)|ConsecutiveOn|hours'].isel(time=slice(None, -1)) + model.hours_per_step.isel(time=slice(None, -1))
        )

        # eq: duration(t) >= duration(t - 1) + dt(t) + (On(t) - 1) * BIG
        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOn|con2b'],
            model.variables['Sink(Wärme)|ConsecutiveOn|hours'].isel(time=slice(1, None))
            >= model.variables['Sink(Wärme)|ConsecutiveOn|hours'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1))
            + (model.variables['Sink(Wärme)|on'].isel(time=slice(1, None)) - 1) * mega
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOn|initial'],
            model.variables['Sink(Wärme)|ConsecutiveOn|hours'].isel(time=0)
            == model.variables['Sink(Wärme)|on'].isel(time=0) * (model.hours_per_step.isel(time=0) * (1 + 3)),
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOn|minimum'],
            model.variables['Sink(Wärme)|ConsecutiveOn|hours']
            >= (model.variables['Sink(Wärme)|on'].isel(time=slice(None, -1)) - model.variables['Sink(Wärme)|on'].isel(time=slice(1, None))) * 2
        )

    def test_consecutive_off_hours(self, basic_flow_system_linopy):
        """Test flow with minimum and maximum consecutive off hours."""
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            on_off_parameters=fx.OnOffParameters(
                consecutive_off_hours_min=4,  # Must stay off for at least 4 hours when shut down
                consecutive_off_hours_max=12,  # Can't be off for more than 12 consecutive hours
            ),
        )

        flow_system.add_elements( fx.Sink('Sink', sink=flow))
        model = create_linopy_model(flow_system)

        assert {'Sink(Wärme)|ConsecutiveOff|hours', 'Sink(Wärme)|off'}.issubset(set(flow.model.variables))

        assert {
            'Sink(Wärme)|ConsecutiveOff|con1',
         'Sink(Wärme)|ConsecutiveOff|con2a',
         'Sink(Wärme)|ConsecutiveOff|con2b',
         'Sink(Wärme)|ConsecutiveOff|initial',
         'Sink(Wärme)|ConsecutiveOff|minimum'
        }.issubset(set(flow.model.constraints))

        assert_var_equal(
            model.variables['Sink(Wärme)|ConsecutiveOff|hours'],
            model.add_variables(lower=0, upper=12, coords=(timesteps,))
        )

        mega = model.hours_per_step.sum('time') + model.hours_per_step.isel(time=0) * 1  # previously off for 1h

        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOff|con1'],
            model.variables['Sink(Wärme)|ConsecutiveOff|hours'] <= model.variables['Sink(Wärme)|off'] * mega
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOff|con2a'],
            model.variables['Sink(Wärme)|ConsecutiveOff|hours'].isel(time=slice(1, None))
            <= model.variables['Sink(Wärme)|ConsecutiveOff|hours'].isel(time=slice(None, -1)) + model.hours_per_step.isel(time=slice(None, -1))
        )

        # eq: duration(t) >= duration(t - 1) + dt(t) + (On(t) - 1) * BIG
        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOff|con2b'],
            model.variables['Sink(Wärme)|ConsecutiveOff|hours'].isel(time=slice(1, None))
            >= model.variables['Sink(Wärme)|ConsecutiveOff|hours'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1))
            + (model.variables['Sink(Wärme)|off'].isel(time=slice(1, None)) - 1) * mega
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOff|initial'],
            model.variables['Sink(Wärme)|ConsecutiveOff|hours'].isel(time=0)
            == model.variables['Sink(Wärme)|off'].isel(time=0) * (model.hours_per_step.isel(time=0) * (1 + 1)),
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOff|minimum'],
            model.variables['Sink(Wärme)|ConsecutiveOff|hours']
            >= (model.variables['Sink(Wärme)|off'].isel(time=slice(None, -1)) - model.variables['Sink(Wärme)|off'].isel(time=slice(1, None))) * 4
        )

    def test_consecutive_off_hours_previous(self, basic_flow_system_linopy):
        """Test flow with minimum and maximum consecutive off hours."""
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            on_off_parameters=fx.OnOffParameters(
                consecutive_off_hours_min=4,  # Must stay off for at least 4 hours when shut down
                consecutive_off_hours_max=12,  # Can't be off for more than 12 consecutive hours
            ),
            previous_flow_rate=np.array([10, 20, 30, 0, 20, 0, 0]) # Previously off for 2 steps
        )

        flow_system.add_elements( fx.Sink('Sink', sink=flow))
        model = create_linopy_model(flow_system)

        assert {'Sink(Wärme)|ConsecutiveOff|hours', 'Sink(Wärme)|off'}.issubset(set(flow.model.variables))

        assert {
            'Sink(Wärme)|ConsecutiveOff|con1',
         'Sink(Wärme)|ConsecutiveOff|con2a',
         'Sink(Wärme)|ConsecutiveOff|con2b',
         'Sink(Wärme)|ConsecutiveOff|initial',
         'Sink(Wärme)|ConsecutiveOff|minimum'
        }.issubset(set(flow.model.constraints))

        assert_var_equal(
            model.variables['Sink(Wärme)|ConsecutiveOff|hours'],
            model.add_variables(lower=0, upper=12, coords=(timesteps,))
        )

        mega = model.hours_per_step.sum('time') + model.hours_per_step.isel(time=0) * 2

        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOff|con1'],
            model.variables['Sink(Wärme)|ConsecutiveOff|hours'] <= model.variables['Sink(Wärme)|off'] * mega
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOff|con2a'],
            model.variables['Sink(Wärme)|ConsecutiveOff|hours'].isel(time=slice(1, None))
            <= model.variables['Sink(Wärme)|ConsecutiveOff|hours'].isel(time=slice(None, -1)) + model.hours_per_step.isel(time=slice(None, -1))
        )

        # eq: duration(t) >= duration(t - 1) + dt(t) + (On(t) - 1) * BIG
        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOff|con2b'],
            model.variables['Sink(Wärme)|ConsecutiveOff|hours'].isel(time=slice(1, None))
            >= model.variables['Sink(Wärme)|ConsecutiveOff|hours'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1))
            + (model.variables['Sink(Wärme)|off'].isel(time=slice(1, None)) - 1) * mega
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOff|initial'],
            model.variables['Sink(Wärme)|ConsecutiveOff|hours'].isel(time=0)
            == model.variables['Sink(Wärme)|off'].isel(time=0) * (model.hours_per_step.isel(time=0) * (1+2)),
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|ConsecutiveOff|minimum'],
            model.variables['Sink(Wärme)|ConsecutiveOff|hours']
            >= (model.variables['Sink(Wärme)|off'].isel(time=slice(None, -1)) - model.variables['Sink(Wärme)|off'].isel(time=slice(1, None))) * 4
        )

    def test_switch_on_constraints(self, basic_flow_system_linopy):
        """Test flow with constraints on the number of startups."""
        flow_system = basic_flow_system_linopy

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            on_off_parameters=fx.OnOffParameters(
                switch_on_total_max=5,  # Maximum 5 startups
                effects_per_switch_on={'Costs': 100},  # 100 EUR startup cost
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', sink=flow))
        model = create_linopy_model(flow_system)

        # Check that variables exist
        assert {'Sink(Wärme)|switch_on', 'Sink(Wärme)|switch_off', 'Sink(Wärme)|switch_on_nr'}.issubset(
            set(flow.model.variables)
        )

        # Check that constraints exist
        assert {
            'Sink(Wärme)|switch_con',
            'Sink(Wärme)|initial_switch_con',
            'Sink(Wärme)|switch_on_or_off',
            'Sink(Wärme)|switch_on_nr',
        }.issubset(set(flow.model.constraints))

        # Check switch_on_nr variable bounds
        assert_var_equal(flow.model.variables['Sink(Wärme)|switch_on_nr'], model.add_variables(lower=0, upper=5))

        # Verify switch_on_nr constraint (limits number of startups)
        assert_conequal(
            model.constraints['Sink(Wärme)|switch_on_nr'],
            flow.model.variables['Sink(Wärme)|switch_on_nr']
            == flow.model.variables['Sink(Wärme)|switch_on'].sum('time'),
        )

        # Check that startup cost effect constraint exists
        assert 'Sink(Wärme)->Costs(operation)' in model.constraints

        # Verify the startup cost effect constraint
        assert_conequal(
            model.constraints['Sink(Wärme)->Costs(operation)'],
            model.variables['Sink(Wärme)->Costs(operation)'] == flow.model.variables['Sink(Wärme)|switch_on'] * 100,
        )

    def test_on_hours_limits(self, basic_flow_system_linopy):
        """Test flow with limits on total on hours."""
        flow_system = basic_flow_system_linopy

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            on_off_parameters=fx.OnOffParameters(
                on_hours_total_min=20,  # Minimum 20 hours of operation
                on_hours_total_max=100,  # Maximum 100 hours of operation
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', sink=flow))
        model = create_linopy_model(flow_system)

        # Check that variables exist
        assert {'Sink(Wärme)|on', 'Sink(Wärme)|on_hours_total'}.issubset(set(flow.model.variables))

        # Check that constraints exist
        assert 'Sink(Wärme)|on_hours_total' in model.constraints

        # Check on_hours_total variable bounds
        assert_var_equal(flow.model.variables['Sink(Wärme)|on_hours_total'], model.add_variables(lower=20, upper=100))

        # Check on_hours_total constraint
        assert_conequal(
            model.constraints['Sink(Wärme)|on_hours_total'],
            flow.model.variables['Sink(Wärme)|on_hours_total']
            == (flow.model.variables['Sink(Wärme)|on'] * model.hours_per_step).sum(),
        )


class TestFlowOnInvestModel:
    """Test the FlowModel class."""

    def test_flow_on_invest_optional(self, basic_flow_system_linopy):
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps
        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(minimum_size=20, maximum_size=200, optional=True),
            relative_minimum=xr.DataArray(0.2, coords=(timesteps,)),
            relative_maximum=xr.DataArray(0.8, coords=(timesteps,)),
            on_off_parameters=fx.OnOffParameters(),
        )
        flow_system.add_elements(fx.Sink('Sink', sink=flow))
        model = create_linopy_model(flow_system)

        assert set(flow.model.variables) == set(
            [
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|flow_rate',
                'Sink(Wärme)|is_invested',
                'Sink(Wärme)|size',
                'Sink(Wärme)|on',
                'Sink(Wärme)|on_hours_total',
            ]
        )

        assert set(flow.model.constraints) == set(
            [
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|on_hours_total',
                'Sink(Wärme)|on_con1',
                'Sink(Wärme)|on_con2',
                'Sink(Wärme)|is_invested_lb',
                'Sink(Wärme)|is_invested_ub',
                'Sink(Wärme)|lb_Sink(Wärme)|flow_rate',
                'Sink(Wärme)|ub_Sink(Wärme)|flow_rate',
            ]
        )

        # flow_rate
        assert_var_equal(
            flow.model.flow_rate,
            model.add_variables(
                lower=0,
                upper=0.8 * 200,
                coords=(timesteps,),
            ),
        )

        # OnOff
        assert_var_equal(
            flow.model.on_off.on,
            model.add_variables(binary=True, coords=(timesteps,)),
        )
        assert_var_equal(
            model.variables['Sink(Wärme)|on_hours_total'],
            model.add_variables(lower=0),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|on_con1'],
            flow.model.variables['Sink(Wärme)|on'] * 0.2 * 20 <= flow.model.variables['Sink(Wärme)|flow_rate'],
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|on_con2'],
            flow.model.variables['Sink(Wärme)|on'] * 0.8 * 200 >= flow.model.variables['Sink(Wärme)|flow_rate'],
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|on_hours_total'],
            flow.model.variables['Sink(Wärme)|on_hours_total']
            == (flow.model.variables['Sink(Wärme)|on'] * model.hours_per_step).sum(),
        )

        # Investment
        assert_var_equal(model['Sink(Wärme)|size'], model.add_variables(lower=0, upper=200))

        mega = 0.2 * 200  # Relative minimum * maximum size
        assert_conequal(
            model.constraints['Sink(Wärme)|lb_Sink(Wärme)|flow_rate'],
            flow.model.variables['Sink(Wärme)|flow_rate']
            >= flow.model.variables['Sink(Wärme)|on'] * mega + flow.model.variables['Sink(Wärme)|size'] * 0.2 - mega,
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|ub_Sink(Wärme)|flow_rate'],
            flow.model.variables['Sink(Wärme)|flow_rate'] <= flow.model.variables['Sink(Wärme)|size'] * 0.8,
        )

    def test_flow_on_invest_non_optional(self, basic_flow_system_linopy):
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps
        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(minimum_size=20, maximum_size=200, optional=False),
            relative_minimum=xr.DataArray(0.2, coords=(timesteps,)),
            relative_maximum=xr.DataArray(0.8, coords=(timesteps,)),
            on_off_parameters=fx.OnOffParameters(),
        )
        flow_system.add_elements(fx.Sink('Sink', sink=flow))
        model = create_linopy_model(flow_system)

        assert set(flow.model.variables) == set(
            [
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|flow_rate',
                'Sink(Wärme)|size',
                'Sink(Wärme)|on',
                'Sink(Wärme)|on_hours_total',
            ]
        )

        assert set(flow.model.constraints) == set(
            [
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|on_hours_total',
                'Sink(Wärme)|on_con1',
                'Sink(Wärme)|on_con2',
                'Sink(Wärme)|lb_Sink(Wärme)|flow_rate',
                'Sink(Wärme)|ub_Sink(Wärme)|flow_rate',
            ]
        )

        # flow_rate
        assert_var_equal(
            flow.model.flow_rate,
            model.add_variables(
                lower=0,
                upper=0.8 * 200,
                coords=(timesteps,),
            ),
        )

        # OnOff
        assert_var_equal(
            flow.model.on_off.on,
            model.add_variables(binary=True, coords=(timesteps,)),
        )
        assert_var_equal(
            model.variables['Sink(Wärme)|on_hours_total'],
            model.add_variables(lower=0),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|on_con1'],
            flow.model.variables['Sink(Wärme)|on'] * 0.2 * 20 <= flow.model.variables['Sink(Wärme)|flow_rate'],
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|on_con2'],
            flow.model.variables['Sink(Wärme)|on'] * 0.8 * 200 >= flow.model.variables['Sink(Wärme)|flow_rate'],
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|on_hours_total'],
            flow.model.variables['Sink(Wärme)|on_hours_total']
            == (flow.model.variables['Sink(Wärme)|on'] * model.hours_per_step).sum(),
        )

        # Investment
        assert_var_equal(model['Sink(Wärme)|size'], model.add_variables(lower=20, upper=200))

        mega = 0.2 * 200  # Relative minimum * maximum size
        assert_conequal(
            model.constraints['Sink(Wärme)|lb_Sink(Wärme)|flow_rate'],
            flow.model.variables['Sink(Wärme)|flow_rate']
            >= flow.model.variables['Sink(Wärme)|on'] * mega + flow.model.variables['Sink(Wärme)|size'] * 0.2 - mega,
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|ub_Sink(Wärme)|flow_rate'],
            flow.model.variables['Sink(Wärme)|flow_rate'] <= flow.model.variables['Sink(Wärme)|size'] * 0.8,
        )


class TestFlowWithFixedProfile:
    """Test Flow with fixed relative profile."""

    def test_fixed_relative_profile(self, basic_flow_system_linopy):
        """Test flow with a fixed relative profile."""
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps

        # Create a time-varying profile (e.g., for a load or renewable generation)
        profile = np.sin(np.linspace(0, 2 * np.pi, len(timesteps))) * 0.5 + 0.5  # Values between 0 and 1

        flow = fx.Flow(
            'Wärme', bus='Fernwärme', size=100, fixed_relative_profile=xr.DataArray(profile, coords=(timesteps,))
        )

        flow_system.add_elements(fx.Sink('Sink', sink=flow))
        model = create_linopy_model(flow_system)

        assert_var_equal(flow.model.variables['Sink(Wärme)|flow_rate'],
                         model.add_variables(lower=profile * 100,
                                             upper=profile * 100,
                                             coords=(timesteps,))
                         )


    def test_fixed_profile_with_investment(self, basic_flow_system_linopy):
        """Test flow with fixed profile and investment."""
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps

        # Create a fixed profile
        profile = np.sin(np.linspace(0, 2 * np.pi, len(timesteps))) * 0.5 + 0.5

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(minimum_size=50, maximum_size=200, optional=True),
            fixed_relative_profile=xr.DataArray(profile, coords=(timesteps,)),
        )

        flow_system.add_elements( fx.Sink('Sink', sink=flow))
        model = create_linopy_model(flow_system)

        assert_var_equal(
            flow.model.variables['Sink(Wärme)|flow_rate'],
            model.add_variables(lower=0, upper=profile * 200, coords=(timesteps,)),
        )

        # The constraint should link flow_rate to size * profile
        assert_conequal(
            model.constraints['Sink(Wärme)|fix_Sink(Wärme)|flow_rate'],
            flow.model.variables['Sink(Wärme)|flow_rate']
            == flow.model.variables['Sink(Wärme)|size'] * xr.DataArray(profile, coords=(timesteps,)),
        )


if __name__ == '__main__':
    pytest.main()
