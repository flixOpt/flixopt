import numpy as np
import pytest
import xarray as xr

import flixopt as fx

from .conftest import assert_conequal, assert_sets_equal, assert_var_equal, create_linopy_model


class TestFlowModel:
    """Test the FlowModel class."""

    def test_flow_minimal(self, basic_flow_system_linopy_coords, coords_config):
        """Test that flow model constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow('Wärme', bus='Fernwärme', size=100)

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))

        model = create_linopy_model(flow_system)

        assert_conequal(
            model.constraints['Sink(Wärme)|total_flow_hours'],
            flow.submodel.variables['Sink(Wärme)|total_flow_hours']
            == (flow.submodel.variables['Sink(Wärme)|flow_rate'] * model.hours_per_step).sum('time'),
        )
        assert_var_equal(flow.submodel.flow_rate, model.add_variables(lower=0, upper=100, coords=model.get_coords()))
        assert_var_equal(
            flow.submodel.total_flow_hours,
            model.add_variables(lower=0, coords=model.get_coords(['period', 'scenario'])),
        )

        assert_sets_equal(
            set(flow.submodel.variables),
            {'Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|flow_rate'},
            msg='Incorrect variables',
        )
        assert_sets_equal(set(flow.submodel.constraints), {'Sink(Wärme)|total_flow_hours'}, msg='Incorrect constraints')

    def test_flow(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            relative_minimum=np.linspace(0, 0.5, timesteps.size),
            relative_maximum=np.linspace(0.5, 1, timesteps.size),
            flow_hours_max=1000,
            flow_hours_min=10,
            load_factor_min=0.1,
            load_factor_max=0.9,
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # total_flow_hours
        assert_conequal(
            model.constraints['Sink(Wärme)|total_flow_hours'],
            flow.submodel.variables['Sink(Wärme)|total_flow_hours']
            == (flow.submodel.variables['Sink(Wärme)|flow_rate'] * model.hours_per_step).sum('time'),
        )

        assert_var_equal(
            flow.submodel.total_flow_hours,
            model.add_variables(lower=10, upper=1000, coords=model.get_coords(['period', 'scenario'])),
        )

        assert flow.relative_minimum.dims == tuple(model.get_coords())
        assert flow.relative_maximum.dims == tuple(model.get_coords())

        assert_var_equal(
            flow.submodel.flow_rate,
            model.add_variables(
                lower=flow.relative_minimum * 100,
                upper=flow.relative_maximum * 100,
                coords=model.get_coords(),
            ),
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|load_factor_min'],
            flow.submodel.variables['Sink(Wärme)|total_flow_hours'] >= model.hours_per_step.sum('time') * 0.1 * 100,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|load_factor_max'],
            flow.submodel.variables['Sink(Wärme)|total_flow_hours'] <= model.hours_per_step.sum('time') * 0.9 * 100,
        )

        assert_sets_equal(
            set(flow.submodel.variables),
            {'Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|flow_rate'},
            msg='Incorrect variables',
        )
        assert_sets_equal(
            set(flow.submodel.constraints),
            {'Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|load_factor_max', 'Sink(Wärme)|load_factor_min'},
            msg='Incorrect constraints',
        )

    def test_effects_per_flow_hour(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        costs_per_flow_hour = xr.DataArray(np.linspace(1, 2, timesteps.size), coords=(timesteps,))
        co2_per_flow_hour = xr.DataArray(np.linspace(4, 5, timesteps.size), coords=(timesteps,))

        flow = fx.Flow(
            'Wärme', bus='Fernwärme', effects_per_flow_hour={'costs': costs_per_flow_hour, 'CO2': co2_per_flow_hour}
        )
        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]), fx.Effect('CO2', 't', ''))
        model = create_linopy_model(flow_system)
        costs, co2 = flow_system.effects['costs'], flow_system.effects['CO2']

        assert_sets_equal(
            set(flow.submodel.variables),
            {'Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|flow_rate'},
            msg='Incorrect variables',
        )
        assert_sets_equal(set(flow.submodel.constraints), {'Sink(Wärme)|total_flow_hours'}, msg='Incorrect constraints')

        assert 'Sink(Wärme)->costs(temporal)' in set(costs.submodel.constraints)
        assert 'Sink(Wärme)->CO2(temporal)' in set(co2.submodel.constraints)

        assert_conequal(
            model.constraints['Sink(Wärme)->costs(temporal)'],
            model.variables['Sink(Wärme)->costs(temporal)']
            == flow.submodel.variables['Sink(Wärme)|flow_rate'] * model.hours_per_step * costs_per_flow_hour,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)->CO2(temporal)'],
            model.variables['Sink(Wärme)->CO2(temporal)']
            == flow.submodel.variables['Sink(Wärme)|flow_rate'] * model.hours_per_step * co2_per_flow_hour,
        )


class TestFlowInvestModel:
    """Test the FlowModel class."""

    def test_flow_invest(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(minimum_size=20, maximum_size=100, mandatory=True),
            relative_minimum=np.linspace(0.1, 0.5, timesteps.size),
            relative_maximum=np.linspace(0.5, 1, timesteps.size),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert_sets_equal(
            set(flow.submodel.variables),
            {
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|flow_rate',
                'Sink(Wärme)|size',
            },
            msg='Incorrect variables',
        )
        assert_sets_equal(
            set(flow.submodel.constraints),
            {
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|flow_rate|ub',
                'Sink(Wärme)|flow_rate|lb',
            },
            msg='Incorrect constraints',
        )

        # size
        assert_var_equal(
            model['Sink(Wärme)|size'],
            model.add_variables(lower=20, upper=100, coords=model.get_coords(['period', 'scenario'])),
        )

        assert flow.relative_minimum.dims == tuple(model.get_coords())
        assert flow.relative_maximum.dims == tuple(model.get_coords())

        # flow_rate
        assert_var_equal(
            flow.submodel.flow_rate,
            model.add_variables(
                lower=flow.relative_minimum * 20,
                upper=flow.relative_maximum * 100,
                coords=model.get_coords(),
            ),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|lb'],
            flow.submodel.variables['Sink(Wärme)|flow_rate']
            >= flow.submodel.variables['Sink(Wärme)|size'] * flow.relative_minimum,
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|ub'],
            flow.submodel.variables['Sink(Wärme)|flow_rate']
            <= flow.submodel.variables['Sink(Wärme)|size'] * flow.relative_maximum,
        )

    def test_flow_invest_optional(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(minimum_size=20, maximum_size=100, mandatory=False),
            relative_minimum=np.linspace(0.1, 0.5, timesteps.size),
            relative_maximum=np.linspace(0.5, 1, timesteps.size),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert_sets_equal(
            set(flow.submodel.variables),
            {'Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|flow_rate', 'Sink(Wärme)|size', 'Sink(Wärme)|invested'},
            msg='Incorrect variables',
        )
        assert_sets_equal(
            set(flow.submodel.constraints),
            {
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|size|lb',
                'Sink(Wärme)|size|ub',
                'Sink(Wärme)|flow_rate|lb',
                'Sink(Wärme)|flow_rate|ub',
            },
            msg='Incorrect constraints',
        )

        assert_var_equal(
            model['Sink(Wärme)|size'],
            model.add_variables(lower=0, upper=100, coords=model.get_coords(['period', 'scenario'])),
        )

        assert_var_equal(
            model['Sink(Wärme)|invested'],
            model.add_variables(binary=True, coords=model.get_coords(['period', 'scenario'])),
        )

        assert flow.relative_minimum.dims == tuple(model.get_coords())
        assert flow.relative_maximum.dims == tuple(model.get_coords())

        # flow_rate
        assert_var_equal(
            flow.submodel.flow_rate,
            model.add_variables(
                lower=0,  # Optional investment
                upper=flow.relative_maximum * 100,
                coords=model.get_coords(),
            ),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|lb'],
            flow.submodel.variables['Sink(Wärme)|flow_rate']
            >= flow.submodel.variables['Sink(Wärme)|size'] * flow.relative_minimum,
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|ub'],
            flow.submodel.variables['Sink(Wärme)|flow_rate']
            <= flow.submodel.variables['Sink(Wärme)|size'] * flow.relative_maximum,
        )

        # Is invested
        assert_conequal(
            model.constraints['Sink(Wärme)|size|ub'],
            flow.submodel.variables['Sink(Wärme)|size'] <= flow.submodel.variables['Sink(Wärme)|invested'] * 100,
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|size|lb'],
            flow.submodel.variables['Sink(Wärme)|size'] >= flow.submodel.variables['Sink(Wärme)|invested'] * 20,
        )

    def test_flow_invest_optional_wo_min_size(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(maximum_size=100, mandatory=False),
            relative_minimum=np.linspace(0.1, 0.5, timesteps.size),
            relative_maximum=np.linspace(0.5, 1, timesteps.size),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert_sets_equal(
            set(flow.submodel.variables),
            {'Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|flow_rate', 'Sink(Wärme)|size', 'Sink(Wärme)|invested'},
            msg='Incorrect variables',
        )
        assert_sets_equal(
            set(flow.submodel.constraints),
            {
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|size|ub',
                'Sink(Wärme)|size|lb',
                'Sink(Wärme)|flow_rate|lb',
                'Sink(Wärme)|flow_rate|ub',
            },
            msg='Incorrect constraints',
        )

        assert_var_equal(
            model['Sink(Wärme)|size'],
            model.add_variables(lower=0, upper=100, coords=model.get_coords(['period', 'scenario'])),
        )

        assert_var_equal(
            model['Sink(Wärme)|invested'],
            model.add_variables(binary=True, coords=model.get_coords(['period', 'scenario'])),
        )

        assert flow.relative_minimum.dims == tuple(model.get_coords())
        assert flow.relative_maximum.dims == tuple(model.get_coords())

        # flow_rate
        assert_var_equal(
            flow.submodel.flow_rate,
            model.add_variables(
                lower=0,  # Optional investment
                upper=flow.relative_maximum * 100,
                coords=model.get_coords(),
            ),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|lb'],
            flow.submodel.variables['Sink(Wärme)|flow_rate']
            >= flow.submodel.variables['Sink(Wärme)|size'] * flow.relative_minimum,
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|ub'],
            flow.submodel.variables['Sink(Wärme)|flow_rate']
            <= flow.submodel.variables['Sink(Wärme)|size'] * flow.relative_maximum,
        )

        # Is invested
        assert_conequal(
            model.constraints['Sink(Wärme)|size|ub'],
            flow.submodel.variables['Sink(Wärme)|size'] <= flow.submodel.variables['Sink(Wärme)|invested'] * 100,
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|size|lb'],
            flow.submodel.variables['Sink(Wärme)|size'] >= flow.submodel.variables['Sink(Wärme)|invested'] * 1e-5,
        )

    def test_flow_invest_wo_min_size_non_optional(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(maximum_size=100, mandatory=True),
            relative_minimum=np.linspace(0.1, 0.5, timesteps.size),
            relative_maximum=np.linspace(0.5, 1, timesteps.size),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert_sets_equal(
            set(flow.submodel.variables),
            {'Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|flow_rate', 'Sink(Wärme)|size'},
            msg='Incorrect variables',
        )
        assert_sets_equal(
            set(flow.submodel.constraints),
            {
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|flow_rate|lb',
                'Sink(Wärme)|flow_rate|ub',
            },
            msg='Incorrect constraints',
        )

        assert_var_equal(
            model['Sink(Wärme)|size'],
            model.add_variables(lower=1e-5, upper=100, coords=model.get_coords(['period', 'scenario'])),
        )

        assert flow.relative_minimum.dims == tuple(model.get_coords())
        assert flow.relative_maximum.dims == tuple(model.get_coords())

        # flow_rate
        assert_var_equal(
            flow.submodel.flow_rate,
            model.add_variables(
                lower=flow.relative_minimum * 1e-5,
                upper=flow.relative_maximum * 100,
                coords=model.get_coords(),
            ),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|lb'],
            flow.submodel.variables['Sink(Wärme)|flow_rate']
            >= flow.submodel.variables['Sink(Wärme)|size'] * flow.relative_minimum,
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|ub'],
            flow.submodel.variables['Sink(Wärme)|flow_rate']
            <= flow.submodel.variables['Sink(Wärme)|size'] * flow.relative_maximum,
        )

    def test_flow_invest_fixed_size(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with fixed size investment."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(fixed_size=75, mandatory=True),
            relative_minimum=0.2,
            relative_maximum=0.9,
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert_sets_equal(
            set(flow.submodel.variables),
            {'Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|flow_rate', 'Sink(Wärme)|size'},
            msg='Incorrect variables',
        )

        # Check that size is fixed to 75
        assert_var_equal(
            flow.submodel.variables['Sink(Wärme)|size'],
            model.add_variables(lower=75, upper=75, coords=model.get_coords(['period', 'scenario'])),
        )

        # Check flow rate bounds
        assert_var_equal(
            flow.submodel.flow_rate, model.add_variables(lower=0.2 * 75, upper=0.9 * 75, coords=model.get_coords())
        )

    def test_flow_invest_with_effects(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with investment effects."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create effects
        co2 = fx.Effect(label='CO2', unit='ton', description='CO2 emissions')

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(
                minimum_size=20,
                maximum_size=100,
                mandatory=False,
                effects_of_investment={'costs': 1000, 'CO2': 5},  # Fixed investment effects
                effects_of_investment_per_size={'costs': 500, 'CO2': 0.1},  # Specific investment effects
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]), co2)
        model = create_linopy_model(flow_system)

        # Check investment effects
        assert 'Sink(Wärme)->costs(periodic)' in model.variables
        assert 'Sink(Wärme)->CO2(periodic)' in model.variables

        # Check fix effects (applied only when invested=1)
        assert_conequal(
            model.constraints['Sink(Wärme)->costs(periodic)'],
            model.variables['Sink(Wärme)->costs(periodic)']
            == flow.submodel.variables['Sink(Wärme)|invested'] * 1000
            + flow.submodel.variables['Sink(Wärme)|size'] * 500,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)->CO2(periodic)'],
            model.variables['Sink(Wärme)->CO2(periodic)']
            == flow.submodel.variables['Sink(Wärme)|invested'] * 5 + flow.submodel.variables['Sink(Wärme)|size'] * 0.1,
        )

    def test_flow_invest_divest_effects(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with divestment effects."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(
                minimum_size=20,
                maximum_size=100,
                mandatory=False,
                effects_of_retirement={'costs': 500},  # Cost incurred when NOT investing
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check divestment effects
        assert 'Sink(Wärme)->costs(periodic)' in model.constraints

        assert_conequal(
            model.constraints['Sink(Wärme)->costs(periodic)'],
            model.variables['Sink(Wärme)->costs(periodic)'] + (model.variables['Sink(Wärme)|invested'] - 1) * 500 == 0,
        )


class TestFlowOnModel:
    """Test the FlowModel class."""

    def test_flow_on(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            relative_minimum=0.2,
            relative_maximum=0.8,
            status_parameters=fx.StatusParameters(),
        )
        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert_sets_equal(
            set(flow.submodel.variables),
            {'Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|flow_rate', 'Sink(Wärme)|status', 'Sink(Wärme)|active_hours'},
            msg='Incorrect variables',
        )

        assert_sets_equal(
            set(flow.submodel.constraints),
            {
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|active_hours',
                'Sink(Wärme)|flow_rate|lb',
                'Sink(Wärme)|flow_rate|ub',
            },
            msg='Incorrect constraints',
        )
        # flow_rate
        assert_var_equal(
            flow.submodel.flow_rate,
            model.add_variables(
                lower=0,
                upper=0.8 * 100,
                coords=model.get_coords(),
            ),
        )

        # Status
        assert_var_equal(
            flow.submodel.status.status,
            model.add_variables(binary=True, coords=model.get_coords()),
        )
        assert_var_equal(
            model.variables['Sink(Wärme)|active_hours'],
            model.add_variables(lower=0, coords=model.get_coords(['period', 'scenario'])),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|lb'],
            flow.submodel.variables['Sink(Wärme)|flow_rate']
            >= flow.submodel.variables['Sink(Wärme)|status'] * 0.2 * 100,
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|ub'],
            flow.submodel.variables['Sink(Wärme)|flow_rate']
            <= flow.submodel.variables['Sink(Wärme)|status'] * 0.8 * 100,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|active_hours'],
            flow.submodel.variables['Sink(Wärme)|active_hours']
            == (flow.submodel.variables['Sink(Wärme)|status'] * model.hours_per_step).sum('time'),
        )

    def test_effects_per_active_hour(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        costs_per_running_hour = np.linspace(1, 2, timesteps.size)
        co2_per_running_hour = np.linspace(4, 5, timesteps.size)

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            status_parameters=fx.StatusParameters(
                effects_per_active_hour={'costs': costs_per_running_hour, 'CO2': co2_per_running_hour}
            ),
        )
        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]), fx.Effect('CO2', 't', ''))
        model = create_linopy_model(flow_system)
        costs, co2 = flow_system.effects['costs'], flow_system.effects['CO2']

        assert_sets_equal(
            set(flow.submodel.variables),
            {
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|flow_rate',
                'Sink(Wärme)|status',
                'Sink(Wärme)|active_hours',
            },
            msg='Incorrect variables',
        )
        assert_sets_equal(
            set(flow.submodel.constraints),
            {
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|flow_rate|lb',
                'Sink(Wärme)|flow_rate|ub',
                'Sink(Wärme)|active_hours',
            },
            msg='Incorrect constraints',
        )

        assert 'Sink(Wärme)->costs(temporal)' in set(costs.submodel.constraints)
        assert 'Sink(Wärme)->CO2(temporal)' in set(co2.submodel.constraints)

        costs_per_running_hour = flow.status_parameters.effects_per_active_hour['costs']
        co2_per_running_hour = flow.status_parameters.effects_per_active_hour['CO2']

        assert costs_per_running_hour.dims == tuple(model.get_coords())
        assert co2_per_running_hour.dims == tuple(model.get_coords())

        assert_conequal(
            model.constraints['Sink(Wärme)->costs(temporal)'],
            model.variables['Sink(Wärme)->costs(temporal)']
            == flow.submodel.variables['Sink(Wärme)|status'] * model.hours_per_step * costs_per_running_hour,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)->CO2(temporal)'],
            model.variables['Sink(Wärme)->CO2(temporal)']
            == flow.submodel.variables['Sink(Wärme)|status'] * model.hours_per_step * co2_per_running_hour,
        )

    def test_consecutive_on_hours(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with minimum and maximum consecutive on hours."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            status_parameters=fx.StatusParameters(
                min_uptime=2,  # Must run for at least 2 hours when turned on
                max_uptime=8,  # Can't run more than 8 consecutive hours
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert {'Sink(Wärme)|uptime', 'Sink(Wärme)|status'}.issubset(set(flow.submodel.variables))

        assert_sets_equal(
            {
                'Sink(Wärme)|uptime|ub',
                'Sink(Wärme)|uptime|forward',
                'Sink(Wärme)|uptime|backward',
                'Sink(Wärme)|uptime|initial',
                'Sink(Wärme)|uptime|lb',
            }
            & set(flow.submodel.constraints),
            {
                'Sink(Wärme)|uptime|ub',
                'Sink(Wärme)|uptime|forward',
                'Sink(Wärme)|uptime|backward',
                'Sink(Wärme)|uptime|initial',
                'Sink(Wärme)|uptime|lb',
            },
            msg='Missing uptime constraints',
        )

        assert_var_equal(
            model.variables['Sink(Wärme)|uptime'],
            model.add_variables(lower=0, upper=8, coords=model.get_coords()),
        )

        mega = model.hours_per_step.sum('time')

        assert_conequal(
            model.constraints['Sink(Wärme)|uptime|ub'],
            model.variables['Sink(Wärme)|uptime'] <= model.variables['Sink(Wärme)|status'] * mega,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|uptime|forward'],
            model.variables['Sink(Wärme)|uptime'].isel(time=slice(1, None))
            <= model.variables['Sink(Wärme)|uptime'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1)),
        )

        # eq: duration(t) >= duration(t - 1) + dt(t) + (On(t) - 1) * BIG
        assert_conequal(
            model.constraints['Sink(Wärme)|uptime|backward'],
            model.variables['Sink(Wärme)|uptime'].isel(time=slice(1, None))
            >= model.variables['Sink(Wärme)|uptime'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1))
            + (model.variables['Sink(Wärme)|status'].isel(time=slice(1, None)) - 1) * mega,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|uptime|initial'],
            model.variables['Sink(Wärme)|uptime'].isel(time=0)
            == model.variables['Sink(Wärme)|status'].isel(time=0) * model.hours_per_step.isel(time=0),
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|uptime|lb'],
            model.variables['Sink(Wärme)|uptime']
            >= (
                model.variables['Sink(Wärme)|status'].isel(time=slice(None, -1))
                - model.variables['Sink(Wärme)|status'].isel(time=slice(1, None))
            )
            * 2,
        )

    def test_consecutive_on_hours_previous(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with minimum and maximum uptime."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            status_parameters=fx.StatusParameters(
                min_uptime=2,  # Must run for at least 2 hours when active
                max_uptime=8,  # Can't run more than 8 consecutive hours
            ),
            previous_flow_rate=np.array([10, 20, 30, 0, 20, 20, 30]),  # Previously active for 3 steps
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert {'Sink(Wärme)|uptime', 'Sink(Wärme)|status'}.issubset(set(flow.submodel.variables))

        assert_sets_equal(
            {
                'Sink(Wärme)|uptime|lb',
                'Sink(Wärme)|uptime|forward',
                'Sink(Wärme)|uptime|backward',
                'Sink(Wärme)|uptime|initial',
            }
            & set(flow.submodel.constraints),
            {
                'Sink(Wärme)|uptime|lb',
                'Sink(Wärme)|uptime|forward',
                'Sink(Wärme)|uptime|backward',
                'Sink(Wärme)|uptime|initial',
            },
            msg='Missing uptime constraints for previous states',
        )

        assert_var_equal(
            model.variables['Sink(Wärme)|uptime'],
            model.add_variables(lower=0, upper=8, coords=model.get_coords()),
        )

        mega = model.hours_per_step.sum('time') + model.hours_per_step.isel(time=0) * 3

        assert_conequal(
            model.constraints['Sink(Wärme)|uptime|ub'],
            model.variables['Sink(Wärme)|uptime'] <= model.variables['Sink(Wärme)|status'] * mega,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|uptime|forward'],
            model.variables['Sink(Wärme)|uptime'].isel(time=slice(1, None))
            <= model.variables['Sink(Wärme)|uptime'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1)),
        )

        # eq: duration(t) >= duration(t - 1) + dt(t) + (On(t) - 1) * BIG
        assert_conequal(
            model.constraints['Sink(Wärme)|uptime|backward'],
            model.variables['Sink(Wärme)|uptime'].isel(time=slice(1, None))
            >= model.variables['Sink(Wärme)|uptime'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1))
            + (model.variables['Sink(Wärme)|status'].isel(time=slice(1, None)) - 1) * mega,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|uptime|initial'],
            model.variables['Sink(Wärme)|uptime'].isel(time=0)
            == model.variables['Sink(Wärme)|status'].isel(time=0) * (model.hours_per_step.isel(time=0) * (1 + 3)),
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|uptime|lb'],
            model.variables['Sink(Wärme)|uptime']
            >= (
                model.variables['Sink(Wärme)|status'].isel(time=slice(None, -1))
                - model.variables['Sink(Wärme)|status'].isel(time=slice(1, None))
            )
            * 2,
        )

    def test_consecutive_off_hours(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with minimum and maximum consecutive inactive hours."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            status_parameters=fx.StatusParameters(
                min_downtime=4,  # Must stay inactive for at least 4 hours when shut down
                max_downtime=12,  # Can't be inactive for more than 12 consecutive hours
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert {'Sink(Wärme)|downtime', 'Sink(Wärme)|inactive'}.issubset(set(flow.submodel.variables))

        assert_sets_equal(
            {
                'Sink(Wärme)|downtime|ub',
                'Sink(Wärme)|downtime|forward',
                'Sink(Wärme)|downtime|backward',
                'Sink(Wärme)|downtime|initial',
                'Sink(Wärme)|downtime|lb',
            }
            & set(flow.submodel.constraints),
            {
                'Sink(Wärme)|downtime|ub',
                'Sink(Wärme)|downtime|forward',
                'Sink(Wärme)|downtime|backward',
                'Sink(Wärme)|downtime|initial',
                'Sink(Wärme)|downtime|lb',
            },
            msg='Missing consecutive inactive hours constraints',
        )

        assert_var_equal(
            model.variables['Sink(Wärme)|downtime'],
            model.add_variables(lower=0, upper=12, coords=model.get_coords()),
        )

        mega = model.hours_per_step.sum('time') + model.hours_per_step.isel(time=0) * 1  # previously inactive for 1h

        assert_conequal(
            model.constraints['Sink(Wärme)|downtime|ub'],
            model.variables['Sink(Wärme)|downtime'] <= model.variables['Sink(Wärme)|inactive'] * mega,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|downtime|forward'],
            model.variables['Sink(Wärme)|downtime'].isel(time=slice(1, None))
            <= model.variables['Sink(Wärme)|downtime'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1)),
        )

        # eq: duration(t) >= duration(t - 1) + dt(t) + (On(t) - 1) * BIG
        assert_conequal(
            model.constraints['Sink(Wärme)|downtime|backward'],
            model.variables['Sink(Wärme)|downtime'].isel(time=slice(1, None))
            >= model.variables['Sink(Wärme)|downtime'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1))
            + (model.variables['Sink(Wärme)|inactive'].isel(time=slice(1, None)) - 1) * mega,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|downtime|initial'],
            model.variables['Sink(Wärme)|downtime'].isel(time=0)
            == model.variables['Sink(Wärme)|inactive'].isel(time=0) * (model.hours_per_step.isel(time=0) * (1 + 1)),
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|downtime|lb'],
            model.variables['Sink(Wärme)|downtime']
            >= (
                model.variables['Sink(Wärme)|inactive'].isel(time=slice(None, -1))
                - model.variables['Sink(Wärme)|inactive'].isel(time=slice(1, None))
            )
            * 4,
        )

    def test_consecutive_off_hours_previous(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with minimum and maximum consecutive inactive hours."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            status_parameters=fx.StatusParameters(
                min_downtime=4,  # Must stay inactive for at least 4 hours when shut down
                max_downtime=12,  # Can't be inactive for more than 12 consecutive hours
            ),
            previous_flow_rate=np.array([10, 20, 30, 0, 20, 0, 0]),  # Previously inactive for 2 steps
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert {'Sink(Wärme)|downtime', 'Sink(Wärme)|inactive'}.issubset(set(flow.submodel.variables))

        assert_sets_equal(
            {
                'Sink(Wärme)|downtime|ub',
                'Sink(Wärme)|downtime|forward',
                'Sink(Wärme)|downtime|backward',
                'Sink(Wärme)|downtime|initial',
                'Sink(Wärme)|downtime|lb',
            }
            & set(flow.submodel.constraints),
            {
                'Sink(Wärme)|downtime|ub',
                'Sink(Wärme)|downtime|forward',
                'Sink(Wärme)|downtime|backward',
                'Sink(Wärme)|downtime|initial',
                'Sink(Wärme)|downtime|lb',
            },
            msg='Missing consecutive inactive hours constraints for previous states',
        )

        assert_var_equal(
            model.variables['Sink(Wärme)|downtime'],
            model.add_variables(lower=0, upper=12, coords=model.get_coords()),
        )

        mega = model.hours_per_step.sum('time') + model.hours_per_step.isel(time=0) * 2

        assert_conequal(
            model.constraints['Sink(Wärme)|downtime|ub'],
            model.variables['Sink(Wärme)|downtime'] <= model.variables['Sink(Wärme)|inactive'] * mega,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|downtime|forward'],
            model.variables['Sink(Wärme)|downtime'].isel(time=slice(1, None))
            <= model.variables['Sink(Wärme)|downtime'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1)),
        )

        # eq: duration(t) >= duration(t - 1) + dt(t) + (On(t) - 1) * BIG
        assert_conequal(
            model.constraints['Sink(Wärme)|downtime|backward'],
            model.variables['Sink(Wärme)|downtime'].isel(time=slice(1, None))
            >= model.variables['Sink(Wärme)|downtime'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1))
            + (model.variables['Sink(Wärme)|inactive'].isel(time=slice(1, None)) - 1) * mega,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|downtime|initial'],
            model.variables['Sink(Wärme)|downtime'].isel(time=0)
            == model.variables['Sink(Wärme)|inactive'].isel(time=0) * (model.hours_per_step.isel(time=0) * (1 + 2)),
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|downtime|lb'],
            model.variables['Sink(Wärme)|downtime']
            >= (
                model.variables['Sink(Wärme)|inactive'].isel(time=slice(None, -1))
                - model.variables['Sink(Wärme)|inactive'].isel(time=slice(1, None))
            )
            * 4,
        )

    def test_switch_on_constraints(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with constraints on the number of startups."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            status_parameters=fx.StatusParameters(
                startup_limit=5,  # Maximum 5 startups
                effects_per_startup={'costs': 100},  # 100 EUR startup cost
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that variables exist
        assert {'Sink(Wärme)|startup', 'Sink(Wärme)|shutdown', 'Sink(Wärme)|startup_count'}.issubset(
            set(flow.submodel.variables)
        )

        # Check that constraints exist
        assert_sets_equal(
            {
                'Sink(Wärme)|switch|transition',
                'Sink(Wärme)|switch|initial',
                'Sink(Wärme)|switch|mutex',
                'Sink(Wärme)|startup_count',
            }
            & set(flow.submodel.constraints),
            {
                'Sink(Wärme)|switch|transition',
                'Sink(Wärme)|switch|initial',
                'Sink(Wärme)|switch|mutex',
                'Sink(Wärme)|startup_count',
            },
            msg='Missing switch constraints',
        )

        # Check startup_count variable bounds
        assert_var_equal(
            flow.submodel.variables['Sink(Wärme)|startup_count'],
            model.add_variables(lower=0, upper=5, coords=model.get_coords(['period', 'scenario'])),
        )

        # Verify startup_count constraint (limits number of startups)
        assert_conequal(
            model.constraints['Sink(Wärme)|startup_count'],
            flow.submodel.variables['Sink(Wärme)|startup_count']
            == flow.submodel.variables['Sink(Wärme)|startup'].sum('time'),
        )

        # Check that startup cost effect constraint exists
        assert 'Sink(Wärme)->costs(temporal)' in model.constraints

        # Verify the startup cost effect constraint
        assert_conequal(
            model.constraints['Sink(Wärme)->costs(temporal)'],
            model.variables['Sink(Wärme)->costs(temporal)'] == flow.submodel.variables['Sink(Wärme)|startup'] * 100,
        )

    def test_on_hours_limits(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with limits on total active hours."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            status_parameters=fx.StatusParameters(
                active_hours_min=20,  # Minimum 20 hours of operation
                active_hours_max=100,  # Maximum 100 hours of operation
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that variables exist
        assert {'Sink(Wärme)|status', 'Sink(Wärme)|active_hours'}.issubset(set(flow.submodel.variables))

        # Check that constraints exist
        assert 'Sink(Wärme)|active_hours' in model.constraints

        # Check active_hours variable bounds
        assert_var_equal(
            flow.submodel.variables['Sink(Wärme)|active_hours'],
            model.add_variables(lower=20, upper=100, coords=model.get_coords(['period', 'scenario'])),
        )

        # Check active_hours constraint
        assert_conequal(
            model.constraints['Sink(Wärme)|active_hours'],
            flow.submodel.variables['Sink(Wärme)|active_hours']
            == (flow.submodel.variables['Sink(Wärme)|status'] * model.hours_per_step).sum('time'),
        )


class TestFlowOnInvestModel:
    """Test the FlowModel class."""

    def test_flow_on_invest_optional(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(minimum_size=20, maximum_size=200, mandatory=False),
            relative_minimum=0.2,
            relative_maximum=0.8,
            status_parameters=fx.StatusParameters(),
        )
        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert_sets_equal(
            set(flow.submodel.variables),
            {
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|flow_rate',
                'Sink(Wärme)|invested',
                'Sink(Wärme)|size',
                'Sink(Wärme)|status',
                'Sink(Wärme)|active_hours',
            },
            msg='Incorrect variables',
        )

        assert_sets_equal(
            set(flow.submodel.constraints),
            {
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|active_hours',
                'Sink(Wärme)|flow_rate|lb1',
                'Sink(Wärme)|flow_rate|ub1',
                'Sink(Wärme)|size|lb',
                'Sink(Wärme)|size|ub',
                'Sink(Wärme)|flow_rate|lb2',
                'Sink(Wärme)|flow_rate|ub2',
            },
            msg='Incorrect constraints',
        )

        # flow_rate
        assert_var_equal(
            flow.submodel.flow_rate,
            model.add_variables(
                lower=0,
                upper=0.8 * 200,
                coords=model.get_coords(),
            ),
        )

        # Status
        assert_var_equal(
            flow.submodel.status.status,
            model.add_variables(binary=True, coords=model.get_coords()),
        )
        assert_var_equal(
            model.variables['Sink(Wärme)|active_hours'],
            model.add_variables(lower=0, coords=model.get_coords(['period', 'scenario'])),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|size|lb'],
            flow.submodel.variables['Sink(Wärme)|size'] >= flow.submodel.variables['Sink(Wärme)|invested'] * 20,
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|size|ub'],
            flow.submodel.variables['Sink(Wärme)|size'] <= flow.submodel.variables['Sink(Wärme)|invested'] * 200,
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|lb1'],
            flow.submodel.variables['Sink(Wärme)|status'] * 0.2 * 20
            <= flow.submodel.variables['Sink(Wärme)|flow_rate'],
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|ub1'],
            flow.submodel.variables['Sink(Wärme)|status'] * 0.8 * 200
            >= flow.submodel.variables['Sink(Wärme)|flow_rate'],
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|active_hours'],
            flow.submodel.variables['Sink(Wärme)|active_hours']
            == (flow.submodel.variables['Sink(Wärme)|status'] * model.hours_per_step).sum('time'),
        )

        # Investment
        assert_var_equal(
            model['Sink(Wärme)|size'],
            model.add_variables(lower=0, upper=200, coords=model.get_coords(['period', 'scenario'])),
        )

        mega = 0.2 * 200  # Relative minimum * maximum size
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|lb2'],
            flow.submodel.variables['Sink(Wärme)|flow_rate']
            >= flow.submodel.variables['Sink(Wärme)|status'] * mega
            + flow.submodel.variables['Sink(Wärme)|size'] * 0.2
            - mega,
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|ub2'],
            flow.submodel.variables['Sink(Wärme)|flow_rate'] <= flow.submodel.variables['Sink(Wärme)|size'] * 0.8,
        )

    def test_flow_on_invest_non_optional(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(minimum_size=20, maximum_size=200, mandatory=True),
            relative_minimum=0.2,
            relative_maximum=0.8,
            status_parameters=fx.StatusParameters(),
        )
        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert_sets_equal(
            set(flow.submodel.variables),
            {
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|flow_rate',
                'Sink(Wärme)|size',
                'Sink(Wärme)|status',
                'Sink(Wärme)|active_hours',
            },
            msg='Incorrect variables',
        )

        assert_sets_equal(
            set(flow.submodel.constraints),
            {
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|active_hours',
                'Sink(Wärme)|flow_rate|lb1',
                'Sink(Wärme)|flow_rate|ub1',
                'Sink(Wärme)|flow_rate|lb2',
                'Sink(Wärme)|flow_rate|ub2',
            },
            msg='Incorrect constraints',
        )

        # flow_rate
        assert_var_equal(
            flow.submodel.flow_rate,
            model.add_variables(
                lower=0,
                upper=0.8 * 200,
                coords=model.get_coords(),
            ),
        )

        # Status
        assert_var_equal(
            flow.submodel.status.status,
            model.add_variables(binary=True, coords=model.get_coords()),
        )
        assert_var_equal(
            model.variables['Sink(Wärme)|active_hours'],
            model.add_variables(lower=0, coords=model.get_coords(['period', 'scenario'])),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|lb1'],
            flow.submodel.variables['Sink(Wärme)|status'] * 0.2 * 20
            <= flow.submodel.variables['Sink(Wärme)|flow_rate'],
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|ub1'],
            flow.submodel.variables['Sink(Wärme)|status'] * 0.8 * 200
            >= flow.submodel.variables['Sink(Wärme)|flow_rate'],
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|active_hours'],
            flow.submodel.variables['Sink(Wärme)|active_hours']
            == (flow.submodel.variables['Sink(Wärme)|status'] * model.hours_per_step).sum('time'),
        )

        # Investment
        assert_var_equal(
            model['Sink(Wärme)|size'],
            model.add_variables(lower=20, upper=200, coords=model.get_coords(['period', 'scenario'])),
        )

        mega = 0.2 * 200  # Relative minimum * maximum size
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|lb2'],
            flow.submodel.variables['Sink(Wärme)|flow_rate']
            >= flow.submodel.variables['Sink(Wärme)|status'] * mega
            + flow.submodel.variables['Sink(Wärme)|size'] * 0.2
            - mega,
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|ub2'],
            flow.submodel.variables['Sink(Wärme)|flow_rate'] <= flow.submodel.variables['Sink(Wärme)|size'] * 0.8,
        )


class TestFlowWithFixedProfile:
    """Test Flow with fixed relative profile."""

    def test_fixed_relative_profile(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with a fixed relative profile."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        # Create a time-varying profile (e.g., for a load or renewable generation)
        profile = np.sin(np.linspace(0, 2 * np.pi, len(timesteps))) * 0.5 + 0.5  # Values between 0 and 1

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            fixed_relative_profile=profile,
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert_var_equal(
            flow.submodel.variables['Sink(Wärme)|flow_rate'],
            model.add_variables(
                lower=flow.fixed_relative_profile * 100,
                upper=flow.fixed_relative_profile * 100,
                coords=model.get_coords(),
            ),
        )

    def test_fixed_profile_with_investment(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with fixed profile and investment."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        # Create a fixed profile
        profile = np.sin(np.linspace(0, 2 * np.pi, len(timesteps))) * 0.5 + 0.5

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=fx.InvestParameters(minimum_size=50, maximum_size=200, mandatory=False),
            fixed_relative_profile=profile,
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert_var_equal(
            flow.submodel.variables['Sink(Wärme)|flow_rate'],
            model.add_variables(lower=0, upper=flow.fixed_relative_profile * 200, coords=model.get_coords()),
        )

        # The constraint should link flow_rate to size * profile
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|fixed'],
            flow.submodel.variables['Sink(Wärme)|flow_rate']
            == flow.submodel.variables['Sink(Wärme)|size'] * flow.fixed_relative_profile,
        )


if __name__ == '__main__':
    pytest.main()
