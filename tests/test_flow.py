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
            on_off_parameters=fx.OnOffParameters(),
        )
        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert_sets_equal(
            set(flow.submodel.variables),
            {'Sink(Wärme)|total_flow_hours', 'Sink(Wärme)|flow_rate', 'Sink(Wärme)|on', 'Sink(Wärme)|on_hours_total'},
            msg='Incorrect variables',
        )

        assert_sets_equal(
            set(flow.submodel.constraints),
            {
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|on_hours_total',
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

        # OnOff
        assert_var_equal(
            flow.submodel.on_off.on,
            model.add_variables(binary=True, coords=model.get_coords()),
        )
        assert_var_equal(
            model.variables['Sink(Wärme)|on_hours_total'],
            model.add_variables(lower=0, coords=model.get_coords(['period', 'scenario'])),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|lb'],
            flow.submodel.variables['Sink(Wärme)|flow_rate'] >= flow.submodel.variables['Sink(Wärme)|on'] * 0.2 * 100,
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|ub'],
            flow.submodel.variables['Sink(Wärme)|flow_rate'] <= flow.submodel.variables['Sink(Wärme)|on'] * 0.8 * 100,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|on_hours_total'],
            flow.submodel.variables['Sink(Wärme)|on_hours_total']
            == (flow.submodel.variables['Sink(Wärme)|on'] * model.hours_per_step).sum('time'),
        )

    def test_effects_per_running_hour(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        costs_per_running_hour = np.linspace(1, 2, timesteps.size)
        co2_per_running_hour = np.linspace(4, 5, timesteps.size)

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            on_off_parameters=fx.OnOffParameters(
                effects_per_running_hour={'costs': costs_per_running_hour, 'CO2': co2_per_running_hour}
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
                'Sink(Wärme)|on',
                'Sink(Wärme)|on_hours_total',
            },
            msg='Incorrect variables',
        )
        assert_sets_equal(
            set(flow.submodel.constraints),
            {
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|flow_rate|lb',
                'Sink(Wärme)|flow_rate|ub',
                'Sink(Wärme)|on_hours_total',
            },
            msg='Incorrect constraints',
        )

        assert 'Sink(Wärme)->costs(temporal)' in set(costs.submodel.constraints)
        assert 'Sink(Wärme)->CO2(temporal)' in set(co2.submodel.constraints)

        costs_per_running_hour = flow.on_off_parameters.effects_per_running_hour['costs']
        co2_per_running_hour = flow.on_off_parameters.effects_per_running_hour['CO2']

        assert costs_per_running_hour.dims == tuple(model.get_coords())
        assert co2_per_running_hour.dims == tuple(model.get_coords())

        assert_conequal(
            model.constraints['Sink(Wärme)->costs(temporal)'],
            model.variables['Sink(Wärme)->costs(temporal)']
            == flow.submodel.variables['Sink(Wärme)|on'] * model.hours_per_step * costs_per_running_hour,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)->CO2(temporal)'],
            model.variables['Sink(Wärme)->CO2(temporal)']
            == flow.submodel.variables['Sink(Wärme)|on'] * model.hours_per_step * co2_per_running_hour,
        )

    def test_consecutive_on_hours(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with minimum and maximum consecutive on hours."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            on_off_parameters=fx.OnOffParameters(
                consecutive_on_hours_min=2,  # Must run for at least 2 hours when turned on
                consecutive_on_hours_max=8,  # Can't run more than 8 consecutive hours
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert {'Sink(Wärme)|consecutive_on_hours', 'Sink(Wärme)|on'}.issubset(set(flow.submodel.variables))

        assert_sets_equal(
            {
                'Sink(Wärme)|consecutive_on_hours|ub',
                'Sink(Wärme)|consecutive_on_hours|forward',
                'Sink(Wärme)|consecutive_on_hours|backward',
                'Sink(Wärme)|consecutive_on_hours|initial',
                'Sink(Wärme)|consecutive_on_hours|lb',
            }
            & set(flow.submodel.constraints),
            {
                'Sink(Wärme)|consecutive_on_hours|ub',
                'Sink(Wärme)|consecutive_on_hours|forward',
                'Sink(Wärme)|consecutive_on_hours|backward',
                'Sink(Wärme)|consecutive_on_hours|initial',
                'Sink(Wärme)|consecutive_on_hours|lb',
            },
            msg='Missing consecutive on hours constraints',
        )

        assert_var_equal(
            model.variables['Sink(Wärme)|consecutive_on_hours'],
            model.add_variables(lower=0, upper=8, coords=model.get_coords()),
        )

        mega = model.hours_per_step.sum('time')

        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_on_hours|ub'],
            model.variables['Sink(Wärme)|consecutive_on_hours'] <= model.variables['Sink(Wärme)|on'] * mega,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_on_hours|forward'],
            model.variables['Sink(Wärme)|consecutive_on_hours'].isel(time=slice(1, None))
            <= model.variables['Sink(Wärme)|consecutive_on_hours'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1)),
        )

        # eq: duration(t) >= duration(t - 1) + dt(t) + (On(t) - 1) * BIG
        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_on_hours|backward'],
            model.variables['Sink(Wärme)|consecutive_on_hours'].isel(time=slice(1, None))
            >= model.variables['Sink(Wärme)|consecutive_on_hours'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1))
            + (model.variables['Sink(Wärme)|on'].isel(time=slice(1, None)) - 1) * mega,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_on_hours|initial'],
            model.variables['Sink(Wärme)|consecutive_on_hours'].isel(time=0)
            == model.variables['Sink(Wärme)|on'].isel(time=0) * model.hours_per_step.isel(time=0),
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_on_hours|lb'],
            model.variables['Sink(Wärme)|consecutive_on_hours']
            >= (
                model.variables['Sink(Wärme)|on'].isel(time=slice(None, -1))
                - model.variables['Sink(Wärme)|on'].isel(time=slice(1, None))
            )
            * 2,
        )

    def test_consecutive_on_hours_previous(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with minimum and maximum consecutive on hours."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            on_off_parameters=fx.OnOffParameters(
                consecutive_on_hours_min=2,  # Must run for at least 2 hours when turned on
                consecutive_on_hours_max=8,  # Can't run more than 8 consecutive hours
            ),
            previous_flow_rate=np.array([10, 20, 30, 0, 20, 20, 30]),  # Previously on for 3 steps
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert {'Sink(Wärme)|consecutive_on_hours', 'Sink(Wärme)|on'}.issubset(set(flow.submodel.variables))

        assert_sets_equal(
            {
                'Sink(Wärme)|consecutive_on_hours|lb',
                'Sink(Wärme)|consecutive_on_hours|forward',
                'Sink(Wärme)|consecutive_on_hours|backward',
                'Sink(Wärme)|consecutive_on_hours|initial',
            }
            & set(flow.submodel.constraints),
            {
                'Sink(Wärme)|consecutive_on_hours|lb',
                'Sink(Wärme)|consecutive_on_hours|forward',
                'Sink(Wärme)|consecutive_on_hours|backward',
                'Sink(Wärme)|consecutive_on_hours|initial',
            },
            msg='Missing consecutive on hours constraints for previous states',
        )

        assert_var_equal(
            model.variables['Sink(Wärme)|consecutive_on_hours'],
            model.add_variables(lower=0, upper=8, coords=model.get_coords()),
        )

        mega = model.hours_per_step.sum('time') + model.hours_per_step.isel(time=0) * 3

        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_on_hours|ub'],
            model.variables['Sink(Wärme)|consecutive_on_hours'] <= model.variables['Sink(Wärme)|on'] * mega,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_on_hours|forward'],
            model.variables['Sink(Wärme)|consecutive_on_hours'].isel(time=slice(1, None))
            <= model.variables['Sink(Wärme)|consecutive_on_hours'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1)),
        )

        # eq: duration(t) >= duration(t - 1) + dt(t) + (On(t) - 1) * BIG
        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_on_hours|backward'],
            model.variables['Sink(Wärme)|consecutive_on_hours'].isel(time=slice(1, None))
            >= model.variables['Sink(Wärme)|consecutive_on_hours'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1))
            + (model.variables['Sink(Wärme)|on'].isel(time=slice(1, None)) - 1) * mega,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_on_hours|initial'],
            model.variables['Sink(Wärme)|consecutive_on_hours'].isel(time=0)
            == model.variables['Sink(Wärme)|on'].isel(time=0) * (model.hours_per_step.isel(time=0) * (1 + 3)),
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_on_hours|lb'],
            model.variables['Sink(Wärme)|consecutive_on_hours']
            >= (
                model.variables['Sink(Wärme)|on'].isel(time=slice(None, -1))
                - model.variables['Sink(Wärme)|on'].isel(time=slice(1, None))
            )
            * 2,
        )

    def test_consecutive_off_hours(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with minimum and maximum consecutive off hours."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            on_off_parameters=fx.OnOffParameters(
                consecutive_off_hours_min=4,  # Must stay off for at least 4 hours when shut down
                consecutive_off_hours_max=12,  # Can't be off for more than 12 consecutive hours
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert {'Sink(Wärme)|consecutive_off_hours', 'Sink(Wärme)|off'}.issubset(set(flow.submodel.variables))

        assert_sets_equal(
            {
                'Sink(Wärme)|consecutive_off_hours|ub',
                'Sink(Wärme)|consecutive_off_hours|forward',
                'Sink(Wärme)|consecutive_off_hours|backward',
                'Sink(Wärme)|consecutive_off_hours|initial',
                'Sink(Wärme)|consecutive_off_hours|lb',
            }
            & set(flow.submodel.constraints),
            {
                'Sink(Wärme)|consecutive_off_hours|ub',
                'Sink(Wärme)|consecutive_off_hours|forward',
                'Sink(Wärme)|consecutive_off_hours|backward',
                'Sink(Wärme)|consecutive_off_hours|initial',
                'Sink(Wärme)|consecutive_off_hours|lb',
            },
            msg='Missing consecutive off hours constraints',
        )

        assert_var_equal(
            model.variables['Sink(Wärme)|consecutive_off_hours'],
            model.add_variables(lower=0, upper=12, coords=model.get_coords()),
        )

        mega = model.hours_per_step.sum('time') + model.hours_per_step.isel(time=0) * 1  # previously off for 1h

        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_off_hours|ub'],
            model.variables['Sink(Wärme)|consecutive_off_hours'] <= model.variables['Sink(Wärme)|off'] * mega,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_off_hours|forward'],
            model.variables['Sink(Wärme)|consecutive_off_hours'].isel(time=slice(1, None))
            <= model.variables['Sink(Wärme)|consecutive_off_hours'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1)),
        )

        # eq: duration(t) >= duration(t - 1) + dt(t) + (On(t) - 1) * BIG
        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_off_hours|backward'],
            model.variables['Sink(Wärme)|consecutive_off_hours'].isel(time=slice(1, None))
            >= model.variables['Sink(Wärme)|consecutive_off_hours'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1))
            + (model.variables['Sink(Wärme)|off'].isel(time=slice(1, None)) - 1) * mega,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_off_hours|initial'],
            model.variables['Sink(Wärme)|consecutive_off_hours'].isel(time=0)
            == model.variables['Sink(Wärme)|off'].isel(time=0) * (model.hours_per_step.isel(time=0) * (1 + 1)),
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_off_hours|lb'],
            model.variables['Sink(Wärme)|consecutive_off_hours']
            >= (
                model.variables['Sink(Wärme)|off'].isel(time=slice(None, -1))
                - model.variables['Sink(Wärme)|off'].isel(time=slice(1, None))
            )
            * 4,
        )

    def test_consecutive_off_hours_previous(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with minimum and maximum consecutive off hours."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            on_off_parameters=fx.OnOffParameters(
                consecutive_off_hours_min=4,  # Must stay off for at least 4 hours when shut down
                consecutive_off_hours_max=12,  # Can't be off for more than 12 consecutive hours
            ),
            previous_flow_rate=np.array([10, 20, 30, 0, 20, 0, 0]),  # Previously off for 2 steps
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert {'Sink(Wärme)|consecutive_off_hours', 'Sink(Wärme)|off'}.issubset(set(flow.submodel.variables))

        assert_sets_equal(
            {
                'Sink(Wärme)|consecutive_off_hours|ub',
                'Sink(Wärme)|consecutive_off_hours|forward',
                'Sink(Wärme)|consecutive_off_hours|backward',
                'Sink(Wärme)|consecutive_off_hours|initial',
                'Sink(Wärme)|consecutive_off_hours|lb',
            }
            & set(flow.submodel.constraints),
            {
                'Sink(Wärme)|consecutive_off_hours|ub',
                'Sink(Wärme)|consecutive_off_hours|forward',
                'Sink(Wärme)|consecutive_off_hours|backward',
                'Sink(Wärme)|consecutive_off_hours|initial',
                'Sink(Wärme)|consecutive_off_hours|lb',
            },
            msg='Missing consecutive off hours constraints for previous states',
        )

        assert_var_equal(
            model.variables['Sink(Wärme)|consecutive_off_hours'],
            model.add_variables(lower=0, upper=12, coords=model.get_coords()),
        )

        mega = model.hours_per_step.sum('time') + model.hours_per_step.isel(time=0) * 2

        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_off_hours|ub'],
            model.variables['Sink(Wärme)|consecutive_off_hours'] <= model.variables['Sink(Wärme)|off'] * mega,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_off_hours|forward'],
            model.variables['Sink(Wärme)|consecutive_off_hours'].isel(time=slice(1, None))
            <= model.variables['Sink(Wärme)|consecutive_off_hours'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1)),
        )

        # eq: duration(t) >= duration(t - 1) + dt(t) + (On(t) - 1) * BIG
        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_off_hours|backward'],
            model.variables['Sink(Wärme)|consecutive_off_hours'].isel(time=slice(1, None))
            >= model.variables['Sink(Wärme)|consecutive_off_hours'].isel(time=slice(None, -1))
            + model.hours_per_step.isel(time=slice(None, -1))
            + (model.variables['Sink(Wärme)|off'].isel(time=slice(1, None)) - 1) * mega,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_off_hours|initial'],
            model.variables['Sink(Wärme)|consecutive_off_hours'].isel(time=0)
            == model.variables['Sink(Wärme)|off'].isel(time=0) * (model.hours_per_step.isel(time=0) * (1 + 2)),
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|consecutive_off_hours|lb'],
            model.variables['Sink(Wärme)|consecutive_off_hours']
            >= (
                model.variables['Sink(Wärme)|off'].isel(time=slice(None, -1))
                - model.variables['Sink(Wärme)|off'].isel(time=slice(1, None))
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
            on_off_parameters=fx.OnOffParameters(
                switch_on_max=5,  # Maximum 5 startups
                effects_per_switch_on={'costs': 100},  # 100 EUR startup cost
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that variables exist
        assert {'Sink(Wärme)|switch|on', 'Sink(Wärme)|switch|off', 'Sink(Wärme)|switch|count'}.issubset(
            set(flow.submodel.variables)
        )

        # Check that constraints exist
        assert_sets_equal(
            {
                'Sink(Wärme)|switch|transition',
                'Sink(Wärme)|switch|initial',
                'Sink(Wärme)|switch|mutex',
                'Sink(Wärme)|switch|count',
            }
            & set(flow.submodel.constraints),
            {
                'Sink(Wärme)|switch|transition',
                'Sink(Wärme)|switch|initial',
                'Sink(Wärme)|switch|mutex',
                'Sink(Wärme)|switch|count',
            },
            msg='Missing switch constraints',
        )

        # Check switch_on_nr variable bounds
        assert_var_equal(
            flow.submodel.variables['Sink(Wärme)|switch|count'],
            model.add_variables(lower=0, upper=5, coords=model.get_coords(['period', 'scenario'])),
        )

        # Verify switch_on_nr constraint (limits number of startups)
        assert_conequal(
            model.constraints['Sink(Wärme)|switch|count'],
            flow.submodel.variables['Sink(Wärme)|switch|count']
            == flow.submodel.variables['Sink(Wärme)|switch|on'].sum('time'),
        )

        # Check that startup cost effect constraint exists
        assert 'Sink(Wärme)->costs(temporal)' in model.constraints

        # Verify the startup cost effect constraint
        assert_conequal(
            model.constraints['Sink(Wärme)->costs(temporal)'],
            model.variables['Sink(Wärme)->costs(temporal)'] == flow.submodel.variables['Sink(Wärme)|switch|on'] * 100,
        )

    def test_on_hours_limits(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with limits on total on hours."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            on_off_parameters=fx.OnOffParameters(
                on_hours_min=20,  # Minimum 20 hours of operation
                on_hours_max=100,  # Maximum 100 hours of operation
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that variables exist
        assert {'Sink(Wärme)|on', 'Sink(Wärme)|on_hours_total'}.issubset(set(flow.submodel.variables))

        # Check that constraints exist
        assert 'Sink(Wärme)|on_hours_total' in model.constraints

        # Check on_hours_total variable bounds
        assert_var_equal(
            flow.submodel.variables['Sink(Wärme)|on_hours_total'],
            model.add_variables(lower=20, upper=100, coords=model.get_coords(['period', 'scenario'])),
        )

        # Check on_hours_total constraint
        assert_conequal(
            model.constraints['Sink(Wärme)|on_hours_total'],
            flow.submodel.variables['Sink(Wärme)|on_hours_total']
            == (flow.submodel.variables['Sink(Wärme)|on'] * model.hours_per_step).sum('time'),
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
            on_off_parameters=fx.OnOffParameters(),
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
                'Sink(Wärme)|on',
                'Sink(Wärme)|on_hours_total',
            },
            msg='Incorrect variables',
        )

        assert_sets_equal(
            set(flow.submodel.constraints),
            {
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|on_hours_total',
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

        # OnOff
        assert_var_equal(
            flow.submodel.on_off.on,
            model.add_variables(binary=True, coords=model.get_coords()),
        )
        assert_var_equal(
            model.variables['Sink(Wärme)|on_hours_total'],
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
            flow.submodel.variables['Sink(Wärme)|on'] * 0.2 * 20 <= flow.submodel.variables['Sink(Wärme)|flow_rate'],
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|ub1'],
            flow.submodel.variables['Sink(Wärme)|on'] * 0.8 * 200 >= flow.submodel.variables['Sink(Wärme)|flow_rate'],
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|on_hours_total'],
            flow.submodel.variables['Sink(Wärme)|on_hours_total']
            == (flow.submodel.variables['Sink(Wärme)|on'] * model.hours_per_step).sum('time'),
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
            >= flow.submodel.variables['Sink(Wärme)|on'] * mega
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
            on_off_parameters=fx.OnOffParameters(),
        )
        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        assert_sets_equal(
            set(flow.submodel.variables),
            {
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|flow_rate',
                'Sink(Wärme)|size',
                'Sink(Wärme)|on',
                'Sink(Wärme)|on_hours_total',
            },
            msg='Incorrect variables',
        )

        assert_sets_equal(
            set(flow.submodel.constraints),
            {
                'Sink(Wärme)|total_flow_hours',
                'Sink(Wärme)|on_hours_total',
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

        # OnOff
        assert_var_equal(
            flow.submodel.on_off.on,
            model.add_variables(binary=True, coords=model.get_coords()),
        )
        assert_var_equal(
            model.variables['Sink(Wärme)|on_hours_total'],
            model.add_variables(lower=0, coords=model.get_coords(['period', 'scenario'])),
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|lb1'],
            flow.submodel.variables['Sink(Wärme)|on'] * 0.2 * 20 <= flow.submodel.variables['Sink(Wärme)|flow_rate'],
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|flow_rate|ub1'],
            flow.submodel.variables['Sink(Wärme)|on'] * 0.8 * 200 >= flow.submodel.variables['Sink(Wärme)|flow_rate'],
        )
        assert_conequal(
            model.constraints['Sink(Wärme)|on_hours_total'],
            flow.submodel.variables['Sink(Wärme)|on_hours_total']
            == (flow.submodel.variables['Sink(Wärme)|on'] * model.hours_per_step).sum('time'),
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
            >= flow.submodel.variables['Sink(Wärme)|on'] * mega
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
