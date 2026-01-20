import numpy as np
import pytest
import xarray as xr

import flixopt as fx

from .conftest import assert_conequal, assert_dims_compatible, assert_sets_equal, assert_var_equal, create_linopy_model


class TestFlowModel:
    """Test the FlowModel class."""

    def test_flow_minimal(self, basic_flow_system_linopy_coords, coords_config):
        """Test that flow model constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow('Wärme', bus='Fernwärme', size=100)

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))

        model = create_linopy_model(flow_system)

        # Constraints are now batched at type-level, select specific flow
        assert_conequal(
            model.constraints['flow|hours_eq'].sel(flow='Sink(Wärme)'),
            flow.submodel.total_flow_hours == (flow.submodel.flow_rate * model.timestep_duration).sum('time'),
        )
        assert_var_equal(flow.submodel.flow_rate, model.add_variables(lower=0, upper=100, coords=model.get_coords()))
        assert_var_equal(
            flow.submodel.total_flow_hours,
            model.add_variables(lower=0, coords=model.get_coords(['period', 'scenario'])),
        )

        # Variables are registered with short names in submodel
        assert_sets_equal(
            set(flow.submodel._variables.keys()),
            {'total_flow_hours', 'flow_rate'},
            msg='Incorrect variables',
        )
        # Constraints are now at type-level (batched), submodel constraints are empty
        assert_sets_equal(
            set(flow.submodel._constraints.keys()), set(), msg='Batched model has no per-element constraints'
        )

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

        # total_flow_hours - now batched at type-level
        assert_conequal(
            model.constraints['flow|hours_eq'].sel(flow='Sink(Wärme)'),
            flow.submodel.total_flow_hours == (flow.submodel.flow_rate * model.timestep_duration).sum('time'),
        )

        assert_var_equal(
            flow.submodel.total_flow_hours,
            model.add_variables(lower=10, upper=1000, coords=model.get_coords(['period', 'scenario'])),
        )

        assert_dims_compatible(flow.relative_minimum, tuple(model.get_coords()))
        assert_dims_compatible(flow.relative_maximum, tuple(model.get_coords()))

        assert_var_equal(
            flow.submodel.flow_rate,
            model.add_variables(
                lower=flow.relative_minimum * 100,
                upper=flow.relative_maximum * 100,
                coords=model.get_coords(),
            ),
        )

        # load_factor constraints - now batched at type-level
        assert_conequal(
            model.constraints['flow|load_factor_min'].sel(flow='Sink(Wärme)'),
            flow.submodel.total_flow_hours >= model.timestep_duration.sum('time') * 0.1 * 100,
        )

        assert_conequal(
            model.constraints['flow|load_factor_max'].sel(flow='Sink(Wärme)'),
            flow.submodel.total_flow_hours <= model.timestep_duration.sum('time') * 0.9 * 100,
        )

        # Submodel uses short names
        assert_sets_equal(
            set(flow.submodel._variables.keys()),
            {'total_flow_hours', 'flow_rate'},
            msg='Incorrect variables',
        )
        # Constraints are now at type-level (batched), submodel constraints are empty
        assert_sets_equal(
            set(flow.submodel._constraints.keys()),
            set(),
            msg='Batched model has no per-element constraints',
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

        # Submodel uses short names
        assert_sets_equal(
            set(flow.submodel._variables.keys()),
            {'total_flow_hours', 'flow_rate'},
            msg='Incorrect variables',
        )
        # Constraints are now at type-level (batched)
        assert_sets_equal(
            set(flow.submodel._constraints.keys()), set(), msg='Batched model has no per-element constraints'
        )

        # In type-level mode, effects don't have per-element submodels
        # Effects are managed by the batched EffectsModel
        assert costs.submodel is None, 'Effect submodels are not created in type-level mode'
        assert co2.submodel is None, 'Effect submodels are not created in type-level mode'

        # Batched temporal shares are managed by the EffectsModel
        assert 'share|temporal' in model.constraints, 'Batched temporal share constraint should exist'

        # The flow's effects are included in the batched constraints
        # Check that the effect factors are correctly computed
        effects_model = flow_system.effects.submodel._batched_model
        assert effects_model is not None, 'Batched EffectsModel should exist'
        assert hasattr(effects_model, 'share_temporal'), 'EffectsModel should have share_temporal'


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

        # In type-level mode, flow.submodel._variables uses short names
        assert_sets_equal(
            set(flow.submodel._variables.keys()),
            {'total_flow_hours', 'flow_rate', 'size'},
            msg='Incorrect variables',
        )
        # Type-level mode has no per-element constraints (they're batched)
        assert_sets_equal(
            set(flow.submodel._constraints.keys()),
            set(),
            msg='Batched model has no per-element constraints',
        )

        # Check batched variables exist
        assert 'flow|size' in model.variables, 'Batched size variable should exist'
        assert 'flow|rate' in model.variables, 'Batched rate variable should exist'

        # Check batched constraints exist
        assert 'flow|rate_invest_lb' in model.constraints, 'Batched rate lower bound constraint should exist'
        assert 'flow|rate_invest_ub' in model.constraints, 'Batched rate upper bound constraint should exist'

        assert_dims_compatible(flow.relative_minimum, tuple(model.get_coords()))
        assert_dims_compatible(flow.relative_maximum, tuple(model.get_coords()))

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

        # In type-level mode, flow.submodel._variables uses short names
        assert_sets_equal(
            set(flow.submodel._variables.keys()),
            {'total_flow_hours', 'flow_rate', 'size', 'invested'},
            msg='Incorrect variables',
        )
        # Type-level mode has no per-element constraints (they're batched)
        assert_sets_equal(
            set(flow.submodel._constraints.keys()),
            set(),
            msg='Batched model has no per-element constraints',
        )

        # Check batched variables exist
        assert 'flow|size' in model.variables, 'Batched size variable should exist'
        assert 'flow|invested' in model.variables, 'Batched invested variable should exist'
        assert 'flow|rate' in model.variables, 'Batched rate variable should exist'

        # Check batched constraints exist
        assert 'flow|rate_invest_lb' in model.constraints, 'Batched rate lower bound constraint should exist'
        assert 'flow|rate_invest_ub' in model.constraints, 'Batched rate upper bound constraint should exist'
        assert 'flow|size|lb' in model.constraints, 'Batched size lower bound constraint should exist'
        assert 'flow|size|ub' in model.constraints, 'Batched size upper bound constraint should exist'

        assert_dims_compatible(flow.relative_minimum, tuple(model.get_coords()))
        assert_dims_compatible(flow.relative_maximum, tuple(model.get_coords()))

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

        assert_dims_compatible(flow.relative_minimum, tuple(model.get_coords()))
        assert_dims_compatible(flow.relative_maximum, tuple(model.get_coords()))

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

        assert_dims_compatible(flow.relative_minimum, tuple(model.get_coords()))
        assert_dims_compatible(flow.relative_maximum, tuple(model.get_coords()))

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
        flow_label = 'Sink(Wärme)'

        # Verify batched variables exist and have flow dimension
        assert 'flow|rate' in model.variables
        assert 'flow|status' in model.variables
        assert 'flow|active_hours' in model.variables
        assert 'flow|hours' in model.variables

        # Verify batched constraints exist
        assert 'flow|rate_status_lb' in model.constraints
        assert 'flow|rate_status_ub' in model.constraints
        assert 'flow|active_hours' in model.constraints
        assert 'flow|hours_eq' in model.constraints

        # Get individual flow variables
        flow_rate = model.variables['flow|rate'].sel(flow=flow_label, drop=True)
        status = model.variables['flow|status'].sel(flow=flow_label, drop=True)
        active_hours = model.variables['flow|active_hours'].sel(flow=flow_label, drop=True)

        # flow_rate
        assert_var_equal(
            flow_rate,
            model.add_variables(
                lower=0,
                upper=0.8 * 100,
                coords=model.get_coords(),
            ),
        )

        # Status
        assert_var_equal(status, model.add_variables(binary=True, coords=model.get_coords()))

        # Upper bound is total hours when active_hours_max is not specified
        total_hours = model.timestep_duration.sum('time')
        assert_var_equal(
            active_hours,
            model.add_variables(lower=0, upper=total_hours, coords=model.get_coords(['period', 'scenario'])),
        )

        # Check batched constraints (select flow for comparison)
        assert_conequal(
            model.constraints['flow|rate_status_lb'].sel(flow=flow_label, drop=True),
            flow_rate >= status * 0.2 * 100,
        )
        assert_conequal(
            model.constraints['flow|rate_status_ub'].sel(flow=flow_label, drop=True),
            flow_rate <= status * 0.8 * 100,
        )

        assert_conequal(
            model.constraints['flow|active_hours'].sel(flow=flow_label, drop=True),
            active_hours == (status * model.timestep_duration).sum('time'),
        )

    def test_effects_per_active_hour(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        costs_per_running_hour = np.linspace(1, 2, timesteps.size)
        co2_per_running_hour = np.linspace(4, 5, timesteps.size)

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            status_parameters=fx.StatusParameters(
                effects_per_active_hour={'costs': costs_per_running_hour, 'CO2': co2_per_running_hour}
            ),
        )
        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]), fx.Effect('CO2', 't', ''))
        model = create_linopy_model(flow_system)
        flow_label = 'Sink(Wärme)'

        # Verify batched variables exist
        assert 'flow|rate' in model.variables
        assert 'flow|status' in model.variables
        assert 'flow|active_hours' in model.variables
        assert 'flow|hours' in model.variables

        # Verify batched constraints exist
        assert 'flow|rate_status_lb' in model.constraints
        assert 'flow|rate_status_ub' in model.constraints
        assert 'flow|active_hours' in model.constraints

        # Verify effect temporal constraint exists
        assert 'effect|temporal' in model.constraints

        costs_per_running_hour = flow.status_parameters.effects_per_active_hour['costs']
        co2_per_running_hour = flow.status_parameters.effects_per_active_hour['CO2']

        assert_dims_compatible(costs_per_running_hour, tuple(model.get_coords()))
        assert_dims_compatible(co2_per_running_hour, tuple(model.get_coords()))

        # Get the status variable for this flow
        _status = model.variables['flow|status'].sel(flow=flow_label, drop=True)

        # Effects are now accumulated in the batched effect|temporal variable
        # The contributions from status * timestep_duration * rate are part of the effect temporal sum
        assert 'effect|temporal' in model.variables
        assert 'costs' in model.variables['effect|temporal'].coords['effect'].values
        assert 'CO2' in model.variables['effect|temporal'].coords['effect'].values

    def test_consecutive_on_hours(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with minimum and maximum consecutive on hours."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            previous_flow_rate=0,  # Required to get initial constraint
            status_parameters=fx.StatusParameters(
                min_uptime=2,  # Must run for at least 2 hours when turned on
                max_uptime=8,  # Can't run more than 8 consecutive hours
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)
        flow_label = 'Sink(Wärme)'

        # Verify batched variables exist
        assert 'flow|uptime' in model.variables
        assert 'flow|status' in model.variables

        # Verify batched constraints exist
        assert 'flow|uptime|ub' in model.constraints
        assert 'flow|uptime|forward' in model.constraints
        assert 'flow|uptime|backward' in model.constraints
        assert 'flow|uptime|initial_ub' in model.constraints

        # Get individual flow variables
        uptime = model.variables['flow|uptime'].sel(flow=flow_label, drop=True)
        status = model.variables['flow|status'].sel(flow=flow_label, drop=True)

        assert_var_equal(uptime, model.add_variables(lower=0, upper=8, coords=model.get_coords()))

        mega = model.timestep_duration.sum('time')

        assert_conequal(
            model.constraints['flow|uptime|ub'].sel(flow=flow_label, drop=True),
            uptime <= status * mega,
        )

        assert_conequal(
            model.constraints['flow|uptime|forward'].sel(flow=flow_label, drop=True),
            uptime.isel(time=slice(1, None))
            <= uptime.isel(time=slice(None, -1)) + model.timestep_duration.isel(time=slice(None, -1)),
        )

        # eq: duration(t) >= duration(t - 1) + dt(t) + (On(t) - 1) * BIG
        assert_conequal(
            model.constraints['flow|uptime|backward'].sel(flow=flow_label, drop=True),
            uptime.isel(time=slice(1, None))
            >= uptime.isel(time=slice(None, -1))
            + model.timestep_duration.isel(time=slice(None, -1))
            + (status.isel(time=slice(1, None)) - 1) * mega,
        )

        assert_conequal(
            model.constraints['flow|uptime|initial_ub'].sel(flow=flow_label, drop=True),
            uptime.isel(time=0) <= status.isel(time=0) * model.timestep_duration.isel(time=0),
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
        flow_label = 'Sink(Wärme)'

        # Verify batched variables exist
        assert 'flow|uptime' in model.variables
        assert 'flow|status' in model.variables

        # Verify batched constraints exist
        assert 'flow|uptime|ub' in model.constraints
        assert 'flow|uptime|forward' in model.constraints
        assert 'flow|uptime|backward' in model.constraints
        assert 'flow|uptime|initial_lb' in model.constraints

        # Get individual flow variables
        uptime = model.variables['flow|uptime'].sel(flow=flow_label, drop=True)
        status = model.variables['flow|status'].sel(flow=flow_label, drop=True)

        assert_var_equal(uptime, model.add_variables(lower=0, upper=8, coords=model.get_coords()))

        mega = model.timestep_duration.sum('time') + model.timestep_duration.isel(time=0) * 3

        assert_conequal(
            model.constraints['flow|uptime|ub'].sel(flow=flow_label, drop=True),
            uptime <= status * mega,
        )

        assert_conequal(
            model.constraints['flow|uptime|forward'].sel(flow=flow_label, drop=True),
            uptime.isel(time=slice(1, None))
            <= uptime.isel(time=slice(None, -1)) + model.timestep_duration.isel(time=slice(None, -1)),
        )

        # eq: duration(t) >= duration(t - 1) + dt(t) + (On(t) - 1) * BIG
        assert_conequal(
            model.constraints['flow|uptime|backward'].sel(flow=flow_label, drop=True),
            uptime.isel(time=slice(1, None))
            >= uptime.isel(time=slice(None, -1))
            + model.timestep_duration.isel(time=slice(None, -1))
            + (status.isel(time=slice(1, None)) - 1) * mega,
        )

        # Check that initial constraint has previous uptime value incorporated
        assert_conequal(
            model.constraints['flow|uptime|initial_lb'].sel(flow=flow_label, drop=True),
            uptime.isel(time=0) >= status.isel(time=0) * (model.timestep_duration.isel(time=0) * (1 + 3)),
        )

    def test_consecutive_off_hours(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with minimum and maximum consecutive inactive hours."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            'Wärme',
            bus='Fernwärme',
            size=100,
            previous_flow_rate=0,  # Required to get initial constraint (was OFF for 1h, so previous_downtime=1)
            status_parameters=fx.StatusParameters(
                min_downtime=4,  # Must stay inactive for at least 4 hours when shut down
                max_downtime=12,  # Can't be inactive for more than 12 consecutive hours
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)
        flow_label = 'Sink(Wärme)'

        # Verify batched variables exist
        assert 'flow|downtime' in model.variables
        assert 'flow|inactive' in model.variables

        # Verify batched constraints exist
        assert 'flow|downtime|ub' in model.constraints
        assert 'flow|downtime|forward' in model.constraints
        assert 'flow|downtime|backward' in model.constraints
        assert 'flow|downtime|initial_ub' in model.constraints

        # Get individual flow variables
        downtime = model.variables['flow|downtime'].sel(flow=flow_label, drop=True)
        inactive = model.variables['flow|inactive'].sel(flow=flow_label, drop=True)

        assert_var_equal(downtime, model.add_variables(lower=0, upper=12, coords=model.get_coords()))

        mega = (
            model.timestep_duration.sum('time') + model.timestep_duration.isel(time=0) * 1
        )  # previously inactive for 1h

        assert_conequal(
            model.constraints['flow|downtime|ub'].sel(flow=flow_label, drop=True),
            downtime <= inactive * mega,
        )

        assert_conequal(
            model.constraints['flow|downtime|forward'].sel(flow=flow_label, drop=True),
            downtime.isel(time=slice(1, None))
            <= downtime.isel(time=slice(None, -1)) + model.timestep_duration.isel(time=slice(None, -1)),
        )

        # eq: duration(t) >= duration(t - 1) + dt(t) + (inactive(t) - 1) * BIG
        assert_conequal(
            model.constraints['flow|downtime|backward'].sel(flow=flow_label, drop=True),
            downtime.isel(time=slice(1, None))
            >= downtime.isel(time=slice(None, -1))
            + model.timestep_duration.isel(time=slice(None, -1))
            + (inactive.isel(time=slice(1, None)) - 1) * mega,
        )

        assert_conequal(
            model.constraints['flow|downtime|initial_ub'].sel(flow=flow_label, drop=True),
            downtime.isel(time=0) <= inactive.isel(time=0) * (model.timestep_duration.isel(time=0) * (1 + 1)),
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

        mega = model.timestep_duration.sum('time') + model.timestep_duration.isel(time=0) * 2

        assert_conequal(
            model.constraints['Sink(Wärme)|downtime|ub'],
            model.variables['Sink(Wärme)|downtime'] <= model.variables['Sink(Wärme)|inactive'] * mega,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|downtime|forward'],
            model.variables['Sink(Wärme)|downtime'].isel(time=slice(1, None))
            <= model.variables['Sink(Wärme)|downtime'].isel(time=slice(None, -1))
            + model.timestep_duration.isel(time=slice(None, -1)),
        )

        # eq: duration(t) >= duration(t - 1) + dt(t) + (On(t) - 1) * BIG
        assert_conequal(
            model.constraints['Sink(Wärme)|downtime|backward'],
            model.variables['Sink(Wärme)|downtime'].isel(time=slice(1, None))
            >= model.variables['Sink(Wärme)|downtime'].isel(time=slice(None, -1))
            + model.timestep_duration.isel(time=slice(None, -1))
            + (model.variables['Sink(Wärme)|inactive'].isel(time=slice(1, None)) - 1) * mega,
        )

        assert_conequal(
            model.constraints['Sink(Wärme)|downtime|initial'],
            model.variables['Sink(Wärme)|downtime'].isel(time=0)
            == model.variables['Sink(Wärme)|inactive'].isel(time=0) * (model.timestep_duration.isel(time=0) * (1 + 2)),
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
            previous_flow_rate=0,  # Required to get initial constraint
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
            == (flow.submodel.variables['Sink(Wärme)|status'] * model.timestep_duration).sum('time'),
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
        # Upper bound is total hours when active_hours_max is not specified
        total_hours = model.timestep_duration.sum('time')
        assert_var_equal(
            model.variables['Sink(Wärme)|active_hours'],
            model.add_variables(lower=0, upper=total_hours, coords=model.get_coords(['period', 'scenario'])),
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
            == (flow.submodel.variables['Sink(Wärme)|status'] * model.timestep_duration).sum('time'),
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
        # Upper bound is total hours when active_hours_max is not specified
        total_hours = model.timestep_duration.sum('time')
        assert_var_equal(
            model.variables['Sink(Wärme)|active_hours'],
            model.add_variables(lower=0, upper=total_hours, coords=model.get_coords(['period', 'scenario'])),
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
            == (flow.submodel.variables['Sink(Wärme)|status'] * model.timestep_duration).sum('time'),
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
