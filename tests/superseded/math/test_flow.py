import numpy as np
import pytest

import flixopt as fx

from ...conftest import create_linopy_model


class TestFlowModel:
    """Test the FlowModel class."""

    def test_flow_minimal(self, basic_flow_system_linopy_coords, coords_config):
        """Test that flow model constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(bus='Fernwärme', flow_id='Wärme', size=100)

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))

        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist
        assert 'flow|rate' in model.variables
        flow_rate = model.variables['flow|rate']
        assert 'Sink(Wärme)' in flow_rate.coords['flow'].values

        # Check bounds
        rate = flow_rate.sel(flow='Sink(Wärme)')
        assert (rate.lower.values >= 0).all()
        assert (rate.upper.values <= 100).all()

    def test_flow(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
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

        # Check that flow rate variables exist
        assert 'flow|rate' in model.variables
        flow_rate = model.variables['flow|rate']
        assert 'Sink(Wärme)' in flow_rate.coords['flow'].values

        # Check flow hours constraints exist
        assert 'flow|hours_min' in model.constraints
        assert 'flow|hours_max' in model.constraints

        # Check load factor constraints exist
        assert 'flow|load_factor_min' in model.constraints
        assert 'flow|load_factor_max' in model.constraints

    def test_effects_per_flow_hour(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        costs_per_flow_hour = np.linspace(1, 2, timesteps.size)
        co2_per_flow_hour = np.linspace(4, 5, timesteps.size)

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            effects_per_flow_hour={'costs': costs_per_flow_hour, 'CO2': co2_per_flow_hour},
        )
        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]), fx.Effect('CO2', 't', ''))
        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist
        assert 'flow|rate' in model.variables
        flow_rate = model.variables['flow|rate']
        assert 'Sink(Wärme)' in flow_rate.coords['flow'].values

        # Check that effect share variable and constraints exist
        assert 'share|temporal' in model.variables
        assert 'share|temporal(costs)' in model.constraints
        assert 'share|temporal(CO2)' in model.constraints


class TestFlowInvestModel:
    """Test the FlowModel class with investment."""

    def test_flow_invest(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            size=fx.InvestParameters(minimum_size=20, maximum_size=100, mandatory=True),
            relative_minimum=np.linspace(0.1, 0.5, timesteps.size),
            relative_maximum=np.linspace(0.5, 1, timesteps.size),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist
        assert 'flow|rate' in model.variables
        flow_rate = model.variables['flow|rate']
        assert 'Sink(Wärme)' in flow_rate.coords['flow'].values

        # Check that investment variables exist
        assert 'flow|size' in model.variables
        size_var = model.variables['flow|size']
        assert 'Sink(Wärme)' in size_var.coords['flow'].values

        # Check size bounds (mandatory investment)
        size = size_var.sel(flow='Sink(Wärme)')
        assert (size.lower.values >= 20).all()
        assert (size.upper.values <= 100).all()

        # Check flow rate constraints exist
        assert 'flow|invest_ub' in model.constraints
        assert 'flow|invest_lb' in model.constraints

    def test_flow_invest_optional(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            size=fx.InvestParameters(minimum_size=20, maximum_size=100, mandatory=False),
            relative_minimum=np.linspace(0.1, 0.5, timesteps.size),
            relative_maximum=np.linspace(0.5, 1, timesteps.size),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that investment variables exist
        assert 'flow|size' in model.variables
        assert 'flow|invested' in model.variables
        size_var = model.variables['flow|size']
        invested_var = model.variables['flow|invested']
        assert 'Sink(Wärme)' in size_var.coords['flow'].values
        assert 'Sink(Wärme)' in invested_var.coords['flow'].values

        # Check size bounds (optional investment)
        size = size_var.sel(flow='Sink(Wärme)')
        assert (size.lower.values >= 0).all()  # Optional
        assert (size.upper.values <= 100).all()

        # Check investment constraints exist
        assert 'flow|size|lb' in model.constraints
        assert 'flow|size|ub' in model.constraints

    def test_flow_invest_optional_wo_min_size(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            size=fx.InvestParameters(maximum_size=100, mandatory=False),
            relative_minimum=np.linspace(0.1, 0.5, timesteps.size),
            relative_maximum=np.linspace(0.5, 1, timesteps.size),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that investment variables exist
        assert 'flow|size' in model.variables
        assert 'flow|invested' in model.variables

        # Check investment constraints exist
        assert 'flow|size|ub' in model.constraints
        assert 'flow|size|lb' in model.constraints

    def test_flow_invest_wo_min_size_non_optional(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            size=fx.InvestParameters(maximum_size=100, mandatory=True),
            relative_minimum=np.linspace(0.1, 0.5, timesteps.size),
            relative_maximum=np.linspace(0.5, 1, timesteps.size),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that investment variables exist
        assert 'flow|size' in model.variables
        size_var = model.variables['flow|size']
        assert 'Sink(Wärme)' in size_var.coords['flow'].values

        # Check size bounds (mandatory, no min_size means 1e-5 lower bound)
        size = size_var.sel(flow='Sink(Wärme)')
        assert (size.lower.values >= 1e-5 - 1e-10).all()
        assert (size.upper.values <= 100).all()

    def test_flow_invest_fixed_size(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with fixed size investment."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            size=fx.InvestParameters(fixed_size=75, mandatory=True),
            relative_minimum=0.2,
            relative_maximum=0.9,
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that investment variables exist
        assert 'flow|size' in model.variables
        size_var = model.variables['flow|size']
        assert 'Sink(Wärme)' in size_var.coords['flow'].values

        # Check size is fixed to 75
        size = size_var.sel(flow='Sink(Wärme)')
        assert (size.lower.values >= 75 - 0.1).all()
        assert (size.upper.values <= 75 + 0.1).all()

        # Check flow rate bounds
        flow_rate = model.variables['flow|rate'].sel(flow='Sink(Wärme)')
        assert (flow_rate.lower.values >= 0.2 * 75 - 0.1).all()
        assert (flow_rate.upper.values <= 0.9 * 75 + 0.1).all()

    def test_flow_invest_with_effects(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with investment effects."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create effects
        co2 = fx.Effect('CO2', unit='ton', description='CO2 emissions')

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
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

        # Check that share variables and constraints exist for periodic effects
        assert 'share|periodic' in model.variables
        assert 'share|periodic(costs)' in model.constraints
        assert 'share|periodic(CO2)' in model.constraints

    def test_flow_invest_divest_effects(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with divestment effects."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            size=fx.InvestParameters(
                minimum_size=20,
                maximum_size=100,
                mandatory=False,
                effects_of_retirement={'costs': 500},  # Cost incurred when NOT investing
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check share periodic constraints exist for divestment effects
        assert 'share|periodic' in model.variables
        assert 'share|periodic(costs)' in model.constraints


class TestFlowOnModel:
    """Test the FlowModel class with status."""

    def test_flow_on(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            size=100,
            relative_minimum=0.2,
            relative_maximum=0.8,
            status_parameters=fx.StatusParameters(),
        )
        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist
        assert 'flow|rate' in model.variables
        flow_rate = model.variables['flow|rate']
        assert 'Sink(Wärme)' in flow_rate.coords['flow'].values

        # Check that status variables exist
        assert 'flow|status' in model.variables
        status_var = model.variables['flow|status']
        assert 'Sink(Wärme)' in status_var.coords['flow'].values

        # Check that active_hours variables exist
        assert 'flow|active_hours' in model.variables
        active_hours = model.variables['flow|active_hours']
        assert 'Sink(Wärme)' in active_hours.coords['flow'].values

        # Check flow rate constraints exist
        assert 'flow|status_lb' in model.constraints
        assert 'flow|status_ub' in model.constraints

        # Check active_hours constraints exist
        assert 'flow|active_hours' in model.constraints

    def test_effects_per_active_hour(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        costs_per_running_hour = np.linspace(1, 2, timesteps.size)
        co2_per_running_hour = np.linspace(4, 5, timesteps.size)

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            size=100,
            status_parameters=fx.StatusParameters(
                effects_per_active_hour={'costs': costs_per_running_hour, 'CO2': co2_per_running_hour}
            ),
        )
        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]), fx.Effect('CO2', 't', ''))
        model = create_linopy_model(flow_system)

        # Check that status variables exist
        assert 'flow|status' in model.variables
        assert 'flow|active_hours' in model.variables

        # Check share temporal variables and constraints exist
        assert 'share|temporal' in model.variables
        assert 'share|temporal(costs)' in model.constraints
        assert 'share|temporal(CO2)' in model.constraints

    def test_consecutive_on_hours(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with minimum and maximum consecutive on hours."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            size=100,
            previous_flow_rate=0,  # Required to get initial constraint
            status_parameters=fx.StatusParameters(
                min_uptime=2,  # Must run for at least 2 hours when turned on
                max_uptime=8,  # Can't run more than 8 consecutive hours
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that uptime variables exist
        assert 'flow|uptime' in model.variables
        assert 'flow|status' in model.variables

        # Check uptime constraints exist
        assert 'flow|uptime|ub' in model.constraints
        assert 'flow|uptime|forward' in model.constraints
        assert 'flow|uptime|backward' in model.constraints
        assert 'flow|uptime|initial' in model.constraints
        assert 'flow|uptime|min' in model.constraints

        # Check uptime variable bounds
        uptime = model.variables['flow|uptime'].sel(flow='Sink(Wärme)')
        assert (uptime.lower.values >= 0).all()
        assert (uptime.upper.values <= 8).all()

    def test_consecutive_on_hours_previous(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with minimum and maximum uptime."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            size=100,
            status_parameters=fx.StatusParameters(
                min_uptime=2,  # Must run for at least 2 hours when active
                max_uptime=8,  # Can't run more than 8 consecutive hours
            ),
            previous_flow_rate=np.array([10, 20, 30, 0, 20, 20, 30]),  # Previously active for 3 steps
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that uptime variables exist
        assert 'flow|uptime' in model.variables
        assert 'flow|status' in model.variables

        # Check uptime constraints exist (including initial)
        assert 'flow|uptime|forward' in model.constraints
        assert 'flow|uptime|backward' in model.constraints
        assert 'flow|uptime|initial' in model.constraints

    def test_consecutive_off_hours(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with minimum and maximum consecutive inactive hours."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            size=100,
            previous_flow_rate=0,  # Required to get initial constraint (was OFF for 1h, so previous_downtime=1)
            status_parameters=fx.StatusParameters(
                min_downtime=4,  # Must stay inactive for at least 4 hours when shut down
                max_downtime=12,  # Can't be inactive for more than 12 consecutive hours
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that downtime variables exist
        assert 'flow|downtime' in model.variables
        assert 'flow|inactive' in model.variables

        # Check downtime constraints exist
        assert 'flow|downtime|ub' in model.constraints
        assert 'flow|downtime|forward' in model.constraints
        assert 'flow|downtime|backward' in model.constraints
        assert 'flow|downtime|initial' in model.constraints
        assert 'flow|downtime|min' in model.constraints

        # Check downtime variable bounds
        downtime = model.variables['flow|downtime'].sel(flow='Sink(Wärme)')
        assert (downtime.lower.values >= 0).all()
        assert (downtime.upper.values <= 12).all()

    def test_consecutive_off_hours_previous(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with minimum and maximum consecutive inactive hours."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            size=100,
            status_parameters=fx.StatusParameters(
                min_downtime=4,  # Must stay inactive for at least 4 hours when shut down
                max_downtime=12,  # Can't be inactive for more than 12 consecutive hours
            ),
            previous_flow_rate=np.array([10, 20, 30, 0, 20, 0, 0]),  # Previously inactive for 2 steps
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that downtime variables exist
        assert 'flow|downtime' in model.variables
        assert 'flow|inactive' in model.variables

        # Check downtime constraints exist (including initial)
        assert 'flow|downtime|forward' in model.constraints
        assert 'flow|downtime|backward' in model.constraints
        assert 'flow|downtime|initial' in model.constraints

    def test_switch_on_constraints(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with constraints on the number of startups."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            size=100,
            previous_flow_rate=0,  # Required to get initial constraint
            status_parameters=fx.StatusParameters(
                startup_limit=5,  # Maximum 5 startups
                effects_per_startup={'costs': 100},  # 100 EUR startup cost
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that switch variables exist
        assert 'flow|startup' in model.variables
        assert 'flow|shutdown' in model.variables
        assert 'flow|startup_count' in model.variables

        # Check that switch constraints exist
        assert 'flow|switch_transition' in model.constraints
        assert 'flow|switch_initial' in model.constraints
        assert 'flow|switch_mutex' in model.constraints
        assert 'flow|startup_count' in model.constraints

        # Check startup_count variable bounds
        startup_count = model.variables['flow|startup_count'].sel(flow='Sink(Wärme)')
        assert (startup_count.lower.values >= 0).all()
        assert (startup_count.upper.values <= 5).all()

        # Check startup cost effect share temporal constraint exists
        assert 'share|temporal' in model.variables
        assert 'share|temporal(costs)' in model.constraints

    def test_on_hours_limits(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with limits on total active hours."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            size=100,
            status_parameters=fx.StatusParameters(
                active_hours_min=20,  # Minimum 20 hours of operation
                active_hours_max=100,  # Maximum 100 hours of operation
            ),
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that status and active_hours variables exist
        assert 'flow|status' in model.variables
        assert 'flow|active_hours' in model.variables

        # Check active_hours constraint exists
        assert 'flow|active_hours' in model.constraints

        # Check active_hours variable bounds
        active_hours = model.variables['flow|active_hours'].sel(flow='Sink(Wärme)')
        assert (active_hours.lower.values >= 20 - 0.1).all()
        assert (active_hours.upper.values <= 100 + 0.1).all()


class TestFlowOnInvestModel:
    """Test the FlowModel class with status and investment."""

    def test_flow_on_invest_optional(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            size=fx.InvestParameters(minimum_size=20, maximum_size=200, mandatory=False),
            relative_minimum=0.2,
            relative_maximum=0.8,
            status_parameters=fx.StatusParameters(),
        )
        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist
        assert 'flow|rate' in model.variables
        flow_rate = model.variables['flow|rate']
        assert 'Sink(Wärme)' in flow_rate.coords['flow'].values

        # Check that investment variables exist
        assert 'flow|size' in model.variables
        assert 'flow|invested' in model.variables

        # Check that status variables exist
        assert 'flow|status' in model.variables
        assert 'flow|active_hours' in model.variables

        # Check investment constraints
        assert 'flow|size|lb' in model.constraints
        assert 'flow|size|ub' in model.constraints

    def test_flow_on_invest_non_optional(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            size=fx.InvestParameters(minimum_size=20, maximum_size=200, mandatory=True),
            relative_minimum=0.2,
            relative_maximum=0.8,
            status_parameters=fx.StatusParameters(),
        )
        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist
        assert 'flow|rate' in model.variables
        flow_rate = model.variables['flow|rate']
        assert 'Sink(Wärme)' in flow_rate.coords['flow'].values

        # Check that investment variables exist
        assert 'flow|size' in model.variables
        # No invested variable for mandatory investment
        size_var = model.variables['flow|size'].sel(flow='Sink(Wärme)')
        assert (size_var.lower.values >= 20).all()
        assert (size_var.upper.values <= 200).all()

        # Check that status variables exist
        assert 'flow|status' in model.variables
        assert 'flow|active_hours' in model.variables


class TestFlowWithFixedProfile:
    """Test Flow with fixed relative profile."""

    def test_fixed_relative_profile(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with a fixed relative profile."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        # Create a time-varying profile (e.g., for a load or renewable generation)
        profile = np.sin(np.linspace(0, 2 * np.pi, len(timesteps))) * 0.5 + 0.5  # Values between 0 and 1

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            size=100,
            fixed_relative_profile=profile,
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist
        assert 'flow|rate' in model.variables
        flow_rate = model.variables['flow|rate'].sel(flow='Sink(Wärme)')

        # Check that flow rate is fixed (lower == upper)
        np.testing.assert_allclose(flow_rate.lower.values.flatten(), flow_rate.upper.values.flatten(), rtol=1e-5)

    def test_fixed_profile_with_investment(self, basic_flow_system_linopy_coords, coords_config):
        """Test flow with fixed profile and investment."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        # Create a fixed profile
        profile = np.sin(np.linspace(0, 2 * np.pi, len(timesteps))) * 0.5 + 0.5

        flow = fx.Flow(
            bus='Fernwärme',
            flow_id='Wärme',
            size=fx.InvestParameters(minimum_size=50, maximum_size=200, mandatory=False),
            fixed_relative_profile=profile,
        )

        flow_system.add_elements(fx.Sink('Sink', inputs=[flow]))
        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist
        assert 'flow|rate' in model.variables
        assert 'Sink(Wärme)' in model.variables['flow|rate'].coords['flow'].values

        # Check that investment variables exist
        assert 'flow|size' in model.variables
        assert 'flow|invested' in model.variables

        # Check investment constraints exist (fixed profile is enforced via invest_lb/invest_ub constraints)
        assert 'flow|invest_lb' in model.constraints
        assert 'flow|invest_ub' in model.constraints


if __name__ == '__main__':
    pytest.main()
