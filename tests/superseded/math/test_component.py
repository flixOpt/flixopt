import numpy as np
import pytest

import flixopt as fx
import flixopt.elements

from ...conftest import (
    assert_almost_equal_numeric,
    create_linopy_model,
)


class TestComponentModel:
    def test_flow_label_check(self):
        """Test that flow model constraints are correctly generated."""
        inputs = [
            fx.Flow(bus='Fernwärme', flow_id='Q_th_Last', relative_minimum=np.ones(10) * 0.1),
            fx.Flow(bus='Fernwärme', flow_id='Q_Gas', relative_minimum=np.ones(10) * 0.1),
        ]
        outputs = [
            fx.Flow(bus='Gas', flow_id='Q_th_Last', relative_minimum=np.ones(10) * 0.01),
            fx.Flow(bus='Gas', flow_id='Q_Gas', relative_minimum=np.ones(10) * 0.01),
        ]
        with pytest.raises(ValueError, match='Flow names must be unique!'):
            _ = flixopt.elements.Component('TestComponent', inputs=inputs, outputs=outputs)

    def test_component(self, basic_flow_system_linopy_coords, coords_config):
        """Test that flow model constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        inputs = [
            fx.Flow(bus='Fernwärme', flow_id='In1', size=100, relative_minimum=np.ones(10) * 0.1),
            fx.Flow(bus='Fernwärme', flow_id='In2', size=100, relative_minimum=np.ones(10) * 0.1),
        ]
        outputs = [
            fx.Flow(bus='Gas', flow_id='Out1', size=100, relative_minimum=np.ones(10) * 0.01),
            fx.Flow(bus='Gas', flow_id='Out2', size=100, relative_minimum=np.ones(10) * 0.01),
        ]
        comp = flixopt.elements.Component('TestComponent', inputs=inputs, outputs=outputs)
        flow_system.add_elements(comp)
        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist with new naming
        flow_rate = model.variables['flow|rate']
        assert 'TestComponent(In1)' in flow_rate.coords['flow'].values
        assert 'TestComponent(In2)' in flow_rate.coords['flow'].values
        assert 'TestComponent(Out1)' in flow_rate.coords['flow'].values
        assert 'TestComponent(Out2)' in flow_rate.coords['flow'].values

        # Check bus balance constraints exist
        assert 'bus|balance' in model.constraints

    def test_on_with_multiple_flows(self, basic_flow_system_linopy_coords, coords_config):
        """Test that component with status and multiple flows is correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        ub_out2 = np.linspace(1, 1.5, 10).round(2)
        inputs = [
            fx.Flow(bus='Fernwärme', flow_id='In1', relative_minimum=np.ones(10) * 0.1, size=100),
        ]
        outputs = [
            fx.Flow(bus='Gas', flow_id='Out1', relative_minimum=np.ones(10) * 0.2, size=200),
            fx.Flow(bus='Gas', flow_id='Out2', relative_minimum=np.ones(10) * 0.3, relative_maximum=ub_out2, size=300),
        ]
        comp = flixopt.elements.Component(
            'TestComponent', inputs=inputs, outputs=outputs, status_parameters=fx.StatusParameters()
        )
        flow_system.add_elements(comp)
        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist
        flow_rate = model.variables['flow|rate']
        assert 'TestComponent(In1)' in flow_rate.coords['flow'].values
        assert 'TestComponent(Out1)' in flow_rate.coords['flow'].values
        assert 'TestComponent(Out2)' in flow_rate.coords['flow'].values

        # Check component status variables exist
        assert 'component|status' in model.variables
        component_status = model.variables['component|status']
        assert 'TestComponent' in component_status.coords['component'].values

        # Check flow status variables exist
        assert 'flow|status' in model.variables
        flow_status = model.variables['flow|status']
        assert 'TestComponent(In1)' in flow_status.coords['flow'].values
        assert 'TestComponent(Out1)' in flow_status.coords['flow'].values
        assert 'TestComponent(Out2)' in flow_status.coords['flow'].values

        # Check active_hours variables exist
        assert 'component|active_hours' in model.variables
        active_hours = model.variables['component|active_hours']
        assert 'TestComponent' in active_hours.coords['component'].values

        # Check constraints for component status
        assert 'component|status|lb' in model.constraints
        assert 'component|status|ub' in model.constraints

        # Check flow rate bounds
        out2_rate = flow_rate.sel(flow='TestComponent(Out2)')
        assert (out2_rate.lower.values >= 0).all()

    def test_on_with_single_flow(self, basic_flow_system_linopy_coords, coords_config):
        """Test that component with status and single flow is correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        inputs = [
            fx.Flow(bus='Fernwärme', flow_id='In1', relative_minimum=np.ones(10) * 0.1, size=100),
        ]
        outputs = []
        comp = flixopt.elements.Component(
            'TestComponent', inputs=inputs, outputs=outputs, status_parameters=fx.StatusParameters()
        )
        flow_system.add_elements(comp)
        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist
        flow_rate = model.variables['flow|rate']
        assert 'TestComponent(In1)' in flow_rate.coords['flow'].values

        # Check status variables exist (component and flow)
        assert 'component|status' in model.variables
        assert 'flow|status' in model.variables

        # Check active_hours variables exist
        assert 'component|active_hours' in model.variables

        # Check component status constraint - for single flow should be equality
        assert 'component|status|eq' in model.constraints

        # Check flow rate bounds
        in1_rate = flow_rate.sel(flow='TestComponent(In1)')
        assert (in1_rate.lower.values >= 0).all()
        assert (in1_rate.upper.values <= 100).all()

    def test_previous_states_with_multiple_flows(self, basic_flow_system_linopy_coords, coords_config):
        """Test that component with previous states is correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        ub_out2 = np.linspace(1, 1.5, 10).round(2)
        inputs = [
            fx.Flow(
                bus='Fernwärme',
                flow_id='In1',
                relative_minimum=np.ones(10) * 0.1,
                size=100,
                previous_flow_rate=np.array([0, 0, 1e-6, 1e-5, 1e-4, 3, 4]),
            ),
        ]
        outputs = [
            fx.Flow(
                bus='Gas', flow_id='Out1', relative_minimum=np.ones(10) * 0.2, size=200, previous_flow_rate=[3, 4, 5]
            ),
            fx.Flow(
                bus='Gas',
                flow_id='Out2',
                relative_minimum=np.ones(10) * 0.3,
                relative_maximum=ub_out2,
                size=300,
                previous_flow_rate=20,
            ),
        ]
        comp = flixopt.elements.Component(
            'TestComponent', inputs=inputs, outputs=outputs, status_parameters=fx.StatusParameters()
        )
        flow_system.add_elements(comp)
        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist
        flow_rate = model.variables['flow|rate']
        assert 'TestComponent(In1)' in flow_rate.coords['flow'].values
        assert 'TestComponent(Out1)' in flow_rate.coords['flow'].values
        assert 'TestComponent(Out2)' in flow_rate.coords['flow'].values

        # Check status variables exist
        assert 'component|status' in model.variables
        assert 'flow|status' in model.variables

        # Check component status constraints
        assert 'component|status|lb' in model.constraints
        assert 'component|status|ub' in model.constraints

    @pytest.mark.parametrize(
        'in1_previous_flow_rate, out1_previous_flow_rate, out2_previous_flow_rate, previous_on_hours',
        [
            (None, None, None, 0),
            (np.array([0, 1e-6, 1e-4, 5]), None, None, 2),
            (np.array([0, 5, 0, 5]), None, None, 1),
            (np.array([0, 5, 0, 0]), 3, 0, 1),
            (np.array([0, 0, 2, 0, 4, 5]), [3, 4, 5], None, 4),
        ],
    )
    def test_previous_states_with_multiple_flows_parameterized(
        self,
        basic_flow_system_linopy_coords,
        coords_config,
        in1_previous_flow_rate,
        out1_previous_flow_rate,
        out2_previous_flow_rate,
        previous_on_hours,
    ):
        """Test that component with different previous states configurations is correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        ub_out2 = np.linspace(1, 1.5, 10).round(2)
        inputs = [
            fx.Flow(
                bus='Fernwärme',
                flow_id='In1',
                relative_minimum=np.ones(10) * 0.1,
                size=100,
                previous_flow_rate=in1_previous_flow_rate,
                status_parameters=fx.StatusParameters(min_uptime=3),
            ),
        ]
        outputs = [
            fx.Flow(
                bus='Gas',
                flow_id='Out1',
                relative_minimum=np.ones(10) * 0.2,
                size=200,
                previous_flow_rate=out1_previous_flow_rate,
            ),
            fx.Flow(
                bus='Gas',
                flow_id='Out2',
                relative_minimum=np.ones(10) * 0.3,
                relative_maximum=ub_out2,
                size=300,
                previous_flow_rate=out2_previous_flow_rate,
            ),
        ]
        comp = flixopt.elements.Component(
            'TestComponent',
            inputs=inputs,
            outputs=outputs,
            status_parameters=fx.StatusParameters(min_uptime=3),
        )
        flow_system.add_elements(comp)
        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist
        flow_rate = model.variables['flow|rate']
        assert 'TestComponent(In1)' in flow_rate.coords['flow'].values
        assert 'TestComponent(Out1)' in flow_rate.coords['flow'].values
        assert 'TestComponent(Out2)' in flow_rate.coords['flow'].values

        # Check status variables exist
        assert 'component|status' in model.variables
        assert 'flow|status' in model.variables

        # Check uptime variables exist when min_uptime is set
        assert 'component|uptime' in model.variables

        # Initial constraint only exists when at least one flow has previous_flow_rate set
        has_previous = any(
            x is not None for x in [in1_previous_flow_rate, out1_previous_flow_rate, out2_previous_flow_rate]
        )
        if has_previous:
            assert 'component|uptime|initial' in model.constraints


class TestTransmissionModel:
    def test_transmission_basic(self, basic_flow_system, highs_solver):
        """Test basic transmission functionality"""
        flow_system = basic_flow_system
        flow_system.add_elements(fx.Bus('Wärme lokal'))

        boiler = fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.5,
            thermal_flow=fx.Flow(bus='Wärme lokal', flow_id='Q_th'),
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
        )

        transmission = fx.Transmission(
            'Rohr',
            relative_losses=0.2,
            absolute_losses=20,
            in1=fx.Flow(
                bus='Wärme lokal',
                flow_id='Rohr1',
                size=fx.InvestParameters(effects_of_investment_per_size=5, maximum_size=1e6),
            ),
            out1=fx.Flow(bus='Fernwärme', flow_id='Rohr2', size=1000),
        )

        flow_system.add_elements(transmission, boiler)

        flow_system.optimize(highs_solver)

        # Assertions using new API (flow_system.solution)
        assert_almost_equal_numeric(
            flow_system.solution['Rohr(Rohr1)|status'].values,
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            'Status does not work properly',
        )

        assert_almost_equal_numeric(
            flow_system.solution['Rohr(Rohr1)|flow_rate'].values * 0.8 - 20,
            flow_system.solution['Rohr(Rohr2)|flow_rate'].values,
            'Losses are not computed correctly',
        )

    def test_transmission_balanced(self, basic_flow_system, highs_solver):
        """Test advanced transmission functionality"""
        flow_system = basic_flow_system
        flow_system.add_elements(fx.Bus('Wärme lokal'))

        boiler = fx.linear_converters.Boiler(
            'Boiler_Standard',
            thermal_efficiency=0.9,
            thermal_flow=fx.Flow(
                bus='Fernwärme', flow_id='Q_th', size=1000, relative_maximum=np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
            ),
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
        )

        boiler2 = fx.linear_converters.Boiler(
            'Boiler_backup',
            thermal_efficiency=0.4,
            thermal_flow=fx.Flow(bus='Wärme lokal', flow_id='Q_th'),
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
        )

        last2 = fx.Sink(
            'Wärmelast2',
            inputs=[
                fx.Flow(
                    bus='Wärme lokal',
                    flow_id='Q_th_Last',
                    size=1,
                    fixed_relative_profile=flow_system.components['Wärmelast'].inputs[0].fixed_relative_profile
                    * np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
                )
            ],
        )

        transmission = fx.Transmission(
            'Rohr',
            relative_losses=0.2,
            absolute_losses=20,
            in1=fx.Flow(
                bus='Wärme lokal',
                flow_id='Rohr1a',
                size=fx.InvestParameters(effects_of_investment_per_size=5, maximum_size=1000),
            ),
            out1=fx.Flow(bus='Fernwärme', flow_id='Rohr1b', size=1000),
            in2=fx.Flow(bus='Fernwärme', flow_id='Rohr2a', size=fx.InvestParameters(maximum_size=1000)),
            out2=fx.Flow(bus='Wärme lokal', flow_id='Rohr2b', size=1000),
            balanced=True,
        )

        flow_system.add_elements(transmission, boiler, boiler2, last2)

        flow_system.optimize(highs_solver)

        # Assertions using new API (flow_system.solution)
        assert_almost_equal_numeric(
            flow_system.solution['Rohr(Rohr1a)|status'].values,
            np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
            'Status does not work properly',
        )

        # Verify output flow matches input flow minus losses (relative 20% + absolute 20)
        in1_flow = flow_system.solution['Rohr(Rohr1a)|flow_rate'].values
        expected_out1_flow = in1_flow * 0.8 - np.array([20 if val > 0.1 else 0 for val in in1_flow])
        assert_almost_equal_numeric(
            flow_system.solution['Rohr(Rohr1b)|flow_rate'].values,
            expected_out1_flow,
            'Losses are not computed correctly',
        )

        assert_almost_equal_numeric(
            flow_system.solution['Rohr(Rohr1a)|size'].item(),
            flow_system.solution['Rohr(Rohr2a)|size'].item(),
            'The Investments are not equated correctly',
        )

    def test_transmission_unbalanced(self, basic_flow_system, highs_solver):
        """Test advanced transmission functionality"""
        flow_system = basic_flow_system
        flow_system.add_elements(fx.Bus('Wärme lokal'))

        boiler = fx.linear_converters.Boiler(
            'Boiler_Standard',
            thermal_efficiency=0.9,
            thermal_flow=fx.Flow(
                bus='Fernwärme', flow_id='Q_th', size=1000, relative_maximum=np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
            ),
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
        )

        boiler2 = fx.linear_converters.Boiler(
            'Boiler_backup',
            thermal_efficiency=0.4,
            thermal_flow=fx.Flow(bus='Wärme lokal', flow_id='Q_th'),
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
        )

        last2 = fx.Sink(
            'Wärmelast2',
            inputs=[
                fx.Flow(
                    bus='Wärme lokal',
                    flow_id='Q_th_Last',
                    size=1,
                    fixed_relative_profile=flow_system.components['Wärmelast'].inputs[0].fixed_relative_profile
                    * np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
                )
            ],
        )

        transmission = fx.Transmission(
            'Rohr',
            relative_losses=0.2,
            absolute_losses=20,
            in1=fx.Flow(
                bus='Wärme lokal',
                flow_id='Rohr1a',
                size=fx.InvestParameters(effects_of_investment_per_size=50, maximum_size=1000),
            ),
            out1=fx.Flow(bus='Fernwärme', flow_id='Rohr1b', size=1000),
            in2=fx.Flow(
                bus='Fernwärme',
                flow_id='Rohr2a',
                size=fx.InvestParameters(
                    effects_of_investment_per_size=100, minimum_size=10, maximum_size=1000, mandatory=True
                ),
            ),
            out2=fx.Flow(bus='Wärme lokal', flow_id='Rohr2b', size=1000),
            balanced=False,
        )

        flow_system.add_elements(transmission, boiler, boiler2, last2)

        flow_system.optimize(highs_solver)

        # Assertions using new API (flow_system.solution)
        assert_almost_equal_numeric(
            flow_system.solution['Rohr(Rohr1a)|status'].values,
            np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
            'Status does not work properly',
        )

        # Verify output flow matches input flow minus losses (relative 20% + absolute 20)
        in1_flow = flow_system.solution['Rohr(Rohr1a)|flow_rate'].values
        expected_out1_flow = in1_flow * 0.8 - np.array([20 if val > 0.1 else 0 for val in in1_flow])
        assert_almost_equal_numeric(
            flow_system.solution['Rohr(Rohr1b)|flow_rate'].values,
            expected_out1_flow,
            'Losses are not computed correctly',
        )

        assert flow_system.solution['Rohr(Rohr1a)|size'].item() > 11

        assert_almost_equal_numeric(
            flow_system.solution['Rohr(Rohr2a)|size'].item(),
            10,
            'Sizing does not work properly',
        )
