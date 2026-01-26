import numpy as np
import pytest

import flixopt as fx
import flixopt.elements

from .conftest import (
    assert_almost_equal_numeric,
    assert_conequal,
    assert_dims_compatible,
    assert_var_equal,
    create_linopy_model,
)


class TestComponentModel:
    def test_flow_label_check(self):
        """Test that flow model constraints are correctly generated."""
        inputs = [
            fx.Flow('Q_th_Last', 'Fernwärme', relative_minimum=np.ones(10) * 0.1),
            fx.Flow('Q_Gas', 'Fernwärme', relative_minimum=np.ones(10) * 0.1),
        ]
        outputs = [
            fx.Flow('Q_th_Last', 'Gas', relative_minimum=np.ones(10) * 0.01),
            fx.Flow('Q_Gas', 'Gas', relative_minimum=np.ones(10) * 0.01),
        ]
        with pytest.raises(ValueError, match='Flow names must be unique!'):
            _ = flixopt.elements.Component('TestComponent', inputs=inputs, outputs=outputs)

    def test_component(self, basic_flow_system_linopy_coords, coords_config):
        """Test that flow model constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        inputs = [
            fx.Flow('In1', 'Fernwärme', size=100, relative_minimum=np.ones(10) * 0.1),
            fx.Flow('In2', 'Fernwärme', size=100, relative_minimum=np.ones(10) * 0.1),
        ]
        outputs = [
            fx.Flow('Out1', 'Gas', size=100, relative_minimum=np.ones(10) * 0.01),
            fx.Flow('Out2', 'Gas', size=100, relative_minimum=np.ones(10) * 0.01),
        ]
        comp = flixopt.elements.Component('TestComponent', inputs=inputs, outputs=outputs)
        flow_system.add_elements(comp)
        model = create_linopy_model(flow_system)

        # Check batched variables exist
        assert 'flow|rate' in model.variables, 'Batched flow rate variable should exist'
        # Note: hours variable removed - computed inline in constraints now

    def test_on_with_multiple_flows(self, basic_flow_system_linopy_coords, coords_config):
        """Test that flow model constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        ub_out2 = np.linspace(1, 1.5, 10).round(2)
        inputs = [
            fx.Flow('In1', 'Fernwärme', relative_minimum=np.ones(10) * 0.1, size=100),
        ]
        outputs = [
            fx.Flow('Out1', 'Gas', relative_minimum=np.ones(10) * 0.2, size=200),
            fx.Flow('Out2', 'Gas', relative_minimum=np.ones(10) * 0.3, relative_maximum=ub_out2, size=300),
        ]
        comp = flixopt.elements.Component(
            'TestComponent', inputs=inputs, outputs=outputs, status_parameters=fx.StatusParameters()
        )
        flow_system.add_elements(comp)
        model = create_linopy_model(flow_system)

        # Check batched variables exist
        assert 'flow|rate' in model.variables, 'Batched flow rate variable should exist'
        assert 'flow|status' in model.variables, 'Batched status variable should exist'
        assert 'flow|active_hours' in model.variables, 'Batched active_hours variable should exist'
        assert 'component|status' in model.variables, 'Batched component status variable should exist'
        assert 'component|active_hours' in model.variables, 'Batched component active_hours variable should exist'

        upper_bound_flow_rate = outputs[1].relative_maximum

        assert_dims_compatible(upper_bound_flow_rate, tuple(model.get_coords()))

        # Access variables using type-level batched model + sel
        flow_rate_out2 = model.variables['flow|rate'].sel(flow='TestComponent(Out2)')
        flow_status_out2 = model.variables['flow|status'].sel(flow='TestComponent(Out2)')
        comp_status = model.variables['component|status'].sel(component='TestComponent')

        # Check variable bounds and types
        assert_var_equal(
            flow_rate_out2,
            model.add_variables(lower=0, upper=300 * upper_bound_flow_rate, coords=model.get_coords()),
        )
        assert_var_equal(comp_status, model.add_variables(binary=True, coords=model.get_coords()))
        assert_var_equal(flow_status_out2, model.add_variables(binary=True, coords=model.get_coords()))

        # Check flow rate constraints exist and have correct bounds
        assert_conequal(
            model.constraints['flow|status_lb'].sel(flow='TestComponent(Out2)'),
            flow_rate_out2 >= flow_status_out2 * 0.3 * 300,
        )
        assert_conequal(
            model.constraints['flow|status_ub'].sel(flow='TestComponent(Out2)'),
            flow_rate_out2 <= flow_status_out2 * 300 * upper_bound_flow_rate,
        )

        # Check component status constraints exist (multi-flow uses lb/ub bounds)
        assert 'component|status|lb' in model.constraints, 'Component status lower bound should exist'
        assert 'component|status|ub' in model.constraints, 'Component status upper bound should exist'
        assert 'TestComponent' in model.constraints['component|status|lb'].coords['component'].values

    def test_on_with_single_flow(self, basic_flow_system_linopy_coords, coords_config):
        """Test that flow model constraints are correctly generated for single-flow components."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        inputs = [
            fx.Flow('In1', 'Fernwärme', relative_minimum=np.ones(10) * 0.1, size=100),
        ]
        outputs = []
        comp = flixopt.elements.Component(
            'TestComponent', inputs=inputs, outputs=outputs, status_parameters=fx.StatusParameters()
        )
        flow_system.add_elements(comp)
        model = create_linopy_model(flow_system)

        # Check batched variables exist
        assert 'flow|rate' in model.variables, 'Batched flow rate variable should exist'
        assert 'flow|status' in model.variables, 'Batched status variable should exist'
        assert 'component|status' in model.variables, 'Batched component status variable should exist'

        # Access individual flow variables using batched model + sel
        flow_label = 'TestComponent(In1)'
        flow_rate = model.variables['flow|rate'].sel(flow=flow_label)
        flow_status = model.variables['flow|status'].sel(flow=flow_label)
        comp_status = model.variables['component|status'].sel(component='TestComponent')

        # Check variable bounds and types
        assert_var_equal(flow_rate, model.add_variables(lower=0, upper=100, coords=model.get_coords()))
        assert_var_equal(comp_status, model.add_variables(binary=True, coords=model.get_coords()))
        assert_var_equal(flow_status, model.add_variables(binary=True, coords=model.get_coords()))

        # Check flow rate constraints exist and have correct bounds
        assert_conequal(
            model.constraints['flow|status_lb'].sel(flow=flow_label),
            flow_rate >= flow_status * 0.1 * 100,
        )
        assert_conequal(
            model.constraints['flow|status_ub'].sel(flow=flow_label),
            flow_rate <= flow_status * 100,
        )

        # Check component status constraint exists (single-flow uses equality constraint)
        assert 'component|status|eq' in model.constraints, 'Component status equality should exist'
        assert 'TestComponent' in model.constraints['component|status|eq'].coords['component'].values

    def test_previous_states_with_multiple_flows(self, basic_flow_system_linopy_coords, coords_config):
        """Test that flow model constraints are correctly generated with previous flow rates."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        ub_out2 = np.linspace(1, 1.5, 10).round(2)
        inputs = [
            fx.Flow(
                'In1',
                'Fernwärme',
                relative_minimum=np.ones(10) * 0.1,
                size=100,
                previous_flow_rate=np.array([0, 0, 1e-6, 1e-5, 1e-4, 3, 4]),
            ),
        ]
        outputs = [
            fx.Flow('Out1', 'Gas', relative_minimum=np.ones(10) * 0.2, size=200, previous_flow_rate=[3, 4, 5]),
            fx.Flow(
                'Out2',
                'Gas',
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

        # Check batched variables exist
        assert 'flow|rate' in model.variables, 'Batched flow rate variable should exist'
        assert 'flow|status' in model.variables, 'Batched status variable should exist'
        assert 'component|status' in model.variables, 'Batched component status variable should exist'

        upper_bound_flow_rate = outputs[1].relative_maximum

        assert_dims_compatible(upper_bound_flow_rate, tuple(model.get_coords()))

        # Access variables using type-level batched model + sel
        flow_rate_out2 = model.variables['flow|rate'].sel(flow='TestComponent(Out2)')
        flow_status_out2 = model.variables['flow|status'].sel(flow='TestComponent(Out2)')
        comp_status = model.variables['component|status'].sel(component='TestComponent')

        # Check variable bounds and types
        assert_var_equal(
            flow_rate_out2,
            model.add_variables(lower=0, upper=300 * upper_bound_flow_rate, coords=model.get_coords()),
        )
        assert_var_equal(comp_status, model.add_variables(binary=True, coords=model.get_coords()))
        assert_var_equal(flow_status_out2, model.add_variables(binary=True, coords=model.get_coords()))

        # Check flow rate constraints exist and have correct bounds
        assert_conequal(
            model.constraints['flow|status_lb'].sel(flow='TestComponent(Out2)'),
            flow_rate_out2 >= flow_status_out2 * 0.3 * 300,
        )
        assert_conequal(
            model.constraints['flow|status_ub'].sel(flow='TestComponent(Out2)'),
            flow_rate_out2 <= flow_status_out2 * 300 * upper_bound_flow_rate,
        )

        # Check component status constraints exist (multi-flow uses lb/ub bounds)
        assert 'component|status|lb' in model.constraints, 'Component status lower bound should exist'
        assert 'component|status|ub' in model.constraints, 'Component status upper bound should exist'
        assert 'TestComponent' in model.constraints['component|status|lb'].coords['component'].values

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
        """Test that flow model constraints are correctly generated with different previous flow rates and constraint factors."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        ub_out2 = np.linspace(1, 1.5, 10).round(2)
        inputs = [
            fx.Flow(
                'In1',
                'Fernwärme',
                relative_minimum=np.ones(10) * 0.1,
                size=100,
                previous_flow_rate=in1_previous_flow_rate,
                status_parameters=fx.StatusParameters(min_uptime=3),
            ),
        ]
        outputs = [
            fx.Flow(
                'Out1', 'Gas', relative_minimum=np.ones(10) * 0.2, size=200, previous_flow_rate=out1_previous_flow_rate
            ),
            fx.Flow(
                'Out2',
                'Gas',
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

        # Initial constraint only exists when at least one flow has previous_flow_rate set
        has_previous = any(
            x is not None for x in [in1_previous_flow_rate, out1_previous_flow_rate, out2_previous_flow_rate]
        )
        if has_previous:
            # Check that uptime initial constraints exist in the model (batched naming)
            # Note: component uptime constraints use |initial_lb and |initial_ub naming
            has_uptime_constraint = (
                'component|uptime|initial_lb' in model.constraints or 'component|uptime|initial_ub' in model.constraints
            )
            assert has_uptime_constraint, 'Uptime initial constraint should exist'
        else:
            # When no previous flow rate, no uptime initialization needed
            pass


class TestTransmissionModel:
    def test_transmission_basic(self, basic_flow_system, highs_solver):
        """Test basic transmission functionality"""
        flow_system = basic_flow_system
        flow_system.add_elements(fx.Bus('Wärme lokal'))

        boiler = fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.5,
            thermal_flow=fx.Flow('Q_th', bus='Wärme lokal'),
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
        )

        transmission = fx.Transmission(
            'Rohr',
            relative_losses=0.2,
            absolute_losses=20,
            in1=fx.Flow(
                'Rohr1', 'Wärme lokal', size=fx.InvestParameters(effects_of_investment_per_size=5, maximum_size=1e6)
            ),
            out1=fx.Flow('Rohr2', 'Fernwärme', size=1000),
        )

        flow_system.add_elements(transmission, boiler)

        flow_system.optimize(highs_solver)

        # Assertions using batched variable naming (flow|status, flow|rate)
        assert_almost_equal_numeric(
            flow_system.solution['flow|status'].sel(flow='Rohr(Rohr1)').values,
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            'Status does not work properly',
        )

        assert_almost_equal_numeric(
            flow_system.solution['flow|rate'].sel(flow='Rohr(Rohr1)').values * 0.8 - 20,
            flow_system.solution['flow|rate'].sel(flow='Rohr(Rohr2)').values,
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
                'Q_th', bus='Fernwärme', size=1000, relative_maximum=np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
            ),
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
        )

        boiler2 = fx.linear_converters.Boiler(
            'Boiler_backup',
            thermal_efficiency=0.4,
            thermal_flow=fx.Flow('Q_th', bus='Wärme lokal'),
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
        )

        last2 = fx.Sink(
            'Wärmelast2',
            inputs=[
                fx.Flow(
                    'Q_th_Last',
                    bus='Wärme lokal',
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
                'Rohr1a',
                bus='Wärme lokal',
                size=fx.InvestParameters(effects_of_investment_per_size=5, maximum_size=1000),
            ),
            out1=fx.Flow('Rohr1b', 'Fernwärme', size=1000),
            in2=fx.Flow('Rohr2a', 'Fernwärme', size=fx.InvestParameters(maximum_size=1000)),
            out2=fx.Flow('Rohr2b', bus='Wärme lokal', size=1000),
            balanced=True,
        )

        flow_system.add_elements(transmission, boiler, boiler2, last2)

        flow_system.optimize(highs_solver)

        # Assertions using batched variable naming (flow|status, flow|rate, flow|size)
        assert_almost_equal_numeric(
            flow_system.solution['flow|status'].sel(flow='Rohr(Rohr1a)').values,
            np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
            'Status does not work properly',
        )

        # Verify output flow matches input flow minus losses (relative 20% + absolute 20)
        in1_flow = flow_system.solution['flow|rate'].sel(flow='Rohr(Rohr1a)').values
        expected_out1_flow = in1_flow * 0.8 - np.array([20 if val > 0.1 else 0 for val in in1_flow])
        assert_almost_equal_numeric(
            flow_system.solution['flow|rate'].sel(flow='Rohr(Rohr1b)').values,
            expected_out1_flow,
            'Losses are not computed correctly',
        )

        assert_almost_equal_numeric(
            flow_system.solution['flow|size'].sel(flow='Rohr(Rohr1a)').item(),
            flow_system.solution['flow|size'].sel(flow='Rohr(Rohr2a)').item(),
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
                'Q_th', bus='Fernwärme', size=1000, relative_maximum=np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
            ),
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
        )

        boiler2 = fx.linear_converters.Boiler(
            'Boiler_backup',
            thermal_efficiency=0.4,
            thermal_flow=fx.Flow('Q_th', bus='Wärme lokal'),
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
        )

        last2 = fx.Sink(
            'Wärmelast2',
            inputs=[
                fx.Flow(
                    'Q_th_Last',
                    bus='Wärme lokal',
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
                'Rohr1a',
                bus='Wärme lokal',
                size=fx.InvestParameters(effects_of_investment_per_size=50, maximum_size=1000),
            ),
            out1=fx.Flow('Rohr1b', 'Fernwärme', size=1000),
            in2=fx.Flow(
                'Rohr2a',
                'Fernwärme',
                size=fx.InvestParameters(
                    effects_of_investment_per_size=100, minimum_size=10, maximum_size=1000, mandatory=True
                ),
            ),
            out2=fx.Flow('Rohr2b', bus='Wärme lokal', size=1000),
            balanced=False,
        )

        flow_system.add_elements(transmission, boiler, boiler2, last2)

        flow_system.optimize(highs_solver)

        # Assertions using batched variable naming (flow|status, flow|rate, flow|size)
        assert_almost_equal_numeric(
            flow_system.solution['flow|status'].sel(flow='Rohr(Rohr1a)').values,
            np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
            'Status does not work properly',
        )

        # Verify output flow matches input flow minus losses (relative 20% + absolute 20)
        in1_flow = flow_system.solution['flow|rate'].sel(flow='Rohr(Rohr1a)').values
        expected_out1_flow = in1_flow * 0.8 - np.array([20 if val > 0.1 else 0 for val in in1_flow])
        assert_almost_equal_numeric(
            flow_system.solution['flow|rate'].sel(flow='Rohr(Rohr1b)').values,
            expected_out1_flow,
            'Losses are not computed correctly',
        )

        assert flow_system.solution['flow|size'].sel(flow='Rohr(Rohr1a)').item() > 11

        assert_almost_equal_numeric(
            flow_system.solution['flow|size'].sel(flow='Rohr(Rohr2a)').item(),
            10,
            'Sizing does not work properly',
        )
