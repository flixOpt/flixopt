import numpy as np
import pytest

import flixopt as fx

from ...conftest import create_linopy_model


class TestLinearConverterModel:
    """Test the LinearConverterModel class."""

    def test_basic_linear_converter(self, basic_flow_system_linopy_coords, coords_config):
        """Test basic initialization and modeling of a LinearConverter."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create input and output flows
        input_flow = fx.Flow('input_bus', flow_id='input', size=100)
        output_flow = fx.Flow('output_bus', flow_id='output', size=100)

        # Create a simple linear converter with constant conversion factor
        converter = fx.LinearConverter(
            'Converter',
            inputs=[input_flow],
            outputs=[output_flow],
            conversion_factors=[{input_flow.label: 0.8, output_flow.label: 1.0}],
        )

        # Add to flow system
        flow_system.add_elements(fx.Bus('input_bus'), fx.Bus('output_bus'), converter)

        # Create model
        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist
        flow_rate = model.variables['flow|rate']
        assert 'Converter(input)' in flow_rate.coords['flow'].values
        assert 'Converter(output)' in flow_rate.coords['flow'].values

        # Check conversion constraint exists
        assert 'converter|conversion' in model.constraints

    def test_linear_converter_time_varying(self, basic_flow_system_linopy_coords, coords_config):
        """Test a LinearConverter with time-varying conversion factors."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        # Create time-varying efficiency (e.g., temperature-dependent)
        varying_efficiency = np.linspace(0.7, 0.9, len(timesteps))

        # Create input and output flows
        input_flow = fx.Flow('input_bus', flow_id='input', size=100)
        output_flow = fx.Flow('output_bus', flow_id='output', size=100)

        # Create a linear converter with time-varying conversion factor
        converter = fx.LinearConverter(
            'Converter',
            inputs=[input_flow],
            outputs=[output_flow],
            conversion_factors=[{input_flow.label: varying_efficiency, output_flow.label: 1.0}],
        )

        # Add to flow system
        flow_system.add_elements(fx.Bus('input_bus'), fx.Bus('output_bus'), converter)

        # Create model
        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist
        flow_rate = model.variables['flow|rate']
        assert 'Converter(input)' in flow_rate.coords['flow'].values
        assert 'Converter(output)' in flow_rate.coords['flow'].values

        # Check conversion constraint exists
        assert 'converter|conversion' in model.constraints

    def test_linear_converter_multiple_factors(self, basic_flow_system_linopy_coords, coords_config):
        """Test a LinearConverter with multiple conversion factors."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create flows
        input_flow1 = fx.Flow('input_bus1', flow_id='input1', size=100)
        input_flow2 = fx.Flow('input_bus2', flow_id='input2', size=100)
        output_flow1 = fx.Flow('output_bus1', flow_id='output1', size=100)
        output_flow2 = fx.Flow('output_bus2', flow_id='output2', size=100)

        # Create a linear converter with multiple inputs/outputs and conversion factors
        converter = fx.LinearConverter(
            'Converter',
            inputs=[input_flow1, input_flow2],
            outputs=[output_flow1, output_flow2],
            conversion_factors=[
                {input_flow1.label: 0.8, output_flow1.label: 1.0},  # input1 -> output1
                {input_flow2.label: 0.5, output_flow2.label: 1.0},  # input2 -> output2
                {input_flow1.label: 0.2, output_flow2.label: 0.3},  # input1 contributes to output2
            ],
        )

        # Add to flow system
        flow_system.add_elements(
            fx.Bus('input_bus1'), fx.Bus('input_bus2'), fx.Bus('output_bus1'), fx.Bus('output_bus2'), converter
        )

        # Create model
        model = create_linopy_model(flow_system)

        # Check constraint for conversion factor (should be named converter|conversion with index dimension)
        assert 'converter|conversion' in model.constraints

    def test_linear_converter_with_status(self, basic_flow_system_linopy_coords, coords_config):
        """Test a LinearConverter with StatusParameters."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create input and output flows
        input_flow = fx.Flow('input_bus', flow_id='input', size=100)
        output_flow = fx.Flow('output_bus', flow_id='output', size=100)

        # Create StatusParameters
        status_params = fx.StatusParameters(
            active_hours_min=10, active_hours_max=40, effects_per_active_hour={'costs': 5}
        )

        # Create a linear converter with StatusParameters
        converter = fx.LinearConverter(
            'Converter',
            inputs=[input_flow],
            outputs=[output_flow],
            conversion_factors=[{input_flow.label: 0.8, output_flow.label: 1.0}],
            status_parameters=status_params,
        )

        # Add to flow system
        flow_system.add_elements(
            fx.Bus('input_bus'),
            fx.Bus('output_bus'),
            converter,
        )

        # Create model
        model = create_linopy_model(flow_system)

        # Verify Status variables exist
        assert 'component|status' in model.variables
        assert 'component|active_hours' in model.variables
        component_status = model.variables['component|status']
        assert 'Converter' in component_status.coords['component'].values

        # Check active_hours constraint
        assert 'component|active_hours' in model.constraints

        # Check conversion constraint
        assert 'converter|conversion' in model.constraints

        # Check status effects - share temporal constraints
        assert 'share|temporal(costs)' in model.constraints

    def test_linear_converter_multidimensional(self, basic_flow_system_linopy_coords, coords_config):
        """Test LinearConverter with multiple inputs, outputs, and connections between them."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create a more complex setup with multiple flows
        input_flow1 = fx.Flow('fuel_bus', flow_id='fuel', size=100)
        input_flow2 = fx.Flow('electricity_bus', flow_id='electricity', size=50)
        output_flow1 = fx.Flow('heat_bus', flow_id='heat', size=70)
        output_flow2 = fx.Flow('cooling_bus', flow_id='cooling', size=30)

        # Create a CHP-like converter with more complex connections
        converter = fx.LinearConverter(
            'MultiConverter',
            inputs=[input_flow1, input_flow2],
            outputs=[output_flow1, output_flow2],
            conversion_factors=[
                # Fuel to heat (primary)
                {input_flow1.label: 0.7, output_flow1.label: 1.0},
                # Electricity to cooling
                {input_flow2.label: 0.3, output_flow2.label: 1.0},
                # Fuel also contributes to cooling
                {input_flow1.label: 0.1, output_flow2.label: 0.5},
            ],
        )

        # Add to flow system
        flow_system.add_elements(
            fx.Bus('fuel_bus'), fx.Bus('electricity_bus'), fx.Bus('heat_bus'), fx.Bus('cooling_bus'), converter
        )

        # Create model
        model = create_linopy_model(flow_system)

        # Check conversion constraint exists
        assert 'converter|conversion' in model.constraints

    def test_edge_case_time_varying_conversion(self, basic_flow_system_linopy_coords, coords_config):
        """Test edge case with extreme time-varying conversion factors."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        # Create fluctuating conversion efficiency (e.g., for a heat pump)
        # Values range from very low (0.1) to very high (5.0)
        fluctuating_cop = np.concatenate(
            [
                np.linspace(0.1, 1.0, len(timesteps) // 3),
                np.linspace(1.0, 5.0, len(timesteps) // 3),
                np.linspace(5.0, 0.1, len(timesteps) // 3 + len(timesteps) % 3),
            ]
        )

        # Create input and output flows
        input_flow = fx.Flow('electricity_bus', flow_id='electricity', size=100)
        output_flow = fx.Flow('heat_bus', flow_id='heat', size=500)  # Higher maximum to allow for COP of 5

        conversion_factors = [{input_flow.label: fluctuating_cop, output_flow.label: np.ones(len(timesteps))}]

        # Create the converter
        converter = fx.LinearConverter(
            'VariableConverter', inputs=[input_flow], outputs=[output_flow], conversion_factors=conversion_factors
        )

        # Add to flow system
        flow_system.add_elements(fx.Bus('electricity_bus'), fx.Bus('heat_bus'), converter)

        # Create model
        model = create_linopy_model(flow_system)

        # Check that the correct constraint was created
        assert 'converter|conversion' in model.constraints

    def test_piecewise_conversion(self, basic_flow_system_linopy_coords, coords_config):
        """Test a LinearConverter with PiecewiseConversion."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create input and output flows
        input_flow = fx.Flow('input_bus', flow_id='input', size=100)
        output_flow = fx.Flow('output_bus', flow_id='output', size=100)

        # Create pieces for piecewise conversion
        # For input flow: two pieces from 0-50 and 50-100
        input_pieces = [fx.Piece(start=0, end=50), fx.Piece(start=50, end=100)]

        # For output flow: two pieces from 0-30 and 30-90
        output_pieces = [fx.Piece(start=0, end=30), fx.Piece(start=30, end=90)]

        # Create piecewise conversion
        piecewise_conversion = fx.PiecewiseConversion(
            {input_flow.label: fx.Piecewise(input_pieces), output_flow.label: fx.Piecewise(output_pieces)}
        )

        # Create a linear converter with piecewise conversion
        converter = fx.LinearConverter(
            'Converter', inputs=[input_flow], outputs=[output_flow], piecewise_conversion=piecewise_conversion
        )

        # Add to flow system
        flow_system.add_elements(fx.Bus('input_bus'), fx.Bus('output_bus'), converter)

        # Create model with the piecewise conversion
        model = create_linopy_model(flow_system)

        # Check that we have the expected pieces (2 in this case)
        # Verify that variables were created for piecewise
        # Check piecewise-related constraints exist
        assert (
            'piecewise|lambda' in model.constraints
            or 'piecewise|inside_piece' in model.constraints
            or any('piecewise' in name.lower() or 'piece' in name.lower() for name in model.constraints)
        )

    def test_piecewise_conversion_with_status(self, basic_flow_system_linopy_coords, coords_config):
        """Test a LinearConverter with PiecewiseConversion and StatusParameters."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create input and output flows
        input_flow = fx.Flow('input_bus', flow_id='input', size=100)
        output_flow = fx.Flow('output_bus', flow_id='output', size=100)

        # Create pieces for piecewise conversion
        input_pieces = [fx.Piece(start=0, end=50), fx.Piece(start=50, end=100)]

        output_pieces = [fx.Piece(start=0, end=30), fx.Piece(start=30, end=90)]

        # Create piecewise conversion
        piecewise_conversion = fx.PiecewiseConversion(
            {input_flow.label: fx.Piecewise(input_pieces), output_flow.label: fx.Piecewise(output_pieces)}
        )

        # Create StatusParameters
        status_params = fx.StatusParameters(
            active_hours_min=10, active_hours_max=40, effects_per_active_hour={'costs': 5}
        )

        # Create a linear converter with piecewise conversion and status parameters
        converter = fx.LinearConverter(
            'Converter',
            inputs=[input_flow],
            outputs=[output_flow],
            piecewise_conversion=piecewise_conversion,
            status_parameters=status_params,
        )

        # Add to flow system
        flow_system.add_elements(
            fx.Bus('input_bus'),
            fx.Bus('output_bus'),
            converter,
        )

        # Create model with the piecewise conversion
        model = create_linopy_model(flow_system)

        # Also check that the Status model is working correctly
        assert 'component|status' in model.variables
        assert 'component|active_hours' in model.constraints

        # Verify that the costs effect is applied through share temporal constraints
        assert 'share|temporal(costs)' in model.constraints


if __name__ == '__main__':
    pytest.main()
