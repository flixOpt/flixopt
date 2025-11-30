import numpy as np
import pytest
import xarray as xr

import flixopt as fx

from .conftest import assert_conequal, assert_var_equal, create_linopy_model


class TestLinearConverterModel:
    """Test the LinearConverterModel class."""

    def test_basic_linear_converter(self, basic_flow_system_linopy_coords, coords_config):
        """Test basic initialization and modeling of a LinearConverter."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create input and output flows
        input_flow = fx.Flow('input', bus='input_bus', size=100)
        output_flow = fx.Flow('output', bus='output_bus', size=100)

        # Create a simple linear converter with constant conversion factor
        converter = fx.LinearConverter(
            label='Converter',
            inputs=[input_flow],
            outputs=[output_flow],
            conversion_factors=[{input_flow.label: 0.8, output_flow.label: 1.0}],
        )

        # Add to flow system
        flow_system.add_elements(fx.Bus('input_bus'), fx.Bus('output_bus'), converter)

        # Create model
        model = create_linopy_model(flow_system)

        # Check variables and constraints
        assert 'Converter(input)|flow_rate' in model.variables
        assert 'Converter(output)|flow_rate' in model.variables
        assert 'Converter|conversion_0' in model.constraints

        # Check conversion constraint (input * 0.8 == output * 1.0)
        assert_conequal(
            model.constraints['Converter|conversion_0'],
            input_flow.submodel.flow_rate * 0.8 == output_flow.submodel.flow_rate * 1.0,
        )

    def test_linear_converter_time_varying(self, basic_flow_system_linopy_coords, coords_config):
        """Test a LinearConverter with time-varying conversion factors."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        timesteps = flow_system.timesteps

        # Create time-varying efficiency (e.g., temperature-dependent)
        varying_efficiency = np.linspace(0.7, 0.9, len(timesteps))
        efficiency_series = xr.DataArray(varying_efficiency, coords=(timesteps,))

        # Create input and output flows
        input_flow = fx.Flow('input', bus='input_bus', size=100)
        output_flow = fx.Flow('output', bus='output_bus', size=100)

        # Create a linear converter with time-varying conversion factor
        converter = fx.LinearConverter(
            label='Converter',
            inputs=[input_flow],
            outputs=[output_flow],
            conversion_factors=[{input_flow.label: efficiency_series, output_flow.label: 1.0}],
        )

        # Add to flow system
        flow_system.add_elements(fx.Bus('input_bus'), fx.Bus('output_bus'), converter)

        # Create model
        model = create_linopy_model(flow_system)

        # Check variables and constraints
        assert 'Converter(input)|flow_rate' in model.variables
        assert 'Converter(output)|flow_rate' in model.variables
        assert 'Converter|conversion_0' in model.constraints

        # Check conversion constraint (input * efficiency_series == output * 1.0)
        assert_conequal(
            model.constraints['Converter|conversion_0'],
            input_flow.submodel.flow_rate * efficiency_series == output_flow.submodel.flow_rate * 1.0,
        )

    def test_linear_converter_multiple_factors(self, basic_flow_system_linopy_coords, coords_config):
        """Test a LinearConverter with multiple conversion factors."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create flows
        input_flow1 = fx.Flow('input1', bus='input_bus1', size=100)
        input_flow2 = fx.Flow('input2', bus='input_bus2', size=100)
        output_flow1 = fx.Flow('output1', bus='output_bus1', size=100)
        output_flow2 = fx.Flow('output2', bus='output_bus2', size=100)

        # Create a linear converter with multiple inputs/outputs and conversion factors
        converter = fx.LinearConverter(
            label='Converter',
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

        # Check constraints for each conversion factor
        assert 'Converter|conversion_0' in model.constraints
        assert 'Converter|conversion_1' in model.constraints
        assert 'Converter|conversion_2' in model.constraints

        # Check conversion constraint 1 (input1 * 0.8 == output1 * 1.0)
        assert_conequal(
            model.constraints['Converter|conversion_0'],
            input_flow1.submodel.flow_rate * 0.8 == output_flow1.submodel.flow_rate * 1.0,
        )

        # Check conversion constraint 2 (input2 * 0.5 == output2 * 1.0)
        assert_conequal(
            model.constraints['Converter|conversion_1'],
            input_flow2.submodel.flow_rate * 0.5 == output_flow2.submodel.flow_rate * 1.0,
        )

        # Check conversion constraint 3 (input1 * 0.2 == output2 * 0.3)
        assert_conequal(
            model.constraints['Converter|conversion_2'],
            input_flow1.submodel.flow_rate * 0.2 == output_flow2.submodel.flow_rate * 0.3,
        )

    def test_linear_converter_with_status(self, basic_flow_system_linopy_coords, coords_config):
        """Test a LinearConverter with StatusParameters."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create input and output flows
        input_flow = fx.Flow('input', bus='input_bus', size=100)
        output_flow = fx.Flow('output', bus='output_bus', size=100)

        # Create StatusParameters
        status_params = fx.StatusParameters(
            active_hours_min=10, active_hours_max=40, effects_per_active_hour={'costs': 5}
        )

        # Create a linear converter with StatusParameters
        converter = fx.LinearConverter(
            label='Converter',
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

        # Verify Status variables and constraints
        assert 'Converter|status' in model.variables
        assert 'Converter|active_hours' in model.variables

        # Check active_hours constraint
        assert_conequal(
            model.constraints['Converter|active_hours'],
            model.variables['Converter|active_hours']
            == (model.variables['Converter|status'] * model.hours_per_step).sum('time'),
        )

        # Check conversion constraint
        assert_conequal(
            model.constraints['Converter|conversion_0'],
            input_flow.submodel.flow_rate * 0.8 == output_flow.submodel.flow_rate * 1.0,
        )

        # Check status effects
        assert 'Converter->costs(temporal)' in model.constraints
        assert_conequal(
            model.constraints['Converter->costs(temporal)'],
            model.variables['Converter->costs(temporal)']
            == model.variables['Converter|status'] * model.hours_per_step * 5,
        )

    def test_linear_converter_multidimensional(self, basic_flow_system_linopy_coords, coords_config):
        """Test LinearConverter with multiple inputs, outputs, and connections between them."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create a more complex setup with multiple flows
        input_flow1 = fx.Flow('fuel', bus='fuel_bus', size=100)
        input_flow2 = fx.Flow('electricity', bus='electricity_bus', size=50)
        output_flow1 = fx.Flow('heat', bus='heat_bus', size=70)
        output_flow2 = fx.Flow('cooling', bus='cooling_bus', size=30)

        # Create a CHP-like converter with more complex connections
        converter = fx.LinearConverter(
            label='MultiConverter',
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

        # Check all expected constraints
        assert 'MultiConverter|conversion_0' in model.constraints
        assert 'MultiConverter|conversion_1' in model.constraints
        assert 'MultiConverter|conversion_2' in model.constraints

        # Check the conversion equations
        assert_conequal(
            model.constraints['MultiConverter|conversion_0'],
            input_flow1.submodel.flow_rate * 0.7 == output_flow1.submodel.flow_rate * 1.0,
        )

        assert_conequal(
            model.constraints['MultiConverter|conversion_1'],
            input_flow2.submodel.flow_rate * 0.3 == output_flow2.submodel.flow_rate * 1.0,
        )

        assert_conequal(
            model.constraints['MultiConverter|conversion_2'],
            input_flow1.submodel.flow_rate * 0.1 == output_flow2.submodel.flow_rate * 0.5,
        )

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
        input_flow = fx.Flow('electricity', bus='electricity_bus', size=100)
        output_flow = fx.Flow('heat', bus='heat_bus', size=500)  # Higher maximum to allow for COP of 5

        conversion_factors = [{input_flow.label: fluctuating_cop, output_flow.label: np.ones(len(timesteps))}]

        # Create the converter
        converter = fx.LinearConverter(
            label='VariableConverter', inputs=[input_flow], outputs=[output_flow], conversion_factors=conversion_factors
        )

        # Add to flow system
        flow_system.add_elements(fx.Bus('electricity_bus'), fx.Bus('heat_bus'), converter)

        # Create model
        model = create_linopy_model(flow_system)

        # Check that the correct constraint was created
        assert 'VariableConverter|conversion_0' in model.constraints

        factor = converter.conversion_factors[0]['electricity']

        assert factor.dims == tuple(model.get_coords())

        # Verify the constraint has the time-varying coefficient
        assert_conequal(
            model.constraints['VariableConverter|conversion_0'],
            input_flow.submodel.flow_rate * factor == output_flow.submodel.flow_rate * 1.0,
        )

    def test_piecewise_conversion(self, basic_flow_system_linopy_coords, coords_config):
        """Test a LinearConverter with PiecewiseConversion."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create input and output flows
        input_flow = fx.Flow('input', bus='input_bus', size=100)
        output_flow = fx.Flow('output', bus='output_bus', size=100)

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
            label='Converter', inputs=[input_flow], outputs=[output_flow], piecewise_conversion=piecewise_conversion
        )

        # Add to flow system
        flow_system.add_elements(fx.Bus('input_bus'), fx.Bus('output_bus'), converter)

        # Create model with the piecewise conversion
        model = create_linopy_model(flow_system)

        # Verify that PiecewiseModel was created and added as a submodel
        assert converter.submodel.piecewise_conversion is not None

        # Get the PiecewiseModel instance
        piecewise_model = converter.submodel.piecewise_conversion

        # Check that we have the expected pieces (2 in this case)
        assert len(piecewise_model.pieces) == 2

        # Verify that variables were created for each piece
        for i, _ in enumerate(piecewise_model.pieces):
            # Each piece should have lambda0, lambda1, and inside_piece variables
            assert f'Converter|Piece_{i}|lambda0' in model.variables
            assert f'Converter|Piece_{i}|lambda1' in model.variables
            assert f'Converter|Piece_{i}|inside_piece' in model.variables
            lambda0 = model.variables[f'Converter|Piece_{i}|lambda0']
            lambda1 = model.variables[f'Converter|Piece_{i}|lambda1']
            inside_piece = model.variables[f'Converter|Piece_{i}|inside_piece']

            assert_var_equal(inside_piece, model.add_variables(binary=True, coords=model.get_coords()))
            assert_var_equal(lambda0, model.add_variables(lower=0, upper=1, coords=model.get_coords()))
            assert_var_equal(lambda1, model.add_variables(lower=0, upper=1, coords=model.get_coords()))

            # Check that the inside_piece constraint exists
            assert f'Converter|Piece_{i}|inside_piece' in model.constraints
            # Check the relationship between inside_piece and lambdas
            assert_conequal(model.constraints[f'Converter|Piece_{i}|inside_piece'], inside_piece == lambda0 + lambda1)

        assert_conequal(
            model.constraints['Converter|Converter(input)|flow_rate|lambda'],
            model.variables['Converter(input)|flow_rate']
            == model.variables['Converter|Piece_0|lambda0'] * 0
            + model.variables['Converter|Piece_0|lambda1'] * 50
            + model.variables['Converter|Piece_1|lambda0'] * 50
            + model.variables['Converter|Piece_1|lambda1'] * 100,
        )

        assert_conequal(
            model.constraints['Converter|Converter(output)|flow_rate|lambda'],
            model.variables['Converter(output)|flow_rate']
            == model.variables['Converter|Piece_0|lambda0'] * 0
            + model.variables['Converter|Piece_0|lambda1'] * 30
            + model.variables['Converter|Piece_1|lambda0'] * 30
            + model.variables['Converter|Piece_1|lambda1'] * 90,
        )

        # Check that we enforce the constraint that only one segment can be active
        assert 'Converter|Converter(input)|flow_rate|single_segment' in model.constraints

        # The constraint should enforce that the sum of inside_piece variables is limited
        # If there's no status parameter, the right-hand side should be 1
        assert_conequal(
            model.constraints['Converter|Converter(input)|flow_rate|single_segment'],
            sum([model.variables[f'Converter|Piece_{i}|inside_piece'] for i in range(len(piecewise_model.pieces))])
            <= 1,
        )

    def test_piecewise_conversion_with_status(self, basic_flow_system_linopy_coords, coords_config):
        """Test a LinearConverter with PiecewiseConversion and StatusParameters."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create input and output flows
        input_flow = fx.Flow('input', bus='input_bus', size=100)
        output_flow = fx.Flow('output', bus='output_bus', size=100)

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
            label='Converter',
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

        # Verify that PiecewiseModel was created and added as a submodel
        assert converter.submodel.piecewise_conversion is not None

        # Get the PiecewiseModel instance
        piecewise_model = converter.submodel.piecewise_conversion

        # Check that we have the expected pieces (2 in this case)
        assert len(piecewise_model.pieces) == 2

        # Verify that the status variable was used as the zero_point for the piecewise model
        # When using StatusParameters, the zero_point should be the status variable
        assert 'Converter|status' in model.variables
        assert piecewise_model.zero_point is not None  # Should be a variable

        # Verify that variables were created for each piece
        for i, _ in enumerate(piecewise_model.pieces):
            # Each piece should have lambda0, lambda1, and inside_piece variables
            assert f'Converter|Piece_{i}|lambda0' in model.variables
            assert f'Converter|Piece_{i}|lambda1' in model.variables
            assert f'Converter|Piece_{i}|inside_piece' in model.variables
            lambda0 = model.variables[f'Converter|Piece_{i}|lambda0']
            lambda1 = model.variables[f'Converter|Piece_{i}|lambda1']
            inside_piece = model.variables[f'Converter|Piece_{i}|inside_piece']

            assert_var_equal(inside_piece, model.add_variables(binary=True, coords=model.get_coords()))
            assert_var_equal(lambda0, model.add_variables(lower=0, upper=1, coords=model.get_coords()))
            assert_var_equal(lambda1, model.add_variables(lower=0, upper=1, coords=model.get_coords()))

            # Check that the inside_piece constraint exists
            assert f'Converter|Piece_{i}|inside_piece' in model.constraints
            # Check the relationship between inside_piece and lambdas
            assert_conequal(model.constraints[f'Converter|Piece_{i}|inside_piece'], inside_piece == lambda0 + lambda1)

        assert_conequal(
            model.constraints['Converter|Converter(input)|flow_rate|lambda'],
            model.variables['Converter(input)|flow_rate']
            == model.variables['Converter|Piece_0|lambda0'] * 0
            + model.variables['Converter|Piece_0|lambda1'] * 50
            + model.variables['Converter|Piece_1|lambda0'] * 50
            + model.variables['Converter|Piece_1|lambda1'] * 100,
        )

        assert_conequal(
            model.constraints['Converter|Converter(output)|flow_rate|lambda'],
            model.variables['Converter(output)|flow_rate']
            == model.variables['Converter|Piece_0|lambda0'] * 0
            + model.variables['Converter|Piece_0|lambda1'] * 30
            + model.variables['Converter|Piece_1|lambda0'] * 30
            + model.variables['Converter|Piece_1|lambda1'] * 90,
        )

        # Check that we enforce the constraint that only one segment can be active
        assert 'Converter|Converter(input)|flow_rate|single_segment' in model.constraints

        # The constraint should enforce that the sum of inside_piece variables is limited
        assert_conequal(
            model.constraints['Converter|Converter(input)|flow_rate|single_segment'],
            sum([model.variables[f'Converter|Piece_{i}|inside_piece'] for i in range(len(piecewise_model.pieces))])
            <= model.variables['Converter|status'],
        )

        # Also check that the Status model is working correctly
        assert 'Converter|active_hours' in model.constraints
        assert_conequal(
            model.constraints['Converter|active_hours'],
            model['Converter|active_hours'] == (model['Converter|status'] * model.hours_per_step).sum('time'),
        )

        # Verify that the costs effect is applied
        assert 'Converter->costs(temporal)' in model.constraints
        assert_conequal(
            model.constraints['Converter->costs(temporal)'],
            model.variables['Converter->costs(temporal)']
            == model.variables['Converter|status'] * model.hours_per_step * 5,
        )


if __name__ == '__main__':
    pytest.main()
