import numpy as np
import pytest
import xarray as xr

import flixopt as fx

from .conftest import create_linopy_model


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

        # Check variables and constraints exist
        assert 'flow|rate' in model.variables  # Batched variable with flow dimension
        assert 'converter|conversion_0' in model.constraints  # Batched constraint

        # Verify constraint has expected dimensions (batched model includes converter dim)
        con = model.constraints['converter|conversion_0']
        assert 'converter' in con.dims
        assert 'time' in con.dims

        # Verify flows exist in the batched model (using type-level access)
        flow_rate = model.variables['flow|rate']
        assert 'Converter(input)' in flow_rate.coords['flow'].values
        assert 'Converter(output)' in flow_rate.coords['flow'].values

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

        # Check variables and constraints exist
        assert 'flow|rate' in model.variables  # Batched variable with flow dimension
        assert 'converter|conversion_0' in model.constraints  # Batched constraint

        # Verify constraint has expected dimensions
        con = model.constraints['converter|conversion_0']
        assert 'converter' in con.dims
        assert 'time' in con.dims

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

        # Check constraints for each conversion factor (batched model uses lowercase 'converter')
        assert 'converter|conversion_0' in model.constraints
        assert 'converter|conversion_1' in model.constraints
        assert 'converter|conversion_2' in model.constraints

        # Verify constraints have expected dimensions
        for i in range(3):
            con = model.constraints[f'converter|conversion_{i}']
            assert 'converter' in con.dims
            assert 'time' in con.dims

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

        # Verify Status variables and constraints exist (batched naming)
        assert 'component|status' in model.variables  # Batched status variable
        assert 'component|active_hours' in model.variables

        # Check conversion constraint exists with expected dimensions
        assert 'converter|conversion_0' in model.constraints
        con = model.constraints['converter|conversion_0']
        assert 'converter' in con.dims
        assert 'time' in con.dims

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
        assert 'converter|conversion_0' in model.constraints
        assert 'converter|conversion_1' in model.constraints
        assert 'converter|conversion_2' in model.constraints

        # Verify constraints have expected dimensions
        for i in range(3):
            con = model.constraints[f'converter|conversion_{i}']
            assert 'converter' in con.dims
            assert 'time' in con.dims

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
        assert 'converter|conversion_0' in model.constraints

        # Verify constraint has expected dimensions
        con = model.constraints['converter|conversion_0']
        assert 'converter' in con.dims
        assert 'time' in con.dims

    def test_piecewise_conversion(self, basic_flow_system_linopy_coords, coords_config):
        """Test a LinearConverter with PiecewiseConversion (batched model)."""
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

        # Verify batched piecewise variables exist (tied to component dimension)
        assert 'converter|piecewise_conversion|inside_piece' in model.variables
        assert 'converter|piecewise_conversion|lambda0' in model.variables
        assert 'converter|piecewise_conversion|lambda1' in model.variables

        # Check dimensions of batched variables
        inside_piece = model.variables['converter|piecewise_conversion|inside_piece']
        assert 'converter' in inside_piece.dims
        assert 'segment' in inside_piece.dims
        assert 'time' in inside_piece.dims

        # Verify batched constraints exist
        assert 'converter|piecewise_conversion|lambda_sum' in model.constraints
        assert 'converter|piecewise_conversion|single_segment' in model.constraints

        # Verify coupling constraints for each flow
        assert 'converter|piecewise_conversion|coupling|Converter(input)' in model.constraints
        assert 'converter|piecewise_conversion|coupling|Converter(output)' in model.constraints

    def test_piecewise_conversion_with_status(self, basic_flow_system_linopy_coords, coords_config):
        """Test a LinearConverter with PiecewiseConversion and StatusParameters (batched model)."""
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
        flow_system.add_elements(fx.Bus('input_bus'), fx.Bus('output_bus'), converter)

        # Create model with the piecewise conversion
        model = create_linopy_model(flow_system)

        # Verify batched piecewise variables exist (tied to component dimension)
        assert 'converter|piecewise_conversion|inside_piece' in model.variables
        assert 'converter|piecewise_conversion|lambda0' in model.variables
        assert 'converter|piecewise_conversion|lambda1' in model.variables

        # Status variable should exist (handled by ComponentsModel)
        assert 'component|status' in model.variables

        # Verify batched constraints exist
        assert 'converter|piecewise_conversion|lambda_sum' in model.constraints
        assert 'converter|piecewise_conversion|single_segment' in model.constraints

        # Verify coupling constraints for each flow
        assert 'converter|piecewise_conversion|coupling|Converter(input)' in model.constraints
        assert 'converter|piecewise_conversion|coupling|Converter(output)' in model.constraints


if __name__ == '__main__':
    pytest.main()
