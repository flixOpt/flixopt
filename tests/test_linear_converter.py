import numpy as np
import pandas as pd
import pytest
import xarray as xr

import flixopt as fx
from flixopt.features import PiecewiseModel

from .conftest import assert_conequal, assert_var_equal, create_linopy_model


class TestLinearConverterModel:
    """Test the LinearConverterModel class."""

    def test_basic_linear_converter(self, basic_flow_system_linopy):
        """Test basic initialization and modeling of a LinearConverter."""
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps

        # Create input and output flows
        input_flow = fx.Flow('input', bus='input_bus', size=100)
        output_flow = fx.Flow('output', bus='output_bus', size=100)

        # Create a simple linear converter with constant conversion factor
        converter = fx.LinearConverter(
            label='Converter',
            inputs=[input_flow],
            outputs=[output_flow],
            conversion_factors=[{input_flow.label: 0.8, output_flow.label: 1.0}]
        )

        # Add to flow system
        flow_system.add_elements(
            fx.Bus('input_bus'),
            fx.Bus('output_bus'),
            converter
        )

        # Create model
        model = create_linopy_model(flow_system)

        # Check variables and constraints
        assert 'Converter(input)|flow_rate' in model.variables
        assert 'Converter(output)|flow_rate' in model.variables
        assert 'Converter|conversion_0' in model.constraints

        # Check conversion constraint (input * 0.8 == output * 1.0)
        assert_conequal(
            model.constraints['Converter|conversion_0'],
            input_flow.model.flow_rate * 0.8 == output_flow.model.flow_rate * 1.0
        )

    def test_linear_converter_time_varying(self, basic_flow_system_linopy):
        """Test a LinearConverter with time-varying conversion factors."""
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps

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
            conversion_factors=[{input_flow.label: efficiency_series, output_flow.label: 1.0}]
        )

        # Add to flow system
        flow_system.add_elements(
            fx.Bus('input_bus'),
            fx.Bus('output_bus'),
            converter
        )

        # Create model
        model = create_linopy_model(flow_system)

        # Check variables and constraints
        assert 'Converter(input)|flow_rate' in model.variables
        assert 'Converter(output)|flow_rate' in model.variables
        assert 'Converter|conversion_0' in model.constraints

        # Check conversion constraint (input * efficiency_series == output * 1.0)
        assert_conequal(
            model.constraints['Converter|conversion_0'],
            input_flow.model.flow_rate * efficiency_series == output_flow.model.flow_rate * 1.0
        )

    def test_linear_converter_multiple_factors(self, basic_flow_system_linopy):
        """Test a LinearConverter with multiple conversion factors."""
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps

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
                {input_flow1.label: 0.2, output_flow2.label: 0.3}   # input1 contributes to output2
            ]
        )

        # Add to flow system
        flow_system.add_elements(
            fx.Bus('input_bus1'),
            fx.Bus('input_bus2'),
            fx.Bus('output_bus1'),
            fx.Bus('output_bus2'),
            converter
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
            input_flow1.model.flow_rate * 0.8 == output_flow1.model.flow_rate * 1.0
        )

        # Check conversion constraint 2 (input2 * 0.5 == output2 * 1.0)
        assert_conequal(
            model.constraints['Converter|conversion_1'],
            input_flow2.model.flow_rate * 0.5 == output_flow2.model.flow_rate * 1.0
        )

        # Check conversion constraint 3 (input1 * 0.2 == output2 * 0.3)
        assert_conequal(
            model.constraints['Converter|conversion_2'],
            input_flow1.model.flow_rate * 0.2 == output_flow2.model.flow_rate * 0.3
        )

    def test_linear_converter_with_on_off(self, basic_flow_system_linopy):
        """Test a LinearConverter with OnOffParameters."""
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps

        # Create input and output flows
        input_flow = fx.Flow('input', bus='input_bus', size=100)
        output_flow = fx.Flow('output', bus='output_bus', size=100)

        # Create OnOffParameters
        on_off_params = fx.OnOffParameters(
            on_hours_total_min=10,
            on_hours_total_max=40,
            effects_per_running_hour={'Costs': 5}
        )

        # Create a linear converter with OnOffParameters
        converter = fx.LinearConverter(
            label='Converter',
            inputs=[input_flow],
            outputs=[output_flow],
            conversion_factors=[{input_flow.label: 0.8, output_flow.label: 1.0}],
            on_off_parameters=on_off_params
        )

        # Add to flow system
        flow_system.add_elements(
            fx.Bus('input_bus'),
            fx.Bus('output_bus'),
            converter,
            fx.Effect('Costs', '€', 'Costs')
        )

        # Create model
        model = create_linopy_model(flow_system)

        # Verify OnOff variables and constraints
        assert 'Converter|on' in model.variables
        assert 'Converter|on_hours_total' in model.variables

        # Check on_hours_total constraint
        assert_conequal(
            model.constraints['Converter|on_hours_total'],
            converter.model.on_off.variables['Converter|on_hours_total'] ==
            (converter.model.on_off.variables['Converter|on'] * model.hours_per_step).sum()
        )

        # Check conversion constraint
        assert_conequal(
            model.constraints['Converter|conversion_0'],
            input_flow.model.flow_rate * 0.8 == output_flow.model.flow_rate * 1.0
        )

        # Check on_off effects
        assert 'Converter->Costs(operation)' in model.constraints
        assert_conequal(
            model.constraints['Converter->Costs(operation)'],
            model.variables['Converter->Costs(operation)'] ==
            converter.model.on_off.variables['Converter|on'] * model.hours_per_step * 5
        )

    def test_piecewise_conversion(self, basic_flow_system_linopy, monkeypatch):
        """Test a LinearConverter with PiecewiseConversion."""
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps

        # Create input and output flows
        input_flow = fx.Flow('input', bus='input_bus', size=100)
        output_flow = fx.Flow('output', bus='output_bus', size=100)

        # Create pieces for piecewise conversion
        pieces = [
            fx.Piece(start=0, end=50),
            fx.Piece(start=50, end=100)
        ]

        # Create piecewise conversion - mocking this since we're testing the model, not the conversion
        piecewise_conversion = fx.PiecewiseConversion({
            input_flow.label: fx.Piecewise(pieces),
            output_flow.label: fx.Piecewise([
                fx.Piece(start=0, end=30),
                fx.Piece(start=30, end=90)
            ])
        })

        # Mock the transform_data method to avoid actual data transformation
        def mock_transform_data(self, flow_system, name_prefix):
            pass

        monkeypatch.setattr(fx.PiecewiseConversion, "transform_data", mock_transform_data)

        # Create a linear converter with piecewise conversion
        converter = fx.LinearConverter(
            label='Converter',
            inputs=[input_flow],
            outputs=[output_flow],
            piecewise_conversion=piecewise_conversion
        )

        # Mock PiecewiseModel to avoid full implementation testing
        class MockPiecewiseModel:
            def __init__(self, **kwargs):
                self.initialized = True
                self.kwargs = kwargs

            def do_modeling(self):
                pass

        # Patch PiecewiseModel with our mock
        monkeypatch.setattr(fx.structure, "PiecewiseModel", MockPiecewiseModel)

        # Add to flow system
        flow_system.add_elements(
            fx.Bus('input_bus'),
            fx.Bus('output_bus'),
            converter
        )

        # Create model - this will use our mocked PiecewiseModel
        model = create_linopy_model(flow_system)

        # Verify that PiecewiseModel was created with correct arguments
        assert len(converter.model.sub_models) == 1
        piecewise_model = converter.model.sub_models[0]
        assert piecewise_model.initialized
        assert piecewise_model.kwargs['label_of_element'] == 'Converter'
        assert piecewise_model.kwargs['label'] == 'Converter'
        assert piecewise_model.kwargs['as_time_series'] == True
        assert piecewise_model.kwargs['zero_point'] == False  # No on_off params provided


if __name__ == '__main__':
    pytest.main()
