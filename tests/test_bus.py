import flixopt as fx

from .conftest import assert_conequal, assert_var_equal, create_linopy_model


class TestBusModel:
    """Test the FlowModel class."""

    def test_bus(self, basic_flow_system_linopy):
        """Test that flow model constraints are correctly generated."""
        flow_system = basic_flow_system_linopy
        bus = fx.Bus('TestBus', excess_penalty_per_flow_hour=None)
        flow_system.add_elements(
            bus,
            fx.Sink('WärmelastTest', sink=fx.Flow('Q_th_Last', 'TestBus')),
            fx.Source('GastarifTest', source=fx.Flow('Q_Gas', 'TestBus')),
        )
        model = create_linopy_model(flow_system)

        assert set(bus.model.variables) == {'WärmelastTest(Q_th_Last)|flow_rate', 'GastarifTest(Q_Gas)|flow_rate'}
        assert set(bus.model.constraints) == {'TestBus|balance'}

        assert_conequal(
            model.constraints['TestBus|balance'],
            model.variables['GastarifTest(Q_Gas)|flow_rate'] == model.variables['WärmelastTest(Q_th_Last)|flow_rate'],
        )

    def test_bus_penalty(self, basic_flow_system_linopy):
        """Test that flow model constraints are correctly generated."""
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps
        bus = fx.Bus('TestBus', penalty_of_output_deficit=1e4)
        flow_system.add_elements(
            bus,
            fx.Sink('WärmelastTest', sink=fx.Flow('Q_th_Last', 'TestBus')),
            fx.Source('GastarifTest', source=fx.Flow('Q_Gas', 'TestBus')),
        )
        model = create_linopy_model(flow_system)

        assert set(bus.model.variables) == {
            'TestBus|output_deficit',
            'TestBus|input_deficit',
            'WärmelastTest(Q_th_Last)|flow_rate',
            'GastarifTest(Q_Gas)|flow_rate',
        }
        assert set(bus.model.constraints) == {'TestBus|balance'}

        assert_var_equal(model.variables['TestBus|output_deficit'], model.add_variables(lower=0, coords=(timesteps,)))
        assert_var_equal(model.variables['TestBus|input_deficit'], model.add_variables(lower=0, coords=(timesteps,)))

        assert_conequal(
            model.constraints['TestBus|balance'],
            model.variables['GastarifTest(Q_Gas)|flow_rate']
            - model.variables['WärmelastTest(Q_th_Last)|flow_rate']
            + model.variables['TestBus|input_deficit']
            - model.variables['TestBus|output_deficit']
            == 0,
        )

        assert_conequal(
            model.constraints['TestBus->Penalty'],
            model.variables['TestBus->Penalty']
            == (model.variables['TestBus|input_deficit'] * 1e5 * model.hours_per_step).sum()
            + (model.variables['TestBus|output_deficit'] * 1e4 * model.hours_per_step).sum(),
        )
