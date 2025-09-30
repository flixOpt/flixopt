import flixopt as fx

from .conftest import assert_conequal, assert_var_equal, create_linopy_model


class TestBusModel:
    """Test the FlowModel class."""

    def test_bus(self, basic_flow_system_linopy_coords, coords_config):
        """Test that flow model constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        bus = fx.Bus('TestBus', excess_penalty_per_flow_hour=None)
        flow_system.add_elements(
            bus,
            fx.Sink('WärmelastTest', inputs=[fx.Flow('Q_th_Last', 'TestBus')]),
            fx.Source('GastarifTest', outputs=[fx.Flow('Q_Gas', 'TestBus')]),
        )
        model = create_linopy_model(flow_system)

        assert set(bus.submodel.variables) == {'WärmelastTest(Q_th_Last)|flow_rate', 'GastarifTest(Q_Gas)|flow_rate'}
        assert set(bus.submodel.constraints) == {'TestBus|balance'}

        assert_conequal(
            model.constraints['TestBus|balance'],
            model.variables['GastarifTest(Q_Gas)|flow_rate'] == model.variables['WärmelastTest(Q_th_Last)|flow_rate'],
        )

    def test_bus_penalty(self, basic_flow_system_linopy_coords, coords_config):
        """Test that flow model constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        bus = fx.Bus('TestBus')
        flow_system.add_elements(
            bus,
            fx.Sink('WärmelastTest', inputs=[fx.Flow('Q_th_Last', 'TestBus')]),
            fx.Source('GastarifTest', outputs=[fx.Flow('Q_Gas', 'TestBus')]),
        )
        model = create_linopy_model(flow_system)

        assert set(bus.submodel.variables) == {
            'TestBus|excess_input',
            'TestBus|excess_output',
            'WärmelastTest(Q_th_Last)|flow_rate',
            'GastarifTest(Q_Gas)|flow_rate',
        }
        assert set(bus.submodel.constraints) == {'TestBus|balance'}

        assert_var_equal(
            model.variables['TestBus|excess_input'], model.add_variables(lower=0, coords=model.get_coords())
        )
        assert_var_equal(
            model.variables['TestBus|excess_output'], model.add_variables(lower=0, coords=model.get_coords())
        )

        assert_conequal(
            model.constraints['TestBus|balance'],
            model.variables['GastarifTest(Q_Gas)|flow_rate']
            - model.variables['WärmelastTest(Q_th_Last)|flow_rate']
            + model.variables['TestBus|excess_input']
            - model.variables['TestBus|excess_output']
            == 0,
        )

        assert_conequal(
            model.constraints['TestBus->Penalty'],
            model.variables['TestBus->Penalty']
            == (model.variables['TestBus|excess_input'] * 1e5 * model.hours_per_step).sum()
            + (model.variables['TestBus|excess_output'] * 1e5 * model.hours_per_step).sum(),
        )

    def test_bus_with_coords(self, basic_flow_system_linopy_coords, coords_config):
        """Test bus behavior across different coordinate configurations."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        bus = fx.Bus('TestBus', excess_penalty_per_flow_hour=None)
        flow_system.add_elements(
            bus,
            fx.Sink('WärmelastTest', inputs=[fx.Flow('Q_th_Last', 'TestBus')]),
            fx.Source('GastarifTest', outputs=[fx.Flow('Q_Gas', 'TestBus')]),
        )
        model = create_linopy_model(flow_system)

        # Same core assertions as your existing test
        assert set(bus.submodel.variables) == {'WärmelastTest(Q_th_Last)|flow_rate', 'GastarifTest(Q_Gas)|flow_rate'}
        assert set(bus.submodel.constraints) == {'TestBus|balance'}

        assert_conequal(
            model.constraints['TestBus|balance'],
            model.variables['GastarifTest(Q_Gas)|flow_rate'] == model.variables['WärmelastTest(Q_th_Last)|flow_rate'],
        )

        # Just verify coordinate dimensions are correct
        gas_var = model.variables['GastarifTest(Q_Gas)|flow_rate']
        if flow_system.scenarios is not None:
            assert 'scenario' in gas_var.dims
        assert 'time' in gas_var.dims
