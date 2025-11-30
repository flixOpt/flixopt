import flixopt as fx

from .conftest import assert_conequal, assert_var_equal, create_linopy_model


class TestBusModel:
    """Test the FlowModel class."""

    def test_bus(self, basic_flow_system_linopy_coords, coords_config):
        """Test that flow model constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        bus = fx.Bus('TestBus', imbalance_penalty_per_flow_hour=None)
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
        bus = fx.Bus('TestBus', imbalance_penalty_per_flow_hour=1e5)
        flow_system.add_elements(
            bus,
            fx.Sink('WärmelastTest', inputs=[fx.Flow('Q_th_Last', 'TestBus')]),
            fx.Source('GastarifTest', outputs=[fx.Flow('Q_Gas', 'TestBus')]),
        )
        model = create_linopy_model(flow_system)

        assert set(bus.submodel.variables) == {
            'TestBus|virtual_supply',
            'TestBus|virtual_demand',
            'WärmelastTest(Q_th_Last)|flow_rate',
            'GastarifTest(Q_Gas)|flow_rate',
        }
        assert set(bus.submodel.constraints) == {'TestBus|balance'}

        assert_var_equal(
            model.variables['TestBus|virtual_supply'], model.add_variables(lower=0, coords=model.get_coords())
        )
        assert_var_equal(
            model.variables['TestBus|virtual_demand'], model.add_variables(lower=0, coords=model.get_coords())
        )

        assert_conequal(
            model.constraints['TestBus|balance'],
            model.variables['GastarifTest(Q_Gas)|flow_rate']
            - model.variables['WärmelastTest(Q_th_Last)|flow_rate']
            + model.variables['TestBus|virtual_supply']
            - model.variables['TestBus|virtual_demand']
            == 0,
        )

        # Penalty is now added as shares to the Penalty effect's temporal model
        # Check that the penalty shares exist
        assert 'TestBus->Penalty(temporal)' in model.constraints
        assert 'TestBus->Penalty(temporal)' in model.variables

        # The penalty share should equal the imbalance (virtual_supply + virtual_demand) times the penalty cost
        # Let's verify the total penalty contribution by checking the effect's temporal model
        penalty_effect = flow_system.effects.penalty_effect
        assert penalty_effect.submodel is not None
        assert 'TestBus' in penalty_effect.submodel.temporal.shares

        assert_conequal(
            model.constraints['TestBus->Penalty(temporal)'],
            model.variables['TestBus->Penalty(temporal)']
            == model.variables['TestBus|virtual_supply'] * 1e5 * model.hours_per_step
            + model.variables['TestBus|virtual_demand'] * 1e5 * model.hours_per_step,
        )

    def test_bus_with_coords(self, basic_flow_system_linopy_coords, coords_config):
        """Test bus behavior across different coordinate configurations."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        bus = fx.Bus('TestBus', imbalance_penalty_per_flow_hour=None)
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
