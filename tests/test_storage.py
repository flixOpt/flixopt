import numpy as np
import pytest

import flixopt as fx

from .conftest import assert_conequal, assert_var_equal, create_linopy_model


class TestStorageModel:
    """Test that storage model variables and constraints are correctly generated."""

    def test_basic_storage(self, basic_flow_system_linopy_coords, coords_config):
        """Test that basic storage model variables and constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create a simple storage
        storage = fx.Storage(
            'TestStorage',
            charging=fx.Flow('Q_th_in', bus='Fernwärme', size=20),
            discharging=fx.Flow('Q_th_out', bus='Fernwärme', size=20),
            capacity_in_flow_hours=30,  # 30 kWh storage capacity
            initial_charge_state=0,  # Start empty
            prevent_simultaneous_charge_and_discharge=True,
        )

        flow_system.add_elements(storage)
        model = create_linopy_model(flow_system)

        # Check that all expected variables exist - linopy model variables are accessed by indexing
        expected_variables = {
            'TestStorage(Q_th_in)|flow_rate',
            'TestStorage(Q_th_in)|total_flow_hours',
            'TestStorage(Q_th_out)|flow_rate',
            'TestStorage(Q_th_out)|total_flow_hours',
            'TestStorage|charge_state',
            'TestStorage|netto_discharge',
        }
        for var_name in expected_variables:
            assert var_name in model.variables, f'Missing variable: {var_name}'

        # Check that all expected constraints exist - linopy model constraints are accessed by indexing
        expected_constraints = {
            'TestStorage(Q_th_in)|total_flow_hours',
            'TestStorage(Q_th_out)|total_flow_hours',
            'TestStorage|netto_discharge',
            'TestStorage|charge_state',
            'TestStorage|initial_charge_state',
        }
        for con_name in expected_constraints:
            assert con_name in model.constraints, f'Missing constraint: {con_name}'

        # Check variable properties
        assert_var_equal(
            model['TestStorage(Q_th_in)|flow_rate'], model.add_variables(lower=0, upper=20, coords=model.get_coords())
        )
        assert_var_equal(
            model['TestStorage(Q_th_out)|flow_rate'], model.add_variables(lower=0, upper=20, coords=model.get_coords())
        )
        assert_var_equal(
            model['TestStorage|charge_state'],
            model.add_variables(lower=0, upper=30, coords=model.get_coords(extra_timestep=True)),
        )

        # Check constraint formulations
        assert_conequal(
            model.constraints['TestStorage|netto_discharge'],
            model.variables['TestStorage|netto_discharge']
            == model.variables['TestStorage(Q_th_out)|flow_rate'] - model.variables['TestStorage(Q_th_in)|flow_rate'],
        )

        charge_state = model.variables['TestStorage|charge_state']
        assert_conequal(
            model.constraints['TestStorage|charge_state'],
            charge_state.isel(time=slice(1, None))
            == charge_state.isel(time=slice(None, -1))
            + model.variables['TestStorage(Q_th_in)|flow_rate'] * model.timestep_duration
            - model.variables['TestStorage(Q_th_out)|flow_rate'] * model.timestep_duration,
        )
        # Check initial charge state constraint
        assert_conequal(
            model.constraints['TestStorage|initial_charge_state'],
            model.variables['TestStorage|charge_state'].isel(time=0) == 0,
        )

    def test_lossy_storage(self, basic_flow_system_linopy_coords, coords_config):
        """Test that basic storage model variables and constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create a simple storage
        storage = fx.Storage(
            'TestStorage',
            charging=fx.Flow('Q_th_in', bus='Fernwärme', size=20),
            discharging=fx.Flow('Q_th_out', bus='Fernwärme', size=20),
            capacity_in_flow_hours=30,  # 30 kWh storage capacity
            initial_charge_state=0,  # Start empty
            eta_charge=0.9,  # Charging efficiency
            eta_discharge=0.8,  # Discharging efficiency
            relative_loss_per_hour=0.05,  # 5% loss per hour
            prevent_simultaneous_charge_and_discharge=True,
        )

        flow_system.add_elements(storage)
        model = create_linopy_model(flow_system)

        # Check that all expected variables exist - linopy model variables are accessed by indexing
        expected_variables = {
            'TestStorage(Q_th_in)|flow_rate',
            'TestStorage(Q_th_in)|total_flow_hours',
            'TestStorage(Q_th_out)|flow_rate',
            'TestStorage(Q_th_out)|total_flow_hours',
            'TestStorage|charge_state',
            'TestStorage|netto_discharge',
        }
        for var_name in expected_variables:
            assert var_name in model.variables, f'Missing variable: {var_name}'

        # Check that all expected constraints exist - linopy model constraints are accessed by indexing
        expected_constraints = {
            'TestStorage(Q_th_in)|total_flow_hours',
            'TestStorage(Q_th_out)|total_flow_hours',
            'TestStorage|netto_discharge',
            'TestStorage|charge_state',
            'TestStorage|initial_charge_state',
        }
        for con_name in expected_constraints:
            assert con_name in model.constraints, f'Missing constraint: {con_name}'

        # Check variable properties
        assert_var_equal(
            model['TestStorage(Q_th_in)|flow_rate'], model.add_variables(lower=0, upper=20, coords=model.get_coords())
        )
        assert_var_equal(
            model['TestStorage(Q_th_out)|flow_rate'], model.add_variables(lower=0, upper=20, coords=model.get_coords())
        )
        assert_var_equal(
            model['TestStorage|charge_state'],
            model.add_variables(lower=0, upper=30, coords=model.get_coords(extra_timestep=True)),
        )

        # Check constraint formulations
        assert_conequal(
            model.constraints['TestStorage|netto_discharge'],
            model.variables['TestStorage|netto_discharge']
            == model.variables['TestStorage(Q_th_out)|flow_rate'] - model.variables['TestStorage(Q_th_in)|flow_rate'],
        )

        charge_state = model.variables['TestStorage|charge_state']
        rel_loss = 0.05
        timestep_duration = model.timestep_duration
        charge_rate = model.variables['TestStorage(Q_th_in)|flow_rate']
        discharge_rate = model.variables['TestStorage(Q_th_out)|flow_rate']
        eff_charge = 0.9
        eff_discharge = 0.8

        assert_conequal(
            model.constraints['TestStorage|charge_state'],
            charge_state.isel(time=slice(1, None))
            == charge_state.isel(time=slice(None, -1)) * (1 - rel_loss) ** timestep_duration
            + charge_rate * eff_charge * timestep_duration
            - discharge_rate / eff_discharge * timestep_duration,
        )

        # Check initial charge state constraint
        assert_conequal(
            model.constraints['TestStorage|initial_charge_state'],
            model.variables['TestStorage|charge_state'].isel(time=0) == 0,
        )

    def test_charge_state_bounds(self, basic_flow_system_linopy_coords, coords_config):
        """Test that basic storage model variables and constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create a simple storage
        storage = fx.Storage(
            'TestStorage',
            charging=fx.Flow('Q_th_in', bus='Fernwärme', size=20),
            discharging=fx.Flow('Q_th_out', bus='Fernwärme', size=20),
            capacity_in_flow_hours=30,  # 30 kWh storage capacity
            initial_charge_state=3,
            prevent_simultaneous_charge_and_discharge=True,
            relative_maximum_charge_state=np.array([0.14, 0.22, 0.3, 0.38, 0.46, 0.54, 0.62, 0.7, 0.78, 0.86]),
            relative_minimum_charge_state=np.array([0.07, 0.11, 0.15, 0.19, 0.23, 0.27, 0.31, 0.35, 0.39, 0.43]),
        )

        flow_system.add_elements(storage)
        model = create_linopy_model(flow_system)

        # Check that all expected variables exist - linopy model variables are accessed by indexing
        expected_variables = {
            'TestStorage(Q_th_in)|flow_rate',
            'TestStorage(Q_th_in)|total_flow_hours',
            'TestStorage(Q_th_out)|flow_rate',
            'TestStorage(Q_th_out)|total_flow_hours',
            'TestStorage|charge_state',
            'TestStorage|netto_discharge',
        }
        for var_name in expected_variables:
            assert var_name in model.variables, f'Missing variable: {var_name}'

        # Check that all expected constraints exist - linopy model constraints are accessed by indexing
        expected_constraints = {
            'TestStorage(Q_th_in)|total_flow_hours',
            'TestStorage(Q_th_out)|total_flow_hours',
            'TestStorage|netto_discharge',
            'TestStorage|charge_state',
            'TestStorage|initial_charge_state',
        }
        for con_name in expected_constraints:
            assert con_name in model.constraints, f'Missing constraint: {con_name}'

        # Check variable properties
        assert_var_equal(
            model['TestStorage(Q_th_in)|flow_rate'], model.add_variables(lower=0, upper=20, coords=model.get_coords())
        )
        assert_var_equal(
            model['TestStorage(Q_th_out)|flow_rate'], model.add_variables(lower=0, upper=20, coords=model.get_coords())
        )
        assert_var_equal(
            model['TestStorage|charge_state'],
            model.add_variables(
                lower=storage.relative_minimum_charge_state.reindex(
                    time=model.get_coords(extra_timestep=True)['time']
                ).ffill('time')
                * 30,
                upper=storage.relative_maximum_charge_state.reindex(
                    time=model.get_coords(extra_timestep=True)['time']
                ).ffill('time')
                * 30,
                coords=model.get_coords(extra_timestep=True),
            ),
        )

        # Check constraint formulations
        assert_conequal(
            model.constraints['TestStorage|netto_discharge'],
            model.variables['TestStorage|netto_discharge']
            == model.variables['TestStorage(Q_th_out)|flow_rate'] - model.variables['TestStorage(Q_th_in)|flow_rate'],
        )

        charge_state = model.variables['TestStorage|charge_state']
        assert_conequal(
            model.constraints['TestStorage|charge_state'],
            charge_state.isel(time=slice(1, None))
            == charge_state.isel(time=slice(None, -1))
            + model.variables['TestStorage(Q_th_in)|flow_rate'] * model.timestep_duration
            - model.variables['TestStorage(Q_th_out)|flow_rate'] * model.timestep_duration,
        )
        # Check initial charge state constraint
        assert_conequal(
            model.constraints['TestStorage|initial_charge_state'],
            model.variables['TestStorage|charge_state'].isel(time=0) == 3,
        )

    def test_storage_with_investment(self, basic_flow_system_linopy_coords, coords_config):
        """Test storage with investment parameters."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create storage with investment parameters
        storage = fx.Storage(
            'InvestStorage',
            charging=fx.Flow('Q_th_in', bus='Fernwärme', size=20),
            discharging=fx.Flow('Q_th_out', bus='Fernwärme', size=20),
            capacity_in_flow_hours=fx.InvestParameters(
                effects_of_investment=100,
                effects_of_investment_per_size=10,
                minimum_size=20,
                maximum_size=100,
                mandatory=False,
            ),
            initial_charge_state=0,
            eta_charge=0.9,
            eta_discharge=0.9,
            relative_loss_per_hour=0.05,
            prevent_simultaneous_charge_and_discharge=True,
        )

        flow_system.add_elements(storage)
        model = create_linopy_model(flow_system)

        # Check investment variables exist
        for var_name in {
            'InvestStorage|charge_state',
            'InvestStorage|size',
            'InvestStorage|invested',
        }:
            assert var_name in model.variables, f'Missing investment variable: {var_name}'

        # Check investment constraints exist
        for con_name in {'InvestStorage|size|ub', 'InvestStorage|size|lb'}:
            assert con_name in model.constraints, f'Missing investment constraint: {con_name}'

        # Check variable properties
        assert_var_equal(
            model['InvestStorage|size'],
            model.add_variables(lower=0, upper=100, coords=model.get_coords(['period', 'scenario'])),
        )
        assert_var_equal(
            model['InvestStorage|invested'],
            model.add_variables(binary=True, coords=model.get_coords(['period', 'scenario'])),
        )
        assert_conequal(
            model.constraints['InvestStorage|size|ub'],
            model.variables['InvestStorage|size'] <= model.variables['InvestStorage|invested'] * 100,
        )
        assert_conequal(
            model.constraints['InvestStorage|size|lb'],
            model.variables['InvestStorage|size'] >= model.variables['InvestStorage|invested'] * 20,
        )

    def test_storage_with_final_state_constraints(self, basic_flow_system_linopy_coords, coords_config):
        """Test storage with final state constraints."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create storage with final state constraints
        storage = fx.Storage(
            'FinalStateStorage',
            charging=fx.Flow('Q_th_in', bus='Fernwärme', size=20),
            discharging=fx.Flow('Q_th_out', bus='Fernwärme', size=20),
            capacity_in_flow_hours=30,
            initial_charge_state=10,  # Start with 10 kWh
            minimal_final_charge_state=15,  # End with at least 15 kWh
            maximal_final_charge_state=25,  # End with at most 25 kWh
            eta_charge=0.9,
            eta_discharge=0.9,
            relative_loss_per_hour=0.05,
        )

        flow_system.add_elements(storage)
        model = create_linopy_model(flow_system)

        # Check final state constraints exist
        expected_constraints = {
            'FinalStateStorage|final_charge_min',
            'FinalStateStorage|final_charge_max',
        }

        for con_name in expected_constraints:
            assert con_name in model.constraints, f'Missing final state constraint: {con_name}'

        assert_conequal(
            model.constraints['FinalStateStorage|initial_charge_state'],
            model.variables['FinalStateStorage|charge_state'].isel(time=0) == 10,
        )

        # Check final state constraint formulations
        assert_conequal(
            model.constraints['FinalStateStorage|final_charge_min'],
            model.variables['FinalStateStorage|charge_state'].isel(time=-1) >= 15,
        )
        assert_conequal(
            model.constraints['FinalStateStorage|final_charge_max'],
            model.variables['FinalStateStorage|charge_state'].isel(time=-1) <= 25,
        )

    def test_storage_cyclic_initialization(self, basic_flow_system_linopy_coords, coords_config):
        """Test storage with cyclic initialization."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create storage with cyclic initialization
        storage = fx.Storage(
            'CyclicStorage',
            charging=fx.Flow('Q_th_in', bus='Fernwärme', size=20),
            discharging=fx.Flow('Q_th_out', bus='Fernwärme', size=20),
            capacity_in_flow_hours=30,
            initial_charge_state='equals_final',  # Cyclic initialization
            eta_charge=0.9,
            eta_discharge=0.9,
            relative_loss_per_hour=0.05,
        )

        flow_system.add_elements(storage)
        model = create_linopy_model(flow_system)

        # Check cyclic constraint exists
        assert 'CyclicStorage|initial_charge_state' in model.constraints, 'Missing cyclic initialization constraint'

        # Check cyclic constraint formulation
        assert_conequal(
            model.constraints['CyclicStorage|initial_charge_state'],
            model.variables['CyclicStorage|charge_state'].isel(time=0)
            == model.variables['CyclicStorage|charge_state'].isel(time=-1),
        )

    @pytest.mark.parametrize(
        'prevent_simultaneous',
        [True, False],
    )
    def test_simultaneous_charge_discharge(self, basic_flow_system_linopy_coords, coords_config, prevent_simultaneous):
        """Test prevent_simultaneous_charge_and_discharge parameter."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create storage with or without simultaneous charge/discharge prevention
        storage = fx.Storage(
            'SimultaneousStorage',
            charging=fx.Flow('Q_th_in', bus='Fernwärme', size=20),
            discharging=fx.Flow('Q_th_out', bus='Fernwärme', size=20),
            capacity_in_flow_hours=30,
            initial_charge_state=0,
            eta_charge=0.9,
            eta_discharge=0.9,
            relative_loss_per_hour=0.05,
            prevent_simultaneous_charge_and_discharge=prevent_simultaneous,
        )

        flow_system.add_elements(storage)
        model = create_linopy_model(flow_system)

        # Binary variables should exist when preventing simultaneous operation
        if prevent_simultaneous:
            binary_vars = {
                'SimultaneousStorage(Q_th_in)|status',
                'SimultaneousStorage(Q_th_out)|status',
            }
            for var_name in binary_vars:
                assert var_name in model.variables, f'Missing binary variable: {var_name}'

            # Check for constraints that enforce either charging or discharging
            constraint_name = 'SimultaneousStorage|prevent_simultaneous_use'
            assert constraint_name in model.constraints, 'Missing constraint to prevent simultaneous operation'

            assert_conequal(
                model.constraints['SimultaneousStorage|prevent_simultaneous_use'],
                model.variables['SimultaneousStorage(Q_th_in)|status']
                + model.variables['SimultaneousStorage(Q_th_out)|status']
                <= 1,
            )

    @pytest.mark.parametrize(
        'mandatory,minimum_size,expected_vars,expected_constraints',
        [
            (False, None, {'InvestStorage|invested'}, {'InvestStorage|size|lb'}),
            (False, 20, {'InvestStorage|invested'}, {'InvestStorage|size|lb'}),
            (True, None, set(), set()),
            (True, 20, set(), set()),
        ],
    )
    def test_investment_parameters(
        self,
        basic_flow_system_linopy_coords,
        coords_config,
        mandatory,
        minimum_size,
        expected_vars,
        expected_constraints,
    ):
        """Test different investment parameter combinations."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create investment parameters
        invest_params = {
            'effects_of_investment': 100,
            'effects_of_investment_per_size': 10,
            'mandatory': mandatory,
            'maximum_size': 100,
        }
        if minimum_size is not None:
            invest_params['minimum_size'] = minimum_size

        # Create storage with specified investment parameters
        storage = fx.Storage(
            'InvestStorage',
            charging=fx.Flow('Q_th_in', bus='Fernwärme', size=20),
            discharging=fx.Flow('Q_th_out', bus='Fernwärme', size=20),
            capacity_in_flow_hours=fx.InvestParameters(**invest_params),
            initial_charge_state=0,
            eta_charge=0.9,
            eta_discharge=0.9,
            relative_loss_per_hour=0.05,
        )

        flow_system.add_elements(storage)
        model = create_linopy_model(flow_system)

        # Check that expected variables exist
        for var_name in expected_vars:
            if not mandatory:  # Optional investment (mandatory=False)
                assert var_name in model.variables, f'Expected variable {var_name} not found'

        # Check that expected constraints exist
        for constraint_name in expected_constraints:
            if not mandatory:  # Optional investment (mandatory=False)
                assert constraint_name in model.constraints, f'Expected constraint {constraint_name} not found'

        # If mandatory is True, invested should be fixed to 1
        if mandatory:
            # Check that the invested variable exists and is fixed to 1
            if 'InvestStorage|invested' in model.variables:
                var = model.variables['InvestStorage|invested']
                # Check if the lower and upper bounds are both 1
                assert var.upper == 1 and var.lower == 1, 'invested variable should be fixed to 1 when mandatory=True'
