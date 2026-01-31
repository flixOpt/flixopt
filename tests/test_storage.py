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

        # Check that batched variables exist
        assert 'flow|rate' in model.variables
        assert 'storage|charge' in model.variables
        assert 'storage|netto' in model.variables

        # Check that batched constraints exist
        assert 'storage|netto_eq' in model.constraints
        assert 'storage|balance' in model.constraints
        assert 'storage|initial_charge_state' in model.constraints

        # Access batched flow rate variable and select individual flows
        flow_rate = model.variables['flow|rate']
        charge_rate = flow_rate.sel(flow='TestStorage(Q_th_in)', drop=True)
        discharge_rate = flow_rate.sel(flow='TestStorage(Q_th_out)', drop=True)

        # Access batched storage variables
        charge_state = model.variables['storage|charge'].sel(storage='TestStorage', drop=True)
        netto_discharge = model.variables['storage|netto'].sel(storage='TestStorage', drop=True)

        # Check variable properties (bounds)
        assert_var_equal(charge_rate, model.add_variables(lower=0, upper=20, coords=model.get_coords()))
        assert_var_equal(discharge_rate, model.add_variables(lower=0, upper=20, coords=model.get_coords()))
        assert_var_equal(
            charge_state,
            model.add_variables(lower=0, upper=30, coords=model.get_coords(extra_timestep=True)),
        )

        # Check constraint formulations
        # netto_discharge = discharge_rate - charge_rate
        assert_conequal(
            model.constraints['storage|netto_eq'].sel(storage='TestStorage', drop=True),
            netto_discharge == discharge_rate - charge_rate,
        )

        # Energy balance: charge_state[t+1] = charge_state[t] + charge*dt - discharge*dt
        assert_conequal(
            model.constraints['storage|balance'].sel(storage='TestStorage', drop=True),
            charge_state.isel(time=slice(1, None))
            == charge_state.isel(time=slice(None, -1))
            + charge_rate * model.timestep_duration
            - discharge_rate * model.timestep_duration,
        )

        # Check initial charge state constraint
        assert_conequal(
            model.constraints['storage|initial_charge_state'].sel(storage='TestStorage', drop=True),
            charge_state.isel(time=0) == 0,
        )

    def test_lossy_storage(self, basic_flow_system_linopy_coords, coords_config):
        """Test storage with charge/discharge efficiency and loss rate."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create a storage with efficiency and loss parameters
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

        # Check that batched variables exist
        assert 'flow|rate' in model.variables
        assert 'storage|charge' in model.variables
        assert 'storage|netto' in model.variables

        # Check that batched constraints exist
        assert 'storage|netto_eq' in model.constraints
        assert 'storage|balance' in model.constraints
        assert 'storage|initial_charge_state' in model.constraints

        # Access batched flow rate variable and select individual flows
        flow_rate = model.variables['flow|rate']
        charge_rate = flow_rate.sel(flow='TestStorage(Q_th_in)', drop=True)
        discharge_rate = flow_rate.sel(flow='TestStorage(Q_th_out)', drop=True)

        # Access batched storage variables
        charge_state = model.variables['storage|charge'].sel(storage='TestStorage', drop=True)
        netto_discharge = model.variables['storage|netto'].sel(storage='TestStorage', drop=True)

        # Check variable properties (bounds)
        assert_var_equal(charge_rate, model.add_variables(lower=0, upper=20, coords=model.get_coords()))
        assert_var_equal(discharge_rate, model.add_variables(lower=0, upper=20, coords=model.get_coords()))
        assert_var_equal(
            charge_state,
            model.add_variables(lower=0, upper=30, coords=model.get_coords(extra_timestep=True)),
        )

        # Check constraint formulations
        assert_conequal(
            model.constraints['storage|netto_eq'].sel(storage='TestStorage', drop=True),
            netto_discharge == discharge_rate - charge_rate,
        )

        rel_loss = 0.05
        timestep_duration = model.timestep_duration
        eff_charge = 0.9
        eff_discharge = 0.8

        # Energy balance with efficiency and loss:
        # charge_state[t+1] = charge_state[t] * (1-loss)^dt + charge*eta_c*dt - discharge*dt/eta_d
        assert_conequal(
            model.constraints['storage|balance'].sel(storage='TestStorage', drop=True),
            charge_state.isel(time=slice(1, None))
            == charge_state.isel(time=slice(None, -1)) * (1 - rel_loss) ** timestep_duration
            + charge_rate * eff_charge * timestep_duration
            - discharge_rate / eff_discharge * timestep_duration,
        )

        # Check initial charge state constraint
        assert_conequal(
            model.constraints['storage|initial_charge_state'].sel(storage='TestStorage', drop=True),
            charge_state.isel(time=0) == 0,
        )

    def test_charge_state_bounds(self, basic_flow_system_linopy_coords, coords_config):
        """Test storage with time-varying charge state bounds."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create a storage with time-varying relative bounds
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

        # Check that batched variables exist
        assert 'flow|rate' in model.variables
        assert 'storage|charge' in model.variables
        assert 'storage|netto' in model.variables

        # Check that batched constraints exist
        assert 'storage|netto_eq' in model.constraints
        assert 'storage|balance' in model.constraints
        assert 'storage|initial_charge_state' in model.constraints

        # Access batched flow rate variable and select individual flows
        flow_rate = model.variables['flow|rate']
        charge_rate = flow_rate.sel(flow='TestStorage(Q_th_in)', drop=True)
        discharge_rate = flow_rate.sel(flow='TestStorage(Q_th_out)', drop=True)

        # Access batched storage variables
        charge_state = model.variables['storage|charge'].sel(storage='TestStorage', drop=True)
        netto_discharge = model.variables['storage|netto'].sel(storage='TestStorage', drop=True)

        # Check variable properties (bounds) - flow rates
        assert_var_equal(charge_rate, model.add_variables(lower=0, upper=20, coords=model.get_coords()))
        assert_var_equal(discharge_rate, model.add_variables(lower=0, upper=20, coords=model.get_coords()))

        # Check variable properties - charge state with time-varying bounds
        assert_var_equal(
            charge_state,
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
            model.constraints['storage|netto_eq'].sel(storage='TestStorage', drop=True),
            netto_discharge == discharge_rate - charge_rate,
        )

        assert_conequal(
            model.constraints['storage|balance'].sel(storage='TestStorage', drop=True),
            charge_state.isel(time=slice(1, None))
            == charge_state.isel(time=slice(None, -1))
            + charge_rate * model.timestep_duration
            - discharge_rate * model.timestep_duration,
        )

        # Check initial charge state constraint
        assert_conequal(
            model.constraints['storage|initial_charge_state'].sel(storage='TestStorage', drop=True),
            charge_state.isel(time=0) == 3,
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

        # Check batched storage variables exist
        assert 'storage|charge' in model.variables
        assert 'storage|size' in model.variables
        assert 'storage|invested' in model.variables

        # Check batched investment constraints exist
        assert 'storage|size|ub' in model.constraints
        assert 'storage|size|lb' in model.constraints

        # Access batched variables and select this storage
        size = model.variables['storage|size'].sel(storage='InvestStorage', drop=True)
        invested = model.variables['storage|invested'].sel(storage='InvestStorage', drop=True)

        # Check variable properties (bounds)
        assert_var_equal(
            size,
            model.add_variables(lower=0, upper=100, coords=model.get_coords(['period', 'scenario'])),
        )
        assert_var_equal(
            invested,
            model.add_variables(binary=True, coords=model.get_coords(['period', 'scenario'])),
        )

        # Check investment constraints
        assert_conequal(
            model.constraints['storage|size|ub'].sel(storage='InvestStorage', drop=True),
            size <= invested * 100,
        )
        assert_conequal(
            model.constraints['storage|size|lb'].sel(storage='InvestStorage', drop=True),
            size >= invested * 20,
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
        assert 'storage|initial_charge_state' in model.constraints
        assert 'storage|final_charge_min' in model.constraints
        assert 'storage|final_charge_max' in model.constraints

        # Access batched storage charge state variable
        charge_state = model.variables['storage|charge'].sel(storage='FinalStateStorage', drop=True)

        # Check initial constraint
        assert_conequal(
            model.constraints['storage|initial_charge_state'].sel(storage='FinalStateStorage', drop=True),
            charge_state.isel(time=0) == 10,
        )

        # Check final state constraint formulations
        assert_conequal(
            model.constraints['storage|final_charge_min'].sel(storage='FinalStateStorage', drop=True),
            charge_state.isel(time=-1) >= 15,
        )
        assert_conequal(
            model.constraints['storage|final_charge_max'].sel(storage='FinalStateStorage', drop=True),
            charge_state.isel(time=-1) <= 25,
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

        # Check cyclic constraint exists (batched constraint name)
        assert 'storage|initial_equals_final' in model.constraints, 'Missing cyclic initialization constraint'

        # Access batched storage charge state variable
        charge_state = model.variables['storage|charge'].sel(storage='CyclicStorage', drop=True)

        # Check cyclic constraint formulation
        assert_conequal(
            model.constraints['storage|initial_equals_final'].sel(storage='CyclicStorage', drop=True),
            charge_state.isel(time=0) == charge_state.isel(time=-1),
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
            # Check batched status variable exists
            assert 'flow|status' in model.variables, 'Missing batched flow status variable'

            # Verify status variable is binary for charge/discharge flows
            status = model.variables['flow|status']
            status_charge = status.sel(flow='SimultaneousStorage(Q_th_in)', drop=True)
            status_discharge = status.sel(flow='SimultaneousStorage(Q_th_out)', drop=True)
            # Verify binary bounds
            assert float(status_charge.lower.min()) == 0
            assert float(status_charge.upper.max()) == 1
            assert float(status_discharge.lower.min()) == 0
            assert float(status_discharge.upper.max()) == 1

            # Check for batched constraint that enforces either charging or discharging
            # Constraint name is 'prevent_simultaneous' with a 'component' dimension
            assert 'storage|prevent_simultaneous' in model.constraints, (
                'Missing constraint to prevent simultaneous operation'
            )

            # Verify this storage is included in the constraint
            constraint = model.constraints['storage|prevent_simultaneous']
            assert 'SimultaneousStorage' in constraint.coords['component'].values

    @pytest.mark.parametrize(
        'mandatory,minimum_size,expected_vars,expected_constraints',
        [
            (False, None, {'storage|invested'}, {'storage|size|lb'}),
            (False, 20, {'storage|invested'}, {'storage|size|lb'}),
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

        # Check that expected batched variables exist
        for var_name in expected_vars:
            if not mandatory:  # Optional investment (mandatory=False)
                assert var_name in model.variables, f'Expected variable {var_name} not found'

        # Check that expected batched constraints exist
        for constraint_name in expected_constraints:
            if not mandatory:  # Optional investment (mandatory=False)
                assert constraint_name in model.constraints, f'Expected constraint {constraint_name} not found'

        # If mandatory is True, invested should be fixed to 1 or not present
        if mandatory:
            # For mandatory investments, there may be no 'invested' variable in the optional subset
            # or if present, it should have upper=lower=1
            if 'storage|invested' in model.variables:
                invested = model.variables['storage|invested']
                # Check if storage dimension exists and if InvestStorage is in it
                if 'storage' in invested.dims and 'InvestStorage' in invested.coords['storage'].values:
                    inv_sel = invested.sel(storage='InvestStorage')
                    # Check if the lower and upper bounds are both 1
                    assert float(inv_sel.upper.min()) == 1 and float(inv_sel.lower.min()) == 1, (
                        'invested variable should be fixed to 1 when mandatory=True'
                    )
