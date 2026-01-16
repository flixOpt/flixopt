"""Tests for the IO conversion utilities for backwards compatibility."""

import pathlib

import pytest
import xarray as xr

from flixopt.io import (
    PARAMETER_RENAMES,
    VALUE_RENAMES,
    _rename_keys_recursive,
    convert_old_dataset,
    convert_old_netcdf,
    load_dataset_from_netcdf,
    save_dataset_to_netcdf,
)


class TestRenameKeysRecursive:
    """Tests for the _rename_keys_recursive function."""

    def test_simple_key_rename(self):
        """Test basic key renaming."""
        old = {'minimum_operation': 100}
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)
        assert 'minimum_temporal' in result
        assert 'minimum_operation' not in result
        assert result['minimum_temporal'] == 100

    def test_nested_key_rename(self):
        """Test key renaming in nested structures."""
        old = {
            'components': {
                'Boiler': {
                    'on_off_parameters': {
                        'on_hours_total_max': 50,
                    }
                }
            }
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)
        assert 'status_parameters' in result['components']['Boiler']
        assert 'on_off_parameters' not in result['components']['Boiler']
        assert result['components']['Boiler']['status_parameters']['on_hours_max'] == 50

    def test_class_name_rename(self):
        """Test that __class__ values are also renamed."""
        old = {
            '__class__': 'OnOffParameters',
            'on_hours_total_max': 100,
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)
        assert result['__class__'] == 'StatusParameters'
        assert result['on_hours_max'] == 100

    def test_value_rename(self):
        """Test value renaming for specific keys."""
        old = {'initial_charge_state': 'lastValueOfSim'}
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)
        assert result['initial_charge_state'] == 'equals_final'

    def test_list_handling(self):
        """Test that lists are processed correctly."""
        old = {
            'flows': [
                {'flow_hours_total_max': 100},
                {'flow_hours_total_min': 50},
            ]
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)
        assert result['flows'][0]['flow_hours_max'] == 100
        assert result['flows'][1]['flow_hours_min'] == 50

    def test_unchanged_keys_preserved(self):
        """Test that keys not in rename map are preserved."""
        old = {'label': 'MyComponent', 'size': 100}
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)
        assert result['label'] == 'MyComponent'
        assert result['size'] == 100

    def test_empty_dict(self):
        """Test handling of empty dict."""
        result = _rename_keys_recursive({}, PARAMETER_RENAMES, VALUE_RENAMES)
        assert result == {}

    def test_empty_list(self):
        """Test handling of empty list."""
        result = _rename_keys_recursive([], PARAMETER_RENAMES, VALUE_RENAMES)
        assert result == []

    def test_scalar_values(self):
        """Test that scalar values are returned unchanged."""
        assert _rename_keys_recursive(42, PARAMETER_RENAMES, VALUE_RENAMES) == 42
        assert _rename_keys_recursive('string', PARAMETER_RENAMES, VALUE_RENAMES) == 'string'
        assert _rename_keys_recursive(None, PARAMETER_RENAMES, VALUE_RENAMES) is None


class TestParameterRenames:
    """Tests to verify all expected parameter renames are in the mapping."""

    def test_effect_parameters(self):
        """Test Effect parameter renames are defined."""
        assert PARAMETER_RENAMES['minimum_operation'] == 'minimum_temporal'
        assert PARAMETER_RENAMES['maximum_operation'] == 'maximum_temporal'
        assert PARAMETER_RENAMES['minimum_invest'] == 'minimum_periodic'
        assert PARAMETER_RENAMES['maximum_invest'] == 'maximum_periodic'
        assert PARAMETER_RENAMES['minimum_investment'] == 'minimum_periodic'
        assert PARAMETER_RENAMES['maximum_investment'] == 'maximum_periodic'
        assert PARAMETER_RENAMES['minimum_operation_per_hour'] == 'minimum_per_hour'
        assert PARAMETER_RENAMES['maximum_operation_per_hour'] == 'maximum_per_hour'

    def test_invest_parameters(self):
        """Test InvestParameters renames are defined."""
        assert PARAMETER_RENAMES['fix_effects'] == 'effects_of_investment'
        assert PARAMETER_RENAMES['specific_effects'] == 'effects_of_investment_per_size'
        assert PARAMETER_RENAMES['divest_effects'] == 'effects_of_retirement'
        assert PARAMETER_RENAMES['piecewise_effects'] == 'piecewise_effects_of_investment'

    def test_flow_parameters(self):
        """Test Flow/OnOffParameters renames are defined."""
        assert PARAMETER_RENAMES['flow_hours_total_max'] == 'flow_hours_max'
        assert PARAMETER_RENAMES['flow_hours_total_min'] == 'flow_hours_min'
        assert PARAMETER_RENAMES['on_hours_total_max'] == 'on_hours_max'
        assert PARAMETER_RENAMES['on_hours_total_min'] == 'on_hours_min'
        assert PARAMETER_RENAMES['switch_on_total_max'] == 'switch_on_max'

    def test_bus_parameters(self):
        """Test Bus parameter renames are defined."""
        assert PARAMETER_RENAMES['excess_penalty_per_flow_hour'] == 'imbalance_penalty_per_flow_hour'

    def test_component_parameters(self):
        """Test component parameter renames are defined."""
        assert PARAMETER_RENAMES['source'] == 'outputs'
        assert PARAMETER_RENAMES['sink'] == 'inputs'
        assert PARAMETER_RENAMES['prevent_simultaneous_sink_and_source'] == 'prevent_simultaneous_flow_rates'

    def test_linear_converter_parameters(self):
        """Test linear converter parameter renames are defined."""
        assert PARAMETER_RENAMES['Q_fu'] == 'fuel_flow'
        assert PARAMETER_RENAMES['P_el'] == 'electrical_flow'
        assert PARAMETER_RENAMES['Q_th'] == 'thermal_flow'
        assert PARAMETER_RENAMES['Q_ab'] == 'heat_source_flow'
        assert PARAMETER_RENAMES['eta'] == 'thermal_efficiency'
        assert PARAMETER_RENAMES['eta_th'] == 'thermal_efficiency'
        assert PARAMETER_RENAMES['eta_el'] == 'electrical_efficiency'
        assert PARAMETER_RENAMES['COP'] == 'cop'

    def test_class_renames(self):
        """Test class name renames are defined."""
        assert PARAMETER_RENAMES['OnOffParameters'] == 'StatusParameters'
        assert PARAMETER_RENAMES['on_off_parameters'] == 'status_parameters'
        assert PARAMETER_RENAMES['FullCalculation'] == 'Optimization'
        assert PARAMETER_RENAMES['AggregatedCalculation'] == 'ClusteredOptimization'
        assert PARAMETER_RENAMES['SegmentedCalculation'] == 'SegmentedOptimization'
        assert PARAMETER_RENAMES['CalculationResults'] == 'Results'
        assert PARAMETER_RENAMES['AggregationParameters'] == 'ClusteringParameters'

    def test_time_series_data_parameters(self):
        """Test TimeSeriesData parameter renames are defined."""
        assert PARAMETER_RENAMES['agg_group'] == 'aggregation_group'
        assert PARAMETER_RENAMES['agg_weight'] == 'aggregation_weight'


class TestValueRenames:
    """Tests for value renaming."""

    def test_initial_charge_state_value(self):
        """Test initial_charge_state value rename is defined."""
        assert VALUE_RENAMES['initial_charge_state']['lastValueOfSim'] == 'equals_final'


class TestConvertOldDataset:
    """Tests for convert_old_dataset function."""

    def test_converts_attrs(self):
        """Test that dataset attrs are converted."""
        ds = xr.Dataset(attrs={'minimum_operation': 100, 'maximum_invest': 500})
        result = convert_old_dataset(ds)
        assert 'minimum_temporal' in result.attrs
        assert 'maximum_periodic' in result.attrs
        assert 'minimum_operation' not in result.attrs
        assert 'maximum_invest' not in result.attrs

    def test_nested_attrs_conversion(self):
        """Test conversion of nested attrs structures."""
        ds = xr.Dataset(
            attrs={
                'components': {
                    'Boiler': {
                        '__class__': 'OnOffParameters',
                        'on_hours_total_max': 100,
                    }
                }
            }
        )
        result = convert_old_dataset(ds)
        assert result.attrs['components']['Boiler']['__class__'] == 'StatusParameters'
        assert result.attrs['components']['Boiler']['on_hours_max'] == 100

    def test_custom_renames(self):
        """Test that custom renames can be provided."""
        ds = xr.Dataset(attrs={'custom_old': 'value'})
        result = convert_old_dataset(ds, key_renames={'custom_old': 'custom_new'}, value_renames={})
        assert 'custom_new' in result.attrs
        assert 'custom_old' not in result.attrs

    def test_returns_equivalent_dataset(self):
        """Test that the function converts and returns equivalent dataset."""
        ds = xr.Dataset(attrs={'minimum_operation': 100})
        result = convert_old_dataset(ds)
        # Check that attrs are converted
        assert result.attrs == {'minimum_temporal': 100}


class TestConvertOldNetcdf:
    """Tests for convert_old_netcdf function."""

    def test_load_and_convert(self, tmp_path):
        """Test loading and converting a netCDF file."""
        # Create an old-style dataset and save it
        old_ds = xr.Dataset(
            {'var1': (['time'], [1, 2, 3])},
            coords={'time': [0, 1, 2]},
            attrs={
                'components': {
                    'Boiler': {
                        '__class__': 'OnOffParameters',
                        'on_hours_total_max': 100,
                    }
                }
            },
        )
        input_path = tmp_path / 'old_system.nc'
        save_dataset_to_netcdf(old_ds, input_path)

        # Convert
        result = convert_old_netcdf(input_path)

        # Verify conversion
        assert result.attrs['components']['Boiler']['__class__'] == 'StatusParameters'
        assert result.attrs['components']['Boiler']['on_hours_max'] == 100

    def test_load_convert_and_save(self, tmp_path):
        """Test loading, converting, and saving to new file."""
        # Create an old-style dataset and save it
        old_ds = xr.Dataset(
            {'var1': (['time'], [1, 2, 3])},
            coords={'time': [0, 1, 2]},
            attrs={'minimum_operation': 100},
        )
        input_path = tmp_path / 'old_system.nc'
        output_path = tmp_path / 'new_system.nc'
        save_dataset_to_netcdf(old_ds, input_path)

        # Convert and save
        convert_old_netcdf(input_path, output_path)

        # Load the new file and verify
        loaded = load_dataset_from_netcdf(output_path)
        assert 'minimum_temporal' in loaded.attrs
        assert loaded.attrs['minimum_temporal'] == 100


class TestFullConversionScenario:
    """Integration tests for full conversion scenarios."""

    def test_complex_flowsystem_structure(self):
        """Test conversion of a complex FlowSystem-like structure."""
        old_structure = {
            '__class__': 'FlowSystem',
            'components': {
                'Boiler': {
                    '__class__': 'LinearConverter',
                    'Q_fu': ':::Boiler|fuel',
                    'eta': 0.9,
                    'on_off_parameters': {
                        '__class__': 'OnOffParameters',
                        'on_hours_total_max': 100,
                        'switch_on_total_max': 10,
                    },
                },
                'HeatPump': {
                    '__class__': 'HeatPumpWithSource',
                    'COP': 3.5,
                    'Q_ab': ':::HeatPump|ambient',
                },
                'Battery': {
                    '__class__': 'Storage',
                    'initial_charge_state': 'lastValueOfSim',
                },
                'Grid': {
                    '__class__': 'Source',
                    'source': [{'__class__': 'Flow', 'flow_hours_total_max': 1000}],
                },
                'Demand': {
                    '__class__': 'Sink',
                    'sink': [{'__class__': 'Flow', 'flow_hours_total_min': 500}],
                },
            },
            'effects': {
                'costs': {
                    '__class__': 'Effect',
                    'minimum_operation': 0,
                    'maximum_invest': 1000,
                    'minimum_operation_per_hour': 0,
                },
            },
            'buses': {
                'heat_bus': {
                    '__class__': 'Bus',
                    'excess_penalty_per_flow_hour': 1000,
                },
            },
        }

        result = _rename_keys_recursive(old_structure, PARAMETER_RENAMES, VALUE_RENAMES)

        # Verify component conversions
        boiler = result['components']['Boiler']
        assert boiler['fuel_flow'] == ':::Boiler|fuel'
        assert boiler['thermal_efficiency'] == 0.9
        assert boiler['status_parameters']['__class__'] == 'StatusParameters'
        assert boiler['status_parameters']['on_hours_max'] == 100
        assert boiler['status_parameters']['switch_on_max'] == 10

        heat_pump = result['components']['HeatPump']
        assert heat_pump['cop'] == 3.5
        assert heat_pump['heat_source_flow'] == ':::HeatPump|ambient'

        battery = result['components']['Battery']
        assert battery['initial_charge_state'] == 'equals_final'

        grid = result['components']['Grid']
        assert 'outputs' in grid
        assert grid['outputs'][0]['flow_hours_max'] == 1000

        demand = result['components']['Demand']
        assert 'inputs' in demand
        assert demand['inputs'][0]['flow_hours_min'] == 500

        # Verify effect conversions
        costs = result['effects']['costs']
        assert costs['minimum_temporal'] == 0
        assert costs['maximum_periodic'] == 1000
        assert costs['minimum_per_hour'] == 0

        # Verify bus conversions
        heat_bus = result['buses']['heat_bus']
        assert heat_bus['imbalance_penalty_per_flow_hour'] == 1000

    def test_invest_parameters_conversion(self):
        """Test conversion of InvestParameters."""
        old_structure = {
            '__class__': 'InvestParameters',
            'fix_effects': {'costs': 1000},
            'specific_effects': {'costs': 100},
            'divest_effects': {'costs': 500},
            'piecewise_effects': {'__class__': 'PiecewiseEffects'},
        }

        result = _rename_keys_recursive(old_structure, PARAMETER_RENAMES, VALUE_RENAMES)

        assert result['effects_of_investment'] == {'costs': 1000}
        assert result['effects_of_investment_per_size'] == {'costs': 100}
        assert result['effects_of_retirement'] == {'costs': 500}
        assert result['piecewise_effects_of_investment']['__class__'] == 'PiecewiseEffects'


class TestEdgeCases:
    """Tests for edge cases and potential issues."""

    def test_effect_dict_keys_not_renamed(self):
        """Effect dict keys are effect labels, not parameter names - should NOT be renamed."""
        old = {
            'effects_per_flow_hour': {'costs': 100, 'CO2': 50},
            'fix_effects': {'costs': 1000},  # key should be renamed, but 'costs' value key should not
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)

        # 'costs' and 'CO2' are effect labels, not parameter names
        assert result['effects_per_flow_hour'] == {'costs': 100, 'CO2': 50}
        # 'fix_effects' key should be renamed to 'effects_of_investment'
        assert 'effects_of_investment' in result
        # But the nested 'costs' key should remain (it's an effect label)
        assert result['effects_of_investment'] == {'costs': 1000}

    def test_deeply_nested_structure(self):
        """Test handling of deeply nested structures (5+ levels)."""
        old = {
            'level1': {
                'level2': {
                    'level3': {
                        'level4': {
                            'level5': {
                                'on_hours_total_max': 100,
                            }
                        }
                    }
                }
            }
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)
        assert result['level1']['level2']['level3']['level4']['level5']['on_hours_max'] == 100

    def test_mixed_old_and_new_parameters(self):
        """Test structure with both old and new parameter names."""
        old = {
            'minimum_operation': 0,  # old
            'minimum_temporal': 10,  # new (should not be double-renamed)
            'maximum_periodic': 1000,  # new
            'maximum_invest': 500,  # old
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)

        # Old should be renamed
        assert 'minimum_temporal' in result
        assert 'maximum_periodic' in result

        # Values should be correct (old one gets overwritten if both exist)
        # This is a potential issue - if both old and new exist, new gets overwritten
        # In practice this shouldn't happen, but let's document the behavior
        assert result['minimum_temporal'] == 10  # new value preserved (processed second)
        assert result['maximum_periodic'] in [500, 1000]  # either could win

    def test_none_values_preserved(self):
        """Test that None values are preserved."""
        old = {
            'minimum_operation': None,
            'some_param': None,
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)
        assert result['minimum_temporal'] is None
        assert result['some_param'] is None

    def test_boolean_values_preserved(self):
        """Test that boolean values are preserved."""
        old = {
            'mandatory': True,
            'is_standard': False,
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)
        assert result['mandatory'] is True
        assert result['is_standard'] is False

    def test_numeric_edge_cases(self):
        """Test numeric edge cases (0, negative, floats)."""
        old = {
            'minimum_operation': 0,
            'maximum_operation': -100,  # negative (unusual but possible)
            'eta': 0.95,
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)
        assert result['minimum_temporal'] == 0
        assert result['maximum_temporal'] == -100
        assert result['thermal_efficiency'] == 0.95

    def test_dataarray_reference_strings_preserved(self):
        """Test that DataArray reference strings are preserved as-is.

        Note: We don't rename inside reference strings like ':::Boiler|Q_fu'
        because those reference the actual DataArray variable names, which
        would need separate handling if they also need renaming.
        """
        old = {
            'Q_fu': ':::Boiler|Q_fu',  # key renamed, but ref string preserved
            'eta': ':::Boiler|eta',
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)

        # Keys should be renamed
        assert 'fuel_flow' in result
        assert 'thermal_efficiency' in result

        # Reference strings should be preserved (they point to DataArray names)
        assert result['fuel_flow'] == ':::Boiler|Q_fu'
        assert result['thermal_efficiency'] == ':::Boiler|eta'

    def test_list_of_dicts(self):
        """Test conversion of lists containing dictionaries."""
        old = {
            'flows': [
                {
                    '__class__': 'Flow',
                    'on_off_parameters': {'__class__': 'OnOffParameters'},
                    'flow_hours_total_max': 100,
                },
                {
                    '__class__': 'Flow',
                    'flow_hours_total_min': 50,
                },
            ]
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)

        assert len(result['flows']) == 2
        assert result['flows'][0]['status_parameters']['__class__'] == 'StatusParameters'
        assert result['flows'][0]['flow_hours_max'] == 100
        assert result['flows'][1]['flow_hours_min'] == 50

    def test_special_characters_in_labels(self):
        """Test that special characters in component labels are preserved."""
        old = {
            'components': {
                'CHP_Unit-1': {
                    '__class__': 'CHP',
                    'eta_th': 0.4,
                },
                'Heat Pump (Main)': {
                    '__class__': 'HeatPump',
                    'COP': 3.5,
                },
            }
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)

        # Labels should be preserved exactly
        assert 'CHP_Unit-1' in result['components']
        assert 'Heat Pump (Main)' in result['components']

        # Parameters should still be renamed
        assert result['components']['CHP_Unit-1']['thermal_efficiency'] == 0.4
        assert result['components']['Heat Pump (Main)']['cop'] == 3.5

    def test_value_rename_only_for_specific_keys(self):
        """Test that value renames only apply to specific keys."""
        old = {
            'initial_charge_state': 'lastValueOfSim',  # should be renamed
            'other_param': 'lastValueOfSim',  # should NOT be renamed (different key)
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)

        assert result['initial_charge_state'] == 'equals_final'
        assert result['other_param'] == 'lastValueOfSim'  # unchanged

    def test_value_rename_with_non_string_value(self):
        """Test that value renames don't break with non-string values."""
        old = {
            'initial_charge_state': 0.5,  # numeric, not string
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)

        # Should be preserved as-is (value rename only applies to strings)
        assert result['initial_charge_state'] == 0.5


class TestRealWorldScenarios:
    """Tests with real-world-like data structures."""

    def test_source_with_investment(self):
        """Test Source component with investment parameters."""
        old = {
            '__class__': 'Source',
            'label': 'GasGrid',
            'source': [
                {
                    '__class__': 'Flow',
                    'label': 'gas',
                    'bus': 'gas_bus',
                    'flow_hours_total_max': 10000,
                    'invest_parameters': {
                        '__class__': 'InvestParameters',
                        'fix_effects': {'costs': 5000},
                        'specific_effects': {'costs': 100},
                    },
                }
            ],
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)

        assert 'outputs' in result
        assert result['outputs'][0]['flow_hours_max'] == 10000
        assert result['outputs'][0]['invest_parameters']['effects_of_investment'] == {'costs': 5000}
        assert result['outputs'][0]['invest_parameters']['effects_of_investment_per_size'] == {'costs': 100}

    def test_storage_with_all_old_parameters(self):
        """Test Storage component with various old parameters."""
        old = {
            '__class__': 'Storage',
            'label': 'Battery',
            'initial_charge_state': 'lastValueOfSim',
            'charging': {
                '__class__': 'Flow',
                'on_off_parameters': {
                    '__class__': 'OnOffParameters',
                    'on_hours_total_max': 100,
                    'on_hours_total_min': 10,
                    'switch_on_total_max': 50,
                },
            },
            'discharging': {
                '__class__': 'Flow',
                'flow_hours_total_max': 500,
            },
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)

        assert result['initial_charge_state'] == 'equals_final'
        assert result['charging']['status_parameters']['on_hours_max'] == 100
        assert result['charging']['status_parameters']['on_hours_min'] == 10
        assert result['charging']['status_parameters']['switch_on_max'] == 50
        assert result['discharging']['flow_hours_max'] == 500

    def test_effect_with_all_old_parameters(self):
        """Test Effect with all old parameter names."""
        old = {
            '__class__': 'Effect',
            'label': 'costs',
            'unit': '€',
            'minimum_operation': 0,
            'maximum_operation': 1000000,
            'minimum_invest': 0,
            'maximum_invest': 500000,
            'minimum_operation_per_hour': 0,
            'maximum_operation_per_hour': 10000,
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)

        assert result['minimum_temporal'] == 0
        assert result['maximum_temporal'] == 1000000
        assert result['minimum_periodic'] == 0
        assert result['maximum_periodic'] == 500000
        assert result['minimum_per_hour'] == 0
        assert result['maximum_per_hour'] == 10000

        # Labels should be preserved
        assert result['label'] == 'costs'
        assert result['unit'] == '€'


class TestFlowSystemFromOldResults:
    """Tests for FlowSystem.from_old_results() method."""

    def test_load_old_results_from_resources(self):
        """Test loading old results files from test resources."""
        import pathlib

        import flixopt as fx

        resources_path = pathlib.Path(__file__).parent / 'ressources'

        # Load old results using new method
        fs = fx.FlowSystem.from_old_results(resources_path, 'Sim1')

        # Verify FlowSystem was loaded
        assert fs is not None
        assert fs.name == 'Sim1'

        # Verify solution was attached
        assert fs.solution is not None
        assert len(fs.solution.data_vars) > 0

    def test_old_results_can_be_saved_new_format(self, tmp_path):
        """Test that old results can be saved in new single-file format."""
        import pathlib

        import flixopt as fx

        resources_path = pathlib.Path(__file__).parent / 'ressources'

        # Load old results
        fs = fx.FlowSystem.from_old_results(resources_path, 'Sim1')

        # Save in new format
        new_path = tmp_path / 'migrated.nc'
        fs.to_netcdf(new_path)

        # Verify the new file exists and can be loaded
        assert new_path.exists()
        loaded = fx.FlowSystem.from_netcdf(new_path)
        assert loaded is not None
        assert loaded.solution is not None


class TestV4APIConversion:
    """Tests for converting v4 API result files to the new format."""

    V4_API_PATH = pathlib.Path(__file__).parent / 'ressources' / 'v4-api'

    # All result names in the v4-api folder
    V4_RESULT_NAMES = [
        '00_minimal',
        '01_simple',
        '02_complex',
        '04_scenarios',
        'io_flow_system_base',
        'io_flow_system_long',
        'io_flow_system_segments',
        'io_simple_flow_system',
        'io_simple_flow_system_scenarios',
    ]

    @pytest.mark.parametrize('result_name', V4_RESULT_NAMES)
    def test_v4_results_can_be_loaded(self, result_name):
        """Test that v4 API results can be loaded."""
        import flixopt as fx

        fs = fx.FlowSystem.from_old_results(self.V4_API_PATH, result_name)

        # Verify FlowSystem was loaded
        assert fs is not None
        assert fs.name == result_name

        # Verify solution was attached
        assert fs.solution is not None
        assert len(fs.solution.data_vars) > 0

        # Verify we have components
        assert len(fs.components) > 0

    @pytest.mark.parametrize('result_name', V4_RESULT_NAMES)
    def test_v4_results_can_be_saved_and_reloaded(self, result_name, tmp_path):
        """Test that v4 API results can be saved in new format and reloaded."""
        import flixopt as fx

        # Load old results
        fs = fx.FlowSystem.from_old_results(self.V4_API_PATH, result_name)

        # Save in new format
        new_path = tmp_path / f'{result_name}_migrated.nc'
        fs.to_netcdf(new_path)

        # Reload and verify
        loaded = fx.FlowSystem.from_netcdf(new_path)
        assert loaded is not None
        assert loaded.solution is not None
        assert len(loaded.solution.data_vars) == len(fs.solution.data_vars)
        assert len(loaded.components) == len(fs.components)

    @pytest.mark.parametrize('result_name', V4_RESULT_NAMES)
    def test_v4_solution_variables_accessible(self, result_name):
        """Test that solution variables from v4 results are accessible."""
        import flixopt as fx

        fs = fx.FlowSystem.from_old_results(self.V4_API_PATH, result_name)

        # Check that we can access solution variables
        for var_name in list(fs.solution.data_vars)[:5]:  # Check first 5 variables
            var = fs.solution[var_name]
            assert var is not None
            # Variables should have data
            assert var.size > 0

    @pytest.mark.parametrize('result_name', V4_RESULT_NAMES)
    def test_v4_reoptimized_objective_matches_original(self, result_name):
        """Test that re-solving the migrated FlowSystem gives the same objective effect."""
        import flixopt as fx

        # Load old results
        fs = fx.FlowSystem.from_old_results(self.V4_API_PATH, result_name)

        # Get the objective effect label
        objective_effect_label = fs.effects.objective_effect.label

        # Get the original effect total from the old solution (sum for multi-scenario)
        old_effect_total = float(fs.solution[objective_effect_label].values.sum())
        old_objective = float(fs.solution['objective'].values.sum())

        # Re-solve the FlowSystem
        fs.optimize(fx.solvers.HighsSolver(mip_gap=0))

        # Get new objective effect total (sum for multi-scenario)
        new_objective = float(fs.solution['objective'].item())
        new_effect_total = float(fs.solution[objective_effect_label].sum().item())

        # Skip comparison for scenarios test case - scenario weights are now always normalized,
        # which changes the objective value when loading old results with non-normalized weights
        if result_name == '04_scenarios':
            pytest.skip('Scenario weights are now always normalized - old results have different weights')

        # Verify objective matches (within tolerance)
        assert new_objective == pytest.approx(old_objective, rel=1e-5, abs=1), (
            f'Objective mismatch for {result_name}: new={new_objective}, old={old_objective}'
        )

        assert new_effect_total == pytest.approx(old_effect_total, rel=1e-5, abs=1), (
            f'Effect {objective_effect_label} mismatch for {result_name}: '
            f'new={new_effect_total}, old={old_effect_total}'
        )
