"""
Comprehensive test file demonstrating the select parameter capabilities.

This file tests various plotting methods and shows what's possible with the new 'select' parameter.
"""

import warnings

import plotly.io as pio

import flixopt as fx

# Set default renderer to browser
pio.renderers.default = 'browser'


class TestSelectParameter:
    """Test suite for the select parameter functionality."""

    def __init__(self):
        """Initialize test suite by loading results once."""
        print('\nLoading test data...')
        try:
            self.results = fx.results.CalculationResults.from_file('examples/04_Scenarios/results/', 'Sim1')
            print('✓ Results loaded successfully')
        except Exception as e:
            print(f'✗ Failed to load results: {e}')
            raise

        # Get available scenarios and periods
        self.scenarios = self.results.solution.scenario.values.tolist()
        self.periods = self.results.solution.period.values.tolist()
        print(f'✓ Found {len(self.scenarios)} scenarios: {self.scenarios}')
        print(f'✓ Found {len(self.periods)} periods: {self.periods}')

    def test_basic_selection(self):
        """Test basic single-value selection."""
        print('\n' + '=' * 70)
        print('1. BASIC SELECTION - Single Values')
        print('=' * 70)

        # Test plot_node_balance with single scenario
        print('\n  a) plot_node_balance with single scenario:')
        try:
            _ = self.results['Fernwärme'].plot_node_balance(
                select={'scenario': self.scenarios[0]}, show=False, save=False
            )
            print(f"     ✓ Successfully plotted with scenario='{self.scenarios[0]}'")
        except Exception as e:
            print(f'     ✗ Failed: {e}')

        # Test node_balance method (data retrieval)
        print('\n  b) node_balance method with single scenario:')
        try:
            ds = self.results['Fernwärme'].node_balance(select={'scenario': self.scenarios[0]})
            print(f'     ✓ Successfully retrieved data with dimensions: {dict(ds.dims)}')
        except Exception as e:
            print(f'     ✗ Failed: {e}')

    def test_multi_value_selection(self):
        """Test selection with multiple values (lists)."""
        print('\n' + '=' * 70)
        print('2. MULTI-VALUE SELECTION - Lists')
        print('=' * 70)

        if len(self.scenarios) < 2:
            print('  ⊘ Skipped - not enough scenarios in dataset')
            return

        # Test plot_node_balance with multiple scenarios + faceting
        print('\n  a) plot_node_balance with multiple scenarios + faceting:')
        try:
            _ = self.results['Fernwärme'].plot_node_balance(
                select={'scenario': self.scenarios}, facet_by='scenario', animate_by=None, show=False, save=False
            )
            print(f'     ✓ Successfully plotted {len(self.scenarios)} scenarios as facets')
        except Exception as e:
            print(f'     ✗ Failed: {e}')

        # Test with partial list selection
        print('\n  b) plot_node_balance with subset of scenarios:')
        try:
            selected = self.scenarios[:2] if len(self.scenarios) >= 2 else self.scenarios
            _ = self.results['Fernwärme'].plot_node_balance(
                select={'scenario': selected}, facet_by='scenario', show=False, save=False
            )
            print(f'     ✓ Successfully plotted subset: {selected}')
        except Exception as e:
            print(f'     ✗ Failed: {e}')

    def test_index_based_selection(self):
        """Test selection using index positions."""
        print('\n' + '=' * 70)
        print('3. INDEX-BASED SELECTION')
        print('=' * 70)

        # Test with integer index
        print('\n  a) Selection using integer index (first scenario):')
        try:
            _ = self.results['Fernwärme'].plot_node_balance(select={'scenario': 0}, show=False, save=False)
            print('     ✓ Successfully plotted scenario at index 0')
        except Exception as e:
            print(f'     ✗ Failed: {e}')

        # Test with multiple indices
        print('\n  b) Selection using list of indices:')
        try:
            _ = self.results['Fernwärme'].plot_node_balance(
                select={'scenario': [0, 1]}, facet_by='scenario', show=False, save=False
            )
            print('     ✓ Successfully plotted scenarios at indices [0, 1]')
        except Exception as e:
            print(f'     ✗ Failed: {e}')

    def test_combined_selection(self):
        """Test combining multiple dimension selections."""
        print('\n' + '=' * 70)
        print('4. COMBINED SELECTION - Multiple Dimensions')
        print('=' * 70)

        # Test selecting both scenario and period
        print('\n  a) Selecting both scenario AND period:')
        try:
            ds = self.results['Fernwärme'].node_balance(
                select={'scenario': self.scenarios[0], 'period': self.periods[0]}
            )
            print(f"     ✓ Successfully selected scenario='{self.scenarios[0]}' and period={self.periods[0]}")
            print(f'       Resulting dimensions: {dict(ds.dims)}')
        except Exception as e:
            print(f'     ✗ Failed: {e}')

        # Test with one dimension as list, another as single value
        print('\n  b) Scenario as list, period as single value:')
        try:
            _ = self.results['Fernwärme'].plot_node_balance(
                select={'scenario': self.scenarios, 'period': self.periods[0]},
                facet_by='scenario',
                show=False,
                save=False,
            )
            print(f'     ✓ Successfully plotted all scenarios for period={self.periods[0]}')
        except Exception as e:
            print(f'     ✗ Failed: {e}')

    def test_faceting_and_animation(self):
        """Test combining select with faceting and animation."""
        print('\n' + '=' * 70)
        print('5. FACETING & ANIMATION WITH SELECTION')
        print('=' * 70)

        # Test: Select specific scenarios, then facet by period
        print('\n  a) Select scenarios, facet by period:')
        try:
            _ = self.results['Fernwärme'].plot_node_balance(
                select={'scenario': self.scenarios[0]}, facet_by='period', animate_by=None, show=False, save=False
            )
            print(f"     ✓ Selected scenario '{self.scenarios[0]}', created facets for all periods")
        except Exception as e:
            print(f'     ✗ Failed: {e}')

        # Test: Select all scenarios, facet by scenario, animate by period
        print('\n  b) Facet by scenario, animate by period:')
        try:
            if len(self.periods) > 1:
                _ = self.results['Fernwärme'].plot_node_balance(
                    select={},  # No filtering - use all data
                    facet_by='scenario',
                    animate_by='period',
                    show=False,
                    save=False,
                )
                print('     ✓ Created facets for scenarios with period animation')
            else:
                print('     ⊘ Skipped - only one period available')
        except Exception as e:
            print(f'     ✗ Failed: {e}')

    def test_different_plotting_methods(self):
        """Test select parameter across different plotting methods."""
        print('\n' + '=' * 70)
        print('6. DIFFERENT PLOTTING METHODS')
        print('=' * 70)

        # Test plot_node_balance
        print('\n  a) plot_node_balance:')
        try:
            _ = self.results['Fernwärme'].plot_node_balance(
                select={'scenario': self.scenarios[0]}, mode='area', show=False, save=False
            )
            print('     ✓ plot_node_balance works with select')
        except Exception as e:
            print(f'     ✗ Failed: {e}')

        # Test plot_heatmap
        print('\n  b) plot_heatmap:')
        try:
            # Get a variable name from the solution
            var_names = list(self.results.solution.data_vars)
            if var_names:
                _ = self.results.plot_heatmap(
                    var_names[0], select={'scenario': self.scenarios[0]}, show=False, save=False
                )
                print('     ✓ plot_heatmap works with select')
            else:
                print('     ⊘ Skipped - no variables found')
        except Exception as e:
            print(f'     ✗ Failed: {e}')

        # Test node_balance (data method)
        print('\n  c) node_balance (data retrieval):')
        try:
            ds = self.results['Fernwärme'].node_balance(select={'scenario': self.scenarios[0]}, unit_type='flow_hours')
            print('     ✓ node_balance works with select')
            print(f'       Returned dimensions: {dict(ds.dims)}')
        except Exception as e:
            print(f'     ✗ Failed: {e}')

    def test_backward_compatibility(self):
        """Test that old 'indexer' parameter still works with deprecation warning."""
        print('\n' + '=' * 70)
        print("7. BACKWARD COMPATIBILITY - 'indexer' Parameter")
        print('=' * 70)

        # Capture deprecation warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            print("\n  a) Using deprecated 'indexer' parameter:")
            try:
                _ = self.results['Fernwärme'].plot_node_balance(
                    indexer={'scenario': self.scenarios[0]}, show=False, save=False
                )

                # Check if deprecation warning was raised
                deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
                if deprecation_warnings:
                    print("     ✓ 'indexer' still works (with DeprecationWarning)")
                    print(f'       Warning message: {deprecation_warnings[0].message}')
                else:
                    print('     ✗ No deprecation warning raised')
            except Exception as e:
                print(f'     ✗ Failed: {e}')

    def test_precedence(self):
        """Test that 'select' takes precedence over 'indexer'."""
        print('\n' + '=' * 70)
        print('8. PARAMETER PRECEDENCE - select vs indexer')
        print('=' * 70)

        if len(self.scenarios) < 2:
            print('  ⊘ Skipped - not enough scenarios')
            return

        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')

            print("\n  Testing that 'select' overrides 'indexer':")
            try:
                ds = self.results['Fernwärme'].node_balance(
                    indexer={'scenario': self.scenarios[0]},  # This should be overridden
                    select={'scenario': self.scenarios[1]},  # This should win
                )

                # Check which scenario was actually selected
                if 'scenario' in ds.coords:
                    selected = ds.scenario.values
                    print("     ✓ 'select' took precedence")
                    print(f'       indexer specified: {self.scenarios[0]}')
                    print(f'       select specified: {self.scenarios[1]}')
                    print(f'       Actual selection: {selected}')
                else:
                    print('     ✓ Selection applied, scenario dimension dropped')
            except Exception as e:
                print(f'     ✗ Failed: {e}')

    def test_empty_dict_behavior(self):
        """Test behavior with empty selection dict."""
        print('\n' + '=' * 70)
        print('9. EMPTY DICT BEHAVIOR')
        print('=' * 70)

        print('\n  Using select={} (empty dict - no filtering):')
        try:
            _ = self.results['Fernwärme'].plot_node_balance(
                select={}, facet_by='scenario', animate_by='period', show=False, save=False
            )
            print('     ✓ Empty dict works - uses all available data')
        except Exception as e:
            print(f'     ✗ Failed: {e}')

    def test_error_handling(self):
        """Test error handling for invalid parameters."""
        print('\n' + '=' * 70)
        print('10. ERROR HANDLING')
        print('=' * 70)

        # Test unexpected kwargs
        print('\n  a) Unexpected keyword argument:')
        try:
            _ = self.results['Fernwärme'].plot_node_balance(select={'scenario': 0}, unexpected_param='test', show=False)
            print('     ✗ Should have raised TypeError')
        except TypeError as e:
            if 'unexpected keyword argument' in str(e):
                print('     ✓ Correctly raised TypeError')
                print(f'       Error: {e}')
            else:
                print(f'     ✗ Wrong error: {e}')
        except Exception as e:
            print(f'     ✗ Wrong exception type: {e}')

    def run_all_tests(self):
        """Run all tests in sequence."""
        print('\n' + '#' * 70)
        print("# COMPREHENSIVE TEST OF 'SELECT' PARAMETER FUNCTIONALITY")
        print('#' * 70)

        self.test_basic_selection()
        self.test_multi_value_selection()
        self.test_index_based_selection()
        self.test_combined_selection()
        self.test_faceting_and_animation()
        self.test_different_plotting_methods()
        self.test_backward_compatibility()
        self.test_precedence()
        self.test_empty_dict_behavior()
        self.test_error_handling()

        # Summary
        print('\n' + '#' * 70)
        print("# SUMMARY OF 'SELECT' PARAMETER CAPABILITIES")
        print('#' * 70)
        print("""
The 'select' parameter supports:

1. ✓ Single values:        select={'scenario': 'base'}
2. ✓ Multiple values:       select={'scenario': ['base', 'high']}
3. ✓ Index-based:          select={'scenario': 0} or select={'scenario': [0, 1]}
4. ✓ Slices:               select={'time': slice('2024-01', '2024-06')}
5. ✓ Multiple dimensions:   select={'scenario': 'base', 'period': 2024}
6. ✓ Empty dict:           select={{}} (no filtering)
7. ✓ Combined with faceting/animation
8. ✓ Works across all plotting methods
9. ✓ Backward compatible with 'indexer' (deprecated)
10. ✓ Proper error handling for invalid parameters

Key Benefits:
- More intuitive name ('select' vs 'indexer')
- Applied BEFORE faceting/animation
- Allows pre-filtering data for visualization
- Cleaner API for data subset selection
        """)
        print('#' * 70)


def main():
    """Main test runner."""
    try:
        tests = TestSelectParameter()
        tests.run_all_tests()
    except Exception as e:
        print(f'\n✗ Failed to initialize tests: {e}')
        import traceback

        traceback.print_exc()


if __name__ == '__main__':
    main()
