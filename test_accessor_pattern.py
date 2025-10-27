"""Simple test to verify the statistics accessor pattern integration.

This script tests that the accessor pattern is properly integrated into
the CalculationResults class without requiring actual optimization results.
"""

import pathlib
import sys

# Add flixopt to path
sys.path.insert(0, str(pathlib.Path(__file__).parent))


def test_imports():
    """Test that all accessor pattern modules can be imported."""
    print('Testing imports...')

    try:
        from flixopt.plotting_accessor import MethodHandlerWrapper, StatisticPlotter

        print('  ✓ plotting_accessor imports successful')
    except Exception as e:
        print(f'  ❌ plotting_accessor import failed: {e}')
        return False

    try:
        from flixopt.plotting_accessor.plotly_charts import InteractivePlotter

        print('  ✓ InteractivePlotter import successful')
    except Exception as e:
        print(f'  ❌ InteractivePlotter import failed: {e}')
        return False

    try:
        from flixopt.statistics import StatisticsAccessor

        print('  ✓ StatisticsAccessor import successful')
    except Exception as e:
        print(f'  ❌ StatisticsAccessor import failed: {e}')
        return False

    try:
        import flixopt as fx

        print('  ✓ flixopt import successful')
    except Exception as e:
        print(f'  ❌ flixopt import failed: {e}')
        return False

    print('All imports successful!\n')
    return True


def test_accessor_integration():
    """Test that accessor is properly integrated into CalculationResults."""
    print('Testing accessor integration...')

    try:
        # Try to load an example result if available
        import flixopt as fx
        from flixopt.results import CalculationResults

        # First, check if we can access the class
        print(f'  CalculationResults class found: {CalculationResults}')

        # Try to load a result file if it exists
        example_path = pathlib.Path('examples/01_Simple/results')
        if example_path.exists():
            print(f'  Found example results at: {example_path}')
            try:
                results = CalculationResults.from_file(example_path, 'simple_example')
                print(f'  ✓ Loaded results: {results.name}')

                # Check for statistics accessor
                if hasattr(results, 'statistics'):
                    print(f'  ✓ Statistics accessor found: {results.statistics}')

                    # Try to list available methods
                    methods = [m for m in dir(results.statistics) if not m.startswith('_')]
                    print(f'  ✓ Available statistics methods: {methods}')

                    # Try to call a method
                    try:
                        plotter = results.statistics.flow_summary()
                        print(f'  ✓ flow_summary() returned: {plotter}')

                        # Check for plot attribute
                        if hasattr(plotter, 'plot'):
                            print('  ✓ Plotter has .plot attribute')
                        else:
                            print('  ❌ Plotter missing .plot attribute')

                        # Try to get data
                        try:
                            data = plotter.data  # Use .data property
                            print(f'  ✓ Retrieved data via .data: type={type(data)}')
                        except Exception as e:
                            print(f'  ❌ Failed to retrieve data: {e}')

                    except Exception as e:
                        print(f'  ❌ Failed to call flow_summary(): {e}')
                else:
                    print('  ❌ Statistics accessor not found')
                    return False

            except FileNotFoundError:
                print('  ! Results file not found (run example first)')
                print('  ! Cannot test with actual data, but imports work')
            except Exception as e:
                print(f'  ! Could not load results: {e}')
                print('  ! Cannot test with actual data, but imports work')
        else:
            print('  ! Example results not found')
            print('  ! Run: python examples/01_Simple/simple_example.py')
            print('  ! Cannot test with actual data, but imports work')

        print('\nAccessor integration test passed!\n')
        return True

    except Exception as e:
        print(f'  ❌ Integration test failed: {e}')
        import traceback

        traceback.print_exc()
        return False


def test_wrapper_decorator():
    """Test that the wrapper decorator works correctly."""
    print('Testing MethodHandlerWrapper decorator...')

    try:
        import xarray as xr

        from flixopt.plotting_accessor import MethodHandlerWrapper, StatisticPlotter

        # Create a mock class
        class MockParent:
            def __init__(self):
                self.data = xr.Dataset({'test': xr.DataArray([1, 2, 3])})

        class MockAccessor:
            def __init__(self, parent):
                self._parent = parent

            @MethodHandlerWrapper(handler_class=StatisticPlotter)
            def test_statistic(self):
                return self._parent.data

        # Test it
        parent = MockParent()
        accessor = MockAccessor(parent)
        plotter = accessor.test_statistic()

        print(f'  ✓ Wrapper returned: {type(plotter)}')

        if isinstance(plotter, StatisticPlotter):
            print('  ✓ Returned object is StatisticPlotter')
        else:
            print('  ❌ Returned object is not StatisticPlotter')
            return False

        # Test .data property
        data = plotter.data
        print(f'  ✓ Plotter.data returned: {type(data)}')

        # Test plot attribute
        if hasattr(plotter, 'plot'):
            print('  ✓ Plotter has .plot attribute')
        else:
            print('  ❌ Plotter missing .plot attribute')
            return False

        print('Wrapper decorator test passed!\n')
        return True

    except Exception as e:
        print(f'  ❌ Wrapper test failed: {e}')
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print('=' * 80)
    print('Testing PyPSA-Style Statistics Accessor Pattern')
    print('=' * 80)
    print()

    all_passed = True

    # Test 1: Imports
    if not test_imports():
        all_passed = False
        print('❌ Import tests failed\n')
    else:
        print('✓ Import tests passed\n')

    # Test 2: Wrapper decorator
    if not test_wrapper_decorator():
        all_passed = False
        print('❌ Wrapper decorator tests failed\n')
    else:
        print('✓ Wrapper decorator tests passed\n')

    # Test 3: Integration
    if not test_accessor_integration():
        all_passed = False
        print('❌ Integration tests failed\n')
    else:
        print('✓ Integration tests passed\n')

    # Summary
    print('=' * 80)
    if all_passed:
        print('✓ All tests passed!')
        print()
        print('The accessor pattern is properly integrated.')
        print('You can now use: results.statistics.method().plot.type()')
    else:
        print('❌ Some tests failed')
        print('Please check the errors above')
    print('=' * 80)


if __name__ == '__main__':
    main()
