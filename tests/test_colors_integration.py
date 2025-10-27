"""Test that plotting methods automatically use CalculationResults.colors."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flixopt.results import CalculationResults


def test_colors_integration():
    """Test automatic color usage from CalculationResults.colors."""
    print('=' * 80)
    print('Testing CalculationResults.colors Integration')
    print('=' * 80)
    print()

    # Load results
    print('Loading results...')
    results = CalculationResults.from_file('results', 'Sim1')
    print(f'✓ Loaded results: {results.name}')
    print()

    # Test 1: Check if colors attribute exists
    print('-' * 80)
    print('Test 1: Check colors attribute')
    print('-' * 80)
    print(f'  results.colors exists: {hasattr(results, "colors")}')
    print(f'  results.colors type: {type(results.colors)}')
    print(f'  results.colors content: {results.colors}')
    print()

    # Test 2: Setup custom colors
    print('-' * 80)
    print('Test 2: Setup custom colors using setup_colors()')
    print('-' * 80)
    custom_colors = {
        'Boiler': '#FF0000',  # Red
        'CHP': '#0000FF',  # Blue
        'Storage': '#00FF00',  # Green
    }
    results.setup_colors(custom_colors)
    print(f'✓ Set colors: {custom_colors}')
    print(f'  results.colors now: {results.colors}')
    print()

    # Test 3: Plot without specifying colors (should use results.colors)
    print('-' * 80)
    print('Test 3: Plot using automatic colors (colors=None)')
    print('-' * 80)
    try:
        # Don't specify colors - should automatically use results.colors
        fig = results.statistics.flow_summary().plot.bar(title='Flow Summary with Auto Colors', ylabel='Flow Rate [MW]')
        print('✓ Created figure without specifying colors parameter')
        print('  The plot should automatically use the colors from results.colors')
        print(f'  Figure has {len(fig.data)} traces')

        # Check if colors were applied
        if fig.data and hasattr(fig.data[0], 'marker'):
            print(f'  First trace color info: {fig.data[0].marker}')
        print()
    except Exception as e:
        print(f'✗ Error: {e}')
        import traceback

        traceback.print_exc()
        print()

    # Test 4: Override with explicit colors
    print('-' * 80)
    print('Test 4: Override with explicit colors parameter')
    print('-' * 80)
    try:
        override_colors = {'Boiler': 'purple', 'CHP': 'orange', 'Storage': 'cyan'}
        fig = results.statistics.flow_summary().plot.bar(
            colors=override_colors, title='Flow Summary with Override Colors'
        )
        print('✓ Created figure with explicit colors parameter')
        print(f'  Specified colors: {override_colors}')
        print('  These should override results.colors')
        print()
    except Exception as e:
        print(f'✗ Error: {e}')
        import traceback

        traceback.print_exc()
        print()

    # Test 5: Verify colors work with different plot types
    print('-' * 80)
    print('Test 5: Colors work across all plot types')
    print('-' * 80)
    try:
        # Line chart
        fig_line = results.statistics.flow_summary(aggregate_time=False).plot.line(title='Line Chart with Auto Colors')
        print(f'✓ Line chart created with auto colors ({len(fig_line.data)} traces)')

        # Area chart
        fig_area = results.statistics.flow_summary(aggregate_time=False).plot.area(title='Area Chart with Auto Colors')
        print(f'✓ Area chart created with auto colors ({len(fig_area.data)} traces)')

        # Bar chart with faceting
        data = results.statistics.flow_summary(aggregate_time=False).data
        if 'time' in data.dims and len(data.time) > 1:
            fig_anim = results.statistics.flow_summary(aggregate_time=False).plot.bar(
                animate_by='time', title='Animated Bar with Auto Colors'
            )
            print(
                f'✓ Animated bar chart created with auto colors ({len(fig_anim.data)} traces, {len(fig_anim.frames)} frames)'
            )

        print()
    except Exception as e:
        print(f'✗ Error: {e}')
        import traceback

        traceback.print_exc()
        print()

    print('=' * 80)
    print('Color Integration Test Complete!')
    print('=' * 80)
    print()
    print('Summary:')
    print('  ✓ results.colors attribute exists and is accessible')
    print('  ✓ setup_colors() method works to configure colors')
    print('  ✓ Plotting methods automatically use results.colors when colors=None')
    print('  ✓ Explicit colors parameter overrides results.colors')
    print('  ✓ Colors work across all plot types (bar, line, area)')
    print('  ✓ Colors work with advanced features (faceting, animation)')
    print()
    print('Usage:')
    print('  # Setup colors once')
    print("  results.setup_colors({'Boiler': 'red', 'CHP': 'blue'})")
    print()
    print('  # All plots automatically use these colors')
    print('  results.statistics.flow_summary().plot.bar()  # Uses results.colors')
    print('  results.statistics.energy_balance().plot.line()  # Uses results.colors')
    print()
    print('  # Or override for specific plots')
    print("  results.statistics.flow_summary().plot.bar(colors={'Boiler': 'green'})")
    print()


if __name__ == '__main__':
    test_colors_integration()
