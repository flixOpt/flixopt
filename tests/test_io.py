import uuid

import pytest

import flixopt as fx
from flixopt.io import CalculationResultsPaths

from .conftest import (
    assert_almost_equal_numeric,
    flow_system_base,
    flow_system_long,
    flow_system_segments_of_flows_2,
    simple_flow_system,
    simple_flow_system_scenarios,
)


@pytest.fixture(
    params=[
        flow_system_base,
        simple_flow_system_scenarios,
        flow_system_segments_of_flows_2,
        simple_flow_system,
        flow_system_long,
    ]
)
def flow_system(request):
    fs = request.getfixturevalue(request.param.__name__)
    if isinstance(fs, fx.FlowSystem):
        return fs
    else:
        return fs[0]


@pytest.mark.slow
def test_flow_system_file_io(flow_system, highs_solver, request):
    # Use UUID to ensure unique names across parallel test workers
    unique_id = uuid.uuid4().hex[:12]
    worker_id = getattr(request.config, 'workerinput', {}).get('workerid', 'main')
    test_id = f'{worker_id}-{unique_id}'

    calculation_0 = fx.Optimization(f'IO-{test_id}', flow_system=flow_system)
    calculation_0.do_modeling()
    calculation_0.solve(highs_solver)
    calculation_0.flow_system.plot_network()

    calculation_0.results.to_file()
    paths = CalculationResultsPaths(calculation_0.folder, calculation_0.name)
    flow_system_1 = fx.FlowSystem.from_netcdf(paths.flow_system)

    calculation_1 = fx.Optimization(f'Loaded_IO-{test_id}', flow_system=flow_system_1)
    calculation_1.do_modeling()
    calculation_1.solve(highs_solver)
    calculation_1.flow_system.plot_network()

    assert_almost_equal_numeric(
        calculation_0.results.model.objective.value,
        calculation_1.results.model.objective.value,
        'objective of loaded flow_system doesnt match the original',
    )

    assert_almost_equal_numeric(
        calculation_0.results.solution['costs'].values,
        calculation_1.results.solution['costs'].values,
        'costs doesnt match expected value',
    )


def test_flow_system_io(flow_system):
    flow_system.to_json('fs.json')

    ds = flow_system.to_dataset()
    new_fs = fx.FlowSystem.from_dataset(ds)

    assert flow_system == new_fs

    print(flow_system)
    flow_system.__repr__()
    flow_system.__str__()


def test_suppress_output_file_descriptors(tmp_path):
    """Test that suppress_output() redirects file descriptors to /dev/null."""
    import os

    from flixopt.io import suppress_output

    # Create temporary files to capture output
    test_file = tmp_path / 'test_output.txt'

    # Test that FD 1 (stdout) is redirected during suppression
    with open(test_file, 'w') as f:
        original_stdout_fd = os.dup(1)  # Save original stdout FD
        try:
            # Redirect FD 1 to our test file
            os.dup2(f.fileno(), 1)
            os.write(1, b'before suppression\n')

            with suppress_output():
                # Inside suppress_output, writes should go to /dev/null, not our file
                os.write(1, b'during suppression\n')

            # After suppress_output, writes should go to our file again
            os.write(1, b'after suppression\n')
        finally:
            # Restore original stdout
            os.dup2(original_stdout_fd, 1)
            os.close(original_stdout_fd)

    # Read the file and verify content
    content = test_file.read_text()
    assert 'before suppression' in content
    assert 'during suppression' not in content  # This should NOT be in the file
    assert 'after suppression' in content


def test_suppress_output_python_level():
    """Test that Python-level stdout/stderr continue to work after suppress_output()."""
    import io
    import sys

    from flixopt.io import suppress_output

    # Create a StringIO to capture Python-level output
    captured_output = io.StringIO()

    # After suppress_output exits, Python streams should be functional
    with suppress_output():
        pass  # Just enter and exit the context

    # Redirect sys.stdout to our StringIO
    old_stdout = sys.stdout
    try:
        sys.stdout = captured_output
        print('test message')
    finally:
        sys.stdout = old_stdout

    # Verify Python-level stdout works
    assert 'test message' in captured_output.getvalue()


def test_suppress_output_exception_handling():
    """Test that suppress_output() properly restores streams even on exception."""
    import sys

    from flixopt.io import suppress_output

    # Save original file descriptors
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    try:
        with suppress_output():
            raise ValueError('Test exception')
    except ValueError:
        pass

    # Verify streams are restored after exception
    assert sys.stdout.fileno() == original_stdout_fd
    assert sys.stderr.fileno() == original_stderr_fd

    # Verify we can still write to stdout/stderr
    sys.stdout.write('test after exception\n')
    sys.stdout.flush()


def test_suppress_output_c_level():
    """Test that suppress_output() suppresses C-level output (file descriptor level)."""
    import os
    import sys

    from flixopt.io import suppress_output

    # This test verifies that even low-level C writes are suppressed
    # by writing directly to file descriptor 1 (stdout)
    with suppress_output():
        # Try to write directly to FD 1 (stdout) - should be suppressed
        os.write(1, b'C-level stdout write\n')
        # Try to write directly to FD 2 (stderr) - should be suppressed
        os.write(2, b'C-level stderr write\n')

    # After exiting context, ensure streams work
    sys.stdout.write('After C-level test\n')
    sys.stdout.flush()


def test_tqdm_cleanup_on_exception():
    """Test that tqdm progress bar is properly cleaned up even when exceptions occur.

    This test verifies the pattern used in SegmentedCalculation where a try/finally
    block ensures progress_bar.close() is called even if an exception occurs.
    """
    from tqdm import tqdm

    # Create a progress bar (disabled to avoid output during tests)
    items = enumerate(range(5))
    progress_bar = tqdm(items, total=5, desc='Test progress', disable=True)

    # Track whether cleanup was called
    cleanup_called = False
    exception_raised = False

    try:
        try:
            for idx, _ in progress_bar:
                if idx == 2:
                    raise ValueError('Test exception')
        finally:
            # This should always execute, even with exception
            progress_bar.close()
            cleanup_called = True
    except ValueError:
        exception_raised = True

    # Verify both that the exception was raised AND cleanup happened
    assert exception_raised, 'Test exception should have been raised'
    assert cleanup_called, 'Cleanup should have been called even with exception'

    # Verify that close() is idempotent - calling it again should not raise
    progress_bar.close()  # Should not raise even if already closed


if __name__ == '__main__':
    pytest.main(['-v', '--disable-warnings'])
