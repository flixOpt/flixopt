import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

# Path to the examples directory (at project root, not in tests/)
EXAMPLES_DIR = Path(__file__).parent.parent.parent / 'examples'

# Examples that have dependencies and must run in sequence
DEPENDENT_EXAMPLES = (
    '02_Complex/complex_example.py',
    '02_Complex/complex_example_results.py',
)


@contextmanager
def working_directory(path):
    """Context manager for changing the working directory."""
    original_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_cwd)


@pytest.mark.parametrize(
    'example_script',
    sorted(
        [p for p in EXAMPLES_DIR.rglob('*.py') if str(p.relative_to(EXAMPLES_DIR)) not in DEPENDENT_EXAMPLES],
        key=lambda path: (str(path.parent), path.name),
    ),
    ids=lambda path: str(path.relative_to(EXAMPLES_DIR)).replace(os.sep, '/'),
)
@pytest.mark.examples
def test_independent_examples(example_script):
    """
    Test independent example scripts.
    Ensures they run without errors.
    Changes the current working directory to the directory of the example script.
    Runs them alphabetically.
    This imitates behaviour of running the script directly.
    """
    with working_directory(example_script.parent):
        timeout = 800
        # Set environment variable to disable interactive plotting
        env = os.environ.copy()
        env['FLIXOPT_CI'] = 'true'
        try:
            result = subprocess.run(
                [sys.executable, example_script.name],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f'Script {example_script} timed out after {timeout} seconds')

        assert result.returncode == 0, (
            f'Script {example_script} failed:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}'
        )


@pytest.mark.examples
def test_dependent_examples():
    """Test examples that must run in order (complex_example.py generates data for complex_example_results.py)."""
    for script_path in DEPENDENT_EXAMPLES:
        script_full_path = EXAMPLES_DIR / script_path

        with working_directory(script_full_path.parent):
            timeout = 600
            # Set environment variable to disable interactive plotting
            env = os.environ.copy()
            env['FLIXOPT_CI'] = 'true'
            try:
                result = subprocess.run(
                    [sys.executable, script_full_path.name],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                pytest.fail(f'Script {script_path} timed out after {timeout} seconds')

            assert result.returncode == 0, f'{script_path} failed:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}'


if __name__ == '__main__':
    pytest.main(['-v', '--disable-warnings', '-m', 'examples'])
