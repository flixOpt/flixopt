import os
import subprocess
import sys
from pathlib import Path

import pytest

# Path to the examples directory
EXAMPLES_DIR = Path(__file__).parent.parent / 'examples'


DEPENDENT_EXAMPLES = ('complex_example.py', 'complex_example_results.py')


@pytest.mark.parametrize(
    'example_script',
    sorted(
        [p for p in EXAMPLES_DIR.rglob('*.py') if p.name not in DEPENDENT_EXAMPLES],
        key=lambda path: (str(path.parent), path.name),
    ),
    ids=lambda path: str(path.relative_to(EXAMPLES_DIR)),
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
    script_dir = example_script.parent
    original_cwd = os.getcwd()

    try:
        # Change the working directory to the script's location
        os.chdir(script_dir)

        # Run the script
        result = subprocess.run(
            [sys.executable, example_script.name],
            capture_output=True,
            text=True,
            timeout=180,  # Add timeout to prevent hanging
        )
        assert result.returncode == 0, (
            f'Script {example_script} failed:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}'
        )

    finally:
        # Restore the original working directory
        os.chdir(original_cwd)


@pytest.mark.examples
def test_dependent_examples():
    """Test examples that must run in order."""
    script_dir = EXAMPLES_DIR / '02_Complex'
    original_cwd = os.getcwd()

    try:
        os.chdir(script_dir)

        # Iterate in the defined order
        for script in DEPENDENT_EXAMPLES:
            result = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                timeout=180,
            )
            assert result.returncode == 0, f'{script} failed:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}'

    finally:
        os.chdir(original_cwd)


if __name__ == '__main__':
    pytest.main(['-v', '--disable-warnings'])
