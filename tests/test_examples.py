import os
import subprocess
import sys
from pathlib import Path

import pytest

# Path to the examples directory
EXAMPLES_DIR = Path(__file__).parent.parent / 'examples'


@pytest.mark.parametrize(
    'example_script',
    sorted(
        EXAMPLES_DIR.rglob('*.py'), key=lambda path: (str(path.parent), path.name)
    ),  # Sort by parent and script name
    ids=lambda path: str(path.relative_to(EXAMPLES_DIR)),  # Show relative file paths
)
@pytest.mark.slow
def test_example_scripts(example_script):
    """
    Test all example scripts in the examples directory.
    Ensures they run without errors.
    Changes the current working directory to the directory of the example script.
    Runs them alphabetically.
    This imitates behaviour of running the script directly
    """
    script_dir = example_script.parent
    original_cwd = os.getcwd()

    try:
        # Change the working directory to the script's location
        os.chdir(script_dir)

        # Run the script
        result = subprocess.run(
            [sys.executable, example_script.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert result.returncode == 0, f'Script {example_script} failed:\n{result.stderr}'

    finally:
        # Restore the original working directory
        os.chdir(original_cwd)


if __name__ == '__main__':
    pytest.main(['-v', '--disable-warnings'])
