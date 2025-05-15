"""
FlixOpt Results Explorer

A module for launching a Streamlit app to explore flixopt calculation results.
"""

import os
import subprocess
import sys
import webbrowser
from pathlib import Path


def explore_results(self, port=8501):
    """
    Launch a Streamlit app to explore the calculation results.
    This function is experimental and might have issues.

    Args:
        port: Port to use for the Streamlit server

    Returns:
        subprocess.Popen: The running Streamlit process
    """

    # Find explorer app path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    explorer_script = os.path.join(current_dir, '_results_explorer_app.py')

    # If the explorer app doesn't exist, inform the user
    if not os.path.exists(explorer_script):
        raise FileNotFoundError(
            f'Explorer app not found at {explorer_script}. '
            'Please ensure the explorer_app.py file is in the flixopt package directory.'
        )

    # Run the Streamlit app - the port argument needs to be separate from the script arguments
    cmd = [
        sys.executable,
        '-m',
        'streamlit',
        'run',
        explorer_script,
        '--server.port',
        str(port),
        '--',  # This separator is important
        str(self.folder),
        self.name,
    ]

    self.to_file() # Save results to file. This is needed to be able to launch the app from the file. # TODO

    # Launch the Streamlit app
    process = subprocess.Popen(cmd)

    # Open browser
    webbrowser.open(f'http://localhost:{port}')

    print(f'Streamlit app launched on port {port}. Press Ctrl+C to stop the app.')

    return process
