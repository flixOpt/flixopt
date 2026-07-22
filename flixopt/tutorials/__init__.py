"""Datasets and example systems for the flixopt tutorials and notebooks.

Two tiers, so every notebook is standalone after ``pip install flixopt`` - no need
to clone the repository or copy files out of GitHub:

* **Synthetic data** (notebooks 01-07) - generated on the fly from numpy/pandas,
  no files and no network. Access by name with :func:`get_data`; see :func:`list_data`.

* **Pre-built example systems** (notebooks 08-09) - downloaded (and cached) from the
  project's GitHub releases with :func:`load_example`; see :func:`list_examples`.
  Needs ``pooch`` (``pip install flixopt[tutorials]``).
"""

from ._examples import ExampleName, list_examples, load_example
from ._tutorial_data import DataName, get_data, list_data

__all__ = [
    'DataName',
    'get_data',
    'list_data',
    'ExampleName',
    'load_example',
    'list_examples',
]
