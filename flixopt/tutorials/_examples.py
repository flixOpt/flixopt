"""Download pre-built example FlowSystems for the advanced notebooks (08-09).

Unlike the synthetic :mod:`._tutorial_data` helpers, these example systems are
built from realistic profiles (BDEW load profiles via ``demandlib``, weather via
``pvlib``) and real input time series. Rather than regenerating them - which would
pull in those heavy dependencies and the raw input data - we build them once,
serialise them with :meth:`flixopt.FlowSystem.to_netcdf`, host the artefacts on the
project's GitHub releases, and download them on demand.

The download is cached on disk (via ``pooch``), so the network is only touched the
first time a given example is requested.

Usage::

    import flixopt as fx

    fs = fx.tutorials.load_example('district_heating')
    fx.tutorials.list_examples()  # -> available names
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from flixopt.flow_system import FlowSystem

# GitHub release holding the example artefacts. Versioned independently of the
# package; bump it (and re-upload the assets) when the example systems change.
DATA_RELEASE = 'tutorial-data-v1'

_BASE_URL_ENV = 'FLIXOPT_DATA_BASE_URL'  # override for testing / self-hosting
_DEFAULT_BASE_URL = f'https://github.com/flixOpt/flixopt/releases/download/{DATA_RELEASE}/'
_REGISTRY_FILENAME = 'registry.txt'

#: The available example systems - the single source of truth for their names.
#: Each is hosted as ``<name>.nc`` and built by ``create_<name>_system`` in
#: ``docs/notebooks/data/generate_example_systems.py``.
ExampleName = Literal[
    'simple',
    'complex',
    'district_heating',
    'operational',
    'seasonal_storage',
    'multiperiod',
]

_INSTALL_HINT = (
    "Downloading example systems needs the 'pooch' package. Install it with "
    '`pip install flixopt[tutorials]` (or `pip install pooch`).'
)


def list_examples() -> list[str]:
    """Return the names of the example systems available via :func:`load_example`."""
    return list(get_args(ExampleName))


def _base_url() -> str:
    url = os.environ.get(_BASE_URL_ENV, _DEFAULT_BASE_URL)
    return url if url.endswith('/') else url + '/'


def _make_pooch():
    try:
        import pooch
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(_INSTALL_HINT) from e

    # Hashes are loaded from the hosted registry.txt so they never drift out of
    # sync with the uploaded artefacts.
    odie = pooch.create(path=pooch.os_cache('flixopt'), base_url=_base_url(), registry=None)
    registry_path = pooch.retrieve(
        url=_base_url() + _REGISTRY_FILENAME,
        known_hash=None,
        path=odie.path,
        fname=_REGISTRY_FILENAME,
    )
    odie.load_registry(registry_path)
    return odie


def load_example(name: ExampleName) -> FlowSystem:
    """Download (and cache) a pre-built example FlowSystem and return it.

    Args:
        name: One of :func:`list_examples` (e.g. ``'district_heating'``).

    Returns:
        The deserialised :class:`flixopt.FlowSystem`.

    Raises:
        ValueError: If ``name`` is not a known example.
        ModuleNotFoundError: If ``pooch`` is not installed.

    Note:
        The first call for a given example downloads the artefact from the project's
        GitHub releases; subsequent calls read it from the local cache.
    """
    if name not in get_args(ExampleName):
        raise ValueError(f'Unknown example {name!r}. Available: {", ".join(list_examples())}.')

    from flixopt.flow_system import FlowSystem

    odie = _make_pooch()
    path = odie.fetch(f'{name}.nc')
    return FlowSystem.from_netcdf(path)
