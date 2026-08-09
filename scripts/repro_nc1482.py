"""Reproducer for Unidata/netcdf4-python#1482.

PermissionError when a *directory* component of the path contains non-ASCII
characters. Reported on Windows + netCDF4 1.7.4 / libnetcdf 4.9.3; macOS and
Linux are said to be unaffected.

Run:  python scripts/repro_nc1482.py
Exit code 1 if any netcdf4-engine case fails.
"""

from __future__ import annotations

import platform
import sys
import tempfile
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

import flixopt as fx
from flixopt import io as fx_io


def _dataset() -> xr.Dataset:
    return xr.Dataset({'a': ('time', np.arange(5.0))}, coords={'time': np.arange(5)})


def _case(name: str, fn) -> bool:
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - we want the type name in the report
        print(f'FAIL  {name}: {type(e).__name__}: {e}')
        return False
    print(f'ok    {name}')
    return True


def main() -> int:
    print(f'platform     : {platform.platform()}')
    print(f'python       : {sys.version.split()[0]}')
    print(f'filesystem   : {sys.getfilesystemencoding()}')
    print(f'netCDF4      : {netCDF4.__version__}')
    print(f'libnetcdf    : {netCDF4.__netcdf4libversion__}')
    print(f'xarray       : {xr.__version__}')
    print(f'flixopt      : {fx.__version__}')
    print()

    ds = _dataset()
    ok = True

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        ascii_dir = root / 'plain'
        ascii_dir.mkdir()
        # Control: non-ASCII *filename* in an ASCII directory - reported to work.
        ok &= _case('ascii dir  / ascii file  / netCDF4 lib', lambda: netCDF4.Dataset(ascii_dir / 'a.nc', 'w').close())
        ok &= _case(
            'ascii dir  / umlaut file / netCDF4 lib', lambda: netCDF4.Dataset(ascii_dir / 'grün.nc', 'w').close()
        )

        uni_dir = root / 'Müller_测试'
        uni_dir.mkdir()
        ok &= _case('umlaut dir / ascii file  / netCDF4 lib', lambda: netCDF4.Dataset(uni_dir / 'a.nc', 'w').close())
        ok &= _case('umlaut dir / umlaut file / netCDF4 lib', lambda: netCDF4.Dataset(uni_dir / 'grün.nc', 'w').close())

        # The path flixopt actually takes: xarray -> engine='netcdf4'.
        ok &= _case(
            'umlaut dir / xr.to_netcdf  engine=netcdf4', lambda: ds.to_netcdf(uni_dir / 'xr.nc', engine='netcdf4')
        )
        ok &= _case(
            'umlaut dir / xr.load       engine=netcdf4',
            lambda: xr.load_dataset(str(uni_dir / 'xr.nc'), engine='netcdf4'),
        )

        ok &= _case(
            'umlaut dir / fx save_dataset_to_netcdf', lambda: fx_io.save_dataset_to_netcdf(ds, uni_dir / 'fx.nc')
        )
        ok &= _case(
            'umlaut dir / fx load_dataset_from_netcdf', lambda: fx_io.load_dataset_from_netcdf(uni_dir / 'fx.nc')
        )

        # Comparison engine - reported to handle the same paths fine.
        try:
            import h5netcdf  # noqa: F401
        except ImportError:
            print('skip  umlaut dir / xr.to_netcdf  engine=h5netcdf (h5netcdf not installed)')
        else:
            _case(
                'umlaut dir / xr.to_netcdf  engine=h5netcdf', lambda: ds.to_netcdf(uni_dir / 'h5.nc', engine='h5netcdf')
            )

    print()
    print('RESULT:', 'all netcdf4 cases passed' if ok else 'REPRODUCED - at least one netcdf4 case failed')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
