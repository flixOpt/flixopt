"""Tests for deprecated Results.to_file() overwrite protection.

This module tests the deprecated Results.to_file() overwrite behavior.
These tests will be removed in v6.0.0.

For new tests, use FlowSystem.solution.to_netcdf() instead.
"""

import pathlib
import tempfile

import pytest

import flixopt as fx


def test_results_overwrite_protection(simple_flow_system, highs_solver):
    """Test that Results.to_file() prevents accidental overwriting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_folder = pathlib.Path(tmpdir) / 'results'

        # Run optimization
        opt = fx.Optimization('test_results', simple_flow_system, folder=test_folder)
        opt.do_modeling()
        opt.solve(highs_solver)

        # First save should succeed
        opt.results.to_file(compression=0, document_model=False, save_linopy_model=False)

        # Second save without overwrite should fail
        with pytest.raises(FileExistsError, match='Results files already exist'):
            opt.results.to_file(compression=0, document_model=False, save_linopy_model=False)

        # Third save with overwrite should succeed
        opt.results.to_file(compression=0, document_model=False, save_linopy_model=False, overwrite=True)


def test_results_overwrite_to_different_folder(simple_flow_system, highs_solver):
    """Test that saving to different folder works without overwrite flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_folder1 = pathlib.Path(tmpdir) / 'results1'
        test_folder2 = pathlib.Path(tmpdir) / 'results2'

        # Run optimization
        opt = fx.Optimization('test_results', simple_flow_system, folder=test_folder1)
        opt.do_modeling()
        opt.solve(highs_solver)

        # Save to first folder
        opt.results.to_file(compression=0, document_model=False, save_linopy_model=False)

        # Save to different folder should work without overwrite flag
        opt.results.to_file(folder=test_folder2, compression=0, document_model=False, save_linopy_model=False)


def test_results_overwrite_with_different_name(simple_flow_system, highs_solver):
    """Test that saving with different name works without overwrite flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_folder = pathlib.Path(tmpdir) / 'results'

        # Run optimization
        opt = fx.Optimization('test_results', simple_flow_system, folder=test_folder)
        opt.do_modeling()
        opt.solve(highs_solver)

        # Save with first name
        opt.results.to_file(compression=0, document_model=False, save_linopy_model=False)

        # Save with different name should work without overwrite flag
        opt.results.to_file(name='test_results_v2', compression=0, document_model=False, save_linopy_model=False)
