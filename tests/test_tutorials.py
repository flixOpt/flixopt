"""Tests for the flixopt.tutorials tutorial-data helpers."""

from typing import get_args

import pandas as pd
import pytest

import flixopt as fx
from flixopt import tutorials


def test_tutorials_exposed_on_package():
    assert fx.tutorials is tutorials


@pytest.mark.parametrize('name', tutorials.list_data())
def test_get_data_returns_timesteps(name):
    """Synthetic tutorial data must work offline and expose a DatetimeIndex."""
    data = tutorials.get_data(name)
    assert isinstance(data, dict)
    assert isinstance(data['timesteps'], pd.DatetimeIndex)
    assert len(data['timesteps']) > 0


def test_list_data_matches_literal():
    assert tutorials.list_data() == list(get_args(tutorials.DataName))


def test_get_data_rejects_unknown_name():
    with pytest.raises(ValueError, match='Unknown dataset'):
        tutorials.get_data('does-not-exist')


def test_list_examples_matches_literal():
    assert tutorials.list_examples() == list(get_args(tutorials.ExampleName))


def test_load_example_rejects_unknown_name():
    """Validation happens before any network access."""
    with pytest.raises(ValueError, match='Unknown example'):
        tutorials.load_example('does-not-exist')
