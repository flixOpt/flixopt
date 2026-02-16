"""Validation tests for input parameter checking.

Tests verify that appropriate errors are raised when invalid or
inconsistent parameters are provided to components and flows.
"""

import numpy as np
import pytest

import flixopt as fx
from flixopt.core import PlausibilityError

from .conftest import make_flow_system


class TestValidation:
    def test_source_and_sink_requires_size_with_prevent_simultaneous(self):
        """Proves: SourceAndSink with prevent_simultaneous_flow_rates=True raises
        PlausibilityError when flows don't have a size.

        prevent_simultaneous internally adds StatusParameters, which require
        a defined size to bound the flow rate. Without size, optimization
        should raise PlausibilityError during model building.
        """
        fs = make_flow_system(3)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([0.1, 0.1, 0.1])),
                ],
            ),
            fx.SourceAndSink(
                'GridConnection',
                outputs=[fx.Flow(bus='Elec', flow_id='buy', effects_per_flow_hour=5)],
                inputs=[fx.Flow(bus='Elec', flow_id='sell', effects_per_flow_hour=-1)],
                prevent_simultaneous_flow_rates=True,
            ),
        )
        with pytest.raises(PlausibilityError, match='status_parameters but no size'):
            fs.optimize(fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=60, log_to_console=False))
