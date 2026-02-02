"""Mathematical correctness tests for piecewise linearization."""

import numpy as np
from numpy.testing import assert_allclose

import flixopt as fx

from .conftest import make_flow_system, solve


class TestPiecewise:
    def test_piecewise_selects_cheap_segment(self):
        """Proves: PiecewiseConversion correctly interpolates within the active segment,
        and the optimizer selects the right segment for a given demand level.

        2-segment converter: seg1 fuel 10→30/heat 5→15 (ratio 2:1),
        seg2 fuel 30→100/heat 15→60 (ratio ≈1.56:1, more efficient).
        Demand=45 falls in segment 2.

        Sensitivity: If piecewise were ignored and a constant ratio used (e.g. 2:1
        from seg1), fuel would be 90 per timestep → cost=180 instead of ≈153.33.
        If the wrong segment were selected, the interpolation would be incorrect.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([45, 45])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.LinearConverter(
                'Converter',
                inputs=[fx.Flow('fuel', bus='Gas')],
                outputs=[fx.Flow('heat', bus='Heat')],
                piecewise_conversion=fx.PiecewiseConversion(
                    {
                        'fuel': fx.Piecewise([fx.Piece(10, 30), fx.Piece(30, 100)]),
                        'heat': fx.Piecewise([fx.Piece(5, 15), fx.Piece(15, 60)]),
                    }
                ),
            ),
        )
        solve(fs)
        # heat=45 in segment 2: fuel = 30 + (45-15)/(60-15) * (100-30) = 30 + 46.667 = 76.667
        # cost per timestep = 76.667, total = 2 * 76.667 ≈ 153.333
        assert_allclose(fs.solution['costs'].item(), 2 * (30 + 30 / 45 * 70), rtol=1e-4)

    def test_piecewise_conversion_at_breakpoint(self):
        """Proves: PiecewiseConversion is consistent at segment boundaries — both
        adjacent segments agree on the flow ratio at the shared breakpoint.

        Demand=15 = end of seg1 = start of seg2. Both give fuel=30.
        Verifies the fuel flow_rate directly.

        Sensitivity: If breakpoint handling were off-by-one or segments didn't
        share boundary values, fuel would differ from 30 (e.g. interpolation
        error or infeasibility at the boundary).
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([15, 15])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.LinearConverter(
                'Converter',
                inputs=[fx.Flow('fuel', bus='Gas')],
                outputs=[fx.Flow('heat', bus='Heat')],
                piecewise_conversion=fx.PiecewiseConversion(
                    {
                        'fuel': fx.Piecewise([fx.Piece(10, 30), fx.Piece(30, 100)]),
                        'heat': fx.Piecewise([fx.Piece(5, 15), fx.Piece(15, 60)]),
                    }
                ),
            ),
        )
        solve(fs)
        # At breakpoint: fuel = 30 per timestep, total = 60
        assert_allclose(fs.solution['costs'].item(), 60.0, rtol=1e-5)
        # Verify fuel flow rate
        assert_allclose(fs.solution['Converter(fuel)|flow_rate'].values[0], 30.0, rtol=1e-5)
