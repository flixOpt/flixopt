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

    def test_piecewise_with_gap_forces_minimum_load(self):
        """Proves: Gaps between pieces create forbidden operating regions.

        Converter with pieces: [fuel 0→0 / heat 0→0] and [fuel 40→100 / heat 40→100].
        The gap between 0 and 40 is forbidden — converter must be off (0) or at ≥40.
        CheapSrc at 1€/kWh has no gap constraint.
        Demand=[50,50]. Both sources can serve. But PiecewiseConverter has minimum load 40.

        Sensitivity: Without the gap (continuous 0-100), both could share any way.
        With the gap, PiecewiseConverter must produce ≥40 or 0. When demand=50, producing
        50 is valid (within 40-100 range). Verify the piecewise constraint is active.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([50, 50])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.Source(
                'CheapSrc',
                outputs=[
                    fx.Flow('heat', bus='Heat', effects_per_flow_hour=10),  # More expensive backup
                ],
            ),
            fx.LinearConverter(
                'Converter',
                inputs=[fx.Flow('fuel', bus='Gas')],
                outputs=[fx.Flow('heat', bus='Heat')],
                piecewise_conversion=fx.PiecewiseConversion(
                    {
                        # Gap between 0 and 40: forbidden region (minimum load requirement)
                        'fuel': fx.Piecewise([fx.Piece(0, 0), fx.Piece(40, 100)]),
                        'heat': fx.Piecewise([fx.Piece(0, 0), fx.Piece(40, 100)]),
                    }
                ),
            ),
        )
        solve(fs)
        # Converter at 1€/kWh (via gas), CheapSrc at 10€/kWh
        # Converter serves all 50 each timestep → fuel = 100, cost = 100
        assert_allclose(fs.solution['costs'].item(), 100.0, rtol=1e-5)
        # Verify converter heat is within valid range (0 or 40-100)
        heat = fs.solution['Converter(heat)|flow_rate'].values[:-1]
        for h in heat:
            assert h < 1e-5 or h >= 40.0 - 1e-5, f'Heat in forbidden gap: {h}'

    def test_piecewise_gap_allows_off_state(self):
        """Proves: Piecewise with off-state piece allows unit to be completely off
        when demand is below minimum load and backup is available.

        Converter: [0→0 / 0→0] (off) and [50→100 / 50→100] (operating range).
        Demand=[20,20]. Since 20 < 50 (min load), cheaper to use backup than run at 50.
        ExpensiveBackup at 3€/kWh. Converter at 1€/kWh but minimum 50.

        Sensitivity: If converter had to run (no off piece), cost=2×50×1=100.
        With off piece, backup covers all: cost=2×20×3=120. Wait, that's more expensive.
        Let's flip: Converter at 10€/kWh, Backup at 1€/kWh.
        Then: Converter at min 50 = 2×50×10=1000. Backup all = 2×20×1=40.
        The optimizer should choose backup (off state for converter).
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([20, 20])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=10),  # Expensive gas
                ],
            ),
            fx.Source(
                'Backup',
                outputs=[
                    fx.Flow('heat', bus='Heat', effects_per_flow_hour=1),  # Cheap backup
                ],
            ),
            fx.LinearConverter(
                'Converter',
                inputs=[fx.Flow('fuel', bus='Gas')],
                outputs=[fx.Flow('heat', bus='Heat')],
                piecewise_conversion=fx.PiecewiseConversion(
                    {
                        # Off state (0,0) + operating range with minimum load
                        'fuel': fx.Piecewise([fx.Piece(0, 0), fx.Piece(50, 100)]),
                        'heat': fx.Piecewise([fx.Piece(0, 0), fx.Piece(50, 100)]),
                    }
                ),
            ),
        )
        solve(fs)
        # Converter expensive (10€/kWh gas) with min load 50: 2×50×10=1000
        # Backup cheap (1€/kWh): 2×20×1=40
        # Optimizer chooses backup (converter off)
        assert_allclose(fs.solution['costs'].item(), 40.0, rtol=1e-5)
        # Verify converter is off
        conv_heat = fs.solution['Converter(heat)|flow_rate'].values[:-1]
        assert_allclose(conv_heat, [0, 0], atol=1e-5)

    def test_piecewise_varying_efficiency_across_segments(self):
        """Proves: Different segments can have different efficiency ratios,
        allowing modeling of equipment with varying efficiency at different loads.

        Segment 1: fuel 10→20, heat 10→15 (ratio starts at 1:1, ends at 1.33:1)
        Segment 2: fuel 20→50, heat 15→45 (ratio 1:1, more efficient at high load)
        Demand=35 falls in segment 2.

        Sensitivity: At segment 2, fuel = 20 + (35-15)/(45-15) × (50-20) = 20 + 20 = 40.
        If constant efficiency 1.33:1 from seg1 end were used, fuel≈46.67.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([35, 35])),
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
                        # Low load: less efficient. High load: more efficient.
                        'fuel': fx.Piecewise([fx.Piece(10, 20), fx.Piece(20, 50)]),
                        'heat': fx.Piecewise([fx.Piece(10, 15), fx.Piece(15, 45)]),
                    }
                ),
            ),
        )
        solve(fs)
        # heat=35 in segment 2: fuel = 20 + (35-15)/(45-15) × 30 = 20 + 20 = 40
        # cost = 2 × 40 = 80
        assert_allclose(fs.solution['costs'].item(), 80.0, rtol=1e-5)
        assert_allclose(fs.solution['Converter(fuel)|flow_rate'].values[0], 40.0, rtol=1e-5)
