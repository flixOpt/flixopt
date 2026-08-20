"""Mathematical correctness tests for component-level features.

Tests for component-specific behavior including:
- Component-level StatusParameters (affects all flows)
- Transmission with losses
- HeatPump with COP
"""

import numpy as np
from numpy.testing import assert_allclose

import flixopt as fx

from .conftest import make_flow_system


class TestComponentStatus:
    """Tests for StatusParameters applied at the component level (not flow level)."""

    def test_component_status_startup_cost(self, optimize):
        """Proves: StatusParameters on LinearConverter applies startup cost when
        the component (all its flows) transitions to active.

        Boiler with component-level status_parameters(effects_per_startup=100).
        Demand=[0,20,0,20]. Two startups.

        Sensitivity: Without startup cost, cost=40 (fuel only).
        With 100€/startup × 2, cost=240.
        """
        fs = make_flow_system(4)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([0, 20, 0, 20])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.LinearConverter(
                'Boiler',
                inputs=[fx.Flow('fuel', bus='Gas', size=100)],  # Size required for component status
                outputs=[fx.Flow('heat', bus='Heat', size=100)],  # Size required for component status
                conversion_factors=[{'fuel': 1, 'heat': 1}],
                status_parameters=fx.StatusParameters(effects_per_startup=100),
            ),
        )
        fs = optimize(fs)
        # fuel=40, 2 startups × 100 = 200, total = 240
        assert_allclose(fs.solution['costs'].item(), 240.0, rtol=1e-5)

    def test_component_status_min_uptime(self, optimize):
        """Proves: min_uptime on component level forces the entire component
        to stay on for consecutive hours.

        LinearConverter with component-level min_uptime=2.
        Demand=[20,10,20]. Component must stay on all 3 hours due to min_uptime blocks.

        Sensitivity: Without min_uptime, could turn on/off freely.
        With min_uptime=2, status is forced into 2-hour blocks.
        """
        fs = make_flow_system(3)
        fs.add_elements(
            fx.Bus('Heat'),  # Strict balance
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([20, 10, 20])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.LinearConverter(
                'Boiler',
                inputs=[fx.Flow('fuel', bus='Gas', size=100)],  # Size required
                outputs=[fx.Flow('heat', bus='Heat', size=100)],
                conversion_factors=[{'fuel': 1, 'heat': 1}],
                status_parameters=fx.StatusParameters(min_uptime=2),
            ),
        )
        fs = optimize(fs)
        # Demand must be met: fuel = 20 + 10 + 20 = 50
        assert_allclose(fs.solution['costs'].item(), 50.0, rtol=1e-5)
        # Verify component is on all 3 hours (min_uptime forces continuous operation)
        status = fs.solution['Boiler(heat)|status'].values[:-1]
        assert all(s > 0.5 for s in status), f'Component should be on all hours: {status}'

    def test_component_status_active_hours_max(self, optimize):
        """Proves: active_hours_max on component level limits total operating hours.

        LinearConverter with active_hours_max=2. Backup available.
        Demand=[10,10,10,10]. Component can only run 2 of 4 hours.

        Sensitivity: Without limit, component runs all 4 hours → cost=40.
        With limit=2, backup covers 2 hours → cost=60.
        """
        fs = make_flow_system(4)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10, 10])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.LinearConverter(
                'CheapBoiler',
                inputs=[fx.Flow('fuel', bus='Gas', size=100)],  # Size required
                outputs=[fx.Flow('heat', bus='Heat', size=100)],  # Size required
                conversion_factors=[{'fuel': 1, 'heat': 1}],
                status_parameters=fx.StatusParameters(active_hours_max=2),
            ),
            fx.linear_converters.Boiler(
                'ExpensiveBackup',
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow('fuel', bus='Gas'),
                thermal_flow=fx.Flow('heat', bus='Heat', size=100),
            ),
        )
        fs = optimize(fs)
        # CheapBoiler: 2 hours × 10 = 20
        # ExpensiveBackup: 2 hours × 10/0.5 = 40
        # total = 60
        assert_allclose(fs.solution['costs'].item(), 60.0, rtol=1e-5)

    def test_component_status_effects_per_active_hour(self, optimize):
        """Proves: effects_per_active_hour on component level adds cost per active hour.

        LinearConverter with effects_per_active_hour=50. Two hours of operation.

        Sensitivity: Without effects_per_active_hour, cost=20 (fuel only).
        With 50€/hour × 2, cost=120.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([10, 10])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.LinearConverter(
                'Boiler',
                inputs=[fx.Flow('fuel', bus='Gas', size=100)],
                outputs=[fx.Flow('heat', bus='Heat', size=100)],
                conversion_factors=[{'fuel': 1, 'heat': 1}],
                status_parameters=fx.StatusParameters(effects_per_active_hour=50),
            ),
        )
        fs = optimize(fs)
        # fuel=20, active_hour_cost=2×50=100, total=120
        assert_allclose(fs.solution['costs'].item(), 120.0, rtol=1e-5)

    def test_component_status_active_hours_min(self, optimize):
        """Proves: active_hours_min on component level forces minimum operating hours.

        Expensive LinearConverter with active_hours_min=2. Cheap backup available.
        Demand=[10,10]. Without constraint, backup would serve all (cost=20).
        With active_hours_min=2, expensive component must run both hours.

        Sensitivity: Without active_hours_min, backup covers all → cost=20.
        With floor=2, expensive component runs → status must be [1,1].
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([10, 10])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.LinearConverter(
                'ExpensiveBoiler',
                inputs=[fx.Flow('fuel', bus='Gas', size=100)],
                outputs=[fx.Flow('heat', bus='Heat', size=100)],
                conversion_factors=[{'fuel': 1, 'heat': 2}],  # eta=0.5 (fuel:heat = 1:2 → eta = 1/2)
                status_parameters=fx.StatusParameters(active_hours_min=2),
            ),
            fx.LinearConverter(
                'CheapBoiler',
                inputs=[fx.Flow('fuel', bus='Gas', size=100)],
                outputs=[fx.Flow('heat', bus='Heat', size=100)],
                conversion_factors=[{'fuel': 1, 'heat': 1}],
            ),
        )
        fs = optimize(fs)
        # ExpensiveBoiler must be on 2 hours (status=1). Verify status.
        status = fs.solution['ExpensiveBoiler(heat)|status'].values[:-1]
        assert_allclose(status, [1, 1], atol=1e-5)

    def test_component_status_max_uptime(self, optimize):
        """Proves: max_uptime on component level limits continuous operation.

        LinearConverter with max_uptime=2, min_uptime=2, previous state was on for 1 hour.
        Cheap boiler, expensive backup. Demand=[10,10,10,10,10].
        With previous_flow_rate and max_uptime=2, boiler can only run 1 more hour at start.

        Sensitivity: Without max_uptime, cheap boiler runs all 5 hours → cost=50.
        With max_uptime=2 and 1 hour carry-over, pattern forces backup use.
        """
        fs = make_flow_system(5)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10, 10, 10])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.LinearConverter(
                'CheapBoiler',
                inputs=[fx.Flow('fuel', bus='Gas', size=100, previous_flow_rate=10)],
                outputs=[fx.Flow('heat', bus='Heat', size=100, previous_flow_rate=10)],
                conversion_factors=[{'fuel': 1, 'heat': 1}],
                status_parameters=fx.StatusParameters(max_uptime=2, min_uptime=2),
            ),
            fx.LinearConverter(
                'ExpensiveBackup',
                inputs=[fx.Flow('fuel', bus='Gas', size=100)],
                outputs=[fx.Flow('heat', bus='Heat', size=100)],
                conversion_factors=[{'fuel': 1, 'heat': 2}],  # eta=0.5 (fuel:heat = 1:2 → eta = 1/2)
            ),
        )
        fs = optimize(fs)
        # With previous 1h uptime + max_uptime=2: can run 1 more hour, then must stop.
        # Pattern forced: [on,off,on,on,off] or similar with blocks of ≤2 consecutive.
        # CheapBoiler runs 3 hours, ExpensiveBackup runs 2 hours.
        # Without max_uptime: 5 hours cheap = 50
        # Verify no more than 2 consecutive on-hours for cheap boiler
        status = fs.solution['CheapBoiler(heat)|status'].values[:-1]
        max_consecutive = 0
        current_consecutive = 0
        for s in status:
            if s > 0.5:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        assert max_consecutive <= 2, f'max_uptime violated: {status}'

    def test_component_status_min_downtime(self, optimize):
        """Proves: min_downtime on component level prevents quick restart.

        CheapBoiler with min_downtime=3, relative_minimum=0.1. Was on before horizon.
        Demand=[20,0,20,0]. With relative_minimum, cannot stay on at t=1 (would overproduce).
        Must turn off at t=1, then min_downtime=3 prevents restart until t=1,2,3 elapsed.

        Sensitivity: Without min_downtime, cheap boiler restarts at t=2 → cost=40.
        With min_downtime=3, backup needed at t=2 → cost=60.
        """
        fs = make_flow_system(4)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([20, 0, 20, 0])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.LinearConverter(
                'CheapBoiler',
                inputs=[fx.Flow('fuel', bus='Gas', size=100, previous_flow_rate=20, relative_minimum=0.1)],
                outputs=[fx.Flow('heat', bus='Heat', size=100, previous_flow_rate=20, relative_minimum=0.1)],
                conversion_factors=[{'fuel': 1, 'heat': 1}],
                status_parameters=fx.StatusParameters(min_downtime=3),
            ),
            fx.LinearConverter(
                'ExpensiveBackup',
                inputs=[fx.Flow('fuel', bus='Gas', size=100)],
                outputs=[fx.Flow('heat', bus='Heat', size=100)],
                conversion_factors=[
                    {'fuel': 1, 'heat': 2}
                ],  # eta=0.5 (fuel:heat = 1:2 → eta = 1/2) (1 fuel → 0.5 heat)
            ),
        )
        fs = optimize(fs)
        # t=0: CheapBoiler on (20). At t=1 demand=0, relative_min forces off.
        # min_downtime=3: must stay off t=1,2,3. Can't restart at t=2.
        # Backup covers t=2: fuel = 20/0.5 = 40.
        # Without min_downtime: CheapBoiler at t=2 (fuel=20), total=40 vs 60.
        assert_allclose(fs.solution['costs'].item(), 60.0, rtol=1e-5)
        # Verify CheapBoiler is off at t=2
        assert fs.solution['CheapBoiler(heat)|status'].values[2] < 0.5

    def test_component_status_max_downtime(self, optimize):
        """Proves: max_downtime on component level forces restart after idle.

        ExpensiveBoiler with max_downtime=1 was on before horizon.
        CheapBackup available. Demand=[10,10,10,10].
        max_downtime=1 means ExpensiveBoiler can be off at most 1 consecutive hour.
        Since ExpensiveBoiler can supply any amount ≤20, CheapBackup can complement.

        Sensitivity: Without max_downtime, all from CheapBackup → cost=40.
        With max_downtime=1, ExpensiveBoiler forced on ≥2 of 4 hours → cost > 40.
        """
        fs = make_flow_system(4)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10, 10])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.LinearConverter(
                'ExpensiveBoiler',
                inputs=[fx.Flow('fuel', bus='Gas', size=40, previous_flow_rate=20)],
                outputs=[fx.Flow('heat', bus='Heat', size=20, relative_minimum=0.5, previous_flow_rate=10)],
                conversion_factors=[
                    {'fuel': 1, 'heat': 2}
                ],  # eta=0.5 (fuel:heat = 1:2 → eta = 1/2) (1 fuel → 0.5 heat)
                status_parameters=fx.StatusParameters(max_downtime=1),
            ),
            fx.LinearConverter(
                'CheapBackup',
                inputs=[fx.Flow('fuel', bus='Gas', size=100)],
                outputs=[fx.Flow('heat', bus='Heat', size=100)],
                conversion_factors=[{'fuel': 1, 'heat': 1}],
            ),
        )
        fs = optimize(fs)
        # max_downtime=1: no two consecutive off-hours for ExpensiveBoiler
        status = fs.solution['ExpensiveBoiler(heat)|status'].values[:-1]
        for i in range(len(status) - 1):
            assert not (status[i] < 0.5 and status[i + 1] < 0.5), f'Consecutive off at t={i},{i + 1}'
        # Without max_downtime, all from CheapBackup: cost=40
        # With constraint, ExpensiveBoiler must run ≥2 hours → cost > 40
        assert fs.solution['costs'].item() > 40.0 + 1e-5

    def test_component_status_startup_limit(self, optimize):
        """Proves: startup_limit on component level caps number of startups.

        CheapBoiler with startup_limit=1, relative_minimum=0.5, was off before horizon.
        ExpensiveBackup available. Demand=[10,0,10].
        With relative_minimum, CheapBoiler can't stay on at t=1 (would overproduce).
        Two peaks would need 2 startups, but limit=1 → backup covers one peak.

        Sensitivity: Without startup_limit, CheapBoiler serves both peaks → cost=20.
        With startup_limit=1, backup serves one peak → cost=30.
        """
        fs = make_flow_system(3)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([10, 0, 10])),
                ],
            ),
            fx.Source(
                'GasSrc',
                outputs=[
                    fx.Flow('gas', bus='Gas', effects_per_flow_hour=1),
                ],
            ),
            fx.LinearConverter(
                'CheapBoiler',
                inputs=[fx.Flow('fuel', bus='Gas', size=20, previous_flow_rate=0, relative_minimum=0.5)],
                outputs=[fx.Flow('heat', bus='Heat', size=20, previous_flow_rate=0, relative_minimum=0.5)],
                conversion_factors=[{'fuel': 1, 'heat': 1}],  # eta=1.0
                status_parameters=fx.StatusParameters(startup_limit=1),
            ),
            fx.LinearConverter(
                'ExpensiveBackup',
                inputs=[fx.Flow('fuel', bus='Gas', size=100)],
                outputs=[fx.Flow('heat', bus='Heat', size=100)],
                conversion_factors=[
                    {'fuel': 1, 'heat': 2}
                ],  # eta=0.5 (fuel:heat = 1:2 → eta = 1/2) (1 fuel → 0.5 heat)
            ),
        )
        fs = optimize(fs)
        # With relative_minimum=0.5 on size=20, when ON must produce ≥10 heat.
        # At t=1 with demand=0, staying on would overproduce → must turn off.
        # So optimally needs: on-off-on = 2 startups.
        # startup_limit=1: only 1 startup allowed.
        # CheapBoiler serves 1 peak: 10 heat needs 10 fuel.
        # ExpensiveBackup serves other peak: 10/0.5 = 20 fuel.
        # Total = 30. Without limit: 2×10 = 20.
        assert_allclose(fs.solution['costs'].item(), 30.0, rtol=1e-5)


class TestTransmission:
    """Tests for Transmission component with losses and structural constraints."""

    def test_transmission_relative_losses(self, optimize):
        """Proves: relative_losses correctly reduces transmitted energy.

        Transmission with relative_losses=0.1 (10% loss).
        CheapSource→Transmission→Demand. Source produces more than demand receives.

        Sensitivity: Without losses, source=100 for demand=100.
        With 10% loss, source≈111.11 for demand=100.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Source'),
            fx.Bus('Sink'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Sink', size=1, fixed_relative_profile=np.array([50, 50])),
                ],
            ),
            fx.Source(
                'CheapSource',
                outputs=[
                    fx.Flow('heat', bus='Source', effects_per_flow_hour=1),
                ],
            ),
            fx.Transmission(
                'Pipe',
                in1=fx.Flow('in', bus='Source', size=200),
                out1=fx.Flow('out', bus='Sink', size=200),
                relative_losses=0.1,
            ),
        )
        fs = optimize(fs)
        # demand=100, with 10% loss: source = 100 / 0.9 ≈ 111.11
        # cost ≈ 111.11
        expected_cost = 100 / 0.9
        assert_allclose(fs.solution['costs'].item(), expected_cost, rtol=1e-4)

    def test_transmission_absolute_losses(self, optimize):
        """Proves: absolute_losses adds fixed loss when transmission is active.

        Transmission with absolute_losses=5. When active, loses 5 kW regardless of flow.
        Demand=20 each hour. Source must provide 20+5=25 when transmission active.

        Sensitivity: Without absolute losses, source=40 for demand=40.
        With absolute_losses=5, source=50 (40 + 2×5).
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Source'),
            fx.Bus('Sink'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Sink', size=1, fixed_relative_profile=np.array([20, 20])),
                ],
            ),
            fx.Source(
                'CheapSource',
                outputs=[
                    fx.Flow('heat', bus='Source', effects_per_flow_hour=1),
                ],
            ),
            fx.Transmission(
                'Pipe',
                in1=fx.Flow('in', bus='Source', size=200),
                out1=fx.Flow('out', bus='Sink', size=200),
                absolute_losses=5,
            ),
        )
        fs = optimize(fs)
        # demand=40, absolute_losses=5 per active hour × 2 = 10
        # source = 40 + 10 = 50
        assert_allclose(fs.solution['costs'].item(), 50.0, rtol=1e-4)

    def test_transmission_bidirectional(self, optimize):
        """Proves: Bidirectional transmission allows flow in both directions.

        Two sources on opposite ends. Demand shifts between buses.
        Optimizer routes through transmission to use cheaper source.

        Sensitivity: Without bidirectional, each bus must use local source.
        With bidirectional, cheap source can serve both sides.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Left'),
            fx.Bus('Right'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'LeftDemand',
                inputs=[
                    fx.Flow('heat', bus='Left', size=1, fixed_relative_profile=np.array([20, 0])),
                ],
            ),
            fx.Sink(
                'RightDemand',
                inputs=[
                    fx.Flow('heat', bus='Right', size=1, fixed_relative_profile=np.array([0, 20])),
                ],
            ),
            fx.Source(
                'LeftSource',
                outputs=[
                    fx.Flow('heat', bus='Left', effects_per_flow_hour=1),
                ],
            ),
            fx.Source(
                'RightSource',
                outputs=[
                    fx.Flow('heat', bus='Right', effects_per_flow_hour=10),  # Expensive
                ],
            ),
            fx.Transmission(
                'Link',
                in1=fx.Flow('left', bus='Left', size=100),
                out1=fx.Flow('right', bus='Right', size=100),
                in2=fx.Flow('right_in', bus='Right', size=100),
                out2=fx.Flow('left_out', bus='Left', size=100),
            ),
        )
        fs = optimize(fs)
        # t=0: LeftDemand=20 from LeftSource @1€ = 20
        # t=1: RightDemand=20 from LeftSource via Transmission @1€ = 20
        # total = 40 (vs 20+200=220 if only local sources)
        assert_allclose(fs.solution['costs'].item(), 40.0, rtol=1e-5)

    def test_transmission_prevent_simultaneous_bidirectional(self, optimize):
        """Proves: prevent_simultaneous_flows_in_both_directions=True prevents both
        directions from being active at the same timestep.

        Two buses, demands alternate sides. Bidirectional transmission with
        prevent_simultaneous=True. Structural check: at no timestep both directions active.

        Sensitivity: Constraint is structural. Cost = 40 (same as unrestricted).
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Left'),
            fx.Bus('Right'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'LeftDemand',
                inputs=[
                    fx.Flow('heat', bus='Left', size=1, fixed_relative_profile=np.array([20, 0])),
                ],
            ),
            fx.Sink(
                'RightDemand',
                inputs=[
                    fx.Flow('heat', bus='Right', size=1, fixed_relative_profile=np.array([0, 20])),
                ],
            ),
            fx.Source(
                'LeftSource',
                outputs=[fx.Flow('heat', bus='Left', effects_per_flow_hour=1)],
            ),
            fx.Transmission(
                'Link',
                in1=fx.Flow('left', bus='Left', size=100),
                out1=fx.Flow('right', bus='Right', size=100),
                in2=fx.Flow('right_in', bus='Right', size=100),
                out2=fx.Flow('left_out', bus='Left', size=100),
                prevent_simultaneous_flows_in_both_directions=True,
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['costs'].item(), 40.0, rtol=1e-5)
        # Structural check: at no timestep both directions active
        in1 = fs.solution['Link(left)|flow_rate'].values[:-1]
        in2 = fs.solution['Link(right_in)|flow_rate'].values[:-1]
        for t in range(len(in1)):
            assert not (in1[t] > 1e-5 and in2[t] > 1e-5), f'Simultaneous bidirectional flow at t={t}'

    def test_transmission_status_startup_cost(self, optimize):
        """Proves: StatusParameters on Transmission applies startup cost
        when the transmission transitions to active.

        Demand=[20, 0, 20, 0] through Transmission with effects_per_startup=50.
        previous_flow_rate=0 and relative_minimum=0.1 force on/off cycling.
        2 startups × 50 + energy 40.

        Sensitivity: Without startup cost, cost=40 (energy only).
        With 50€/startup × 2, cost=140.
        """
        fs = make_flow_system(4)
        fs.add_elements(
            fx.Bus('Source'),
            fx.Bus('Sink'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Sink', size=1, fixed_relative_profile=np.array([20, 0, 20, 0])),
                ],
            ),
            fx.Source(
                'CheapSource',
                outputs=[fx.Flow('heat', bus='Source', effects_per_flow_hour=1)],
            ),
            fx.Transmission(
                'Pipe',
                in1=fx.Flow('in', bus='Source', size=200, previous_flow_rate=0, relative_minimum=0.1),
                out1=fx.Flow('out', bus='Sink', size=200, previous_flow_rate=0, relative_minimum=0.1),
                status_parameters=fx.StatusParameters(effects_per_startup=50),
            ),
        )
        fs = optimize(fs)
        # energy = 40, 2 startups × 50 = 100. Total = 140.
        assert_allclose(fs.solution['costs'].item(), 140.0, rtol=1e-5)


class TestHeatPump:
    """Tests for HeatPump component with COP."""

    def test_heatpump_cop(self, optimize):
        """Proves: HeatPump correctly applies COP to compute electrical consumption.

        HeatPump with cop=3. For 30 kW heat, needs 10 kW electricity.

        Sensitivity: If COP were ignored (=1), elec=30 → cost=30.
        With cop=3, elec=10 → cost=10.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([30, 30])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour=1),
                ],
            ),
            fx.linear_converters.HeatPump(
                'HP',
                cop=3.0,
                electrical_flow=fx.Flow('elec', bus='Elec'),
                thermal_flow=fx.Flow('heat', bus='Heat'),
            ),
        )
        fs = optimize(fs)
        # heat=60, cop=3 → elec=20, cost=20
        assert_allclose(fs.solution['costs'].item(), 20.0, rtol=1e-5)

    def test_heatpump_variable_cop(self, optimize):
        """Proves: HeatPump accepts time-varying COP array.

        cop=[2, 4]. t=0: 20kW heat needs 10kW elec. t=1: 20kW heat needs 5kW elec.

        Sensitivity: If scalar cop=3 used, elec=13.33. Only time-varying gives 15.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([20, 20])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour=1),
                ],
            ),
            fx.linear_converters.HeatPump(
                'HP',
                cop=np.array([2.0, 4.0]),
                electrical_flow=fx.Flow('elec', bus='Elec'),
                thermal_flow=fx.Flow('heat', bus='Heat'),
            ),
        )
        fs = optimize(fs)
        # t=0: 20/2=10, t=1: 20/4=5, total elec=15, cost=15
        assert_allclose(fs.solution['costs'].item(), 15.0, rtol=1e-5)


class TestCoolingTower:
    """Tests for CoolingTower component."""

    def test_cooling_tower_specific_electricity(self, optimize):
        """Proves: CoolingTower correctly applies specific_electricity_demand.

        CoolingTower with specific_electricity_demand=0.1 (kWel/kWth).
        For 100 kWth rejected, needs 10 kWel.

        Sensitivity: If specific_electricity_demand ignored, cost=0.
        With specific_electricity_demand=0.1, cost=20 for 200 kWth.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Source(
                'HeatSource',
                outputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([100, 100])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[
                    fx.Flow('elec', bus='Elec', effects_per_flow_hour=1),
                ],
            ),
            fx.linear_converters.CoolingTower(
                'CT',
                specific_electricity_demand=0.1,  # 0.1 kWel per kWth
                thermal_flow=fx.Flow('heat', bus='Heat'),
                electrical_flow=fx.Flow('elec', bus='Elec'),
            ),
        )
        fs = optimize(fs)
        # heat=200, specific_elec=0.1 → elec = 200 * 0.1 = 20
        assert_allclose(fs.solution['costs'].item(), 20.0, rtol=1e-5)


class TestPower2Heat:
    """Tests for Power2Heat component."""

    def test_power2heat_efficiency(self, optimize):
        """Proves: Power2Heat applies thermal_efficiency to electrical input.

        Power2Heat with thermal_efficiency=0.9. Demand=40 heat over 2 timesteps.
        Elec needed = 40 / 0.9 ≈ 44.44.

        Sensitivity: If efficiency ignored (=1), elec=40 → cost=40.
        With eta=0.9, elec=44.44 → cost≈44.44.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([20, 20])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[fx.Flow('elec', bus='Elec', effects_per_flow_hour=1)],
            ),
            fx.linear_converters.Power2Heat(
                'P2H',
                thermal_efficiency=0.9,
                electrical_flow=fx.Flow('elec', bus='Elec'),
                thermal_flow=fx.Flow('heat', bus='Heat'),
            ),
        )
        fs = optimize(fs)
        # heat=40, eta=0.9 → elec = 40/0.9 ≈ 44.44
        assert_allclose(fs.solution['costs'].item(), 40.0 / 0.9, rtol=1e-5)


class TestHeatPumpWithSource:
    """Tests for HeatPumpWithSource component with COP and heat source."""

    def test_heatpump_with_source_cop(self, optimize):
        """Proves: HeatPumpWithSource applies COP to compute electrical consumption,
        drawing the remainder from a heat source.

        HeatPumpWithSource cop=3. Demand=60 heat over 2 timesteps.
        Elec = 60/3 = 20. Heat source provides 60 - 20 = 40.

        Sensitivity: If cop=1, elec=60 → cost=60. With cop=3, cost=20.
        """
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Bus('Elec'),
            fx.Bus('HeatSource'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('heat', bus='Heat', size=1, fixed_relative_profile=np.array([30, 30])),
                ],
            ),
            fx.Source(
                'Grid',
                outputs=[fx.Flow('elec', bus='Elec', effects_per_flow_hour=1)],
            ),
            fx.Source(
                'FreeHeat',
                outputs=[fx.Flow('heat', bus='HeatSource')],
            ),
            fx.linear_converters.HeatPumpWithSource(
                'HP',
                cop=3.0,
                electrical_flow=fx.Flow('elec', bus='Elec'),
                heat_source_flow=fx.Flow('source', bus='HeatSource'),
                thermal_flow=fx.Flow('heat', bus='Heat'),
            ),
        )
        fs = optimize(fs)
        # heat=60, cop=3 → elec=20, cost=20
        assert_allclose(fs.solution['costs'].item(), 20.0, rtol=1e-5)


class TestSourceAndSink:
    """Tests for SourceAndSink component (e.g. grid connection for buy/sell)."""

    def test_source_and_sink_prevent_simultaneous(self, optimize):
        """Proves: SourceAndSink with prevent_simultaneous_flow_rates=True prevents
        buying and selling in the same timestep.

        Solar=[30, 30, 0]. Demand=[10, 10, 10]. GridConnection: buy @5€, sell @-1€.
        t0,t1: excess 20 → sell 20 (revenue 20 each = -40). t2: deficit 10 → buy 10 (50).

        Sensitivity: Cost = 50 - 40 = 10.
        """
        fs = make_flow_system(3)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Sink(
                'Demand',
                inputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                ],
            ),
            fx.Source(
                'Solar',
                outputs=[
                    fx.Flow('elec', bus='Elec', size=1, fixed_relative_profile=np.array([30, 30, 0])),
                ],
            ),
            fx.SourceAndSink(
                'GridConnection',
                outputs=[fx.Flow('buy', bus='Elec', size=100, effects_per_flow_hour=5)],
                inputs=[fx.Flow('sell', bus='Elec', size=100, effects_per_flow_hour=-1)],
                prevent_simultaneous_flow_rates=True,
            ),
        )
        fs = optimize(fs)
        # t0: sell 20 → -20€. t1: sell 20 → -20€. t2: buy 10 → 50€. Total = 10€.
        assert_allclose(fs.solution['costs'].item(), 10.0, rtol=1e-5)
