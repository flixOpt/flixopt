# Plan: Comprehensive test_math Coverage Expansion

All tests use the existing `optimize` fixture (3 modes: `solve`, `save->reload->solve`, `solve->save->reload`).

---

## Part A — Single-period gaps

### A1. Storage (`test_storage.py`, existing `TestStorage`)

- [ ] **`test_storage_relative_minimum_charge_state`**
  - 3 ts, Grid=[1, 100, 1], Demand=[0, 80, 0]
  - Storage: capacity=100, initial=0, **relative_minimum_charge_state=0.3**
  - SOC must stay >= 30. Charge 100 @t0, discharge max 70 @t1, grid covers 10 @100.
  - **Cost = 1100** (without: 80)

- [ ] **`test_storage_maximal_final_charge_state`**
  - 2 ts, Bus imbalance_penalty=5, Grid=[1,100], Demand=[0, 50]
  - Storage: capacity=100, initial=80, **maximal_final_charge_state=20**
  - Must discharge 60 (demand 50 + 10 excess penalized @5).
  - **Cost = 50** (without: 0)

- [ ] **`test_storage_relative_minimum_final_charge_state`**
  - 2 ts, Grid=[1, 100], Demand=[0, 50]
  - Storage: capacity=100, initial=0, **relative_minimum_final_charge_state=0.7**
  - Final SOC >= 70. Charge 100, discharge 30, grid covers 20 @100.
  - **Cost = 2100** (without: 50)

- [ ] **`test_storage_relative_maximum_final_charge_state`**
  - Same as maximal_final but relative: **relative_maximum_final_charge_state=0.2** on capacity=100.
  - **Cost = 50** (without: 0)

- [ ] **`test_storage_balanced_invest`**
  - 3 ts, Grid=[1, 100, 100], Demand=[0, 80, 80]
  - Storage: capacity=200, initial=0, **balanced=True**
    - charge: InvestParams(max=200, per_size=0.5)
    - discharge: InvestParams(max=200, per_size=0.5)
  - Balanced forces charge_size = discharge_size = 160. Invest=160. Grid=160.
  - **Cost = 320** (without balanced: 280, since discharge_size could be 80)

### A2. Transmission (`test_components.py`, existing `TestTransmission`)

- [ ] **`test_transmission_prevent_simultaneous_bidirectional`**
  - 2 ts, 2 buses. Demand alternates sides.
  - **prevent_simultaneous_flows_in_both_directions=True**
  - Structural check: at no timestep both directions active.
  - **Cost = 40** (same as unrestricted in this case; constraint is structural)

- [ ] **`test_transmission_status_startup_cost`**
  - 4 ts, Demand=[20, 0, 20, 0] through Transmission
  - **status_parameters=StatusParameters(effects_per_startup=50)**
  - 2 startups * 50 + energy 40.
  - **Cost = 140** (without: 40)

### A3. New component classes (`test_components.py`)

- [ ] **`TestPower2Heat` — `test_power2heat_efficiency`**
  - 2 ts, Demand=[20, 20], Grid @1
  - Power2Heat: thermal_efficiency=0.9
  - Elec = 40/0.9 = 44.44
  - **Cost = 40/0.9** (without eta: 40)

- [ ] **`TestHeatPumpWithSource` — `test_heatpump_with_source_cop`**
  - 2 ts, Demand=[30, 30], Grid @1 (elec), free heat source
  - HeatPumpWithSource: cop=3. Elec = 60/3 = 20.
  - **Cost = 20** (with cop=1: 60)

- [ ] **`TestSourceAndSink` — `test_source_and_sink_prevent_simultaneous`**
  - 3 ts, Solar=[30, 30, 0], Demand=[10, 10, 10]
  - SourceAndSink `GridConnection`: buy @5, sell @-1, prevent_simultaneous=True
  - t0,t1: sell 20 (revenue 20 each). t2: buy 10 (cost 50).
  - **Cost = 10** (50 - 40 revenue)

### A4. Flow status (`test_flow_status.py`)

- [ ] **`test_max_uptime_standalone`**
  - 5 ts, Demand=[10]*5
  - CheapBoiler eta=1.0, **StatusParameters(max_uptime=2)**, previous_flow_rate=0
  - ExpensiveBoiler eta=0.5 (backup)
  - Cheap: on(0,1), off(2), on(3,4) = 40 fuel. Expensive covers t2: 20 fuel.
  - **Cost = 60** (without: 50)

---

## Part B — Multi-period, scenarios, clustering

### B1. conftest.py helpers

```python
def make_multi_period_flow_system(n_timesteps=3, periods=None, weight_of_last_period=None):
    ts = pd.date_range('2020-01-01', periods=n_timesteps, freq='h')
    if periods is None:
        periods = [2020, 2025]
    return fx.FlowSystem(ts, periods=pd.Index(periods, name='period'),
                         weight_of_last_period=weight_of_last_period)

def make_scenario_flow_system(n_timesteps=3, scenarios=None, scenario_weights=None):
    ts = pd.date_range('2020-01-01', periods=n_timesteps, freq='h')
    if scenarios is None:
        scenarios = ['low', 'high']
    return fx.FlowSystem(ts, scenarios=pd.Index(scenarios, name='scenario'),
                         scenario_weights=scenario_weights)
```

**Note:** Multi-period objective assertion — `fs.solution['costs'].item()` only works for scalar results. For multi-period, need to verify how to access the total objective (e.g., `fs.solution['objective'].item()` or `fs.model.model.objective.value`). Verify during implementation.

### B2. Multi-period (`test_multi_period.py`, new `TestMultiPeriod`)

- [ ] **`test_period_weights_affect_objective`**
  - 2 ts, periods=[2020, 2025], weight_of_last_period=5
  - Grid @1, Demand=[10, 10]. Per-period cost=20. Weights=[5, 5].
  - **Objective = 200** (10*20 would be wrong if weights not applied)

- [ ] **`test_flow_hours_max_over_periods`**
  - 2 ts, periods=[2020, 2025], weight_of_last_period=5
  - Dirty @1, Clean @10. Demand=[10, 10].
  - Dirty flow: **flow_hours_max_over_periods=50**
  - Weights [5,5]: 5*fh0 + 5*fh1 <= 50 => fh0+fh1 <= 10.
  - Dirty 5/period, Clean 15/period. Per-period cost=155.
  - **Objective = 1550** (without: 200)

- [ ] **`test_flow_hours_min_over_periods`**
  - Same setup but **flow_hours_min_over_periods=50** on expensive source.
  - Forces min production from expensive source.
  - **Objective = 650** (without: 200)

- [ ] **`test_effect_maximum_over_periods`**
  - CO2 effect with **maximum_over_periods=50**, Dirty emits CO2=1/kWh.
  - Same math as flow_hours_max: caps total dirty across periods.
  - **Objective = 1550** (without: 200)

- [ ] **`test_effect_minimum_over_periods`**
  - CO2 with **minimum_over_periods=50**, both sources @1 cost, imbalance_penalty=0.
  - Demand=[2, 2]. Must overproduce dirty to meet min CO2.
  - **Objective = 50** (without: 40)

- [ ] **`test_invest_linked_periods`**
  - InvestParameters with **linked_periods=(2020, 2025)**.
  - Verify invested sizes equal across periods (structural check).

- [ ] **`test_effect_period_weights`**
  - costs effect with **period_weights=[1, 10]** (overrides default [5, 5]).
  - Grid @1, Demand=[10, 10]. Per-period cost=20.
  - **Objective = 1*20 + 10*20 = 220** (default weights would give 200)

### B3. Scenarios (`test_scenarios.py`, new `TestScenarios`)

- [ ] **`test_scenario_weights_affect_objective`**
  - 2 ts, scenarios=['low', 'high'], weights=[0.3, 0.7]
  - Demand: low=[10, 10], high=[30, 30] (xr.DataArray with scenario dim)
  - **Objective = 0.3*20 + 0.7*60 = 48**

- [ ] **`test_scenario_independent_sizes`**
  - Same setup + InvestParams on flow.
  - With **scenario_independent_sizes=True**: same size forced across scenarios.
  - Size=30 (peak high). Invest cost weighted=30. Ops=48.
  - **Objective = 78** (without: 72, where low invests 10, high invests 30)

- [ ] **`test_scenario_independent_flow_rates`**
  - **scenario_independent_flow_rates=True**, weights=[0.5, 0.5]
  - Flow rates must match across scenarios. Rate=30 (max of demands).
  - **Objective = 60** (without: 40)

### B4. Clustering (`test_clustering.py`, new `TestClustering`)

These tests are structural/approximate (clustering is heuristic). Require `tsam` (`pytest.importorskip`).

- [ ] **`test_clustering_basic_objective`**
  - 48 ts, cluster to 2 typical days. Compare clustered vs full objective.
  - Assert within 10% tolerance.

- [ ] **`test_storage_cluster_mode_cyclic`**
  - Clustered system with Storage(cluster_mode='cyclic').
  - Structural: SOC start == SOC end within each cluster.

- [ ] **`test_storage_cluster_mode_intercluster`**
  - Storage(cluster_mode='intercluster').
  - Structural: intercluster SOC variables exist, objective differs from cyclic.

- [ ] **`test_status_cluster_mode_cyclic`**
  - Boiler with StatusParameters(cluster_mode='cyclic').
  - Structural: status wraps within each cluster.

---

## Summary

| Section | File | Tests | Type |
|---------|------|-------|------|
| A1 | test_storage.py | 5 | Exact analytical |
| A2 | test_components.py | 2 | Exact analytical |
| A3 | test_components.py | 3 | Exact analytical |
| A4 | test_flow_status.py | 1 | Exact analytical |
| B1 | conftest.py | — | Helpers |
| B2 | test_multi_period.py | 7 | Exact analytical |
| B3 | test_scenarios.py | 3 | Exact analytical |
| B4 | test_clustering.py | 4 | Approximate/structural |

**Total: 25 new tests** (x3 optimize modes = 75 test runs)

## Implementation order
1. conftest.py helpers (B1)
2. Single-period gaps (A1-A4, independent, can parallelize)
3. Multi-period tests (B2)
4. Scenario tests (B3)
5. Clustering tests (B4)

## Verification
Run `python -m pytest tests/test_math/ -v --tb=short` — all tests should pass across all 3 optimize modes (solve, save->reload->solve, solve->save->reload).
