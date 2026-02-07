# Linopy Variable Names

Overview of all `add_variables()` calls in the production codebase.

Variable names are now **explicit and fully qualified** at all call sites — no auto-prefixing.
`TypeModel` is subscriptable: `self['flow|rate']` returns the linopy variable.

## elements.py — FlowsModel (prefix `flow|`)

| Assigned To | `name=` | Dims |
|-------------|---------|------|
| `rate` | `'flow\|rate'` | implicit temporal |
| `status` | `'flow\|status'` | implicit temporal |
| `size` | `'flow\|size'` | `('period','scenario')` |
| `invested` | `'flow\|invested'` | `('period','scenario')` |
| `active_hours` | `'flow\|active_hours'` | `('period','scenario')` |
| `startup` | `'flow\|startup'` | implicit temporal |
| `shutdown` | `'flow\|shutdown'` | implicit temporal |
| `inactive` | `'flow\|inactive'` | implicit temporal |
| `startup_count` | `'flow\|startup_count'` | `('period','scenario')` |
| `share_var` | `f'{name_prefix}\|share'` | — |

## elements.py — BusesModel (prefix `bus|`)

| Assigned To | `name=` | Dims |
|-------------|---------|------|
| (via add_variables) | `'bus\|virtual_supply'` | temporal_dims |
| (via add_variables) | `'bus\|virtual_demand'` | temporal_dims |
| `share_var` | `f'{label}->Penalty(temporal)'` | — |

## elements.py — ComponentsModel (prefix `component|`)

| Assigned To | `name=` | Dims |
|-------------|---------|------|
| (via add_variables) | `'component\|status'` | implicit temporal |
| `active_hours` | `'component\|active_hours'` | `('period','scenario')` |
| `startup` | `'component\|startup'` | implicit temporal |
| `shutdown` | `'component\|shutdown'` | implicit temporal |
| `inactive` | `'component\|inactive'` | implicit temporal |
| `startup_count` | `'component\|startup_count'` | `('period','scenario')` |

## components.py — StoragesModel (prefix `storage|`)

| Assigned To | `name=` | Dims |
|-------------|---------|------|
| `charge_state` | `'storage\|charge'` | extra_timestep |
| `netto_discharge` | `'storage\|netto'` | temporal |
| `size_var` | `'storage\|size'` | — |
| `invested_var` | `'storage\|invested'` | — |
| `share_var` | `f'{prefix}\|share'` | — |

## components.py — InterclusterStoragesModel (prefix `intercluster_storage|`)

| Assigned To | `name=` | Dims |
|-------------|---------|------|
| `charge_state` | `f'{dim}\|charge_state'` | extra_timestep |
| `netto_discharge` | `f'{dim}\|netto_discharge'` | temporal |
| `soc_boundary` | `f'{dim}\|SOC_boundary'` | — |
| `size_var` | `f'{dim}\|size'` | — |
| `invested_var` | `f'{dim}\|invested'` | — |

## effects.py

| Assigned To | `name=` | Dims |
|-------------|---------|------|
| `self.periodic` | `'effect\|periodic'` | periodic_coords |
| `self.temporal` | `'effect\|temporal'` | periodic_coords |
| `self.per_timestep` | `'effect\|per_timestep'` | temporal_coords |
| `self.total` | `'effect\|total'` | periodic_coords |
| `self.total_over_periods` | `'effect\|total_over_periods'` | over_periods_coords |
| `var` | `name` (param) | coords (param) |

## features.py

| Assigned To | `name=` | Dims |
|-------------|---------|------|
| `inside_piece` | `f'{prefix}\|inside_piece'` | full_coords |
| `lambda0` | `f'{prefix}\|lambda0'` | full_coords |
| `lambda1` | `f'{prefix}\|lambda1'` | full_coords |

## modeling.py

| Assigned To | `name=` | Dims |
|-------------|---------|------|
| `tracker` | `name` (param) | coords |
| `duration` | `name` (param) | state.coords |

## Access Patterns

```python
# TypeModel is subscriptable
rate = flows_model['flow|rate']           # __getitem__
exists = 'flow|status' in flows_model     # __contains__
size = storages_model.get('storage|size') # .get() with default

# Cross-model access
flow_rate = self._flows_model['flow|rate']

# get_variable() with optional element slicing
rate_for_boiler = flows_model.get_variable('flow|rate', 'Boiler(gas_in)')
```

## Naming Conventions

1. **Pipe-delimited hierarchy**: All names use `'type|variable'` — e.g. `'flow|rate'`, `'storage|charge'`, `'component|status'`
2. **Consistent across all models**: No more bare names — every variable has its type prefix
3. **`netto` vs `net`**: `'netto'` (German/Dutch) used instead of English `'net'`
4. **Special separator**: `f'{label}->Penalty(temporal)'` uses `->` instead of `|`
