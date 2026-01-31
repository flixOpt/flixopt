# Linopy Variable Names

Overview of all `add_variables()` calls in the production codebase.

## elements.py

| Line | Assigned To | `name=` | Dims |
|------|-------------|---------|------|
| 786 | (return) | `'rate'` | implicit temporal |
| 799 | (return) | `'status'` | implicit temporal |
| 812 | (return) | `'size'` | `('period','scenario')` |
| 826 | (return) | `'invested'` | `('period','scenario')` |
| 1186 | `share_var` | `f'{name_prefix}\|share'` | — |
| 1288 | (return) | `'active_hours'` | `('period','scenario')` |
| 1302 | (return) | `'startup'` | implicit temporal |
| 1310 | (return) | `'shutdown'` | implicit temporal |
| 1318 | (return) | `'inactive'` | implicit temporal |
| 1326 | (return) | `'startup_count'` | `('period','scenario')` |
| 1605 | (return) | `'virtual_supply'` | temporal_dims |
| 1614 | (return) | `'virtual_demand'` | temporal_dims |
| 1707 | `share_var` | `f'{label}->Penalty(temporal)'` | — |
| 1852 | (return) | `'status'` | implicit temporal |
| 1985 | (return) | `'active_hours'` | `('period','scenario')` |
| 1999 | (return) | `'startup'` | implicit temporal |
| 2007 | (return) | `'shutdown'` | implicit temporal |
| 2015 | (return) | `'inactive'` | implicit temporal |
| 2023 | (return) | `'startup_count'` | `('period','scenario')` |

## components.py

| Line | Assigned To | `name=` | Dims |
|------|-------------|---------|------|
| 1040 | `charge_state` | `'storage\|charge'` | extra_timestep |
| 1054 | `netto_discharge` | `'storage\|netto'` | temporal |
| 1366 | `size_var` | `'storage\|size'` | — |
| 1382 | `invested_var` | `'storage\|invested'` | — |
| 1620 | `share_var` | `f'{prefix}\|share'` | — |
| 1725 | `charge_state` | `f'{dim}\|charge_state'` | extra_timestep |
| 1735 | `netto_discharge` | `f'{dim}\|netto_discharge'` | temporal |
| 1800 | `soc_boundary` | `f'{dim}\|SOC_boundary'` | — |
| 2076 | `size_var` | `f'{dim}\|size'` | — |
| 2091 | `invested_var` | `f'{dim}\|invested'` | — |

## effects.py

| Line | Assigned To | `name=` | Dims |
|------|-------------|---------|------|
| 410 | `self.periodic` | `'effect\|periodic'` | periodic_coords |
| 423 | `self.temporal` | `'effect\|temporal'` | periodic_coords |
| 446 | `self.per_timestep` | `'effect\|per_timestep'` | temporal_coords |
| 462 | `self.total` | `'effect\|total'` | periodic_coords |
| 494 | `self.total_over_periods` | `'effect\|total_over_periods'` | over_periods_coords |
| 595 | `var` | `name` (param) | coords (param) |

## features.py

| Line | Assigned To | `name=` | Dims |
|------|-------------|---------|------|
| 732 | `inside_piece` | `f'{prefix}\|inside_piece'` | full_coords |
| 741 | `lambda0` | `f'{prefix}\|lambda0'` | full_coords |
| 748 | `lambda1` | `f'{prefix}\|lambda1'` | full_coords |

## modeling.py

| Line | Assigned To | `name=` | Dims |
|------|-------------|---------|------|
| 332/336 | `tracker` | `name` (param) | coords |
| 404 | `duration` | `name` (param) | state.coords |

## structure.py

| Line | Assigned To | `name=` | Dims |
|------|-------------|---------|------|
| 609 | `variable` | `full_name` | coords (param) |

## Naming Pattern Observations

1. **Pipe-delimited hierarchy**: Most names use `'category|variable'` — e.g. `'storage|charge'`, `'effect|total'`
2. **Inconsistency in elements.py vs components.py**: elements.py uses bare names (`'rate'`, `'status'`, `'size'`) while components.py uses prefixed names (`'storage|charge'`, `'storage|size'`)
3. **Duplicate logic**: On/Off variables appear twice in elements.py (lines ~1288-1326 and ~1985-2023) with identical names
4. **`netto` vs `net`**: `'netto'` (German/Dutch) used instead of English `'net'`
5. **Inconsistent charge naming**: `'storage|charge'` (line 1040) vs `f'{dim}|charge_state'` (line 1725)
6. **Special separator**: `f'{label}->Penalty(temporal)'` uses `->` instead of `|`
