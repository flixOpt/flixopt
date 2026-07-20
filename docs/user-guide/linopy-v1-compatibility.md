# linopy v1 Semantics Compatibility

!!! info "Who this is for"
    This is a **contributor / maintainer** guide. flixopt end users never
    interact with linopy directly, so nothing in your models or scripts changes.
    This page documents how flixopt's constraint-construction code stays correct
    under linopy's upcoming *v1 arithmetic convention* and how to keep new code
    compatible.

---

## Background

[linopy PR #717](https://github.com/PyPSA/linopy/pull/717) introduces a global
switch:

```python
import linopy
linopy.options["semantics"] = "legacy"  # current default
linopy.options["semantics"] = "v1"      # opt-in; becomes default before linopy 1.0
```

Legacy linopy silently *guessed* in three situations where a guess can quietly
change a model. **v1** replaces each silent guess with an explicit rule and
**raises** where legacy would have guessed, so a wrong model surfaces as a build
error instead of a wrong number at solve time:

| Legacy behaviour | Under v1 |
|---|---|
| Coordinate mismatch on a shared dim → aligned by position / inner join | **Raises** — align by label explicitly |
| A conflicting auxiliary (non-dimension) coordinate → silently dropped | **Raises** — drop or relabel explicitly |
| `NaN` in a constant / a shifted-in variable → filled with `0` (or `1` for `/`) | **Raises** / absence propagates |

flixopt builds and solves **identically** under both settings. The full test
suite is verified under both `"legacy"` and `"v1"`.

---

## The three patterns flixopt uses

### 1. Adjacent-step constraints → `modeling._lead`

Constraints that relate `x[t+1]` to `x[t]` (storage balance, on/off duration
tracking, state transitions, level tracking, linked-period sizing) slice the
same variable two ways:

```python
x.isel(time=slice(1, None))    # x[t+1] — labels t[1:]
x.isel(time=slice(None, -1))   # x[t]   — labels t[:-1]
```

The two slices carry **different labels** on `time`, so combining them is a
coordinate mismatch. Legacy aligned them by position; v1 raises.

**Fix:** relabel the leading slice onto the trailing slice's labels with the
`_lead` helper:

```python
from flixopt.modeling import _lead

# Before (relies on legacy positional alignment):
x.isel(time=slice(1, None)) - x.isel(time=slice(None, -1))

# After (explicit, identical under both semantics):
_lead(x, 'time') - x.isel(time=slice(None, -1))
```

`_lead(var, dim)` returns `var[dim, 1:]` relabelled onto the `var[dim, :-1]`
coordinate. The positional pairing is unchanged (value at index *i* still pairs
with value at index *i*); the resulting constraint keeps the `labels[:-1]`
coordinate. This is exactly the `.assign_coords` fix linopy's migration guide
prescribes for a shared-dim label mismatch.

!!! warning "Do **not** reach for `.shift()` here"
    `x - x.shift(time=1)` looks tempting — it keeps the full coordinate — but its
    boundary handling **differs between legacy and v1**: legacy fills the
    shifted-in slot with `0` (creating a real `x[t0] ≤ …` constraint row), while
    v1 marks that row absent (dropped). That is precisely the legacy↔v1
    divergence we must avoid, and the boundary of these constraints is handled by
    a **separate** initial constraint anyway. Use `_lead` / `.assign_coords`.

### 2. Scalar-endpoint constraints → `drop=True`

A scalar integer index keeps the indexed coordinate as a **leftover scalar
aux-coord**:

```python
x.isel(time=0)    # drops the `time` *dimension*, keeps `time` = first timestamp
x.isel(time=-1)   # keeps `time` = last timestamp
```

Cyclic / initial / final constraints compare two such endpoints:

```python
x.isel(time=0) == x.isel(time=-1)   # `time` = first vs last → conflict
```

Legacy silently drops the conflicting coord; v1 raises. **Fix:** pass
`drop=True` so the meaningless scalar coord never appears:

```python
x.isel(time=0, drop=True) == x.isel(time=-1, drop=True)
```

This applies to any scalar `.isel(...)` whose result is combined with another
operand — `time`, `scenario`, `cluster_boundary`, and to a value *derived* from
such a select (e.g. `previous_status.isel(time=-1, drop=True)`).

### 3. Constraint mutation → `Constraint.update`

The `Constraint.lhs` / `.rhs` **setters** are deprecated in linopy 0.8. Use
`update()`:

```python
# Before:
con.lhs += extra_term

# After:
con.update(lhs=con.lhs + extra_term)
```

---

## Writing v1-safe constraints (checklist)

- Combining two slices of the same variable along a dim? → use `_lead`, never
  raw `isel(slice(1, None))` against `isel(slice(None, -1))`.
- Scalar `.isel(dim=k)` whose result feeds arithmetic or a comparison? → add
  `drop=True`.
- Mutating an existing constraint? → `con.update(lhs=...)`, not `con.lhs = ...`.
- Passing a plain numpy array / list into linopy arithmetic? → wrap it in a
  named `DataArray` so it aligns by label, not by size.
- Never introduce a user-supplied `NaN` as a stand-in for "absent" — express
  absence with `mask=` / `.where(...)` on the variable.

---

## Testing under v1

Run the suite with the option flipped (e.g. via a `conftest`/plugin that sets
`linopy.options["semantics"] = "v1"` at configure time), and turn the legacy
warning into an error to surface every remaining site under legacy:

```python
import warnings
from linopy import LinopySemanticsWarning
warnings.filterwarnings("error", category=LinopySemanticsWarning)
```

A green run under **both** `"legacy"` and `"v1"` is the compatibility contract.

---

## Known non-issue: cyclic-cluster SOC degeneracy

A handful of clustered-storage tests assert an exact absolute charge-state
trajectory. In `cluster_mode='cyclic'` the initial/final charge-state
constraints are intentionally skipped, so the **absolute** SOC level of each
representative cluster is unpinned — only the per-cluster *delta* is fixed. The
problem is therefore degenerate in the SOC level: multiple equally-optimal
solutions exist, and different solver/linopy builds may return a different (but
equally valid) vertex. Objective and all flow rates are unaffected. This is a
test-robustness matter, **independent of the semantics setting** (it reproduces
identically under `"legacy"` and `"v1"`).
