# Plotting API Design for flixopt

## Overview

This document outlines the design for a new, user-friendly plotting interface for the `Results` class. The API follows a layered approach that serves users at different skill levels while always providing access to the underlying data.

## Design Principles

1. **Data always accessible**: Every plot method returns a `PlotResult` with `.data` and `.figure`
2. **Sensible defaults**: Colors from `results.colors`, time on x-axis, etc.
3. **Consistent interface**: Same kwargs work across plot types
4. **Plotly-only** (for now): Single backend simplifies implementation
5. **Composable**: Can chain modifications before rendering
6. **xarray-native**: Leverage xarray's selection/slicing capabilities

## Architecture

```
Results
├── .plot (PlotAccessor)
│   ├── .balance()
│   ├── .heatmap()
│   ├── .storage()
│   ├── .flows()
│   ├── .compare()
│   ├── .sankey()
│   └── .effects()
│
├── ['Element'] (ComponentResults / BusResults)
│   └── .plot (ElementPlotAccessor)
│       ├── .balance()
│       ├── .heatmap()
│       └── .storage()  # Only for storage components
```

---

## Core Classes

### 1. PlotResult

Container returned by all plot methods. Holds both data and figure.

```python
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go


@dataclass
class PlotResult:
    """Container returned by all plot methods. Holds both data and figure."""

    data: pd.DataFrame
    """Prepared data used for the plot. Ready for export or custom plotting."""

    figure: go.Figure
    """Plotly figure object. Can be modified with update_layout(), update_traces(), etc."""

    def show(self) -> 'PlotResult':
        """Display the figure. Returns self for chaining."""
        self.figure.show()
        return self

    def update(self, **layout_kwargs) -> 'PlotResult':
        """Update figure layout. Returns self for chaining.

        Example:
            result.update(title='Custom Title', height=600).show()
        """
        self.figure.update_layout(**layout_kwargs)
        return self

    def update_traces(self, **trace_kwargs) -> 'PlotResult':
        """Update figure traces. Returns self for chaining."""
        self.figure.update_traces(**trace_kwargs)
        return self

    def to_html(self, path: str | Path) -> 'PlotResult':
        """Save figure as interactive HTML."""
        self.figure.write_html(path)
        return self

    def to_image(self, path: str | Path, **kwargs) -> 'PlotResult':
        """Save figure as static image (png, svg, pdf, etc.)."""
        self.figure.write_image(path, **kwargs)
        return self

    def to_csv(self, path: str | Path, **kwargs) -> 'PlotResult':
        """Export the underlying data to CSV."""
        self.data.to_csv(path, **kwargs)
        return self
```

---

### 2. PlotAccessor

Attached to `Results` as `results.plot`.

```python
from typing import Literal, Any

# Type aliases
SelectType = dict[str, Any]  # xarray-style selection: {'time': slice(...), 'scenario': 'base'}
FilterType = str | list[str]  # For include/exclude: 'Boiler' or ['Boiler', 'CHP']


class PlotAccessor:
    """Plot accessor for Results. Access via results.plot.<method>()"""

    def __init__(self, results: 'Results'):
        self._results = results

    @property
    def colors(self) -> dict[str, str]:
        """Global colors from Results."""
        return self._results.colors
```

---

## Plot Methods

### 2.1 balance()

Plot node balance (inputs vs outputs) for a Bus or Component.

```python
def balance(
    self,
    node: str,
    *,
    # Data selection (xarray-style)
    select: SelectType | None = None,
    # Flow filtering
    include: FilterType | None = None,
    exclude: FilterType | None = None,
    # Data transformation
    unit: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
    aggregate: Literal['sum', 'mean', 'max', 'min'] | None = None,
    # Visual style
    mode: Literal['bar', 'line', 'area'] = 'bar',
    colors: dict[str, str] | None = None,
    # Faceting & animation
    facet_col: str | None = 'scenario',
    facet_row: str | None = None,
    animate_by: str | None = 'period',
    # Display
    show: bool | None = None,  # None = CONFIG.Plotting.default_show
    **plotly_kwargs,
) -> PlotResult:
    """Plot node balance (inputs vs outputs) for a Bus or Component.

    Args:
        node: Label of the Bus or Component to plot.
        select: xarray-style selection dict. Supports:
            - Single values: {'scenario': 'base'}
            - Multiple values: {'scenario': ['base', 'high']}
            - Slices: {'time': slice('2024-01', '2024-06')}
        include: Only include flows matching these patterns (substring match).
        exclude: Exclude flows matching these patterns.
        unit: 'flow_rate' (power, kW) or 'flow_hours' (energy, kWh).
        aggregate: Aggregate over time dimension before plotting.
        mode: Plot style - 'bar', 'line', or 'area'.
        colors: Override colors (merged with global colors).
        facet_col: Dimension for column facets.
        facet_row: Dimension for row facets.
        animate_by: Dimension to animate over.
        show: Whether to display the plot.
        **plotly_kwargs: Passed to plotly express.

    Returns:
        PlotResult with .data (DataFrame) and .figure (go.Figure).

    Examples:
        # Basic usage
        results.plot.balance('ElectricityBus')

        # Select time range
        results.plot.balance('Bus', select={'time': slice('2024-01', '2024-03')})

        # Filter specific flows
        results.plot.balance('Bus', include=['Boiler', 'CHP'], exclude=['Grid'])

        # Energy instead of power
        results.plot.balance('Bus', unit='flow_hours')

        # Aggregate to total
        results.plot.balance('Bus', aggregate='sum', mode='bar')

        # Get data for custom use
        df = results.plot.balance('Bus').data
    """
    ...
```

**DataFrame Schema:**
```
| time | flow | value | direction | [scenario] | [period] |
```

- `time`: pd.DatetimeIndex - Timestep
- `flow`: str - Flow label (e.g., 'Boiler|Q_th')
- `value`: float - Flow rate or flow hours
- `direction`: str - 'input' or 'output'
- `scenario`: str - Optional, if multiple scenarios
- `period`: int - Optional, if multiple periods

---

### 2.2 heatmap()

Plot heatmap of time series data with time reshaping.

```python
def heatmap(
    self,
    variables: str | list[str],
    *,
    # Data selection
    select: SelectType | None = None,
    # Reshaping
    reshape: tuple[str, str] = ('D', 'h'),  # (outer, inner) frequency
    # Visual style
    colorscale: str = 'viridis',
    # Faceting & animation (for multiple variables)
    facet_col: str | None = None,  # 'variable' auto-facets multiple vars
    animate_by: str | None = None,
    # Display
    show: bool | None = None,
    **plotly_kwargs,
) -> PlotResult:
    """Plot heatmap of time series data with time reshaping.

    Args:
        variables: Single variable name or list of variables.
            Example: 'Boiler|on' or ['Boiler|on', 'CHP|on']
        select: xarray-style selection.
        reshape: How to reshape time axis - (outer, inner).
            Common patterns:
            - ('D', 'h'): Days × Hours (default)
            - ('W', 'D'): Weeks × Days
            - ('MS', 'D'): Months × Days
        colorscale: Plotly colorscale name.
        facet_col: Facet dimension. Use 'variable' for multi-var plots.
        animate_by: Animation dimension.
        show: Whether to display.

    Returns:
        PlotResult with reshaped data ready for heatmap.

    Examples:
        # Single variable
        results.plot.heatmap('Boiler|on')

        # Multiple variables with faceting
        results.plot.heatmap(['Boiler|on', 'CHP|on'], facet_col='variable')

        # Weekly pattern
        results.plot.heatmap('Load|flow_rate', reshape=('W', 'h'))
    """
    ...
```

**DataFrame Schema:**
```
| outer | inner | value | [variable] |
```

- `outer`: pd.DatetimeIndex - Outer grouping (e.g., date)
- `inner`: int | str - Inner grouping (e.g., hour)
- `value`: float - Variable value
- `variable`: str - Optional, if multiple variables

---

### 2.3 storage()

Plot storage component with charge state and flow balance.

```python
def storage(
    self,
    component: str,
    *,
    # Data selection
    select: SelectType | None = None,
    # What to show
    show_balance: bool = True,
    show_charge_state: bool = True,
    # Visual style
    mode: Literal['bar', 'line', 'area'] = 'area',
    colors: dict[str, str] | None = None,
    # Faceting
    facet_col: str | None = 'scenario',
    animate_by: str | None = 'period',
    # Display
    show: bool | None = None,
    **plotly_kwargs,
) -> PlotResult:
    """Plot storage component with charge state and flow balance.

    Creates a dual-axis plot showing:
    - Charge/discharge flows (left axis, as area/bar)
    - State of charge (right axis, as line)

    Args:
        component: Storage component label.
        select: xarray-style selection.
        show_balance: Show charge/discharge flows.
        show_charge_state: Show state of charge line.
        mode: Style for balance plot.
        colors: Override colors.
        facet_col: Facet dimension.
        animate_by: Animation dimension.
        show: Whether to display.

    Returns:
        PlotResult with combined storage data.
    """
    ...
```

**DataFrame Schema:**
```
| time | variable | value | [scenario] | [period] |
```

- `time`: pd.DatetimeIndex
- `variable`: str - 'charge_state', 'charge', 'discharge'
- `value`: float
- `scenario`: str - Optional
- `period`: int - Optional

---

### 2.4 flows()

Plot flow rates filtered by start/end nodes or component.

```python
def flows(
    self,
    *,
    # Flow filtering
    start: str | list[str] | None = None,
    end: str | list[str] | None = None,
    component: str | list[str] | None = None,
    # Data selection
    select: SelectType | None = None,
    # Transformation
    unit: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
    aggregate: Literal['sum', 'mean', 'max', 'min'] | None = None,
    # Visual style
    mode: Literal['bar', 'line', 'area'] = 'line',
    colors: dict[str, str] | None = None,
    # Faceting
    facet_col: str | None = None,
    animate_by: str | None = None,
    # Display
    show: bool | None = None,
    **plotly_kwargs,
) -> PlotResult:
    """Plot flow rates filtered by start/end nodes or component.

    Args:
        start: Filter by source node(s).
        end: Filter by destination node(s).
        component: Filter by parent component(s).
        select: xarray-style selection.
        unit: 'flow_rate' or 'flow_hours'.
        aggregate: Aggregate over time.
        mode: Plot style.
        colors: Override colors.

    Examples:
        # All flows from a bus
        results.plot.flows(start='ElectricityBus')

        # Flows for specific component
        results.plot.flows(component='Boiler')

        # Total energy by flow
        results.plot.flows(unit='flow_hours', aggregate='sum')
    """
    ...
```

**DataFrame Schema:**
```
| time | flow | value | start | end | component | [scenario] | [period] |
```

---

### 2.5 compare()

Compare multiple elements side-by-side or overlaid.

```python
def compare(
    self,
    elements: list[str],
    *,
    variable: str = 'flow_rate',
    # Data selection
    select: SelectType | None = None,
    # Visual style
    mode: Literal['overlay', 'facet'] = 'overlay',
    colors: dict[str, str] | None = None,
    # Display
    show: bool | None = None,
    **plotly_kwargs,
) -> PlotResult:
    """Compare multiple elements side-by-side or overlaid.

    Args:
        elements: List of element labels to compare.
        variable: Which variable to compare.
        select: xarray-style selection.
        mode: 'overlay' (same axes) or 'facet' (subplots).
        colors: Override colors.

    Examples:
        results.plot.compare(['Boiler', 'CHP', 'HeatPump'], variable='on')
    """
    ...
```

---

### 2.6 sankey()

Plot Sankey diagram of energy/material flows.

```python
def sankey(
    self,
    *,
    # Time handling
    timestep: int | str | None = None,  # Index, timestamp, or None for sum
    aggregate: Literal['sum', 'mean'] = 'sum',
    # Data selection
    select: SelectType | None = None,
    # Display
    show: bool | None = None,
    **plotly_kwargs,
) -> PlotResult:
    """Plot Sankey diagram of energy/material flows.

    Args:
        timestep: Specific timestep to show, or None for aggregation.
        aggregate: How to aggregate if timestep is None.
        select: xarray-style selection.

    Examples:
        # Total flows over all time
        results.plot.sankey()

        # Specific timestep
        results.plot.sankey(timestep=100)

        # Average flows
        results.plot.sankey(aggregate='mean')
    """
    ...
```

---

### 2.7 effects()

Plot effect (cost, emissions, etc.) breakdown.

```python
def effects(
    self,
    effect: str = 'cost',
    *,
    by: Literal['component', 'flow', 'time'] = 'component',
    # Data selection
    select: SelectType | None = None,
    # Visual style
    mode: Literal['bar', 'pie', 'treemap'] = 'bar',
    colors: dict[str, str] | None = None,
    # Display
    show: bool | None = None,
    **plotly_kwargs,
) -> PlotResult:
    """Plot effect (cost, emissions, etc.) breakdown.

    Args:
        effect: Effect name ('cost', 'emissions', etc.).
        by: Group by 'component', 'flow', or 'time'.
        select: xarray-style selection.
        mode: Chart type.

    Examples:
        results.plot.effects('cost', by='component', mode='pie')
        results.plot.effects('emissions', by='time', mode='area')
    """
    ...
```

---

## Element-Level PlotAccessor

Attached to individual element results (ComponentResults, BusResults).

```python
class ElementPlotAccessor:
    """Plot accessor for individual element results."""

    def __init__(self, element_results: '_ElementResults'):
        self._element = element_results
        self._results = element_results._results

    def balance(self, **kwargs) -> PlotResult:
        """Plot balance for this element. Same kwargs as PlotAccessor.balance()."""
        return self._results.plot.balance(self._element.label, **kwargs)

    def heatmap(self, variable: str | list[str] | None = None, **kwargs) -> PlotResult:
        """Plot heatmap for this element's variables.

        Args:
            variable: Variable suffix (e.g., 'on') or full name.
                      If None, shows all time-series variables.
        """
        # Resolve to full variable names
        ...

    def storage(self, **kwargs) -> PlotResult:
        """Plot storage state (only for storage components)."""
        if not self._element.is_storage:
            raise ValueError(f'{self._element.label} is not a storage component')
        return self._results.plot.storage(self._element.label, **kwargs)
```

---

## Usage Examples

### Quick Plots

```python
from flixopt import Results

results = Results.from_file('results', 'optimization')

# Basic usage - shows immediately (if CONFIG.Plotting.default_show is True)
results.plot.balance('ElectricityBus')
results.plot.storage('Battery')
results.plot.heatmap('Boiler|on')
```

### Customized Plots

```python
# Select time range and scenario
results.plot.balance('Bus',
    select={'time': slice('2024-06', '2024-08'), 'scenario': 'high'},
    include=['Solar', 'Wind'],
    unit='flow_hours',
    mode='area'
)

# Multiple variables in heatmap
results.plot.heatmap(['Boiler|on', 'CHP|on'], facet_col='variable')
```

### Data Access

```python
# Get DataFrame for export or custom plotting
df = results.plot.balance('Bus').data
df.to_csv('bus_balance.csv')

# Custom aggregation with pandas
df_agg = df.groupby('flow')['value'].sum()
df_agg.plot.bar()  # Use pandas/matplotlib
```

### Figure Modification

```python
# Get result without showing
result = results.plot.balance('Bus', show=False)

# Modify the figure
result.update(title='Custom Title', template='plotly_dark')
result.figure.add_annotation(x='2024-06-15', y=100, text='Peak')

# Show when ready
result.show()
```

### Chaining

```python
(results.plot.balance('Bus')
    .update(title='Energy Balance', height=800)
    .to_html('balance.html')
    .show())
```

### Element-Level Plotting

```python
# Access via element
results['Boiler'].plot.balance()
results['Battery'].plot.storage()
results['CHP'].plot.heatmap('on')
```

---

## Configuration

Uses existing `CONFIG.Plotting.default_show` for auto-show behavior.

Colors are resolved in this order:
1. Per-plot `colors` kwarg (highest priority)
2. `results.colors` (global colors set via `setup_colors()`)
3. Auto-assigned from default colorscale (for missing colors)

---

## Implementation Notes

### Accessor Attachment

The `plot` accessor should be a cached property on `Results`:

```python
@property
def plot(self) -> PlotAccessor:
    if self._plot_accessor is None:
        self._plot_accessor = PlotAccessor(self)
    return self._plot_accessor
```

### Default Facet/Animation Behavior

Current defaults:
- `facet_col='scenario'` - Auto-facet by scenario if present
- `animate_by='period'` - Auto-animate by period if present

These are ignored if the dimension doesn't exist in the data.

### Include/Exclude Semantics

Uses substring matching:
- `include='Boiler'` matches any flow containing 'Boiler'
- `include=['Boiler', 'CHP']` matches flows containing 'Boiler' OR 'CHP'
- `exclude='Grid'` removes flows containing 'Grid'

Applied after `include`, so you can do:
```python
include=['*Solar*', '*Wind*'], exclude=['*Curtailment*']
```

---

## Open Questions

1. **Accessor attachment**: Should `plot` be a property (lazy) or set in `__init__`?
   - **Recommendation**: Lazy property (cached)

2. **Default facet/animate**: Should `facet_col='scenario'`, `animate_by='period'` be the defaults, or `None` (explicit opt-in)?
   - **Recommendation**: Keep current defaults, they're ignored if dimension doesn't exist

3. **Include/exclude semantics**: Substring match, glob, or regex?
   - **Recommendation**: Start with substring, consider glob later

---

## Migration Path

The new API coexists with existing methods:
- `results.plot.balance('Bus')` (new)
- `results['Bus'].plot_node_balance()` (existing, keep for backwards compatibility)

Eventually deprecate old methods with warnings pointing to new API.
