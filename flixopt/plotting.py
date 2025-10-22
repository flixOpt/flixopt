"""Comprehensive visualization toolkit for flixopt optimization results and data analysis.

This module provides a unified plotting interface supporting both Plotly (interactive)
and Matplotlib (static) backends for visualizing energy system optimization results.
It offers specialized plotting functions for time series, heatmaps, network diagrams,
and statistical analyses commonly needed in energy system modeling.

Key Features:
    **Dual Backend Support**: Seamless switching between Plotly and Matplotlib
    **Energy System Focus**: Specialized plots for power flows, storage states, emissions
    **Color Management**: Intelligent color processing with ColorProcessor and component-based
                         ComponentColorManager for stable, pattern-matched coloring
    **Export Capabilities**: High-quality export for reports and publications
    **Integration Ready**: Designed for use with CalculationResults and standalone analysis

Main Plot Types:
    - **Time Series**: Flow rates, power profiles, storage states over time
    - **Heatmaps**: High-resolution temporal data visualization with customizable aggregation
    - **Network Diagrams**: System topology with flow visualization
    - **Statistical Plots**: Distribution analysis, correlation studies, performance metrics
    - **Comparative Analysis**: Multi-scenario and sensitivity study visualizations

The module integrates seamlessly with flixopt's result classes while remaining
accessible for standalone data visualization tasks.
"""

from __future__ import annotations

import fnmatch
import itertools
import logging
import os
import pathlib
import re
from typing import TYPE_CHECKING, Any, Literal

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline
import xarray as xr
from plotly.exceptions import PlotlyError

from .config import CONFIG

# Optional dependency for flow-level color shading
try:
    from colour import Color

    HAS_COLOUR = True
except ImportError:
    HAS_COLOUR = False

if TYPE_CHECKING:
    import pyvis

logger = logging.getLogger('flixopt')

# Define the colors for the 'portland' colormap in matplotlib
_portland_colors = [
    [12 / 255, 51 / 255, 131 / 255],  # Dark blue
    [10 / 255, 136 / 255, 186 / 255],  # Light blue
    [242 / 255, 211 / 255, 56 / 255],  # Yellow
    [242 / 255, 143 / 255, 56 / 255],  # Orange
    [217 / 255, 30 / 255, 30 / 255],  # Red
]

# Check if the colormap already exists before registering it
if hasattr(plt, 'colormaps'):  # Matplotlib >= 3.7
    registry = plt.colormaps
    if 'portland' not in registry:
        registry.register(mcolors.LinearSegmentedColormap.from_list('portland', _portland_colors))
else:  # Matplotlib < 3.7
    if 'portland' not in [c for c in plt.colormaps()]:
        plt.register_cmap(name='portland', cmap=mcolors.LinearSegmentedColormap.from_list('portland', _portland_colors))


ColorType = str | list[str] | dict[str, str]
"""Flexible color specification type supporting multiple input formats for visualization.

Color specifications can take several forms to accommodate different use cases:

**Named Colormaps** (str):
    - Standard colormaps: 'viridis', 'plasma', 'cividis', 'tab10', 'Set1'
    - Energy-focused: 'portland' (custom flixopt colormap for energy systems)
    - Backend-specific maps available in Plotly and Matplotlib

**Color Lists** (list[str]):
    - Explicit color sequences: ['red', 'blue', 'green', 'orange']
    - HEX codes: ['#FF0000', '#0000FF', '#00FF00', '#FFA500']
    - Mixed formats: ['red', '#0000FF', 'green', 'orange']

**Label-to-Color Mapping** (dict[str, str]):
    - Explicit associations: {'Wind': 'skyblue', 'Solar': 'gold', 'Gas': 'brown'}
    - Ensures consistent colors across different plots and datasets
    - Ideal for energy system components with semantic meaning

Examples:
    ```python
    # Named colormap
    colors = 'viridis'  # Automatic color generation

    # Explicit color list
    colors = ['red', 'blue', 'green', '#FFD700']

    # Component-specific mapping
    colors = {
        'Wind_Turbine': 'skyblue',
        'Solar_Panel': 'gold',
        'Natural_Gas': 'brown',
        'Battery': 'green',
        'Electric_Load': 'darkred'
    }
    ```

Color Format Support:
    - **Named Colors**: 'red', 'blue', 'forestgreen', 'darkorange'
    - **HEX Codes**: '#FF0000', '#0000FF', '#228B22', '#FF8C00'
    - **RGB Tuples**: (255, 0, 0), (0, 0, 255) [Matplotlib only]
    - **RGBA**: 'rgba(255,0,0,0.8)' [Plotly only]

References:
    - HTML Color Names: https://htmlcolorcodes.com/color-names/
    - Matplotlib Colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    - Plotly Built-in Colorscales: https://plotly.com/python/builtin-colorscales/
"""

PlottingEngine = Literal['plotly', 'matplotlib']
"""Identifier for the plotting engine to use."""


class ColorProcessor:
    """Intelligent color management system for consistent multi-backend visualization.

    This class provides unified color processing across Plotly and Matplotlib backends,
    ensuring consistent visual appearance regardless of the plotting engine used.
    It handles color palette generation, named colormap translation, and intelligent
    color cycling for complex datasets with many categories.

    Key Features:
        **Backend Agnostic**: Automatic color format conversion between engines
        **Palette Management**: Support for named colormaps, custom palettes, and color lists
        **Intelligent Cycling**: Smart color assignment for datasets with many categories
        **Fallback Handling**: Graceful degradation when requested colormaps are unavailable
        **Energy System Colors**: Built-in palettes optimized for energy system visualization

    Color Input Types:
        - **Named Colormaps**: 'viridis', 'plasma', 'portland', 'tab10', etc.
        - **Color Lists**: ['red', 'blue', 'green'] or ['#FF0000', '#0000FF', '#00FF00']
        - **Label Dictionaries**: {'Generator': 'red', 'Storage': 'blue', 'Load': 'green'}

    Examples:
        Basic color processing:

        ```python
        # Initialize for Plotly backend
        processor = ColorProcessor(engine='plotly', default_colormap='viridis')

        # Process different color specifications
        colors = processor.process_colors('plasma', ['Gen1', 'Gen2', 'Storage'])
        colors = processor.process_colors(['red', 'blue', 'green'], ['A', 'B', 'C'])
        colors = processor.process_colors({'Wind': 'skyblue', 'Solar': 'gold'}, ['Wind', 'Solar', 'Gas'])

        # Switch to Matplotlib
        processor = ColorProcessor(engine='matplotlib')
        mpl_colors = processor.process_colors('tab10', component_labels)
        ```

        Energy system visualization:

        ```python
        # Specialized energy system palette
        energy_colors = {
            'Natural_Gas': '#8B4513',  # Brown
            'Electricity': '#FFD700',  # Gold
            'Heat': '#FF4500',  # Red-orange
            'Cooling': '#87CEEB',  # Sky blue
            'Hydrogen': '#E6E6FA',  # Lavender
            'Battery': '#32CD32',  # Lime green
        }

        processor = ColorProcessor('plotly')
        flow_colors = processor.process_colors(energy_colors, flow_labels)
        ```

    Args:
        engine: Plotting backend ('plotly' or 'matplotlib'). Determines output color format.
        default_colormap: Fallback colormap when requested palettes are unavailable.
            Common options: 'viridis', 'plasma', 'tab10', 'portland'.

    """

    def __init__(self, engine: PlottingEngine = 'plotly', default_colormap: str | None = None):
        """Initialize the color processor with specified backend and defaults."""
        if engine not in ['plotly', 'matplotlib']:
            raise TypeError(f'engine must be "plotly" or "matplotlib", but is {engine}')
        self.engine = engine
        self.default_colormap = (
            default_colormap if default_colormap is not None else CONFIG.Plotting.default_qualitative_colorscale
        )

    def _generate_colors_from_colormap(self, colormap_name: str, num_colors: int) -> list[Any]:
        """
        Generate colors from a named colormap.

        Args:
            colormap_name: Name of the colormap
            num_colors: Number of colors to generate

        Returns:
            list of colors in the format appropriate for the engine
        """
        if self.engine == 'plotly':
            # First try qualitative color sequences (Dark24, Plotly, Set1, etc.)
            colormap_name = colormap_name.title()
            if hasattr(px.colors.qualitative, colormap_name):
                color_list = getattr(px.colors.qualitative, colormap_name)
                # Cycle through colors if we need more than available
                return [color_list[i % len(color_list)] for i in range(num_colors)]

            # Then try sequential/continuous colorscales (viridis, plasma, etc.)
            try:
                colorscale = px.colors.get_colorscale(colormap_name)
            except PlotlyError as e:
                logger.error(f"Colorscale '{colormap_name}' not found in Plotly. Using {self.default_colormap}: {e}")
                # Try default as qualitative first
                if hasattr(px.colors.qualitative, self.default_colormap):
                    color_list = getattr(px.colors.qualitative, self.default_colormap)
                    return [color_list[i % len(color_list)] for i in range(num_colors)]
                # Otherwise use default as sequential
                colorscale = px.colors.get_colorscale(self.default_colormap)

            # Generate evenly spaced points
            color_points = [i / (num_colors - 1) for i in range(num_colors)] if num_colors > 1 else [0]
            return px.colors.sample_colorscale(colorscale, color_points)

        else:  # matplotlib
            try:
                cmap = plt.get_cmap(colormap_name, num_colors)
            except ValueError as e:
                logger.error(f"Colormap '{colormap_name}' not found in Matplotlib. Using {self.default_colormap}: {e}")
                cmap = plt.get_cmap(self.default_colormap, num_colors)

            return [cmap(i) for i in range(num_colors)]

    def _handle_color_list(self, colors: list[str], num_labels: int) -> list[str]:
        """
        Handle a list of colors, cycling if necessary.

        Args:
            colors: list of color strings
            num_labels: Number of labels that need colors

        Returns:
            list of colors matching the number of labels
        """
        if len(colors) == 0:
            logger.error(f'Empty color list provided. Using {self.default_colormap} instead.')
            return self._generate_colors_from_colormap(self.default_colormap, num_labels)

        if len(colors) < num_labels:
            logger.warning(
                f'Not enough colors provided ({len(colors)}) for all labels ({num_labels}). Colors will cycle.'
            )
            # Cycle through the colors
            color_iter = itertools.cycle(colors)
            return [next(color_iter) for _ in range(num_labels)]
        else:
            # Trim if necessary
            if len(colors) > num_labels:
                logger.warning(
                    f'More colors provided ({len(colors)}) than labels ({num_labels}). Extra colors will be ignored.'
                )
            return colors[:num_labels]

    def _handle_color_dict(self, colors: dict[str, str], labels: list[str]) -> list[str]:
        """
        Handle a dictionary mapping labels to colors.

        Args:
            colors: Dictionary mapping labels to colors
            labels: list of labels that need colors

        Returns:
            list of colors in the same order as labels
        """
        if len(colors) == 0:
            logger.warning(f'Empty color dictionary provided. Using {self.default_colormap} instead.')
            return self._generate_colors_from_colormap(self.default_colormap, len(labels))

        # Find missing labels
        missing_labels = sorted(set(labels) - set(colors.keys()))
        if missing_labels:
            logger.warning(
                f'Some labels have no color specified: {missing_labels}. Using {self.default_colormap} for these.'
            )

            # Generate colors for missing labels
            missing_colors = self._generate_colors_from_colormap(self.default_colormap, len(missing_labels))

            # Create a copy to avoid modifying the original
            colors_copy = colors.copy()
            for i, label in enumerate(missing_labels):
                colors_copy[label] = missing_colors[i]
        else:
            colors_copy = colors

        # Create color list in the same order as labels
        return [colors_copy[label] for label in labels]

    def process_colors(
        self,
        colors: ColorType,
        labels: list[str],
        return_mapping: bool = False,
    ) -> list[Any] | dict[str, Any]:
        """
        Process colors for the specified labels.

        Args:
            colors: Color specification (colormap name, list of colors, or label-to-color mapping)
            labels: list of data labels that need colors assigned
            return_mapping: If True, returns a dictionary mapping labels to colors;
                           if False, returns a list of colors in the same order as labels

        Returns:
            Either a list of colors or a dictionary mapping labels to colors
        """
        if len(labels) == 0:
            logger.error('No labels provided for color assignment.')
            return {} if return_mapping else []

        # Process based on type of colors input
        if isinstance(colors, str):
            color_list = self._generate_colors_from_colormap(colors, len(labels))
        elif isinstance(colors, list):
            color_list = self._handle_color_list(colors, len(labels))
        elif isinstance(colors, dict):
            color_list = self._handle_color_dict(colors, labels)
        else:
            logger.error(
                f'Unsupported color specification type: {type(colors)}. Using {self.default_colormap} instead.'
            )
            color_list = self._generate_colors_from_colormap(self.default_colormap, len(labels))

        # Return either a list or a mapping
        if return_mapping:
            return {label: color_list[i] for i, label in enumerate(labels)}
        else:
            return color_list


# Type aliases for ComponentColorManager
MatchType = Literal['prefix', 'suffix', 'contains', 'glob', 'regex']


class ComponentColorManager:
    """Manage stable colors for flow system components with pattern-based grouping.

    This class provides component-centric color management where each component gets
    a stable color assigned once, ensuring consistent coloring across all plots.
    Components can be grouped using pattern matching, and each group uses a different colormap.

    Key Features:
        - **Stable colors**: Components assigned colors once based on sorted order
        - **Pattern-based grouping**: Auto-group components using patterns (prefix, contains, regex, etc.)
        - **Variable extraction**: Auto-extract component names from variable names
        - **Flexible colormaps**: Use Plotly sequential palettes or custom colors
        - **Override support**: Manually override specific component colors
        - **Zero configuration**: Works automatically with sensible defaults

    Available Color Families (14 single-hue palettes):
        Cool: blues, greens, teals, purples, mint, emrld, darkmint
        Warm: reds, oranges, peach, pinks, burg, sunsetdark
        Neutral: greys

    Example Usage:
        Basic usage (automatic, each component gets distinct color):

        ```python
        manager = ComponentColorManager(components=['Boiler1', 'Boiler2', 'CHP1'])
        color = manager.get_color('Boiler1')  # Always same color
        ```

        Grouped coloring (components in same group get shades of same color):

        ```python
        manager = ComponentColorManager(components=['Boiler1', 'Boiler2', 'CHP1', 'Storage1'])
        manager.add_grouping_rule('Boiler', 'Heat_Producers', 'reds', 'prefix')
        manager.add_grouping_rule('CHP', 'Heat_Producers', 'reds', 'prefix')
        manager.add_grouping_rule('Storage', 'Storage', 'blues', 'contains')
        manager.apply_colors()

        # Boiler1, Boiler2, CHP1 get different shades of red
        # Storage1 gets blue
        ```

        Override specific components:

        ```python
        manager.override({'Boiler1': '#FF0000'})  # Force Boiler1 to red
        ```

        Get colors for variables (extracts component automatically):

        ```python
        colors = manager.get_variable_colors(['Boiler1(Bus_A)|flow', 'CHP1(Bus_B)|flow'])
        # Returns: {'Boiler1(Bus_A)|flow': '#...', 'CHP1(Bus_B)|flow': '#...'}
        ```
    """

    # Class-level color family defaults
    DEFAULT_FAMILIES = {
        'blues': px.colors.sequential.Blues[1:8],
        'greens': px.colors.sequential.Greens[1:8],
        'reds': px.colors.sequential.Reds[1:8],
        'purples': px.colors.sequential.Purples[1:8],
        'oranges': px.colors.sequential.Oranges[1:8],
        'teals': px.colors.sequential.Teal[1:8],
        'greys': px.colors.sequential.Greys[1:8],
        'pinks': px.colors.sequential.Pinkyl[1:8],
        'peach': px.colors.sequential.Peach[1:8],
        'burg': px.colors.sequential.Burg[1:8],
        'sunsetdark': px.colors.sequential.Sunsetdark[1:8],
        'mint': px.colors.sequential.Mint[1:8],
        'emrld': px.colors.sequential.Emrld[1:8],
        'darkmint': px.colors.sequential.Darkmint[1:8],
    }

    def __init__(
        self,
        components: list[str] | None = None,
        flows: dict[str, list[str]] | None = None,
        enable_flow_shading: bool = False,
        flow_variation_strength: float = 0.04,
        default_colormap: str = 'Dark24',
    ) -> None:
        """Initialize component color manager.

        Args:
            components: List of all component names in the system (optional if flows provided)
            flows: Dict mapping component names to their flow labels (e.g., {'Boiler': ['Q_th', 'Q_fu']})
            enable_flow_shading: If True, create subtle color variations for flows of same component
            flow_variation_strength: Lightness variation per flow (0.05-0.15, default: 0.08 = 8%)
            default_colormap: Default colormap for ungrouped components (default: 'Dark24')
        """
        # Extract components from flows dict if provided
        if flows is not None:
            self.flows = {comp: sorted(set(flow_list)) for comp, flow_list in flows.items()}
            self.components = sorted(self.flows.keys())
        elif components is not None:
            self.components = sorted(set(components))
            self.flows = {}
        else:
            raise ValueError('Must provide either components or flows parameter')

        self.default_colormap = default_colormap
        self.color_families = self.DEFAULT_FAMILIES.copy()

        # Flow shading settings (requires optional 'colour' library)
        if enable_flow_shading and not HAS_COLOUR:
            logger.error(
                'Flow shading requested but optional dependency "colour" is not installed. '
                'Install it with: pip install flixopt[flow_colors]\n'
                'Flow shading will be disabled.'
            )
            self.enable_flow_shading = False
        else:
            self.enable_flow_shading = enable_flow_shading
        self.flow_variation_strength = flow_variation_strength

        # Pattern-based grouping rules
        self._grouping_rules: list[dict[str, str]] = []

        # Computed colors: {component_name: color}
        self._component_colors: dict[str, str] = {}

        # Manual overrides (highest priority)
        self._overrides: dict[str, str] = {}

        # Variable color cache for performance: {variable_name: color}
        self._variable_cache: dict[str, str] = {}

        # Auto-assign default colors
        self._assign_default_colors()

    def __repr__(self) -> str:
        """Return detailed representation of ComponentColorManager."""
        flow_info = f', flow_shading={self.enable_flow_shading}' if self.enable_flow_shading else ''
        return (
            f'ComponentColorManager(components={len(self.components)}, '
            f'rules={len(self._grouping_rules)}, '
            f'overrides={len(self._overrides)}, '
            f"default_colormap='{self.default_colormap}'{flow_info})"
        )

    def __str__(self) -> str:
        """Return human-readable summary of ComponentColorManager."""
        lines = [
            'ComponentColorManager',
            f'  Components: {len(self.components)}',
        ]

        # Show first few components as examples
        if self.components:
            sample = self.components[:5]
            if len(self.components) > 5:
                sample_str = ', '.join(sample) + f', ... ({len(self.components) - 5} more)'
            else:
                sample_str = ', '.join(sample)
            lines.append(f'    [{sample_str}]')

        lines.append(f'  Grouping rules: {len(self._grouping_rules)}')
        if self._grouping_rules:
            for rule in self._grouping_rules[:3]:  # Show first 3 rules
                lines.append(
                    f"    - {rule['match_type']}('{rule['pattern']}') → "
                    f"group '{rule['group_name']}' ({rule['colormap']})"
                )
            if len(self._grouping_rules) > 3:
                lines.append(f'    ... and {len(self._grouping_rules) - 3} more')

        lines.append(f'  Overrides: {len(self._overrides)}')
        if self._overrides:
            for comp, color in list(self._overrides.items())[:3]:
                lines.append(f'    - {comp}: {color}')
            if len(self._overrides) > 3:
                lines.append(f'    ... and {len(self._overrides) - 3} more')

        lines.append(f'  Default colormap: {self.default_colormap}')

        return '\n'.join(lines)

    @classmethod
    def from_flow_system(cls, flow_system, enable_flow_shading: bool = False, **kwargs):
        """Create ComponentColorManager from a FlowSystem.

        Automatically extracts all components and their flows from the FlowSystem.

        Args:
            flow_system: FlowSystem instance to extract components and flows from
            enable_flow_shading: Enable subtle color variations for flows (default: False)
            **kwargs: Additional arguments passed to ComponentColorManager.__init__

        Returns:
            ComponentColorManager instance

        Examples:
            ```python
            # Basic usage
            manager = ComponentColorManager.from_flow_system(flow_system)

            # With flow shading
            manager = ComponentColorManager.from_flow_system(
                flow_system, enable_flow_shading=True, flow_variation_strength=0.10
            )
            ```
        """
        from .flow_system import FlowSystem

        if not isinstance(flow_system, FlowSystem):
            raise TypeError(f'Expected FlowSystem, got {type(flow_system).__name__}')

        # Extract flows from all components
        flows = {}
        for component_label, component in flow_system.components.items():
            flow_labels = [flow.label for flow in component.inputs + component.outputs]
            if flow_labels:  # Only add if component has flows
                flows[component_label] = flow_labels

        return cls(flows=flows, enable_flow_shading=enable_flow_shading, **kwargs)

    def add_custom_family(self, name: str, colors: list[str]) -> ComponentColorManager:
        """Add a custom color family.

        Args:
            name: Name for the color family.
            colors: List of hex color codes.

        Returns:
            Self for method chaining.
        """
        self.color_families[name] = colors
        return self

    def add_rule(
        self, pattern: str, colormap: str, match_type: MatchType | None = None, group_name: str | None = None
    ) -> ComponentColorManager:
        """Add color rule with optional auto-detection (flexible API).

        By default, automatically detects the matching strategy from pattern syntax.
        Optionally override with explicit match_type and group_name for full control.

        Auto-detection rules:
            - 'Solar' → prefix matching
            - 'Solar*' → glob matching
            - '~Storage' → contains matching (strip ~)
            - 'Solar$' → suffix matching (strip $)
            - '.*Solar.*' → regex matching

        Colors are automatically applied after adding the rule.

        Args:
            pattern: Pattern to match components
            colormap: Colormap name ('reds', 'blues', 'greens', etc.')
            match_type: Optional explicit match type. If None (default), auto-detects from pattern.
                Options: 'prefix', 'suffix', 'contains', 'glob', 'regex'
            group_name: Optional group name for organization. If None, auto-generates from pattern.

        Returns:
            Self for method chaining

        Examples:
            Simple auto-detection (most common):

            ```python
            manager.add_rule('Solar', 'oranges')         # Auto: prefix
            manager.add_rule('Wind*', 'blues')           # Auto: glob
            manager.add_rule('~Storage', 'greens')       # Auto: contains
            ```

            Override auto-detection when needed:

            ```python
            # Force prefix matching even though it has special chars
            manager.add_rule('Solar*', 'oranges', match_type='prefix')

            # Explicit regex when pattern is ambiguous
            manager.add_rule('Solar.+', 'oranges', match_type='regex')
            ```

            Full explicit control:

            ```python
            manager.add_rule('Solar', 'oranges', 'prefix', 'renewables')
            ```

            Chained configuration:

            ```python
            manager.add_rule('Solar*', 'oranges')\
                   .add_rule('Wind*', 'blues')\
                   .add_rule('Battery', 'greens', 'prefix', 'storage')
            ```
        """
        # Auto-detect match type if not provided
        if match_type is None:
            match_type = self._detect_match_type(pattern)

        # Clean pattern based on match type (strip special markers)
        clean_pattern = pattern
        if match_type == 'contains' and pattern.startswith('~'):
            clean_pattern = pattern[1:]  # Strip ~ prefix
        elif match_type == 'suffix' and pattern.endswith('$') and not any(c in pattern for c in r'.[]()^|+\\'):
            clean_pattern = pattern[:-1]  # Strip $ suffix (only if not part of regex)

        # Auto-generate group name if not provided
        if group_name is None:
            group_name = clean_pattern.replace('*', '').replace('?', '').replace('.', '')[:20]

        # Delegate to _add_grouping_rule
        return self._add_grouping_rule(clean_pattern, group_name, colormap, match_type)

    def _add_grouping_rule(
        self, pattern: str, group_name: str, colormap: str, match_type: MatchType = 'prefix'
    ) -> ComponentColorManager:
        """Add pattern rule for grouping components (low-level API).

        Args:
            pattern: Pattern to match component names against
            group_name: Name of the group (used for organization)
            colormap: Colormap name for this group ('reds', 'blues', etc.')
            match_type: Type of pattern matching (default: 'prefix')
                - 'prefix': Match if component starts with pattern
                - 'suffix': Match if component ends with pattern
                - 'contains': Match if pattern appears in component name
                - 'glob': Unix wildcards (* and ?)
                - 'regex': Regular expression matching
        """
        valid_types = ('prefix', 'suffix', 'contains', 'glob', 'regex')
        if match_type not in valid_types:
            raise ValueError(f"match_type must be one of {valid_types}, got '{match_type}'")

        self._grouping_rules.append(
            {'pattern': pattern, 'group_name': group_name, 'colormap': colormap, 'match_type': match_type}
        )
        # Auto-apply colors after adding rule for immediate effect
        self.apply_colors()
        return self

    def apply_colors(self) -> None:
        """Apply grouping rules and assign colors to all components.

        This recomputes colors for all components based on current grouping rules.
        Components are grouped, then within each group they get sequential colors
        from the group's colormap (based on sorted order for stability).

        Call this after adding/changing grouping rules to update colors.
        """
        # Group components by matching rules
        groups: dict[str, dict] = {}

        for component in self.components:
            matched = False
            for rule in self._grouping_rules:
                if self._match_pattern(component, rule['pattern'], rule['match_type']):
                    group_name = rule['group_name']
                    if group_name not in groups:
                        groups[group_name] = {'components': [], 'colormap': rule['colormap']}
                    groups[group_name]['components'].append(component)
                    matched = True
                    break  # First match wins

            if not matched:
                # Unmatched components go to default group
                if '_ungrouped' not in groups:
                    groups['_ungrouped'] = {'components': [], 'colormap': self.default_colormap}
                groups['_ungrouped']['components'].append(component)

        # Assign colors within each group (stable sorted order)
        self._component_colors = {}
        for group_data in groups.values():
            colormap = self._get_colormap_colors(group_data['colormap'])
            sorted_components = sorted(group_data['components'])  # Stable!

            for idx, component in enumerate(sorted_components):
                self._component_colors[component] = colormap[idx % len(colormap)]

        # Apply overrides (highest priority)
        self._component_colors.update(self._overrides)

        # Clear variable cache since colors have changed
        self._variable_cache.clear()

    def override(self, component_colors: dict[str, str]) -> None:
        """Override colors for specific components.

        These overrides have highest priority and persist even after regrouping.

        Args:
            component_colors: Dict mapping component names to colors

        Examples:
            ```python
            manager.override({'Boiler1': '#FF0000', 'CHP1': '#00FF00'})
            ```
        """
        self._overrides.update(component_colors)
        self._component_colors.update(component_colors)

        # Clear variable cache since colors have changed
        self._variable_cache.clear()

    def get_color(self, component: str) -> str:
        """Get color for a component.

        Args:
            component: Component name

        Returns:
            Hex color string (defaults to grey if component unknown)
        """
        return self._component_colors.get(component, '#808080')

    def extract_component(self, variable: str) -> str:
        """Extract component name from variable name.

        Uses default extraction logic: split on '(' or '|' to get component.

        Args:
            variable: Variable name (e.g., 'Boiler1(Bus_A)|flow_rate')

        Returns:
            Component name (e.g., 'Boiler1')

        Examples:
            ```python
            extract_component('Boiler1(Bus_A)|flow')  # Returns: 'Boiler1'
            extract_component('CHP1|power')  # Returns: 'CHP1'
            extract_component('Storage')  # Returns: 'Storage'
            ```
        """
        component, _ = self._extract_component_and_flow(variable)
        return component

    def _extract_component_and_flow(self, variable: str) -> tuple[str, str | None]:
        """Extract both component and flow name from variable name.

        Parses variable formats:
        - 'Component(Flow)|attribute' → ('Component', 'Flow')
        - 'Component|attribute' → ('Component', None)
        - 'Component' → ('Component', None)

        Args:
            variable: Variable name

        Returns:
            Tuple of (component_name, flow_name or None)

        Examples:
            ```python
            _extract_component_and_flow('Boiler(Q_th)|flow_rate')  # ('Boiler', 'Q_th')
            _extract_component_and_flow('CHP(P_el)|flow_rate')  # ('CHP', 'P_el')
            _extract_component_and_flow('Boiler|investment')  # ('Boiler', None)
            ```
        """
        # Try "Component(Flow)|attribute" format
        if '(' in variable and ')' in variable:
            component = variable.split('(')[0]
            flow = variable.split('(')[1].split(')')[0]
            return component, flow

        # Try "Component|attribute" format (no flow)
        if '|' in variable:
            return variable.split('|')[0], None

        # Just the component name itself
        return variable, None

    def get_variable_color(self, variable: str) -> str:
        """Get color for a variable (extracts component automatically).

        If flow_shading is enabled, generates subtle color variations for different
        flows of the same component.

        Args:
            variable: Variable name

        Returns:
            Hex color string
        """
        # Check cache first
        if variable in self._variable_cache:
            return self._variable_cache[variable]

        # Extract component and flow
        component, flow = self._extract_component_and_flow(variable)

        # Get base color for component
        base_color = self.get_color(component)

        # Apply flow shading if enabled and flow is present
        if self.enable_flow_shading and flow is not None and component in self.flows:
            # Get sorted flow list for this component
            component_flows = self.flows[component]

            if flow in component_flows and len(component_flows) > 1:
                # Generate shades for all flows
                shades = self._create_flow_shades(base_color, len(component_flows))

                # Assign shade based on flow's position in sorted list
                flow_idx = component_flows.index(flow)
                color = shades[flow_idx]
            else:
                # Flow not in predefined list or only one flow - use base color
                color = base_color
        else:
            # No flow shading or no flow info - use base color
            color = base_color

        # Cache and return
        self._variable_cache[variable] = color
        return color

    def get_variable_colors(self, variables: list[str]) -> dict[str, str]:
        """Get colors for multiple variables.

        This is the main API used by plotting functions.

        Args:
            variables: List of variable names

        Returns:
            Dict mapping variable names to colors
        """
        return {var: self.get_variable_color(var) for var in variables}

    def to_dict(self) -> dict[str, str]:
        """Get complete component→color mapping.

        Returns:
            Dict of all components and their assigned colors
        """
        return self._component_colors.copy()

    # ==================== INTERNAL METHODS ====================

    def _assign_default_colors(self) -> None:
        """Assign default colors to all components (no grouping)."""
        colormap = self._get_colormap_colors(self.default_colormap)

        for idx, component in enumerate(self.components):
            self._component_colors[component] = colormap[idx % len(colormap)]

    def _detect_match_type(self, pattern: str) -> MatchType:
        """Auto-detect match type from pattern syntax.

        Detection logic:
            - Contains '~' prefix → 'contains' (strip ~ from pattern)
            - Ends with '$' → 'suffix'
            - Contains '*' or '?' → 'glob'
            - Contains regex special chars (^[]().|+\\) → 'regex'
            - Otherwise → 'prefix' (default)

        Args:
            pattern: Pattern string to analyze

        Returns:
            Detected match type

        Examples:
            >>> _detect_match_type('Solar')  # 'prefix'
            >>> _detect_match_type('Solar*')  # 'glob'
            >>> _detect_match_type('~Storage')  # 'contains'
            >>> _detect_match_type('.*Solar.*')  # 'regex'
            >>> _detect_match_type('Solar$')  # 'suffix'
        """
        # Check for explicit contains marker
        if pattern.startswith('~'):
            return 'contains'

        # Check for suffix marker (but only if not a regex pattern)
        if pattern.endswith('$') and len(pattern) > 1 and not any(c in pattern for c in r'.[]()^|+\\'):
            return 'suffix'

        # Check for regex special characters (before glob, since .* is regex not glob)
        # Exclude * and ? which are also glob chars
        regex_only_chars = r'.[]()^|+\\'
        if any(char in pattern for char in regex_only_chars):
            return 'regex'

        # Check for simple glob wildcards
        if '*' in pattern or '?' in pattern:
            return 'glob'

        # Default to prefix matching
        return 'prefix'

    @staticmethod
    def _load_config_from_file(file_path: str | pathlib.Path) -> dict[str, str]:
        """Load color configuration from YAML or JSON file.

        Args:
            file_path: Path to YAML or JSON configuration file

        Returns:
            Dictionary mapping patterns to colormaps

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported or invalid

        Examples:
            YAML file (colors.yaml):
            ```yaml
            Solar*: oranges
            Wind*: blues
            Battery: greens
            ~Storage: teals
            ```

            JSON file (colors.json):
            ```json
            {
                "Solar*": "oranges",
                "Wind*": "blues",
                "Battery": "greens"
            }
            ```
        """
        import json

        file_path = pathlib.Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f'Color configuration file not found: {file_path}')

        # Determine file type from extension
        suffix = file_path.suffix.lower()

        if suffix in ['.yaml', '.yml']:
            try:
                import yaml
            except ImportError as e:
                raise ImportError(
                    'PyYAML is required to load YAML config files. Install it with: pip install pyyaml'
                ) from e

            with open(file_path, encoding='utf-8') as f:
                config = yaml.safe_load(f)

        elif suffix == '.json':
            with open(file_path, encoding='utf-8') as f:
                config = json.load(f)

        else:
            raise ValueError(f'Unsupported file format: {suffix}. Supported formats: .yaml, .yml, .json')

        # Validate config structure
        if not isinstance(config, dict):
            raise ValueError(f'Invalid config file structure. Expected dict, got {type(config).__name__}')

        # Ensure all values are strings
        for key, value in config.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError(f'Invalid config entry: {key}: {value}. Both keys and values must be strings.')

        return config

    def _match_pattern(self, value: str, pattern: str, match_type: str) -> bool:
        """Check if value matches pattern.

        Args:
            value: String to test
            pattern: Pattern to match against
            match_type: Type of matching

        Returns:
            True if matches
        """
        if match_type == 'prefix':
            return value.startswith(pattern)
        elif match_type == 'suffix':
            return value.endswith(pattern)
        elif match_type == 'contains':
            return pattern in value
        elif match_type == 'glob':
            return fnmatch.fnmatch(value, pattern)
        elif match_type == 'regex':
            try:
                return bool(re.search(pattern, value))
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e
        return False

    def _get_colormap_colors(self, colormap_name: str) -> list[str]:
        """Get list of colors from colormap name."""

        # Check custom families first
        if colormap_name in self.color_families:
            return self.color_families[colormap_name]

        # Try qualitative palettes (best for discrete components)
        if hasattr(px.colors.qualitative, colormap_name.title()):
            return getattr(px.colors.qualitative, colormap_name.title())

        # Try sequential palettes
        if hasattr(px.colors.sequential, colormap_name.title()):
            return getattr(px.colors.sequential, colormap_name.title())

        # Fall back to ColorProcessor for matplotlib colormaps
        processor = ColorProcessor(engine='plotly')
        try:
            colors = processor._generate_colors_from_colormap(colormap_name, 10)
            return colors
        except Exception:
            logger.warning(f"Colormap '{colormap_name}' not found, using 'Dark24' instead")
            return px.colors.qualitative.Dark24

    def _create_flow_shades(self, base_color: str, num_flows: int) -> list[str]:
        """Generate subtle color variations from a single base color using HSL.

        Uses the `colour` library for robust color manipulation. If `colour` is not
        available, returns the base color for all flows.

        Args:
            base_color: Color string (hex like '#D62728' or rgb like 'rgb(255, 0, 0)')
            num_flows: Number of distinct shades needed

        Returns:
            List of hex colors with subtle lightness variations
        """
        if num_flows == 1:
            return [base_color]

        # Fallback if colour library not available (defensive check)
        if not HAS_COLOUR:
            return [base_color] * num_flows

        # Parse color using colour library (handles hex, rgb(), etc.)
        color = Color(base_color)
        h, s, lightness = color.hsl

        # Create symmetric variations around base lightness
        # For 3 flows with strength 0.08: [-0.08, 0, +0.08]
        # For 5 flows: [-0.16, -0.08, 0, +0.08, +0.16]
        center_idx = (num_flows - 1) / 2
        shades = []

        for idx in range(num_flows):
            delta_lightness = (idx - center_idx) * self.flow_variation_strength
            new_lightness = np.clip(lightness + delta_lightness, 0.1, 0.9)

            # Create new color with adjusted lightness
            new_color = Color(hsl=(h, s, new_lightness))
            shades.append(new_color.hex_l)

        return shades


def _ensure_dataset(data: xr.Dataset | pd.DataFrame) -> xr.Dataset:
    """
    Ensure the input data is an xarray Dataset, converting from DataFrame if needed.

    Args:
        data: Input data, either xarray Dataset or pandas DataFrame.

    Returns:
        xarray Dataset.

    Raises:
        TypeError: If data is neither Dataset nor DataFrame.
    """
    if isinstance(data, xr.Dataset):
        return data
    elif isinstance(data, pd.DataFrame):
        # Convert DataFrame to Dataset
        return data.to_xarray()
    else:
        raise TypeError(f'Data must be xr.Dataset or pd.DataFrame, got {type(data).__name__}')


def _validate_plotting_data(data: xr.Dataset, allow_empty: bool = False) -> None:
    """
    Validate input data for plotting and raise clear errors for common issues.

    Args:
        data: xarray Dataset to validate.
        allow_empty: Whether to allow empty datasets (no variables).

    Raises:
        ValueError: If data is invalid for plotting.
        TypeError: If data contains non-numeric types.
    """
    # Check for empty data
    if not allow_empty and len(data.data_vars) == 0:
        raise ValueError('Empty Dataset provided (no variables). Cannot create plot.')

    # Check if dataset has any data (xarray uses nbytes for total size)
    if all(data[var].size == 0 for var in data.data_vars) if len(data.data_vars) > 0 else True:
        if not allow_empty and len(data.data_vars) > 0:
            raise ValueError('Dataset has zero size. Cannot create plot.')
        if len(data.data_vars) == 0:
            return  # Empty dataset, nothing to validate
        return

    # Check for non-numeric data types
    for var in data.data_vars:
        dtype = data[var].dtype
        if not np.issubdtype(dtype, np.number):
            raise TypeError(
                f"Variable '{var}' has non-numeric dtype '{dtype}'. "
                f'Plotting requires numeric data types (int, float, etc.).'
            )

    # Warn about NaN/Inf values
    for var in data.data_vars:
        if data[var].isnull().any():
            logger.debug(f"Variable '{var}' contains NaN values which may affect visualization.")
        if np.isinf(data[var].values).any():
            logger.debug(f"Variable '{var}' contains Inf values which may affect visualization.")


def resolve_colors(
    data: xr.Dataset,
    colors: ColorType | ComponentColorManager,
    engine: PlottingEngine = 'plotly',
) -> dict[str, str]:
    """Resolve colors parameter to a color mapping dict.

    This public utility function handles all color parameter types and applies the
    color manager intelligently based on the data structure. Can be used standalone
    or as part of CalculationResults.

    Args:
        data: Dataset to create colors for. Variable names from data_vars are used as labels.
        colors: Color specification or a ComponentColorManager to use
        engine: Plotting engine ('plotly' or 'matplotlib')

    Returns:
        Dictionary mapping variable names to colors

    Examples:
        With CalculationResults:

        >>> resolved_colors = resolve_colors(data, results.color_manager)

        Standalone usage:

        >>> manager = plotting.ComponentColorManager(['Solar', 'Wind', 'Coal'])
        >>> manager.add_grouping_rule('Solar', 'renewables', 'oranges', match_type='prefix')
        >>> resolved_colors = resolve_colors(data, manager)

        Without manager:

        >>> resolved_colors = resolve_colors(data, 'viridis')
    """
    # Get variable names from Dataset (always strings and unique)
    labels = list(data.data_vars.keys())

    # If explicit dict provided, use it directly
    if isinstance(colors, dict):
        return colors

    # If string or list, use ColorProcessor (traditional behavior)
    if isinstance(colors, (str, list)):
        processor = ColorProcessor(engine=engine)
        return processor.process_colors(colors, labels, return_mapping=True)

    if isinstance(colors, ComponentColorManager):
        # Use color manager to resolve colors for variables
        return colors.get_variable_colors(labels)

    raise TypeError(f'Wrong type passed to resolve_colors(): {type(colors)}')


def with_plotly(
    data: xr.Dataset | pd.DataFrame,
    mode: Literal['stacked_bar', 'line', 'area', 'grouped_bar'] = 'stacked_bar',
    colors: ColorType | ComponentColorManager | None = None,
    title: str = '',
    ylabel: str = '',
    xlabel: str = '',
    fig: go.Figure | None = None,
    facet_by: str | list[str] | None = None,
    animate_by: str | None = None,
    facet_cols: int | None = None,
    shared_yaxes: bool = True,
    shared_xaxes: bool = True,
    trace_kwargs: dict[str, Any] | None = None,
    layout_kwargs: dict[str, Any] | None = None,
    **px_kwargs: Any,
) -> go.Figure:
    """
    Plot data with Plotly using facets (subplots) and/or animation for multidimensional data.

    Uses Plotly Express for convenient faceting and animation with automatic styling.
    For simple plots without faceting, can optionally add to an existing figure.

    Args:
        data: An xarray Dataset to plot.
        mode: The plotting mode. Use 'stacked_bar' for stacked bar charts, 'line' for lines,
              'area' for stacked area charts, or 'grouped_bar' for grouped bar charts.
        colors: Color specification. Can be:
            - A colormap name (e.g., 'viridis', 'plasma')
            - A list of color strings (e.g., ['#ff0000', '#00ff00'])
            - A dict mapping labels to colors (e.g., {'Solar': '#FFD700'})
            - A ComponentColorManager instance for pattern-based color rules with component grouping
        title: The main title of the plot.
        ylabel: The label for the y-axis.
        xlabel: The label for the x-axis.
        fig: A Plotly figure object to plot on (only for simple plots without faceting).
             If not provided, a new figure will be created.
        facet_by: Dimension(s) to create facets for. Creates a subplot grid.
              Can be a single dimension name or list of dimensions (max 2 for facet_row and facet_col).
              If the dimension doesn't exist in the data, it will be silently ignored.
        animate_by: Dimension to animate over. Creates animation frames.
              If the dimension doesn't exist in the data, it will be silently ignored.
        facet_cols: Number of columns in the facet grid (used when facet_by is single dimension).
        shared_yaxes: Whether subplots share y-axes.
        shared_xaxes: Whether subplots share x-axes.
        trace_kwargs: Optional dict of parameters to pass to fig.update_traces().
                     Use this to customize trace properties (e.g., marker style, line width).
        layout_kwargs: Optional dict of parameters to pass to fig.update_layout().
                      Use this to customize layout properties (e.g., width, height, legend position).
        **px_kwargs: Additional keyword arguments passed to the underlying Plotly Express function
                    (px.bar, px.line, px.area). These override default arguments if provided.

    Returns:
        A Plotly figure object containing the faceted/animated plot.

    Examples:
        Simple plot:

        ```python
        fig = with_plotly(dataset, mode='area', title='Energy Mix')
        ```

        Facet by scenario:

        ```python
        fig = with_plotly(dataset, facet_by='scenario', facet_cols=2)
        ```

        Animate by period:

        ```python
        fig = with_plotly(dataset, animate_by='period')
        ```

        Facet and animate:

        ```python
        fig = with_plotly(dataset, facet_by='scenario', animate_by='period')
        ```

        Pattern-based colors with ComponentColorManager:

        ```python
        manager = ComponentColorManager(['Solar', 'Wind', 'Battery', 'Gas'])
        manager.add_grouping_rule('Solar', 'renewables', 'oranges', match_type='prefix')
        manager.add_grouping_rule('Wind', 'renewables', 'blues', match_type='prefix')
        manager.add_grouping_rule('Battery', 'storage', 'greens', match_type='contains')
        manager.apply_colors()
        fig = with_plotly(dataset, colors=manager, mode='area')
        ```
    """
    if colors is None:
        colors = CONFIG.Plotting.default_qualitative_colorscale

    if mode not in ('stacked_bar', 'line', 'area', 'grouped_bar'):
        raise ValueError(f"'mode' must be one of {{'stacked_bar','line','area', 'grouped_bar'}}, got {mode!r}")

    # Apply CONFIG defaults if not explicitly set
    if facet_cols is None:
        facet_cols = CONFIG.Plotting.default_facet_cols

    # Ensure data is a Dataset and validate it
    data = _ensure_dataset(data)
    _validate_plotting_data(data, allow_empty=True)

    # Handle empty data
    if len(data.data_vars) == 0:
        logger.error('"with_plotly() got an empty Dataset.')
        return go.Figure()

    # Handle all-scalar datasets (where all variables have no dimensions)
    # This occurs when all variables are scalar values with dims=()
    if all(len(data[var].dims) == 0 for var in data.data_vars):
        # Create a simple DataFrame with variable names as x-axis
        variables = list(data.data_vars.keys())
        values = [float(data[var].values) for var in data.data_vars]

        # Resolve colors
        color_discrete_map = resolve_colors(data, colors, engine='plotly')
        marker_colors = [color_discrete_map.get(var, '#636EFA') for var in variables]

        # Create simple plot based on mode using go (not px) for better color control
        if mode in ('stacked_bar', 'grouped_bar'):
            fig = go.Figure(data=[go.Bar(x=variables, y=values, marker_color=marker_colors)])
        elif mode == 'line':
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=variables,
                        y=values,
                        mode='lines+markers',
                        marker=dict(color=marker_colors, size=8),
                        line=dict(color='lightgray'),
                    )
                ]
            )
        elif mode == 'area':
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=variables,
                        y=values,
                        fill='tozeroy',
                        marker=dict(color=marker_colors, size=8),
                        line=dict(color='lightgray'),
                    )
                ]
            )

        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, showlegend=False)
        return fig

    # Warn if fig parameter is used with faceting
    if fig is not None and (facet_by is not None or animate_by is not None):
        logger.warning('The fig parameter is ignored when using faceting or animation. Creating a new figure.')
        fig = None

    # Convert Dataset to long-form DataFrame for Plotly Express
    # Structure: time, variable, value, scenario, period, ... (all dims as columns)
    dim_names = list(data.dims)
    df_long = data.to_dataframe().reset_index().melt(id_vars=dim_names, var_name='variable', value_name='value')

    # Validate facet_by and animate_by dimensions exist in the data
    available_dims = [col for col in df_long.columns if col not in ['variable', 'value']]

    # Check facet_by dimensions
    if facet_by is not None:
        if isinstance(facet_by, str):
            if facet_by not in available_dims:
                logger.debug(
                    f"Dimension '{facet_by}' not found in data. Available dimensions: {available_dims}. "
                    f'Ignoring facet_by parameter.'
                )
                facet_by = None
        elif isinstance(facet_by, list):
            # Filter out dimensions that don't exist
            missing_dims = [dim for dim in facet_by if dim not in available_dims]
            facet_by = [dim for dim in facet_by if dim in available_dims]
            if missing_dims:
                logger.debug(
                    f'Dimensions {missing_dims} not found in data. Available dimensions: {available_dims}. '
                    f'Using only existing dimensions: {facet_by if facet_by else "none"}.'
                )
            if len(facet_by) == 0:
                facet_by = None

    # Check animate_by dimension
    if animate_by is not None and animate_by not in available_dims:
        logger.debug(
            f"Dimension '{animate_by}' not found in data. Available dimensions: {available_dims}. "
            f'Ignoring animate_by parameter.'
        )
        animate_by = None

    # Setup faceting parameters for Plotly Express
    facet_row = None
    facet_col = None
    if facet_by:
        if isinstance(facet_by, str):
            # Single facet dimension - use facet_col with facet_col_wrap
            facet_col = facet_by
        elif len(facet_by) == 1:
            facet_col = facet_by[0]
        elif len(facet_by) == 2:
            # Two facet dimensions - use facet_row and facet_col
            facet_row = facet_by[0]
            facet_col = facet_by[1]
        else:
            raise ValueError(f'facet_by can have at most 2 dimensions, got {len(facet_by)}')

    # Process colors using resolve_colors (handles validation and all color types)
    color_discrete_map = resolve_colors(data, colors, engine='plotly')

    # Get unique variable names for area plot processing
    all_vars = df_long['variable'].unique().tolist()

    # Determine which dimension to use for x-axis
    # Collect dimensions used for faceting and animation
    used_dims = set()
    if facet_row:
        used_dims.add(facet_row)
    if facet_col:
        used_dims.add(facet_col)
    if animate_by:
        used_dims.add(animate_by)

    # Find available dimensions for x-axis (not used for faceting/animation)
    x_candidates = [d for d in available_dims if d not in used_dims]

    # Use 'time' if available, otherwise use the first available dimension
    if 'time' in x_candidates:
        x_dim = 'time'
    elif len(x_candidates) > 0:
        x_dim = x_candidates[0]
    else:
        # Fallback: use the first dimension (shouldn't happen in normal cases)
        x_dim = available_dims[0] if available_dims else 'time'

    # Create plot using Plotly Express based on mode
    common_args = {
        'data_frame': df_long,
        'x': x_dim,
        'y': 'value',
        'color': 'variable',
        'facet_row': facet_row,
        'facet_col': facet_col,
        'animation_frame': animate_by,
        'color_discrete_map': color_discrete_map,
        'title': title,
        'labels': {'value': ylabel, x_dim: xlabel, 'variable': ''},
    }

    # Add facet_col_wrap for single facet dimension
    if facet_col and not facet_row:
        common_args['facet_col_wrap'] = facet_cols

    # Apply user-provided Plotly Express kwargs (overrides defaults)
    common_args.update(px_kwargs)

    if mode == 'stacked_bar':
        fig = px.bar(**common_args)
        fig.update_traces(marker_line_width=0)
        fig.update_layout(barmode='relative', bargap=0, bargroupgap=0)
    elif mode == 'grouped_bar':
        fig = px.bar(**common_args)
        fig.update_layout(barmode='group', bargap=0.2, bargroupgap=0)
    elif mode == 'line':
        fig = px.line(**common_args, line_shape='hv')  # Stepped lines
    elif mode == 'area':
        # Use Plotly Express to create the area plot (preserves animation, legends, faceting)
        fig = px.area(**common_args, line_shape='hv')

        # Classify each variable based on its values
        variable_classification = {}
        for var in all_vars:
            var_data = df_long[df_long['variable'] == var]['value']
            var_data_clean = var_data[(var_data < -1e-5) | (var_data > 1e-5)]

            if len(var_data_clean) == 0:
                variable_classification[var] = 'zero'
            else:
                has_pos, has_neg = (var_data_clean > 0).any(), (var_data_clean < 0).any()
                variable_classification[var] = (
                    'mixed' if has_pos and has_neg else ('negative' if has_neg else 'positive')
                )

        # Log warning for mixed variables
        mixed_vars = [v for v, c in variable_classification.items() if c == 'mixed']
        if mixed_vars:
            logger.warning(f'Variables with both positive and negative values: {mixed_vars}. Plotted as dashed lines.')

        all_traces = list(fig.data)
        for frame in fig.frames:
            all_traces.extend(frame.data)

        for trace in all_traces:
            cls = variable_classification.get(trace.name, None)
            # Only stack positive and negative, not mixed or zero
            trace.stackgroup = cls if cls in ('positive', 'negative') else None

            if cls in ('positive', 'negative'):
                # Stacked area: add opacity to avoid hiding layers, remove line border
                if hasattr(trace, 'line') and trace.line.color:
                    trace.fillcolor = trace.line.color
                    trace.line.width = 0
            elif cls == 'mixed':
                # Mixed variables: show as dashed line, not stacked
                if hasattr(trace, 'line'):
                    trace.line.width = 2
                    trace.line.dash = 'dash'
                if hasattr(trace, 'fill'):
                    trace.fill = None

    # Update layout with basic styling (Plotly Express handles sizing automatically)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
    )

    # Update axes to share if requested (Plotly Express already handles this, but we can customize)
    if not shared_yaxes:
        fig.update_yaxes(matches=None)
    if not shared_xaxes:
        fig.update_xaxes(matches=None)

    # Apply user-provided trace and layout customizations
    if trace_kwargs:
        fig.update_traces(**trace_kwargs)
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)

    return fig


def with_matplotlib(
    data: xr.Dataset | pd.DataFrame,
    mode: Literal['stacked_bar', 'line'] = 'stacked_bar',
    colors: ColorType | ComponentColorManager | None = None,
    title: str = '',
    ylabel: str = '',
    xlabel: str = 'Time in h',
    figsize: tuple[int, int] = (12, 6),
    plot_kwargs: dict[str, Any] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot data with Matplotlib using stacked bars or stepped lines.

    Args:
        data: An xarray Dataset to plot. After conversion to DataFrame,
              the index represents time and each column represents a separate data series (variables).
        mode: Plotting mode. Use 'stacked_bar' for stacked bar charts or 'line' for stepped lines.
        colors: Color specification. Can be:
            - A colormap name (e.g., 'viridis', 'plasma')
            - A list of color strings (e.g., ['#ff0000', '#00ff00'])
            - A dict mapping column names to colors (e.g., {'Column1': '#ff0000'})
            - A ComponentColorManager instance for pattern-based color rules with grouping and sorting
        title: The title of the plot.
        ylabel: The ylabel of the plot.
        xlabel: The xlabel of the plot.
        figsize: Specify the size of the figure (width, height) in inches.
        plot_kwargs: Optional dict of parameters to pass to ax.bar() or ax.step() plotting calls.
                    Use this to customize plot properties (e.g., linewidth, alpha, edgecolor).

    Returns:
        A tuple containing the Matplotlib figure and axes objects used for the plot.

    Notes:
        - If `mode` is 'stacked_bar', bars are stacked for both positive and negative values.
          Negative values are stacked separately without extra labels in the legend.
        - If `mode` is 'line', stepped lines are drawn for each data series.

    Examples:
        With ComponentColorManager:

        ```python
        manager = ComponentColorManager(['Solar', 'Wind', 'Coal'])
        manager.add_grouping_rule('Solar', 'renewables', 'oranges', match_type='prefix')
        manager.add_grouping_rule('Wind', 'renewables', 'blues', match_type='prefix')
        manager.apply_colors()
        fig, ax = with_matplotlib(dataset, colors=manager, mode='line')
        ```
    """
    if colors is None:
        colors = CONFIG.Plotting.default_qualitative_colorscale

    if mode not in ('stacked_bar', 'line'):
        raise ValueError(f"'mode' must be one of {{'stacked_bar','line'}} for matplotlib, got {mode!r}")

    # Ensure data is a Dataset and validate it
    data = _ensure_dataset(data)
    _validate_plotting_data(data, allow_empty=True)

    # Create new figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Initialize plot_kwargs if not provided
    if plot_kwargs is None:
        plot_kwargs = {}

    # Handle all-scalar datasets (where all variables have no dimensions)
    # This occurs when all variables are scalar values with dims=()
    if all(len(data[var].dims) == 0 for var in data.data_vars):
        # Create simple bar/line plot with variable names as x-axis
        variables = list(data.data_vars.keys())
        values = [float(data[var].values) for var in data.data_vars]

        # Resolve colors
        color_discrete_map = resolve_colors(data, colors, engine='matplotlib')
        colors_list = [color_discrete_map.get(var, '#808080') for var in variables]

        # Create plot based on mode
        if mode == 'stacked_bar':
            ax.bar(variables, values, color=colors_list, **plot_kwargs)
        elif mode == 'line':
            ax.plot(
                variables,
                values,
                marker='o',
                color=colors_list[0] if len(set(colors_list)) == 1 else None,
                **plot_kwargs,
            )
            # If different colors, plot each point separately
            if len(set(colors_list)) > 1:
                ax.clear()
                for i, (var, val) in enumerate(zip(variables, values, strict=False)):
                    ax.plot([i], [val], marker='o', color=colors_list[i], label=var, **plot_kwargs)
                ax.set_xticks(range(len(variables)))
                ax.set_xticklabels(variables)

        ax.set_xlabel(xlabel, ha='center')
        ax.set_ylabel(ylabel, va='center')
        ax.set_title(title)
        ax.grid(color='lightgrey', linestyle='-', linewidth=0.5, axis='y')
        fig.tight_layout()

        return fig, ax

    # Resolve colors first (includes validation)
    color_discrete_map = resolve_colors(data, colors, engine='matplotlib')

    # Convert Dataset to DataFrame for matplotlib plotting (naturally wide-form)
    df = data.to_dataframe()

    # Get colors in column order
    processed_colors = [color_discrete_map.get(str(col), '#808080') for col in df.columns]

    if mode == 'stacked_bar':
        cumulative_positive = np.zeros(len(df))
        cumulative_negative = np.zeros(len(df))
        width = df.index.to_series().diff().dropna().min()  # Minimum time difference

        for i, column in enumerate(df.columns):
            positive_values = np.clip(df[column], 0, None)  # Keep only positive values
            negative_values = np.clip(df[column], None, 0)  # Keep only negative values
            # Plot positive bars
            ax.bar(
                df.index,
                positive_values,
                bottom=cumulative_positive,
                color=processed_colors[i],
                label=column,
                width=width,
                align='center',
                **plot_kwargs,
            )
            cumulative_positive += positive_values.values
            # Plot negative bars
            ax.bar(
                df.index,
                negative_values,
                bottom=cumulative_negative,
                color=processed_colors[i],
                label='',  # No label for negative bars
                width=width,
                align='center',
                **plot_kwargs,
            )
            cumulative_negative += negative_values.values

    elif mode == 'line':
        for i, column in enumerate(df.columns):
            ax.step(df.index, df[column], where='post', color=processed_colors[i], label=column, **plot_kwargs)

    # Aesthetics
    ax.set_xlabel(xlabel, ha='center')
    ax.set_ylabel(ylabel, va='center')
    ax.set_title(title)
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    ax.legend(
        loc='upper center',  # Place legend at the bottom center
        bbox_to_anchor=(0.5, -0.15),  # Adjust the position to fit below plot
        ncol=5,
        frameon=False,  # Remove box around legend
    )
    fig.tight_layout()

    return fig, ax


def reshape_data_for_heatmap(
    data: xr.DataArray,
    reshape_time: tuple[Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'], Literal['W', 'D', 'h', '15min', 'min']]
    | Literal['auto']
    | None = 'auto',
    facet_by: str | list[str] | None = None,
    animate_by: str | None = None,
    fill: Literal['ffill', 'bfill'] | None = 'ffill',
) -> xr.DataArray:
    """
    Reshape data for heatmap visualization, handling time dimension intelligently.

    This function decides whether to reshape the 'time' dimension based on the reshape_time parameter:
    - 'auto': Automatically reshapes if only 'time' dimension would remain for heatmap
    - Tuple: Explicitly reshapes time with specified parameters
    - None: No reshaping (returns data as-is)

    All non-time dimensions are preserved during reshaping.

    Args:
        data: DataArray to reshape for heatmap visualization.
        reshape_time: Reshaping configuration:
                     - 'auto' (default): Auto-reshape if needed based on facet_by/animate_by
                     - Tuple (timeframes, timesteps_per_frame): Explicit time reshaping
                     - None: No reshaping
        facet_by: Dimension(s) used for faceting (used in 'auto' decision).
        animate_by: Dimension used for animation (used in 'auto' decision).
        fill: Method to fill missing values: 'ffill' or 'bfill'. Default is 'ffill'.

    Returns:
        Reshaped DataArray. If time reshaping is applied, 'time' dimension is replaced
        by 'timestep' and 'timeframe'. All other dimensions are preserved.

    Examples:
        Auto-reshaping:

        ```python
        # Will auto-reshape because only 'time' remains after faceting/animation
        data = reshape_data_for_heatmap(data, reshape_time='auto', facet_by='scenario', animate_by='period')
        ```

        Explicit reshaping:

        ```python
        # Explicitly reshape to daily pattern
        data = reshape_data_for_heatmap(data, reshape_time=('D', 'h'))
        ```

        No reshaping:

        ```python
        # Keep data as-is
        data = reshape_data_for_heatmap(data, reshape_time=None)
        ```
    """
    # If no time dimension, return data as-is
    if 'time' not in data.dims:
        return data

    # Handle None (disabled) - return data as-is
    if reshape_time is None:
        return data

    # Determine timeframes and timesteps_per_frame based on reshape_time parameter
    if reshape_time == 'auto':
        # Check if we need automatic time reshaping
        facet_dims_used = []
        if facet_by:
            facet_dims_used = [facet_by] if isinstance(facet_by, str) else list(facet_by)
        if animate_by:
            facet_dims_used.append(animate_by)

        # Get dimensions that would remain for heatmap
        potential_heatmap_dims = [dim for dim in data.dims if dim not in facet_dims_used]

        # Auto-reshape if only 'time' dimension remains
        if len(potential_heatmap_dims) == 1 and potential_heatmap_dims[0] == 'time':
            logger.debug(
                "Auto-applying time reshaping: Only 'time' dimension remains after faceting/animation. "
                "Using default timeframes='D' and timesteps_per_frame='h'. "
                "To customize, use reshape_time=('D', 'h') or disable with reshape_time=None."
            )
            timeframes, timesteps_per_frame = 'D', 'h'
        else:
            # No reshaping needed
            return data
    elif isinstance(reshape_time, tuple):
        # Explicit reshaping
        timeframes, timesteps_per_frame = reshape_time
    else:
        raise ValueError(f"reshape_time must be 'auto', a tuple like ('D', 'h'), or None. Got: {reshape_time}")

    # Validate that time is datetime
    if not np.issubdtype(data.coords['time'].dtype, np.datetime64):
        raise ValueError(f'Time dimension must be datetime-based, got {data.coords["time"].dtype}')

    # Define formats for different combinations
    formats = {
        ('YS', 'W'): ('%Y', '%W'),
        ('YS', 'D'): ('%Y', '%j'),  # day of year
        ('YS', 'h'): ('%Y', '%j %H:00'),
        ('MS', 'D'): ('%Y-%m', '%d'),  # day of month
        ('MS', 'h'): ('%Y-%m', '%d %H:00'),
        ('W', 'D'): ('%Y-w%W', '%w_%A'),  # week and day of week
        ('W', 'h'): ('%Y-w%W', '%w_%A %H:00'),
        ('D', 'h'): ('%Y-%m-%d', '%H:00'),  # Day and hour
        ('D', '15min'): ('%Y-%m-%d', '%H:%M'),  # Day and minute
        ('h', '15min'): ('%Y-%m-%d %H:00', '%M'),  # minute of hour
        ('h', 'min'): ('%Y-%m-%d %H:00', '%M'),  # minute of hour
    }

    format_pair = (timeframes, timesteps_per_frame)
    if format_pair not in formats:
        raise ValueError(f'{format_pair} is not a valid format. Choose from {list(formats.keys())}')
    period_format, step_format = formats[format_pair]

    # Check if resampling is needed
    if data.sizes['time'] > 1:
        # Use NumPy for more efficient timedelta computation
        time_values = data.coords['time'].values  # Already numpy datetime64[ns]
        # Calculate differences and convert to minutes
        time_diffs = np.diff(time_values).astype('timedelta64[s]').astype(float) / 60.0
        if time_diffs.size > 0:
            min_time_diff_min = np.nanmin(time_diffs)
            time_intervals = {'min': 1, '15min': 15, 'h': 60, 'D': 24 * 60, 'W': 7 * 24 * 60}
            if time_intervals[timesteps_per_frame] > min_time_diff_min:
                logger.warning(
                    f'Resampling data from {min_time_diff_min:.2f} min to '
                    f'{time_intervals[timesteps_per_frame]:.2f} min. Mean values are displayed.'
                )

    # Resample along time dimension
    resampled = data.resample(time=timesteps_per_frame).mean()

    # Apply fill if specified
    if fill == 'ffill':
        resampled = resampled.ffill(dim='time')
    elif fill == 'bfill':
        resampled = resampled.bfill(dim='time')

    # Create period and step labels
    time_values = pd.to_datetime(resampled.coords['time'].values)
    period_labels = time_values.strftime(period_format)
    step_labels = time_values.strftime(step_format)

    # Handle special case for weekly day format
    if '%w_%A' in step_format:
        step_labels = pd.Series(step_labels).replace('0_Sunday', '7_Sunday').values

    # Add period and step as coordinates
    resampled = resampled.assign_coords(
        {
            'timeframe': ('time', period_labels),
            'timestep': ('time', step_labels),
        }
    )

    # Convert to multi-index and unstack
    resampled = resampled.set_index(time=['timeframe', 'timestep'])
    result = resampled.unstack('time')

    # Ensure timestep and timeframe come first in dimension order
    # Get other dimensions
    other_dims = [d for d in result.dims if d not in ['timestep', 'timeframe']]

    # Reorder: timestep, timeframe, then other dimensions
    result = result.transpose('timestep', 'timeframe', *other_dims)

    return result


def plot_network(
    node_infos: dict,
    edge_infos: dict,
    path: str | pathlib.Path | None = None,
    controls: bool
    | list[
        Literal['nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer']
    ] = True,
    show: bool = False,
) -> pyvis.network.Network | None:
    """
    Visualizes the network structure of a FlowSystem using PyVis, using info-dictionaries.

    Args:
        path: Path to save the HTML visualization. `False`: Visualization is created but not saved. `str` or `Path`: Specifies file path (default: 'results/network.html').
        controls: UI controls to add to the visualization. `True`: Enables all available controls. `list`: Specify controls, e.g., ['nodes', 'layout'].
            Options: 'nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer'.
            You can play with these and generate a Dictionary from it that can be applied to the network returned by this function.
            network.set_options()
            https://pyvis.readthedocs.io/en/latest/tutorial.html
        show: Whether to open the visualization in the web browser.
            The calculation must be saved to show it. If no path is given, it defaults to 'network.html'.
    Returns:
        The `Network` instance representing the visualization, or `None` if `pyvis` is not installed.

    Notes:
    - This function requires `pyvis`. If not installed, the function prints a warning and returns `None`.
    - Nodes are styled based on type (e.g., circles for buses, boxes for components) and annotated with node information.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        logger.critical("Plotting the flow system network was not possible. Please install pyvis: 'pip install pyvis'")
        return None

    net = Network(directed=True, height='100%' if controls is False else '800px', font_color='white')

    for node_id, node in node_infos.items():
        net.add_node(
            node_id,
            label=node['label'],
            shape={'Bus': 'circle', 'Component': 'box'}[node['class']],
            color={'Bus': '#393E46', 'Component': '#00ADB5'}[node['class']],
            title=node['infos'].replace(')', '\n)'),
            font={'size': 14},
        )

    for edge in edge_infos.values():
        net.add_edge(
            edge['start'],
            edge['end'],
            label=edge['label'],
            title=edge['infos'].replace(')', '\n)'),
            font={'color': '#4D4D4D', 'size': 14},
            color='#222831',
        )

    # Enhanced physics settings
    net.barnes_hut(central_gravity=0.8, spring_length=50, spring_strength=0.05, gravity=-10000)

    if controls:
        net.show_buttons(filter_=controls)  # Adds UI buttons to control physics settings
    if not show and not path:
        return net
    elif path:
        path = pathlib.Path(path) if isinstance(path, str) else path
        net.write_html(path.as_posix())
    elif show:
        path = pathlib.Path('network.html')
        net.write_html(path.as_posix())

    if show:
        try:
            import webbrowser

            worked = webbrowser.open(f'file://{path.resolve()}', 2)
            if not worked:
                logger.error(f'Showing the network in the Browser went wrong. Open it manually. Its saved under {path}')
        except Exception as e:
            logger.error(
                f'Showing the network in the Browser went wrong. Open it manually. Its saved under {path}: {e}'
            )


def pie_with_plotly(
    data: xr.Dataset | pd.DataFrame,
    colors: ColorType | ComponentColorManager | None = None,
    title: str = '',
    legend_title: str = '',
    hole: float = 0.0,
    fig: go.Figure | None = None,
    hover_template: str = '%{label}: %{value} (%{percent})',
    text_info: str = 'percent+label+value',
    text_position: str = 'inside',
) -> go.Figure:
    """
    Create a pie chart with Plotly to visualize the proportion of values in a Dataset.

    Args:
        data: An xarray Dataset containing the data to plot. All dimensions will be summed
              to get the total for each variable.
        colors: Color specification, can be:
            - A string with a colorscale name (e.g., 'viridis', 'plasma')
            - A list of color strings (e.g., ['#ff0000', '#00ff00'])
            - A dictionary mapping variable names to colors (e.g., {'Solar': '#ff0000'})
            - A ComponentColorManager instance for pattern-based color rules
        title: The title of the plot.
        legend_title: The title for the legend.
        hole: Size of the hole in the center for creating a donut chart (0.0 to 1.0).
        fig: A Plotly figure object to plot on. If not provided, a new figure will be created.
        hover_template: Template for hover text. Use %{label}, %{value}, %{percent}.
        text_info: What to show on pie segments: 'label', 'percent', 'value', 'label+percent',
                  'label+value', 'percent+value', 'label+percent+value', or 'none'.
        text_position: Position of text: 'inside', 'outside', 'auto', or 'none'.

    Returns:
        A Plotly figure object containing the generated pie chart.

    Notes:
        - Negative values are not appropriate for pie charts and will be converted to absolute values with a warning.
        - All dimensions are summed to get total values for each variable.
        - Scalar variables (with no dimensions) are used directly.

    Examples:
        Simple pie chart:

        ```python
        fig = pie_with_plotly(dataset, colors='viridis', title='Energy Mix')
        ```

        With ComponentColorManager:

        ```python
        manager = ComponentColorManager(['Solar', 'Wind', 'Coal'])
        manager.add_grouping_rule('Solar', 'renewables', 'oranges', match_type='prefix')
        manager.add_grouping_rule('Wind', 'renewables', 'blues', match_type='prefix')
        manager.apply_colors()
        fig = pie_with_plotly(dataset, colors=manager, title='Renewable Energy')
        ```
    """
    if colors is None:
        colors = CONFIG.Plotting.default_qualitative_colorscale

    # Ensure data is a Dataset and validate it
    data = _ensure_dataset(data)
    _validate_plotting_data(data, allow_empty=True)

    if len(data.data_vars) == 0:
        logger.error('Empty Dataset provided for pie chart. Returning empty figure.')
        return go.Figure()

    # Sum all dimensions for each variable to get total values
    labels = []
    values = []

    for var in data.data_vars:
        var_data = data[var]

        # Sum across all dimensions to get total
        if len(var_data.dims) > 0:
            total_value = var_data.sum().item()
        else:
            # Scalar variable
            total_value = var_data.item()

        # Check for negative values
        if total_value < 0:
            logger.warning(f'Negative value detected for {var}: {total_value}. Using absolute value.')
            total_value = abs(total_value)

        labels.append(str(var))
        values.append(total_value)

    # Use resolve_colors for consistent color handling
    color_discrete_map = resolve_colors(data, colors, engine='plotly')
    processed_colors = [color_discrete_map.get(label, '#636EFA') for label in labels]

    # Create figure if not provided
    fig = fig if fig is not None else go.Figure()

    # Add pie trace
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=hole,
            marker=dict(colors=processed_colors),
            textinfo=text_info,
            textposition=text_position,
            insidetextorientation='radial',
            hovertemplate=hover_template,
        )
    )

    # Update layout for better aesthetics
    fig.update_layout(
        title=title,
        legend_title=legend_title,
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
        font=dict(size=14),  # Increase font size for better readability
    )

    return fig


def pie_with_matplotlib(
    data: xr.Dataset | pd.DataFrame,
    colors: ColorType | ComponentColorManager | None = None,
    title: str = '',
    legend_title: str = 'Categories',
    hole: float = 0.0,
    figsize: tuple[int, int] = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a pie chart with Matplotlib to visualize the proportion of values in a Dataset.

    Args:
        data: An xarray Dataset containing the data to plot. All dimensions will be summed
              to get the total for each variable.
        colors: Color specification, can be:
            - A string with a colormap name (e.g., 'viridis', 'plasma')
            - A list of color strings (e.g., ['#ff0000', '#00ff00'])
            - A dictionary mapping variable names to colors (e.g., {'Solar': '#ff0000'})
            - A ComponentColorManager instance for pattern-based color rules
        title: The title of the plot.
        legend_title: The title for the legend.
        hole: Size of the hole in the center for creating a donut chart (0.0 to 1.0).
        figsize: The size of the figure (width, height) in inches.

    Returns:
        A tuple containing the Matplotlib figure and axes objects used for the plot.

    Notes:
        - Negative values are not appropriate for pie charts and will be converted to absolute values with a warning.
        - All dimensions are summed to get total values for each variable.
        - Scalar variables (with no dimensions) are used directly.

    Examples:
        Simple pie chart:

        ```python
        fig, ax = pie_with_matplotlib(dataset, colors='viridis', title='Energy Mix')
        ```

        With ComponentColorManager:

        ```python
        manager = ComponentColorManager(['Solar', 'Wind', 'Coal'])
        manager.add_grouping_rule('Solar', 'renewables', 'oranges', match_type='prefix')
        manager.add_grouping_rule('Wind', 'renewables', 'blues', match_type='prefix')
        manager.apply_colors()
        fig, ax = pie_with_matplotlib(dataset, colors=manager, title='Renewable Energy')
        ```
    """
    if colors is None:
        colors = CONFIG.Plotting.default_qualitative_colorscale

    # Ensure data is a Dataset and validate it
    data = _ensure_dataset(data)
    _validate_plotting_data(data, allow_empty=True)

    if len(data.data_vars) == 0:
        logger.error('Empty Dataset provided for pie chart. Returning empty figure.')
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    # Sum all dimensions for each variable to get total values
    labels = []
    values = []

    for var in data.data_vars:
        var_data = data[var]

        # Sum across all dimensions to get total
        if len(var_data.dims) > 0:
            total_value = var_data.sum().item()
        else:
            # Scalar variable
            total_value = var_data.item()

        # Check for negative values
        if total_value < 0:
            logger.warning(f'Negative value detected for {var}: {total_value}. Using absolute value.')
            total_value = abs(total_value)

        labels.append(str(var))
        values.append(total_value)

    # Use resolve_colors for consistent color handling
    color_discrete_map = resolve_colors(data, colors, engine='matplotlib')
    processed_colors = [color_discrete_map.get(label, '#808080') for label in labels]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Draw the pie chart
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=processed_colors,
        autopct='%1.1f%%',
        startangle=90,
        shadow=False,
        wedgeprops=dict(width=0.5) if hole > 0 else None,  # Set width for donut
    )

    # Adjust the wedgeprops to make donut hole size consistent with plotly
    # For matplotlib, the hole size is determined by the wedge width
    # Convert hole parameter to wedge width
    if hole > 0:
        # Adjust hole size to match plotly's hole parameter
        # In matplotlib, wedge width is relative to the radius (which is 1)
        # For plotly, hole is a fraction of the radius
        wedge_width = 1 - hole
        for wedge in wedges:
            wedge.set_width(wedge_width)

    # Customize the appearance
    # Make autopct text more visible
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')

    # Set aspect ratio to be equal to ensure a circular pie
    ax.set_aspect('equal')

    # Add title
    if title:
        ax.set_title(title, fontsize=16)

    # Create a legend if there are many segments
    if len(labels) > 6:
        ax.legend(wedges, labels, title=legend_title, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))

    # Apply tight layout
    fig.tight_layout()

    return fig, ax


def dual_pie_with_plotly(
    data_left: xr.Dataset | pd.DataFrame,
    data_right: xr.Dataset | pd.DataFrame,
    colors: ColorType | ComponentColorManager | None = None,
    title: str = '',
    subtitles: tuple[str, str] = ('Left Chart', 'Right Chart'),
    legend_title: str = '',
    hole: float = 0.2,
    lower_percentage_group: float = 5.0,
    hover_template: str = '%{label}: %{value} (%{percent})',
    text_info: str = 'percent+label',
    text_position: str = 'inside',
) -> go.Figure:
    """
    Create two pie charts side by side with Plotly, with consistent coloring across both charts.

    Args:
        data_left: Dataset for the left pie chart. Variables are summed across all dimensions.
        data_right: Dataset for the right pie chart. Variables are summed across all dimensions.
        colors: Color specification, can be:
            - A string with a colorscale name (e.g., 'viridis', 'plasma')
            - A list of color strings (e.g., ['#ff0000', '#00ff00'])
            - A dictionary mapping variable names to colors (e.g., {'Solar': '#ff0000'})
            - A ComponentColorManager instance for pattern-based color rules
        title: The main title of the plot.
        subtitles: Tuple containing the subtitles for (left, right) charts.
        legend_title: The title for the legend.
        hole: Size of the hole in the center for creating donut charts (0.0 to 1.0).
        lower_percentage_group: Group segments whose cumulative share is below this percentage (0–100) into "Other".
        hover_template: Template for hover text. Use %{label}, %{value}, %{percent}.
        text_info: What to show on pie segments: 'label', 'percent', 'value', 'label+percent',
                  'label+value', 'percent+value', 'label+percent+value', or 'none'.
        text_position: Position of text: 'inside', 'outside', 'auto', or 'none'.

    Returns:
        A Plotly figure object containing the generated dual pie chart.
    """
    if colors is None:
        colors = CONFIG.Plotting.default_qualitative_colorscale

    from plotly.subplots import make_subplots

    # Ensure data is a Dataset and validate it
    data_left = _ensure_dataset(data_left)
    data_right = _ensure_dataset(data_right)
    _validate_plotting_data(data_left, allow_empty=True)
    _validate_plotting_data(data_right, allow_empty=True)

    # Check for empty data
    if len(data_left.data_vars) == 0 and len(data_right.data_vars) == 0:
        logger.error('Both datasets are empty. Returning empty figure.')
        return go.Figure()

    # Create a subplot figure
    fig = make_subplots(
        rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'pie'}]], subplot_titles=subtitles, horizontal_spacing=0.05
    )

    # Helper function to extract labels and values from Dataset
    def dataset_to_pie_data(dataset):
        labels = []
        values = []

        for var in dataset.data_vars:
            var_data = dataset[var]

            # Sum across all dimensions
            if len(var_data.dims) > 0:
                total_value = float(var_data.sum().values)
            else:
                total_value = float(var_data.values)

            # Handle negative values
            if total_value < 0:
                logger.warning(f'Negative value for {var}: {total_value}. Using absolute value.')
                total_value = abs(total_value)

            # Only include if value > 0
            if total_value > 0:
                labels.append(str(var))
                values.append(total_value)

        return labels, values

    # Get data for left and right
    left_labels, left_values = dataset_to_pie_data(data_left)
    right_labels, right_values = dataset_to_pie_data(data_right)

    # Get unique set of all labels for consistent coloring across both pies
    # Merge both datasets for color resolution
    combined_vars = list(set(data_left.data_vars) | set(data_right.data_vars))
    combined_ds = xr.Dataset(
        {var: data_left[var] if var in data_left.data_vars else data_right[var] for var in combined_vars}
    )

    # Use resolve_colors for consistent color handling
    color_discrete_map = resolve_colors(combined_ds, colors, engine='plotly')
    color_map = {label: color_discrete_map.get(label, '#636EFA') for label in left_labels + right_labels}

    # Function to create a pie trace with consistently mapped colors
    def create_pie_trace(labels, values, side):
        if not labels:
            return None

        trace_colors = [color_map[label] for label in labels]

        return go.Pie(
            labels=labels,
            values=values,
            name=side,
            marker=dict(colors=trace_colors),
            hole=hole,
            textinfo=text_info,
            textposition=text_position,
            insidetextorientation='radial',
            hovertemplate=hover_template,
            sort=True,  # Sort values by default (largest first)
        )

    # Add left pie if data exists
    left_trace = create_pie_trace(left_labels, left_values, subtitles[0])
    if left_trace:
        left_trace.domain = dict(x=[0, 0.48])
        fig.add_trace(left_trace, row=1, col=1)

    # Add right pie if data exists
    right_trace = create_pie_trace(right_labels, right_values, subtitles[1])
    if right_trace:
        right_trace.domain = dict(x=[0.52, 1])
        fig.add_trace(right_trace, row=1, col=2)

    # Update layout
    fig.update_layout(
        title=title,
        legend_title=legend_title,
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
        font=dict(size=14),
        margin=dict(t=80, b=50, l=30, r=30),
    )

    return fig


def dual_pie_with_matplotlib(
    data_left: pd.Series,
    data_right: pd.Series,
    colors: ColorType | None = None,
    title: str = '',
    subtitles: tuple[str, str] = ('Left Chart', 'Right Chart'),
    legend_title: str = '',
    hole: float = 0.2,
    lower_percentage_group: float = 5.0,
    figsize: tuple[int, int] = (14, 7),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Create two pie charts side by side with Matplotlib, with consistent coloring across both charts.

    Args:
        data_left: Series for the left pie chart.
        data_right: Series for the right pie chart.
        colors: Color specification, can be:
            - A string with a colormap name (e.g., 'viridis', 'plasma')
            - A list of color strings (e.g., ['#ff0000', '#00ff00'])
            - A dictionary mapping category names to colors (e.g., {'Category1': '#ff0000'})
        title: The main title of the plot.
        subtitles: Tuple containing the subtitles for (left, right) charts.
        legend_title: The title for the legend.
        hole: Size of the hole in the center for creating donut charts (0.0 to 1.0).
        lower_percentage_group: Whether to group small segments (below percentage) into an "Other" category.
        figsize: The size of the figure (width, height) in inches.

    Returns:
        A tuple containing the Matplotlib figure and list of axes objects used for the plot.
    """
    if colors is None:
        colors = CONFIG.Plotting.default_qualitative_colorscale

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Check for empty data
    if data_left.empty and data_right.empty:
        logger.error('Both datasets are empty. Returning empty figure.')
        return fig, axes

    # Process series to handle negative values and apply minimum percentage threshold
    def preprocess_series(series: pd.Series):
        """
        Preprocess a series for pie chart display by handling negative values
        and grouping the smallest parts together if they collectively represent
        less than the specified percentage threshold.
        """
        # Handle negative values
        if (series < 0).any():
            logger.error('Negative values detected in data. Using absolute values for pie chart.')
            series = series.abs()

        # Remove zeros
        series = series[series > 0]

        # Apply minimum percentage threshold if needed
        if lower_percentage_group and not series.empty:
            total = series.sum()
            if total > 0:
                # Sort series by value (ascending)
                sorted_series = series.sort_values()

                # Calculate cumulative percentage contribution
                cumulative_percent = (sorted_series.cumsum() / total) * 100

                # Find entries that collectively make up less than lower_percentage_group
                to_group = cumulative_percent <= lower_percentage_group

                if to_group.sum() > 1:
                    # Create "Other" category for the smallest values that together are < threshold
                    other_sum = sorted_series[to_group].sum()

                    # Keep only values that aren't in the "Other" group
                    result_series = series[~series.index.isin(sorted_series[to_group].index)]

                    # Add the "Other" category if it has a value
                    if other_sum > 0:
                        result_series['Other'] = other_sum

                    return result_series

        return series

    # Preprocess data
    data_left_processed = preprocess_series(data_left)
    data_right_processed = preprocess_series(data_right)

    # Convert Series to DataFrames for pie_with_matplotlib
    df_left = pd.DataFrame(data_left_processed).T if not data_left_processed.empty else pd.DataFrame()
    df_right = pd.DataFrame(data_right_processed).T if not data_right_processed.empty else pd.DataFrame()

    # Get unique set of all labels for consistent coloring
    all_labels = sorted(set(data_left_processed.index) | set(data_right_processed.index))

    # Get consistent color mapping for both charts using our unified function
    color_map = ColorProcessor(engine='matplotlib').process_colors(colors, all_labels, return_mapping=True)

    # Configure colors for each DataFrame based on the consistent mapping
    left_colors = [color_map[col] for col in df_left.columns] if not df_left.empty else []
    right_colors = [color_map[col] for col in df_right.columns] if not df_right.empty else []

    # Helper function to draw pie chart on a specific axis
    def draw_pie_on_axis(ax, data_series, colors_list, subtitle, hole_size):
        """Draw a pie chart on a specific matplotlib axis."""
        if data_series.empty:
            ax.set_title(subtitle)
            ax.axis('off')
            return

        labels = list(data_series.index)
        values = list(data_series.values)

        # Draw the pie chart
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=colors_list,
            autopct='%1.1f%%',
            startangle=90,
            shadow=False,
            wedgeprops=dict(width=0.5) if hole_size > 0 else None,
        )

        # Adjust hole size
        if hole_size > 0:
            wedge_width = 1 - hole_size
            for wedge in wedges:
                wedge.set_width(wedge_width)

        # Customize text
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color('white')

        # Set aspect ratio and title
        ax.set_aspect('equal')
        if subtitle:
            ax.set_title(subtitle, fontsize=14)

    # Create left pie chart
    draw_pie_on_axis(axes[0], data_left_processed, left_colors, subtitles[0], hole)

    # Create right pie chart
    draw_pie_on_axis(axes[1], data_right_processed, right_colors, subtitles[1], hole)

    # Add main title
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)

    # Adjust layout
    fig.tight_layout()

    # Create a unified legend if both charts have data
    if not df_left.empty and not df_right.empty:
        # Remove individual legends
        for ax in axes:
            if ax.get_legend():
                ax.get_legend().remove()

        # Create handles for the unified legend
        handles = []
        labels_for_legend = []

        for label in all_labels:
            color = color_map[label]
            patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
            handles.append(patch)
            labels_for_legend.append(label)

        # Add unified legend
        fig.legend(
            handles=handles,
            labels=labels_for_legend,
            title=legend_title,
            loc='lower center',
            bbox_to_anchor=(0.5, 0),
            ncol=min(len(all_labels), 5),  # Limit columns to 5 for readability
        )

        # Add padding at the bottom for the legend
        fig.subplots_adjust(bottom=0.2)

    return fig, axes


def heatmap_with_plotly(
    data: xr.DataArray,
    colors: ColorType | None = None,
    title: str = '',
    facet_by: str | list[str] | None = None,
    animate_by: str | None = None,
    facet_cols: int | None = None,
    reshape_time: tuple[Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'], Literal['W', 'D', 'h', '15min', 'min']]
    | Literal['auto']
    | None = 'auto',
    fill: Literal['ffill', 'bfill'] | None = 'ffill',
    **imshow_kwargs: Any,
) -> go.Figure:
    """
    Plot a heatmap visualization using Plotly's imshow with faceting and animation support.

    This function creates heatmap visualizations from xarray DataArrays, supporting
    multi-dimensional data through faceting (subplots) and animation. It automatically
    handles dimension reduction and data reshaping for optimal heatmap display.

    Automatic Time Reshaping:
        If only the 'time' dimension remains after faceting/animation (making the data 1D),
        the function automatically reshapes time into a 2D format using default values
        (timeframes='D', timesteps_per_frame='h'). This creates a daily pattern heatmap
        showing hours vs days.

    Args:
        data: An xarray DataArray containing the data to visualize. Should have at least
              2 dimensions, or a 'time' dimension that can be reshaped into 2D.
        colors: Color specification (colormap name, list, or dict). Common options:
                'viridis', 'plasma', 'RdBu', 'portland'.
        title: The main title of the heatmap.
        facet_by: Dimension to create facets for. Creates a subplot grid.
                  Can be a single dimension name or list (only first dimension used).
                  Note: px.imshow only supports single-dimension faceting.
                  If the dimension doesn't exist in the data, it will be silently ignored.
        animate_by: Dimension to animate over. Creates animation frames.
                    If the dimension doesn't exist in the data, it will be silently ignored.
        facet_cols: Number of columns in the facet grid (used with facet_by).
        reshape_time: Time reshaping configuration:
                     - 'auto' (default): Automatically applies ('D', 'h') if only 'time' dimension remains
                     - Tuple like ('D', 'h'): Explicit time reshaping (days vs hours)
                     - None: Disable time reshaping (will error if only 1D time data)
        fill: Method to fill missing values when reshaping time: 'ffill' or 'bfill'. Default is 'ffill'.
        **imshow_kwargs: Additional keyword arguments to pass to plotly.express.imshow.
                        Common options include:
                        - aspect: 'auto', 'equal', or a number for aspect ratio
                        - zmin, zmax: Minimum and maximum values for color scale
                        - labels: Dict to customize axis labels

    Returns:
        A Plotly figure object containing the heatmap visualization.

    Examples:
        Simple heatmap:

        ```python
        fig = heatmap_with_plotly(data_array, colors='RdBu', title='Temperature Map')
        ```

        Facet by scenario:

        ```python
        fig = heatmap_with_plotly(data_array, facet_by='scenario', facet_cols=2)
        ```

        Animate by period:

        ```python
        fig = heatmap_with_plotly(data_array, animate_by='period')
        ```

        Automatic time reshaping (when only time dimension remains):

        ```python
        # Data with dims ['time', 'scenario', 'period']
        # After faceting and animation, only 'time' remains -> auto-reshapes to (timestep, timeframe)
        fig = heatmap_with_plotly(data_array, facet_by='scenario', animate_by='period')
        ```

        Explicit time reshaping:

        ```python
        fig = heatmap_with_plotly(data_array, facet_by='scenario', animate_by='period', reshape_time=('W', 'D'))
        ```
    """
    if colors is None:
        colors = CONFIG.Plotting.default_sequential_colorscale

    # Apply CONFIG defaults if not explicitly set
    if facet_cols is None:
        facet_cols = CONFIG.Plotting.default_facet_cols

    # Handle empty data
    if data.size == 0:
        return go.Figure()

    # Apply time reshaping using the new unified function
    data = reshape_data_for_heatmap(
        data, reshape_time=reshape_time, facet_by=facet_by, animate_by=animate_by, fill=fill
    )

    # Get available dimensions
    available_dims = list(data.dims)

    # Validate and filter facet_by dimensions
    if facet_by is not None:
        if isinstance(facet_by, str):
            if facet_by not in available_dims:
                logger.debug(
                    f"Dimension '{facet_by}' not found in data. Available dimensions: {available_dims}. "
                    f'Ignoring facet_by parameter.'
                )
                facet_by = None
        elif isinstance(facet_by, list):
            missing_dims = [dim for dim in facet_by if dim not in available_dims]
            facet_by = [dim for dim in facet_by if dim in available_dims]
            if missing_dims:
                logger.debug(
                    f'Dimensions {missing_dims} not found in data. Available dimensions: {available_dims}. '
                    f'Using only existing dimensions: {facet_by if facet_by else "none"}.'
                )
            if len(facet_by) == 0:
                facet_by = None

    # Validate animate_by dimension
    if animate_by is not None and animate_by not in available_dims:
        logger.debug(
            f"Dimension '{animate_by}' not found in data. Available dimensions: {available_dims}. "
            f'Ignoring animate_by parameter.'
        )
        animate_by = None

    # Determine which dimensions are used for faceting/animation
    facet_dims = []
    if facet_by:
        facet_dims = [facet_by] if isinstance(facet_by, str) else facet_by
    if animate_by:
        facet_dims.append(animate_by)

    # Get remaining dimensions for the heatmap itself
    heatmap_dims = [dim for dim in available_dims if dim not in facet_dims]

    if len(heatmap_dims) < 2:
        # Handle single-dimension case by adding variable name as a dimension
        if len(heatmap_dims) == 1:
            # Get the variable name, or use a default
            var_name = data.name if data.name else 'value'

            # Expand the DataArray by adding a new dimension with the variable name
            data = data.expand_dims({'variable': [var_name]})

            # Update available dimensions
            available_dims = list(data.dims)
            heatmap_dims = [dim for dim in available_dims if dim not in facet_dims]

            logger.debug(f'Only 1 dimension remaining for heatmap. Added variable dimension: {var_name}')
        else:
            # No dimensions at all - cannot create a heatmap
            logger.error(
                f'Heatmap requires at least 1 dimension. '
                f'After faceting/animation, {len(heatmap_dims)} dimension(s) remain: {heatmap_dims}'
            )
            return go.Figure()

    # Setup faceting parameters for Plotly Express
    # Note: px.imshow only supports facet_col, not facet_row
    facet_col_param = None
    if facet_by:
        if isinstance(facet_by, str):
            facet_col_param = facet_by
        elif len(facet_by) == 1:
            facet_col_param = facet_by[0]
        elif len(facet_by) >= 2:
            # px.imshow doesn't support facet_row, so we can only facet by one dimension
            # Use the first dimension and warn about the rest
            facet_col_param = facet_by[0]
            logger.warning(
                f'px.imshow only supports faceting by a single dimension. '
                f'Using {facet_by[0]} for faceting. Dimensions {facet_by[1:]} will be ignored. '
                f'Consider using animate_by for additional dimensions.'
            )

    # Create the imshow plot - px.imshow can work directly with xarray DataArrays
    common_args = {
        'img': data,
        'color_continuous_scale': colors if isinstance(colors, str) else CONFIG.Plotting.default_sequential_colorscale,
        'title': title,
    }

    # Add faceting if specified
    if facet_col_param:
        common_args['facet_col'] = facet_col_param
        if facet_cols:
            common_args['facet_col_wrap'] = facet_cols

    # Add animation if specified
    if animate_by:
        common_args['animation_frame'] = animate_by

    # Merge in additional imshow kwargs
    common_args.update(imshow_kwargs)

    try:
        fig = px.imshow(**common_args)
    except Exception as e:
        logger.error(f'Error creating imshow plot: {e}. Falling back to basic heatmap.')
        # Fallback: create a simple heatmap without faceting
        fallback_args = {
            'img': data.values,
            'color_continuous_scale': colors
            if isinstance(colors, str)
            else CONFIG.Plotting.default_sequential_colorscale,
            'title': title,
        }
        fallback_args.update(imshow_kwargs)
        fig = px.imshow(**fallback_args)

    # Update layout with basic styling
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
    )

    return fig


def heatmap_with_matplotlib(
    data: xr.DataArray,
    colors: ColorType | None = None,
    title: str = '',
    figsize: tuple[float, float] = (12, 6),
    reshape_time: tuple[Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'], Literal['W', 'D', 'h', '15min', 'min']]
    | Literal['auto']
    | None = 'auto',
    fill: Literal['ffill', 'bfill'] | None = 'ffill',
    vmin: float | None = None,
    vmax: float | None = None,
    imshow_kwargs: dict[str, Any] | None = None,
    cbar_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a heatmap visualization using Matplotlib's imshow.

    This function creates a basic 2D heatmap from an xarray DataArray using matplotlib's
    imshow function. For multi-dimensional data, only the first two dimensions are used.

    Args:
        data: An xarray DataArray containing the data to visualize. Should have at least
              2 dimensions. If more than 2 dimensions exist, additional dimensions will
              be reduced by taking the first slice.
        colors: Color specification. Should be a colormap name (e.g., 'viridis', 'RdBu').
        title: The title of the heatmap.
        figsize: The size of the figure (width, height) in inches.
        reshape_time: Time reshaping configuration:
                     - 'auto' (default): Automatically applies ('D', 'h') if only 'time' dimension
                     - Tuple like ('D', 'h'): Explicit time reshaping (days vs hours)
                     - None: Disable time reshaping
        fill: Method to fill missing values when reshaping time: 'ffill' or 'bfill'. Default is 'ffill'.
        vmin: Minimum value for color scale. If None, uses data minimum.
        vmax: Maximum value for color scale. If None, uses data maximum.
        imshow_kwargs: Optional dict of parameters to pass to ax.imshow().
                      Use this to customize image properties (e.g., interpolation, aspect).
        cbar_kwargs: Optional dict of parameters to pass to plt.colorbar().
                    Use this to customize colorbar properties (e.g., orientation, label).
        **kwargs: Additional keyword arguments passed to ax.imshow().
                 Common options include:
                 - interpolation: 'nearest', 'bilinear', 'bicubic', etc.
                 - alpha: Transparency level (0-1)
                 - extent: [left, right, bottom, top] for axis limits

    Returns:
        A tuple containing the Matplotlib figure and axes objects used for the plot.

    Notes:
        - Matplotlib backend doesn't support faceting or animation. Use plotly engine for those features.
        - The y-axis is automatically inverted to display data with origin at top-left.
        - A colorbar is added to show the value scale.

    Examples:
        ```python
        fig, ax = heatmap_with_matplotlib(data_array, colors='RdBu', title='Temperature')
        plt.savefig('heatmap.png')
        ```

        Time reshaping:

        ```python
        fig, ax = heatmap_with_matplotlib(data_array, reshape_time=('D', 'h'))
        ```
    """
    if colors is None:
        colors = CONFIG.Plotting.default_sequential_colorscale

    # Initialize kwargs if not provided
    if imshow_kwargs is None:
        imshow_kwargs = {}
    if cbar_kwargs is None:
        cbar_kwargs = {}

    # Merge any additional kwargs into imshow_kwargs
    # This allows users to pass imshow options directly
    imshow_kwargs.update(kwargs)

    # Handle empty data
    if data.size == 0:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    # Apply time reshaping using the new unified function
    # Matplotlib doesn't support faceting/animation, so we pass None for those
    data = reshape_data_for_heatmap(data, reshape_time=reshape_time, facet_by=None, animate_by=None, fill=fill)

    # Handle single-dimension case by adding variable name as a dimension
    if isinstance(data, xr.DataArray) and len(data.dims) == 1:
        var_name = data.name if data.name else 'value'
        data = data.expand_dims({'variable': [var_name]})
        logger.debug(f'Only 1 dimension in data. Added variable dimension: {var_name}')

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data values
    # If data has more than 2 dimensions, we need to reduce it
    if isinstance(data, xr.DataArray):
        # Get the first 2 dimensions
        dims = list(data.dims)
        if len(dims) > 2:
            logger.warning(
                f'Data has {len(dims)} dimensions: {dims}. '
                f'Only the first 2 will be used for the heatmap. '
                f'Use the plotly engine for faceting/animation support.'
            )
            # Select only the first 2 dimensions by taking first slice of others
            selection = {dim: 0 for dim in dims[2:]}
            data = data.isel(selection)

        values = data.values
        x_labels = data.dims[1] if len(data.dims) > 1 else 'x'
        y_labels = data.dims[0] if len(data.dims) > 0 else 'y'
    else:
        values = data
        x_labels = 'x'
        y_labels = 'y'

    # Process colormap
    cmap = colors if isinstance(colors, str) else CONFIG.Plotting.default_sequential_colorscale

    # Create the heatmap using imshow with user customizations
    imshow_defaults = {'cmap': cmap, 'aspect': 'auto', 'origin': 'upper', 'vmin': vmin, 'vmax': vmax}
    imshow_defaults.update(imshow_kwargs)  # User kwargs override defaults
    im = ax.imshow(values, **imshow_defaults)

    # Add colorbar with user customizations
    cbar_defaults = {'ax': ax, 'orientation': 'horizontal', 'pad': 0.1, 'aspect': 15, 'fraction': 0.05}
    cbar_defaults.update(cbar_kwargs)  # User kwargs override defaults
    cbar = plt.colorbar(im, **cbar_defaults)

    # Set colorbar label if not overridden by user
    if 'label' not in cbar_kwargs:
        cbar.set_label('Value')

    # Set labels and title
    ax.set_xlabel(str(x_labels).capitalize())
    ax.set_ylabel(str(y_labels).capitalize())
    ax.set_title(title)

    # Apply tight layout
    fig.tight_layout()

    return fig, ax


def export_figure(
    figure_like: go.Figure | tuple[plt.Figure, plt.Axes],
    default_path: pathlib.Path,
    default_filetype: str | None = None,
    user_path: pathlib.Path | None = None,
    show: bool | None = None,
    save: bool = False,
    dpi: int | None = None,
) -> go.Figure | tuple[plt.Figure, plt.Axes]:
    """
    Export a figure to a file and or show it.

    Args:
        figure_like: The figure to export. Can be a Plotly figure or a tuple of Matplotlib figure and axes.
        default_path: The default file path if no user filename is provided.
        default_filetype: The default filetype if the path doesnt end with a filetype.
        user_path: An optional user-specified file path.
        show: Whether to display the figure. If None, uses CONFIG.Plotting.default_show (default: None).
        save: Whether to save the figure (default: False).
        dpi: DPI (dots per inch) for saving Matplotlib figures. If None, uses CONFIG.Plotting.default_dpi.

    Raises:
        ValueError: If no default filetype is provided and the path doesn't specify a filetype.
        TypeError: If the figure type is not supported.
    """
    # Apply CONFIG defaults if not explicitly set
    if show is None:
        show = CONFIG.Plotting.default_show

    if dpi is None:
        dpi = CONFIG.Plotting.default_dpi

    filename = user_path or default_path
    filename = filename.with_name(filename.name.replace('|', '__'))
    if filename.suffix == '':
        if default_filetype is None:
            raise ValueError('No default filetype provided')
        filename = filename.with_suffix(default_filetype)

    if isinstance(figure_like, plotly.graph_objs.Figure):
        fig = figure_like

        # Apply default dimensions if configured
        layout_updates = {}
        if CONFIG.Plotting.default_figure_width is not None:
            layout_updates['width'] = CONFIG.Plotting.default_figure_width
        if CONFIG.Plotting.default_figure_height is not None:
            layout_updates['height'] = CONFIG.Plotting.default_figure_height
        if layout_updates:
            fig.update_layout(**layout_updates)

        if filename.suffix != '.html':
            logger.warning(f'To save a Plotly figure, using .html. Adjusting suffix for {filename}')
            filename = filename.with_suffix('.html')

        try:
            is_test_env = 'PYTEST_CURRENT_TEST' in os.environ

            if is_test_env:
                # Test environment: never open browser, only save if requested
                if save:
                    fig.write_html(str(filename))
                # Ignore show flag in tests
            else:
                # Production environment: respect show and save flags
                if save and show:
                    # Save and auto-open in browser
                    plotly.offline.plot(fig, filename=str(filename))
                elif save and not show:
                    # Save without opening
                    fig.write_html(str(filename))
                elif show and not save:
                    # Show interactively without saving
                    fig.show()
                # If neither save nor show: do nothing
        finally:
            # Cleanup to prevent socket warnings
            if hasattr(fig, '_renderer'):
                fig._renderer = None

        return figure_like

    elif isinstance(figure_like, tuple):
        fig, ax = figure_like
        if show:
            # Only show if using interactive backend and not in test environment
            backend = matplotlib.get_backend().lower()
            is_interactive = backend not in {'agg', 'pdf', 'ps', 'svg', 'template'}
            is_test_env = 'PYTEST_CURRENT_TEST' in os.environ

            if is_interactive and not is_test_env:
                plt.show()

        if save:
            fig.savefig(str(filename), dpi=dpi)
            plt.close(fig)  # Close figure to free memory

        return fig, ax

    raise TypeError(f'Figure type not supported: {type(figure_like)}')
