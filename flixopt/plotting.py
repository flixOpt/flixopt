"""Comprehensive visualization toolkit for flixopt optimization results and data analysis.

This module provides a unified plotting interface supporting both Plotly (interactive)
and Matplotlib (static) backends for visualizing energy system optimization results.
It offers specialized plotting functions for time series, heatmaps, network diagrams,
and statistical analyses commonly needed in energy system modeling.

Key Features:
    **Dual Backend Support**: Seamless switching between Plotly and Matplotlib
    **Energy System Focus**: Specialized plots for power flows, storage states, emissions
    **Color Management**: Intelligent color processing with ColorProcessor and pattern-based
                         XarrayColorMapper for grouped coloring
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

    def __init__(self, engine: PlottingEngine = 'plotly', default_colormap: str = 'viridis'):
        """Initialize the color processor with specified backend and defaults."""
        if engine not in ['plotly', 'matplotlib']:
            raise TypeError(f'engine must be "plotly" or "matplotlib", but is {engine}')
        self.engine = engine
        self.default_colormap = default_colormap

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
            try:
                colorscale = px.colors.get_colorscale(colormap_name)
            except PlotlyError as e:
                logger.error(f"Colorscale '{colormap_name}' not found in Plotly. Using {self.default_colormap}: {e}")
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


# Type aliases for XarrayColorMapper
MatchType = Literal['prefix', 'suffix', 'contains', 'glob', 'regex']
CoordValues = xr.DataArray | np.ndarray | list[Any]


class XarrayColorMapper:
    """Map xarray coordinate values to colors based on naming patterns.

    A simple, maintainable utility class for mapping xarray coordinate values
    to colors based on naming patterns. Enables visual grouping in plots where
    similar items get similar colors.

    Key Features:
        - Pattern-based color assignment (prefix, suffix, contains, glob, regex)
        - ColorBrewer sequential palettes for grouped coloring (cycles through shades)
        - Discrete color support for exact color matching across all items
        - Override support for special cases
        - Coordinate reordering for visual grouping in plots
        - Full type hints and comprehensive documentation

    Example Usage:
        Using color families (items cycle through shades):

        ```python
        mapper = (
            XarrayColorMapper()
            .add_rule('Product_A', 'blues', 'prefix')  # Product_A1, A2, A3 get different blue shades
            .add_rule('Product_B', 'greens', 'prefix')  # Product_B1, B2, B3 get different green shades
            .add_override({'Special': '#FFD700'})
        )
        ```

        Using discrete colors (all matching items get the same color):

        ```python
        mapper = (
            XarrayColorMapper()
            .add_rule('Solar', '#FFA500', 'prefix')  # All Solar* items get exact orange
            .add_rule('Wind', 'skyblue', 'prefix')  # All Wind* items get exact skyblue
            .add_rule('Battery', 'rgb(50,205,50)', 'contains')  # All *Battery* get lime green
        )

        # Apply to data
        color_map = mapper.apply_to_dataarray(da, 'component')

        # Plot with Plotly
        fig = px.bar(df, x='component', y='value', color='component', color_discrete_map=color_map)
        ```
    """

    # Class-level defaults (easy to update in one place)
    DEFAULT_FAMILIES = {
        'blues': px.colors.sequential.Blues[2:7],
        'greens': px.colors.sequential.Greens[2:7],
        'reds': px.colors.sequential.Reds[2:7],
        'purples': px.colors.sequential.Purples[2:7],
        'oranges': px.colors.sequential.Oranges[2:7],
        'teals': px.colors.sequential.Teal[2:7],
        'greys': px.colors.sequential.Greys[2:7],
        'pinks': px.colors.sequential.Pinkyl[2:7],
    }

    def __init__(self, color_families: dict[str, list[str]] | None = None, sort_within_groups: bool = True) -> None:
        """
        Initialize with ColorBrewer families.

        Parameters:
        -----------
        color_families : dict, optional
            Custom color families. If None, uses DEFAULT_FAMILIES
        sort_within_groups : bool, default True
            Whether to sort values within groups by default
        """
        if color_families is None:
            self.color_families = self.DEFAULT_FAMILIES.copy()
        else:
            self.color_families = color_families.copy()

        self.sort_within_groups = sort_within_groups
        self.rules: list[dict[str, str]] = []
        self.overrides: dict[str, str] = {}

    def add_custom_family(self, name: str, colors: list[str]) -> XarrayColorMapper:
        """
        Add a custom color family.

        Parameters:
        -----------
        name : str
            Name for the color family
        colors : List[str]
            List of hex color codes

        Returns:
        --------
        XarrayColorMapper : Self for method chaining
        """
        self.color_families[name] = colors
        return self

    def add_rule(self, pattern: str, family_or_color: str, match_type: MatchType = 'prefix') -> XarrayColorMapper:
        """
        Add a pattern-based rule to assign color families or discrete colors.

        Parameters:
        -----------
        pattern : str
            Pattern to match against coordinate values
        family_or_color : str
            Either a color family name (e.g., 'blues', 'greens') or a discrete color.
            Discrete colors can be:
            - Hex colors: '#FF0000', '#00FF00'
            - RGB/RGBA strings: 'rgb(255,0,0)', 'rgba(255,0,0,0.5)'
            - Named colors: 'red', 'blue', 'skyblue'
        match_type : {'prefix', 'suffix', 'contains', 'glob', 'regex'}, default 'prefix'
            Type of pattern matching to use:
            - 'prefix': Match if value starts with pattern
            - 'suffix': Match if value ends with pattern
            - 'contains': Match if pattern appears anywhere in value
            - 'glob': Unix-style wildcards (* matches anything, ? matches one char)
            - 'regex': Match using regular expression

        Returns:
        --------
        XarrayColorMapper : Self for method chaining

        Examples:
        ---------
        Using color families (cycles through shades):

        mapper.add_rule('Product_A', 'blues', 'prefix')
        mapper.add_rule('_test', 'greens', 'suffix')

        Using discrete colors (all matches get the same color):

        mapper.add_rule('Solar', '#FFA500', 'prefix')  # All Solar* items get orange
        mapper.add_rule('Wind', 'skyblue', 'prefix')  # All Wind* items get skyblue
        mapper.add_rule('Battery', 'rgb(50,205,50)', 'contains')  # All *Battery* get lime green
        """
        valid_types = ('prefix', 'suffix', 'contains', 'glob', 'regex')
        if match_type not in valid_types:
            raise ValueError(f"match_type must be one of {valid_types}, got '{match_type}'")

        # Check if family_or_color is a discrete color or a family name
        if family_or_color in self.color_families:
            # It's a known family - store as family rule
            self.rules.append(
                {'pattern': pattern, 'family': family_or_color, 'match_type': match_type, 'is_discrete': False}
            )
        else:
            # Otherwise treat as discrete color - no error handling of invalid colors!
            self.rules.append(
                {'pattern': pattern, 'discrete_color': family_or_color, 'match_type': match_type, 'is_discrete': True}
            )

        return self

    def add_override(self, color_dict: dict[str, str]) -> XarrayColorMapper:
        """
        Override colors for specific values (takes precedence over rules).

        Parameters:
        -----------
        color_dict : Dict[str, str]
            Mapping of {value: hex_color}

        Returns:
        --------
        XarrayColorMapper : Self for method chaining

        Examples:
        ---------
        mapper.add_override({'Special': '#FFD700'})
        mapper.add_override({
            'Product_A1': '#FF00FF',
            'Product_B2': '#00FFFF'
        })
        """
        for val, col in color_dict.items():
            self.overrides[str(val)] = col
        return self

    def create_color_map(
        self,
        coord_values: CoordValues,
        sort_within_groups: bool | None = None,
        fallback_family: str = 'greys',
    ) -> dict[str, str]:
        """
        Create color mapping for coordinate values.

        Parameters:
        -----------
        coord_values : xr.DataArray, np.ndarray, or list
            Coordinate values to map
        sort_within_groups : bool, optional
            Sort values within each group. If None, uses instance default
        fallback_family : str, default 'greys'
            Color family for unmatched values

        Returns:
        --------
        Dict[str, str] : Mapping of {value: hex_color}
        """
        if sort_within_groups is None:
            sort_within_groups = self.sort_within_groups

        # Convert to string list
        if isinstance(coord_values, xr.DataArray):
            categories: list[str] = [str(val) for val in coord_values.values]
        elif isinstance(coord_values, np.ndarray):
            categories = [str(val) for val in coord_values]
        else:
            categories = [str(val) for val in coord_values]

        # Remove duplicates while preserving order
        seen: set = set()
        categories = [x for x in categories if not (x in seen or seen.add(x))]

        # Group by rules
        groups = self._group_categories(categories)
        color_map: dict[str, str] = {}

        # Assign colors to groups
        for group_key, group_categories in groups.items():
            if group_key == '_unmatched':
                # Unmatched items use fallback family
                family = self.color_families.get(fallback_family, self.color_families['greys'])
                if sort_within_groups:
                    group_categories = sorted(group_categories)
                for idx, category in enumerate(group_categories):
                    color_map[category] = family[idx % len(family)]

            elif group_key.startswith('_discrete_'):
                # Discrete color group - all items get the same color
                discrete_color = group_key.replace('_discrete_', '', 1)
                if sort_within_groups:
                    group_categories = sorted(group_categories)
                for category in group_categories:
                    color_map[category] = discrete_color

            else:
                # Family-based group - cycle through family colors
                family = self.color_families[group_key]
                if sort_within_groups:
                    group_categories = sorted(group_categories)
                for idx, category in enumerate(group_categories):
                    color_map[category] = family[idx % len(family)]

        # Apply overrides
        color_map.update(self.overrides)

        return color_map

    def apply_to_dataarray(self, da: xr.DataArray, coord_dim: str) -> dict[str, str]:
        """
        Create color map for a DataArray coordinate dimension.

        Parameters:
        -----------
        da : xr.DataArray
            The data array
        coord_dim : str
            Coordinate dimension name

        Returns:
        --------
        Dict[str, str] : Color mapping for that dimension

        Raises:
        -------
        ValueError : If coord_dim is not found in the DataArray
        """
        if coord_dim not in da.coords:
            raise ValueError(f"Coordinate '{coord_dim}' not found. Available: {list(da.coords.keys())}")

        return self.create_color_map(da.coords[coord_dim])

    def reorder_coordinate(
        self, da: xr.DataArray, coord_dim: str, sort_within_groups: bool | None = None
    ) -> xr.DataArray:
        """
        Reorder a DataArray coordinate so values with the same color are adjacent.

        This is useful for creating plots where similar items (same color group)
        appear next to each other, making visual groupings clear.

        Parameters:
        -----------
        da : xr.DataArray
            The data array to reorder
        coord_dim : str
            The coordinate dimension to reorder
        sort_within_groups : bool, optional
            Whether to sort values within each group. If None, uses instance default

        Returns:
        --------
        xr.DataArray : New DataArray with reordered coordinate

        Examples:
        ---------
        # Original order: ['Product_B1', 'Product_A1', 'Product_B2', 'Product_A2']
        # After reorder: ['Product_A1', 'Product_A2', 'Product_B1', 'Product_B2']

        mapper = XarrayColorMapper()
        mapper.add_rule('Product_A', 'blues', 'prefix')
        mapper.add_rule('Product_B', 'greens', 'prefix')

        da_reordered = mapper.reorder_coordinate(da, 'product')
        """
        if coord_dim not in da.coords:
            raise ValueError(f"Coordinate '{coord_dim}' not found. Available: {list(da.coords.keys())}")

        if sort_within_groups is None:
            sort_within_groups = self.sort_within_groups

        # Get coordinate values
        coord_values = da.coords[coord_dim].values
        categories = [str(val) for val in coord_values]

        # Group categories
        groups = self._group_categories(categories)

        # Build new order: group by group, optionally sorted within each
        new_order = []
        for group_name in groups.keys():
            group_categories = groups[group_name]
            if sort_within_groups:
                group_categories = sorted(group_categories)
            new_order.extend(group_categories)

        # Convert back to original dtype if needed
        original_values = list(coord_values)
        # Map string back to original values
        str_to_original = {str(v): v for v in original_values}
        reordered_values = [str_to_original[cat] for cat in new_order]

        # Reindex the DataArray
        return da.sel({coord_dim: reordered_values})

    def get_rules(self) -> list[dict[str, str]]:
        """Return a copy of current rules for inspection."""
        return self.rules.copy()

    def get_overrides(self) -> dict[str, str]:
        """Return a copy of current overrides for inspection."""
        return self.overrides.copy()

    def get_families(self) -> dict[str, list[str]]:
        """Return a copy of available color families."""
        return self.color_families.copy()

    def _match_rule(self, value: str, rule: dict[str, str]) -> bool:
        """
        Check if value matches a rule.

        Parameters:
        -----------
        value : str
            Value to check
        rule : dict
            Rule dictionary with 'pattern' and 'match_type' keys

        Returns:
        --------
        bool : True if value matches the rule
        """
        pattern = rule['pattern']
        match_type = rule['match_type']

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
                return bool(re.match(pattern, value))
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

        return False

    def _group_categories(self, categories: list[str]) -> dict[str, list[str]]:
        """
        Group categories by matching rules.

        Parameters:
        -----------
        categories : List[str]
            List of category values to group

        Returns:
        --------
        Dict[str, List[str]] : Mapping of {group_key: [matching_values]}

        Note: For discrete color rules, group_key is '_discrete_<color>',
              for family rules, group_key is the family name.
        """
        groups: dict[str, list[str]] = {}
        unmatched: list[str] = []

        for category in categories:
            matched = False
            for rule in self.rules:
                if self._match_rule(category, rule):
                    if rule.get('is_discrete', False):
                        # For discrete colors, use a special group key
                        group_key = f'_discrete_{rule["discrete_color"]}'
                    else:
                        # For families, use the family name
                        group_key = rule['family']

                    if group_key not in groups:
                        groups[group_key] = []
                    groups[group_key].append(category)
                    matched = True
                    break  # First match wins

            if not matched:
                unmatched.append(category)

        if unmatched:
            groups['_unmatched'] = unmatched

        return groups


def resolve_colors(
    data: xr.DataArray | xr.Dataset,
    colors: ColorType | XarrayColorMapper,
    coord_dim: str = 'variable',
    engine: PlottingEngine = 'plotly',
) -> dict[str, str]:
    """Resolve colors parameter to a color mapping dict.

    This public utility function handles all color parameter types and applies the
    color mapper intelligently based on the data structure. Can be used standalone
    or as part of CalculationResults.

    Args:
        data: DataArray or Dataset to create colors for
        colors: Color specification or a XarrayColorMapper to use
        coord_dim: Coordinate dimension to map colors to
        engine: Plotting engine ('plotly' or 'matplotlib')

    Returns:
        Dictionary mapping coordinate values to colors

    Examples:
        With CalculationResults:

        >>> resolved_colors = resolve_colors(data, results.color_mapper)

        Standalone usage:

        >>> mapper = plotting.XarrayColorMapper()
        >>> mapper.add_rule('Solar', 'oranges', 'prefix')
        >>> resolved_colors = resolve_colors(data, mapper)

        Without mapper:

        >>> resolved_colors = resolve_colors(data, 'viridis')
    """
    # If explicit dict provided, use it directly
    if isinstance(colors, dict):
        return colors

    # If string or list, use ColorProcessor (traditional behavior)
    if isinstance(colors, (str, list)):
        if isinstance(data, xr.DataArray):
            if coord_dim in data.coords:
                labels = [str(v) for v in data.coords[coord_dim].values]
            else:
                labels = []
        elif isinstance(data, xr.Dataset):
            labels = [str(v) for v in data.data_vars]
        else:
            labels = []

        if labels:
            processor = ColorProcessor(engine=engine)
            return processor.process_colors(colors, labels, return_mapping=True)

    if isinstance(colors, XarrayColorMapper):
        color_mapper = colors
        if color_mapper.sort_within_groups:
            # Check if coord_dim exists and reorder
            if isinstance(data, xr.DataArray) and coord_dim in data.coords:
                data = color_mapper.reorder_coordinate(data, coord_dim)
            elif isinstance(data, xr.Dataset):
                # For Dataset, we'll work with the variables directly
                pass

        # Apply color mapper to get dict
        if isinstance(data, xr.DataArray):
            if coord_dim in data.coords:
                return color_mapper.apply_to_dataarray(data, coord_dim)
        elif isinstance(data, xr.Dataset):
            # For Dataset, map colors to variable names
            labels = [str(v) for v in data.data_vars]
            return color_mapper.create_color_map(labels)

    raise TypeError(f'Wrong type passed to resolve_colors(): {type(colors)}')


def with_plotly(
    data: pd.DataFrame | xr.DataArray | xr.Dataset,
    mode: Literal['stacked_bar', 'line', 'area', 'grouped_bar'] = 'stacked_bar',
    colors: ColorType = 'viridis',
    title: str = '',
    ylabel: str = '',
    xlabel: str = 'Time in h',
    fig: go.Figure | None = None,
    facet_by: str | list[str] | None = None,
    animate_by: str | None = None,
    facet_cols: int = 3,
    shared_yaxes: bool = True,
    shared_xaxes: bool = True,
) -> go.Figure:
    """
    Plot data with Plotly using facets (subplots) and/or animation for multidimensional data.

    Uses Plotly Express for convenient faceting and animation with automatic styling.
    For simple plots without faceting, can optionally add to an existing figure.

    Args:
        data: A DataFrame or xarray DataArray/Dataset to plot.
        mode: The plotting mode. Use 'stacked_bar' for stacked bar charts, 'line' for lines,
              'area' for stacked area charts, or 'grouped_bar' for grouped bar charts.
        colors: Color specification (colormap, list, or dict mapping labels to colors).
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

    Returns:
        A Plotly figure object containing the faceted/animated plot.

    Examples:
        Simple plot:

        ```python
        fig = with_plotly(df, mode='area', title='Energy Mix')
        ```

        Facet by scenario:

        ```python
        fig = with_plotly(ds, facet_by='scenario', facet_cols=2)
        ```

        Animate by period:

        ```python
        fig = with_plotly(ds, animate_by='period')
        ```

        Facet and animate:

        ```python
        fig = with_plotly(ds, facet_by='scenario', animate_by='period')
        ```
    """
    if mode not in ('stacked_bar', 'line', 'area', 'grouped_bar'):
        raise ValueError(f"'mode' must be one of {{'stacked_bar','line','area', 'grouped_bar'}}, got {mode!r}")

    # Handle empty data
    if isinstance(data, pd.DataFrame) and data.empty:
        return go.Figure()
    elif isinstance(data, xr.DataArray) and data.size == 0:
        return go.Figure()
    elif isinstance(data, xr.Dataset) and len(data.data_vars) == 0:
        return go.Figure()

    # Warn if fig parameter is used with faceting
    if fig is not None and (facet_by is not None or animate_by is not None):
        logger.warning('The fig parameter is ignored when using faceting or animation. Creating a new figure.')
        fig = None

    # Convert xarray to long-form DataFrame for Plotly Express
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        # Convert to long-form (tidy) DataFrame
        # Structure: time, variable, value, scenario, period, ... (all dims as columns)
        if isinstance(data, xr.Dataset):
            # Stack all data variables into long format
            df_long = data.to_dataframe().reset_index()
            # Melt to get: time, scenario, period, ..., variable, value
            id_vars = [dim for dim in data.dims]
            value_vars = list(data.data_vars)
            df_long = df_long.melt(id_vars=id_vars, value_vars=value_vars, var_name='variable', value_name='value')
        else:
            # DataArray
            df_long = data.to_dataframe().reset_index()
            if data.name:
                df_long = df_long.rename(columns={data.name: 'value'})
            else:
                # Unnamed DataArray, find the value column
                non_dim_cols = [col for col in df_long.columns if col not in data.dims]
                if len(non_dim_cols) != 1:
                    raise ValueError(
                        f'Expected exactly one non-dimension column for unnamed DataArray, '
                        f'but found {len(non_dim_cols)}: {non_dim_cols}'
                    )
                value_col = non_dim_cols[0]
                df_long = df_long.rename(columns={value_col: 'value'})
            df_long['variable'] = data.name or 'data'
    else:
        # Already a DataFrame - convert to long format for Plotly Express
        df_long = data.reset_index()
        if 'time' not in df_long.columns:
            # First column is probably time
            df_long = df_long.rename(columns={df_long.columns[0]: 'time'})
        # Melt to long format
        id_vars = [
            col
            for col in df_long.columns
            if col in ['time', 'scenario', 'period']
            or col in (facet_by if isinstance(facet_by, list) else [facet_by] if facet_by else [])
        ]
        value_vars = [col for col in df_long.columns if col not in id_vars]
        df_long = df_long.melt(id_vars=id_vars, value_vars=value_vars, var_name='variable', value_name='value')

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

    # Process colors
    all_vars = df_long['variable'].unique().tolist()
    processed_colors = ColorProcessor(engine='plotly').process_colors(colors, all_vars)
    color_discrete_map = {var: color for var, color in zip(all_vars, processed_colors, strict=True)}

    # Create plot using Plotly Express based on mode
    common_args = {
        'data_frame': df_long,
        'x': 'time',
        'y': 'value',
        'color': 'variable',
        'facet_row': facet_row,
        'facet_col': facet_col,
        'animation_frame': animate_by,
        'color_discrete_map': color_discrete_map,
        'title': title,
        'labels': {'value': ylabel, 'time': xlabel, 'variable': ''},
    }

    # Add facet_col_wrap for single facet dimension
    if facet_col and not facet_row:
        common_args['facet_col_wrap'] = facet_cols

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

    return fig


def with_matplotlib(
    data: pd.DataFrame,
    mode: Literal['stacked_bar', 'line'] = 'stacked_bar',
    colors: ColorType = 'viridis',
    title: str = '',
    ylabel: str = '',
    xlabel: str = 'Time in h',
    figsize: tuple[int, int] = (12, 6),
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a DataFrame with Matplotlib using stacked bars or stepped lines.

    Args:
        data: A DataFrame containing the data to plot. The index should represent time (e.g., hours),
              and each column represents a separate data series.
        mode: Plotting mode. Use 'stacked_bar' for stacked bar charts or 'line' for stepped lines.
        colors: Color specification, can be:
            - A string with a colormap name (e.g., 'viridis', 'plasma')
            - A list of color strings (e.g., ['#ff0000', '#00ff00'])
            - A dictionary mapping column names to colors (e.g., {'Column1': '#ff0000'})
        title: The title of the plot.
        ylabel: The ylabel of the plot.
        xlabel: The xlabel of the plot.
        figsize: Specify the size of the figure
        fig: A Matplotlib figure object to plot on. If not provided, a new figure will be created.
        ax: A Matplotlib axes object to plot on. If not provided, a new axes will be created.

    Returns:
        A tuple containing the Matplotlib figure and axes objects used for the plot.

    Notes:
        - If `mode` is 'stacked_bar', bars are stacked for both positive and negative values.
          Negative values are stacked separately without extra labels in the legend.
        - If `mode` is 'line', stepped lines are drawn for each data series.
    """
    if mode not in ('stacked_bar', 'line'):
        raise ValueError(f"'mode' must be one of {{'stacked_bar','line'}} for matplotlib, got {mode!r}")

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    processed_colors = ColorProcessor(engine='matplotlib').process_colors(colors, list(data.columns))

    if mode == 'stacked_bar':
        cumulative_positive = np.zeros(len(data))
        cumulative_negative = np.zeros(len(data))
        width = data.index.to_series().diff().dropna().min()  # Minimum time difference

        for i, column in enumerate(data.columns):
            positive_values = np.clip(data[column], 0, None)  # Keep only positive values
            negative_values = np.clip(data[column], None, 0)  # Keep only negative values
            # Plot positive bars
            ax.bar(
                data.index,
                positive_values,
                bottom=cumulative_positive,
                color=processed_colors[i],
                label=column,
                width=width,
                align='center',
            )
            cumulative_positive += positive_values.values
            # Plot negative bars
            ax.bar(
                data.index,
                negative_values,
                bottom=cumulative_negative,
                color=processed_colors[i],
                label='',  # No label for negative bars
                width=width,
                align='center',
            )
            cumulative_negative += negative_values.values

    elif mode == 'line':
        for i, column in enumerate(data.columns):
            ax.step(data.index, data[column], where='post', color=processed_colors[i], label=column)

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
    data: pd.DataFrame,
    colors: ColorType = 'viridis',
    title: str = '',
    legend_title: str = '',
    hole: float = 0.0,
    fig: go.Figure | None = None,
) -> go.Figure:
    """
    Create a pie chart with Plotly to visualize the proportion of values in a DataFrame.

    Args:
        data: A DataFrame containing the data to plot. If multiple rows exist,
              they will be summed unless a specific index value is passed.
        colors: Color specification, can be:
            - A string with a colorscale name (e.g., 'viridis', 'plasma')
            - A list of color strings (e.g., ['#ff0000', '#00ff00'])
            - A dictionary mapping column names to colors (e.g., {'Column1': '#ff0000'})
        title: The title of the plot.
        legend_title: The title for the legend.
        hole: Size of the hole in the center for creating a donut chart (0.0 to 1.0).
        fig: A Plotly figure object to plot on. If not provided, a new figure will be created.

    Returns:
        A Plotly figure object containing the generated pie chart.

    Notes:
        - Negative values are not appropriate for pie charts and will be converted to absolute values with a warning.
        - If the data contains very small values (less than 1% of the total), they can be grouped into an "Other" category
          for better readability.
        - By default, the sum of all columns is used for the pie chart. For time series data, consider preprocessing.

    """
    if data.empty:
        logger.error('Empty DataFrame provided for pie chart. Returning empty figure.')
        return go.Figure()

    # Create a copy to avoid modifying the original DataFrame
    data_copy = data.copy()

    # Check if any negative values and warn
    if (data_copy < 0).any().any():
        logger.error('Negative values detected in data. Using absolute values for pie chart.')
        data_copy = data_copy.abs()

    # If data has multiple rows, sum them to get total for each column
    if len(data_copy) > 1:
        data_sum = data_copy.sum()
    else:
        data_sum = data_copy.iloc[0]

    # Get labels (column names) and values
    labels = data_sum.index.tolist()
    values = data_sum.values.tolist()

    # Apply color mapping using the unified color processor
    processed_colors = ColorProcessor(engine='plotly').process_colors(colors, labels)

    # Create figure if not provided
    fig = fig if fig is not None else go.Figure()

    # Add pie trace
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=hole,
            marker=dict(colors=processed_colors),
            textinfo='percent+label+value',
            textposition='inside',
            insidetextorientation='radial',
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
    data: pd.DataFrame,
    colors: ColorType = 'viridis',
    title: str = '',
    legend_title: str = 'Categories',
    hole: float = 0.0,
    figsize: tuple[int, int] = (10, 8),
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a pie chart with Matplotlib to visualize the proportion of values in a DataFrame.

    Args:
        data: A DataFrame containing the data to plot. If multiple rows exist,
              they will be summed unless a specific index value is passed.
        colors: Color specification, can be:
            - A string with a colormap name (e.g., 'viridis', 'plasma')
            - A list of color strings (e.g., ['#ff0000', '#00ff00'])
            - A dictionary mapping column names to colors (e.g., {'Column1': '#ff0000'})
        title: The title of the plot.
        legend_title: The title for the legend.
        hole: Size of the hole in the center for creating a donut chart (0.0 to 1.0).
        figsize: The size of the figure (width, height) in inches.
        fig: A Matplotlib figure object to plot on. If not provided, a new figure will be created.
        ax: A Matplotlib axes object to plot on. If not provided, a new axes will be created.

    Returns:
        A tuple containing the Matplotlib figure and axes objects used for the plot.

    Notes:
        - Negative values are not appropriate for pie charts and will be converted to absolute values with a warning.
        - If the data contains very small values (less than 1% of the total), they can be grouped into an "Other" category
          for better readability.
        - By default, the sum of all columns is used for the pie chart. For time series data, consider preprocessing.

    """
    if data.empty:
        logger.error('Empty DataFrame provided for pie chart. Returning empty figure.')
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    # Create a copy to avoid modifying the original DataFrame
    data_copy = data.copy()

    # Check if any negative values and warn
    if (data_copy < 0).any().any():
        logger.error('Negative values detected in data. Using absolute values for pie chart.')
        data_copy = data_copy.abs()

    # If data has multiple rows, sum them to get total for each column
    if len(data_copy) > 1:
        data_sum = data_copy.sum()
    else:
        data_sum = data_copy.iloc[0]

    # Get labels (column names) and values
    labels = data_sum.index.tolist()
    values = data_sum.values.tolist()

    # Apply color mapping using the unified color processor
    processed_colors = ColorProcessor(engine='matplotlib').process_colors(colors, labels)

    # Create figure and axis if not provided
    if fig is None or ax is None:
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
    data_left: pd.Series,
    data_right: pd.Series,
    colors: ColorType = 'viridis',
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
        data_left: Series for the left pie chart.
        data_right: Series for the right pie chart.
        colors: Color specification, can be:
            - A string with a colorscale name (e.g., 'viridis', 'plasma')
            - A list of color strings (e.g., ['#ff0000', '#00ff00'])
            - A dictionary mapping category names to colors (e.g., {'Category1': '#ff0000'})
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
    from plotly.subplots import make_subplots

    # Check for empty data
    if data_left.empty and data_right.empty:
        logger.error('Both datasets are empty. Returning empty figure.')
        return go.Figure()

    # Create a subplot figure
    fig = make_subplots(
        rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'pie'}]], subplot_titles=subtitles, horizontal_spacing=0.05
    )

    # Process series to handle negative values and apply minimum percentage threshold
    def preprocess_series(series: pd.Series):
        """
        Preprocess a series for pie chart display by handling negative values
        and grouping the smallest parts together if they collectively represent
        less than the specified percentage threshold.

        Args:
            series: The series to preprocess

        Returns:
            A preprocessed pandas Series
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

    data_left_processed = preprocess_series(data_left)
    data_right_processed = preprocess_series(data_right)

    # Get unique set of all labels for consistent coloring
    all_labels = sorted(set(data_left_processed.index) | set(data_right_processed.index))

    # Get consistent color mapping for both charts using our unified function
    color_map = ColorProcessor(engine='plotly').process_colors(colors, all_labels, return_mapping=True)

    # Function to create a pie trace with consistently mapped colors
    def create_pie_trace(data_series, side):
        if data_series.empty:
            return None

        labels = data_series.index.tolist()
        values = data_series.values.tolist()
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
    left_trace = create_pie_trace(data_left_processed, subtitles[0])
    if left_trace:
        left_trace.domain = dict(x=[0, 0.48])
        fig.add_trace(left_trace, row=1, col=1)

    # Add right pie if data exists
    right_trace = create_pie_trace(data_right_processed, subtitles[1])
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
    colors: ColorType = 'viridis',
    title: str = '',
    subtitles: tuple[str, str] = ('Left Chart', 'Right Chart'),
    legend_title: str = '',
    hole: float = 0.2,
    lower_percentage_group: float = 5.0,
    figsize: tuple[int, int] = (14, 7),
    fig: plt.Figure | None = None,
    axes: list[plt.Axes] | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Create two pie charts side by side with Matplotlib, with consistent coloring across both charts.
    Leverages the existing pie_with_matplotlib function.

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
        fig: A Matplotlib figure object to plot on. If not provided, a new figure will be created.
        axes: A list of Matplotlib axes objects to plot on. If not provided, new axes will be created.

    Returns:
        A tuple containing the Matplotlib figure and list of axes objects used for the plot.
    """
    # Check for empty data
    if data_left.empty and data_right.empty:
        logger.error('Both datasets are empty. Returning empty figure.')
        if fig is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
        return fig, axes

    # Create figure and axes if not provided
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

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

    # Create left pie chart
    if not df_left.empty:
        pie_with_matplotlib(data=df_left, colors=left_colors, title=subtitles[0], hole=hole, fig=fig, ax=axes[0])
    else:
        axes[0].set_title(subtitles[0])
        axes[0].axis('off')

    # Create right pie chart
    if not df_right.empty:
        pie_with_matplotlib(data=df_right, colors=right_colors, title=subtitles[1], hole=hole, fig=fig, ax=axes[1])
    else:
        axes[1].set_title(subtitles[1])
        axes[1].axis('off')

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
    colors: ColorType = 'viridis',
    title: str = '',
    facet_by: str | list[str] | None = None,
    animate_by: str | None = None,
    facet_cols: int = 3,
    reshape_time: tuple[Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'], Literal['W', 'D', 'h', '15min', 'min']]
    | Literal['auto']
    | None = 'auto',
    fill: Literal['ffill', 'bfill'] | None = 'ffill',
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
        # Need at least 2 dimensions for a heatmap
        logger.error(
            f'Heatmap requires at least 2 dimensions for rows and columns. '
            f'After faceting/animation, only {len(heatmap_dims)} dimension(s) remain: {heatmap_dims}'
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
        'color_continuous_scale': colors if isinstance(colors, str) else 'viridis',
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

    try:
        fig = px.imshow(**common_args)
    except Exception as e:
        logger.error(f'Error creating imshow plot: {e}. Falling back to basic heatmap.')
        # Fallback: create a simple heatmap without faceting
        fig = px.imshow(
            data.values,
            color_continuous_scale=colors if isinstance(colors, str) else 'viridis',
            title=title,
        )

    # Update layout with basic styling
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
    )

    return fig


def heatmap_with_matplotlib(
    data: xr.DataArray,
    colors: ColorType = 'viridis',
    title: str = '',
    figsize: tuple[float, float] = (12, 6),
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    reshape_time: tuple[Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'], Literal['W', 'D', 'h', '15min', 'min']]
    | Literal['auto']
    | None = 'auto',
    fill: Literal['ffill', 'bfill'] | None = 'ffill',
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
        fig: A Matplotlib figure object to plot on. If not provided, a new figure will be created.
        ax: A Matplotlib axes object to plot on. If not provided, a new axes will be created.
        reshape_time: Time reshaping configuration:
                     - 'auto' (default): Automatically applies ('D', 'h') if only 'time' dimension
                     - Tuple like ('D', 'h'): Explicit time reshaping (days vs hours)
                     - None: Disable time reshaping
        fill: Method to fill missing values when reshaping time: 'ffill' or 'bfill'. Default is 'ffill'.

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
    # Handle empty data
    if data.size == 0:
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    # Apply time reshaping using the new unified function
    # Matplotlib doesn't support faceting/animation, so we pass None for those
    data = reshape_data_for_heatmap(data, reshape_time=reshape_time, facet_by=None, animate_by=None, fill=fill)

    # Create figure and axes if not provided
    if fig is None or ax is None:
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
    cmap = colors if isinstance(colors, str) else 'viridis'

    # Create the heatmap using imshow
    im = ax.imshow(values, cmap=cmap, aspect='auto', origin='upper')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, aspect=15, fraction=0.05)
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
    show: bool = True,
    save: bool = False,
) -> go.Figure | tuple[plt.Figure, plt.Axes]:
    """
    Export a figure to a file and or show it.

    Args:
        figure_like: The figure to export. Can be a Plotly figure or a tuple of Matplotlib figure and axes.
        default_path: The default file path if no user filename is provided.
        default_filetype: The default filetype if the path doesnt end with a filetype.
        user_path: An optional user-specified file path.
        show: Whether to display the figure (default: True).
        save: Whether to save the figure (default: False).

    Raises:
        ValueError: If no default filetype is provided and the path doesn't specify a filetype.
        TypeError: If the figure type is not supported.
    """
    filename = user_path or default_path
    filename = filename.with_name(filename.name.replace('|', '__'))
    if filename.suffix == '':
        if default_filetype is None:
            raise ValueError('No default filetype provided')
        filename = filename.with_suffix(default_filetype)

    if isinstance(figure_like, plotly.graph_objs.Figure):
        fig = figure_like
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
            fig.savefig(str(filename), dpi=300)
            plt.close(fig)  # Close figure to free memory

        return fig, ax

    raise TypeError(f'Figure type not supported: {type(figure_like)}')
