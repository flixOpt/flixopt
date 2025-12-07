"""Simplified color handling for visualization.

This module provides clean color processing that transforms various input formats
into a label-to-color mapping dictionary, without needing to know about the plotting engine.
"""

from __future__ import annotations

import logging

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.exceptions import PlotlyError

logger = logging.getLogger('flixopt')

# Type alias for flexible color input
ColorType = str | list[str] | dict[str, str] | None
ColorType = str | list[str] | dict[str, str]
"""Flexible color specification type supporting multiple input formats for visualization.

Color specifications can take several forms to accommodate different use cases:

**Named colorscales** (str):
    - Standard colorscales: 'turbo', 'plasma', 'cividis', 'tab10', 'Set1'
    - Energy-focused: 'portland' (custom flixopt colorscale for energy systems)
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
    # Named colorscale
    colors = 'turbo'  # Automatic color generation

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
    - Matplotlib colorscales: https://matplotlib.org/stable/tutorials/colors/colorscales.html
    - Plotly Built-in Colorscales: https://plotly.com/python/builtin-colorscales/
"""


def _rgb_string_to_hex(color: str) -> str:
    """Convert Plotly RGB/RGBA string format to hex.

    Args:
        color: Color in format 'rgb(R, G, B)', 'rgba(R, G, B, A)' or already in hex

    Returns:
        Color in hex format '#RRGGBB'
    """
    color = color.strip()

    # If already hex, return as-is
    if color.startswith('#'):
        return color

    # Try to parse rgb() or rgba()
    try:
        if color.startswith('rgb('):
            # Extract RGB values from 'rgb(R, G, B)' format
            rgb_str = color[4:-1]  # Remove 'rgb(' and ')'
        elif color.startswith('rgba('):
            # Extract RGBA values from 'rgba(R, G, B, A)' format
            rgb_str = color[5:-1]  # Remove 'rgba(' and ')'
        else:
            return color

        # Split on commas and parse first three components
        components = rgb_str.split(',')
        if len(components) < 3:
            return color

        # Parse and clamp the first three components
        r = max(0, min(255, int(round(float(components[0].strip())))))
        g = max(0, min(255, int(round(float(components[1].strip())))))
        b = max(0, min(255, int(round(float(components[2].strip())))))

        return f'#{r:02x}{g:02x}{b:02x}'
    except (ValueError, IndexError):
        # If parsing fails, return original
        return color


def process_colors(
    colors: None | str | list[str] | dict[str, str],
    labels: list[str],
    default_colorscale: str = 'turbo',
) -> dict[str, str]:
    """Process color input and return a label-to-color mapping.

    This function takes flexible color input and always returns a dictionary
    mapping each label to a specific color string. The plotting engine can then
    use this mapping as needed.

    Args:
        colors: Color specification in one of four formats:
            - None: Use the default colorscale
            - str: Name of a colorscale (e.g., 'turbo', 'plasma', 'Set1', 'portland')
            - list[str]: List of color strings (hex, named colors, etc.)
            - dict[str, str]: Direct label-to-color mapping
        labels: List of labels that need colors assigned
        default_colorscale: Fallback colorscale name if requested scale not found

    Returns:
        Dictionary mapping each label to a color string

    Examples:
        >>> # Using None - applies default colorscale
        >>> process_colors(None, ['A', 'B', 'C'])
        {'A': '#0d0887', 'B': '#7e03a8', 'C': '#cc4778'}

        >>> # Using a colorscale name
        >>> process_colors('plasma', ['A', 'B', 'C'])
        {'A': '#0d0887', 'B': '#7e03a8', 'C': '#cc4778'}

        >>> # Using a list of colors
        >>> process_colors(['red', 'blue', 'green'], ['A', 'B', 'C'])
        {'A': 'red', 'B': 'blue', 'C': 'green'}

        >>> # Using a pre-made mapping
        >>> process_colors({'A': 'red', 'B': 'blue'}, ['A', 'B', 'C'])
        {'A': 'red', 'B': 'blue', 'C': '#0d0887'}  # C gets color from default scale
    """
    if not labels:
        return {}

    # Case 1: Already a mapping dictionary
    if isinstance(colors, dict):
        return _fill_missing_colors(colors, labels, default_colorscale)

    # Case 2: None or colorscale name (string)
    if colors is None or isinstance(colors, str):
        colorscale_name = colors if colors is not None else default_colorscale
        color_list = _get_colors_from_scale(colorscale_name, len(labels), default_colorscale)
        return dict(zip(labels, color_list, strict=False))

    # Case 3: List of colors
    if isinstance(colors, list):
        if len(colors) == 0:
            logger.warning(f'Empty color list provided. Using {default_colorscale} instead.')
            color_list = _get_colors_from_scale(default_colorscale, len(labels), default_colorscale)
            return dict(zip(labels, color_list, strict=False))

        if len(colors) < len(labels):
            logger.debug(
                f'Not enough colors provided ({len(colors)}) for all labels ({len(labels)}). Colors will cycle.'
            )

        # Cycle through colors if we don't have enough
        return {label: colors[i % len(colors)] for i, label in enumerate(labels)}

    raise TypeError(f'colors must be None, str, list, or dict, got {type(colors)}')


def _fill_missing_colors(
    color_mapping: dict[str, str],
    labels: list[str],
    default_colorscale: str,
) -> dict[str, str]:
    """Fill in missing labels in a color mapping using a colorscale.

    Args:
        color_mapping: Partial label-to-color mapping
        labels: All labels that need colors
        default_colorscale: Colorscale to use for missing labels

    Returns:
        Complete label-to-color mapping
    """
    missing_labels = [label for label in labels if label not in color_mapping]

    if not missing_labels:
        return color_mapping.copy()

    # Log warning about missing labels
    logger.debug(f'Labels missing colors: {missing_labels}. Using {default_colorscale} for these.')

    # Get colors for missing labels
    missing_colors = _get_colors_from_scale(default_colorscale, len(missing_labels), default_colorscale)

    # Combine existing and new colors
    result = color_mapping.copy()
    result.update(dict(zip(missing_labels, missing_colors, strict=False)))
    return result


def _get_colors_from_scale(
    colorscale_name: str,
    num_colors: int,
    fallback_scale: str,
) -> list[str]:
    """Extract a list of colors from a named colorscale.

    Tries to get colors from the named scale (Plotly first, then Matplotlib),
    falls back to the fallback scale if not found.

    Args:
        colorscale_name: Name of the colorscale to try
        num_colors: Number of colors needed
        fallback_scale: Fallback colorscale name if first fails

    Returns:
        List of color strings (hex format)
    """
    # Try to get the requested colorscale
    colors = _try_get_colorscale(colorscale_name, num_colors)

    if colors is not None:
        return colors

    # Fallback to default
    logger.warning(f"Colorscale '{colorscale_name}' not found. Using '{fallback_scale}' instead.")

    colors = _try_get_colorscale(fallback_scale, num_colors)

    if colors is not None:
        return colors

    # Ultimate fallback: just use basic colors
    logger.warning(f"Fallback colorscale '{fallback_scale}' also not found. Using basic colors.")
    basic_colors = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf',
    ]
    return [basic_colors[i % len(basic_colors)] for i in range(num_colors)]


def _try_get_colorscale(colorscale_name: str, num_colors: int) -> list[str] | None:
    """Try to get colors from Plotly or Matplotlib colorscales.

    Tries Plotly colorscales first (both qualitative and sequential),
    then falls back to Matplotlib colorscales.

    Args:
        colorscale_name: Name of the colorscale
        num_colors: Number of colors needed

    Returns:
        List of color strings (hex format) if successful, None if colorscale not found
    """
    # First try Plotly qualitative (discrete) color sequences
    colorscale_title = colorscale_name.title()
    if hasattr(px.colors.qualitative, colorscale_title):
        color_list = getattr(px.colors.qualitative, colorscale_title)
        # Convert to hex format for matplotlib compatibility
        return [_rgb_string_to_hex(color_list[i % len(color_list)]) for i in range(num_colors)]

    # Then try Plotly sequential/continuous colorscales
    try:
        colorscale = px.colors.get_colorscale(colorscale_name)
        # Sample evenly from the colorscale
        if num_colors == 1:
            sample_points = [0.5]
        else:
            sample_points = [i / (num_colors - 1) for i in range(num_colors)]
        colors = px.colors.sample_colorscale(colorscale, sample_points)
        # Convert to hex format for matplotlib compatibility
        return [_rgb_string_to_hex(c) for c in colors]
    except (PlotlyError, ValueError):
        pass

    # Finally try Matplotlib colorscales
    try:
        cmap = plt.get_cmap(colorscale_name)

        # Sample evenly from the colorscale
        if num_colors == 1:
            colors = [cmap(0.5)]
        else:
            colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

        # Convert RGBA tuples to hex strings
        return [mcolors.rgb2hex(color[:3]) for color in colors]

    except (ValueError, KeyError):
        return None
