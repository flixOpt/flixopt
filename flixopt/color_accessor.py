"""Color accessor for centralized color management in FlowSystem.

This module provides the ColorAccessor class that enables consistent color
assignment across all visualization methods with context-aware logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .color_processing import process_colors
from .config import CONFIG

if TYPE_CHECKING:
    from .flow_system import FlowSystem


class ColorAccessor:
    """Centralized color management for FlowSystem. Access via ``flow_system.colors``.

    ColorAccessor provides a unified interface for managing colors across all
    visualization methods. It supports context-aware color resolution:
    - When plotting a bus balance: colors are based on components
    - When plotting a component balance: colors are based on bus carriers
    - Sankey diagrams: colors are based on bus carriers

    Color Resolution Priority:
        1. Explicit colors passed to plot methods (always override)
        2. Component/bus-specific colors set via setup()
        3. Element meta_data['color'] if present
        4. Carrier colors from flow_system.colors or CONFIG.Carriers
        5. Default colorscale

    Examples:
        Basic setup:

        ```python
        # Configure colors for components
        flow_system.colors.setup(
            {
                'Boiler': '#D35400',
                'CHP': '#8E44AD',
                'HeatPump': '#27AE60',
            }
        )

        # Override carrier colors for this system
        flow_system.colors.set_carrier_color('electricity', '#FFC300')

        # Plots automatically use configured colors
        flow_system.statistics.plot.balance('Electricity')  # Colors by component
        flow_system.statistics.plot.balance('CHP')  # Colors by carrier
        flow_system.statistics.plot.sankey()  # Buses use carrier colors
        ```

        Loading from file:

        ```python
        flow_system.colors.setup('colors.json')
        # or
        flow_system.colors.setup(Path('colors.yaml'))
        ```
    """

    def __init__(self, flow_system: FlowSystem) -> None:
        self._fs = flow_system
        self._component_colors: dict[str, str] = {}
        self._bus_colors: dict[str, str] = {}
        self._carrier_colors: dict[str, str] = {}

    def setup(self, config: dict[str, str] | str | Path) -> ColorAccessor:
        """Configure colors from a dictionary or file.

        The config dictionary maps element labels to colors. Elements can be
        components, buses, or carriers. The type is inferred from the label.

        Args:
            config: Either a dictionary mapping labels to colors, or a path
                to a JSON/YAML file containing such a mapping.

        Returns:
            Self for method chaining.

        Examples:
            ```python
            # From dictionary
            flow_system.colors.setup(
                {
                    'Boiler': '#D35400',  # Component
                    'HeatPump': '#27AE60',  # Component
                    'electricity': '#FFD700',  # Carrier (lowercase = carrier)
                    'heat': '#FF6B6B',  # Carrier
                }
            )

            # From file
            flow_system.colors.setup('my_colors.json')
            ```
        """
        if isinstance(config, (str, Path)):
            from . import io as fx_io

            config = fx_io.load_yaml(Path(config))

        for label, color in config.items():
            # Check if it's a known carrier (has attribute on CONFIG.Carriers or lowercase)
            if hasattr(CONFIG.Carriers, label) or label.islower():
                self._carrier_colors[label] = color
            # Check if it's a component
            elif label in self._fs.components:
                self._component_colors[label] = color
            # Check if it's a bus
            elif label in self._fs.buses:
                self._bus_colors[label] = color
            # Otherwise treat as component (most common case)
            else:
                self._component_colors[label] = color

        return self

    def set_component_color(self, label: str, color: str) -> ColorAccessor:
        """Set color for a specific component.

        Args:
            label: Component label.
            color: Color string (hex, named color, etc.).

        Returns:
            Self for method chaining.
        """
        self._component_colors[label] = color
        return self

    def set_bus_color(self, label: str, color: str) -> ColorAccessor:
        """Set color for a specific bus.

        Args:
            label: Bus label.
            color: Color string (hex, named color, etc.).

        Returns:
            Self for method chaining.
        """
        self._bus_colors[label] = color
        return self

    def set_carrier_color(self, carrier: str, color: str) -> ColorAccessor:
        """Set color for a carrier, overriding CONFIG.Carriers default.

        Args:
            carrier: Carrier name (e.g., 'electricity', 'heat').
            color: Color string (hex, named color, etc.).

        Returns:
            Self for method chaining.
        """
        self._carrier_colors[carrier] = color
        return self

    def for_component(self, label: str) -> str | None:
        """Get color for a component.

        Resolution order:
        1. Explicit component color from setup()
        2. Component's color attribute (auto-assigned or user-specified)
        3. Component's meta_data['color'] if present (legacy support)
        4. None (let caller use default colorscale)

        Args:
            label: Component label.

        Returns:
            Color string or None if not configured.
        """
        # Check explicit color from setup()
        if label in self._component_colors:
            return self._component_colors[label]

        # Check component's color attribute
        if label in self._fs.components:
            component = self._fs.components[label]
            if component.color:
                return component.color

            # Check meta_data (legacy support)
            if component.meta_data and 'color' in component.meta_data:
                return component.meta_data['color']

        return None

    def for_bus(self, label: str) -> str | None:
        """Get color for a bus.

        Buses get their color from their carrier. This provides consistent
        coloring where all heat buses are red, electricity buses are yellow, etc.

        Resolution order:
        1. Explicit bus color from setup()
        2. Carrier color (if bus has carrier set)
        3. None (let caller use default colorscale)

        Args:
            label: Bus label.

        Returns:
            Color string or None if not configured.
        """
        # Check explicit bus color from setup()
        if label in self._bus_colors:
            return self._bus_colors[label]

        # Check carrier color
        if label in self._fs.buses:
            bus = self._fs.buses[label]
            if bus.carrier:
                return self.for_carrier(bus.carrier)

        return None

    def for_carrier(self, carrier: str) -> str | None:
        """Get color for a carrier.

        Resolution order:
        1. Explicit carrier color override from setup()
        2. FlowSystem-registered carrier (via add_carrier())
        3. CONFIG.Carriers default
        4. None if carrier not found

        Args:
            carrier: Carrier name.

        Returns:
            Color string or None if not configured.
        """
        carrier_lower = carrier.lower()

        # Check explicit color override
        if carrier_lower in self._carrier_colors:
            return self._carrier_colors[carrier_lower]

        # Check FlowSystem-registered carriers
        carrier_obj = self._fs.get_carrier(carrier_lower)
        if carrier_obj:
            return carrier_obj.color

        return None

    def for_flow(self, label: str, context: Literal['bus', 'component']) -> str | None:
        """Get color for a flow based on plotting context.

        Context determines which parent element's color to use:
        - 'bus': Plotting a bus balance, so color by the flow's parent component
        - 'component': Plotting a component, so color by the flow's connected bus/carrier

        Args:
            label: Flow label (label_full format, e.g., 'Boiler(Q_th)').
            context: Either 'bus' or 'component'.

        Returns:
            Color string or None if not configured.
        """
        # Find the flow
        if label not in self._fs.flows:
            return None

        flow = self._fs.flows[label]

        if context == 'bus':
            # Plotting a bus balance → color by component
            return self.for_component(flow.component)
        else:
            # Plotting a component → color by bus/carrier
            bus_label = flow.bus if isinstance(flow.bus, str) else flow.bus.label
            return self.for_bus(bus_label)

    def get_color_map_for_balance(
        self,
        node: str,
        flow_labels: list[str],
        fallback_colorscale: str | None = None,
    ) -> dict[str, str]:
        """Get a complete color mapping for a balance plot.

        This method creates a color map for all flows in a balance plot,
        using context-aware logic (component colors for bus plots,
        carrier colors for component plots).

        Args:
            node: The bus or component being plotted.
            flow_labels: List of flow labels to color.
            fallback_colorscale: Colorscale for flows without configured colors.

        Returns:
            Dictionary mapping each flow label to a color.
        """
        if fallback_colorscale is None:
            fallback_colorscale = CONFIG.Plotting.default_qualitative_colorscale

        # Determine context based on node type
        if node in self._fs.buses:
            context: Literal['bus', 'component'] = 'bus'
        else:
            context = 'component'

        # Build color map from configured colors
        color_map = {}
        labels_without_colors = []

        for label in flow_labels:
            color = self.for_flow(label, context)
            if color is not None:
                color_map[label] = color
            else:
                labels_without_colors.append(label)

        # Fill remaining with colorscale
        if labels_without_colors:
            fallback_colors = process_colors(fallback_colorscale, labels_without_colors)
            color_map.update(fallback_colors)

        return color_map

    def get_color_map_for_sankey(
        self,
        node_labels: list[str],
        fallback_colorscale: str | None = None,
    ) -> dict[str, str]:
        """Get a complete color mapping for a sankey diagram.

        Sankey nodes (buses and components) are colored based on:
        - Buses: Use carrier color or explicit bus color
        - Components: Use explicit component color or fallback

        Args:
            node_labels: List of node labels (buses and components).
            fallback_colorscale: Colorscale for nodes without configured colors.

        Returns:
            Dictionary mapping each node label to a color.
        """
        if fallback_colorscale is None:
            fallback_colorscale = CONFIG.Plotting.default_qualitative_colorscale

        color_map = {}
        labels_without_colors = []

        for label in node_labels:
            # Try bus color first (includes carrier resolution)
            color = self.for_bus(label)
            if color is None:
                # Try component color
                color = self.for_component(label)

            if color is not None:
                color_map[label] = color
            else:
                labels_without_colors.append(label)

        # Fill remaining with colorscale
        if labels_without_colors:
            fallback_colors = process_colors(fallback_colorscale, labels_without_colors)
            color_map.update(fallback_colors)

        return color_map

    def reset(self) -> None:
        """Clear all color configurations."""
        self._component_colors.clear()
        self._bus_colors.clear()
        self._carrier_colors.clear()

    def to_dict(self) -> dict:
        """Convert color configuration to a dictionary for serialization.

        Returns:
            Dictionary with component, bus, and carrier color mappings.
        """
        return {
            'component_colors': self._component_colors.copy(),
            'bus_colors': self._bus_colors.copy(),
            'carrier_colors': self._carrier_colors.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict, flow_system: FlowSystem) -> ColorAccessor:
        """Create a ColorAccessor from a serialized dictionary.

        Args:
            data: Dictionary from to_dict().
            flow_system: The FlowSystem this accessor belongs to.

        Returns:
            New ColorAccessor instance with restored configuration.
        """
        accessor = cls(flow_system)
        accessor._component_colors = data.get('component_colors', {}).copy()
        accessor._bus_colors = data.get('bus_colors', {}).copy()
        accessor._carrier_colors = data.get('carrier_colors', {}).copy()
        return accessor

    def __repr__(self) -> str:
        n_components = len(self._component_colors)
        n_buses = len(self._bus_colors)
        n_carriers = len(self._carrier_colors)
        return f'ColorAccessor({n_components} components, {n_buses} buses, {n_carriers} carriers)'
