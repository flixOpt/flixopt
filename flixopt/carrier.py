"""Carrier class for energy/material type definitions.

Carriers represent types of energy or materials that flow through buses,
such as electricity, heat, gas, or water. They provide consistent styling
and metadata across visualizations.
"""

from __future__ import annotations

from .structure import ContainerMixin, Interface


class Carrier(Interface):
    """Definition of an energy or material carrier type.

    Carriers represent the type of energy or material flowing through a Bus.
    They provide consistent color, unit, and description across all visualizations
    and can be shared between multiple buses of the same type.

    Inherits from Interface to provide serialization capabilities.

    Args:
        name: Identifier for the carrier (e.g., 'electricity', 'heat', 'gas').
        color: Hex color string for visualizations (e.g., '#FFD700').
        unit: Unit string for display (e.g., 'kW', 'kW_th', 'm³/h').
        description: Optional human-readable description.

    Examples:
        Creating custom carriers:

        ```python
        import flixopt as fx

        # Define custom carriers
        electricity = fx.Carrier('electricity', '#FFD700', 'kW', 'Electrical power')
        district_heat = fx.Carrier('district_heat', '#FF6B6B', 'kW_th', 'District heating')
        hydrogen = fx.Carrier('hydrogen', '#00CED1', 'kg/h', 'Hydrogen fuel')

        # Register with FlowSystem
        flow_system.add_carrier(electricity)
        flow_system.add_carrier(district_heat)

        # Use with buses (just reference by name)
        elec_bus = fx.Bus('MainGrid', carrier='electricity')
        heat_bus = fx.Bus('HeatingNetwork', carrier='district_heat')
        ```

        Using predefined carriers from CONFIG:

        ```python
        # Access built-in carriers
        elec = fx.CONFIG.Carriers.electricity
        heat = fx.CONFIG.Carriers.heat

        # Use directly
        bus = fx.Bus('Grid', carrier='electricity')
        ```

        Adding custom carriers to CONFIG:

        ```python
        # Add a new carrier globally
        fx.CONFIG.Carriers.add(fx.Carrier('biogas', '#228B22', 'kW', 'Biogas'))

        # Now available as
        fx.CONFIG.Carriers.biogas
        ```

    Note:
        Carriers are compared by name for equality, allowing flexible usage
        patterns where the same carrier type can be referenced by name string
        or Carrier object interchangeably.
    """

    def __init__(
        self,
        name: str,
        color: str = '',
        unit: str = '',
        description: str = '',
    ) -> None:
        """Initialize a Carrier.

        Args:
            name: Identifier for the carrier (normalized to lowercase).
            color: Hex color string for visualizations.
            unit: Unit string for display.
            description: Optional human-readable description.
        """
        self.name = name.lower()
        self.color = color
        self.unit = unit
        self.description = description

    def transform_data(self, name_prefix: str = '') -> None:
        """Transform data to match FlowSystem dimensions.

        Carriers don't have time-series data, so this is a no-op.

        Args:
            name_prefix: Ignored for Carrier.
        """
        pass  # Carriers have no data to transform

    @property
    def label(self) -> str:
        """Label for container keying (alias for name)."""
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Carrier):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other.lower()
        return False

    def __repr__(self):
        return f"Carrier('{self.name}', color='{self.color}', unit='{self.unit}')"

    def __str__(self):
        return self.name


class CarrierContainer(ContainerMixin['Carrier']):
    """Container for Carrier objects.

    Uses carrier.name for keying. Provides dict-like access to carriers
    registered with a FlowSystem.

    Examples:
        ```python
        # Access via FlowSystem
        carriers = flow_system.carriers

        # Dict-like access
        elec = carriers['electricity']
        'heat' in carriers  # True/False

        # Iteration
        for name in carriers:
            print(name)
        ```
    """

    def __init__(self, carriers: list[Carrier] | dict[str, Carrier] | None = None):
        """Initialize a CarrierContainer.

        Args:
            carriers: Initial carriers to add.
        """
        super().__init__(elements=carriers, element_type_name='carriers')

    def _get_label(self, carrier: Carrier) -> str:
        """Extract name from Carrier for keying."""
        return carrier.name
