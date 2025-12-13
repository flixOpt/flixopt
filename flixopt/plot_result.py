"""Plot result container for unified plotting API.

This module provides the PlotResult class that wraps plotting outputs
across the entire flixopt package, ensuring a consistent interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import plotly.graph_objects as go
    import xarray as xr


@dataclass
class PlotResult:
    """Container returned by all plot methods. Holds both data and figure.

    This class provides a unified interface for all plotting methods across
    the flixopt package, enabling consistent method chaining and export options.

    Attributes:
        data: Prepared xarray Dataset used for the plot.
        figure: Plotly figure object.

    Examples:
        Basic usage with chaining:

        >>> result = flow_system.statistics.plot.balance('Bus')
        >>> result.show().to_html('plot.html')

        Accessing underlying data:

        >>> result = flow_system.statistics.plot.flows()
        >>> df = result.data.to_dataframe()
        >>> result.to_csv('data.csv')

        Customizing the figure:

        >>> result = clustering.plot()
        >>> result.update(title='My Custom Title').show()
    """

    data: xr.Dataset
    figure: go.Figure

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebook display."""
        return self.figure.to_html(full_html=False, include_plotlyjs='cdn')

    def show(self) -> PlotResult:
        """Display the figure. Returns self for chaining."""
        self.figure.show()
        return self

    def update(self, **layout_kwargs: Any) -> PlotResult:
        """Update figure layout. Returns self for chaining.

        Args:
            **layout_kwargs: Arguments passed to plotly's update_layout().

        Returns:
            Self for method chaining.

        Examples:
            >>> result.update(title='New Title', height=600)
        """
        self.figure.update_layout(**layout_kwargs)
        return self

    def update_traces(self, **trace_kwargs: Any) -> PlotResult:
        """Update figure traces. Returns self for chaining.

        Args:
            **trace_kwargs: Arguments passed to plotly's update_traces().

        Returns:
            Self for method chaining.

        Examples:
            >>> result.update_traces(line_width=2, marker_size=8)
        """
        self.figure.update_traces(**trace_kwargs)
        return self

    def to_html(self, path: str | Path) -> PlotResult:
        """Save figure as interactive HTML. Returns self for chaining.

        Args:
            path: File path for the HTML output.

        Returns:
            Self for method chaining.
        """
        self.figure.write_html(str(path))
        return self

    def to_image(self, path: str | Path, **kwargs: Any) -> PlotResult:
        """Save figure as static image. Returns self for chaining.

        Args:
            path: File path for the image (format inferred from extension).
            **kwargs: Additional arguments passed to write_image().

        Returns:
            Self for method chaining.

        Examples:
            >>> result.to_image('plot.png', scale=2)
            >>> result.to_image('plot.svg')
        """
        self.figure.write_image(str(path), **kwargs)
        return self

    def to_csv(self, path: str | Path, **kwargs: Any) -> PlotResult:
        """Export the underlying data to CSV. Returns self for chaining.

        Args:
            path: File path for the CSV output.
            **kwargs: Additional arguments passed to to_csv().

        Returns:
            Self for method chaining.
        """
        self.data.to_dataframe().to_csv(path, **kwargs)
        return self

    def to_netcdf(self, path: str | Path, **kwargs: Any) -> PlotResult:
        """Export the underlying data to netCDF. Returns self for chaining.

        Args:
            path: File path for the netCDF output.
            **kwargs: Additional arguments passed to to_netcdf().

        Returns:
            Self for method chaining.
        """
        self.data.to_netcdf(path, **kwargs)
        return self
