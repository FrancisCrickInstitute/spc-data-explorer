"""
Components package for the Phenotype Clustering Interactive Visualization App.

This package contains modules that define the user interface components
and layout structure of the Dash application. Each module focuses on
a specific aspect of the UI to maintain clean separation of concerns.

Modules:
- layout: Main application layout and overall structure
- controls: Control panels including dropdowns, sliders, and toggles
- search: Search interface components for compound ID, MOA, and DMSO searches

The components package provides a modular approach to UI construction,
making it easy to modify individual interface elements without affecting
the entire application structure.
"""

try:
    from .layout import create_layout
    from .controls import create_control_panels
    from .search import create_search_interface
except ImportError:
    from components.layout import create_layout
    from components.controls import create_control_panels
    from components.search import create_search_interface

__all__ = [
    'create_layout',
    'create_control_panels', 
    'create_search_interface'
]