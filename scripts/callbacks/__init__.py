"""
Callbacks package for the Phenotype Clustering Interactive Visualization App.

This package contains all Dash callback functions that handle user interactions
and update the application state. Each module focuses on specific functionality
to maintain clean separation of concerns and improve maintainability.

Modules:
- plot_callbacks: Callbacks for main plot updates, axis controls, and visualization
- image_callbacks: Callbacks for image display, hover effects, and click interactions  
- search_callbacks: Callbacks for search functionality, dropdown population, and results
- landmark_callbacks: Callbacks for landmark distance analysis plot

The callbacks package implements the interactive behavior of the application,
connecting user interface events to data processing and display updates.
Each callback is designed to be modular and testable.
"""

try:
    from .plot_callbacks import register_plot_callbacks
    from .image_callbacks import register_image_callbacks
    from .search_callbacks import register_search_callbacks
    from .landmark_callbacks import register_landmark_callbacks  
except ImportError:
    from plot_callbacks import register_plot_callbacks
    from image_callbacks import register_image_callbacks
    from search_callbacks import register_search_callbacks
    from landmark_callbacks import register_landmark_callbacks 

__all__ = [
    'register_plot_callbacks',
    'register_image_callbacks', 
    'register_search_callbacks',
    'register_landmark_callbacks' 
]