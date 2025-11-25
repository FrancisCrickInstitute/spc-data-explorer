"""
Utilities package for the Phenotype Clustering Interactive Visualization App.

This package contains utility modules for image processing, color generation,
and other helper functions used throughout the application.

Modules:
- image_utils: Image processing, thumbnail finding, and text overlay functions
- color_utils: Color generation and palette management utilities

The utils package provides reusable functionality that can be imported
and used across different components of the application.
"""

from .image_utils import (
    find_thumbnail, 
    add_text_to_image, 
    load_image_base64,
    ImageNotFoundError
)
from .color_utils import generate_colors

__all__ = [
    'find_thumbnail', 
    'add_text_to_image', 
    'load_image_base64',
    'ImageNotFoundError',
    'generate_colors'
]