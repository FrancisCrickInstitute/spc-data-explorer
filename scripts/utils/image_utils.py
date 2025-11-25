"""
Image utilities module for the Phenotype Clustering Interactive Visualization App.

This module provides functions for finding, loading, and processing thumbnail images
used in the phenotype clustering visualization. It handles image path resolution,
text overlay addition, and base64 encoding for web display.

Key Functions:
- find_thumbnail(): Locate thumbnail images with fallback logic
- add_text_to_image(): Add text overlays to images with proper formatting
- load_image_base64(): Convert images to base64 for web display

The module includes robust error handling and fallback mechanisms to ensure
the application continues working even when some images are missing.
"""

import os
import glob
import base64
import io
from pathlib import Path
from typing import Optional
import logging

from PIL import Image, ImageDraw, ImageFont

try:
    from ..config_loader import get_config
except ImportError:
    from config_loader import get_config

# Set up logging
logger = logging.getLogger(__name__)


class ImageNotFoundError(Exception):
    """Custom exception for when thumbnail images cannot be found."""
    pass


def find_thumbnail(plate: str, well: str, site: str = "01", 
                  scaling_mode: str = "fixed") -> Optional[Path]:
    """
    Find the thumbnail image for a given plate, well, site, and scaling mode.
    """
    Config = get_config()
    
    # Standardize well format (ensure A01 not A1)
    well = _standardize_well_format(well)
    
    # Normalize THUMBNAIL_DIRS to always be a list
    thumbnail_dirs = Config.THUMBNAIL_DIRS if isinstance(Config.THUMBNAIL_DIRS, list) else [Config.THUMBNAIL_DIRS]
    
    # Try each thumbnail directory
    for thumbnail_base_dir in thumbnail_dirs:
        if not thumbnail_base_dir.exists():
            continue
        
        # Check scaling subdirectory
        if scaling_mode == "auto":
            scaling_dir = thumbnail_base_dir / "auto"
        else:
            scaling_dir = thumbnail_base_dir / "fixed"
            
        if not scaling_dir.exists():
            continue
        
        # Check plate subdirectory
        plate_dir = scaling_dir / str(plate)
        if not plate_dir.exists():
            continue
        
        # Build the expected path for the selected scaling mode
        thumbnail_path = plate_dir / f"{plate}_{well}_{site}.png"
        
        # Check if the exact file exists
        if thumbnail_path.exists():
            return thumbnail_path
        
        # Try with different site numbers if the exact one wasn't found
        for i in range(1, 11):  # Try sites 01-10
            site_str = f"{i:02d}"
            alt_path = plate_dir / f"{plate}_{well}_{site_str}.png"
            
            if alt_path.exists():
                return alt_path
        
        # If still not found, use glob pattern as last resort
        pattern = str(plate_dir / f"{plate}_{well}_*.png")
        matches = glob.glob(pattern)
        if matches:
            return Path(matches[0])
    
    # Final fallback: if auto-scaling thumbnail not found, try fixed-scaling
    if scaling_mode == "auto":
        return find_thumbnail(plate, well, site, "fixed")
    
    return None


def add_text_to_image(img_path: Optional[Path], text: str, 
                     add_label: bool = False) -> str:
    """
    Add text overlay to an image and return as base64 encoded string.
    
    This function loads an image, optionally adds a text overlay in the top-right
    corner with proper formatting (white text with black outline), and returns
    the result as a base64 encoded string suitable for web display.
    
    Args:
        img_path: Path to the image file, or None
        text: Text to overlay on the image
        add_label: Whether to actually add the text overlay
        
    Returns:
        str: Base64 encoded image data URI, or empty string if failed
    """
    if img_path is None or not img_path.exists():
        logger.warning(f"Cannot load image: {img_path}")
        return ""
    
    try:
        # Open and process the image
        with Image.open(img_path) as img:
            # Convert to RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
                logger.debug("Converted image to RGB mode")
            
            # Only add text if requested and text is provided
            if add_label and text:
                img = _add_text_overlay(img, text)
            
            # Convert to base64
            return _image_to_base64(img)
            
    except Exception as e:
        logger.error(f"Error processing image {img_path}: {str(e)}")
        return ""


def load_image_base64(img_path: Optional[Path]) -> str:
    """
    Load an image file and convert to base64 for display (without text overlay).
    
    Args:
        img_path: Path to the image file
        
    Returns:
        str: Base64 encoded image data URI, or empty string if failed
    """
    if img_path is None or not img_path.exists():
        logger.warning(f"Cannot load image: {img_path}")
        return ""
    
    try:
        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{img_data}"
    except Exception as e:
        logger.error(f"Error loading image {img_path}: {str(e)}")
        return ""


def _standardize_well_format(well: str) -> str:
    """
    Standardize well format to ensure consistent A01 format (not A1).
    
    Args:
        well: Well identifier
        
    Returns:
        str: Standardized well format
    """
    # Handle None
    if well is None:
        return well
    
    # Convert to string if it's not already (handles int, float, etc.)
    well = str(well)
    
    # Check length after conversion
    if not well or len(well) < 2:
        return well
    
    # For wells like "A1", standardize to "A01"
    if len(well) == 2 and well[0].isalpha() and well[1].isdigit():
        return f"{well[0]}{int(well[1:]):02d}"
    
    return well


def _add_text_overlay(img: Image.Image, text: str) -> Image.Image:
    """
    Add text overlay to an image with proper formatting.
    
    Args:
        img: PIL Image object
        text: Text to overlay
        
    Returns:
        Image.Image: Image with text overlay
    """
    # Create a copy to draw on
    img_with_text = img.copy()
    draw = ImageDraw.Draw(img_with_text)
    
    # Load font with fallback
    font = _load_font()
    
    # Get image and text dimensions
    img_width, img_height = img_with_text.size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position text in top right corner with margin
    margin = 10
    x = img_width - text_width - margin
    y = margin
    
    # Add text with outline for better visibility
    _draw_text_with_outline(draw, (x, y), text, font)
    
    return img_with_text


def _load_font() -> ImageFont.ImageFont:
    """
    Load a font with fallback to default if system fonts not available.
    
    Returns:
        ImageFont.ImageFont: Loaded font object
    """
    Config = get_config()
    try:
        # Try system fonts in order of preference
        for font_path in Config.FONT_PATHS:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, Config.FONT_SIZE)
                logger.debug(f"Loaded font: {font_path}")
                return font
        
        # If no system font found, use default
        logger.debug("Using default font")
        return ImageFont.load_default()
        
    except Exception as e:
        logger.warning(f"Font loading error: {e}, using default")
        return ImageFont.load_default()


def _draw_text_with_outline(draw: ImageDraw.ImageDraw, position: tuple, 
                           text: str, font: ImageFont.ImageFont) -> None:
    """
    Draw text with a black outline for better visibility.
    
    Args:
        draw: ImageDraw object
        position: (x, y) position for text
        text: Text to draw
        font: Font to use
    """
    x, y = position
    
    # Draw black outline (8-directional)
    for dx in [-1, -1, -1, 0, 0, 1, 1, 1]:
        for dy in [-1, 0, 1, -1, 1, -1, 0, 1]:
            draw.text((x + dx, y + dy), text, font=font, fill='black')
    
    # Draw main white text
    draw.text((x, y), text, font=font, fill='white')


def _image_to_base64(img: Image.Image) -> str:
    """
    Convert PIL Image to base64 data URI.
    
    Args:
        img: PIL Image object
        
    Returns:
        str: Base64 encoded data URI
    """
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_data = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_data}"


def extract_site_from_path(img_path: Path) -> str:
    """
    Extract site number from thumbnail file path.
    
    Args:
        img_path: Path to thumbnail file
        
    Returns:
        str: Site number (e.g., "01", "05") or "Unknown"
    """
    if not img_path or "_" not in img_path.name:
        return "Unknown"
    
    try:
        # Expected format: plate_well_site.png
        parts = img_path.stem.split("_")
        if len(parts) >= 3:
            return parts[2]  # Site part
    except Exception as e:
        logger.debug(f"Could not extract site from {img_path}: {e}")
    
    return "Unknown"
