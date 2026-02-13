"""
Color utilities module for the Phenotype Clustering Interactive Visualization App.

This module provides functions for generating colors and managing color palettes
for the visualization dashboard. It includes utilities for creating distinct
color sets and managing color mappings for different data categories.

Key Functions:
- generate_colors(): Generate n distinct colors using HSV color space
- get_color_column_config(): Get predefined color configurations for data columns
- determine_color_type(): Automatically determine if a column should use continuous or discrete coloring

The module ensures consistent and visually appealing color schemes throughout
the application while providing flexibility for different data types.
"""

import colorsys
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import plotly.express as px
import logging

# Set up logging
logger = logging.getLogger(__name__)

# 100 maximally distinct colors generated via `pastel distinct 100`
PASTEL_DISTINCT_100 = [
    '#88fd70', '#0000ff', '#fe003b', '#08b1f4', '#240001', '#ff03c8', '#fcab0a', '#006b3b',
    '#f1beb9', '#4c0072', '#06fffb', '#765404', '#fbff01', '#0178ff', '#ea9eed', '#bf1b67',
    '#004962', '#edffa7', '#738b13', '#28ff07', '#c346f5', '#7f2000', '#936659', '#9fc3b6',
    '#ff9654', '#0554ff', '#ff656f', '#8c7bb8', '#ff68ca', '#00bb51', '#c8ae46', '#007200',
    '#0536b2', '#944fb6', '#a0ffaa', '#9eff03', '#34002e', '#5fcda8', '#52000c', '#ff7da8',
    '#1399ff', '#b34d5c', '#00180a', '#435400', '#a595ff', '#697277', '#6c00ab', '#008d7f',
    '#a200ff', '#f8f7ff', '#004e42', '#e881ff', '#adb900', '#000047', '#fff36d', '#08c90e',
    '#673a77', '#ff0095', '#a98e57', '#003369', '#ffd2ff', '#a8a9c4', '#b67900', '#6f8e66',
    '#4555b1', '#ffd502', '#b30033', '#ff00ff', '#b95b9b', '#890070', '#b4633b', '#ffefc5',
    '#ff2600', '#1bffca', '#1187a4', '#5c3545', '#001027', '#b20000', '#492700', '#0700c4',
    '#ff0f6b', '#525234', '#ff8000', '#fc9987', '#56a700', '#c1849d', '#ffc485', '#0cc1d2',
    '#d1fe56', '#a0e1ff', '#a5d35f', '#136eaf', '#74033f', '#00ff99', '#c8ffd4', '#003100',
    '#5aa85e', '#b10bac', '#946dff', '#ff633c',
]


def generate_colors(n: int) -> List[str]:
    """
    Generate n distinct colors using HSV color space.
    
    This function creates visually distinct colors by distributing hue values
    evenly around the color wheel while maintaining consistent saturation
    and brightness for optimal visibility.
    
    Args:
        n: Number of colors to generate
        
    Returns:
        List[str]: List of RGB color strings in format 'rgb(r,g,b)'
    """
    if n <= 0:
        return []
    
    # Generate HSV tuples with even hue distribution
    hsv_tuples = [(x/n, 0.8, 0.9) for x in range(n)]
    
    # Convert HSV to RGB
    rgb_tuples = [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_tuples]
    
    # Format as RGB strings
    rgb_strings = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                   for r, g, b in rgb_tuples]
    
    logger.debug(f"Generated {n} distinct colors")
    return rgb_strings


def get_color_column_config() -> List[Tuple[str, bool, str, Any]]:
    """
    Get predefined color column configurations.
    
    Returns a list of tuples defining how different data columns should be
    colored in the visualization. Each tuple contains:
    (column_name, is_continuous, display_name, color_palette)
    
    Returns:
        List[Tuple]: Color configuration tuples
    """
    color_columns = [
        # Categorical columns
        ('library', False, 'Library', px.colors.qualitative.Bold),
        ('landmark_label', False, 'Landmark Status', px.colors.qualitative.Set1),
        ('moa_first', False, 'MOA (First)', px.colors.qualitative.Dark24),
        ('moa_compound_uM', False, 'MOA with Concentration', px.colors.qualitative.Dark24),             
        ('treatment', False, 'Treatment', px.colors.qualitative.Light24),
        ('manual_annotation', False, 'Broad Annotation', PASTEL_DISTINCT_100),
        
        # Landmark label variants
        ('landmark_label_mad', False, 'Landmark Status (MAD)', px.colors.qualitative.Set1),
        ('landmark_label_std', False, 'Landmark Status (StdDev)', px.colors.qualitative.Set1),
        ('landmark_label_var', False, 'Landmark Status (Variance)', px.colors.qualitative.Set1),
        
        # Plate and well columns
        ('plate', False, 'Plate', px.colors.qualitative.Alphabet),
        ('well', False, 'Well', px.colors.qualitative.Bold),
        ('well_row', False, 'Well Row', px.colors.qualitative.Bold),
        ('well_column', False, 'Well Column', px.colors.qualitative.Bold),
        ('compound_name', False, 'Compound Name', px.colors.qualitative.Vivid),
        ('compound_uM', False, 'Compound Concentration (µM)', px.colors.qualitative.Vivid),
        ('compound_type', False, 'Compound Type', px.colors.qualitative.Safe),  # ADDED compound_type
        
        # Continuous metrics
        ('cosine_distance_from_dmso', True, 'Cosine Distance from DMSO', 'Plasma'),
        ('mad_cosine', True, 'MAD Cosine', 'Inferno'),
        ('var_cosine', True, 'Variance Cosine', 'Cividis'),
        ('std_cosine', True, 'Std Dev Cosine', 'Magma'),
        ('median_distance', True, 'Median Distance', 'Viridis'),
        ('well_count', False, 'Well Count', px.colors.qualitative.Prism),
        ('cell_count', True, 'Cell Count', 'Viridis'),
        ('closest_landmark_distance', True, 'Closest Landmark Distance', 'Plasma'),
        ('second_closest_landmark_distance', True, '2nd Closest Landmark Distance', 'Inferno'),
        ('third_closest_landmark_distance', True, '3rd Closest Landmark Distance', 'Cividis'),
        ('score_harmonic_mean_2term_mad_cosine', True, 'Score Harmonic Mean (2-term MAD)', 'Magma'),
        ('score_harmonic_mean_3term_mad_cosine', True, 'Score Harmonic Mean (3-term MAD)', 'Turbo'),
    ]
    
    return color_columns


def filter_available_color_columns(df: pd.DataFrame) -> List[Tuple[str, bool, str, Any]]:
    """
    Filter color column configurations to only include columns present in the dataframe.
    
    Args:
        df: DataFrame to check for column availability
        
    Returns:
        List[Tuple]: Filtered color configuration tuples
    """
    all_color_columns = get_color_column_config()
    available_color_columns = []
    
    for col_info in all_color_columns:
        col_name = col_info[0]
        
        if col_name in df.columns:
            # Check if column has actual data (non-NaN values)
            non_nan_count = df[col_name].notna().sum()
            if non_nan_count > 0:
                available_color_columns.append(col_info)
                logger.debug(f"Color column '{col_name}' available with {non_nan_count} values")
            else:
                logger.warning(f"Skipping color column '{col_name}' - no non-NaN values")
        else:
            logger.debug(f"Skipping color column '{col_name}' - not found in data")
    
    logger.info(f"Found {len(available_color_columns)} available color columns")
    return available_color_columns


def determine_color_type(df: pd.DataFrame, column: str, 
                        predefined_config: Optional[Tuple] = None) -> Tuple[bool, str, Any]:
    """
    Automatically determine if a column should use continuous or discrete coloring.
    
    Args:
        df: DataFrame containing the column
        column: Column name to analyze
        predefined_config: Optional predefined configuration tuple
        
    Returns:
        Tuple[bool, str, Any]: (is_continuous, display_name, color_palette)
    """
    if predefined_config:
        # Use predefined configuration
        _, is_continuous, display_name, color_palette = predefined_config
        return is_continuous, display_name, color_palette
    
    # Auto-detect based on data characteristics
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in dataframe")
        return False, column.replace('_', ' ').title(), None
    
    series = df[column].dropna()
    
    if len(series) == 0:
        logger.warning(f"Column '{column}' has no non-null values")
        return False, column.replace('_', ' ').title(), None
    
    # Check if numeric
    is_numeric = pd.api.types.is_numeric_dtype(series)
    n_unique = series.nunique()
    
    # Decision logic for continuous vs discrete
    if is_numeric and n_unique > 12:
        # Numeric with many unique values -> continuous
        is_continuous = True
        color_palette = 'Viridis'
        logger.debug(f"Column '{column}' detected as continuous ({n_unique} unique values)")
    else:
        # Categorical or numeric with few unique values -> discrete
        is_continuous = False
        color_palette = px.colors.qualitative.Set1
        logger.debug(f"Column '{column}' detected as discrete ({n_unique} unique values)")
    
    display_name = column.replace('_', ' ').title()
    return is_continuous, display_name, color_palette


def get_color_parameters(df: pd.DataFrame, color_column: str, 
                        color_info: Optional[Tuple] = None) -> Dict[str, Any]:
    """
    Get appropriate color parameters for plotly plots.
    
    Args:
        df: DataFrame containing the data
        color_column: Name of the column to color by
        color_info: Optional predefined color configuration
        
    Returns:
        Dict[str, Any]: Color parameters for plotly
    """
    is_continuous, display_name, color_palette = determine_color_type(
        df, color_column, color_info
    )
    
    color_params = {}
    
    if is_continuous:
        # Continuous coloring parameters
        if color_palette:
            color_params['color_continuous_scale'] = color_palette
        else:
            color_params['color_continuous_scale'] = 'Viridis'
        
        # Handle NaN values in continuous data
        if df[color_column].isna().any():
            non_na_values = df[color_column].dropna()
            if len(non_na_values) > 0:
                color_params['range_color'] = [non_na_values.min(), non_na_values.max()]
    
    else:
        # Discrete coloring parameters
        if color_palette:
            color_params['color_discrete_sequence'] = color_palette
        else:
            # Choose palette based on column characteristics
            color_params['color_discrete_sequence'] = _choose_discrete_palette(color_column)
        
        # Add category orders for specific columns
        category_orders = _get_category_orders(df, color_column)
        if category_orders:
            color_params['category_orders'] = {color_column: category_orders}
    
    return color_params


def _choose_discrete_palette(color_column: str) -> List[str]:
    """
    Choose an appropriate discrete color palette based on column name.
    
    Args:
        color_column: Name of the column
        
    Returns:
        List[str]: Color palette
    """
    if 'landmark' in color_column.lower():
        return px.colors.qualitative.Set1
    elif color_column.lower() in ['moa', 'moa_first']:
        return px.colors.qualitative.Dark24
    elif color_column.lower() == 'plate':
        return px.colors.qualitative.Alphabet
    elif 'well' in color_column.lower():
        return px.colors.qualitative.Bold
    elif color_column.lower() == 'library':
        return px.colors.qualitative.Bold
    elif color_column.lower() == 'compound_type':
        return px.colors.qualitative.Safe
    # ADD THIS LINE for manual_annotation
    elif color_column.lower() == 'manual_annotation':
        return PASTEL_DISTINCT_100
    else:
        return px.colors.qualitative.Light24


def _get_category_orders(df: pd.DataFrame, color_column: str) -> Optional[List[str]]:
    """
    Get category ordering for specific columns that benefit from ordered display.
    
    Args:
        df: DataFrame containing the data
        color_column: Column name
        
    Returns:
        Optional[List[str]]: Ordered categories, or None if no special ordering needed
    """
    try:
            from ..config_loader import get_config
    except ImportError:
            from config_loader import get_config
    Config = get_config()
    
    if color_column == 'well':
        return Config.get_well_list()
    elif color_column == 'well_column':
        return Config.get_column_list()
    elif color_column == 'well_row':
        return list(Config.WELL_ROWS)
    # Alpha numerical 
    else:
        # For any other categorical column, return sorted unique values
        if color_column in df.columns:
            unique_vals = df[color_column].dropna().unique()
            # Convert to string and sort alphanumerically
            sorted_vals = sorted([str(val) for val in unique_vals], key=lambda x: (x.lower(), x))
            return sorted_vals
    
    return None


def create_color_options(available_color_columns: List[Tuple]) -> List[Dict[str, str]]:
    """
    Create dropdown options for color selection.
    
    Args:
        available_color_columns: List of available color column configurations
        
    Returns:
        List[Dict]: Options formatted for Dash dropdown
    """
    return [
        {'label': display_name, 'value': col_name} 
        for col_name, _, display_name, _ in available_color_columns
    ]