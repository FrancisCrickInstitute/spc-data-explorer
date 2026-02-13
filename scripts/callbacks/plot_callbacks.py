"""
Plot callbacks module for the Phenotype Clustering Interactive Visualization App.

UPDATED (2025-01-21): Now supports BOTH SPC and CellProfiler datasets!

This module contains all callback functions related to plot generation, axis controls,
and visualization updates. It handles the main scatter plot creation with different
plot types, color schemes, and interactive features.

Key Functions:
- register_plot_callbacks(): Register all plot-related callbacks with the app
- update_axis_dropdowns(): Handle plot type changes and axis control enablement
- update_plot(): Main callback for generating and updating the scatter plot
- _prepare_plot_data(): Data preprocessing for plotting
- _get_color_parameters(): Color parameter configuration for different data types

The module implements sophisticated plotting logic with support for multiple
dimensionality reduction techniques, flexible coloring schemes, and interactive features.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback_context, no_update, State
import traceback
import logging
from typing import Dict, Any, Tuple, List, Optional

try:
    from ..config_loader import get_config
    from ..utils.color_utils import (
        get_color_parameters, 
        determine_color_type,
        filter_available_color_columns
    )
except ImportError:
    from config_loader import get_config
    from utils.color_utils import (
        get_color_parameters, 
        determine_color_type,
        filter_available_color_columns
    )

# Set up logging
logger = logging.getLogger(__name__)


def register_plot_callbacks(app, df_spc: pd.DataFrame, df_cp: pd.DataFrame,
                           available_color_columns: List[Tuple],
                           plot_types: List[Dict[str, Any]],
                           available_metrics: List[str]) -> None:
    """
    Register all plot-related callbacks with the Dash app.
    
    UPDATED (2025-01-21): Now accepts BOTH SPC and CP dataframes!
    
    Args:
        app: Dash application instance
        df_spc: SPC dataframe
        df_cp: CellProfiler dataframe
        available_color_columns: Available color column configurations
        plot_types: Available plot type configurations
        available_metrics: Available metric columns for plotting
    """
    
    def _get_dataframe_for_plot_type(plot_type: str) -> Optional[pd.DataFrame]:
        """
        Determine which dataframe to use based on plot type.
        
        Args:
            plot_type: Selected plot type value
            
        Returns:
            Appropriate dataframe (df_spc or df_cp) or None
        """
        if plot_type and ('cp' in plot_type or 'cellprofiler' in plot_type.lower()):
            # Use CellProfiler data
            logger.debug(f"Plot type '{plot_type}' → using CP dataframe")
            return df_cp if df_cp is not None else df_spc
        else:
            # Use SPC data
            logger.debug(f"Plot type '{plot_type}' → using SPC dataframe")
            return df_spc if df_spc is not None else df_cp
    
    @app.callback(
        [Output('x-axis-dropdown', 'disabled'),
         Output('y-axis-dropdown', 'disabled'),
         Output('x-axis-dropdown', 'value'),
         Output('y-axis-dropdown', 'value')],
        [Input('plot-type-dropdown', 'value')]
    )
    def update_axis_dropdowns(plot_type: str) -> Tuple[bool, bool, Optional[str], Optional[str]]:
        """
        Update axis dropdown states based on selected plot type.
        
        When a predefined plot type is selected, disable custom axis controls.
        When 'custom' is selected, enable axis controls for manual selection.
        
        Args:
            plot_type: Selected plot type value
            
        Returns:
            Tuple of (x_disabled, y_disabled, x_value, y_value)
        """
        try:
            if plot_type == 'custom':
                # Enable custom axis selection
                default_x = 'UMAP1' if 'UMAP1' in available_metrics else (
                    available_metrics[0] if available_metrics else None
                )
                default_y = 'UMAP2' if 'UMAP2' in available_metrics else (
                    available_metrics[1] if len(available_metrics) > 1 else None
                )
                
                logger.debug(f"Custom plot mode: x={default_x}, y={default_y}")
                return False, False, default_x, default_y
            else:
                # Use predefined plot configuration
                plot_info = next((p for p in plot_types if p['value'] == plot_type), None)
                if plot_info:
                    logger.debug(f"Predefined plot: {plot_type}, x={plot_info['x']}, y={plot_info['y']}")
                    return True, True, plot_info['x'], plot_info['y']
                
                logger.warning(f"Unknown plot type: {plot_type}")
                return True, True, None, None
                
        except Exception as e:
            logger.error(f"Error updating axis dropdowns: {str(e)}")
            return True, True, None, None
    
    
    @app.callback(
        Output('main-plot', 'figure'),
        [Input('plot-type-dropdown', 'value'),
         Input('x-axis-dropdown', 'value'),
         Input('y-axis-dropdown', 'value'),
         Input('color-dropdown', 'value'),
         Input('point-size-slider', 'value'),
         Input('plot-search-dropdown', 'value')]
    )
    def update_plot(plot_type: str, x_col: str, y_col: str, 
                   color_col: str, point_size: int,
                   highlight_compound: str = None) -> Dict[str, Any]: 
        """
        Generate and update the main scatter plot.
        
        This is the primary callback for plot generation, handling different
        plot types, color schemes, and styling options for BOTH datasets.
        
        Args:
            plot_type: Selected plot type
            x_col: X-axis column (for custom plots)
            y_col: Y-axis column (for custom plots)
            color_col: Column to color points by
            point_size: Size of plot points
            
        Returns:
            Dict: Plotly figure dictionary
        """

        try:
            logger.info(f"Updating plot: type={plot_type}, color={color_col}, size={point_size}")
            
            # ===== SELECT DATAFRAME BASED ON PLOT TYPE =====
            df = _get_dataframe_for_plot_type(plot_type)
            
            if df is None:
                logger.error("Both dataframes are None - cannot create plot")
                return _create_empty_figure("No data available")
            
            logger.info(f"Using dataframe with {len(df)} rows")
            # ==============================================
            
            # Validate inputs
            if not all([plot_type, color_col]):
                logger.warning("Missing required plot parameters")
                return _create_empty_figure("Missing required parameters")
            
            # Determine x and y columns
            x, y = _determine_plot_axes(plot_type, x_col, y_col, plot_types)
            if not x or not y:
                logger.warning(f"Could not determine plot axes for type: {plot_type}")
                return _create_empty_figure("Invalid plot configuration")
            
            # Validate columns exist
            missing_cols = [col for col in [x, y, color_col] if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                return _create_empty_figure(f"Missing columns: {', '.join(missing_cols)}")
            
            # Prepare data for plotting
            plot_df = _prepare_plot_data(df, x, y, color_col)
            if plot_df.empty:
                logger.warning("No data available for plotting after filtering")
                return _create_empty_figure("No data available for plotting")
            
            # Get color parameters
            color_info = next((c for c in available_color_columns if c[0] == color_col), None)
            color_params = get_color_parameters(plot_df, color_col, color_info)
            
            # Create the plot
            fig = _create_scatter_plot(
                plot_df, x, y, color_col, point_size, 
                plot_type, color_params, color_info, plot_types,
                highlight_compound=highlight_compound  # ← ADD THIS
            )
            
            logger.info(f"Plot created successfully with {len(plot_df)} points")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating plot: {str(e)}")
            logger.error(traceback.format_exc())
            return _create_error_figure(str(e))
        
        
    @app.callback(
        Output('plot-search-dropdown', 'options'),
        Input('plot-search-dropdown', 'search_value'),
        State('plot-search-dropdown', 'value'),
        State('plot-type-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_plot_search_options(search_value: str, current_value: str, plot_type: str):
        """
        Update dropdown options for plot compound search.
        Searches in treatment, PP_ID_uM, and moa_compound_uM columns.
        Shows: MOA@conc (PP_ID@conc) when MOA exists, otherwise just PP_ID@conc
        """
        # If user has already selected something, keep that option available
        if current_value and not search_value:
            return [{'label': current_value, 'value': current_value}]
        
        if not search_value or len(search_value) < 2:
            return []
        
        # Get the appropriate dataframe
        dataframe = _get_dataframe_for_plot_type(plot_type)
        if dataframe is None:
            return []
        
        search_lower = search_value.lower()
        options = []
        seen_values = set()  # Track PP_ID_uM values we've added
        
        # 1. Search in 'moa_compound_uM' column FIRST (MOA with concentration, e.g., MAPK@0.1)
        #    This takes priority - shows "MOA@conc (PP_ID@conc)" format
        if 'moa_compound_uM' in dataframe.columns:
            matches = dataframe[
                dataframe['moa_compound_uM'].astype(str).str.lower().str.contains(search_lower, na=False)
            ]
            
            # Get unique moa_compound_uM values with their PP_ID_uM
            for _, row in matches.drop_duplicates(subset=['moa_compound_uM', 'PP_ID_uM']).head(30).iterrows():
                moa_uM = row.get('moa_compound_uM', '')
                pp_id_uM = row.get('PP_ID_uM', row.get('treatment', ''))
                
                if pd.notna(moa_uM) and str(moa_uM).strip() not in ['', 'nan', 'None', 'Unknown']:
                    # Create label like "MAPK@0.1 (CG0031@0.1)"
                    if pp_id_uM and pd.notna(pp_id_uM) and str(pp_id_uM).strip() not in ['', 'nan', 'None']:
                        label = f"{moa_uM} ({pp_id_uM})"
                        value = str(pp_id_uM)
                    else:
                        label = str(moa_uM)
                        value = str(moa_uM)
                    
                    if value not in seen_values:
                        options.append({'label': label, 'value': value})
                        seen_values.add(value)
        
        # 2. Search in 'PP_ID_uM' column (only if not already found via MOA search)
        if 'PP_ID_uM' in dataframe.columns:
            matches = dataframe[
                dataframe['PP_ID_uM'].astype(str).str.lower().str.contains(search_lower, na=False)
            ]['PP_ID_uM'].unique()
            
            for pp_id in matches[:30]:
                if pp_id not in seen_values:
                    options.append({'label': str(pp_id), 'value': str(pp_id)})
                    seen_values.add(pp_id)
        
        # 3. Search in 'treatment' column (only if not already found and different from PP_ID_uM)
        if 'treatment' in dataframe.columns:
            matches = dataframe[
                dataframe['treatment'].astype(str).str.lower().str.contains(search_lower, na=False)
            ]
            
            for _, row in matches.drop_duplicates(subset=['treatment']).head(30).iterrows():
                treatment = row.get('treatment', '')
                pp_id_uM = row.get('PP_ID_uM', '')
                
                # Skip if treatment is the same as PP_ID_uM (avoid duplicates)
                if str(treatment) == str(pp_id_uM):
                    continue
                
                # Skip if this PP_ID_uM was already added via MOA search
                if pp_id_uM and str(pp_id_uM) in seen_values:
                    continue
                
                # Use PP_ID_uM as value if available, otherwise use treatment
                if pp_id_uM and pd.notna(pp_id_uM) and str(pp_id_uM).strip() not in ['', 'nan', 'None']:
                    value = str(pp_id_uM)
                else:
                    value = str(treatment)
                
                if value not in seen_values:
                    options.append({'label': str(treatment), 'value': value})
                    seen_values.add(value)
        
        return sorted(options, key=lambda x: x['label'])[:50]
    
    
    @app.callback(
        Output('plot-search-dropdown', 'value'),
        Input('clear-plot-highlight-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def clear_plot_highlight(n_clicks):
        """Clear the plot highlight search."""
        return None


def _determine_plot_axes(plot_type: str, x_col: str, y_col: str, 
                        plot_types: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    Determine the x and y axes based on plot type selection.
    
    Args:
        plot_type: Selected plot type
        x_col: Custom x column selection
        y_col: Custom y column selection
        plot_types: Available plot type configurations
        
    Returns:
        Tuple of (x_column, y_column)
    """
    if plot_type == 'custom':
        return x_col, y_col
    else:
        plot_info = next((p for p in plot_types if p['value'] == plot_type), None)
        if plot_info:
            return plot_info['x'], plot_info['y']
        return None, None


def _prepare_plot_data(df: pd.DataFrame, x: str, y: str, color_col: str) -> pd.DataFrame:
    """
    Prepare and clean data for plotting.
    
    Args:
        df: Input dataframe
        x: X-axis column name
        y: Y-axis column name
        color_col: Color column name
        
    Returns:
        pd.DataFrame: Cleaned dataframe ready for plotting
    """
    Config = get_config()
    # Create a copy for processing
    plot_df = df.copy()
    
    # Ensure specific columns are treated as categorical
    categorical_columns = ['plate', 'well_column', 'compound_uM', 'well_count']
    for col in categorical_columns:
        if col in plot_df.columns:
            plot_df[col] = plot_df[col].astype(str)
    
    # Prioritize PP columns if available (handle both SPC and CP naming)
    if 'PP_ID' in plot_df.columns:
        if 'compound_name' in plot_df.columns:
            plot_df['compound_name'] = plot_df['PP_ID'].fillna(plot_df['compound_name'])
    elif 'Metadata_PP_ID' in plot_df.columns:
        if 'Metadata_chemical_name' in plot_df.columns:
            plot_df['Metadata_chemical_name'] = plot_df['Metadata_PP_ID'].fillna(plot_df['Metadata_chemical_name'])

    if 'PP_ID_uM' in plot_df.columns:
        if 'treatment' in plot_df.columns:
            plot_df['treatment'] = plot_df['PP_ID_uM'].fillna(plot_df['treatment'])
    elif 'Metadata_PP_ID_uM' in plot_df.columns:
        if 'treatment' in plot_df.columns:
            plot_df['treatment'] = plot_df['Metadata_PP_ID_uM'].fillna(plot_df['treatment'])
    
    # Create ordered categorical types for well-related columns
    if 'well' in plot_df.columns:
        all_wells = Config.get_well_list()
        plot_df['well'] = pd.Categorical(plot_df['well'], categories=all_wells, ordered=True)
    
    if 'well_column' in plot_df.columns:
        # Ensure proper zero-padding
        if pd.api.types.is_numeric_dtype(plot_df['well_column']):
            plot_df['well_column'] = plot_df['well_column'].astype(int).astype(str).str.zfill(2)
        else:
            plot_df['well_column'] = plot_df['well_column'].str.zfill(2)
        
        all_columns = Config.get_column_list()
        plot_df['well_column'] = pd.Categorical(plot_df['well_column'], categories=all_columns, ordered=True)
    
    if 'well_row' in plot_df.columns:
        all_rows = list(Config.WELL_ROWS)
        plot_df['well_row'] = pd.Categorical(plot_df['well_row'], categories=all_rows, ordered=True)
    
    # Remove rows with missing data in ONLY x and y (keep color_col even if NaN)
    required_cols = [x, y]
    plot_df = plot_df.dropna(subset=required_cols)
    
    # Fill missing values in color column with 'Unknown'
    if color_col in plot_df.columns:
        plot_df[color_col] = plot_df[color_col].fillna('Unknown')
        # Also convert to string to ensure consistent handling
        plot_df[color_col] = plot_df[color_col].astype(str)
        # Replace 'nan' strings that might have been created
        plot_df[color_col] = plot_df[color_col].replace(['nan', 'None', ''], 'Unknown')
    
    return plot_df


def _create_scatter_plot(plot_df: pd.DataFrame, x: str, y: str, color_col: str,
                        point_size: int, plot_type: str, color_params: Dict,
                        color_info: Optional[Tuple], plot_types: List[Dict],
                        highlight_compound: str = None) -> go.Figure:  # ← ADD PARAMETER
    """
    Create the scatter plot with proper configuration.
    
    Args:
        plot_df: Prepared dataframe for plotting
        x: X-axis column name
        y: Y-axis column name
        color_col: Color column name
        point_size: Size of plot points
        plot_type: Selected plot type
        color_params: Color parameter configuration
        color_info: Color information tuple
        plot_types: Available plot type configurations
        
    Returns:
        go.Figure: Configured plotly figure
    """
    Config = get_config()
    
    # Determine data type from plot_type
    data_type = 'cp' if (plot_type and ('cp' in plot_type or 'cellprofiler' in plot_type.lower())) else 'spc'
    
    # Build customdata columns for hover and click functionality
    custom_data_cols = _build_customdata_columns(plot_df, data_type)
    
    # Get display name for color column
    display_name = color_info[2] if color_info else color_col.replace('_', ' ').title()
    
    # Determine color column to use (handle renaming)
    color_to_use = color_col
    if color_col not in plot_df.columns and f'{color_col}_display' in plot_df.columns:
        color_to_use = f'{color_col}_display'
    
    # Get axis titles
    plot_info = next((p for p in plot_types if p['value'] == plot_type), None)
    if plot_info:
        x_title = plot_info.get('x_title', x)
        y_title = plot_info.get('y_title', y)
    else:
        x_title = x
        y_title = y
    
    # Create the figure
    if color_params.get('is_continuous', False):
        # Continuous color scale
        fig = px.scatter(
            plot_df,
            x=x,
            y=y,
            color=color_to_use,
            color_continuous_scale=color_params.get('color_scale', 'Viridis'),
            custom_data=custom_data_cols,
            hover_name='treatment' if 'treatment' in plot_df.columns else None
        )
    else:
        # Discrete categories
        fig = px.scatter(
            plot_df,
            x=x,
            y=y,
            color=color_to_use,
            color_discrete_sequence=color_params.get('color_discrete_sequence'),
            custom_data=custom_data_cols,
            hover_name='treatment' if 'treatment' in plot_df.columns else None,
            category_orders=color_params.get('category_orders', {})
        )
    
    # Build hover template
    hover_template = _build_hover_template(custom_data_cols, plot_df, data_type)
    
    fig.update_traces(
        hovertemplate=hover_template,
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    
    # Update marker styling
    fig.update_traces(
        marker=dict(size=point_size, line=dict(width=0)),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            font_color="black",
            bordercolor="rgba(0,0,0,0.2)"
        )
    )
    
    # Calculate axis ranges
    x_min, x_max = plot_df[x].min(), plot_df[x].max()
    y_min, y_max = plot_df[y].min(), plot_df[y].max()
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_padding = x_range * 0.05
    y_padding = y_range * 0.05
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title=display_name,
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        title=dict(x=0.5, font=dict(size=16, color='#2c3e50')),
        xaxis=dict(
            range=[x_min - x_padding, x_max + x_padding],
            autorange=False,
            fixedrange=False
        ),
        yaxis=dict(
            range=[y_min - y_padding, y_max + y_padding],
            autorange=False,
            fixedrange=False,
            scaleanchor='x',
            scaleratio=1,
        ),
    )
    
    # Configure legend
    _configure_legend(fig, plot_df, color_to_use, color_info)
    
   # ===== ADD HIGHLIGHT RING IF COMPOUND IS SELECTED =====
    if highlight_compound:
        # Find the compound in the dataframe
        mask = (
            (plot_df['treatment'].astype(str) == highlight_compound) |
            (plot_df.get('PP_ID_uM', pd.Series()).astype(str) == highlight_compound)
        )
        
        matched_rows = plot_df[mask]
        
        if not matched_rows.empty:
            # Get coordinates of matched points
            x_coords = matched_rows[x].values
            y_coords = matched_rows[y].values
            
            # Calculate radius based on POINT SIZE, not data range
            # Convert point size (in pixels) to data coordinates
            x_range = plot_df[x].max() - plot_df[x].min()
            y_range = plot_df[y].max() - plot_df[y].min()
            
            # Approximate: point_size pixels ≈ this fraction of the plot
            # Assuming ~800px plot width, point_size of 5 = 5/800 = 0.00625 of range
            scale_factor = point_size / 2000  # Adjust divisor to tune size
            radius_x = x_range * scale_factor
            radius_y = y_range * scale_factor
            
            # Add circles as SHAPES (these render on top with layer='above')
            for x_coord, y_coord in zip(x_coords, y_coords):
                # Single circle, just slightly bigger than the point
                fig.add_shape(
                    type="circle",
                    x0=x_coord - radius_x,
                    x1=x_coord + radius_x,
                    y0=y_coord - radius_y,
                    y1=y_coord + radius_y,
                    line=dict(color="black", width=2),
                    fillcolor="rgba(0,0,0,0)",
                    layer="above"
                )
            
            # Add to legend manually with a dummy trace
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color='black', symbol='circle-open', line=dict(width=2)),
                name=f'⚫ {highlight_compound}',
                showlegend=True
            ))
            
            logger.info(f"Highlighted {len(matched_rows)} points for compound: {highlight_compound}")
    # ===== END HIGHLIGHT RING =====
    
    return fig


def _build_customdata_columns(plot_df: pd.DataFrame, data_type: str = 'spc') -> List[str]:
    """
    Build list of columns to include in customdata for hover/click functionality.
    Uses Config.get_hover_columns() for the appropriate data type.
    
    Args:
        plot_df: Dataframe with plot data
        data_type: 'spc' or 'cp'
        
    Returns:
        List: Column names for customdata
    """
    Config = get_config()
    
    # Get the appropriate column list for this data type
    hover_columns = Config.get_hover_columns(data_type)
    
    # Filter to only columns that exist in the dataframe
    custom_data_cols = [col for col in hover_columns if col in plot_df.columns]
    
    logger.info(f" Built customdata for {data_type.upper()} with {len(custom_data_cols)} columns")
    
    return custom_data_cols


def _build_hover_template(custom_data_cols: List[str], plot_df: pd.DataFrame, 
                          data_type: str = 'spc') -> str:
    """
    Build hover template string for plotly figure using DYNAMIC indices.
    Now with colored text for landmark rows!
    """
    Config = get_config()
    
    # Get the display mapping for this data type
    hover_display = Config.get_hover_display(data_type)
    
    # Define colors for different row types
    LANDMARK_COLORS = {
        '1st': '#1565c0',  # Sapphire blue
        '2nd': '#6a1b9a',  # Amethyst purple
        '3rd': '#00838f',  # Teal
    }
    
    hover_items = []
    
    for col_name, display_label in hover_display:
        if col_name in custom_data_cols:
            idx = custom_data_cols.index(col_name)
            
            # Check if this is a landmark row and apply color
            if '1st L' in display_label or '1st Landmark' in display_label:
                # 1st landmark - dark blue
                hover_items.append(
                    f"<span style='color:{LANDMARK_COLORS['1st']}'>{display_label}: %{{customdata[{idx}]}}</span>"
                )
            elif '2nd L' in display_label or '2nd Landmark' in display_label:
                # 2nd landmark - medium blue
                hover_items.append(
                    f"<span style='color:{LANDMARK_COLORS['2nd']}'>{display_label}: %{{customdata[{idx}]}}</span>"
                )
            elif '3rd L' in display_label or '3rd Landmark' in display_label:
                # 3rd landmark - light blue
                hover_items.append(
                    f"<span style='color:{LANDMARK_COLORS['3rd']}'>{display_label}: %{{customdata[{idx}]}}</span>"
                )
            else:
                # Regular row - no color styling
                hover_items.append(f"{display_label}: %{{customdata[{idx}]}}")
    
    hovertemplate = "<b>%{hovertext}</b><br>" + "<br>".join(hover_items) + "<extra></extra>"
    
    return hovertemplate


def _configure_legend(fig: go.Figure, plot_df: pd.DataFrame, 
                     color_col: str, color_info: Optional[Tuple]) -> None:
    """
    Configure legend appearance for different data types.
    
    Args:
        fig: Plotly figure
        plot_df: Plot dataframe
        color_col: Color column name
        color_info: Color information tuple
    """
    # Check if this is a continuous color scale
    is_continuous = color_info[1] if color_info else (
        pd.api.types.is_numeric_dtype(plot_df[color_col]) and 
        plot_df[color_col].nunique() > 12
    )
    
    if not is_continuous:
        n_categories = plot_df[color_col].nunique()
        
        if n_categories > 10:
            # Configure legend for many categories
            fig.update_layout(
                legend=dict(
                    itemsizing='constant',
                    itemwidth=30,
                    font=dict(size=8),
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='rgba(0,0,0,0.2)',
                    borderwidth=1,
                    traceorder='normal'
                )
            )
            
            if n_categories > 30:
                logger.info(f"High cardinality column: {color_col} with {n_categories} categories")


def _create_empty_figure(message: str = "No data to display") -> Dict[str, Any]:
    """
    Create an empty figure with a message.
    
    Args:
        message: Message to display
        
    Returns:
        Dict: Empty figure dictionary
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="#666666")
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='white'
    )
    return fig


def _create_error_figure(error_message: str) -> Dict[str, Any]:
    """
    Create an error figure to display when plot generation fails.
    
    Args:
        error_message: Error message to display
        
    Returns:
        Dict: Error figure dictionary
    """
    fig = go.Figure()
    fig.add_annotation(
        text=f"Error creating plot: {error_message}",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="red")
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='white',
        title="Plot Error"
    )
    return fig