"""
Landmark Analysis callbacks - UPDATED FOR NEW SLIM FORMAT (2025-01-20)

This version works with the new simplified parquet structure:
- test_distances.parquet: One treatment per row, distances as columns
- reference_distances.parquet: Same structure
- landmark_metadata.parquet: Metadata for all landmarks

"""

from dash import Input, Output, State, html, dcc
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
from typing import Optional, Tuple
import logging

try:
    from data.landmark_loader import (
        get_landmark_options,
        load_distances_for_landmark,
        get_landmark_info
    )
    from config_loader import get_config
    from utils.image_utils import find_thumbnail, add_text_to_image
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.landmark_loader import (
        get_landmark_options,
        load_distances_for_landmark,
        get_landmark_info
    )
    from config_loader import get_config
    from utils.image_utils import find_thumbnail, add_text_to_image

logger = logging.getLogger(__name__)


def register_landmark_callbacks(app, df_spc, df_cp):
    """
    Register landmark analysis callbacks for standalone section.
    
    UPDATED (2025-01-21): Now accepts BOTH SPC and CP dataframes!
    
    Args:
        app: Dash application instance
        df_spc: SPC main visualization dataframe (with landmark columns)
        df_cp: CellProfiler main visualization dataframe (with landmark columns)
    """
    
    # NOTE: Hover image callback is in image_callbacks.py and handles both main-plot and landmark-plot-graph
    
    @app.callback(
        [Output('landmark-name-selector', 'options'),
         Output('landmark-name-selector', 'value')],
        [Input('landmark-dataset-selector', 'value'),
         Input('landmark-data-type-selector', 'value')],
        prevent_initial_call=False
    )
    def populate_landmark_options(dataset_type, data_type):
        """
        Load landmark names for dropdown.
        
        UNIFIED APPROACH: Both CP and SPC now use the same slim format!
        """
        logger.info(f"=" * 80)
        logger.info(f"populate_landmark_options called:")
        logger.info(f"  dataset_type={dataset_type}, data_type={data_type}")
        
        if not dataset_type or not data_type:
            logger.warning(f"Missing parameters")
            return [], None
        
        Config = get_config()
        
        try:
            # Use unified function for both CP and SPC
            logger.info(f"Using NEW slim format helper for {data_type.upper()}")
            options, default_value = get_landmark_options(Config, dataset_type, data_type)
            logger.info(f" Loaded {len(options)} {data_type.upper()} landmark options")
            return options, default_value
            
        except Exception as e:
            logger.error(f" ERROR loading landmark data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return [], None
    
    
    @app.callback(
        [Output('landmark-plot-container', 'children'),
         Output('landmark-info-box', 'children')],
        [Input('landmark-name-selector', 'value'),
         Input('landmark-dataset-selector', 'value'),
         Input('landmark-data-type-selector', 'value'),
         Input('landmark-color-selector', 'value'),
         Input('landmark-size-slider', 'value'),
         Input('landmark-plot-search-dropdown', 'value')],
        prevent_initial_call=True
    )
    def update_landmark_plot(landmark_value, dataset_type, data_type, color_column, point_size,
                             highlight_compound=None):
        """
        Load selected landmark's data and create plot + info box.
        
        UNIFIED APPROACH: Both CP and SPC now use the same slim format!
        """
        logger.info(f"=" * 80)
        logger.info(f"update_landmark_plot called:")
        logger.info(f"  landmark_value={landmark_value}")
        logger.info(f"  dataset_type={dataset_type}")
        logger.info(f"  data_type={data_type}")
        
        if not landmark_value or not dataset_type or not data_type:
            logger.warning("Missing required parameters")
            return (
                html.Div("Please select a dataset type, data type, and landmark", 
                        style={'padding': '20px', 'textAlign': 'center'}),
                html.Div()
            )
        
        Config = get_config()
        
        try:
            # Parse landmark value (just the treatment name for both CP and SPC)
            landmark_treatment = landmark_value
            
            logger.info(f"Loading {data_type.upper()} data for landmark: {landmark_treatment}")
            
            # Determine which viz dataframe to use
            viz_df = df_cp if data_type == 'cp' else df_spc
            
            # Load the distance data (only the columns we need) - UNIFIED FUNCTION
            # NOW PASSING viz_df to get landmark columns and SMILES!
            df_plot = load_distances_for_landmark(
                Config, dataset_type, data_type, landmark_treatment, viz_df=viz_df
            )
            
            if df_plot is None or df_plot.empty:
                return (
                    html.Div("Failed to load landmark data", 
                            style={'padding': '20px', 'color': 'red'}),
                    html.Div()
                )
            
            logger.info(f" Loaded {len(df_plot)} treatments for plotting")
            
            # Get landmark metadata for display - UNIFIED FUNCTION
            landmark_info = get_landmark_info(Config, data_type, landmark_treatment)
            
            if landmark_info:
                # Column names differ between CP and SPC
                if data_type == 'cp':
                    target = landmark_info.get('Metadata_annotated_target_first', 'Unknown')
                    pp_id = landmark_info.get('Metadata_PP_ID_uM', 'Unknown')
                else:  # spc
                    target = landmark_info.get('moa_first', 'Unknown')
                    pp_id = landmark_info.get('PP_ID_uM', 'Unknown')
                
                landmark_label = f"{target} ({pp_id})"
            else:
                landmark_label = landmark_treatment
            
            # Column names for plotting (same for both)
            dmso_col = 'query_dmso_distance'
            distance_col = 'landmark_distance'  # We renamed it in the loader
            
            # Create the plot
            fig = create_landmark_plot(
                df_plot, 
                landmark_label, 
                distance_col, 
                color_column, 
                point_size,
                dmso_distance_col=dmso_col,
                data_type=data_type,
                highlight_compound=highlight_compound
            )
            
            # Create info box
            info_box = create_landmark_info_box(landmark_info, df_plot, landmark_label, data_type)
            
            return dcc.Graph(
                id='landmark-plot-graph', 
                figure=fig, 
                style={'height': '700px'},
                clear_on_unhover=True,  # ← ADD THIS LINE
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    'dragmode': 'pan'
                }
            ), info_box
            
        except Exception as e:
            logger.error(f" ERROR in update_landmark_plot: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return (
                html.Div(f"Error: {str(e)}", 
                        style={'padding': '20px', 'color': 'red'}),
                html.Div()
            )

    # ===== NEW CALLBACKS FOR LANDMARK PLOT SEARCH =====
    @app.callback(
        Output('landmark-plot-search-dropdown', 'options'),
        Input('landmark-plot-search-dropdown', 'search_value'),
        [State('landmark-plot-search-dropdown', 'value'),
         State('landmark-data-type-selector', 'value')],
        prevent_initial_call=True
    )
    def update_landmark_plot_search_options(search_value, current_value, data_type):
        """Update dropdown options for landmark plot compound search."""
        if current_value and not search_value:
            return [{'label': current_value, 'value': current_value}]
        
        if not search_value or len(search_value) < 2:
            return []
        
        # Use appropriate dataframe
        dataframe = df_cp if data_type == 'cp' else df_spc
        if dataframe is None:
            return []
        
        search_lower = search_value.lower()
        options = []
        seen_values = set()
        
        # Search in moa_compound_uM, PP_ID_uM, and treatment columns
        for col in ['moa_compound_uM', 'PP_ID_uM', 'treatment']:
            if col in dataframe.columns:
                matches = dataframe[
                    dataframe[col].astype(str).str.lower().str.contains(search_lower, na=False)
                ][col].unique()
                
                for val in matches[:30]:
                    if val not in seen_values and pd.notna(val):
                        options.append({'label': str(val), 'value': str(val)})
                        seen_values.add(val)
        
        return sorted(options, key=lambda x: x['label'])[:50]
    
    
    @app.callback(
        Output('landmark-plot-search-dropdown', 'value'),
        Input('clear-landmark-plot-highlight-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def clear_landmark_plot_highlight(n_clicks):
        """Clear the landmark plot highlight search."""
        return None
    

def create_landmark_info_box(landmark_info: dict, df_plot: pd.DataFrame, 
                             landmark_label: str, data_type: str) -> html.Div:
    """
    Create the information box showing landmark metadata.
    
    Args:
        landmark_info: Dictionary of landmark metadata
        df_plot: DataFrame being plotted (for stats)
        landmark_label: Display label for the landmark
        data_type: 'cp' or 'spc' (for column name handling)
    
    Returns:
        html.Div with formatted info
    """
    info_items = [
        html.H4(
            f"Landmark: {landmark_label}",
            style={'color': '#2c3e50', 'marginBottom': '15px', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}
        )
    ]
    
    # Add landmark metadata if available
    if landmark_info:
        # Define fields based on data type
        if data_type == 'cp':
            metadata_fields = {
                'Metadata_PP_ID_uM': 'PP ID',
                'Metadata_annotated_target': 'Target',
                'Metadata_annotated_target_description_truncated_10': 'Target Description',
                'Metadata_perturbation_name': 'Perturbation',
                'Metadata_chemical_name': 'Chemical Name',
                'Metadata_chemical_description': 'Description',
                'Metadata_compound_type': 'Compound Type',
                'Metadata_manual_annotation': 'Annotation',
                'Metadata_library': 'Library'
            }
        else:  # spc
            metadata_fields = {
                'PP_ID_uM': 'PP ID',
                'annotated_target': 'Target',
                'annotated_target_description_truncated_10': 'Target Description',
                'perturbation_name': 'Perturbation',
                'chemical_name': 'Chemical Name',
                'chemical_description': 'Description',
                'manual_annotation': 'Annotation',
                'library': 'Library'
            }
        
        for field, display_name in metadata_fields.items():
            if field in landmark_info:
                value = landmark_info[field]
                if pd.notna(value) and str(value).strip() not in ['', 'nan', 'None']:
                    info_items.append(
                        html.Div([
                            html.Strong(f"{display_name}: ", style={'color': '#555'}),
                            html.Span(str(value), style={'color': '#333'})
                        ], style={'marginBottom': '8px', 'fontSize': '14px'})
                    )
    
    # Add plot statistics
    if not df_plot.empty:
        info_items.append(
            html.Hr(style={'margin': '15px 0'})
        )
        info_items.append(
            html.H5("Plot Statistics", style={'color': '#2c3e50', 'marginBottom': '10px'})
        )
        
        n_points = len(df_plot)
        info_items.append(
            html.Div([
                html.Strong("Number of Perturbations: ", style={'color': '#555'}),
                html.Span(f"{n_points:,}", style={'color': '#333'})
            ], style={'marginBottom': '8px', 'fontSize': '14px'})
        )
        
        if 'landmark_distance' in df_plot.columns:
            min_dist = df_plot['landmark_distance'].min()
            max_dist = df_plot['landmark_distance'].max()
            mean_dist = df_plot['landmark_distance'].mean()
            
            info_items.extend([
                html.Div([
                    html.Strong("Min distance: ", style={'color': '#555'}),
                    html.Span(f"{min_dist:.6f}", style={'color': '#333'})
                ], style={'marginBottom': '8px', 'fontSize': '14px'}),
                html.Div([
                    html.Strong("Max distance: ", style={'color': '#555'}),
                    html.Span(f"{max_dist:.6f}", style={'color': '#333'})
                ], style={'marginBottom': '8px', 'fontSize': '14px'}),
                html.Div([
                    html.Strong("Mean distance: ", style={'color': '#555'}),
                    html.Span(f"{mean_dist:.6f}", style={'color': '#333'})
                ], style={'marginBottom': '8px', 'fontSize': '14px'})
            ])
    
    if len(info_items) == 1:
        info_items.append(
            html.Div("No information available", 
                    style={'color': '#999', 'fontStyle': 'italic'})
        )
    
    return html.Div(
        info_items,
        style={
            'padding': '15px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '5px',
            'border': '1px solid #ddd',
            'maxHeight': '700px',
            'overflowY': 'auto'
        }
    )


def create_landmark_plot(df: pd.DataFrame, landmark_label: str, 
                        distance_col: str, color_col: str, 
                        point_size: int = 5,
                        dmso_distance_col: str = 'query_dmso_distance',
                        data_type: str = 'cp',
                        highlight_compound: str = None) -> go.Figure:
    """
    Create landmark distance scatter plot.
    
    Args:
        df: DataFrame with distance data
        landmark_label: Display label for landmark
        distance_col: Column name for landmark distance (y-axis)
        color_col: Column to color by
        point_size: Size of points
        dmso_distance_col: Column name for DMSO distance (x-axis)
    
    Returns:
        Plotly Figure object
    """
    
    logger.info(f"=" * 80)
    logger.info(f"create_landmark_plot called:")
    logger.info(f"  landmark_label: {landmark_label}")
    logger.info(f"  DataFrame shape: {df.shape}")
    
    if df is None or df.empty:
        logger.error("  DataFrame is None or empty!")
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="red")
        )
        return fig
    
    # Ensure color column exists
    if color_col not in df.columns:
        logger.warning(f"Color column '{color_col}' not found, using 'library'")
        color_col = 'library' if 'library' in df.columns else 'treatment'
    
    df = df.copy()
    df[color_col] = df[color_col].fillna('Unknown')
    
    # Get x and y data
    if dmso_distance_col not in df.columns:
        logger.warning(f"DMSO distance column '{dmso_distance_col}' not found")
        x_data = pd.Series([0] * len(df))
    else:
        x_data = df[dmso_distance_col]
    
    y_data = df[distance_col]
    
    # Log statistics
    logger.info(f"  X-axis (DMSO) - Min: {x_data.min():.6f}, Max: {x_data.max():.6f}")
    logger.info(f"  Y-axis (Landmark) - Min: {y_data.min():.6f}, Max: {y_data.max():.6f}")
    
    # =========================================================================
    # SIMPLIFIED CUSTOMDATA BUILDING - Uses Config.get_hover_columns()
    # =========================================================================
    Config = get_config()
    
    # Get the appropriate column list for this data type (SPC or CP)
    hover_columns = Config.get_hover_columns(data_type)
    
    # Filter to only columns that exist in the dataframe
    customdata_cols = [col for col in hover_columns if col in df.columns]
    
    logger.info(f"📦 Building customdata for {data_type.upper()} landmark plot:")
    logger.info(f"   Requested columns: {len(hover_columns)}")
    logger.info(f"   Available columns: {len(customdata_cols)}")
    
    # Ensure we have at least treatment column
    if 'treatment' not in customdata_cols and 'treatment' in df.columns:
        customdata_cols.insert(0, 'treatment')
    
    customdata = df[customdata_cols].values
    
    # Get plate/well column names for this data type
    plate_col = Config.get_plate_column(data_type)
    well_col = Config.get_well_column(data_type)
    
    # Build hover template using Config.get_hover_display() for correct labels
    hover_display = Config.get_hover_display(data_type)
    
    # Define colors for landmark rows
    LANDMARK_COLORS = {
        '1st': '#1565c0',  # Sapphire blue
        '2nd': '#6a1b9a',  # Amethyst purple  
        '3rd': '#00838f',  # Teal
    }
    
    # Add colours to plotly hover

    hover_template = '<b>%{text}</b><br>'
    hover_template += f'DMSO Distance: %{{x:.4f}}<br>'
    hover_template += f'{landmark_label} Distance: %{{y:.4f}}<br>'
    
    # Add fields dynamically with correct indices and colors for landmarks
    for col_name, display_label in hover_display:
        if col_name in customdata_cols:
            idx = customdata_cols.index(col_name)
            
            # Check if this is a landmark row and apply color
            if '1st L' in display_label or '1st Landmark' in display_label:
                hover_template += f"<span style='color:{LANDMARK_COLORS['1st']}'>{display_label}: %{{customdata[{idx}]}}</span><br>"
            elif '2nd L' in display_label or '2nd Landmark' in display_label:
                hover_template += f"<span style='color:{LANDMARK_COLORS['2nd']}'>{display_label}: %{{customdata[{idx}]}}</span><br>"
            elif '3rd L' in display_label or '3rd Landmark' in display_label:
                hover_template += f"<span style='color:{LANDMARK_COLORS['3rd']}'>{display_label}: %{{customdata[{idx}]}}</span><br>"
            else:
                hover_template += f'{display_label}: %{{customdata[{idx}]}}<br>'
    
    hover_template += '<extra></extra>'
    
    logger.debug(f"Built landmark hover template with {len([c for c in hover_display if c[0] in customdata_cols])} items")
    
    # Create figure
    fig = go.Figure()
    
    # Determine if continuous or discrete coloring
    is_numeric = pd.api.types.is_numeric_dtype(df[color_col])
    n_unique = df[color_col].nunique()
    
    # Prepare text labels (treatment names)
    text_labels = df['treatment'].astype(str)
    
    if is_numeric and n_unique > 12:
        # Continuous coloring
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            marker=dict(
                size=point_size,
                color=df[color_col],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_col.replace('_', ' ').title())
            ),
            text=text_labels,
            customdata=customdata,
            hovertemplate=hover_template,
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial",
                font_color="black",
                bordercolor="rgba(0,0,0,0.2)"
            ),
            name='',
            showlegend=False
        ))
    else:
        # Discrete coloring
        color_series = df[color_col]
        if isinstance(color_series, pd.DataFrame):
            color_series = color_series.iloc[:, 0]
        
        for category in sorted(color_series.dropna().unique()):
            mask = color_series == category
            
            fig.add_trace(go.Scatter(
                x=x_data[mask],
                y=y_data[mask],
                mode='markers',
                marker=dict(size=point_size),
                name=str(category),
                text=text_labels[mask],
                customdata=customdata[mask],
                hovertemplate=hover_template,
                    hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial",
                    font_color="black",
                    bordercolor="rgba(0,0,0,0.2)"
                ),
                showlegend=True
            ))
    
    # Update layout
    fig.update_layout(
        title=f'Landmark Distance Analysis: {landmark_label}',
        xaxis_title='Distance from DMSO',
        yaxis_title=f'Distance to {landmark_label}',
        hovermode='closest',
        height=700,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#ddd',
            borderwidth=1,
            font=dict(size=9)
        ),
        margin=dict(r=180)
    )
    
    logger.info(f" Created plot for {landmark_label} with {len(df)} points")

    # ===== ADD HIGHLIGHT RING IF COMPOUND IS SELECTED =====
    if highlight_compound:
        mask = (
            (df['treatment'].astype(str) == highlight_compound) |
            (df.get('PP_ID_uM', pd.Series()).astype(str) == highlight_compound)
        )
        matched_rows = df[mask]
        
        if not matched_rows.empty:
            x_coords = matched_rows[dmso_distance_col].values if dmso_distance_col in df.columns else []
            y_coords = matched_rows[distance_col].values
            
            # Calculate radius based on data range
            x_range = x_data.max() - x_data.min()
            y_range = y_data.max() - y_data.min()
            scale_factor = point_size / 2000
            radius_x = x_range * scale_factor
            radius_y = y_range * scale_factor
            
            for x_coord, y_coord in zip(x_coords, y_coords):
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
            
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color='black', symbol='circle-open', line=dict(width=2)),
                name=f'⚫ {highlight_compound}',
                showlegend=True
            ))
            
            logger.info(f"Highlighted {len(matched_rows)} points for compound: {highlight_compound}")
    # ===== END HIGHLIGHT RING =====
    
    return fig  # existing line
