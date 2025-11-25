"""
Image callbacks module for the Phenotype Clustering Interactive Visualization App.

UPDATED (2025-01-21): Now supports BOTH SPC and CellProfiler datasets!

This module contains all callback functions related to image display, hover effects,
and click interactions. It handles the fixed position hover images, tooltip display,
and detailed image views when users click on plot points.

Key Functions:
- register_image_callbacks(): Register all image-related callbacks with the app
- update_fixed_image(): Handle hover image display in top-right corner
- display_hover_data(): Show tooltip information on hover
- display_image(): Show detailed image view when clicking points

The module provides seamless integration between plot interactions and microscopy
image display, with support for text overlays and different scaling modes.
"""

import random
import pandas as pd
from dash import Input, Output, State, no_update, html
from dash.exceptions import PreventUpdate
import traceback
import logging
from typing import List, Tuple, Any, Optional

try:
    from ..config_loader import get_config
    from ..utils.image_utils import find_thumbnail, add_text_to_image, extract_site_from_path
    from ..utils.smiles_utils import smiles_to_image_base64, is_valid_smiles
except ImportError:
    from config_loader import get_config
    from utils.image_utils import find_thumbnail, add_text_to_image, extract_site_from_path
    from utils.smiles_utils import smiles_to_image_base64, is_valid_smiles

# Set up logging
logger = logging.getLogger(__name__)

def _safe_get_column(hover_info, column_name, fallback='Unknown'):
    """Get column value with proper fallback handling"""
    value = hover_info.get(column_name, fallback)
    if not value or pd.isna(value) or str(value).strip() in ['Unknown', 'nan', '']:
        return fallback
    return str(value)

def register_image_callbacks(app, df_spc: pd.DataFrame, df_cp: pd.DataFrame, 
                            hover_data_cols: List[str]) -> None:
    """
    Register all image-related callbacks with the Dash app.
    
    UPDATED (2025-01-21): Now accepts BOTH SPC and CP dataframes!
    
    Args:
        app: Dash application instance
        df_spc: SPC dataframe
        df_cp: CellProfiler dataframe
        hover_data_cols: List of columns to include in hover data
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
            logger.debug(f"Plot type '{plot_type}' → using CP dataframe")
            return df_cp if df_cp is not None else df_spc
        else:
            logger.debug(f"Plot type '{plot_type}' → using SPC dataframe")
            return df_spc if df_spc is not None else df_cp

    def _create_detailed_image_display(hover_info: dict, img_src: str) -> html.Div:
        """
        Create detailed image display for clicked points with enhanced landmark information.
        """
        print(f"DEBUG: Available hover_info keys: {list(hover_info.keys())}")

        # Extract information with proper fallbacks (handle both SPC and CP columns)
        compound = hover_info.get('PP_ID', hover_info.get('Metadata_PP_ID', ''))
        if pd.isna(compound) or str(compound).strip() == '' or str(compound) == 'None':
            compound = hover_info.get('compound_name', hover_info.get('Metadata_chemical_name', 'Unknown'))
        
        treatment = hover_info.get('PP_ID_uM', hover_info.get('Metadata_PP_ID_uM', ''))
        if pd.isna(treatment) or str(treatment).strip() == '' or str(treatment) == 'None':
            treatment = hover_info.get('treatment', 'Unknown')
        
        concentration = hover_info.get('compound_uM', hover_info.get('Metadata_compound_uM', 'Unknown'))
        
        # Handle plate/well - different column names for SPC vs CP
        plate = hover_info.get('plate', hover_info.get('Metadata_plate_barcode', 'Unknown'))
        well = hover_info.get('well', hover_info.get('Metadata_well', 'Unknown'))
        
        library = hover_info.get('library', hover_info.get('Metadata_library', 'Unknown'))
        moa = _safe_get_column(hover_info, 'moa_truncated_10', 'Unknown')
        target_description = _safe_get_column(hover_info, 'target_description_truncated_10', 
                                             _safe_get_column(hover_info, 'Metadata_annotated_target_description_truncated_10', 'Unknown'))
        
        # Get validity status
        validity = hover_info.get('valid_for_phenotypic_makeup', None)
        validity_symbol = '✓' if validity == True else '✗'
        validity_color = '#2ecc71' if validity == True else '#e74c3c'

        # Get chemical description and SMILES (handle both naming conventions)
        chemical_description = _safe_get_column(hover_info, 'chemical_description', 
                                               _safe_get_column(hover_info, 'Metadata_chemical_description', 'Unknown'))
        smiles = hover_info.get('SMILES', hover_info.get('Metadata_SMILES', ''))

        # Format concentration
        concentration_text = f"{concentration} µM" if concentration not in [None, 'Unknown'] else "Unknown"
        
        # Create metrics display with visual indicators
        metrics_display = []
        metric_info = {
            'cosine_distance_from_dmso': {'name': 'Cosine Distance from DMSO', 'format': '.4f'},
            'mad_cosine': {'name': 'MAD Cosine', 'format': '.4f'},
            'var_cosine': {'name': 'Variance Cosine', 'format': '.4f'},
            'std_cosine': {'name': 'Std Dev Cosine', 'format': '.4f'},
            'median_distance': {'name': 'Median Distance', 'format': '.4f'},
            'closest_landmark_distance': {'name': 'Closest Landmark Distance', 'format': '.4f'}
        }

        # Debug
        print(f"DEBUG: Looking for metrics: {list(metric_info.keys())}")
        
        # NOTE: For metrics, we need access to the dataframe - passed via closure
        # Get the current dataframe based on data type (this will be set in the callback)
        dataframe = df_spc if df_spc is not None else df_cp  # Default fallback
        
        for key, info in metric_info.items():
            if key in hover_info and hover_info[key] is not None:
                value = hover_info[key]
                if isinstance(value, (int, float)):
                    # Get min and max from the dataframe for this metric
                    if key in dataframe.columns:
                        min_val = dataframe[key].min()
                        max_val = dataframe[key].max()
                        
                        # Calculate position percentage
                        if max_val > min_val:
                            position_pct = ((value - min_val) / (max_val - min_val)) * 100
                        else:
                            position_pct = 50  # Default to middle if no range
                        
                        # Create metric display with visual range indicator
                        metrics_display.append(
                            html.Div([
                                html.P(f"{info['name']}: {value:{info['format']}}", 
                                    style={'marginBottom': '2px', 'fontSize': '12px'}),
                                html.Div([
                                    # Background bar
                                    html.Div(style={
                                        'width': '100%',
                                        'height': '10px',
                                        'backgroundColor': '#e0e0e0',
                                        'borderRadius': '5px',
                                        'position': 'relative',
                                        'marginBottom': '8px'
                                    }),
                                    # Position indicator
                                    html.Div(style={
                                        'position': 'absolute',
                                        'top': '0',
                                        'left': f'{position_pct}%',
                                        'width': '3px',
                                        'height': '10px',
                                        'backgroundColor': '#3498db',
                                        'borderRadius': '2px',
                                        'transform': 'translateX(-50%)'
                                    }),
                                    # Min/Max labels
                                    html.Div([
                                        html.Span(f"{min_val:.2f}", style={
                                            'position': 'absolute',
                                            'left': '0',
                                            'top': '12px',
                                            'fontSize': '10px',
                                            'color': '#666'
                                        }),
                                        html.Span(f"{max_val:.2f}", style={
                                            'position': 'absolute',
                                            'right': '0',
                                            'top': '12px',
                                            'fontSize': '10px',
                                            'color': '#666'
                                        })
                                    ])
                                ], style={'position': 'relative', 'width': '100%', 'marginBottom': '20px'})
                            ])
                        )
                    else:
                        # Fallback to simple text if column not in dataframe
                        metrics_display.append(html.P(f"{info['name']}: {value:{info['format']}}"))
                else:
                    metrics_display.append(html.P(f"{info['name']}: {value}"))
        
        # CREATE COMPREHENSIVE LANDMARK INFORMATION SECTION
        landmark_info = []
        
        def get_landmark_info(prefix):
            # Handle both SPC and CP landmark column naming
            landmark = _safe_get_column(hover_info, f'{prefix}_landmark_moa_first', 
                                       _safe_get_column(hover_info, f'{prefix}_landmark_Metadata_annotated_target_first', None))
            
            pp_id_um = _safe_get_column(hover_info, f'{prefix}_landmark_PP_ID_uM',
                                       _safe_get_column(hover_info, f'{prefix}_landmark_Metadata_PP_ID_uM', None))
            
            target_desc = _safe_get_column(hover_info, f'{prefix}_landmark_annotated_target_description_truncated_10',
                                          _safe_get_column(hover_info, f'{prefix}_landmark_Metadata_annotated_target_description_truncated_10', None))
            
            broad_annotation = _safe_get_column(hover_info, f'{prefix}_landmark_manual_annotation',
                                               _safe_get_column(hover_info, f'{prefix}_landmark_Metadata_manual_annotation', None))
            distance = hover_info.get(f'{prefix}_landmark_distance', None)
            
            # Only show if we have actual landmark data
            if landmark and landmark != 'Unknown':
                info_parts = [f"Landmark: {landmark}"]
                
                if pp_id_um and pp_id_um != 'Unknown':
                    info_parts.append(f"PP_ID: {pp_id_um}")
                if target_desc and target_desc != 'Unknown':
                    info_parts.append(f"Target: {target_desc}")
                if broad_annotation and broad_annotation != 'Unknown':
                    # Truncate broad annotation if it's too long
                    if len(str(broad_annotation)) > 100:
                        broad_annotation = str(broad_annotation)[:100] + "..."
                    info_parts.append(f"Broad Annotation: {broad_annotation}")
                if distance is not None and not pd.isna(distance):
                    try:
                        distance_float = float(distance)
                        info_parts.append(f"Distance: {distance_float:.4f}")
                    except (ValueError, TypeError):
                        info_parts.append(f"Distance: {distance}")
                        
                return info_parts
            return None

        # Get information for all three closest landmarks
        for i, prefix in enumerate(['closest', 'second_closest', 'third_closest'], 1):
            landmark_data = get_landmark_info(prefix)
            if landmark_data:
                # Create a styled container for each landmark
                landmark_info.append(
                    html.Div([
                        html.P(f"{i}{'st' if i==1 else 'nd' if i==2 else 'rd'} Closest:", 
                              style={'fontWeight': 'bold', 'marginBottom': '5px', 'color': '#2c3e50'}),
                        html.Div([
                            html.P(part, style={'marginBottom': '2px', 'fontSize': '11px'}) 
                            for part in landmark_data
                        ], style={'marginLeft': '10px', 'marginBottom': '10px'})
                    ])
                )
        
        # Also check for general landmark_label if no specific landmark data
        if not landmark_info:
            general_landmark = _safe_get_column(hover_info, 'landmark_label', None)
            if general_landmark and general_landmark != 'Unknown':
                landmark_info.append(
                    html.P(f"Landmark Status: {general_landmark}", 
                          style={'fontSize': '12px', 'color': '#34495e'})
                )
        
        # Create the main layout
        left_column_content = [
            html.H4("Basic Information", style={'color': '#34495e', 'marginBottom': '10px'}),
            html.P(f"Treatment: {treatment}"),
            html.P(f"Plate: {plate}"),
            html.P(f"Well: {well}"),
            html.P(f"Concentration: {concentration_text}"),
            html.P(f"Library: {library}"),
            html.P(f"MOA: {moa}"),
            html.P(f"Target Description: {target_description}"),
            html.P(f"Chemical Description: {chemical_description}"),
            html.P([
                "First closest landmark within distance threshold: ",
                html.Span(validity_symbol, style={'color': validity_color, 'fontWeight': 'bold', 'fontSize': '14px'})
            ], style={'fontSize': '11px', 'marginBottom': '2px'}),
            
            html.Hr(style={'margin': '15px 0'}),
            
            html.H4("Metrics", style={'color': '#34495e', 'marginBottom': '10px'}),
            *metrics_display,
            
            html.Hr(style={'margin': '15px 0'}),
            
            # ENHANCED LANDMARK SECTION
            html.H4("Landmark Information", style={'color': '#34495e', 'marginBottom': '10px'}),
        ]
        
        # Add landmark information to left column content
        if landmark_info:
            left_column_content.extend(landmark_info)
        else:
            left_column_content.append(
                html.P("No landmark information available", 
                      style={'fontSize': '11px', 'color': '#666'})
            )
        
        # Add SMILES structure to left column if available
        Config = get_config()
        if Config.SHOW_CHEMICAL_STRUCTURES and smiles and str(smiles).strip() not in ['Unknown', 'nan', '', 'None']:
            structure_img = smiles_to_image_base64(smiles, width=400, height=400)
            
            if structure_img:
                left_column_content.extend([
                    html.Hr(style={'margin': '15px 0'}),
                    html.H4("Chemical Structure", style={'color': '#34495e', 'marginBottom': '10px'}),
                    html.Div([
                        html.Img(
                            src=structure_img,
                            style={
                                'maxWidth': '350px',
                                'maxHeight': '350px',
                                'border': '1px solid #ddd',
                                'borderRadius': '4px',
                                'backgroundColor': 'white'
                            }
                        ),
                        html.P(f"SMILES: {smiles}", 
                            style={
                                'fontSize': '10px', 
                                'color': '#6c757d', 
                                'marginTop': '8px',
                                'wordBreak': 'break-all',
                                'lineHeight': '1.2'
                            })
                    ], style={'textAlign': 'left'})
                ])
            else:
                # Show SMILES text if structure generation failed
                left_column_content.extend([
                    html.Hr(style={'margin': '15px 0'}),
                    html.H4("SMILES", style={'color': '#34495e', 'marginBottom': '10px'}),
                    html.P(smiles, 
                        style={
                            'fontSize': '10px', 
                            'color': '#6c757d', 
                            'wordBreak': 'break-all',
                            'lineHeight': '1.2',
                            'backgroundColor': '#f8f9fa',
                            'padding': '8px',
                            'borderRadius': '4px'
                        })
                ])

        # Add CHEMOPROTEOMICS section to click display
        has_chemprot = any(
            hover_info.get(col) and str(hover_info.get(col)).strip() not in ['Unknown', 'nan', '', 'None', 'null']
            for col in ['cell_line', 'experiment_type', 'gene_site_1']
        )
        
        if has_chemprot:
            left_column_content.extend([
                html.Hr(style={'margin': '15px 0'}),
                html.H4("Chemoproteomics Data", style={'color': '#34495e', 'marginBottom': '10px'}),
            ])
            
            # Cell line
            cell_line = _safe_get_column(hover_info, 'cell_line', None)
            if cell_line and cell_line != 'Unknown':
                left_column_content.append(
                    html.P(f"Cell Line: {cell_line}", 
                        style={'fontSize': '12px', 'marginBottom': '5px', 'fontWeight': 'bold'})
                )
            
            # Experiment type
            exp_type = _safe_get_column(hover_info, 'experiment_type', None)
            if exp_type and exp_type != 'Unknown':
                left_column_content.append(
                    html.P(f"Experiment Type: {exp_type}", 
                        style={'fontSize': '12px', 'marginBottom': '5px'})
                )
            
            # Compound concentration
            chemprot_conc = hover_info.get('compound_concentration_uM', None)
            if chemprot_conc and not pd.isna(chemprot_conc):
                left_column_content.append(
                    html.P(f"Chemoproteomics Concentration: {chemprot_conc} µM", 
                        style={'fontSize': '12px', 'marginBottom': '5px'})
                )
            
            # Gene sites (protein targets)
            gene_sites = []
            for i in range(1, 6):
                gene_site = _safe_get_column(hover_info, f'gene_site_{i}', None)
                if gene_site and gene_site != 'Unknown':
                    gene_sites.append(gene_site)
            
            if gene_sites:
                left_column_content.append(
                    html.P("Protein Targets:", 
                        style={'fontSize': '12px', 'fontWeight': 'bold', 'marginTop': '8px', 'marginBottom': '5px'})
                )
                
                for site in gene_sites:
                    left_column_content.append(
                        html.P(f"• {site}", 
                            style={'fontSize': '11px', 'marginLeft': '15px', 'marginBottom': '3px'})
                    )
        
        # Add GENE DESCRIPTION section to click display
        gene_description = hover_info.get('gene_description', None)
        moa_first = hover_info.get('moa_first', None)
        
        if gene_description and moa_first and str(gene_description).strip() not in ['Unknown', 'nan', '', 'null']:
            left_column_content.extend([
                html.Hr(style={'margin': '15px 0'}),
                html.H4(f"Gene: {moa_first}", style={'color': '#34495e', 'marginBottom': '10px'}),
                html.Div([
                    html.P(str(gene_description), 
                        style={
                            'fontSize': '11px', 
                            'lineHeight': '1.4', 
                            'color': '#495057',
                            'textAlign': 'left',
                            'wordWrap': 'break-word',
                            'whiteSpace': 'normal'
                        })
                ], style={
                    'backgroundColor': '#e8f4fd',
                    'padding': '10px',
                    'borderRadius': '4px',
                    'border': '1px solid #bee5eb',
                    'maxHeight': '300px',
                    'overflowY': 'auto'
                })
            ])
        
        return html.Div([
            html.H3(f"Compound: {compound}", style={'color': '#2c3e50', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
            html.Div([
                # Left column with metadata and SMILES
                html.Div(left_column_content, style={
                    'width': '35%', 
                    'display': 'inline-block', 
                    'verticalAlign': 'top',
                    'paddingRight': '20px'
                }),
                
                # Right column with image
                html.Div([
                    html.H4("Image", style={
                        'color': '#34495e', 
                        'marginBottom': '10px',
                        'textAlign': 'left'
                    }),
                    html.Img(
                        src=img_src, 
                        style={
                            'maxWidth': '100%', 
                            'maxHeight': '700px',
                            'border': '2px solid #bdc3c7',
                            'borderRadius': '8px',
                            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                            'display': 'block'
                        }
                    )
                ], style={
                    'width': '65%',
                    'display': 'inline-block',
                    'textAlign': 'left'
                })
            ])
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '20px',
            'borderRadius': '8px',
            'border': '1px solid #dee2e6'
            
        })


    @app.callback(
    [Output('fixed-image-container', 'children'),
     Output('fixed-image-container', 'style'),
     Output('last-hover-time', 'data')],  # ← ADD THIS
    [Input('main-plot', 'hoverData'),
     Input('landmark-plot-graph', 'hoverData'),
     Input('scaling-mode', 'value'),
     Input('add-label-checkbox', 'value'),
     Input('label-type-radio', 'value'),
     Input('plot-type-dropdown', 'value')]  # ← ADDED plot type
     )
    
    def update_fixed_image(main_hover_data: dict, landmark_hover_data: dict,
                          scaling_mode: str, add_label_checkbox: List[str], 
                          label_type: str, plot_type: str) -> Tuple[List, dict]:
        """
        Update the fixed position image container on plot hover.
        
        This callback displays microscopy images in a fixed position container
        when users hover over plot points, providing immediate visual feedback.
        NOW HANDLES BOTH main plot and landmark plot hover events.
        
        Args:
            main_hover_data: Plotly hover data from main plot
            landmark_hover_data: Plotly hover data from landmark plot
            scaling_mode: Image scaling mode ('fixed' or 'auto')
            add_label_checkbox: List containing 'add_label' if enabled
            label_type: Type of label to add ('treatment' or 'moa_compound_uM')
            plot_type: Current plot type (for dataframe selection)
            
        Returns:
            Tuple of (children, style) for the fixed image container
        """
        try:
            # ===== SELECT DATAFRAME BASED ON PLOT TYPE =====
            dataframe = _get_dataframe_for_plot_type(plot_type)
            
            if dataframe is None:
                base_style = {'display': 'none'}
                return [], base_style, 0
            # ==============================================
            
            Config = get_config()
            # FORCE the container width
            base_style = {
                'position': 'fixed', 
                'top': '10px', 
                'right': '20px',
                'width': '720px',
                'minWidth': '720px',
                'maxWidth': '720px',
                'backgroundColor': 'white',
                'padding': '10px',
                'borderRadius': '5px',
                'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)',
                'border': '1px solid #ddd',
                'zIndex': 1000,
                'maxHeight': '100vh',
                'overflowY': 'auto',
                'overflowX': 'hidden'
            }
            
            # Determine which plot triggered the hover using callback_context
            from dash import callback_context
            ctx = callback_context
            
            # Check which input triggered the callback
            if not ctx.triggered:
                hover_data = None
            else:
                trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                # # ===== ADD DEBUG LOGGING =====
                # logger.info(f"🔍 HOVER DEBUG: Trigger ID = {trigger_id}")
                # logger.info(f"   Main hover data: {main_hover_data is not None}")
                # logger.info(f"   Landmark hover data: {landmark_hover_data is not None}")
                # # ============================
                
                if trigger_id == 'landmark-plot-graph':
                    hover_data = landmark_hover_data
                elif trigger_id == 'main-plot':
                    hover_data = main_hover_data
                else:
                    # Triggered by something else (like scaling mode change)
                    # Use whichever has data
                    hover_data = landmark_hover_data if landmark_hover_data else main_hover_data

            
            # Hide if no hover data
            if not hover_data:
                base_style['display'] = 'none'
                return [], base_style, 0
            
            # Determine data type from plot_type
            data_type = 'cp' if (plot_type and ('cp' in plot_type or 'cellprofiler' in plot_type.lower())) else 'spc'
            
            # Build custom_data_cols the same way the plot does - using get_hover_columns()
            custom_data_cols = [col for col in Config.get_hover_columns(data_type) if col in dataframe.columns]
            logger.debug(f"Image callback using {data_type.upper()} columns: {len(custom_data_cols)} cols")
            hover_info = _extract_hover_info(hover_data, custom_data_cols, dataframe)
            
            # DEBUG
            print("\n=== IMAGE CALLBACK DEBUG ===")
            print(f"Hover data received: {hover_data is not None}")
            print(f"Hover info extracted: {hover_info is not None}")
            if hover_info:
                print(f"Plate: {hover_info.get('plate', hover_info.get('Metadata_plate_barcode', 'MISSING'))}")
                print(f"Well: {hover_info.get('well', hover_info.get('Metadata_well', 'MISSING'))}")
            print("===========================\n")
            
            if not hover_info:
                base_style['display'] = 'none'
                return [], base_style, 0
            
            # Get plate and well information (handle both SPC and CP naming)
            plate = hover_info.get('plate', hover_info.get('Metadata_plate_barcode', None))
            well = hover_info.get('well', hover_info.get('Metadata_well', None))
            
            if not plate or not well:
                logger.debug("Missing plate or well information in hover data")
                base_style['display'] = 'none'
                return [], base_style, 0
            
            # Generate random site and find thumbnail
            site = random.randint(*Config.get_site_range())
            site_str = f"{site:02d}"
            logger.debug(f"Using random site: {site_str}")
            
            thumbnail_path = find_thumbnail(plate, well, site_str, scaling_mode)
            if not thumbnail_path:
                logger.debug(f"No thumbnail found for plate={plate}, well={well}")
                base_style['display'] = 'none'
                return [], base_style, 0
            
            # Determine labeling
            add_label = 'add_label' in add_label_checkbox
            label_text = _get_label_text(hover_info, label_type, add_label)
            
            # Create image with potential text overlay
            img_src = add_text_to_image(thumbnail_path, label_text, add_label)
            if not img_src:
                base_style['display'] = 'none'
                return [], base_style, 0
            
            # Create container content
            children = _create_fixed_image_content(
                img_src, hover_info, thumbnail_path, scaling_mode, label_text, add_label
            )
            
            # Show container
            base_style['display'] = 'block'
            import time
            return children, base_style, time.time()  # ← ADD timestamp
            
        except Exception as e:
            logger.error(f"Error updating fixed image: {str(e)}")
            logger.error(traceback.format_exc())
            base_style['display'] = 'none'
            return [], base_style, 0
    
    @app.callback(
        Output('image-display', 'children'),
        [Input('main-plot', 'clickData')],
        [State('last-clicked-point', 'data'),
         State('add-label-checkbox', 'value'),
         State('label-type-radio', 'value'),
         State('scaling-mode', 'value'),
         State('plot-type-dropdown', 'value')]  # ← ADDED plot type
    )
    def display_image(click_data: dict, last_clicked: str, 
                        add_label_checkbox: List[str], label_type: str,
                        scaling_mode: str, plot_type: str) -> Any:  # ← ADDED plot_type
        """
        Display detailed image view when clicking on plot points.
        
        Shows a larger, detailed view of the microscopy image with comprehensive
        metadata when users click on specific points in the plot.
        
        Args:
            click_data: Plotly click data
            last_clicked: Previously clicked point identifier
            add_label_checkbox: List containing 'add_label' if enabled
            label_type: Type of label to add
            scaling_mode: Image scaling mode
            plot_type: Current plot type (for dataframe selection)
            
        Returns:
            HTML components for detailed image display
        """
        try:
            # ===== SELECT DATAFRAME BASED ON PLOT TYPE =====
            dataframe = _get_dataframe_for_plot_type(plot_type)
            
            if dataframe is None:
                return html.Div("No data available")
            # ==============================================
            
            Config = get_config()
            
            if not click_data:
                return html.Div(
                    "Click on a point to display its image",
                    style={'textAlign': 'center', 'padding': '20px', 'color': '#666'}
                )
            
            # Determine data type from plot_type
            data_type = 'cp' if (plot_type and ('cp' in plot_type or 'cellprofiler' in plot_type.lower())) else 'spc'
    
            # Build custom_data_cols EXACTLY the same way as plot - using get_hover_columns()
            custom_data_cols = [col for col in Config.get_hover_columns(data_type) if col in dataframe.columns]
            logger.debug(f"Click callback using {data_type.upper()} columns: {len(custom_data_cols)} cols")
            
            hover_info = _extract_hover_info(click_data, custom_data_cols, dataframe)
            if not hover_info:
                return html.Div("No data available for clicked point")
            
            # Get plate and well (handle both SPC and CP)
            plate = hover_info.get('plate', hover_info.get('Metadata_plate_barcode', None))
            well = hover_info.get('well', hover_info.get('Metadata_well', None))

            # Convert to string
            if plate is not None:
                plate = str(plate)
            if well is not None:
                well = str(well)
            
            if not plate or not well:
                return html.Div("Missing plate or well information for this point")
            
            # Check if same point clicked again
            current_point = f"{plate}_{well}"
            if last_clicked == current_point:
                raise PreventUpdate
            
            # Find thumbnail and create image with user's scaling mode
            thumbnail_path = find_thumbnail(plate, well, scaling_mode=scaling_mode)
            
            # Determine labeling
            add_label = 'add_label' in add_label_checkbox
            label_text = _get_label_text(hover_info, label_type, add_label)
            
            img_src = add_text_to_image(thumbnail_path, label_text, add_label)
            
            if not img_src:
                return html.Div([
                    html.H3(f"Image not found for Plate: {plate}, Well: {well}"),
                    html.P("Try other points to see images")
                ])
            
            # Create detailed display
            return _create_detailed_image_display(hover_info, img_src)
            
        except PreventUpdate:
            raise
        except Exception as e:
            logger.error(f"Error displaying image: {str(e)}")
            logger.error(traceback.format_exc())
            return html.Div([
                html.H3("Error displaying image"),
                html.P(f"Error message: {str(e)}")
            ])
        
        # ===== CLOSE BUTTON CALLBACK =====
    @app.callback(
        [Output('fixed-image-container', 'children', allow_duplicate=True),
         Output('fixed-image-container', 'style', allow_duplicate=True)],
        [Input('close-hover-image-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def close_hover_image(n_clicks):
        """Close the hover image container when X button is clicked."""
        if n_clicks:
            return [], {'display': 'none'}
        return no_update, no_update
    
    # ===== AUTO-HIDE AFTER TIMEOUT =====
    @app.callback(
        [Output('fixed-image-container', 'children', allow_duplicate=True),
         Output('fixed-image-container', 'style', allow_duplicate=True)],
        [Input('hover-timeout-interval', 'n_intervals')],
        [State('last-hover-time', 'data'),
         State('fixed-image-container', 'style')],
        prevent_initial_call=True
    )
    def auto_hide_hover_image(n_intervals, last_hover_time, current_style):
        """Auto-hide hover image after 1 second of no new hovers."""
        import time
        
        # # ===== ADD DEBUG LOGGING =====
        # logger.info(f" AUTO-HIDE CHECK: Interval #{n_intervals}")
        # logger.info(f"   Container visible: {current_style and current_style.get('display') == 'block'}")
        # if last_hover_time:
        #     elapsed = time.time() - last_hover_time
        #     logger.info(f"   Time since last hover: {elapsed:.1f}s")
        # # ============================
        
        if current_style and current_style.get('display') == 'block':
            # Check if it's been more than 100 seconds since last hover
            if last_hover_time and (time.time() - last_hover_time) > 100.0:
                logger.info("    HIDING NOW!")
                return [], {'display': 'none'}
        
        return no_update, no_update


def _extract_hover_info(hover_data: dict, hover_data_cols: List[str], dataframe: pd.DataFrame) -> dict:
    """
    Extract hover information from plot data using DYNAMIC column mapping.
    Intelligently detects source plot by checking customdata content.
    """
    try:
        pt = hover_data['points'][0]
        custom_data = pt.get('customdata', [])
        
        # DEBUG: Print what we received
        print(f"DEBUG: Received {len(custom_data)} custom data items")
        print("DEBUG: Raw customdata values:")
        for i, value in enumerate(custom_data):
            print(f"  [{i}] = {value}")
        print("=" * 50)

        # ===== ADD THIS NEW DEBUG BLOCK HERE =====
        print("DEBUG: Column mapping (hover_data_cols → value):")
        for i, col_name in enumerate(hover_data_cols):
            value = custom_data[i] if i < len(custom_data) else "N/A"
            print(f"  [{i:2d}] {col_name:45s} = {str(value)[:40]}")
        print("=" * 50)
        
        hover_info = {}
        
        # ===== INTELLIGENT DETECTION: Check if this looks like landmark data =====
        # Landmark plot has: treatment, plate, well as first 3 items
        # Main plot has: plate, well, SMILES as first 3 items
        
        is_landmark_plot = False
        if len(custom_data) >= 3:
            # Check if item[0] looks like a treatment (contains '@')
            # and item[1] is numeric (plate number)
            try:
                item_0 = str(custom_data[0])
                item_1 = custom_data[1]
                
                # Landmark pattern: "treatment@conc", plate_number, well
                # Main plot pattern: plate_number, well, SMILES
                
                if '@' in item_0 and str(item_1).replace('.','').isdigit():
                    is_landmark_plot = True
                    print("DEBUG: Detected LANDMARK PLOT (treatment@conc pattern)")
                else:
                    print("DEBUG: Detected MAIN PLOT (plate-first pattern)")
                    
            except (ValueError, TypeError):
                pass
        
        # ===== HANDLE LANDMARK PLOT =====
        if is_landmark_plot:
            # Landmark customdata order from landmark_callbacks.py
            landmark_order = [
                'treatment',              # 0
                'plate',                  # 1
                'well',                   # 2
                'compound_uM',            # 3
                'library',                # 4
                'PP_ID',                  # 5
                'PP_ID_uM',               # 6
                'annotated_target',       # 7
                'manual_annotation',      # 8
            ]
            
            for i, col_name in enumerate(landmark_order):
                if i < len(custom_data):
                    hover_info[col_name] = custom_data[i]
                else:
                    hover_info[col_name] = None
            
            # Add CP aliases for compatibility
            hover_info['Metadata_plate_barcode'] = hover_info.get('plate')
            hover_info['Metadata_well'] = hover_info.get('well')
            hover_info['Metadata_compound_uM'] = hover_info.get('compound_uM')
            hover_info['Metadata_library'] = hover_info.get('library')
            hover_info['Metadata_PP_ID'] = hover_info.get('PP_ID')
            hover_info['Metadata_PP_ID_uM'] = hover_info.get('PP_ID_uM')
            
            if 'PP_ID' in hover_info and hover_info['PP_ID']:
                hover_info['compound_name'] = hover_info['PP_ID']
            if 'PP_ID_uM' in hover_info and hover_info['PP_ID_uM']:
                hover_info['treatment'] = hover_info['PP_ID_uM']
            
            print(f"DEBUG: Extracted from landmark plot - Plate: {hover_info.get('plate')}, Well: {hover_info.get('well')}")
            return hover_info
        
        # ===== HANDLE MAIN PLOT =====
        print("DEBUG: Using MAIN PLOT customdata structure")
        
        # Map customdata values to column names by index
        for i, col_name in enumerate(hover_data_cols):
            if i < len(custom_data):
                hover_info[col_name] = custom_data[i]
            else:
                hover_info[col_name] = None
        
        print(f"DEBUG: Mapped {len(hover_info)} columns from customdata")
        print(f"DEBUG: Key values:")
        print(f"  Treatment: {hover_info.get('treatment', hover_info.get('PP_ID_uM'))}")
        print(f"  Plate: {hover_info.get('plate')}")
        print(f"  Well: {hover_info.get('well')}")
        
        # Get point index for metrics lookup
        point_index = pt.get('pointIndex', None)
        if point_index is not None and point_index < len(dataframe):
            row = dataframe.iloc[point_index]
            
            # Add metric columns from dataframe if not in hover_info
            metric_columns = [
                'cosine_distance_from_dmso', 'mad_cosine', 'var_cosine', 
                'std_cosine', 'median_distance', 'closest_landmark_distance'
            ]
            
            for metric_col in metric_columns:
                if metric_col in dataframe.columns and metric_col not in hover_info:
                    hover_info[metric_col] = row[metric_col]
        
        return hover_info
        
    except (KeyError, IndexError, TypeError) as e:
        logger.debug(f"Error extracting hover info: {e}")
        return {}

def _get_label_text(hover_info: dict, label_type: str, add_label: bool) -> str:
    """
    Get the appropriate label text based on settings.
    
    Args:
        hover_info: Hover information dictionary
        label_type: Type of label ('treatment' or 'moa_compound_uM')
        add_label: Whether to add label
        
    Returns:
        str: Label text or empty string
    """
    if not add_label:
        return ""
    
    if label_type == 'treatment':
        return str(hover_info.get('treatment', ''))
    elif label_type == 'moa_compound_uM':
        return str(hover_info.get('moa_compound_uM', ''))
    
    return ""

def _create_fixed_image_content(img_src: str, hover_info: dict, thumbnail_path, 
                               scaling_mode: str, label_text: str, add_label: bool) -> List:
    """
    Create content for the fixed position image container - COMPACT LAYOUT.
    Image + text/structure side-by-side at top, then landmarks/chemprot/gene below.
    """
    # Extract site information
    site_num = extract_site_from_path(thumbnail_path) if thumbnail_path else "Unknown"
    
    # Get display information (handle both SPC and CP naming)
    compound = hover_info.get('PP_ID') or hover_info.get('Metadata_PP_ID') or hover_info.get('compound_name', 'Unknown')
    treatment = hover_info.get('PP_ID_uM') or hover_info.get('Metadata_PP_ID_uM') or hover_info.get('treatment', 'Unknown')
    concentration = hover_info.get('compound_uM', hover_info.get('Metadata_compound_uM', 'Unknown'))
    concentration_text = f"{concentration} µM" if concentration not in [None, 'Unknown'] else "Unknown"
    plate = hover_info.get('plate', hover_info.get('Metadata_plate_barcode', 'Unknown'))
    well = hover_info.get('well', hover_info.get('Metadata_well', 'Unknown'))
    target_description = _safe_get_column(hover_info, 'target_description_truncated_10', 
                                         _safe_get_column(hover_info, 'Metadata_annotated_target_description_truncated_10', 'Unknown'))
    moa = _safe_get_column(hover_info, 'moa_truncated_10', 'Unknown')
    chemical_description = _safe_get_column(hover_info, 'chemical_description', 
                                           _safe_get_column(hover_info, 'Metadata_chemical_description', 'Unknown'))
    smiles = hover_info.get('SMILES', hover_info.get('Metadata_SMILES', ''))
    gene_description = hover_info.get('gene_description', None)
    moa_first = hover_info.get('moa_first', None)
    
    # === LEFT COLUMN: Image ===
    left_column = html.Div([
        html.Img(
            src=img_src, 
            style={
                'width': '340px', 
                'height': '340px', 
                'border': '1px solid #ddd',
                'borderRadius': '4px'
            }
        ),
        html.P(f"Site: {site_num} | {('Auto' if scaling_mode == 'auto' else 'Fixed')}", 
            style={'fontSize': '9px', 'color': '#666', 'marginTop': '4px', 'marginBottom': '0px'})
    ], style={
        'display': 'inline-block',
        'verticalAlign': 'top',
        'width': '340px',
        'marginRight': '10px'
    })
    
    # === RIGHT COLUMN: Text info + Chemical structure ===
    right_column_items = [
        html.P(f"Compound: {compound}", 
            style={'fontWeight': 'bold', 'margin': '0 0 2px 0', 'fontSize': '11px', 'color': 'black'}),
        html.P(f"Treatment: {treatment}", 
            style={'margin': '0 0 2px 0', 'fontSize': '10px', 'color': 'black'}),
        html.P(f"Plate: {plate}, Well: {well}", 
            style={'margin': '0 0 2px 0', 'fontSize': '10px', 'color': 'black'}),
        html.P(f"Conc: {concentration_text}", 
            style={'margin': '0 0 2px 0', 'fontSize': '10px', 'color': 'black'}),
        html.P(f"MOA: {moa}",
            style={'margin': '0 0 2px 0', 'fontSize': '10px', 'color': 'black'}),
        html.P(f"Target: {target_description}",
            style={'margin': '0 0 2px 0', 'fontSize': '10px', 'color': 'black'}),
        html.P(f"Chem: {chemical_description}",
            style={'margin': '0 0 5px 0', 'fontSize': '10px', 'color': 'black'})
    ]
    
    # Add SMILES structure to right column if available
    Config = get_config()
    if Config.SHOW_CHEMICAL_STRUCTURES and smiles and str(smiles).strip() not in ['Unknown', 'nan', '', 'None']:
        structure_img = smiles_to_image_base64(smiles, width=200, height=200)
        
        if structure_img:
            right_column_items.append(
                html.Div([
                    html.P("Structure:", style={'fontWeight': 'bold', 'fontSize': '10px', 'marginBottom': '3px', 'color': 'black'}),
                    html.Img(
                        src=structure_img,
                        style={
                            'maxWidth': '180px',
                            'maxHeight': '180px',
                            'border': '1px solid #ddd',
                            'borderRadius': '4px',
                            'backgroundColor': 'white'
                        }
                    )
                ], style={'marginTop': '5px'})
            )
    
    right_column = html.Div(
        right_column_items,
        style={
            'display': 'inline-block',
            'verticalAlign': 'top',
            'width': '350px',
            'maxHeight': '350px',
            'overflowY': 'auto'
        }
    )
    
    # === TOP SECTION: Image and text side by side ===
    content = [
 
            # ===== ADD CLOSE BUTTON =====
            html.Button(
                "✕",
                id='close-hover-image-btn',
                n_clicks=0,
                style={
                    'position': 'absolute',
                    'top': '5px',
                    'right': '5px',
                    'backgroundColor': 'transparent',
                    'border': 'none',
                    'fontSize': '24px',
                    'cursor': 'pointer',
                    'color': '#666',
                    'fontWeight': 'bold',
                    'zIndex': 1001,
                    'padding': '5px 10px'
                },
                title='Close hover preview'
            ),
           html.Div([left_column, right_column], 
                style={'marginBottom': '8px', 'borderBottom': '1px solid #ddd', 'paddingBottom': '8px'})
    ]
    
    # === BOTTOM SECTION: Landmarks (compact) ===
    landmark_content = []
    
    for i, prefix in enumerate(['closest', 'second_closest', 'third_closest'], 1):
        # CP columns use different naming than SPC
        # SPC: {prefix}_landmark_moa_first
        # CP:  {prefix}_landmark_Metadata_annotated_target_first
        
        # Get landmark name (the MOA/target)
        landmark = _safe_get_column(hover_info, f'{prefix}_landmark_moa_first', None)
        if not landmark or landmark == 'Unknown':
            landmark = _safe_get_column(hover_info, f'{prefix}_landmark_Metadata_annotated_target_first', None)
        
        # Get distance
        distance = hover_info.get(f'{prefix}_landmark_distance', None)
        
        # Get target description
        target = _safe_get_column(hover_info, f'{prefix}_landmark_annotated_target_description_truncated_10', None)
        if not target or target == 'Unknown':
            target = _safe_get_column(hover_info, f'{prefix}_landmark_Metadata_annotated_target_description_truncated_10', None)
        
        # Get PP_ID for display
        pp_id = _safe_get_column(hover_info, f'{prefix}_landmark_PP_ID_uM', None)
        if not pp_id or pp_id == 'Unknown':
            pp_id = _safe_get_column(hover_info, f'{prefix}_landmark_Metadata_PP_ID_uM', None)
        
        if landmark and landmark != 'Unknown':
            parts = [f"{i}. {landmark}"]
            if pp_id and pp_id != 'Unknown':
                parts.append(f"({pp_id})")
            if target and target != 'Unknown':
                parts.append(f"→ {target}")
            if distance is not None and not pd.isna(distance):
                try:
                    parts.append(f"[{float(distance):.4f}]")
                except (ValueError, TypeError):
                    pass
            
            # Define colors for each landmark level
            landmark_colors = {
                1: '#1565c0',  # Sapphire blue
                2: '#6a1b9a',  # Amethyst purple
                3: '#00838f',  # Teal
            }

            landmark_content.append(
                html.P(" ".join(parts),
                    style={
                        'fontSize': '9px', 
                        'color': landmark_colors.get(i, '#555'),
                        'marginBottom': '1px',
                        'fontWeight': 'bold' if i == 1 else 'normal'
                    })
            )
    
    if landmark_content:
        content.append(
            html.Div([
                html.P("Landmarks:", 
                    style={'fontWeight': 'bold', 'fontSize': '10px', 'marginBottom': '2px', 'color': 'black'}),
                *landmark_content
            ], style={
                'backgroundColor': '#f8f9fa',
                'padding': '5px',
                'borderRadius': '3px',
                'marginBottom': '5px'
            })
        )
    
    # === CHEMOPROTEOMICS SECTION (compact) ===
    has_chemprot = any(
        hover_info.get(col) and str(hover_info.get(col)).strip() not in ['Unknown', 'nan', '', 'None', 'null']
        for col in ['cell_line', 'experiment_type', 'gene_site_1']
    )
    
    if has_chemprot:
        chemprot_items = []
        
        cell_line = _safe_get_column(hover_info, 'cell_line', None)
        exp_type = _safe_get_column(hover_info, 'experiment_type', None)
        
        if cell_line and cell_line != 'Unknown':
            chemprot_items.append(html.Span(f"{cell_line}", style={'fontWeight': 'bold', 'color': '#856404'}))
        if exp_type and exp_type != 'Unknown':
            chemprot_items.append(html.Span(f" | {exp_type}", style={'color': '#856404'}))
        
        gene_sites = []
        for i in range(1, 6):
            gene_site = _safe_get_column(hover_info, f'gene_site_{i}', None)
            if gene_site and gene_site != 'Unknown':
                gene_sites.append(gene_site)
        
        if gene_sites:
            chemprot_items.append(html.Br())
            chemprot_items.append(
                html.Span(f"Targets: {', '.join(gene_sites)}", 
                    style={'fontSize': '9px', 'color': '#856404'})
            )
        
        if chemprot_items:
            content.append(
                html.Div([
                    html.P("Chemoproteomics:", 
                        style={'fontWeight': 'bold', 'fontSize': '10px', 'marginBottom': '2px', 'color': '#856404'}),
                    html.Div(chemprot_items, style={'fontSize': '9px'})
                ], style={
                    'backgroundColor': '#fff3cd',
                    'padding': '5px',
                    'borderRadius': '3px',
                    'marginBottom': '5px'
                })
            )
    
    # === GENE DESCRIPTION SECTION (compact, truncated) ===
    if gene_description and moa_first and str(gene_description).strip() not in ['Unknown', 'nan', '', 'null']:
        gene_desc_display = str(gene_description)
        if len(gene_desc_display) > 200:
            gene_desc_display = gene_desc_display[:200] + "..."
        
        content.append(
            html.Div([
                html.P(f"Gene: {moa_first}", 
                    style={'fontWeight': 'bold', 'fontSize': '10px', 'marginBottom': '2px', 'color': 'black'}),
                html.P(gene_desc_display, 
                    style={'fontSize': '9px', 'lineHeight': '1.3', 'marginBottom': '0px', 'color': 'black'})
            ], style={
                'backgroundColor': '#e8f4fd',
                'padding': '5px',
                'borderRadius': '3px',
                'maxHeight': '80px',
                'overflowY': 'auto'
            })
            
        )
    
    return content


def _create_tooltip_content(hover_info: dict) -> List:
    """
    Create tooltip content for hover display.
    
    Args:
        hover_info: Hover information dictionary
        
    Returns:
        List: HTML components for tooltip
    """
    # Handle both SPC and CP naming
    compound = hover_info.get('compound_name', hover_info.get('Metadata_chemical_name', 'Unknown'))
    treatment = hover_info.get('treatment', 'Unknown')
    concentration = hover_info.get('compound_uM', hover_info.get('Metadata_compound_uM', 'Unknown'))
    plate = hover_info.get('plate', hover_info.get('Metadata_plate_barcode', 'Unknown'))
    well = hover_info.get('well', hover_info.get('Metadata_well', 'Unknown'))
    
    # Format concentration
    concentration_text = f"{concentration} µM" if concentration not in [None, 'Unknown'] else "Unknown"
    
    return [
        html.Div([
            html.P(f"Compound: {compound}", style={'fontWeight': 'bold', 'marginBottom': '4px'}),
            html.P(f"Treatment: {treatment}", style={'marginBottom': '4px'}),
            html.P(f"Plate: {plate}, Well: {well}", style={'marginBottom': '4px'}),
            html.P(f"Concentration: {concentration_text}", style={'marginBottom': '0px'})
        ], style={'padding': '8px'})
    ]