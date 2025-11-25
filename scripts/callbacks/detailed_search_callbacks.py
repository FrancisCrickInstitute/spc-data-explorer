"""
Callback for detailed compound search functionality - COMPLETE VERSION
"""

from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import pandas as pd
from typing import Any, List
import logging

logger = logging.getLogger(__name__)


def register_detailed_search_callbacks(app, df_spc: pd.DataFrame, df_cp: pd.DataFrame):
    """
    Register callbacks for the detailed compound search feature.
    
    Args:
        app: The Dash app instance
        dataframe: The loaded dataframe
    """
    
    # Import necessary functions
    try:
        from utils.image_utils import find_thumbnail, add_text_to_image
        from config_loader import get_config
        from utils.smiles_utils import smiles_to_image_base64
    except ImportError:
        try:
            from ..utils.image_utils import find_thumbnail, add_text_to_image
            from ..config_loader import get_config
            from ..utils.smiles_utils import smiles_to_image_base64
        except ImportError:
            logger.error("Could not import required modules")
            return
    
    Config = get_config()

    def _get_dataframe_for_plot_type(plot_type: str) -> pd.DataFrame:
        """Determine which dataframe to use based on plot type."""
        if plot_type and ('cp' in plot_type or 'cellprofiler' in plot_type.lower()):
            logger.debug(f"Plot type '{plot_type}' → using CP dataframe")
            return df_cp if df_cp is not None else df_spc
        else:
            logger.debug(f"Plot type '{plot_type}' → using SPC dataframe")
            return df_spc if df_spc is not None else df_cp
    
    def _safe_get_column(hover_info: dict, column_name: str, fallback: str = 'Unknown') -> str:
        """Safely get column value with fallback."""
        value = hover_info.get(column_name, fallback)
        if not value or pd.isna(value) or str(value).strip() in ['Unknown', 'nan', '']:
            return fallback
        return str(value)
    

    @app.callback(
        Output('detailed-compound-search-dropdown', 'options'),
        Input('detailed-compound-search-dropdown', 'search_value'),
        State('detailed-compound-search-dropdown', 'value'),
        State('plot-type-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_detailed_search_options(search_value: str, current_value: str, plot_type: str):
        """
        Update dropdown options for detailed compound search.
        """
        # DEBUG
        print(f"\n=== DETAILED SEARCH DEBUG ===")
        print(f"Search value: '{search_value}'")
        print(f"Plot type: '{plot_type}'")
        
        # SELECT DATAFRAME
        dataframe = _get_dataframe_for_plot_type(plot_type)
        if dataframe is None:
            print("ERROR: dataframe is None!")
            return []
        
        print(f"Dataframe shape: {dataframe.shape}")
        print(f"'treatment' in columns: {'treatment' in dataframe.columns}")
        print(f"'PP_ID_uM' in columns: {'PP_ID_uM' in dataframe.columns}")
        
        # If user has already selected something, don't clear it
        if current_value and not search_value:
            if '|' in current_value:
                column_type, value = current_value.split('|', 1)
                return [{'label': f"{column_type.title()}: {value}", 'value': current_value}]
        
        if not search_value or len(search_value) < Config.MIN_SEARCH_LENGTH:
            print(f"Search too short (min: {Config.MIN_SEARCH_LENGTH})")
            return []
        
        search_lower = search_value.lower()
        options = []
        seen = set()
        
        # Search in 'treatment' column
        if 'treatment' in dataframe.columns:
            treatment_matches = dataframe[
                dataframe['treatment'].astype(str).str.lower().str.contains(search_lower, na=False)
            ]['treatment'].unique()
            
            print(f"Treatment matches found: {len(treatment_matches)}")
            
            for treatment in treatment_matches[:50]:
                if treatment not in seen:
                    options.append({
                        'label': f"Treatment: {treatment}",
                        'value': f"treatment|{treatment}"
                    })
                    seen.add(treatment)
        
        # Search in 'PP_ID_uM' column
        if 'PP_ID_uM' in dataframe.columns:
            pp_id_matches = dataframe[
                dataframe['PP_ID_uM'].astype(str).str.lower().str.contains(search_lower, na=False)
            ]['PP_ID_uM'].unique()
            
            print(f"PP_ID_uM matches found: {len(pp_id_matches)}")
            
            for pp_id in pp_id_matches[:50]:
                if pp_id not in seen:
                    compound_info = dataframe[dataframe['PP_ID_uM'] == pp_id].iloc[0]
                    compound_name = compound_info.get('compound_name', '')
                    
                    label = f"PP_ID: {pp_id}"
                    if compound_name and str(compound_name) != 'nan':
                        label += f" ({compound_name})"
                    
                    options.append({
                        'label': label,
                        'value': f"PP_ID_uM|{pp_id}"
                    })
                    seen.add(pp_id)
        
        print(f"Total options returned: {len(options)}")
        print("=== END DEBUG ===\n")
        
        return sorted(options, key=lambda x: x['label'])[:Config.MAX_SEARCH_RESULTS]
    
    
    @app.callback(
        Output('detailed-compound-search-result', 'children'),
        Input('detailed-compound-search-button', 'n_clicks'),
        [State('detailed-compound-search-dropdown', 'value'),
         State('scaling-mode', 'value'),
         State('plot-type-dropdown', 'value')],  # ← ADD THIS
        prevent_initial_call=True
    )

    def display_detailed_compound_info(n_clicks: int, selected_value: str, scaling_mode: str, plot_type:str) -> Any:
        """
        Display detailed compound information - FULL VERSION matching UMAP click.
        """

        # SELECT DATAFRAME
        dataframe = _get_dataframe_for_plot_type(plot_type)
        if dataframe is None:
            return html.Div(
                "No data available",
                style={'textAlign': 'center', 'padding': '20px', 'color': '#e74c3c'}
            )
        
        if not selected_value:
            return html.Div(
                "Please select a compound from the dropdown",
                style={'textAlign': 'center', 'padding': '20px', 'color': '#999'}
            )
        
        try:
            # Parse the selected value
            column_type, search_value = selected_value.split('|', 1)
            
            # Find the matching row
            if column_type == 'treatment':
                matching_rows = dataframe[dataframe['treatment'] == search_value]
            elif column_type == 'PP_ID_uM':
                matching_rows = dataframe[dataframe['PP_ID_uM'] == search_value]
            else:
                return html.Div("Invalid selection format")
            
            if matching_rows.empty:
                return html.Div(
                    f"No data found for {column_type}: {search_value}",
                    style={'textAlign': 'center', 'padding': '20px', 'color': '#e74c3c'}
                )
            
            # Use the first matching row
            row = matching_rows.iloc[0]
            hover_info = row.to_dict()
            
            # Get plate and well
            plate = row.get('plate', None)
            well = row.get('well', None)
            
            if not plate or not well:
                return html.Div("Missing plate or well information")
            
            # Find thumbnail
            # Find thumbnail (use the scaling mode from UI)
            thumbnail_path = find_thumbnail(plate, well, scaling_mode=scaling_mode)
            img_src = add_text_to_image(thumbnail_path, "", add_label=False)
            
            if not img_src:
                return html.Div([
                    html.H3(f"Image not found for Plate: {plate}, Well: {well}"),
                    html.P("Data found but image file is missing")
                ])
            
            # Extract all info (matching UMAP click display)
            compound = hover_info.get('PP_ID', '')
            if pd.isna(compound) or str(compound).strip() == '':
                compound = hover_info.get('compound_name', 'Unknown')
            
            treatment = hover_info.get('PP_ID_uM', '')
            if pd.isna(treatment) or str(treatment).strip() == '':
                treatment = hover_info.get('treatment', 'Unknown')
            
            concentration = hover_info.get('compound_uM', 'Unknown')
            compound_type = _safe_get_column(hover_info, 'compound_type', 'Unknown')
            library = hover_info.get('library', 'Unknown')
            moa = _safe_get_column(hover_info, 'moa_truncated_10', 'Unknown')
            target_description = _safe_get_column(hover_info, 'target_description_truncated_10', 'Unknown')
            chemical_description = _safe_get_column(hover_info, 'chemical_description', 'Unknown')
            smiles = hover_info.get('SMILES', '')
            
            # Validity
            validity = hover_info.get('valid_for_phenotypic_makeup', None)
            validity_symbol = '✓' if validity == True else '✗'
            validity_color = '#2ecc71' if validity == True else '#e74c3c'
            
            concentration_text = f"{concentration} µM" if concentration not in [None, 'Unknown'] else "Unknown"
            
            # Build metrics display with visual bars
            metrics_display = []
            metric_info = {
                'cosine_distance_from_dmso': {'name': 'Cosine Distance from DMSO', 'format': '.4f'},
                'mad_cosine': {'name': 'MAD Cosine', 'format': '.4f'},
                'var_cosine': {'name': 'Variance Cosine', 'format': '.4f'},
                'std_cosine': {'name': 'Std Dev Cosine', 'format': '.4f'},
                'median_distance': {'name': 'Median Distance', 'format': '.4f'},
                'closest_landmark_distance': {'name': 'Closest Landmark Distance', 'format': '.4f'}
            }
            
            for key, info in metric_info.items():
                if key in hover_info and hover_info[key] is not None:
                    value = hover_info[key]
                    if isinstance(value, (int, float)):
                        if key in dataframe.columns:
                            min_val = dataframe[key].min()
                            max_val = dataframe[key].max()
                            
                            if max_val > min_val:
                                position_pct = ((value - min_val) / (max_val - min_val)) * 100
                            else:
                                position_pct = 50
                            
                            metrics_display.append(
                                html.Div([
                                    html.P(f"{info['name']}: {value:{info['format']}}", 
                                        style={'marginBottom': '2px', 'fontSize': '12px'}),
                                    html.Div([
                                        html.Div(style={
                                            'width': '100%',
                                            'height': '10px',
                                            'backgroundColor': '#e0e0e0',
                                            'borderRadius': '5px',
                                            'position': 'relative',
                                            'marginBottom': '8px'
                                        }),
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
                                        ], style={'position': 'relative'})
                                    ], style={'position': 'relative', 'height': '25px'})
                                ])
                            )
            
            # Build landmark information
            landmark_info = []
            
            def get_landmark_info(prefix: str) -> List[str]:
                """Extract landmark information for a given prefix."""
                moa_col = f'{prefix}_landmark_moa_first'
                pp_id_col = f'{prefix}_landmark_PP_ID_uM'
                target_col = f'{prefix}_landmark_annotated_target_description_truncated_10'
                annotation_col = f'{prefix}_landmark_manual_annotation'
                distance_col = f'{prefix}_landmark_distance'
                
                moa_val = _safe_get_column(hover_info, moa_col, None)
                pp_id_val = _safe_get_column(hover_info, pp_id_col, None)
                target_val = _safe_get_column(hover_info, target_col, None)
                annotation_val = _safe_get_column(hover_info, annotation_col, None)
                distance = hover_info.get(distance_col, None)
                
                if moa_val or pp_id_val:
                    info_parts = []
                    if moa_val and moa_val != 'Unknown':
                        info_parts.append(f"Landmark: {moa_val}")
                    if pp_id_val and pp_id_val != 'Unknown':
                        info_parts.append(f"PP_ID: {pp_id_val}")
                    if target_val and target_val != 'Unknown':
                        info_parts.append(f"Target: {target_val}")
                    if annotation_val and annotation_val != 'Unknown':
                        info_parts.append(f"Broad Annotation: {annotation_val}")
                    if distance is not None and not pd.isna(distance):
                        try:
                            info_parts.append(f"Distance: {float(distance):.4f}")
                        except (ValueError, TypeError):
                            info_parts.append(f"Distance: {distance}")
                    return info_parts
                return None
            
            for i, prefix in enumerate(['closest', 'second_closest', 'third_closest'], 1):
                landmark_data = get_landmark_info(prefix)
                if landmark_data:
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
            
            # Build left column content
            left_column_content = [
                html.H4("Basic Information", style={'color': '#34495e', 'marginBottom': '10px'}),
                html.P(f"Treatment: {treatment}"),
                html.P(f"Plate: {plate}"),
                html.P(f"Well: {well}"),
                html.P(f"Concentration: {concentration_text}"),
                html.P(f"Compound Type: {compound_type}"),
                html.P(f"Library: {library}"),
                html.P(f"MOA: {moa}"),
                html.P(f"Target Description: {target_description}"),
                html.P(f"Chemical Description: {chemical_description}"),
                html.P([
                    "Landmarks Within Range: ",
                    html.Span(validity_symbol, style={'color': validity_color, 'fontWeight': 'bold', 'fontSize': '14px'})
                ], style={'fontSize': '11px', 'marginBottom': '2px'}),
                
                html.Hr(style={'margin': '15px 0'}),
                
                html.H4("Metrics", style={'color': '#34495e', 'marginBottom': '10px'}),
                *metrics_display,
                
                html.Hr(style={'margin': '15px 0'}),
                
                html.H4("Landmark Information", style={'color': '#34495e', 'marginBottom': '10px'}),
            ]
            
            # Add landmarks
            if landmark_info:
                left_column_content.extend(landmark_info)
            else:
                left_column_content.append(
                    html.P("No landmark information available", 
                          style={'fontSize': '11px', 'color': '#666'})
                )
            
            # Add chemical structure if available and enabled
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
            
            # Final layout with image and info
            return html.Div([
                html.Div([
                    # Image on the right
                    html.Div([
                        html.Img(src=img_src, style={
                            'maxWidth': '100%',
                            'height': 'auto',
                            'border': '2px solid #ddd',
                            'borderRadius': '5px'
                        })
                    ], style={
                        'width': '40%',
                        'display': 'inline-block',
                        'verticalAlign': 'top',
                        'padding': '10px'
                    }),
                    
                    # Info on the left
                    html.Div(left_column_content, style={
                        'width': '58%',
                        'display': 'inline-block',
                        'verticalAlign': 'top',
                        'padding': '10px'
                    })
                ])
            ], style={
                'padding': '20px',
                'backgroundColor': 'white',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
            
        except Exception as e:
            logger.error(f"Error in detailed compound search: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return html.Div([
                html.H3("Error displaying compound information", style={'color': '#e74c3c'}),
                html.P(f"Error: {str(e)}", style={'fontSize': '12px', 'color': '#666'})
            ])