"""
Search callbacks module for the Phenotype Clustering Interactive Visualization App.

This module contains all callback functions related to search functionality,
including dropdown population, search execution, and result display for
compound IDs, MOAs, target descriptions, and DMSO controls.

Key Functions:
- register_search_callbacks(): Register all search-related callbacks with the app
- update_treatment_dropdown_options(): Dynamic compound ID search dropdown population
- update_moa_dropdown_options(): Dynamic MOA search dropdown population
- update_target_dropdown_options(): Dynamic target description search dropdown population
- display_treatment_result(): Handle compound ID search results
- display_moa_result(): Handle MOA search results
- display_target_result(): Handle target description search results
- display_dmso_image(): Handle DMSO control display

The module implements sophisticated search functionality with real-time dropdown
updates, multiple search result displays, and seamless integration with image display.
"""

import random
import pandas as pd
from dash import Input, Output, State, callback_context, html
import logging
from typing import List, Dict, Any, Tuple, Optional

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


def register_search_callbacks(app, df_spc: pd.DataFrame, df_cp: pd.DataFrame) -> None:
    """
    Register all search-related callbacks with the Dash app.
    
    Args:
        app: Dash application instance
        df: The main dataframe
    """

    # Helper function

    def _get_dataframe_for_plot_type(plot_type: str) -> Optional[pd.DataFrame]:
        """Determine which dataframe to use based on plot type."""
        if plot_type and ('cp' in plot_type or 'cellprofiler' in plot_type.lower()):
            logger.debug(f"Plot type '{plot_type}' → using CP dataframe")
            return df_cp if df_cp is not None else df_spc
        else:
            logger.debug(f"Plot type '{plot_type}' → using SPC dataframe")
            return df_spc if df_spc is not None else df_cp
    
    # Treatment (Compound ID) search callbacks
    @app.callback(
        [Output('treatment-search-dropdown', 'options'),
         Output('treatment-search-dropdown', 'value')],
        [Input('treatment-search-dropdown', 'search_value'),
         Input('treatment-search-dropdown', 'value')],
        [State('treatment-search-dropdown', 'options'),
         State('plot-type-dropdown', 'value')]
    )
    def update_treatment_dropdown_options(search_value: str, current_value: str, 
                                    current_options: List[Dict], plot_type: str) -> Tuple[List[Dict], str]:
        """
        Update treatment dropdown options based on user search input.
        
        Implements dynamic search functionality that populates dropdown options
        as the user types, with intelligent handling of selections and search states.
        
        Args:
            search_value: Current search text
            current_value: Currently selected value
            current_options: Current dropdown options
            
        Returns:
            Tuple of (new_options, current_value)
        """
        try:
            # SELECT DATAFRAME
            df = _get_dataframe_for_plot_type(plot_type)
            if df is None:
                return [], current_value
            
            Config = get_config() 
            ctx = callback_context
            triggered = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
            
            # Convert current options to dictionary for easy lookup
            current_options_dict = {
                option['value']: option for option in current_options
            } if current_options else {}
            
            # Handle value selection (not search)
            if triggered == 'treatment-search-dropdown' and ctx.triggered[0]['prop_id'].endswith('.value'):
                if current_value:
                    logger.debug(f"Treatment value selected: {current_value}")
                    # Ensure selected value is in options
                    if current_value not in current_options_dict:
                        new_option = {'label': str(current_value), 'value': current_value}
                        new_options = current_options + [new_option] if current_options else [new_option]
                        return new_options, current_value
                return current_options, current_value
            
            # Handle search input
            if not search_value or len(search_value) < Config.MIN_SEARCH_LENGTH:
                # If no search but we have a selected value, keep it in options
                if current_value:
                    return [{'label': str(current_value), 'value': current_value}], current_value
                return [], current_value
            
            # Perform search - use PP_ID_uM if available, otherwise treatment
            search_value_lower = search_value.lower()

            # Create a column that prioritizes PP_ID_uM over treatment
            if 'PP_ID_uM' in df.columns:
                search_values = df['PP_ID_uM'].fillna(df['treatment']).dropna().unique()
            else:
                search_values = df['treatment'].dropna().unique()

            filtered_treatments = [
                t for t in search_values 
                if search_value_lower in str(t).lower()
            ]
            
            # Limit results
            if len(filtered_treatments) > Config.MAX_SEARCH_RESULTS:
                filtered_treatments = filtered_treatments[:Config.MAX_SEARCH_RESULTS]
            
            options = [{'label': str(t), 'value': str(t)} for t in filtered_treatments]
            
            # Ensure current value is included
            if current_value and current_value not in [opt['value'] for opt in options]:
                options.append({'label': str(current_value), 'value': current_value})
            
            logger.debug(f"Found {len(filtered_treatments)} matching treatments for '{search_value}'")
            return options, current_value
            
        except Exception as e:
            logger.error(f"Error updating treatment dropdown: {str(e)}")
            return [], current_value
    
    
    # MOA search callbacks
    @app.callback(
        [Output('moa-search-dropdown', 'options'),
         Output('moa-search-dropdown', 'value')],
        [Input('moa-search-dropdown', 'search_value'),
         Input('moa-search-dropdown', 'value')],
        [State('moa-search-dropdown', 'options'),
         State('plot-type-dropdown', 'value')]
    )
    def update_moa_dropdown_options(search_value: str, current_value: str, 
                                   current_options: List[Dict], plot_type: str) -> Tuple[List[Dict], str]:
        """
        Update MOA dropdown options based on user search input.
        
        Similar to treatment search but searches in MOA-related columns
        with priority given to moa_compound_uM if available.
        
        Args:
            search_value: Current search text
            current_value: Currently selected value
            current_options: Current dropdown options
            
        Returns:
            Tuple of (new_options, current_value)
        """
        try:
            # SELECT DATAFRAME
            df = _get_dataframe_for_plot_type(plot_type)
            if df is None:
                return [], current_value
            
            Config = get_config()
            ctx = callback_context
            triggered = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
            
            # Convert current options to dictionary
            current_options_dict = {
                option['value']: option for option in current_options
            } if current_options else {}
            
            # Handle value selection
            if triggered == 'moa-search-dropdown' and ctx.triggered[0]['prop_id'].endswith('.value'):
                if current_value:
                    logger.debug(f"MOA value selected: {current_value}")
                    if current_value not in current_options_dict:
                        new_option = {'label': str(current_value), 'value': current_value}
                        new_options = current_options + [new_option] if current_options else [new_option]
                        return new_options, current_value
                return current_options, current_value
            
            # Handle search input
            if not search_value or len(search_value) < Config.MIN_SEARCH_LENGTH:
                if current_value:
                    return [{'label': str(current_value), 'value': current_value}], current_value
                return [], current_value
            
            # NEW CODE (replace with this):
            # Determine which MOA column to search
            moa_column = _get_moa_search_column(df)

            # Perform search - get unique MOA + PP_ID_uM combinations
            search_value_lower = search_value.lower()

            # Create combinations of MOA + PP_ID_uM for uniqueness
            if 'PP_ID_uM' in df.columns:
                # Group by MOA and get the first PP_ID_uM for each MOA
                moa_combinations = df.groupby(moa_column)['PP_ID_uM'].first().reset_index()
                # Filter based on search
                filtered_combinations = moa_combinations[
                    moa_combinations[moa_column].astype(str).str.lower().str.contains(search_value_lower, na=False)
                ]
                
                # Create display labels: "MOA (PP_ID_uM)"
                options = []
                for _, row in filtered_combinations.iterrows():
                    moa_val = str(row[moa_column])
                    pp_id_val = str(row['PP_ID_uM']) if pd.notna(row['PP_ID_uM']) else "Unknown"
                    
                    if pp_id_val != "Unknown" and pp_id_val != "nan":
                        label = f"{moa_val} ({pp_id_val})"
                    else:
                        label = moa_val
                        
                    # Store value as "MOA|PP_ID_uM" for easy parsing
                    value = f"{moa_val}|{pp_id_val}"
                    options.append({'label': label, 'value': value})
            else:
                # Fallback to just MOA if PP_ID_uM not available
                matching_moas = df[moa_column].dropna().unique()
                filtered_moas = [
                    m for m in matching_moas 
                    if search_value_lower in str(m).lower()
                ]
                options = [{'label': str(m), 'value': str(m)} for m in filtered_moas]

            # Limit results
            if len(options) > Config.MAX_SEARCH_RESULTS:
                options = options[:Config.MAX_SEARCH_RESULTS]

            # Ensure current value is included
            if current_value and current_value not in [opt['value'] for opt in options]:
                options.append({'label': str(current_value), 'value': current_value})

            # FIXED: Use the correct variable for logging
            if 'PP_ID_uM' in df.columns:
                logger.debug(f"Found {len(options)} matching MOA+PP_ID combinations for '{search_value}'")
            else:
                logger.debug(f"Found {len(options)} matching MOAs for '{search_value}'")

            return options, current_value
        
        except Exception as e:
            logger.error(f"Error updating MOA dropdown: {str(e)}")
            return [], current_value
    
    # TARGET DESCRIPTION search callbacks
    @app.callback(
        [Output('target-search-dropdown', 'options'),
         Output('target-search-dropdown', 'value')],
        [Input('target-search-dropdown', 'search_value'),
         Input('target-search-dropdown', 'value')],
        [State('target-search-dropdown', 'options'),
         State('plot-type-dropdown', 'value')]
    )
    def update_target_dropdown_options(search_value: str, current_value: str, 
                                     current_options: List[Dict], plot_type: str) -> Tuple[List[Dict], str]:
        """
        Update target description dropdown options based on user search input.
        
        Args:
            search_value: Current search text
            current_value: Currently selected value
            current_options: Current dropdown options
            
        Returns:
            Tuple of (new_options, current_value)
        """
        try:
            # SELECT DATAFRAME
            df = _get_dataframe_for_plot_type(plot_type)
            if df is None:
                return [], current_value
            
            Config = get_config()
            ctx = callback_context
            triggered = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
            
            # Convert current options to dictionary
            current_options_dict = {
                option['value']: option for option in current_options
            } if current_options else {}
            
            # Handle value selection
            if triggered == 'target-search-dropdown' and ctx.triggered[0]['prop_id'].endswith('.value'):
                if current_value:
                    logger.debug(f"Target description value selected: {current_value}")
                    if current_value not in current_options_dict:
                        new_option = {'label': str(current_value), 'value': current_value}
                        new_options = current_options + [new_option] if current_options else [new_option]
                        return new_options, current_value
                return current_options, current_value
            
            # Handle search input
            if not search_value or len(search_value) < Config.MIN_SEARCH_LENGTH:
                if current_value:
                    return [{'label': str(current_value), 'value': current_value}], current_value
                return [], current_value
            
            # Perform search in target_description column
            # NEW: Support both SPC and CP column names
            search_value_lower = search_value.lower()
            
            # Determine which target column to use (CP vs SPC)
            if 'target_description' in df.columns:
                target_col = 'target_description'
            elif 'Metadata_annotated_target_description' in df.columns:
                target_col = 'Metadata_annotated_target_description'  # CP column
            elif 'annotated_target_description' in df.columns:
                target_col = 'annotated_target_description'
            else:
                logger.warning("target_description column not found in dataframe")
                return [], current_value
            
            # Perform the search
            search_values = df[target_col].dropna().unique()
            filtered_targets = [
                t for t in search_values 
                if search_value_lower in str(t).lower()
            ]
            
            # Limit results
            if len(filtered_targets) > Config.MAX_SEARCH_RESULTS:
                filtered_targets = filtered_targets[:Config.MAX_SEARCH_RESULTS]
            
            options = [{'label': str(t), 'value': str(t)} for t in filtered_targets]
            
            # Ensure current value is included
            if current_value and current_value not in [opt['value'] for opt in options]:
                options.append({'label': str(current_value), 'value': current_value})
            
            logger.debug(f"Found {len(filtered_targets)} matching target descriptions for '{search_value}'")
            return options, current_value
            
        except Exception as e:
            logger.error(f"Error updating target description dropdown: {str(e)}")
            return [], current_value
    
    
    # Search result display callbacks
    @app.callback(
        Output('treatment-result-container', 'children'),
        [Input('treatment-search-button', 'n_clicks')],
        [State('treatment-search-dropdown', 'value'),
         State('scaling-mode', 'value'),
         State('add-treatment-label-checkbox', 'value'),
         State('plot-type-dropdown', 'value')],
        prevent_initial_call=True
    )
    def display_treatment_result(n_clicks: int, selected_treatment: str, 
                                scaling_mode: str, add_treatment_label: List[str],
                                plot_type: str) -> Any:
        """
        Display compound ID search results.
        
        Args:
            n_clicks: Number of button clicks
            selected_treatment: Selected treatment/compound ID
            scaling_mode: Image scaling mode
            add_treatment_label: List containing 'add_label' if enabled
            
        Returns:
            HTML components for treatment search results
        """
        try:
            # SELECT DATAFRAME
            df = _get_dataframe_for_plot_type(plot_type)
            if df is None:
                return html.Div("No data available")
            
            if n_clicks and selected_treatment:
                logger.info(f"Displaying treatment result for: {selected_treatment}")

                # Determine label source with fallback
                label_source = 'PP_ID_uM' if 'PP_ID_uM' in df.columns else 'treatment'

                return create_search_result_content(
                    df=df,
                    search_value=selected_treatment,
                    search_column='treatment',
                    search_type="Compound ID",
                    add_label='add_label' in add_treatment_label,
                    label_text_source=label_source,
                    scaling_mode=scaling_mode
                )
            return []
            
        except Exception as e:
            logger.error(f"Error displaying treatment result: {str(e)}")
            return html.Div([
                html.H4("Error", style={'color': 'red'}),
                html.P(f"Could not display result: {str(e)}")
            ])
    
    
    @app.callback(
        Output('moa-result-container', 'children'),
        [Input('moa-search-button', 'n_clicks')],
        [State('moa-search-dropdown', 'value'),
         State('scaling-mode', 'value'),
         State('add-moa-label-checkbox', 'value'),
         State('plot-type-dropdown', 'value')],
        prevent_initial_call=True
    )
    def display_moa_result(n_clicks: int, selected_moa: str, 
                          scaling_mode: str, add_moa_label: List[str],
                          plot_type: str) -> Any:
        """
        Display MOA search results.
        
        Args:
            n_clicks: Number of button clicks
            selected_moa: Selected MOA
            scaling_mode: Image scaling mode
            add_moa_label: List containing 'add_label' if enabled
            
        Returns:
            HTML components for MOA search results
        """
        try:
            # SELECT DATAFRAME
            df = _get_dataframe_for_plot_type(plot_type)
            if df is None:
                return html.Div("No data available")
            
            if n_clicks and selected_moa:
                logger.info(f"Displaying MOA result for: {selected_moa}")
                
                search_column = _get_moa_search_column(df)
                
                return create_search_result_content(
                    df=df,
                    search_value=selected_moa,
                    search_column=search_column,
                    search_type="MOA",
                    add_label='add_label' in add_moa_label,
                    label_text_source='moa_compound_uM',
                    scaling_mode=scaling_mode
                )
            return []
            
        except Exception as e:
            logger.error(f"Error displaying MOA result: {str(e)}")
            return html.Div([
                html.H4("Error", style={'color': 'red'}),
                html.P(f"Could not display result: {str(e)}")
            ])
    
    
    @app.callback(
        Output('target-result-container', 'children'),
        [Input('target-search-button', 'n_clicks')],
        [State('target-search-dropdown', 'value'),
         State('scaling-mode', 'value'),
         State('add-target-label-checkbox', 'value'),
         State('plot-type-dropdown', 'value')],
        prevent_initial_call=True
    )
    def display_target_result(n_clicks: int, selected_target: str, 
                            scaling_mode: str, add_target_label: List[str],
                            plot_type: str) -> Any:
        """
        Display target description search results.
        
        Args:
            n_clicks: Number of button clicks
            selected_target: Selected target description
            scaling_mode: Image scaling mode
            add_target_label: List containing 'add_label' if enabled
            
        Returns:
            HTML components for target description search results
        """
        try:
            # SELECT DATAFRAME
            df = _get_dataframe_for_plot_type(plot_type)
            if df is None:
                return html.Div("No data available")
            
            if n_clicks and selected_target:
                logger.info(f"Displaying target description result for: {selected_target}")
                
                # NEW: Determine correct target column for both SPC and CP
                if 'target_description' in df.columns:
                    target_col = 'target_description'
                elif 'Metadata_annotated_target_description' in df.columns:
                    target_col = 'Metadata_annotated_target_description'
                else:
                    target_col = 'annotated_target_description'
                
                return create_search_result_content(
                    df=df,
                    search_value=selected_target,
                    search_column=target_col,  # Use detected column
                    search_type="Target Description",
                    add_label='add_label' in add_target_label,
                    label_text_source=target_col,  # Use detected column
                    scaling_mode=scaling_mode
                )
            return []
            
        except Exception as e:
            logger.error(f"Error displaying target description result: {str(e)}")
            return html.Div([
                html.H4("Error", style={'color': 'red'}),
                html.P(f"Could not display result: {str(e)}")
            ])
    
    
    @app.callback(
        Output('dmso-result-container', 'children'),
        [Input('dmso-button', 'n_clicks')],
        [State('scaling-mode', 'value'),
         State('add-label-checkbox', 'value'),
         State('label-type-radio', 'value'),
         State('plot-type-dropdown', 'value')],
        prevent_initial_call=True
    )
    def display_dmso_image(n_clicks: int, scaling_mode: str, 
                          add_label_checkbox: List[str], label_type: str,
                          plot_type: str) -> Any:
        """
        Display DMSO control images.
        
        Args:
            n_clicks: Number of button clicks
            scaling_mode: Image scaling mode
            add_label_checkbox: List containing 'add_label' if enabled
            label_type: Type of label to add
            
        Returns:
            HTML components for DMSO display
        """
        try:

            # SELECT DATAFRAME
            df = _get_dataframe_for_plot_type(plot_type)
            if df is None:
                return html.Div("No data available")
            
            Config = get_config()
            if not n_clicks:
                return []
            
            logger.info("Displaying DMSO control image")
            
            # Find DMSO treatments
            dmso_data = df[df['treatment'].str.contains('DMSO', case=False, na=False)]
            
            if dmso_data.empty:
                return html.Div([
                    html.H4("No DMSO Data Found", style={'color': '#e74c3c'}),
                    html.P("No treatments containing 'DMSO' were found in the dataset")
                ])
            
            # Get random DMSO sample
            random_row = dmso_data.sample(n=1).iloc[0]
            plate = random_row.get('plate', None)
            well = random_row.get('well', None)
            
            if not plate or not well:
                return html.Div([
                    html.H4("Missing Information", style={'color': '#e74c3c'}),
                    html.P("Missing plate or well information for DMSO sample")
                ])
            
            # Generate random site and find thumbnail
            site = random.randint(*Config.get_site_range())
            site_str = f"{site:02d}"
            
            thumbnail_path = find_thumbnail(plate, well, site_str, scaling_mode)
            
            if not thumbnail_path:
                return html.Div([
                    html.H4("Image Not Found", style={'color': '#e74c3c'}),
                    html.P(f"Could not find image for Plate: {plate}, Well: {well}")
                ])
            
            # Create image with potential label
            add_label = 'add_label' in add_label_checkbox
            label_text = ""
            if add_label:
                if label_type == 'treatment':
                    label_text = str(random_row.get('treatment', ''))
                elif label_type == 'moa_compound_uM':
                    label_text = str(random_row.get('moa_compound_uM', ''))
            
            img_src = add_text_to_image(thumbnail_path, label_text, add_label)
            
            if not img_src:
                return html.Div([
                    html.H4("Image Processing Error", style={'color': '#e74c3c'}),
                    html.P("Could not process the DMSO image")
                ])
            
            # Create DMSO display
            return _create_dmso_display(random_row, img_src, thumbnail_path, scaling_mode, label_text, add_label)
            
        except Exception as e:
            logger.error(f"Error displaying DMSO image: {str(e)}")
            return html.Div([
                html.H4("Error", style={'color': 'red'}),
                html.P(f"Could not display DMSO image: {str(e)}")
            ])
    
    
    # "Show Another" button callbacks
    @app.callback(
        Output('treatment-search-button', 'n_clicks'),
        [Input('show-another-treatment-button', 'n_clicks')],
        [State('treatment-search-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def trigger_new_treatment_search(another_clicks: int, current_clicks: int) -> int:
        """Trigger new treatment search when 'Show Another' is clicked."""
        if another_clicks:
            return current_clicks + 1
        return current_clicks
    
    
    @app.callback(
        Output('moa-search-button', 'n_clicks'),
        [Input('show-another-moa-button', 'n_clicks')],
        [State('moa-search-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def trigger_new_moa_search(another_clicks: int, current_clicks: int) -> int:
        """Trigger new MOA search when 'Show Another' is clicked."""
        if another_clicks:
            return current_clicks + 1
        return current_clicks
    
    
    @app.callback(
        Output('target-search-button', 'n_clicks'),
        [Input('show-another-target-button', 'n_clicks')],
        [State('target-search-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def trigger_new_target_search(another_clicks: int, current_clicks: int) -> int:
        """Trigger new target description search when 'Show Another' is clicked."""
        if another_clicks:
            return current_clicks + 1
        return current_clicks
    
    
    @app.callback(
        Output('dmso-button', 'n_clicks'),
        [Input('show-another-dmso-button', 'n_clicks')],
        [State('dmso-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def trigger_new_dmso_search(another_clicks: int, current_clicks: int) -> int:
        """Trigger new DMSO display when 'Show Another DMSO' is clicked."""
        if another_clicks:
            return current_clicks + 1
        return current_clicks


def create_search_result_content(df: pd.DataFrame, search_value: str, search_column: str,
                                search_type: str, add_label: bool, label_text_source: str,
                                scaling_mode: str) -> html.Div:
    """
    Create search result content for display.
    
    This function generates the HTML content for search results, including
    image display and metadata information.
    
    Args:
        df: Main dataframe
        search_value: Value being searched for
        search_column: Column to search in
        search_type: Type of search for display
        add_label: Whether to add text label
        label_text_source: Column to get label text from
        scaling_mode: Image scaling mode
        
    Returns:
        html.Div: Search result display component
    """
    try:
        Config = get_config()
        logger.debug(f"Creating search result for {search_type}: {search_value}")

        # Parse MOA value if it contains PP_ID_uM
        if search_type == "MOA" and '|' in search_value:
            search_value = search_value.split('|', 1)[0]  # Use just the MOA part
    
       # Find matching data - use PP_ID_uM if available, otherwise treatment
        if search_column == 'treatment' and 'PP_ID_uM' in df.columns:
            # Search in PP_ID_uM first, then fall back to treatment
            search_data = df[
                (df['PP_ID_uM'] == search_value) | 
                ((df['PP_ID_uM'].isna()) & (df['treatment'] == search_value))
            ]
        else:
            search_data = df[df[search_column] == search_value]
        
        if search_data.empty:
            return html.Div([
                html.H4(f"No Data Found", style={'color': '#e74c3c'}),
                html.P(f"No data found for {search_type}: {search_value}"),
                html.P("Please try another search term")
            ])
        
        # Get random sample
        random_row = search_data.sample(n=1).iloc[0]
        plate = random_row.get('plate', None)
        well = random_row.get('well', None)
        
        if not plate or not well:
            return html.Div([
                html.H4("Missing Information", style={'color': '#e74c3c'}),
                html.P(f"Missing plate or well information for {search_type}: {search_value}")
            ])
        
        # Generate random site and find thumbnail
        site = random.randint(*Config.get_site_range())
        site_str = f"{site:02d}"
        
        thumbnail_path = find_thumbnail(plate, well, site_str, scaling_mode)
        
        if not thumbnail_path:
            return html.Div([
                html.H4("Image Not Found", style={'color': '#e74c3c'}),
                html.P(f"Image not found for {search_type}: {search_value}"),
                html.P(f"Plate: {plate}, Well: {well}")
            ])
        
        # Create image with potential label with fallback
        if add_label:
            # Try primary label source first
            label_value = random_row.get(label_text_source, '')
            
            # If primary is NaN or empty, try fallback
            if pd.isna(label_value) or str(label_value).strip() == '':
                # For treatment search, fallback from PP_ID_uM to treatment
                if label_text_source == 'PP_ID_uM':
                    label_value = random_row.get('treatment', '')
                # For MOA search, fallback from moa_compound_uM to moa_first or moa
                elif label_text_source == 'moa_compound_uM':
                    label_value = random_row.get('moa_first', random_row.get('moa', ''))
                # For target description search, keep as is
                elif label_text_source == 'target_description':
                    label_value = random_row.get('target_description', '')
            
            label_text = str(label_value) if not pd.isna(label_value) else ""
        else:
            label_text = ""
        img_src = add_text_to_image(thumbnail_path, label_text, add_label)
        
        if not img_src:
            return html.Div([
                html.H4("Image Processing Error", style={'color': '#e74c3c'}),
                html.P(f"Could not process image for {search_type}: {search_value}")
            ])
        
        # Create display
        return _create_search_display(random_row, img_src, thumbnail_path, search_type, 
                                    search_value, scaling_mode, label_text, add_label)
        
    except Exception as e:
        logger.error(f"Error creating search result content: {str(e)}")
        return html.Div([
            html.H4("Error", style={'color': 'red'}),
            html.P(f"Error creating {search_type} result: {str(e)}")
        ])


def _get_moa_search_column(df: pd.DataFrame) -> str:
    """
    Determine which MOA column to use for searching.
    Works for both SPC and CellProfiler data.
    
    Args:
        df: Main dataframe
        
    Returns:
        str: Column name to use for MOA search
    """
    # NEW: Check for CP column first (Metadata_annotated_target_first is CP's MOA equivalent)
    if 'Metadata_annotated_target_first' in df.columns:
        return 'Metadata_annotated_target_first'
    # SPC columns (in priority order)
    elif 'moa_compound_uM' in df.columns:
        return 'moa_compound_uM'
    elif 'moa_first' in df.columns:
        return 'moa_first'
    else:
        return 'moa'


# In search_callbacks.py - Update the _create_search_display function

def _create_search_display(row_data: pd.Series, img_src: str, thumbnail_path,
                          search_type: str, search_value: str, scaling_mode: str,
                          label_text: str, add_label: bool) -> html.Div:
    """
    Create the search result display component.
    """
    # Extract site information
    site_num = extract_site_from_path(thumbnail_path) if thumbnail_path else "Unknown"
    
    # Extract data for display with proper fallbacks
    compound = row_data.get('PP_ID', '')
    if pd.isna(compound) or str(compound).strip() == '':
        compound = row_data.get('compound_name', 'Unknown')
    
    treatment = row_data.get('PP_ID_uM', '')
    if pd.isna(treatment) or str(treatment).strip() == '':
        treatment = row_data.get('treatment', 'Unknown')
    
    concentration = row_data.get('compound_uM', 'Unknown')
    concentration_text = f"{concentration} µM" if concentration not in [None, 'Unknown'] else "Unknown"
    library = row_data.get('library', 'Unknown')
    moa = row_data.get('moa', 'Unknown')
    plate = row_data.get('plate', 'Unknown')
    well = row_data.get('well', 'Unknown')
    target_description = row_data.get('target_description', 'Unknown')
    
    # NEW: Get chemical description and SMILES
    chemical_description = row_data.get('chemical_description', 'Unknown')
    smiles = row_data.get('SMILES', '')
    
    # Truncate target description for title if it's a target description search
    display_title = search_value
    if search_type == "Target Description" and search_value != "Unknown":
        words = str(search_value).split()
        if len(words) > 6:
            display_title = ' '.join(words[:6]) + "..."
        else:
            display_title = search_value
    
    # Create the main content
    main_content = [
        html.H4(f"{search_type}: {display_title}",
               style={'marginBottom': '10px', 'fontSize': '14px', 'color': '#2c3e50'}),
        
        # Image display
        html.Div([
            html.Img(
                src=img_src, 
                style={
                    'maxWidth': '100%', 
                    'maxHeight': '400px', 
                    'border': '1px solid #ddd',
                    'borderRadius': '4px'
                }
            ),
            html.P(f"Site: {site_num}", 
                  style={'fontSize': '10px', 'color': '#666', 'marginTop': '5px', 'marginBottom': '2px'}),
            html.P(f"Scaling: {'Auto-scaled' if scaling_mode == 'auto' else 'Fixed-scale'}", 
                  style={'fontSize': '10px', 'color': '#666', 'marginBottom': '2px'}),
            html.P(f"Label: {label_text if add_label and label_text else 'None'}", 
                  style={'fontSize': '10px', 'color': '#666', 'marginBottom': '5px'})
        ], style={'textAlign': 'center', 'marginBottom': '10px'}),
        
        # Metadata
        html.Div([
            html.P(f"Compound: {compound}", 
                  style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '12px'}),
            html.P(f"Treatment: {treatment}", 
                  style={'fontSize': '11px', 'marginBottom': '2px'}),
            html.P(f"Concentration: {concentration_text}", 
                  style={'fontSize': '11px', 'marginBottom': '2px'}),
            html.P(f"Plate: {plate}, Well: {well}", 
                  style={'fontSize': '11px', 'marginBottom': '2px'}),
            html.P(f"Library: {library}", 
                  style={'fontSize': '11px', 'marginBottom': '2px'}),
            html.P(f"MOA: {moa}", 
                  style={'fontSize': '11px', 'marginBottom': '2px'}),
            html.P(f"Target Description: {target_description}",
                  style={'fontSize': '11px', 'marginBottom': '2px'}),
            # NEW: Add chemical description
            html.P(f"Chemical Description: {chemical_description}",
                  style={'fontSize': '11px', 'marginBottom': '2px'})
        ])
    ]
    
    # NEW: Add SMILES structure at the bottom if available with higher resolution
    # CHANGE TO:
    Config = get_config()
    if Config.SHOW_CHEMICAL_STRUCTURES and smiles and str(smiles).strip() not in ['Unknown', 'nan', '', 'None']:
        structure_img = smiles_to_image_base64(smiles, width=300, height=300)  # INCREASED from 150x150
        
        if structure_img:
            main_content.append(
                html.Div([
                    html.P("Chemical Structure:", 
                        style={
                            'fontWeight': 'bold', 
                            'marginTop': '10px',
                            'marginBottom': '5px', 
                            'fontSize': '11px'
                        }),
                    html.Img(
                        src=structure_img,
                        style={
                            'maxWidth': '250px',  # INCREASED from 150px
                            'maxHeight': '250px',  # INCREASED from 150px
                            'border': '1px solid #ddd',
                            'borderRadius': '4px',
                            'backgroundColor': 'white'
                        }
                    ),
                    html.P(f"Chemical Description: {chemical_description}", 
                        style={
                            'fontSize': '9px', 
                            'color': '#666', 
                            'marginTop': '4px',
                            'wordBreak': 'break-word',
                            'lineHeight': '1.1'
                        })
                ], style={'textAlign': 'center', 'marginTop': '5px'})
            )
        else:
            # Show chemical description if structure generation failed
            main_content.append(
                html.Div([
                    html.P(f"Chemical Description: {chemical_description}", 
                        style={
                            'fontSize': '9px', 
                            'color': '#666', 
                            'marginTop': '5px',
                            'wordBreak': 'break-word',
                            'lineHeight': '1.1'
                        })
                ])
            )
    
    return html.Div(main_content)


def _create_dmso_display(row_data: pd.Series, img_src: str, thumbnail_path,
                        scaling_mode: str, label_text: str, add_label: bool) -> html.Div:
    """
    Create the DMSO control display component.
    
    Args:
        row_data: Data for the DMSO sample
        img_src: Base64 encoded image source
        thumbnail_path: Path to thumbnail file
        scaling_mode: Image scaling mode
        label_text: Label text
        add_label: Whether label is added
        
    Returns:
        html.Div: DMSO display component
    """
    # Extract site information
    site_num = extract_site_from_path(thumbnail_path) if thumbnail_path else "Unknown"
    
    # Extract data for display
    compound = row_data.get('PP_ID', row_data.get('compound_name', 'Unknown'))
    treatment = row_data.get('PP_ID_uM', row_data.get('treatment', 'Unknown'))
    concentration = row_data.get('compound_uM', 'Unknown')
    concentration_text = f"{concentration} µM" if concentration not in [None, 'Unknown'] else "Unknown"
    library = row_data.get('library', 'Unknown')
    moa = row_data.get('moa', 'Unknown')
    plate = row_data.get('plate', 'Unknown')
    well = row_data.get('well', 'Unknown')
    target_description = row_data.get('target_description', 'Unknown')  # NEW
    
    return html.Div([
        html.H4("DMSO Control Image", 
               style={'marginBottom': '10px', 'fontSize': '14px', 'color': '#e67e22'}),
        
        # Image display
        html.Div([
            html.Img(
                src=img_src, 
                style={
                    'maxWidth': '100%', 
                    'maxHeight': '400px', 
                    'border': '1px solid #ddd',
                    'borderRadius': '4px'
                }
            ),
            html.P(f"Site: {site_num}", 
                  style={'fontSize': '10px', 'color': '#666', 'marginTop': '5px', 'marginBottom': '2px'}),
            html.P(f"Scaling: {'Auto-scaled' if scaling_mode == 'auto' else 'Fixed-scale'}", 
                  style={'fontSize': '10px', 'color': '#666', 'marginBottom': '2px'}),
            html.P(f"Label: {label_text if add_label and label_text else 'None'}", 
                  style={'fontSize': '10px', 'color': '#666', 'marginBottom': '5px'})
        ], style={'textAlign': 'center', 'marginBottom': '10px'}),
        
        # Metadata
        html.Div([
            html.P(f"Compound: {compound}", 
                  style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '12px'}),
            html.P(f"Treatment: {treatment}", 
                  style={'fontSize': '11px', 'marginBottom': '2px'}),
            html.P(f"Concentration: {concentration_text}", 
                  style={'fontSize': '11px', 'marginBottom': '2px'}),
            html.P(f"Plate: {plate}, Well: {well}", 
                  style={'fontSize': '11px', 'marginBottom': '2px'}),
            html.P(f"Library: {library}", 
                  style={'fontSize': '11px', 'marginBottom': '2px'}),
            html.P(f"MOA: {moa}", 
                  style={'fontSize': '11px', 'marginBottom': '2px'}),
            html.P(f"Target Description: {target_description}",  # NEW LINE
                  style={'fontSize': '11px', 'marginBottom': '2px'})
        ])
    ])