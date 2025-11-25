"""
Search interface module for the Phenotype Clustering Interactive Visualization App.

This module creates the search interface components that allow users to search
for specific compounds, MOAs, and DMSO controls. It provides three distinct
search panels with their own controls and labeling options.

Key Functions:
- create_search_interface(): Main function to create the complete search interface
- create_compound_search(): Creates the compound ID search panel
- create_moa_search(): Creates the MOA search panel  
- create_dmso_search(): Creates the DMSO control search panel

The search interface provides an intuitive way for users to find and display
specific examples from their dataset with customizable image labeling options.
"""

from dash import html, dcc
from typing import Dict, Any

try:
    from ..config_loader import get_config
except ImportError:
    from config_loader import get_config

# In search.py - Update create_search_interface() function

def create_search_interface() -> html.Div:
    """
    Create the complete search interface with four search panels.
    
    Creates four side-by-side search panels for compound ID, MOA, target description,
    and DMSO searches, each with their own controls and labeling options.
    
    Returns:
        html.Div: Complete search interface
    """
    return html.Div([
        html.H3(
            "Search and Display Examples",
            style={
                'textAlign': 'center',
                'marginBottom': '20px',
                'color': '#2c3e50',
                'fontFamily': 'Arial, sans-serif'
            }
        ),
        
        html.Div([
            # Left - Compound ID Search
            create_compound_search(),
            
            # Center Left - MOA Search  
            create_moa_search(),
            
            # Center Right - Target Description Search
            create_target_search(),
            
            # Right - DMSO Search
            create_dmso_search()
        ], style={
            'display': 'flex', 
            'justifyContent': 'flex-start',  # Changed back from space-between
            'gap': '10px',  # Consistent with result containers
            'overflowX': 'auto',
            'paddingBottom': '10px',
            'width': '100%',
            'minWidth': '1200px'  # Match the result container minWidth
        })
    ], style={
        'marginTop': '20px',
        'marginBottom': '30px',
        'padding': '20px',
        'border': '2px solid #e9ecef',
        'borderRadius': '10px',
        'backgroundColor': '#f8f9fa',
        'width': '100%'
    })


# In search.py - Update create_target_search() with better dropdown styling

def create_target_search() -> html.Div:
    """
    Create the target description search panel.
    
    This panel allows users to search for specific target descriptions and display
    their corresponding microscopy images with optional labeling.
    
    Returns:
        html.Div: Target description search panel
    """
    return html.Div([
        # Header
        html.Label(
            "Search for Target Description:",
            style=_get_search_header_style()
        ),
        
        # Search dropdown with improved styling
        dcc.Dropdown(
            id='target-search-dropdown',
            options=[],
            placeholder="Type target description...",
            multi=False,
            clearable=True,
            persistence=True,
            persistence_type='session',
            style={
                'marginBottom': '10px',
                'fontSize': '10px',  # Smaller font
                'width': '100%'
            },
            # Improved dropdown properties
            optionHeight=280,  # Match the new min-height
            maxHeight=600,     # Increased to accommodate the massive options
            searchable=True,  # Ensure it's searchable
            className='target-description-dropdown'
        ),
        
        # Enhanced user hint
        html.P(
            "💡 Scroll horizontally in dropdown to read full descriptions",
            style={
                'fontSize': '9px',
                'color': '#666',
                'marginBottom': '10px',
                'fontStyle': 'italic',
                'lineHeight': '1.2'
            }
        ),
        
        # Labeling option
        html.Div([
            dcc.Checklist(
                id='add-target-label-checkbox',
                options=[{
                    'label': 'Add target description label to images', 
                    'value': 'add_label'
                }],
                value=[],
                style=_get_checkbox_style()
            )
        ], style={'marginBottom': '10px'}),
        
        # Action buttons
        html.Div([
            html.Button(
                'View Image',
                id='target-search-button',
                n_clicks=0,
                style=_get_primary_button_style()
            ),
            html.Button(
                'Show Another',
                id='show-another-target-button',
                n_clicks=0,
                style=_get_secondary_button_style()
            )
        ])
    ], style={
        **_get_search_panel_style(),
        'flexShrink': 0  # Prevent shrinking in flex container
    })


def create_compound_search() -> html.Div:
    """
    Create the compound ID search panel.
    
    This panel allows users to search for specific compound IDs and display
    their corresponding microscopy images with optional labeling.
    
    Returns:
        html.Div: Compound ID search panel
    """
    return html.Div([
        # Header
        html.Label(
            "Search for Compound ID:",
            style=_get_search_header_style()
        ),
        
        # Search dropdown
        dcc.Dropdown(
            id='treatment-search-dropdown',
            options=[],
            placeholder="Type compound ID...",
            multi=False,
            clearable=True,
            persistence=True,
            persistence_type='session',
            style={'marginBottom': '10px'}
        ),
        
        # Labeling option
        html.Div([
            dcc.Checklist(
                id='add-treatment-label-checkbox',
                options=[{
                    'label': 'Add compound ID label to images', 
                    'value': 'add_label'
                }],
                value=[],
                style=_get_checkbox_style()
            )
        ], style={'marginBottom': '10px'}),
        
        # Action buttons
        html.Div([
            html.Button(
                'View Image',
                id='treatment-search-button',
                n_clicks=0,
                style=_get_primary_button_style()
            ),
            html.Button(
                'Show Another',
                id='show-another-treatment-button',
                n_clicks=0,
                style=_get_secondary_button_style()
            )
        ])
    ], style=_get_search_panel_style())


def create_moa_search() -> html.Div:
    """
    Create the MOA (Mechanism of Action) search panel.
    
    This panel allows users to search for specific MOAs and display
    corresponding microscopy images with optional labeling.
    
    Returns:
        html.Div: MOA search panel
    """
    return html.Div([
        # Header
        html.Label(
            "Search for MOA:",
            style=_get_search_header_style()
        ),
        
        # Search dropdown
        dcc.Dropdown(
            id='moa-search-dropdown',
            options=[],
            placeholder="Type MOA...",
            multi=False,
            clearable=True,
            persistence=True,
            persistence_type='session',
            style={'marginBottom': '10px'}
        ),
        
        # Labeling option
        html.Div([
            dcc.Checklist(
                id='add-moa-label-checkbox',
                options=[{
                    'label': 'Add MOA label to images', 
                    'value': 'add_label'
                }],
                value=[],
                style=_get_checkbox_style()
            )
        ], style={'marginBottom': '10px'}),
        
        # Action buttons
        html.Div([
            html.Button(
                'View Image',
                id='moa-search-button',
                n_clicks=0,
                style=_get_primary_button_style()
            ),
            html.Button(
                'Show Another',
                id='show-another-moa-button',
                n_clicks=0,
                style=_get_secondary_button_style()
            )
        ])
    ], style=_get_search_panel_style())


def create_dmso_search() -> html.Div:
    """
    Create the DMSO control search panel.
    
    This panel allows users to display random DMSO control images
    for comparison purposes.
    
    Returns:
        html.Div: DMSO search panel
    """
    return html.Div([
        # Header
        html.Label(
            "DMSO Control:",
            style=_get_search_header_style()
        ),
        
        # Description
        html.P(
            "Display random DMSO control image",
            style={
                'fontSize': '12px',
                'marginBottom': '15px',
                'color': '#6c757d',
                'fontStyle': 'italic'
            }
        ),
        
        # Action buttons
        html.Div([
            html.Button(
                'Display DMSO Image',
                id='dmso-button',
                n_clicks=0,
                style={
                    **_get_primary_button_style(),
                    'backgroundColor': '#FF9800',
                    'marginBottom': '10px',
                    'width': '100%'
                }
            ),
            html.Button(
                'Show Another DMSO',
                id='show-another-dmso-button',
                n_clicks=0,
                style={
                    **_get_secondary_button_style(),
                    'backgroundColor': '#FF6600',
                    'width': '100%'
                }
            )
        ])
    ], style=_get_search_panel_style())


def _get_search_panel_style() -> Dict[str, Any]:
    """
    Get the standard styling for search panels.
    """
    return {
        'width': '22%',  # Match the result container width exactly
        'minWidth': '22%',  # Ensure consistency
        'maxWidth': '22%',  # Lock the width
        'padding': '15px',
        'border': '1px solid #ddd',
        'borderRadius': '8px',
        'backgroundColor': 'white',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'minHeight': '200px',
        'flexShrink': 0
    }




def create_detailed_compound_search() -> html.Div:
    """
    Create the detailed compound search panel.
    
    This panel allows users to search for compounds by either treatment name or PP_ID
    and displays detailed information exactly like clicking on a UMAP point, including:
    - Full microscopy image
    - Basic compound information
    - Metrics
    - Landmark information
    - compound_type (if available)
    
    Returns:
        html.Div: Detailed compound search panel
    """
    return html.Div([
        html.H3(
            "Detailed Perturbation Search",
            style={
                'textAlign': 'center',
                'marginBottom': '15px',
                'marginTop': '30px',
                'color': '#2c3e50',
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '22px'
            }
        ),
        
        html.P(
            "Search by treatment name (e.g., GSK_ID@0.1) or PP/CR/CG/SC/CC/JP number (e.g., PP001234)",
            style={
                'textAlign': 'center',
                'fontSize': '12px',
                'color': '#666',
                'marginBottom': '15px',
                'fontStyle': 'italic'
            }
        ),
        
        html.Div([
            # Search dropdown
            dcc.Dropdown(
                id='detailed-compound-search-dropdown',
                options=[],
                placeholder="Type treatment name or PP number...",
                multi=False,
                clearable=True,
                persistence=True,
                persistence_type='session',
                style={
                    'marginBottom': '15px',
                    'fontSize': '14px',
                    'width': '100%'
                }
            ),
            
            # Action button
            html.Button(
                'View Detailed Information',
                id='detailed-compound-search-button',
                n_clicks=0,
                style={
                    'backgroundColor': '#007bff',
                    'color': 'white',
                    'padding': '10px 20px',
                    'border': 'none',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontSize': '14px',
                    'fontWeight': 'bold',
                    'width': '100%',
                    'transition': 'background-color 0.3s'
                }
            ),
            
            # Result container (will be populated by callback)
            html.Div(
                id='detailed-compound-search-result',
                style={
                    'marginTop': '20px',
                    'padding': '20px',
                    'border': '1px solid #ddd',
                    'borderRadius': '5px',
                    'backgroundColor': '#f9f9f9',
                    'minHeight': '100px'
                }
            )
        ], style={
            'maxWidth': '1200px',
            'margin': '0 auto',
            'padding': '20px',
            'border': '2px solid #007bff',
            'borderRadius': '8px',
            'backgroundColor': '#f0f8ff'
        })
    ], style={
        'marginTop': '30px',
        'marginBottom': '30px',
        'padding': '20px 0'
    })



def _get_search_header_style() -> Dict[str, Any]:
    """
    Get the standard styling for search panel headers.
    
    Returns:
        Dict[str, Any]: Style dictionary for headers
    """
    return {
        'fontWeight': 'bold',
        'marginBottom': '10px',
        'display': 'block',
        'color': '#2c3e50',
        'fontSize': '14px'
    }


def _get_primary_button_style() -> Dict[str, Any]:
    """
    Get the standard styling for primary action buttons.
    
    Returns:
        Dict[str, Any]: Style dictionary for primary buttons
    """
    return {
        'backgroundColor': '#4CAF50',
        'color': 'white',
        'padding': '8px 16px',
        'border': 'none',
        'borderRadius': '4px',
        'cursor': 'pointer',
        'marginRight': '8px',
        'fontSize': '12px',
        'fontWeight': 'bold',
        'transition': 'background-color 0.3s ease'
    }


def _get_secondary_button_style() -> Dict[str, Any]:
    """
    Get the standard styling for secondary action buttons.
    
    Returns:
        Dict[str, Any]: Style dictionary for secondary buttons
    """
    return {
        'backgroundColor': '#2196F3',
        'color': 'white',
        'padding': '8px 16px',
        'border': 'none',
        'borderRadius': '4px',
        'cursor': 'pointer',
        'fontSize': '12px',
        'fontWeight': 'bold',
        'transition': 'background-color 0.3s ease'
    }


def _get_checkbox_style() -> Dict[str, Any]:
    """
    Get the standard styling for checkboxes.
    
    Returns:
        Dict[str, Any]: Style dictionary for checkboxes
    """
    return {
        'marginBottom': '5px',
        'fontSize': '11px',
        'color': '#495057'
    }


def create_search_help() -> html.Div:
    """
    Create a help section explaining how to use the search interface.
    
    Returns:
        html.Div: Search help component
    """
    return html.Div([
        html.H4("How to Use Search", style={'marginBottom': '10px'}),
        html.Ul([
            html.Li("Compound ID: Search for specific compound identifiers"),
            html.Li("MOA: Search for mechanisms of action"),
            html.Li("DMSO: Display random control images for comparison"),
            html.Li("Use 'Show Another' to see different examples of the same type"),
            html.Li("Enable labels to add text overlays to images"),
            html.Li("All three search results can be displayed simultaneously")
        ], style={'fontSize': '12px', 'color': '#6c757d'}),
    ], style={
        'marginTop': '15px',
        'padding': '15px',
        'border': '1px solid #e9ecef',
        'borderRadius': '5px',
        'backgroundColor': '#f8f9fa'
    })


def create_search_stats() -> html.Div:
    """
    Create a component to display search statistics.
    
    Returns:
        html.Div: Search statistics component
    """
    return html.Div([
        html.Div(
            id='search-stats-display',
            style={
                'padding': '10px',
                'border': '1px solid #ddd',
                'borderRadius': '5px',
                'backgroundColor': '#f8f9fa',
                'marginTop': '15px',
                'fontSize': '12px',
                'color': '#6c757d'
            }
        )
    ])


def create_batch_search() -> html.Div:
    """
    Create an advanced batch search interface (future enhancement).
    
    Returns:
        html.Div: Batch search component
    """
    return html.Div([
        html.H4("Batch Search", style={'marginBottom': '10px'}),
        html.P("Upload a list of compound IDs or MOAs to search multiple items at once."),
        dcc.Upload(
            id='batch-upload',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px 0',
                'cursor': 'pointer'
            },
            multiple=False
        ),
        html.Button(
            'Process Batch',
            id='batch-process-button',
            n_clicks=0,
            disabled=True,
            style={
                'backgroundColor': '#6c757d',
                'color': 'white',
                'padding': '8px 16px',
                'border': 'none',
                'borderRadius': '4px',
                'cursor': 'not-allowed'
            }
        )
    ], style={
        'marginTop': '20px',
        'padding': '15px',
        'border': '1px solid #ddd',
        'borderRadius': '5px',
        'backgroundColor': '#f8f9fa',
        'display': 'none'  # Hidden by default
    })