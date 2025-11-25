"""
Layout module for the Phenotype Clustering Interactive Visualization App.

This module defines the main application layout structure, organizing all
UI components into a cohesive dashboard interface. It creates the overall
page structure and positions major components like plots, controls, and
search interfaces.

Key Functions:
- create_layout(): Main function to create the complete app layout
- create_fixed_image_container(): Creates the hover image display overlay
- create_result_containers(): Creates the three search result containers
- create_main_plot_area(): Creates the primary visualization area
- create_landmark_analysis_section(): Creates the standalone landmark section

The layout module ensures consistent styling and responsive design across
all components while maintaining clean separation from business logic.
"""

from dash import html, dcc
import pandas as pd
from typing import List, Tuple, Any

try:
    from .controls import create_control_panels
    from .search import create_search_interface
    from .search import create_detailed_compound_search
    from ..config_loader import get_config
except ImportError:
    from components.controls import create_control_panels
    from components.search import create_search_interface
    from components.search import create_detailed_compound_search
    from config_loader import get_config

def create_layout(df: pd.DataFrame, 
                 available_color_columns: List[Tuple[str, bool, str, Any]],
                 plot_type_options: List[dict],
                 available_metrics: List[str]) -> html.Div:
    """
    Create the main application layout.
    
    This function assembles all UI components into the complete dashboard layout,
    including the header, controls, search interface, result containers, and
    main visualization area.
    
    Args:
        df: The loaded dataframe
        available_color_columns: Available color column configurations
        plot_type_options: Available plot type options
        available_metrics: Available metric columns for plotting
        
    Returns:
        html.Div: Complete application layout
    """
    return html.Div([
        # Application header
        _create_header(),
        
        # Fixed position image container for hover display
        _create_fixed_image_container(),
        
        # Data store for selected treatment
        dcc.Store(id='selected-treatment-store', data=None),
        
        # Data stores for landmark distance data
        dcc.Store(id='landmark-test-data-store', data=None),
        dcc.Store(id='landmark-reference-data-store', data=None),
        
        # Main control panels
        create_control_panels(
            available_color_columns=available_color_columns,
            plot_type_options=plot_type_options,
            available_metrics=available_metrics
        ),
        
        # Search interface
        create_search_interface(),
        
        # Result containers for search results
        _create_result_containers(),
        
        # Detailed compound search
        create_detailed_compound_search(),
        
        # Main plot area
        _create_main_plot_area(),
        
        # Tooltip and additional components
        _create_additional_components(),

        # Landmark analysis section (standalone, below main plot)
        _create_landmark_analysis_section(available_color_columns)
    ])


def _create_header() -> html.H1:
    """
    Create the application header.
    
    Returns:
        html.H1: Application title header
    """
    Config = get_config()
    return html.H1(
        Config.APP_TITLE,
        style={
            'textAlign': 'center',
            'marginBottom': '20px',
            'color': '#2c3e50',
            'fontFamily': 'Arial, sans-serif'
        }
    )


def _create_fixed_image_container() -> html.Div:
    """
    Create the fixed position container for hover image display.
    
    This container appears in the top-right corner when hovering over
    plot points, showing the corresponding microscopy image.
    
    Returns:
        html.Div: Fixed position image container with timeout components
    """
    return html.Div([
        # Main hover image container
        html.Div(
            id='fixed-image-container',
            style={
                'position': 'fixed', 
                'top': '-80px', 
                'right': '20px',
                'width': '720px',
                'backgroundColor': 'white',
                'padding': '10px',
                'borderRadius': '5px',
                'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)',
                'border': '1px solid #ddd',
                'zIndex': 1000,
                'display': 'none',
                'color': 'black',
                'maxHeight': '90vh',
                'overflowY': 'auto'
            },
            children=[]
        ),
        # Interval for auto-hide checking
        dcc.Interval(
            id='hover-timeout-interval',
            interval=500,  # Check every 500ms
            n_intervals=0
        ),
        # Store for last hover timestamp
        dcc.Store(id='last-hover-time', data=0)
    ])


def _create_result_containers() -> html.Div:
    """
    Create the four search result containers displayed side by side.
    
    These containers show the results from compound ID, MOA, target description,
    and DMSO searches with their corresponding microscopy images and metadata.
    
    Returns:
        html.Div: Container with four result panels
    """
    container_style = {
        'width': '22%',  # Reduced from 30% to fit 4 containers
        'padding': '10px',
        'marginTop': '20px',
        'marginBottom': '20px',
        'border': '1px solid #ddd',
        'borderRadius': '5px',
        'minHeight': '400px',
        'flexShrink': 0  # Prevent shrinking
    }
    
    return html.Div([
        # Left - Compound ID results
        html.Div(
            id='treatment-result-container',
            style=container_style
        ),
        
        # Center Left - MOA results  
        html.Div(
            id='moa-result-container',
            style=container_style
        ),
        
        # Center Right - Target Description results
        html.Div(
            id='target-result-container',
            style=container_style
        ),
        
        # Right - DMSO results
        html.Div(
            id='dmso-result-container',
            style=container_style
        )
    ], style={
    'display': 'flex', 
    'justifyContent': 'flex-start',
    'gap': '35px',  # Increased from 15px to 35px
    'marginTop': '20px',
    'overflowX': 'auto',
    'paddingBottom': '10px',
    'paddingLeft': '15px',  # Keep left position perfect
    'minWidth': '1200px'
})


# def _create_main_plot_area() -> html.Div:
#     """
#     Create the main plotting area with the interactive scatter plot.
    
#     Returns:
#         html.Div: Container with the main plot
#     """
#     Config = get_config()
#     return html.Div([
#         dcc.Graph(
#             id='main-plot', 
#             style={'height': Config.PLOT_HEIGHT, 'width': f'{Config.PLOT_WIDTH}px'},
#             clear_on_unhover=True,
#             config={
#                 'displayModeBar': True,
#                 'displaylogo': False,
#                 'modeBarButtonsToRemove': [
#                     'lasso2d', 'select2d', 'autoScale2d', 'hoverClosestCartesian'
#                 ],
#                 # ADD THIS LINE to enable drag panning
#                 'dragmode': 'pan'
#             }
#         )
#     ], style={
#         'marginTop': '20px',
#         'marginBottom': '20px'
#     })

def _create_main_plot_area() -> html.Div:
    """
    Create the main plotting area with the interactive scatter plot.
    Now includes a search box to highlight compounds on the plot.
    
    Returns:
        html.Div: Container with the main plot
    """
    Config = get_config()
    return html.Div([
        # Search box to highlight compound on plot
        html.Div([
            html.Label("Find compound on plot:", 
                      style={'fontWeight': 'bold', 'marginRight': '10px', 'fontSize': '14px'}),
            dcc.Dropdown(
                id='plot-search-dropdown',
                options=[],
                placeholder="Type treatment name or PP ID to highlight on plot...",
                multi=False,
                clearable=True,
                style={'width': '500px', 'display': 'inline-block', 'verticalAlign': 'middle'},
                persistence=False
            ),
            html.Button(
                '✕ Clear',
                id='clear-plot-highlight-btn',
                n_clicks=0,
                style={
                    'marginLeft': '10px',
                    'padding': '6px 12px',
                    'backgroundColor': '#dc3545',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontSize': '12px',
                    'verticalAlign': 'middle'
                }
            )
        ], style={
            'marginBottom': '15px',
            'padding': '10px',
            'backgroundColor': '#f0f8ff',
            'borderRadius': '5px',
            'border': '1px solid #bee5eb'
        }),
        
        # The main plot
        dcc.Graph(
            id='main-plot', 
            style={'height': f'{Config.PLOT_DEFAULT_HEIGHT}px', 'width': f'{Config.PLOT_WIDTH}px'},
            clear_on_unhover=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': [
                    'lasso2d', 'select2d', 'autoScale2d', 'hoverClosestCartesian'
                ],
                'dragmode': 'pan'
            }
        )
    ], style={
        'marginTop': '20px',
        'marginBottom': '20px'
    })


def _create_additional_components() -> html.Div:
    """
    Create additional components like tooltips and data stores.
    
    Returns:
        html.Div: Container with additional components
    """
    return html.Div([
        # Tooltip for hover information
        dcc.Tooltip(id='graph-tooltip', className='custom-tooltip'),
        
        # Store for tracking last clicked point
        dcc.Store(id='last-clicked-point', data=None),
        
        # Image display area for clicked points
        html.Div([
            html.Div(
                id='image-display', 
                style={
                    'marginTop': '20px',
                    'padding': '20px',
                    'border': '1px solid #ddd',
                    'borderRadius': '5px',
                    'backgroundColor': '#f9f9f9'
                }
            )
        ])
    ])


def _create_landmark_analysis_section(available_color_columns: List[Tuple]) -> html.Div:
    """
    Create the standalone landmark analysis section below the main plot.
    
    Returns:
        html.Div: Complete landmark analysis section
    """
    return html.Div([
        # Section divider
        html.Hr(style={'margin': '40px 0', 'border': '2px solid #e0e0e0'}),
        
        # Section header
        html.H2(
            "Landmark Distance Analysis",
            style={
                'textAlign': 'center',
                'color': '#2c3e50',
                'marginBottom': '30px',
                'fontFamily': 'Arial, sans-serif'
            }
        ),
        
        # Controls row
        html.Div([
            # Data type selector (SPC vs CP)
            html.Div([
                html.Label("Data Type:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Dropdown(
                    id='landmark-data-type-selector',
                    options=[
                        {'label': 'SPC (Machine Learning)', 'value': 'spc'},
                        {'label': 'CellProfiler', 'value': 'cp'}
                    ],
                    value='spc',
                    clearable=False,
                    style={'width': '250px'}
                )
            ], style={'display': 'inline-block', 'marginRight': '30px', 'verticalAlign': 'top'}),
            
            # Dataset selector
            html.Div([
                html.Label("Dataset:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Dropdown(
                    id='landmark-dataset-selector',
                    options=[
                        {'label': 'Test Compounds', 'value': 'test'},
                        {'label': 'Reference Compounds', 'value': 'reference'}
                    ],
                    value='test',
                    clearable=False,
                    style={'width': '250px'}
                )
            ], style={'display': 'inline-block', 'marginRight': '30px', 'verticalAlign': 'top'}),
            
            # Landmark selector
            html.Div([
                html.Label("Landmark:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Dropdown(
                    id='landmark-name-selector',
                    options=[],
                    value=None,
                    style={'width': '500px'}
                )
            ], style={'display': 'inline-block', 'marginRight': '30px', 'verticalAlign': 'top'}),
            
            # Color by selector
            html.Div([
                html.Label("Color by:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Dropdown(
                    id='landmark-color-selector',
                    options=[{'label': col[2], 'value': col[0]} 
                            for col in available_color_columns],
                    value='library',
                    clearable=False,
                    style={'width': '250px'}
                )
            ], style={'display': 'inline-block', 'marginRight': '30px', 'verticalAlign': 'top'}),
            
            # Point size slider
            html.Div([
                html.Label("Point size:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Slider(
                    id='landmark-size-slider',
                    min=2,
                    max=15,
                    step=1,
                    value=5,
                    marks={i: str(i) for i in range(2, 16, 2)},
                    tooltip={'placement': 'bottom'}
                )
            ], style={'display': 'inline-block', 'width': '250px', 'verticalAlign': 'top'})
        ], style={
            'padding': '20px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '5px',
            'border': '1px solid #ddd',
            'marginBottom': '20px'
        }),
        
        # Plot and info box container
        html.Div([
            # Plot container (left, wider)
            html.Div(
                id='landmark-plot-container',
                children=[
                    html.Div(
                        "Select a dataset and landmark to begin analysis",
                        style={
                            'padding': '100px 40px',
                            'textAlign': 'center',
                            'color': '#999',
                            'fontSize': '18px',
                            'fontStyle': 'italic'
                        }
                    )
                ],
                style={
                    'width': '70%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'padding': '10px'
                }
            ),
            
            # Info box (right, narrower)
            html.Div(
                id='landmark-info-box',
                children=[],
                style={
                    'width': '28%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'padding': '10px'
                }
            )
        ], style={
            'border': '1px solid #ddd',
            'borderRadius': '5px',
            'padding': '10px',
            'backgroundColor': 'white',
            'minHeight': '750px'
        }),
        
        # Hover image container for landmark plot
        html.Div(
            id='landmark-hover-image-container',
            style={'display': 'none'},
            children=[]
        )
        
    ], style={
        'marginTop': '50px',
        'marginBottom': '50px',
        'padding': '30px',
        'backgroundColor': '#fafafa',
        'borderRadius': '10px',
        'border': '2px solid #e0e0e0'
    })


def create_info_panel(df: pd.DataFrame) -> html.Div:
    """
    Create an information panel showing dataset statistics.
    
    This optional component can be added to provide users with
    an overview of the loaded data.
    
    Args:
        df: The loaded dataframe
        
    Returns:
        html.Div: Information panel
    """
    n_treatments = df['treatment'].nunique() if 'treatment' in df.columns else 0
    n_plates = df['plate'].nunique() if 'plate' in df.columns else 0
    n_wells = df['well'].nunique() if 'well' in df.columns else 0
    
    return html.Div([
        html.H4("Dataset Information", style={'marginBottom': '10px'}),
        html.P(f"Total data points: {len(df):,}"),
        html.P(f"Unique treatments: {n_treatments:,}"),
        html.P(f"Plates: {n_plates}"),
        html.P(f"Wells: {n_wells}"),
        html.P(f"Columns: {len(df.columns)}")
    ], style={
        'padding': '15px',
        'border': '1px solid #ddd',
        'borderRadius': '5px',
        'backgroundColor': '#f8f9fa',
        'marginTop': '20px'
    })


def create_help_modal() -> html.Div:
    """
    Create a help modal with usage instructions.
    
    Returns:
        html.Div: Help modal component
    """
    return html.Div([
        html.H3("How to Use This Dashboard"),
        html.Ul([
            html.Li("Use the plot controls to select different visualization types"),
            html.Li("Color points by different data attributes using the Color By dropdown"),
            html.Li("Hover over points to see images in the top-right corner"),
            html.Li("Click points to see detailed information below the plot"),
            html.Li("Use the search boxes to find specific compounds, MOAs, or DMSO controls"),
            html.Li("Toggle image scaling between fixed (comparable) and auto (per-image)"),
            html.Li("Add text labels to images using the labeling controls")
        ]),
        html.P("For questions or issues, please contact the development team.")
    ], style={
        'padding': '20px',
        'border': '1px solid #ddd',
        'borderRadius': '5px',
        'backgroundColor': '#e8f4f8',
        'marginTop': '20px',
        'display': 'none'  # Initially hidden
    }, id='help-modal')