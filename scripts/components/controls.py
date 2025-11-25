"""
Controls module for the Phenotype Clustering Interactive Visualization App.

This module creates the control panels and UI components for the dashboard,
including plot type selection, axis controls, color selection, point size
adjustment, and image labeling options.

Key Functions:
- create_control_panels(): Main function to create all control panels
- create_plot_controls(): Creates plot type and axis selection controls
- create_style_controls(): Creates color, size, and scaling controls
- create_labeling_controls(): Creates image labeling control panel

The controls module focuses on creating intuitive and responsive UI elements
that allow users to interact with and customize the visualization.
"""

from dash import html, dcc
from typing import List, Tuple, Any

try:
    from ..config_loader import get_config
    from ..utils.color_utils import create_color_options
except ImportError:
    from config_loader import get_config
    from utils.color_utils import create_color_options

def create_control_panels(available_color_columns: List[Tuple[str, bool, str, Any]],
                         plot_type_options: List[dict],
                         available_metrics: List[str]) -> html.Div:
    """
    Create all control panels for the dashboard.
    
    This function assembles all control components into organized panels,
    including plot controls, styling options, and labeling controls.
    
    Args:
        available_color_columns: Available color column configurations
        plot_type_options: Available plot type options
        available_metrics: Available metric columns for plotting
        
    Returns:
        html.Div: Complete control panels section
    """
    color_options = create_color_options(available_color_columns)
    
    return html.Div([
        # Plot type and axis controls
        create_plot_controls(plot_type_options, available_metrics),
        
        # Color, size, and scaling controls
        create_style_controls(color_options, available_color_columns),
        
        # Image labeling controls
        create_labeling_controls()
    ])


def create_plot_controls(plot_type_options: List[dict], 
                        available_metrics: List[str]) -> html.Div:
    """
    Create plot type and axis selection controls.
    
    Args:
        plot_type_options: Available plot type options
        available_metrics: Available metric columns for custom plots
        
    Returns:
        html.Div: Plot control panel
    """
    return html.Div([
        # Plot type selection
        html.Div([
            html.Label(
                "Plot Type:",
                style={
                    'fontWeight': 'bold',
                    'marginBottom': '5px',
                    'display': 'block'
                }
            ),
            dcc.Dropdown(
                id='plot-type-dropdown',
                options=plot_type_options,
                value=plot_type_options[0]['value'] if plot_type_options else None,
                clearable=False,
                style={'marginBottom': '10px'}
            )
        ], style={
            'width': '30%', 
            'display': 'inline-block', 
            'padding': '10px',
            'verticalAlign': 'top'
        }),
        
        # X-axis selection (for custom plots)
        html.Div([
            html.Label(
                "X-Axis (Custom):",
                style={
                    'fontWeight': 'bold',
                    'marginBottom': '5px',
                    'display': 'block'
                }
            ),
            dcc.Dropdown(
                id='x-axis-dropdown',
                options=[{'label': col, 'value': col} for col in available_metrics],
                value='UMAP1' if 'UMAP1' in available_metrics else (
                    available_metrics[0] if available_metrics else None
                ),
                disabled=True,
                style={'marginBottom': '10px'}
            )
        ], style={
            'width': '30%', 
            'display': 'inline-block', 
            'padding': '10px',
            'verticalAlign': 'top'
        }),
        
        # Y-axis selection (for custom plots)
        html.Div([
            html.Label(
                "Y-Axis (Custom):",
                style={
                    'fontWeight': 'bold',
                    'marginBottom': '5px',
                    'display': 'block'
                }
            ),
            dcc.Dropdown(
                id='y-axis-dropdown',
                options=[{'label': col, 'value': col} for col in available_metrics],
                value='UMAP2' if 'UMAP2' in available_metrics else (
                    available_metrics[1] if len(available_metrics) > 1 else None
                ),
                disabled=True,
                style={'marginBottom': '10px'}
            )
        ], style={
            'width': '30%', 
            'display': 'inline-block', 
            'padding': '10px',
            'verticalAlign': 'top'
        })
    ], style={
        'border': '1px solid #ddd',
        'borderRadius': '5px',
        'padding': '15px',
        'marginBottom': '20px',
        'backgroundColor': '#f8f9fa'
    })


def create_style_controls(color_options: List[dict], 
                         available_color_columns: List[Tuple]) -> html.Div:
    """
    Create styling controls for color, point size, and image scaling.
    
    Args:
        color_options: Available color options for dropdown
        available_color_columns: Available color column configurations
        
    Returns:
        html.Div: Style control panel
    """
    Config = get_config()
    return html.Div([
        # Color selection
        html.Div([
            html.Label(
                "Color By:",
                style={
                    'fontWeight': 'bold',
                    'marginBottom': '5px',
                    'display': 'block'
                }
            ),
            dcc.Dropdown(
                id='color-dropdown',
                options=color_options,
                value=available_color_columns[0][0] if available_color_columns else None,
                clearable=False,
                style={'marginBottom': '10px'}
            )
        ], style={
            'width': '30%', 
            'display': 'inline-block', 
            'padding': '10px',
            'verticalAlign': 'top'
        }),
        
        # Point size control
        html.Div([
            html.Label(
                "Point Size:",
                style={
                    'fontWeight': 'bold',
                    'marginBottom': '5px',
                    'display': 'block'
                }
            ),
            dcc.Slider(
                id='point-size-slider',
                min=Config.POINT_SIZE_MIN,
                max=Config.POINT_SIZE_MAX,
                value=Config.POINT_SIZE_DEFAULT,
                marks={i: str(i) for i in range(
                    Config.POINT_SIZE_MIN, 
                    Config.POINT_SIZE_MAX + 1, 
                    2
                )},
                step=1,
                tooltip={'placement': 'bottom', 'always_visible': False}
            )
        ], style={
            'width': '30%', 
            'display': 'inline-block', 
            'padding': '10px',
            'verticalAlign': 'top'
        }),
        
        # Image scaling toggle
        html.Div([
            html.Label(
                "Image Scaling:",
                style={
                    'fontWeight': 'bold',
                    'marginBottom': '5px',
                    'display': 'block'
                }
            ),
            dcc.RadioItems(
                id='scaling-mode',
                options=[
                    {
                        'label': 'Fixed Scale (Comparable)', 
                        'value': 'fixed'
                    },
                    {
                        'label': 'Auto Scale (Per Image)', 
                        'value': 'auto'
                    }
                ],
                value='fixed',
                inline=True,
                style={'marginTop': '5px'}
            )
        ], style={
            'width': '40%', 
            'display': 'inline-block', 
            'padding': '10px',
            'verticalAlign': 'top'
        })
    ], style={
        'border': '1px solid #ddd',
        'borderRadius': '5px',
        'padding': '15px',
        'marginBottom': '20px',
        'backgroundColor': '#f8f9fa'
    })


def create_labeling_controls() -> html.Div:
    """
    Create image labeling control panel.
    
    This panel allows users to add text labels to images and choose
    the type of label (treatment or MOA).
    
    Returns:
        html.Div: Labeling control panel
    """
    return html.Div([
        html.Div([
            html.Label(
                "Image Labeling Options:",
                style={
                    'fontWeight': 'bold',
                    'marginBottom': '10px',
                    'display': 'block'
                }
            ),
            
            # Enable/disable labeling checkbox
            dcc.Checklist(
                id='add-label-checkbox',
                options=[{
                    'label': 'Add treatment or MOA label to images', 
                    'value': 'add_label'
                }],
                value=[],
                style={'marginBottom': '10px'}
            ),
            
            # Label type selection
            html.Label(
                "Label Type:",
                style={
                    'fontWeight': 'bold',
                    'marginBottom': '5px',
                    'display': 'block'
                }
            ),
            dcc.RadioItems(
                id='label-type-radio',
                options=[
                    {'label': 'Treatment', 'value': 'treatment'},
                    {'label': 'MOA', 'value': 'moa_compound_uM'}
                ],
                value='treatment',
                inline=True,
                style={'marginTop': '5px'}
            )
        ], style={
            'width': '100%', 
            'display': 'inline-block', 
            'padding': '10px'
        })
    ], style={
        'marginTop': '10px',
        'marginBottom': '20px',
        'border': '1px solid #ddd',
        'padding': '15px',
        'borderRadius': '5px',
        'backgroundColor': '#f0f8ff'
    })


def create_landmark_analysis_controls() -> html.Div:
    """
    Create button to open landmark analysis modal.
    
    Returns:
        html.Div: Landmark analysis button
    """
    return html.Div([
        html.Button(
            "Open Landmark Distance Analysis: £100 per session",
            id='open-landmark-modal-btn',
            n_clicks=0,
            style={
                'padding': '12px 24px',
                'fontSize': '16px',
                'backgroundColor': '#4CAF50',
                'color': 'white',
                'border': 'none',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'fontWeight': 'bold',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.2)'
            }
        )
    ], style={
        'marginTop': '20px',
        'marginBottom': '20px',
        'textAlign': 'center'
    })


def create_info_display() -> html.Div:
    """
    Create an information display area for showing plot statistics.
    
    Returns:
        html.Div: Information display component
    """
    return html.Div([
        html.Div(
            id='plot-info-display',
            style={
                'padding': '10px',
                'border': '1px solid #ddd',
                'borderRadius': '5px',
                'backgroundColor': '#f9f9f9',
                'marginTop': '10px',
                'minHeight': '50px'
            }
        )
    ])


def create_advanced_controls() -> html.Div:
    """
    Create advanced control options (collapsible panel).
    
    Returns:
        html.Div: Advanced controls panel
    """
    return html.Div([
        html.Details([
            html.Summary(
                "Advanced Options",
                style={
                    'fontWeight': 'bold',
                    'cursor': 'pointer',
                    'padding': '10px',
                    'backgroundColor': '#e9ecef',
                    'borderRadius': '5px',
                    'marginBottom': '10px'
                }
            ),
            html.Div([
                # Plot size controls
                html.Label("Plot Height:"),
                dcc.Slider(
                    id='plot-height-slider',
                    min=400,
                    max=1000,
                    value=600,
                    marks={i: f"{i}px" for i in range(400, 1001, 200)},
                    step=50,
                    tooltip={'placement': 'bottom'}
                ),
                
                # Opacity control
                html.Label("Point Opacity:", style={'marginTop': '15px'}),
                dcc.Slider(
                    id='opacity-slider',
                    min=0.1,
                    max=1.0,
                    value=0.8,
                    marks={i/10: f"{i/10}" for i in range(1, 11, 2)},
                    step=0.1,
                    tooltip={'placement': 'bottom'}
                )
            ], style={'padding': '10px'})
        ])
    ], style={
        'marginTop': '20px',
        'border': '1px solid #ddd',
        'borderRadius': '5px',
        'backgroundColor': '#f8f9fa'
    })


def create_export_controls() -> html.Div:
    """
    Create export and download controls.
    
    Returns:
        html.Div: Export controls panel
    """
    return html.Div([
        html.H4("Export Options", style={'marginBottom': '10px'}),
        html.Button(
            "Download Plot as PNG",
            id='download-plot-button',
            className='btn btn-primary',
            style={
                'marginRight': '10px',
                'padding': '8px 16px',
                'backgroundColor': '#007bff',
                'color': 'white',
                'border': 'none',
                'borderRadius': '4px',
                'cursor': 'pointer'
            }
        ),
        html.Button(
            "Export Data",
            id='export-data-button',
            className='btn btn-secondary',
            style={
                'padding': '8px 16px',
                'backgroundColor': '#6c757d',
                'color': 'white',
                'border': 'none',
                'borderRadius': '4px',
                'cursor': 'pointer'
            }
        )
    ], style={
        'padding': '15px',
        'border': '1px solid #ddd',
        'borderRadius': '5px',
        'backgroundColor': '#f8f9fa',
        'marginTop': '20px'
    })