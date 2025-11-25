"""
Main application entry point for the Phenotype Clustering Interactive Visualization App.

UPDATED (2025-01-21): Now loads and manages BOTH SPC and CellProfiler datasets!

This is the primary entry point for the application. It orchestrates the loading of BOTH
datasets, creation of the Dash app, registration of callbacks, and server startup. The 
application provides an interactive dashboard for exploring phenotype clustering data 
with integrated microscopy image display for both SPC and CellProfiler data.

Key Functions:
- create_app(): Main function to create and configure the Dash application
- main(): Entry point that handles data loading and app startup
- setup_logging(): Configure application logging
- validate_environment(): Check system requirements and paths

Usage:
    Run directly from command line:
        python main.py
    
    Or import and run programmatically:
        from main import create_app, main
        main()

The application will start a web server (default: http://localhost:8090) that serves
the interactive dashboard interface with dual dataset support.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional
import traceback

# Add the project root to the path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import application modules
try:
    # Try relative imports first (when run as module)
    from config_loader import get_config
    from data import load_both_datasets, DataLoadError
    from components import create_layout
    from utils.color_utils import filter_available_color_columns, create_color_options
    from callbacks import register_plot_callbacks, register_image_callbacks, register_search_callbacks
    from callbacks.detailed_search_callbacks import register_detailed_search_callbacks
    from callbacks.landmark_callbacks import register_landmark_callbacks
except ImportError:
    # Fall back to absolute imports (when run directly)
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config_loader import get_config
    from data import load_both_datasets, DataLoadError
    from components import create_layout
    from utils.color_utils import filter_available_color_columns, create_color_options
    from callbacks import register_plot_callbacks, register_image_callbacks, register_search_callbacks
    from callbacks.landmark_callbacks import register_landmark_callbacks

# Dash imports
from dash import Dash

# Set up logging
def setup_logging(debug_mode: bool = True) -> None:
    """Configure application logging with appropriate levels and formatting."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if debug_mode:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

def validate_environment(Config) -> bool:
    """
    Validate that the environment is properly configured for the application.
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Validating application environment...")
    
    # Check configuration paths - NOW CHECKING FOR BOTH FILES IN data/ subdirectory
    paths_valid = True
    
    # Check SPC paths (files are in data/ subdirectory)
    spc_viz_file = Path(Config.SPC_ANALYSIS_DIR) / "data" / "spc_for_viz_app.csv"
    if not spc_viz_file.exists():
        logger.warning(f"SPC visualization file not found: {spc_viz_file}")
        paths_valid = False
    else:
        logger.info(f"✓ Found SPC file: {spc_viz_file}")
    
    # Check CP paths (files are in data/ subdirectory)
    cp_viz_file = Path(Config.CP_ANALYSIS_DIR) / "data" / "cp_for_viz_app.csv"
    if not cp_viz_file.exists():
        logger.warning(f"CP visualization file not found: {cp_viz_file}")
        paths_valid = False
    else:
        logger.info(f"✓ Found CP file: {cp_viz_file}")
    
    # At least ONE dataset must be available
    if not spc_viz_file.exists() and not cp_viz_file.exists():
        logger.error("Path validation failed - NEITHER SPC nor CP visualization files found")
        return False
    
    if not paths_valid:
        logger.warning("Some files missing, but at least one dataset is available - continuing...")
    
    return True  # ← Make sure to return True here!


def create_app(df_spc, df_cp, Config) -> Dash:
    """
    Create and configure the Dash application with BOTH datasets.
    
    Args:
        df_spc: SPC dataframe
        df_cp: CellProfiler dataframe
        Config: Configuration object
    
    Returns:
        Dash: Configured Dash application
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Creating Dash application...")
    
    # Create Dash app with custom configuration
    app = Dash(
        __name__, 
        suppress_callback_exceptions=True,
        title=Config.APP_TITLE,
        update_title="Loading...",
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"},
            {"name": "description", "content": "Interactive phenotype clustering visualisation dashboard"},
        ],
        external_stylesheets=[
            'https://codepen.io/chriddyp/pen/bWLwgP.css'
        ]
    )

    # Add custom CSS styling
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                /* Target dropdown specific styling */
                .target-description-dropdown .Select-menu-outer {
                    max-height: 200px !important;
                    overflow-y: auto !important;
                    z-index: 9999 !important;
                    border: 1px solid #ccc !important;
                    border-radius: 4px !important;
                }
                
                .target-description-dropdown .Select-option {
                    white-space: nowrap !important;
                    overflow-x: auto !important;
                    overflow-y: hidden !important;
                    padding: 100px 80px !important;
                    margin: 32px 0 !important;
                    line-height: 3.0 !important;
                    border-bottom: 1px solid #f0f0f0 !important;
                    cursor: pointer !important;
                    font-size: 10px !important;
                    min-height: 280px !important;
                    display: flex !important;
                    align-items: center !important;
                }
                                            
                .target-description-dropdown .Select-option:hover {
                    background-color: #f5f5f5 !important;
                }
                
                .target-description-dropdown .Select-option:last-child {
                    border-bottom: none !important;
                }
                
                /* Custom scrollbar for horizontal scrolling */
                .target-description-dropdown .Select-option::-webkit-scrollbar {
                    height: 8px;
                }
                
                .target-description-dropdown .Select-option::-webkit-scrollbar-track {
                    background: #f1f1f1;
                    border-radius: 4px;
                }
                
                .target-description-dropdown .Select-option::-webkit-scrollbar-thumb {
                    background: #c1c1c1;
                    border-radius: 4px;
                }
                
                .target-description-dropdown .Select-option::-webkit-scrollbar-thumb:hover {
                    background: #a1a1a1;
                }
                
                /* Ensure dropdown container can accommodate horizontal scroll */
                .target-description-dropdown .Select-menu {
                    min-width: 320px !important;
                    max-width: 700px !important;
                }
                
                /* Modern Dash dropdown styling (for newer versions) */
                .target-description-dropdown .dash-dropdown .Select-menu-outer,
                .target-description-dropdown .dash-dropdown .dropdown-menu {
                    max-height: 200px !important;
                    overflow-y: auto !important;
                }
                
               .target-description-dropdown .dash-dropdown .Select-option,
               .target-description-dropdown .dash-dropdown .dropdown-item {
                    white-space: nowrap !important;
                    overflow-x: auto !important;
                    overflow-y: hidden !important;
                    padding: 100px 80px !important;
                    margin: 32px 0 !important;
                    line-height: 3.0 !important;
                    border-bottom: 1px solid #f0f0f0 !important;
                    min-height: 280px !important;
                    display: flex !important;
                    align-items: center !important;
                }
                
                /* Fix for dropdown input field */
                .target-description-dropdown .Select-input input {
                    font-size: 10px !important;
                    padding: 5px !important;
                }
                
                .target-description-dropdown .Select-placeholder {
                    font-size: 10px !important;
                    color: #999 !important;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Prepare data for UI components - USE SPC AS PRIMARY FOR NOW
    # (Callbacks will switch between df_spc and df_cp based on user selection)
    logger.info("Preparing UI configuration using SPC as primary...")
    df_primary = df_spc if df_spc is not None else df_cp
    
    if df_primary is None:
        logger.error("Both dataframes are None - cannot create app")
        raise ValueError("No valid dataframes to create application")
    
    # Get available metrics for plotting
    available_metrics = [
        col for col in df_primary.columns 
        if any(pattern in col for pattern in Config.METRIC_COLUMN_PATTERNS)
    ]
    logger.info(f"Found {len(available_metrics)} metric columns for plotting")
    
    # Get available color columns
    available_color_columns = filter_available_color_columns(df_primary)
    logger.info(f"Found {len(available_color_columns)} color columns")
    
    # Define plot types - will work for both SPC and CP
    plot_types = [
        # SPC plots
        {'label': 'UMAP (SPC)', 'value': 'umap_spc', 'x': 'UMAP1', 'y': 'UMAP2', 
         'x_title': 'UMAP Component 1 (SPC)', 'y_title': 'UMAP Component 2 (SPC)'},
        {'label': 't-SNE (SPC)', 'value': 'tsne_spc', 'x': 'TSNE1', 'y': 'TSNE2', 
         'x_title': 't-SNE Component 1 (SPC)', 'y_title': 't-SNE Component 2 (SPC)'},
        
        # CellProfiler plots - Check both datasets for column names
        {'label': 'UMAP (CellProfiler)', 'value': 'umap_cp', 
         'x': 'umap_n15_d0.1_x', 'y': 'umap_n15_d0.1_y', 
         'x_title': 'UMAP Component 1 (CellProfiler)', 'y_title': 'UMAP Component 2 (CellProfiler)'},
        {'label': 't-SNE (CellProfiler)', 'value': 'tsne_cp', 
         'x': 'tsne_p30_x', 'y': 'tsne_p30_y', 
         'x_title': 't-SNE Component 1 (CellProfiler)', 'y_title': 't-SNE Component 2 (CellProfiler)'},
        
        # Metric plots
        {'label': 'DMSO vs MAD', 'value': 'dmso_mad', 'x': 'cosine_distance_from_dmso', 'y': 'mad_cosine',
         'x_title': 'Cosine Distance from DMSO', 'y_title': 'MAD Cosine'},
        {'label': 'DMSO vs Variance', 'value': 'dmso_var', 'x': 'cosine_distance_from_dmso', 'y': 'var_cosine',
         'x_title': 'Cosine Distance from DMSO', 'y_title': 'Variance Cosine'},
        {'label': 'DMSO vs StdDev', 'value': 'dmso_std', 'x': 'cosine_distance_from_dmso', 'y': 'std_cosine',
         'x_title': 'Cosine Distance from DMSO', 'y_title': 'Standard Deviation Cosine'},

        # Landmark distance plots
        {'label': '1st Landmark Distance vs DMSO Distance',
        'value': 'landmark1_vs_dmso',
        'x': 'cosine_distance_from_dmso',
        'y': 'closest_landmark_distance',
        'x_title': 'Distance from DMSO',
        'y_title': '1st Closest Landmark Distance'},

        {'label': '2nd Landmark Distance vs DMSO Distance',
        'value': 'landmark2_vs_dmso',
        'x': 'cosine_distance_from_dmso',
        'y': 'second_closest_landmark_distance',
        'x_title': 'Distance from DMSO',
        'y_title': '2nd Closest Landmark Distance'},

        {'label': '3rd Landmark Distance vs DMSO Distance',
        'value': 'landmark3_vs_dmso',
        'x': 'cosine_distance_from_dmso',
        'y': 'third_closest_landmark_distance',
        'x_title': 'Distance from DMSO',
        'y_title': '3rd Closest Landmark Distance'},
        
        # Custom plot option
        {'label': 'Custom', 'value': 'custom', 'x': None, 'y': None}
    ]
    
    # Filter plot types to only include those with available data in EITHER dataset
    filtered_plot_types = []
    for plot in plot_types:
        if plot['value'] == 'custom':
            filtered_plot_types.append(plot)
        elif plot['x'] and plot['y']:
            # Check if columns exist in EITHER dataset
            spc_has = (df_spc is not None and plot['x'] in df_spc.columns and plot['y'] in df_spc.columns)
            cp_has = (df_cp is not None and plot['x'] in df_cp.columns and plot['y'] in df_cp.columns)
            
            if spc_has or cp_has:
                filtered_plot_types.append(plot)
            else:
                logger.debug(f"Skipping plot type '{plot['label']}' - missing columns in both datasets")
    
    plot_type_options = [{'label': p['label'], 'value': p['value']} for p in filtered_plot_types]
    logger.info(f"Available plot types: {[p['label'] for p in filtered_plot_types]}")
    
    # Create application layout
    logger.info("Creating application layout...")
    
    try:
        layout = create_layout(
            df=df_primary,  # Pass primary df for initial layout creation
            available_color_columns=available_color_columns,
            plot_type_options=plot_type_options,
            available_metrics=available_metrics
        )
        
        if layout is None:
            logger.error("create_layout returned None!")
            raise ValueError("Layout creation failed - returned None")
        
        logger.info(f"Layout created successfully, type: {type(layout)}")
        app.layout = layout
        
    except Exception as e:
        logger.error(f"Error creating layout: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create a simple fallback layout for debugging
        from dash import html
        logger.info("Creating fallback layout...")
        app.layout = html.Div([
            html.H1("Fallback Layout - Layout Creation Failed"),
            html.P(f"Error: {str(e)}"),
            html.P("Check the logs for more details.")
        ])
        logger.info("Fallback layout created")
    
    # Register all callbacks with BOTH dataframes
    logger.info("Registering application callbacks...")
    
    # Hover data columns for callbacks - use Config.HOVER_DATA_COLUMNS directly
    # The config already has all the columns we want, so just filter to what exists
    hover_data_cols = [col for col in Config.HOVER_DATA_COLUMNS if col in df_primary.columns]
    
    # Add SMILES and gene_description if not already there
    for col in ['SMILES', 'Metadata_SMILES', 'gene_description']:
        if col in df_primary.columns and col not in hover_data_cols:
            hover_data_cols.append(col)
    
    logger.info(f" Built hover_data_cols with {len(hover_data_cols)} columns from Config.HOVER_DATA_COLUMNS")
    
    # Log any missing columns for debugging
    missing_cols = [col for col in Config.HOVER_DATA_COLUMNS if col not in df_primary.columns]
    if missing_cols:
        logger.debug(f"⚠️  {len(missing_cols)} columns from HOVER_DATA_COLUMNS not in primary dataframe")
    
    try:
        # Register callbacks - NOW WITH BOTH DATAFRAMES
        register_plot_callbacks(
            app=app, 
            df_spc=df_spc,  # Pass SPC dataframe
            df_cp=df_cp,    # Pass CP dataframe
            available_color_columns=available_color_columns,
            plot_types=plot_types,
            available_metrics=available_metrics
        )
        
        register_image_callbacks(
            app=app, 
            df_spc=df_spc,  # Pass SPC dataframe
            df_cp=df_cp,    # Pass CP dataframe
            hover_data_cols=hover_data_cols
        )
        
        register_search_callbacks(
            app=app, 
            df_spc=df_spc,  # Pass SPC dataframe
            df_cp=df_cp     # Pass CP dataframe
        )
        
        register_detailed_search_callbacks(
            app, 
            df_spc=df_spc,  # Pass SPC dataframe
            df_cp=df_cp     # Pass CP dataframe
        ) 
        
        # Register landmark analysis callbacks - UPDATED to pass both dataframes
        register_landmark_callbacks(app, df_spc, df_cp)  # Pass both for merger with viz data
        
        logger.info("All callbacks registered successfully")
        
    except Exception as e:
        logger.error(f"Error registering callbacks: {str(e)}")
        logger.error(traceback.format_exc())
        logger.warning("Continuing with app creation despite callback errors")
    
    logger.info("Dash application created successfully")
    return app


def print_startup_info(df_spc, df_cp, Config) -> None:
    """
    Print startup information and data summary for BOTH datasets.
    
    Args:
        df_spc: SPC dataframe
        df_cp: CellProfiler dataframe
        Config: Configuration object
    """
    logger = logging.getLogger(__name__)
    
    print(f"\n{'='*60}")
    print(f"{Config.APP_TITLE}")
    print(f"{'='*60}")
    
    # SPC Dataset Info
    if df_spc is not None:
        print(f"\n SPC Dataset Information:")
        print(f"   • Total data points: {len(df_spc):,}")
        print(f"   • Columns: {len(df_spc.columns)}")
        
        if 'treatment' in df_spc.columns:
            print(f"   • Unique treatments: {df_spc['treatment'].nunique():,}")
        if 'plate' in df_spc.columns:
            print(f"   • Plates: {df_spc['plate'].nunique()}")
        if 'well' in df_spc.columns:
            print(f"   • Wells: {df_spc['well'].nunique():,}")
    else:
        print(f"\n SPC Dataset: NOT LOADED")
    
    # CP Dataset Info
    if df_cp is not None:
        print(f"\n CellProfiler Dataset Information:")
        print(f"   • Total data points: {len(df_cp):,}")
        print(f"   • Columns: {len(df_cp.columns)}")
        
        if 'treatment' in df_cp.columns:
            print(f"   • Unique treatments: {df_cp['treatment'].nunique():,}")
        if 'Metadata_plate_barcode' in df_cp.columns:
            print(f"   • Plates: {df_cp['Metadata_plate_barcode'].nunique()}")
        if 'Metadata_well' in df_cp.columns:
            print(f"   • Wells: {df_cp['Metadata_well'].nunique():,}")
    else:
        print(f"\n CellProfiler Dataset: NOT LOADED")
    
    print(f"\n  Server Information:")
    print(f"   • URL: http://{Config.HOST}:{Config.PORT}")
    print(f"   • Debug mode: {'Enabled' if Config.DEBUG else 'Disabled'}")
    
    print(f"\n Data Paths:")
    print(f"   • SPC directory: {Config.SPC_ANALYSIS_DIR}")
    print(f"   • CP directory: {Config.CP_ANALYSIS_DIR}")
    print(f"   • Thumbnail directory: {Config.THUMBNAIL_DIRS}")
    
    print(f"\n{'='*60}")
    print("✨ Starting server... (Ctrl+C to stop)")
    print(f"{'='*60}\n")


def main() -> None:
    """
    Main application entry point.
    
    This function orchestrates the complete application startup process:
    1. Configure logging
    2. Load configuration once
    3. Validate environment
    4. Load BOTH datasets (SPC + CP)
    5. Create Dash application
    6. Start the web server
    """
    # Setup logging first with default debug mode
    setup_logging(debug_mode=True)
    logger = logging.getLogger(__name__)
    
    try:
        # Get config with interactive selection - ONLY ONCE
        Config = get_config()
        
        # Update logging level based on config
        if not Config.DEBUG:
            logging.getLogger().setLevel(logging.INFO)
        
        logger.info(f"Starting {Config.APP_TITLE}")
        
        # Validate environment
        if not validate_environment(Config):
            logger.error("Environment validation failed - exiting")
            sys.exit(1)
        
        # Load BOTH datasets
        logger.info("Loading application data...")
        df_spc, df_cp = load_both_datasets(Config)
        
        # Check if at least one dataset loaded successfully
        if df_spc is None and df_cp is None:
            logger.error("Both datasets failed to load")
            print("\n ERROR: Could not load either SPC or CellProfiler data files.")
            sys.exit(1)
        
        if df_spc is not None:
            logger.info(f" SPC data: {len(df_spc)} rows, {len(df_spc.columns)} columns")
        else:
            logger.warning("⚠️  SPC data not loaded - SPC features will be unavailable")
        
        if df_cp is not None:
            logger.info(f" CP data: {len(df_cp)} rows, {len(df_cp.columns)} columns")
        else:
            logger.warning("⚠️  CP data not loaded - CellProfiler features will be unavailable")
        
        # Print startup info and create app
        print_startup_info(df_spc, df_cp, Config)
        app = create_app(df_spc, df_cp, Config)
        
        # Start the server
        if os.environ.get("USER", "") == "warchas":
            app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
        else:
            app.run_server(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
        
    except KeyboardInterrupt:
        print("\n Application stopped. Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\n ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()