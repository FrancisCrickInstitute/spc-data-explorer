"""
Data loading module for the Phenotype Clustering Interactive Visualization App.

UPDATED (2025-01-21): Now loads BOTH SPC and CellProfiler datasets!

This module handles loading and initial processing of both visualization data CSV files.
It provides robust error handling, data validation, and creates display-friendly
column mappings for the dashboard interface.

Key Functions:
- load_both_datasets(): Main function to load both SPC and CP data
- _load_single_dataset(): Helper to load one dataset
- _add_display_columns(): Add computed columns for better UI display
- _validate_required_columns(): Ensure essential columns are present

The module handles missing files gracefully and provides informative error messages
to help with troubleshooting data loading issues.
"""

import glob
import pandas as pd
import traceback
from pathlib import Path
from typing import Optional, List, Tuple
import logging

# Handle imports for both relative and absolute import contexts
try:
    from ..config_loader import get_config
except ImportError:
    from config_loader import get_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Custom exception for data loading errors."""
    pass


def load_both_datasets(Config=None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load BOTH SPC and CellProfiler visualization datasets.
    
    This is the main entry point for loading data in the application.
    
    Args:
        Config: Configuration object (optional, will be loaded if None)
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (df_spc, df_cp) or (None, None) if loading fails
    """
    if Config is None:
        Config = get_config()
    
    logger.info("="*80)
    logger.info("LOADING BOTH DATASETS (SPC + CellProfiler)")
    logger.info("="*80)
    
    # Load SPC dataset
    logger.info("\n[1/2] Loading SPC dataset...")
    df_spc = _load_single_dataset(Config, 'spc')
    
    if df_spc is not None:
        logger.info(f" SPC dataset loaded: {len(df_spc):,} rows, {len(df_spc.columns)} columns")
    else:
        logger.error(" Failed to load SPC dataset")
    
    # Load CP dataset
    logger.info("\n[2/2] Loading CellProfiler dataset...")
    df_cp = _load_single_dataset(Config, 'cp')
    
    if df_cp is not None:
        logger.info(f" CP dataset loaded: {len(df_cp):,} rows, {len(df_cp.columns)} columns")
    else:
        logger.error(" Failed to load CellProfiler dataset")
    
    # Summary
    logger.info("\n" + "="*80)
    if df_spc is not None and df_cp is not None:
        logger.info(" BOTH DATASETS LOADED SUCCESSFULLY")
    elif df_spc is not None or df_cp is not None:
        logger.warning("⚠️  PARTIAL SUCCESS - One dataset failed to load")
    else:
        logger.error(" BOTH DATASETS FAILED TO LOAD")
    logger.info("="*80 + "\n")
    
    return df_spc, df_cp


def _load_single_dataset(Config, data_type: str) -> Optional[pd.DataFrame]:
    """
    Load a single dataset (either SPC or CP).
    
    Args:
        Config: Configuration object
        data_type: 'spc' or 'cp'
    
    Returns:
        pd.DataFrame or None if loading fails
    """
    try:
        if data_type == 'spc':
            analysis_dir = Config.SPC_ANALYSIS_DIR
            file_name = "spc_for_viz_app.csv"
        elif data_type == 'cp':
            analysis_dir = Config.CP_ANALYSIS_DIR
            file_name = "cp_for_viz_app.csv"
        else:
            logger.error(f"Invalid data_type: {data_type}")
            return None
        
        # Files are in the data/ subdirectory
        file_path = Path(analysis_dir) / "data" / file_name
        
        if not file_path.exists():
            logger.warning(f"CSV file not found: {file_path}")
            logger.info(f"Attempting to load from Parquet files instead...")
            
            # Parquet files are in different subdirs for SPC vs CP
            if data_type == 'spc':
                parquet_dir = Path(analysis_dir) / "analysis" / "landmark_distances"
            else:  # cp
                parquet_dir = Path(analysis_dir) / "landmark_analysis"
            
            # Try loading from parquet
            df = _load_from_parquet(parquet_dir, data_type)
            if df is not None:
                return df
            
            logger.error(f"Could not load {data_type.upper()} data from CSV or Parquet")
            return None
        
        logger.info(f"Loading {data_type.upper()} data from: {file_path}")
        
        # Load CSV
        df = pd.read_csv(file_path, low_memory=False)
        
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Add display columns
        df = _add_display_columns(df, data_type)
        
        # Validate required columns
        if not _validate_required_columns(df, data_type):
            logger.error(f"Required columns missing in {data_type.upper()} dataset")
            return None
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading {data_type.upper()} data: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def _load_from_parquet(parquet_dir: Path, Config, data_type: str) -> Optional[pd.DataFrame]:
    """Load data from parquet files as fallback."""
    try:
        test_path = parquet_dir / "test_distances.parquet"
        ref_path = parquet_dir / "reference_distances.parquet"
        landmark_path = parquet_dir / "landmark_metadata.parquet"
        
        if not test_path.exists():
            logger.error(f"Parquet files not found in: {parquet_dir}")
            return None
        
        logger.info(f"Loading test distances from: {test_path}")
        df_test = pd.read_parquet(test_path)
        
        logger.info(f"Loading reference distances from: {ref_path}")
        df_ref = pd.read_parquet(ref_path)
        
        # Combine
        df = pd.concat([df_test, df_ref], ignore_index=True)
        logger.info(f"Combined: {len(df):,} rows")
        
        # Add display columns and return
        return _add_display_columns(df, data_type)
        
    except Exception as e:
        logger.error(f"Error loading from Parquet: {e}")
        return None


def _merge_chemoproteomics_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge chemoproteomics data into the main dataframe.
    
    Args:
        df: Main dataframe
    
    Returns:
        pd.DataFrame: Dataframe with chemoproteomics data merged (if available)
    """
    try:
        # Fixed path to chemoproteomics file
        chemprot_path = Path("/Volumes/hts/working/Joe_Tuersley/Prosperity/chemoproteomics_hit_list_oct_2025/chemoproteomics_for_cell_paint.csv")
        
        if not chemprot_path.exists():
            logger.debug(f"  Chemoproteomics file not found at: {chemprot_path}")
            return df
        
        logger.info(f"  Loading chemoproteomics data from: {chemprot_path.name}")
        chemprot_df = pd.read_csv(chemprot_path)
        
        # Columns to merge
        chemprot_cols_to_merge = [
            'compound_id',  # Join key
            'cell_line',
            'experiment_type',
            'compound_concentration_uM',
            'gene_site_1',
            'gene_site_2',
            'gene_site_3',
            'gene_site_4',
            'gene_site_5'
        ]
        
        # Check which columns actually exist
        available_chemprot_cols = [col for col in chemprot_cols_to_merge if col in chemprot_df.columns]
        
        if len(available_chemprot_cols) <= 1:  # Only compound_id or less
            logger.debug("  No chemoproteomics columns to merge")
            return df
        
        # Check if compound_id exists in main dataframe
        if 'compound_id' not in df.columns:
            logger.debug("  compound_id column not found - cannot merge chemoproteomics data")
            return df
        
        # Remove duplicates and merge
        chemprot_subset = chemprot_df[available_chemprot_cols].drop_duplicates(subset=['compound_id'])
        
        # Perform the merge
        df_before_merge = len(df)
        df = df.merge(chemprot_subset, on='compound_id', how='left', suffixes=('', '_chemprot'))
        df_after_merge = len(df)
        
        # Check merge success
        if df_before_merge != df_after_merge:
            logger.warning(f"  Merge changed row count: {df_before_merge} -> {df_after_merge}")
        
        # Count how many rows got chemoproteomics data
        non_null_count = df['experiment_type'].notna().sum() if 'experiment_type' in df.columns else 0
        logger.info(f"   Merged chemoproteomics data: {non_null_count} rows have chemprot info")
        
        return df
        
    except Exception as e:
        logger.warning(f"  Could not merge chemoproteomics data: {str(e)}")
        return df


def _log_column_info(df: pd.DataFrame, data_type: str) -> None:
    """
    Log information about the loaded dataframe columns.
    
    Args:
        df: The loaded dataframe
        data_type: 'spc' or 'cp'
    """
    Config = get_config()
    
    # Check for dimensionality reduction columns
    if data_type == 'spc':
        dim_red_cols = [col for col in df.columns 
                       if col.startswith('UMAP') or col.startswith('TSNE')]
    else:  # cp
        dim_red_cols = [col for col in df.columns 
                       if 'umap' in col.lower() or 'tsne' in col.lower()]
    
    if dim_red_cols:
        logger.info(f"  Found dimensionality reduction columns: {len(dim_red_cols)} columns")
        logger.debug(f"    {dim_red_cols[:5]}...")  # Show first 5
    else:
        logger.warning("  No dimensionality reduction columns found")
    
    # Check for essential phenotype columns
    if data_type == 'spc':
        essential_cols = ['treatment', 'plate', 'well', 'compound_name']
    else:  # cp
        essential_cols = ['treatment', 'Metadata_plate_barcode', 'Metadata_well']
    
    missing_essential = [col for col in essential_cols if col not in df.columns]
    
    if missing_essential:
        logger.warning(f"  Missing some essential columns: {missing_essential}")


def _validate_required_columns(df: pd.DataFrame, data_type: str) -> bool:
    """
    Validate that the dataframe contains minimum required columns.
    
    Args:
        df: The dataframe to validate
        data_type: 'spc' or 'cp'
        
    Returns:
        bool: True if all required columns are present
    """
    Config = get_config()
    
    # Define required columns based on data type
    if data_type == 'spc':
        required_cols = ['treatment', 'plate', 'well']
    else:  # cp
        required_cols = ['treatment', 'Metadata_plate_barcode', 'Metadata_well']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"  Missing required columns: {missing_cols}")
        return False
    
    # Check for at least one plotting dimension
    plot_cols = [col for col in df.columns 
                if any(pattern in col for pattern in Config.METRIC_COLUMN_PATTERNS)]
    
    if not plot_cols:
        logger.warning("  No plotting columns found - limited visualization available")
    
    return True


def _add_display_columns(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """
    Add computed columns for better display in the UI.
    
    Args:
        df: Input dataframe
        data_type: 'spc' or 'cp'
        
    Returns:
        pd.DataFrame: Dataframe with additional display columns
    """
    logger.info("  Adding display columns...")

   # ===== CREATE ALIASES FOR CP COLUMNS =====
    if data_type == 'cp':
        # Create aliases from Metadata_* columns to standard names
        # This allows CP data to use the same column names as SPC data
        aliases = {
        # Basic identifiers
        'treatment': 'Metadata_treatment',
        'library': 'Metadata_library',
        'plate': 'Metadata_plate_barcode',
        'well': 'Metadata_well',
        'cell_count': 'cell_count',
        
        # Compound information
        'compound_name': 'Metadata_chemical_name',
        'compound_uM': 'Metadata_compound_uM',
        'PP_ID': 'Metadata_PP_ID',
        'PP_ID_uM': 'Metadata_PP_ID_uM',
        'SMILES': 'Metadata_SMILES',
        'chemical_description': 'Metadata_chemical_description',
        'compound_type': 'Metadata_compound_type',
        'perturbation_name': 'Metadata_perturbation_name',
        'supplier_ID': 'Metadata_supplier_ID',
        
        # MOA/Target information
        'moa_first': 'Metadata_annotated_target_first',
        'moa_truncated_10': 'Metadata_annotated_target_truncated_10',
        'moa_compound_uM': 'Metadata_annotated_target_first_compound_uM',
        'target_description_truncated_10': 'Metadata_annotated_target_description_truncated_10',
        'manual_annotation': 'Metadata_manual_annotation',
        'annotated_target': 'Metadata_annotated_target',                                    # ← NEW
        'annotated_target_description': 'Metadata_annotated_target_description',            # ← NEW
        'annotated_target_description_truncated_10': 'Metadata_annotated_target_description_truncated_10',  # ← NEW
        
        # Control information
        'control_type': 'Metadata_control_type',                                            # ← NEW
        'control_name': 'Metadata_control_name',                                            # ← NEW
        'is_control': 'Metadata_is_control',                                                # ← NEW
        
        # Cell type
        'cell_type': 'Metadata_cell_type',                                                  # ← NEW
        
        # Metrics (CP → SPC naming)
        'mad_cosine': 'query_mad',
        'cosine_distance_from_dmso': 'query_dmso_distance',
    }
    
        # Landmark columns (closest, second_closest, third_closest)
        for prefix in ['closest', 'second_closest', 'third_closest']:
            # Basic landmark info
            aliases[f'{prefix}_landmark_moa_first'] = f'{prefix}_landmark_Metadata_annotated_target_first'
            aliases[f'{prefix}_landmark_PP_ID'] = f'{prefix}_landmark_Metadata_PP_ID'
            aliases[f'{prefix}_landmark_PP_ID_uM'] = f'{prefix}_landmark_Metadata_PP_ID_uM'
            aliases[f'{prefix}_landmark_annotated_target_description_truncated_10'] = f'{prefix}_landmark_Metadata_annotated_target_description_truncated_10'
            aliases[f'{prefix}_landmark_manual_annotation'] = f'{prefix}_landmark_Metadata_manual_annotation'
            aliases[f'{prefix}_landmark_compound_uM'] = f'{prefix}_landmark_Metadata_compound_uM'
            aliases[f'{prefix}_landmark_SMILES'] = f'{prefix}_landmark_Metadata_SMILES'
            aliases[f'{prefix}_landmark_chemical_description'] = f'{prefix}_landmark_Metadata_chemical_description'  # ← NEW (if exists)
            aliases[f'{prefix}_landmark_compound_type'] = f'{prefix}_landmark_Metadata_compound_type'                # ← NEW
            aliases[f'{prefix}_landmark_chemical_name'] = f'{prefix}_landmark_Metadata_chemical_name'                # ← NEW
            aliases[f'{prefix}_landmark_perturbation_name'] = f'{prefix}_landmark_Metadata_perturbation_name'        # ← NEW
            aliases[f'{prefix}_landmark_annotated_target'] = f'{prefix}_landmark_Metadata_annotated_target'          # ← NEW
            aliases[f'{prefix}_landmark_annotated_target_description'] = f'{prefix}_landmark_Metadata_annotated_target_description'  # ← NEW
            aliases[f'{prefix}_landmark_moa_truncated_10'] = f'{prefix}_landmark_Metadata_annotated_target_truncated_10'             # ← NEW
            
            for alias, metadata_col in aliases.items():
                if alias not in df.columns and metadata_col in df.columns:
                    df[alias] = df[metadata_col]
                    logger.info(f"    Created '{alias}' alias from {metadata_col}")

                    # ===== SPECIAL HANDLING: Columns that need to be CREATED, not aliased =====
            
            # Create landmark_label from is_landmark (CP has boolean, SPC has string)
            if 'landmark_label' not in df.columns and 'is_landmark' in df.columns:
                df['landmark_label'] = df['is_landmark'].apply(
                    lambda x: 'Reference Landmark' if x == True else 'Other'
                )
                logger.info("    Created 'landmark_label' from 'is_landmark'")
            
            # Create landmark_label_mad (CP doesn't have MAD-based landmark labels)
            if 'landmark_label_mad' not in df.columns and 'is_landmark' in df.columns:
                df['landmark_label_mad'] = df['is_landmark'].apply(
                    lambda x: 'Reference Landmark' if x == True else 'Other'
                )
                logger.info("    Created 'landmark_label_mad' from 'is_landmark'")
            
            # Verify cell_count exists
            if 'cell_count' in df.columns:
                logger.info("    'cell_count' column already exists")
        

    # ==========================================
    # Handle different column naming conventions
    if data_type == 'spc':
        compound_col = 'compound_name'
        pp_id_col = 'PP_ID'
        conc_col = 'compound_uM'
        well_col = 'well'
        moa_col = 'moa'
    else:  # cp
        compound_col = 'Metadata_chemical_name'
        pp_id_col = 'Metadata_PP_ID'
        conc_col = 'Metadata_compound_uM'
        well_col = 'Metadata_well'
        moa_col = 'moa'  # May not exist in CP
    
    # Ensure we have compound_name column (for SPC)
    if data_type == 'spc' and compound_col not in df.columns:
        if 'treatment' in df.columns:
            df[compound_col] = df['treatment']
            logger.info("    Created compound_name from treatment column")
    
    # Ensure we have compound_uM column (for SPC)
    if data_type == 'spc' and conc_col not in df.columns and 'concentration' in df.columns:
        df[conc_col] = df['concentration']
        logger.info("    Created compound_uM from concentration column")
    
    # Create display_name column
    if pp_id_col in df.columns:
        display_compound = df[pp_id_col].fillna(df[compound_col] if compound_col in df.columns else df['treatment'])
    elif compound_col in df.columns:
        display_compound = df[compound_col]
    else:
        display_compound = df['treatment']
    
    df['display_name'] = display_compound.astype(str)
    
    # Add concentration to display name if available
    if conc_col in df.columns:
        concentration_mask = ~df[conc_col].isna()
        df.loc[concentration_mask, 'display_name'] = (
            display_compound.loc[concentration_mask].astype(str) + 
            ' (' + df.loc[concentration_mask, conc_col].astype(str) + ' µM)'
        )
        logger.info("    Enhanced display names with concentration information")
    
    # Standardize well format (ensure A01 not A1)
    if well_col in df.columns:
        df[well_col] = df[well_col].apply(_standardize_well_format)
        logger.info("    Standardized well format")
    
    # Create MOA first column if MOA contains comma-separated values
    if moa_col in df.columns:
        if df[moa_col].astype(str).str.contains(',').any():
            df['moa_first'] = df[moa_col].astype(str).apply(
                lambda x: x.split(',')[0].strip() if ',' in str(x) else str(x)
            )
            logger.info("    Created moa_first column from comma-separated MOA values")
    
    logger.info(f"  Final {data_type.upper()} dataframe shape: {df.shape}")
    return df


def _standardize_well_format(well: str) -> str:
    """
    Standardize well format to ensure consistent A01 format (not A1).
    
    Args:
        well: Well identifier (e.g., 'A1', 'A01', 'B12')
        
    Returns:
        str: Standardized well format (e.g., 'A01', 'B12')
    """
    if pd.isna(well):
        return well
    
    well = str(well).strip()
    
    # If format is like "A1", convert to "A01"
    if len(well) == 2 and well[0].isalpha() and well[1].isdigit():
        return f"{well[0]}{int(well[1:]):02d}"
    
    # If already in correct format or other format, return as-is
    return well


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the loaded data for debugging and validation.
    
    Args:
        df: The loaded dataframe
        
    Returns:
        dict: Summary statistics and information about the data
    """
    # Try SPC columns first, then CP columns
    treatment_col = 'treatment'
    plate_col = 'plate' if 'plate' in df.columns else 'Metadata_plate_barcode'
    well_col = 'well' if 'well' in df.columns else 'Metadata_well'
    
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'n_treatments': df[treatment_col].nunique() if treatment_col in df.columns else 0,
        'n_plates': df[plate_col].nunique() if plate_col in df.columns else 0,
        'n_wells': df[well_col].nunique() if well_col in df.columns else 0,
        'has_umap': any('umap' in col.lower() for col in df.columns),
        'has_tsne': any('tsne' in col.lower() for col in df.columns),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    return summary