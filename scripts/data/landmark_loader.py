"""
Helper functions for loading landmark data in NEW SLIM format.

UPDATED (2025-01-21): Now supports BOTH CellProfiler AND SPC using the same slim format!

NEW FORMAT:
- test_distances.parquet: One treatment per row, with distances to ALL landmarks as columns
- reference_distances.parquet: Same structure as test
- landmark_metadata.parquet: Metadata for all landmarks (one landmark per row)

This module handles loading and mapping the data efficiently with caching for both data types.
"""

import pandas as pd
import pyarrow.parquet as pq
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import json

logger = logging.getLogger(__name__)

# Global cache for landmark metadata and dropdown options
_LANDMARK_METADATA_CACHE = {}  # Separate cache per data type
_LANDMARK_OPTIONS_CACHE = {}


def load_landmark_metadata(Config, data_type: str) -> Optional[pd.DataFrame]:
    """
    Load the landmark metadata file (with caching).
    
    Works for both CP and SPC data types.
    
    Args:
        Config: Configuration object with path methods
        data_type: 'cp' or 'spc'
    
    Returns:
        DataFrame with landmark metadata, or None if error
    """
    global _LANDMARK_METADATA_CACHE
    
    cache_key = data_type
    
    # Return cached version if available
    if cache_key in _LANDMARK_METADATA_CACHE:
        logger.debug(f"Using cached {data_type.upper()} landmark metadata")
        return _LANDMARK_METADATA_CACHE[cache_key]
    
    try:
        # Get appropriate path
        if data_type == 'cp':
            metadata_path = Config.get_cp_landmark_metadata_path()
        else:  # spc
            metadata_path = Config.get_spc_landmark_metadata_path()
        
        if not metadata_path.exists():
            logger.error(f"Landmark metadata file not found: {metadata_path}")
            return None
        
        logger.info(f"Loading {data_type.upper()} landmark metadata from {metadata_path.name}")
        df = pd.read_parquet(metadata_path)
        
        logger.info(f" Loaded metadata for {len(df):,} {data_type.upper()} landmarks")
        
        # Cache it
        _LANDMARK_METADATA_CACHE[cache_key] = df
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading {data_type.upper()} landmark metadata: {str(e)}")
        return None


def get_landmark_options(Config, dataset_type: str, data_type: str) -> Tuple[List[Dict], Optional[str]]:
    """
    Get landmark options with DISK CACHING.
    
    Builds dropdown options from landmark metadata. Works for both CP and SPC!
    
    CP label format: "GPCR (PP0001@10.0)" 
      from Metadata_annotated_target_first + Metadata_PP_ID_uM
    
    SPC label format: "ADRA1D (SC0138@1.0)"
      from moa_first + PP_ID_uM
    
    Args:
        Config: Configuration object
        dataset_type: 'test' or 'reference'
        data_type: 'cp' or 'spc'
    
    Returns:
        Tuple of (options_list, default_value)
    """
    
    try:
        # Check memory cache first
        cache_key = f"{dataset_type}_{data_type}"
        if cache_key in _LANDMARK_OPTIONS_CACHE:
            logger.info(f" Using cached landmark options (from memory)")
            options = _LANDMARK_OPTIONS_CACHE[cache_key]
            default_value = options[0]['value'] if options else None
            return options, default_value
        
        # Get paths based on data type
        if data_type == 'cp':
            if dataset_type == 'test':
                distance_path = Config.get_cp_landmark_distance_test_path()
            else:
                distance_path = Config.get_cp_landmark_distance_reference_path()
        else:  # spc
            if dataset_type == 'test':
                distance_path = Config.get_spc_landmark_distance_test_path()
            else:
                distance_path = Config.get_spc_landmark_distance_reference_path()
        
        # Check disk cache
        cache_file = distance_path.parent / f"{dataset_type}_{data_type}_landmark_options_cache.json"
        
        if cache_file.exists():
            cache_mtime = cache_file.stat().st_mtime
            parquet_mtime = distance_path.stat().st_mtime
            
            if cache_mtime > parquet_mtime:
                logger.info(f" Loading cached landmark options from {cache_file.name}")
                with open(cache_file, 'r') as f:
                    options = json.load(f)
                
                # Store in memory cache
                _LANDMARK_OPTIONS_CACHE[cache_key] = options
                
                default_value = options[0]['value'] if options else None
                return options, default_value
        
        # No valid cache - build from scratch
        logger.info(f"Building {data_type.upper()} landmark options from parquet files...")
        logger.info(f"  (This will take 30-60 seconds, but a cache file will be created)")
        
        if not distance_path.exists():
            logger.error(f"Required file not found: {distance_path}")
            return [], None
        
        # Load landmark metadata
        landmark_meta_df = load_landmark_metadata(Config, data_type)
        if landmark_meta_df is None:
            return [], None
        
        logger.info(f"  Found {len(landmark_meta_df):,} landmarks in metadata")
        
        # Get list of distance columns from the parquet file
        parquet_file = pq.ParquetFile(distance_path)
        all_columns = parquet_file.schema.names
        
        # Find all columns ending with "_distance"
        distance_cols = [col for col in all_columns if col.endswith('_distance')]
        
        # Remove "query_dmso_distance" if present (that's not a landmark)
        distance_cols = [col for col in distance_cols if col != 'query_dmso_distance']
        
        n_landmarks = len(distance_cols)
        logger.info(f"  Found {n_landmarks} landmark distance columns")
        
        if n_landmarks == 0:
            logger.error("   No landmark distance columns found!")
            return [], None
        
        # Build options list
        landmark_options = []
        
        # Define column names based on data type
        if data_type == 'cp':
            target_col = 'Metadata_annotated_target_first'
            pp_id_col = 'Metadata_PP_ID_uM'
        else:  # spc
            target_col = 'moa_first'
            pp_id_col = 'PP_ID_uM'
        
        for distance_col in distance_cols:
            # Extract treatment name from column name
            # e.g., "(R)-9s@0.1_distance" -> "(R)-9s@0.1"
            landmark_treatment = distance_col.replace('_distance', '')
            
            # Look up in metadata
            landmark_row = landmark_meta_df[landmark_meta_df['treatment'] == landmark_treatment]
            
            if len(landmark_row) == 0:
                logger.debug(f"  No metadata for landmark: {landmark_treatment}")
                continue
            
            landmark_row = landmark_row.iloc[0]
            
            # Build display label from target + PP_ID
            target = landmark_row.get(target_col)
            pp_id_um = landmark_row.get(pp_id_col)
            
            # Skip if both are missing (NO fallback as per requirements)
            if pd.isna(target) or str(target).strip() in ['', 'nan', 'None']:
                logger.debug(f"  Skipping landmark {landmark_treatment}: no {target_col}")
                continue
            
            if pd.isna(pp_id_um) or str(pp_id_um).strip() in ['', 'nan', 'None']:
                logger.debug(f"  Skipping landmark {landmark_treatment}: no {pp_id_col}")
                continue
            
            # Create label: "ADRA1D (SC0138@1.0)" or "GPCR (PP0001@10.0)"
            display_label = f"{target} ({pp_id_um})"
            
            # Value is the treatment name (without _distance suffix)
            value = landmark_treatment
            
            landmark_options.append({'label': display_label, 'value': value})
        
        # Sort alphabetically by target name (label)
        landmark_options.sort(key=lambda x: x['label'].lower())
        
        logger.info(f" Created {len(landmark_options)} {data_type.upper()} landmark options")
        
        # Save to disk cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(landmark_options, f, indent=2)
            logger.info(f"   Saved cache to {cache_file.name}")
        except Exception as e:
            logger.warning(f"  Could not save cache file: {e}")
        
        # Save to memory cache
        _LANDMARK_OPTIONS_CACHE[cache_key] = landmark_options
        
        default_value = landmark_options[0]['value'] if landmark_options else None
        return landmark_options, default_value
        
    except Exception as e:
        logger.error(f" Error getting {data_type.upper()} landmark options: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return [], None


def load_distances_for_landmark(Config, dataset_type: str, data_type: str,
                                landmark_treatment: str,
                                viz_df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """
    Load distance data for a specific landmark using NEW SLIM FORMAT.
    
    Works for both CP and SPC data types!
    
    Loads only the columns we need:
    - treatment (the query treatment)
    - Metadata columns
    - query_dmso_distance (x-axis)
    - {landmark_treatment}_distance (y-axis)
    
    UPDATED: Now merges with viz_df to get landmark columns and SMILES!
    
    Args:
        Config: Configuration object
        dataset_type: 'test' or 'reference'
        data_type: 'cp' or 'spc'
        landmark_treatment: The landmark treatment name, e.g., "(R)-9s@0.1"
        viz_df: Optional main visualization DataFrame to merge with for landmark info
    
    Returns:
        DataFrame with selective columns including landmarks, or None if error
    """
    
    try:
        # Get the appropriate distance file path
        if data_type == 'cp':
            if dataset_type == 'test':
                distance_path = Config.get_cp_landmark_distance_test_path()
            else:
                distance_path = Config.get_cp_landmark_distance_reference_path()
        else:  # spc
            if dataset_type == 'test':
                distance_path = Config.get_spc_landmark_distance_test_path()
            else:
                distance_path = Config.get_spc_landmark_distance_reference_path()
        
        if not distance_path.exists():
            logger.error(f"{data_type.upper()} distance file not found: {distance_path}")
            return None
        
        # Build column name for the landmark distance
        distance_col = f'{landmark_treatment}_distance'
        
        # Check if column exists
        parquet_file = pq.ParquetFile(distance_path)
        all_columns = parquet_file.schema.names
        
        if distance_col not in all_columns:
            logger.error(f"Distance column '{distance_col}' not found in parquet file!")
            return None
        
        # Define columns to load based on data type
        if data_type == 'cp':
            base_cols = [
                'treatment',
                'query_dmso_distance',
                'Metadata_plate_barcode',
                'Metadata_well',
                'Metadata_compound_uM',
                'Metadata_library',
                'Metadata_PP_ID',
                'Metadata_PP_ID_uM',
                'Metadata_annotated_target',
                'Metadata_annotated_target_first',
                'Metadata_annotated_target_description_truncated_10',
                'Metadata_perturbation_name',
                'Metadata_chemical_name',
                'Metadata_chemical_description',
                'Metadata_compound_type',
                'Metadata_manual_annotation',
                'Metadata_SMILES',
                'library',
                'is_reference',
                'n_replicates',
                'query_mad'
            ]
        else:  # spc
            base_cols = [
                'treatment',
                'query_dmso_distance',
                'plate',
                'well',
                'compound_uM',
                'library',
                'PP_ID',
                'PP_ID_uM',
                'annotated_target',
                'moa_first',
                'annotated_target_description_truncated_10',
                'perturbation_name',
                'chemical_name',
                'chemical_description',
                'manual_annotation',
                'SMILES',
                'is_reference',
                'query_mad'
            ]
        
        # Filter to only columns that exist
        cols_to_load = [col for col in base_cols if col in all_columns]
        
        # Add the specific landmark distance column
        cols_to_load.append(distance_col)
        
        logger.info(f"Loading {len(cols_to_load)} columns for {data_type.upper()} landmark '{landmark_treatment}'")
        
        # Load the data
        df = pd.read_parquet(distance_path, columns=cols_to_load, engine='pyarrow')
        
        # Rename distance column for convenience
        df = df.rename(columns={distance_col: 'landmark_distance'})
        
        logger.info(f"✓ Loaded {len(df):,} treatments with landmark data")
        
        # =========================================================================
        # SIMPLIFIED MERGE - Use Config.get_hover_columns() for the data type
        # =========================================================================
        if viz_df is not None and not viz_df.empty:
            logger.info(f"🔄 Merging with viz dataframe to get additional columns...")
            
            # Get the hover columns for this data type from config
            try:
                hover_columns = Config.get_hover_columns(data_type)
            except AttributeError:
                # Fallback if Config doesn't have the new method yet
                hover_columns = []
                logger.warning("Config.get_hover_columns() not available, using fallback")
            
            # Filter to columns that exist in viz_df but NOT already in df
            existing_cols = set(df.columns)
            cols_to_merge = [col for col in hover_columns 
                            if col in viz_df.columns and col not in existing_cols]
            
            # Always ensure treatment is included for merge key
            if cols_to_merge:
                merge_cols = ['treatment'] + cols_to_merge
                viz_subset = viz_df[merge_cols].drop_duplicates(subset=['treatment']).copy()
                
                # Merge on treatment
                df = df.merge(viz_subset, on='treatment', how='left', suffixes=('', '_viz'))
                
                logger.info(f"✓ Merged {len(cols_to_merge)} columns from viz dataframe")
                landmark_cols = [c for c in cols_to_merge if 'landmark' in c.lower()]
                logger.info(f"   Landmark columns: {len(landmark_cols)}")
            else:
                logger.info(f"✓ All needed columns already present in distance data")
        else:
            logger.warning(f"⚠ No viz_df provided - some columns may not be available")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading {data_type.upper()} distances for landmark: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def get_landmark_info(Config, data_type: str, landmark_treatment: str) -> Optional[Dict]:
    """
    Get metadata information for a specific landmark.
    
    Args:
        Config: Configuration object
        data_type: 'cp' or 'spc'
        landmark_treatment: The landmark treatment name
    
    Returns:
        Dict with landmark metadata, or None if not found
    """
    try:
        landmark_meta_df = load_landmark_metadata(Config, data_type)
        
        if landmark_meta_df is None:
            return None
        
        landmark_row = landmark_meta_df[landmark_meta_df['treatment'] == landmark_treatment]
        
        if len(landmark_row) == 0:
            logger.warning(f"Landmark '{landmark_treatment}' not found in metadata")
            return None
        
        return landmark_row.iloc[0].to_dict()
        
    except Exception as e:
        logger.error(f"Error getting landmark info: {str(e)}")
        return None


def clear_landmark_cache():
    """Clear all cached landmark metadata and options (useful for testing)."""
    global _LANDMARK_METADATA_CACHE, _LANDMARK_OPTIONS_CACHE
    _LANDMARK_METADATA_CACHE = {}
    _LANDMARK_OPTIONS_CACHE = {}
    logger.info("Cleared all landmark caches")