"""
Configuration module for the Phenotype Clustering Interactive Visualization App.

This module contains all configuration settings, paths, and constants used throughout
the application. It provides centralized configuration management and path validation
to ensure the application can locate required data and image directories.

Key Features:
- Centralized configuration management
- Path validation with helpful error messages
- Environment-specific settings (dev/prod)
- Image processing and UI constants

For sharing, get IP address in terminal with:
ifconfig | grep "inet" | grep -v 127.0.0.1
"""

import os
from pathlib import Path
from typing import List, Tuple

# Image thumbnail directories

# CRISPR genome
# /Volumes/proj-prosperity/hts/raw/projects/20250501_HaCaT_CRISPR_genome_cell_paint/data/images/thumbnails

# GSK Chemogenetic V1, JUMP V2, SCG V1, CCA
# /Volumes/proj-prosperity/hts/raw/projects/20250612_HaCaT_GSK_chemogenetic_JUMP_SGC_CCA_cell_paint/data/images/thumbnails

# GSK fragments V3, GSK clickable V1, JUMP V1
# /Volumes/proj-prosperity/hts/raw/projects/20250219_GSK_fragments_V3_clickable_V1_with_reference_sets/data/images/thumbnails

# HTC V1
# /Volumes/proj-prosperity/hts/raw/projects/20240705_harry_htc_v1_hacat_4conc/data/images/thumbnails



class Config:
    """Application configuration settings and constants."""
    
    # ============================================================================
    # PATHS - Update these to match your environment
    # ============================================================================
    if os.environ.get("USER", "") == "warchas":
        # SPC Analysis Directory
        SPC_ANALYSIS_DIR = Path("/nemo/stp/hts/working/Joe_Tuersley/code/spc-cosine-analysis/analysis/20251122_HaCaT_gsk_all_no_CRISPR_no_HTC_V1_no_CCA_V1_ResNet18/spc_analysis_20251124_174723")
        
        # CellProfiler Analysis Directory
        CP_ANALYSIS_DIR = Path("/nemo/project/proj-prosperity/hts/raw/projects/20251111_gsk_prosperity_all_datasets/cellprofiler/processed_data/20251120_124820_from_well_results")
        
        THUMBNAIL_DIRS = [
            # CRISPR genome
            Path("/nemo/project/proj-prosperity/hts/raw/projects/20250501_HaCaT_CRISPR_genome_cell_paint/data/images/thumbnails"),
            # GSK Chemogenetic V1, JUMP V2, SCG V1, CCA
            Path("/nemo/project/proj-prosperity/hts/raw/projects/20250612_HaCaT_GSK_chemogenetic_JUMP_SGC_CCA_cell_paint/data/images/thumbnails"),
            # GSK fragments V3, GSK clickable V1, JUMP V1
            Path("/nemo/project/proj-prosperity/hts/raw/projects/20250219_GSK_fragments_V3_clickable_V1_with_reference_sets/data/images/thumbnails"),
            # HTC V1
            Path("/nemo/project/proj-prosperity/hts/raw/projects/20240705_harry_htc_v1_hacat_4conc/data/images/thumbnails"),
            # HTC V2
            Path("/nemo/project/proj-prosperity/hts/raw/projects/20251020_HaCaT_GSK_HTC_V1_V2_cell_paint/data/images/thumbnails")
        ]
    else:
        # SPC Analysis Directory
        SPC_ANALYSIS_DIR = Path("/Volumes/hts/working/Joe_Tuersley/code/spc-cosine-analysis/analysis/20251122_HaCaT_gsk_all_no_CRISPR_no_HTC_V1_no_CCA_V1_ResNet18/spc_analysis_20251124_174723")
        
        # CellProfiler Analysis Directory
        CP_ANALYSIS_DIR = Path("/Volumes/proj-prosperity/hts/raw/projects/20251111_gsk_prosperity_all_datasets/cellprofiler/processed_data/20251120_124820_from_well_results")
        
        THUMBNAIL_DIRS = [
            # CRISPR genome
            Path("/Volumes/proj-prosperity/hts/raw/projects/20250501_HaCaT_CRISPR_genome_cell_paint/data/images/thumbnails"),
            # GSK Chemogenetic V1, JUMP V2, SCG V1, CCA
            Path("/Volumes/proj-prosperity/hts/raw/projects/20250612_HaCaT_GSK_chemogenetic_JUMP_SGC_CCA_cell_paint/data/images/thumbnails"),
            # GSK fragments V3, GSK clickable V1, JUMP V1
            Path("/Volumes/proj-prosperity/hts/raw/projects/20250219_GSK_fragments_V3_clickable_V1_with_reference_sets/data/images/thumbnails"),
            # HTC V1
            Path("/Volumes/proj-prosperity/hts/raw/projects/20240705_harry_htc_v1_hacat_4conc/data/images/thumbnails"),
            # HTC V2
            Path("/Volumes/proj-prosperity/hts/raw/projects/20251020_HaCaT_GSK_HTC_V1_V2_cell_paint/data/images/thumbnails")
        ]
    
    # For backwards compatibility
    ANALYSIS_DIR = SPC_ANALYSIS_DIR
    
    # ============================================================================
    # APP SETTINGS
    # ============================================================================
    DEBUG = False
    HOST = '0.0.0.0'
    PORT = 1009
    APP_TITLE = "Phenotype Clustering Interactive Visualisation"
    
    # Toggle chemical structure display
    SHOW_CHEMICAL_STRUCTURES = True  # Set to True to re-enable chemical structures
    
    # ============================================================================
    # IMAGE SETTINGS
    # ============================================================================
    DEFAULT_IMAGE_SIZE = (500, 500)
    LARGE_IMAGE_SIZE = (650, 650)
    RANDOM_SITE_RANGE = (1, 9)  # Range for random site selection
    
    # Font settings for image labeling
    FONT_SIZE = 16
    FONT_PATHS = [
        "/System/Library/Fonts/Arial.ttf",  # macOS
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        "C:/Windows/Fonts/arial.ttf"  # Windows
    ]
    
    # ============================================================================
    # UI SETTINGS
    # ============================================================================
    PLOT_HEIGHT = '80vh'
    PLOT_WIDTH = 1200
    PLOT_DEFAULT_HEIGHT = 1000
    
    # Point size range for scatter plots
    POINT_SIZE_MIN = 2
    POINT_SIZE_MAX = 10
    POINT_SIZE_DEFAULT = 5
    
    # Search settings
    MIN_SEARCH_LENGTH = 3  # Minimum characters before search activates
    MAX_SEARCH_RESULTS = 100  # Maximum results to show in dropdowns
    
    # ============================================================================
    # DATA COLUMN MAPPINGS - FIXED FOR YOUR CELLPROFILER DATA
    # ============================================================================
    # Columns to search for metrics in plots - Updated to match your actual data structure
    METRIC_COLUMN_PATTERNS = [
    # Original SPC metrics
    'mad_cosine', 'var_cosine', 'std_cosine', 'closest_landmark_distance',
    'cosine_distance_from_dmso', 'median_distance', 
    'UMAP', 'TSNE',  # This will match UMAP1, UMAP2, TSNE1, TSNE2
    
    # CellProfiler columns - simplified patterns
    'cellprofiler_umap',  # This will match all cellprofiler_umap_* columns
    'cellprofiler_tsne',  # This will match all cellprofiler_tsne_* columns
    'cellprofiler_pca',   # This will match all cellprofiler_pca_* columns
]
    # Columns for hover data display - modify as desired
    HOVER_DATA_COLUMNS = [
        # Basic compound information
        'plate',
        'well',
        'SMILES',
        'treatment', 
        'compound_name', 
        'compound_uM',
        'compound_type',  # ADDED compound_type
        'moa_truncated_10', 
        'moa_compound_uM',
        'library', 
        'landmark_label',
        'moa_first', 
        'PP_ID', 
        'PP_ID_uM',
        'target_description_truncated_10',
        'chemical_description',
        'manual_annotation',
        'gene_description',
        'valid_for_phenotypic_makeup',
        'validity_display',
        
        # LANDMARK INFORMATION - 1st, 2nd, 3rd closest landmarks
        'closest_landmark_moa_first',
        'closest_landmark_PP_ID',
        'closest_landmark_PP_ID_uM',
        'closest_landmark_annotated_target_description_truncated_10',
        'closest_landmark_manual_annotation',
        'closest_landmark_distance',
        
        'second_closest_landmark_moa_first',
        'second_closest_landmark_PP_ID', 
        'second_closest_landmark_PP_ID_uM',
        'second_closest_landmark_annotated_target_description_truncated_10',
        'second_closest_landmark_manual_annotation',
        'second_closest_landmark_distance',
        
        'third_closest_landmark_moa_first',
        'third_closest_landmark_PP_ID',
        'third_closest_landmark_PP_ID_uM',
        'third_closest_landmark_annotated_target_description_truncated_10',
        'third_closest_landmark_manual_annotation',
        'valid_for_phenotypic_makeup',
        'third_closest_landmark_distance',
        'cell_line',   
        'experiment_type',
        'compound_concentration_uM',
        'gene_site_1',
        'gene_site_2',
        'gene_site_3',
        'gene_site_4',
        'gene_site_5',
        ]
    
    # ============================================================================
    # SIMPLIFIED HOVER DATA COLUMNS - SEPARATE FOR SPC AND CP
    # ============================================================================
    # These are the EXACT columns that exist in each dataset.
    # Order matters - first columns are used for plate/well detection.
    
    # --- SPC HOVER COLUMNS ---
    SPC_HOVER_COLUMNS = [
        'plate', 'well', 'treatment', 'SMILES',
        'compound_name', 'compound_uM', 'PP_ID', 'PP_ID_uM', 'library',
        'moa_first', 'moa_truncated_10', 'moa_compound_uM',
        'annotated_target_description_truncated_10', 'target_description_truncated_10',
        'chemical_description', 'manual_annotation', 'compound_type',
        'landmark_label', 'is_landmark', 'valid_for_phenotypic_makeup',
        'cosine_distance_from_dmso', 'mad_cosine',
        # 1st Landmark
        'closest_landmark_moa_first', 'closest_landmark_PP_ID', 'closest_landmark_PP_ID_uM',
        'closest_landmark_distance', 'closest_landmark_annotated_target_description_truncated_10',
        'closest_landmark_manual_annotation', 'closest_landmark_SMILES',
        # 2nd Landmark - ADDED target description
        'second_closest_landmark_moa_first', 'second_closest_landmark_PP_ID_uM',
        'second_closest_landmark_distance', 
        'second_closest_landmark_annotated_target_description_truncated_10', 
        'second_closest_landmark_manual_annotation',
        # 3rd Landmark - ADDED target description
        'third_closest_landmark_moa_first', 'third_closest_landmark_PP_ID_uM',
        'third_closest_landmark_distance', 
        'third_closest_landmark_annotated_target_description_truncated_10',  
        'third_closest_landmark_manual_annotation',
        'gene_description',
    ]
    
    CP_HOVER_COLUMNS = [
        'plate', 'well', 'treatment', 'SMILES',
        'compound_name', 'compound_uM', 'PP_ID', 'PP_ID_uM', 'library',
        'moa_first', 'moa_truncated_10', 'moa_compound_uM',
        'target_description_truncated_10',
        'chemical_description', 'manual_annotation', 'compound_type',
        'is_landmark', 'valid_for_phenotypic_makeup',
        'cosine_distance_from_dmso', 'mad_cosine',
        # 1st Landmark
        'closest_landmark_moa_first', 'closest_landmark_PP_ID', 'closest_landmark_PP_ID_uM',
        'closest_landmark_distance', 'closest_landmark_annotated_target_description_truncated_10',
        'closest_landmark_manual_annotation', 'closest_landmark_SMILES',
        # 2nd Landmark - ADDED target description
        'second_closest_landmark_moa_first', 'second_closest_landmark_PP_ID_uM',
        'second_closest_landmark_distance',
        'second_closest_landmark_annotated_target_description_truncated_10',  
        'second_closest_landmark_manual_annotation',
        # 3rd Landmark - ADDED target description
        'third_closest_landmark_moa_first', 'third_closest_landmark_PP_ID_uM',
        'third_closest_landmark_distance',
        'third_closest_landmark_annotated_target_description_truncated_10',  
        'third_closest_landmark_manual_annotation',
    ]

    
# ================================================================================
    # HOVER DISPLAY MAPPINGS - Using aliases, these are nearly identical
    # ============================================================================
    # Format: (column_name, display_label)
    # loader.py creates aliases so CP data has same column names as SPC
    
    # ============================================================================
    # HOVER DISPLAY MAPPINGS - Define what to show in plot hover tooltips
    # ============================================================================
    # Format: (column_name, display_label)
    
    SPC_HOVER_DISPLAY = [
        # Basic info
        ('plate', 'Plate'),
        ('well', 'Well'),
        ('treatment', 'Treatment'),
        ('compound_name', 'Compound'),
        ('compound_uM', 'Conc (µM)'),
        ('library', 'Library'),
        ('PP_ID_uM', 'PP ID'),
        
        # Target/MOA
        ('moa_first', 'MOA'),
        ('annotated_target_description_truncated_10', 'Target Desc'),
        ('chemical_description', 'Chem Desc'),
        ('manual_annotation', 'Broad Annotation'),
        
        # Status
        ('landmark_label', 'Landmark Status'),
        ('valid_for_phenotypic_makeup', 'Valid for Phenotypic Makeup'),
        
        # Metrics
        ('cosine_distance_from_dmso', 'DMSO Distance'),
        ('mad_cosine', 'MAD Cosine'),
        
        # 1st Closest Landmark
        ('closest_landmark_moa_first', '1st Landmark'),
        ('closest_landmark_PP_ID_uM', '1st LM PP ID'),
        ('closest_landmark_distance', '1st LM Distance'),
        ('closest_landmark_annotated_target_description_truncated_10', '1st LM Target Desc'),
        ('closest_landmark_manual_annotation', '1st LM Broad Annotation'),
        
        # 2nd Closest Landmark
        ('second_closest_landmark_moa_first', '2nd Landmark'),
        ('second_closest_landmark_PP_ID_uM', '2nd LM PP ID'),
        ('second_closest_landmark_distance', '2nd LM Distance'),
        ('second_closest_landmark_annotated_target_description_truncated_10', '2nd LM Target Desc'),
        ('second_closest_landmark_manual_annotation', '2nd LM Broad Annotation'),
        
        # 3rd Closest Landmark
        ('third_closest_landmark_moa_first', '3rd Landmark'),
        ('third_closest_landmark_PP_ID_uM', '3rd LM PP ID'),
        ('third_closest_landmark_distance', '3rd LM Distance'),
        ('third_closest_landmark_annotated_target_description_truncated_10', '3rd LM Target Desc'),
        ('third_closest_landmark_manual_annotation', '3rd LM Broad Annotation'),
    ]
    
    # CP uses same column names thanks to aliases created in loader.py
    CP_HOVER_DISPLAY = [
        # Basic info
        ('plate', 'Plate'),
        ('well', 'Well'),
        ('treatment', 'Treatment'),
        ('compound_name', 'Compound'),
        ('compound_uM', 'Conc (µM)'),
        ('library', 'Library'),
        ('PP_ID_uM', 'PP ID'),
        
        # Target/MOA
        ('moa_first', 'Target'),
        ('target_description_truncated_10', 'Target Desc'),
        ('chemical_description', 'Chem Desc'),
        ('manual_annotation', 'Broad Annotation'),
        
        # Status
        ('is_landmark', 'Is Landmark'),
        ('valid_for_phenotypic_makeup', 'Valid for Phenotypic Makeup'),
        
        # Metrics
        ('cosine_distance_from_dmso', 'DMSO Distance'),
        ('mad_cosine', 'MAD Cosine'),
        
        # 1st Closest Landmark
        ('closest_landmark_moa_first', '1st Landmark'),
        ('closest_landmark_PP_ID_uM', '1st LM PP ID'),
        ('closest_landmark_distance', '1st LM Distance'),
        ('closest_landmark_annotated_target_description_truncated_10', '1st LM Target Desc'),
        ('closest_landmark_manual_annotation', '1st LM Broad Annotation'),
        
        # 2nd Closest Landmark
        ('second_closest_landmark_moa_first', '2nd Landmark'),
        ('second_closest_landmark_PP_ID_uM', '2nd LM PP ID'),
        ('second_closest_landmark_distance', '2nd LM Distance'),
        ('second_closest_landmark_annotated_target_description_truncated_10', '2nd LM Target Desc'),
        ('second_closest_landmark_manual_annotation', '2nd LM Broad Annotation'),
        
        # 3rd Closest Landmark
        ('third_closest_landmark_moa_first', '3rd Landmark'),
        ('third_closest_landmark_PP_ID_uM', '3rd LM PP ID'),
        ('third_closest_landmark_distance', '3rd LM Distance'),
        ('third_closest_landmark_annotated_target_description_truncated_10', '3rd LM Target Desc'),
        ('third_closest_landmark_manual_annotation', '3rd LM Broad Annotation'),
    ]
    
    @classmethod
    def get_hover_columns(cls, data_type: str) -> list:
        """Get the appropriate hover columns for SPC or CP data."""
        if data_type.lower() == 'cp':
            return cls.CP_HOVER_COLUMNS
        else:
            return cls.SPC_HOVER_COLUMNS
    
    @classmethod
    def get_hover_display(cls, data_type: str) -> list:
        """Get the hover display mapping for SPC or CP data."""
        if data_type.lower() == 'cp':
            return cls.CP_HOVER_DISPLAY
        else:
            return cls.SPC_HOVER_DISPLAY
    
    @classmethod
    def get_plate_column(cls, data_type: str) -> str:
        """Get the plate column name (same for both now due to aliases)."""
        return 'plate'
    
    @classmethod
    def get_well_column(cls, data_type: str) -> str:
        """Get the well column name (same for both now due to aliases)."""
        return 'well'
    
    @classmethod
    def get_smiles_column(cls, data_type: str) -> str:
        """Get the SMILES column name (same for both now due to aliases)."""
        return 'SMILES'
    
    # ============================================================================
    # WELL PLATE SETTINGS
    # ============================================================================
    WELL_ROWS = 'ABCDEFGHIJKLMNOP'  # Standard 384-well plate rows
    WELL_COLUMNS_MAX = 24  # Standard 384-well plate columns
    
    # ============================================================================
    # Class methods for main data files:
      # SPC analysis
      # Cell Profiler analysis
    # ============================================================================
    
    @classmethod
    def get_visualization_data_path(cls) -> Path:
        """Get the path to the main visualization data file."""
        return cls.SPC_ANALYSIS_DIR / "data" / "spc_for_viz_app.csv"
    
    @classmethod
    def get_cp_visualization_data_path(cls) -> Path:
        """Get the path to the CellProfiler visualization data file."""
        return cls.CP_ANALYSIS_DIR / "data" / "cp_for_viz_app.csv"
    
    # ============================================================================
    # Landmark Paths:
      # SPC analysis
      # Cell Profiler analysis
    # ============================================================================

    # ============================================================================
    # SPC LANDMARK PATHS
    # ============================================================================
    @classmethod
    def get_spc_landmark_distance_test_path(cls) -> Path:
        """Get the path to the SPC test landmark distance file - NEW SLIM FORMAT."""
        return cls.SPC_ANALYSIS_DIR / "analysis" / "landmark_distances" / "test_distances.parquet"

    @classmethod
    def get_spc_landmark_distance_reference_path(cls) -> Path:
        """Get the path to the SPC reference landmark distance file - NEW SLIM FORMAT."""
        return cls.SPC_ANALYSIS_DIR / "analysis" / "landmark_distances" / "reference_distances.parquet"

    @classmethod
    def get_spc_landmark_metadata_path(cls) -> Path:
        """
        Get the path to the SPC landmark metadata file - NEW SLIM FORMAT.
        """
        return cls.SPC_ANALYSIS_DIR / "analysis" / "landmark_distances" / "landmark_metadata.parquet"

    # ============================================================================
    # CELLPROFILER LANDMARK PATHS
    # ============================================================================
    @classmethod
    def get_cp_landmark_distance_test_path(cls) -> Path:
        """Get the path to the test landmark distance file (CellProfiler data) - SLIM FORMAT."""
        return cls.CP_ANALYSIS_DIR / "landmark_analysis" / "test_distances.parquet"
    
    @classmethod
    def get_cp_landmark_distance_reference_path(cls) -> Path:
        """Get the path to the reference landmark distance file (CellProfiler data) - SLIM FORMAT."""
        return cls.CP_ANALYSIS_DIR / "landmark_analysis" / "reference_distances.parquet"
    
    @classmethod
    def get_cp_landmark_metadata_path(cls) -> Path:
        """
        Get the path to the landmark metadata file (CellProfiler data) - SLIM FORMAT.
        """
        return cls.CP_ANALYSIS_DIR / "landmark_analysis" / "landmark_metadata.parquet"
    

    @classmethod
    def validate_paths(cls) -> bool:
        """
        Validate that required paths exist and are accessible.
        
        Returns:
            bool: True if all paths are valid, False otherwise
        """
        valid = True
        
        # Check SPC analysis directory
        if not cls.SPC_ANALYSIS_DIR.exists():
            print(f"ERROR: SPC analysis directory not found: {cls.SPC_ANALYSIS_DIR}")
            valid = False
        else:
            print(f"✓ Found SPC analysis directory: {cls.SPC_ANALYSIS_DIR}")

            # Check SPC landmark paths (NEW slim format)
            spc_landmark_test = cls.get_spc_landmark_distance_test_path()
            spc_landmark_ref = cls.get_spc_landmark_distance_reference_path()
            spc_landmark_meta = cls.get_spc_landmark_metadata_path()

            if spc_landmark_test.exists():
                print(f"  ✓ Found SPC test distances (slim format): {spc_landmark_test.name}")
            else:
                print(f"  ✗ SPC test distances not found: {spc_landmark_test}")

            if spc_landmark_ref.exists():
                print(f"  ✓ Found SPC reference distances (slim format): {spc_landmark_ref.name}")
            else:
                print(f"  ✗ SPC reference distances not found: {spc_landmark_ref}")

            if spc_landmark_meta.exists():
                print(f"  ✓ Found SPC landmark metadata: {spc_landmark_meta.name}")
            else:
                print(f"  ✗ SPC landmark metadata not found: {spc_landmark_meta}")

        
        # Check CP analysis directory
        if not cls.CP_ANALYSIS_DIR.exists():
            print(f"WARNING: CP analysis directory not found: {cls.CP_ANALYSIS_DIR}")
            print("CellProfiler landmark analysis will not be available")
        else:
            print(f"✓ Found CP analysis directory: {cls.CP_ANALYSIS_DIR}")
            
            # Check for new slim format files
            cp_test_path = cls.get_cp_landmark_distance_test_path()
            cp_ref_path = cls.get_cp_landmark_distance_reference_path()
            cp_meta_path = cls.get_cp_landmark_metadata_path()
            
            if cp_test_path.exists():
                print(f"  ✓ Found CP test distances (slim format): {cp_test_path.name}")
            else:
                print(f"  ✗ CP test distances not found: {cp_test_path}")
                
            if cp_ref_path.exists():
                print(f"  ✓ Found CP reference distances (slim format): {cp_ref_path.name}")
            else:
                print(f"  ✗ CP reference distances not found: {cp_ref_path}")
                
            if cp_meta_path.exists():
                print(f"  ✓ Found CP landmark metadata: {cp_meta_path.name}")
            else:
                print(f"  ✗ CP landmark metadata not found: {cp_meta_path}")
        
        # Check thumbnail directories (warning only, not critical)
        thumbnails_found = False
        for thumbnail_dir in cls.THUMBNAIL_DIRS:
            if thumbnail_dir.exists():
                thumbnails_found = True
                print(f"✓ Found thumbnail directory: {thumbnail_dir}")
            else:
                print(f"WARNING: Thumbnail directory not found: {thumbnail_dir}")
        
        if not thumbnails_found:
            print("WARNING: No thumbnail directories found - image display will not work")
        

        # Check exists: main visualization data files (SPC and CellProfiler)

        # Check SPC visualization data file
        viz_data_path = cls.get_visualization_data_path()
        if not viz_data_path.exists():
            print(f"ERROR: SPC visualization data file not found: {viz_data_path}")
            valid = False
        else:
            print(f"✓ Found SPC visualization data file: {viz_data_path.name}")

        # Check CP visualization data file
        cp_viz_data_path = cls.get_cp_visualization_data_path()
        if not cp_viz_data_path.exists():
            print(f"ERROR: CP visualization data file not found: {cp_viz_data_path}")
            valid = False
        else:
            print(f"✓ Found CP visualization data file: {cp_viz_data_path.name}")
    


    @classmethod
    def get_well_list(cls) -> List[str]:
        """Generate ordered list of all possible wells (A01-P24)."""
        wells = []
        for row in cls.WELL_ROWS:
            for col in range(1, cls.WELL_COLUMNS_MAX + 1):
                wells.append(f"{row}{col:02d}")
        return wells
    
    @classmethod
    def get_column_list(cls) -> List[str]:
        """Generate ordered list of well columns with zero padding."""
        return [f"{i:02d}" for i in range(1, cls.WELL_COLUMNS_MAX + 1)]
    
    @classmethod
    def get_site_range(cls) -> Tuple[int, int]:
        """Get the range for random site selection."""
        return cls.RANDOM_SITE_RANGE


# Environment-specific overrides (optional)
class DevelopmentConfig(Config):
    """Development-specific configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production-specific configuration."""
    DEBUG = False
    HOST = '127.0.0.0'  # More restrictive in production


# Default configuration
config = Config