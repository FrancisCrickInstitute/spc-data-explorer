# SPC Data Explorer

An interactive dashboard for exploring Cell Painting phenotypic screening data, built with Dash and Plotly. Supports both **SPC (Spherical Phenotype Clustering)** and **CellProfiler** analysis pipelines with unified visualisation capabilities.


## Overview

This application provides an interactive web-based interface for exploring high-content screening data from Cell Painting assays. It was developed to:

- **Visualise compound phenotypes** in reduced dimensionality space (UMAP/t-SNE)
- **Compare analysis methods** by switching between SPC and CellProfiler pipelines
- **Identify phenotypic neighbours** through landmark-based distance analysis
- **Explore compound metadata** including MOA, target annotations, and chemical structures

The dashboard integrates multiple data sources including morphological features, compound annotations, chemoproteomics data, and microscopy images to provide a comprehensive view of phenotypic screening results.

---

## Features

### Dual Pipeline Support
The application supports data from two distinct analysis pipelines:
- **SPC (Spherical Phenotype Clustering)**: A machine learning approach using ResNet-based feature extraction and cosine similarity metrics
- **CellProfiler**: Traditional morphological profiling with standardised feature sets

Each pipeline has its own column naming conventions, which the app handles transparently through configurable column mappings.

### Interactive Visualisation
- **UMAP/t-SNE scatter plots** for both SPC and CellProfiler datasets
- **Dynamic colour mapping** by:
  - Library source (GSK, JUMP, SGC, etc.)
  - Mechanism of Action (MOA)
  - Landmark proximity status
  - Plate/well location
  - Various distance metrics
- **Compound search** with autocomplete supporting:
  - PP_ID (e.g., `PPXXXX@1.0`)
  - Treatment names (e.g., `CompoundXXXX@0.1`)
  - MOA/gene names (e.g., `UNG@0.1 (CR000023@0.1)`)
- **Visual highlighting** of selected compounds on the plot

### Microscopy Image Integration
- **Hover preview**: See microscopy thumbnails instantly when hovering over data points
- **Click for details**: Full compound information panel with larger image
- **Multiple scaling modes**: Fixed (comparable across images) or auto-scaled (per-image optimisation)
- **Text overlays**: Optional treatment or MOA labels on images
- **Multi-site support**: Random site selection from available fields of view

### Landmark Analysis
Reference compounds ("landmarks") with known mechanisms serve as anchors for phenotypic interpretation:
- **Distance calculations** to three closest landmarks for each compound
- **Validity indicators** showing if compounds fall within meaningful distance thresholds
- **Detailed landmark information** including:
  - MOA/target annotations
  - PP_ID identifiers
  - Cosine distances
  - Broad Institute annotations

### Rich Metadata Display
Hover and click interactions reveal comprehensive compound information:
- **Basic info**: Treatment, plate, well, concentration, library
- **Annotations**: MOA, target description, Broad annotation
- **Chemical structure**: Rendered from SMILES using RDKit
- **Chemoproteomics**: Protein targets from pulldown experiments
- **Gene descriptions**: Functional annotations for target genes
- **Distance metrics**: MAD cosine, variance, standard deviation measures

---

## Input Files

The app loads data from two upstream analysis pipelines. Each pipeline produces a set of files that must be present in specific locations.

### Directory Structure Overview

```
SPC_ANALYSIS_DIR/
├── data/
│   └── spc_for_viz_app.csv              # Main visualisation data (primary)
└── analysis/
    └── landmark_distances/
        ├── test_distances.parquet        # Landmark distances for test compounds
        ├── reference_distances.parquet   # Landmark distances for reference compounds
        ├── landmark_metadata.parquet     # Metadata for all landmarks
        ├── test_spc_landmark_options_cache.json      # Auto-generated cache
        └── reference_spc_landmark_options_cache.json # Auto-generated cache

CP_ANALYSIS_DIR/
├── data/
│   └── cp_for_viz_app.csv               # Main visualisation data (primary)
└── landmark_analysis/
    ├── test_distances.parquet            # Landmark distances for test compounds
    ├── reference_distances.parquet       # Landmark distances for reference compounds
    ├── landmark_metadata.parquet         # Metadata for all landmarks
    ├── test_cp_landmark_options_cache.json       # Auto-generated cache
    └── reference_cp_landmark_options_cache.json  # Auto-generated cache

THUMBNAIL_DIR/
├── fixed/                                # Fixed intensity scaling
│   └── {plate}/
│       └── {plate}_{well}_{site}.png
└── auto/                                 # Auto-scaled per image
    └── {plate}/
        └── {plate}_{well}_{site}.png
```

### Main Visualisation Data

These CSV files contain the pre-computed coordinates and metadata for plotting.

| Pipeline | File | Location | Description |
|----------|------|----------|-------------|
| SPC | `spc_for_viz_app.csv` | `{SPC_ANALYSIS_DIR}/data/` | UMAP/t-SNE coordinates, landmark info, compound scores |
| CellProfiler | `cp_for_viz_app.csv` | `{CP_ANALYSIS_DIR}/data/` | UMAP/t-SNE coordinates, landmark info, metadata |

**SPC columns include**: `UMAP1`, `UMAP2`, `TSNE1`, `TSNE2`, `plate`, `well`, `treatment`, `PP_ID`, `PP_ID_uM`, `library`, `moa_first`, `moa_compound_uM`, `SMILES`, `mad_cosine`, `cosine_distance_from_dmso`, `closest_landmark_*`, `second_closest_landmark_*`, `third_closest_landmark_*`, `score_harmonic_mean_*`

**CellProfiler columns include**: `UMAP1`, `UMAP2`, `TSNE1`, `TSNE2`, `Metadata_plate_barcode`, `Metadata_well`, `treatment`, `Metadata_PP_ID`, `Metadata_PP_ID_uM`, `Metadata_library`, `Metadata_annotated_target_first`, `Metadata_SMILES`, `closest_landmark_Metadata_*`, etc.

### Landmark Distance Files (Slim Format)

These parquet files power the landmark analysis modal. They use a "slim format" where each row is one compound/treatment, with distances to all landmarks stored as separate columns.

| Pipeline | File | Location | Description |
|----------|------|----------|-------------|
| SPC | `test_distances.parquet` | `{SPC_ANALYSIS_DIR}/analysis/landmark_distances/` | Distances from test compounds to all landmarks |
| SPC | `reference_distances.parquet` | `{SPC_ANALYSIS_DIR}/analysis/landmark_distances/` | Distances from reference compounds to all landmarks |
| SPC | `landmark_metadata.parquet` | `{SPC_ANALYSIS_DIR}/analysis/landmark_distances/` | Metadata for each landmark (MOA, PP_ID, etc.) |
| CellProfiler | `test_distances.parquet` | `{CP_ANALYSIS_DIR}/landmark_analysis/` | Distances from test compounds to all landmarks |
| CellProfiler | `reference_distances.parquet` | `{CP_ANALYSIS_DIR}/landmark_analysis/` | Distances from reference compounds to all landmarks |
| CellProfiler | `landmark_metadata.parquet` | `{CP_ANALYSIS_DIR}/landmark_analysis/` | Metadata for each landmark |

**Slim format structure** (`test_distances.parquet` / `reference_distances.parquet`):
- One row per treatment/compound
- `treatment` column: unique identifier
- `query_dmso_distance`: distance from DMSO control
- `query_mad`: MAD-based dispersion metric
- `{landmark_treatment}_distance`: distance to each landmark (one column per landmark)
- Metadata columns: plate, well, library, PP_ID, MOA, SMILES, etc.

**Landmark metadata structure** (`landmark_metadata.parquet`):
- One row per landmark
- `treatment`: landmark identifier (e.g., `CompoundX@1.0`)
- `moa_first` / `Metadata_annotated_target_first`: mechanism of action
- `PP_ID_uM` / `Metadata_PP_ID_uM`: compound identifier with concentration
- Additional annotation columns

### Landmark Options Cache Files (Auto-generated)

The landmark loader creates JSON cache files to avoid rebuilding dropdown options on each app startup (which takes 30-60 seconds). These are generated automatically and stored alongside the parquet files:

| File | Location | Description |
|------|----------|-------------|
| `test_spc_landmark_options_cache.json` | `{SPC_ANALYSIS_DIR}/analysis/landmark_distances/` | Cached dropdown options for SPC test landmarks |
| `reference_spc_landmark_options_cache.json` | `{SPC_ANALYSIS_DIR}/analysis/landmark_distances/` | Cached dropdown options for SPC reference landmarks |
| `test_cp_landmark_options_cache.json` | `{CP_ANALYSIS_DIR}/landmark_analysis/` | Cached dropdown options for CP test landmarks |
| `reference_cp_landmark_options_cache.json` | `{CP_ANALYSIS_DIR}/landmark_analysis/` | Cached dropdown options for CP reference landmarks |

The cache is automatically invalidated and rebuilt if the source parquet file is newer than the cache file. To force a rebuild, simply delete the relevant `.json` cache file.

### Thumbnail Images

RGB thumbnail images (500×500 pixels) displayed on hover and click.

| Scaling Mode | Location | Description |
|--------------|----------|-------------|
| Fixed | `{THUMBNAIL_DIR}/fixed/{plate}/{plate}_{well}_{site}.png` | Pre-defined intensity limits for cross-image comparison |
| Auto | `{THUMBNAIL_DIR}/auto/{plate}/{plate}_{well}_{site}.png` | Per-image 1st-99th percentile scaling |

Multiple thumbnail directories can be specified in the config (e.g., one per imaging project).

### Fallback Loading Logic

The data loader (`loader.py`) implements fallback logic:
1. **Primary**: Load from CSV file (`spc_for_viz_app.csv` or `cp_for_viz_app.csv`)
2. **Fallback**: If CSV not found, attempt to load and combine parquet files from `landmark_distances/` or `landmark_analysis/` directories

---

## Data Sources

This visualisation app displays data generated by two upstream analysis pipelines.

### SPC Analysis Pipeline
- **Repository**: [spc-cosine-analysis](https://github.com/FrancisCrickInstitute/spc-cosine-analysis)
- **Generates**:
  - `spc_for_viz_app.csv` — Main visualisation data with UMAP/t-SNE coordinates
  - `test_distances.parquet` — Landmark distances for test compounds
  - `reference_distances.parquet` — Landmark distances for reference compounds
  - `landmark_metadata.parquet` — Landmark compound metadata
- **Input**: SPC embeddings from [spc-distributed](https://github.com/FrancisCrickInstitute/spc-distributed)

### CellProfiler Analysis Pipeline  
- **Repository**: [cellprofiler_processing](https://github.com/FrancisCrickInstitute/cellprofiler_processing)
- **Generates**:
  - `cp_for_viz_app.csv` — Main visualisation data with UMAP/t-SNE coordinates
  - `test_distances.parquet` — Landmark distances for test compounds
  - `reference_distances.parquet` — Landmark distances for reference compounds
  - `landmark_metadata.parquet` — Landmark compound metadata
- **Input**: CellProfiler feature extraction output

### Column Name Mapping

The app harmonises different column naming conventions between pipelines:

| SPC Column | CellProfiler Column | Description |
|------------|---------------------|-------------|
| `plate` | `Metadata_plate_barcode` | Plate identifier |
| `well` | `Metadata_well` | Well position |
| `PP_ID` | `Metadata_PP_ID` | Compound identifier |
| `PP_ID_uM` | `Metadata_PP_ID_uM` | Compound ID with concentration |
| `library` | `Metadata_library` | Source library |
| `moa_first` | `Metadata_annotated_target_first` | Primary MOA/target |
| `SMILES` | `Metadata_SMILES` | Chemical structure |
| `compound_name` | `Metadata_chemical_name` | Compound name |
| `compound_uM` | `Metadata_compound_uM` | Concentration |
| `mad_cosine` | `query_mad` | MAD-based dispersion |
| `cosine_distance_from_dmso` | `query_dmso_distance` | Distance from DMSO |

---

## Project Structure

```
spc-data-explorer/
└── scripts/
    ├── main.py                      # Application entry point
    ├── config_loader.py             # Configuration management (singleton pattern)
    ├── environment.yml              # Conda environment specification
    ├── requirements.txt             # Pip dependencies (alternative to conda)
    │
    ├── callbacks/
    │   ├── __init__.py
    │   ├── plot_callbacks.py        # Main scatter plot generation and updates
    │   ├── image_callbacks.py       # Hover/click image display with metadata
    │   ├── search_callbacks.py      # MOA-based compound search functionality
    │   ├── detailed_search_callbacks.py  # Advanced search with multiple criteria
    │   └── landmark_callbacks.py    # Landmark analysis modal and calculations
    │
    ├── components/
    │   ├── __init__.py
    │   ├── layout.py                # Dashboard layout structure
    │   ├── controls.py              # UI control panels (dropdowns, sliders)
    │   └── search.py                # Search component builders
    │
    ├── config/
    │   └── config_20251118_TEST_INPUTS.py  # ✅ RECOMMENDED: Latest config
    │
    ├── data/
    │   ├── __init__.py
    │   ├── loader.py                # Main data loading with column harmonisation
    │   └── landmark_loader.py       # Landmark data processing and validation
    │
    ├── utils/
    │   ├── __init__.py
    │   ├── color_utils.py           # Colour palette management for categories
    │   ├── image_utils.py           # Thumbnail finding and image processing
    │   └── smiles_utils.py          # Chemical structure rendering via RDKit
    │
    └── generate_thumbnails/
        ├── scripts/
        │   └── generate_thumbnails_perc_and_auto_thresh_V1.py  # Thumbnail generator
        └── submit/
            └── thumbnails_*.sh      # SLURM submission scripts for each dataset
```

### Module Descriptions

| Module | File | Description |
|--------|------|-------------|
| **Entry Point** | `main.py` | Initialises Dash app, loads data, registers callbacks |
| **Config Loader** | `config_loader.py` | Singleton pattern for interactive config selection |
| **Data Loader** | `data/loader.py` | Loads SPC and CP datasets, creates column aliases |
| **Landmark Loader** | `data/landmark_loader.py` | Loads slim-format parquet files, builds dropdown options |
| **Plot Callbacks** | `callbacks/plot_callbacks.py` | Generates scatter plots, handles plot type switching |
| **Image Callbacks** | `callbacks/image_callbacks.py` | Hover/click handlers, thumbnail display, metadata panels |
| **Search Callbacks** | `callbacks/search_callbacks.py` | Compound search with autocomplete |
| **Landmark Callbacks** | `callbacks/landmark_callbacks.py` | Landmark analysis modal, distance scatter plots |
| **Layout** | `components/layout.py` | Dashboard HTML structure and component arrangement |
| **Controls** | `components/controls.py` | Dropdowns, sliders, radio buttons for UI |
| **Colour Utils** | `utils/color_utils.py` | Colour palettes for categorical/continuous data |
| **Image Utils** | `utils/image_utils.py` | Thumbnail path resolution, image loading, text overlay |
| **SMILES Utils** | `utils/smiles_utils.py` | RDKit-based chemical structure rendering |

### Key Configuration Note

> ✅ **Use `config_20251118_TEST_INPUTS.py` as your starting point.** 

This is the latest configuration file that includes:
- Separate loading logic for SPC and CellProfiler datasets
- Correct column name mappings for both pipelines
- Hover column definitions for each data type
- Plot type configurations for all four views (SPC UMAP/t-SNE, CP UMAP/t-SNE)

---

## Thumbnail Generation

The `generate_thumbnails/` directory contains scripts for creating RGB thumbnail images from multi-channel Cell Painting microscopy data. These thumbnails are displayed in the dashboard when hovering over or clicking on data points.

### Overview

Cell Painting assays typically acquire 4-5 fluorescent channels per field of view. The thumbnail generator combines these channels into false-colour RGB thumbnails (500×500 pixels) suitable for quick visual inspection.

### Scaling Modes

The script produces two versions of each thumbnail:

| Mode | Directory | Description | Best For |
|------|-----------|-------------|----------|
| **Fixed** | `fixed/` | Pre-defined intensity limits based on dataset-wide percentiles | Comparing phenotypes across treatments, identifying outliers |
| **Auto** | `auto/` | Per-image 1st-99th percentile scaling | Examining morphological details, dim images, QC checking |

### Channel Mapping

Fluorescent channels are mapped to RGB colours:
- **Blue**: Nuclear stains (HOECHST 33342, DAPI)
- **Green**: Alexa 488, FITC (ER, actin, cytoplasmic markers)
- **Red**: Alexa 568, MitoTracker Deep Red, Cy5 (mitochondria, membrane)

### Usage

Basic usage:
```bash
python generate_thumbnails_perc_and_auto_thresh_V1.py \
    /path/to/max_projected_images \
    /path/to/output/thumbnails \
    --scaling both
```

Scan directories first (planning mode):
```bash
python generate_thumbnails_perc_and_auto_thresh_V1.py \
    /path/to/images \
    /path/to/thumbnails \
    --scan-only \
    --input-dirs /other/path1 /other/path2
```

### SLURM Submission

For HPC environments, use the submission scripts in `submit/`:

```bash
sbatch thumbnails_20251020_HaCaT_HTC_V1_V2_cell_paint.sh
```

Example SLURM configuration:
```bash
#SBATCH --job-name=thumbnails
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=ncpu
```

### Output Structure

```
thumbnails/
├── fixed/                    # Fixed intensity scaling
│   ├── {plate_barcode}/
│   │   ├── {plate}_{well}_{site}.png
│   │   └── ...
│   └── ...
└── auto/                     # Auto-scaled per image
    ├── {plate_barcode}/
    │   └── ...
    └── ...
```

---

## Installation

### Prerequisites
- Python 3.11+
- Conda (recommended) or pip
- Access to data files (parquet/CSV) and thumbnail images

### Setup with Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/FrancisCrickInstitute/spc-data-explorer.git
cd spc-data-explorer/scripts

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate spc_visualisation
```

### Setup with Pip (Alternative)

```bash
# Clone the repository
git clone https://github.com/FrancisCrickInstitute/spc-data-explorer.git
cd spc-data-explorer/scripts

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### RDKit Note
RDKit is required for chemical structure rendering. It's easiest to install via conda:
```bash
conda install -c conda-forge rdkit
```

---

## Configuration

### 1. Create Your Configuration File

```bash
# Copy the recommended config
cp config/config_20251118_TEST_INPUTS.py config/config_myproject.py
```

### 2. Update Paths

Edit your configuration file to point to your data:

```python
class Config:
    # SPC Analysis Directory (contains data/ and analysis/ subdirs)
    SPC_ANALYSIS_DIR = Path("/path/to/spc_analysis_output")
    
    # CellProfiler Analysis Directory (contains data/ and landmark_analysis/ subdirs)
    CP_ANALYSIS_DIR = Path("/path/to/cellprofiler_output")
    
    # Thumbnail directories (list of paths, each containing fixed/ and auto/ subdirs)
    THUMBNAIL_DIRS = [
        Path("/path/to/thumbnails/project1"),
        Path("/path/to/thumbnails/project2"),
    ]
```

### 3. Environment-Specific Paths

The template supports automatic path switching based on username:

```python
if os.environ.get("USER", "") == "your_cluster_username":
    # Cluster paths (e.g., /nemo/...)
    SPC_ANALYSIS_DIR = Path("/nemo/path/to/analysis")
else:
    # Local paths (e.g., mounted volumes)
    SPC_ANALYSIS_DIR = Path("/Volumes/path/to/analysis")
```

### 4. Verify Configuration

The config includes a `validate_paths()` method that checks all required files exist:

```python
from config_loader import get_config
Config = get_config()
Config.validate_paths()
```

This will print a checklist showing which files are found/missing.

---

## Usage

### Starting the Application

```bash
cd scripts/

# Interactive config selection (prompts you to choose)
python main.py

# Or specify config directly
python main.py --config config_myproject

# Or use environment variable
export SPC_CONFIG=config_myproject
python main.py
```

The app will start at `http://127.0.0.1:8090` (or the port specified in your config).

### Dashboard Navigation

1. **Select Plot Type**: Choose from:
   - SPC UMAP / t-SNE
   - CellProfiler UMAP / t-SNE
   - Custom axes

2. **Colour By**: Select metadata column for point colouring:
   - Library, MOA, landmark status
   - Plate, well location
   - Distance metrics (continuous colour scales)

3. **Search Compounds**: Type to search by:
   - Compound ID: `PPXXXX`
   - Treatment: `CompoundXXXX`
   - Gene/MOA: `UNG` → shows `UNG@0.1 (CR000023@0.1)`

4. **Interact with Plot**:
   - **Hover**: See microscopy image preview + key metadata
   - **Click**: Open detailed compound panel with full information
   - **Zoom/Pan**: Standard Plotly interactions

5. **Adjust Settings**:
   - Point size slider
   - Image scaling mode (fixed/auto)
   - Optional text labels on images

6. **Landmark Analysis Modal**:
   - Click "Open Landmark Distance Analysis" button
   - Select a landmark from the dropdown
   - View scatter plot of all compounds vs selected landmark distance

---

## Troubleshooting

### Common Issues

**"No data available" error**
- Check that your data paths in the config file are correct
- Run `Config.validate_paths()` to see which files are missing
- Verify the CSV/parquet files exist and are readable
- Ensure required columns are present in your data

**Images not displaying**
- Verify thumbnail directory paths are correct
- Check that `fixed/` and `auto/` subdirectories exist within each thumbnail directory
- Confirm image naming convention: `{plate}_{well}_{site}.png`
- Check the app logs for "Cannot load image" warnings

**Landmark analysis not working**
- Ensure all three parquet files exist: `test_distances.parquet`, `reference_distances.parquet`, `landmark_metadata.parquet`
- Check they are in the correct subdirectory (`analysis/landmark_distances/` for SPC, `landmark_analysis/` for CP)
- Verify the slim format structure with distance columns ending in `_distance`
- If dropdown options seem stale/incorrect, delete the `*_landmark_options_cache.json` files to force a rebuild

**Slow performance with large datasets**
- Consider filtering data before loading
- Reduce the number of hover columns in config
- First app startup builds landmark caches (30-60 seconds) — subsequent loads are fast
- The landmark loader caches options to disk (`.json` files) after first load

**RDKit import errors**
- Install RDKit via conda: `conda install -c conda-forge rdkit`
- If using pip, RDKit installation can be complex - conda is recommended

**Column not found errors**
- Check that your data has the expected column names
- For CellProfiler data, columns should be prefixed with `Metadata_`
- The loader creates aliases automatically, but source columns must exist

---

## Related Repositories

- [spc-distributed](https://github.com/FrancisCrickInstitute/spc-distributed) — Upstream SPC embedding generation using ResNet models
- [spc-cosine-analysis](https://github.com/FrancisCrickInstitute/spc-cosine-analysis) — SPC-based phenotypic analysis pipeline (generates input for this app)
- [cellprofiler_processing](https://github.com/FrancisCrickInstitute/cellprofiler_processing) — CellProfiler-based analysis pipeline (generates input for this app)