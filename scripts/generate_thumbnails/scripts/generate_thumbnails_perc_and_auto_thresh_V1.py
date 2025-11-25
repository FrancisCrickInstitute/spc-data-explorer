"""
Thumbnail Generator for Cell Painting Microscopy Images
========================================================

This script generates RGB thumbnail images from multi-channel Cell Painting 
microscopy data. It produces two versions of each thumbnail with different 
intensity scaling approaches, enabling both cross-image comparability and 
per-image optimisation.

Overview
--------
Cell Painting assays typically acquire 4-5 fluorescent channels per field of view.
This script combines these channels into false-colour RGB thumbnails suitable for
quick visual inspection in the SPC visualisation dashboard.

Scaling Modes
-------------
The script generates thumbnails with two intensity scaling approaches:

1. **Fixed Scaling** (`fixed/` directory)
   - Uses pre-defined intensity limits based on dataset-wide percentiles
   - Enables direct visual comparison across all images
   - Consistent appearance regardless of individual image intensities
   - Best for: comparing phenotypes across treatments, identifying outliers
   
   Default intensity limits (derived from dataset-wide percentile analysis):
   - HOECHST 33342 (nuclear): 89 - 14,081
   - Alexa 488 (green): 121 - 26,431  
   - Alexa 568 (red): 149 - 42,714
   - MitoTracker Deep Red: 183 - 23,096

2. **Auto Scaling** (`auto/` directory)
   - Uses per-image 1st-99th percentile scaling
   - Optimises dynamic range for each individual image
   - Shows maximum detail within each image
   - Best for: examining morphological details, dim images, QC checking

Channel Mapping
---------------
Fluorescent channels are mapped to RGB colours as follows:
- **Blue**: Nuclear stains (HOECHST 33342, DAPI)
- **Green**: Alexa 488, FITC (typically ER, actin, or other cytoplasmic markers)
- **Red**: Alexa 568, MitoTracker Deep Red, Cy5 (mitochondria, membrane markers)

The script handles multiple naming conventions:
- New format: "A01_01_HOECHST 33342.tiff"
- Old format: "A01_1_0_hoechst.tiff"

Input Requirements
------------------
- Directory structure: {input_dir}/{plate_barcode}/*.tiff
- Multi-channel TIFF files (16-bit recommended)
- Supported channels: HOECHST/DAPI, Alexa 488/FITC, Alexa 568/TRITC, MitoTracker
- Brightfield channels are automatically skipped

Output Structure
----------------
    {output_dir}/
    ├── fixed/                    # Fixed intensity scaling
    │   ├── {plate_barcode}/
    │   │   ├── {plate}_{well}_{site}.png
    │   │   └── ...
    │   └── ...
    └── auto/                     # Auto-scaled per image
        ├── {plate_barcode}/
        │   ├── {plate}_{well}_{site}.png
        │   └── ...
        └── ...

Thumbnail specifications:
- Size: 500 x 500 pixels
- Format: PNG (8-bit RGB)
- Naming: {plate_barcode}_{well}_{site}.png

Usage
-----
Basic usage:
    python generate_thumbnails_perc_and_auto_thresh_V1.py \\
        /path/to/max_projected_images \\
        /path/to/output/thumbnails

With options:
    python generate_thumbnails_perc_and_auto_thresh_V1.py \\
        /path/to/images \\
        /path/to/thumbnails \\
        --scaling both \\        # 'fixed', 'auto', or 'both'
        --debug                  # Enable verbose output

Scan multiple directories (planning mode):
    python generate_thumbnails_perc_and_auto_thresh_V1.py \\
        /path/to/images \\
        /path/to/thumbnails \\
        --scan-only \\
        --input-dirs /other/path1 /other/path2

SLURM Submission
----------------
For HPC environments, use a submission script like:

    #!/bin/bash
    #SBATCH --job-name=thumbnails
    #SBATCH --time=168:00:00
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=16
    #SBATCH --mem=64G
    #SBATCH --partition=ncpu

    PYTHON="/path/to/conda/envs/make_thumbnails/bin/python"
    SCRIPT="/path/to/generate_thumbnails_perc_and_auto_thresh_V1.py"
    
    $PYTHON $SCRIPT "/path/to/images" "/path/to/thumbnails" --scaling both

Performance
-----------
- Uses multiprocessing (16 cores by default) for parallel processing
- Memory usage scales with image size and number of channels
- Typical throughput: ~1000-2000 thumbnails per hour on HPC

Dependencies
------------
- numpy
- pandas  
- scikit-image (skimage)
- tqdm

Notes
-----
- Plates with barcodes starting with '0000' are automatically filtered out
  (these are typically test/calibration plates)
- Missing channels result in a black channel in the RGB output (with warning)
- The script reports statistics before and after processing
"""


import multiprocessing
import argparse
import os
from dataclasses import dataclass
from glob import glob
import pandas as pd 
import numpy as np
from skimage import io, transform, util
from tqdm import tqdm
import sys
import re

THUMBNAIL_SIZE = (500, 500)

# Update the CH_LIMITS dictionary to use your calculated percentiles as (low, high) tuples. Can just keep these the same across datasets now for comparability.
CH_LIMITS = {
    # New naming format with calculated percentiles
    "HOECHST 33342": (89, 14081),     # Using both low and high percentiles
    "Alexa 488": (121, 26431),
    "Alexa 568": (149, 42714),
    "MitoTracker Deep Red": (183, 23096),
    
    # Old naming format (keep these for backward compatibility)
    "hoechst": (89, 14081),
    "alexa488": (121, 26431), 
    "alexa568": (149, 42714),
    "mitotracker": (183, 23096)
}

@dataclass
class Metadata:
    barcode: str
    well: str
    site: str
    channel: str
    path: str

def scan_all_input_folders(input_dirs):
    """Scan all input folders for plates and report totals"""
    print("\n====== PLATE BARCODE SUMMARY ACROSS ALL INPUT FOLDERS ======")
    
    all_plates = {}  # Dictionary to track plates and their source folder
    total_tiff_files = 0
    
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"Directory does not exist: {input_dir}")
            continue
            
        print(f"\nScanning input folder: {input_dir}")
        
        # Find all plates in this directory
        plate_dirs = glob(os.path.join(input_dir, "*/"))
        plates_in_folder = [os.path.basename(d.rstrip('/')) for d in plate_dirs]
        
        # Count TIFF files
        folder_tiff_count = len(glob(os.path.join(input_dir, "*", "*.tiff")))
        total_tiff_files += folder_tiff_count
        
        print(f"  Found {len(plates_in_folder)} plates with {folder_tiff_count} TIFF files")
        
        # List all plates in this folder
        if plates_in_folder:
            print("  Plates in this folder:")
            for plate in sorted(plates_in_folder):
                # Count TIFF files for this plate
                plate_tiff_count = len(glob(os.path.join(input_dir, plate, "*.tiff")))
                print(f"    {plate}: {plate_tiff_count} TIFF files")
                
                # Add to overall plate tracking
                if plate not in all_plates:
                    all_plates[plate] = []
                all_plates[plate].append(input_dir)
    
    # Print overall summary
    print("\n=== OVERALL SUMMARY ===")
    print(f"Total unique plates across all folders: {len(all_plates)}")
    print(f"Total TIFF files across all folders: {total_tiff_files}")
    
    # Print plates in multiple folders
    duplicate_plates = {plate: folders for plate, folders in all_plates.items() if len(folders) > 1}
    if duplicate_plates:
        print("\nPlates found in multiple folders:")
        for plate, folders in duplicate_plates.items():
            formatted_folders = [os.path.basename(f) for f in folders]
            print(f"  {plate}: Found in {len(folders)} folders: {', '.join(formatted_folders)}")
    
    print("=================================================\n")
    
    return all_plates, total_tiff_files

def report_plates(search_dir: str):
    """Report on plates found in a single directory"""
    print(f"\n====== PLATE REPORT: {os.path.basename(search_dir)} ======")
    
    plate_dirs = glob(os.path.join(search_dir, "*/"))
    plate_names = [os.path.basename(d.rstrip('/')) for d in plate_dirs]
    
    print(f"Processing {len(plate_names)} plates in this directory:")
    for plate in sorted(plate_names):
        # Count TIFF files for this plate
        tiff_count = len(glob(os.path.join(search_dir, plate, "*.tiff")))
        print(f"  {plate}: {tiff_count} TIFF files")
    
    total_tiffs = len(glob(os.path.join(search_dir, "*", "*.tiff")))
    print(f"Total TIFF files in this directory: {total_tiffs}")
    print("========================\n")

def load_img_paths(search_dir: str):
    """Find all image files in the directory structure"""
    print(f"Searching for images in: {search_dir}")
    
    # Report on plates in this specific directory
    report_plates(search_dir)
    
    # Look for TIFF files in plate directories
    all_img_paths = glob(os.path.join(search_dir, "*", "*.tiff"))
    
    print(f"Total TIFF files found: {len(all_img_paths)}")
    if len(all_img_paths) > 0:
        print("Example paths:")
        for i, path in enumerate(all_img_paths[:5]):
            print(f"  {i+1}. {path}")
    
    if len(all_img_paths) == 0:
        print("ERROR: No images found. Check the input directory.")
        return pd.DataFrame()
    
    print("Extracting metadata from filenames...")
    metadatas = []
    for path in tqdm(all_img_paths, desc="Processing image paths"):
        try:
            metadatas.append(get_metadata(path))
        except Exception as e:
            print(f"Error processing path {path}: {e}")
    
    return pd.DataFrame(metadatas)

def get_metadata(path: str) -> Metadata:
    """Extract metadata from the filename and path - handles multiple formats"""
    # Extract barcode from path (parent directory)
    barcode = path.split(os.sep)[-2]
    
    # Extract well, site, and channel from filename
    basename = os.path.basename(path).split(".")[0]
    
    # Try different filename patterns
    
    # Pattern 1: New format - A01_01_HOECHST 33342.tiff
    pattern1 = re.match(r'([A-P]\d+)_(\d+)_(.+)', basename)
    
    # Pattern 2: Old format - A01_1_0_hoechst.tiff
    pattern2 = re.match(r'([A-P]\d+)_(\d+)_\d+_(.+)', basename)
    
    # Pattern 3: Format with number prefix - A01_01_0_mitotracker.tiff
    pattern3 = re.match(r'([A-P]\d+)_(\d+)_\d+_(.+)', basename)
    
    if pattern1:
        well, site, channel = pattern1.groups()
    elif pattern2 or pattern3:
        well, site, channel = pattern2.groups() if pattern2 else pattern3.groups()
        # Remove any prefix numbers from the channel (like "0_")
        channel = re.sub(r'^\d+_', '', channel)
    else:
        # Fall back to simple splitting
        parts = basename.split("_")
        if len(parts) >= 4:  # Old format or format with prefix
            well, site, prefix, channel = parts[:4]
            # Check if prefix is numeric and remove it from channel if needed
            if prefix.isdigit() and channel in ['mitotracker', 'alexa488', 'alexa568', 'hoechst', 'brightfield']:
                pass  # channel is already correctly extracted
            else:
                # Might be part of channel name or another format
                # Try to extract channel by removing any numeric prefix
                channel = "_".join(parts[2:])
                channel = re.sub(r'^\d+_', '', channel)
        elif len(parts) >= 3:  # New format
            well, site, channel = parts
        else:
            well = parts[0] if len(parts) > 0 else "unknown"
            site = parts[1] if len(parts) > 1 else "unknown"
            channel = parts[2] if len(parts) > 2 else "unknown"
            # Remove any prefix numbers from the channel
            channel = re.sub(r'^\d+_', '', channel)
    
    # Standardize well format (e.g., "A1" to "A01")
    if re.match(r'[A-P]\d$', well):
        well = f"{well[0]}{int(well[1:]):02d}"
    
    # Standardize site format (e.g., "1" to "01")
    if re.match(r'^\d$', site):
        site = f"{int(site):02d}"
    
    return Metadata(barcode, well, site, channel, path)

def load_stack(group, auto_scale=False, percentile_low=1, percentile_high=99):
    """
    Load and combine channels for a well and site
    
    Args:
        group: DataFrame group containing channel information
        auto_scale: Whether to use auto-scaling instead of fixed limits
        percentile_low: Low percentile for auto-scaling (default: 1)
        percentile_high: High percentile for auto-scaling (default: 99)
    """
    # Get all channels in this group
    channels = group['channel'].tolist()
    
    # Print debug info for channels in this group
    print(f"Processing group with channels: {channels}")
    
    # Initialize images
    nuclear_img = None
    green_img = None
    red_img = None
    
    # Process each channel based on name
    for _, row in group.iterrows():
        channel = row['channel']
        path = row['path']
        
        # Determine channel type based on name - case insensitive matching
        channel_lower = channel.lower()
        
        # Get intensity limits for fixed scaling - default if not found
        intensity_limits = (0, 15000)  # Default limit
        for known_channel, limits in CH_LIMITS.items():
            if known_channel.lower() in channel_lower:
                intensity_limits = limits
                break
        
        low_limit, high_limit = intensity_limits
        print(f"  Channel: {channel}, Intensity limits: {low_limit} to {high_limit}")
                
        # Load image
        try:
            img = io.imread(path)
            
            # Apply scaling
            if auto_scale:
                # Auto-scale using percentiles specific to this image
                p_low = np.percentile(img, percentile_low)
                p_high = np.percentile(img, percentile_high)
                
                # Clip and rescale to [0,1]
                img = np.clip(img, p_low, p_high)
                img = (img - p_low) / (p_high - p_low)
                print(f"  Auto-scaling {channel}: p{percentile_low}={p_low}, p{percentile_high}={p_high}")
            else:
                # Fixed scaling using predefined limits
                img = np.clip(img, low_limit, high_limit)
                img = (img - low_limit) / (high_limit - low_limit)
            
            # Assign image to correct color channel based on name
            if 'hoechst' in channel_lower or 'dapi' in channel_lower:
                nuclear_img = img
                print(f"  Assigned {channel} to nuclear (blue) channel")
            elif 'alexa488' in channel_lower or 'alexa 488' in channel_lower or 'fitc' in channel_lower:
                if green_img is None:
                    green_img = img
                else:
                    green_img = (green_img + img) / 2
                print(f"  Assigned {channel} to green channel")
            elif 'alexa568' in channel_lower or 'alexa 568' in channel_lower or 'tritc' in channel_lower:
                if red_img is None:
                    red_img = img
                else:
                    red_img = (red_img + img) / 2
                print(f"  Assigned {channel} to red channel")
            elif 'mitotracker' in channel_lower or 'cy5' in channel_lower or 'deep red' in channel_lower:
                if red_img is None:
                    red_img = img
                else:
                    red_img = (red_img + img) / 2
                print(f"  Assigned {channel} to red channel")
            elif 'brightfield' in channel_lower:
                # Skip brightfield for now
                print(f"  Skipping brightfield channel")
            else:
                print(f"  WARNING: Unknown channel type: {channel}")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    
    # Default to empty arrays if any channel is missing
    if nuclear_img is None or green_img is None or red_img is None:
        # Use first non-None image to get dimensions
        first_img = next((img for img in [nuclear_img, green_img, red_img] if img is not None), None)
        
        if first_img is None:
            # If all images are None, create a small placeholder
            img_shape = (1024, 1024)  # Default size
            print(f"Warning: No valid channels found for {group.iloc[0]['barcode']} {group.iloc[0]['well']} {group.iloc[0]['site']}")
        else:
            img_shape = first_img.shape
            
        if nuclear_img is None:
            nuclear_img = np.zeros(img_shape)
            print("  Created empty nuclear (blue) channel")
        if green_img is None:
            green_img = np.zeros(img_shape)
            print("  Created empty green channel")
        if red_img is None:
            red_img = np.zeros(img_shape)
            print("  Created empty red channel")
    
    # Create the RGB image
    rgb_img = np.dstack([
        np.clip(red_img, 0, 1),       # Red
        np.clip(green_img, 0, 1),     # Green
        np.clip(nuclear_img, 0, 1)    # Blue
    ])
    
    print(f"  Created RGB stack with shape: {rgb_img.shape}")
        
    return rgb_img

def group_to_savepath(group):
    """Determine the output filename for this image group"""
    row = group.iloc[0]
    barcode = row.barcode
    well = row.well
    site = row.site
    return f"{barcode}_{well}_{site}.png"

def save_thumbnail(img, save_path: str):
    """Save the thumbnail image"""
    img = util.img_as_ubyte(img)
    io.imsave(save_path, img, check_contrast=False)

def make_thumbnail(group):
    """Process a single group of images (one well location) and create both fixed and auto-scaled versions"""
    try:
        # Only print for the first few groups to avoid excessive output
        debug_print = len(group) < 20  # Limit debug printing
        
        # Check if we have all expected channels
        channels = set(group['channel'].str.lower())
        
        # Use case-insensitive matching to check for channels
        has_nuclear = any('hoechst' in ch or 'dapi' in ch for ch in channels)
        has_green = any('488' in ch or 'fitc' in ch for ch in channels)
        has_red = any('568' in ch or 'mito' in ch or 'deep red' in ch or 'cy5' in ch for ch in channels)
        
        # Print warning if missing important channels
        if not has_nuclear or not has_green or not has_red:
            missing = []
            if not has_nuclear: missing.append('nuclear (HOECHST/DAPI)')
            if not has_green: missing.append('green (Alexa488)')
            if not has_red: missing.append('red (MitoTracker/Alexa568)')
            print(f"Warning: Missing channels {', '.join(missing)} for {group.iloc[0]['barcode']} {group.iloc[0]['well']} {group.iloc[0]['site']}")
            print(f"  Available channels: {channels}")
        
        save_path = group_to_savepath(group)
        barcode = group.iloc[0].barcode
        
        # 1. Create fixed-scale version
        if debug_print:
            print(f"Creating fixed-scale image for {barcode} {group.iloc[0]['well']} {group.iloc[0]['site']}")
            
        fixed_img = load_stack(group, auto_scale=False)
        fixed_thumbnail = transform.resize(fixed_img, THUMBNAIL_SIZE, anti_aliasing=True)
        
        # Create directory for fixed-scale images
        fixed_save_dir = os.path.join(SAVE_DIR, "fixed", barcode)
        os.makedirs(fixed_save_dir, exist_ok=True)
        
        # Save fixed-scale thumbnail
        fixed_save_path = os.path.join(fixed_save_dir, save_path)
        save_thumbnail(fixed_thumbnail, fixed_save_path)
        
        if debug_print:
            print(f"  Saved fixed-scale thumbnail to: {fixed_save_path}")
        
        # 2. Create auto-scaled version
        if debug_print:
            print(f"Creating auto-scaled image for {barcode} {group.iloc[0]['well']} {group.iloc[0]['site']}")
            
        auto_img = load_stack(group, auto_scale=True)
        auto_thumbnail = transform.resize(auto_img, THUMBNAIL_SIZE, anti_aliasing=True)
        
        # Create directory for auto-scaled images
        auto_save_dir = os.path.join(SAVE_DIR, "auto", barcode)
        os.makedirs(auto_save_dir, exist_ok=True)
        
        # Save auto-scaled thumbnail
        auto_save_path = os.path.join(auto_save_dir, save_path)
        save_thumbnail(auto_thumbnail, auto_save_path)
        
        if debug_print:
            print(f"  Saved auto-scaled thumbnail to: {auto_save_path}")
            
    except Exception as e:
        print(f"Error processing group: {e}")
        print(f"Group details: {group[['barcode', 'well', 'site', 'channel']]}")

def report_thumbnails(save_dir):
    """Report on generated thumbnails for both fixed and auto-scaled versions"""
    print("\n====== THUMBNAIL GENERATION RESULTS ======")
    
    # Report for fixed-scale images
    fixed_dir = os.path.join(save_dir, "fixed")
    if os.path.exists(fixed_dir):
        # Find all plate directories
        plate_dirs = glob(os.path.join(fixed_dir, "*/"))
        plate_names = [os.path.basename(d.rstrip('/')) for d in plate_dirs]
        
        print(f"Generated fixed-scale thumbnails for {len(plate_names)} plates:")
        for plate in sorted(plate_names):
            # Count PNG files for this plate
            png_count = len(glob(os.path.join(fixed_dir, plate, "*.png")))
            print(f"  {plate}: {png_count} thumbnails")
        
        total_fixed_thumbnails = len(glob(os.path.join(fixed_dir, "*", "*.png")))
        print(f"Total fixed-scale thumbnails generated: {total_fixed_thumbnails}")
    
    # Report for auto-scaled images
    auto_dir = os.path.join(save_dir, "auto")
    if os.path.exists(auto_dir):
        # Find all plate directories
        plate_dirs = glob(os.path.join(auto_dir, "*/"))
        plate_names = [os.path.basename(d.rstrip('/')) for d in plate_dirs]
        
        print(f"Generated auto-scaled thumbnails for {len(plate_names)} plates:")
        for plate in sorted(plate_names):
            # Count PNG files for this plate
            png_count = len(glob(os.path.join(auto_dir, plate, "*.png")))
            print(f"  {plate}: {png_count} thumbnails")
        
        total_auto_thumbnails = len(glob(os.path.join(auto_dir, "*", "*.png")))
        print(f"Total auto-scaled thumbnails generated: {total_auto_thumbnails}")
    
    print("======================================\n")

def process_single_directory(raw_img_dir, save_dir):
    """Process a single input directory"""
    global SAVE_DIR
    SAVE_DIR = save_dir
    
    print(f"\n====== PROCESSING DIRECTORY: {raw_img_dir} ======")
    print(f"Output directory: {save_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading image paths...")
    img_df = load_img_paths(raw_img_dir)
    
    if len(img_df) == 0:
        print("No images found to process. Exiting.")
        return 1
    
    # Filter out plates that start with four zeros
    original_count = len(img_df)
    img_df = img_df[~img_df['barcode'].str.startswith('0000')]
    filtered_count = original_count - len(img_df)
    
    if filtered_count > 0:
        print(f"\nFiltered out {filtered_count} images from plates starting with '0000'")
        
        # Print the filtered plate barcodes if any were removed
        filtered_barcodes = set()
        for path in glob(os.path.join(raw_img_dir, "0000*")):
            barcode = os.path.basename(path.rstrip('/'))
            filtered_barcodes.add(barcode)
        
        if filtered_barcodes:
            print("Skipped plates:")
            for barcode in sorted(filtered_barcodes):
                print(f"  {barcode}")
    
    # Print unique channels found
    channels = img_df['channel'].unique()
    print(f"\nUnique channels found: {channels}")
    
    # Group by barcode, well, and site
    print("Grouping images by barcode, well, and site...")
    groups = [group for _, group in img_df.groupby(["barcode", "well", "site"])]
    print(f"\nNumber of groups to process: {len(groups)}")
    
    if len(groups) == 0:
        print("No valid groups to process. Exiting.")
        return 1
    
    # Print some group examples
    if len(groups) > 0:
        print("\nExample of first group:")
        print(groups[0][['barcode', 'well', 'site', 'channel']])
    
    # Process groups with multiprocessing
    print("\nProcessing groups with multiprocessing...")
    with multiprocessing.Pool(16) as p:
        for _ in tqdm(p.imap_unordered(make_thumbnail, groups), total=len(groups)):
            pass
    
    # Report results
    report_thumbnails(save_dir)
    
    return 0

def main():
    parser = argparse.ArgumentParser(description="Generate thumbnails from image stacks.")
    parser.add_argument("raw_img_dir", help="Directory containing raw images")
    parser.add_argument("save_dir", help="Directory to save thumbnails")
    parser.add_argument("--scan-only", action="store_true", help="Only scan directories without processing")
    parser.add_argument("--input-dirs", nargs='+', help="Additional input directories to scan (but not process)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--scaling", choices=["fixed", "auto", "both"], default="both",
                        help="Which scaling method to use (fixed, auto, or both)")
    
    args = parser.parse_args()
    
    # Pass scaling option to the global scope for use in make_thumbnail
    global SCALING_MODE
    SCALING_MODE = args.scaling
    
    raw_img_dir = args.raw_img_dir
    save_dir = args.save_dir
    
    # If we have additional directories to scan
    if args.input_dirs:
        all_dirs = [raw_img_dir] + args.input_dirs
        print(f"Scanning all {len(all_dirs)} directories...")
        scan_all_input_folders(all_dirs)
    
    # If scan-only mode, exit after scanning
    if args.scan_only:
        print("Scan-only mode - exiting without processing images.")
        return 0
    
    # Process the single input directory
    exit_code = process_single_directory(raw_img_dir, save_dir)
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)