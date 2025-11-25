#!/bin/bash
#SBATCH --job-name=make_thumbnails
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=ncpu

# Python environment path
PYTHON="/camp/home/tuerslj/.conda/envs/make_thumbnails/bin/python"

# Output directory
OUTPUT_DIR="/nemo/project/proj-prosperity/hts/raw/projects/20250219_GSK_fragments_V3_clickable_V1_with_reference_sets/data/images/thumbnails"

# Script path
SCRIPT="/nemo/stp/hts/working/Joe_Tuersley/code/spherical-phenotype-clustering-2/generate_thumbnails/generate_thumbnails_perc_and_auto_thresh_V1.py"

# Define all input directories
INPUT_DIRS=(
    "/nemo/project/proj-prosperity/hts/raw/projects/20250219_GSK_fragments_V3_clickable_V1/data/images/max_projected_images"
    "/nemo/project/proj-prosperity/hts/raw/projects/20250303_HaCaT_CRISPR_cell_paint_pilot_4_day_week_0/data/images/max_projected_images"
    "/nemo/project/proj-prosperity/hts/raw/projects/20230911_prosperity_jump_hacat_repeat/data/images/max_projected_images/20x_confocal"
    "/nemo/project/proj-prosperity/hts/raw/projects/20231012_prosperity_crispr_validation_v2_repeat_hacat_20x_confocal/data/images/max_projected_images"
)

# First, scan all directories to get an overview
echo "Scanning all directories..."
FIRST_DIR="${INPUT_DIRS[0]}"
OTHER_DIRS="${INPUT_DIRS[@]:1}"
$PYTHON $SCRIPT "$FIRST_DIR" "$OUTPUT_DIR" --scan-only --input-dirs ${OTHER_DIRS[@]}

# Now process each directory one by one
for INPUT_DIR in "${INPUT_DIRS[@]}"; do
    echo "Processing: $INPUT_DIR"
    $PYTHON $SCRIPT "$INPUT_DIR" "$OUTPUT_DIR" --scaling both
done

echo "Job completed"
