#!/bin/bash
#SBATCH --job-name=thumbnails_HTC_V1_V2
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=ncpu

# Python environment path
PYTHON="/camp/home/tuerslj/.conda/envs/make_thumbnails/bin/python"

# Output directory
OUTPUT_DIR="/nemo/project/proj-prosperity/hts/raw/projects/20251020_HaCaT_GSK_HTC_V1_V2_cell_paint/data/images/thumbnails"

# Script path
SCRIPT="/nemo/stp/hts/working/Joe_Tuersley/code/spc-data-explorer/scripts/generate_thumbnails/scripts/generate_thumbnails_perc_and_auto_thresh_V1.py"

# Define all input directories
INPUT_DIRS=(
    "/nemo/project/proj-prosperity/hts/raw/projects/20251020_HaCaT_GSK_HTC_V1_V2_cell_paint/data/images/max_projected_images"
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
