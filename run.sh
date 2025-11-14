#!/bin/bash

# ==============================================================================
# Batch Training and Testing Script for Image Prompt Optimization
#
# This script iterates through all images in a specified directory,
# runs the RL training for each image, and then immediately evaluates
# the trained policy.
# ==============================================================================

# --- Configuration ---
# Please set these default paths and parameters to match your setup.

# Path to your image generation model (e.g., Stable Diffusion)
TARGET_MODEL_PATH="stabilityai/sdxl-turbo"
RECORD_DIR="./logs"
# Path to your VL-Model for local mutations (e.g., Qwen-VL)
# This is used if you are NOT using GPT-4o for mutations.
IMAGE_MODEL_PATH="Qwen/Qwen2-VL-2B-Instruct"
# Your OpenAI API Key (leave empty if using only local models)
WORK_DIR="/base_prompt"
OPENAI_API_KEY=""
mkdir -p $WORK_DIR
CUDA_ID=0

# --- Script Logic ---

# 1. Check for input directory argument
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_image_directory>"
    exit 1
fi

IMAGE_DIR=$1
echo "Starting batch processing for images in: $IMAGE_DIR"
echo "======================================================="
count=0
# 2. Loop through all image files in the directory
for image_path in "$IMAGE_DIR"/*.png "$IMAGE_DIR"/*.jpg "$IMAGE_DIR"/*.jpeg; do

    if [[ "$image_path" == *original* ]]; then
        echo "⏩ $image_path"
        continue
    fi
    
    count=$((count+1))
    echo "Processing: $image_path"

    # 3. Define variables for the current image
    image_filename=$(basename "$image_path")

    # --- MODIFICATION STARTS HERE ---

    # Get the base name of the image without its extension (e.g., "004" from "004.png")
    image_basename="${image_filename%.*}"
    
    # Define the path for the corresponding .txt record file
    record_file="$RECORD_DIR/fuzz_results_${image_basename}.csv"
    
    # Check if the .txt file exists
    if [ -f "$record_file" ]; then
        # If the file exists, print a message and skip to the next image
        echo "Record '$record_file' found for '$image_filename'. Skipping."
        continue
    fi
    
    echo ""
    echo "-------------------------------------------------------"
    echo "Processing Image: $image_filename Path: $image_path"
    echo "-------------------------------------------------------"



    # --- 4. Run Training ---
    echo "[TRAINING] Starting RL training for $image_filename..."

    python train_imm_blip.py \
       --image_dir "$image_path" \
       --work_dir "$WORK_DIR" \
       --target_model_path "$TARGET_MODEL_PATH"\
       --image_range "$TARGET_MODEL_PATH" \
       --step 2000\

    echo "[TRAINING] RL Training finished for $image_filename."

    # --- 5. Run Testing ---
    echo "[FUZZ] Starting modifier mutation for $image_filename..."

    
    python fuzz_run.py \
        --target_image_path "$image_path" \
        --target_model_path "$TARGET_MODEL_PATH" \
        --image_model_path "$IMAGE_MODEL_PATH" \
        --openai_key "$OPENAI_API_KEY" \
        --work_dir "$WORK_DIR" \
        --record_dir "$RECORD_DIR" \
     

        

    echo "[FUZZ] Evaluation finished for $image_filename."

done

echo ""
echo "======================================================="
echo "All images processed."
echo "======================================================="
