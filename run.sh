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
TARGET_MODEL_PATH="/project/pi_shiqingma_umass_edu/mingzheli/model/sdxl-turbo"
RECORD_DIR="/home/mingzhel_umass_edu/Modifier_fuzz/logs_structured_mcts_fuzzer_sdxl_lexica_baseonly"
# Path to your VL-Model for local mutations (e.g., Qwen-VL)
# This is used if you are NOT using GPT-4o for mutations.
IMAGE_MODEL_PATH="/project/pi_shiqingma_umass_edu/mingzheli/model/Qwen2-VL-2B-Instruct"
# IMAGE_MODEL_PATH="/project/pi_shiqingma_umass_edu/mingzheli/model/Qwen2.5-VL-3B-Instruct"
# IMAGE_MODEL_PATH="gpt-4o"
# Your OpenAI API Key (leave empty if using only local models)
WORK_DIR="/home/mingzhel_umass_edu/Modifier_fuzz/base_prompt_results_sdxl_lexica"
OPENAI_API_KEY="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"
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

# 2. Loop through all image files in the directory
for image_path in "$IMAGE_DIR"/*.png "$IMAGE_DIR"/*.jpg "$IMAGE_DIR"/*.jpeg; do
    # count=$((count+1))
    # if [ $count -lt 4 ]; then
    #     continue
    # fi
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

    # python train_imm_blip.py \
    #     --image_dir "$image_path" \
    #     --work_dir "$WORK_DIR" \
    #     --target_model_path "$TARGET_MODEL_PATH"



    echo "[TRAINING] RL Training finished for $image_filename."

    # --- 5. Run Testing ---
    echo "[FUZZ] Starting modifier mutation for $image_filename..."


    python fuzz_run.py \
        --target_image_path "$image_path" \
        --target_model_path "$TARGET_MODEL_PATH" \
        --image_model_path "$IMAGE_MODEL_PATH" \
        --openai_key "$OPENAI_API_KEY" \
        --work_dir "$WORK_DIR" \
        --record_dir "$RECORD_DIR" 

        

    echo "[FUZZ] Evaluation finished for $image_filename."

done

echo ""
echo "======================================================="
echo "All images processed."
echo "======================================================="
