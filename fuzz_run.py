# run_fuzzer.py

import os
import argparse
import torch
import numpy as np
import random
import openai
import json 
import os   
from fuzzer_core import ImagePromptFuzzer

def main():
    parser = argparse.ArgumentParser(description='MCTS Fuzzer for Image Prompt Optimization')

    # --- Task Core Parameters ---
    parser.add_argument('--target_image_path', type=str, default="/home/mingzhel_umass_edu/inverse/LatentTracer/data/flickr30k/004.png", help='Path to the target image file.')
    parser.add_argument('--target_model_path', type=str, default='/project/pi_shiqingma_umass_edu/mingzheli/model/sdxl-turbo', help='Path or Hub ID for the target image generation model.')

    # --- Mutator Model Parameters ---
    # For VLM, only OpenAI models are supported in this script. Adapt utils_image.py for local models.
    parser.add_argument('--image_model_path', type=str, default='/project/pi_shiqingma_umass_edu/mingzheli/model/Qwen2-VL-2B-Instruct', help='VLM model used to generate the initial prompt.')
    parser.add_argument('--model_path', type=str, default='gpt-3.5-turbo', help='LLM model used for prompt mutation.')
    parser.add_argument('--work_dir', type=str, default="./results", help='Path to the workplace.')
    # --- Fuzzing Control Parameters ---
    parser.add_argument('--max_query', type=int, default=100, help='The maximum number of image generations before stopping.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--cuda_id', type=int, default=0, help='GPU ID to use.')
    parser.add_argument('--record_dir', type=str, help='Directory to save intermediate results.')
    # parser.add_argument('--base_prompt', type=str, default="guy jumps into the pool play the floating floating floating", help='base prompt')

    # --- Logging Parameters ---
    parser.add_argument('--openai_key', type=str, default='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', help='OpenAI key for Mutator LLM if using OpenAI models.')
    args = parser.parse_args()
    
    openai.api_key = args.openai_key
    json_filename = os.path.join(args.work_dir, "best_prompts.json")
    image_key = args.target_image_path
    
    # 先初始化一个默认的 base_prompt
    args.base_prompt = "a high quality photo of" # 如果加载失败，会使用这个默认值

    print(f"\n--- Attempting to load base prompt for '{image_key}' from {json_filename} ---")
    
    if os.path.exists(json_filename):
        try:
            with open(json_filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查当前图片路径是否存在于 JSON 文件中
            if image_key in data and "best_prompt" in data[image_key]:
                loaded_prompt = data[image_key]["best_prompt"]
                args.base_prompt = loaded_prompt
                print(f"✅ Successfully loaded prompt: '{args.base_prompt}'")
            else:
                print(f"⚠️ Warning: Prompt for '{image_key}' not found in {json_filename}. Using default.")
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"❌ Error reading {json_filename}: {e}. Using default prompt.")
    else:
        print(f"⚠️ Warning: File '{json_filename}' not found. Using default prompt.")

    # --- Set Seeds for Reproducibility ---
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 50)
    print("Starting MCTS Image Prompt Fuzzer")
    print(f"Parameters: {args}")
    print("=" * 50)

    # Initialize and run the fuzzer
    fuzzer = ImagePromptFuzzer(args)
    fuzzer.run()


if __name__ == "__main__":
    main()