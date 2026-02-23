# PromptMiner

Our paper, *PROMPTMINER: Black-Box Prompt Stealing against Text-to-Image Generative Models via Reinforcement Learning and Fuzz Optimization*, has been accepted to CVPR 2026.

PromptMiner has two stages:

1. `train_imm_blip.py`: Phase I (imitation learning + RL with BLIP + PPO) to generate a base prompt and save it to `best_prompts.json`
2. `fuzz_run.py`: Phase II prompt mutation / fuzzing based on the Phase I base prompt, producing per-image search logs `fuzz_results_*.csv`

`run.sh` batch-processes images in a directory and runs both stages in sequence.

## 1. Installation

```bash
conda create -n promptminer python=3.10 -y
conda activate promptminer

# Use -r (not -t)
pip install -r requirements.txt
```


## 2. Models Used by the Code

- BLIP (Phase I / base prompt generation): [`Salesforce/blip-image-captioning-large`](https://huggingface.co/Salesforce/blip-image-captioning-large) (the code also uses a base BLIP variant in some paths)
- VLM for local mutation (Phase II): [`Qwen/Qwen2-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- Diffusion model (target image generation model): e.g. [`stabilityai/sdxl-turbo`](https://huggingface.co/stabilityai/sdxl-turbo)

The example dataset directory in this repo uses `SDXL_Turbo`.

## 3. Data Preparation (Most Important)

### Input format for batch running `run.sh`

`run.sh` expects an image directory as input. It will iterate over:

- `*.png`
- `*.jpg`
- `*.jpeg`

For this project, the image directory should follow this layout:

```bash
data/<dataset_name>/<model_name>/
```

This is important because other scripts (especially `eval.py` and the result export workflow) assume the same dataset/model directory organization.

Example directory (included in this repo):

```bash
./data/lexica_test/SDXL_Turbo
```

You can also use your own directory as long as it contains the target images.

## 4. Edit `run.sh` Before Running

Open `run.sh` and check at least these settings:

```bash
TARGET_MODEL_PATH="stabilityai/sdxl-turbo"   # target diffusion model
RECORD_DIR="./logs"                           # Phase II CSV log output
IMAGE_MODEL_PATH="Qwen/Qwen2-VL-2B-Instruct"  # local VLM (change if needed)
WORK_DIR="./results/base_prompt"              # Phase I output directory (use a writable path)
OPENAI_API_KEY=""                             # leave empty if not using OpenAI
```

Notes:

- If `WORK_DIR` is set to `/base_prompt` (absolute path), many machines will fail due to permission issues. Use a relative path instead (for example `./results/base_prompt`).
- `run.sh` skips an image if `RECORD_DIR/fuzz_results_<image>.csv` already exists, which is useful for resume/restart runs.

## 5. One-Command Batch Run (Recommended)

Run this inside the `PromptMiner` directory:

```bash
bash run.sh ./data/lexica_test/SDXL_Turbo
```

After the run finishes, you should see:

- `WORK_DIR/best_prompts.json`: the best base prompt found in Phase I for each image
- `RECORD_DIR/fuzz_results_<image>.csv`: full Phase II fuzzing logs (including `clip_similarity` and `full_prompt`)

## 6. Extract the Best Prompt from Each CSV

```bash
python utils/extract_best_prompt.py ./logs
```

Notes:

- The script selects the `full_prompt` with the maximum `clip_similarity` from each `fuzz_results_*.csv`
- The current script writes results to a fixed directory: `./results/lexica_test/SDXL_Turbo` (see line 7 in `utils/extract_best_prompt.py`)
- If you use your own dataset, update `output_dir` in that script

## 7. Run Each Stage Separately (Useful for Debugging)

### Phase I: Base prompt generation (BLIP + PPO)

```bash
python train_imm_blip.py \
  --image_dir ./data/lexica_test/SDXL_Turbo/000.png \
  --work_dir ./results/base_prompt \
  --target_model_path stabilityai/sdxl-turbo \
  --image_range lexica_sdxl \
  --step 2000
```

Output: `./results/base_prompt/best_prompts.json`

### Phase II: Fuzzing / mutation

```bash
python fuzz_run.py \
  --target_image_path ./data/lexica_test/SDXL_Turbo/000.png \
  --target_model_path stabilityai/sdxl-turbo \
  --image_model_path Qwen/Qwen2-VL-2B-Instruct \
  --work_dir ./results/base_prompt \
  --record_dir ./logs \
  --openai_key ""
```

Notes:

- `fuzz_run.py` defaults to `--use_base_prompt True`, so it reads the base prompt from `WORK_DIR/best_prompts.json`
- The keys in `best_prompts.json` are image path strings, so `--target_image_path` must match the same path format used in Phase I (for example, both relative paths or both absolute paths)

## 8. `eval.py` (Not a single-image quick evaluation script)

`eval.py` is a batch benchmark evaluation script. By default, it iterates over multiple datasets and diffusion models:

- Datasets: `flickr30k`, `lexica_test`, `mscoco`
- Models: `FLUX_1_dev`, `SD15`, `SD35_medium`, `SDXL_Turbo`

Before running it directly, make sure the expected directory structure and result files already exist under `./results/...` and `./data/...`; otherwise it will fail with missing-file errors.

```bash
python eval.py
```
