#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import numpy as np
import bert_score
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict
from lpips import LPIPS
import open_clip
from PIL import Image
from torchvision import transforms
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from diffusers import AutoPipelineForText2Image
from sentence_transformers import SentenceTransformer, util
import gc
# ======= 全局路径配置 =======
ROOT_DIR = Path("/project/pi_shiqingma_umass_edu/zwen_umass_edu/baselines/blip_10per")
DATA_ROOT = Path("/project/pi_shiqingma_umass_edu/mingzheli/CVPR_Inversion/data")

DATASETS = ["diffusiondb", "flickr30k", "lexica_test", "mscoco"]
MODELS = ["FLUX_1_dev", "SD15", "SD35_medium", "SDXL_Turbo"]

PPL_MODEL_ID = "openai-community/gpt2"
CACHE_DIR = "/project/pi_shiqingma_umass_edu/mingzheli/.cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======= 模型配置 =======
MODEL_SPECS = {
    "SD15": {
        "model_id": "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5",
        "height": 512, "width": 512,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
    },
    "SDXL_Turbo": {
        "model_id": "/project/pi_shiqingma_umass_edu/mingzheli/model/sdxl-turbo",
        "height": 1024, "width": 1024,
        "num_inference_steps": 1,
        "guidance_scale": 0.0,
    },
    "FLUX_1_dev": {
        "model_id": "/project/pi_shiqingma_umass_edu/mingzheli/model/FLUX.1-dev",
        "height": 1024, "width": 1024,
        "num_inference_steps": 30,
        "guidance_scale": 3.5,
    },
    "SD35_medium": {
        "model_id": "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-3.5-medium",
        "height": 1024, "width": 1024,
        "num_inference_steps": 30,
        "guidance_scale": 4.5,
    },
}

# ======= 通用模型加载 =======
print("🔹 Loading shared models...")
sbert_model = SentenceTransformer(
    "/project/pi_shiqingma_umass_edu/mingzheli/model/all-mpnet-base-v2",
    device=DEVICE
)

transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
lpips_model = LPIPS(net="alex")

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai", device=DEVICE,
    cache_dir="/project/pi_shiqingma_umass_edu/mingzheli/model"
)

ppl_model = GPT2LMHeadModel.from_pretrained(PPL_MODEL_ID, cache_dir=CACHE_DIR).to(DEVICE)
ppl_tokenizer = GPT2TokenizerFast.from_pretrained(PPL_MODEL_ID)


# ======= 工具函数 =======
def calculate_clip_similarity(image_a, image_b):
    processed_a = clip_preprocess(image_a).unsqueeze(0).to(DEVICE)
    processed_b = clip_preprocess(image_b).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat_a = clip_model.encode_image(processed_a)
        feat_b = clip_model.encode_image(processed_b)
        feat_a /= feat_a.norm(dim=-1, keepdim=True)
        feat_b /= feat_b.norm(dim=-1, keepdim=True)
    return (feat_a @ feat_b.T).item()


def ppl(text: str):
    """Compute normalized perplexity (nPPL) for a given text."""
    max_length = ppl_model.config.n_positions
    stride = 512
    encodings = ppl_tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    total_nll = 0.0
    total_tokens = 0
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = ppl_model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        total_nll += neg_log_likelihood
        total_tokens += trg_len
        prev_end_loc = end_loc

        if end_loc == seq_len:
            break

    # 归一化困惑度（Normalized Perplexity）
    mean_nll = total_nll / total_tokens
    normalized_ppl = torch.exp(mean_nll)

    return normalized_ppl



def get_best_text(image_path, prompts, pipe, spec, seed=0):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    orig_image = Image.open(image_path).convert("RGB")

    best_clip = -1e9
    best_lpips = 1e9
    best_text = ""

    for prompt in tqdm(prompts, desc=f"Evaluating {Path(image_path).name}"):
        with torch.no_grad():
            out = pipe(prompt,
                       height=spec["height"],
                       width=spec["width"],
                       num_inference_steps=spec["num_inference_steps"],
                       guidance_scale=spec["guidance_scale"],
                       generator=generator)
            pred_img = out.images[0]

        clip_sim = calculate_clip_similarity(orig_image, pred_img)
        orig_tensor = transform(orig_image).unsqueeze(0) * 2 - 1
        pred_tensor = transform(pred_img).unsqueeze(0) * 2 - 1
        lpips_val = lpips_model(orig_tensor, pred_tensor).item()

        if clip_sim > best_clip:
            best_clip = clip_sim
            best_lpips = lpips_val
            best_text = prompt

    return best_clip, best_lpips, best_text


# ======= 主循环：16 个子文件夹 =======
def main():
    for dataset in DATASETS:
        for model_name in MODELS:
            print(f"\n🚀 Running {dataset} / {model_name}...")
            spec = MODEL_SPECS[model_name]

            results_dir = ROOT_DIR / dataset / model_name
            data_dir = DATA_ROOT / dataset / model_name
            result_file = results_dir / "result.json"
            prompts_file = data_dir / "prompts.json"

            pipe = AutoPipelineForText2Image.from_pretrained(spec["model_id"]).to(DEVICE)

            with open(prompts_file, "r") as f:
                meta_data = json.load(f)

            if result_file.exists():
                with open(result_file, "r") as f:
                    best_texts = json.load(f)
            else:
                best_texts = {}

            txt_files = [f for f in os.listdir(results_dir) if f.endswith(".txt") and len(f) == 7]
            for txt_file in txt_files:
                with open(results_dir / txt_file, "r") as file:
                    results = [line.strip() for line in file if line.strip()]

                image_name = Path(txt_file).stem
                image_path = data_dir / f"{image_name}.png"

                best_clip, best_lpips, best_text = get_best_text(image_path, results, pipe, spec)
                ref_text = meta_data[image_name]
                P, R, F1 = bert_score.score([best_text], [ref_text], lang="en", verbose=False)

                emb_best = sbert_model.encode([best_text], convert_to_tensor=True)
                emb_ref = sbert_model.encode([ref_text], convert_to_tensor=True)
                sbert_score = util.cos_sim(emb_best, emb_ref).item()
                ppl_score = ppl(best_text).item()

                best_texts[str(image_path)] = {
                    "text": best_text,
                    "cos_sim": best_clip,
                    "lpips_sim": best_lpips,
                    "P": float(P),
                    "R": float(R),
                    "F1": float(F1),
                    "sbert": sbert_score,
                    "ppl": ppl_score,
                }

                with open(result_file, "w") as file:
                    json.dump(best_texts, file, indent=4)

            print(f"✅ Finished {dataset}/{model_name}")
            # ===== 显存清理 =====
            del pipe
            torch.cuda.empty_cache()
            gc.collect()
            print(f"🧹 GPU memory cleared after finishing {model_name}")


if __name__ == "__main__":
    main()
