import os
import json
import torch
import numpy as np
import bert_score
from pathlib import Path
from tqdm.auto import tqdm
from lpips import LPIPS
import open_clip
from PIL import Image
from torchvision import transforms
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from diffusers import AutoPipelineForText2Image

# this path is where you store all the txt files with generated prompts(000.txt, 001.txt), and one prompt per line
RESULTS_DIR = Path("/home/mingzhel_umass_edu/Modifier_fuzz/results_sdxl_txt_mscoco_5000pre_lexica_baseonly")
# this path is where you store the lexica dataset, you should have prompts.json and images(000.png, 001.png) in this folder
LEXICA_DIR = Path("/home/mingzhel_umass_edu/inverse/LatentTracer/data/lexica")
# this is where the final results will be saved
RESULT_FILE = RESULTS_DIR / "result.json"
# this is the path to the prompts.json file in the lexica dataset(datasets can be changed to other datasets as well)
PROMPTS_FILE = LEXICA_DIR / "prompts.json"
# modify this path to your model path
MODEL_ID = "/project/pi_shiqingma_umass_edu/mingzheli/model/sdxl-turbo"
PPL_MODEL_ID = "openai-community/gpt2"
CACHE_DIR = "/project/pi_shiqingma_umass_edu/mingzheli/.cache"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===== 图像变换 =====
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# ===== 初始化模型 =====
lpips_model = LPIPS(net="alex")

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai", device=DEVICE, cache_dir="/project/pi_shiqingma_umass_edu/mingzheli/model"
)

pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID).to(DEVICE)

ppl_model = GPT2LMHeadModel.from_pretrained(PPL_MODEL_ID, cache_dir=CACHE_DIR).to(DEVICE)
ppl_tokenizer = GPT2TokenizerFast.from_pretrained(PPL_MODEL_ID)


# ===== 工具函数 =====
def calculate_clip_similarity(image_a, image_b):
    """计算两张图像的 CLIP 相似度"""
    processed_a = clip_preprocess(image_a).unsqueeze(0).to(DEVICE)
    processed_b = clip_preprocess(image_b).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat_a = clip_model.encode_image(processed_a)
        feat_b = clip_model.encode_image(processed_b)
        feat_a /= feat_a.norm(dim=-1, keepdim=True)
        feat_b /= feat_b.norm(dim=-1, keepdim=True)
    return (feat_a @ feat_b.T).item()


def ppl(text):
    """计算文本的 perplexity"""
    max_length = ppl_model.config.n_positions
    stride = 512
    encodings = ppl_tokenizer(text, return_tensors="pt")

    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = ppl_model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return torch.exp(torch.stack(nlls).mean())


def get_best_text(image_path, prompts, seed=0):
    """给定候选 prompt，找到生成效果最佳的文本"""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    orig_image = Image.open(image_path).convert("RGB")

    best_loss_clip = 0.0
    best_loss_lpips = 1.0
    best_text = ""

    for prompt in tqdm(prompts, desc=f"Evaluating {Path(image_path).name}"):
        with torch.no_grad():
            pred_imgs = pipe(
                prompt,
                guidance_scale=0,
                num_inference_steps=1,
                generator=generator
            ).images

            eval_loss_clip = calculate_clip_similarity(orig_image, pred_imgs[0])

            orig_tensor = transform(orig_image).unsqueeze(0) * 2 - 1
            pred_tensor = transform(pred_imgs[0]).unsqueeze(0) * 2 - 1
            eval_loss_lpips = lpips_model(orig_tensor, pred_tensor)

            if best_loss_clip < eval_loss_clip:
                best_loss_clip = eval_loss_clip
                best_loss_lpips = eval_loss_lpips
                best_text = prompt

    print(f"\n[Best] CLIP={best_loss_clip:.3f}, LPIPS={best_loss_lpips.item():.3f}, Text={best_text}")
    return best_loss_clip, best_loss_lpips.item(), best_text


# ===== 主流程 =====
def main():
    txt_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".txt") and len(f) == 7]

    with open(PROMPTS_FILE, "r") as file:
        meta_data = json.load(file)

    # 如果已有结果，先加载
    if RESULT_FILE.exists():
        with open(RESULT_FILE, "r") as file:
            best_texts = json.load(file)
    else:
        best_texts = {}

    for i, txt_file in enumerate(txt_files, 1):
        print(f"\n[{i}/{len(txt_files)}] Processing {txt_file} ...")

        with open(RESULTS_DIR / txt_file, "r") as file:
            results = file.readlines()

        image_name = Path(txt_file).stem
        image_path = LEXICA_DIR / f"{image_name}.png"

        best_loss_clip, best_loss_lpips, best_text = get_best_text(image_path, results)

        P, R, F1 = bert_score.score(
            [best_text],
            [meta_data[f"{image_name}.png"]],
            lang="en",
            verbose=False
        )

        ppl_score = ppl(best_text)

        best_texts[str(image_path)] = {
            "text": best_text,
            "cos_sim": best_loss_clip,
            "lpips_sim": best_loss_lpips,
            "P": P.item(),
            "R": R.item(),
            "F1": F1.item(),
            "ppl": ppl_score.item()
        }

        # 每次迭代保存
        with open(RESULT_FILE, "w") as file:
            json.dump(best_texts, file, indent=4)


if __name__ == "__main__":
    main()
